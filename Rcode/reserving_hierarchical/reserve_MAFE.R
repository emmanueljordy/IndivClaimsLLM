require(hirem)
require(tidyverse)
library(tidyr)
library(data.table)
library(gbm)




df <- as.data.table(read.table("./data/raw/EH_10000_short.txt", sep = ",",header = T))
df$rep.year <- as.integer(as.character(df$rep.year.fact))
df$claim.nr <- 1:nrow(df)
# Put in long format
df.long <- df %>% pivot_longer(cols = starts_with(c('open_obs', 'settlement_obs', 'size_obs')),
                               names_to = c(".value", "obs"),
                               names_sep = "_")

# Add development year variable
df.long <- df.long %>%
  dplyr::mutate('dev.year.fact' =  recode_factor(obs, obs1 = 1, obs2 = 2, obs3 = 3, obs4 = 4, obs5 = 5, obs6 = 6,
                                                 obs7 = 7, obs8 = 8, obs9 = 9), .before = 'open') %>%
  dplyr::mutate('dev.year' = as.numeric(as.character(dev.year.fact)), .before = 'dev.year.fact') %>%
  dplyr::select(-c(obs))

# Add calendar year and payment variable
df.long <- df.long %>% dplyr::mutate('calendar.year' = rep.year + dev.year - 1, .after = 'dev.year.fact') %>%
  dplyr::mutate('payment' = (size > 0)*1, .before = 'size')

reserving_data <- df.long



fact_cols <- which(grepl( "fact" , names( reserving_data )))
reserving_data <- reserving_data %>%
  mutate_at(vars(fact_cols), funs(as.factor))
reserving_data$type <- as.factor(reserving_data$type)
reserving_data <- na.omit(reserving_data)

# Creating the interaction effect
reserving_data$monthDev12 <- as.character(reserving_data$rep.month)
reserving_data$monthDev12[reserving_data$dev.year > 3] <- 'dev.year > 3'
reserving_data$devYearMonth <- factor(paste(reserving_data$dev.year, reserving_data$monthDev12, sep = '-'))
# Observed and prediction data set
observed_data   <- reserving_data %>% filter(calendar.year <= 9)
prediction_data <- reserving_data %>% filter(calendar.year > 9)


# Calculating the weights
reported_claims <- observed_data %>%
  dplyr::filter(dev.year == 1) %>%
  group_by(rep.year) %>% 
  dplyr::summarise(count = n()) %>%
  pull(count)

denominator <- tail(rev(cumsum(reported_claims)), -1)
numerator <- head(cumsum(rev(reported_claims)), -1)
weight <- c(10^(-6), numerator / denominator)
if(length(weight) <9){
  weight <- rep(weight, length.out =9)
}


names(weight) <- paste0('dev.year',1:(length(weight)))
weight







# Results of hierarchical model calibration
gbm_param_settle <- list('n.trees' = 225, 'interaction.depth' = 1, 'shrinkage' = 0.05)
gbm_param_pay    <- list('n.trees' = 125, 'interaction.depth' = 3, 'shrinkage' = 0.05)
gbm_param_size   <- list('n.trees' = 700, 'interaction.depth' = 1, 'shrinkage' = 0.05)

# Model specifications
model_gbm <- hirem(reserving_data) %>%
  split_data(observed = function(df) df %>% filter(calendar.year <= 9, open == 1)) %>%
  layer_gbm('settlement', distribution = 'bernoulli', bag.fraction = 0.75, n.minobsinnode = 100,
            n.trees = gbm_param_settle$n.trees, interaction.depth = gbm_param_settle$interaction.depth,
            shrinkage = gbm_param_settle$shrinkage, select_trees = 'last') %>%
  layer_gbm('payment', distribution = 'bernoulli', bag.fraction = 0.75, n.minobsinnode = 100,            
            n.trees = gbm_param_pay$n.trees, interaction.depth = gbm_param_pay$interaction.depth,
            shrinkage = gbm_param_pay$shrinkage, select_trees = 'last') %>%
  layer_gbm('size', distribution = 'gamma', bag.fraction = 0.75, n.minobsinnode = 100,
            n.trees = gbm_param_size$n.trees, interaction.depth = gbm_param_size$interaction.depth,
            shrinkage = gbm_param_size$shrinkage, select_trees = 'last',
            filter = function(data){data$payment == 1})


# Covariates
covariates_gbm <- c('type', 'dev.year.fact', 'rep.month', 'rep.year.fact', 'rep.delay', 'calendar.year')

# Fitting the hierarchical GBM
model_gbm <- fit(model_gbm,
                 weights = weight,
                 weight.var = 'dev.year',
                 balance.var = 'dev.year',
                 settlement = paste0('settlement ~ 1 + ', paste0(covariates_gbm, collapse = ' + ')),
                 payment = paste0('payment ~ 1 + ', paste0(c(covariates_gbm, 'settlement'), collapse = ' + ')),
                 size = paste0('size ~ 1 + ', paste0(c(covariates_gbm,'settlement'), collapse = ' + ')))


# Update function
update <- function(data) {
  data$dev.year <- data$dev.year + 1
  data$dev.year.fact <- factor(data$dev.year, levels = 1:9)
  
  data$calendar.year <- data$calendar.year + 1
  
  data$monthDev12[data$dev.year > 3] <- 'dev.year > 3'
  data$devYearMonth <- factor(paste(data$dev.year, data$monthDev12, sep = '-'))
  
  data
}


model_gbm <- register_updater(model_gbm, update)




simul_gbm <- simulate(model_gbm,
                      nsim = 100,
                      filter = function(data){dplyr::filter(data, dev.year <= 9, settlement == 0)},
                      data = model_gbm$data_observed %>% dplyr::filter(calendar.year == 9),
                      balance.correction = TRUE)

# Incremental run-off triangles
triangle_open    <- construct_triangle(data = observed_data %>% filter(open == 1), group.var1 = 'rep.year', 
                                       group.var2 = 'dev.year', value = 'open', cumulative = FALSE)
triangle_payment <- construct_triangle(data = observed_data, group.var1 = 'rep.year',
                                       group.var2 = 'dev.year', value = 'payment', cumulative = FALSE)
triangle_size    <- construct_triangle(data = observed_data, group.var1 = 'rep.year',
                                       group.var2 = 'dev.year', value = 'size', cumulative = FALSE)

# Number of open claims in the year following the evaluation date
settle.evalyear <- observed_data %>% 
  filter(open == 1, calendar.year == 9) %>%
  group_by(rep.year, dev.year) %>%
  summarise(settlement = sum(settlement))

# The number of open claims in the year after the evaluation date
triangle_open[row(triangle_open) + col(triangle_open) == 11] <- 
  (triangle_open[row(triangle_open) + col(triangle_open) == 10] - rev(settle.evalyear$settlement))[1:8]



nsim <- 100

# Predictions
obs_open_total <- prediction_data %>% filter(calendar.year != 10) %>% summarise(Total = sum(open)) %>% pull(Total)
gbm_open_total <- simul_gbm %>% filter(calendar.year != 10) %>% summarise(Total = sum(open)/nsim) %>% pull(Total)

# Print Results
c('Actual' =  obs_open_total, 'Hierarchical GBM' = gbm_open_total)


# Predictions
obs_pay_total <- prediction_data %>% filter(calendar.year != 10) %>% summarise(Total = sum(payment)) %>% pull(Total)

gbm_pay_total <- simul_gbm %>% filter(calendar.year != 10) %>% summarise(Total = sum(payment)/nsim) %>% pull(Total)

# Print Results
c('Actual' = obs_pay_total, 'Hierarchical GBM' = gbm_pay_total)

# Predictions
obs_size_total <- prediction_data %>% filter(calendar.year != 9) %>% summarise(Total = sum(size)) %>% pull(Total)


gbm_size_total <- simul_gbm %>% filter(calendar.year != 9) %>% summarise(Total = sum(size)/nsim) %>% pull(Total)

# Print Results
c('Actual' = obs_size_total, 'Hierarchical GBM' = gbm_size_total)

simul_gbm_indiv_BE <- simul_gbm%>% filter(calendar.year != 9)%>% group_by(claim.nr,simulation)%>% summarise(Total = sum(size))
simul_gbm_indiv_BE <- (reshape2::dcast(simul_gbm_indiv_BE , ... ~ simulation, value.var = "Total"))

