library(plyr)
library(lubridate)
library(tidyr)
library(dplyr)
library(knitr)
library(kableExtra)
library(ggplot2)
library(tidyverse)
options(dplyr.summarise.inform = FALSE)

# Portfolio size
n <- 13000
set.seed(0)

# Probability distribution of the covariate 'Type'
prob.Type = c(0.60,0.25,0.15)

# Probability distribution of the unobservable covariate 'Hidden'
prob.Hidden = c(0.35,0.45,0.20)

# The assumed maximal reporting delay (in years)
max_rep_delay = 2

# The assumed maximal settlement delay (in years)
max_set_delay = 20


# Occurence date simulation -> Every possible date in between 2010-2020
date_occ_seq <- seq(as.Date("2010/1/1"), as.Date("2020/12/31"), "day")
date_occ     <- sample(date_occ_seq, size = n, replace = TRUE)

base_year <- year(date_occ_seq[1]) + max_rep_delay
df <- data.frame('occ.date' = as.character(date_occ), 'occ.year' = year(date_occ),
                 'occ.month' = factor(month(date_occ), levels = 1:12))
df <- df %>% mutate('occ.year.fact' = factor(occ.year, levels = sort(unique(occ.year))),
                    .after = 'occ.year')


# Covariate 'Type'
x1 <- sample(c("T1","T2","T3"), size = n, replace = TRUE,  prob = prob.Type)
x1 <- factor(x1, levels = c("T1",'T2',"T3"))

# Covariate 'Hidden'
xh <- sample(c("L","M","H"), size = n, replace = TRUE, prob = prob.Hidden)
xh <- factor(xh, levels = c("L","M","H"))

# Add to dataframe and define the claim number
df <- data.frame('claim.nr' = 1:n, 'type' = x1, 'hidden' = xh, df)
row.names(df) <- paste('claim',1:n)


# Reporting delay simulation - in days (max two years)
ndays        <- 365.25*max_rep_delay
delay_report <- floor(ndays * rbeta(n, c(1,2,3)[as.numeric(x1)], 10))
df <- df %>% dplyr::mutate('rep.delay' = as.numeric(delay_report)/365.25)

# Reporting date, year and month
date_report <- date_occ + delay_report

# Expand the portfolio with the claim's reporting information
df <- df %>%
  dplyr::mutate('rep.date'      = as.character(date_report),
                'rep.year'      = year(date_report) - base_year + 1,
                'rep.year.fact' = factor(rep.year, levels = sort(unique(rep.year)), labels = sort(unique(year(date_report)))),
                'rep.month'     = factor(month(date_report), levels = 1:12))


ggplot(df, aes(x = rep.delay*365.25, group = type, fill = type)) + 
  theme_bw(base_size = 15) + 
  geom_histogram(binwidth = 15, col = 'gray50') + 
  xlab('Reporting delay (days)') + 
  ylab('Frequency') 

# Simulating the 30 continuous payment dates per claim
n.payments <- 30
lijst <- lapply(1:n, function(x)
  as.numeric(date_report[x]) +
    floor(cumsum(c(rexp(1, rate = c(6,5,4)[as.numeric(x1[x])]),
                   rexp(n.payments - 1, rate = c(2,1.5,1)[as.numeric(x1[x])])))*365.25))
df_pay <- as.data.frame(data.table::transpose(lijst))
colnames(df_pay) <- paste0("Pay",1:n.payments)


# 'df_pay' contains the continuous payment dates, 'df_pay_copy' contains the continuous payment delays (in years)
df_pay_copy <- df_pay
df_pay_copy$Report <- as.numeric(date_report)
for(j in 1:30){
  df_pay_copy[,j] <- (df_pay_copy[,j] - df_pay_copy$Report)/365.25
}
df_pay_copy <- as.data.frame(t(df_pay_copy %>% dplyr::select(-c(Report))))
df_pay      <- as.data.frame(t(df_pay))



# Mean of the log-normal distribution
meanlog <- function(x1, x2){
  base <- log(c(100,200,400)[as.numeric(x1)])
  ext  <- 0.1*(payments)^(c(1.50,1.25,1.40)[as.numeric(x2)])
  rep(base + ext, times = n.payments)
}

# Simulate the 30 payment sizes for each claim
payments <- c(as.matrix(df_pay_copy))
df_size <- matrix(rlnorm(length(payments),
                         meanlog = meanlog(x1 = x1, x2 = xh),
                         sdlog =  1), nrow = n, ncol = 30, byrow = T)

# Arrange in a data frame 
df_size <- as.data.frame(t(df_size))
rownames(df_size) <- paste0("size", 1:n.payments)


df_size_t <- as.data.frame(t(df_size)) %>% mutate(type = x1) %>%
  tidyr::gather(key = 'number', value = 'size', - type)
avg_size  <- df_size_t %>% group_by(type, number) %>% summarise(S = mean(size, na.rm = TRUE))
avg_size$number <- factor(avg_size$number, levels = paste0('size',1:30))

ggplot(avg_size, aes(x = number, y = log(S), fill = type, colour = type)) +
  theme_bw(base_size = 15) + geom_bar(stat = 'identity', col = 'gray50', position = 'dodge') + xlab('k-th payment date') +
  ylab('log(size)') + scale_x_discrete(breaks = paste0('size',c(seq(1,30,3),30)), labels = c(seq(1,30,3),30))


# Assumed maximum settlement delay (in days)
ndays <- 365.25*max_set_delay

# Generate the settlement delays for each claim
settlement_delay <- floor(rbeta(n, shape1 = 1, shape2 = 8*c(1,0.75,0.5)[as.numeric(x1)])*ndays)

# Add claim's settlement information to the portfolio
date_settlement <- date_report + settlement_delay
df <- df %>% dplyr::mutate("settlement.date" = as.character(date_settlement)) %>%
  dplyr::mutate("settlement.year" = year(date_settlement))


ggplot(data.frame('settlement' = settlement_delay, df), aes(x = settlement, group = type, fill = type)) +
  theme_bw(base_size = 15) + geom_histogram(binwidth = 200,col = 'gray50') + xlab('Settlement delay (days)') +
  ylab('Frequency') + geom_vline(xintercept = 365.25*9, linetype = 'dashed', colour = 'gray50')


# Remove payments that occur after settlement date
max_delays <- as.numeric(date_settlement)
df_pay  <- t(df_pay)
df_size <- t(df_size)
ind <- which(df_pay > max_delays, arr.ind = TRUE)
df_pay[ind] = df_size[ind] <- NA
df_pay  <- t(df_pay)
df_size <- t(df_size)


# Define the end dates of each year (to group the data by years)
breaks <- as.numeric(as.Date(paste0(2009:2040,"-12-31")))
names(breaks) <- 2009:2040

# Group the continuous payments dates by development year since reporting
calc_obs_year <- function(k){
  pay_dates <- na.omit(as.numeric(df_pay[,k]))
  rep.year  <- df$rep.year[k] + base_year - 1
  breaks.k  <- breaks[as.character((rep.year - 1):(rep.year + 8))]
  vec0      <- (diff(rank(c(breaks.k+0.0001, pay_dates)))-1)[1:(length(breaks.k)-1)]
  vec1      <- as.numeric(rep(names(vec0), times = vec0)) - rep.year + 1
  vec       <- c(vec1, rep(NA, length(pay_dates) - length(vec1)))
  vec
}
obs_years <- lapply(1:n, function(k) calc_obs_year(k))



# Aggregrate claim sizes belonging to same development year since reporting
agg_fu <- function(k){
  obsy  <- obs_years[[k]]
  vec   <- if(length(obsy) > 0) sapply(na.omit(df_size[,k]) %>% split(obsy), FUN = sum) else NULL
  vec.add <- rep(0, 9 - length(vec))
  names(vec.add) <- as.character(1:9)[! as.character(1:9) %in% names(vec)]
  vec <- c(vec,vec.add)
  vec[order(names(vec))]
}

df.size <- t(sapply(1:n, function(k) agg_fu(k)))
colnames(df.size) <- paste0('size_obs', 1:9)



df.size.t <- as.data.frame(df.size) %>% mutate(type = x1) %>%
  tidyr::gather(key = 'OY', value = 'size', - type)
avg_size  <- df.size.t %>% group_by(type, OY) %>% summarise(S = mean(size[size > 0], na.rm = TRUE))

ggplot(avg_size, aes(x = OY, y = log(S), fill = type, colour = type)) +
  theme_bw(base_size = 15) + geom_bar(stat = 'identity', col = 'gray50', position = 'dodge') + 
  xlab('Development year since reporting') + ylab('log(Size)') + 
  scale_x_discrete(breaks = paste0('size_obs',1:9), labels = c(1:9))



# Is open indicator
create_open <- function(ones){
  ones.vec <- rep(0, 9)
  ones.vec[1:(min(ones+1,9))] <- 1
  ones.vec
}
df.open <- t(sapply(df$settlement.year - df$rep.year, create_open))
colnames(df.open) <- paste0("open_obs", 1:9)

# Settlement indicator (close)
create_settlement <- function(zeros){
  zeros.vec <- rep(1,9)
  zeros.vec[0:min(zeros,9)] <- 0
  zeros.vec
}
df.settlement <- t(sapply(df$settlement.year - df$rep.year, create_settlement))
colnames(df.settlement) <- paste0("settlement_obs", 1:9)



# Add to dataset
df <- df %>% bind_cols(as_tibble(df.open)) %>% bind_cols(as_tibble(df.settlement)) %>%
  bind_cols(as_tibble(df.size))



# Remove claims with reporting year later than 2028
df <- df %>% filter(rep.year <= 9, rep.year >= 1)
df <- df %>% mutate(rep.year.fact = factor(rep.year, levels = 1:9))

df <- df[1:10000,]

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


df.long

write.table(df.long
,"EH_10000_long.txt",sep=";",row.names=FALSE)

write.table(df[1:10000,]
            ,"EH_10000_short.txt",sep=";",row.names=FALSE)
