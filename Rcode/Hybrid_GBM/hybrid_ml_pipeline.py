"""
Hybrid ML Pipeline - Statistical Base + XGBoost Adjustment
Combines robust statistical estimates with ML-learned adjustment factors
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost', '-q'])
    import xgboost as xgb

from datetime import datetime
import json


class HybridMLModel:
    """
    True hybrid model: Statistical base estimate × ML adjustment factor.
    Uses XGBoost to learn adjustment factors rather than direct predictions.
    """
    
    def __init__(self, target_bias=1.05):
        self.target_bias = target_bias
        self.xgb_models = {}
        self.base_calibrations = {}
        self.lob_statistics = {}
        
    def analyze_lob_characteristics(self, train_features, lob):
        """Analyze LoB payment characteristics."""
        n_samples = len(train_features['cc'])
        
        # Calculate payment patterns
        n_with_future = 0
        future_reserves = []
        payment_counts = []
        
        for i in range(n_samples):
            future = 0
            n_future = 0
            
            for t in range(1, 12):
                if f'PayInd{t:02d}' in train_features:
                    mask = train_features.get(f'Time_Predict_Payment{t:02d}', np.ones(n_samples))
                    if mask[i] > 0:
                        ind = train_features[f'PayInd{t:02d}'][i]
                        amt = train_features[f'LogPay{t:02d}'][i]
                        if ind > 0:
                            future += np.exp(amt)
                            n_future += 1
            
            if future > 0:
                n_with_future += 1
                future_reserves.append(future)
                payment_counts.append(n_future)
            else:
                future_reserves.append(0)
                payment_counts.append(0)
        
        future_reserves = np.array(future_reserves)
        payment_rate = n_with_future / n_samples if n_samples > 0 else 0
        
        # Robust statistics on positive reserves
        if (future_reserves > 0).sum() > 0:
            positive_reserves = future_reserves[future_reserves > 0]
            
            # Remove outliers
            q1, q3 = np.percentile(positive_reserves, [25, 75])
            iqr = q3 - q1
            outlier_bound = q3 + 1.5 * iqr
            clean_reserves = positive_reserves[positive_reserves <= outlier_bound]
            
            if len(clean_reserves) > 0:
                median_reserve = np.median(clean_reserves)
                mean_reserve = np.mean(clean_reserves)
                std_reserve = np.std(clean_reserves)
            else:
                median_reserve = np.median(positive_reserves)
                mean_reserve = np.mean(positive_reserves)
                std_reserve = np.std(positive_reserves)
        else:
            median_reserve = 100
            mean_reserve = 100
            std_reserve = 50
        
        self.lob_statistics[lob] = {
            'payment_rate': payment_rate,
            'median_reserve': median_reserve,
            'mean_reserve': mean_reserve,
            'std_reserve': std_reserve,
            'n_samples': n_samples
        }
        
        return payment_rate, median_reserve, mean_reserve
    
    def calculate_base_estimate(self, features, i):
        """Calculate robust statistical base estimate."""
        n_samples = len(features['cc'])
        
        # Historical payment analysis
        historical_sum = 0
        n_payments = 0
        last_payment_time = -1
        payment_amounts = []
        
        for t in range(11):
            if f'PayInd{t:02d}' in features:
                ind = features[f'PayInd{t:02d}'][i]
                amt = features[f'LogPay{t:02d}'][i]
                if ind > 0:
                    payment = np.exp(amt)
                    historical_sum += payment
                    n_payments += 1
                    last_payment_time = t
                    payment_amounts.append(payment)
        
        # Base estimate using payment history
        if n_payments > 0:
            avg_payment = historical_sum / n_payments
            
            # Estimate remaining development
            if last_payment_time < 10:
                remaining_periods = 11 - last_payment_time
                
                # Use payment trend
                if len(payment_amounts) >= 2:
                    recent = np.mean(payment_amounts[-min(3, len(payment_amounts)):])
                    early = np.mean(payment_amounts[:min(3, len(payment_amounts))])
                    trend = recent / early if early > 0 else 0.8
                else:
                    trend = 0.8
                
                # Apply decay
                decay = 0.85 ** (last_payment_time / 3)
                base = avg_payment * remaining_periods * decay * trend
            else:
                # Tail factor
                base = historical_sum * 0.1
        else:
            # No payments yet - use conservative estimate
            base = 500
        
        return max(base, 50)
    
    def create_features(self, features, i):
        """Create feature vector for ML adjustment."""
        n_samples = len(features['cc'])
        
        # Basic claim features
        ml_features = [
            features['cc'][i],
            features['age'][i],
            features['inj_part'][i],
            features.get('AY', np.zeros(n_samples))[i],
            features.get('AQ', np.zeros(n_samples))[i],
            features.get('RepDel', np.zeros(n_samples))[i]
        ]
        
        # Payment history features
        n_payments = 0
        total_paid = 0
        last_payment = -1
        max_payment = 0
        payment_times = []
        
        for t in range(11):
            if f'PayInd{t:02d}' in features:
                ind = features[f'PayInd{t:02d}'][i]
                amt = features[f'LogPay{t:02d}'][i]
                if ind > 0:
                    payment = np.exp(amt)
                    n_payments += 1
                    total_paid += payment
                    last_payment = t
                    max_payment = max(max_payment, payment)
                    payment_times.append(t)
        
        ml_features.extend([
            n_payments,
            np.log1p(total_paid),
            last_payment,
            np.log1p(max_payment),
            11 - last_payment if last_payment >= 0 else 11
        ])
        
        # Payment velocity
        if len(payment_times) > 1:
            velocity = n_payments / (max(payment_times) - min(payment_times) + 1)
        else:
            velocity = 0.5
        ml_features.append(velocity)
        
        # Base estimate (important feature)
        base = self.calculate_base_estimate(features, i)
        ml_features.append(np.log1p(base))
        
        return ml_features
    
    def train_model(self, train_features, val_features, lob):
        """Train XGBoost to learn adjustment factors."""
        # Analyze LoB
        payment_rate, median_reserve, mean_reserve = self.analyze_lob_characteristics(train_features, lob)
        print(f"    Payment rate: {payment_rate:.1%}")
        print(f"    Median reserve: {median_reserve:.2f}")
        
        n_samples = len(train_features['cc'])
        
        # Prepare training data
        X_train = []
        y_train = []  # Adjustment factors
        base_estimates = []
        actual_reserves = []
        
        for i in range(n_samples):
            # Base estimate
            base = self.calculate_base_estimate(train_features, i)
            base_estimates.append(base)
            
            # Features
            features = self.create_features(train_features, i)
            X_train.append(features)
            
            # Actual future reserve
            actual = 0
            for t in range(1, 12):
                if f'PayInd{t:02d}' in train_features:
                    mask = train_features.get(f'Time_Predict_Payment{t:02d}', np.ones(n_samples))
                    if mask[i] > 0:
                        ind = train_features[f'PayInd{t:02d}'][i]
                        amt = train_features[f'LogPay{t:02d}'][i]
                        if ind > 0:
                            actual += np.exp(amt)
            actual_reserves.append(actual)
            
            # Target: adjustment factor (actual / base)
            if base > 0 and actual > 0:
                adjustment = actual / base
                # Cap extreme adjustments for stability
                adjustment = np.clip(adjustment, 0.01, 100)
            else:
                adjustment = 1.0
            y_train.append(np.log1p(adjustment))  # Log transform for better learning
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        base_estimates = np.array(base_estimates)
        actual_reserves = np.array(actual_reserves)
        
        # Train XGBoost if enough data
        if (actual_reserves > 0).sum() > 10:
            # Train model
            model = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.05,
                reg_alpha=0.05,
                reg_lambda=0.5,
                random_state=42,
                verbosity=0
            )
            
            # Focus on claims with actual reserves
            train_mask = actual_reserves > 0
            if train_mask.sum() > 5:
                model.fit(X_train[train_mask], y_train[train_mask], verbose=False)
                
                # Evaluate on training data
                train_adj_pred = np.expm1(model.predict(X_train))
                train_pred = base_estimates * train_adj_pred
                
                # Calculate base calibration
                valid_mask = (actual_reserves > 10) & (train_pred > 0)
                if valid_mask.sum() > 0:
                    ratios = actual_reserves[valid_mask] / train_pred[valid_mask]
                    
                    # Remove outliers
                    q1, q3 = np.percentile(ratios, [25, 75])
                    iqr = q3 - q1
                    clean_mask = (ratios >= q1 - 1.5*iqr) & (ratios <= q3 + 1.5*iqr)
                    
                    if clean_mask.sum() > 0:
                        base_calibration = np.median(ratios[clean_mask])
                    else:
                        base_calibration = np.median(ratios)
                else:
                    base_calibration = 1.0
                
                # Apply sparsity-based adjustment
                if payment_rate < 0.1:
                    calibration = base_calibration * 3.0
                elif payment_rate < 0.3:
                    calibration = base_calibration * 1.5
                else:
                    calibration = base_calibration
                
                # Apply target bias
                calibration *= self.target_bias
                
                self.xgb_models[lob] = model
                self.base_calibrations[lob] = np.clip(calibration, 0.5, 50)
            else:
                # Not enough positive samples
                self.xgb_models[lob] = None
                self.base_calibrations[lob] = 2.0 if payment_rate > 0.3 else 10.0
        else:
            # Fallback calibration
            self.xgb_models[lob] = None
            if payment_rate < 0.1:
                self.base_calibrations[lob] = 30.0
            elif payment_rate < 0.3:
                self.base_calibrations[lob] = 10.0
            else:
                self.base_calibrations[lob] = 2.0
        
        print(f"    Calibration: {self.base_calibrations[lob]:.3f}")
        print(f"    ML Model: {'XGBoost' if self.xgb_models.get(lob) else 'Statistical'}")
        
        return self.base_calibrations[lob]
    
    def predict(self, features, lob):
        """Predict using base × ML adjustment × calibration."""
        if lob not in self.base_calibrations:
            return 0
        
        calibration = self.base_calibrations[lob]
        model = self.xgb_models.get(lob)
        
        n_samples = len(features['cc'])
        total_reserve = 0
        
        for i in range(n_samples):
            # Check if prediction needed
            needs_pred = False
            for t in range(1, 12):
                mask_key = f'Time_Predict_Payment{t:02d}'
                if mask_key in features and features[mask_key][i] > 0:
                    needs_pred = True
                    break
            
            if not needs_pred:
                continue
            
            # Base estimate
            base = self.calculate_base_estimate(features, i)
            
            # ML adjustment if available
            if model is not None:
                feat_vec = self.create_features(features, i)
                adj_log = model.predict(np.array([feat_vec]))[0]
                adjustment = np.expm1(adj_log)
                adjustment = np.clip(adjustment, 0.1, 10)
            else:
                adjustment = 1.0
            
            # Final prediction
            prediction = base * adjustment * calibration
            total_reserve += prediction
        
        return total_reserve


def run_hybrid_ml_pipeline():
    """Run hybrid ML pipeline."""
    from data_processing import DataProcessor
    
    print("\n" + "="*80)
    print("HYBRID ML PIPELINE")
    print("Statistical Base × XGBoost Adjustment × Adaptive Calibration")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results_dir = Path("./Results")
    results_dir.mkdir(exist_ok=True)
    tables_dir = results_dir / "Tables"
    tables_dir.mkdir(exist_ok=True)
    
    datasets = [
        "../../data/raw/data.txt",
        "../../data/BI/dat_gpt.txt",
        "../../data/BI/dat_openelm.txt",
        "../../data/BI/dat_qwen3.txt",
        "../../data/BI/dat_smollm2.txt"
    ]
    
    all_results = []
    
    for data_path in datasets:
        if not Path(data_path).exists():
            continue
        
        dataset_name = Path(data_path).stem
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")
        
        processor = DataProcessor(data_path)
        lobs = processor.get_available_lobs()
        
        model = HybridMLModel(target_bias=1.05)
        dataset_results = []
        
        for lob in lobs:
            print(f"\n  LoB {lob}:")
            
            try:
                # Prepare data
                train_features, val_features, info = processor.prepare_lob_data(lob)
                n_samples = len(train_features['cc'])
                print(f"    Samples: {n_samples}")
                
                if n_samples < 20:
                    print(f"    Skipping - insufficient samples")
                    continue
                
                # Train
                calibration = model.train_model(train_features, val_features, lob)
                
                # Predict
                pred_features, _, true_reserves = processor.prepare_prediction_data(lob)
                estimated = model.predict(pred_features, lob)
                
                estimated = round(estimated)
                true_reserves = round(true_reserves)
                
                if true_reserves <= 0:
                    print(f"    Skipping - invalid true reserves")
                    continue
                
                bias = estimated - true_reserves
                bias_percent = (bias / true_reserves * 100)
                
                print(f"    Results:")
                print(f"      Estimated: {estimated:,}")
                print(f"      True:      {true_reserves:,}")
                print(f"      Bias:      {bias_percent:.1f}%")
                
                # Status
                if abs(bias_percent - 5) <= 5:
                    status = "SUCCESS"
                    print(f"      *** SUCCESS - Target achieved! ***")
                elif abs(bias_percent - 5) <= 15:
                    status = "GOOD"
                    print(f"      ** GOOD - Close to target **")
                elif abs(bias_percent) <= 50:
                    status = "ACCEPTABLE"
                    print(f"      * ACCEPTABLE *")
                else:
                    status = "NEEDS_IMPROVEMENT"
                
                result = {
                    'Dataset': dataset_name,
                    'LoB': lob,
                    'Estimated': estimated,
                    'True': true_reserves,
                    'Bias': bias,
                    'Bias_Percent': bias_percent,
                    'Calibration': calibration,
                    'Payment_Rate': model.lob_statistics[lob]['payment_rate'],
                    'ML_Model': 'XGBoost' if model.xgb_models.get(lob) else 'Statistical',
                    'Status': status
                }
                
                dataset_results.append(result)
                all_results.append(result)
                
            except Exception as e:
                print(f"    Error: {e}")
        
        if dataset_results:
            avg_bias = np.mean([abs(r['Bias_Percent']) for r in dataset_results])
            print(f"\n  Dataset Average |Bias|: {avg_bias:.1f}%")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = tables_dir / "hybrid_ml_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Create report
        txt_path = tables_dir / "hybrid_ml_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*90 + "\n")
            f.write("HYBRID ML PIPELINE RESULTS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*90 + "\n\n")
            
            f.write("METHOD:\n")
            f.write("- Statistical base estimate from payment history\n")
            f.write("- XGBoost learns adjustment factors (not direct predictions)\n")
            f.write("- Adaptive calibration based on payment sparsity\n")
            f.write("- Final prediction = Base × ML_Adjustment × Calibration\n\n")
            
            # Detailed results
            for dataset in df['Dataset'].unique():
                dataset_df = df[df['Dataset'] == dataset]
                f.write(f"\nDATASET: {dataset}\n")
                f.write("-"*90 + "\n")
                f.write("LoB | Predicted Reserves | True Reserves | Bias | Bias% | Cal | PayRate | Model | Status\n")
                f.write("-"*90 + "\n")
                
                for _, row in dataset_df.iterrows():
                    f.write(f"{row['LoB']:3} | {row['Estimated']:18,} | {row['True']:13,} | "
                           f"{row['Bias']:12,} | {row['Bias_Percent']:7.1f}% | "
                           f"{row['Calibration']:5.2f} | {row['Payment_Rate']:7.1%} | "
                           f"{row['ML_Model']:10} | {row['Status']}\n")
                
                avg_bias = dataset_df['Bias_Percent'].abs().mean()
                f.write("-"*90 + "\n")
                f.write(f"Dataset Average |Bias|: {avg_bias:.1f}%\n")
            
            # Summary
            f.write("\n" + "="*90 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("="*90 + "\n")
            
            overall_bias = df['Bias_Percent'].abs().mean()
            success_count = (df['Status'] == 'SUCCESS').sum()
            
            f.write(f"Overall Average |Bias|: {overall_bias:.1f}%\n")
            f.write(f"Success Rate: {success_count}/{len(df)} LoBs\n")
            f.write(f"XGBoost Used: {(df['ML_Model'] == 'XGBoost').sum()}/{len(df)}\n")
        
        print(f"\nResults saved to: {csv_path}")
        print(f"Report saved to: {txt_path}")
    
    return all_results


if __name__ == "__main__":
    results = run_hybrid_ml_pipeline()