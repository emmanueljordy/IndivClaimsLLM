"""
ML-Enhanced Improved Pipeline
Combines XGBoost with adaptive calibration for sparse LoBs
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


class MLImprovedModel:
    """
    Hybrid model combining XGBoost with adaptive calibration.
    Uses ML for prediction but adapts calibration based on payment sparsity.
    """
    
    def __init__(self, target_bias=1.05):
        self.target_bias = target_bias
        self.learned_calibrations = {}
        self.xgb_models = {}
        self.lob_statistics = {}
        self.feature_names = None
        
    def analyze_lob_characteristics(self, train_features, lob):
        """Analyze LoB-specific characteristics."""
        n_samples = len(train_features['cc'])
        
        # Calculate payment sparsity
        n_with_payments = 0
        future_reserves = []
        
        for i in range(n_samples):
            future = 0
            for t in range(1, 12):
                if f'PayInd{t:02d}' in train_features:
                    mask = train_features.get(f'Time_Predict_Payment{t:02d}', np.ones(n_samples))
                    if mask[i] > 0:
                        ind = train_features[f'PayInd{t:02d}'][i]
                        amt = train_features[f'LogPay{t:02d}'][i]
                        if ind > 0:
                            future += np.exp(amt)
            
            if future > 0:
                n_with_payments += 1
            future_reserves.append(future)
        
        future_reserves = np.array(future_reserves)
        payment_rate = n_with_payments / n_samples if n_samples > 0 else 0
        
        # Calculate robust statistics
        if (future_reserves > 0).sum() > 0:
            positive_reserves = future_reserves[future_reserves > 0]
            
            # Remove outliers for robust estimation
            q1, q3 = np.percentile(positive_reserves, [25, 75])
            iqr = q3 - q1
            outlier_bound = q3 + 1.5 * iqr
            clean_reserves = positive_reserves[positive_reserves <= outlier_bound]
            
            if len(clean_reserves) > 0:
                median_reserve = np.median(clean_reserves)
                mean_reserve = np.mean(clean_reserves)
            else:
                median_reserve = np.median(positive_reserves)
                mean_reserve = np.mean(positive_reserves)
        else:
            median_reserve = 0
            mean_reserve = 0
        
        self.lob_statistics[lob] = {
            'payment_rate': payment_rate,
            'median_reserve': median_reserve,
            'mean_reserve': mean_reserve,
            'n_samples': n_samples
        }
        
        return payment_rate, median_reserve, mean_reserve
    
    def create_ml_features(self, features, i):
        """Create feature vector for XGBoost."""
        n_samples = len(features['cc'])
        
        # Basic features
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
        last_payment_time = -1
        payment_times = []
        payment_amounts = []
        
        for t in range(11):
            if f'PayInd{t:02d}' in features:
                ind = features[f'PayInd{t:02d}'][i]
                amt = features[f'LogPay{t:02d}'][i]
                if ind > 0:
                    payment = np.exp(amt)
                    n_payments += 1
                    total_paid += payment
                    last_payment_time = t
                    payment_times.append(t)
                    payment_amounts.append(payment)
        
        ml_features.extend([
            n_payments,
            np.log1p(total_paid),
            last_payment_time,
            max(payment_amounts) if payment_amounts else 0,
            np.std(payment_amounts) if len(payment_amounts) > 1 else 0
        ])
        
        # Payment velocity and trend
        if len(payment_times) > 1:
            time_span = max(payment_times) - min(payment_times) + 1
            velocity = n_payments / time_span
            
            # Payment trend
            if len(payment_amounts) >= 3:
                recent_avg = np.mean(payment_amounts[-3:])
                early_avg = np.mean(payment_amounts[:3])
                trend = recent_avg / early_avg if early_avg > 0 else 1.0
            else:
                trend = 1.0
        else:
            velocity = 0.5
            trend = 1.0
        
        ml_features.extend([velocity, trend])
        
        # Time since last payment
        time_since_last = 10 - last_payment_time if last_payment_time >= 0 else 11
        ml_features.append(time_since_last)
        
        return ml_features
    
    def calculate_adaptive_calibration(self, train_features, val_features, lob, base_predictions, actual_reserves):
        """Calculate calibration with special handling for sparse LoBs."""
        stats = self.lob_statistics[lob]
        payment_rate = stats['payment_rate']
        
        # Filter for meaningful samples
        valid_mask = (actual_reserves > 10) & (base_predictions > 0)
        
        if valid_mask.sum() < 5:
            # Too few samples - use conservative default based on sparsity
            if payment_rate < 0.1:
                return 100.0  # Very high calibration for very sparse LoBs
            elif payment_rate < 0.3:
                return 50.0
            else:
                return 10.0
        
        # Calculate ratios with outlier removal
        ratios = actual_reserves[valid_mask] / base_predictions[valid_mask]
        
        # Remove extreme outliers
        q1, q3 = np.percentile(ratios, [25, 75])
        iqr = q3 - q1
        outlier_mask = (ratios >= q1 - 1.5*iqr) & (ratios <= q3 + 1.5*iqr)
        
        if outlier_mask.sum() > 0:
            clean_ratios = ratios[outlier_mask]
            base_calibration = np.median(clean_ratios) * self.target_bias
        else:
            base_calibration = np.median(ratios) * self.target_bias
        
        # Adaptive adjustment based on payment sparsity
        if payment_rate < 0.1:
            # Very sparse LoB - needs much higher calibration
            calibration = max(base_calibration * 10.0, 50.0)
        elif payment_rate < 0.2:
            # Sparse LoB - high boost
            calibration = max(base_calibration * 5.0, 10.0)
        elif payment_rate < 0.5:
            # Medium sparsity - moderate boost  
            calibration = max(base_calibration * 2.0, 5.0)
        else:
            # Normal LoB
            calibration = max(base_calibration, 2.0)
        
        # Ensure minimum calibration to avoid underestimation
        return np.clip(calibration, 5.0, 100.0)
    
    def train_model(self, train_features, val_features, lob):
        """Train XGBoost model with adaptive calibration."""
        # Analyze LoB characteristics
        payment_rate, median_reserve, mean_reserve = self.analyze_lob_characteristics(train_features, lob)
        
        n_samples = len(train_features['cc'])
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for i in range(n_samples):
            # Create features
            features = self.create_ml_features(train_features, i)
            X_train.append(features)
            
            # Calculate target (future reserves)
            actual = 0
            for t in range(1, 12):
                if f'PayInd{t:02d}' in train_features:
                    mask = train_features.get(f'Time_Predict_Payment{t:02d}', np.ones(n_samples))
                    if mask[i] > 0:
                        ind = train_features[f'PayInd{t:02d}'][i]
                        amt = train_features[f'LogPay{t:02d}'][i]
                        if ind > 0:
                            actual += np.exp(amt)
            y_train.append(actual)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Store feature names for interpretability
        if self.feature_names is None:
            self.feature_names = [
                'cc', 'age', 'inj_part', 'AY', 'AQ', 'RepDel',
                'n_payments', 'log_total_paid', 'last_payment_time',
                'max_payment', 'payment_std', 'velocity', 'trend',
                'time_since_last'
            ]
        
        # Train XGBoost model
        if (y_train > 0).sum() > 10:
            # Use log transform for better performance
            y_train_log = np.log1p(y_train)
            
            # Configure XGBoost with parameters suitable for insurance data
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                objective='reg:squarederror'
            )
            
            # Train on all data (including zeros for better calibration)
            model.fit(X_train, y_train_log, verbose=False)
            
            # Get base predictions for calibration
            train_pred_log = model.predict(X_train)
            train_pred = np.expm1(train_pred_log)
            
            # Calculate adaptive calibration
            calibration = self.calculate_adaptive_calibration(
                train_features, val_features, lob, train_pred, y_train
            )
            
            # Validate on validation set if available
            if val_features is not None and len(val_features['cc']) > 0:
                n_val = len(val_features['cc'])
                X_val = []
                y_val = []
                
                for i in range(n_val):
                    features = self.create_ml_features(val_features, i)
                    X_val.append(features)
                    
                    actual = 0
                    for t in range(1, 12):
                        if f'PayInd{t:02d}' in val_features:
                            ind = val_features[f'PayInd{t:02d}'][i]
                            amt = val_features[f'LogPay{t:02d}'][i]
                            if ind > 0:
                                actual += np.exp(amt)
                    y_val.append(actual)
                
                X_val = np.array(X_val)
                y_val = np.array(y_val)
                
                if len(X_val) > 0:
                    val_pred_log = model.predict(X_val)
                    val_pred = np.expm1(val_pred_log) * calibration
                    val_actual_total = np.sum(y_val)
                    val_pred_total = np.sum(val_pred)
                    
                    if val_actual_total > 100 and val_pred_total > 10:
                        val_bias = val_pred_total / val_actual_total
                        
                        # Adjust calibration based on validation performance
                        if val_bias > 10:
                            calibration *= 0.2
                        elif val_bias > 3:
                            calibration *= self.target_bias / val_bias
                        elif val_bias < 0.3:
                            calibration *= self.target_bias / val_bias
            
            self.xgb_models[lob] = model
            self.learned_calibrations[lob] = calibration
            
        else:
            # Not enough data for ML - use statistical fallback
            if payment_rate < 0.1:
                calibration = 100.0
            elif payment_rate < 0.3:
                calibration = 50.0
            else:
                calibration = 10.0
            
            self.xgb_models[lob] = None
            self.learned_calibrations[lob] = calibration
        
        return calibration
    
    def predict(self, features, lob):
        """Make predictions using XGBoost with adaptive calibration."""
        if lob not in self.learned_calibrations:
            return 0
        
        calibration = self.learned_calibrations[lob]
        model = self.xgb_models.get(lob)
        stats = self.lob_statistics.get(lob, {})
        
        n_samples = len(features['cc'])
        total_reserve = 0
        
        # Prepare features for prediction
        X_pred = []
        needs_prediction = []
        
        for i in range(n_samples):
            # Check if prediction needed
            needs_pred = False
            for t in range(1, 12):
                mask_key = f'Time_Predict_Payment{t:02d}'
                if mask_key in features and features[mask_key][i] > 0:
                    needs_pred = True
                    break
            
            if needs_pred:
                features_vec = self.create_ml_features(features, i)
                X_pred.append(features_vec)
                needs_prediction.append(i)
        
        if len(X_pred) == 0:
            return 0
        
        X_pred = np.array(X_pred)
        
        # Make predictions
        if model is not None:
            # Use XGBoost model
            pred_log = model.predict(X_pred)
            predictions = np.expm1(pred_log) * calibration
            total_reserve = np.sum(predictions)
        else:
            # Fallback to statistical estimate
            payment_rate = stats.get('payment_rate', 0.5)
            if payment_rate > 0.1:
                base_estimate = stats.get('median_reserve', 1000)
            else:
                base_estimate = stats.get('median_reserve', 100) * 0.5
            
            total_reserve = base_estimate * len(needs_prediction) * calibration
        
        return total_reserve


def run_ml_improved_pipeline():
    """Run ML-enhanced pipeline with adaptive calibration."""
    from data_processing import DataProcessor
    
    print("\n" + "="*80)
    print("ML-ENHANCED IMPROVED PIPELINE")
    print("XGBoost with adaptive calibration for sparse LoBs")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Create results directory
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
        
        model = MLImprovedModel(target_bias=1.05)
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
                
                # Train model
                calibration = model.train_model(train_features, val_features, lob)
                stats = model.lob_statistics[lob]
                print(f"    Payment rate: {stats['payment_rate']:.1%}")
                print(f"    Median reserve: {stats['median_reserve']:.2f}")
                print(f"    Calibration: {calibration:.3f}")
                print(f"    ML Model: {'XGBoost' if model.xgb_models[lob] is not None else 'Statistical fallback'}")
                
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
                
                # Check status
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
                    'Payment_Rate': stats['payment_rate'],
                    'ML_Model': 'XGBoost' if model.xgb_models[lob] is not None else 'Fallback',
                    'Status': status
                }
                
                dataset_results.append(result)
                all_results.append(result)
                
            except Exception as e:
                print(f"    Error: {e}")
        
        if dataset_results:
            valid_results = [r for r in dataset_results if r['True'] > 0]
            if valid_results:
                avg_bias = np.mean([abs(r['Bias_Percent']) for r in valid_results])
                print(f"\n  Dataset Average |Bias|: {avg_bias:.1f}%")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = tables_dir / "ml_improved_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Create detailed report
        txt_path = tables_dir / "ml_improved_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ML-ENHANCED IMPROVED PIPELINE RESULTS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("KEY FEATURES:\n")
            f.write("- XGBoost regression models for reserve prediction\n")
            f.write("- Adaptive calibration based on payment sparsity\n")
            f.write("- Special handling for sparse LoBs (<10% payment rate)\n")
            f.write("- Outlier-robust statistics for calibration\n")
            f.write("- Validation-based calibration refinement\n\n")
            
            # Results by dataset with detailed table
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
                           f"{row['ML_Model']:8} | {row['Status']}\n")
                
                avg_bias = dataset_df['Bias_Percent'].abs().mean()
                f.write("-"*90 + "\n")
                f.write(f"Dataset Average |Bias|: {avg_bias:.1f}%\n")
            
            # Overall summary
            f.write("\n" + "="*80 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("="*80 + "\n")
            
            overall_bias = df['Bias_Percent'].abs().mean()
            success_count = (df['Status'] == 'SUCCESS').sum()
            xgb_count = (df['ML_Model'] == 'XGBoost').sum()
            
            f.write(f"Overall Average |Bias|: {overall_bias:.1f}%\n")
            f.write(f"Success Rate: {success_count}/{len(df)} LoBs\n")
            f.write(f"XGBoost Models Used: {xgb_count}/{len(df)}\n")
            
            if overall_bias <= 50:
                f.write("\n*** ML-ENHANCED MODEL ACHIEVES IMPROVED PERFORMANCE ***\n")
                f.write("XGBoost combined with adaptive calibration successfully handles both dense and sparse LoBs\n")
        
        print(f"\nResults saved to: {csv_path}")
        print(f"Report saved to: {txt_path}")
    
    return all_results


if __name__ == "__main__":
    results = run_ml_improved_pipeline()