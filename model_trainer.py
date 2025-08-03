"""
Self-improving model trainer for Palisade price prediction
"""

import pandas as pd
import numpy as np
from database import PalisadeDatabase
from datetime import datetime
import json

class PalisadeModelTrainer:
    """Handles model retraining with new data submissions"""
    
    def __init__(self, db_path="palisade_data.db"):
        self.db = PalisadeDatabase(db_path)
        
    def prepare_training_data(self, min_records=10):
        """Prepare data for model training"""
        df = self.db.get_training_data()
        
        if len(df) < min_records:
            return None, f"Insufficient data for training. Need at least {min_records} records, have {len(df)}"
        
        # Create one-hot encoded features (same format as original model)
        encoded_df = pd.get_dummies(df, columns=['Trim', 'Drivetrain', 'ExtColor', 'IntColor'], prefix=['Trim', 'Drivetrain', 'ExtColor', 'IntColor'])
        
        # Ensure all expected columns exist (fill missing with 0)
        expected_columns = [
            'Trim_Calligraphy','Trim_Limited','Trim_SEL','Trim_SEL Convenience',
            'Drivetrain_AWD','Drivetrain_FWD','City_mpg',
            'ExtColor_Abyss Black','ExtColor_Classy Blue','ExtColor_Creamy White','ExtColor_Ecotronic Gray',
            'IntColor_Black','IntColor_Brown','IntColor_Gray','IntColor_Gray/Navy','IntColor_Navy/Brown'
        ]
        
        for col in expected_columns:
            if col not in encoded_df.columns:
                encoded_df[col] = 0
        
        # Select features in correct order
        X = encoded_df[expected_columns]
        y = encoded_df['Price']
        
        return X, y
    
    def calculate_model_coefficients(self, X, y):
        """Calculate new model coefficients from training data"""
        
        # Group data by features to calculate base prices and adjustments
        df_combined = X.copy()
        df_combined['Price'] = y
        
        # Calculate trim base prices
        trim_prices = {}
        for trim in ['SEL', 'SEL Convenience', 'Limited', 'Calligraphy']:
            trim_col = f'Trim_{trim}'
            if trim_col in df_combined.columns:
                trim_data = df_combined[df_combined[trim_col] == 1]['Price']
                if len(trim_data) > 0:
                    trim_prices[trim] = trim_data.mean()
                else:
                    # Use previous known values as fallback
                    fallback_prices = {'SEL': 44520, 'SEL Convenience': 47495, 'Limited': 53345, 'Calligraphy': 57837}
                    trim_prices[trim] = fallback_prices.get(trim, 45000)
        
        # Calculate drivetrain effect
        awd_prices = df_combined[df_combined['Drivetrain_AWD'] == 1]['Price']
        fwd_prices = df_combined[df_combined['Drivetrain_FWD'] == 1]['Price']
        
        if len(awd_prices) > 0 and len(fwd_prices) > 0:
            fwd_premium = fwd_prices.mean() - awd_prices.mean()
        else:
            fwd_premium = 1346  # Fallback from original data
        
        # Calculate color effects
        ext_color_adjustments = {}
        int_color_adjustments = {}
        
        # Exterior colors
        for color in ['Abyss Black', 'Classy Blue', 'Creamy White', 'Ecotronic Gray']:
            col_name = f'ExtColor_{color}'
            if col_name in df_combined.columns:
                color_data = df_combined[df_combined[col_name] == 1]['Price']
                if len(color_data) > 0:
                    # Calculate relative to Abyss Black baseline
                    baseline_data = df_combined[df_combined['ExtColor_Abyss Black'] == 1]['Price']
                    baseline_price = baseline_data.mean() if len(baseline_data) > 0 else trim_prices.get('SEL', 45000)
                    ext_color_adjustments[color] = color_data.mean() - baseline_price
                else:
                    ext_color_adjustments[color] = 0
        
        # Interior colors
        for color in ['Black', 'Brown', 'Gray', 'Gray/Navy', 'Navy/Brown']:
            col_name = f'IntColor_{color}'
            if col_name in df_combined.columns:
                color_data = df_combined[df_combined[col_name] == 1]['Price']
                if len(color_data) > 0:
                    # Calculate relative to Black baseline
                    baseline_data = df_combined[df_combined['IntColor_Black'] == 1]['Price']
                    baseline_price = baseline_data.mean() if len(baseline_data) > 0 else trim_prices.get('SEL', 45000)
                    int_color_adjustments[color] = color_data.mean() - baseline_price
                else:
                    int_color_adjustments[color] = 0
        
        # Calculate MPG effect
        mpg_effect = 0
        if 'City_mpg' in df_combined.columns:
            # Simple correlation between MPG and price
            mpg_corr = df_combined[['City_mpg', 'Price']].corr().iloc[0, 1]
            if not np.isnan(mpg_corr):
                mpg_effect = mpg_corr * 1000  # Scale factor
        
        return {
            'trim_prices': trim_prices,
            'fwd_premium': fwd_premium,
            'ext_color_adjustments': ext_color_adjustments,
            'int_color_adjustments': int_color_adjustments,
            'mpg_effect': mpg_effect
        }
    
    def test_model_accuracy(self, X, y, coefficients):
        """Test model accuracy against training data"""
        
        def predict_with_coefficients(row, coeffs):
            # Determine trim
            trim = 'SEL'  # Default
            for t in ['Calligraphy', 'Limited', 'SEL Convenience', 'SEL']:
                if row.get(f'Trim_{t}', 0) == 1:
                    trim = t
                    break
            
            price = coeffs['trim_prices'].get(trim, 45000)
            
            # Add drivetrain premium
            if row.get('Drivetrain_FWD', 0) == 1:
                price += coeffs['fwd_premium']
            
            # Add color adjustments
            for color, adj in coeffs['ext_color_adjustments'].items():
                if row.get(f'ExtColor_{color}', 0) == 1:
                    price += adj * 0.5  # Scale down
                    break
            
            for color, adj in coeffs['int_color_adjustments'].items():
                if row.get(f'IntColor_{color}', 0) == 1:
                    price += adj * 0.3  # Scale down
                    break
            
            # MPG effect
            city_mpg = row.get('City_mpg', 19)
            if city_mpg < 19:
                price += (19 - city_mpg) * abs(coeffs['mpg_effect'] * 0.5)
            elif city_mpg > 19:
                price -= (city_mpg - 19) * abs(coeffs['mpg_effect'] * 0.25)
            
            return max(min(price, 65000), 40000)  # Bounds
        
        # Calculate predictions
        predictions = []
        for _, row in X.iterrows():
            pred = predict_with_coefficients(row, coefficients)
            predictions.append(pred)
        
        # Calculate accuracy metrics
        predictions = np.array(predictions)
        actual = y.values
        
        differences = np.abs(actual - predictions)
        percent_errors = (differences / actual) * 100
        
        avg_error = np.mean(differences)
        avg_percent_error = np.mean(percent_errors)
        max_error = np.max(differences)
        
        return {
            'avg_error': avg_error,
            'avg_percent_error': avg_percent_error,
            'max_error': max_error,
            'predictions': predictions,
            'actual': actual
        }
    
    def retrain_model(self, min_improvement_pct=1.0):
        """Retrain model if enough new data and improvement threshold met"""
        
        # Get current data
        X, y = self.prepare_training_data()
        if X is None:
            return False, y  # Error message
        
        # Calculate new coefficients
        new_coefficients = self.calculate_model_coefficients(X, y)
        
        # Test accuracy
        accuracy = self.test_model_accuracy(X, y, new_coefficients)
        
        # Check if model should be updated (simple threshold-based)
        data_count = len(X)
        avg_error_pct = accuracy['avg_percent_error']
        
        # Update model if error is reasonable
        if avg_error_pct < 10.0:  # Accept models with < 10% average error
            
            # Save new model coefficients
            model_config = {
                'coefficients': new_coefficients,
                'accuracy': accuracy,
                'training_date': datetime.now().isoformat(),
                'data_count': data_count
            }
            
            # Save to file
            with open('enhanced_model_config.json', 'w') as f:
                json.dump(model_config, f, indent=2, default=str)
            
            # Log training history
            conn = self.db.db_path
            import sqlite3
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_history (data_count, avg_error, notes)
                VALUES (?, ?, ?)
            ''', (data_count, avg_error_pct, f"Retrained with {data_count} records"))
            conn.commit()
            conn.close()
            
            return True, f"Model retrained successfully! Average error: {avg_error_pct:.1f}% with {data_count} records"
        else:
            return False, f"Model accuracy insufficient ({avg_error_pct:.1f}% error). Keeping current model."
    
    def should_retrain(self, new_submissions_threshold=3):
        """Check if model should be retrained based on new submissions"""
        stats = self.db.get_data_stats()
        
        # Simple heuristic: retrain if we have new submissions
        if stats['recent_submissions'] >= new_submissions_threshold:
            return True, f"{stats['recent_submissions']} new submissions warrant retraining"
        
        return False, "No retraining needed"

def load_enhanced_model():
    """Load the enhanced model configuration"""
    try:
        with open('enhanced_model_config.json', 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    # Test the trainer
    trainer = PalisadeModelTrainer()
    should_retrain, message = trainer.should_retrain()
    print(f"Should retrain: {should_retrain} - {message}")
    
    if should_retrain:
        success, result = trainer.retrain_model()
        print(f"Retraining result: {result}")