"""
Model performance analytics and improvement tracking
"""

import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os

class ModelAnalytics:
    """Track and analyze model improvement over time"""
    
    def __init__(self, db_path="palisade_data.db"):
        self.db_path = db_path
        
    def get_model_history(self):
        """Get model training history with performance metrics"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM model_history 
            ORDER BY training_date ASC
        ''', conn)
        conn.close()
        return df
    
    def get_data_growth_over_time(self):
        """Track how training data has grown over time"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT 
                DATE(submission_date) as date,
                COUNT(*) as daily_submissions,
                SUM(COUNT(*)) OVER (ORDER BY DATE(submission_date)) as cumulative_records,
                source
            FROM vehicle_data 
            GROUP BY DATE(submission_date), source
            ORDER BY submission_date ASC
        ''', conn)
        conn.close()
        return df
    
    def analyze_prediction_accuracy(self, test_size=0.2):
        """Analyze current model accuracy using cross-validation approach"""
        from database import PalisadeDatabase
        from model_trainer import PalisadeModelTrainer
        
        db = PalisadeDatabase(self.db_path)
        trainer = PalisadeModelTrainer(self.db_path)
        
        # Get all training data
        X, y = trainer.prepare_training_data()
        if X is None:
            return None
        
        # Split into train/test (chronological split to avoid data leakage)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if len(X_test) == 0:
            return None
        
        # Calculate coefficients on training data only
        df_train = X_train.copy()
        df_train['Price'] = y_train
        coefficients = trainer.calculate_model_coefficients(X_train, y_train)
        
        # Test on held-out data
        accuracy = trainer.test_model_accuracy(X_test, y_test, coefficients)
        
        return {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': accuracy,
            'coefficients': coefficients
        }
    
    def get_data_diversity_metrics(self):
        """Analyze diversity of training data"""
        conn = sqlite3.connect(self.db_path)
        
        # Get feature distributions
        trim_dist = pd.read_sql_query('SELECT trim, COUNT(*) as count FROM vehicle_data GROUP BY trim', conn)
        drivetrain_dist = pd.read_sql_query('SELECT drivetrain, COUNT(*) as count FROM vehicle_data GROUP BY drivetrain', conn)
        ext_color_dist = pd.read_sql_query('SELECT ext_color, COUNT(*) as count FROM vehicle_data GROUP BY ext_color', conn)
        int_color_dist = pd.read_sql_query('SELECT int_color, COUNT(*) as count FROM vehicle_data GROUP BY int_color', conn)
        
        # Price distribution
        price_stats = pd.read_sql_query('''
            SELECT 
                MIN(price) as min_price,
                MAX(price) as max_price,
                AVG(price) as avg_price,
                COUNT(DISTINCT ROUND(price/1000)*1000) as price_buckets,
                COUNT(*) as total_records
            FROM vehicle_data
        ''', conn)
        
        conn.close()
        
        return {
            'trim_distribution': trim_dist,
            'drivetrain_distribution': drivetrain_dist,
            'ext_color_distribution': ext_color_dist,
            'int_color_distribution': int_color_dist,
            'price_statistics': price_stats
        }
    
    def calculate_diversity_score(self):
        """Calculate a diversity score for the dataset"""
        diversity_metrics = self.get_data_diversity_metrics()
        
        # Calculate entropy for categorical features
        def entropy(counts):
            proportions = counts / counts.sum()
            return -np.sum(proportions * np.log2(proportions + 1e-10))
        
        trim_entropy = entropy(diversity_metrics['trim_distribution']['count'].values)
        drivetrain_entropy = entropy(diversity_metrics['drivetrain_distribution']['count'].values)
        ext_color_entropy = entropy(diversity_metrics['ext_color_distribution']['count'].values)
        int_color_entropy = entropy(diversity_metrics['int_color_distribution']['count'].values)
        
        # Maximum possible entropies (for normalization)
        max_trim_entropy = np.log2(4)  # 4 trim levels
        max_drivetrain_entropy = np.log2(2)  # 2 drivetrains
        max_ext_color_entropy = np.log2(4)  # 4 exterior colors
        max_int_color_entropy = np.log2(5)  # 5 interior colors
        
        # Normalize and average
        diversity_score = np.mean([
            trim_entropy / max_trim_entropy,
            drivetrain_entropy / max_drivetrain_entropy,
            ext_color_entropy / max_ext_color_entropy,
            int_color_entropy / max_int_color_entropy
        ])
        
        return {
            'overall_diversity_score': diversity_score,
            'trim_diversity': trim_entropy / max_trim_entropy,
            'drivetrain_diversity': drivetrain_entropy / max_drivetrain_entropy,
            'ext_color_diversity': ext_color_entropy / max_ext_color_entropy,
            'int_color_diversity': int_color_entropy / max_int_color_entropy
        }
    
    def create_improvement_dashboard(self):
        """Create visualizations for model improvement tracking"""
        
        # Model history
        history_df = self.get_model_history()
        
        # Data growth
        growth_df = self.get_data_growth_over_time()
        
        # Current accuracy
        accuracy_analysis = self.analyze_prediction_accuracy()
        
        # Diversity metrics
        diversity_score = self.calculate_diversity_score()
        diversity_metrics = self.get_data_diversity_metrics()
        
        charts = {}
        
        # 1. Model Accuracy Over Time
        if len(history_df) > 0:
            fig_accuracy = go.Figure()
            fig_accuracy.add_trace(go.Scatter(
                x=pd.to_datetime(history_df['training_date']),
                y=history_df['avg_error'],
                mode='lines+markers',
                name='Model Error %',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            fig_accuracy.update_layout(
                title="Model Accuracy Improvement Over Time",
                xaxis_title="Date",
                yaxis_title="Average Error %",
                height=400
            )
            charts['accuracy_trend'] = fig_accuracy
        
        # 2. Data Growth Over Time
        if len(growth_df) > 0:
            fig_growth = go.Figure()
            fig_growth.add_trace(go.Scatter(
                x=pd.to_datetime(growth_df['date']),
                y=growth_df['cumulative_records'],
                mode='lines+markers',
                name='Total Records',
                fill='tonexty'
            ))
            fig_growth.update_layout(
                title="Training Data Growth Over Time",
                xaxis_title="Date",
                yaxis_title="Cumulative Records",
                height=400
            )
            charts['data_growth'] = fig_growth
        
        # 3. Feature Diversity
        fig_diversity = go.Figure()
        
        # Trim distribution
        trim_data = diversity_metrics['trim_distribution']
        fig_diversity.add_trace(go.Bar(
            x=trim_data['trim'],
            y=trim_data['count'],
            name='Trim Levels',
            marker_color='lightblue'
        ))
        
        fig_diversity.update_layout(
            title="Feature Distribution - Trim Levels",
            xaxis_title="Trim Level",
            yaxis_title="Count",
            height=400
        )
        charts['feature_diversity'] = fig_diversity
        
        # 4. Price Distribution
        conn = sqlite3.connect(self.db_path)
        price_data = pd.read_sql_query('SELECT price FROM vehicle_data', conn)
        conn.close()
        
        if len(price_data) > 0:
            fig_price = px.histogram(
                price_data, 
                x='price', 
                nbins=20,
                title="Price Distribution in Training Data"
            )
            fig_price.update_layout(height=400)
            charts['price_distribution'] = fig_price
        
        return {
            'charts': charts,
            'metrics': {
                'current_accuracy': accuracy_analysis,
                'diversity_score': diversity_score,
                'total_records': len(price_data) if len(price_data) > 0 else 0,
                'model_versions': len(history_df)
            }
        }
    
    def log_model_performance(self, data_count, avg_error, notes=""):
        """Log model performance for tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO model_history (data_count, avg_error, notes)
            VALUES (?, ?, ?)
        ''', (data_count, avg_error, notes))
        conn.commit()
        conn.close()
    
    def generate_improvement_report(self):
        """Generate a comprehensive improvement report"""
        dashboard = self.create_improvement_dashboard()
        metrics = dashboard['metrics']
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_records': metrics['total_records'],
                'model_versions': metrics['model_versions'],
                'diversity_score': metrics['diversity_score']['overall_diversity_score'],
                'current_accuracy': metrics['current_accuracy']['accuracy']['avg_percent_error'] if metrics['current_accuracy'] else None
            },
            'recommendations': []
        }
        
        # Generate recommendations based on analysis
        diversity = metrics['diversity_score']
        
        if diversity['trim_diversity'] < 0.7:
            report['recommendations'].append("Need more diverse trim levels - consider incentivizing SEL and Limited submissions")
        
        if diversity['ext_color_diversity'] < 0.7:
            report['recommendations'].append("Limited exterior color variety - need more submissions across all color options")
        
        if metrics['total_records'] < 25:
            report['recommendations'].append("Small dataset - accuracy will improve significantly with more submissions")
        
        if metrics['current_accuracy'] and metrics['current_accuracy']['accuracy']['avg_percent_error'] > 3.0:
            report['recommendations'].append("High prediction error - may need data quality review or model architecture changes")
        
        return report

if __name__ == "__main__":
    # Test the analytics
    analytics = ModelAnalytics()
    
    # Generate dashboard
    dashboard = analytics.create_improvement_dashboard()
    print("Dashboard generated with", len(dashboard['charts']), "charts")
    
    # Generate report
    report = analytics.generate_improvement_report()
    print("Improvement Report:")
    print(json.dumps(report, indent=2, default=str))