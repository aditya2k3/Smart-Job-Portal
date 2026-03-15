import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import statistics

class ModelPerformancePredictor:
    """Advanced ML model performance prediction and monitoring system"""
    
    def __init__(self):
        self.metrics_history = []
        self.model_configs = {
            'resume_matcher': {
                'name': 'Resume-Job Matcher',
                'type': 'Classification',
                'target_metric': 'accuracy',
                'current_performance': 0.85,
                'features': ['skills', 'experience', 'education', 'keywords']
            },
            'salary_predictor': {
                'name': 'Salary Predictor',
                'type': 'Regression',
                'target_metric': 'r2_score',
                'current_performance': 0.78,
                'features': ['experience', 'skills', 'location', 'company_size']
            },
            'skill_recommender': {
                'name': 'Skill Recommender',
                'type': 'Recommendation',
                'target_metric': 'precision_at_k',
                'current_performance': 0.72,
                'features': ['current_skills', 'job_market', 'trends']
            }
        }
        
    def simulate_model_training(self, model_name: str, epochs: int = 100) -> Dict:
        """Simulate model training with performance metrics"""
        config = self.model_configs[model_name]
        
        # Generate realistic training curves
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []
        
        base_loss = 2.0
        base_metric = config['current_performance']
        
        for epoch in range(epochs):
            # Training loss decreases with some noise
            train_loss = base_loss * (0.95 ** (epoch / 10)) + random.uniform(-0.05, 0.05)
            train_losses.append(max(0.1, train_loss))
            
            # Validation loss with overfitting simulation
            if epoch < epochs * 0.7:
                val_loss = base_loss * (0.93 ** (epoch / 10)) + random.uniform(-0.08, 0.08)
            else:
                # Simulate overfitting
                val_loss = train_loss * (1 + (epoch - epochs * 0.7) / (epochs * 0.3) * 0.3)
            val_losses.append(max(0.1, val_loss))
            
            # Metrics improve then plateau
            if epoch < epochs * 0.6:
                train_metric = base_metric * (1 - 0.9 ** (epoch / 20)) + random.uniform(-0.02, 0.02)
                val_metric = base_metric * (1 - 0.85 ** (epoch / 20)) + random.uniform(-0.03, 0.03)
            else:
                # Plateau with slight degradation
                train_metric = base_metric * 0.98 + random.uniform(-0.01, 0.01)
                val_metric = base_metric * 0.95 + random.uniform(-0.02, 0.02)
            
            train_metrics.append(min(1.0, max(0.0, train_metric)))
            val_metrics.append(min(1.0, max(0.0, val_metric)))
        
        return {
            'model_name': model_name,
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'final_train_metric': train_metrics[-1],
            'final_val_metric': val_metrics[-1],
            'overfitting_detected': val_metrics[-1] < train_metrics[-1] * 0.9
        }
    
    def predict_future_performance(self, model_name: str, days_ahead: int = 30) -> Dict:
        """Predict model performance over time with various scenarios"""
        config = self.model_configs[model_name]
        current_perf = config['current_performance']
        
        # Generate different scenarios
        scenarios = {
            'optimistic': [],
            'realistic': [],
            'pessimistic': []
        }
        
        base_drift = random.uniform(-0.002, 0.001)  # Daily performance drift
        
        for day in range(days_ahead):
            # Optimistic scenario: improvement with new data
            opt_improvement = 0.001 * (1 - day / days_ahead)  # Diminishing returns
            scenarios['optimistic'].append(
                min(1.0, current_perf + opt_improvement * day + random.uniform(0, 0.002))
            )
            
            # Realistic scenario: slight improvement then plateau
            if day < 15:
                real_improvement = 0.0005 * (1 - day / 15)
            else:
                real_improvement = -0.0001  # Slight degradation
            scenarios['realistic'].append(
                max(0.5, current_perf + real_improvement * day + random.uniform(-0.001, 0.001))
            )
            
            # Pessimistic scenario: data drift and degradation
            pessimistic_drift = base_drift * day - 0.0003 * day
            scenarios['pessimistic'].append(
                max(0.4, current_perf + pessimistic_drift + random.uniform(-0.002, 0))
            )
        
        return {
            'model_name': model_name,
            'days_ahead': days_ahead,
            'current_performance': current_perf,
            'scenarios': scenarios,
            'confidence_intervals': self._calculate_confidence_intervals(scenarios)
        }
    
    def _calculate_confidence_intervals(self, scenarios: Dict) -> Dict:
        """Calculate confidence intervals for predictions"""
        intervals = {}
        for scenario, values in scenarios.items():
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0
            intervals[scenario] = {
                'mean': mean,
                'std': stdev,
                'upper_95': mean + 1.96 * stdev,
                'lower_95': mean - 1.96 * stdev
            }
        return intervals
    
    def analyze_feature_importance(self, model_name: str) -> Dict:
        """Analyze and simulate feature importance for models"""
        config = self.model_configs[model_name]
        features = config['features']
        
        # Generate realistic feature importance scores
        importance_scores = []
        remaining_weight = 1.0
        
        for i, feature in enumerate(features):
            if i == len(features) - 1:
                # Last feature gets remaining weight
                score = remaining_weight
            else:
                # Random distribution favoring first features
                score = random.uniform(0.1, remaining_weight * 0.6)
                remaining_weight -= score
            
            importance_scores.append({
                'feature': feature,
                'importance': score,
                'impact': self._calculate_feature_impact(score)
            })
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'model_name': model_name,
            'feature_importance': importance_scores,
            'top_features': importance_scores[:3],
            'weak_features': importance_scores[-2:]
        }
    
    def _calculate_feature_impact(self, importance: float) -> str:
        """Categorize feature impact based on importance score"""
        if importance > 0.4:
            return 'critical'
        elif importance > 0.2:
            return 'high'
        elif importance > 0.1:
            return 'medium'
        else:
            return 'low'
    
    def generate_performance_report(self, model_name: str) -> Dict:
        """Generate comprehensive performance report for a model"""
        config = self.model_configs[model_name]
        
        # Get training simulation
        training_data = self.simulate_model_training(model_name)
        
        # Get future predictions
        future_predictions = self.predict_future_performance(model_name)
        
        # Get feature importance
        feature_analysis = self.analyze_feature_importance(model_name)
        
        # Calculate health metrics
        health_score = self._calculate_model_health(training_data, future_predictions)
        
        return {
            'model_name': model_name,
            'model_type': config['type'],
            'target_metric': config['target_metric'],
            'current_performance': config['current_performance'],
            'health_score': health_score,
            'training_analysis': training_data,
            'future_predictions': future_predictions,
            'feature_analysis': feature_analysis,
            'recommendations': self._generate_recommendations(health_score, training_data),
            'generated_at': datetime.now().isoformat()
        }
    
    def _calculate_model_health(self, training_data: Dict, predictions: Dict) -> Dict:
        """Calculate overall model health score"""
        val_metric = training_data['final_val_metric']
        overfitting = training_data['overfitting_detected']
        
        # Base health from validation performance
        base_health = val_metric * 100
        
        # Penalty for overfitting
        if overfitting:
            base_health -= 15
        
        # Future performance trend
        realistic_trend = predictions['scenarios']['realistic']
        trend_score = (realistic_trend[-1] - realistic_trend[0]) * 100
        base_health += trend_score * 10
        
        # Categorize health
        if base_health >= 85:
            status = 'excellent'
            color = '#10b981'
        elif base_health >= 70:
            status = 'good'
            color = '#3b82f6'
        elif base_health >= 55:
            status = 'fair'
            color = '#f59e0b'
        else:
            status = 'poor'
            color = '#ef4444'
        
        return {
            'score': max(0, min(100, base_health)),
            'status': status,
            'color': color,
            'overfitting_detected': overfitting,
            'trend_direction': 'improving' if trend_score > 0 else 'declining'
        }
    
    def _generate_recommendations(self, health_score: Dict, training_data: Dict) -> List[str]:
        """Generate actionable recommendations based on model performance"""
        recommendations = []
        
        if health_score['score'] < 70:
            recommendations.append("Consider retraining with more diverse data")
        
        if training_data['overfitting_detected']:
            recommendations.append("Implement regularization techniques")
            recommendations.append("Increase training data size")
        
        if health_score['trend_direction'] == 'declining':
            recommendations.append("Monitor for data drift")
            recommendations.append("Update model with recent data")
        
        if health_score['score'] < 60:
            recommendations.append("Review feature engineering")
            recommendations.append("Consider model architecture changes")
        
        if not recommendations:
            recommendations.append("Model is performing well - continue monitoring")
        
        return recommendations
    
    def get_dashboard_data(self) -> Dict:
        """Get comprehensive data for performance dashboard"""
        dashboard_data = {
            'summary': {
                'total_models': len(self.model_configs),
                'healthy_models': 0,
                'models_needing_attention': 0,
                'average_performance': 0
            },
            'models': {},
            'system_metrics': self._get_system_metrics(),
            'alerts': []
        }
        
        total_performance = 0
        
        for model_name in self.model_configs:
            report = self.generate_performance_report(model_name)
            dashboard_data['models'][model_name] = report
            
            total_performance += report['current_performance']
            
            if report['health_score']['score'] >= 70:
                dashboard_data['summary']['healthy_models'] += 1
            else:
                dashboard_data['summary']['models_needing_attention'] += 1
                
            # Generate alerts for problematic models
            if report['health_score']['score'] < 60:
                dashboard_data['alerts'].append({
                    'model': model_name,
                    'severity': 'high',
                    'message': f"{model_name} performance is below acceptable levels"
                })
            elif report['training_analysis']['overfitting_detected']:
                dashboard_data['alerts'].append({
                    'model': model_name,
                    'severity': 'medium',
                    'message': f"{model_name} shows signs of overfitting"
                })
        
        dashboard_data['summary']['average_performance'] = total_performance / len(self.model_configs)
        
        return dashboard_data
    
    def _get_system_metrics(self) -> Dict:
        """Get system-wide performance metrics"""
        return {
            'total_predictions_today': random.randint(1000, 5000),
            'average_response_time': random.uniform(50, 200),  # ms
            'system_uptime': '99.9%',
            'data_volume_processed': f"{random.uniform(1.5, 5.2):.1f} GB",
            'active_models': len(self.model_configs),
            'last_training': (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat()
        }
