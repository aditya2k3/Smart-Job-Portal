from flask import Blueprint, render_template, jsonify, request
from model_performance import ModelPerformancePredictor
import json
from datetime import datetime

# Create blueprint for ML dashboard
ml_dashboard = Blueprint('ml_dashboard', __name__, 
                         template_folder='templates',
                         url_prefix='/ml-dashboard')

# Initialize predictor
predictor = ModelPerformancePredictor()

@ml_dashboard.route('/')
def dashboard_home():
    """Main ML performance dashboard"""
    return render_template('ml_dashboard.html')

@ml_dashboard.route('/api/performance-data')
def get_performance_data():
    """Get comprehensive performance data for dashboard"""
    try:
        dashboard_data = predictor.get_dashboard_data()
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ml_dashboard.route('/api/model/<model_name>/report')
def get_model_report(model_name):
    """Get detailed performance report for specific model"""
    try:
        if model_name not in predictor.model_configs:
            return jsonify({'error': 'Model not found'}), 404
        
        report = predictor.generate_performance_report(model_name)
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ml_dashboard.route('/api/model/<model_name>/train')
def simulate_training(model_name):
    """Simulate model training process"""
    try:
        epochs = request.args.get('epochs', 100, type=int)
        training_data = predictor.simulate_model_training(model_name, epochs)
        return jsonify(training_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ml_dashboard.route('/api/model/<model_name>/predict')
def predict_performance(model_name):
    """Predict future performance for model"""
    try:
        days = request.args.get('days', 30, type=int)
        predictions = predictor.predict_future_performance(model_name, days)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ml_dashboard.route('/api/model/<model_name>/features')
def get_feature_importance(model_name):
    """Get feature importance analysis for model"""
    try:
        feature_analysis = predictor.analyze_feature_importance(model_name)
        return jsonify(feature_analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ml_dashboard.route('/api/system/health')
def system_health():
    """Get overall system health metrics"""
    try:
        system_metrics = predictor._get_system_metrics()
        return jsonify(system_metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
