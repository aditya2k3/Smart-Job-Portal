# Machine Learning Performance Dashboard

Advanced ML model performance monitoring and prediction system for the Smart Job Portal.

## 🚀 Features

### Model Performance Monitoring
- **Real-time Performance Tracking**: Monitor accuracy, precision, recall, and other metrics
- **Health Score Assessment**: Comprehensive model health evaluation
- **Overfitting Detection**: Automatic detection of overfitting patterns
- **Performance Prediction**: Future performance forecasting with multiple scenarios

### Advanced Analytics
- **Training Visualization**: Real-time training curves and loss plots
- **Feature Importance Analysis**: Identify most influential features
- **Trend Analysis**: Performance trends over time
- **Comparative Analysis**: Compare multiple models side-by-side

### Interactive Dashboard
- **Live Updates**: Real-time dashboard with auto-refresh
- **Interactive Charts**: Detailed performance visualizations
- **Alert System**: Automated alerts for performance issues
- **Detailed Reports**: Comprehensive model analysis reports

## 📊 Available Models

### 1. Resume-Job Matcher
- **Type**: Classification
- **Target Metric**: Accuracy
- **Current Performance**: 85%
- **Features**: Skills, Experience, Education, Keywords

### 2. Salary Predictor
- **Type**: Regression
- **Target Metric**: R² Score
- **Current Performance**: 78%
- **Features**: Experience, Skills, Location, Company Size

### 3. Skill Recommender
- **Type**: Recommendation System
- **Target Metric**: Precision@K
- **Current Performance**: 72%
- **Features**: Current Skills, Job Market, Trends

## 🛠 Architecture

### Core Components

#### ModelPerformancePredictor (`model_performance.py`)
```python
class ModelPerformancePredictor:
    - simulate_model_training()
    - predict_future_performance()
    - analyze_feature_importance()
    - generate_performance_report()
    - get_dashboard_data()
```

#### Dashboard API (`dashboard.py`)
```python
# API Endpoints:
GET /ml-dashboard/api/performance-data
GET /ml-dashboard/api/model/{name}/report
GET /ml-dashboard/api/model/{name}/train
GET /ml-dashboard/api/model/{name}/predict
GET /ml-dashboard/api/model/{name}/features
GET /ml-dashboard/api/system/health
```

#### Frontend Dashboard (`templates/ml_dashboard.html`)
- Real-time performance monitoring
- Interactive charts with Chart.js
- Responsive design with TailwindCSS
- Modal-based detailed views

## 📈 Performance Metrics

### Health Score Calculation
The health score is calculated based on:
- **Base Performance**: Validation metric score (0-100)
- **Overfitting Penalty**: -15 points if overfitting detected
- **Future Trend**: ±10 points based on predicted performance trend

### Health Categories
- **Excellent** (85-100): Model performing optimally
- **Good** (70-84): Acceptable performance with minor issues
- **Fair** (55-69): Performance needs attention
- **Poor** (0-54): Immediate action required

### Prediction Scenarios
- **Optimistic**: Best-case scenario with continuous improvement
- **Realistic**: Expected performance with normal conditions
- **Pessimistic**: Worst-case scenario with data drift

## 🔧 Usage

### Access the Dashboard
1. Start the main application: `python app.py`
2. Navigate to: `http://localhost:5000/ml-dashboard`
3. View real-time performance metrics

### API Usage Examples

#### Get All Performance Data
```bash
curl http://localhost:5000/ml-dashboard/api/performance-data
```

#### Get Specific Model Report
```bash
curl http://localhost:5000/ml-dashboard/api/model/resume_matcher/report
```

#### Predict Future Performance
```bash
curl "http://localhost:5000/ml-dashboard/api/model/resume_matcher/predict?days=30"
```

### Integration with Main App
The ML dashboard is automatically integrated with the main application:
- Navigation link appears in the main header
- Shared authentication and session management
- Consistent styling and UI components

## 🎨 Dashboard Features

### System Overview
- Total models count
- Healthy vs. unhealthy models
- Average performance across all models
- Active alerts count

### Model Cards
- Current performance metrics
- Health score visualization
- Quick status indicators
- Detailed analysis links

### Performance Charts
- **Training Performance**: Loss curves over epochs
- **Future Predictions**: Multi-scenario forecasting
- **Feature Importance**: Top contributing features
- **Trend Analysis**: Performance over time

### Alert System
- High severity alerts for critical issues
- Medium severity for warnings
- Automated recommendations
- Real-time notifications

## 🔍 Model Analysis

### Training Simulation
The system simulates realistic training scenarios:
- Gradual loss reduction
- Validation overfitting detection
- Performance plateau identification
- Noise and variability simulation

### Feature Analysis
- Automatic feature importance ranking
- Impact categorization (critical, high, medium, low)
- Feature contribution visualization
- Weak feature identification

### Recommendations Engine
Generates actionable recommendations based on:
- Current health score
- Training patterns
- Performance trends
- Overfitting detection

## 📊 Data Visualization

### Chart Types
- **Line Charts**: Training curves and predictions
- **Bar Charts**: Feature importance
- **Gauge Charts**: Health scores
- **Progress Bars**: Performance metrics

### Color Coding
- **Green**: Excellent performance
- **Blue**: Good performance
- **Yellow**: Fair performance
- **Red**: Poor performance

## 🚀 Deployment

### Production Considerations
- **Database Integration**: Replace simulated data with real metrics
- **Authentication**: Add user authentication for dashboard access
- **Scaling**: Handle multiple concurrent users
- **Monitoring**: Add uptime and performance monitoring

### Customization
- Add new models to the `model_configs` dictionary
- Customize metrics and thresholds
- Modify visualization styles
- Extend API endpoints

## 🔮 Future Enhancements

### Planned Features
- **Real-time Training**: Connect to actual ML training pipelines
- **A/B Testing**: Compare model versions
- **Automated Retraining**: Trigger retraining based on performance
- **Export Reports**: PDF and CSV report generation
- **Mobile App**: Native mobile dashboard application

### Advanced Analytics
- **Model Explainability**: SHAP and LIME integration
- **Data Drift Detection**: Automated drift monitoring
- **Ensemble Methods**: Multiple model combination strategies
- **Hyperparameter Optimization**: Automated tuning suggestions

## 📞 Support

For technical support or questions:
- Check the main application README
- Review API documentation
- Examine browser console for frontend issues
- Check Flask logs for backend errors

---

**Monitor your ML models with confidence!** 🎯
