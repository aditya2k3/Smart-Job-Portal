"""
Real Machine Learning Model Trainer for Smart Job Portal
Trains actual ML models on job portal data with real algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RealMLTrainer:
    """Real Machine Learning Model Training for Job Portal"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.training_history = {}
        
    def generate_training_data(self, n_samples=1000):
        """Generate realistic training data for job matching"""
        np.random.seed(42)
        
        # Job categories and skills
        job_categories = ['Software Engineer', 'Data Scientist', 'Product Manager', 'DevOps Engineer', 'UX Designer', 'ML Engineer']
        skills_db = [
            'python', 'java', 'javascript', 'react', 'nodejs', 'aws', 'docker', 
            'kubernetes', 'sql', 'nosql', 'machine learning', 'deep learning',
            'tensorflow', 'pytorch', 'git', 'ci/cd', 'agile', 'scrum',
            'html', 'css', 'vue', 'angular', 'mongodb', 'postgresql',
            'tableau', 'power bi', 'excel', 'communication', 'leadership'
        ]
        
        education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
        experience_levels = ['Entry', 'Junior', 'Mid', 'Senior', 'Lead', 'Principal']
        locations = ['San Francisco', 'New York', 'Austin', 'Remote', 'Seattle', 'Boston']
        
        data = []
        
        for i in range(n_samples):
            # Generate realistic resume
            job_category = np.random.choice(job_categories)
            experience_level = np.random.choice(experience_levels, p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05])
            education = np.random.choice(education_levels, p=[0.05, 0.5, 0.35, 0.1])
            location = np.random.choice(locations)
            
            # Generate skills based on job category
            if 'Software' in job_category or 'DevOps' in job_category or 'ML' in job_category:
                core_skills = np.random.choice(['python', 'java', 'javascript'], 1)[0]
                additional_skills = np.random.choice(skills_db, np.random.randint(3, 8), replace=False)
            elif 'Data' in job_category:
                core_skills = np.random.choice(['python', 'sql', 'machine learning'], 1)[0]
                additional_skills = np.random.choice(skills_db, np.random.randint(3, 7), replace=False)
            elif 'Product' in job_category:
                core_skills = np.random.choice(['agile', 'scrum', 'communication'], 1)[0]
                additional_skills = np.random.choice(skills_db, np.random.randint(2, 5), replace=False)
            else:
                core_skills = np.random.choice(skills_db, 1)[0]
                additional_skills = np.random.choice(skills_db, np.random.randint(2, 6), replace=False)
            
            all_skills = [core_skills] + list(additional_skills)
            
            # Generate experience years based on level
            exp_years_map = {'Entry': 1, 'Junior': 2, 'Mid': 5, 'Senior': 8, 'Lead': 12, 'Principal': 15}
            experience_years = exp_years_map[experience_level] + np.random.randint(-1, 3)
            experience_years = max(1, experience_years)
            
            # Generate job requirements
            num_required = min(np.random.randint(3, 6), len(all_skills))
            required_skills = np.random.choice(all_skills, num_required, replace=False)
            required_experience = experience_years + np.random.randint(-2, 3)
            required_experience = max(1, required_experience)
            
            # Calculate match score (target variable)
            skill_match = len(set(all_skills) & set(required_skills)) / len(set(required_skills))
            exp_match = min(experience_years / required_experience, 1.0)
            education_score = {'High School': 0.3, 'Bachelor': 0.7, 'Master': 0.9, 'PhD': 1.0}[education]
            location_match = 1.0 if location == 'Remote' else np.random.choice([1.0, 0.8, 0.6], p=[0.3, 0.5, 0.2])
            
            # Overall match score
            match_score = (skill_match * 0.4 + exp_match * 0.3 + education_score * 0.2 + location_match * 0.1) * 100
            
            # Generate salary based on role and experience
            base_salaries = {
                'Software Engineer': 80000, 'Data Scientist': 90000, 'Product Manager': 95000,
                'DevOps Engineer': 85000, 'UX Designer': 75000, 'ML Engineer': 110000
            }
            base_salary = base_salaries.get(job_category, 80000)
            exp_multiplier = 1 + (experience_years / 10)
            salary = base_salary * exp_multiplier * np.random.uniform(0.9, 1.3)
            
            data.append({
                'job_category': job_category,
                'experience_level': experience_level,
                'experience_years': experience_years,
                'education': education,
                'location': location,
                'skills': ', '.join(all_skills),
                'required_skills': ', '.join(required_skills),
                'required_experience': required_experience,
                'skill_match': skill_match,
                'experience_match': exp_match,
                'education_score': education_score,
                'location_match': location_match,
                'match_score': match_score,
                'salary': salary
            })
        
        return pd.DataFrame(data)
    
    def train_resume_job_matcher(self):
        """Train real ML model for resume-job matching"""
        print("🤖 Training Resume-Job Matching Model...")
        
        # Generate training data
        data = self.generate_training_data(2000)
        
        # Feature engineering
        features = ['experience_years', 'skill_match', 'experience_match', 'education_score', 'location_match']
        X = data[features]
        y = (data['match_score'] >= 70).astype(int)  # Binary classification: Good match or not
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models and select best
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        best_model = None
        best_score = 0
        model_results = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = accuracy_score(y_test, y_pred)
            
            model_results[name] = {
                'accuracy': score,
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
        
        # Store the best model
        self.models['resume_job_matcher'] = best_model
        self.scalers['resume_job_matcher'] = scaler
        self.training_history['resume_job_matcher'] = {
            'model_type': best_model_name,
            'features': features,
            'performance': model_results[best_model_name],
            'all_results': model_results,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"✅ Best model: {best_model_name} with accuracy: {best_score:.3f}")
        return model_results
    
    def train_salary_predictor(self):
        """Train real ML model for salary prediction"""
        print("💰 Training Salary Prediction Model...")
        
        # Generate training data
        data = self.generate_training_data(1500)
        
        # Feature engineering
        categorical_features = ['job_category', 'experience_level', 'education', 'location']
        numerical_features = ['experience_years', 'skill_match', 'experience_match']
        
        X = data[categorical_features + numerical_features]
        y = data['salary']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Train models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'LinearRegression': LinearRegression(),
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        best_model = None
        best_score = -float('inf')
        best_model_name = None
        model_results = {}
        
        for name, model in models.items():
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_results[name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'mae': np.mean(np.abs(y_test - y_pred))
            }
            
            if mse < best_score:
                best_score = mse
                best_model = pipeline
                best_model_name = name
        
        # Store the best model
        if best_model is not None:
            self.models['salary_predictor'] = best_model
            self.training_history['salary_predictor'] = {
                'model_type': best_model_name,
                'features': categorical_features + numerical_features,
                'performance': model_results[best_model_name],
                'all_results': model_results,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            print(f"✅ Best model: {best_model_name} with RMSE: {np.sqrt(best_score):.2f}")
        else:
            print("❌ No suitable model found for salary prediction")
        
        return model_results
    
    def train_skill_recommender(self):
        """Train real ML model for skill recommendations"""
        print("🎯 Training Skill Recommendation Model...")
        
        # Generate training data
        data = self.generate_training_data(1200)
        
        # Create skill profiles
        all_skills = set()
        for skills in data['skills'].str.split(', '):
            all_skills.update(skills)
        all_skills = list(all_skills)
        
        # Create skill presence matrix
        skill_matrix = []
        for skills in data['skills'].str.split(', '):
            skill_vector = [1 if skill in skills else 0 for skill in all_skills]
            skill_matrix.append(skill_vector)
        
        X = np.array(skill_matrix)
        
        # Target: recommended skills based on job category and experience
        y_categories = data['job_category']
        y_experience = data['experience_level']
        
        # Split data
        X_train, X_test, y_train_cat, y_test_cat = train_test_split(
            X, y_categories, test_size=0.2, random_state=42, stratify=y_categories
        )
        
        # Train multi-label classifier for skill recommendations
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train_cat)
        
        # Evaluate
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test_cat, y_pred)
        
        # Store model and skill mapping
        self.models['skill_recommender'] = {
            'model': rf_model,
            'skill_list': all_skills
        }
        self.training_history['skill_recommender'] = {
            'model_type': 'RandomForest',
            'accuracy': accuracy,
            'num_skills': len(all_skills),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"✅ Skill Recommender trained with accuracy: {accuracy:.3f}")
        return {'accuracy': accuracy}
    
    def train_text_classifier(self):
        """Train NLP model for job description classification"""
        print("📝 Training Text Classification Model...")
        
        # Generate training data
        data = self.generate_training_data(1000)
        
        # Combine job descriptions from skills and requirements
        job_descriptions = data['skills'] + ' ' + data['required_skills']
        job_categories = data['job_category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            job_descriptions, job_categories, test_size=0.2, random_state=42, stratify=job_categories
        )
        
        # Create pipeline with TF-IDF and classifier
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store model
        self.models['text_classifier'] = pipeline
        self.training_history['text_classifier'] = {
            'model_type': 'RandomForest + TF-IDF',
            'accuracy': accuracy,
            'feature_count': 1000,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"✅ Text Classifier trained with accuracy: {accuracy:.3f}")
        return {'accuracy': accuracy}
    
    def evaluate_all_models(self):
        """Comprehensive evaluation of all trained models"""
        print("\n📊 Comprehensive Model Evaluation")
        print("=" * 50)
        
        evaluation_results = {}
        
        for model_name, history in self.training_history.items():
            print(f"\n🔍 {model_name.upper()}:")
            print("-" * 30)
            
            if 'performance' in history:
                perf = history['performance']
                if isinstance(perf, dict):
                    for metric, value in perf.items():
                        print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  Performance: {perf}")
            
            print(f"  Model Type: {history['model_type']}")
            print(f"  Training Samples: {history['training_samples']}")
            print(f"  Test Samples: {history['test_samples']}")
            
            evaluation_results[model_name] = history
        
        return evaluation_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations of model performance"""
        print("\n📈 Creating Performance Visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Machine Learning Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Resume-Job Matcher Performance
        if 'resume_job_matcher' in self.training_history:
            history = self.training_history['resume_job_matcher']
            models = list(history['all_results'].keys())
            accuracies = [history['all_results'][model]['accuracy'] for model in models]
            
            axes[0, 0].bar(models, accuracies, color=['#667eea', '#764ba2', '#f093fb', '#f5576c'])
            axes[0, 0].set_title('Resume-Job Matcher: Model Accuracy')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Salary Predictor Performance
        if 'salary_predictor' in self.training_history:
            history = self.training_history['salary_predictor']
            models = list(history['all_results'].keys())
            r2_scores = [history['all_results'][model]['r2'] for model in models]
            
            axes[0, 1].bar(models, r2_scores, color=['#667eea', '#764ba2', '#f093fb', '#f5576c'])
            axes[0, 1].set_title('Salary Predictor: R² Scores')
            axes[0, 1].set_ylabel('R² Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Feature Importance for Resume Matcher
        if 'resume_job_matcher' in self.models:
            model = self.models['resume_job_matcher']
            if hasattr(model, 'feature_importances_'):
                features = self.training_history['resume_job_matcher']['features']
                importances = model.feature_importances_
                
                axes[0, 2].barh(features, importances, color='#667eea')
                axes[0, 2].set_title('Resume Matcher: Feature Importance')
                axes[0, 2].set_xlabel('Importance')
        
        # 4. Model Accuracy Comparison
        model_names = []
        accuracies = []
        colors = []
        
        for name, history in self.training_history.items():
            if 'accuracy' in history:
                model_names.append(name.replace('_', ' ').title())
                accuracies.append(history['accuracy'])
                colors.append('#667eea')
            elif 'performance' in history and 'accuracy' in history['performance']:
                model_names.append(name.replace('_', ' ').title())
                accuracies.append(history['performance']['accuracy'])
                colors.append('#764ba2')
        
        if model_names:
            axes[1, 0].bar(model_names, accuracies, color=colors)
            axes[1, 0].set_title('Model Accuracy Comparison')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Training Data Distribution
        data = self.generate_training_data(500)
        axes[1, 1].hist(data['match_score'], bins=20, color='#f093fb', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Match Score Distribution')
        axes[1, 1].set_xlabel('Match Score')
        axes[1, 1].set_ylabel('Frequency')
        
        # 6. Salary vs Experience Scatter
        axes[1, 2].scatter(data['experience_years'], data['salary'], alpha=0.6, color='#f5576c')
        axes[1, 2].set_title('Salary vs Experience')
        axes[1, 2].set_xlabel('Experience (Years)')
        axes[1, 2].set_ylabel('Salary ($)')
        
        plt.tight_layout()
        plt.savefig('ml_model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'ml_model_performance.png'")
    
    def save_models(self):
        """Save all trained models"""
        print("\n💾 Saving Trained Models...")
        
        # Create models directory if it doesn't exist
        import os
        os.makedirs('trained_models', exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            if name == 'skill_recommender':
                # Special handling for skill recommender
                joblib.dump(model, f'trained_models/{name}.pkl')
            elif name == 'text_classifier':
                joblib.dump(model, f'trained_models/{name}.pkl')
            else:
                joblib.dump(model, f'trained_models/{name}.pkl')
        
        # Save scalers and encoders
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'trained_models/{name}_scaler.pkl')
        
        # Save training history
        with open('trained_models/training_history.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_serializable = {}
            for key, value in self.training_history.items():
                history_serializable[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        history_serializable[key][k] = v.tolist()
                    elif isinstance(v, (np.int64, np.int32)):
                        history_serializable[key][k] = int(v)
                    elif isinstance(v, (np.float64, np.float32)):
                        history_serializable[key][k] = float(v)
                    else:
                        history_serializable[key][k] = v
            
            json.dump(history_serializable, f, indent=2)
        
        print("✅ All models and training history saved successfully!")
    
    def load_models(self):
        """Load pre-trained models"""
        print("\n📂 Loading Pre-trained Models...")
        
        import os
        if not os.path.exists('trained_models'):
            print("❌ No trained models found. Please train models first.")
            return False
        
        try:
            # Load models
            model_files = {
                'resume_job_matcher': 'resume_job_matcher.pkl',
                'salary_predictor': 'salary_predictor.pkl',
                'skill_recommender': 'skill_recommender.pkl',
                'text_classifier': 'text_classifier.pkl'
            }
            
            for name, filename in model_files.items():
                if os.path.exists(f'trained_models/{filename}'):
                    self.models[name] = joblib.load(f'trained_models/{filename}')
                    print(f"✅ Loaded {name}")
            
            # Load scalers
            if os.path.exists('trained_models/resume_job_matcher_scaler.pkl'):
                self.scalers['resume_job_matcher'] = joblib.load('trained_models/resume_job_matcher_scaler.pkl')
            
            # Load training history
            if os.path.exists('trained_models/training_history.json'):
                with open('trained_models/training_history.json', 'r') as f:
                    self.training_history = json.load(f)
            
            print("✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_resume_match(self, resume_features):
        """Predict resume-job match using trained model"""
        if 'resume_job_matcher' not in self.models:
            print("❌ Resume-job matcher model not trained or loaded")
            return None
        
        model = self.models['resume_job_matcher']
        scaler = self.scalers.get('resume_job_matcher')
        
        # Prepare features
        feature_order = self.training_history['resume_job_matcher']['features']
        X = np.array([[resume_features.get(feature, 0) for feature in feature_order]])
        
        # Scale features
        if scaler:
            X = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else [0, 0]
        
        return {
            'match_probability': probability[1] if len(probability) > 1 else prediction,
            'is_good_match': bool(prediction),
            'confidence': max(probability) if len(probability) > 1 else 0.5
        }
    
    def predict_salary(self, job_features):
        """Predict salary using trained model"""
        if 'salary_predictor' not in self.models:
            print("❌ Salary predictor model not trained or loaded")
            return None
        
        model = self.models['salary_predictor']
        
        # Prepare features as DataFrame
        import pandas as pd
        feature_order = self.training_history['salary_predictor']['features']
        X = pd.DataFrame([job_features], columns=feature_order)
        
        # Predict
        prediction = model.predict(X)[0]
        
        return {
            'predicted_salary': prediction,
            'salary_range': {
                'min': prediction * 0.9,
                'max': prediction * 1.1
            }
        }
    
    def recommend_skills(self, current_skills, target_category):
        """Recommend skills based on current profile and target job"""
        if 'skill_recommender' not in self.models:
            print("❌ Skill recommender model not trained or loaded")
            return None
        
        model_data = self.models['skill_recommender']
        model = model_data['model']
        skill_list = model_data['skill_list']
        
        # Create skill vector
        skill_vector = [1 if skill in current_skills else 0 for skill in skill_list]
        X = np.array([skill_vector])
        
        # Get predictions for different categories
        predictions = model.predict_proba(X)[0]
        classes = model.classes_
        
        # Find recommended skills for target category
        target_index = np.where(classes == target_category)[0]
        if len(target_index) == 0:
            return []
        
        # Get feature importance for target category
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            skill_importance = list(zip(skill_list, importances))
            skill_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Filter out current skills
            recommended = [skill for skill, importance in skill_importance 
                         if skill not in current_skills][:10]
            
            return recommended
        
        return []
    
    def train_all_models(self):
        """Train all machine learning models"""
        print("🚀 Starting Comprehensive ML Training Pipeline")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Train all models
        self.train_resume_job_matcher()
        self.train_salary_predictor()
        self.train_skill_recommender()
        self.train_text_classifier()
        
        # Evaluate all models
        evaluation_results = self.evaluate_all_models()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save models
        self.save_models()
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"\n🎉 Training Pipeline Completed!")
        print(f"⏱️ Total Training Time: {training_time}")
        print(f"📊 Models Trained: {len(self.models)}")
        print(f"💾 Models Saved: trained_models/")
        
        return evaluation_results

def main():
    """Main function to run ML training"""
    print("🤖 Smart Job Portal - Real ML Model Training")
    print("=" * 60)
    
    trainer = RealMLTrainer()
    
    # Check if we should load existing models or train new ones
    import os
    if os.path.exists('trained_models'):
        print("Found existing trained models.")
        choice = input("Load existing models? (y/n): ").lower().strip()
        
        if choice == 'y':
            if trainer.load_models():
                print("\n📊 Available Models:")
                for name in trainer.models.keys():
                    print(f"  ✅ {name}")
                
                # Test predictions
                print("\n🧪 Testing Model Predictions...")
                
                # Test resume matching
                test_features = {
                    'experience_years': 5,
                    'skill_match': 0.8,
                    'experience_match': 0.9,
                    'education_score': 0.8,
                    'location_match': 1.0
                }
                result = trainer.predict_resume_match(test_features)
                print(f"Resume Match Prediction: {result}")
                
                # Test salary prediction
                salary_features = {
                    'job_category': 'Software Engineer',
                    'experience_level': 'Mid',
                    'education': 'Bachelor',
                    'location': 'San Francisco',
                    'experience_years': 5,
                    'skill_match': 0.8,
                    'experience_match': 0.9
                }
                salary_result = trainer.predict_salary(salary_features)
                print(f"Salary Prediction: {salary_result}")
                
                return
    
    # Train new models
    print("Training new models...")
    evaluation_results = trainer.train_all_models()
    
    # Display summary
    print("\n📋 Training Summary:")
    print("=" * 40)
    for model_name, results in evaluation_results.items():
        print(f"\n🔍 {model_name.replace('_', ' ').title()}:")
        if 'accuracy' in results:
            print(f"  Accuracy: {results['accuracy']:.3f}")
        if 'performance' in results:
            perf = results['performance']
            if isinstance(perf, dict):
                for metric, value in perf.items():
                    print(f"  {metric}: {value:.3f}")
        print(f"  Model: {results['model_type']}")

if __name__ == "__main__":
    main()
