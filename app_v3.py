from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import re
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Add machine learning directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'machine_learning'))

app = Flask(__name__)
app.secret_key = 'smart_job_portal_v3_real_ml_2024'
CORS(app)

# Import ML dashboard
try:
    from machine_learning.dashboard import ml_dashboard
    app.register_blueprint(ml_dashboard)
    ML_DASHBOARD_AVAILABLE = True
except ImportError:
    ML_DASHBOARD_AVAILABLE = False

# Load real trained ML models
def load_trained_models():
    """Load the real trained ML models"""
    models = {}
    try:
        # Load resume-job matcher
        models['resume_matcher'] = joblib.load('machine_learning/trained_models/resume_job_matcher.pkl')
        models['resume_matcher_scaler'] = joblib.load('machine_learning/trained_models/resume_job_matcher_scaler.pkl')
        
        # Load skill recommender
        skill_rec_data = joblib.load('machine_learning/trained_models/skill_recommender.pkl')
        models['skill_recommender'] = skill_rec_data['model']
        models['skill_list'] = skill_rec_data['skill_list']
        
        # Load text classifier
        models['text_classifier'] = joblib.load('machine_learning/trained_models/text_classifier.pkl')
        
        # Load training history
        with open('machine_learning/trained_models/training_history.json', 'r') as f:
            models['training_history'] = json.load(f)
        
        print("✅ Real ML models loaded successfully!")
        return models
    except Exception as e:
        print(f"❌ Error loading ML models: {e}")
        return None

# Load models at startup
ml_models = load_trained_models()

class RealAIJobMatcher:
    """Real AI Job Matching using trained ML models"""
    
    def __init__(self):
        self.models = ml_models
        if not self.models:
            print("⚠️ ML models not loaded, using fallback logic")
            self.fallback_mode = True
        else:
            self.fallback_mode = False
    
    def extract_skills_advanced(self, text):
        """Advanced skill extraction with comprehensive database"""
        skills_database = {
            'technical': {
                'programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust', 'swift', 'kotlin'],
                'web': ['html', 'css', 'react', 'vue', 'angular', 'nodejs', 'express', 'django', 'flask', 'spring', 'laravel'],
                'mobile': ['react-native', 'flutter', 'swift', 'kotlin', 'ios', 'android', 'xamarin'],
                'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins'],
                'data': ['sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'cassandra'],
                'ai_ml': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'nlp', 'computer vision', 'reinforcement learning'],
                'devops': ['ci/cd', 'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence'],
                'testing': ['jest', 'mocha', 'selenium', 'cypress', 'junit', 'pytest', 'postman']
            },
            'soft_skills': {
                'leadership': ['leadership', 'management', 'team lead', 'mentoring', 'coaching'],
                'communication': ['communication', 'presentation', 'public speaking', 'negotiation'],
                'analytical': ['problem solving', 'critical thinking', 'analytical skills', 'research'],
                'creativity': ['creativity', 'innovation', 'design thinking', 'brainstorming'],
                'collaboration': ['teamwork', 'collaboration', 'interpersonal skills', 'emotional intelligence']
            }
        }
        
        text_lower = text.lower()
        extracted_skills = {
            'technical': [],
            'soft_skills': [],
            'certifications': [],
            'tools': []
        }
        
        # Extract technical skills
        for category, skills in skills_database['technical'].items():
            for skill in skills:
                if skill in text_lower:
                    extracted_skills['technical'].append(skill)
        
        # Extract soft skills
        for category, skills in skills_database['soft_skills'].items():
            for skill in skills:
                if skill in text_lower:
                    extracted_skills['soft_skills'].append(skill)
        
        # Extract certifications
        cert_patterns = [
            r'(certified|certification|certificate)\s+([a-z]+)',
            r'(aws|azure|gcp|pmp|cfa|cisa|cism)\s+(certified|certification)',
            r'bachelor\'?s?\s+degree',
            r'master\'?s?\s+degree',
            r'phd|ph\.d\.|doctorate'
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    extracted_skills['certifications'].append(' '.join(match))
                else:
                    extracted_skills['certifications'].append(match)
        
        return extracted_skills
    
    def calculate_experience_features(self, resume_text, job_requirements):
        """Calculate experience-related features"""
        exp_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)',
            r'(entry|junior|mid|senior|lead|principal|staff)\s*(?:level|lvl)?'
        ]
        
        resume_exp = 1  # Default
        job_exp = 1
        
        # Extract from resume
        for pattern in exp_patterns:
            matches = re.findall(pattern, resume_text.lower())
            if matches:
                if len(matches[0]) == 2:
                    resume_exp = max(int(matches[0][0]), int(matches[0][1]))
                elif len(matches[0]) == 1:
                    try:
                        resume_exp = int(matches[0][0])
                    except ValueError:
                        level_map = {'entry': 1, 'junior': 2, 'mid': 4, 'senior': 6, 'lead': 8, 'principal': 10, 'staff': 12}
                        if matches[0][0] in level_map:
                            resume_exp = level_map[matches[0][0]]
                break
        
        # Extract from job
        for pattern in exp_patterns:
            matches = re.findall(pattern, job_requirements.lower())
            if matches:
                if len(matches[0]) == 2:
                    job_exp = max(int(matches[0][0]), int(matches[0][1]))
                elif len(matches[0]) == 1:
                    try:
                        job_exp = int(matches[0][0])
                    except ValueError:
                        level_map = {'entry': 1, 'junior': 2, 'mid': 4, 'senior': 6, 'lead': 8, 'principal': 10, 'staff': 12}
                        if matches[0][0] in level_map:
                            job_exp = level_map[matches[0][0]]
                break
        
        # Calculate features
        skill_match = 0.5  # Will be calculated separately
        exp_match = min(resume_exp / job_exp, 1.0)
        education_score = 0.7  # Default
        location_match = 1.0  # Default
        
        return {
            'experience_years': resume_exp,
            'skill_match': skill_match,
            'experience_match': exp_match,
            'education_score': education_score,
            'location_match': location_match
        }
    
    def predict_match_real_ml(self, resume_text, job_description):
        """Predict match using real trained ML models"""
        if self.fallback_mode:
            return self.predict_match_fallback(resume_text, job_description)
        
        try:
            # Extract skills
            resume_skills = self.extract_skills_advanced(resume_text)
            job_skills = self.extract_skills_advanced(job_description)
            
            # Calculate features
            features = self.calculate_experience_features(resume_text, job_description)
            
            # Calculate skill match
            all_resume_skills = resume_skills['technical'] + resume_skills['soft_skills']
            all_job_skills = job_skills['technical'] + job_skills['soft_skills']
            
            if all_job_skills:
                skill_match = len(set(all_resume_skills) & set(all_job_skills)) / len(set(all_job_skills))
            else:
                skill_match = 0.5
            
            features['skill_match'] = skill_match
            
            # Prepare features for ML model
            feature_order = ['experience_years', 'skill_match', 'experience_match', 'education_score', 'location_match']
            X = np.array([[features.get(feat, 0) for feat in feature_order]])
            
            # Scale features
            scaler = self.models['resume_matcher_scaler']
            X_scaled = scaler.transform(X)
            
            # Predict with real ML model
            model = self.models['resume_matcher']
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]
            
            # Get feature importance if available
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_order, model.feature_importances_))
            
            return {
                'overall_match': round(probability[1] * 100, 1) if len(probability) > 1 else round(prediction * 100, 1),
                'is_good_match': bool(prediction) if len(probability) <= 1 else bool(probability[1] > 0.5),
                'confidence': max(probability) if len(probability) > 1 else 0.5,
                'features': features,
                'feature_importance': feature_importance,
                'matched_skills': list(set(all_resume_skills) & set(all_job_skills)),
                'missing_skills': list(set(all_job_skills) - set(all_resume_skills)),
                'resume_skills': all_resume_skills,
                'job_skills': all_job_skills
            }
            
        except Exception as e:
            print(f"❌ Error in real ML prediction: {e}")
            return self.predict_match_fallback(resume_text, job_description)
    
    def predict_match_fallback(self, resume_text, job_description):
        """Fallback prediction logic when ML models are not available"""
        resume_skills = self.extract_skills_advanced(resume_text)
        job_skills = self.extract_skills_advanced(job_description)
        
        all_resume_skills = resume_skills['technical'] + resume_skills['soft_skills']
        all_job_skills = job_skills['technical'] + job_skills['soft_skills']
        
        if all_job_skills:
            skill_match = len(set(all_resume_skills) & set(all_job_skills)) / len(set(all_job_skills))
        else:
            skill_match = 0.5
        
        # Simple scoring
        overall_score = skill_match * 100
        
        return {
            'overall_match': round(overall_score, 1),
            'is_good_match': overall_score >= 70,
            'confidence': 0.7,
            'features': {'skill_match': skill_match},
            'matched_skills': list(set(all_resume_skills) & set(all_job_skills)),
            'missing_skills': list(set(all_job_skills) - set(all_resume_skills)),
            'resume_skills': all_resume_skills,
            'job_skills': all_job_skills,
            'fallback_mode': True
        }
    
    def recommend_skills_real_ml(self, current_skills, target_category=None):
        """Recommend skills using real ML models"""
        if self.fallback_mode:
            return self.recommend_skills_fallback(current_skills, target_category)
        
        try:
            skill_list = self.models['skill_list']
            model = self.models['skill_recommender']
            
            # Create skill vector
            skill_vector = [1 if skill in current_skills else 0 for skill in skill_list]
            X = np.array([skill_vector])
            
            # Get predictions
            predictions = model.predict_proba(X)[0]
            classes = model.classes_
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                skill_importance = list(zip(skill_list, importances))
                skill_importance.sort(key=lambda x: x[1], reverse=True)
                
                # Filter out current skills
                recommended = [skill for skill, importance in skill_importance 
                             if skill not in current_skills][:10]
                
                return recommended
            
            return []
            
        except Exception as e:
            print(f"❌ Error in skill recommendation: {e}")
            return self.recommend_skills_fallback(current_skills, target_category)
    
    def recommend_skills_fallback(self, current_skills, target_category=None):
        """Fallback skill recommendations"""
        all_skills = [
            'python', 'java', 'javascript', 'react', 'nodejs', 'aws', 'docker',
            'kubernetes', 'sql', 'nosql', 'machine learning', 'tensorflow',
            'git', 'ci/cd', 'agile', 'communication', 'leadership'
        ]
        
        # Simple recommendation based on what's missing
        recommended = [skill for skill in all_skills if skill not in current_skills]
        return recommended[:10]
    
    def classify_job_category(self, job_text):
        """Classify job category using real ML models"""
        if self.fallback_mode:
            return self.classify_job_fallback(job_text)
        
        try:
            model = self.models['text_classifier']
            prediction = model.predict([job_text])[0]
            probability = model.predict_proba([job_text])[0]
            
            return {
                'category': prediction,
                'confidence': max(probability),
                'all_probabilities': dict(zip(model.classes_, probability))
            }
            
        except Exception as e:
            print(f"❌ Error in job classification: {e}")
            return self.classify_job_fallback(job_text)
    
    def classify_job_fallback(self, job_text):
        """Fallback job classification"""
        text_lower = job_text.lower()
        
        categories = {
            'Software Engineer': ['software', 'developer', 'programming', 'code', 'python', 'java', 'javascript'],
            'Data Scientist': ['data', 'analytics', 'machine learning', 'statistics', 'python', 'r'],
            'Product Manager': ['product', 'management', 'strategy', 'agile', 'scrum'],
            'DevOps Engineer': ['devops', 'infrastructure', 'cloud', 'docker', 'kubernetes', 'ci/cd'],
            'UX Designer': ['design', 'ux', 'ui', 'user experience', 'figma', 'sketch'],
            'ML Engineer': ['machine learning', 'ai', 'deep learning', 'tensorflow', 'pytorch']
        }
        
        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score
        
        if scores:
            best_category = max(scores, key=scores.get)
            return {
                'category': best_category,
                'confidence': min(scores[best_category] / len(keywords), 1.0),
                'all_probabilities': scores
            }
        
        return {
            'category': 'General',
            'confidence': 0.5,
            'all_probabilities': {}
        }

# Initialize real AI matcher
real_matcher = RealAIJobMatcher()

# Enhanced job database
enhanced_jobs = [
    {
        'id': 1,
        'title': 'Senior Full Stack Developer',
        'company': 'TechCorp Solutions',
        'location': 'San Francisco, CA (Hybrid)',
        'type': 'Full-time',
        'salary': '$140k - $180k',
        'remote_option': 'Hybrid',
        'industry': 'technology',
        'description': '''We are seeking a Senior Full Stack Developer to join our innovative team. 
        You'll work on cutting-edge projects using React, Node.js, and cloud technologies.
        
        Requirements:
        - 5+ years of full-stack development experience
        - Strong proficiency in JavaScript, React, Node.js
        - Experience with cloud platforms (AWS/Azure)
        - Excellent problem-solving and communication skills
        - Bachelor's degree in Computer Science or related field
        
        Responsibilities:
        - Design and develop scalable web applications
        - Collaborate with cross-functional teams
        - Mentor junior developers
        - Drive technical innovation and best practices'''
    },
    {
        'id': 2,
        'title': 'Machine Learning Engineer',
        'company': 'DataTech Analytics',
        'location': 'Remote',
        'type': 'Full-time',
        'salary': '$130k - $160k',
        'remote_option': 'Fully Remote',
        'industry': 'technology',
        'description': '''DataTech Analytics is looking for an experienced Machine Learning Engineer to join our AI research team.
        
        Requirements:
        - 3+ years of machine learning experience
        - Strong Python programming skills
        - Experience with TensorFlow, PyTorch, or similar frameworks
        - Knowledge of deep learning and NLP
        - Master's degree in Computer Science, Statistics, or related field
        
        Responsibilities:
        - Develop and deploy machine learning models
        - Conduct research and implement state-of-the-art algorithms
        - Collaborate with data scientists and engineers
        - Optimize model performance and scalability'''
    },
    {
        'id': 3,
        'title': 'Product Manager',
        'company': 'StartupHub Inc',
        'location': 'New York, NY',
        'type': 'Full-time',
        'salary': '$120k - $150k',
        'remote_option': 'Hybrid',
        'industry': 'technology',
        'description': '''StartupHub is seeking a dynamic Product Manager to lead our product development initiatives.
        
        Requirements:
        - 3+ years of product management experience
        - Strong analytical and communication skills
        - Experience with agile development methodologies
        - Technical background or experience with tech products
        - MBA or equivalent experience preferred
        
        Responsibilities:
        - Define product vision and strategy
        - Work with engineering teams to deliver products
        - Conduct market research and user analysis
        - Drive product roadmap and prioritization'''
    },
    {
        'id': 4,
        'title': 'DevOps Engineer',
        'company': 'CloudScale Systems',
        'location': 'Austin, TX',
        'type': 'Full-time',
        'salary': '$110k - $140k',
        'remote_option': 'Hybrid',
        'industry': 'technology',
        'description': '''CloudScale Systems is looking for a skilled DevOps Engineer to optimize our infrastructure.
        
        Requirements:
        - 4+ years of DevOps experience
        - Strong knowledge of AWS, Docker, Kubernetes
        - Experience with CI/CD pipelines
        - Scripting skills (Python, Bash)
        - Understanding of security best practices
        
        Responsibilities:
        - Design and maintain cloud infrastructure
        - Implement CI/CD pipelines
        - Monitor system performance and reliability
        - Automate deployment and scaling processes'''
    },
    {
        'id': 5,
        'title': 'UX/UI Designer',
        'company': 'Creative Digital Agency',
        'location': 'Los Angeles, CA',
        'type': 'Full-time',
        'salary': '$90k - $120k',
        'remote_option': 'Hybrid',
        'industry': 'technology',
        'description': '''Join our creative team as a UX/UI Designer and shape user experiences across digital products.
        
        Requirements:
        - 3+ years of UX/UI design experience
        - Proficiency in Figma, Sketch, or Adobe Creative Suite
        - Strong portfolio demonstrating design process
        - Understanding of user-centered design principles
        - Bachelor's degree in Design or related field
        
        Responsibilities:
        - Create wireframes, prototypes, and high-fidelity designs
        - Conduct user research and usability testing
        - Collaborate with product and engineering teams
        - Maintain and evolve our design system'''
    }
]

# User database
users_db = {}
job_applications = []

@app.route('/')
def index_v3():
    return render_template('index_v3.html', ML_DASHBOARD_AVAILABLE=ML_DASHBOARD_AVAILABLE)

@app.route('/api/analyze-resume-real-ml', methods=['POST'])
def analyze_resume_real_ml():
    """Real ML-powered resume analysis endpoint"""
    data = request.get_json()
    resume_text = data.get('resume_text', '')
    
    if not resume_text:
        return jsonify({'error': 'Resume text is required'}), 400
    
    matches = []
    for job in enhanced_jobs:
        match_result = real_matcher.predict_match_real_ml(resume_text, job['description'])
        match_result.update(job)
        matches.append(match_result)
    
    matches.sort(key=lambda x: x['overall_match'], reverse=True)
    
    # Generate analysis
    analysis = {
        'total_jobs': len(enhanced_jobs),
        'high_matches': len([m for m in matches if m['overall_match'] >= 80]),
        'medium_matches': len([m for m in matches if 60 <= m['overall_match'] < 80]),
        'low_matches': len([m for m in matches if m['overall_match'] < 60]),
        'average_match': sum(m['overall_match'] for m in matches) / len(matches),
        'top_match': matches[0] if matches else None,
        'ml_model_info': {
            'models_loaded': not real_matcher.fallback_mode,
            'model_types': list(ml_models.keys()) if ml_models else [],
            'training_history': ml_models.get('training_history', {}) if ml_models else {}
        }
    }
    
    return jsonify({
        'matches': matches,
        'analysis': analysis,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/skills-recommendation', methods=['POST'])
def get_skills_recommendation():
    """Get skill recommendations using real ML"""
    data = request.get_json()
    current_skills = data.get('current_skills', [])
    target_category = data.get('target_category')
    
    recommendations = real_matcher.recommend_skills_real_ml(current_skills, target_category)
    
    return jsonify({
        'recommendations': recommendations,
        'current_skills': current_skills,
        'target_category': target_category,
        'ml_model_used': not real_matcher.fallback_mode
    })

@app.route('/api/job-classification', methods=['POST'])
def classify_job():
    """Classify job category using real ML"""
    data = request.get_json()
    job_text = data.get('job_text', '')
    
    if not job_text:
        return jsonify({'error': 'Job text is required'}), 400
    
    classification = real_matcher.classify_job_category(job_text)
    
    return jsonify({
        'classification': classification,
        'ml_model_used': not real_matcher.fallback_mode
    })

@app.route('/api/ml-model-info')
def get_ml_model_info():
    """Get information about loaded ML models"""
    if ml_models:
        return jsonify({
            'models_loaded': True,
            'model_count': len([k for k in ml_models.keys() if not k.endswith('_scaler') and k != 'training_history']),
            'model_types': [k for k in ml_models.keys() if not k.endswith('_scaler') and k != 'training_history'],
            'training_history': ml_models.get('training_history', {}),
            'fallback_mode': real_matcher.fallback_mode
        })
    else:
        return jsonify({
            'models_loaded': False,
            'error': 'ML models not available',
            'fallback_mode': True
        })

@app.route('/api/jobs-enhanced')
def get_enhanced_jobs():
    """Get enhanced job listings"""
    industry = request.args.get('industry')
    job_type = request.args.get('type')
    remote = request.args.get('remote')
    
    filtered_jobs = enhanced_jobs.copy()
    
    if industry:
        filtered_jobs = [job for job in filtered_jobs if job['industry'] == industry]
    
    if job_type:
        filtered_jobs = [job for job in filtered_jobs if job['type'] == job_type]
    
    if remote:
        if remote == 'yes':
            filtered_jobs = [job for job in filtered_jobs if job['remote_option'] in ['Remote', 'Fully Remote']]
        elif remote == 'no':
            filtered_jobs = [job for job in filtered_jobs if job['remote_option'] == 'On-site']
    
    return jsonify(filtered_jobs)

@app.route('/api/job/<int:job_id>/enhanced')
def get_enhanced_job(job_id):
    """Get detailed job information"""
    job = next((job for job in enhanced_jobs if job['id'] == job_id), None)
    if job:
        return jsonify(job)
    return jsonify({'error': 'Job not found'}), 404

@app.route('/api/apply-job', methods=['POST'])
def apply_for_job():
    """Job application endpoint"""
    data = request.get_json()
    job_id = data.get('job_id')
    user_email = data.get('user_email')
    resume_text = data.get('resume_text', '')
    cover_letter = data.get('cover_letter', '')
    
    if not job_id or not user_email:
        return jsonify({'error': 'Job ID and user email are required'}), 400
    
    # Create application record
    application = {
        'id': len(job_applications) + 1,
        'job_id': job_id,
        'user_email': user_email,
        'resume_text': resume_text,
        'cover_letter': cover_letter,
        'applied_at': datetime.now().isoformat(),
        'status': 'submitted'
    }
    
    job_applications.append(application)
    
    return jsonify({
        'success': True,
        'application_id': application['id'],
        'message': 'Application submitted successfully!'
    })

@app.route('/api/user-profile', methods=['GET', 'POST'])
def user_profile():
    """User profile management"""
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        # Create or update user profile
        profile = {
            'email': email,
            'name': data.get('name', ''),
            'phone': data.get('phone', ''),
            'location': data.get('location', ''),
            'resume_text': data.get('resume_text', ''),
            'skills': data.get('skills', []),
            'experience': data.get('experience', ''),
            'education': data.get('education', ''),
            'preferences': data.get('preferences', {}),
            'updated_at': datetime.now().isoformat()
        }
        
        users_db[email] = profile
        return jsonify({'success': True, 'profile': profile})
    
    else:  # GET
        email = request.args.get('email')
        if email and email in users_db:
            return jsonify(users_db[email])
        return jsonify({'error': 'Profile not found'}), 404

if __name__ == '__main__':
    print("🚀 Starting Smart Job Portal V3 with Real ML Models")
    print(f"🤖 ML Models Loaded: {ml_models is not None}")
    print(f"📊 Fallback Mode: {real_matcher.fallback_mode}")
    app.run(debug=True, port=5000)
