from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import re
import json
import os
import sys
from datetime import datetime, timedelta
import random
import hashlib

# Add machine learning directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'machine_learning'))

app = Flask(__name__)
app.secret_key = 'smart_job_portal_v2_2024'
CORS(app)

# Import ML dashboard
try:
    from machine_learning.dashboard import ml_dashboard
    app.register_blueprint(ml_dashboard)
    ML_DASHBOARD_AVAILABLE = True
except ImportError:
    ML_DASHBOARD_AVAILABLE = False

class AdvancedAIJobMatcher:
    """Enhanced AI Job Matching System with more sophisticated algorithms"""
    
    def __init__(self):
        self.skills_weight = 0.35
        self.experience_weight = 0.25
        self.education_weight = 0.15
        self.location_weight = 0.10
        self.culture_weight = 0.10
        self.growth_weight = 0.05
        
        # Expanded skills database
        self.skills_database = {
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
        
        # Industry-specific keywords
        self.industries = {
            'technology': ['software', 'tech', 'it', 'startup', 'saas', 'fintech', 'healthtech'],
            'finance': ['banking', 'finance', 'investment', 'insurance', 'accounting'],
            'healthcare': ['medical', 'healthcare', 'hospital', 'pharmaceutical', 'biotech'],
            'education': ['education', 'e-learning', 'university', 'training', 'academic'],
            'retail': ['retail', 'ecommerce', 'sales', 'customer service', 'merchandising'],
            'manufacturing': ['manufacturing', 'production', 'logistics', 'supply chain', 'quality control']
        }
    
    def extract_comprehensive_skills(self, text):
        """Advanced skill extraction with categorization"""
        text_lower = text.lower()
        extracted_skills = {
            'technical': [],
            'soft_skills': [],
            'certifications': [],
            'tools': []
        }
        
        # Extract technical skills
        for category, skills in self.skills_database['technical'].items():
            for skill in skills:
                if skill in text_lower:
                    extracted_skills['technical'].append(skill)
        
        # Extract soft skills
        for category, skills in self.skills_database['soft_skills'].items():
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
        
        # Extract tools and platforms
        tool_patterns = [
            r'(microsoft|google|adobe|salesforce|slack|zoom|teams|jira|asana|trello)',
            r'(office|excel|powerpoint|word|outlook|photoshop|illustrator)',
            r'(slack|discord|skype|zoom|teams|webex)'
        ]
        
        for pattern in tool_patterns:
            matches = re.findall(pattern, text_lower)
            extracted_skills['tools'].extend(matches)
        
        return extracted_skills
    
    def calculate_experience_level(self, resume_text, job_requirements):
        """Enhanced experience calculation with level categorization"""
        exp_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)',
            r'(entry|junior|mid|senior|lead|principal|staff)\s*(?:level|lvl)?'
        ]
        
        resume_exp = 0
        job_exp = 1
        resume_level = 'entry'
        job_level = 'entry'
        
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
                        # Handle level-based matching
                        level_map = {'entry': 1, 'junior': 2, 'mid': 4, 'senior': 6, 'lead': 8, 'principal': 10, 'staff': 12}
                        resume_level = matches[0][0]
                        if resume_level in level_map:
                            resume_exp = level_map[resume_level]
                break
        
        # Extract from job requirements
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
                        job_level = matches[0][0]
                        if job_level in level_map:
                            job_exp = level_map[job_level]
                break
        
        # Calculate experience match with level consideration
        if resume_exp >= job_exp:
            exp_score = min(1.0, 1.0 - (resume_exp - job_exp) * 0.05)  # Slight penalty for overqualification
        else:
            exp_score = resume_exp / job_exp
        
        return {
            'score': exp_score,
            'resume_years': resume_exp,
            'required_years': job_exp,
            'resume_level': resume_level,
            'required_level': job_level
        }
    
    def analyze_cultural_fit(self, resume_text, job_description):
        """Analyze cultural fit based on keywords and values"""
        culture_keywords = {
            'innovation': ['innovative', 'creative', 'disruptive', 'breakthrough', 'pioneering'],
            'collaboration': ['team', 'collaborative', 'partnership', 'synergy', 'cooperative'],
            'growth': ['growth', 'learning', 'development', 'mentorship', 'advancement'],
            'work_life_balance': ['balance', 'flexible', 'remote', 'hybrid', 'autonomy'],
            'results_driven': ['results', 'outcome', 'impact', 'achievement', 'performance'],
            'diversity': ['diverse', 'inclusive', 'multicultural', 'equality', 'belonging']
        }
        
        resume_culture = {}
        job_culture = {}
        
        for category, keywords in culture_keywords.items():
            resume_culture[category] = sum(1 for keyword in keywords if keyword in resume_text.lower())
            job_culture[category] = sum(1 for keyword in keywords if keyword in job_description.lower())
        
        # Calculate cultural fit score
        total_match = 0
        total_job = 0
        
        for category in culture_keywords:
            if job_culture[category] > 0:
                total_match += min(resume_culture[category], job_culture[category])
                total_job += job_culture[category]
        
        culture_score = total_match / total_job if total_job > 0 else 0.5
        
        return {
            'score': culture_score,
            'resume_culture': resume_culture,
            'job_culture': job_culture
        }
    
    def predict_career_growth(self, resume_text, job_description):
        """Predict career growth potential"""
        growth_indicators = [
            'leadership', 'management', 'mentor', 'train', 'guide', 'lead',
            'promoted', 'promotion', 'advance', 'growth', 'develop',
            'initiative', 'innovate', 'improve', 'optimize', 'scale'
        ]
        
        resume_growth = sum(1 for indicator in growth_indicators if indicator in resume_text.lower())
        job_growth = sum(1 for indicator in growth_indicators if indicator in job_description.lower())
        
        growth_score = min(1.0, (resume_growth + 1) / (job_growth + 1))
        
        return {
            'score': growth_score,
            'growth_indicators_found': resume_growth,
            'growth_opportunities': job_growth
        }
    
    def advanced_match_resume_job(self, resume_text, job_description):
        """Comprehensive resume-job matching with multiple dimensions"""
        
        # Extract comprehensive skills
        resume_skills = self.extract_comprehensive_skills(resume_text)
        job_skills = self.extract_comprehensive_skills(job_description)
        
        # Calculate skill match with weighting
        skill_matches = {
            'technical': self.calculate_skill_match(resume_skills['technical'], job_skills['technical']),
            'soft_skills': self.calculate_skill_match(resume_skills['soft_skills'], job_skills['soft_skills']),
            'certifications': self.calculate_skill_match(resume_skills['certifications'], job_skills['certifications']),
            'tools': self.calculate_skill_match(resume_skills['tools'], job_skills['tools'])
        }
        
        # Weighted skill score
        skill_score = (
            skill_matches['technical'] * 0.5 +
            skill_matches['soft_skills'] * 0.2 +
            skill_matches['certifications'] * 0.2 +
            skill_matches['tools'] * 0.1
        )
        
        # Calculate other dimensions
        experience_analysis = self.calculate_experience_level(resume_text, job_description)
        culture_fit = self.analyze_cultural_fit(resume_text, job_description)
        career_growth = self.predict_career_growth(resume_text, job_description)
        
        # Calculate overall score
        overall_score = (
            skill_score * self.skills_weight +
            experience_analysis['score'] * self.experience_weight +
            self.calculate_education_score(resume_text, job_description) * self.education_weight +
            self.calculate_location_match(resume_text, job_description) * self.location_weight +
            culture_fit['score'] * self.culture_weight +
            career_growth['score'] * self.growth_weight
        ) * 100
        
        return {
            'overall_match': round(overall_score, 1),
            'dimension_scores': {
                'skills': round(skill_score * 100, 1),
                'experience': round(experience_analysis['score'] * 100, 1),
                'education': round(self.calculate_education_score(resume_text, job_description) * 100, 1),
                'location': round(self.calculate_location_match(resume_text, job_description) * 100, 1),
                'culture': round(culture_fit['score'] * 100, 1),
                'growth': round(career_growth['score'] * 100, 1)
            },
            'skill_analysis': {
                'resume_skills': resume_skills,
                'job_skills': job_skills,
                'skill_matches': skill_matches,
                'missing_skills': self.find_missing_skills(resume_skills, job_skills)
            },
            'experience_analysis': experience_analysis,
            'culture_fit': culture_fit,
            'career_growth': career_growth,
            'recommendations': self.generate_recommendations(overall_score, skill_matches, experience_analysis)
        }
    
    def calculate_skill_match(self, resume_skills, job_skills):
        """Calculate skill match percentage"""
        if not job_skills:
            return 0.5
        
        resume_set = set(resume_skills)
        job_set = set(job_skills)
        
        if not job_set:
            return 0.5
        
        match = len(resume_set & job_set) / len(job_set)
        return match
    
    def find_missing_skills(self, resume_skills, job_skills):
        """Find missing skills with priority"""
        missing = {}
        
        for category in job_skills:
            job_set = set(job_skills[category])
            resume_set = set(resume_skills.get(category, []))
            missing_skills = list(job_set - resume_set)
            
            if missing_skills:
                missing[category] = missing_skills
        
        return missing
    
    def calculate_education_score(self, resume_text, job_requirements):
        """Enhanced education scoring with degree types"""
        education_levels = {
            'phd': 8, 'doctorate': 8, 'doctoral': 8,
            'master': 7, 'ms': 7, 'm.sc': 7, 'm.s': 7,
            'bachelor': 6, 'bs': 6, 'b.sc': 6, 'b.s': 6,
            'associate': 4, 'diploma': 3, 'certificate': 2
        }
        
        resume_edu = 0
        job_edu = 4  # Default to bachelor's
        
        for level, score in education_levels.items():
            if level in resume_text.lower():
                resume_edu = max(resume_edu, score)
            if level in job_requirements.lower():
                job_edu = max(job_edu, score)
        
        return min(resume_edu / job_edu, 1.0) if job_edu > 0 else 0.5
    
    def calculate_location_match(self, resume_text, job_requirements):
        """Enhanced location matching with remote work consideration"""
        resume_locations = re.findall(r'([a-zA-Z\s]+,?\s*[a-zA-Z]{2,3})', resume_text)
        job_locations = re.findall(r'([a-zA-Z\s]+,?\s*[a-zA-Z]{2,3})', job_requirements)
        
        # Check for remote work
        remote_keywords = ['remote', 'work from home', 'wfh', 'hybrid', 'telecommute']
        is_remote = any(keyword in job_requirements.lower() for keyword in remote_keywords)
        
        if is_remote:
            return 1.0  # Perfect match for remote positions
        
        if not job_locations:
            return 0.8  # Neutral if no location specified
        
        if not resume_locations:
            return 0.5  # Partial if no resume location
        
        # Simple location matching
        for job_loc in job_locations:
            for resume_loc in resume_locations:
                if job_loc.strip().lower() in resume_loc.strip().lower():
                    return 1.0
        
        return 0.3  # Low but not zero for different locations
    
    def generate_recommendations(self, overall_score, skill_matches, experience_analysis):
        """Generate personalized recommendations"""
        recommendations = []
        
        if overall_score < 70:
            recommendations.append("Consider gaining more experience or additional skills to improve your match")
        
        if skill_matches['technical'] < 0.6:
            recommendations.append("Focus on developing technical skills mentioned in the job requirements")
        
        if skill_matches['soft_skills'] < 0.5:
            recommendations.append("Highlight your soft skills and consider developing communication and teamwork abilities")
        
        if experience_analysis['score'] < 0.7:
            if experience_analysis['resume_years'] < experience_analysis['required_years']:
                recommendations.append(f"Consider gaining {experience_analysis['required_years'] - experience_analysis['resume_years']} more years of relevant experience")
            else:
                recommendations.append("Highlight your leadership and mentorship experience")
        
        if not recommendations:
            recommendations.append("Your profile matches well! Focus on tailoring your resume for this specific position")
        
        return recommendations

# Initialize advanced matcher
advanced_matcher = AdvancedAIJobMatcher()

# Enhanced job database with more realistic data
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
        - Drive technical innovation and best practices
        
        We offer competitive salary, comprehensive benefits, and opportunities for professional growth.''',
        'culture_keywords': ['innovation', 'collaboration', 'growth', 'flexible'],
        'benefits': ['Health insurance', '401k matching', 'Unlimited PTO', 'Remote work options', 'Learning budget']
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
        - Optimize model performance and scalability
        
        We offer remote-first culture, competitive compensation, and cutting-edge projects.''',
        'culture_keywords': ['innovation', 'research', 'growth', 'autonomy'],
        'benefits': ['Fully remote', 'Flexible hours', 'Conference budget', 'Health insurance', 'Stock options']
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
        - Drive product roadmap and prioritization
        
        Join us in building the next generation of startup tools!''',
        'culture_keywords': ['innovation', 'leadership', 'growth', 'collaboration'],
        'benefits': ['Equity options', 'Flexible work', 'Professional development', 'Health benefits', 'Gym membership']
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
        - Automate deployment and scaling processes
        
        We offer a collaborative environment and cutting-edge technology stack.''',
        'culture_keywords': ['collaboration', 'innovation', 'results', 'growth'],
        'benefits': ['Remote work', 'Training budget', 'Health insurance', '401k', 'Tool allowance']
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
        - Maintain and evolve our design system
        
        We value creativity, innovation, and user-centered thinking.''',
        'culture_keywords': ['creativity', 'collaboration', 'innovation', 'growth'],
        'benefits': ['Creative freedom', 'Flexible schedule', 'Design budget', 'Health insurance', 'Remote options']
    }
]

# User database (in production, use a real database)
users_db = {}
job_applications = []

@app.route('/')
def index_v2():
    return render_template('index_v2.html', ML_DASHBOARD_AVAILABLE=ML_DASHBOARD_AVAILABLE)

@app.route('/api/analyze-resume-advanced', methods=['POST'])
def analyze_resume_advanced():
    """Advanced resume analysis endpoint"""
    data = request.get_json()
    resume_text = data.get('resume_text', '')
    
    if not resume_text:
        return jsonify({'error': 'Resume text is required'}), 400
    
    matches = []
    for job in enhanced_jobs:
        match_result = advanced_matcher.advanced_match_resume_job(resume_text, job['description'])
        match_result.update(job)
        matches.append(match_result)
    
    matches.sort(key=lambda x: x['overall_match'], reverse=True)
    
    # Generate comprehensive analysis
    analysis = {
        'total_jobs': len(enhanced_jobs),
        'high_matches': len([m for m in matches if m['overall_match'] >= 80]),
        'medium_matches': len([m for m in matches if 60 <= m['overall_match'] < 80]),
        'low_matches': len([m for m in matches if m['overall_match'] < 60]),
        'average_match': sum(m['overall_match'] for m in matches) / len(matches),
        'top_match': matches[0] if matches else None
    }
    
    return jsonify({
        'matches': matches,
        'analysis': analysis,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/jobs-enhanced')
def get_enhanced_jobs():
    """Get enhanced job listings with filters"""
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

@app.route('/api/salary-insights')
def get_salary_insights():
    """Get salary insights by role and location"""
    insights = {
        'software_engineer': {
            'entry': {'min': 60, 'max': 90, 'avg': 75},
            'mid': {'min': 90, 'max': 130, 'avg': 110},
            'senior': {'min': 130, 'max': 180, 'avg': 155}
        },
        'data_scientist': {
            'entry': {'min': 70, 'max': 100, 'avg': 85},
            'mid': {'min': 100, 'max': 140, 'avg': 120},
            'senior': {'min': 140, 'max': 190, 'avg': 165}
        },
        'product_manager': {
            'entry': {'min': 80, 'max': 110, 'avg': 95},
            'mid': {'min': 110, 'max': 150, 'avg': 130},
            'senior': {'min': 150, 'max': 200, 'avg': 175}
        }
    }
    
    return jsonify(insights)

@app.route('/api/career-advice')
def get_career_advice():
    """Get career advice and recommendations"""
    advice = {
        'resume_tips': [
            'Use quantifiable achievements to demonstrate impact',
            'Tailor your resume for each job application',
            'Include relevant keywords from the job description',
            'Keep your resume concise and focused (1-2 pages)',
            'Use action verbs to start bullet points'
        ],
        'interview_tips': [
            'Research the company and role thoroughly',
            'Prepare examples using the STAR method',
            'Practice common interview questions',
            'Prepare thoughtful questions to ask the interviewer',
            'Follow up with a thank-you email'
        ],
        'skill_development': [
            'Focus on in-demand technical skills for your field',
            'Develop soft skills like communication and leadership',
            'Consider certifications to validate your expertise',
            'Build a portfolio of projects to showcase your skills',
            'Network with professionals in your industry'
        ]
    }
    
    return jsonify(advice)

@app.route('/api/market-trends')
def get_market_trends():
    """Get job market trends and insights"""
    trends = {
        'growing_skills': [
            {'skill': 'Machine Learning', 'growth': 35, 'demand': 'High'},
            {'skill': 'Cloud Computing', 'growth': 28, 'demand': 'High'},
            {'skill': 'Cybersecurity', 'growth': 25, 'demand': 'High'},
            {'skill': 'Data Analysis', 'growth': 22, 'demand': 'High'},
            {'skill': 'DevOps', 'growth': 20, 'demand': 'High'}
        ],
        'industry_trends': [
            {'industry': 'Technology', 'growth': 15, 'jobs_added': 150000},
            {'industry': 'Healthcare', 'growth': 12, 'jobs_added': 85000},
            {'industry': 'Finance', 'growth': 8, 'jobs_added': 45000},
            {'industry': 'Education', 'growth': 6, 'jobs_added': 32000}
        ],
        'remote_work': {
            'percentage': 35,
            'trending': 'Increasing',
            'popular_roles': ['Software Developer', 'Data Analyst', 'Digital Marketer', 'Customer Support']
        }
    }
    
    return jsonify(trends)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
