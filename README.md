# Smart Job Portal - AI Powered Career Matching Platform

An intelligent job portal that uses AI and machine learning to match resumes with job requirements, providing percentage-based matching scores and personalized recommendations.

## 🚀 Features

### AI-Powered Matching
- **Resume Analysis**: Get detailed percentage matching scores for your resume against job requirements
- **Skill Extraction**: AI automatically identifies and extracts skills from your resume
- **Gap Analysis**: Shows missing skills and suggests what to learn to improve job prospects
- **Multi-factor Scoring**: Considers skills, experience, education, and location preferences

### Smart Recommendations
- **Personalized Job Matches**: Jobs ranked by compatibility percentage
- **Skill Development Suggestions**: Recommends skills to learn based on job market demands
- **Career Path Insights**: Analysis of your profile vs. job requirements

### Modern Interface
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Real-time Analysis**: Instant AI processing of your resume
- **Interactive Dashboard**: Visual representation of match scores and skill gaps

## 🛠 Technology Stack

- **Backend**: Python Flask with scikit-learn ML models
- **Frontend**: HTML5, TailwindCSS, JavaScript (Vanilla)
- **AI/ML**: TF-IDF Vectorization, Cosine Similarity, NLP Processing
- **Database**: SQLite (easily upgradeable to PostgreSQL)
- **Deployment**: Ready for Heroku, Vercel, or any cloud platform

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd "Smart Job Portal"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:5000`

## 🎯 How to Use

### For Job Seekers

1. **Paste Your Resume**: Copy and paste your resume text into the analysis area
2. **Get Instant Analysis**: Click "Analyze Resume" to receive:
   - Overall match percentages for available jobs
   - Skill breakdown (matched vs missing skills)
   - Experience and education compatibility scores
3. **Browse Recommendations**: View jobs ranked by compatibility
4. **Skill Development**: See suggested skills to improve your prospects

### Sample Resume Format
The system works best with resumes that include:
- Professional summary
- Skills section
- Work experience with duration
- Education details
- Contact information

## 🔧 API Endpoints

### Resume Analysis
- `POST /api/analyze-resume`
  - Input: `{ "resume_text": "your resume content" }`
  - Output: Match scores, skill analysis, job recommendations

### Job Listings
- `GET /api/jobs` - Returns all available job postings
- `GET /api/job/{id}` - Returns specific job details

### Skills Analysis
- `POST /api/skills-suggestion`
  - Input: `{ "resume_text": "your resume content" }`
  - Output: Current skills and suggested skills to learn

## 🚀 Deployment Options

### Option 1: Heroku (Recommended for Beginners)
1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: gunicorn app:app
   ```
3. Deploy:
   ```bash
   heroku create your-app-name
   git add .
   git commit -m "Initial deploy"
   git push heroku main
   ```

### Option 2: Vercel (Serverless)
1. Install Vercel CLI
2. Create `vercel.json` configuration
3. Deploy:
   ```bash
   vercel --prod
   ```

### Option 3: PythonAnywhere (Free Hosting)
1. Create account at pythonanywhere.com
2. Upload files via web interface
3. Configure web app in dashboard
4. Install requirements and run

### Option 4: Self-Hosting
- **VPS**: DigitalOcean, Linode, AWS EC2
- **Docker**: Create Dockerfile for containerized deployment
- **Local Server**: Run on company intranet

## 💰 Monetization Options

### For Resume Service
1. **Freemium Model**:
   - Free: Basic analysis (3 jobs per day)
   - Premium: $9.99/month - Unlimited analysis, advanced features

2. **Pay-per-Analysis**:
   - $1.99 for detailed resume report
   - $4.99 for resume + interview preparation

3. **Enterprise Plans**:
   - Companies: $99/month for job posting + candidate matching
   - Universities: Campus recruitment packages

### Integration Opportunities
- **LinkedIn Integration**: Import resumes directly
- **ATS Integration**: Connect with Applicant Tracking Systems
- **API Access**: Sell API keys to other job platforms

## 🎨 Customization

### Adding New Jobs
Edit the `sample_jobs` list in `app.py`:
```python
{
    'id': 4,
    'title': 'Your Job Title',
    'company': 'Company Name',
    'location': 'City, State',
    'type': 'Full-time',
    'salary': '$X - $Y',
    'description': 'Job description with requirements...'
}
```

### Modifying AI Weights
Adjust matching criteria in `AIJobMatcher` class:
```python
self.skills_weight = 0.4      # Skill importance
self.experience_weight = 0.3  # Experience importance
self.education_weight = 0.2   # Education importance
self.location_weight = 0.1    # Location preference
```

### Adding New Skills
Update the `common_skills` list in `extract_skills` method to include industry-specific skills.

## 🔒 Security Considerations

- Input validation for resume text
- Rate limiting for API endpoints
- HTTPS for production deployment
- Sanitization of user inputs
- GDPR compliance for EU users

## 📈 Scaling Up

### Database Migration
- Switch from SQLite to PostgreSQL for better performance
- Add Redis for caching
- Implement user authentication system

### Enhanced AI Features
- Integrate OpenAI GPT for resume improvement suggestions
- Add salary prediction models
- Implement career path recommendations
- Add industry trend analysis

### Enterprise Features
- Multi-tenant architecture for different companies
- Advanced analytics dashboard
- Bulk resume processing
- Integration with HR systems

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📞 Support

For questions or support:
- Email: support@smartjobportal.com
- Documentation: Check this README file
- Issues: Report via GitHub issues

## 📄 License

This project is open source 


---

**Start your AI-powered job search today!** 🚀
