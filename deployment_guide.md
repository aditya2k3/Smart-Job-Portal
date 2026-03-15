# Deployment Guide - Smart Job Portal

## Quick Start Options

### 🚀 Option 1: Local Testing (5 minutes)
```bash
cd "Smart Job Portal"
pip install -r requirements.txt
python app.py
# Visit http://localhost:5000
```

### ☁️ Option 2: Heroku Deployment (10 minutes)

#### Step 1: Prepare for Heroku
1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
2. Login: `heroku login`
3. Create `Procfile` in project root:
```
web: gunicorn app:app
```

#### Step 2: Deploy Commands
```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial commit"

# Create Heroku app
heroku create your-smart-job-portal

# Deploy
git push heroku main

# Open your app
heroku open
```

### 🌐 Option 3: Vercel Deployment (8 minutes)

#### Step 1: Install Vercel CLI
```bash
npm i -g vercel
```

#### Step 2: Create vercel.json
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

#### Step 3: Deploy
```bash
vercel --prod
```

### 🐳 Option 4: Docker Deployment

#### Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

#### Build and Run
```bash
# Build image
docker build -t smart-job-portal .

# Run container
docker run -p 5000:5000 smart-job-portal
```

### 🏢 Option 5: PythonAnywhere (Free)

#### Steps:
1. Sign up at https://www.pythonanywhere.com
2. Go to "Web" tab → "Add a new web app"
3. Choose "Flask" framework
4. Upload your files via web interface
5. Install requirements in Bash console:
   ```bash
   pip install -r requirements.txt
   ```
6. Configure WSGI file to point to your app

## 🎯 Best Deployment Practices

### Security
1. **Environment Variables**: Store sensitive data in environment variables
2. **HTTPS**: Always use HTTPS in production
3. **Rate Limiting**: Implement rate limiting for API endpoints

### Performance
1. **Caching**: Add Redis for caching results
2. **CDN**: Use CloudFlare for static assets
3. **Database**: Upgrade to PostgreSQL for better performance

### Monitoring
1. **Logging**: Implement proper logging
2. **Health Checks**: Add `/health` endpoint
3. **Analytics**: Track user interactions

## 💰 Cost Estimates

### Free Options
- **PythonAnywhere**: Free tier (limited)
- **Heroku**: Free tier (sleeps after inactivity)
- **Vercel**: Free tier (limited bandwidth)

### Paid Options (Monthly)
- **Heroku Hobby**: $7/month
- **DigitalOcean**: $5/month (1GB RAM)
- **AWS EC2**: $3.50/month (t2.nano)

## 🔧 Production Configuration

### Environment Variables
Create `.env` file:
```
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
DATABASE_URL=your-database-url
```

### Production Requirements
Update `requirements.txt`:
```
gunicorn==21.2.0
psycopg2-binary==2.9.7
redis==4.6.0
```

## 📱 Mobile App Options

### React Native App
```bash
# Create mobile app
npx react-native init SmartJobPortalMobile
# Add API calls to your Flask backend
```

### Progressive Web App (PWA)
Add to your HTML:
```html
<link rel="manifest" href="manifest.json">
<meta name="theme-color" content="#667eea">
```

## 🌍 Domain Setup

### Custom Domain (Heroku)
```bash
heroku domains:add yourdomain.com
heroku certs:add /path/to/cert.pem /path/to/key.pem
```

### DNS Configuration
- A record: Points to your server IP
- CNAME: Points to provider (Heroku, Vercel)

## 📊 Scaling Considerations

### When to Scale Up
- More than 1000 concurrent users
- Slow response times (>2 seconds)
- High CPU/memory usage

### Scaling Options
1. **Horizontal**: Add more server instances
2. **Vertical**: Upgrade server resources
3. **Database**: Separate database server
4. **CDN**: Distribute static assets globally

## 🔄 CI/CD Pipeline

### GitHub Actions Example
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Production
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Heroku
        uses: akhileshns/heroku-deploy@v3.12.12
        with:
          heroku_api_key: ${{secrets.HEROKU_API_KEY}}
          heroku_app_name: your-app-name
          heroku_email: your-email@example.com
```

## 🎓 Next Steps

After deployment:
1. **Analytics**: Add Google Analytics
2. **SEO**: Optimize meta tags and descriptions
3. **Testing**: Implement automated tests
4. **Backup**: Regular database backups
5. **Monitoring**: Set up uptime monitoring

Choose the deployment option that best fits your needs and budget!
