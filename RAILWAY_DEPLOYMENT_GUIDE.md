# Railway Deployment - Complete Beginner's Guide

## 🎯 What You Need Before Starting
1. GitHub account (free)
2. Railway account (free tier available)
3. Your project ready to deploy

---

## 📋 STEP 1: Prepare Your Project for GitHub

### Clean Up Unnecessary Files ✅ (Already Done!)
The following files have been removed:
- ❌ `test_*.py` (test files)
- ❌ `*.bat`, `*.ps1` (Windows scripts)
- ❌ `*.md` documentation files (except README.md)
- ❌ `__pycache__/` (Python cache)
- ❌ `venv/` (will be in .gitignore)

### Files You Need to Deploy ✅ (Already Created!)
- ✅ `Procfile` - Tells Railway how to run your app
- ✅ `requirements.txt` - Lists all Python packages (includes gunicorn)
- ✅ `runtime.txt` - Specifies Python version
- ✅ `.gitignore` - Lists files to ignore in git
- ✅ `README.md` - Project documentation

---

## 📦 STEP 2: Update .gitignore

Make sure your `.gitignore` includes:
```
venv/
__pycache__/
*.pyc
.env
*.log
.DS_Store
```

---

## 🐙 STEP 3: Push to GitHub

### A. Create a New Repository on GitHub
1. Go to https://github.com
2. Click the **"+"** button (top right) → **"New repository"**
3. Fill in:
   - **Repository name**: `fake-job-detection`
   - **Description**: "AI-powered fake job detection system"
   - **Visibility**: Public or Private (your choice)
   - ⚠️ **DO NOT** check "Add README" (you already have one)
4. Click **"Create repository"**

### B. Initialize Git in Your Project
Open PowerShell in your project folder:

```powershell
cd "c:\Users\pavan\fake job detection"

# Initialize git (if not already)
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit - Fake job detection app"

# Add your GitHub repository as remote
# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/fake-job-detection.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Example:**
If your GitHub username is `pavanreddy`, the command would be:
```powershell
git remote add origin https://github.com/pavanreddy/fake-job-detection.git
```

### C. Verify on GitHub
1. Refresh your GitHub repository page
2. You should see all your files uploaded

---

## 🚂 STEP 4: Deploy on Railway

### A. Create Railway Account
1. Go to https://railway.app
2. Click **"Login"**
3. Choose **"Login with GitHub"** (easiest option)
4. Authorize Railway to access your GitHub

### B. Create New Project
1. After logging in, click **"New Project"**
2. Choose **"Deploy from GitHub repo"**
3. Railway will ask for GitHub permissions:
   - Click **"Configure GitHub App"**
   - Select **"Only select repositories"**
   - Choose your `fake-job-detection` repository
   - Click **"Install & Authorize"**

### C. Select Your Repository
1. You'll see a list of your repositories
2. Click on **`fake-job-detection`**
3. Railway will automatically detect it's a Python app

### D. Configure Deployment Settings

#### 1. **Add Environment Variables** (Optional but Recommended)
After deployment starts, click on your service:
1. Go to **"Variables"** tab
2. Add these variables:
   ```
   THRESHOLD = 0.5
   ```
3. Click **"Add"**

#### 2. **Configure Build Settings** (Automatic)
Railway automatically detects:
- ✅ Python 3.11 (from `runtime.txt`)
- ✅ Dependencies from `requirements.txt`
- ✅ Start command from `Procfile`

---

## 🎉 STEP 5: Deployment Process

### What Happens Automatically:
1. **Build Phase** (5-10 minutes):
   - Railway installs Python 3.11
   - Installs all packages from `requirements.txt`
   - Downloads DistilBERT model (~250MB)
   - Sets up Gunicorn web server

2. **Deploy Phase** (1-2 minutes):
   - Starts your Flask app with Gunicorn
   - Binds to Railway's assigned port
   - Makes your app publicly accessible

### Watch the Build Logs:
1. Click on **"Deployments"** tab
2. Click on the latest deployment
3. View **"Build Logs"** and **"Deploy Logs"**
4. Wait for: ✅ **"Build successful"** → ✅ **"Deployment successful"**

---

## 🌐 STEP 6: Get Your Live URL

### A. Find Your App URL
1. In your Railway project dashboard
2. Click on your service
3. Go to **"Settings"** tab
4. Scroll to **"Networking"** section
5. Click **"Generate Domain"**
6. Railway will create a URL like:
   ```
   https://fake-job-detection-production-xxxx.up.railway.app
   ```
7. **Copy this URL** - this is your live app!

### B. Test Your Deployment
Open your browser or use curl:

**Test 1: Health Check**
```bash
curl https://your-app-name.up.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Test 2: Predict with Text**
```bash
curl -X POST https://your-app-name.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "We are hiring software engineer with python experience"}'
```

**Test 3: Open in Browser**
```
https://your-app-name.up.railway.app/
```

---

## 🔧 STEP 7: Important Railway Settings

### A. Resource Limits (Free Tier)
Railway free tier includes:
- ✅ $5 free credits per month
- ✅ 512 MB RAM
- ✅ Shared CPU
- ✅ 1 GB disk space

**Your app uses:**
- ~450-500 MB RAM (DistilBERT model)
- ~800 MB disk (models + dependencies)

⚠️ **Note**: Free tier should work, but may sleep after inactivity.

### B. Upgrade to Hobby Plan (Optional)
If free tier has issues:
- **Hobby Plan**: $5/month
- 8 GB RAM
- More reliable uptime
- No sleep mode

### C. Environment Variables You Can Set

In Railway Variables tab:

| Variable | Value | Purpose |
|----------|-------|---------|
| `PORT` | Auto-set by Railway | Don't change this |
| `THRESHOLD` | `0.5` | Prediction threshold (0-1) |
| `FLASK_ENV` | `production` | Disable debug mode |

---

## 🐛 STEP 8: Troubleshooting

### Issue 1: Build Fails
**Check Build Logs for:**
- ❌ Python version mismatch
  - **Fix**: Edit `runtime.txt` to `python-3.11.0`
- ❌ Missing dependencies
  - **Fix**: Check `requirements.txt` has all packages
- ❌ Out of memory
  - **Fix**: Upgrade to Hobby plan

### Issue 2: Deployment Succeeds But App Doesn't Respond
**Check Deploy Logs for:**
- ❌ Port binding error
  - **Fix**: Ensure `Procfile` uses `$PORT`
- ❌ Model files missing
  - **Fix**: Ensure `models/` folder is committed to git
- ❌ Import errors
  - **Fix**: Check all Python files are in repository

### Issue 3: "Model not found" Error
**Fix:**
1. Verify `models/` folder exists in GitHub
2. Check these files exist:
   - `models/rf_model_calibrated.joblib`
   - `models/scaler.joblib`
   - `models/feature_info.joblib`
   - `models/tokenizer/` (folder)
3. Redeploy:
   ```powershell
   git add models/
   git commit -m "Add model files"
   git push
   ```

### Issue 4: Selenium/ChromeDriver Errors
Railway provides Chrome automatically, but if issues:
1. Go to Railway dashboard → Settings
2. Add **Nixpacks** configuration (Railway docs)
3. Or disable Selenium features for deployment

---

## 🔄 STEP 9: Update Your Deployed App

When you make changes locally:

```powershell
# 1. Make your changes in code
# 2. Test locally
python app.py

# 3. Commit changes
git add .
git commit -m "Description of changes"

# 4. Push to GitHub
git push

# 5. Railway auto-deploys!
# Watch the deployment in Railway dashboard
```

Railway automatically redeploys when you push to GitHub! 🎉

---

## 📊 STEP 10: Monitor Your App

### A. View Logs
1. Railway dashboard → Your service
2. Click **"View Logs"**
3. See real-time application logs

### B. Check Metrics
1. Go to **"Metrics"** tab
2. Monitor:
   - CPU usage
   - Memory usage
   - Network traffic
   - Response times

### C. Set Up Alerts (Optional)
1. Settings → Notifications
2. Get alerts for:
   - Deployment failures
   - High resource usage
   - App crashes

---

## 🎯 Quick Reference Commands

### Local Development
```powershell
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

### Git Commands
```powershell
# Check status
git status

# Add changes
git add .

# Commit
git commit -m "Your message"

# Push to GitHub (triggers Railway deployment)
git push
```

### Testing Deployed API
```bash
# Health check
curl https://your-app.up.railway.app/health

# Predict with text
curl -X POST https://your-app.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "job description"}'

# Predict with URL
curl -X POST https://your-app.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "https://linkedin.com/jobs/view/12345"}'
```

---

## ✅ Deployment Checklist

Before deploying, make sure:
- [ ] All test files removed
- [ ] `venv/` in `.gitignore`
- [ ] `Procfile` created
- [ ] `requirements.txt` includes `gunicorn`
- [ ] `runtime.txt` specifies Python version
- [ ] `models/` folder committed to git
- [ ] Code pushed to GitHub
- [ ] Railway account created
- [ ] GitHub connected to Railway
- [ ] Environment variables set (if any)
- [ ] Domain generated in Railway
- [ ] Health check endpoint tested
- [ ] Prediction endpoint tested

---

## 🎓 Summary

1. ✅ **Cleaned** unnecessary files
2. ✅ **Created** deployment files (Procfile, runtime.txt)
3. 📤 **Push** to GitHub
4. 🚂 **Deploy** on Railway from GitHub repo
5. 🌐 **Generate** domain in Railway settings
6. ✅ **Test** your live API
7. 🔄 **Update** by pushing to GitHub (auto-deploys)

**Your app will be live at**: `https://your-app-name.up.railway.app` 🎉

---

## 💡 Pro Tips

1. **Use Railway CLI** (optional):
   ```bash
   npm i -g @railway/cli
   railway login
   railway logs
   ```

2. **Database** (if needed in future):
   - Railway → Add PostgreSQL
   - Connection string auto-added to env vars

3. **Custom Domain** (optional):
   - Settings → Networking → Custom Domain
   - Add your own domain (e.g., fakejobdetector.com)

4. **Cost Management**:
   - Monitor usage in Railway dashboard
   - Set spending limits in account settings
   - Free tier: $5/month credits

---

## 📞 Need Help?

- **Railway Docs**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **Check logs**: Railway dashboard → View Logs

Good luck with your first deployment! 🚀
