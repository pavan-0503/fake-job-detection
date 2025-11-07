# Railway Deployment Setup Guide

## ✅ Problem: "Model not found" Error

Your app is deploying successfully, but showing "Model not found. Please train the model first by running train_model.py" because the models need to be downloaded from Google Drive at startup.

## 🔧 Solution: Set Environment Variable in Railway

### Step 1: Go to Railway Dashboard
1. Open your Railway project: https://railway.app/
2. Click on your project: **fake-job-detection**
3. Click on your service: **web**

### Step 2: Add Environment Variable
1. Click on the **Variables** tab (in the top menu)
2. Click **+ New Variable** button
3. Add the variable:
   - **Variable Name**: `GOOGLE_DRIVE_MODEL_ID`
   - **Variable Value**: `16cFNpCAVWqM_qZDuZYbFi_iEmWjkyjin`
4. Click **Add** or press Enter

### Step 3: Redeploy
Railway will automatically redeploy with the new environment variable.

**Expected deployment logs after fix:**
```
🔍 Checking for models...
⚠️  Models not found locally. Attempting download...
📥 Downloading from Google Drive (File ID: 16cFNpCAVWqM_qZDuZYbFi_iEmWjkyjin)...
🔄 Attempting to download from Google Drive (ID: 16cFNpCAV...)
📦 Downloading models (XX.XX MB)...
✅ Download complete! (XX.XX MB)
Extracting models...
Extraction complete!
Cleaned up zip file
✅ Models downloaded and extracted successfully!
✅ Models ready!
```

## 📋 Verification Steps

After Railway redeploys:

1. **Check Deploy Logs**: Look for "✅ Models ready!" message
2. **Test Health Endpoint**: Visit `https://your-app.up.railway.app/health`
   - Should return: `{"status": "healthy", "model_loaded": true}`
3. **Test Prediction**: Use the web interface to analyze a job posting

## 🔍 Troubleshooting

### If download still fails:

1. **Verify Google Drive File Sharing**:
   - File ID: `16cFNpCAVWqM_qZDuZYbFi_iEmWjkyjin`
   - Share link: https://drive.google.com/file/d/16cFNpCAVWqM_qZDuZYbFi_iEmWjkyjin/view
   - Check that sharing is set to: **"Anyone with the link"**
   - File should be a ZIP file containing the models

2. **Check Railway Logs**:
   - Go to Railway → Your Service → **Deployments** tab
   - Click on the latest deployment
   - Check logs for error messages starting with "❌"

3. **Common Issues**:
   - **"ERROR: Google Drive file ID not provided!"** 
     → Environment variable not set correctly
   
   - **"Received HTML instead of file"**
     → Google Drive file not publicly accessible
   
   - **"Downloaded file too small"**
     → File may be restricted or ID incorrect

## 📦 Models Included

The `models.zip` file should contain:
- `models/rf_model_calibrated.joblib` - Trained Random Forest model
- `models/scaler.joblib` - Feature scaler
- `models/feature_info.joblib` - Feature names and metadata
- `models/tokenizer/` - DistilBERT tokenizer files

## 🚀 Alternative: Railway Volumes (For Future)

For better performance, you can use Railway Volumes to persist models:
1. Create a volume in Railway
2. Mount it to `/app/models`
3. Models will persist between deployments (faster startup)

However, the Google Drive approach works fine for initial setup.

## ✅ Success Indicators

Your app is working when you see:
- ✅ Deployment shows "Running" status
- ✅ Logs show "✅ Models ready!"
- ✅ `/health` endpoint returns model_loaded: true
- ✅ Web interface shows "Analyze Job Posting" form (not error message)
- ✅ Predictions work correctly
