# 🚀 Railway Deployment with Google Drive Models

## 📤 **STEP 1: Upload Models to Google Drive**

### A. Models Already Compressed ✅
Your `models.zip` file is ready in the project folder!

### B. Upload to Google Drive

1. **Go to**: https://drive.google.com
2. **Click**: "New" → "File upload"
3. **Select**: `models.zip` (in your project folder)
4. **Wait** for upload to complete (~50 MB)

### C. Get Shareable Link

1. **Right-click** on uploaded `models.zip`
2. **Click**: "Share"
3. **Change access** to: **"Anyone with the link"** → **"Viewer"**
4. **Copy** the link (looks like this):
   ```
   https://drive.google.com/file/d/1ABC123XYZ456DEF789GHI/view?usp=sharing
   ```

### D. Extract FILE_ID

From the link above, copy the FILE_ID:
```
https://drive.google.com/file/d/1ABC123XYZ456DEF789GHI/view?usp=sharing
                                 ^^^^^^^^^^^^^^^^^^^^^
                                 This is your FILE_ID
```

**Example:**
- Link: `https://drive.google.com/file/d/1kT9xB2pL7mN3vQ8wR4sY6uZ/view?usp=sharing`
- FILE_ID: `1kT9xB2pL7mN3vQ8wR4sY6uZ`

---

## 🗑️ **STEP 2: Remove Models from Git**

Models are too large for Railway's build timeout. Remove them:

```powershell
cd "c:\Users\pavan\fake job detection"

# Remove models from git tracking
git rm -r --cached models/
git rm --cached merged_job_postings.csv
git rm --cached models.zip

# Commit changes
git add .
git commit -m "Remove large files, add model downloader for Railway"
git push
```

---

## ⚙️ **STEP 3: Configure Railway Environment Variable**

1. **Go to Railway Dashboard**: https://railway.app
2. **Click** on your `fake-job-detection` project
3. **Click** on your service (web)
4. **Go to** "Variables" tab
5. **Click** "New Variable"
6. **Add**:
   - **Name**: `GOOGLE_DRIVE_MODEL_ID`
   - **Value**: `YOUR_FILE_ID_HERE` (paste the FILE_ID from Step 1D)
7. **Click** "Add"

**Example:**
```
Name:  GOOGLE_DRIVE_MODEL_ID
Value: 1kT9xB2pL7mN3vQ8wR4sY6uZ
```

---

## 🚂 **STEP 4: Deploy on Railway**

Railway will automatically redeploy after you push the changes.

### What Happens:
1. **Build Phase** (5-8 minutes):
   - Installs Python and dependencies
   - Much faster without large model files!

2. **Deploy Phase** (2-3 minutes):
   - App starts
   - Checks for models locally (not found)
   - Downloads `models.zip` from Google Drive (50 MB)
   - Extracts models
   - Starts serving requests

### Watch the Logs:
In Railway dashboard → View Logs, you'll see:
```
🔍 Checking for models...
⚠ Models not found locally
Downloading models (50.85 MB)...
████████████████████ 100%
Download complete!
Extracting models...
Extraction complete!
✅ Models ready!
[INFO] Starting gunicorn...
```

---

## ✅ **STEP 5: Test Your Deployment**

After deployment succeeds:

### Health Check:
```bash
curl https://your-app-name.up.railway.app/health
```

Expected:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Prediction Test:
```bash
curl -X POST https://your-app-name.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "We are hiring software engineer"}'
```

---

## 🔧 **Troubleshooting**

### Issue 1: "Failed to download models"
**Cause**: Wrong FILE_ID or Google Drive link not public

**Fix**:
1. Check FILE_ID is correct
2. Ensure Google Drive file is shared as "Anyone with the link"
3. Check Railway logs for exact error
4. Update `GOOGLE_DRIVE_MODEL_ID` environment variable

### Issue 2: "Models not found" error
**Cause**: Download failed during startup

**Fix**:
1. Check Railway logs for download errors
2. Verify Google Drive link works (try in browser)
3. Redeploy: Railway → Deployments → "Redeploy"

### Issue 3: Download takes too long
**Cause**: Large model file (50 MB)

**Solution**: This is normal! First startup takes 2-3 minutes to download models.
Subsequent requests will be fast since models are cached.

---

## 📊 **Summary**

| Step | Time | What Happens |
|------|------|--------------|
| 1. Upload to Google Drive | 2 min | Upload models.zip (50 MB) |
| 2. Remove from Git | 1 min | Commit and push |
| 3. Set Environment Variable | 1 min | Add GOOGLE_DRIVE_MODEL_ID to Railway |
| 4. Build | 5-8 min | Railway installs dependencies (faster!) |
| 5. First Deploy | 2-3 min | Downloads models from Google Drive |
| **Total** | **~15 min** | App is live! |

---

## 💡 **Benefits of This Approach**

✅ **Fast Builds**: No 50 MB upload to Railway  
✅ **Free Tier Compatible**: Stays within Railway limits  
✅ **Easy Updates**: Update models in Google Drive, redeploy app  
✅ **No Timeouts**: Build completes in <10 minutes  

---

## 🎯 **Quick Checklist**

- [ ] `models.zip` uploaded to Google Drive
- [ ] Google Drive link set to "Anyone with the link"
- [ ] FILE_ID extracted from share link
- [ ] Models removed from git (`git rm -r --cached models/`)
- [ ] Changes committed and pushed
- [ ] `GOOGLE_DRIVE_MODEL_ID` environment variable set in Railway
- [ ] Railway deployment succeeded
- [ ] Health check returns `model_loaded: true`
- [ ] Prediction endpoint tested
- [ ] App is live! 🎉

---

**Your app will be live at**: `https://fake-job-detection-production.up.railway.app`

Good luck! 🚀
