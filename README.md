# NEPSE Trend Predictor - Web App

LSTM-based NEPSE stock trend predictor with a full web dashboard.

## Deploy Online FREE on Render.com

### Step 1: Upload to GitHub
1. Go to https://github.com and create an account
2. Click "New Repository" → name it `nepse-predictor`
3. Upload all files from this folder

### Step 2: Deploy on Render
1. Go to https://render.com and sign up (free)
2. Click "New" → "Web Service"
3. Connect your GitHub repo
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
5. Click "Deploy"
6. Wait 5-10 minutes → your app is LIVE!

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Then open: http://localhost:5000
