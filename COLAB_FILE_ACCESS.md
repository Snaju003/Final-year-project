# Google Colab Setup - File Access Methods

When you see this error in Colab:
```
fatal: could not read Username for 'https://github.com': No such device or address
[Errno 2] No such file or directory: 'Final-year-project'
```

It means the notebook can't access your GitHub repository. Here are solutions:

---

## **Solution 1: Make Repository Public (Recommended)**

If your repository is currently **private**:

```bash
# On your local machine:
git remote -v  # Check your remote URL

# Go to GitHub.com → Your repository → Settings → Visibility
# Change from Private → Public

# Then push any latest changes:
git push origin main
```

**Then in Colab:** Just run the cells normally - they'll clone successfully.

---

## **Solution 2: Upload Project to Google Drive (No Network Needed)**

This is the **most reliable method** if your repo is private or network is unstable.

### Step-by-Step:

1. **On your local machine**, zip your entire project:
   ```bash
   # Windows PowerShell
   Compress-Archive -Path "E:\Final-year-project" -DestinationPath "Final-year-project.zip"
   
   # Or use 7-Zip / WinRAR GUI
   ```

2. **Upload to Google Drive:**
   - Go to https://drive.google.com/
   - Create new folder: `MyDrive/Final-year-project-backup`
   - Upload the ZIP file
   - Right-click ZIP → Open with → Google Drive (to preview)

3. **In Colab cell, use this code:**
   ```python
   from google.colab import drive
   import shutil
   import zipfile
   
   # Mount Drive
   drive.mount('/content/drive', force_remount=True)
   
   # Extract ZIP
   zip_path = '/content/drive/MyDrive/Final-year-project.zip'
   extract_path = '/content/Final-year-project'
   
   with zipfile.ZipFile(zip_path, 'r') as zip_ref:
       zip_ref.extractall('/content')
   
   %cd /content/Final-year-project
   print("✅ Project extracted and ready")
   ```

---

## **Solution 3: Upload Project Folder Directly (Easiest)**

**Don't need ZIP file? Use this instead:**

1. **In Google Drive:**
   - Upload the entire `Final-year-project` folder to `MyDrive/`

2. **In Colab:**
   ```python
   from google.colab import drive
   import shutil
   
   # Mount Drive
   drive.mount('/content/drive', force_remount=True)
   
   # Copy from Drive
   shutil.copytree(
       '/content/drive/MyDrive/Final-year-project',
       '/content/Final-year-project'
   )
   
   %cd /content/Final-year-project
   print("✅ Project ready")
   ```

---

## **Solution 4: Use GitHub Token (For Private Repos)**

If you want to keep repo private but still clone from Colab:

1. **Create GitHub Personal Access Token:**
   - Go to https://github.com/settings/tokens
   - Click "Generate new token"
   - Select `repo` scope
   - Copy the token

2. **In Colab:**
   ```python
   # Clone with token
   token = "ghp_your_token_here"
   !git clone https://{token}@github.com/Snaju003/Final-year-project.git
   ```

3. **⚠️ SECURITY WARNING:**
   - Never commit token to public repo
   - Use Colab secrets instead:
     - Click 🔑 icon in Colab left panel
     - Add secret named `GITHUB_TOKEN`
     - Access via: `from google.colab import userdata; token = userdata.get('GITHUB_TOKEN')`

---

## **Solution 5: Use Gradio Share / Google Colab Share**

For easy sharing without files:
```python
# In Colab, share a notebook link directly
# File → Share → Get link (anyone with link can view)
```

---

## **Quick Comparison**

| Method | Pros | Cons | Speed |
|--------|------|------|-------|
| **Public GitHub** | Easy, always synced | Must be public | Fast |
| **Google Drive ZIP** | Secure, reliable | Manual upload | Fast |
| **Drive Folder** | Simple setup | Large uploads | Fast |
| **GitHub Token** | Private repo works | Security risk | Fast |
| **Direct Upload** | No Drive needed | Size limited | Medium |

---

## **Recommended Setup for Your Project**

Since your repo might be private, I recommend:

1. **Upload to Google Drive** (most reliable):
   ```bash
   # Step 1: On your machine, zip the project
   Compress-Archive -Path "E:\Final-year-project" -DestinationPath "fyp.zip"
   
   # Step 2: Upload fyp.zip to Google Drive
   # Step 3: In Colab, extract it
   ```

2. **Or make repo public** if you're okay with that:
   - Settings → Visibility → Public
   - Then just use the normal clone in Colab

---

## **Updated Notebook Cells (Auto-Detect)**

The notebook now has **auto-detection logic**:

```python
# METHOD 1: Try GitHub
try:
    !git clone https://github.com/Snaju003/Final-year-project.git
    %cd Final-year-project
except:
    # METHOD 2: Fall back to Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    # Copy from Drive...
```

---

## **Troubleshooting**

### Q: "No such device or address"
**A:** GitHub is unreachable. Use Google Drive method instead.

### Q: "Directory not found"
**A:** Project wasn't uploaded to Drive. Check path: `/content/drive/MyDrive/Final-year-project/`

### Q: "Permission denied"
**A:** Make sure you're in the correct directory. Use `%cd` to change.

### Q: "Module not found (src.inference)"
**A:** You're not in the project root. Check `os.getcwd()` returns the right path.

---

## **Verify Setup is Working**

Run this to confirm:
```python
import os
print("Current dir:", os.getcwd())
print("Files:", os.listdir('.')[:10])
print("Has src/:", os.path.exists('src'))
print("Has models/:", os.path.exists('models'))
print("Has requirements.txt:", os.path.exists('requirements.txt'))
```

All should return `True` ✅

---

**Still stuck?** Check:
1. Is the folder named exactly `Final-year-project`?
2. Did you extract the ZIP correctly?
3. Is the path `/content/drive/MyDrive/Final-year-project/` correct?

Let me know if you need help with any specific method!
