# How to Create a GitHub Repository for This Project

Follow these steps to create a GitHub repository and push your code:

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `multilingual-pdf-qa` (or your preferred name)
   - **Description**: "Multilingual PDF Q&A system with Hindi/English/Hinglish support using RAG"
   - **Visibility**: Choose **Public** or **Private**
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

## Step 2: Initialize Git in Your Project

Open PowerShell in your project directory and run:

```powershell
# Navigate to your project directory
cd C:\Users\mihir\Desktop\bisag_project

# Initialize git repository
git init

# Add all files (respecting .gitignore)
git add .

# Make your first commit
git commit -m "Initial commit: Multilingual PDF QA system with RAG"
```

## Step 3: Connect to GitHub and Push

```powershell
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/multilingual-pdf-qa.git

# Rename main branch (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Note**: If you're using SSH instead of HTTPS:
```powershell
git remote add origin git@github.com:YOUR_USERNAME/multilingual-pdf-qa.git
```

## Step 4: Verify

1. Go to your GitHub repository page
2. You should see all your files uploaded
3. The README.md will be displayed on the repository homepage

## What's Included in .gitignore

The `.gitignore` file excludes:
- Virtual environment (`venv/`)
- Environment variables (`.env`)
- ChromaDB data (`chroma_db/`)
- Python cache files (`__pycache__/`)
- Tesseract binaries (large files)
- IDE configuration files
- Temporary files

## Future Updates

To push future changes:

```powershell
git add .
git commit -m "Description of your changes"
git push
```

## Optional: Add a License

If you want to add a license:

1. Go to your repository on GitHub
2. Click **"Add file"** → **"Create new file"**
3. Name it `LICENSE`
4. GitHub will suggest templates - choose one (e.g., MIT, Apache 2.0)
5. Commit the file

## Troubleshooting

### If you get authentication errors:
- Use a Personal Access Token instead of password
- Or set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

### If you need to update .gitignore:
```powershell
# Remove files from git cache if they were already tracked
git rm -r --cached .
git add .
git commit -m "Update .gitignore"
git push
```
