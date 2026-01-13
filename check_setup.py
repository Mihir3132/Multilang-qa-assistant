"""Quick setup verification script for the PDF QA project."""

import sys
from pathlib import Path

print("=" * 60)
print("PDF QA Project - Setup Verification")
print("=" * 60)

# Check Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# Check required packages
required_packages = [
    "langchain",
    "langchain_community",
    "langchain_huggingface",
    "chromadb",
    "fitz",  # PyMuPDF
    "pytesseract",
    "PIL",  # Pillow
    "sentence_transformers",
    "streamlit",
    "ollama",
]

print("\n📦 Checking Python packages...")
missing = []
for pkg in required_packages:
    try:
        if pkg == "fitz":
            import fitz
        elif pkg == "PIL":
            from PIL import Image
        else:
            __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} - MISSING")
        missing.append(pkg)

if missing:
    print(f"\n⚠ Missing packages: {', '.join(missing)}")
    print("  Run: pip install -r requirements.txt")
else:
    print("\n✓ All required packages installed")

# Check Tesseract
print("\n🔍 Checking Tesseract OCR...")
try:
    import pytesseract
    
    # Try to set from config if available
    try:
        from config import settings
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
            print(f"  Using Tesseract from: {settings.tesseract_cmd}")
    except Exception:
        pass
    
    version = pytesseract.get_tesseract_version()
    print(f"  ✓ Tesseract version: {version}")
    
    # Check for Hindi language pack
    try:
        langs = pytesseract.get_languages()
        if "hin" in langs:
            print(f"  ✓ Hindi language pack found")
        else:
            print(f"  ⚠ Hindi language pack not found (OCR for Hindi may not work)")
        if "eng" in langs:
            print(f"  ✓ English language pack found")
    except Exception:
        print(f"  ⚠ Could not check language packs")
except Exception as e:
    print(f"  ✗ Tesseract not found: {e}")
    print("  ⚠ OCR features will be disabled. Text-based PDFs will still work.")
    print("  Install from: https://github.com/UB-Mannheim/tesseract/wiki")

# Check .env file
print("\n⚙️  Checking configuration...")
env_file = Path(".env")
if env_file.exists():
    print("  ✓ .env file found")
    with open(env_file) as f:
        content = f.read()
        if "CHROMA_DB_DIR" in content:
            print("  ✓ CHROMA_DB_DIR configured")
        if "OLLAMA_MODEL" in content:
            print("  ✓ OLLAMA_MODEL configured")
        if "TESSERACT_CMD" in content:
            print("  ✓ TESSERACT_CMD configured")
else:
    print("  ⚠ .env file not found")
    print("  Create .env with:")
    print("    CHROMA_DB_DIR=./chroma_db")
    print("    OLLAMA_MODEL=llama3.2:3b-instruct-q4_K_S")

# Check Ollama (optional check)
print("\n🤖 Checking Ollama...")
try:
    import subprocess
    result = subprocess.run(
        ["ollama", "--version"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        print(f"  ✓ Ollama found: {result.stdout.strip()}")
        
        # Try to check if model is available
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "llama3.2" in result.stdout.lower() or "3b" in result.stdout.lower():
            print("  ✓ LLaMA 3.2 3B model appears to be available")
        else:
            print("  ⚠ LLaMA 3.2 3B model not found. Run: ollama pull llama3.2:3b-instruct-q4_K_S")
    else:
        print("  ✗ Ollama not found or not working")
except FileNotFoundError:
    print("  ✗ Ollama not found in PATH")
    print("  Install from: https://ollama.com/download")
except Exception as e:
    print(f"  ⚠ Could not verify Ollama: {e}")

print("\n" + "=" * 60)
print("Setup check complete!")
print("=" * 60)
print("\nTo run the app:")
print("  streamlit run app.py")
print("  or")
print("  python -m streamlit run app.py")
