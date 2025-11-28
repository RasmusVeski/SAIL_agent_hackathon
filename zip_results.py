import os
import zipfile
import datetime
import sys

def package_submission():
    # 1. Setup Paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"submission_{timestamp}.zip"
    
    # Folders we want to include (relative to root)
    targets = [
        "logs",
        "app/services"
    ]
    
    # Files to exclude
    exclusions = {".DS_Store", "__pycache__", ".pyc", ".git"}

    print(f"📦 Packaging submission into '{zip_filename}'...")
    
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for target in targets:
                if not os.path.exists(target):
                    print(f"⚠️  Warning: Target '{target}' not found. Skipping.")
                    continue
                
                # Walk through the directory
                for root, dirs, files in os.walk(target):
                    # Modify dirs in-place to skip __pycache__ during traversal
                    dirs[:] = [d for d in dirs if d not in exclusions]
                    
                    for file in files:
                        if file in exclusions or file.endswith(".pyc"):
                            continue
                            
                        # Absolute path on disk
                        file_path = os.path.join(root, file)
                        
                        # Relative path inside the zip (keeps structure clean)
                        # e.g. app/services/agents/main.py
                        zipf.write(file_path, file_path)
                        # print(f"  + Added: {file_path}") # Uncomment for verbose
        
        print("-" * 40)
        print(f"✅ Success! Created: {zip_filename}")
        print(f"   Size: {os.path.getsize(zip_filename) / 1024:.2f} KB")
        print("-" * 40)
        print("Please upload this file to the submission portal.")

    except Exception as e:
        print(f"❌ Error creating zip: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure we are running from the project root
    if not os.path.exists("app"):
        print("❌ Error: Please run this script from the project root directory.")
        sys.exit(1)
        
    package_submission()