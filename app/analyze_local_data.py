import os
import sys
from dotenv import load_dotenv

# Load env to get the correct path
load_dotenv()

def analyze_local_data():
    # 1. Construct Path
    base_dir = os.getenv("BASE_DATA_DIR", "./app/sharded_data")
    data_dir_name = os.getenv("DATA_DIR_NAME", "client_0")  # Default to client_0
    target_path = os.path.join(base_dir, data_dir_name)

    print(f"🔍 INSPECTING: {target_path}")

    # 2. Check if Path Exists
    if not os.path.exists(target_path):
        print("❌ ERROR: Directory does not exist.")
        return

    contents = os.listdir(target_path)
    if not contents:
        print("⚠️  WARNING: Directory is EMPTY (0 files/folders).")
        return

    # 3. Detect class folders
    class_folders = [
        d for d in contents
        if os.path.isdir(os.path.join(target_path, d))
    ]

    if not class_folders:
        print("⚠️  WARNING: Directory contains files but NO class folders.")
        print(f"   Contents: {contents[:5]}...")
        return

    # Try to sort numerically when possible
    try:
        class_folders.sort(key=int)
    except ValueError:
        class_folders.sort()

    total_global_classes = 40
    total_local_folders = len(class_folders)

    print("-" * 50)
    print(f"📁 CLASS FOLDERS FOUND: {total_local_folders}")
    print("-" * 50)
    print("📁 CLASS DISTRIBUTION:")
    print()

    empty_classes = []
    nonempty_classes = []
    total_images = 0

    # 4. Print per-class counts and track nonempty classes
    for cls in class_folders:
        cls_path = os.path.join(target_path, cls)
        images = [
            f for f in os.listdir(cls_path)
            if os.path.isfile(os.path.join(cls_path, f))
        ]
        count = len(images)
        total_images += count

        if count > 0:
            nonempty_classes.append(cls)
            status = f"{count} images"
        else:
            empty_classes.append(cls)
            status = "EMPTY"

        print(f"  • Class {cls:>2}: {status}")

    print("-" * 50)
    print(f"🖼️ TOTAL IMAGES: {total_images}")
    print(f"📊 CLASSES WITH DATA: {len(nonempty_classes)} / {total_global_classes}")
    print()

    if empty_classes:
        print(f"⚠️ EMPTY CLASS FOLDERS: {empty_classes}")
    else:
        print("✨ All class folders contain data.")

if __name__ == "__main__":
    analyze_local_data()
