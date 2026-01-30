"""
Environment verification script for handwritten OCR setup.
Run with: python verify_env.py (from project root, with ocr_env activated)
"""
import sys

def check(name, import_fn):
    """Try importing and report success or failure."""
    try:
        import_fn()
        print(f"[OK] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False

def main():
    print("Python:", sys.version)
    print("-" * 50)
    results = []

    results.append(check("torch", lambda: __import__("torch")))
    results.append(check("transformers", lambda: __import__("transformers")))
    results.append(check("cv2 (opencv-python)", lambda: __import__("cv2")))
    results.append(check("layoutparser", lambda: __import__("layoutparser")))

    print("-" * 50)
    if all(results):
        print("All required imports succeeded.")
        return 0
    print("One or more imports failed.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
