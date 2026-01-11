import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    print("🚀 Setting up environment...")
    
    # Check if virtual environment exists
    if not Path("venv").exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Get the pip executable in the venv
    if os.name == 'nt': # Windows
        pip_path = Path("venv/Scripts/pip")
        python_path = Path("venv/Scripts/python")
    else: # Unix/Mac
        pip_path = Path("venv/bin/pip")
        python_path = Path("venv/bin/python")
    
    if not pip_path.exists():
        print("❌ Error: Could not find pip in venv. Please create venv manually.")
        return False

    print("📥 Installing dependencies from requirements.txt...")
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"])
    
    print("\n✅ Setup complete!")
    return str(python_path)

def launch_gui(python_path):
    print("🎨 Checking system dependencies...")
    
    # Critical modules to verify
    critical_modules = ["torch", "torchvision", "streamlit", "PIL", "yaml", "cv2"]
    missing_modules = []
    
    for module in critical_modules:
        try:
            # Handle PIL vs Pillow and opencv vs cv2
            check_name = "PIL" if module == "pillow" else ("cv2" if module == "opencv-python" else module)
            subprocess.run([python_path, "-c", f"import {check_name}"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_modules.append(module)

    if missing_modules:
        print(f"📦 Missing modules found: {', '.join(missing_modules)}")
        print("📥 Installing all dependencies from requirements.txt...")
        try:
            subprocess.run([python_path, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ All dependencies installed successfully.")
        except subprocess.CalledProcessError:
            print("❌ Error: Failed to install dependencies. Please try manually: pip install -r requirements.txt")
            return

    gui_script = Path("code/Combined-Inference/gui_streamlit.py")
    
    if not gui_script.exists():
        print(f"❌ Error: Could not find {gui_script}")
        return
    
    # Run the Streamlit GUI
    print("🚀 Launching GUI...")
    subprocess.run([python_path, "-m", "streamlit", "run", str(gui_script)])

if __name__ == "__main__":
    # Check if requirements already installed in current env or venv
    # For simplicity, we'll try to use the venv if it exists, otherwise offer to set it up
    
    venv_python = "venv/bin/python" if os.name != 'nt' else "venv/Scripts/python"
    
    if Path(venv_python).exists():
        launch_gui(venv_python)
    else:
        print("Welcome to the Smartphone-DSLR Image Pipeline!")
        choice = input("Would you like to auto-setup the environment? (y/n): ")
        if choice.lower() == 'y':
            python_path = setup_environment()
            if python_path:
                launch_gui(python_path)
        else:
            print("Please install the dependencies manually using:")
            print("pip install -r requirements.txt")
            print("\nThen run the GUI with:")
            print("python3 -m streamlit run code/Combined-Inference/gui_streamlit.py")
