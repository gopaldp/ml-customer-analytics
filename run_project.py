import subprocess
import sys

def run_pipeline():
    """Run the complete ML pipeline"""
    
    print("🚀 Starting ML Customer Analytics Pipeline...")
    
    # Step 1: Generate data
    print("\n📊 Generating synthetic data...")
    subprocess.run([sys.executable, "src/data_generation.py"])
    
    # Step 2: Preprocess data
    print("\n🔧 Preprocessing data...")
    subprocess.run([sys.executable, "src/data_preprocessing.py"])
    
    # Step 3: Train models
    print("\n🤖 Training ML models...")
    subprocess.run([sys.executable, "src/models.py"])
    
    # Step 4: Launch dashboard
    print("\n🎨 Launching dashboard...")
    print("Dashboard will open at: http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])

if __name__ == "__main__":
    run_pipeline()
