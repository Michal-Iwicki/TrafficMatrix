import sys
import subprocess
from pathlib import Path

def main():
    # Get the absolute path of the directory containing this script (project root)
    root_dir = Path(__file__).parent.absolute()
    
    # Define the path to the main Streamlit application file
    app_path = root_dir / "streamlit_app" / "app.py"
    
    # Check if the application file exists in the specified location
    if not app_path.exists():
        print(f"Error: Application file not found at path: {app_path}")
        sys.exit(1)
        
    print("🚀 Starting the Streamlit application...")
    print(f"Working directory: {root_dir}")
    
    # sys.executable ensures the script uses the same Python interpreter 
    # (and the same virtual environment) that was used to run this wrapper script
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    
    try:
        # Run the Streamlit process
        subprocess.run(command, cwd=root_dir, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Application closed gracefully.")
    except Exception as e:
        print(f"\n❌ An error occurred while starting the application: {e}")

if __name__ == "__main__":
    main()