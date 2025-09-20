#!/usr/bin/env python3
"""
Simple script to run data fetching and model training
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the data fetching and training process"""
    print("=" * 60)
    print("Smart Health Surveillance - Data Fetching & Training")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("app").exists():
        print("Error: Please run this script from the SIH_Backend directory")
        sys.exit(1)
    
    try:
        print("Starting data fetching and model training...")
        print("This may take several minutes as we fetch real data from various sources...")
        print()
        
        # Run the comprehensive data fetching and training script
        result = subprocess.run([
            sys.executable, 
            "scripts/run_data_fetching_and_training.py"
        ], check=True, capture_output=True, text=True)
        
        print("Data fetching and training completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        
        if result.stderr:
            print("\nWarnings/Errors:")
            print(result.stderr)
        
        print("\nNext steps:")
        print("1. Start the services: ./scripts/start_services.sh")
        print("2. Test the API: python scripts/test_api.py")
        print("3. View API docs: http://localhost:8000/docs")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running data fetching and training: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
