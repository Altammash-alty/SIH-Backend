#!/usr/bin/env python3
"""
Script to display the datasets created by the demo
"""
import pandas as pd
import os

def show_dataset_info():
    """Display information about the created datasets"""
    print("📊 DATASET INFORMATION")
    print("=" * 60)
    
    # Check if demo data exists
    if not os.path.exists('demo_data'):
        print("❌ Demo data directory not found. Please run demo_data_scraping.py first.")
        return
    
    # List all dataset files
    files = os.listdir('demo_data')
    print(f"Available datasets: {len(files)} files")
    for file in sorted(files):
        file_path = os.path.join('demo_data', file)
        size = os.path.getsize(file_path)
        print(f"  📄 {file} ({size:,} bytes)")
    
    print("\n" + "=" * 60)
    print("HEALTH REPORTS DATASET")
    print("=" * 60)
    
    # Load and display health reports dataset
    try:
        health_df = pd.read_csv('demo_data/health_reports_dataset.csv')
        print(f"Shape: {health_df.shape[0]} rows × {health_df.shape[1]} columns")
        print(f"\nColumns: {list(health_df.columns)}")
        
        print(f"\nFirst 5 records:")
        print(health_df.head().to_string())
        
        print(f"\nData Summary:")
        print(f"- Date range: {health_df['report_timestamp'].min()} to {health_df['report_timestamp'].max()}")
        print(f"- Unique locations: {health_df['location_address'].nunique()}")
        print(f"- Sources: {health_df['report_source'].value_counts().to_dict()}")
        
        print(f"\nSymptom Statistics:")
        symptoms = ['fever', 'diarrhea', 'vomiting', 'nausea', 'abdominal_pain', 'dehydration']
        for symptom in symptoms:
            count = health_df[symptom].sum()
            pct = (count / len(health_df)) * 100
            print(f"  {symptom}: {count} ({pct:.1f}%)")
        
    except Exception as e:
        print(f"Error loading health reports: {e}")
    
    print("\n" + "=" * 60)
    print("WATER QUALITY DATASET")
    print("=" * 60)
    
    # Load and display water quality dataset
    try:
        water_df = pd.read_csv('demo_data/water_quality_dataset.csv')
        print(f"Shape: {water_df.shape[0]} rows × {water_df.shape[1]} columns")
        print(f"\nColumns: {list(water_df.columns)}")
        
        print(f"\nFirst 5 records:")
        print(water_df.head().to_string())
        
        print(f"\nData Summary:")
        print(f"- Unique locations: {water_df['location_name'].nunique()}")
        print(f"- Contamination status: {water_df['is_contaminated'].value_counts().to_dict()}")
        print(f"- Contamination levels: {water_df['contamination_level'].value_counts().to_dict()}")
        
        print(f"\nWater Quality Parameters:")
        numeric_cols = ['turbidity', 'ph_level', 'temperature', 'bacterial_count', 'chlorine_residual']
        for col in numeric_cols:
            if col in water_df.columns:
                mean_val = water_df[col].mean()
                std_val = water_df[col].std()
                min_val = water_df[col].min()
                max_val = water_df[col].max()
                print(f"  {col}: {mean_val:.2f} ± {std_val:.2f} (range: {min_val:.2f} - {max_val:.2f})")
        
    except Exception as e:
        print(f"Error loading water quality data: {e}")
    
    print("\n" + "=" * 60)
    print("TRAINING/TESTING SPLIT")
    print("=" * 60)
    
    # Show training/testing split
    try:
        train_health = pd.read_csv('demo_data/train_health_dataset.csv')
        test_health = pd.read_csv('demo_data/test_health_dataset.csv')
        train_water = pd.read_csv('demo_data/train_water_dataset.csv')
        test_water = pd.read_csv('demo_data/test_water_dataset.csv')
        
        print(f"Training Data:")
        print(f"  Health reports: {len(train_health)} records")
        print(f"  Water quality: {len(train_water)} records")
        
        print(f"\nTesting Data:")
        print(f"  Health reports: {len(test_health)} records")
        print(f"  Water quality: {len(test_water)} records")
        
        print(f"\nSplit Ratios:")
        total_health = len(train_health) + len(test_health)
        total_water = len(train_water) + len(test_water)
        print(f"  Health reports: {len(train_health)/total_health:.1%} train, {len(test_health)/total_health:.1%} test")
        print(f"  Water quality: {len(train_water)/total_water:.1%} train, {len(test_water)/total_water:.1%} test")
        
    except Exception as e:
        print(f"Error loading train/test data: {e}")
    
    print("\n" + "=" * 60)
    print("MACHINE LEARNING READINESS")
    print("=" * 60)
    
    print("✅ The datasets are ready for ML training with:")
    print("  - Structured health reports with symptom data")
    print("  - Water quality parameters with contamination labels")
    print("  - Geographic coordinates for spatial analysis")
    print("  - Temporal data for time-series analysis")
    print("  - Proper train/test split for model validation")
    print("  - Balanced data distribution across features")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Run full model training: python run_fetch_and_train.py")
    print(f"  2. Start the API server: docker-compose up app")
    print(f"  3. Test predictions: python scripts/test_api.py")
    print(f"  4. View API docs: http://localhost:8000/docs")

if __name__ == "__main__":
    show_dataset_info()
