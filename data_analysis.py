# -*- coding: utf-8 -*-
"""
Smart Campus AI - Occupancy Data Analysis
Data analysis script for occupancy detection using sensor data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import os

def load_and_inspect_data():
    """Load and perform initial inspection of the occupancy data"""
    print("Loading occupancy data...")
    df = pd.read_csv('Data.csv')
    
    print("First few rows:")
    print(df.head())
    
    # Initial statistical summary
    summary_all = df.describe(include='all')
    print("\nStatistical Summary:")
    print(summary_all)
    
    # Number of rows and columns
    print(f"\nDataset shape:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    return df

def clean_data(df):
    """Clean the data by removing missing values and duplicates"""
    print("\nCleaning data...")
    
    # Display data info
    print("Data info:")
    print(df.info())
    
    # Remove missing values
    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f"Removed {initial_rows - len(df)} rows with missing values")
    
    # Remove duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Check data types
    data_types = df.dtypes
    print("\nData types:")
    print(data_types)
    
    # Convert date column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    print("\nDate column converted to datetime format")
    print(df[['date']].head())
    
    return df

def analyze_distributions(df):
    """Analyze and visualize data distributions"""
    print("\nAnalyzing data distributions...")
    
    # Round temperature values for frequency analysis
    df['Rounded_Temp'] = df['Temperature'].round().astype(int)
    
    # Temperature distribution (rounded)
    print("Temperature distribution (rounded):")
    print(df['Rounded_Temp'].value_counts().sort_index())
    
    # Occupancy distribution
    print("\nOccupancy distribution:")
    print(df['Occupancy'].value_counts())
    
    # Create visualizations
    create_distribution_plots(df)
    
    return df

def create_distribution_plots(df):
    """Create distribution plots for key variables"""
    
    # Create visualizations directory if it doesn't exist
    visualizations_dir = 'visualizations'
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Set matplotlib to non-interactive backend to avoid popup windows
    plt.ioff()
    
    # Temperature histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Temperature'], bins=30, kde=True, color='skyblue')
    plt.title("Histogram of Temperature (°C)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(visualizations_dir, 'temperature_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {visualizations_dir}/temperature_histogram.png")
    
    # Occupancy bar chart
    plt.figure(figsize=(6, 4))
    df['Occupancy'].value_counts().plot(kind='bar', color=['gray', 'green'])
    plt.title("Occupancy Count")
    plt.xticks([0, 1], ['Not Occupied', 'Occupied'], rotation=0)
    plt.ylabel("Count")
    plt.grid(axis='y')
    plt.savefig(os.path.join(visualizations_dir, 'occupancy_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {visualizations_dir}/occupancy_distribution.png")

def detect_and_remove_outliers(df):
    """Detect and remove outliers using various methods"""
    print("\nDetecting and removing outliers...")
    
    # Create visualizations directory if it doesn't exist
    visualizations_dir = 'visualizations'
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Set seaborn style
    sns.set(style="whitegrid")
    plt.ioff()  # Turn off interactive mode
    
    # Light distribution before outlier removal
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Light'], bins=100, kde=False)
    plt.title("Light Distribution (Before Outlier Removal)")
    plt.xlabel("Light")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(visualizations_dir, 'light_distribution_before.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {visualizations_dir}/light_distribution_before.png")
    
    # Light boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['Light'])
    plt.title("Boxplot - Light (Before Outlier Removal)")
    plt.savefig(os.path.join(visualizations_dir, 'light_boxplot_before.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {visualizations_dir}/light_boxplot_before.png")
    
    # Remove Temperature outliers using IQR method
    Q1 = df['Temperature'].quantile(0.25)
    Q3 = df['Temperature'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound_temp = Q1 - 1.5 * IQR
    upper_bound_temp = Q3 + 1.5 * IQR
    
    initial_rows = len(df)
    df = df[(df['Temperature'] >= lower_bound_temp) & (df['Temperature'] <= upper_bound_temp)]
    print(f"Removed {initial_rows - len(df)} temperature outliers using IQR method")
    
    # Temperature boxplot after outlier removal
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['Temperature'])
    plt.title("Boxplot - Temperature (After Outlier Removal)")
    plt.savefig(os.path.join(visualizations_dir, 'temperature_boxplot_after.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {visualizations_dir}/temperature_boxplot_after.png")
    
    # Temperature histogram after outlier removal
    plt.figure(figsize=(6, 4))
    sns.histplot(df['Temperature'], bins=30, kde=True)
    plt.title("Histogram - Temperature (After Outlier Removal)")
    plt.xlabel("Temperature")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(visualizations_dir, 'temperature_histogram_after.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {visualizations_dir}/temperature_histogram_after.png")
    
    # Remove CO2 outliers using Z-score method
    z_scores = zscore(df['CO2'])
    initial_rows = len(df)
    df = df[(np.abs(z_scores) < 1.5)]
    print(f"Removed {initial_rows - len(df)} CO2 outliers using Z-score method")
    
    # CO2 boxplot after outlier removal
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['CO2'])
    plt.title("Boxplot - CO2 (After Outlier Removal)")
    plt.savefig(os.path.join(visualizations_dir, 'co2_boxplot_after.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {visualizations_dir}/co2_boxplot_after.png")
    
    # CO2 histogram after outlier removal
    plt.figure(figsize=(6, 4))
    sns.histplot(df['CO2'], bins=30, kde=True)
    plt.title("Histogram - CO2 (After Outlier Removal)")
    plt.xlabel("CO2")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(visualizations_dir, 'co2_histogram_after.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {visualizations_dir}/co2_histogram_after.png")
    
    # Light histogram after all outlier removals
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Light'], bins=100, kde=False)
    plt.title("Histogram - Light (After Outlier Removal)")
    plt.xlabel("Light")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(visualizations_dir, 'light_histogram_after.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {visualizations_dir}/light_histogram_after.png")
    
    return df

def main():
    """Main function to run the complete data analysis pipeline"""
    print("=== Smart Campus AI - Occupancy Data Analysis ===\n")
    
    # Load and inspect data
    df = load_and_inspect_data()
    
    # Clean the data
    df = clean_data(df)
    
    # Analyze distributions
    df = analyze_distributions(df)
    
    # Detect and remove outliers
    df = detect_and_remove_outliers(df)
    
    print(f"\nFinal dataset shape: {df.shape}")
    print("Data analysis completed successfully!")
    print("\nGenerated visualization files in 'visualizations/' directory:")
    print("- visualizations/temperature_histogram.png")
    print("- visualizations/occupancy_distribution.png")
    print("- visualizations/light_distribution_before.png")
    print("- visualizations/light_boxplot_before.png")
    print("- visualizations/temperature_boxplot_after.png")
    print("- visualizations/temperature_histogram_after.png")
    print("- visualizations/co2_boxplot_after.png")
    print("- visualizations/co2_histogram_after.png")
    print("- visualizations/light_histogram_after.png")
    
    return df

if __name__ == "__main__":
    # Run the analysis
    cleaned_data = main()
