import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import os
import gdown

# -------------------
# 0) Setup
# -------------------
viz_dir = "visualization"
os.makedirs(viz_dir, exist_ok=True)

# -------------------
# 1) Load Data
# -------------------
file_name = "data.csv"
file_id = "1KPw56B4kYr5zaPb_edTgyMcFSzp6soGq"
if not os.path.exists(file_name):
    gdown.download(id=file_id, output=file_name, quiet=False)

df = pd.read_csv(file_name)
df.head()

# -------------------
# Data Overview
# -------------------
summary_all = df.describe(include='all')
print(summary_all)

print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# -------------------
# Data Info
# -------------------
print("Data info:")
print(df.info())

# -------------------
# Data Cleaning
# -------------------
# Remove missing values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Check data types
data_types = df.dtypes
print("Data types:")
print(data_types)

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])
print("After date conversion:")
print(df[['date']].head())

# -------------------
# EDA
# -------------------
# Round temperature for frequency analysis
df['Rounded_Temp'] = df['Temperature'].round().astype(int)

# Temperature distribution
print("Temperature distribution:")
print(df['Rounded_Temp'].value_counts())

# Occupancy distribution
print("\nOccupancy distribution:")
print(df['Occupancy'].value_counts())

# Temperature histogram
plt.figure(figsize=(10, 5))
sns.histplot(df['Temperature'], bins=30, kde=True, color='skyblue')
plt.title("Histogram of Temperature (°C)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(os.path.join(viz_dir, '01_temperature_histogram.png'), dpi=300, bbox_inches='tight')
plt.close()

# Occupancy histogram
plt.figure(figsize=(6, 4))
df['Occupancy'].value_counts().plot(kind='bar', color=['gray', 'green'])
plt.title("Occupancy Count")
plt.xticks([0, 1], ['Not Occupied', 'Occupied'], rotation=0)
plt.ylabel("Count")
plt.grid(axis='y')
plt.savefig(os.path.join(viz_dir, '02_occupancy_histogram.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------
# Outlier Detection
# -------------------

# Light histogram
sns.set(style="whitegrid")
plt.figure(figsize=(8, 4))
sns.histplot(df['Light'], bins=100, kde=False)
plt.title("Light Distribution")
plt.xlabel("Light")
plt.ylabel("Frequency")
plt.savefig(os.path.join(viz_dir, '03_light_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Light boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Light'])
plt.title("Boxplot - Light")
plt.savefig(os.path.join(viz_dir, '04_light_boxplot.png'), dpi=300, bbox_inches='tight')
plt.close()

# Filter out outliers in Temperature
Q1 = df['Temperature'].quantile(0.25)
Q3 = df['Temperature'].quantile(0.75)
IQR = Q3 - Q1
lower_bound_temp = Q1 - 1.5 * IQR
upper_bound_temp = Q3 + 1.5 * IQR

# IQR method for Temperature
df = df[(df['Temperature'] >= lower_bound_temp) & (df['Temperature'] <= upper_bound_temp)]

# Boxplot for Temperature
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Temperature'])
plt.title("Boxplot - Temperature")
plt.savefig(os.path.join(viz_dir, '05_temperature_boxplot.png'), dpi=300, bbox_inches='tight')
plt.close()

# Histogram for Temperature
plt.figure(figsize=(6, 4))
sns.histplot(df['Temperature'], bins=30, kde=True)
plt.title("Histogram - Temperature")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.savefig(os.path.join(viz_dir, '06_temperature_histogram_cleaned.png'), dpi=300, bbox_inches='tight')
plt.close()

# Z-score method for CO2
z_scores = zscore(df['CO2'])
df = df[(np.abs(z_scores) < 1.5)]

# Boxplot for CO2
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['CO2'])
plt.title("Boxplot - CO2")
plt.savefig(os.path.join(viz_dir, '07_co2_boxplot.png'), dpi=300, bbox_inches='tight')
plt.close()

# Histogram for CO2
plt.figure(figsize=(6, 4))
sns.histplot(df['CO2'], bins=30, kde=True)
plt.title("Histogram - CO2")
plt.xlabel("CO2")
plt.ylabel("Frequency")
plt.savefig(os.path.join(viz_dir, '08_co2_histogram.png'), dpi=300, bbox_inches='tight')
plt.close()

# Histogram for Light
plt.figure(figsize=(8, 4))
sns.histplot(df['Light'], bins=100, kde=False)
plt.title("Histogram - Light")
plt.xlabel("Light")
plt.ylabel("Frequency")
plt.savefig(os.path.join(viz_dir, '09_light_histogram_final.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll visualizations have been saved to the '{viz_dir}' folder:")
print("- 01_temperature_histogram.png")
print("- 02_occupancy_histogram.png") 
print("- 03_light_distribution.png")
print("- 04_light_boxplot.png")
print("- 05_temperature_boxplot.png")
print("- 06_temperature_histogram_cleaned.png")
print("- 07_co2_boxplot.png")
print("- 08_co2_histogram.png")
print("- 09_light_histogram_final.png")

