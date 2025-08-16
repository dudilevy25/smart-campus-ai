import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

# -------------------
# 0) Setup
# -------------------
viz_dir = "visualization"
os.makedirs(viz_dir, exist_ok=True)

sns.set(style="whitegrid")

# -------------------
# 1) Load Data
# -------------------
file_name = "data.csv"
file_id = "1KPw56B4kYr5zaPb_edTgyMcFSzp6soGq"
if not os.path.exists(file_name):
    gdown.download(id=file_id, output=file_name, quiet=False)

df = pd.read_csv(file_name)

# -------------------
# 2) Basic Overview
# -------------------
print("=== Head ===")
print(df.head())
print("\n=== Describe (all) ===")
print(df.describe(include="all"))

print(f"\nRows: {df.shape[0]}")
print(f"Cols: {df.shape[1]}")

print("\n=== Info (before typing) ===")
df.info()

# -------------------
# 3) Typing & Cleaning
# -------------------
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

for col in df.columns:
    if col != 'date':
        df[col] = pd.to_numeric(df[col], errors='ignore')

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print("\n=== Info (after typing & cleaning) ===")
df.info()

# -------------------
# 4) Rounding rules
# -------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if col == 'HumidityRatio':
        df[col] = pd.to_numeric(df[col], errors='coerce').round(3)
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(0)

print("\n=== Numeric columns after rounding ===")
print(df[numeric_cols].head())

# -------------------
# 5) EDA (distributions)
# -------------------
if 'Temperature' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Temperature'].dropna().astype(float), bins=30, kde=True)
    plt.title("Histogram of Temperature (°C)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir, '01_temperature_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()

if 'Occupancy' in df.columns:
    plt.figure(figsize=(6, 4))
    df['Occupancy'].value_counts().sort_index().plot(kind='bar')
    plt.title("Occupancy Count")
    plt.xticks(rotation=0)
    plt.ylabel("Count")
    plt.grid(axis='y')
    plt.savefig(os.path.join(viz_dir, '02_occupancy_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()

# -------------------
# 6) Boxplots + Histograms (no outlier removal)
# -------------------
if 'Light' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Light'].dropna().astype(float), bins=100, kde=False)
    plt.title("Light Distribution")
    plt.savefig(os.path.join(viz_dir, '03_light_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['Light'].dropna().astype(float))
    plt.title("Boxplot - Light")
    plt.savefig(os.path.join(viz_dir, '04_light_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

if 'Temperature' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['Temperature'].dropna().astype(float))
    plt.title("Boxplot - Temperature")
    plt.savefig(os.path.join(viz_dir, '05_temperature_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df['Temperature'].dropna().astype(float), bins=30, kde=True)
    plt.title("Histogram - Temperature")
    plt.savefig(os.path.join(viz_dir, '06_temperature_histogram_cleaned.png'), dpi=300, bbox_inches='tight')
    plt.close()

if 'CO2' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['CO2'].dropna().astype(float))
    plt.title("Boxplot - CO2")
    plt.savefig(os.path.join(viz_dir, '07_co2_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df['CO2'].dropna().astype(float), bins=30, kde=True)
    plt.title("Histogram - CO2")
    plt.savefig(os.path.join(viz_dir, '08_co2_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()

if 'Humidity' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Humidity'].dropna().astype(float), bins=30, kde=True)
    plt.title("Histogram - Humidity")
    plt.savefig(os.path.join(viz_dir, '15_humidity_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['Humidity'].dropna().astype(float))
    plt.title("Boxplot - Humidity")
    plt.savefig(os.path.join(viz_dir, '17_humidity_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

if 'HumidityRatio' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df['HumidityRatio'].dropna().astype(float), bins=30, kde=True)
    plt.title("Histogram - HumidityRatio")
    plt.savefig(os.path.join(viz_dir, '18_humidityratio_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['HumidityRatio'].dropna().astype(float))
    plt.title("Boxplot - HumidityRatio")
    plt.savefig(os.path.join(viz_dir, '19_humidityratio_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

# -------------------
# 7) Time-Series Trends
# -------------------
if 'date' in df.columns:
    df = df.sort_values('date').reset_index(drop=True)
    ts = df.set_index('date')

    if 'HumidityRatio' in ts.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(ts.index, ts['HumidityRatio'], linewidth=0.8)
        plt.title("HumidityRatio over Time")
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, '20_humidityratio_trend.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if 'Temperature' in ts.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(ts.index, ts['Temperature'], linewidth=0.8)
        plt.title("Temperature over Time")
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, '10_temperature_trend.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if 'Humidity' in ts.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(ts.index, ts['Humidity'], linewidth=0.8)
        plt.title("Humidity over Time")
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, '11_humidity_trend.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if 'Light' in ts.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(ts.index, ts['Light'], linewidth=0.8)
        plt.title("Light over Time")
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, '12_light_trend.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if 'CO2' in ts.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(ts.index, ts['CO2'], linewidth=0.8)
        plt.title("CO2 over Time")
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, '13_co2_trend.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if 'Occupancy' in ts.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(ts.index, ts['Occupancy'], linewidth=0.8, drawstyle="steps-post")
        plt.title("Occupancy over Time")
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, '14_occupancy_trend.png'), dpi=300, bbox_inches='tight')
        plt.close()

# -------------------
# 7b) Time-Series Trends - All Features Together
# -------------------
if 'date' in df.columns:
    ts = df.set_index('date')

    # ניקוי: נשמור רק עמודות מספריות כדי למנוע בעיות
    ts_numeric = ts.select_dtypes(include=[np.number])

    if not ts_numeric.empty:
        plt.figure(figsize=(14, 7))
        for col in ts_numeric.columns:
            plt.plot(ts_numeric.index, ts_numeric[col], label=col, linewidth=0.8)

        plt.title("All Features over Time")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, '21_all_features_trend.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
# -------------------
# 8) Correlation Heatmap
# -------------------
if len(numeric_cols) > 0:
    plt.figure(figsize=(8, 6))
    corr = df[numeric_cols].astype(float).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(viz_dir, '16_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


# -------------------
# 9) Summary
# -------------------
print(f"\nAll visualizations have been saved to the '{viz_dir}' folder.")

# -------------------
# 10) Auto EDA Reports (HTML)
# -------------------
os.makedirs("reports", exist_ok=True)

# ydata-profiling – דוח אחד מקיף
try:
    from ydata_profiling import ProfileReport
    profile = ProfileReport(
        df,
        title="EDA Report - ydata-profiling",
        explorative=True,  # כולל עוד כרטיסיות וגרפים
        minimal=False
    )
    profile.to_file("reports/eda_ydata_profiling.html")
    print("✓ Created reports/eda_ydata_profiling.html")
except Exception as e:
    print("ydata-profiling failed:", e)

print("\nAuto EDA reports saved under 'reports/'")
