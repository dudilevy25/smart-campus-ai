# Smart Campus AI

A comprehensive data analysis project for occupancy detection using environmental sensor data from smart campus systems.

## 🏢 Project Overview

This project analyzes occupancy patterns in campus buildings using environmental sensor readings including temperature, humidity, light, and CO2 levels. The analysis provides insights for optimizing building energy efficiency and space utilization.

## 📁 Project Structure

```
smart-campus-ai/
├── Data.csv                           # Occupancy dataset (20,560 records)
├── data_analysis.py                   # Main data analysis script
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── visualizations/                    # Generated analysis plots
    ├── temperature_histogram.png
    ├── occupancy_distribution.png
    ├── light_distribution_before.png
    ├── light_boxplot_before.png
    ├── temperature_boxplot_after.png
    ├── temperature_histogram_after.png
    ├── co2_boxplot_after.png
    ├── co2_histogram_after.png
    └── light_histogram_after.png
```

## 📊 Dataset Information

The dataset contains **20,560 environmental sensor readings** with the following features:

| Feature | Description | Range |
|---------|-------------|-------|
| **date** | Timestamp of measurement | 2015-02-02 to 2015-02-18 |
| **Temperature** | Room temperature (°C) | 19.0 - 24.4°C |
| **Humidity** | Relative humidity (%) | 16.7 - 39.5% |
| **Light** | Light intensity | Variable |
| **CO2** | CO2 concentration (ppm) | 412.8 - 2,076.5 ppm |
| **HumidityRatio** | Humidity ratio | 0.0027 - 0.0065 |
| **Occupancy** | Target variable | 0 (not occupied) / 1 (occupied) |

### 🎯 Key Dataset Insights
- **76.9%** of readings show unoccupied rooms (15,810 records)
- **23.1%** of readings show occupied rooms (4,750 records)
- Most common temperature: **20°C** (7,305 readings)
- Clean dataset with **no missing values or duplicates**

## 🛠️ Installation & Setup

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import pandas, numpy, matplotlib, seaborn, scipy; print('All dependencies installed successfully!')"
   ```

## 🚀 Usage

### Run Complete Analysis
```bash
python data_analysis.py
```

This will:
1. **Load and inspect** the occupancy dataset
2. **Clean the data** (remove missing values and duplicates)
3. **Analyze distributions** of all sensor variables
4. **Detect and remove outliers** using statistical methods
5. **Generate 9 visualization files** showing data patterns

### Expected Output
```
=== Smart Campus AI - Occupancy Data Analysis ===

Loading occupancy data...
Dataset shape: 20560 rows × 7 columns
Cleaning data...
Analyzing data distributions...
Detecting and removing outliers...
Removed 519 temperature outliers using IQR method
Removed 1993 CO2 outliers using Z-score method

Final dataset shape: (18,048, 8)
Data analysis completed successfully!
```

## 📈 Analysis Features

### 🧹 **Data Cleaning Pipeline**
- **Missing Value Detection**: Automatic identification and removal
- **Duplicate Removal**: Eliminates redundant records
- **Data Type Conversion**: Proper datetime formatting
- **Quality Validation**: Comprehensive data integrity checks

### 🔍 **Outlier Detection Methods**
- **IQR Method** for Temperature data (removed 519 outliers)
- **Z-Score Method** for CO2 data (removed 1,993 outliers)
- **Statistical Validation** ensures data quality for ML models

### 📊 **Comprehensive Visualizations**
1. **Distribution Analysis**
   - Temperature histograms with KDE curves
   - Occupancy distribution bar charts
   - Light intensity patterns

2. **Outlier Visualization**
   - Before/after boxplots for outlier detection
   - Statistical distribution comparisons
   - Data quality assessment plots

3. **Export Features**
   - High-resolution PNG files (300 DPI)
   - Professional formatting for reports
   - Automatic file naming and organization

## 🎨 Generated Visualizations

After running the analysis, you'll find these visualization files:

| File | Description |
|------|-------------|
| `temperature_histogram.png` | Temperature distribution across all readings |
| `occupancy_distribution.png` | Bar chart showing occupied vs unoccupied patterns |
| `light_distribution_before.png` | Light intensity before outlier removal |
| `light_boxplot_before.png` | Boxplot showing light outliers |
| `temperature_boxplot_after.png` | Clean temperature data visualization |
| `temperature_histogram_after.png` | Temperature distribution after cleaning |
| `co2_boxplot_after.png` | CO2 levels after outlier removal |
| `co2_histogram_after.png` | CO2 distribution analysis |
| `light_histogram_after.png` | Final light intensity distribution |

## 🔬 Technical Details

### **Data Processing Pipeline**
1. **Load** → Read CSV data with pandas
2. **Inspect** → Statistical summaries and data profiling
3. **Clean** → Remove missing values and duplicates
4. **Transform** → Convert data types and create derived features
5. **Analyze** → Statistical distribution analysis
6. **Visualize** → Generate comprehensive plots
7. **Export** → Save cleaned dataset and visualizations

### **Statistical Methods Used**
- **Interquartile Range (IQR)** for temperature outlier detection
- **Z-Score Analysis** for CO2 outlier identification
- **Descriptive Statistics** for data summarization
- **Distribution Analysis** using histograms and KDE

## 🔮 Future Enhancements

- [ ] **Machine Learning Models** for occupancy prediction
- [ ] **Time Series Analysis** for temporal patterns
- [ ] **Interactive Dashboards** with Plotly/Dash
- [ ] **Real-time Data Integration** with IoT sensors
- [ ] **Energy Optimization Algorithms** based on occupancy patterns

## 🛡️ Data Quality Assurance

- ✅ **No missing values** in the dataset
- ✅ **No duplicate records** found
- ✅ **519 temperature outliers** removed using IQR method
- ✅ **1,993 CO2 outliers** removed using Z-score method
- ✅ **Final clean dataset**: 18,048 records ready for analysis

## 📋 Dependencies

- **pandas** ≥ 1.5.0 - Data manipulation and analysis
- **numpy** ≥ 1.21.0 - Numerical computing
- **matplotlib** ≥ 3.5.0 - Data visualization
- **seaborn** ≥ 0.11.0 - Statistical data visualization
- **scipy** ≥ 1.9.0 - Scientific computing and statistics

---

*This project demonstrates comprehensive data analysis techniques for smart building occupancy detection, providing a foundation for energy-efficient campus management systems.*