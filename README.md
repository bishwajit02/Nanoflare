# Solar Nanoflare Detection Analysis

A comprehensive Python-based analysis toolkit for detecting and analyzing solar nanoflares using GOES X-ray sensor data. This project implements advanced signal processing techniques including baseline correction, Savitzky-Golay filtering, and statistical event detection to identify small-scale solar energy release events.

## 🌟 Features

- **Multi-year GOES X-ray data analysis** (2017-2025)
- **Advanced baseline correction** using pybaselines library
- **Signal smoothing** with Savitzky-Golay filters
- **Statistical nanoflare detection** using 3-sigma thresholding
- **Enhanced visualizations** with Seaborn and Matplotlib
- **Comprehensive event characterization** and analysis
- **Export capabilities** for detected events

## 📊 Data Sources

The project analyzes GOES X-ray sensor data from the following channels:

- **XRSA**: 0.05-0.4 nm (soft X-rays)
- **XRSB**: 0.1-0.8 nm (hard X-rays)

Data files are stored in the `data/` directory with naming convention: `YYYY_xrsa_xrsb.csv`

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### 1. Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd solar-nanoflare-analysis

# Or simply download and extract the project files
```

### 2. Install Required Libraries

#### Core Scientific Libraries

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

#### Specialized Libraries

```bash
pip install pybaselines
```

#### Complete Installation (All at once)

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn pybaselines
```

### 3. Verify Installation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, optimize, stats
from sklearn.preprocessing import StandardScaler
from pybaselines import Baseline

print("All libraries imported successfully!")
```

## 📚 Required Libraries

### Core Data Science

- **pandas** (≥1.3.0) - Data manipulation and analysis
- **numpy** (≥1.21.0) - Numerical computing and array operations
- **matplotlib** (≥3.4.0) - Basic plotting and visualization

### Advanced Visualization

- **seaborn** (≥0.11.0) - Statistical data visualization
- **scipy** (≥1.7.0) - Scientific computing tools
  - `scipy.signal` - Signal processing functions
  - `scipy.optimize` - Optimization algorithms
  - `scipy.stats` - Statistical functions
  - `scipy.ndimage` - Image processing functions

### Machine Learning

- **scikit-learn** (≥1.0.0) - Machine learning utilities
  - `sklearn.preprocessing.StandardScaler` - Data standardization

### Specialized Analysis

- **pybaselines** (≥1.0.0) - Advanced baseline correction algorithms

## 🎯 Usage Examples

### 1. Basic Analysis (Jupyter Notebook)

```python
# Load and analyze data
import pandas as pd
import numpy as np

# Load GOES X-ray data
df = pd.read_csv('data/2023_xrsa_xrsb.csv')
df_clean = df.dropna()

# Extract time series data
time_minutes = df_clean['time_minutes'].values
xrsa_flux = df_clean['xrsa_flux_observed'].values
xrsb_flux = df_clean['xrsb_flux_observed'].values
```

### 2. Baseline Correction

```python
from pybaselines import Baseline

# Initialize baseline correction
baseline_fitter = Baseline(time_minutes)

# Apply AsLS baseline correction
xrsa_result = baseline_fitter.asls(log_xrsa, lam=1e6, p=0.01)
xrsa_baseline = xrsa_result[0]
xrsa_corrected = log_xrsa - xrsa_baseline
```

### 3. Signal Smoothing

```python
from scipy.signal import savgol_filter

# Apply Savitzky-Golay smoothing
window_length = 21  # Must be odd
polyorder = 3
xrsa_smoothed = savgol_filter(xrsa_corrected, window_length, polyorder)
```

### 4. Nanoflare Detection

```python
# Calculate residuals
xrsa_residuals = xrsa_detrended - xrsa_smooth

# Set detection threshold (3-sigma)
threshold_xrsa = 3 * np.std(xrsa_residuals)

# Detect events
xrsa_events = np.where(np.abs(xrsa_residuals) > threshold_xrsa)[0]
```

## 📁 Project Structure

```
solar-nanoflare-analysis/
├── README.md                           # This file
├── data/                              # GOES X-ray data files
│   ├── 2017_xrsa_xrsb.csv
│   ├── 2018_xrsa_xrsb.csv
│   ├── 2019_xrsa_xrsb.csv
│   ├── 2020_xrsa_xrsb.csv
│   ├── 2021_xrsa_xrsb.csv
│   ├── 2022_xrsa_xrsb.csv
│   ├── 2023_xrsa_xrsb.csv
│   ├── 2024_xrsa_xrsb.csv
│   └── 2025_xrsa_xrsb.csv
├── nanoflare_detection_analysis-1.ipynb  # Main analysis notebook
├── mfd.ipynb                           # Additional analysis notebook
├── mfd_fixed.ipynb                     # Fixed version of mfd notebook
├── focused_nanoflare_analysis.py       # Focused analysis script
└── nanoflare_events_detected.csv       # Output file with detected events
```

## 🔬 Analysis Workflow

1. **Data Loading**: Import GOES X-ray sensor data
2. **Data Cleaning**: Remove missing values and outliers
3. **Baseline Correction**: Remove long-term trends using pybaselines
4. **Signal Smoothing**: Apply Savitzky-Golay filters
5. **Event Detection**: Identify nanoflares using statistical thresholds
6. **Visualization**: Create comprehensive plots and analysis
7. **Export Results**: Save detected events to CSV

## 📈 Key Analysis Techniques

### Baseline Correction Methods

- **AsLS (Asymmetric Least Squares)**: Preserves peak shapes while removing drift
- **Linear Detrending**: Removes long-term linear trends
- **Gaussian Smoothing**: Isolates baseline from transient events

### Signal Processing

- **Savitzky-Golay Filter**: Smooths signals while preserving peak characteristics
- **Gaussian Filter**: Advanced smoothing with configurable kernel width
- **Residual Analysis**: Separates baseline from event signatures

### Event Detection

- **3-Sigma Thresholding**: Statistical significance testing
- **Multi-channel Correlation**: Cross-validation between XRSA and XRSB
- **Event Characterization**: Amplitude, timing, and significance analysis

## 🎨 Visualization Features

- **Enhanced Styling**: Professional-grade plots with custom color schemes
- **Multi-panel Layouts**: Comprehensive analysis views
- **Interactive Elements**: Zoom, pan, and exploration capabilities
- **Publication Quality**: High-resolution outputs suitable for research papers

## 📊 Output Files

- **nanoflare_events_detected.csv**: Comprehensive list of detected events with:
  - Time stamps (minutes from start)
  - Channel (XRSA/XRSB)
  - Residual amplitude
  - Detection threshold
  - Statistical significance

## 🔧 Configuration Options

### Time Windows

- **Focused Analysis**: 6-hour to 24-hour windows
- **Full Dataset**: Complete year analysis
- **Custom Ranges**: User-defined time periods

### Detection Parameters

- **Threshold Multiplier**: Adjustable from 2-sigma to 5-sigma
- **Smoothing Kernel**: Configurable Gaussian and Savitzky-Golay parameters
- **Baseline Methods**: Multiple correction algorithms available

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required libraries are installed
2. **Memory Issues**: Large datasets may require chunked processing
3. **Baseline Correction Failures**: Adjust `lam` and `p` parameters
4. **Visualization Problems**: Check matplotlib backend configuration

### Performance Tips

- Use smaller time windows for interactive analysis
- Process data in chunks for large datasets
- Adjust smoothing parameters based on data characteristics
- Monitor memory usage during processing

## 📚 References

- GOES X-ray Sensor Documentation
- Solar Physics and Nanoflare Research Papers
- Signal Processing and Statistical Analysis Methods
- Python Scientific Computing Best Practices

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs and issues
- Suggest new features and improvements
- Submit pull requests with enhancements
- Share analysis results and findings

## 📄 License

This project is provided for educational and research purposes. Please ensure proper attribution when using this code in publications or other work.

## 📞 Support

For questions, issues, or collaboration opportunities, please:

1. Check the troubleshooting section above
2. Review the code comments and documentation
3. Open an issue in the project repository
4. Contact the development team

---

**Happy Solar Analysis! 🌞✨**
