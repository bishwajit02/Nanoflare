# Solar X-Ray Data Analysis - Streamlit GUI

A web-based interactive interface for analyzing solar X-ray data using Gaussian fitting and R-squared analysis.

## Overview

This Streamlit application provides a user-friendly GUI for the solar X-ray data analysis workflow originally implemented in `rsquare_fixed.ipynb`. It allows real-time parameter adjustment and visualization of results through an intuitive web interface.

## Features

### 🎛️ Interactive Controls

- **Peak Detection Parameters**:
  - `height_percentile` (50-95%): Controls peak detection sensitivity
  - `prominence_factor` (1-20): Controls peak prominence threshold
- **Baseline Correction Parameters**:
  - `p` (0.001-0.1): Asymmetry parameter for baseline correction
  - `lam` (1e6-1e15): Regularization parameter for baseline correction
- **Analysis Configuration**:
  - Data file selection from available CSV files
  - Time range specification (start/end minutes)
  - Gaussian component range (min/max)

### 📊 Visualizations

1. **Baseline Correction Analysis**: Original vs corrected data with detected peaks
2. **R-Squared Analysis**: R² values vs number of Gaussian components
3. **Best Gaussian Fits**: Optimal fits and component breakdowns

### 📈 Real-time Metrics

- Number of peaks detected for XRSA and XRSB
- Optimal number of Gaussian components
- R-squared scores table

## Installation

### Prerequisites

- Python 3.7+
- Required packages (see `requirements_streamlit.txt`)

### Setup

1. Install dependencies:

   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. Ensure your data files are in the `data/` directory with the following structure:
   ```
   data/
   ├── 2017_xrsa_xrsb.csv
   ├── 2018_xrsa_xrsb.csv
   ├── ...
   └── 2025_xrsa_xrsb.csv
   ```

## Usage

### Starting the Application

```bash
streamlit run rsquare_streamlit.py
```

The application will start and display a URL (typically `http://localhost:8501`) in your terminal.

### Using the Interface

1. **Configure Parameters** (Sidebar):

   - Select your data file from the dropdown
   - Set time range for analysis
   - Adjust peak detection parameters
   - Configure baseline correction parameters
   - Set Gaussian component range

2. **Run Analysis**:

   - Click the "🚀 Run Analysis" button
   - Wait for processing to complete

3. **View Results**:
   - Check metrics in the main panel
   - Examine the three visualization plots
   - Review R-squared scores table

### Parameter Guidelines

#### Peak Detection

- **Height Percentile**: Higher values (80-95%) = more selective, fewer peaks
- **Prominence Factor**: Higher values (5-20) = more selective, fewer peaks

#### Baseline Correction

- **Asymmetry Parameter (p)**:
  - Lower values (0.001-0.01) = more aggressive correction
  - Higher values (0.01-0.1) = gentler correction
- **Regularization Parameter (λ)**:
  - Lower values (1e6-1e9) = more flexible baseline
  - Higher values (1e10-1e15) = smoother baseline

## File Structure

```
rsquare_streamlit.py          # Main Streamlit application
requirements_streamlit.txt    # Python dependencies
README_streamlit.md          # This documentation
data/                        # CSV data files directory
├── 2017_xrsa_xrsb.csv
├── 2018_xrsa_xrsb.csv
└── ...
```

## Data Format

The application expects CSV files with the following columns:

- `time_minutes`: Time in minutes
- `xrsa_flux_observed`: XRSA flux observations
- `xrsb_flux_observed`: XRSB flux observations

## Technical Details

### Dependencies

- `streamlit`: Web application framework
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Plotting
- `scipy`: Scientific computing (signal processing, optimization)
- `scikit-learn`: Machine learning (R-squared calculation)
- `pybaselines`: Advanced baseline correction (optional)

### Fallback Behavior

- If `pybaselines` is not available, the app falls back to simple linear detrending
- If Gaussian fitting fails, it uses a simplified component creation method

## Troubleshooting

### Common Issues

1. **"No CSV files found in data directory"**:

   - Ensure CSV files are in the `data/` directory
   - Check file naming convention

2. **"Analysis failed"**:

   - Try adjusting parameters (especially time range)
   - Check data file format
   - Ensure sufficient data points in selected time range

3. **Import errors**:

   - Install missing dependencies: `pip install -r requirements_streamlit.txt`
   - Check Python version compatibility

4. **Port already in use**:
   - Streamlit will automatically find an available port
   - Or specify a different port: `streamlit run rsquare_streamlit.py --server.port 8502`

### Performance Tips

- Start with smaller time ranges for faster analysis
- Use fewer Gaussian components for quicker processing
- The app caches results in session state for faster re-analysis

## Output Interpretation

### R-Squared Values

- Values closer to 1.0 indicate better fit
- Compare XRSA vs XRSB performance
- Look for optimal number of components (peak R²)

### Peak Detection

- More peaks = more sensitive detection
- Fewer peaks = more selective detection
- Balance sensitivity vs noise

### Baseline Correction

- Well-corrected data should show clear peaks above baseline
- Over-correction may remove real signal
- Under-correction may leave baseline artifacts

## Support

For issues or questions:

1. Check this README for common solutions
2. Verify data format and file structure
3. Test with different parameter combinations
4. Check terminal output for error messages
