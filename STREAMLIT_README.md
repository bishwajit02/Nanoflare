# Solar X-Ray Data Analysis - Streamlit GUI

This is a Streamlit GUI version of the `rsquare_fixed.ipynb` notebook for analyzing solar X-ray data.

## Features

- **Interactive Parameter Control**: Adjust key parameters through the sidebar:

  - `height_percentile`: Controls peak detection sensitivity (50-95%)
  - `prominence_factor`: Controls peak prominence threshold (1-20)
  - `p`: Asymmetry parameter for baseline correction (0.001-0.1)
  - `lam`: Regularization parameter for baseline correction (1e6-1e15)

- **File Selection**: Choose from available CSV files in the `data/` directory
- **Time Range Control**: Set start and end times for analysis
- **Gaussian Analysis**: Configure minimum and maximum Gaussian components
- **Three Main Visualizations**:
  1. Baseline Correction Analysis
  2. R-Squared Analysis
  3. Best Gaussian Fits

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements_streamlit.txt
```

## Usage

1. Make sure your data files are in the `data/` directory
2. Run the Streamlit app:

```bash
streamlit run rsquare_streamlit.py
```

3. Open your browser to the URL shown in the terminal (usually `http://localhost:8501`)

## Controls

### Sidebar Parameters

- **Data File**: Select from available CSV files
- **Time Range**: Set start and end times in minutes
- **Gaussian Analysis**: Set min/max number of Gaussian components
- **Peak Detection**:
  - Height Percentile: Higher values = more selective peak detection
  - Prominence Factor: Higher values = more selective peak detection
- **Baseline Correction**:
  - Asymmetry Parameter (p): Controls baseline asymmetry
  - Regularization Parameter (λ): Controls baseline smoothness

### Main Interface

- Click "🚀 Run Analysis" to start the analysis
- View results in three main sections:
  1. **Metrics**: Key statistics about the analysis
  2. **Visualizations**: Interactive plots
  3. **R-Squared Scores**: Detailed table of R² values

## Notes

- The app uses `pybaselines` for advanced baseline correction if available
- Falls back to simple detrending if `pybaselines` is not installed
- All plots are interactive and can be zoomed/panned
- Results are cached in session state for faster re-analysis

## Troubleshooting

- If you get import errors, make sure all dependencies are installed
- If data files are not found, check that CSV files are in the `data/` directory
- If analysis fails, try adjusting the parameters or time range
