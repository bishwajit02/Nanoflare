import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Solar X-Ray Data Analysis",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

def check_dependencies():
    """Check if required packages are available"""
    try:
        from pybaselines import Baseline
        return True
    except ImportError:
        return False

def load_data(file_path):
    """Load and validate data with proper error handling"""
    try:
        if not os.path.exists(file_path):
            st.error(f"Data file not found: {file_path}")
            return None
        
        data = pd.read_csv(file_path)
        
        if data.empty:
            st.error("Data file is empty")
            return None
        
        required_columns = ['time_minutes', 'xrsa_flux_observed', 'xrsb_flux_observed']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Check for NaN values
        nan_counts = data[required_columns].isnull().sum()
        if nan_counts.any():
            st.warning(f"Found NaN values: {nan_counts.to_dict()}")
            data = data.dropna(subset=required_columns)
            st.info(f"Cleaned data shape: {data.shape}")
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def apply_baseline_correction(time_data, xrsa_data, xrsb_data, use_pybaselines=True, p=0.005, lam=1e12):
    """Apply baseline correction with configurable parameters"""
    if use_pybaselines:
        try:
            from pybaselines import Baseline
            baseline_fitter = Baseline()
            
            # XRSA baseline correction
            xrsa_baseline, _ = baseline_fitter.asls(xrsa_data, lam=lam, p=p)
            xrsa_corrected = xrsa_data - xrsa_baseline
            
            # XRSB baseline correction
            xrsb_baseline, _ = baseline_fitter.asls(xrsb_data, lam=lam, p=p)
            xrsb_corrected = xrsb_data - xrsb_baseline
            
            return xrsa_baseline, xrsa_corrected, xrsb_baseline, xrsb_corrected
            
        except Exception as e:
            st.warning(f"pybaselines failed: {e}, using fallback method")
            use_pybaselines = False
    
    if not use_pybaselines:
        # Fallback to simple linear detrending
        xrsa_baseline = signal.detrend(xrsa_data)
        xrsa_corrected = xrsa_data - xrsa_baseline
        
        xrsb_baseline = signal.detrend(xrsb_data)
        xrsb_corrected = xrsb_data - xrsb_baseline
        
        return xrsa_baseline, xrsa_corrected, xrsb_baseline, xrsb_corrected

def find_peaks_robust(signal_data, time_data, height_percentile=75, prominence_factor=5):
    """Find peaks with configurable parameters"""
    try:
        # Calculate adaptive thresholds
        height_threshold = np.percentile(signal_data, height_percentile)
        prominence_threshold = np.std(signal_data) / prominence_factor
        
        peaks, properties = signal.find_peaks(
            signal_data,
            height=height_threshold,
            prominence=prominence_threshold,
            distance=20
        )
        
        return peaks, properties
        
    except Exception as e:
        st.warning(f"Peak detection failed: {e}")
        return np.array([]), {}

def create_optimized_gaussian_components(time_data, signal_data, num_gaussians):
    """Create optimized Gaussian components using actual curve fitting"""
    def multi_gaussian(x, *params):
        """Multiple Gaussian function"""
        result = np.zeros_like(x)
        n_gaussians = len(params) // 3
        
        for i in range(n_gaussians):
            amplitude = params[i*3]
            mean = params[i*3 + 1]
            sigma = abs(params[i*3 + 2])  # Ensure positive sigma
            result += amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
        
        return result
    
    try:
        # Find peaks for initial parameter estimation
        peaks, _ = signal.find_peaks(signal_data, height=np.percentile(signal_data, 75))
        
        if len(peaks) >= num_gaussians:
            # Use actual peaks if available
            peak_amplitudes = signal_data[peaks]
            sorted_indices = np.argsort(peak_amplitudes)[::-1]
            selected_peaks = peaks[sorted_indices[:num_gaussians]]
            means = time_data[selected_peaks]
            amplitudes = signal_data[selected_peaks] * 0.8
        else:
            # Evenly spaced means
            time_range = time_data.max() - time_data.min()
            means = np.linspace(time_data.min() + time_range * 0.1, 
                              time_data.max() - time_range * 0.1, num_gaussians)
            amplitudes = np.interp(means, time_data, signal_data) * 0.8
        
        # Calculate sigmas based on time range
        base_sigma = (time_data.max() - time_data.min()) / (num_gaussians * 2)
        sigmas = np.full(num_gaussians, base_sigma)
        
        # Prepare initial parameters for curve fitting
        p0 = []
        for i in range(num_gaussians):
            p0.extend([amplitudes[i], means[i], sigmas[i]])
        
        # Set bounds for parameters
        bounds_lower = [0, time_data.min(), 0.1] * num_gaussians
        bounds_upper = [np.max(signal_data) * 2, time_data.max(), 
                       (time_data.max() - time_data.min()) / 2] * num_gaussians
        
        # Fit the multi-Gaussian model
        popt, _ = curve_fit(multi_gaussian, time_data, signal_data, p0=p0, 
                           bounds=(bounds_lower, bounds_upper), maxfev=5000)
        
        # Generate components and total fit
        total_fit = multi_gaussian(time_data, *popt)
        components = []
        
        for i in range(num_gaussians):
            amp, mean, sig = popt[i*3:i*3+3]
            component = amp * np.exp(-((time_data - mean) ** 2) / (2 * sig ** 2))
            components.append(component)
        
        return total_fit, components, popt
        
    except Exception as e:
        # Fallback to simple component creation
        return create_simple_gaussian_components(time_data, signal_data, num_gaussians)

def create_simple_gaussian_components(time_data, signal_data, num_gaussians):
    """Fallback method for creating Gaussian components"""
    def gaussian(x, amplitude, mean, sigma):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
    
    time_range = time_data.max() - time_data.min()
    means = np.linspace(time_data.min() + time_range * 0.1, 
                       time_data.max() - time_range * 0.1, num_gaussians)
    amplitudes = np.interp(means, time_data, signal_data) * 0.8
    base_sigma = time_range / (num_gaussians * 2)
    sigmas = np.full(num_gaussians, base_sigma)
    
    components = []
    total_fit = np.zeros_like(time_data, dtype=np.float64)
    
    for i in range(num_gaussians):
        component = gaussian(time_data, amplitudes[i], means[i], sigmas[i])
        components.append(component)
        total_fit += component
    
    return total_fit, components, np.concatenate([amplitudes, means, sigmas])

def run_analysis(file_path, start_time, end_time, min_gaussians, max_gaussians, 
                height_percentile, prominence_factor, p, lam):
    """Main analysis function with configurable parameters"""
    
    # 1. Load data
    data = load_data(file_path)
    if data is None:
        return None
    
    # 2. Extract and filter data
    time_minutes = data['time_minutes'].values
    xrsa_original = data['xrsa_flux_observed'].values
    xrsb_original = data['xrsb_flux_observed'].values
    
    # Filter for specified time range
    mask = (time_minutes >= start_time) & (time_minutes <= end_time)
    time_filtered = time_minutes[mask]
    xrsa_filtered = xrsa_original[mask]
    xrsb_filtered = xrsb_original[mask]
    
    if len(time_filtered) == 0:
        st.error("No data found in specified time range")
        return None
    
    # 3. Apply baseline correction
    use_pybaselines = check_dependencies()
    xrsa_baseline, xrsa_corrected, xrsb_baseline, xrsb_corrected = apply_baseline_correction(
        time_filtered, xrsa_filtered, xrsb_filtered, use_pybaselines, p, lam
    )
    
    # 4. Find peaks
    xrsa_peaks, xrsa_properties = find_peaks_robust(xrsa_corrected, time_filtered, height_percentile, prominence_factor)
    xrsb_peaks, xrsb_properties = find_peaks_robust(xrsb_corrected, time_filtered, height_percentile, prominence_factor)
    
    # 5. Gaussian components analysis
    gaussian_range = range(min_gaussians, max_gaussians + 1)
    results = {'XRSA': {}, 'XRSB': {}}
    
    for signal_name, signal_data in [('XRSA', xrsa_corrected), ('XRSB', xrsb_corrected)]:
        r2_scores = []
        all_fits = []
        all_components = []
        all_parameters = []
        
        for num_gaussians in gaussian_range:
            try:
                gaussian_fit, components, params = create_optimized_gaussian_components(
                    time_filtered, signal_data, num_gaussians
                )
                
                r2 = r2_score(signal_data, gaussian_fit)
                
                r2_scores.append(r2)
                all_fits.append(gaussian_fit)
                all_components.append(components)
                all_parameters.append(params)
                
            except Exception as e:
                r2_scores.append(0)
                all_fits.append(np.zeros_like(signal_data))
                all_components.append([])
                all_parameters.append([])
        
        # Store results
        results[signal_name]['r2_scores'] = r2_scores
        results[signal_name]['fits'] = all_fits
        results[signal_name]['components'] = all_components
        results[signal_name]['parameters'] = all_parameters
    
    return results, time_filtered, xrsa_corrected, xrsb_corrected, xrsa_peaks, xrsb_peaks, xrsa_original[mask], xrsb_original[mask], xrsa_baseline, xrsb_baseline

def plot_baseline_correction(time_data, xrsa_original, xrsa_baseline, xrsa_corrected,
                           xrsb_original, xrsb_baseline, xrsb_corrected,
                           xrsa_peaks, xrsb_peaks):
    """Plot baseline correction results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # XRSA plots
    axes[0, 0].plot(time_data/60, xrsa_corrected, 'b-', label='XRSA Baseline-Corrected', linewidth=1.5)
    axes[0, 0].plot(time_data[xrsa_peaks]/60, xrsa_corrected[xrsa_peaks], 'ro', 
             label=f'XRSA Peaks ({len(xrsa_peaks)})', markersize=6)
    axes[0, 0].set_title('XRSA: Baseline-Corrected with Peaks')
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Flux')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time_data/60, xrsa_original, 'g-', alpha=0.7, label='XRSA Original', linewidth=1)
    axes[0, 1].plot(time_data/60, xrsa_baseline, 'r-', label='XRSA Baseline', linewidth=2)
    axes[0, 1].set_title('XRSA: Original vs Baseline')
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Flux')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # XRSB plots
    axes[1, 0].plot(time_data/60, xrsb_corrected, 'b-', label='XRSB Baseline-Corrected', linewidth=1.5)
    axes[1, 0].plot(time_data[xrsb_peaks]/60, xrsb_corrected[xrsb_peaks], 'ro', 
             label=f'XRSB Peaks ({len(xrsb_peaks)})', markersize=6)
    axes[1, 0].set_title('XRSB: Baseline-Corrected with Peaks')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Flux')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time_data/60, xrsb_original, 'g-', alpha=0.7, label='XRSB Original', linewidth=1)
    axes[1, 1].plot(time_data/60, xrsb_baseline, 'r-', label='XRSB Baseline', linewidth=2)
    axes[1, 1].set_title('XRSB: Original vs Baseline')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Flux')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_r2_analysis(gaussian_range, results):
    """Plot R-squared analysis results"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the lines
    ax.plot(gaussian_range, results['XRSA']['r2_scores'], 'bo-', 
             linewidth=2, markersize=8, label='XRSA')
    ax.plot(gaussian_range, results['XRSB']['r2_scores'], 'ro-', 
             linewidth=2, markersize=8, label='XRSB')
    
    # Add R² value annotations for XRSA
    for i, (g, r2) in enumerate(zip(gaussian_range, results['XRSA']['r2_scores'])):
        ax.annotate(f'{r2:.3f}', 
                    xy=(g, r2), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9, color='blue', weight='bold')
    
    # Add R² value annotations for XRSB
    for i, (g, r2) in enumerate(zip(gaussian_range, results['XRSB']['r2_scores'])):
        ax.annotate(f'{r2:.3f}', 
                    xy=(g, r2), 
                    xytext=(0, -15), 
                    textcoords='offset points',
                    ha='center', va='top',
                    fontsize=9, color='red', weight='bold')
    
    ax.set_title('Gaussian Components vs R-Squared', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Gaussian Components', fontsize=12)
    ax.set_ylabel('R-Squared', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xticks(gaussian_range)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig

def plot_best_fits(time_data, xrsa_data, xrsb_data, results, gaussian_range):
    """Plot best fit compositions"""
    # Find optimal number of components
    optimal_xrsa = gaussian_range[np.argmax(results['XRSA']['r2_scores'])]
    optimal_xrsb = gaussian_range[np.argmax(results['XRSB']['r2_scores'])]
    
    # Plot best fits
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # XRSA Best Fit
    xrsa_idx = optimal_xrsa - gaussian_range[0]
    axes[0, 0].plot(time_data/60, xrsa_data, 'b-', label='XRSA Data', linewidth=1.5, alpha=0.8)
    axes[0, 0].plot(time_data/60, results['XRSA']['fits'][xrsa_idx], 'r-',
                   label=f'Gaussian Fit (G={optimal_xrsa})', linewidth=2)
    axes[0, 0].set_title(f'XRSA: Best Gaussian Fit\nG={optimal_xrsa}, R²={results["XRSA"]["r2_scores"][xrsa_idx]:.3f}')
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Flux')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # XRSA Components
    colors = plt.cm.viridis(np.linspace(0, 1, len(results['XRSA']['components'][xrsa_idx])))
    for i, component in enumerate(results['XRSA']['components'][xrsa_idx]):
        if i < 5:  # Show only first 5 components to avoid clutter
            axes[0, 1].plot(time_data/60, component, '--', color=colors[i], alpha=0.6, linewidth=1,
                           label=f'Component {i+1}')
    axes[0, 1].plot(time_data/60, results['XRSA']['fits'][xrsa_idx], 'r-', label='Total Fit', linewidth=2)
    axes[0, 1].set_title(f'XRSA: Gaussian Components (G={optimal_xrsa})')
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Flux')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # XRSB Best Fit
    xrsb_idx = optimal_xrsb - gaussian_range[0]
    axes[1, 0].plot(time_data/60, xrsb_data, 'b-', label='XRSB Data', linewidth=1.5, alpha=0.8)
    axes[1, 0].plot(time_data/60, results['XRSB']['fits'][xrsb_idx], 'r-',
                   label=f'Gaussian Fit (G={optimal_xrsb})', linewidth=2)
    axes[1, 0].set_title(f'XRSB: Best Gaussian Fit\nG={optimal_xrsb}, R²={results["XRSB"]["r2_scores"][xrsb_idx]:.3f}')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Flux')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # XRSB Components
    colors = plt.cm.plasma(np.linspace(0, 1, len(results['XRSB']['components'][xrsb_idx])))
    for i, component in enumerate(results['XRSB']['components'][xrsb_idx]):
        if i < 5:  # Show only first 5 components to avoid clutter
            axes[1, 1].plot(time_data/60, component, '--', color=colors[i], alpha=0.6, linewidth=1,
                           label=f'Component {i+1}')
    axes[1, 1].plot(time_data/60, results['XRSB']['fits'][xrsb_idx], 'r-', label='Total Fit', linewidth=2)
    axes[1, 1].set_title(f'XRSB: Gaussian Components (G={optimal_xrsb})')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Flux')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Main Streamlit app
def main():
    # Header
    st.markdown('<h1 class="main-header">☀️ Solar X-Ray Data Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🔧 Configuration")
    
    # File selection
    data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    if not data_files:
        st.error("No CSV files found in the 'data' directory")
        return
    
    selected_file = st.sidebar.selectbox("Select Data File", data_files, index=0)
    data_file_path = f"data/{selected_file}"
    
    # Time range controls
    st.sidebar.subheader("⏰ Time Range")
    start_time = st.sidebar.number_input("Start Time (minutes)", min_value=0, value=361, step=1)
    end_time = st.sidebar.number_input("End Time (minutes)", min_value=start_time+1, value=720, step=1)
    
    # Gaussian range controls
    st.sidebar.subheader("🔬 Gaussian Analysis")
    min_gaussians = st.sidebar.number_input("Min Gaussian Components", min_value=1, value=10, step=1)
    max_gaussians = st.sidebar.number_input("Max Gaussian Components", min_value=min_gaussians+1, value=20, step=1)
    
    # Peak detection parameters
    st.sidebar.subheader("📊 Peak Detection")
    height_percentile = st.sidebar.slider("Height Percentile", min_value=50, max_value=95, value=75, step=5)
    prominence_factor = st.sidebar.slider("Prominence Factor", min_value=1, max_value=20, value=5, step=1)
    
    # Baseline correction parameters
    st.sidebar.subheader("🔄 Baseline Correction")
    p = st.sidebar.number_input("Asymmetry Parameter (p)", min_value=0.001, max_value=0.1, value=0.005, step=0.001, format="%.3f")
    lam = st.sidebar.number_input("Regularization Parameter (λ)", min_value=1e6, max_value=1e15, value=1e12, step=1e6, format="%.0e")
    
    # Analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Running analysis..."):
            # Run the analysis
            results = run_analysis(data_file_path, start_time, end_time, min_gaussians, max_gaussians,
                                 height_percentile, prominence_factor, p, lam)
            
            if results is not None:
                results_dict, time_data, xrsa_data, xrsb_data, xrsa_peaks, xrsb_peaks, xrsa_original, xrsb_original, xrsa_baseline, xrsb_baseline = results
                
                # Store results in session state
                st.session_state['results'] = results_dict
                st.session_state['time_data'] = time_data
                st.session_state['xrsa_data'] = xrsa_data
                st.session_state['xrsb_data'] = xrsb_data
                st.session_state['xrsa_peaks'] = xrsa_peaks
                st.session_state['xrsb_peaks'] = xrsb_peaks
                st.session_state['xrsa_original'] = xrsa_original
                st.session_state['xrsb_original'] = xrsb_original
                st.session_state['xrsa_baseline'] = xrsa_baseline
                st.session_state['xrsb_baseline'] = xrsb_baseline
                st.session_state['gaussian_range'] = range(min_gaussians, max_gaussians + 1)
                
                st.success("Analysis completed successfully!")
            else:
                st.error("Analysis failed. Please check your parameters and try again.")
    
    # Display results if available
    if 'results' in st.session_state:
        st.header("📊 Analysis Results")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("XRSA Peaks Found", len(st.session_state['xrsa_peaks']))
        
        with col2:
            st.metric("XRSB Peaks Found", len(st.session_state['xrsb_peaks']))
        
        with col3:
            optimal_xrsa = st.session_state['gaussian_range'][np.argmax(st.session_state['results']['XRSA']['r2_scores'])]
            st.metric("Optimal XRSA Components", optimal_xrsa)
        
        with col4:
            optimal_xrsb = st.session_state['gaussian_range'][np.argmax(st.session_state['results']['XRSB']['r2_scores'])]
            st.metric("Optimal XRSB Components", optimal_xrsb)
        
        # Display plots
        st.header("📈 Visualizations")
        
        # Plot 1: Baseline Correction
        st.subheader("1. Baseline Correction Analysis")
        fig1 = plot_baseline_correction(
            st.session_state['time_data'], 
            st.session_state['xrsa_original'], 
            st.session_state['xrsa_baseline'], 
            st.session_state['xrsa_data'],
            st.session_state['xrsb_original'], 
            st.session_state['xrsb_baseline'], 
            st.session_state['xrsb_data'],
            st.session_state['xrsa_peaks'], 
            st.session_state['xrsb_peaks']
        )
        st.pyplot(fig1)
        
        # Plot 2: R-squared Analysis
        st.subheader("2. R-Squared Analysis")
        fig2 = plot_r2_analysis(st.session_state['gaussian_range'], st.session_state['results'])
        st.pyplot(fig2)
        
        # Plot 3: Best Fits
        st.subheader("3. Best Gaussian Fits")
        fig3 = plot_best_fits(
            st.session_state['time_data'], 
            st.session_state['xrsa_data'], 
            st.session_state['xrsb_data'], 
            st.session_state['results'], 
            st.session_state['gaussian_range']
        )
        st.pyplot(fig3)
        
        # Display R² scores table
        st.subheader("📋 R-Squared Scores")
        r2_data = {
            'Gaussian Components': list(st.session_state['gaussian_range']),
            'XRSA R²': st.session_state['results']['XRSA']['r2_scores'],
            'XRSB R²': st.session_state['results']['XRSB']['r2_scores']
        }
        r2_df = pd.DataFrame(r2_data)
        st.dataframe(r2_df, use_container_width=True)

if __name__ == "__main__":
    main()
