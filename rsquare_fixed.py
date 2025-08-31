#!/usr/bin/env python3
"""
Improved R-Squared Analysis for Solar X-ray Data
Fixed version with error handling, flexibility, and better analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
DEFAULT_FILE = 'data/2017_xrsa_xrsb.csv'
DEFAULT_START_TIME = 361      # Start time in minutes
DEFAULT_END_TIME = 720      # End time in minutes (6 hours)
DEFAULT_MIN_GAUSSIANS = 10  # Minimum Gaussian components
DEFAULT_MAX_GAUSSIANS = 20  # Maximum Gaussian components

# === HELPER FUNCTIONS ===

def check_dependencies():
    """Check if required packages are available"""
    try:
        from pybaselines import Baseline
        print("✅ pybaselines available - using advanced baseline correction")
        return True
    except ImportError:
        print("⚠️  pybaselines not available - using simple detrending")
        return False

def load_data(file_path):
    """Load and validate data with proper error handling"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        data = pd.read_csv(file_path)
        
        if data.empty:
            raise ValueError("Data file is empty")
        
        required_columns = ['time_minutes', 'xrsa_flux_observed', 'xrsb_flux_observed']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for NaN values
        nan_counts = data[required_columns].isnull().sum()
        if nan_counts.any():
            print(f"⚠️  Found NaN values: {nan_counts.to_dict()}")
            data = data.dropna(subset=required_columns)
            print(f"✅ Cleaned data shape: {data.shape}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def apply_baseline_correction(time_data, xrsa_data, xrsb_data, use_pybaselines=True):
    """Apply baseline correction with fallback option"""
    if use_pybaselines:
        try:
            from pybaselines import Baseline
            baseline_fitter = Baseline()
            
            # XRSA baseline correction
            xrsa_baseline, _ = baseline_fitter.asls(xrsa_data, lam=1e7, p=0.001)
            xrsa_corrected = xrsa_data - xrsa_baseline
            
            # XRSB baseline correction
            xrsb_baseline, _ = baseline_fitter.asls(xrsb_data, lam=1e7, p=0.001)
            xrsb_corrected = xrsb_data - xrsb_baseline
            
            print("✅ Baseline correction completed using pybaselines AsLS method")
            return xrsa_baseline, xrsa_corrected, xrsb_baseline, xrsb_corrected
            
        except Exception as e:
            print(f"⚠️  pybaselines failed: {e}, using fallback method")
            use_pybaselines = False
    
    if not use_pybaselines:
        # Fallback to simple linear detrending
        from scipy import signal
        
        # XRSA detrending
        xrsa_baseline = signal.detrend(xrsa_data)
        xrsa_corrected = xrsa_data - xrsa_baseline
        
        # XRSB detrending
        xrsb_baseline = signal.detrend(xrsb_data)
        xrsb_corrected = xrsb_data - xrsb_baseline
        
        print("✅ Applied linear detrending as fallback")
        return xrsa_baseline, xrsa_corrected, xrsb_baseline, xrsb_corrected

def find_peaks_robust(signal_data, time_data, height_percentile=75, prominence_factor=3):
    """Find peaks with robust parameters"""
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
        print(f"⚠️  Peak detection failed: {e}")
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
        print(f"⚠️  Gaussian fitting failed for {num_gaussians} components: {e}")
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

# === MAIN ANALYSIS FUNCTION ===

def run_analysis(file_path, start_time, end_time, min_gaussians, max_gaussians):
    """Main analysis function"""
    print("🔬 R-SQUARED ANALYSIS FOR SOLAR X-RAY DATA")
    print("=" * 60)
    
    # 1. Load data
    print("📁 Loading data...")
    data = load_data(file_path)
    if data is None:
        return None
    
    print(f"✅ Data loaded successfully!")
    print(f"   Shape: {data.shape}")
    print(f"   Columns: {data.columns.tolist()}")
    
    # 2. Extract and filter data
    print(f"\n⏰ Filtering data for time range {start_time}-{end_time} minutes...")
    
    time_minutes = data['time_minutes'].values
    xrsa_original = data['xrsa_flux_observed'].values
    xrsb_original = data['xrsb_flux_observed'].values
    
    # Filter for specified time range
    mask = (time_minutes >= start_time) & (time_minutes <= end_time)
    time_filtered = time_minutes[mask]
    xrsa_filtered = xrsa_original[mask]
    xrsb_filtered = xrsb_original[mask]
    
    if len(time_filtered) == 0:
        print("❌ No data found in specified time range")
        return None
    
    print(f"✅ Filtered data: {len(time_filtered)} points")
    
    # 3. Apply baseline correction
    print("\n🔄 Applying baseline correction...")
    use_pybaselines = check_dependencies()
    
    xrsa_baseline, xrsa_corrected, xrsb_baseline, xrsb_corrected = apply_baseline_correction(
        time_filtered, xrsa_filtered, xrsb_filtered, use_pybaselines
    )
    
    # 4. Find peaks
    print("\n📊 Finding peaks...")
    
    xrsa_peaks, xrsa_properties = find_peaks_robust(xrsa_corrected, time_filtered)
    xrsb_peaks, xrsb_properties = find_peaks_robust(xrsb_corrected, time_filtered)
    
    print(f"✅ XRSA peaks found: {len(xrsa_peaks)}")
    print(f"✅ XRSB peaks found: {len(xrsb_peaks)}")  # Fixed the bug here
    
    # 5. Gaussian components analysis
    print(f"\n🔬 Performing Gaussian fit analysis ({min_gaussians}-{max_gaussians} components)...")
    
    gaussian_range = range(min_gaussians, max_gaussians + 1)
    results = {'XRSA': {}, 'XRSB': {}}
    
    for signal_name, signal_data in [('XRSA', xrsa_corrected), ('XRSB', xrsb_corrected)]:
        print(f"\n   Analyzing {signal_name}...")
        
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
                
                print(f"   G={num_gaussians:2d}: R² = {r2:.4f}")
                
            except Exception as e:
                print(f"   G={num_gaussians:2d}: Failed - {e}")
                r2_scores.append(0)
                all_fits.append(np.zeros_like(signal_data))
                all_components.append([])
                all_parameters.append([])
        
        # Store results
        results[signal_name]['r2_scores'] = r2_scores
        results[signal_name]['fits'] = all_fits
        results[signal_name]['components'] = all_components
        results[signal_name]['parameters'] = all_parameters
    
    return results, time_filtered, xrsa_corrected, xrsb_corrected, xrsa_peaks, xrsb_peaks

# === VISUALIZATION FUNCTIONS ===

def plot_baseline_correction(time_data, xrsa_original, xrsa_baseline, xrsa_corrected,
                           xrsb_original, xrsb_baseline, xrsb_corrected,
                           xrsa_peaks, xrsb_peaks):
    """Plot baseline correction results"""
    plt.figure(figsize=(15, 10))
    
    # XRSA plots
    plt.subplot(2, 2, 1)
    plt.plot(time_data/60, xrsa_corrected, 'b-', label='XRSA Baseline-Corrected', linewidth=1.5)
    plt.plot(time_data[xrsa_peaks]/60, xrsa_corrected[xrsa_peaks], 'ro', 
             label=f'XRSA Peaks ({len(xrsa_peaks)})', markersize=6)
    plt.title('XRSA: Baseline-Corrected with Peaks')
    plt.xlabel('Time (hours)')
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(time_data/60, xrsa_original, 'g-', alpha=0.7, label='XRSA Original', linewidth=1)
    plt.plot(time_data/60, xrsa_baseline, 'r-', label='XRSA Baseline', linewidth=2)
    plt.title('XRSA: Original vs Baseline')
    plt.xlabel('Time (hours)')
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # XRSB plots
    plt.subplot(2, 2, 3)
    plt.plot(time_data/60, xrsb_corrected, 'b-', label='XRSB Baseline-Corrected', linewidth=1.5)
    plt.plot(time_data[xrsb_peaks]/60, xrsb_corrected[xrsb_peaks], 'ro', 
             label=f'XRSB Peaks ({len(xrsb_peaks)})', markersize=6)
    plt.title('XRSB: Baseline-Corrected with Peaks')
    plt.xlabel('Time (hours)')
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(time_data/60, xrsb_original, 'g-', alpha=0.7, label='XRSB Original', linewidth=1)
    plt.plot(time_data/60, xrsb_baseline, 'r-', label='XRSB Baseline', linewidth=2)
    plt.title('XRSB: Original vs Baseline')
    plt.xlabel('Time (hours)')
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_r2_analysis(gaussian_range, results):
    """Plot R-squared analysis results with R² values displayed"""
    plt.figure(figsize=(12, 8))
    
    # Plot the lines
    plt.plot(gaussian_range, results['XRSA']['r2_scores'], 'bo-', 
             linewidth=2, markersize=8, label='XRSA')
    plt.plot(gaussian_range, results['XRSB']['r2_scores'], 'ro-', 
             linewidth=2, markersize=8, label='XRSB')
    
    # Add R² value annotations for XRSA
    for i, (g, r2) in enumerate(zip(gaussian_range, results['XRSA']['r2_scores'])):
        plt.annotate(f'{r2:.3f}', 
                    xy=(g, r2), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9, color='blue', weight='bold')
    
    # Add R² value annotations for XRSB
    for i, (g, r2) in enumerate(zip(gaussian_range, results['XRSB']['r2_scores'])):
        plt.annotate(f'{r2:.3f}', 
                    xy=(g, r2), 
                    xytext=(0, -15), 
                    textcoords='offset points',
                    ha='center', va='top',
                    fontsize=9, color='red', weight='bold')
    
    plt.title('Gaussian Components vs R-Squared', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Gaussian Components', fontsize=12)
    plt.ylabel('R-Squared', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(gaussian_range)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_best_fits(time_data, xrsa_data, xrsb_data, results, gaussian_range):
    """Plot best fit compositions"""
    # Find optimal number of components
    optimal_xrsa = gaussian_range[np.argmax(results['XRSA']['r2_scores'])]
    optimal_xrsb = gaussian_range[np.argmax(results['XRSB']['r2_scores'])]
    
    print(f"\n🎯 Optimal components - XRSA: {optimal_xrsa}, XRSB: {optimal_xrsb}")
    
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
    axes[1, 0].set_title(f'XRSB: Best Gaussian Fit\nG={optimal_xrsb}, R²={results["XRSB"]["r2_scores"][xrsa_idx]:.3f}')
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
    plt.show()

# === MAIN EXECUTION ===

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='R-Squared Analysis for Solar X-ray Data')
    parser.add_argument('--data', type=str, default=DEFAULT_FILE, 
                       help=f'Data file path (default: {DEFAULT_FILE})')
    parser.add_argument('--start', type=int, default=DEFAULT_START_TIME,
                       help=f'Start time in minutes (default: {DEFAULT_START_TIME})')
    parser.add_argument('--end', type=int, default=DEFAULT_END_TIME,
                       help=f'End time in minutes (default: {DEFAULT_END_TIME})')
    parser.add_argument('--min-gaussians', type=int, default=DEFAULT_MIN_GAUSSIANS,
                       help=f'Minimum Gaussian components (default: {DEFAULT_MIN_GAUSSIANS})')
    parser.add_argument('--max-gaussians', type=int, default=DEFAULT_MAX_GAUSSIANS,
                       help=f'Maximum Gaussian components (default: {DEFAULT_MAX_GAUSSIANS})')
    parser.add_argument('--show-config', action='store_true',
                       help='Show current configuration and exit')
    
    args = parser.parse_args()
    
    # Show configuration if requested
    if args.show_config:
        print("🔧 CURRENT CONFIGURATION:")
        print(f"   Data file: {args.data}")
        print(f"   Time range: {args.start}-{args.end} minutes")
        print(f"   Gaussian range: {args.min_gaussians}-{args.max_gaussians}")
        print(f"   Duration: {(args.end - args.start)/60:.1f} hours")
        exit()
    
    # Run analysis
    results = run_analysis(args.data, args.start, args.end, 
                          args.min_gaussians, args.max_gaussians)
    
    if results is None:
        print("❌ Analysis failed. Exiting.")
        exit()
    
    results_dict, time_data, xrsa_data, xrsb_data, xrsa_peaks, xrsb_peaks = results
    
    # Load original data for baseline plotting
    data = load_data(args.data)
    time_minutes = data['time_minutes'].values
    xrsa_original = data['xrsa_flux_observed'].values
    xrsb_original = data['xrsb_flux_observed'].values
    
    # Apply baseline correction for plotting
    use_pybaselines = check_dependencies()
    xrsa_baseline, _, xrsb_baseline, _ = apply_baseline_correction(
        time_data, xrsa_original[args.start:args.end+1], xrsb_original[args.start:args.end+1], use_pybaselines
    )
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    
    plot_baseline_correction(time_data, xrsa_original[args.start:args.end+1], xrsa_baseline, xrsa_data,
                           xrsb_original[args.start:args.end+1], xrsb_baseline, xrsb_data,
                           xrsa_peaks, xrsb_peaks)
    
    gaussian_range = range(args.min_gaussians, args.max_gaussians + 1)
    plot_r2_analysis(gaussian_range, results_dict)
    plot_best_fits(time_data, xrsa_data, xrsb_data, results_dict, gaussian_range)
    
    # Display summary results
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    
    optimal_xrsa = gaussian_range[np.argmax(results_dict['XRSA']['r2_scores'])]
    optimal_xrsb = gaussian_range[np.argmax(results_dict['XRSB']['r2_scores'])]
    
    print(f"\n🎯 Optimal Gaussian Components:")
    print(f"   XRSA: {optimal_xrsa} components (R² = {max(results_dict['XRSA']['r2_scores']):.4f})")
    print(f"   XRSB: {optimal_xrsb} components (R² = {max(results_dict['XRSB']['r2_scores']):.4f})")
    
    print(f"\n📊 R-squared values for all components:")
    print("Gaussians |   XRSA R²   |   XRSB R²")
    print("-" * 35)
    for i, g in enumerate(gaussian_range):
        print(f"    {g:2d}    |   {results_dict['XRSA']['r2_scores'][i]:.4f}   |   {results_dict['XRSB']['r2_scores'][i]:.4f}")
    
    # Save results to DataFrame
    results_df = pd.DataFrame({
        'Gaussian_Components': list(gaussian_range),
        'XRSA_R2': results_dict['XRSA']['r2_scores'],
        'XRSB_R2': results_dict['XRSB']['r2_scores']
    })
    
    print(f"\n💾 Results DataFrame:")
    print(results_df.to_string(index=False))
    
    # Save to CSV
    output_file = f"r2_analysis_results_{os.path.basename(args.data).replace('.csv', '')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n🎉 Analysis completed successfully!")
    print("\n📖 USAGE INFORMATION:")
    print(f"   Basic usage: python {__file__}")
    print(f"   Custom data: python {__file__} --data data/2023_xrsa_xrsb.csv")
    print(f"   Custom range: python {__file__} --start 100 --end 500")
    print(f"   Show config: python {__file__} --show-config")
    print(f"   Help: python {__file__} --help")
