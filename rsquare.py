import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from pybaselines import Baseline
import warnings
warnings.filterwarnings('ignore')

# 1) Load the file in colab
# Load your dataset
file_path = 'data/2024_xrsa_xrsb.csv'
data = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print(f"Data shape: {data.shape}")
print("Columns:", data.columns.tolist())
print("\nData preview:")
print(data.head())

# 2) Use baseline correction pybaseline asls
print("\nApplying ASLS baseline correction...")

# Extract data
time_minutes = data['time_minutes'].values
xrsa_original = data['xrsa_flux_observed'].values
xrsb_original = data['xrsb_flux_observed'].values

# Apply ASLS baseline correction
baseline_fitter = Baseline()

# XRSA baseline correction
xrsa_baseline, xrsa_params = baseline_fitter.asls(xrsa_original, lam=1e7, p=0.001)
xrsa_corrected = xrsa_original - xrsa_baseline

# XRSB baseline correction
xrsb_baseline, xrsb_params = baseline_fitter.asls(xrsb_original, lam=1e7, p=0.001)
xrsb_corrected = xrsb_original - xrsb_baseline

print("Baseline correction completed!")

# Create DataFrame with corrected data
corrected_data = data.copy()
corrected_data['xrsa_baseline'] = xrsa_baseline
corrected_data['xrsa_corrected'] = xrsa_corrected
corrected_data['xrsb_baseline'] = xrsb_baseline
corrected_data['xrsb_corrected'] = xrsb_corrected

# 3) Find peaks using scipy for 6h time range (0-360 minutes)
print("\nFinding peaks in 6h time range (0-360 minutes)...")

# Filter for 6h time range
mask_6h = (time_minutes >= 0) & (time_minutes <= 360)
time_6h = time_minutes[mask_6h]
xrsa_6h = xrsa_corrected[mask_6h]
xrsb_6h = xrsb_corrected[mask_6h]

# Find peaks for XRSA
xrsa_peaks, xrsa_properties = signal.find_peaks(
    xrsa_6h,
    height=np.percentile(xrsa_6h, 75),
    prominence=np.std(xrsa_6h)/3,
    distance=20
)

# Find peaks for XRSB
xrsb_peaks, xrsb_properties = signal.find_peaks(
    xrsb_6h,
    height=np.percentile(xrsb_6h, 75),
    prominence=np.std(xrsb_6h)/3,
    distance=20
)

print(f"XRSA peaks found: {len(xrsa_peaks)}")
print(f"XRSB peaks found: {len(xrsb_peaks)}")

# Plot baseline correction and peaks
plt.figure(figsize=(15, 10))

# XRSA plot
plt.subplot(2, 2, 1)
plt.plot(time_6h/60, xrsa_6h, 'b-', label='XRSA Baseline-Corrected', linewidth=1.5)
plt.plot(time_6h[xrsa_peaks]/60, xrsa_6h[xrsa_peaks], 'ro', label='XRSA Peaks', markersize=6)
plt.title('XRSA: Baseline-Corrected with Peaks')
plt.xlabel('Time (hours)')
plt.ylabel('Flux')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(time_6h/60, xrsa_original[mask_6h], 'g-', alpha=0.7, label='XRSA Original', linewidth=1)
plt.plot(time_6h/60, xrsa_baseline[mask_6h], 'r-', label='XRSA Baseline', linewidth=2)
plt.title('XRSA: Original vs Baseline')
plt.xlabel('Time (hours)')
plt.ylabel('Flux')
plt.legend()
plt.grid(True, alpha=0.3)

# XRSB plot
plt.subplot(2, 2, 3)
plt.plot(time_6h/60, xrsb_6h, 'b-', label='XRSB Baseline-Corrected', linewidth=1.5)
plt.plot(time_6h[xrsb_peaks]/60, xrsb_6h[xrsb_peaks], 'ro', label='XRSB Peaks', markersize=6)
plt.title('XRSB: Baseline-Corrected with Peaks')
plt.xlabel('Time (hours)')
plt.ylabel('Flux')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(time_6h/60, xrsb_original[mask_6h], 'g-', alpha=0.7, label='XRSB Original', linewidth=1)
plt.plot(time_6h/60, xrsb_baseline[mask_6h], 'r-', label='XRSB Baseline', linewidth=2)
plt.title('XRSB: Original vs Baseline')
plt.xlabel('Time (hours)')
plt.ylabel('Flux')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4) Gaussian components analysis
print("\nPerforming Gaussian fit analysis with components 10 to 20...")

def create_gaussian_components(time, signal_data, num_gaussians):
    """Create Gaussian components with optimized parameters"""
    def gaussian(x, amplitude, mean, sigma):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    time_range = time.max() - time.min()

    # Use peak information to initialize means
    peaks, _ = signal.find_peaks(signal_data, height=np.percentile(signal_data, 75))

    if len(peaks) >= num_gaussians:
        # Use actual peaks if available
        peak_amplitudes = signal_data[peaks]
        sorted_indices = np.argsort(peak_amplitudes)[::-1]
        selected_peaks = peaks[sorted_indices[:num_gaussians]]
        means = time[selected_peaks]
        amplitudes = signal_data[selected_peaks] * 0.8
    else:
        # Evenly spaced means
        means = np.linspace(time.min() + time_range * 0.1, time.max() - time_range * 0.1, num_gaussians)
        amplitudes = np.interp(means, time, signal_data) * 0.8

    # Calculate sigmas based on time range and number of components
    base_sigma = time_range / (num_gaussians * 2)
    sigmas = np.full(num_gaussians, base_sigma)

    # Create components
    components = []
    total_fit = np.zeros_like(time, dtype=np.float64)

    for i in range(num_gaussians):
        component = gaussian(time, amplitudes[i], means[i], sigmas[i])
        components.append(component)
        total_fit += component

    return total_fit, components, amplitudes, means, sigmas

def calculate_gaussian_fit(signal_data, time_data, num_gaussians):
    """Calculate Gaussian fit and R-squared"""
    gaussian_fit, components, amplitudes, means, sigmas = create_gaussian_components(time_data, signal_data, num_gaussians)
    r2 = r2_score(signal_data, gaussian_fit)
    return gaussian_fit, components, amplitudes, means, sigmas, r2

# Analyze for Gaussian components 10 to 20
gaussian_range = range(10, 21)
results = {'XRSA': {}, 'XRSB': {}}

for signal_name, signal_data in [('XRSA', xrsa_6h), ('XRSB', xrsb_6h)]:
    print(f"\nAnalyzing {signal_name}...")

    r2_scores = []
    all_fits = []
    all_components = []
    all_amplitudes = []
    all_means = []
    all_sigmas = []

    for num_gaussians in gaussian_range:
        gaussian_fit, components, amplitudes, means, sigmas, r2 = calculate_gaussian_fit(
            signal_data, time_6h, num_gaussians
        )

        r2_scores.append(r2)
        all_fits.append(gaussian_fit)
        all_components.append(components)
        all_amplitudes.append(amplitudes)
        all_means.append(means)
        all_sigmas.append(sigmas)

        print(f"G={num_gaussians:2d}: R² = {r2:.4f}")

    # Store results
    results[signal_name]['r2_scores'] = r2_scores
    results[signal_name]['fits'] = all_fits
    results[signal_name]['components'] = all_components
    results[signal_name]['amplitudes'] = all_amplitudes
    results[signal_name]['means'] = all_means
    results[signal_name]['sigmas'] = all_sigmas

# 6) Plot Gaussian Vs R-Square
print("\nPlotting Gaussian Components vs R-Squared...")

plt.figure(figsize=(12, 6))
plt.plot(gaussian_range, results['XRSA']['r2_scores'], 'bo-', linewidth=2, markersize=8, label='XRSA')
plt.plot(gaussian_range, results['XRSB']['r2_scores'], 'ro-', linewidth=2, markersize=8, label='XRSB')

plt.title('Gaussian Components vs R-Squared (6h Time Range)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Gaussian Components', fontsize=12)
plt.ylabel('R-Squared', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xticks(gaussian_range)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 5) Draw best fit composition for each components
print("\nDrawing best fit compositions...")

# Find optimal number of components for each signal
optimal_xrsa = gaussian_range[np.argmax(results['XRSA']['r2_scores'])]
optimal_xrsb = gaussian_range[np.argmax(results['XRSB']['r2_scores'])]

print(f"Optimal components - XRSA: {optimal_xrsa}, XRSB: {optimal_xrsb}")

# Plot best fits
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# XRSA Best Fit
xrsa_idx = optimal_xrsa - 10
axes[0, 0].plot(time_6h/60, xrsa_6h, 'b-', label='XRSA Data', linewidth=1.5, alpha=0.8)
axes[0, 0].plot(time_6h/60, results['XRSA']['fits'][xrsa_idx], 'r-',
               label=f'Gaussian Fit (G={optimal_xrsa})', linewidth=2)
axes[0, 0].set_title(f'XRSA: Best Gaussian Fit\nG={optimal_xrsa}, R²={results["XRSA"]["r2_scores"][xrsa_idx]:.3f}')
axes[0, 0].set_xlabel('Time (hours)')
axes[0, 0].set_ylabel('Flux')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# XRSA Components
colors = plt.cm.viridis(np.linspace(0, 1, len(results['XRSA']['components'][xrsa_idx])))
for i, component in enumerate(results['XRSA']['components'][xrsa_idx]):
    axes[0, 1].plot(time_6h/60, component, '--', color=colors[i], alpha=0.6, linewidth=1,
                   label=f'Component {i+1}' if i < 3 else "")
axes[0, 1].plot(time_6h/60, results['XRSA']['fits'][xrsa_idx], 'r-', label='Total Fit', linewidth=2)
axes[0, 1].set_title(f'XRSA: Gaussian Components (G={optimal_xrsa})')
axes[0, 1].set_xlabel('Time (hours)')
axes[0, 1].set_ylabel('Flux')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# XRSB Best Fit
xrsb_idx = optimal_xrsb - 10
axes[1, 0].plot(time_6h/60, xrsb_6h, 'b-', label='XRSB Data', linewidth=1.5, alpha=0.8)
axes[1, 0].plot(time_6h/60, results['XRSB']['fits'][xrsb_idx], 'r-',
               label=f'Gaussian Fit (G={optimal_xrsb})', linewidth=2)
axes[1, 0].set_title(f'XRSB: Best Gaussian Fit\nG={optimal_xrsb}, R²={results["XRSB"]["r2_scores"][xrsb_idx]:.3f}')
axes[1, 0].set_xlabel('Time (hours)')
axes[1, 0].set_ylabel('Flux')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# XRSB Components
colors = plt.cm.plasma(np.linspace(0, 1, len(results['XRSB']['components'][xrsb_idx])))
for i, component in enumerate(results['XRSB']['components'][xrsb_idx]):
    axes[1, 1].plot(time_6h/60, component, '--', color=colors[i], alpha=0.6, linewidth=1,
                   label=f'Component {i+1}' if i < 3 else "")
axes[1, 1].plot(time_6h/60, results['XRSB']['fits'][xrsb_idx], 'r-', label='Total Fit', linewidth=2)
axes[1, 1].set_title(f'XRSB: Gaussian Components (G={optimal_xrsb})')
axes[1, 1].set_xlabel('Time (hours)')
axes[1, 1].set_ylabel('Flux')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Display summary results
print("\n" + "="*60)
print("SUMMARY RESULTS")
print("="*60)

print(f"\nOptimal Gaussian Components:")
print(f"XRSA: {optimal_xrsa} components (R² = {results['XRSA']['r2_scores'][xrsa_idx]:.4f})")
print(f"XRSB: {optimal_xrsb} components (R² = {results['XRSB']['r2_scores'][xrsb_idx]:.4f})")

print(f"\nR-squared values for all components:")
print("Gaussians |   XRSA R²   |   XRSB R²")
print("-" * 35)
for i, g in enumerate(gaussian_range):
    print(f"    {g:2d}    |   {results['XRSA']['r2_scores'][i]:.4f}   |   {results['XRSB']['r2_scores'][i]:.4f}")

# Save results to DataFrame
results_df = pd.DataFrame({
    'Gaussian_Components': list(gaussian_range),
    'XRSA_R2': results['XRSA']['r2_scores'],
    'XRSB_R2': results['XRSB']['r2_scores']
})

print(f"\nResults DataFrame:")
print(results_df.to_string(index=False))

print("\nAnalysis completed successfully!")