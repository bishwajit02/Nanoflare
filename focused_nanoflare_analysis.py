# ============================================================================
# FOCUSED NANOFLARE ANALYSIS: Small Time Window (6 hours or 1 day)
# ============================================================================
# This script implements baseline correction and nanoflare detection for a specific time window
# Choose your parameters below:

# CONFIGURATION - MODIFY THESE VALUES AS NEEDED
YEAR = 2023                    # Choose year: 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025
START_DAY = 100               # Start from day X of the year (0-364)
WINDOW_HOURS = 6              # Time window in hours (6 hours = 360 minutes)
# Alternative: Use 1 day = 24 hours

print(f"🔍 FOCUSED ANALYSIS CONFIGURATION")
print(f"📅 Year: {YEAR}")
print(f"🌅 Start Day: {START_DAY} (day of year)")
print(f"⏱️  Window: {WINDOW_HOURS} hours")
print("=" * 60)

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# Set style for enhanced visualizations
plt.style.use('default')
sns.set_palette("viridis")
sns.set_context("talk", font_scale=1.2)

# Load the specified year's data
data_path = f'data/{YEAR}_xrsa_xrsb.csv'
df = pd.read_csv(data_path)
print(f"📊 Loaded {YEAR} dataset: {df.shape[0]:,} samples")

# Clean the data
df_clean = df.dropna()
print(f"🧹 Cleaned dataset: {df_clean.shape[0]:,} samples")

# Convert time to days for easier selection
df_clean['time_days'] = df_clean['time_minutes'] / 1440  # 1440 minutes = 1 day

# Select the time window
start_minutes = START_DAY * 1440  # Convert day to minutes
end_minutes = start_minutes + (WINDOW_HOURS * 60)  # Add window duration

# Extract the time window
df_window = df_clean[
    (df_clean['time_minutes'] >= start_minutes) & 
    (df_clean['time_minutes'] < end_minutes)
].copy()

print(f"⏰ Time window: {start_minutes/1440:.2f} to {end_minutes/1440:.2f} days")
print(f"📈 Samples in window: {len(df_window)}")
print(f"🕐 Duration: {WINDOW_HOURS} hours ({WINDOW_HOURS*60} minutes)")

# Extract data for analysis
time_minutes = df_window['time_minutes'].values
time_hours = time_minutes / 60  # Convert to hours for better plotting
xrsa_flux = df_window['xrsa_flux_observed'].values
xrsb_flux = df_window['xrsb_flux_observed'].values

# Calculate log values
epsilon = 1e-12
log_xrsa = np.log10(np.maximum(xrsa_flux, epsilon))
log_xrsb = np.log10(np.maximum(xrsb_flux, epsilon))

print(f"\n📊 DATA STATISTICS:")
print(f"XRSA flux range: {xrsa_flux.min():.2e} to {xrsa_flux.max():.2e}")
print(f"XRSB flux range: {xrsb_flux.min():.2e} to {xrsb_flux.max():.2e}")
print(f"Log XRSA range: {log_xrsa.min():.2f} to {log_xrsa.max():.2f}")
print(f"Log XRSB range: {log_xrsb.min():.2f} to {log_xrsb.max():.2f}")

# ============================================================================
# STEP 1: BASELINE CORRECTION
# ============================================================================
print("\n🔄 STEP 1: APPLYING BASELINE CORRECTION")

# Install and import pybaselines if needed
import subprocess
import sys

try:
    from pybaselines import Baseline
    print("✅ pybaselines already installed")
except ImportError:
    print("📦 Installing pybaselines...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybaselines"])
    from pybaselines import Baseline
    print("✅ pybaselines installed and imported successfully")

# Initialize baseline correction
baseline_fitter = Baseline(time_minutes)

# Apply AsLS baseline correction to XRSA
print("🔄 Correcting XRSA baseline...")
xrsa_result = baseline_fitter.asls(log_xrsa, lam=1e6, p=0.01)
xrsa_baseline = xrsa_result[0]
xrsa_corrected = log_xrsa - xrsa_baseline

# Apply AsLS baseline correction to XRSB
print("🔄 Correcting XRSB baseline...")
xrsb_result = baseline_fitter.asls(log_xrsb, lam=1e6, p=0.01)
xrsb_baseline = xrsb_result[0]
xrsb_corrected = log_xrsb - xrsb_baseline

print("✅ Baseline correction completed!")

# ============================================================================
# STEP 2: SAVITZKY-GOLAY SMOOTHING
# ============================================================================
print("\n🔄 STEP 2: APPLYING SAVITZKY-GOLAY SMOOTHING")

from scipy.signal import savgol_filter

# Smoothing parameters
window_length = 21  # Must be odd
polyorder = 3

# Apply smoothing
xrsa_smoothed = savgol_filter(xrsa_corrected, window_length, polyorder)
xrsb_smoothed = savgol_filter(xrsb_corrected, window_length, polyorder)

print("✅ Smoothing completed!")

# ============================================================================
# STEP 3: NANOFLARE DETECTION
# ============================================================================
print("\n🔄 STEP 3: DETECTING NANOFLARES")

# Linear detrending
p_xrsa = np.poly1d(np.polyfit(time_minutes, xrsa_corrected, 1))
p_xrsb = np.poly1d(np.polyfit(time_minutes, xrsb_corrected, 1))

xrsa_detrended = xrsa_corrected - p_xrsa(time_minutes)
xrsb_detrended = xrsb_corrected - p_xrsb(time_minutes)

# Gaussian smoothing for baseline isolation
sigma_smooth = 2.0

xrsa_smooth = gaussian_filter1d(xrsa_detrended, sigma=sigma_smooth)
xrsb_smooth = gaussian_filter1d(xrsb_detrended, sigma=sigma_smooth)

# Calculate residuals
xrsa_residuals = xrsa_detrended - xrsa_smooth
xrsb_residuals = xrsb_detrended - xrsb_smooth

# Set detection thresholds (3-sigma)
threshold_xrsa = 3 * np.std(xrsa_residuals)
threshold_xrsb = 3 * np.std(xrsb_residuals)

# Detect events
xrsa_events = np.where(np.abs(xrsa_residuals) > threshold_xrsa)[0]
xrsb_events = np.where(np.abs(xrsb_residuals) > threshold_xrsb)[0]

print("✅ Nanoflare detection completed!")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("🎯 NANOFLARE DETECTION RESULTS")
print("=" * 60)

print(f"📅 Analysis Period: {YEAR}, Day {START_DAY} ({WINDOW_HOURS} hours)")
print(f"⏱️  Time Window: {time_hours[0]:.1f} to {time_hours[-1]:.1f} hours")
print(f"📊 Total Samples: {len(time_minutes)}")

print(f"\n🔍 DETECTION THRESHOLDS:")
print(f"XRSA threshold (3σ): {threshold_xrsa:.6f}")
print(f"XRSB threshold (3σ): {threshold_xrsb:.6f}")

print(f"\n⚡ EVENTS DETECTED:")
print(f"XRSA events: {len(xrsa_events)}")
print(f"XRSB events: {len(xrsb_events)}")

if len(xrsa_events) > 0:
    print(f"XRSA event times (hours): {time_hours[xrsa_events]}")
if len(xrsb_events) > 0:
    print(f"XRSB event times (hours): {time_hours[xrsb_events]}")

# Calculate event rates
event_rate_xrsa = len(xrsa_events) / WINDOW_HOURS
event_rate_xrsb = len(xrsb_events) / WINDOW_HOURS

print(f"\n📈 EVENT RATES:")
print(f"XRSA: {event_rate_xrsa:.2f} events/hour")
print(f"XRSB: {event_rate_xrsb:.2f} events/hour")

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)

# ============================================================================
# VISUALIZATION: COMPREHENSIVE PLOT
# ============================================================================
print("\n🎨 Creating comprehensive visualization...")

fig, axes = plt.subplots(3, 2, figsize=(16, 15))
fig.suptitle(f'Nanoflare Detection Analysis: {YEAR}, Day {START_DAY} ({WINDOW_HOURS} hours)', 
             fontsize=16, fontweight='bold')

# Row 1: Original vs Corrected
axes[0,0].plot(time_hours, log_xrsa, 'b-', alpha=0.7, label='Original log(XRSA)')
axes[0,0].plot(time_hours, xrsa_baseline, 'r-', linewidth=2, label='Baseline')
axes[0,0].set_title('XRSA: Original Signal with Baseline')
axes[0,0].set_ylabel('log₁₀(XRSA Flux)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(time_hours, log_xrsb, 'b-', alpha=0.7, label='Original log(XRSB)')
axes[0,1].plot(time_hours, xrsb_baseline, 'r-', linewidth=2, label='Baseline')
axes[0,1].set_title('XRSB: Original Signal with Baseline')
axes[0,1].set_ylabel('log₁₀(XRSB Flux)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Row 2: Baseline Corrected
axes[1,0].plot(time_hours, xrsa_corrected, 'g-', linewidth=1.5, label='Baseline Corrected')
axes[1,0].plot(time_hours, xrsa_smoothed, 'r-', linewidth=2, label='Smoothed')
axes[1,0].set_title('XRSA: Baseline Corrected & Smoothed')
axes[1,0].set_ylabel('Corrected log₁₀(XRSA)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(time_hours, xrsb_corrected, 'g-', linewidth=1.5, label='Baseline Corrected')
axes[1,1].plot(time_hours, xrsb_smoothed, 'r-', linewidth=2, label='Smoothed')
axes[1,1].set_title('XRSB: Baseline Corrected & Smoothed')
axes[1,1].set_ylabel('Corrected log₁₀(XRSB)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# Row 3: Residuals and Events
axes[2,0].plot(time_hours, xrsa_residuals, 'crimson', linewidth=1.5, alpha=0.8, label='Residuals')
axes[2,0].axhline(y=threshold_xrsa, color='red', linestyle='--', label=f'3σ Threshold (+{threshold_xrsa:.4f})')
axes[2,0].axhline(y=-threshold_xrsa, color='red', linestyle='--')
if len(xrsa_events) > 0:
    axes[2,0].scatter(time_hours[xrsa_events], xrsa_residuals[xrsa_events], 
                      color='orange', s=50, zorder=5, label='Detected Events')
axes[2,0].set_title('XRSA: Residuals & Event Detection')
axes[2,0].set_xlabel('Time (hours)')
axes[2,0].set_ylabel('Residuals')
axes[2,0].legend()
axes[2,0].grid(True, alpha=0.3)

axes[2,1].plot(time_hours, xrsb_residuals, 'navy', linewidth=1.5, alpha=0.8, label='Residuals')
axes[2,1].axhline(y=threshold_xrsb, color='blue', linestyle='--', label=f'3σ Threshold (+{threshold_xrsb:.4f})')
axes[2,1].axhline(y=-threshold_xrsb, color='blue', linestyle='--')
if len(xrsb_events) > 0:
    axes[2,1].scatter(time_hours[xrsb_events], xrsb_residuals[xrsb_events], 
                      color='orange', s=50, zorder=5, label='Detected Events')
axes[2,1].set_title('XRSB: Residuals & Event Detection')
axes[2,1].set_xlabel('Time (hours)')
axes[2,1].set_ylabel('Residuals')
axes[2,1].legend()
axes[2,1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("🎨 Visualization completed!")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n💾 Saving results...")

# Create results summary
results_summary = {
    'Year': YEAR,
    'Start_Day': START_DAY,
    'Window_Hours': WINDOW_HOURS,
    'Start_Time_Hours': time_hours[0],
    'End_Time_Hours': time_hours[-1],
    'Total_Samples': len(time_minutes),
    'XRSA_Events': len(xrsa_events),
    'XRSB_Events': len(xrsb_events),
    'XRSA_Threshold': threshold_xrsa,
    'XRSB_Threshold': threshold_xrsb,
    'XRSA_Event_Rate': event_rate_xrsa,
    'XRSB_Event_Rate': event_rate_xrsb
}

# Save to CSV
results_df = pd.DataFrame([results_summary])
results_filename = f'nanoflare_analysis_{YEAR}_day{START_DAY}_{WINDOW_HOURS}h.csv'
results_df.to_csv(results_filename, index=False)
print(f"✅ Results saved to: {results_filename}")

print("\n🎉 FOCUSED ANALYSIS COMPLETED! 🎉")
print("You can now modify the configuration variables at the top to analyze different time periods.")
