import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, peak_widths
from astropy.timeseries import BoxLeastSquares, LombScargle
import warnings
warnings.filterwarnings('ignore')

def estimate_transit_depth(time, flux, period, duration):
    """Enhanced transit depth estimation with better phase folding"""
    if not np.isfinite(period) or not np.isfinite(duration) or period <= 0 or duration <= 0:
        return np.nan
    
    try:
        phase = ((time - time.min()) % period) / period
        
        # Define transit window more carefully
        transit_width = min(duration / period, 0.3)  # Cap at 30% of period
        in_transit = (phase < transit_width/2) | (phase > 1 - transit_width/2)
        
        # Require minimum points for reliable estimate
        if np.sum(in_transit) < 3 or np.sum(~in_transit) < 10:
            return np.nan
        
        out_flux = flux[~in_transit]
        in_flux = flux[in_transit]
        
        # Use robust statistics
        out_median = np.nanmedian(out_flux)
        in_median = np.nanmedian(in_flux)
        
        if out_median <= 0:
            return np.nan
        
        depth = 1.0 - in_median / out_median
        return float(max(0, depth))  # Depth should be positive
        
    except Exception:
        return np.nan

def estimate_duration_from_bls(bls, time, flux, period):
    """Enhanced duration estimation using BLS grid search"""
    if not np.isfinite(period) or period <= 0:
        return np.nan
    
    try:
        # Create duration grid based on period
        min_dur = max(0.001, 0.01 * period)
        max_dur = min(0.3 * period, 0.5)
        durations = np.logspace(np.log10(min_dur), np.log10(max_dur), 30)
        
        powers = []
        for d in durations:
            try:
                res = bls.power([period], [d])
                powers.append(np.nanmax(res.power))
            except:
                powers.append(0.0)
        
        if len(powers) == 0:
            return np.nan
            
        best_idx = np.nanargmax(powers)
        return float(durations[best_idx])
        
    except Exception:
        return np.nan

def run_enhanced_bls(time, flux):
    """Enhanced BLS analysis with better period grid and statistics"""
    try:
        baseline = time.max() - time.min()
        if baseline <= 0:
            return None, {'orbital_period': np.nan, 'duration': np.nan, 't0': np.nan, 'bls_power': np.nan}
        
        # Enhanced period grid - focus on likely exoplanet periods
        min_period = max(0.2, 0.01 * baseline)
        max_period = min(50.0, 0.8 * baseline)  # Cap at 50 days
        
        # Logarithmic spacing for better coverage
        n_periods = min(5000, int(baseline * 100))
        periods = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)
        
        # Duration grid - more realistic transit durations
        durations = np.logspace(np.log10(0.001), np.log10(0.3), 20)
        
        # Run BLS
        bls = BoxLeastSquares(time, flux)
        power = bls.power(periods, durations)
        
        # Find best period
        best_idx = np.nanargmax(power.power)
        best_period = float(power.period[best_idx])
        best_duration = float(power.duration[best_idx])
        best_t0 = float(power.transit_time[best_idx])
        best_power = float(np.nanmax(power.power))
        
        # Calculate period accuracy (how well-defined the peak is)
        period_accuracy = calculate_period_accuracy(power.power, best_idx)
        
        # Calculate depth accuracy
        depth_accuracy = calculate_depth_accuracy(bls, best_period, best_duration, time, flux)
        
        return bls, {
            'orbital_period': best_period,
            'duration': best_duration,
            't0': best_t0,
            'bls_power': best_power,
            'bls_period_accuracy': period_accuracy,
            'bls_depth_accuracy': depth_accuracy
        }
        
    except Exception:
        return None, {'orbital_period': np.nan, 'duration': np.nan, 't0': np.nan, 'bls_power': np.nan,
                     'bls_period_accuracy': np.nan, 'bls_depth_accuracy': np.nan}

def calculate_period_accuracy(power_array, best_idx):
    """Calculate how well-defined the period peak is"""
    try:
        if len(power_array) < 5:
            return np.nan
        
        best_power = power_array[best_idx]
        
        # Calculate ratio to surrounding power
        start_idx = max(0, best_idx - 10)
        end_idx = min(len(power_array), best_idx + 11)
        surrounding = np.concatenate([power_array[start_idx:best_idx], power_array[best_idx+1:end_idx]])
        
        if len(surrounding) == 0:
            return np.nan
        
        median_power = np.nanmedian(surrounding)
        if median_power <= 0:
            return 0.0
        
        return float(min(1.0, best_power / median_power / 10.0))  # Normalize
        
    except Exception:
        return np.nan

def calculate_depth_accuracy(bls, period, duration, time, flux):
    """Calculate consistency of transit depth across multiple transits"""
    try:
        if not np.isfinite(period) or period <= 0:
            return np.nan
        
        # Fold the light curve
        phase = ((time - time.min()) % period) / period
        
        # Get individual transits
        n_transits = int((time.max() - time.min()) / period)
        if n_transits < 2:
            return np.nan
        
        depths = []
        for i in range(n_transits):
            start_time = time.min() + i * period
            end_time = start_time + period
            
            mask = (time >= start_time) & (time < end_time)
            if np.sum(mask) < 10:
                continue
            
            t_transit = time[mask]
            f_transit = flux[mask]
            
            depth = estimate_transit_depth(t_transit, f_transit, period, duration)
            if np.isfinite(depth):
                depths.append(depth)
        
        if len(depths) < 2:
            return np.nan
        
        # Return consistency measure (1 - coefficient of variation)
        depths = np.array(depths)
        mean_depth = np.mean(depths)
        std_depth = np.std(depths)
        
        if mean_depth <= 0:
            return np.nan
        
        cv = std_depth / mean_depth
        return float(max(0.0, 1.0 - cv))
        
    except Exception:
        return np.nan

def periodogram_lomb_scargle(time, flux):
    """Enhanced Lomb-Scargle periodogram with better frequency grid"""
    try:
        baseline = time.max() - time.min()
        if baseline <= 0 or len(time) < 10:
            return {'frequency': np.array([0.0]), 'power': np.array([0.0])}
        
        # Enhanced frequency grid
        fmin = 0.5 / baseline  # Nyquist consideration
        dt = np.median(np.diff(np.sort(time)))
        fmax = min(24.0, 0.5 / dt) if dt > 0 else 24.0  # Cap at daily frequency
        
        # Logarithmic spacing for better resolution at low frequencies
        freq = np.logspace(np.log10(fmin), np.log10(fmax), 8000)
        
        # Normalize flux
        flux_norm = flux - np.nanmedian(flux)
        flux_norm = flux_norm / np.nanstd(flux_norm) if np.nanstd(flux_norm) > 0 else flux_norm
        
        ls = LombScargle(time, flux_norm)
        power = ls.power(freq)
        
        return {'frequency': freq, 'power': power}
        
    except Exception:
        return {'frequency': np.array([0.0]), 'power': np.array([0.0])}

def phase_fold(time, flux, period, bins=100):
    """Enhanced phase folding with outlier rejection"""
    try:
        if not np.isfinite(period) or period <= 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        phase = ((time - time.min()) % period) / period
        
        # Sort by phase
        order = np.argsort(phase)
        phase_sorted = phase[order]
        flux_sorted = flux[order]
        
        # Create bins
        bins_edges = np.linspace(0, 1, bins + 1)
        bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
        
        # Bin the data with outlier rejection
        binned = []
        for i in range(bins):
            mask = (phase_sorted >= bins_edges[i]) & (phase_sorted < bins_edges[i+1])
            if np.sum(mask) == 0:
                binned.append(np.nan)
                continue
            
            bin_flux = flux_sorted[mask]
            
            # Simple outlier rejection (3-sigma clipping)
            if len(bin_flux) > 3:
                median_flux = np.nanmedian(bin_flux)
                std_flux = np.nanstd(bin_flux)
                if std_flux > 0:
                    outlier_mask = np.abs(bin_flux - median_flux) < 3 * std_flux
                    bin_flux = bin_flux[outlier_mask]
            
            if len(bin_flux) > 0:
                binned.append(np.nanmedian(bin_flux))
            else:
                binned.append(np.nan)
        
        return phase_sorted, flux_sorted, bin_centers, np.array(binned)
        
    except Exception:
        return np.array([]), np.array([]), np.array([]), np.array([])

def calculate_statistical_features(time, flux, period, duration):
    """Calculate advanced statistical features"""
    try:
        features = {}
        
        # Red noise estimation
        if len(flux) > 10:
            # Simple red noise estimate using autocorrelation
            autocorr = np.corrcoef(flux[:-1], flux[1:])[0, 1] if len(flux) > 1 else 0
            features['flux_autocorrelation'] = float(autocorr) if np.isfinite(autocorr) else 0.0
            features['red_noise_level'] = float(np.abs(autocorr)) if np.isfinite(autocorr) else 0.0
        else:
            features['flux_autocorrelation'] = 0.0
            features['red_noise_level'] = 0.0
        
        # Period stability (consistency across different time segments)
        if np.isfinite(period) and period > 0 and len(time) > 50:
            n_segments = 4
            segment_size = len(time) // n_segments
            period_estimates = []
            
            for i in range(n_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(time)
                
                if end_idx - start_idx < 20:
                    continue
                
                t_seg = time[start_idx:end_idx]
                f_seg = flux[start_idx:end_idx]
                
                try:
                    bls_seg = BoxLeastSquares(t_seg, f_seg)
                    periods_test = np.linspace(max(0.5, period * 0.8), period * 1.2, 50)
                    power_seg = bls_seg.power(periods_test, duration)
                    best_period_seg = periods_test[np.nanargmax(power_seg.power)]
                    period_estimates.append(best_period_seg)
                except:
                    continue
            
            if len(period_estimates) > 1:
                period_std = np.std(period_estimates)
                period_mean = np.mean(period_estimates)
                if period_mean > 0:
                    features['period_stability'] = float(max(0.0, 1.0 - period_std / period_mean))
                else:
                    features['period_stability'] = 0.0
            else:
                features['period_stability'] = 0.0
        else:
            features['period_stability'] = 0.0
        
        return features
        
    except Exception:
        return {
            'flux_autocorrelation': 0.0,
            'red_noise_level': 0.0,
            'period_stability': 0.0
        }

def extract_all_features(df: pd.DataFrame, stellar_params=None) -> dict:
    """Enhanced feature extraction with comprehensive exoplanet indicators"""
    t = df['time'].values.astype(float)
    f = df['flux'].values.astype(float)
    
    if len(t) == 0 or len(f) == 0:
        # Return default values for empty data
        return {key: np.nan for key in [
            'orbital_period', 'transit_duration', 'transit_depth', 'transit_count', 'snr_proxy',
            'ingress_slope', 'egress_slope', 'curve_symmetry', 'shape_asymmetry', 'shape_peakedness',
            'flux_range', 'flux_scatter', 'flux_minimum', 'flux_median', 'flux_std',
            'bls_power', 'bls_period_accuracy', 'bls_depth_accuracy',
            'estimated_planet_radius_re', 'planet_star_radius_ratio', 'equilibrium_temperature',
            'stellar_radius', 'stellar_mass', 'stellar_temperature', 'stellar_metallicity',
            'stellar_surface_gravity', 'stellar_magnitude',
            'period_snr', 'depth_snr', 'duration_snr', 'period_stability',
            'flux_autocorrelation', 'red_noise_level'
        ]}
    
    # Global flux statistics
    flux_range = float(np.nanmax(f) - np.nanmin(f))
    flux_scatter = float(np.nanstd(f))
    flux_min = float(np.nanmin(f))
    flux_median = float(np.nanmedian(f))
    flux_std = float(np.nanstd(f))
    asymmetry = float(skew(f, nan_policy='omit'))
    peakedness = float(kurtosis(f, nan_policy='omit'))
    
    # Enhanced BLS search
    try:
        bls, bls_res = run_enhanced_bls(t, f)
        period = bls_res['orbital_period']
        duration = bls_res['duration']
        t0 = bls_res['t0']
        bls_power = bls_res['bls_power']
        bls_period_accuracy = bls_res['bls_period_accuracy']
        bls_depth_accuracy = bls_res['bls_depth_accuracy']
    except Exception:
        period = np.nan
        duration = np.nan
        t0 = np.nan
        bls_power = np.nan
        bls_period_accuracy = np.nan
        bls_depth_accuracy = np.nan
    
    # Enhanced transit depth estimate
    depth = estimate_transit_depth(t, f, period, duration)
    
    # Transit count and SNR calculations
    baseline = t.max() - t.min() if len(t) > 1 else np.nan
    transit_count = float(np.floor(baseline / period)) if np.isfinite(period) and period > 0 else 0.0
    
    # Enhanced SNR calculation
    if np.isfinite(depth) and np.isfinite(flux_scatter) and flux_scatter > 0:
        snr = float((depth / flux_scatter) * np.sqrt(max(transit_count, 1)))
    else:
        snr = np.nan
    
    # Enhanced ingress/egress slope calculation
    try:
        if np.isfinite(period) and np.isfinite(t0) and np.isfinite(duration) and period > 0:
            phase = ((t - t0) % period) / period
            transit_width = min(duration / period, 0.3)
            
            # Define ingress and egress regions more precisely
            ingress_mask = (phase < transit_width/4) | (phase > 1 - transit_width/4)
            egress_mask = ((phase > transit_width/4) & (phase < transit_width/2)) | \
                         ((phase > 1 - transit_width/2) & (phase < 1 - transit_width/4))
            
            if np.sum(ingress_mask) > 5 and np.sum(egress_mask) > 5:
                ingress_points = f[ingress_mask]
                egress_points = f[egress_mask]
                t_ingress = t[ingress_mask]
                t_egress = t[egress_mask]
                
                if len(ingress_points) > 1 and len(egress_points) > 1:
                    ingress_slope = float(np.polyfit(t_ingress, ingress_points, 1)[0])
                    egress_slope = float(np.polyfit(t_egress, egress_points, 1)[0])
                else:
                    ingress_slope = np.nan
                    egress_slope = np.nan
            else:
                ingress_slope = np.nan
                egress_slope = np.nan
        else:
            ingress_slope = np.nan
            egress_slope = np.nan
    except Exception:
        ingress_slope = np.nan
        egress_slope = np.nan
    
    # Enhanced symmetry calculation
    try:
        if np.isfinite(period) and np.isfinite(t0) and period > 0:
            phase = ((t - t0) % period) / period
            first_half = f[(phase >= 0.0) & (phase < 0.5)]
            second_half = f[(phase >= 0.5) & (phase <= 1.0)]
            
            if len(first_half) > 5 and len(second_half) > 5:
                symmetry = float(np.abs(np.nanmedian(first_half) - np.nanmedian(second_half)))
            else:
                symmetry = np.nan
        else:
            symmetry = np.nan
    except Exception:
        symmetry = np.nan
    
    # Stellar parameters processing
    stellar_radius = np.nan
    stellar_mass = np.nan
    stellar_temperature = np.nan
    stellar_metallicity = np.nan
    stellar_surface_gravity = np.nan
    stellar_magnitude = np.nan
    
    if stellar_params and isinstance(stellar_params, dict):
        stellar_radius = float(stellar_params.get('radius_sun', np.nan))
        stellar_mass = float(stellar_params.get('mass_sun', np.nan))
        stellar_temperature = float(stellar_params.get('teff', np.nan))
        stellar_metallicity = float(stellar_params.get('feh', np.nan))
        stellar_surface_gravity = float(stellar_params.get('logg', np.nan))
        stellar_magnitude = float(stellar_params.get('mag_tess', stellar_params.get('mag_kepler', np.nan)))
    
    # Planet characteristics
    planet_radius_re = np.nan
    planet_star_radius_ratio = np.nan
    equilibrium_temperature = np.nan
    
    if np.isfinite(depth) and depth > 0:
        planet_star_radius_ratio = float(np.sqrt(depth))
        
        if np.isfinite(stellar_radius) and stellar_radius > 0:
            planet_radius_re = float(planet_star_radius_ratio * stellar_radius * 109.1)  # Earth radii
        
        if np.isfinite(stellar_temperature) and np.isfinite(stellar_radius) and np.isfinite(period):
            # Calculate equilibrium temperature
            if stellar_temperature > 0 and stellar_radius > 0 and period > 0:
                # Simplified calculation assuming circular orbit
                a_over_rstar = (period / 365.25) ** (2/3) * (stellar_mass if np.isfinite(stellar_mass) else 1.0) ** (1/3)
                if a_over_rstar > 0:
                    equilibrium_temperature = float(stellar_temperature * np.sqrt(stellar_radius / (2 * a_over_rstar)))
    
    # Statistical features
    stat_features = calculate_statistical_features(t, f, period, duration)
    
    # Enhanced SNR calculations
    period_snr = snr * (bls_period_accuracy if np.isfinite(bls_period_accuracy) else 0.5) if np.isfinite(snr) else np.nan
    depth_snr = (depth / flux_scatter) if np.isfinite(depth) and np.isfinite(flux_scatter) and flux_scatter > 0 else np.nan
    duration_snr = (duration / (baseline / len(t))) if np.isfinite(duration) and np.isfinite(baseline) and len(t) > 0 else np.nan
    
    # Curve symmetry (overall)
    curve_symmetry = float(np.abs(asymmetry)) if np.isfinite(asymmetry) else np.nan
    
    # Assemble all features
    features = {
        # Basic transit features
        'orbital_period': float(period) if np.isfinite(period) else np.nan,
        'transit_duration': float(duration) if np.isfinite(duration) else np.nan,
        'transit_depth': float(depth) if np.isfinite(depth) else np.nan,
        'transit_count': float(transit_count),
        'snr_proxy': float(snr) if np.isfinite(snr) else np.nan,
        'ingress_slope': float(ingress_slope) if np.isfinite(ingress_slope) else np.nan,
        'egress_slope': float(egress_slope) if np.isfinite(egress_slope) else np.nan,
        'curve_symmetry': float(curve_symmetry) if np.isfinite(curve_symmetry) else np.nan,
        'shape_asymmetry': float(asymmetry) if np.isfinite(asymmetry) else np.nan,
        'shape_peakedness': float(peakedness) if np.isfinite(peakedness) else np.nan,
        
        # Flux characteristics
        'flux_range': float(flux_range) if np.isfinite(flux_range) else np.nan,
        'flux_scatter': float(flux_scatter) if np.isfinite(flux_scatter) else np.nan,
        'flux_minimum': float(flux_min) if np.isfinite(flux_min) else np.nan,
        'flux_median': float(flux_median) if np.isfinite(flux_median) else np.nan,
        'flux_std': float(flux_std) if np.isfinite(flux_std) else np.nan,
        
        # BLS features
        'bls_power': float(bls_power) if np.isfinite(bls_power) else np.nan,
        'bls_period_accuracy': float(bls_period_accuracy) if np.isfinite(bls_period_accuracy) else np.nan,
        'bls_depth_accuracy': float(bls_depth_accuracy) if np.isfinite(bls_depth_accuracy) else np.nan,
        
        # Planet characteristics
        'estimated_planet_radius_re': float(planet_radius_re) if np.isfinite(planet_radius_re) else np.nan,
        'planet_star_radius_ratio': float(planet_star_radius_ratio) if np.isfinite(planet_star_radius_ratio) else np.nan,
        'equilibrium_temperature': float(equilibrium_temperature) if np.isfinite(equilibrium_temperature) else np.nan,
        
        # Stellar characteristics
        'stellar_radius': float(stellar_radius) if np.isfinite(stellar_radius) else np.nan,
        'stellar_mass': float(stellar_mass) if np.isfinite(stellar_mass) else np.nan,
        'stellar_temperature': float(stellar_temperature) if np.isfinite(stellar_temperature) else np.nan,
        'stellar_metallicity': float(stellar_metallicity) if np.isfinite(stellar_metallicity) else np.nan,
        'stellar_surface_gravity': float(stellar_surface_gravity) if np.isfinite(stellar_surface_gravity) else np.nan,
        'stellar_magnitude': float(stellar_magnitude) if np.isfinite(stellar_magnitude) else np.nan,
        
        # Statistical features
        'period_snr': float(period_snr) if np.isfinite(period_snr) else np.nan,
        'depth_snr': float(depth_snr) if np.isfinite(depth_snr) else np.nan,
        'duration_snr': float(duration_snr) if np.isfinite(duration_snr) else np.nan,
        'period_stability': float(stat_features['period_stability']),
        'flux_autocorrelation': float(stat_features['flux_autocorrelation']),
        'red_noise_level': float(stat_features['red_noise_level'])
    }
    
    return features

# Keep backward compatibility functions
def run_bls(time, flux):
    """Backward compatibility wrapper"""
    bls, results = run_enhanced_bls(time, flux)
    return bls, {
        'orbital_period': results['orbital_period'],
        'duration': results['duration'],
        't0': results['t0'],
        'bls_power': results['bls_power']
    }