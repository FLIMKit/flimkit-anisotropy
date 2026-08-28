from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as nd_shift
from scipy.optimize import least_squares, minimize


@dataclass
class AnisotropyResult:
    time_ns: np.ndarray
    parallel_decay: np.ndarray
    perpendicular_decay: np.ndarray
    anisotropy_decay: np.ndarray
    anisotropy_cube: np.ndarray
    anisotropy_map: np.ndarray
    parallel_intensity: np.ndarray
    perpendicular_intensity: np.ndarray
    parallel_background: float
    perpendicular_background: float
    spatial_window: int
    stride: int
    perpendicular_shift: tuple
    valid_mask: np.ndarray
    map_valid_mask: np.ndarray
    total_counts: np.ndarray
    window_origins: np.ndarray
    g_factor: float
    parallel_exposure: float
    perpendicular_exposure: float
    metadata: dict = field(default_factory=dict)
    polarized_fit: 'PolarizedFitResult | None' = None
    multicomponent_fit: 'MulticomponentFitResult | None' = None
    late_window_stability: 'LateWindowStabilityResult | None' = None


@dataclass
class LateWindowStabilityResult:
    time_ns: np.ndarray
    nested_scale: np.ndarray
    nested_standard_error: np.ndarray
    rolling_scale: np.ndarray
    rolling_bins: int
    selected_start_bin: int
    selected_scale: float
    selected_parallel_photons: float
    selected_perpendicular_photons: float
    interpretation: str = 'effective_late_window_scale'


@dataclass
class MultistartFitRecord:
    start_index: int
    parameter_vector: np.ndarray
    rotational_correlation_times_ns: np.ndarray
    component_weights: np.ndarray
    poisson_deviance: float
    success: bool
    message: str
    parameters_at_bounds: tuple


@dataclass
class MulticomponentModelFit:
    component_count: int
    rotational_correlation_times_ns: np.ndarray
    component_weights: np.ndarray
    intensity_lifetime_ns: float
    initial_anisotropy: float
    amplitude: float
    parallel_model: np.ndarray
    perpendicular_model: np.ndarray
    parallel_residual: np.ndarray
    perpendicular_residual: np.ndarray
    poisson_deviance: float
    degrees_of_freedom: int
    deviance_per_degree_of_freedom: float
    bic: float
    aicc: float
    success: bool
    message: str
    parameters_at_bounds: tuple
    identifiable: bool
    identifiability_warnings: tuple
    common_irf_shift_bins: float
    parallel_background: float
    perpendicular_background: float
    component_bounds_ns: np.ndarray
    multistart_count: int
    multistart_records: tuple
    eligible_for_selection: bool
    parameter_names: tuple


@dataclass
class MulticomponentFitResult:
    max_components: int
    selected_component_count: int
    candidates: tuple

    @property
    def selected_fit(self):
        if self.selected_component_count == 0:
            return None
        return self.candidates[self.selected_component_count - 1]

    @property
    def has_resolved_model(self):
        return bool(
            self.selected_fit is not None and self.selected_fit.identifiable)

    @property
    def resolved_component_count(self):
        return self.selected_component_count if self.has_resolved_model else 0


@dataclass
class PolarizedFitResult:
    intensity_lifetime_ns: float
    rotational_correlation_ns: float
    initial_anisotropy: float
    amplitude: float
    parallel_model: np.ndarray
    perpendicular_model: np.ndarray
    parallel_residual: np.ndarray
    perpendicular_residual: np.ndarray
    poisson_deviance: float
    success: bool
    message: str
    parameters_at_bounds: tuple = ()
    common_irf_shift_bins: float = 0.0
    parallel_background: float = 0.0
    perpendicular_background: float = 0.0


def apply_translation(data, shift_yx):
    data = np.asarray(data, dtype=float)
    if data.ndim < 2:
        raise ValueError('Spatial data must have at least two dimensions')
    if len(shift_yx) != 2 or not np.all(np.isfinite(shift_yx)):
        raise ValueError('shift_yx must contain two finite values')
    shift_vector = tuple(float(value) for value in shift_yx)
    shift_vector += (0.0,) * (data.ndim - 2)
    return nd_shift(data, shift_vector, order=1, mode='constant',
                    cval=0.0, prefilter=False)


def estimate_translation(reference, moving, max_shift=3.0):
    reference = np.asarray(reference, dtype=float)
    moving = np.asarray(moving, dtype=float)
    if reference.ndim != 2 or moving.ndim != 2:
        raise ValueError('Registration images must be two-dimensional')
    if reference.shape != moving.shape:
        raise ValueError('Registration images must have the same shape')
    if not np.isfinite(max_shift) or max_shift <= 0:
        raise ValueError('max_shift must be positive and finite')
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(moving)):
        raise ValueError('Registration images must contain only finite values')

    reference = np.log1p(np.maximum(reference, 0.0))
    moving = np.log1p(np.maximum(moving, 0.0))
    if np.std(reference) <= np.finfo(float).eps or np.std(moving) <= np.finfo(float).eps:
        raise ValueError('Registration images must contain spatial variation')
    margin = min(12, max(0, min(reference.shape) // 8))

    def correlation(first, second):
        if margin:
            first = first[margin:-margin, margin:-margin]
            second = second[margin:-margin, margin:-margin]
        first = first.ravel()
        second = second.ravel()
        if np.std(first) == 0 or np.std(second) == 0:
            return -1.0
        return float(np.corrcoef(first, second)[0, 1])

    def objective(offset):
        shifted = nd_shift(moving, offset, order=1, mode='nearest',
                           prefilter=False)
        return -correlation(reference, shifted)

    fitted = minimize(
        objective, np.zeros(2), method='Powell',
        bounds=[(-max_shift, max_shift), (-max_shift, max_shift)],
        options={'xtol': 1e-4, 'ftol': 1e-10, 'maxiter': 100})
    if not fitted.success:
        raise RuntimeError(f'Image registration failed: {fitted.message}')
    boundary_tolerance = max(1e-3, max_shift * 1e-3)
    if np.any(np.abs(fitted.x) >= max_shift - boundary_tolerance):
        raise RuntimeError(
            'Image registration reached the search boundary; increase max_shift '
            'or disable automatic registration')
    if not np.isfinite(fitted.fun) or -float(fitted.fun) < 0.1:
        raise RuntimeError('Image registration confidence is too low')
    return tuple(float(value) for value in fitted.x)


def spatial_window_sum(data, window_size=1, stride=1):
    data = np.asarray(data, dtype=float)
    if data.ndim < 2:
        raise ValueError('Spatial data must have at least two dimensions')
    if not isinstance(window_size, (int, np.integer)) or window_size < 1:
        raise ValueError('window_size must be a positive integer')
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError('stride must be a positive integer')
    if window_size > data.shape[0] or window_size > data.shape[1]:
        raise ValueError('window_size cannot exceed the image dimensions')

    cumulative = data.cumsum(axis=0).cumsum(axis=1)
    pad = [(1, 0), (1, 0)] + [(0, 0)] * (data.ndim - 2)
    cumulative = np.pad(cumulative, pad, mode='constant')
    windowed = (cumulative[window_size:, window_size:]
                - cumulative[:-window_size, window_size:]
                - cumulative[window_size:, :-window_size]
                + cumulative[:-window_size, :-window_size])
    return windowed[::stride, ::stride]


def _select_time_bins(data, selector, name):
    if np.isscalar(selector):
        raise ValueError(f'{name} must preserve the TCSPC axis')
    try:
        selected = data[..., selector]
    except (IndexError, TypeError) as exc:
        raise ValueError(f'{name} is not a valid time-bin selector') from exc
    if selected.ndim != data.ndim or selected.shape[-1] == 0:
        raise ValueError(f'{name} must select at least one time bin')
    return selected


def subtract_background(data, background_bins):
    data = np.asarray(data, dtype=float)
    selected = _select_time_bins(data, background_bins, 'background_bins')
    background = np.median(selected, axis=-1)
    corrected = data - background[..., None]
    return corrected, background


def _shift_irf_bins(irf, shift_bins, periodic):
    if shift_bins == 0.0:
        return irf
    bins = np.arange(irf.size, dtype=float)
    source = bins - shift_bins
    if not periodic:
        return np.interp(source, bins, irf, left=0.0, right=0.0)
    source = np.mod(source, irf.size)
    lower = np.floor(source).astype(int)
    fraction = source - lower
    upper = (lower + 1) % irf.size
    return (1.0 - fraction) * irf[lower] + fraction * irf[upper]


def polarized_decay_models(time_ns, parallel_irf, perpendicular_irf,
                             intensity_lifetime_ns,
                             rotational_correlation_ns,
                             initial_anisotropy, amplitude,
                             g_factor=1.0, parallel_exposure=1.0,
                             perpendicular_exposure=1.0,
                             parallel_background=0.0,
                             perpendicular_background=0.0,
                             repetition_period_ns=None,
                             common_irf_shift_bins=0.0):
    time_ns = np.asarray(time_ns, dtype=float)
    parallel_irf = np.asarray(parallel_irf, dtype=float)
    perpendicular_irf = np.asarray(perpendicular_irf, dtype=float)
    if time_ns.ndim != 1 or time_ns.size == 0:
        raise ValueError('time_ns must be a non-empty one-dimensional array')
    if parallel_irf.shape != time_ns.shape or perpendicular_irf.shape != time_ns.shape:
        raise ValueError('Each polarization IRF must match time_ns')
    positive = (intensity_lifetime_ns, rotational_correlation_ns,
                amplitude, g_factor, parallel_exposure,
                perpendicular_exposure)
    if not np.all(np.isfinite(positive)) or np.any(np.asarray(positive) <= 0):
        raise ValueError('Lifetimes, amplitude, G, and exposures must be positive and finite')
    if not np.isfinite(initial_anisotropy):
        raise ValueError('initial_anisotropy must be finite')
    backgrounds = (parallel_background, perpendicular_background)
    if not np.all(np.isfinite(backgrounds)) or np.any(np.asarray(backgrounds) < 0):
        raise ValueError('Backgrounds must be finite and non-negative')
    if (repetition_period_ns is not None
            and (not np.isfinite(repetition_period_ns)
                 or repetition_period_ns <= 0)):
        raise ValueError('repetition_period_ns must be positive and finite')
    if repetition_period_ns is not None:
        histogram_duration_ns = time_ns.size * float(np.diff(time_ns)[0])
        if not np.isclose(
                repetition_period_ns, histogram_duration_ns,
                rtol=1e-6, atol=float(np.diff(time_ns)[0]) / 2.0):
            raise ValueError(
                'repetition_period_ns must match the TCSPC histogram duration')
    if not np.isfinite(common_irf_shift_bins):
        raise ValueError('common_irf_shift_bins must be finite')

    def normalize_irf(irf):
        if np.any(~np.isfinite(irf)) or np.any(irf < 0) or irf.sum() <= 0:
            raise ValueError('IRFs must be finite, non-negative, and non-zero')
        return irf / irf.sum()

    elapsed_ns = time_ns - time_ns[0]
    intensity = amplitude * np.exp(-elapsed_ns / intensity_lifetime_ns)
    effective_ns = 1.0 / (1.0 / intensity_lifetime_ns
                          + 1.0 / rotational_correlation_ns)
    polarized = amplitude * initial_anisotropy * np.exp(
        -elapsed_ns / effective_ns)
    if repetition_period_ns is not None:
        intensity /= -np.expm1(-repetition_period_ns / intensity_lifetime_ns)
        polarized /= -np.expm1(-repetition_period_ns / effective_ns)
    parallel_impulse = parallel_exposure * (intensity + 2.0 * polarized) / 3.0
    perpendicular_impulse = (perpendicular_exposure * (intensity - polarized)
                             / (3.0 * g_factor))

    def convolve(signal, irf):
        irf = normalize_irf(irf)
        if common_irf_shift_bins != 0.0:
            irf = _shift_irf_bins(
                irf, common_irf_shift_bins,
                periodic=repetition_period_ns is not None)
            irf = normalize_irf(irf)
        if repetition_period_ns is None:
            return np.convolve(signal, irf, mode='full')[:time_ns.size]
        return np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(irf)))

    parallel_model = convolve(parallel_impulse, parallel_irf)
    perpendicular_model = convolve(perpendicular_impulse, perpendicular_irf)
    return (parallel_model + parallel_background,
            perpendicular_model + perpendicular_background)


def _validate_time_axis(time_ns, minimum_size=2):
    time_ns = np.asarray(time_ns, dtype=float)
    if time_ns.ndim != 1 or time_ns.size < minimum_size:
        raise ValueError(
            f'time_ns must be one-dimensional with at least {minimum_size} values')
    if np.any(~np.isfinite(time_ns)):
        raise ValueError('time_ns must be finite')
    spacing = np.diff(time_ns)
    if np.any(spacing <= 0):
        raise ValueError('time_ns must be strictly increasing')
    if not np.allclose(
            spacing, spacing[0], rtol=1e-6,
            atol=max(np.finfo(float).eps, abs(float(spacing[0])) * 1e-9)):
        raise ValueError('time_ns must be evenly spaced')
    return time_ns, spacing


def multicomponent_polarized_decay_models(
        time_ns, parallel_irf, perpendicular_irf, intensity_lifetime_ns,
        rotational_correlation_times_ns, component_weights,
        initial_anisotropy, amplitude, g_factor=1.0,
        parallel_exposure=1.0, perpendicular_exposure=1.0,
        parallel_background=0.0, perpendicular_background=0.0,
        repetition_period_ns=None, common_irf_shift_bins=0.0):
    time_ns, spacing = _validate_time_axis(time_ns)
    parallel_irf = np.asarray(parallel_irf, dtype=float)
    perpendicular_irf = np.asarray(perpendicular_irf, dtype=float)
    correlation_times = np.asarray(
        rotational_correlation_times_ns, dtype=float)
    weights = np.asarray(component_weights, dtype=float)
    if (parallel_irf.shape != time_ns.shape
            or perpendicular_irf.shape != time_ns.shape):
        raise ValueError('Each polarization IRF must match time_ns')
    if correlation_times.ndim != 1 or weights.ndim != 1:
        raise ValueError('Component times and weights must be one-dimensional')
    if correlation_times.size != weights.size:
        raise ValueError('Component times and weights must have the same length')
    if correlation_times.size < 1 or correlation_times.size > 3:
        raise ValueError('Choose between one and three anisotropy components')
    if (np.any(~np.isfinite(correlation_times))
            or np.any(correlation_times <= 0)):
        raise ValueError('Component times must be positive and finite')
    if np.any(np.diff(correlation_times) <= 0):
        raise ValueError('Component times must be strictly increasing')
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('Component weights must be finite and non-negative')
    if not np.isclose(weights.sum(), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError('Component weights must sum to one')
    positive = (intensity_lifetime_ns, amplitude, g_factor,
                parallel_exposure, perpendicular_exposure)
    if not np.all(np.isfinite(positive)) or np.any(np.asarray(positive) <= 0):
        raise ValueError('Lifetimes, amplitude, G, and exposures must be positive and finite')
    if not np.isfinite(initial_anisotropy):
        raise ValueError('initial_anisotropy must be finite')
    backgrounds = (parallel_background, perpendicular_background)
    if not np.all(np.isfinite(backgrounds)) or np.any(np.asarray(backgrounds) < 0):
        raise ValueError('Backgrounds must be finite and non-negative')
    if (repetition_period_ns is not None
            and (not np.isfinite(repetition_period_ns)
                 or repetition_period_ns <= 0)):
        raise ValueError('repetition_period_ns must be positive and finite')
    if repetition_period_ns is not None:
        histogram_duration_ns = time_ns.size * float(spacing[0])
        if not np.isclose(
                repetition_period_ns, histogram_duration_ns,
                rtol=1e-6, atol=float(spacing[0]) / 2.0):
            raise ValueError(
                'repetition_period_ns must match the TCSPC histogram duration')
    if not np.isfinite(common_irf_shift_bins):
        raise ValueError('common_irf_shift_bins must be finite')

    def normalize_irf(irf):
        if np.any(~np.isfinite(irf)) or np.any(irf < 0) or irf.sum() <= 0:
            raise ValueError('IRFs must be finite, non-negative, and non-zero')
        return irf / irf.sum()

    elapsed_ns = time_ns - time_ns[0]
    intensity = amplitude * np.exp(-elapsed_ns / intensity_lifetime_ns)
    polarized = np.zeros_like(elapsed_ns)
    for correlation_time, weight in zip(correlation_times, weights):
        effective_ns = 1.0 / (
            1.0 / intensity_lifetime_ns + 1.0 / correlation_time)
        component = np.exp(-elapsed_ns / effective_ns)
        if repetition_period_ns is not None:
            component /= -np.expm1(-repetition_period_ns / effective_ns)
        polarized += weight * component
    polarized *= amplitude * initial_anisotropy
    if repetition_period_ns is not None:
        intensity /= -np.expm1(
            -repetition_period_ns / intensity_lifetime_ns)
    parallel_impulse = parallel_exposure * (
        intensity + 2.0 * polarized) / 3.0
    perpendicular_impulse = perpendicular_exposure * (
        intensity - polarized) / (3.0 * g_factor)

    def convolve(signal, irf):
        irf = normalize_irf(irf)
        if common_irf_shift_bins != 0.0:
            irf = _shift_irf_bins(
                irf, common_irf_shift_bins,
                periodic=repetition_period_ns is not None)
            irf = normalize_irf(irf)
        if repetition_period_ns is None:
            return np.convolve(signal, irf, mode='full')[:time_ns.size]
        return np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(irf)))

    return (
        convolve(parallel_impulse, parallel_irf) + parallel_background,
        convolve(perpendicular_impulse, perpendicular_irf)
        + perpendicular_background)


def fit_polarized_decays(parallel, perpendicular, time_ns,
                          parallel_irf, perpendicular_irf,
                          intensity_lifetime_ns,
                          g_factor=1.0, parallel_exposure=1.0,
                          perpendicular_exposure=1.0,
                          initial_parallel_background=0.0,
                          initial_perpendicular_background=0.0,
                          repetition_period_ns=None, fit_bins=None,
                          initial_rotational_ns=1.0,
                          initial_anisotropy=0.2,
                          rotational_bounds_ns=None,
                          anisotropy_bounds=(-0.2, 0.4),
                          common_shift_bounds_bins=(-2.0, 2.0)):
    parallel = np.asarray(parallel, dtype=float)
    perpendicular = np.asarray(perpendicular, dtype=float)
    time_ns = np.asarray(time_ns, dtype=float)
    if parallel.shape != time_ns.shape or perpendicular.shape != time_ns.shape:
        raise ValueError('Polarized decays must match time_ns')
    if (np.any(~np.isfinite(parallel)) or np.any(parallel < 0)
            or np.any(~np.isfinite(perpendicular))
            or np.any(perpendicular < 0)):
        raise ValueError('Polarized decays must be finite and non-negative')
    if time_ns.size < 3 or np.any(~np.isfinite(time_ns)):
        raise ValueError('time_ns must contain at least three finite values')
    spacing = np.diff(time_ns)
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0]):
        raise ValueError('time_ns must be increasing and evenly spaced')
    if not np.isfinite(intensity_lifetime_ns) or intensity_lifetime_ns <= 0:
        raise ValueError('intensity_lifetime_ns must be positive and finite')
    for background in (initial_parallel_background,
                       initial_perpendicular_background):
        if not np.isfinite(background) or background < 0:
            raise ValueError('Initial backgrounds must be finite and non-negative')
    if fit_bins is None:
        fit_bins = slice(None)
    selected_parallel = _select_time_bins(parallel, fit_bins, 'fit_bins')
    selected_perpendicular = _select_time_bins(
        perpendicular, fit_bins, 'fit_bins')

    bin_width_ns = float(spacing[0])
    duration_ns = (time_ns[-1] - time_ns[0]) + bin_width_ns
    if rotational_bounds_ns is None:
        rotational_upper = min(
            repetition_period_ns / 2.0
            if repetition_period_ns is not None else duration_ns / 2.0,
            10.0 * intensity_lifetime_ns)
        rotational_bounds_ns = (2.0 * bin_width_ns, rotational_upper)
    for lower, upper in (rotational_bounds_ns, anisotropy_bounds,
                         common_shift_bounds_bins):
        if (not np.isfinite(lower) or not np.isfinite(upper)
                or lower >= upper):
            raise ValueError('Fit bounds must be finite and increasing')
    if not rotational_bounds_ns[0] < initial_rotational_ns < rotational_bounds_ns[1]:
        raise ValueError('initial_rotational_ns must lie inside its bounds')
    if not anisotropy_bounds[0] < initial_anisotropy < anisotropy_bounds[1]:
        raise ValueError('initial_anisotropy must lie inside its bounds')
    if not common_shift_bounds_bins[0] < 0.0 < common_shift_bounds_bins[1]:
        raise ValueError('common_shift_bounds_bins must contain zero')

    corrected_total = (
        (parallel - initial_parallel_background) / parallel_exposure
        + 2.0 * g_factor
        * (perpendicular - initial_perpendicular_background)
        / perpendicular_exposure)
    initial_amplitude = max(float(np.max(corrected_total)), 1.0)

    def unpack(params):
        return (float(np.exp(params[0])), float(params[1]),
                float(np.exp(params[2])), float(params[3]),
                float(params[4]), float(params[5]))

    def models(params):
        rotational_ns, r0, amplitude, shift_bins, parallel_bg, perpendicular_bg = (
            unpack(params))
        return polarized_decay_models(
            time_ns, parallel_irf, perpendicular_irf,
            intensity_lifetime_ns=intensity_lifetime_ns,
            rotational_correlation_ns=rotational_ns,
            initial_anisotropy=r0, amplitude=amplitude,
            g_factor=g_factor, parallel_exposure=parallel_exposure,
            perpendicular_exposure=perpendicular_exposure,
            parallel_background=parallel_bg,
            perpendicular_background=perpendicular_bg,
            repetition_period_ns=repetition_period_ns,
            common_irf_shift_bins=shift_bins)

    def poisson_residual(observed, expected):
        expected = np.maximum(expected, 1e-12)
        contribution = expected - observed
        positive = observed > 0
        contribution[positive] += observed[positive] * np.log(
            observed[positive] / expected[positive])
        contribution = np.maximum(2.0 * contribution, 0.0)
        return np.sign(expected - observed) * np.sqrt(contribution)

    def residuals(params):
        parallel_model, perpendicular_model = models(params)
        return np.concatenate([
            poisson_residual(
                selected_parallel,
                _select_time_bins(parallel_model, fit_bins, 'fit_bins')),
            poisson_residual(
                selected_perpendicular,
                _select_time_bins(perpendicular_model, fit_bins, 'fit_bins')),
        ])

    initial = np.array([
        np.log(initial_rotational_ns), initial_anisotropy,
        np.log(initial_amplitude), 0.0,
        initial_parallel_background,
        initial_perpendicular_background])
    max_background = max(float(np.max(parallel)),
                         float(np.max(perpendicular)), 1.0) * 10.0
    lower = np.array([
        np.log(rotational_bounds_ns[0]), anisotropy_bounds[0],
        np.log(1e-12), common_shift_bounds_bins[0],
        0.0, 0.0])
    upper = np.array([
        np.log(rotational_bounds_ns[1]), anisotropy_bounds[1],
        np.log(max(initial_amplitude * 1e6, 1e6)),
        common_shift_bounds_bins[1],
        max_background, max_background])
    fitted = least_squares(
        residuals, initial, bounds=(lower, upper), method='trf',
        max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    rotational_ns, r0, amplitude, shift_bins, parallel_bg, perpendicular_bg = (
        unpack(fitted.x))
    parallel_model, perpendicular_model = models(fitted.x)
    final_residuals = residuals(fitted.x)
    parameter_names = (
        'rotational_correlation_ns', 'initial_anisotropy', 'amplitude',
        'common_irf_shift_bins', 'parallel_background',
        'perpendicular_background')
    parameters_at_bounds = tuple(
        name for name, active in zip(parameter_names, fitted.active_mask)
        if active != 0)
    return PolarizedFitResult(
        intensity_lifetime_ns=float(intensity_lifetime_ns),
        rotational_correlation_ns=rotational_ns,
        initial_anisotropy=r0,
        amplitude=amplitude,
        parallel_model=parallel_model,
        perpendicular_model=perpendicular_model,
        parallel_residual=parallel - parallel_model,
        perpendicular_residual=perpendicular - perpendicular_model,
        poisson_deviance=float(np.sum(final_residuals ** 2)),
        success=bool(fitted.success),
        message=str(fitted.message),
        parameters_at_bounds=parameters_at_bounds,
        common_irf_shift_bins=shift_bins,
        parallel_background=parallel_bg,
        perpendicular_background=perpendicular_bg)


def _softmax(values):
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    exponential = np.exp(shifted)
    return exponential / exponential.sum()


def _automatic_component_times(parameters, count, bounds_ns):
    gaps = _softmax(np.r_[parameters[:count], 0.0])
    fractions = np.cumsum(gaps[:-1])
    lower, upper = bounds_ns
    return lower + (upper - lower) * fractions


def _manual_component_times(parameters, component_bounds_ns):
    fractions = 1.0 / (1.0 + np.exp(-parameters))
    return (component_bounds_ns[:, 0]
            + fractions * (component_bounds_ns[:, 1]
                           - component_bounds_ns[:, 0]))


def _irf_fwhm_ns(irf, bin_width_ns):
    irf = np.asarray(irf, dtype=float)
    above_half = np.flatnonzero(irf >= 0.5 * np.max(irf))
    if above_half.size == 0:
        return float(bin_width_ns)
    return float((above_half[-1] - above_half[0] + 1) * bin_width_ns)


def fit_multicomponent_polarized_decays(
        parallel, perpendicular, time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns, g_factor=1.0, parallel_exposure=1.0,
        perpendicular_exposure=1.0, initial_parallel_background=0.0,
        initial_perpendicular_background=0.0, repetition_period_ns=None,
        fit_bins=None, max_components=3, multistart=6,
        rotational_bounds_ns=None, component_bounds_ns=None,
        initial_anisotropy=0.2, anisotropy_bounds=(-0.2, 0.4),
        common_shift_bounds_bins=(-2.0, 2.0)):
    if (not isinstance(max_components, (int, np.integer))
            or not 1 <= max_components <= 3):
        raise ValueError('max_components must be between one and three')
    if (not isinstance(multistart, (int, np.integer))
            or not 1 <= multistart <= 32):
        raise ValueError('multistart must be between 1 and at most 32')
    parallel = np.asarray(parallel, dtype=float)
    perpendicular = np.asarray(perpendicular, dtype=float)
    time_ns, spacing = _validate_time_axis(time_ns, minimum_size=3)
    parallel_irf = np.asarray(parallel_irf, dtype=float)
    perpendicular_irf = np.asarray(perpendicular_irf, dtype=float)
    if parallel.shape != time_ns.shape or perpendicular.shape != time_ns.shape:
        raise ValueError('Polarized decays must match time_ns')
    if parallel_irf.shape != time_ns.shape or perpendicular_irf.shape != time_ns.shape:
        raise ValueError('Each polarization IRF must match time_ns')
    if (np.any(~np.isfinite(parallel)) or np.any(parallel < 0)
            or np.any(~np.isfinite(perpendicular))
            or np.any(perpendicular < 0)):
        raise ValueError('Polarized counts must be finite and non-negative')
    positive = np.asarray([
        intensity_lifetime_ns, g_factor, parallel_exposure,
        perpendicular_exposure], dtype=float)
    if np.any(~np.isfinite(positive)) or np.any(positive <= 0):
        raise ValueError('Lifetime, G, and exposures must be positive and finite')
    initial_backgrounds = np.asarray([
        initial_parallel_background, initial_perpendicular_background],
        dtype=float)
    if np.any(~np.isfinite(initial_backgrounds)) or np.any(initial_backgrounds < 0):
        raise ValueError('Initial backgrounds must be finite and non-negative')
    for lower, upper in (anisotropy_bounds, common_shift_bounds_bins):
        if (not np.isfinite(lower) or not np.isfinite(upper)
                or lower >= upper):
            raise ValueError('Fit bounds must be finite and increasing')
    if not anisotropy_bounds[0] < initial_anisotropy < anisotropy_bounds[1]:
        raise ValueError('initial_anisotropy must lie inside its bounds')
    if not common_shift_bounds_bins[0] < 0.0 < common_shift_bounds_bins[1]:
        raise ValueError('common_shift_bounds_bins must contain zero')
    if fit_bins is None:
        fit_bins = slice(None)
    selected_parallel = _select_time_bins(parallel, fit_bins, 'fit_bins')
    selected_perpendicular = _select_time_bins(
        perpendicular, fit_bins, 'fit_bins')

    bin_width_ns = float(spacing[0])
    duration_ns = (time_ns[-1] - time_ns[0]) + bin_width_ns
    if rotational_bounds_ns is None:
        rotational_upper = min(
            repetition_period_ns / 2.0
            if repetition_period_ns is not None else duration_ns / 2.0,
            10.0 * intensity_lifetime_ns)
        rotational_bounds_ns = (2.0 * bin_width_ns, rotational_upper)
    if (len(rotational_bounds_ns) != 2
            or not np.all(np.isfinite(rotational_bounds_ns))
            or rotational_bounds_ns[0] <= 0
            or rotational_bounds_ns[0] >= rotational_bounds_ns[1]):
        raise ValueError(
            'rotational_bounds_ns must contain positive increasing bounds')
    rotational_bounds_ns = tuple(float(value) for value in rotational_bounds_ns)

    fit_bounds_ns = rotational_bounds_ns
    if component_bounds_ns is not None:
        shared_bounds = np.asarray(component_bounds_ns, dtype=float)
        if (shared_bounds.shape != (2,)
                or np.any(~np.isfinite(shared_bounds))
                or np.any(shared_bounds <= 0)
                or shared_bounds[0] >= shared_bounds[1]):
            raise ValueError(
                'component_bounds_ns must contain one positive increasing '
                'lower/upper range shared by every candidate')
        fit_bounds_ns = tuple(float(value) for value in shared_bounds)

    corrected_total = (
        (parallel - initial_parallel_background) / parallel_exposure
        + 2.0 * g_factor
        * (perpendicular - initial_perpendicular_background)
        / perpendicular_exposure)
    initial_amplitude = max(float(np.max(corrected_total)), 1.0)
    max_background = max(
        float(np.max(parallel)), float(np.max(perpendicular)), 1.0) * 10.0
    rng = np.random.default_rng(7300 + max_components)
    observations = selected_parallel.size + selected_perpendicular.size
    irf_resolution_ns = max(
        _irf_fwhm_ns(parallel_irf, bin_width_ns),
        _irf_fwhm_ns(perpendicular_irf, bin_width_ns))
    candidates = []

    def poisson_residual(observed, expected):
        expected = np.maximum(expected, 1e-12)
        contribution = expected - observed
        positive_counts = observed > 0
        contribution[positive_counts] += observed[positive_counts] * np.log(
            observed[positive_counts] / expected[positive_counts])
        contribution = np.maximum(2.0 * contribution, 0.0)
        return np.sign(expected - observed) * np.sqrt(contribution)

    for component_count in range(1, max_components + 1):
        candidate_bounds = np.tile(
            np.asarray(fit_bounds_ns), (component_count, 1))
        time_parameter_count = component_count
        weight_parameter_count = max(0, component_count - 1)
        shape_parameter_count = time_parameter_count + weight_parameter_count
        nuisance_start = shape_parameter_count

        def unpack(
                parameters, component_count=component_count,
                fit_bounds_ns=fit_bounds_ns,
                shape_parameter_count=shape_parameter_count,
                nuisance_start=nuisance_start):
            correlation_times = _automatic_component_times(
                parameters, component_count, fit_bounds_ns)
            if component_count == 1:
                weights = np.ones(1, dtype=float)
            else:
                weights = _softmax(np.r_[
                    parameters[component_count:shape_parameter_count], 0.0])
            r0, log_amplitude, shift_bins, parallel_bg, perpendicular_bg = (
                parameters[nuisance_start:nuisance_start + 5])
            return (
                correlation_times, weights, float(r0),
                float(np.exp(log_amplitude)), float(shift_bins),
                float(parallel_bg), float(perpendicular_bg))

        def models(parameters):
            times, weights, r0, amplitude, shift_bins, parallel_bg, perpendicular_bg = (
                unpack(parameters))
            return multicomponent_polarized_decay_models(
                time_ns, parallel_irf, perpendicular_irf,
                intensity_lifetime_ns=intensity_lifetime_ns,
                rotational_correlation_times_ns=times,
                component_weights=weights, initial_anisotropy=r0,
                amplitude=amplitude, g_factor=g_factor,
                parallel_exposure=parallel_exposure,
                perpendicular_exposure=perpendicular_exposure,
                parallel_background=parallel_bg,
                perpendicular_background=perpendicular_bg,
                repetition_period_ns=repetition_period_ns,
                common_irf_shift_bins=shift_bins)

        def residuals(parameters):
            parallel_model, perpendicular_model = models(parameters)
            return np.concatenate([
                poisson_residual(
                    selected_parallel,
                    _select_time_bins(parallel_model, fit_bins, 'fit_bins')),
                poisson_residual(
                    selected_perpendicular,
                    _select_time_bins(
                        perpendicular_model, fit_bins, 'fit_bins')),
            ])

        lower = np.r_[
            np.full(shape_parameter_count, -8.0), anisotropy_bounds[0],
            np.log(1e-12), common_shift_bounds_bins[0], 0.0, 0.0]
        upper = np.r_[
            np.full(shape_parameter_count, 8.0), anisotropy_bounds[1],
            np.log(max(initial_amplitude * 1e6, 1e6)),
            common_shift_bounds_bins[1], max_background, max_background]
        parameter_names = (
            tuple(f'component_time_parameter_{index + 1}'
                  for index in range(component_count))
            + tuple(f'component_weight_parameter_{index + 1}'
                    for index in range(weight_parameter_count))
            + ('initial_anisotropy', 'log_amplitude',
               'common_irf_shift_bins',
               'parallel_background', 'perpendicular_background'))
        solutions = []
        start_records = []
        for start_index in range(multistart):
            time_start = np.linspace(-1.5, 1.5, component_count)
            weight_start = np.zeros(weight_parameter_count)
            if start_index:
                time_start += rng.normal(0.0, 1.2, component_count)
                weight_start += rng.normal(0.0, 0.8, weight_parameter_count)
            initial = np.r_[
                time_start, weight_start, initial_anisotropy,
                np.log(initial_amplitude), 0.0, initial_backgrounds]
            initial = np.minimum(
                np.maximum(initial, lower + 1e-9), upper - 1e-9)
            try:
                fitted_start = least_squares(
                    residuals, initial, bounds=(lower, upper), method='trf',
                    max_nfev=5000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
                start_deviance = float(np.sum(residuals(fitted_start.x) ** 2))
                start_times, start_weights, *_ = unpack(fitted_start.x)
                start_bound_hits = tuple(
                    name for name, active in zip(
                        parameter_names, fitted_start.active_mask)
                    if active != 0)
                start_records.append(MultistartFitRecord(
                    start_index=int(start_index),
                    parameter_vector=np.asarray(fitted_start.x, dtype=float).copy(),
                    rotational_correlation_times_ns=np.asarray(
                        start_times, dtype=float).copy(),
                    component_weights=np.asarray(start_weights, dtype=float).copy(),
                    poisson_deviance=start_deviance,
                    success=bool(fitted_start.success),
                    message=str(fitted_start.message),
                    parameters_at_bounds=start_bound_hits))
                if fitted_start.success:
                    solutions.append((start_deviance, fitted_start))
            except (ValueError, FloatingPointError) as exc:
                start_times, start_weights, *_ = unpack(initial)
                start_records.append(MultistartFitRecord(
                    start_index=int(start_index),
                    parameter_vector=np.asarray(initial, dtype=float).copy(),
                    rotational_correlation_times_ns=np.asarray(
                        start_times, dtype=float).copy(),
                    component_weights=np.asarray(start_weights, dtype=float).copy(),
                    poisson_deviance=float('inf'), success=False,
                    message=f'{type(exc).__name__}: {exc}',
                    parameters_at_bounds=()))
        if not solutions:
            parameter_count = 2 * component_count + 4
            degrees_of_freedom = observations - parameter_count
            failed_warnings = ['no optimizer start converged']
            if component_count == 3:
                failed_warnings.append(
                    'three-component parameter robustness is not validated')
            failed_warnings.append(
                'physical resolution not established: uncertainty validation '
                'unavailable')
            candidates.append(MulticomponentModelFit(
                component_count=component_count,
                rotational_correlation_times_ns=np.full(
                    component_count, np.nan),
                component_weights=np.full(component_count, np.nan),
                intensity_lifetime_ns=float(intensity_lifetime_ns),
                initial_anisotropy=float('nan'), amplitude=float('nan'),
                parallel_model=np.full(time_ns.shape, np.nan),
                perpendicular_model=np.full(time_ns.shape, np.nan),
                parallel_residual=np.full(time_ns.shape, np.nan),
                perpendicular_residual=np.full(time_ns.shape, np.nan),
                poisson_deviance=float('inf'),
                degrees_of_freedom=int(degrees_of_freedom),
                deviance_per_degree_of_freedom=float('inf'),
                bic=float('inf'), aicc=float('inf'), success=False,
                message='No optimizer start converged',
                parameters_at_bounds=(), identifiable=False,
                identifiability_warnings=tuple(failed_warnings),
                common_irf_shift_bins=float('nan'),
                parallel_background=float('nan'),
                perpendicular_background=float('nan'),
                component_bounds_ns=np.asarray(candidate_bounds, dtype=float),
                multistart_count=int(multistart),
                multistart_records=tuple(start_records),
                eligible_for_selection=False,
                parameter_names=parameter_names))
            continue
        solutions.sort(key=lambda item: item[0])
        deviance, fitted = solutions[0]
        times, weights, r0, amplitude, shift_bins, parallel_bg, perpendicular_bg = (
            unpack(fitted.x))
        parallel_model, perpendicular_model = models(fitted.x)
        parameter_count = 2 * component_count + 4
        degrees_of_freedom = observations - parameter_count
        deviance_per_degree_of_freedom = (
            deviance / degrees_of_freedom
            if degrees_of_freedom > 0 else np.inf)
        bic = deviance + parameter_count * np.log(observations)
        if observations <= parameter_count + 1:
            aicc = np.inf
        else:
            aicc = (deviance + 2.0 * parameter_count
                    + 2.0 * parameter_count * (parameter_count + 1)
                    / (observations - parameter_count - 1))
        parameters_at_bounds = tuple(
            name for name, active in zip(parameter_names, fitted.active_mask)
            if active != 0)
        warnings = []
        if not fitted.success:
            warnings.append('optimizer did not converge')
        if parameters_at_bounds:
            warnings.append(
                'fit parameter reached a bound: '
                + ', '.join(parameters_at_bounds))
        if deviance_per_degree_of_freedom > 2.0:
            warnings.append(
                'Poisson deviance per degree of freedom exceeds 2')
        if component_count == 3:
            warnings.append(
                'three-component parameter robustness is not validated')
        warnings.append(
            'physical resolution not established: uncertainty validation '
            'unavailable')
        if np.any(weights < 0.05):
            warnings.append('component weight below 0.05')
        if (component_count > 1
                and np.any(times[1:] / times[:-1] < 1.5)):
            warnings.append('component times are not separated by 1.5x')
        if np.any(np.abs(fitted.x[:component_count]) >= 7.5):
            warnings.append('component time reached a bound')
        if times[0] < irf_resolution_ns:
            warnings.append('fastest component is below the IRF resolution')
        if times[-1] >= 0.95 * fit_bounds_ns[1]:
            warnings.append('slowest component reached the observation bound')
        competitive = [
            unpack(solution.x) for other_deviance, solution in solutions
            if other_deviance <= deviance + 2.0]
        if len(competitive) > 1:
            time_sets = np.asarray([solution[0] for solution in competitive])
            weight_sets = np.asarray([solution[1] for solution in competitive])
            relative_time_spread = np.max(
                np.ptp(time_sets, axis=0) / np.maximum(times, 1e-12))
            weight_spread = np.max(np.ptp(weight_sets, axis=0))
            if relative_time_spread > 0.2 or weight_spread > 0.1:
                warnings.append('competitive multistart solutions disagree')
        candidates.append(MulticomponentModelFit(
            component_count=component_count,
            rotational_correlation_times_ns=np.asarray(times, dtype=float),
            component_weights=np.asarray(weights, dtype=float),
            intensity_lifetime_ns=float(intensity_lifetime_ns),
            initial_anisotropy=r0, amplitude=amplitude,
            parallel_model=parallel_model,
            perpendicular_model=perpendicular_model,
            parallel_residual=poisson_residual(parallel, parallel_model),
            perpendicular_residual=poisson_residual(
                perpendicular, perpendicular_model),
            poisson_deviance=deviance,
            degrees_of_freedom=int(degrees_of_freedom),
            deviance_per_degree_of_freedom=float(
                deviance_per_degree_of_freedom),
            bic=float(bic), aicc=float(aicc),
            success=bool(fitted.success), message=str(fitted.message),
            parameters_at_bounds=parameters_at_bounds,
            identifiable=not warnings,
            identifiability_warnings=tuple(warnings),
            common_irf_shift_bins=shift_bins,
            parallel_background=parallel_bg,
            perpendicular_background=perpendicular_bg,
            component_bounds_ns=np.asarray(candidate_bounds, dtype=float),
            multistart_count=int(multistart),
            multistart_records=tuple(start_records),
            eligible_for_selection=True,
            parameter_names=parameter_names))

    eligible_candidates = [
        candidate for candidate in candidates
        if candidate.eligible_for_selection]
    if eligible_candidates:
        best_bic = min(candidate.bic for candidate in eligible_candidates)
        selected_component_count = next(
            candidate.component_count for candidate in eligible_candidates
            if candidate.bic <= best_bic + 2.0)
    else:
        selected_component_count = 0
    for candidate in candidates:
        if (candidate.eligible_for_selection
                and candidate.component_count > selected_component_count > 0):
            candidate.identifiability_warnings += (
                'simpler model preferred by BIC',)
            candidate.identifiable = False
    return MulticomponentFitResult(
        max_components=int(max_components),
        selected_component_count=int(selected_component_count),
        candidates=tuple(candidates))


def calculate_late_window_stability(
        parallel, perpendicular, time_ns, parallel_exposure=1.0,
        perpendicular_exposure=1.0, selected_start_ns=0.0,
        rolling_bins=None):
    parallel = np.asarray(parallel, dtype=float)
    perpendicular = np.asarray(perpendicular, dtype=float)
    time_ns = np.asarray(time_ns, dtype=float)
    if (parallel.ndim != 1 or perpendicular.ndim != 1
            or time_ns.ndim != 1 or parallel.shape != perpendicular.shape
            or parallel.shape != time_ns.shape or parallel.size < 2):
        raise ValueError(
            'Late-window decays and time axis must be matching one-dimensional arrays')
    if (np.any(~np.isfinite(parallel)) or np.any(parallel < 0)
            or np.any(~np.isfinite(perpendicular))
            or np.any(perpendicular < 0)
            or np.any(~np.isfinite(time_ns)) or np.any(np.diff(time_ns) <= 0)):
        raise ValueError(
            'Late-window counts must be finite and non-negative and time must increase')
    exposures = np.asarray(
        [parallel_exposure, perpendicular_exposure], dtype=float)
    if np.any(~np.isfinite(exposures)) or np.any(exposures <= 0):
        raise ValueError('Exposure values must be positive and finite')
    if rolling_bins is None:
        rolling_bins = min(5, parallel.size)
    if (not isinstance(rolling_bins, (int, np.integer))
            or rolling_bins < 1 or rolling_bins > parallel.size):
        raise ValueError('rolling_bins must fit inside the TCSPC time axis')
    if (not np.isfinite(selected_start_ns)
            or selected_start_ns < time_ns[0]
            or selected_start_ns > time_ns[-1]):
        raise ValueError('Late-window start must lie inside the TCSPC time axis')

    parallel_sums = np.cumsum(parallel[::-1])[::-1]
    perpendicular_sums = np.cumsum(perpendicular[::-1])[::-1]
    valid = (parallel_sums > 0) & (perpendicular_sums > 0)
    nested_scale = np.full(time_ns.shape, np.nan, dtype=float)
    nested_standard_error = np.full(time_ns.shape, np.nan, dtype=float)
    nested_scale[valid] = (
        (parallel_sums[valid] / parallel_exposure)
        / (perpendicular_sums[valid] / perpendicular_exposure))
    nested_standard_error[valid] = nested_scale[valid] * np.sqrt(
        1.0 / parallel_sums[valid] + 1.0 / perpendicular_sums[valid])
    rolling_parallel = np.convolve(
        parallel, np.ones(rolling_bins), mode='valid')
    rolling_perpendicular = np.convolve(
        perpendicular, np.ones(rolling_bins), mode='valid')
    rolling_scale = np.full(time_ns.shape, np.nan, dtype=float)
    rolling_valid = (rolling_parallel > 0) & (rolling_perpendicular > 0)
    rolling_scale[:rolling_parallel.size][rolling_valid] = (
        (rolling_parallel[rolling_valid] / parallel_exposure)
        / (rolling_perpendicular[rolling_valid] / perpendicular_exposure))
    selected_start_bin = int(np.searchsorted(time_ns, selected_start_ns))
    if not valid[selected_start_bin]:
        raise ValueError('Selected late window must contain photons in both channels')
    return LateWindowStabilityResult(
        time_ns=time_ns, nested_scale=nested_scale,
        nested_standard_error=nested_standard_error,
        rolling_scale=rolling_scale, rolling_bins=int(rolling_bins),
        selected_start_bin=selected_start_bin,
        selected_scale=float(nested_scale[selected_start_bin]),
        selected_parallel_photons=float(parallel_sums[selected_start_bin]),
        selected_perpendicular_photons=float(
            perpendicular_sums[selected_start_bin]))


def calculate_anisotropy(parallel, perpendicular, g_factor=1.0,
                          min_denominator=0.0, parallel_exposure=1.0,
                          perpendicular_exposure=1.0):
    parallel = np.asarray(parallel, dtype=float)
    perpendicular = np.asarray(perpendicular, dtype=float)
    if parallel.shape != perpendicular.shape:
        raise ValueError('Parallel and perpendicular data must have the same shape')
    if not np.isfinite(g_factor) or g_factor <= 0:
        raise ValueError('g_factor must be positive and finite')
    if (not np.isfinite(parallel_exposure) or parallel_exposure <= 0
            or not np.isfinite(perpendicular_exposure)
            or perpendicular_exposure <= 0):
        raise ValueError('Exposure values must be positive and finite')

    parallel_rate = parallel / parallel_exposure
    scaled_perpendicular = g_factor * perpendicular / perpendicular_exposure
    denominator = parallel_rate + 2.0 * scaled_perpendicular
    valid = np.isfinite(denominator) & (denominator > min_denominator)
    result = np.full(parallel.shape, np.nan, dtype=float)
    np.divide(parallel_rate - scaled_perpendicular, denominator,
              out=result, where=valid)
    return result


def analyze_anisotropy(parallel, perpendicular, time_ns, background_bins,
                       analysis_bins=None, g_factor=1.0, spatial_window=1,
                       stride=1, min_bin_photons=0.0,
                       min_map_photons=0.0,
                       perpendicular_shift=(0.0, 0.0),
                       parallel_exposure=1.0,
                       perpendicular_exposure=1.0):
    parallel = np.asarray(parallel, dtype=float)
    perpendicular = np.asarray(perpendicular, dtype=float)
    time_ns = np.asarray(time_ns, dtype=float)
    if parallel.ndim != 3 or perpendicular.ndim != 3:
        raise ValueError('Polarization data must have shape (Y, X, H)')
    if parallel.shape != perpendicular.shape:
        raise ValueError('Parallel and perpendicular stacks must have the same shape')
    if time_ns.ndim != 1 or time_ns.size != parallel.shape[-1]:
        raise ValueError('time_ns must match the TCSPC axis')
    if (not np.isfinite(min_bin_photons) or min_bin_photons < 0
            or not np.isfinite(min_map_photons) or min_map_photons < 0):
        raise ValueError('Photon thresholds must be finite and non-negative')
    if analysis_bins is None:
        analysis_bins = slice(None)

    parallel_decay_raw = parallel.sum(axis=(0, 1))
    perpendicular_decay_raw = perpendicular.sum(axis=(0, 1))
    parallel_decay, parallel_background = subtract_background(
        parallel_decay_raw, background_bins)
    perpendicular_decay, perpendicular_background = subtract_background(
        perpendicular_decay_raw, background_bins)
    anisotropy_decay = calculate_anisotropy(
        parallel_decay, perpendicular_decay, g_factor=g_factor,
        parallel_exposure=parallel_exposure,
        perpendicular_exposure=perpendicular_exposure)
    decay_observed = parallel_decay_raw + perpendicular_decay_raw
    anisotropy_decay[decay_observed < min_bin_photons] = np.nan

    parallel_pooled = spatial_window_sum(
        parallel, window_size=spatial_window, stride=stride)
    perpendicular_registered = apply_translation(
        perpendicular, perpendicular_shift)
    perpendicular_pooled = spatial_window_sum(
        perpendicular_registered, window_size=spatial_window, stride=stride)
    perpendicular_support = apply_translation(
        np.ones(perpendicular.shape[:2]), perpendicular_shift)
    support_pooled = spatial_window_sum(
        perpendicular_support[..., None], window_size=spatial_window,
        stride=stride)[..., 0]
    full_registration_support = np.isclose(
        support_pooled, spatial_window ** 2, rtol=0.0, atol=1e-6)
    parallel_corrected, _ = subtract_background(
        parallel_pooled, background_bins)
    perpendicular_corrected, _ = subtract_background(
        perpendicular_pooled, background_bins)
    anisotropy_cube = calculate_anisotropy(
        parallel_corrected, perpendicular_corrected, g_factor=g_factor,
        parallel_exposure=parallel_exposure,
        perpendicular_exposure=perpendicular_exposure)
    observed_cube = parallel_pooled + perpendicular_pooled
    bin_valid = ((observed_cube >= min_bin_photons)
                 & np.isfinite(anisotropy_cube)
                 & full_registration_support[..., None])
    anisotropy_cube[~bin_valid] = np.nan

    parallel_selected = _select_time_bins(
        parallel_corrected, analysis_bins, 'analysis_bins')
    perpendicular_selected = _select_time_bins(
        perpendicular_corrected, analysis_bins, 'analysis_bins')
    observed_selected = _select_time_bins(
        observed_cube, analysis_bins, 'analysis_bins')
    parallel_map = parallel_selected.sum(axis=-1)
    perpendicular_map = perpendicular_selected.sum(axis=-1)
    anisotropy_map = calculate_anisotropy(
        parallel_map, perpendicular_map, g_factor=g_factor,
        parallel_exposure=parallel_exposure,
        perpendicular_exposure=perpendicular_exposure)
    total_counts = observed_selected.sum(axis=-1)
    map_valid = ((total_counts >= min_map_photons)
                 & np.isfinite(anisotropy_map)
                 & full_registration_support)
    anisotropy_map[~map_valid] = np.nan
    origins_y = np.arange(anisotropy_cube.shape[0]) * stride
    origins_x = np.arange(anisotropy_cube.shape[1]) * stride
    grid_y, grid_x = np.meshgrid(origins_y, origins_x, indexing='ij')
    window_origins = np.stack([grid_y, grid_x], axis=-1)

    return AnisotropyResult(
        time_ns=time_ns,
        parallel_decay=parallel_decay,
        perpendicular_decay=perpendicular_decay,
        anisotropy_decay=anisotropy_decay,
        anisotropy_cube=anisotropy_cube,
        anisotropy_map=anisotropy_map,
        parallel_intensity=parallel_map,
        perpendicular_intensity=perpendicular_map,
        parallel_background=float(parallel_background),
        perpendicular_background=float(perpendicular_background),
        spatial_window=spatial_window,
        stride=stride,
        perpendicular_shift=tuple(float(value)
                                  for value in perpendicular_shift),
        valid_mask=bin_valid,
        map_valid_mask=map_valid,
        total_counts=total_counts,
        window_origins=window_origins,
        g_factor=float(g_factor),
        parallel_exposure=float(parallel_exposure),
        perpendicular_exposure=float(perpendicular_exposure),
    )


def analyze_ptu_pair(parallel_path, perpendicular_path, background_bins,
                     analysis_bins=None, g_factor=1.0, spatial_window=1,
                     stride=1, min_bin_photons=0.0,
                     min_map_photons=0.0, auto_register=False,
                     perpendicular_shift=(0.0, 0.0), reader_class=None,
                     parallel_channel=None, perpendicular_channel=None,
                     parallel_exposure=1.0, perpendicular_exposure=1.0):
    if parallel_channel is None or perpendicular_channel is None:
        raise ValueError('Photon channels must be selected explicitly')
    channels = (parallel_channel, perpendicular_channel)
    if any(isinstance(channel, (bool, np.bool_))
           or not isinstance(channel, (int, np.integer))
           or channel < 0 for channel in channels):
        raise ValueError('Photon channels must be non-negative integers')
    if reader_class is None:
        from flimkit.formats.PTU.reader import PTUFile
        reader_class = PTUFile

    with reader_class(parallel_path, verbose=False) as parallel_file:
        parallel = parallel_file.pixel_stack(channel=parallel_channel)
        parallel_time = np.asarray(parallel_file.time_ns, dtype=float)
    with reader_class(perpendicular_path, verbose=False) as perpendicular_file:
        perpendicular = perpendicular_file.pixel_stack(
            channel=perpendicular_channel)
        perpendicular_time = np.asarray(perpendicular_file.time_ns, dtype=float)

    if parallel.shape != perpendicular.shape:
        raise ValueError('Polarization PTU stacks must have the same shape')
    if (parallel_time.shape != perpendicular_time.shape
            or not np.allclose(parallel_time, perpendicular_time)):
        raise ValueError('Polarization PTUs must have matching time axes')
    if auto_register:
        perpendicular_shift = estimate_translation(
            parallel.sum(axis=-1), perpendicular.sum(axis=-1))

    result = analyze_anisotropy(
        parallel, perpendicular, parallel_time,
        background_bins=background_bins, analysis_bins=analysis_bins,
        g_factor=g_factor, spatial_window=spatial_window, stride=stride,
        min_bin_photons=min_bin_photons,
        min_map_photons=min_map_photons,
        perpendicular_shift=perpendicular_shift,
        parallel_exposure=parallel_exposure,
        perpendicular_exposure=perpendicular_exposure)
    result.metadata.update({
        'parallel_file': Path(parallel_path).name,
        'perpendicular_file': Path(perpendicular_path).name,
        'parallel_role': 'parallel',
        'perpendicular_role': 'perpendicular',
        'parallel_channel': int(parallel_channel),
        'perpendicular_channel': int(perpendicular_channel),
        'min_bin_photons': float(min_bin_photons),
        'min_map_photons': float(min_map_photons),
        'auto_registration': bool(auto_register),
    })
    if isinstance(background_bins, slice):
        result.metadata.update({
            'background_start_bin': background_bins.start,
            'background_stop_bin': background_bins.stop,
            'background_step': background_bins.step,
        })
    if isinstance(analysis_bins, slice):
        result.metadata.update({
            'analysis_start_bin': analysis_bins.start,
            'analysis_stop_bin': analysis_bins.stop,
            'analysis_step': analysis_bins.step,
        })
    return result


def save_anisotropy_npz(result, path):
    payload = {
        'time_ns': result.time_ns,
        'parallel_decay': result.parallel_decay,
        'perpendicular_decay': result.perpendicular_decay,
        'anisotropy_decay': result.anisotropy_decay,
        'anisotropy_cube': result.anisotropy_cube,
        'anisotropy_map': result.anisotropy_map,
        'parallel_intensity': result.parallel_intensity,
        'perpendicular_intensity': result.perpendicular_intensity,
        'parallel_background': result.parallel_background,
        'perpendicular_background': result.perpendicular_background,
        'valid_mask': result.valid_mask,
        'map_valid_mask': result.map_valid_mask,
        'total_counts': result.total_counts,
        'window_origins': result.window_origins,
        'perpendicular_shift': np.asarray(result.perpendicular_shift),
        'g_factor': result.g_factor,
        'parallel_exposure': result.parallel_exposure,
        'perpendicular_exposure': result.perpendicular_exposure,
        'spatial_window': result.spatial_window,
        'stride': result.stride,
    }
    if result.late_window_stability is not None:
        stability = result.late_window_stability
        payload.update({
            'late_window_time_ns': stability.time_ns,
            'late_window_nested_scale': stability.nested_scale,
            'late_window_nested_standard_error': (
                stability.nested_standard_error),
            'late_window_rolling_scale': stability.rolling_scale,
            'late_window_rolling_bins': stability.rolling_bins,
            'late_window_selected_start_bin': stability.selected_start_bin,
            'late_window_selected_scale': stability.selected_scale,
            'late_window_selected_parallel_photons': (
                stability.selected_parallel_photons),
            'late_window_selected_perpendicular_photons': (
                stability.selected_perpendicular_photons),
            'late_window_interpretation': stability.interpretation,
        })
    comparison = getattr(result, 'multicomponent_fit', None)
    if comparison is not None:
        payload.update({
            'advanced_max_components': comparison.max_components,
            'advanced_selected_component_count': (
                comparison.selected_component_count),
            'advanced_residual_type': 'signed_poisson_deviance',
        })
        for candidate in comparison.candidates:
            prefix = f'advanced_candidate_{candidate.component_count}'
            payload.update({
                f'{prefix}_times_ns': (
                    candidate.rotational_correlation_times_ns),
                f'{prefix}_weights': candidate.component_weights,
                f'{prefix}_intensity_lifetime_ns': (
                    candidate.intensity_lifetime_ns),
                f'{prefix}_initial_anisotropy': candidate.initial_anisotropy,
                f'{prefix}_amplitude': candidate.amplitude,
                f'{prefix}_parallel_model': candidate.parallel_model,
                f'{prefix}_perpendicular_model': candidate.perpendicular_model,
                f'{prefix}_parallel_residual': candidate.parallel_residual,
                f'{prefix}_perpendicular_residual': (
                    candidate.perpendicular_residual),
                f'{prefix}_poisson_deviance': candidate.poisson_deviance,
                f'{prefix}_degrees_of_freedom': candidate.degrees_of_freedom,
                f'{prefix}_deviance_per_degree_of_freedom': (
                    candidate.deviance_per_degree_of_freedom),
                f'{prefix}_bic': candidate.bic,
                f'{prefix}_aicc': candidate.aicc,
                f'{prefix}_success': candidate.success,
                f'{prefix}_eligible_for_selection': (
                    candidate.eligible_for_selection),
                f'{prefix}_message': candidate.message,
                f'{prefix}_parameters_at_bounds': np.asarray(
                    candidate.parameters_at_bounds, dtype=str),
                f'{prefix}_identifiable': candidate.identifiable,
                f'{prefix}_warnings': np.asarray(
                    candidate.identifiability_warnings, dtype=str),
                f'{prefix}_common_irf_shift_bins': (
                    candidate.common_irf_shift_bins),
                f'{prefix}_parallel_background': candidate.parallel_background,
                f'{prefix}_perpendicular_background': (
                    candidate.perpendicular_background),
                f'{prefix}_component_bounds_ns': candidate.component_bounds_ns,
                f'{prefix}_multistart_count': candidate.multistart_count,
                f'{prefix}_parameter_names': np.asarray(
                    candidate.parameter_names, dtype=str),
                f'{prefix}_start_indices': np.asarray([
                    record.start_index
                    for record in candidate.multistart_records], dtype=int),
                f'{prefix}_start_parameter_vectors': np.stack([
                    record.parameter_vector
                    for record in candidate.multistart_records]),
                f'{prefix}_start_times_ns': np.stack([
                    record.rotational_correlation_times_ns
                    for record in candidate.multistart_records]),
                f'{prefix}_start_weights': np.stack([
                    record.component_weights
                    for record in candidate.multistart_records]),
                f'{prefix}_start_poisson_deviance': np.asarray([
                    record.poisson_deviance
                    for record in candidate.multistart_records], dtype=float),
                f'{prefix}_start_success': np.asarray([
                    record.success
                    for record in candidate.multistart_records], dtype=bool),
                f'{prefix}_start_messages': np.asarray([
                    record.message
                    for record in candidate.multistart_records], dtype=str),
                f'{prefix}_start_parameters_at_bounds': np.asarray([
                    ';'.join(record.parameters_at_bounds)
                    for record in candidate.multistart_records], dtype=str),
            })
    if result.polarized_fit is not None:
        fit = result.polarized_fit
        payload.update({
            'fit_intensity_lifetime_ns': fit.intensity_lifetime_ns,
            'fit_rotational_correlation_ns': fit.rotational_correlation_ns,
            'fit_initial_anisotropy': fit.initial_anisotropy,
            'fit_amplitude': fit.amplitude,
            'fit_parallel_observed': (
                result.parallel_decay[:len(fit.parallel_model)]
                + result.parallel_background),
            'fit_perpendicular_observed': (
                result.perpendicular_decay[:len(fit.perpendicular_model)]
                + result.perpendicular_background),
            'fit_parallel_model': fit.parallel_model,
            'fit_perpendicular_model': fit.perpendicular_model,
            'fit_parallel_residual': fit.parallel_residual,
            'fit_perpendicular_residual': fit.perpendicular_residual,
            'fit_poisson_deviance': fit.poisson_deviance,
            'fit_common_irf_shift_bins': fit.common_irf_shift_bins,
            'fit_parallel_background': fit.parallel_background,
            'fit_perpendicular_background': fit.perpendicular_background,
            'fit_parameters_at_bounds': np.asarray(
                fit.parameters_at_bounds, dtype=str),
            'fit_success': fit.success,
            'fit_message': fit.message,
        })
    reserved_keys = set(payload)
    for key, value in result.metadata.items():
        if not isinstance(key, str) or key in reserved_keys:
            continue
        if key.endswith('_file'):
            value = Path(value).name
        if isinstance(value, (str, int, float, bool, np.number)):
            output_key = f'metadata_{key}' if key in {'file', 'allow_pickle'} else key
            if output_key not in payload:
                payload[output_key] = value
    np.savez_compressed(path, **payload)
