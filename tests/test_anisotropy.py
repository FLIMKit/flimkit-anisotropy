import numpy as np
import pytest


def test_polarized_decay_models_follow_lakowicz_channel_equations():
    from flimkit_anisotropy.anisotropy import polarized_decay_models

    time_ns = np.arange(6, dtype=float)
    delta_irf = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fluorescence = 300.0 * np.exp(-time_ns / 4.0)
    anisotropy = 0.3 * np.exp(-time_ns / 2.0)

    parallel, perpendicular = polarized_decay_models(
        time_ns, delta_irf, delta_irf,
        intensity_lifetime_ns=4.0, rotational_correlation_ns=2.0,
        initial_anisotropy=0.3, amplitude=300.0,
        g_factor=2.0, parallel_exposure=2.0,
        perpendicular_exposure=1.0,
        parallel_background=5.0, perpendicular_background=7.0)

    expected_parallel = 5.0 + 2.0 * fluorescence * (1.0 + 2.0 * anisotropy) / 3.0
    expected_perpendicular = 7.0 + fluorescence * (1.0 - anisotropy) / 6.0
    np.testing.assert_allclose(parallel, expected_parallel)
    np.testing.assert_allclose(perpendicular, expected_perpendicular)


def test_polarized_decay_models_include_previous_laser_pulses():
    from flimkit_anisotropy.anisotropy import polarized_decay_models

    time_ns = np.arange(4, dtype=float)
    delta_irf = np.array([1.0, 0.0, 0.0, 0.0])
    tau_ns = 2.0
    theta_ns = 1.0
    period_ns = 4.0
    amplitude = 90.0
    r0 = 0.3

    parallel, perpendicular = polarized_decay_models(
        time_ns, delta_irf, delta_irf,
        intensity_lifetime_ns=tau_ns,
        rotational_correlation_ns=theta_ns,
        initial_anisotropy=r0, amplitude=amplitude,
        repetition_period_ns=period_ns)

    intensity = (amplitude * np.exp(-time_ns / tau_ns)
                 / (1.0 - np.exp(-period_ns / tau_ns)))
    effective_ns = 1.0 / (1.0 / tau_ns + 1.0 / theta_ns)
    polarized = (amplitude * r0 * np.exp(-time_ns / effective_ns)
                 / (1.0 - np.exp(-period_ns / effective_ns)))
    np.testing.assert_allclose(parallel, (intensity + 2.0 * polarized) / 3.0)
    np.testing.assert_allclose(perpendicular, (intensity - polarized) / 3.0)


def test_polarized_decay_models_reject_mismatched_laser_period():
    from flimkit_anisotropy.anisotropy import polarized_decay_models

    time_ns = np.arange(8, dtype=float)
    irf = np.eye(1, 8, 0).ravel()

    with pytest.raises(ValueError, match='histogram duration'):
        polarized_decay_models(
            time_ns, irf, irf, intensity_lifetime_ns=3.0,
            rotational_correlation_ns=1.0, initial_anisotropy=0.2,
            amplitude=100.0, repetition_period_ns=6.0)


def test_polarized_decay_models_apply_common_irf_shift():
    from flimkit_anisotropy.anisotropy import polarized_decay_models

    time_ns = np.arange(8, dtype=float)
    delta_irf = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    unshifted, _ = polarized_decay_models(
        time_ns, delta_irf, delta_irf,
        intensity_lifetime_ns=3.0, rotational_correlation_ns=1.0,
        initial_anisotropy=0.2, amplitude=100.0)
    shifted, _ = polarized_decay_models(
        time_ns, delta_irf, delta_irf,
        intensity_lifetime_ns=3.0, rotational_correlation_ns=1.0,
        initial_anisotropy=0.2, amplitude=100.0,
        common_irf_shift_bins=1.0)

    np.testing.assert_allclose(shifted[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(shifted[1:], unshifted[:-1])


def test_global_polarized_fit_recovers_rotation_shift_and_backgrounds():
    from flimkit_anisotropy.anisotropy import (
        fit_polarized_decays, polarized_decay_models)

    time_ns = np.arange(128, dtype=float) * 0.1
    bins = np.arange(time_ns.size, dtype=float)
    parallel_irf = np.exp(-0.5 * ((bins - 8.0) / 1.2) ** 2)
    perpendicular_irf = np.exp(-0.5 * ((bins - 11.0) / 1.8) ** 2)
    parallel, perpendicular = polarized_decay_models(
        time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.2,
        rotational_correlation_ns=1.4,
        initial_anisotropy=0.32, amplitude=12000.0,
        g_factor=1.3, parallel_exposure=1.5,
        perpendicular_exposure=0.8,
        parallel_background=2.5, perpendicular_background=7.0,
        repetition_period_ns=12.8, common_irf_shift_bins=0.7)

    fitted = fit_polarized_decays(
        parallel, perpendicular, time_ns,
        parallel_irf=parallel_irf, perpendicular_irf=perpendicular_irf,
        intensity_lifetime_ns=3.2,
        g_factor=1.3, parallel_exposure=1.5,
        perpendicular_exposure=0.8,
        initial_parallel_background=0.0,
        initial_perpendicular_background=0.0,
        repetition_period_ns=12.8,
        initial_rotational_ns=2.0, initial_anisotropy=0.2)

    assert fitted.success
    assert fitted.intensity_lifetime_ns == 3.2
    assert fitted.rotational_correlation_ns == pytest.approx(1.4, rel=1e-4)
    assert fitted.initial_anisotropy == pytest.approx(0.32, rel=1e-4)
    assert fitted.common_irf_shift_bins == pytest.approx(0.7, abs=1e-4)
    assert fitted.parallel_background == pytest.approx(2.5, rel=1e-3)
    assert fitted.perpendicular_background == pytest.approx(7.0, rel=1e-3)
    np.testing.assert_allclose(fitted.parallel_model, parallel, rtol=1e-5)
    np.testing.assert_allclose(fitted.perpendicular_model, perpendicular, rtol=1e-5)


def test_global_fit_avoids_unequal_irf_bias_from_divided_anisotropy():
    from scipy.optimize import curve_fit
    from flimkit_anisotropy.anisotropy import (
        fit_polarized_decays, polarized_decay_models)

    time_ns = np.arange(256, dtype=float) * 0.1
    bins = np.arange(time_ns.size, dtype=float)
    parallel_irf = np.exp(-0.5 * ((bins - 8.0) / 1.2) ** 2)
    perpendicular_irf = np.exp(-0.5 * ((bins - 11.0) / 2.6) ** 2)
    perpendicular_irf *= 0.9 + 0.1 * np.exp(
        -np.maximum(bins - 11.0, 0.0) / 5.0)
    parallel, perpendicular = polarized_decay_models(
        time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.0, rotational_correlation_ns=0.8,
        initial_anisotropy=0.36, amplitude=30000.0,
        g_factor=0.78, parallel_exposure=1.0,
        perpendicular_exposure=1.6,
        parallel_background=2.5, perpendicular_background=7.0,
        repetition_period_ns=25.6, common_irf_shift_bins=0.7)

    fitted = fit_polarized_decays(
        parallel, perpendicular, time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.0, g_factor=0.78,
        parallel_exposure=1.0, perpendicular_exposure=1.6,
        initial_parallel_background=2.5,
        initial_perpendicular_background=7.0,
        repetition_period_ns=25.6)

    parallel_rate = parallel - 2.5
    perpendicular_rate = 0.78 * (perpendicular - 7.0) / 1.6
    ratio = ((parallel_rate - perpendicular_rate)
             / (parallel_rate + 2.0 * perpendicular_rate))
    peak = int(np.argmax(parallel + 2.0 * 0.78 * perpendicular / 1.6))
    fit_slice = slice(peak, peak + 70)
    relative_time = time_ns[fit_slice] - time_ns[peak]
    ratio_parameters, _ = curve_fit(
        lambda time, r0, theta: r0 * np.exp(-time / theta),
        relative_time, ratio[fit_slice], p0=(0.3, 1.0),
        bounds=([-0.2, 0.05], [0.8, 10.0]))

    assert fitted.rotational_correlation_ns == pytest.approx(0.8, rel=0.005)
    assert abs(ratio_parameters[1] / 0.8 - 1.0) > 0.2


def test_global_polarized_fit_reports_parameters_at_bounds():
    from flimkit_anisotropy.anisotropy import (
        fit_polarized_decays, polarized_decay_models)

    time_ns = np.arange(64, dtype=float) * 0.1
    irf = np.eye(1, 64, 5).ravel()
    parallel, perpendicular = polarized_decay_models(
        time_ns, irf, irf,
        intensity_lifetime_ns=3.0, rotational_correlation_ns=1.0,
        initial_anisotropy=-0.3, amplitude=5000.0,
        repetition_period_ns=6.4)

    fitted = fit_polarized_decays(
        parallel, perpendicular, time_ns, irf, irf,
        intensity_lifetime_ns=3.0, repetition_period_ns=6.4,
        initial_anisotropy=-0.1)

    assert 'initial_anisotropy' in fitted.parameters_at_bounds


def test_multicomponent_models_reduce_to_single_component_with_strict_tolerance():
    from flimkit_anisotropy.anisotropy import (
        multicomponent_polarized_decay_models, polarized_decay_models)

    time_ns = np.arange(128, dtype=float) * 0.1
    bins = np.arange(time_ns.size, dtype=float)
    parallel_irf = np.exp(-0.5 * ((bins - 8.0) / 1.2) ** 2)
    perpendicular_irf = np.exp(-0.5 * ((bins - 11.0) / 1.8) ** 2)
    expected = polarized_decay_models(
        time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.2, rotational_correlation_ns=1.4,
        initial_anisotropy=0.32, amplitude=12000.0,
        g_factor=1.3, parallel_exposure=1.5,
        perpendicular_exposure=0.8,
        parallel_background=2.5, perpendicular_background=7.0,
        repetition_period_ns=12.8, common_irf_shift_bins=0.7)

    actual = multicomponent_polarized_decay_models(
        time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.2,
        rotational_correlation_times_ns=[1.4], component_weights=[1.0],
        initial_anisotropy=0.32, amplitude=12000.0,
        g_factor=1.3, parallel_exposure=1.5,
        perpendicular_exposure=0.8,
        parallel_background=2.5, perpendicular_background=7.0,
        repetition_period_ns=12.8, common_irf_shift_bins=0.7)

    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)


def test_multicomponent_models_use_ordered_normalized_components():
    from flimkit_anisotropy.anisotropy import multicomponent_polarized_decay_models

    time_ns = np.arange(6, dtype=float)
    irf = np.eye(1, 6, 0).ravel()
    amplitude = 300.0
    lifetime_ns = 4.0
    r0 = 0.3
    times_ns = np.array([1.0, 3.0])
    weights = np.array([0.25, 0.75])

    parallel, perpendicular = multicomponent_polarized_decay_models(
        time_ns, irf, irf, intensity_lifetime_ns=lifetime_ns,
        rotational_correlation_times_ns=times_ns,
        component_weights=weights, initial_anisotropy=r0,
        amplitude=amplitude)

    intensity = amplitude * np.exp(-time_ns / lifetime_ns)
    anisotropy = r0 * sum(
        weight * np.exp(-time_ns / theta)
        for theta, weight in zip(times_ns, weights))
    polarized = intensity * anisotropy
    np.testing.assert_allclose(parallel, (intensity + 2.0 * polarized) / 3.0)
    np.testing.assert_allclose(perpendicular, (intensity - polarized) / 3.0)


@pytest.mark.parametrize(
    'times, weights, message', [
        ([2.0, 1.0], [0.5, 0.5], 'strictly increasing'),
        ([1.0, 2.0], [0.5, -0.5], 'non-negative'),
        ([1.0, 2.0], [0.4, 0.4], 'sum to one'),
        ([1.0], [0.5, 0.5], 'same length'),
    ])
def test_multicomponent_models_reject_invalid_components(times, weights, message):
    from flimkit_anisotropy.anisotropy import multicomponent_polarized_decay_models

    time_ns = np.arange(8, dtype=float)
    irf = np.eye(1, 8, 0).ravel()
    with pytest.raises(ValueError, match=message):
        multicomponent_polarized_decay_models(
            time_ns, irf, irf, intensity_lifetime_ns=3.0,
            rotational_correlation_times_ns=times,
            component_weights=weights, initial_anisotropy=0.3,
            amplitude=100.0)


def _multicomponent_synthetic_truth(
        times_ns, weights, seed=None, amplitude=4e6, g_factor=1.35):
    from flimkit_anisotropy.anisotropy import multicomponent_polarized_decay_models

    time_ns = np.arange(132, dtype=float) * (12.8 / 132.0)
    bins = np.arange(time_ns.size, dtype=float)
    parallel_irf = np.exp(-0.5 * ((bins - 8.0) / 1.1) ** 2)
    perpendicular_irf = np.exp(-0.5 * ((bins - 10.0) / 1.5) ** 2)
    parallel, perpendicular = multicomponent_polarized_decay_models(
        time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.0,
        rotational_correlation_times_ns=times_ns,
        component_weights=weights, initial_anisotropy=0.34,
        amplitude=amplitude, g_factor=g_factor,
        parallel_background=15.0, perpendicular_background=22.0,
        repetition_period_ns=12.8, common_irf_shift_bins=0.4)
    if seed is not None:
        rng = np.random.default_rng(seed)
        parallel = rng.poisson(parallel).astype(float)
        perpendicular = rng.poisson(perpendicular).astype(float)
    return time_ns, parallel_irf, perpendicular_irf, parallel, perpendicular


@pytest.mark.parametrize('seed', [31, 57, 89])
def test_clear_two_component_signal_is_recovered_but_not_called_resolved(seed):
    from flimkit_anisotropy.anisotropy import fit_multicomponent_polarized_decays

    truth_times = np.array([0.45, 3.2])
    truth_weights = np.array([0.35, 0.65])
    time_ns, parallel_irf, perpendicular_irf, parallel, perpendicular = (
        _multicomponent_synthetic_truth(truth_times, truth_weights, seed=seed))

    result = fit_multicomponent_polarized_decays(
        parallel, perpendicular, time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.0, g_factor=1.35,
        repetition_period_ns=12.8, max_components=2, multistart=6)

    assert result.selected_component_count == 2
    np.testing.assert_allclose(
        result.selected_fit.rotational_correlation_times_ns,
        truth_times, rtol=0.18)
    np.testing.assert_allclose(
        result.selected_fit.component_weights, truth_weights, atol=0.08)
    assert not result.has_resolved_model
    assert any('uncertainty validation unavailable' in warning
               for warning in result.selected_fit.identifiability_warnings)


@pytest.mark.parametrize('case', [
    'low_counts', 'weak_component', 'g_mismatch', 'irf_mismatch'])
def test_challenging_two_component_cases_remain_fail_closed(case):
    from flimkit_anisotropy.anisotropy import fit_multicomponent_polarized_decays

    weights = [0.35, 0.65]
    amplitude = 4e6
    truth_g = 1.35
    if case == 'low_counts':
        amplitude = 1e5
    elif case == 'weak_component':
        weights = [0.08, 0.92]
    elif case == 'g_mismatch':
        truth_g = 1.55
    time_ns, parallel_irf, perpendicular_irf, parallel, perpendicular = (
        _multicomponent_synthetic_truth(
            [0.45, 3.2], weights, seed=211, amplitude=amplitude,
            g_factor=truth_g))
    fit_parallel_irf = parallel_irf
    if case == 'irf_mismatch':
        fit_parallel_irf = np.roll(parallel_irf, 2)

    result = fit_multicomponent_polarized_decays(
        parallel, perpendicular, time_ns, fit_parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.0, g_factor=1.35,
        repetition_period_ns=12.8, max_components=2, multistart=5)

    assert all(np.isfinite(candidate.bic) for candidate in result.candidates)
    assert not result.has_resolved_model
    assert result.resolved_component_count == 0
    assert all(not candidate.identifiable for candidate in result.candidates)


def test_multicomponent_fit_selects_three_but_keeps_it_exploratory():
    from flimkit_anisotropy.anisotropy import fit_multicomponent_polarized_decays

    truth_times = np.array([0.35, 1.4, 5.0])
    truth_weights = np.array([0.25, 0.35, 0.40])
    time_ns, parallel_irf, perpendicular_irf, parallel, perpendicular = (
        _multicomponent_synthetic_truth(truth_times, truth_weights, seed=103))

    result = fit_multicomponent_polarized_decays(
        parallel, perpendicular, time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.0, g_factor=1.35,
        repetition_period_ns=12.8, max_components=3, multistart=6)

    assert result.selected_component_count == 3
    selected = result.selected_fit
    assert selected.success
    assert not selected.identifiable
    assert 'three-component parameter robustness is not validated' in (
        selected.identifiability_warnings)
    np.testing.assert_allclose(
        selected.rotational_correlation_times_ns, truth_times, rtol=0.18)
    np.testing.assert_allclose(selected.component_weights, truth_weights, atol=0.08)
    assert np.all(np.diff(selected.rotational_correlation_times_ns) > 0)
    assert selected.component_weights.sum() == pytest.approx(1.0)
    assert not result.has_resolved_model
    assert result.resolved_component_count == 0


@pytest.mark.parametrize('time_ns, message', [
    (np.array([0.0]), 'at least 2'),
    (np.array([0.0, np.nan]), 'finite'),
    (np.array([0.0, -0.1, -0.2]), 'strictly increasing'),
    (np.array([0.0, 0.1, 0.25]), 'evenly spaced'),
])
def test_multicomponent_model_rejects_invalid_time_axis(time_ns, message):
    from flimkit_anisotropy.anisotropy import multicomponent_polarized_decay_models

    irf = np.ones(time_ns.size, dtype=float)
    with pytest.raises(ValueError, match=message):
        multicomponent_polarized_decay_models(
            time_ns, irf, irf,
            intensity_lifetime_ns=3.0,
            rotational_correlation_times_ns=[1.0],
            component_weights=[1.0], initial_anisotropy=0.3,
            repetition_period_ns=12.8, common_irf_shift_bins=0.0,
            parallel_background=0.0, perpendicular_background=0.0,
            parallel_exposure=1.0, perpendicular_exposure=1.0,
            g_factor=1.0, amplitude=100.0)


def test_manual_bounds_apply_consistently_to_every_candidate():
    from flimkit_anisotropy.anisotropy import fit_multicomponent_polarized_decays

    time_ns, parallel_irf, perpendicular_irf, parallel, perpendicular = (
        _multicomponent_synthetic_truth(
            [0.35, 1.4, 5.0], [0.3, 0.35, 0.35], seed=23))
    manual_bounds = np.array([0.2, 6.0])

    result = fit_multicomponent_polarized_decays(
        parallel, perpendicular, time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.0, repetition_period_ns=12.8,
        max_components=3, multistart=2,
        component_bounds_ns=manual_bounds)

    for component_count, candidate in enumerate(result.candidates, start=1):
        np.testing.assert_allclose(
            candidate.component_bounds_ns,
            np.tile(manual_bounds, (component_count, 1)))


def test_multicomponent_result_defaults_to_bic_comparison_mode():
    from flimkit_anisotropy.anisotropy import MulticomponentFitResult

    result = MulticomponentFitResult(
        max_components=1, selected_component_count=0, candidates=())

    assert result.selection_mode == 'bic_comparison'


def test_fixed_k_uses_one_bound_range_per_exponential_component():
    from flimkit_anisotropy.anisotropy import fit_multicomponent_polarized_decays

    time_ns, parallel_irf, perpendicular_irf, parallel, perpendicular = (
        _multicomponent_synthetic_truth([0.55, 3.5], [0.4, 0.6], seed=31))
    component_bounds = np.array([[0.2, 1.0], [1.5, 8.0]])

    result = fit_multicomponent_polarized_decays(
        parallel, perpendicular, time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.0, repetition_period_ns=12.8,
        max_components=3, fixed_component_count=2, multistart=3,
        component_bounds_ns=component_bounds)

    assert result.selection_mode == 'fixed_k'
    assert result.max_components == 2
    assert result.selected_component_count == 2
    assert len(result.candidates) == 1
    assert result.selected_fit is result.candidates[0]
    assert result.selected_fit.component_count == 2
    np.testing.assert_allclose(
        result.selected_fit.component_bounds_ns, component_bounds)
    assert 0.2 <= result.selected_fit.rotational_correlation_times_ns[0] <= 1.0
    assert 1.5 <= result.selected_fit.rotational_correlation_times_ns[1] <= 8.0


def test_fixed_k_touching_ranges_still_produce_strictly_ordered_times():
    from flimkit_anisotropy.anisotropy import _manual_component_times

    bounds = np.array([[0.1, 0.8], [0.8, 2.0], [2.0, 10.0]])
    for parameters in (
            np.full(3, -8.0), np.zeros(3), np.full(3, 8.0)):
        times = _manual_component_times(parameters, bounds)
        assert np.all(np.diff(times) > 0)
        assert np.all(times > bounds[:, 0])
        assert np.all(times < bounds[:, 1])


def test_multicomponent_fit_rejects_spurious_extra_components():
    from flimkit_anisotropy.anisotropy import fit_multicomponent_polarized_decays

    time_ns, parallel_irf, perpendicular_irf, parallel, perpendicular = (
        _multicomponent_synthetic_truth([1.5], [1.0], seed=101))

    result = fit_multicomponent_polarized_decays(
        parallel, perpendicular, time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.0, g_factor=1.35,
        repetition_period_ns=12.8, max_components=3, multistart=5)

    assert result.selected_component_count == 1
    assert len(result.candidates) == 3
    assert not result.candidates[0].identifiable
    assert any('uncertainty validation unavailable' in warning
               for warning in result.candidates[0].identifiability_warnings)
    assert not result.has_resolved_model
    for candidate in result.candidates[1:]:
        assert not candidate.identifiable
        assert 'simpler model preferred by BIC' in candidate.identifiability_warnings
    assert result.candidates[0].bic < result.candidates[1].bic
    assert result.candidates[0].bic < result.candidates[2].bic


def test_multicomponent_fit_validates_component_controls():
    from flimkit_anisotropy.anisotropy import fit_multicomponent_polarized_decays

    time_ns = np.arange(16, dtype=float) * 0.1
    irf = np.eye(1, 16, 1).ravel()
    observed = np.ones(16)
    with pytest.raises(ValueError, match='between one and three'):
        fit_multicomponent_polarized_decays(
            observed, observed, time_ns, irf, irf,
            intensity_lifetime_ns=3.0, max_components=4)
    with pytest.raises(ValueError, match='multistart'):
        fit_multicomponent_polarized_decays(
            observed, observed, time_ns, irf, irf,
            intensity_lifetime_ns=3.0, max_components=2, multistart=0)
    with pytest.raises(ValueError, match='at most 32'):
        fit_multicomponent_polarized_decays(
            observed, observed, time_ns, irf, irf,
            intensity_lifetime_ns=3.0, max_components=2, multistart=33)
    with pytest.raises(ValueError, match='fixed_component_count'):
        fit_multicomponent_polarized_decays(
            observed, observed, time_ns, irf, irf,
            intensity_lifetime_ns=3.0, fixed_component_count=4)
    with pytest.raises(ValueError, match='fixed_component_count'):
        fit_multicomponent_polarized_decays(
            observed, observed, time_ns, irf, irf,
            intensity_lifetime_ns=3.0, fixed_component_count=True,
            component_bounds_ns=((0.2, 4.0),))
    with pytest.raises(ValueError, match='fixed-K mode'):
        fit_multicomponent_polarized_decays(
            observed, observed, time_ns, irf, irf,
            intensity_lifetime_ns=3.0,
            component_bounds_ns=((0.2, 0.8), (1.0, 4.0)))
    with pytest.raises(ValueError, match='exactly one range per component'):
        fit_multicomponent_polarized_decays(
            observed, observed, time_ns, irf, irf,
            intensity_lifetime_ns=3.0, fixed_component_count=2)
    with pytest.raises(ValueError, match='exactly one range per component'):
        fit_multicomponent_polarized_decays(
            observed, observed, time_ns, irf, irf,
            intensity_lifetime_ns=3.0, fixed_component_count=2,
            component_bounds_ns=(0.2, 4.0))
    with pytest.raises(ValueError, match='ordered, and non-overlapping'):
        fit_multicomponent_polarized_decays(
            observed, observed, time_ns, irf, irf,
            intensity_lifetime_ns=3.0, fixed_component_count=2,
            component_bounds_ns=((0.2, 2.0), (1.0, 4.0)))
    with pytest.raises(ValueError, match='one range per component'):
        fit_multicomponent_polarized_decays(
            observed, observed, time_ns, irf, irf,
            intensity_lifetime_ns=3.0, fixed_component_count=3,
            component_bounds_ns=((0.2, 0.8), (1.0, 4.0)))


def test_periodic_delta_irf_shift_is_defined_across_optimizer_bounds():
    from flimkit_anisotropy.anisotropy import (
        fit_multicomponent_polarized_decays,
        multicomponent_polarized_decay_models)

    time_ns = np.arange(64, dtype=float) * 0.2
    irf = np.zeros(64, dtype=float)
    irf[0] = 1.0
    parallel, perpendicular = multicomponent_polarized_decay_models(
        time_ns, irf, irf, intensity_lifetime_ns=3.0,
        rotational_correlation_times_ns=[0.6, 3.0],
        component_weights=[0.4, 0.6], initial_anisotropy=0.3,
        amplitude=1e5, parallel_background=2.0,
        perpendicular_background=3.0, repetition_period_ns=12.8,
        common_irf_shift_bins=0.4)

    shifted_parallel, shifted_perpendicular = (
        multicomponent_polarized_decay_models(
            time_ns, irf, irf, intensity_lifetime_ns=3.0,
            rotational_correlation_times_ns=[0.6, 3.0],
            component_weights=[0.4, 0.6], initial_anisotropy=0.3,
            amplitude=1e5, repetition_period_ns=12.8,
            common_irf_shift_bins=2.0))
    assert np.all(np.isfinite(shifted_parallel))
    assert np.all(np.isfinite(shifted_perpendicular))

    result = fit_multicomponent_polarized_decays(
        parallel, perpendicular, time_ns, irf, irf,
        intensity_lifetime_ns=3.0, repetition_period_ns=12.8,
        max_components=2, multistart=2)
    assert result.selected_component_count in {1, 2}
    assert all(len(candidate.multistart_records) == 2
               for candidate in result.candidates)
    assert all(any(record.success for record in candidate.multistart_records)
               for candidate in result.candidates)


def test_nonconverged_starts_cannot_win_model_selection():
    from types import SimpleNamespace
    from unittest.mock import patch
    from flimkit_anisotropy.anisotropy import fit_multicomponent_polarized_decays

    time_ns = np.arange(16, dtype=float) * 0.2
    irf = np.zeros(16, dtype=float)
    irf[2] = 1.0
    observed = np.linspace(100.0, 5.0, 16)
    parameters = np.array([0.0, 0.2, np.log(100.0), 0.0, 0.0, 0.0])
    failed = SimpleNamespace(
        x=parameters, active_mask=np.zeros(6, dtype=int), success=False,
        message='failed start')
    converged = SimpleNamespace(
        x=parameters, active_mask=np.zeros(6, dtype=int), success=True,
        message='converged start')

    with patch('flimkit_anisotropy.anisotropy.least_squares',
               side_effect=[failed, converged]):
        result = fit_multicomponent_polarized_decays(
            observed, observed, time_ns, irf, irf,
            intensity_lifetime_ns=3.0, max_components=1, multistart=2)

    assert result.selected_component_count == 1
    assert result.selected_fit.success
    assert result.selected_fit.message == 'converged start'
    assert [record.success for record in result.selected_fit.multistart_records] == [
        False, True]


def test_all_nonconverged_starts_return_ineligible_audit_candidate():
    from types import SimpleNamespace
    from unittest.mock import patch
    from flimkit_anisotropy.anisotropy import fit_multicomponent_polarized_decays

    time_ns = np.arange(16, dtype=float) * 0.2
    irf = np.zeros(16, dtype=float)
    irf[2] = 1.0
    observed = np.linspace(100.0, 5.0, 16)
    parameters = np.array([0.0, 0.2, np.log(100.0), 0.0, 0.0, 0.0])
    failed = SimpleNamespace(
        x=parameters, active_mask=np.zeros(6, dtype=int), success=False,
        message='failed start')

    with patch('flimkit_anisotropy.anisotropy.least_squares',
               side_effect=[failed, failed]):
        result = fit_multicomponent_polarized_decays(
            observed, observed, time_ns, irf, irf,
            intensity_lifetime_ns=3.0, max_components=1, multistart=2)

    assert result.selected_component_count == 0
    assert result.selected_fit is None
    assert not result.has_resolved_model
    candidate = result.candidates[0]
    assert not candidate.eligible_for_selection
    assert not candidate.success
    assert np.isinf(candidate.bic)
    assert len(candidate.multistart_records) == 2
    assert all(not record.success for record in candidate.multistart_records)
    assert 'no optimizer start converged' in candidate.identifiability_warnings


def test_multicomponent_fit_preserves_every_start_and_poisson_residuals():
    from flimkit_anisotropy.anisotropy import fit_multicomponent_polarized_decays

    time_ns, parallel_irf, perpendicular_irf, parallel, perpendicular = (
        _multicomponent_synthetic_truth([1.5], [1.0], seed=17))
    result = fit_multicomponent_polarized_decays(
        parallel, perpendicular, time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.0, repetition_period_ns=12.8,
        max_components=2, multistart=4)

    candidate = result.candidates[1]
    assert len(candidate.multistart_records) == 4
    assert [record.start_index for record in candidate.multistart_records] == list(range(4))
    for record in candidate.multistart_records:
        assert record.parameter_vector.shape == (8,)
        assert record.rotational_correlation_times_ns.shape == (2,)
        assert record.component_weights.shape == (2,)
        assert np.isfinite(record.poisson_deviance)
        assert isinstance(record.success, bool)
        assert isinstance(record.message, str)
        assert isinstance(record.parameters_at_bounds, tuple)

    def signed_poisson(observed, expected):
        expected = np.maximum(expected, 1e-12)
        contribution = expected - observed
        positive = observed > 0
        contribution[positive] += observed[positive] * np.log(
            observed[positive] / expected[positive])
        return np.sign(expected - observed) * np.sqrt(
            np.maximum(2.0 * contribution, 0.0))

    np.testing.assert_allclose(
        candidate.parallel_residual,
        signed_poisson(parallel, candidate.parallel_model))
    np.testing.assert_allclose(
        candidate.perpendicular_residual,
        signed_poisson(perpendicular, candidate.perpendicular_model))


def test_multicomponent_fit_rejects_gross_model_mismatch():
    from flimkit_anisotropy.anisotropy import fit_multicomponent_polarized_decays

    time_ns = np.arange(16, dtype=float) * 0.1
    irf = np.eye(1, 16, 1).ravel()
    parallel = np.tile([1.0, 10000.0], 8)
    perpendicular = np.tile([10000.0, 1.0], 8)

    result = fit_multicomponent_polarized_decays(
        parallel, perpendicular, time_ns, irf, irf,
        intensity_lifetime_ns=3.0, repetition_period_ns=1.6,
        max_components=1, multistart=2)

    selected = result.selected_fit
    assert selected.deviance_per_degree_of_freedom > 2.0
    assert 'Poisson deviance per degree of freedom exceeds 2' in (
        selected.identifiability_warnings)
    assert not result.has_resolved_model
    assert result.resolved_component_count == 0


def test_late_window_stability_uses_exposure_corrected_nested_counts():
    from flimkit_anisotropy.anisotropy import calculate_late_window_stability

    time_ns = np.arange(4, dtype=float)
    parallel = np.array([10.0, 20.0, 30.0, 40.0])
    perpendicular = np.array([5.0, 10.0, 15.0, 20.0])

    result = calculate_late_window_stability(
        parallel, perpendicular, time_ns,
        parallel_exposure=2.0, perpendicular_exposure=1.0,
        selected_start_ns=1.5)

    np.testing.assert_allclose(result.nested_scale, 1.0)
    np.testing.assert_allclose(
        result.nested_standard_error,
        np.sqrt(1.0 / np.array([100.0, 90.0, 70.0, 40.0])
                + 1.0 / np.array([50.0, 45.0, 35.0, 20.0])))
    assert result.selected_start_bin == 2
    assert result.selected_scale == pytest.approx(1.0)
    assert result.selected_parallel_photons == 70.0
    assert result.selected_perpendicular_photons == 35.0
    assert result.interpretation == 'effective_late_window_scale'


def test_late_window_stability_reports_short_rolling_ratio():
    from flimkit_anisotropy.anisotropy import calculate_late_window_stability

    result = calculate_late_window_stability(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([5.0, 10.0, 15.0, 20.0]),
        np.arange(4, dtype=float),
        parallel_exposure=2.0, perpendicular_exposure=1.0,
        selected_start_ns=2.0, rolling_bins=2)

    np.testing.assert_allclose(result.rolling_scale[:3], 1.0)
    assert np.isnan(result.rolling_scale[3])
    assert result.rolling_bins == 2


def test_late_window_stability_rejects_invalid_inputs():
    from flimkit_anisotropy.anisotropy import calculate_late_window_stability

    time_ns = np.arange(4, dtype=float)
    with pytest.raises(ValueError, match='finite and non-negative'):
        calculate_late_window_stability(
            np.array([1.0, 2.0, -1.0, 4.0]), np.ones(4), time_ns)
    with pytest.raises(ValueError, match='contain photons in both channels'):
        calculate_late_window_stability(
            np.ones(4), np.zeros(4), time_ns, selected_start_ns=2.0)
    with pytest.raises(ValueError, match='inside the TCSPC time axis'):
        calculate_late_window_stability(
            np.ones(4), np.ones(4), time_ns, selected_start_ns=4.0)


def test_calculate_anisotropy_uses_parallel_perpendicular_formula():
    from flimkit_anisotropy.anisotropy import calculate_anisotropy

    parallel = np.array([60.0, 30.0])
    perpendicular = np.array([20.0, 10.0])

    result = calculate_anisotropy(parallel, perpendicular)

    np.testing.assert_allclose(result, [0.4, 0.4])


def test_calculate_anisotropy_applies_g_factor():
    from flimkit_anisotropy.anisotropy import calculate_anisotropy

    result = calculate_anisotropy(
        np.array([80.0]), np.array([20.0]), g_factor=2.0)

    np.testing.assert_allclose(result, [0.25])


def test_calculate_anisotropy_normalizes_exposure():
    from flimkit_anisotropy.anisotropy import calculate_anisotropy

    result = calculate_anisotropy(
        np.array([120.0]), np.array([20.0]),
        parallel_exposure=2.0, perpendicular_exposure=1.0)

    np.testing.assert_allclose(result, [0.4])


def test_calculate_anisotropy_masks_low_denominators():
    from flimkit_anisotropy.anisotropy import calculate_anisotropy

    result = calculate_anisotropy(
        np.array([40.0, 400.0]), np.array([20.0, 100.0]),
        min_denominator=100.0)

    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1], 0.5)


def test_spatial_window_sum_uses_overlapping_valid_windows():
    from flimkit_anisotropy.anisotropy import spatial_window_sum

    cube = np.arange(16, dtype=float).reshape(4, 4, 1)

    pooled = spatial_window_sum(cube, window_size=3, stride=1)

    assert pooled.shape == (2, 2, 1)
    np.testing.assert_allclose(pooled[..., 0], [[45.0, 54.0],
                                                [81.0, 90.0]])


def test_subtract_background_uses_per_decay_prepeak_median():
    from flimkit_anisotropy.anisotropy import subtract_background

    data = np.array([[2.0, 2.0, 6.0, 10.0],
                     [1.0, 3.0, 7.0, 11.0]])

    corrected, background = subtract_background(data, slice(0, 2))

    np.testing.assert_allclose(background, [2.0, 2.0])
    np.testing.assert_allclose(corrected, [[0.0, 0.0, 4.0, 8.0],
                                           [-1.0, 1.0, 5.0, 9.0]])


def test_analyze_anisotropy_has_no_global_fit_by_default():
    from flimkit_anisotropy.anisotropy import analyze_anisotropy

    data = np.ones((1, 1, 3), dtype=float)
    result = analyze_anisotropy(
        data, data, np.arange(3, dtype=float),
        background_bins=slice(0, 1))

    assert result.polarized_fit is None


def test_analyze_anisotropy_returns_decay_and_overlapping_map():
    from flimkit_anisotropy.anisotropy import analyze_anisotropy

    parallel = np.full((5, 5, 6), 2.0)
    perpendicular = np.full((5, 5, 6), 3.0)
    parallel[..., 2:] = 22.0
    perpendicular[..., 2:] = 13.0

    result = analyze_anisotropy(
        parallel, perpendicular, np.arange(6, dtype=float),
        background_bins=slice(0, 2), analysis_bins=slice(2, 6),
        spatial_window=3, stride=1)

    assert result.anisotropy_cube.shape == (3, 3, 6)
    assert result.window_origins.shape == (3, 3, 2)
    np.testing.assert_array_equal(result.window_origins[0, 0], [0, 0])
    np.testing.assert_array_equal(result.window_origins[-1, -1], [2, 2])
    np.testing.assert_allclose(result.anisotropy_decay[2:], 0.25)
    np.testing.assert_allclose(result.anisotropy_map, 0.25)
    np.testing.assert_allclose(result.parallel_background, 50.0)
    np.testing.assert_allclose(result.perpendicular_background, 75.0)


def test_estimate_translation_finds_shift_to_align_moving_image():
    from scipy.ndimage import shift
    from flimkit_anisotropy.anisotropy import estimate_translation

    y, x = np.mgrid[:64, :64]
    reference = np.exp(-((y - 24.0) ** 2 + (x - 35.0) ** 2) / 40.0)
    reference += 0.6 * np.exp(-((y - 43.0) ** 2 + (x - 15.0) ** 2) / 18.0)
    moving = shift(reference, (1.25, -0.75), order=1, mode='constant')

    estimated = estimate_translation(reference, moving)

    np.testing.assert_allclose(estimated, (-1.25, 0.75), atol=0.2)


def test_estimate_translation_rejects_featureless_images():
    from flimkit_anisotropy.anisotropy import estimate_translation

    image = np.ones((16, 16), dtype=float)

    with pytest.raises(ValueError, match='spatial variation'):
        estimate_translation(image, image)


def test_estimate_translation_rejects_shift_at_search_boundary():
    from scipy.ndimage import shift
    from flimkit_anisotropy.anisotropy import estimate_translation

    y, x = np.mgrid[:64, :64]
    reference = np.exp(-((y - 30.0) ** 2 + (x - 30.0) ** 2) / 30.0)
    moving = shift(reference, (4.0, 0.0), order=1, mode='constant')

    with pytest.raises(RuntimeError, match='search boundary'):
        estimate_translation(reference, moving, max_shift=1.0)


def test_estimate_translation_rejects_low_confidence_alignment():
    from flimkit_anisotropy.anisotropy import estimate_translation

    generator = np.random.default_rng(20260808)
    reference = generator.random((64, 64))
    moving = generator.random((64, 64))

    with pytest.raises(RuntimeError, match='confidence'):
        estimate_translation(reference, moving)


def test_apply_translation_moves_only_spatial_axes():
    from flimkit_anisotropy.anisotropy import apply_translation

    cube = np.zeros((5, 5, 2), dtype=float)
    cube[2, 1] = [1.0, 2.0]

    registered = apply_translation(cube, (0.0, 1.0))

    np.testing.assert_allclose(registered[2, 2], [1.0, 2.0])
    np.testing.assert_allclose(registered.sum(axis=(0, 1)), [1.0, 2.0])


def test_analyze_anisotropy_applies_perpendicular_registration_shift():
    from flimkit_anisotropy.anisotropy import analyze_anisotropy

    parallel = np.ones((5, 5, 4), dtype=float)
    perpendicular = np.ones((5, 5, 4), dtype=float)
    parallel[2, 2, 2:] = 21.0
    perpendicular[2, 1, 2:] = 11.0

    result = analyze_anisotropy(
        parallel, perpendicular, np.arange(4, dtype=float),
        background_bins=slice(0, 2), analysis_bins=slice(2, 4),
        perpendicular_shift=(0.0, 1.0))

    np.testing.assert_allclose(result.anisotropy_map[2, 2], 0.25)
    np.testing.assert_allclose(result.perpendicular_shift, (0.0, 1.0))


def test_registration_masks_windows_without_full_perpendicular_support():
    from flimkit_anisotropy.anisotropy import analyze_anisotropy

    parallel = np.ones((4, 4, 3), dtype=float)
    perpendicular = np.ones((4, 4, 3), dtype=float)
    parallel[..., 2] = 11.0
    perpendicular[..., 2] = 6.0

    result = analyze_anisotropy(
        parallel, perpendicular, np.arange(3, dtype=float),
        background_bins=slice(0, 2), analysis_bins=slice(2, 3),
        perpendicular_shift=(0.0, 1.0))

    assert not result.valid_mask[:, 0].any()
    assert not result.map_valid_mask[:, 0].any()
    assert result.map_valid_mask[:, 1:].all()


def test_analyze_ptu_pair_uses_reader_contract_and_explicit_orientation():
    from flimkit_anisotropy.anisotropy import analyze_ptu_pair

    parallel = np.full((3, 3, 4), 2.0)
    perpendicular = np.full((3, 3, 4), 3.0)
    parallel[..., 2:] = 22.0
    perpendicular[..., 2:] = 13.0
    stacks = {'parallel.ptu': parallel, 'perpendicular.ptu': perpendicular}
    channels = {}

    class FakeReader:
        def __init__(self, path, verbose=False):
            self.path = path
            self.time_ns = np.arange(4, dtype=float)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def pixel_stack(self, channel=None):
            channels[self.path] = channel
            return stacks[self.path]

    result = analyze_ptu_pair(
        'parallel.ptu', 'perpendicular.ptu',
        parallel_channel=1, perpendicular_channel=2,
        background_bins=slice(0, 2), analysis_bins=slice(2, 4),
        min_bin_photons=7.0, min_map_photons=11.0,
        reader_class=FakeReader)

    np.testing.assert_allclose(result.anisotropy_decay[2:], 0.25)
    assert channels == {'parallel.ptu': 1, 'perpendicular.ptu': 2}
    assert result.metadata['parallel_role'] == 'parallel'
    assert result.metadata['perpendicular_role'] == 'perpendicular'
    assert result.metadata['background_start_bin'] == 0
    assert result.metadata['background_stop_bin'] == 2
    assert result.metadata['analysis_start_bin'] == 2
    assert result.metadata['analysis_stop_bin'] == 4
    assert result.metadata['min_bin_photons'] == 7.0
    assert result.metadata['min_map_photons'] == 11.0


def test_analyze_ptu_pair_rejects_invalid_photon_channels():
    from flimkit_anisotropy.anisotropy import analyze_ptu_pair

    with pytest.raises(ValueError, match='non-negative integers'):
        analyze_ptu_pair(
            'parallel.ptu', 'perpendicular.ptu',
            parallel_channel=-1, perpendicular_channel=0,
            background_bins=slice(0, 1), reader_class=object)


def test_analyze_anisotropy_masks_bins_using_observed_counts():
    from flimkit_anisotropy.anisotropy import analyze_anisotropy

    parallel = np.zeros((1, 1, 4), dtype=float)
    perpendicular = np.zeros((1, 1, 4), dtype=float)
    parallel[..., 2:] = 40.0
    perpendicular[..., 2:] = 40.0

    result = analyze_anisotropy(
        parallel, perpendicular, np.arange(4, dtype=float),
        background_bins=slice(0, 2), analysis_bins=slice(2, 4),
        min_bin_photons=100.0)

    assert np.isnan(result.anisotropy_cube[..., 2:]).all()


def test_valid_masks_exclude_nonfinite_anisotropy_values():
    from flimkit_anisotropy.anisotropy import analyze_anisotropy

    parallel = np.array([[[20.0, 20.0, 10.0]]])
    perpendicular = np.array([[[20.0, 20.0, 10.0]]])

    result = analyze_anisotropy(
        parallel, perpendicular, np.arange(3, dtype=float),
        background_bins=slice(0, 2), analysis_bins=slice(2, 3))

    assert not result.valid_mask[0, 0, 2]
    assert not result.map_valid_mask[0, 0]
    assert np.isnan(result.anisotropy_cube[0, 0, 2])
    assert np.isnan(result.anisotropy_map[0, 0])


def test_analyze_anisotropy_rejects_nonfinite_photon_thresholds():
    from flimkit_anisotropy.anisotropy import analyze_anisotropy

    data = np.ones((1, 1, 3), dtype=float)

    with pytest.raises(ValueError, match='Photon thresholds'):
        analyze_anisotropy(
            data, data, np.arange(3, dtype=float),
            background_bins=slice(0, 1), min_map_photons=np.nan)


def test_analyze_anisotropy_rejects_scalar_background_selector():
    from flimkit_anisotropy.anisotropy import analyze_anisotropy

    data = np.ones((1, 1, 3), dtype=float)

    with pytest.raises(ValueError, match='background_bins'):
        analyze_anisotropy(
            data, data, np.arange(3, dtype=float), background_bins=0)


def test_analyze_anisotropy_rejects_scalar_analysis_selector():
    from flimkit_anisotropy.anisotropy import analyze_anisotropy

    data = np.ones((1, 1, 3), dtype=float)

    with pytest.raises(ValueError, match='analysis_bins'):
        analyze_anisotropy(
            data, data, np.arange(3, dtype=float),
            background_bins=slice(0, 1), analysis_bins=2)


def test_save_anisotropy_npz_preserves_masks_and_safe_metadata(tmp_path):
    from flimkit_anisotropy.anisotropy import (
        PolarizedFitResult, analyze_anisotropy,
        calculate_late_window_stability, save_anisotropy_npz)

    parallel = np.ones((2, 2, 4), dtype=float)
    perpendicular = np.ones((2, 2, 4), dtype=float)
    result = analyze_anisotropy(
        parallel, perpendicular, np.arange(4, dtype=float),
        background_bins=slice(0, 1), analysis_bins=slice(1, 4))
    result.metadata.update({
        'parallel_file': 'parallel.ptu',
        'perpendicular_file': 'perpendicular.ptu',
        'g_factor': 999.0,
        'file': 'metadata-value',
    })
    result.polarized_fit = PolarizedFitResult(
        intensity_lifetime_ns=3.2,
        rotational_correlation_ns=1.4,
        initial_anisotropy=0.32,
        amplitude=12000.0,
        parallel_model=np.arange(3, dtype=float),
        perpendicular_model=np.arange(3, dtype=float) + 1.0,
        parallel_residual=np.zeros(3),
        perpendicular_residual=np.ones(3),
        poisson_deviance=5.0,
        success=True,
        message='complete',
        common_irf_shift_bins=0.7,
        parallel_background=2.5,
        perpendicular_background=7.0)
    result.late_window_stability = calculate_late_window_stability(
        np.array([20.0, 30.0, 40.0, 50.0]),
        np.array([10.0, 15.0, 20.0, 25.0]),
        np.arange(4, dtype=float), selected_start_ns=2.0,
        rolling_bins=2)
    path = tmp_path / 'result.npz'

    save_anisotropy_npz(result, path)

    saved = np.load(path, allow_pickle=False)
    np.testing.assert_array_equal(saved['valid_mask'], result.valid_mask)
    np.testing.assert_array_equal(saved['window_origins'], result.window_origins)
    assert saved['parallel_background'].item() == result.parallel_background
    assert saved['perpendicular_background'].item() == result.perpendicular_background
    assert saved['g_factor'].item() == result.g_factor
    assert 'metadata_g_factor' not in saved.files
    assert saved['metadata_file'].item() == 'metadata-value'
    assert saved['parallel_file'].item() == 'parallel.ptu'
    assert '/private/' not in saved['parallel_file'].item()
    assert saved['fit_intensity_lifetime_ns'].item() == 3.2
    assert saved['fit_rotational_correlation_ns'].item() == 1.4
    assert saved['fit_initial_anisotropy'].item() == 0.32
    assert saved['fit_common_irf_shift_bins'].item() == 0.7
    assert saved['fit_parallel_background'].item() == 2.5
    assert saved['fit_perpendicular_background'].item() == 7.0
    np.testing.assert_array_equal(
        saved['fit_parallel_observed'],
        (result.parallel_decay + result.parallel_background)[:3])
    np.testing.assert_array_equal(
        saved['fit_parallel_model'], result.polarized_fit.parallel_model)
    np.testing.assert_array_equal(
        saved['late_window_time_ns'],
        result.late_window_stability.time_ns)
    np.testing.assert_array_equal(
        saved['late_window_nested_scale'],
        result.late_window_stability.nested_scale)
    np.testing.assert_array_equal(
        saved['late_window_rolling_scale'],
        result.late_window_stability.rolling_scale)
    assert saved['late_window_selected_scale'].item() == pytest.approx(2.0)


def test_save_anisotropy_npz_preserves_all_multicomponent_candidates(tmp_path):
    from flimkit_anisotropy.anisotropy import (
        analyze_anisotropy, fit_multicomponent_polarized_decays,
        save_anisotropy_npz)

    time_ns, parallel_irf, perpendicular_irf, parallel, perpendicular = (
        _multicomponent_synthetic_truth([0.45, 3.2], [0.35, 0.65], seed=102))
    comparison = fit_multicomponent_polarized_decays(
        parallel, perpendicular, time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.0, g_factor=1.35,
        repetition_period_ns=12.8, max_components=2,
        fixed_component_count=2, multistart=4,
        component_bounds_ns=((0.2, 0.9), (1.1, 6.0)))
    result = analyze_anisotropy(
        parallel[None, None, :], perpendicular[None, None, :], time_ns,
        background_bins=slice(0, 2), analysis_bins=slice(0, 40),
        g_factor=1.35, min_bin_photons=0.0, min_map_photons=0.0)
    result.multicomponent_fit = comparison
    result.metadata.update({
        'parallel_irf_file': 'parallel_irf.ptu',
        'perpendicular_irf_file': 'perpendicular_irf.ptu',
        'repetition_period_ns': 12.8,
        'global_fit_bins': 132,
        'analysis_mode': 'advanced',
        'advanced_fit_mode': 'fixed_k',
        'advanced_fixed_component_count': 2,
        'advanced_component_range_mode': 'per-component',
    })
    path = tmp_path / 'advanced.npz'

    save_anisotropy_npz(result, path)

    saved = np.load(path, allow_pickle=False)
    assert saved['advanced_max_components'].item() == 2
    assert saved['advanced_selected_component_count'].item() == 2
    assert saved['advanced_selection_mode'].item() == 'fixed_k'
    assert saved['advanced_cross_k_bic_selection'].item() is False
    for candidate in comparison.candidates:
        prefix = f'advanced_candidate_{candidate.component_count}'
        np.testing.assert_allclose(
            saved[f'{prefix}_times_ns'],
            candidate.rotational_correlation_times_ns)
        np.testing.assert_allclose(
            saved[f'{prefix}_weights'], candidate.component_weights)
        assert saved[f'{prefix}_bic'].item() == pytest.approx(candidate.bic)
        assert saved[f'{prefix}_aicc'].item() == pytest.approx(candidate.aicc)
        assert saved[f'{prefix}_identifiable'].item() == candidate.identifiable
        np.testing.assert_array_equal(
            saved[f'{prefix}_component_bounds_ns'],
            candidate.component_bounds_ns)
        np.testing.assert_array_equal(
            saved[f'{prefix}_parallel_model'], candidate.parallel_model)
        np.testing.assert_allclose(
            saved[f'{prefix}_start_parameter_vectors'],
            np.stack([record.parameter_vector
                      for record in candidate.multistart_records]))
        np.testing.assert_allclose(
            saved[f'{prefix}_start_times_ns'],
            np.stack([record.rotational_correlation_times_ns
                      for record in candidate.multistart_records]))
        np.testing.assert_allclose(
            saved[f'{prefix}_start_weights'],
            np.stack([record.component_weights
                      for record in candidate.multistart_records]))
        np.testing.assert_allclose(
            saved[f'{prefix}_start_poisson_deviance'],
            [record.poisson_deviance for record in candidate.multistart_records])
        np.testing.assert_array_equal(
            saved[f'{prefix}_start_success'],
            [record.success for record in candidate.multistart_records])
        assert saved[f'{prefix}_start_messages'].dtype.kind in {'U', 'S'}
        assert len(saved[f'{prefix}_parameter_names']) == len(
            candidate.multistart_records[0].parameter_vector)
        assert saved[f'{prefix}_eligible_for_selection'].item() == (
            candidate.eligible_for_selection)
        assert saved[f'{prefix}_warnings'].dtype.kind in {'U', 'S'}
    assert saved['advanced_residual_type'].item() == 'signed_poisson_deviance'
    assert saved['parallel_irf_file'].item() == 'parallel_irf.ptu'
    assert saved['perpendicular_irf_file'].item() == 'perpendicular_irf.ptu'
    assert saved['repetition_period_ns'].item() == pytest.approx(12.8)
