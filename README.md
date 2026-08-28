# flimkit-anisotropy

[![DOI](https://zenodo.org/badge/1331757585.svg)](https://doi.org/10.5281/zenodo.21918921)

Time-resolved fluorescence anisotropy analysis, as a [FLIMKit](https://github.com/FLIMKit/FLIMKit) add-on.

Written by Zhen Yuan Yeo. The history here is the original commits from [FLIMKit#35](https://github.com/FLIMKit/FLIMKit/pull/35), extracted rather than copied, so authorship is preserved.

## What it does

The tool takes matched sequential parallel and perpendicular PTU acquisitions and offers three methods:

- A direct r(t) diagnostic, giving background-corrected anisotropy traces and registered, neighbourhood-pooled maps. It is a diagnostic only: division and IRF convolution do not commute, so it is not used to estimate rotational correlation times.
- A global polarized-decay fit, following the channel equations and direct global analysis in Lakowicz, chapter 11, section 11.2.2. It jointly models the raw parallel and perpendicular decays with separate measured IRFs loaded directly from PicoQuant PTU files or from supported spreadsheet/text exports, a fixed known fluorescence lifetime, one shared polarization-channel scale and exposure factors, periodic excitation, one rotational correlation time, a common IRF timing shift, and separate channel backgrounds.
- An advanced one-to-three-component polarized-decay fit. It compares ordered rotational-correlation models under the same raw-count, separate-IRF model. Component weights are non-negative and sum to one. The user chooses the largest model to compare, from one to three components, and can use automatic bounds or one positive shared manual lower/upper range applied to every candidate.

When PTU IRFs are selected, the plugin sums their photon histograms over the recording and uses one laser period. It rejects an IRF PTU when its TCSPC timing resolution or laser period does not match the sample PTU. Small trailing-bin differences caused by period rounding are cropped or zero-padded.

Also supported: explicit parallel and perpendicular file selection, optional subpixel registration between acquisitions, spatial pooling with photon thresholds and validity masks, Poisson-deviance residual fitting with parameter-bound warnings, a late-window scale-stability diagnostic, and CSV and NPZ export with provenance.

## Shared polarization-channel scale

One shared scale is applied to the entire image and to the direct, preferred global, and advanced fits. Its source must be labelled as one of:

- **Assumed scale**: a user-supplied value without calibration evidence. `G = 1` is explicitly warned as an uncalibrated assumption.
- **Calibrated G**: a user-supplied value that the user declares was measured independently. The software records this declaration but cannot verify it.
- **Effective late-window scale**: estimated from exposure-normalized, full-FOV photon sums from the chosen late-window start to the end of the laser period.

The **Scale diagnostic...** plot shows a short rolling parallel/perpendicular ratio, nested ratios to the period end, and an approximate Poisson 95% interval from the delta method. At low counts this interval can extend below zero and its coverage is not reliable. Nested windows share photons and are therefore correlated. The late-window calculation uses raw counts without background subtraction and retains previous-pulse fluorescence. It assumes that anisotropy approaches zero in the selected tail. Its result is an effective channel scale, **not** a calibrated physical G-factor. Continued drift means that the selected window is not a stable plateau.

The G input defaults to `1`. In advanced mode, the interface tells users to first use the effective late-window scale feature for a better G guess when no independent calibration is available. That late-window value remains an effective scale, not calibrated physical G.

## Advanced one-to-three-component fit

The advanced anisotropy model is

```text
r(t) = r(0) * sum_j w_j exp(-t / theta_j)
```

with positive, strictly ordered correlation times, non-negative component weights, and weights that sum to one. Every candidate is fitted from 1 to 32 deterministic starts. The tool reports Poisson deviance, AICc, BIC, component times, weights, effective bounds, optimizer status, fitted backgrounds, timing shift, curves, and signed Poisson-deviance residuals. BIC selects the simplest candidate within 2 BIC units of the minimum. Optional manual mode applies one shared lower/upper correlation-time range to every candidate, so model comparison uses consistent bounds. CSV and NPZ exports preserve every start's latent parameters, physical times and weights, deviance, convergence state, message, and bound hits, plus IRF filenames and fit provenance. The bounded softmax uses finite latent limits, so weights approach zero but do not become exactly zero.

No advanced candidate is labelled physically resolved in this release. Profile likelihoods or equivalent two-sided uncertainty checks, bound perturbations, and broader replicate validation are not yet implemented. The main result therefore reports BIC and numerical adequacy warnings without showing component times or weights. Numerical optimizer values are available only in **Fit details...**, labelled as optimizer values rather than physical estimates. Additional warnings flag Poisson deviance per degree of freedom above 2, fitted bounds, weights below 0.05, adjacent times separated by less than 1.5-fold, IRF/window limits, multistart disagreement, and preference for a simpler model. The deviance threshold is a conservative screen, not a calibrated goodness-of-fit test. Three-component values remain exploratory.

The fits remain deliberately constrained. An effective or assumed scale is not calibration, fitted r(0) is a time-zero model parameter and may not equal the fundamental anisotropy, and a bound-hit result is rejected rather than accepted. Multiple fluorescence lifetimes, per-pixel rotational fitting, and calibrated confidence intervals are not implemented. Rotational correlation times have not yet been validated against a known standard.

## Installing

Needs FLIMKit 0.10.0 or newer, which is the release that added the add-on system.

```bash
pip install git+https://github.com/FLIMKit/flimkit-anisotropy
```

It registers through the `flimkit.plugins` entry point, so it appears as `Tools > Time-Resolved Anisotropy...` the next time FLIMKit starts. `Help > Plugins...` shows whether it loaded.

## Tests

```bash
pip install -e '.[test]'
pytest
```

`test_global_fit_mode_draws_polarized_models_and_residuals` asserts pixel positions in the matplotlib layout and can fail on a machine whose font metrics differ from the one it was written on.

## Acknowledgements

Zhen Yuan developed the scientific implementation with assistance from OpenAI's GPT-5.6 Sol, operated through Hermes Agent by Nous Research. This assistance was used to translate the equations and methodological ideas described in Lakowicz's textbook into software, and to support the development of tests, the graphical interface, tooltips and documentation. Zhen Yuan directed and reviewed this work and remains responsible for the final implementation and scientific interpretation.

Alexander Hunt preserved the original development history, adapted the project to the FLIMKit plugin system, and contributed the package structure, continuous integration, documentation workflow and release automation.

## Licence

MIT, same as FLIMKit.
