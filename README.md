# flimkit-anisotropy

Time-resolved fluorescence anisotropy analysis, as a [FLIMKit](https://github.com/FLIMKit/FLIMKit) add-on.

Written by Zhen Yuan. The history here is the original commits from [FLIMKit#35](https://github.com/FLIMKit/FLIMKit/pull/35), extracted rather than copied, so authorship is preserved.

## What it does

The tool takes matched sequential parallel and perpendicular PTU acquisitions and offers two methods:

- A direct r(t) diagnostic, giving background-corrected anisotropy traces and registered, neighbourhood-pooled maps. It is a diagnostic only: division and IRF convolution do not commute, so it is not used to estimate rotational correlation times.
- A global polarized-decay fit, following the channel equations and direct global analysis in Lakowicz, chapter 11, section 11.2.2. It jointly models the raw parallel and perpendicular decays with separate measured IRFs, a fixed known fluorescence lifetime, fixed G and exposure factors, periodic excitation, one rotational correlation time, a common IRF timing shift, and separate channel backgrounds.

Also supported: explicit parallel and perpendicular file selection, optional subpixel registration between acquisitions, spatial pooling with photon thresholds and validity masks, Poisson-deviance residual fitting with parameter-bound warnings, and CSV and NPZ export with provenance.

The fit is deliberately constrained. G = 1 is an assumption unless channel sensitivity has been calibrated, the fitted r(0) is the resolved time-zero anisotropy and may not equal the fundamental anisotropy, and a bound-hit result is reported as constrained rather than accepted. Multiple lifetimes, multiple rotational components, per-pixel rotational fitting and calibrated confidence intervals are not implemented. The rotational correlation time has not yet been validated against a known standard.

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
