# flimkit-anisotropy Documentation

Time-resolved fluorescence anisotropy analysis, as a [FLIMKit](https://github.com/FLIMKit/FLIMKit) add-on.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Methods](#methods)
4. [Inputs](#inputs)
5. [Outputs](#outputs)
6. [Scope and Limitations](#scope-and-limitations)
7. [Development](#development)

---

## Overview

Fluorescence anisotropy reports how far a fluorophore rotates between absorbing and emitting a photon. Measured against time, it separates rotational motion from fluorescence decay, which gives access to the rotational correlation time and through it to local viscosity, molecular volume and binding state.

This package adds that analysis to FLIMKit as an add-on. It registers a `Tools > Time-Resolved Anisotropy...` entry through the `flimkit.plugins` registry and takes matched sequential parallel and perpendicular PTU acquisitions.

Written by Zhen Yuan Yeo. The commit history here is the original from [FLIMKit#35](https://github.com/FLIMKit/FLIMKit/pull/35), extracted rather than copied.

---

## Installation

Requires FLIMKit 0.10.0 or newer, which is the release that added the add-on system.

```bash
pip install git+https://github.com/FLIMKit/flimkit-anisotropy
```

It registers through the `flimkit.plugins` entry point, so the menu entry appears the next time FLIMKit starts. `Help > Plugins...` in FLIMKit reports whether it loaded and what it registered.

---

## Methods

### Direct r(t) diagnostic

Background-corrected anisotropy traces and registered, neighbourhood-pooled maps.

This is a diagnostic only. Division and IRF convolution do not commute, so the trace is not used to estimate a rotational correlation time.

### Global polarized-decay fit

The preferred method, following the channel equations and direct global analysis in Lakowicz, chapter 11, section 11.2.2.

It jointly models the raw parallel and perpendicular photon decays with separate measured IRFs, a fixed known fluorescence lifetime, fixed G and relative exposure factors, periodic excitation, one rotational correlation time, a common IRF timing shift, and separate channel backgrounds. Residuals are fitted on Poisson deviance, and a parameter that reaches its bound is reported rather than accepted silently.

---

## Inputs

- A parallel and a perpendicular PTU acquisition, selected explicitly rather than inferred.
- A separate measured IRF for each polarisation channel, as a LAS X export.
- A fluorescence lifetime from an independent analysis, which the fit holds fixed.
- G and relative exposure factors, which the fit also holds fixed.

Optional: subpixel registration between the two acquisitions, spatial pooling, photon thresholds and validity masks.

---

## Outputs

CSV and NPZ export carrying the fit results, the polarized models and residuals, relative time, and provenance for the inputs that produced them.

---

## Scope and Limitations

The fit is deliberately constrained, and the constraints matter when reading a result:

- G = 1 is an assumption unless channel sensitivity has been calibrated.
- The fitted r(0) is the resolved time-zero anisotropy and may not equal the fundamental anisotropy.
- A bound-hit result is reported as constrained or model-incompatible.
- The rotational correlation time has not yet been validated against a known standard.

Not implemented: multiple lifetimes, multiple rotational components, per-pixel rotational fitting, calibrated confidence intervals.

---

## Development

```bash
pip install -e '.[test]'
pytest
```

`test_global_fit_mode_draws_polarized_models_and_residuals` asserts pixel positions in a matplotlib layout and can fail on a machine whose font metrics differ from the one it was written on.

Documentation lives in `Docs/documentation.md` and is published to the wiki by `.github/workflows/wiki.yml` on push to `main`. Do not edit the wiki pages directly, they are overwritten on the next sync.

The wiki needs one page created by hand before the first sync can run, because GitHub does not give a wiki a git repository until it has a page and will not create one by push. Any content will do, since the sync overwrites it.
