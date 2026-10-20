# How non-negative least squares works

[![CI](https://github.com/jklynch/mr-fitty-nnls-tutorial/actions/workflows/ci.yml/badge.svg)](https://github.com/jklynch/mr-fitty-nnls-tutorial/actions/workflows/ci.yml)

A tutorial, in Jupyter notebooks, on non-negative least squares (NNLS): find the weights
$w \ge 0$ that make $Rw$ as close as possible to $s$. It is written for undergraduate math
majors who have taken a first course in linear algebra, and it is motivated by a real problem:
working out which reference X-ray spectra, mixed in which proportions, explain a measured one.
That is what [MrFitty](https://github.com/jklynch/mr-fitty) does, millions of times per search,
and the last notebook builds the fast solver it uses.

| Notebook | What it covers |
|---|---|
| [1. The geometry of least squares](rendered/01_least_squares_geometry.ipynb) | LS as projection onto a subspace; NNLS as projection onto a cone |
| [2. Optimality and enumeration](rendered/02_optimality_and_enumeration.ipynb) | Weight space, the gradient and the KKT conditions; why trying every support is exact |
| [3. The active-set method](rendered/03_active_set_lawson_hanson.ipynb) | Lawson–Hanson's walk from support to support, why it stops at the answer, and what it costs against enumeration |
| [4. Cholesky and conditioning](rendered/04_numerics_cholesky_conditioning.ipynb) | Floating point, condition numbers and why $R^\top R$ squares them; Cholesky derived entry by entry; one check that covers every support |
| [5. Many small fits, fast](rendered/05_many_small_fits_fast.ipynb) | Cross-validation's hundreds of thousands of fits: a SciPy loop, batched NumPy, then compiled Numba, timed live and checked against SciPy |

The links above go to executed copies with every figure in place. To run them yourself, with
the interactive widgets:

```bash
uv sync
make lab
```

`make lab` configures this project's JupyterLab so that double-clicking a `notebooks/*.py` file
opens it as a notebook. (If you start JupyterLab some other way, right-click the file and choose
*Open With → Jupytext Notebook*.) [Jupytext](https://jupytext.readthedocs.io) keeps the `.py`
(committed) and a paired `.ipynb` (git-ignored) in step.

## Development

```bash
make test         # package tests against SciPy, then every notebook executed
make render       # rebuild rendered/*.ipynb from notebooks/*.py
make lint
uv run pre-commit install
```

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs `make lint`,
`make test-package`, and every notebook rendered and executed from its `.py` source, on each
push to `main` and each pull request.

### The tests

`make test` runs two kinds of test. The package tests in `tests/` check the code in
`src/nnls_tutorial/` and run in under a second. Executing every notebook covers the plotting and
widget modules, which have no unit tests of their own.

The package tests fall into four groups:

1. **Solvers give the right answer** (`test_solvers.py`).
   - Each of the three NNLS solvers (enumeration, enumeration through $G$ and $c$, and the active
     set) is compared with `scipy.optimize.nnls` on 50 random problems. Each answer is also
     checked against the KKT conditions, which prove it correct without trusting SciPy.
   - All three are tested on a matrix with a duplicate column, where only the fit is unique.
   - On the bundled spectra, enumeration reproduces MrFitty's published best fit, and the active
     set matches enumeration on all ten references.
   - `nnls_enumerate_many` matches enumerating one target at a time.
2. **The building blocks** (`test_solvers.py`): `supports`, `candidates` with dependent supports,
   `support_fit`, the quadratic expansion of $f$ (which checks the gradient too), Cholesky, the
   triangular solves, the least-squares routes, and the cone-projection identity from
   `geometry.py`.
3. **The theorems, checked numerically.**
   - The active-set walk goes downhill and never repeats a support (Theorems 14 and 15).
   - Notebook 4's results (`test_conditioning.py`): $\kappa(R^\top R) = \kappa(R)^2$, the trace
     bound, no support worse conditioned than the whole, $\kappa = \cot(\theta/2)$ for a unit
     pair, and the normal equations losing about twice QR's digits.
4. **The fast versions agree with the reference** (`test_batched.py`).
   - The batched and compiled solvers match one SciPy call per fit on 40 random problems with
     holdouts.
   - The two Gram-matrix routes agree, and batched Cholesky matches NumPy.
   - The two condition checks bracket each other, and draws that fail the check go to SciPy.
   - The compiled search matches the SciPy search on the bundled spectra.

Tests over many random problems loop over their seeds inside one test, and a failure names the
seed. The random problems come from `tests/problems.py`.

See [SPEC.md](SPEC.md) for the plan of the whole series.
