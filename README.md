# How non-negative least squares works

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

See [SPEC.md](SPEC.md) for the plan of the whole series.
