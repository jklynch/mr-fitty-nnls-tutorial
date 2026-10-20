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
| 2. Optimality and enumeration | *coming* |
| 3. The active-set method | *coming* |
| 4. Cholesky and conditioning | *coming* |
| 5. Many small fits, fast | *coming* |

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
