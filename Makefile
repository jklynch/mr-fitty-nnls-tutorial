NOTEBOOKS := $(wildcard notebooks/*.py)
RENDERED := $(patsubst notebooks/%.py,rendered/%.ipynb,$(NOTEBOOKS))

.PHONY: sync lab lab-settings render test test-package test-notebooks lint clean

sync:
	uv sync

# open the .py notebooks as notebooks; jupytext keeps the paired .ipynb in step.
# overrides.json makes "Jupytext Notebook" the default viewer for .py files in this
# environment's JupyterLab only; it is installed into the environment, not your home directory.
lab: lab-settings
	uv run jupyter lab notebooks

lab-settings:
	uv sync
	mkdir -p "$$(uv run python -c 'import sys; print(sys.prefix)')/share/jupyter/lab/settings"
	cp jupyter/overrides.json "$$(uv run python -c 'import sys; print(sys.prefix)')/share/jupyter/lab/settings/overrides.json"

render: $(RENDERED)

rendered/%.ipynb: notebooks/%.py $(wildcard src/nnls_tutorial/*.py)
	uv run jupytext --to ipynb --output $@ $<
	uv run jupyter nbconvert --to notebook --execute --inplace $@

test: test-package test-notebooks

test-package:
	uv run pytest

# executes the rendered notebooks top to bottom
test-notebooks: render
	uv run pytest --nbmake rendered/

lint:
	uv run ruff check src tests notebooks
	uv run ruff format --check src tests notebooks

clean:
	rm -f rendered/*.ipynb
