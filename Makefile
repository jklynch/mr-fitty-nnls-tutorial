NOTEBOOKS := $(wildcard notebooks/*.py)
RENDERED := $(patsubst notebooks/%.py,rendered/%.ipynb,$(NOTEBOOKS))

# Seconds any one cell may run when notebooks are executed. Every cell takes well under a
# second; this turns an occasional lost kernel message, which otherwise waits forever, into
# an error that names the cell.
CELL_TIMEOUT := 10

.PHONY: sync lab lab-stop lab-settings render test test-package test-notebooks lint clean

sync:
	uv sync

# JupyterLab runs in the background; its output goes to $(LAB_LOG) and `make lab-stop` stops it.
LAB_LOG := .jupyter-lab.log
LAB_PID := .jupyter-lab.pid

# open the .py notebooks as notebooks; jupytext keeps the paired .ipynb in step.
# overrides.json makes "Jupytext Notebook" the default viewer for .py files in this
# environment's JupyterLab only; it is installed into the environment, not your home directory.
lab: lab-settings
	@if [ -f $(LAB_PID) ] && kill -0 "$$(cat $(LAB_PID))" 2>/dev/null; then \
		echo "JupyterLab is already running (pid $$(cat $(LAB_PID))); see $(LAB_LOG)"; \
		exit 0; \
	fi; \
	nohup uv run jupyter lab notebooks > $(LAB_LOG) 2>&1 & echo $$! > $(LAB_PID); \
	for i in $$(seq 30); do \
		url=$$(grep -o 'http://127\.0\.0\.1:[0-9]*/lab?token=[0-9a-f]*' $(LAB_LOG) | head -1); \
		if [ -n "$$url" ]; then echo "JupyterLab: $$url"; exit 0; fi; \
		sleep 1; \
	done; \
	echo "JupyterLab started (pid $$(cat $(LAB_PID))) but printed no URL yet; see $(LAB_LOG)"

# uv run passes the signal on to jupyter
lab-stop:
	@if [ -f $(LAB_PID) ] && kill "$$(cat $(LAB_PID))" 2>/dev/null; then \
		echo "stopped JupyterLab (pid $$(cat $(LAB_PID)))"; \
	else \
		echo "JupyterLab is not running"; \
	fi; \
	rm -f $(LAB_PID)

lab-settings:
	uv sync
	mkdir -p "$$(uv run python -c 'import sys; print(sys.prefix)')/share/jupyter/lab/settings"
	cp jupyter/overrides.json "$$(uv run python -c 'import sys; print(sys.prefix)')/share/jupyter/lab/settings/overrides.json"

render: $(RENDERED)

rendered/%.ipynb: notebooks/%.py $(wildcard src/nnls_tutorial/*.py)
	uv run jupytext --to ipynb --output $@ $<
	uv run jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=$(CELL_TIMEOUT) $@

test: test-package test-notebooks

test-package:
	uv run pytest

# executes the rendered notebooks top to bottom
test-notebooks: render
	uv run pytest --nbmake --nbmake-timeout=$(CELL_TIMEOUT) rendered/

lint:
	uv run ruff check src tests notebooks
	uv run ruff format --check src tests notebooks

clean:
	rm -f rendered/*.ipynb
