# Targets:
#   sync            install the locked dependencies into .venv
#   lab, lab-stop   start JupyterLab in the background, and stop it
#   lab-settings    open notebooks/*.py as notebooks in this project's JupyterLab (lab runs it)
#   render          execute notebooks/*.py into rendered/*.ipynb, where they are out of date
#   test            test-package, then test-notebooks
#   test-package    pytest on src/nnls_tutorial (tests/)
#   test-notebooks  render, then execute every rendered notebook under pytest
#   lint            ruff check and ruff format --check
#   clean           delete rendered/*.ipynb (they are committed: see below)
#
# Variables can be overridden on the command line, as CI does:
#   make -B test-notebooks CELL_TIMEOUT=60

NOTEBOOKS := $(wildcard notebooks/*.py)
RENDERED := $(patsubst notebooks/%.py,rendered/%.ipynb,$(NOTEBOOKS))

# Seconds any one cell may run when notebooks are executed. Every cell takes well under a
# second; this turns an occasional lost kernel message, which otherwise waits forever, into
# an error that names the cell.
CELL_TIMEOUT := 10

# Talk to kernels over Unix sockets, not unencrypted TCP. With IPC, Jupyter treats `ip` as a
# socket path; it must be absolute, because the kernel starts in a different directory from the
# process that connects to it. The sockets go in Jupyter's runtime directory, which is private
# to you, and which a fresh machine (such as a CI runner) doesn't have yet.
# These use `=`, not `:=`, so that uv runs only when a target uses them, not on every make.
# A Unix socket path can be at most 107 characters: a deeply nested JUPYTER_RUNTIME_DIR makes
# every kernel fail to start, with an error saying the ipc path is too long.
JUPYTER_RUNTIME = $(shell d="$$(uv run jupyter --runtime-dir)" && mkdir -p -m 700 "$$d" && echo "$$d")
KERNEL_IPC = --KernelManager.transport=ipc --KernelManager.ip=$(JUPYTER_RUNTIME)/kernel

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
	nohup uv run jupyter lab notebooks $(KERNEL_IPC) > $(LAB_LOG) 2>&1 & echo $$! > $(LAB_PID); \
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

# A notebook re-renders when its .py source or any package module is newer than its rendered
# copy; make -B render re-renders them all, as CI does. Every render changes the outputs (the
# timings in notebook 5 especially), so it leaves a diff even when no code changed.
render: $(RENDERED)

rendered/%.ipynb: notebooks/%.py $(wildcard src/nnls_tutorial/*.py)
	uv run jupytext --to ipynb --output $@ $<
	uv run jupyter nbconvert --to notebook --execute --inplace $(KERNEL_IPC) \
		--ExecutePreprocessor.timeout=$(CELL_TIMEOUT) $@

test: test-package test-notebooks

test-package:
	uv run pytest

# Renders first, so the notebooks are up to date with their .py sources, then executes each
# rendered notebook top to bottom. A notebook that has to re-render therefore runs twice.
test-notebooks: render
	uv run pytest --nbmake --nbmake-timeout=$(CELL_TIMEOUT) rendered/

lint:
	uv run ruff check src tests notebooks
	uv run ruff format --check src tests notebooks

# rendered/*.ipynb are committed, because they are what readers without a kernel see on
# GitHub. Bring them back with make render (new outputs) or git restore rendered/ (the
# committed ones).
clean:
	rm -f rendered/*.ipynb
