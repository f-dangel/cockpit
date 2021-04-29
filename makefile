.PHONY: help
.PHONY: black black-check flake8
.PHONY: test
.PHONY: conda-env
.PHONY: black isort format
.PHONY: black-check isort-check format-check, code-standard-check
.PHONY: flake8
.PHONY: pydocstyle-check
.PHONY: darglint-check
.PHONY: build-docs
.PHONY: clean-all

.DEFAULT: help
help:
	@echo "test"
	@echo "        Run pytest on the project and report coverage"
	@echo "black"
	@echo "        Run black on the project"
	@echo "black-check"
	@echo "        Check if black would change files"
	@echo "flake8"
	@echo "        Run flake8 on the project"
	@echo "pydocstyle-check"
	@echo "        Run pydocstyle on the project"
	@echo "darglint-check"
	@echo "        Run darglint on the project"
	@echo "code-standard-check"
	@echo "        Run all linters on the project to check quality standards."
	@echo "conda-env"
	@echo "        Create conda environment 'cockpit' with dev setup"
	@echo "build-docs"
	@echo "        Build the docs"
	@echo "clean-all"
	@echo "        Removes all unnecessary files."

### TESTING ###
# Run pytest with the Matplotlib backend agg to not show plots
test:
	@MPLBACKEND=agg pytest -vx --cov=cockpit .

### LINTING & FORMATTING ###

# Uses black.toml config instead of pyproject.toml to avoid pip issues. See
# - https://github.com/psf/black/issues/683
# - https://github.com/pypa/pip/pull/6370
# - https://pip.pypa.io/en/stable/reference/pip/#pep-517-and-518-support
black:
	@black . --config=black.toml

black-check:
	@black . --config=black.toml --check

flake8:
	@flake8 .

pydocstyle-check:
	@pydocstyle --count .

darglint-check:
	@darglint --verbosity 2 .

isort:
	@isort .

isort-check:
	@isort . --check

format:
	@make black
	@make isort
	@make black-check

format-check: black-check isort-check pydocstyle-check darglint-check

code-standard-check:
	@make black
	@make isort
	@make black-check
	@make flake8
	@make pydocstyle-check

### CONDA ###
conda-env:
	@conda env create --file .conda_env.yml

### DOCS ###
build-docs:
	@find . -type d -name "automod" -exec rm -r {} +
	@cd docs && make clean && make html

### CLEAN ###
clean-all:
	@find . -name '*.pyc' -delete
	@find . -name '*.pyo' -delete
	@find . -name '*~' -delete
	@find . -type d -name "__pycache__" -delete
	@rm -fr .pytest_cache/
	@rm -fr .benchmarks/