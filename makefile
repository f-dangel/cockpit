.PHONY: help
.PHONY: black black-check flake8
.PHONY: test test-light examples
.PHONY: conda-env
.PHONY: black isort format
.PHONY: black-check isort-check format-check
.PHONY: flake8
.PHONY: pydocstyle-check
.PHONY: darglint-check
.PHONY: build-docs

.DEFAULT: help
help:
	@echo "test"
	@echo "        Run pytest on the project and report coverage"
	@echo "test-light"
	@echo "        Run pytest on 'small' tests and report coverage"
	@echo "examples"
	@echo "        Run examples"
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
	@echo "conda-env"
	@echo "        Create conda environment 'cockpit' with dev setup"
	@echo "build-docs"
	@echo "        Build the docs"

### TESTING ###
test:
	@pytest -vx --cov=cockpit --ignore=tests/test_deepobs tests

test-light:
	@pytest -vx --cov=cockpit --ignore=tests/test_deepobs tests

examples:
	@cd examples && python pytorch_mnist_minimal.py
	# @cd examples && python new_api.py
	# @cd examples && python pytorch_mnist.py
	# @cd examples && python deepobs_quadratic_deep.py


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
	@darglint --verbosity 2 cockpit

isort:
	@isort --apply

isort-check:
	@isort --check

format:
	@make black
	@make isort
	@make black-check

format-check: black-check isort-check pydocstyle-check darglint-check

### CONDA ###
conda-env:
	@conda env create --file .conda_env.yml

### DOCS ###
build-docs:
	@cd docs && make clean && make html
