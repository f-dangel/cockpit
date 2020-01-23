.PHONY: help
.PHONY: black black-check flake8
.PHONY: test
.PHONY: install conda-env

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
	@echo "conda-env"
	@echo "        Create conda environment 'backboard' with dev setup"

###
# Test
test:
	@pytest -vx --ignore=src .

###
# Linter and autoformatter

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

###
# Installation

install:
	@pip install -r requirements.txt
	@pip install .

###
# Conda environment
conda-env:
	@conda env create --file .conda_env.yml
