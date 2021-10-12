PY = python3
VENV = venv
BIN = $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help predict train coverage lint clean fclean


help:
	@echo " 	make venv"
	@echo "		build virtual environment and install dependencies."
	@echo "	make coverage"
	@echo "		run pytest with coverage."
	@echo "	make lint"
	@echo "		run flake8."
	@echo "	make clean"
	@echo "		clean thetas.csv."
	@echo "	make fclean"
	@echo "		clean build files."
	@echo "	make predict"
	@echo "		run a prediction with current thetas."
	@echo "	make train [FLAGS]"
	@echo "		- FLAGS='train.py flags (see make train FLAGS='-h')'"
	@echo "		- eg: make train FLAGS='-v -lr 0.001 -i 1000 -m=0.5 -s normalize'"
	@echo "		train the model."

$(VENV): setup.py
	$(PY) -m venv $(VENV)
	$(BIN)/pip install wheel
	$(BIN)/pip install -e .[dev]
	touch $(VENV)

coverage: $(VENV)
	$(BIN)/coverage erase
	$(BIN)/coverage run --include=linear_regression/* -m pytest
	$(BIN)/coverage report -m

lint: $(VENV)
	$(BIN)/flake8 linear_regression tests setup.py

clean:
	@rm -rf data/thetas.csv
	@echo "clean: removed thetas.csv"

fclean: clean
	@find . -type f -name *.pyc -delete
	@find . -type d -name __pycache__ -delete
	@rm -rf $(VENV)
	@rm -rf .pytest_cache
	@rm -rf .coverage
	@rm -rf .eggs
	@rm -rf linear_regression.egg-info
	@echo "fclean: removed build files"

predict: $(VENV)
	$(BIN)/python linear_regression/predict.py

train: $(VENV)
	$(BIN)/python linear_regression/train.py $(FLAGS)
