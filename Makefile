.PHONY = help train predict clean

VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

.DEFAULT = help
help:
	@echo "---------------HELP-----------------"
	@echo "To setup the project type make setup"
	@echo "To run the prediction program type make predict"
	@echo "To run the training program type make train"
	@echo "To clean the project type make clean"
	@echo "------------------------------------"

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

train: $(VENV)/bin/activate
ifeq ($(PLOT),y)
	$(PYTHON) src/train.py -v
else
	$(PYTHON) src/train.py
endif

predict: $(VENV)/bin/activate
	$(PYTHON) src/predict.py

clean:
	rm -f data/thetas.csv
	rm -rf __pycache__
	rm -rf venv
