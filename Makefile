#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = tc5061-proyecto-equipo37
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

# Prueba de carga; se sobrescriben así: make loadtest LOAD_USERS=150 LOAD_TIME=5m
LOCUST ?= locust
LOAD_HOST ?= http://localhost:8000
LOAD_USERS ?= 50
LOAD_RATE ?= $(shell r=$$(($(LOAD_USERS) / 10)); echo $$((r > 0 ? r : 1)))
LOAD_TIME ?= 5m
LOAD_OUT ?= reports/load_test/local

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) MiBici/dataset.py


## Run Locust load test against the API and record docker stats
.PHONY: loadtest
loadtest:
	@mkdir -p "$(LOAD_OUT)"
	@curl -fsS -m 5 "$(LOAD_HOST)/health" > /dev/null || \
		{ echo "La API no responde en $(LOAD_HOST). Ejecuta: docker compose up -d --build --wait"; exit 1; }
	@while true; do \
		docker stats --no-stream \
			--format "$$(date +%T),{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}" \
			$$(docker compose ps -q); \
	done > "$(LOAD_OUT)/stats_$(LOAD_USERS)u.csv" & \
	STATS_PID=$$!; \
	trap 'kill $$STATS_PID 2>/dev/null' EXIT; \
	$(LOCUST) -f tests/load/locustfile.py --headless --host "$(LOAD_HOST)" \
		-u $(LOAD_USERS) -r $(LOAD_RATE) -t $(LOAD_TIME) --only-summary \
		--csv "$(LOAD_OUT)/run_$(LOAD_USERS)u" --csv-full-history \
		--html "$(LOAD_OUT)/run_$(LOAD_USERS)u.html"


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
