# CONFIG ######################################################################

.DEFAULT_GOAL := help

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PP=$(MAKEFILE_DIR)/portfolio-project
PP_APP=app.py

MODULE1=$(MAKEFILE_DIR)/module-1
MODULE2=$(MAKEFILE_DIR)/module-2
MODULE3=$(MAKEFILE_DIR)/module-3
MODULE4=$(MAKEFILE_DIR)/module-4
MODULE5=$(MAKEFILE_DIR)/module-5
MODULE6=$(MAKEFILE_DIR)/module-6
MODULE7=$(MAKEFILE_DIR)/module-7
MODULE8=$(MAKEFILE_DIR)/module-8

FACIAL_RECOGNITION_BASIC_PATH = $(MODULE1)
FACIAL_RECOGNITION_BASIC_APP=$(MODULE1)/app.py

# PYTHON CONFIG ###############################################################

# ubuntu

# PYTHON_CONFIG=python3
# PYTHON_PIP_CONFIG=pip
# VNV_ACTIVATE=venv/bin/activate

# windows

PYTHON_CONFIG=python.exe
PYTHON_PIP_CONFIG=python.exe -m pip
VNV_ACTIVATE=venv/Scripts/activate

# TARGETS #####################################################################

.PHONY: help
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*' $(MAKEFILE_LIST) | sort

.PHONY: pp-setup
pp-setup: ## setup dependencies and precursors for portfolio project
	@echo "pp: setting up portfolio project virtual env"
	@cd $(PP) && $(PYTHON_CONFIG) -m venv venv && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_PIP_CONFIG) install --upgrade pip && \
		$(PYTHON_PIP_CONFIG) install -r requirements.txt

.PHONY: pp
pp-draw: ## executes portfolio project Annotation Draw
	@echo "pp: starting portfolio project annotation drawing"
	@cd $(PP) && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_CONFIG) $(PP)/$(PP_APP)
	@echo "pp: completed portfolio project annotation drawing"

.PHONY: basic-facial-app-setup
basic-facial-app-setup: ## setup dependencies and precursors for the basic facial recognition app
	@echo "setting up dependencies and precursors for the basic facial recognition app"
	@cd $(FACIAL_RECOGNITION_BASIC_PATH) && conda env create -f environment.yml

.PHONY: basic-facial-app
basic-facial-app: ## executes the basic facial recognition app
	@echo "starting the basic facial recognition app"
	@cd $(FACIAL_RECOGNITION_BASIC_PATH) && \
		python $(FACIAL_RECOGNITION_BASIC_APP)
	@echo "completed the basic facial recognition app"
