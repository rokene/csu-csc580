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

MILESTONE_1_FACE_PATH=$(MAKEFILE_DIR)/milestone-1
MILESTONE_1_FACE_APP=$(MILESTONE_1_FACE_PATH)/app.py

FACIAL_RECOGNITION_BASIC_PATH=$(MODULE1)
FACIAL_RECOGNITION_BASIC_APP=$(FACIAL_RECOGNITION_BASIC_PATH)/app.py

HANDWRITTEN_DIGITS_PATH=$(MODULE2)
HANDWRITTEN_DIGITS_APP=$(HANDWRITTEN_DIGITS_PATH)/app.py

FUEL_EFFICIENCY_PATH=$(MODULE3)
FUEL_EFFICIENCY_APP=$(FUEL_EFFICIENCY_PATH)/app.py

TOX21_PATH=$(MODULE4)
TOX21_APP=$(TOX21_PATH)/app.py

IRISRF_PATH=$(MODULE5)
IRISRF_APP=$(IRISRF_PATH)/app.py

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

.PHONY: m1-face-setup
m1-face-setup: ## setup dependencies and precursors for m1 face
	@echo "setting up dependencies and precursors for m1 face"
	@cd $(MILESTONE_1_FACE_PATH) && conda env create -f environment.yml

.PHONY: m1-face
m1-face: ## executes m1-face
	@echo "starting m1-faceapp"
	@cd $(MILESTONE_1_FACE_PATH) && \
		python $(MILESTONE_1_FACE_APP)
	@echo "completed m1-face app"

.PHONY: irisrf-setup
irisrf-setup: ## setup dependencies and precursors for irisrf
	@echo "setting up dependencies and precursors for irisrf"
	@cd $(IRISRF_PATH) && conda env create -f environment.yml

.PHONY: irisrf
irisrf: ## executes irisrf
	@echo "starting irisrf app"
	@cd $(IRISRF_PATH) && \
		python $(IRISRF_APP) --mode $(MODE)
	@echo "completed irisrf app"

.PHONY: tox21-setup
tox21-setup: ## setup dependencies and precursors for tox21
	@echo "setting up dependencies and precursors for tox21"
	@cd $(TOX21_PATH) && conda env create -f environment.yml

.PHONY: tox21
tox21: ## executes tox21
	@echo "starting tox21 app"
	@cd $(TOX21_PATH) && \
		python $(TOX21_APP) --mode $(MODE) $(if $(MODEL),--model $(MODEL))
	@echo "completed tox21 app"

.PHONY: tox21-tensorboard
tox21-tensorboard: ## executes tox21 tensorboard
	@echo "starting tox21 tensorboard"
	@cd $(TOX21_PATH) && \
		tensorboard --logdir=logs/fcnet-tox21-tf2

.PHONY: basic-fuel-efficency-setup
basic-fuel-efficency-setup: ## setup dependencies and precursors for the basic fuel efficiency app
	@echo "setting up dependencies and precursors for the basic fuel efficiency app"
	@cd $(FUEL_EFFICIENCY_PATH) && conda env create -f environment.yml

.PHONY: basic-fuel-efficency
basic-fuel-efficency: ## executes the basic fuel efficiency app
	@echo "starting the basic fuel efficiency app"
	@cd $(FUEL_EFFICIENCY_PATH) && \
		python $(FUEL_EFFICIENCY_APP) --mode $(MODE) $(if $(MODEL),--model $(MODEL))
	@echo "completed the basic fuel efficiency app"

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

.PHONY: handwritten-digits-ml-app-setup
handwritten-digits-ml-app-setup: ## setup dependencies and precursors for the basic hand written analog to digital numbers
	@echo "setting up dependencies and precursors for the hand written analog to digital numbers"
	@cd $(HANDWRITTEN_DIGITS_PATH) && conda env create -f environment.yml

.PHONY: handwritten-digits-ml-app-train
handwritten-digits-ml-app-train: ## executes the basic hand written analog to digital numbers training
	@echo "starting the hand written analog to digital numbers"
	@cd $(HANDWRITTEN_DIGITS_PATH) && \
		python $(HANDWRITTEN_DIGITS_APP) --mode train
	@echo "completed the hand written analog to digital numbers"

.PHONY: handwritten-digits-ml-app-infer
handwritten-digits-ml-app-infer: ## executes the basic hand written analog to digital numbers inference
	@echo "starting the hand written analog to digital numbers"
	@cd $(HANDWRITTEN_DIGITS_PATH) && \
		python $(HANDWRITTEN_DIGITS_APP) --mode infer
	@echo "completed the hand written analog to digital numbers"

.PHONY: handwritten-digits-ml-app-lrtest
handwritten-digits-ml-app-lrtest: ## executes the learning rate test in basic hand written analog to digital numbers test learning rates
	@echo "starting learning rate test the hand written analog to digital numbers"
	@cd $(HANDWRITTEN_DIGITS_PATH) && \
		python $(HANDWRITTEN_DIGITS_APP) --mode lrtest
	@echo "completed the learning rate test hand written analog to digital numbers"

.PHONY: handwritten-digits-ml-app-ltest
handwritten-digits-ml-app-ltest: ## executes the layers test in basic hand written analog to digital numbers test learning rates
	@echo "starting layers test the hand written analog to digital numbers"
	@cd $(HANDWRITTEN_DIGITS_PATH) && \
		python $(HANDWRITTEN_DIGITS_APP) --mode layerstest
	@echo "completed the layers test hand written analog to digital numbers"

.PHONY: handwritten-digits-ml-app-btest
handwritten-digits-ml-app-btest: ## executes the batch size test in basic hand written analog to digital numbers test learning rates
	@echo "starting batch size test the hand written analog to digital numbers"
	@cd $(HANDWRITTEN_DIGITS_PATH) && \
		python $(HANDWRITTEN_DIGITS_APP) --mode batchtest
	@echo "completed the batch size test hand written analog to digital numbers"

