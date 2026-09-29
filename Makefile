RUN_DIR ?= results/compact_v4
MODEL ?= gpt-5.5
BACKEND ?= sdk
ATTEMPTS ?= 6
EVAL_ATTEMPTS ?= 3
RESUME ?= false

ifeq ($(RESUME),true)
TRAIN_FLAGS := -Resume
endif

ifeq ($(KB),true)
EVAL_MODE := kb
else ifeq ($(KB),false)
EVAL_MODE := baseline
else
EVAL_MODE := both
endif

.PHONY: train eval a

train:
	powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1 -Phase train -RunDir "$(RUN_DIR)" -Model "$(MODEL)" -Backend "$(BACKEND)" -Attempts $(ATTEMPTS) $(TRAIN_FLAGS)

eval:
	powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1 -Phase eval -RunDir "$(RUN_DIR)" -EvalMode "$(EVAL_MODE)" -EvalAttempts $(EVAL_ATTEMPTS)

# Allows the earlier spelling `make train a`.
a:
	@echo Set A is the training set.
