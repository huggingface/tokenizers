.PHONY: bundle job jobs fetch

FLAVOR  ?= cpu-performance
TIMEOUT ?= 120m
IMAGE   := rust:1.93.0-bookworm
OVERRIDES := $(foreach v,MAIN ROUNDS SECS,$(if $($(v)),-e $(v)=$($(v))))

bundle:
	python3 bundle.py

job: bundle
	hf jobs run --detach --label app=measure --label bench=added-token-matcher --flavor $(FLAVOR) --timeout $(TIMEOUT) \
	  -e FLAVOR=$(FLAVOR) $(OVERRIDES) $(IMAGE) bash -c "$$(cat start_job.sh)"

# two independent hosts of the same flavor
jobs:
	$(MAKE) job
	$(MAKE) job

fetch:
	python3 analyze.py $(ID)
