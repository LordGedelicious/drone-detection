PY ?= python
SPLIT_MANIFEST ?= splits/seed42_48-6-6.json
MODEL ?= baseline
EPOCHS ?= 50
SPLIT ?= test
ASSIGN ?= neighbors
WEIGHTS ?= checkpoints/$(MODEL)_best.pth

.PHONY: help split train finetune eval infer bakeoff-screen bakeoff-full docker-build docker-shell clean

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

split:  ## print + freeze the scene split to a manifest
	$(PY) -m src.core.split --write $(SPLIT_MANIFEST)

train:  ## train one model   (MODEL=baseline|fpn|p2  EPOCHS=50)
	$(PY) train.py --model $(MODEL) --epochs $(EPOCHS) --split-manifest $(SPLIT_MANIFEST)

finetune:  ## fine-tune from a checkpoint   (WEIGHTS=... )
	$(PY) finetune.py --weights $(WEIGHTS) --split-manifest $(SPLIT_MANIFEST)

eval:  ## evaluate a checkpoint   (WEIGHTS=...  SPLIT=val|test)
	$(PY) eval.py --weights $(WEIGHTS) --split $(SPLIT) --profile --split-manifest $(SPLIT_MANIFEST)

infer:  ## run inference   (WEIGHTS=...  SOURCE=path)
	$(PY) infer.py --weights $(WEIGHTS) --source $(SOURCE)

bakeoff-screen:  ## short screening run for baseline/fpn/p2, then compare
	bash scripts/run_bakeoff.sh screen $(SPLIT_MANIFEST)

bakeoff-full:  ## full run for one model, eval on test   (MODEL=...  ASSIGN=neighbors|single)
	bash scripts/run_bakeoff.sh full $(SPLIT_MANIFEST) $(MODEL) $(ASSIGN)

docker-build:  ## build the container image
	docker compose build

docker-shell:  ## interactive shell in the container
	docker compose run --rm --entrypoint bash trainer

clean:  ## remove caches and inference outputs
	rm -rf runs __pycache__ src/*/__pycache__ src/**/__pycache__
