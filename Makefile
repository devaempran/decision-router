# Shortcuts for bash users (the PowerShell equivalent is pipeline/demo/start_demo.ps1).
.PHONY: help demo up down logs redis demo-mock demo-gpu publish test lint

help:
	@echo "make demo       Docker only, no GPU: mock model + pipeline + dashboard + demo tickets -> http://127.0.0.1:5055"
	@echo "make up         the same stack without the demo publisher (write your own events)"
	@echo "make down       stop and remove the containers"
	@echo "make demo-mock  no Docker for the app (Redis still runs in Docker): mock model, native Python"
	@echo "make demo-gpu   real Qwen3 model on this machine's GPU (needs the service set up, see service/README.md)"
	@echo "make publish    publish the demo tickets (after demo-mock or demo-gpu is running)"
	@echo "make test lint  run the service and pipeline tests / ruff"

demo:
	docker compose --profile demo up --build

up:
	docker compose up --build -d

down:
	docker compose --profile demo down

logs:
	docker compose logs -f pipeline

redis:
	docker start dr-redis 2>/dev/null || docker run -d --name dr-redis -p 6380:6379 redis:7

demo-mock: redis
	pip install -q redis flask numpy waitress
	DECISIONLM_MOCK=1 python pipeline/run.py start --config pipeline/demo/pipeline.demo.toml

demo-gpu: redis
	python pipeline/run.py start --config pipeline/demo/pipeline.demo.toml

publish:
	python pipeline/demo/publisher/publish.py --edge --dupes 5

test:
	cd service && python -m pytest -q
	cd pipeline && python -m pytest -q

lint:
	cd service && ruff check .
	cd pipeline && ruff check .
