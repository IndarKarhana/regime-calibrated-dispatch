.PHONY: install data osrm-prepare osrm-up osrm-down sim train eval clean

OSRM_DIR  := data/osrm
PBF_FILE  := $(OSRM_DIR)/new-york-latest.osm.pbf
PBF_URL   := https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf
OSRM_IMG  := ghcr.io/project-osrm/osrm-backend

install:
	pip install -e ".[dev]"

# ── Data ────────────────────────────────────────────────
data:
	python scripts/download_tlc.py

# ── OSRM ────────────────────────────────────────────────
$(PBF_FILE):
	mkdir -p $(OSRM_DIR)
	curl -L -o $(PBF_FILE) $(PBF_URL)

osrm-prepare: $(PBF_FILE)
	docker run -t -v "$$(pwd)/$(OSRM_DIR):/data" $(OSRM_IMG) \
		osrm-extract -p /opt/car.lua /data/new-york-latest.osm.pbf
	docker run -t -v "$$(pwd)/$(OSRM_DIR):/data" $(OSRM_IMG) \
		osrm-partition /data/new-york-latest.osrm
	docker run -t -v "$$(pwd)/$(OSRM_DIR):/data" $(OSRM_IMG) \
		osrm-customize /data/new-york-latest.osrm

osrm-up:
	docker compose up -d osrm

osrm-check:
	@python -c "import requests; from pathlib import Path; import yaml; c=yaml.safe_load(open('config/default.yaml')); o=c['osrm']; u=f\"{o['host']}:{o['port']}/nearest/v1/car/-73.985,40.748\"; r=requests.get(u,timeout=3); print('OK' if r.status_code==200 else f'FAIL {r.status_code}')" 2>/dev/null || echo "FAIL (requests error or OSRM down)"

osrm-down:
	docker compose down

# ── Run ─────────────────────────────────────────────────
sim:
	python -m src.simulator.engine

train:
	python -m src.rl.train

eval:
	python -m src.evaluation.ablation_runner

clean:
	rm -rf results/__pycache__ src/**/__pycache__
