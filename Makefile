.PHONY: install data osrm-prepare osrm-up osrm-down sim train eval path2-gate-results path2-theory-results path2-spatial-gate-results path2-gate-sweep path2-external-baselines path2-amod-smoke paper-results-cached paper-results-informs-cached paper-ts-submission paper-arxiv-v2 paper-results clean

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

path2-gate-results:
	python scripts/train_gate.py --epochs 120 --loss pairwise --top-k 30 --temperature 0.10 --random-simplex-trials 256 --out-dir results/path2_gate_pairwise
	python scripts/evaluate_gate.py --gate results/path2_gate_pairwise/similarity_gate.pt --out-dir results/path2_gate_pairwise --downstream-smoke --seeds 42

path2-theory-results:
	python scripts/verify_wait_bound.py

path2-spatial-gate-results:
	python scripts/train_spatial_gate.py --epochs 100 --loss pairwise --top-k 30 --temperature 0.10 --out-dir results/path2_gate_spatial
	python scripts/evaluate_gate.py --gate results/path2_gate_spatial/similarity_gate.pt --out-dir results/path2_gate_spatial_10seed --downstream-smoke --seeds 42,43,44,45,46,47,48,49,50,51 --include-all-static
	python scripts/analyze_gate_results.py results/path2_gate_spatial_10seed/gate_downstream_smoke.csv --out-dir results/path2_gate_spatial_10seed/analysis --focal-method learned_gate
	python scripts/verify_wait_bound.py --gate results/path2_gate_spatial/similarity_gate.pt --downstream results/path2_gate_spatial_10seed/gate_downstream_smoke.csv --out-dir results/path2_theory_spatial_gate_10seed

path2-external-baselines:
	python scripts/external_baselines_eval.py --seeds 42,43,44,45,46,47,48,49,50,51 --out-dir results/path2_external_baselines_gpr_mf050_10seed --oracle-lookahead 15 --methods batch_replay,wen2017_rebalancing,share_lp_hand_tuned,scenario_chance_mpc,gpr_chance_mpc_lite,spatial_gate_share_lp,oracle_mpc --share-move-fraction 0.50 --chance-move-fraction 0.50 --chance-lookahead 15 --chance-quantile 0.80 --chance-risk-weight 0.70 --gpr-chance-move-fraction 0.50 --gpr-chance-lookahead 15 --gpr-chance-quantile 0.90 --gpr-chance-risk-weight 0.80
	python scripts/analyze_gate_results.py results/path2_external_baselines_gpr_mf050_10seed/external_baselines.csv --out-dir results/path2_external_baselines_gpr_mf050_10seed/analysis_scenario_chance_mpc --focal-method scenario_chance_mpc --replay-method batch_replay
	python scripts/analyze_gate_results.py results/path2_external_baselines_gpr_mf050_10seed/external_baselines.csv --out-dir results/path2_external_baselines_gpr_mf050_10seed/analysis_gpr_chance_mpc --focal-method gpr_chance_mpc_lite --replay-method batch_replay
	python scripts/analyze_gate_results.py results/path2_external_baselines_gpr_mf050_10seed/external_baselines.csv --out-dir results/path2_external_baselines_gpr_mf050_10seed/analysis_share_lp --focal-method share_lp_hand_tuned --replay-method batch_replay
	python scripts/analyze_gate_results.py results/path2_external_baselines_gpr_mf050_10seed/external_baselines.csv --out-dir results/path2_external_baselines_gpr_mf050_10seed/analysis_spatial_share_lp --focal-method spatial_gate_share_lp --replay-method batch_replay

path2-gate-sweep:
	python scripts/sweep_spatial_gate_weights.py --epochs 60 --out-dir results/path2_gate_sweep

path2-amod-smoke:
	python scripts/amod_eval.py --seeds 42,43,44 --out-dir results/path2_amod_3seed

paper-results-cached:
	python scripts/build_path2_paper_tables.py
	python scripts/build_path2_review_response_tables.py
	python scripts/build_path2_figures.py
	cd paper && pdflatex -interaction=nonstopmode main_v2.tex && pdflatex -interaction=nonstopmode main_v2.tex && pdflatex -interaction=nonstopmode main_v2.tex

paper-results-informs-cached:
	python scripts/build_path2_paper_tables.py
	python scripts/build_path2_review_response_tables.py
	python scripts/build_path2_figures.py
	cd paper && pdflatex -interaction=nonstopmode main_v2_informs.tex && pdflatex -interaction=nonstopmode main_v2_informs.tex && pdflatex -interaction=nonstopmode main_v2_informs.tex

paper-ts-submission:
	python scripts/build_path2_paper_tables.py
	python scripts/build_path2_review_response_tables.py
	python scripts/build_path2_figures.py
	cd paper && pdflatex -interaction=nonstopmode main_ts_submission.tex && pdflatex -interaction=nonstopmode main_ts_submission.tex && pdflatex -interaction=nonstopmode main_ts_submission.tex

paper-arxiv-v2:
	python scripts/build_path2_paper_tables.py
	python scripts/build_path2_figures.py
	cd paper && pdflatex -interaction=nonstopmode main_v2_arxiv.tex && pdflatex -interaction=nonstopmode main_v2_arxiv.tex && pdflatex -interaction=nonstopmode main_v2_arxiv.tex

paper-results: path2-spatial-gate-results path2-gate-sweep path2-external-baselines path2-amod-smoke
	python scripts/train_contextual_dqn.py --out-dir results/path2_contextual_dqn
	python scripts/external_baselines_eval.py --seeds 42,43,44 --out-dir results/path2_contextual_dqn_3seed --oracle-lookahead 15 --methods batch_replay,lin2018_contextual_dqn --dqn-checkpoint results/path2_contextual_dqn/contextual_dqn_warmstart.npz
	python scripts/analyze_gate_results.py results/path2_contextual_dqn_3seed/external_baselines.csv --out-dir results/path2_contextual_dqn_3seed/analysis --focal-method lin2018_contextual_dqn --replay-method batch_replay
	python scripts/build_path2_paper_tables.py
	python scripts/build_path2_review_response_tables.py
	python scripts/build_path2_figures.py
	cd paper && pdflatex -interaction=nonstopmode main_v2.tex && pdflatex -interaction=nonstopmode main_v2.tex && pdflatex -interaction=nonstopmode main_v2.tex

clean:
	rm -rf results/__pycache__ src/**/__pycache__
