# One-click reproduction of arXiv:2606.03952 figures.
# Run from the repository root:  make -C repro <target>
#
# Targets
#   build      headless binary (build-headless/)
#   metadata   packing metadata incl. free-volume columns (needs RFV binary)
#   manifest   sweep manifest + generated scenes
#   pilot      short verification: gates + c2/c3/c4 groups + static figures
#   sweep      the full c1 main sweep (720 runs; a weekend of core-time)
#   figures    every analysis module that has data available
#   all        metadata manifest sweep figures

REPO := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PY := python3
JOBS ?=
RFV ?= rod_free_volume   # path to rod-free-volume binary (external repo)

.PHONY: build metadata manifest pilot sweep figures all

build:
	cmake -S $(REPO) -B $(REPO)/build-headless -DBUILD_HEADLESS=ON \
	      -DCMAKE_BUILD_TYPE=Release
	cmake --build $(REPO)/build-headless -j

metadata: build
	cd $(REPO) && $(PY) repro/build_packing_metadata.py
	cd $(REPO) && $(PY) repro/fill_free_volume.py --binary $(RFV)

manifest:
	cd $(REPO) && $(PY) repro/gen_manifest.py

pilot: manifest
	cd $(REPO) && $(PY) repro/run_sweep.py --group c2 --group c3 \
	    --group c4 $(if $(JOBS),--jobs $(JOBS))
	cd $(REPO) && $(PY) -m repro.analysis.fig2
	cd $(REPO) && $(PY) -m repro.analysis.fig4ab
	cd $(REPO) && $(PY) -m repro.analysis.figS1
	cd $(REPO) && $(PY) -m repro.analysis.figS4
	cd $(REPO) && $(PY) -m repro.analysis.end_matter

sweep: manifest
	cd $(REPO) && $(PY) repro/run_sweep.py --group c1 \
	    $(if $(JOBS),--jobs $(JOBS))

figures:
	cd $(REPO) && $(PY) -m repro.analysis.fig2
	cd $(REPO) && $(PY) -m repro.analysis.fig4ab
	cd $(REPO) && $(PY) -m repro.analysis.figS1
	-cd $(REPO) && $(PY) -m repro.analysis.figS4
	-cd $(REPO) && $(PY) -m repro.analysis.end_matter
	-cd $(REPO) && $(PY) -m repro.analysis.fig4cd

all: metadata manifest sweep figures
