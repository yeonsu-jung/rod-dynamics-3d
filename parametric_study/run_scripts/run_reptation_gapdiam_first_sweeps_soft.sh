#!/usr/bin/env zsh

setopt errexit nounset pipefail

cd /Users/yeonsu/GitHub/rod-dynamics-3d

py() {
  /Users/yeonsu/anaconda3/bin/conda run -p /Users/yeonsu/anaconda3 --no-capture-output python /Users/yeonsu/.vscode/extensions/ms-python.python-2026.4.0-darwin-arm64/python_files/get_output_via_markers.py "$@"
}

gaps=(0.001 0.002 0.005 0.01 0.02 0.05 0.1)
mus=(0.01 0.01668100537200059 0.027825594022071243 0.046415888336127774 0.0774263682681127 0.1291549665014884 0.21544346900318834 0.3593813663804626 0.5994842503189409 1.0)
ars=(25 100 400 1000)

common_args=(
  --exe ./build/rigidbody_viewer_3d
  --scene assets/scenes/reptation_soft.json
  --rod-length 1.0
  --gaps
  "${gaps[@]}"
  --gap-radius-basis diameter
  --mus
  "${mus[@]}"
  --trials 20
  --jobs 4
  --no-stop-ke
  --perrod
  --perrod-stride 100
  --perrod-max 25000
)

run_dataset() {
  local ar="$1"
  local out_dir="$2"
  shift 2

  mkdir -p "$out_dir"
  echo "=== sweep: $out_dir ==="
  py scripts/sweep_reptation.py \
    --out-dir "$out_dir" \
    --aspect-ratio "$ar" \
    "${common_args[@]}" \
    "$@"

  echo "=== analyze first stop: $out_dir ==="
  py scripts/analyze_reptation_tangent_stop.py \
    --input-dir "$out_dir" \
    --output "$out_dir/tangent_stop_summary.csv" \
    --threshold 1e-5 \
    --dt 0.001 \
    --mode first \
    --window 1

  echo "=== plot first stop: $out_dir ==="
  py scripts/plot_sliding_length_vs_gap_over_mu.py \
    --input "$out_dir/tangent_stop_summary.csv" \
    --scatter-output "$out_dir/sliding_length_vs_gap_over_mu_scatter.png" \
    --summary-output "$out_dir/sliding_length_vs_gap_over_mu_summary.png" \
    --csv-output "$out_dir/sliding_length_vs_gap_over_mu_summary.csv"
}

for ar in "${ars[@]}"; do
  run_dataset "$ar" \
    "results/reptation_ar${ar}_soft_const_vn0p1_vt0_va0p1_w0_gapdiam_mugeom10_first" \
    --fixed-reptation --fixed-vn 0.1 --fixed-vt 0.0 --fixed-va 0.1 --fixed-w 0.0

  run_dataset "$ar" \
    "results/reptation_ar${ar}_soft_isogi_sv0p1_sw0p2_gapdiam_mugeom10_first" \
    --init-mode gaussian-isotropic --sigma-v 0.1 --sigma-w 0.2

  echo "=== completed requested soft-contact init families for AR=$ar ==="
done

echo "=== all requested soft-contact reptation sweeps completed ==="