#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${REPO_ROOT}/data"
CODE_DIR="${REPO_ROOT}/code"

CONFIG_PATH="${CODE_DIR}/config_decoding.yaml"
DECODING_SCRIPT="${CODE_DIR}/02_decoding.py"
PLOTTING_SCRIPT="${CODE_DIR}/03_plot_decoding_results.py"

FIGURES_ONLY=false
OVERRIDE=false
for arg in "$@"; do
  case "$arg" in
    --figures-only)
      FIGURES_ONLY=true
      ;;
    --override)
      OVERRIDE=true
      ;;
    -h|--help)
      cat <<'EOF'
Usage: run_decoding_all_derivatives.sh [--figures-only] [--override]

Options:
  --figures-only   Regenerate figures for all derivative folders only (no decoding, no skip).
  --override       Force re-run and overwrite outputs even if results_decoding already exists.
  -h, --help       Show this help.
EOF
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: Missing config file: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DECODING_SCRIPT}" ]]; then
  echo "ERROR: Missing decoding script: ${DECODING_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${PLOTTING_SCRIPT}" ]]; then
  echo "ERROR: Missing plotting script: ${PLOTTING_SCRIPT}" >&2
  exit 1
fi

TMP_BACKUP="$(mktemp)"
cp "${CONFIG_PATH}" "${TMP_BACKUP}"

restore_config() {
  cp "${TMP_BACKUP}" "${CONFIG_PATH}" || true
  rm -f "${TMP_BACKUP}" || true
}
trap restore_config EXIT

mapfile -t DERIV_DIRS < <(
  find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d \( -name "derivative*" -o -name "derivatives*" \) | sort
)

if [[ ${#DERIV_DIRS[@]} -eq 0 ]]; then
  echo "No derivative folders found under ${DATA_ROOT}"
  exit 0
fi

for DERIV_ROOT in "${DERIV_DIRS[@]}"; do
  RESULTS_DIR="${DERIV_ROOT}/results_decoding"
  FIGURES_DIR="${DERIV_ROOT}/figures_decoding"

  if [[ "${FIGURES_ONLY}" == false && "${OVERRIDE}" == false && -d "${RESULTS_DIR}" ]]; then
    echo "[SKIP] ${DERIV_ROOT} (results_decoding exists)"
    continue
  fi

  if [[ "${FIGURES_ONLY}" == true ]]; then
    echo "[FIG ] ${DERIV_ROOT}"
  else
    echo "[RUN ] ${DERIV_ROOT}"
  fi

  if [[ "${OVERRIDE}" == true ]]; then
    if [[ "${FIGURES_ONLY}" == true ]]; then
      rm -rf "${FIGURES_DIR}"
    else
      rm -rf "${RESULTS_DIR}" "${FIGURES_DIR}"
    fi
  fi

  mkdir -p "${RESULTS_DIR}" "${FIGURES_DIR}"

  python3 - "${CONFIG_PATH}" "${DATA_ROOT}" "${DERIV_ROOT}" "${RESULTS_DIR}" "${FIGURES_DIR}" <<'PY'
# filepath: /home/miliczpl/MySSD/Code/EEG/run_decoding_all_derivatives.sh
import sys
import yaml
from pathlib import Path

config_path, bids_root, deriv_root, results_dir, figures_dir = sys.argv[1:]
p = Path(config_path)

with p.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("paths", {})
cfg["paths"]["bids_root"] = str(Path(bids_root))
cfg["paths"]["deriv_root"] = str(Path(deriv_root))
cfg["paths"]["results_dir"] = str(Path(results_dir))
cfg["paths"]["figures_dir"] = str(Path(figures_dir))

with p.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

  if [[ "${FIGURES_ONLY}" == false ]]; then
    if ! python3 "${DECODING_SCRIPT}"; then
      echo "[FAIL] Decoding failed for ${DERIV_ROOT}" >&2
      continue
    fi
  fi

  if ! python3 "${PLOTTING_SCRIPT}"; then
    echo "[FAIL] Plotting failed for ${DERIV_ROOT}" >&2
    continue
  fi

  if [[ "${FIGURES_ONLY}" == true ]]; then
    echo "[DONE] figures ${DERIV_ROOT}"
  else
    echo "[DONE] ${DERIV_ROOT}"
  fi
done

if [[ "${FIGURES_ONLY}" == true ]]; then
  echo "All derivative folders processed (figures only)."
else
  echo "All derivative folders processed."
fi
