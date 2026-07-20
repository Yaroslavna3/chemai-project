#!/bin/bash

set -euo pipefail

resolve_project_root() {
    if [[ -n "${FREEDPP_PROJECT_ROOT:-}" ]]; then
        cd "$FREEDPP_PROJECT_ROOT"
    else
        cd "$(dirname "${BASH_SOURCE[0]}")/.."
    fi
    pwd
}

activate_runtime() {
    local project_root="$1"
    local runtime_root="${FREEDPP_RUNTIME_ROOT:-$project_root/.runtime}"
    export HOME="${FREEDPP_HOME:-$runtime_root/home}"
    export DGL_HOME="${DGL_HOME:-$HOME/.dgl}"
    export PYTHONPATH="$project_root:${PYTHONPATH:-}"
    mkdir -p "$HOME" "$DGL_HOME" "$project_root/experiments"

    set +u
    if [[ -n "${FREEDPP_CONDA_SH:-}" ]]; then
        source "$FREEDPP_CONDA_SH"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    fi

    if [[ -n "${FREEDPP_CONDA_ENV:-}" ]]; then
        conda activate "$FREEDPP_CONDA_ENV"
    else
        echo "FREEDPP_CONDA_ENV is not set; using current Python environment."
    fi
    set -u
}

target_args() {
    local project_root="$1"
    local target="$2"
    case "$target" in
        1kzn)
            echo "--receptor $project_root/receptors/1kzn.pdbqt --box_center 19.321,29.918,36.345 --box_size 15.05,19.71,19.12"
            ;;
        3fqs)
            echo "--receptor $project_root/receptors/3fqs/3fqs.pdbqt --box_center 1.751,0.781,13.386 --box_size 13.73,11.20,14.72"
            ;;
        *)
            echo "Unknown target: $target" >&2
            return 2
            ;;
    esac
}
objective_args() {
    local trajectory="$1"
    case "$trajectory" in
        llm_docking)
            echo "--objectives DrugLikeness,DockingScore --weights 10.0,1.0"
            ;;
        all_properties)
            echo "--objectives DrugLikeness,DockingScore,LogP,HeavyAtomCount,NumHAcceptors,NumHDonors,PAINS,Glaxo --weights 10.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0"
            ;;
        docking_metrics)
            echo "--objectives DockingScore,LogP,HeavyAtomCount,NumHAcceptors,NumHDonors,PAINS,Glaxo --weights 1.0,1.0,1.0,1.0,1.0,1.0,1.0"
            ;;
        *)
            echo "Unknown trajectory: $trajectory" >&2
            return 2
            ;;
    esac
}
