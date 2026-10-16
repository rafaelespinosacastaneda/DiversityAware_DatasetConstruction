#!/bin/bash
set -euo pipefail
# Run from the directory containing run_al.py.
# Activate your Python environment before running this script.
# All 13 policies run sequentially without time limits.
# Seeds used in the paper: 0 1 2 3 4
RANDOM_SEEDS=(0)
# Available datasets: jarvis18 jarvis22 mp18 mp21
DATASETS=(jarvis18)
GROWING_CRITERIA=(
    QBC
    grow_EA_QBC_1Obj_withDiversity
    growing_EA_2obj_qbc_only
    grow_EA_QBC_2Obj_twoTargets_plusDiversity
    grow_disagreementHV_topM_kmedoids_OneTarget
    grow_DPP_TS_QBC_OneTarget
    grow_TS_QBC_no_diversity_OneTarget
    grow2QBC
    grow_disagreementHV_topM_kmedoids_TwoTargets
    grow_disagreementHV_no_diversity_TwoTargets
    grow_DPP_TS_QBC_TwoTargets
    grow_TS_QBC_no_diversity_TwoTargets
    random
)
# Available single outcomes: bandgap e_form bulk_modulus
TARGETS_1=(bandgap)
# Available pairs: eform_bulk_modulus bandgap_eform bandgap_bulkmodulus
TARGETS_2=(bandgap_bulkmodulus)
TARGETS_ALL=(all)
targets_for_criterion() {
    local crit="$1"
    case "$crit" in
        QBC|grow_EA_QBC_1Obj_withDiversity|grow_disagreementHV_topM_kmedoids_OneTarget|grow_DPP_TS_QBC_OneTarget|grow_TS_QBC_no_diversity_OneTarget)
            echo "${TARGETS_1[*]}" ;;
        grow2QBC|growing_EA_2obj_qbc_only|grow_EA_QBC_2Obj_twoTargets_plusDiversity|grow_disagreementHV_topM_kmedoids_TwoTargets|grow_disagreementHV_no_diversity_TwoTargets|grow_DPP_TS_QBC_TwoTargets|grow_TS_QBC_no_diversity_TwoTargets)
            echo "${TARGETS_2[*]}" ;;
        random)
            echo "${TARGETS_ALL[*]}" ;;
        *)
            echo "ERROR: Unknown growingCriteria '$crit'." >&2
            return 1 ;;
    esac
}
for randomSeed in "${RANDOM_SEEDS[@]}"; do
    for growingCriteria in "${GROWING_CRITERIA[@]}"; do
        target_list="$(targets_for_criterion "$growingCriteria")"
        read -r -a TARGETS <<< "$target_list"
        for dataset in "${DATASETS[@]}"; do
            for target in "${TARGETS[@]}"; do
                outputDir="al/$growingCriteria/$dataset/$target/$randomSeed"
                mkdir -p "$outputDir"
                echo
                echo "Running: seed=$randomSeed criterion=$growingCriteria dataset=$dataset target=$target"
                python -u run_al.py \
                    --growingCriteria "$growingCriteria" \
                    --dataset "$dataset" \
                    --target "$target" \
                    --outputDir "$outputDir" \
                    --randomSeed "$randomSeed"
            done
        done
    done
done
