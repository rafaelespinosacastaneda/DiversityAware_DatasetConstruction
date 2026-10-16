#!/bin/bash
set -euo pipefail
# Run from the directory containing run_all_thermo.py.
# Activate your Python environment before running this script.
# Uncomment the policies you want to run; grow_QBC is enabled by default.
# Each selected policy runs its full acquisition campaign.
RANDOM_SEEDS=(0)
DATASETS=(ThermoEl)
GROWING_CRITERIA=(
    grow_QBC
    # grow_EA_DeltaDiversity
    # grow_disagreementHV_topM_kmedoids_OneTarget
    # grow_DPP_TS_QBC_OneTarget
    # grow_TS_QBC_no_diversity_OneTarget
    # grow_2QBC
    # grow_EA_3Obj_TwoTargets
    # grow_EA_2Obj_TwoTargets_QBC_ONLY
    # grow_disagreementHV_topM_kmedoids_TwoTargets
    # grow_disagreementHV_no_diversity_TwoTargets
    # grow_DPP_TS_QBC_TwoTargets
    # grow_TS_QBC_no_diversity_TwoTargets
    # grow_3QBC
    # grow_EA_4Obj_ThreeTargets
    # grow_EA_3Obj_ThreeTargets_QBC_ONLY
    # grow_disagreementHV_topM_kmedoids_ThreeTargets
    # grow_disagreementHV_no_diversity_ThreeTargets
    # grow_DPP_TS_QBC_ThreeTargets
    # grow_TS_QBC_no_diversity_ThreeTargets
    # grow_random
)
# Choose one or more targets in each group.
# Available single outcomes: electric_cond seebeck_coef zT thermal_cond
TARGETS_1=(electric_cond)
# Available pairs: el_seeb el_zT el_ThC seeb_zT seeb_ThC zT_ThC
TARGETS_2=(el_seeb)
# Available triples: el_seeb_zT el_seeb_ThC el_zT_ThC seeb_zT_ThC
TARGETS_3=(el_seeb_zT)
TARGETS_ALL=(all)
targets_for_criterion() {
    local crit="$1"
    case "$crit" in
        grow_QBC|grow_EA_DeltaDiversity|grow_disagreementHV_topM_kmedoids_OneTarget|grow_DPP_TS_QBC_OneTarget|grow_TS_QBC_no_diversity_OneTarget)
            echo "${TARGETS_1[*]}" ;;
        grow_2QBC|grow_EA_3Obj_TwoTargets|grow_EA_2Obj_TwoTargets_QBC_ONLY|grow_disagreementHV_topM_kmedoids_TwoTargets|grow_disagreementHV_no_diversity_TwoTargets|grow_DPP_TS_QBC_TwoTargets|grow_TS_QBC_no_diversity_TwoTargets)
            echo "${TARGETS_2[*]}" ;;
        grow_3QBC|grow_EA_4Obj_ThreeTargets|grow_EA_3Obj_ThreeTargets_QBC_ONLY|grow_disagreementHV_topM_kmedoids_ThreeTargets|grow_disagreementHV_no_diversity_ThreeTargets|grow_DPP_TS_QBC_ThreeTargets|grow_TS_QBC_no_diversity_ThreeTargets)
            echo "${TARGETS_3[*]}" ;;
        grow_random)
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
                echo "Running: seed=$randomSeed criterion=$growingCriteria dataset=$dataset target=$target"
                python -u run_all_thermo.py \
                    --growingCriteria "$growingCriteria" \
                    --dataset "$dataset" \
                    --target "$target" \
                    --outputDir "$outputDir" \
                    --randomSeed "$randomSeed"
            done
        done
    done
done
