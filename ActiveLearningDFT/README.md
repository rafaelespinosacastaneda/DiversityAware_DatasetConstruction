## DFT datasets

The MP18, MP21, JARVIS18, and JARVIS22 featurized datasets used in this
repository are available from the following Zenodo archive:

Li, K., Persaud, D., Choudhary, K., DeCost, B., Greenwood, M., and
Hattrick-Simpers, J. (2024). *Exploiting redundancy in large materials
datasets for efficient machine learning with less data* (version v3).
Zenodo. https://doi.org/10.5281/zenodo.12763938


The code automatically downloads the required file if it is not already
present in the data directory. The downloaded datasets undergo additional
data curation in the preprocessing code before the active-learning experiments.


## Acquisition policies

The repository includes the DFT acquisition policies listed below. The tables map the names used in the paper to the command-line identifiers accepted by `run_al.py`.

Select the policies to run by editing `GROWING_CRITERIA` in `run_al.bash`. All 13 policies are enabled in the provided script. Comment out any policies you do not want to run. Selected policies run sequentially.

### Policy names and command-line identifiers

| Policy name in the paper | One targeted outcome | Two targeted outcomes |
|---|---|---|
| QBC | `QBC` | `grow2QBC` |
| HV + k-medoids | `grow_disagreementHV_topM_kmedoids_OneTarget` | `grow_disagreementHV_topM_kmedoids_TwoTargets` |
| TS-Dis. | `grow_TS_QBC_no_diversity_OneTarget` | `grow_TS_QBC_no_diversity_TwoTargets` |
| DPP TS-Dis. | `grow_DPP_TS_QBC_OneTarget` | `grow_DPP_TS_QBC_TwoTargets` |
| NSGA-II Dis. | — | `growing_EA_2obj_qbc_only` |
| NSGA-II Dis. + diversity | `grow_EA_QBC_1Obj_withDiversity` | `grow_EA_QBC_2Obj_twoTargets_plusDiversity` |

Random sampling uses the command-line identifier `random` with `target=all`.

The repository also provides the following variant, included in the Supplementary Information:

| Additional policy | Two targeted outcomes |
|---|---|
| HV without diversity | `grow_disagreementHV_no_diversity_TwoTargets` |

### Naming convention

- **QBC** stands for query by committee.
- **HV** stands for hypervolume.
- **TS** stands for Thompson sampling.
- **DPP** stands for determinantal point process.
- **Dis.** denotes model disagreement.
- **NSGA-II** stands for Non-dominated Sorting Genetic Algorithm II.
- Historical code identifiers retain `QBC` in the TS-Dis., DPP TS-Dis., and NSGA-II names.
- `OneTarget` and `TwoTargets` indicate the number of targeted material properties.
- `grow2QBC` is the command-line identifier for QBC targeting two outcomes.
- In the NSGA-II identifiers, `1Obj` and `2Obj` count the targeted disagreement objectives. The `withDiversity` and `plusDiversity` variants include an additional diversity objective.
- Command-line identifiers must match the dispatch conditions in `run_al.py`. They are not always identical to the Python function names.

The NSGA-II command-line identifiers call the following functions:

| Command-line identifier | Python function |
|---|---|
| `growing_EA_2obj_qbc_only` | `grow_EA_QBC_2Obj_twoTargets_from3Outcomes` |
| `grow_EA_QBC_1Obj_withDiversity` | `grow_EA_QBC_1Obj_withDiversity_from3Outcomes` |
| `grow_EA_QBC_2Obj_twoTargets_plusDiversity` | `grow_EA_QBC_2Obj_twoTargets_plusDiversity_from3Outcomes` |

Single-target QBC uses `QBC`, which calls `grow_QBC`. Random sampling uses `random`, which calls `grow_random`.

### Available datasets and targets

The available datasets are `jarvis18`, `jarvis22`, `mp18`, and `mp21`. The provided script selects `jarvis18`. To run all four datasets, set:

```bash
DATASETS=(jarvis18 jarvis22 mp18 mp21)
```

The target identifiers are:

| Target group | Code identifier | Material properties |
|---|---|---|
| One outcome | `bandgap` | Band gap |
| One outcome | `e_form` | Formation energy |
| One outcome | `bulk_modulus` | Bulk modulus |
| Two outcomes | `eform_bulk_modulus` | Formation energy and bulk modulus |
| Two outcomes | `bandgap_eform` | Band gap and formation energy |
| Two outcomes | `bandgap_bulkmodulus` | Band gap and bulk modulus |
| All outcomes | `all` | All three properties; used for random sampling |

Configure one or more targets in each group:

```bash
TARGETS_1=(bandgap)
TARGETS_2=(bandgap_bulkmodulus)
TARGETS_ALL=(all)
```

### Running the demo

Activate the Python environment containing the required dependencies, then run the following command from the directory containing `run_al.py`:

```bash
bash run_al.bash
```

The provided script enables all 13 policies:

```bash
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
```

Edit `GROWING_CRITERIA` to select policies, and edit `DATASETS`, `RANDOM_SEEDS`, and the target arrays to configure the runs.

Each selected policy runs its full acquisition campaign without a time limit. Results are saved under:

```text
al/<policy>/<dataset>/<target>/<seed>/
```

Here, `<policy>` is the command-line identifier, such as `QBC`, `growing_EA_2obj_qbc_only`, or `random`.

You can also select a policy directly through the Python command-line interface. For example:

```bash
python run_al.py \
    --growingCriteria grow_EA_QBC_2Obj_twoTargets_plusDiversity \
    --dataset jarvis18 \
    --target bandgap_bulkmodulus \
    --outputDir al/grow_EA_QBC_2Obj_twoTargets_plusDiversity/jarvis18/bandgap_bulkmodulus/0 \
    --randomSeed 0
```

This example runs NSGA-II Dis. + diversity targeting band gap and bulk modulus in JARVIS18.

The two-target NSGA-II policies require their identifiers to be included in the target-initialization condition in `run_al.py`, so that `target1` and `target2` are assigned before the policy function is called. This correction is included in the updated runner.

To use all target combinations, set:

```bash
TARGETS_1=(bandgap e_form bulk_modulus)
TARGETS_2=(eform_bulk_modulus bandgap_eform bandgap_bulkmodulus)
TARGETS_ALL=(all)
```

The random seeds used in the paper were:

```bash
RANDOM_SEEDS=(0 1 2 3 4)
```
