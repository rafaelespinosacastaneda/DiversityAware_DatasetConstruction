## Acquisition policies

The repository includes the DFT acquisition policies listed below. The tables map the names used in the paper to their corresponding code identifiers.

To select a policy, uncomment its identifier in `GROWING_CRITERIA` in `run_al.bash`. You can enable multiple policies to run them sequentially.

### Policy names and code identifiers

| Policy name in the paper | One targeted outcome | Two targeted outcomes |
|---|---|---|
| QBC | `grow_QBC` | `grow2QBC` |
| HV + k-medoids | `grow_disagreementHV_topM_kmedoids_OneTarget` | `grow_disagreementHV_topM_kmedoids_TwoTargets` |
| TS-Dis. | `grow_TS_QBC_no_diversity_OneTarget` | `grow_TS_QBC_no_diversity_TwoTargets` |
| DPP TS-Dis. | `grow_DPP_TS_QBC_OneTarget` | `grow_DPP_TS_QBC_TwoTargets` |

Random sampling uses `grow_random` with `target=all`.

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
- Historical code identifiers retain `QBC` in the TS-Dis. and DPP TS-Dis. names.
- `OneTarget` and `TwoTargets` indicate the number of targeted material properties.
- `grow2QBC` is the code identifier for QBC targeting two outcomes.

### Available datasets and targets

The available datasets are `jarvis18`, `jarvis22`, `mp18`, and `mp21`. Configure the datasets to run in `run_al.bash`:

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

The demo enables `grow_QBC` by default. Edit `GROWING_CRITERIA` to select other policies, and edit `DATASETS`, `RANDOM_SEEDS`, and the target arrays to configure the runs.

Each selected policy runs its full acquisition campaign. Results are saved under:

```text
al/<policy>/<dataset>/<target>/<seed>/
```

You can also select a policy directly through the Python command-line interface. For example:

```bash
python run_al.py \
    --growingCriteria grow_DPP_TS_QBC_TwoTargets \
    --dataset jarvis18 \
    --target bandgap_bulkmodulus \
    --outputDir al/grow_DPP_TS_QBC_TwoTargets/jarvis18/bandgap_bulkmodulus/0 \
    --randomSeed 0
```

This example runs DPP TS-Dis. targeting band gap and bulk modulus in JARVIS18.

To run all policies listed above, uncomment every policy in `GROWING_CRITERIA` in `run_al.bash`. To use all target combinations, set:

```bash
TARGETS_1=(bandgap e_form bulk_modulus)
TARGETS_2=(eform_bulk_modulus bandgap_eform bandgap_bulkmodulus)
TARGETS_ALL=(all)
```

The random seeds used in the paper were:

```bash
RANDOM_SEEDS=(0 1 2 3 4)
```
