
This folder contains the code used for the dataset construction with the  sysTEm curated experimental dataset .
---

This folder contains two subfolders:

- `systemDataCuration`
- `ActiveLearningDatasetConstruction`

---

## 1. `systemDataCuration`

The `systemDataCuration` folder contains the code and raw data used to curate and featurize the sysTEm dataset.

### `DataCuration_sysTEmDataset.ipynb`

This notebook featurizes the chemical compositions in `sysTEm_dataset.xlsx`, curates the dataset and drops correlated features.

Before running the notebook, make sure that:

1. The sysTEm dataset has been downloaded.
2. The local file path to `sysTEm_dataset.xlsx` is correctly specified in the notebook.

The notebook outputs two CSV files named:

```text
CLEANED_DATA.csv
CLEANED_DATA_AND_FILTERED.csv
```
The first file is the data curated version (no dropping of correlated features) of the sysTEm pool. The second is the file used as the candidate pool for the active-learning dataset construction experiments.

### `sysTEm_dataset.xlsx`

This is the original raw sysTEm dataset.

The dataset is available from the following GitHub repository:

```text
https://github.com/tankylz/sysTEm_dataset
```

Dataset DOI:

```text
10.26434/chemrxiv-2025-4gxmc
```

---

## 2. `ActiveLearningDatasetConstruction`

The `ActiveLearningDatasetConstruction` folder contains the scripts used to run the active-learning dataset construction experiments.

### `run_al_thermo.bash`

The file by default runs a single seed and single policy. This can be used as DEMO. 

This Bash script sequentially runs all random seeds and dataset construction strategies by calling `run_all_thermo.py`.

To run this script, open a terminal in the `ActiveLearningDatasetConstruction` folder and execute:

```bash
bash run_al_thermo.bash
```

### `run_all_thermo.py`

This is the main script for running one dataset construction experiment with one random seed.

The script uses:

```text
CLEANED_DATA_AND_FILTERED.csv
```

as the candidate pool.

It calls a dataset construction policy defined in:

```text
uncertainty_Thermo.py
```

The script outputs CSV files containing the performance metrics (RMSE, R2 and MAE) of Random Forest and XGBoost models predicting
on hold-out test data the different outcome properties zT, Thermal Conductivity, Electrical Conductivity and Seebeck Coefficient.

The output files are saved in a folder named:

```text
al
```

Inside the `al` folder, results are organized into subfolders according to the dataset construction policy used.

### `uncertainty_Thermo.py`

This Python module contains the different dataset construction policies used in the active-learning experiments.

The repository includes the acquisition policies listed below. The tables map the names used in the paper to their corresponding code identifiers.

To select a policy, uncomment its identifier in `GROWING_CRITERIA` in `run_al_thermo.bash`. You can enable multiple policies to run them sequentially.

### Policy names and code identifiers

| Policy name in the paper | One targeted outcome | Two targeted outcomes | Three targeted outcomes |
|---|---|---|---|
| QBC | `grow_QBC` | `grow_2QBC` | `grow_3QBC` |
| HV + k-medoids | `grow_disagreementHV_topM_kmedoids_OneTarget` | `grow_disagreementHV_topM_kmedoids_TwoTargets` | `grow_disagreementHV_topM_kmedoids_ThreeTargets` |
| TS-Dis. | `grow_TS_QBC_no_diversity_OneTarget` | `grow_TS_QBC_no_diversity_TwoTargets` | `grow_TS_QBC_no_diversity_ThreeTargets` |
| DPP TS-Dis. | `grow_DPP_TS_QBC_OneTarget` | `grow_DPP_TS_QBC_TwoTargets` | `grow_DPP_TS_QBC_ThreeTargets` |
| NSGA-II Dis. | — | `grow_EA_2Obj_TwoTargets_QBC_ONLY` | `grow_EA_3Obj_ThreeTargets_QBC_ONLY` |
| NSGA-II Dis. + diversity | `grow_EA_DeltaDiversity` | `grow_EA_3Obj_TwoTargets` | `grow_EA_4Obj_ThreeTargets` |

Random sampling uses `grow_random` with `target=all`.

The repository also provides the following additional variants not included in the main text of the paper but included in the Supplementary Information:

| Additional policy | Two targeted outcomes | Three targeted outcomes |
|---|---|---|
| HV without diversity | `grow_disagreementHV_no_diversity_TwoTargets` | `grow_disagreementHV_no_diversity_ThreeTargets` |

### Naming convention

- **QBC** stands for query by committee.
- **HV** stands for hypervolume.
- **TS** stands for Thompson sampling.
- **DPP** stands for determinantal point process.
- **Dis.** denotes model disagreement.
- **NSGA-II** stands for Non-dominated Sorting Genetic Algorithm II.
- Historical code identifiers retain `QBC` in the TS-Dis. and DPP TS-Dis. names.
- For NSGA-II policies, `2Obj`, `3Obj`, and `4Obj` indicate the number of optimization objectives, including the diversity objective when present.
- `OneTarget`, `TwoTargets`, and `ThreeTargets` indicate the number of targeted material properties.
- `QBC_ONLY` identifies the NSGA-II variants that optimize disagreement without an additional diversity objective.

## Experimental target identifiers

The curated experimental thermoelectric dataset contains four material properties: electrical conductivity, thermal conductivity, Seebeck coefficient, and thermoelectric figure of merit (zT). The following identifiers specify which properties are targeted during acquisition.

| Target group | Code identifier | Material properties |
|---|---|---|
| One outcome | `electric_cond` | Electrical conductivity |
| One outcome | `seebeck_coef` | Seebeck coefficient |
| One outcome | `zT` | Thermoelectric figure of merit (zT) |
| One outcome | `thermal_cond` | Thermal conductivity |
| Two outcomes | `el_seeb` | Electrical conductivity and Seebeck coefficient |
| Two outcomes | `el_zT` | Electrical conductivity and zT |
| Two outcomes | `el_ThC` | Electrical conductivity and thermal conductivity |
| Two outcomes | `seeb_zT` | Seebeck coefficient and zT |
| Two outcomes | `seeb_ThC` | Seebeck coefficient and thermal conductivity |
| Two outcomes | `zT_ThC` | zT and thermal conductivity |
| Three outcomes | `el_seeb_zT` | Electrical conductivity, Seebeck coefficient, and zT |
| Three outcomes | `el_seeb_ThC` | Electrical conductivity, Seebeck coefficient, and thermal conductivity |
| Three outcomes | `el_zT_ThC` | Electrical conductivity, zT, and thermal conductivity |
| Three outcomes | `seeb_zT_ThC` | Seebeck coefficient, zT, and thermal conductivity |
| All outcomes | `all` | All four properties; used for random sampling |

### Target configuration

Edit the target arrays in `run_al_thermo.bash` to select one or more target combinations in each group. For example:

    TARGETS_1=(electric_cond)
    TARGETS_2=(el_seeb)
    TARGETS_3=(el_seeb_zT)
    TARGETS_ALL=(all)


Each selected acquisition policy uses its corresponding target group. Properties excluded from acquisition are evaluated as untargeted outcomes. The `all` identifier is used for the random-sampling baseline.


### Running the demo

Activate the Python environment containing the required dependencies, then run the following command from the directory containing `run_all_thermo.py`:

    bash run_al_thermo.bash

The demo enables `grow_QBC` by default. Edit `GROWING_CRITERIA` to select other policies, and edit `RANDOM_SEEDS` and the target arrays to configure the runs.

Each selected policy runs its full acquisition campaign. Results are saved under:

    al/<policy>/<dataset>/<target>/<seed>/

You can also select a policy directly through the Python command-line interface. For example:

    python run_all_thermo.py \
        --growingCriteria grow_EA_3Obj_TwoTargets \
        --dataset ThermoEl \
        --target el_seeb \
        --outputDir al/grow_EA_3Obj_TwoTargets/ThermoEl/el_seeb/0 \
        --randomSeed 0

This example runs NSGA-II Dis. + diversity targeting electrical conductivity and the Seebeck coefficient.


To run all the policies modify the `run_al_thermo.bash` file to include every policy. Random seeds used in paper 0 1 2 3 4. 

