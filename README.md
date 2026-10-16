# DiversityAware_DatasetConstruction
This repository contains the code needed to reproduce the results reported in the paper Diversity-Aware Active Learning Builds Materials Datasets for Current and Future Objectives.
The results can be downloaded also directly from the zenodo repository [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23424148.svg)](https://doi.org/10.5281/zenodo.23424148). 

The repository is organized into three main folders:

- `ActiveLearningDFT`
- `ActiveLearningExperimental`
- `PlottingCodes`

The `ActiveLearningDFT` folder contains the code used for the DFT dataset construction experiments. These experiments use JARVIS18, JARVIS22, MP18, and MP21 as candidate pools. This folder also includes the Zenodo link to the corresponding DFT dataset construction results.

The `ActiveLearningExperimental` folder contains the code used for the experimental dataset construction experiments. These experiments use the sysTEm dataset as the candidate pool. 

The `PlottingCodes` folder contains the codes used to generate all plots in the paper.


---
# Installation
Installation time should range from 15- 25 minutes .

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

This repository provides two installation options:

1. **Public/local installation** using `requirements.txt`
2. **Compute Canada installation** using `requirements_compute_canada.txt`

Users with no Compute Canada access should use the first option.

Create the environment and install the required packages (by default uses `requirements.txt` ):
```bash

bash install_env.sh
```
This script creates a Python virtual environment named test and installs the packages listed in requirements.txt.
After installation, activate the environment with:
```bash
source test/bin/activate
```

On Windows Git Bash, use:
```bash
source test/Scripts/activate
```

---

