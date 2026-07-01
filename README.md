# landslide-stability-assessment
# AL-SMF: Active-Learning Surrogate Model Framework for Landslide Susceptibility Assessment

This repository contains the official Python implementation of the active-learning surrogate model framework described in our manuscript: 
*"A rapid slope stability assessment framework coupling multi-directional limit equilibrium analysis with an active-learning surrogate model: A case study of Qingshui County, Gansu, China"*.

---

## 1. Code Purpose
The scripts in this repository are developed to compress thousands of expensive physical simulations into intelligent inferences using an active-learning Support Vector Machine (SVM) surrogate model. This framework systematically handles the processing of slope-unit terrain features, integrates multi-directional limit equilibrium factor of safety (FoS) labels, and optimizes failure-sample identification under severe class imbalance.

---

## 2. Repository Structure

The project follows a rigorous engineering layout with nested pipeline stages matching the multi-round active learning framework:

```text
landslide-stability-assessment/
├── data/
│   └── ssap_dat/
│       └── 0_Main.dat                # Source geometric/stratigraphic physical inputs
├── src/
│   ├── stage1/
│   │   └── First-round sample selection.py   # Active Learning Round 1 uncertainty mining
│   ├── stage2/
│   │   ├── First-round data addition.py      # Augmenting training pool with Round 1 labels
│   │   └── Second-round sample selection.py  # Active Learning Round 2 uncertainty mining
│   ├── stage3/
│   │   ├── Second-round data addition.py     # Augmenting training pool with Round 2 labels
│   │   └── Third-round sample selection.py   # Active Learning Round 3 uncertainty mining
│   ├── stage4/
│   │   ├── Third-round data addition.py      # Augmenting training pool with Round 3 labels
│   │   └── Fourth-round sample selection.py  # Active Learning Round 4 uncertainty mining
│   ├── stage5/
│   │   ├── Fourth-round data addition.py     # Augmenting training pool with Round 4 labels
│   │   └── Fifth-round sample selection.py   # Active Learning Round 5 final optimization
│   ├── Initial sample.py             # Selection of the 1,000 initial baseline samples
│   ├── Model.py                      # Machine learning surrogate model definition (SVM/Ensembles)
│   ├── mod_data_generation.py        # Core terrain feature variable generation
│   └── ssap_dataset_creation.py      # Compiling multi-directional FoS from SSAP outputs
└── README.md                         # Project documentation and execution guide
