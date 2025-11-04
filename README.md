# MindScoPy

**MindScoPy** is a Python package implementing user-state modeling and trajectory analysis techniques for EEG brain–computer interfaces (BCIs).  
It provides tools for unsupervised EEG segmentation, Markov chain modeling, and trajectory-based variability analysis — corresponding to the frameworks introduced in the following works:

1. **Markov chain-based user assessment**  
   Ivanov, N., Lio, A., & Chau, T. (2023). *Towards user-centric BCI design: Markov chain-based user assessment for mental imagery EEG-BCIs.*  
   *Journal of Neural Engineering, 20*(6), 066037. [doi:10.1088/1741-2552/ad17f2](https://doi.org/10.1088/1741-2552/ad17f2)

2. **Multi-class intra-trial trajectory (MITT) analysis**  
   Ivanov, N., Wong, M., & Chau, T. (2025). *A multi-class intra-trial trajectory analysis technique to visualize and quantify variability of mental imagery EEG signals.*  
   *International Journal of Neural Systems.* [doi:10.1142/S0129065725500753](https://doi.org/10.1142/S0129065725500753)

---

## Repository structure
```
│   bci_comp_iv2a_preprocessing.py
│   example_markov_chain_analysis.ipynb
│   example_trajectory.ipynb
│   markov_metric_analysis.py
│   trajectory_variance_analysis.py
│
└───mindscopy
    │   classification.py
    │   cluster_base.py
    │   markov_model.py
    │   trajectory.py
    │   __init__.py
    │
    ├───preprocessing
    │   │   artifact_removal.py
    │   │   feature_extraction.py
    │   │   misc.py
    │   │   rebias.py
    │   │   __init__.py
    │
    ├───utils
    │   │   cluster_identification.py
    │   │   transition_matrix.py
    │   │   visualization.py
    │   │   __init__.py
```
---

## Overview

- **`markov_model.py`** – Implements user-state transition modeling, steady-state metrics, and entropy-based stability measures.  
- **`trajectory.py`** – Defines subspace trajectory analysis for visualizing intra-trial EEG variability.  
- **`cluster_base.py`** – Provides the unsupervised clustering and subspace construction base classes.  
- **`classification.py`** – Contains implementations of a CSP-LDA classifier and run-wise classification accuracy metric used for metric validation.  
- **`preprocessing/`** – Includes EEG artifact rejection, feature extraction, and covariance matrix re-biasing utilities.  
- **`utils/`** – Helper modules for Markov chain model construction, visualization, and cluster identity management.  
- **Example notebooks** (`example_markov_chain_analysis.ipynb`, `example_trajectory.ipynb`) demonstrate full analysis workflows using the BCI Competition IV-2a dataset.
- **Example analysis scripts** (`markov_metric_analysis.py`, `trajectory_variance_analysis.py`) provide full implementations of the analyses performed in their respective journal articles using data from the BCI Competition IV-2a dataset.
- **`bci_comp_iv2a_preprocessing.py`** - Implements required preprocessing and data preparation steps for raw BCI competition IV-2a data.
---


## Citation
If you use this code in your research, please cite the following journal articles:
```bibtex
@article{ivanov2023markovbci,
  title={Towards user-centric {BCI} design: Markov chain-based user assessment for mental imagery {EEG}-{BCI}s},
  author={Ivanov, Nicolas and Lio, Aaron and Chau, Tom},
  journal={Journal of Neural Engineering},
  year={2023},
  volume={20},
  number={6},
  pages={066037},
  doi={10.1088/1741-2552/ad17f2}
}

@article{ivanov2025trajectorybci,
  title={A multi-class intra-trial trajectory analysis technique to visualize and quantify variability of mental imagery {EEG} signals},
  author={Ivanov, Nicolas and Wong, Madeline and Chau, Tom},
  journal={International Journal of Neural Systems},
  year={2025},
  doi={10.1142/S0129065725500753}
}
```
