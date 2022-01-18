[![Documentation Status](https://readthedocs.org/projects/hydrocnhs/badge/?version=latest)](https://hydrocnhs.readthedocs.io/en/latest/?badge=latest)


# HydroCNHS
A Python Package of Hydrological Model for Coupled Natural Human Systems

<img src="https://github.com/philip928lin/HydroCNHS/blob/main/docs/figs/CAWS.png?raw=true" alt="Complex Adaptive Water System" width="500"/>

Modeling coupled natural human systems (CNHS) to inform water resources management policies or describe hydrological cycles in the Anthropocene has become popular in recent years. To fulfill this need, we develop a semi-distributed Hydrological model for modeling Coupled Natural Human Systems, HydroCNHS. HydroCNHS is an open-source Python package supporting four APIs for coupling purposes that enable users to integrate their human decision models (e.g., can be programmed with the agent-based modeling concept) into HydroCNHS. The four APIs, Dam API, RiverDiv API, Conveying API, and InSitu API, are designed to couple customized man-made infrastructures (i.e., defined as agents) like reservoirs, off-stream diversions, trans-basin aqueducts, and drainage systems, respectively. Each of the APIs has a unique coupling structure in the HydroCNHS to ensure within-subbasin and inter-subbasin (i.e., river) routing logic. Moreover, HydroCNHS use a single model file to organize input settings from the hydrological model and user-provided human models and allow parallel calibration.


## Install
```
pip install hydrocnhs
```

## User Manual & Example
Click [here](https://hydrocnhs.readthedocs.io/en/latest/?badge=latest)!

## How to cite?
Lin, C. Y., Yang, Y. C. E., & Wi S. (2022). HydroCNHS: A Python Package of Hydrological Model for Coupled Natural Human Systems, Journal of Water Resources Planning and Management. (Pending)

