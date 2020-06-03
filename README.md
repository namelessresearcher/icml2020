# Node-Queries

This repository contains python code used for the experiments of the paper "Estimating the Number of Induced Subgraphs from Incomplete Data and Neighborhood Queries". 

## Prerequisites

All the procedures have been implemented in Python 2.7 using the SNAP platform (more information about installing Snap.py can be found [here](https://snap.stanford.edu/snappy/index.html)). [NumPy](https://numpy.org/) package is used for vector manipulation and [matplotlib](https://matplotlib.org/) for plotting.


## Explanation of Files

The contents are organized as follows.

* `functions.py`: contains the definitions of the sample generating procedures as well as the definitions of our edge and triangle estimators.
* `main.py`: contains an example of using the edge and triangle estimators.
* `plot.py`: contains the code used for creating the plots in the paper.
* `ego-gplus.txt`: is the dataset used. 

The other datasets used in the paper can be downloaded from [SNAP](https://snap.stanford.edu/data/index.html) and [network repository](http://networkrepository.com/).
