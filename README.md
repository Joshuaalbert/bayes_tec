[![Build Status](https://travis-ci.org/Joshuaalbert/bayes_tec.svg?branch=master)](https://travis-ci.org/Joshuaalbert/bayes_tec)

# bayes_tec
Bayesian model for directional and temporal TEC model for phase-wrapped phase screens

# Installation

1. Steup Conda environment

  - If you do not have `conda` install it from [here](https://anaconda.org/anaconda/conda)
  - Create conda environemnt
  ``` bash
  conda create -n bayestf python=3
  ```
  (You can change `bayestf` to whatever you want.)
  - Add the following alias to your `.bashrc` or equivalent:
  ``` bash
   echo alias bayestf='source activate bayestf' >> ~/.bashrc
   # open new terminal or source ~/.bashrc
  ```
  - Activate your conda environment (which you must do every time you open a new terminal and want to work on bayes TEC stuff.)
  ``` bash
  bayestf
  # or if you did not add the alias
  source activate bayestf
  ```
2. Install
  - Install `GPflow` 'develop' branch
  ``` bash
  cd ~/git
  git clone https://github.com/GPflow/GPflow.git
  cd GPflow; git checkout develop; python setup.py install; cd ..
  ```
  - This repo
  ``` bash
  git clone https://github.com/Joshuaalbert/bayes_tec.git
  cd bayes_tec; pip install -r pip-requirements.txt
  ```
  - Install the package
  ``` bash
  python setup.py install
  ```
3. Run tests (broken at the moment)
  ``` bash
  pytest
  ```
