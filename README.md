# Setup



## Installation

### m1 specific steps

it seems that it is easier to install tensorflow properly first and then figure out jax

```bash
 conda create -n dl_jax python=3.9
 conda activate dl_jax
 # set up tf for m1
 conda install -c apple tensorflow-deps
 pip install tensorflow-metal
 pip install tensorflow-macos

 # pip will complain but requried for tensorflow
 pip install numpy --upgrade

 pip install tensorflow_probability==0.23

 # for m1 jax -- for some reason need to specify most recent release?
 # must use conda-forge channel for m1
 conda install jax==0.4.8 -c conda-forge

 # now that tensorflow and jax is installed properly we can install stgp
 pip install -e . 
```

### Conda ENV
```bash
    conda create -n dl_jax python=3.11
    conda activate dl_jax
    pip install -r requirements.txt
```

### Notebooks
```bash
	conda install ipykernel
	conda install -c conda-forge nb_conda_kernels
```