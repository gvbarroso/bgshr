# Diversity reduction due to linked deleterious mutation

This is ``bgshr``, a package for computing the expected reduction in diversity 
due to linked deleterious mutations.

### Installation

Dependencies are listed in [requirements.txt](requirements.txt). `bgshr` can be 
installed directly from github using `pip`. At the moment, the `dev` branch has 
the most complete suite of features. In this sample we create a Python virtual 
environment and install the `dev` branch:
```
python -m venv ~/path/to/venv
source ~/path/to/venv/bin/activate
pip install git+https://github.com/apragsdale/bgshr.git@dev
```

Alternatively, we can clone and build the package locally:
```
git clone https://github.com/apragsdale/bgshr.git
```

Then `cd` into the cloned directory and perform a local installation. If we 
wish, we can switch to the dev branch with `git checkout dev` before installing.
```
cd bgshr
pip install .
```

### Dependencies

If `bgshr` was cloned locally, we can install dependencies listed in 
[requirements.txt](requirements.txt) using:
```
pip install -r requirements.txt
```

Minimal dependencies are the ubiquitous `numpy`, `scipy` and `pandas`. It is
highly useful to also have `ipython` installed. `jupyter` is necessary to run
the example notebooks in [examples/](examples/).


### Usage

Calculation of diversity reduction (B values) uses a lookup table of two-locus
predictions for relative reduction in diversity, and combines the effects
across many loci multiplicatively. A first-order correction for interference
can be applied by adjusting parameters based on the local rescaling of
effective population sizes. We generate lookup tables using the `moments++`
mode,l available here: https://github.com/gvbarroso/momentspp/tree/main.

In addition to the lookup table, we need one or more arrays of constrained
genomic elements that experience selected mutations, an array indicating 
estimates of the per-base deleterious mutation rate, and a recombination map or 
per-base recombination rate. Along with these data we require parameters for
the relevant gamma-distributed DFEs, which may include a point mass of neutral
mutations.

Example usage is shown in [human-chr22-example.ipynb](examples/human-chr22-example.ipynb).

Current features:
- Discretized gamma DFEs
- Handling of mutation maps
- Allows multiple element types (with their own DFEs)
- Likelihood function
- Inference of drift-effective Ne 

