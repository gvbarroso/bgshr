# Diversity reduction due to linked deleterious mutation

This is ``bgshr``, a package for computing expected reduction in diversity due
to linked deleterious mutations.

### Installation

For now, clone and build the package locally:
```
git clone git@github.com:apragsdale/bgshr.git
```

Then `cd` into the cloned directory and
```
pip install .
```

### Usage

Calculation of diversity reduction (B values) uses a lookup table of two-locus
predictions for relative reduction in diversity, and combines the effects
across many loci multiplicatively. A first-order correction for interference
can be applied by adjusting parameters based on the local rescaling of
effective population sizes.

In addition to the lookup table, we need an array of "elements" that allow for
selected mutations, the per-base deleterious mutation rate, and a recombination
map or per-base recombination rate.

Should include an example notebook to document usage.

Future features:

- DFE
- mutation map
- multiple element types (with their own DFEs)
- likelihood function
- inference
