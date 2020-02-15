import numpy
from sklearn import mixture

FILENAME = "mcdonalds-normalized-data.tsv"

# Note: you'll have to remove the last "name" column in the file (or
# some other such thing), so that all the columns are numeric.
x = numpy.loadtxt(open(FILENAME, "rb"), delimiter="\t", skiprows=1, usecols=range(0, 14))
dpgmm = mixture.BayesianGaussianMixture(n_components=25, weight_concentration_prior_type='dirichlet_process',init_params="random", max_iter=100)
dpgmm.fit(x)
clusters = dpgmm.predict(x)
print score
