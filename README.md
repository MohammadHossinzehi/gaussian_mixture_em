# mixem

Gaussian mixture models built from first principles in NumPy: expectation maximisation with four covariance structures, a variational Bayesian mixture that decides for itself how many clusters the data supports, k means++ initialisation, and BIC/AIC model selection. No SciPy, no scikit-learn; the only dependency is NumPy.

```
$ python -m mixem demo
  k  covariance             BIC
  3  full              5719.279  <- best
  4  full              5752.326
  ...
variational Bayes, 10 components offered, 3 kept
```

## Why this exists

A Gaussian mixture is the workhorse behind soft clustering, density estimation, anomaly scoring and speaker models, but most people only ever meet it as `sklearn.mixture.GaussianMixture(...).fit(X)`. This project is the version you can read end to end. Every update is the closed form from the textbook (Bishop, *Pattern Recognition and Machine Learning*, chapters 9 and 10), written so the code and the equations line up, and every claim the maths makes is checked by a test.

## Features

| Module | What it does |
| --- | --- |
| `mixem.GaussianMixture` | Maximum likelihood EM with `full`, `tied`, `diag` and `spherical` covariances, multiple restarts, `reg_covar` regularisation, `predict`, `predict_proba`, `score_samples`, `sample`, `bic`, `aic` |
| `mixem.VariationalGaussianMixture` | Full covariance Bayesian mixture with Dirichlet, Normal and Wishart priors fitted by coordinate ascent; reports the complete evidence lower bound every iteration and prunes components it does not need |
| `mixem.select_model` | Grid search over component count and covariance type by BIC or AIC with a printable ranking table |
| `mixem.kmeans`, `mixem.kmeans_plusplus` | D squared seeding and Lloyd iterations, used for initialisation and usable on their own |
| `python -m mixem` | CLI: `fit`, `select`, `vb` on any CSV, plus a self contained `demo` |

## Quick start

```bash
git clone https://github.com/MohammadHossinzehi/gaussian_mixture_em.git
cd gaussian_mixture_em
pip install numpy pytest

python -m mixem demo                          # built in synthetic data
python examples/compare_covariances.py        # covariance types side by side
python -m pytest                              # 37 tests, a few seconds
```

On your own data (a CSV of numeric columns, header row optional):

```bash
python -m mixem fit data.csv --k 4 --cov diag --labels-out labels.txt
python -m mixem select data.csv --kmax 10 --criterion bic
python -m mixem vb data.csv --kmax 15 --alpha0 0.001
```

As a library:

```python
import numpy as np
from mixem import GaussianMixture, VariationalGaussianMixture, select_model

gm = GaussianMixture(3, covariance_type="full", n_init=5, seed=0).fit(X)
gm.predict_proba(X)      # soft assignments, rows sum to 1
gm.score_samples(X)      # log p(x), useful for anomaly scores
gm.sample(1000)          # new data from the fitted density

res = select_model(X, k_range=range(1, 9))
print(res.summary())

vb = VariationalGaussianMixture(n_components=15, alpha0=1e-3).fit(X)
vb.effective_components()  # how many clusters survived
```

`pip install .` also works and installs a `mixem` console script.

## How it works

**E step.** Every density is evaluated in log space. For full and tied covariances the code factors each covariance as `L L^T`, solves `L y = x - mu`, and reads the Mahalanobis term as `|y|^2` and the log determinant as twice the sum of `log diag(L)`. Responsibilities come from a stable log sum exp, so a point a million units from every component still gets a finite score instead of `NaN`.

**M step.** Weights, means and covariances have closed forms for each covariance type. `reg_covar` is added to the diagonal so a component that collapses onto a single point cannot drive the likelihood to infinity, which is the classic failure mode of maximum likelihood mixtures.

**Variational Bayes.** The Bayesian model puts a Dirichlet prior on the weights and a Normal Wishart prior on each component. Coordinate ascent alternates between updating `q(Z)` and updating `q(pi) q(mu, Lambda)`, and the full ELBO (PRML equations 10.70 to 10.77) is recomputed every iteration. With a small `alpha0` the posterior drives unneeded components to zero weight, so you can offer 15 components and get back the 3 the data actually contains.

**Special functions.** The digamma function is implemented with the recurrence `psi(x) = psi(x + 1) - 1/x` up to `x >= 10` followed by the asymptotic series, plus reflection for negative arguments. It agrees with the harmonic number identity to 1e-12.

## Testing approach

The tests check properties the mathematics guarantees rather than snapshot numbers:

* **Exact evidence oracle.** With one component the variational posterior *is* the exact conjugate posterior, so the ELBO must equal the closed form Normal Wishart log marginal likelihood. The test checks this in 1, 2 and 4 dimensions to nine significant digits, which pins down every one of the seven ELBO terms at once.
* **Monotonicity.** EM log likelihood (all four covariance types) and the variational ELBO must never decrease. k means inertia must never increase.
* **The M step is a true maximiser.** For random responsibilities, the closed form parameters beat 40 random perturbations of themselves on the expected complete data log likelihood.
* **Naive reference.** The Cholesky based log density is compared against `inv` and `det` evaluated directly, and `score_samples` against an explicit sum of weighted densities.
* **Brute force optimum.** On a 9 point problem, k means with restarts matches the best of every possible labelling.
* **Recovery.** Synthetic clusters are recovered (means, covariances, weights, labels); BIC picks the true k; variational Bayes keeps exactly 3 of 10 components; with plenty of data the Bayesian and maximum likelihood fits agree.
* **Sampling.** 60,000 draws from a fitted model reproduce its weights, means and covariances.

## Design decisions

* **NumPy only.** `logsumexp`, `digamma` and `gammaln` live in `mixem/_special.py` so the project installs anywhere NumPy does.
* **Natural shapes for covariances.** `covariances_` is `(K, d, d)`, `(d, d)`, `(K, d)` or `(K,)` depending on type, matching how the parameters are actually stored; `full_covariances()` expands to `(K, d, d)` when you want a uniform view.
* **Restarts keep the best likelihood**, not the last one, and the full per iteration history is exposed as `history_` or `elbo_history_` so convergence can be inspected or plotted.
* **Readable over clever.** Loops over components are kept where they make the equations obvious; K is small in practice and the heavy lifting is still vectorised over samples.

## Project layout

```
mixem/
  _special.py      logsumexp, digamma, gammaln
  kmeans.py        k means++ and Lloyd
  gmm.py           EM GaussianMixture, log densities, M step, BIC/AIC
  bayesian.py      VariationalGaussianMixture and its ELBO
  selection.py     select_model over k and covariance type
  __main__.py      command line interface
examples/
  compare_covariances.py
tests/             37 pytest tests
```

## License

MIT
