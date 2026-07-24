# Methodology

Technical detail behind each of the three analyses in
`src/campaign_analysis.py`.

---

## 1. Purchase Rate by Customer Type

```
purchase_rate = (sessions with Purchase = 1) / (total sessions)
```

Sessions are first filtered to `Month in ["Nov", "Dec"]`, then grouped by
`CustomerType` before the rate is computed for each group. This isolates
peak-season behavior, since a full-year average would dilute the effect of
the busiest shopping months.

---

## 2. Strongest Correlation in Page-Duration Metrics

Pearson correlation (`pandas.DataFrame.corr`) is computed pairwise across:

- `Administrative_Duration`
- `Informational_Duration`
- `ProductRelated_Duration`

The pair with the largest absolute correlation coefficient is reported.
Pearson's r measures linear association; values are interpreted as:

| \|r\| range | Strength |
|---|---|
| 0.0 – 0.3 | Weak |
| 0.3 – 0.7 | Moderate |
| 0.7 – 1.0 | Strong |

---

## 3. Campaign Success Probability (Binomial Model)

**Why binomial:** each of the 500 campaign sessions is treated as an
independent trial with a binary outcome (purchase / no purchase) and a
constant probability of success — the exact conditions the binomial
distribution assumes.

### Step 1 — Boosted purchase rate

```python
boosted_rate = current_rate * 1.15   # 15% multiplicative increase
```

Note this is a **multiplicative** boost (rate × 1.15), not an additive one
(rate + 15 percentage points).

### Step 2 — Probability of at least 100 sales

```python
from scipy import stats

prob_below_100 = stats.binom.cdf(99, n=500, p=boosted_rate)
prob_at_least_100 = 1 - prob_below_100
```

`binom.cdf(k, n, p)` returns `P(X ≤ k)`, so `P(X ≥ 100)` requires the
complement of `P(X ≤ 99)`.

### Step 3 — Expected value and standard deviation

```python
import numpy as np

expected_sales = n * boosted_rate
std_dev = np.sqrt(n * boosted_rate * (1 - boosted_rate))
```

These follow directly from the binomial distribution's mean (`n·p`) and
variance (`n·p·(1-p)`).

### Step 4 — 95% confidence interval

Two equivalent approaches are used:

**A. Normal approximation** (valid for large `n`):

```python
ci_lower = expected_sales - 1.96 * std_dev
ci_upper = expected_sales + 1.96 * std_dev
```

`1.96` is the z-score corresponding to 95% coverage of a normal
distribution (`scipy.stats.norm.ppf(0.975)`).

**B. Exact binomial interval** (used in the script, via SciPy):

```python
ci_lower, ci_upper = stats.binom.interval(0.95, n, boosted_rate)
```

This queries the binomial distribution's quantile function directly
(equivalent to `stats.binom.ppf(0.025, n, p)` and
`stats.binom.ppf(0.975, n, p)`), and is more precise than the normal
approximation for smaller `n` or extreme `p`.

### Interpreting the interval

The 95% CI is the range within which sales would fall in 95 out of 100
repeated runs of an equivalent 500-session campaign. It is used here for
**inventory and staffing planning**: rather than provisioning for exactly
the 100-sale target, operations can provision for the full likely range,
reducing the risk of both stockouts and excess inventory.

### Validating the interval

The interval can be sanity-checked with a Monte Carlo simulation:

```python
import numpy as np

outcomes = np.random.binomial(n, boosted_rate, size=100_000)
share_within_ci = np.mean((outcomes >= ci_lower) & (outcomes <= ci_upper))
# share_within_ci should be ≈ 0.95
```

---

## Summary of Tools

| Tool | Used for |
|---|---|
| `pandas` | Filtering, grouping, `.corr()` |
| `numpy` | `sqrt`, array operations for std dev |
| `scipy.stats.binom` | `.cdf()`, `.pmf()`, `.interval()` — the binomial distribution itself |
| `matplotlib` | Plotting the probability mass function |
