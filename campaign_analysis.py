"""
campaign_analysis.py

Peak Season Campaign Success Probability Forecasting
------------------------------------------------------
Analyzes online shopping session data to answer three business questions
for a returning-customer marketing campaign:

1. What are the November/December purchase rates by customer type?
2. What is the strongest correlation in time spent across page types
   for sessions in November/December?
3. If a campaign boosts the returning-customer purchase rate by 15%,
   what is the probability of at least 100 sales out of 500 sessions?

Usage:
    python src/campaign_analysis.py

Expects `data/online_shopping_session_data.csv` with the schema described
in data/README.md.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

DATA_PATH = "data/online_shopping_session_data.csv"
OUTPUT_IMAGE_PATH = "images/campaign_binomial_distribution.png"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the shopping session dataset."""
    return pd.read_csv(path)


def filter_peak_season(df: pd.DataFrame) -> pd.DataFrame:
    """Return only November and December sessions."""
    return df[df["Month"].isin(["Nov", "Dec"])]


def calculate_purchase_rates(shopping_nov_dec: pd.DataFrame) -> dict:
    """
    Calculate purchase rate by customer type.

    Returns
    -------
    dict
        {"Returning_Customer": float, "New_Customer": float}
    """
    counts = shopping_nov_dec.groupby(["CustomerType"])["Purchase"].value_counts()

    total_new = counts["New_Customer"].sum()
    total_returning = counts["Returning_Customer"].sum()

    purchased_new = counts[("New_Customer", 1)]
    purchased_returning = counts[("Returning_Customer", 1)]

    return {
        "Returning_Customer": purchased_returning / total_returning,
        "New_Customer": purchased_new / total_new,
    }


def calculate_top_correlation(shopping_nov_dec: pd.DataFrame) -> dict:
    """
    Calculate pairwise Pearson correlations across duration columns
    and return the strongest pair.

    Returns
    -------
    dict
        {"pair": (col_a, col_b), "correlation": float}
    """
    duration_cols = [
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
    ]

    corr_matrix = shopping_nov_dec[duration_cols].corr()

    best_pair, best_corr = None, 0.0
    for i, col_a in enumerate(duration_cols):
        for col_b in duration_cols[i + 1 :]:
            value = corr_matrix.loc[col_a, col_b]
            if abs(value) > abs(best_corr):
                best_pair, best_corr = (col_a, col_b), value

    return {"pair": best_pair, "correlation": best_corr}


def calculate_campaign_probability(
    current_rate: float,
    boost_multiplier: float = 1.15,
    n_sessions: int = 500,
    target_sales: int = 100,
) -> dict:
    """
    Model the campaign as a binomial distribution and calculate the
    probability of hitting at least `target_sales`.

    Returns
    -------
    dict with probability, expected sales, std dev, and 95% CI.
    """
    boosted_rate = current_rate * boost_multiplier

    prob_below_target = stats.binom.cdf(target_sales - 1, n_sessions, boosted_rate)
    prob_at_least_target = 1 - prob_below_target

    expected_sales = n_sessions * boosted_rate
    std_dev = np.sqrt(n_sessions * boosted_rate * (1 - boosted_rate))
    ci_lower, ci_upper = stats.binom.interval(0.95, n_sessions, boosted_rate)

    return {
        "boosted_rate": boosted_rate,
        "prob_at_least_target": prob_at_least_target,
        "expected_sales": expected_sales,
        "std_dev": std_dev,
        "confidence_interval_95": (ci_lower, ci_upper),
    }


def plot_binomial_distribution(
    n_sessions: int,
    p: float,
    target_sales: int,
    output_path: str = OUTPUT_IMAGE_PATH,
) -> None:
    """Save a bar chart of the binomial probability mass function."""
    k_values = np.arange(1, n_sessions + 1)
    pmf_values = stats.binom.pmf(k_values, n_sessions, p)

    plt.figure(figsize=(9, 5))
    plt.bar(k_values, pmf_values, color="steelblue", alpha=0.8)
    plt.axvline(target_sales, color="red", linestyle="--", label=f"sales={target_sales}")
    plt.xlabel("Number of sales")
    plt.ylabel("Probability")
    plt.title("Binomial Distribution of Campaign Sales Outcomes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    df = load_data()
    nov_dec = filter_peak_season(df)

    purchase_rates = calculate_purchase_rates(nov_dec)
    print("purchase_rates =", purchase_rates)

    top_correlation = calculate_top_correlation(nov_dec)
    print("top_correlation =", top_correlation)

    campaign = calculate_campaign_probability(
        current_rate=purchase_rates["Returning_Customer"]
    )
    prob_at_least_100_sales = campaign["prob_at_least_target"]
    print("prob_at_least_100_sales =", prob_at_least_100_sales)
    print(f"Expected sales: {campaign['expected_sales']:.1f}")
    print(f"Standard deviation: {campaign['std_dev']:.2f}")
    print(f"95% confidence interval: {campaign['confidence_interval_95']}")

    plot_binomial_distribution(
        n_sessions=500,
        p=campaign["boosted_rate"],
        target_sales=100,
    )
    print(f"Saved distribution chart to {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
