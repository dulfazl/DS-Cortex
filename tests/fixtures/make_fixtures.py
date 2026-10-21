"""Regenerate the tiny router/audit fixtures: python -m tests.fixtures.make_fixtures"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


def main() -> None:
    rng = np.random.default_rng(0)
    n = 200
    # 1 classification (churn)
    age = rng.integers(18, 80, n)
    income = rng.normal(50000, 15000, n).round(2)
    plan = rng.choice(["basic", "pro", "team"], n)
    logit = -1 + 0.03 * (50 - age) + (plan == "basic") * 0.8
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"age": age, "income": income, "plan": plan, "churn": churn})
    df.loc[::17, "income"] = np.nan
    df.to_csv(OUT / "classification.csv", index=False)
    # 2 regression (house price)
    sqft = rng.normal(1500, 400, n).round()
    rooms = rng.integers(1, 6, n)
    price = (sqft * 150 + rooms * 10000 + rng.normal(0, 20000, n)).round(2)
    pd.DataFrame({"sqft": sqft, "rooms": rooms, "neighborhood": rng.choice(["north", "south", "east"], n),
                  "price": price}).to_csv(OUT / "regression.csv", index=False)
    # 3 clustering (customers, no target)
    centers = np.array([[20, 200], [60, 800], [40, 1500]])
    lab = rng.integers(0, 3, n)
    pts = centers[lab] + rng.normal(0, [5, 80], (n, 2))
    pd.DataFrame({"visits_per_month": pts[:, 0].round(1), "avg_basket": pts[:, 1].round(2),
                  "tenure_months": rng.integers(1, 60, n)}).to_csv(OUT / "clustering.csv", index=False)
    # 4 anomaly (transactions, no label)
    amt = rng.lognormal(3, 0.5, n).round(2)
    amt[:6] = [5000, 7200, 6100, 9000, 8800, 7700]
    pd.DataFrame({"amount": amt, "hour": rng.integers(0, 24, n),
                  "merchant_category": rng.choice(["food", "travel", "retail"], n)}).to_csv(OUT / "anomaly.csv", index=False)
    # 5 time series (monthly sales)
    dates = pd.date_range("2015-01-01", periods=96, freq="MS")
    t = np.arange(96)
    sales = (1000 + 5 * t + 120 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 30, 96)).round(1)
    pd.DataFrame({"month": dates.strftime("%Y-%m-%d"), "sales": sales,
                  "promo": rng.integers(0, 2, 96)}).to_csv(OUT / "timeseries.csv", index=False)
    # 6 offline RL / bandit (logged decisions)
    m = 400
    x1, x2 = rng.normal(size=m), rng.normal(size=m)
    action = rng.choice(["offer_a", "offer_b", "offer_c"], m, p=[0.5, 0.3, 0.2])
    prop = np.select([action == "offer_a", action == "offer_b"], [0.5, 0.3], 0.2)
    best = np.where(x1 > 0, "offer_b", "offer_a")
    reward = (rng.random(m) < np.where(action == best, 0.6, 0.2)).astype(int)
    pd.DataFrame({"user_age_z": x1.round(3), "user_spend_z": x2.round(3), "action": action,
                  "propensity": prop, "reward": reward}).to_csv(OUT / "bandit.csv", index=False)
    # 7 two plausible targets (vague)
    pd.DataFrame({"sqft": sqft, "price": price, "rent": (price / 200 + rng.normal(0, 50, n)).round(2)}).to_csv(
        OUT / "two_targets.csv", index=False)


if __name__ == "__main__":
    main()
