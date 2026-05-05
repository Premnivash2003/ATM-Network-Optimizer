"""
ATM-Net Optimizer — Module 1: Synthetic Data Generator
=======================================================
Generates realistic ATM transaction data for a simulated city.

Design decisions:
- 20 ATMs spread across 3 natural city zones (downtown, suburbs, shopping)
- 500 customers, each assigned a "home zone" and 2-3 preferred ATMs
- ~9000 transactions with realistic timestamps, amounts, and usage patterns
- Cluster structure ensures the SNA graph will have meaningful communities

Output: data/transactions.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ─── City Zone Definitions ───────────────────────────────────────────────────
# Each zone has a centroid (lat, lon) and a spread radius (in degrees ~km)
ZONES = {
    "downtown": {
        "center": (12.9716, 80.2209),   # Chennai-like coordinates
        "spread": 0.015,
        "n_atms": 7,
        "label": "Downtown",
    },
    "suburbs": {
        "center": (12.9450, 80.2450),
        "spread": 0.020,
        "n_atms": 7,
        "label": "Suburbs",
    },
    "shopping": {
        "center": (13.0000, 80.2100),
        "spread": 0.012,
        "n_atms": 6,
        "label": "Shopping District",
    },
}

# Transaction amount ranges by time of day (INR amounts)
AMOUNT_PROFILES = {
    "morning":   {"mean": 3000, "std": 1000},   # 06–11h: moderate
    "afternoon": {"mean": 2000, "std":  800},   # 11–17h: smaller
    "evening":   {"mean": 4500, "std": 1500},   # 17–21h: largest (salary day)
    "night":     {"mean": 2500, "std":  900},   # 21–06h: emergency
}


def generate_atm_locations(zones: dict) -> pd.DataFrame:
    """
    Place ATMs across zones using a Gaussian distribution around each
    zone center, so they form natural spatial clusters.
    """
    records = []
    atm_id = 1

    for zone_key, zone in zones.items():
        lat_c, lon_c = zone["center"]
        spread = zone["spread"]
        n = zone["n_atms"]

        for _ in range(n):
            lat = np.random.normal(lat_c, spread)
            lon = np.random.normal(lon_c, spread)
            records.append({
                "atm_id":   f"ATM_{atm_id:03d}",
                "zone":     zone_key,
                "zone_label": zone["label"],
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
            })
            atm_id += 1

    df = pd.DataFrame(records)
    print(f"  ✓ Generated {len(df)} ATMs across {len(zones)} zones")
    return df


def assign_customer_atms(atm_df: pd.DataFrame, n_customers: int = 500) -> dict:
    """
    Assign each customer 2–3 preferred ATMs.

    Strategy:
    - Each customer belongs to a primary zone (home/work area)
    - They use 1–2 ATMs from their primary zone most often
    - With 20% probability they also use 1 ATM from a neighboring zone
      (simulates commute/shopping behavior)

    Returns a dict: customer_id → list of (atm_id, weight) tuples
    """
    zone_names = list(ZONES.keys())
    customer_atms = {}

    for cust_num in range(1, n_customers + 1):
        cust_id = f"CUST_{cust_num:04d}"

        # Pick primary zone proportional to ATM density
        primary_zone = np.random.choice(zone_names)
        zone_atms = atm_df[atm_df["zone"] == primary_zone]["atm_id"].tolist()

        # 1–2 ATMs from primary zone
        n_primary = np.random.choice([1, 2], p=[0.4, 0.6])
        n_primary = min(n_primary, len(zone_atms))
        primary_picks = random.sample(zone_atms, n_primary)

        selected = {atm: np.random.uniform(0.5, 1.0) for atm in primary_picks}

        # 20% chance of cross-zone ATM (commuter behavior)
        if random.random() < 0.20:
            other_zones = [z for z in zone_names if z != primary_zone]
            secondary_zone = random.choice(other_zones)
            sec_atms = atm_df[atm_df["zone"] == secondary_zone]["atm_id"].tolist()
            sec_pick = random.choice(sec_atms)
            selected[sec_pick] = np.random.uniform(0.1, 0.4)  # lower weight

        # Normalize weights so they sum to 1 (used as transaction probabilities)
        total = sum(selected.values())
        customer_atms[cust_id] = {k: v / total for k, v in selected.items()}

    print(f"  ✓ Assigned ATM preferences to {n_customers} customers")
    return customer_atms


def get_amount(hour: int) -> float:
    """Return a random transaction amount based on the time of day."""
    if 6 <= hour < 11:
        profile = AMOUNT_PROFILES["morning"]
    elif 11 <= hour < 17:
        profile = AMOUNT_PROFILES["afternoon"]
    elif 17 <= hour < 21:
        profile = AMOUNT_PROFILES["evening"]
    else:
        profile = AMOUNT_PROFILES["night"]

    amount = np.random.normal(profile["mean"], profile["std"])
    # ATMs dispense in multiples of 500 INR; clamp to realistic range
    amount = max(500, min(20000, round(amount / 500) * 500))
    return float(amount)


def generate_transactions(
    atm_df: pd.DataFrame,
    customer_atms: dict,
    n_transactions: int = 9000,
    start_date: str = "2024-01-01",
    end_date: str = "2024-06-30",
) -> pd.DataFrame:
    """
    Generate individual ATM transactions.

    Each transaction is sampled by:
    1. Picking a random customer
    2. Picking one of their preferred ATMs (weighted by preference)
    3. Picking a random timestamp in the date range
    4. Generating a time-appropriate withdrawal amount
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
    date_range_seconds = int((end_dt - start_dt).total_seconds())

    customer_ids = list(customer_atms.keys())

    # Build ATM coordinate lookup for fast join
    atm_coords = atm_df.set_index("atm_id")[["latitude", "longitude"]].to_dict("index")

    records = []
    for _ in range(n_transactions):
        # 1. Pick customer
        cust_id = random.choice(customer_ids)
        atm_prefs = customer_atms[cust_id]

        # 2. Pick ATM weighted by preference
        atms   = list(atm_prefs.keys())
        weights = list(atm_prefs.values())
        atm_id = random.choices(atms, weights=weights, k=1)[0]

        # 3. Timestamp — skewed toward business hours using a truncated normal
        offset_sec = int(np.random.uniform(0, date_range_seconds))
        ts = start_dt + timedelta(seconds=offset_sec)

        # Bias toward daytime: resample if hour is 1–5 AM (low-traffic window)
        if 1 <= ts.hour <= 5:
            if random.random() < 0.70:          # 70% chance to skip dead hours
                continue

        # 4. Amount
        amount = get_amount(ts.hour)

        records.append({
            "customer_id": cust_id,
            "atm_id":      atm_id,
            "latitude":    atm_coords[atm_id]["latitude"],
            "longitude":   atm_coords[atm_id]["longitude"],
            "timestamp":   ts.strftime("%Y-%m-%d %H:%M:%S"),
            "amount":      amount,
        })

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    print(f"  ✓ Generated {len(df):,} transactions "
          f"({start_date} → {end_date})")
    return df


def save_data(atm_df: pd.DataFrame, tx_df: pd.DataFrame, data_dir: str = "data") -> None:
    """Persist both datasets to CSV."""
    os.makedirs(data_dir, exist_ok=True)

    atm_path = os.path.join(data_dir, "atm_locations.csv")
    tx_path  = os.path.join(data_dir, "transactions.csv")

    atm_df.to_csv(atm_path, index=False)
    tx_df.to_csv(tx_path,  index=False)

    print(f"  ✓ Saved ATM locations  → {atm_path}")
    print(f"  ✓ Saved transactions   → {tx_path}")


def print_summary(atm_df: pd.DataFrame, tx_df: pd.DataFrame) -> None:
    """Print a quick sanity-check summary."""
    print("\n" + "=" * 50)
    print("  DATA GENERATION SUMMARY")
    print("=" * 50)
    print(f"  ATMs            : {len(atm_df)}")
    print(f"  Zones           : {atm_df['zone'].nunique()}")
    print(f"  Transactions    : {len(tx_df):,}")
    print(f"  Unique customers: {tx_df['customer_id'].nunique()}")
    print(f"  Date range      : {tx_df['timestamp'].min()[:10]} "
          f"→ {tx_df['timestamp'].max()[:10]}")
    print(f"  Avg amount (INR): ₹{tx_df['amount'].mean():,.0f}")
    print(f"  Avg tx/customer : {len(tx_df) / tx_df['customer_id'].nunique():.1f}")
    print()

    print("  Transactions per zone:")
    zone_map = atm_df.set_index("atm_id")["zone_label"].to_dict()
    tx_df["zone"] = tx_df["atm_id"].map(zone_map)
    for zone, count in tx_df["zone"].value_counts().items():
        print(f"    {zone:<25}: {count:,}")
    print("=" * 50)


# ─── Entry Point ─────────────────────────────────────────────────────────────
def run(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: generate → validate → save.
    Returns (atm_df, tx_df) so other modules can import and call this directly.
    """
    print("\n[Module 1] Generating synthetic ATM data...")

    atm_df       = generate_atm_locations(ZONES)
    customer_atms = assign_customer_atms(atm_df, n_customers=500)
    tx_df        = generate_transactions(
                       atm_df,
                       customer_atms,
                       n_transactions=10_000,   # generate slightly more; dead-hour
                                                 # filtering brings it to ~8500-9500
                   )

    # Validation guards
    assert len(atm_df) == 20,           "Expected exactly 20 ATMs"
    assert tx_df["customer_id"].nunique() >= 400, "Too few unique customers"
    assert 8_000 <= len(tx_df) <= 10_000,         "Transaction count out of range"
    assert tx_df.isnull().sum().sum() == 0,        "Null values found"

    save_data(atm_df, tx_df, data_dir)
    print_summary(atm_df, tx_df)

    return atm_df, tx_df


if __name__ == "__main__":
    # Run from project root:  python src/data_generator.py
    run(data_dir="data")
