"""Create an unscaled pairwise feature counting distinct SMS-active days."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "sms.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "interim" / "sms_pair_days.csv"
SECONDS_PER_DAY = 86_400
PAIR_COLUMNS = ["user_a", "user_b"]


def build_sms_pair_days(sms: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unordered SMS pair and its distinct active-day count."""
    required_columns = {"timestamp", "sender", "recipient"}
    missing_columns = required_columns.difference(sms.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required SMS columns: {missing}")

    if sms[list(required_columns)].isna().any().any():
        raise ValueError("SMS input contains missing values; no imputation was applied.")

    pair_events = sms[["timestamp", "sender", "recipient"]].copy()
    pair_events["user_a"] = pair_events[["sender", "recipient"]].min(axis=1)
    pair_events["user_b"] = pair_events[["sender", "recipient"]].max(axis=1)

    # The raw timestamps are elapsed seconds, so integer division assigns each
    # event to its elapsed study day without changing or scaling the feature.
    pair_events["texting_day"] = (
        pair_events["timestamp"] // SECONDS_PER_DAY
    ).astype("int64")

    pair_days = (
        pair_events.loc[
            pair_events["user_a"].lt(pair_events["user_b"]),
            PAIR_COLUMNS + ["texting_day"],
        ]
        .drop_duplicates()
        .groupby(PAIR_COLUMNS, as_index=False)
        .agg(days_texted=("texting_day", "nunique"))
        .sort_values(PAIR_COLUMNS, ignore_index=True)
    )

    if pair_days[PAIR_COLUMNS].duplicated().any():
        raise AssertionError("Output contains duplicate user pairs.")
    if not pair_days["days_texted"].ge(1).all():
        raise AssertionError("Every SMS-observed pair must have at least one texting day.")

    return pair_days


def main() -> None:
    sms = pd.read_csv(INPUT_FILE)
    pair_days = build_sms_pair_days(sms)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pair_days.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(pair_days):,} SMS pairs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
