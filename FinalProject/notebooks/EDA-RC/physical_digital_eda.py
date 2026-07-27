"""Visualization helpers for the Copenhagen Networks Study EDA notebooks.

This restored source uses the analysis-ready dyad table in ``data/processed``
and the unchanged interaction files in ``data/raw``. Bluetooth observations
represent co-presence, not confirmed friendship.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


FIVE_MINUTES = 300
DAY_SECONDS = 86_400
DEFAULT_RSSI_THRESHOLD = -80
DEFAULT_MIN_CLOSE_INTERVALS = 12
DEFAULT_MIN_CLOSE_DAYS = 3


@dataclass
class EDAData:
    project_dir: Path
    figures_dir: Path
    dyads: pd.DataFrame
    availability: pd.DataFrame
    bluetooth: pd.DataFrame
    close_bluetooth: pd.DataFrame
    calls: pd.DataFrame
    sms: pd.DataFrame
    participants: np.ndarray
    summary: pd.Series


def _read_raw(raw_dir: Path, filename: str) -> pd.DataFrame:
    frame = pd.read_csv(raw_dir / filename)
    frame.columns = [column.lstrip("# ").strip() for column in frame.columns]
    return frame


def _canonicalize(frame: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    result = frame.copy()
    result["u"] = np.minimum(result[source], result[target]).astype("int16")
    result["v"] = np.maximum(result[source], result[target]).astype("int16")
    return result


def load_and_prepare(
    project_dir: str | Path | None = None,
    **_: object,
) -> EDAData:
    """Load the recovered processed features and raw event data."""
    project = (
        Path(project_dir)
        if project_dir is not None
        else Path(__file__).resolve().parents[1]
    )
    raw_dir = project / "data" / "raw"
    processed_dir = project / "data" / "processed"
    figures_dir = project / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dyad_path = processed_dir / "physical_digital_dyads.csv"
    availability_path = processed_dir / "participant_availability_by_day.csv"
    if not dyad_path.exists() or not availability_path.exists():
        raise FileNotFoundError(
            "The processed EDA tables are missing. Expected "
            f"{dyad_path} and {availability_path}."
        )

    dyads = pd.read_csv(dyad_path)
    availability = pd.read_csv(availability_path, index_col=0)
    availability.index = availability.index.astype(int)
    availability.columns = availability.columns.astype(int)

    bluetooth = _read_raw(raw_dir, "bt_symmetric.csv")
    valid = bluetooth.loc[
        (bluetooth["user_b"] >= 0)
        & (bluetooth["user_a"] != bluetooth["user_b"])
    ].copy()
    valid = _canonicalize(valid, "user_a", "user_b")
    valid["day"] = (valid["timestamp"] // DAY_SECONDS).astype("int8")
    valid["hour"] = (
        (valid["timestamp"] % DAY_SECONDS) // 3600
    ).astype("int8")
    valid["weekday"] = ((valid["day"] + 6) % 7).astype("int8")
    valid["close"] = valid["rssi"] >= DEFAULT_RSSI_THRESHOLD
    close = valid.loc[valid["close"]].copy()
    close["weekend"] = (close["day"] % 7).isin([0, 6])

    calls = _read_raw(raw_dir, "calls.csv")
    sms = _read_raw(raw_dir, "sms.csv")
    participants = np.sort(bluetooth["user_a"].unique())

    summary_path = processed_dir / "eda_summary.csv"
    if summary_path.exists():
        summary_frame = pd.read_csv(summary_path, index_col=0)
        summary = summary_frame.iloc[:, 0]
        summary.name = "value"
    else:
        summary = pd.Series(
            {
                "observation_days": 28,
                "participants": len(participants),
                "possible_dyads": len(dyads),
                "bluetooth_rows": len(bluetooth),
                "facebook_edges_in_universe": int(dyads["facebook"].sum()),
                "call_rows": len(calls),
                "sms_rows": len(sms),
                "physical_ties": int(dyads["physical_tie"].sum()),
            },
            name="value",
        )

    return EDAData(
        project_dir=project,
        figures_dir=figures_dir,
        dyads=dyads,
        availability=availability,
        bluetooth=valid,
        close_bluetooth=close,
        calls=calls,
        sms=sms,
        participants=participants,
        summary=summary,
    )


def _save(
    fig: plt.Figure,
    data: EDAData,
    filename: str,
    tight: bool = True,
) -> plt.Figure:
    if tight:
        fig.tight_layout()
    fig.savefig(data.figures_dir / filename, dpi=180, bbox_inches="tight")
    return fig


def plot_availability(data: EDAData) -> plt.Figure:
    ordered = data.availability.loc[
        data.availability.mean(axis=1).sort_values().index
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [4, 1]}
    )
    sns.heatmap(
        ordered,
        cmap="viridis",
        vmin=0,
        vmax=1,
        yticklabels=False,
        cbar_kws={"label": "Fraction of 5-minute bins available"},
        ax=axes[0],
    )
    axes[0].set(
        title="Bluetooth data availability by participant and relative day",
        xlabel="Relative day (day 0 is Sunday)",
        ylabel="Participants, sorted by mean availability",
    )
    means = ordered.mean(axis=1)
    sns.histplot(means, bins=20, color="#4C78A8", ax=axes[1])
    axes[1].axvline(
        means.median(),
        color="#E45756",
        linestyle="--",
        label=f"Median = {means.median():.1%}",
    )
    axes[1].set(
        title="Participant availability",
        xlabel="Available fraction",
        ylabel="Participants",
        xlim=(0, 1),
    )
    axes[1].legend()
    return _save(fig, data, "01_bluetooth_availability.png")


def plot_temporal_activity(data: EDAData) -> plt.Figure:
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    close = data.close_bluetooth

    def heat(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame.groupby(["weekday", "hour"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(index=range(7), columns=range(24), fill_value=0)
        )

    endpoints = pd.concat(
        [
            close[["weekday", "hour", "u"]].rename(columns={"u": "user"}),
            close[["weekday", "hour", "v"]].rename(columns={"v": "user"}),
        ],
        ignore_index=True,
    )
    active = (
        endpoints.groupby(["weekday", "hour"], observed=True)["user"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )

    def communication(frame: pd.DataFrame) -> pd.DataFrame:
        day = frame["timestamp"] // DAY_SECONDS
        prepared = frame.assign(
            weekday=(day + 6) % 7,
            hour=(frame["timestamp"] % DAY_SECONDS) // 3600,
        )
        return heat(prepared)

    matrices = [
        (heat(close), "Close Bluetooth observations"),
        (active, "Distinct active participants"),
        (communication(data.sms), "SMS messages"),
        (communication(data.calls), "Calls"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
    for axis, (matrix, title) in zip(axes.flat, matrices):
        sns.heatmap(
            matrix,
            cmap="mako",
            yticklabels=weekday_names,
            cbar_kws={"label": "Count across four weeks"},
            ax=axis,
        )
        axis.set(title=title, xlabel="Hour of day", ylabel="")
    return _save(fig, data, "02_temporal_activity_heatmaps.png")


def plot_rssi_and_threshold_sensitivity(data: EDAData) -> plt.Figure:
    fb_pairs = set(
        map(
            tuple,
            data.dyads.loc[data.dyads["facebook"] == 1, ["u", "v"]].to_numpy(),
        )
    )
    records = []
    for threshold in np.arange(-90, -64, 5):
        selected = data.bluetooth.loc[data.bluetooth["rssi"] >= threshold]
        grouped = (
            selected.groupby(["u", "v"], observed=True)
            .agg(intervals=("day", "size"), days=("day", "nunique"))
            .reset_index()
        )
        ties = grouped.loc[
            (grouped["intervals"] >= DEFAULT_MIN_CLOSE_INTERVALS)
            & (grouped["days"] >= DEFAULT_MIN_CLOSE_DAYS),
            ["u", "v"],
        ]
        tie_set = set(map(tuple, ties.to_numpy()))
        records.append(
            {
                "threshold": threshold,
                "physical_ties": len(tie_set),
                "facebook_share": len(tie_set & fb_pairs) / len(tie_set),
            }
        )
    sensitivity = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(
        data.bluetooth.loc[data.bluetooth["rssi"] <= 0, "rssi"],
        bins=50,
        color="#4C78A8",
        ax=axes[0],
    )
    axes[0].axvline(
        -80,
        color="#E45756",
        linestyle="--",
        label="Default threshold = -80 dBm",
    )
    axes[0].set(
        title="RSSI distribution for participant-to-participant detections",
        xlabel="RSSI (dBm; higher means closer)",
        ylabel="Observations",
        yscale="log",
    )
    axes[0].legend()
    right = axes[1].twinx()
    axes[1].plot(
        sensitivity["threshold"],
        sensitivity["physical_ties"],
        marker="o",
        color="#4C78A8",
        label="Physical ties",
    )
    right.plot(
        sensitivity["threshold"],
        sensitivity["facebook_share"],
        marker="s",
        color="#E45756",
        label="Facebook share",
    )
    axes[1].set(
        title="Sensitivity of the physical-tie definition",
        xlabel="RSSI threshold (dBm)",
        ylabel="Number of physical ties",
    )
    right.set(ylabel="Share also connected on Facebook", ylim=(0, 1))
    lines = axes[1].lines + right.lines
    axes[1].legend(lines, [line.get_label() for line in lines])
    return _save(fig, data, "03_rssi_threshold_sensitivity.png")


def plot_facebook_proximity_distributions(data: EDAData) -> plt.Figure:
    frame = data.dyads.assign(
        relationship=np.where(
            data.dyads["facebook"] == 1,
            "Facebook friends",
            "Not Facebook friends",
        ),
        log_close=np.log1p(data.dyads["close_intervals"]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.ecdfplot(
        data=frame,
        x="close_intervals",
        hue="relationship",
        complementary=True,
        log_scale=(True, False),
        ax=axes[0],
    )
    axes[0].set(
        title="Tail distribution of repeated close proximity",
        xlabel="Close five-minute intervals (log scale)",
        ylabel="Share with at least this many intervals",
    )
    sns.boxplot(
        data=frame,
        x="relationship",
        y="log_close",
        hue="relationship",
        legend=False,
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set(
        title="Close proximity by Facebook status",
        xlabel="",
        ylabel="log(1 + close intervals)",
    )
    return _save(fig, data, "04_facebook_vs_proximity.png")


def plot_layer_upset(data: EDAData, top_n: int = 12) -> plt.Figure:
    membership = pd.DataFrame(
        {
            "Facebook": data.dyads["facebook"].astype(bool),
            "Calls": data.dyads["call_count"].gt(0),
            "SMS": data.dyads["sms_count"].gt(0),
            "Physical": data.dyads["physical_tie"].astype(bool),
        }
    )
    counts = (
        membership.value_counts()
        .rename("dyads")
        .reset_index()
        .head(top_n)
        .reset_index(drop=True)
    )
    x = np.arange(len(counts))
    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 1.6], hspace=0.05)
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1], sharex=top)
    top.bar(x, counts["dyads"], color="#4C78A8")
    top.set_yscale("log")
    top.set(
        title="Exact overlap of digital and physical network layers",
        ylabel="Dyads (log scale)",
        ylim=(0.8, counts["dyads"].max() * 3),
    )
    for column, value in enumerate(counts["dyads"]):
        top.text(
            column,
            value * 1.15,
            f"{value:,}",
            ha="center",
            fontsize=8,
            rotation=45,
        )
    layers = ["Facebook", "Calls", "SMS", "Physical"]
    for column, row in enumerate(counts.itertuples(index=False)):
        active = []
        for y, layer in enumerate(layers):
            enabled = bool(getattr(row, layer))
            bottom.scatter(
                column,
                y,
                s=65,
                color="#222222" if enabled else "#D9D9D9",
                zorder=3,
            )
            if enabled:
                active.append(y)
        if len(active) > 1:
            bottom.plot(
                [column, column],
                [min(active), max(active)],
                color="#222222",
                linewidth=2,
            )
    bottom.set(
        yticks=range(4),
        yticklabels=layers,
        xticks=x,
        xlabel="Exact layer combination",
        ylim=(-0.6, 3.4),
    )
    bottom.invert_yaxis()
    return _save(fig, data, "05_layer_overlap_upset.png", tight=False)


def plot_digital_vs_physical(data: EDAData) -> plt.Figure:
    frame = data.dyads.assign(
        communications=data.dyads["call_count"] + data.dyads["sms_count"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for axis, value, title in zip(
        axes, [0, 1], ["Not Facebook friends", "Facebook friends"]
    ):
        subset = frame.loc[frame["facebook"] == value]
        image = axis.hexbin(
            np.log1p(subset["communications"]),
            np.log1p(subset["close_intervals"]),
            gridsize=35,
            bins="log",
            mincnt=1,
            cmap="mako",
        )
        fig.colorbar(image, ax=axis, label="log10(dyads per hexagon)")
        axis.set(
            title=title,
            xlabel="log(1 + calls + SMS)",
            ylabel="log(1 + close intervals)",
        )
    return _save(fig, data, "06_digital_vs_physical_hexbin.png")


def plot_overlap_strength_curve(data: EDAData) -> plt.Figure:
    ranked = data.dyads.sort_values(
        ["close_intervals", "close_days"], ascending=False
    ).reset_index(drop=True)
    x = (np.arange(len(ranked)) + 1) / len(ranked)
    fig, axis = plt.subplots(figsize=(9, 6))
    for column, label, color in [
        ("sms_count", "SMS captured", "#4C78A8"),
        ("call_count", "Calls captured", "#F58518"),
    ]:
        axis.plot(
            x,
            ranked[column].cumsum() / ranked[column].sum(),
            label=label,
            color=color,
        )
    axis.plot([0, 1], [0, 1], "--", color="#999999", label="Uniform baseline")
    axis.set(
        title="Digital communication captured by strongest physical dyads",
        xlabel="Top fraction of dyads ranked by close proximity",
        ylabel="Cumulative fraction of communications",
        xlim=(0, 0.25),
        ylim=(0, 1),
    )
    axis.legend()
    return _save(fig, data, "07_overlap_strength_curve.png")


def plot_ego_multiplex(data: EDAData, max_neighbors: int = 24) -> plt.Figure:
    dyads = data.dyads.copy()
    dyads["strength"] = (
        3 * dyads["facebook"]
        + np.log1p(dyads["sms_count"])
        + np.log1p(dyads["call_count"])
        + np.log1p(dyads["close_intervals"])
    )
    endpoints = pd.concat(
        [
            dyads[["u", "strength"]].rename(columns={"u": "user"}),
            dyads[["v", "strength"]].rename(columns={"v": "user"}),
        ]
    )
    ego = int(endpoints.groupby("user")["strength"].sum().idxmax())
    incident = dyads.loc[
        ((dyads["u"] == ego) | (dyads["v"] == ego))
        & (
            (dyads["facebook"] == 1)
            | (dyads["digital_contact"] == 1)
            | (dyads["physical_tie"] == 1)
        )
    ].copy()
    incident["neighbor"] = np.where(
        incident["u"] == ego, incident["v"], incident["u"]
    )
    neighbors = (
        incident.nlargest(max_neighbors, "strength")["neighbor"].astype(int).tolist()
    )
    nodes = [ego] + neighbors
    selected = dyads.loc[dyads["u"].isin(nodes) & dyads["v"].isin(nodes)]
    angles = np.linspace(0, 2 * np.pi, len(neighbors), endpoint=False)
    positions = {ego: np.array([0.0, 0.0])}
    positions.update(
        {
            node: np.array([np.cos(angle), np.sin(angle)])
            for node, angle in zip(neighbors, angles)
        }
    )
    fig, axis = plt.subplots(figsize=(9, 9))
    for row in selected.itertuples(index=False):
        start, end = positions[int(row.u)], positions[int(row.v)]
        if row.close_intervals > 0:
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#B8B8B8",
                linewidth=0.4 + min(np.log1p(row.close_intervals) / 2, 3),
                alpha=0.55,
            )
        if row.facebook:
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#4C78A8",
                linewidth=1.5,
            )
        if row.call_count > 0 or row.sms_count > 0:
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#F58518",
                linestyle="--",
            )
    for node, position in positions.items():
        axis.scatter(
            *position,
            s=180 if node == ego else 60,
            color="#E45756" if node == ego else "#72B7B2",
            edgecolor="white",
            zorder=4,
        )
        axis.text(position[0], position[1] + 0.05, str(node), ha="center", fontsize=8)
    axis.set(
        title=f"Multiplex ego network for participant {ego}",
        xlim=(-1.2, 1.2),
        ylim=(-1.2, 1.2),
        aspect="equal",
    )
    axis.axis("off")
    axis.legend(
        handles=[
            Line2D([0], [0], color="#4C78A8", label="Facebook"),
            Line2D([0], [0], color="#F58518", linestyle="--", label="Call or SMS"),
            Line2D([0], [0], color="#B8B8B8", linewidth=3, label="Proximity"),
        ]
    )
    return _save(fig, data, "08_multiplex_ego_network.png")


def _digital_groups(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                (frame["facebook"] == 0) & (frame["digital_contact"] == 0),
                (frame["facebook"] == 1) & (frame["digital_contact"] == 0),
                (frame["facebook"] == 0) & (frame["digital_contact"] == 1),
                (frame["facebook"] == 1) & (frame["digital_contact"] == 1),
            ],
            [
                "No digital edge",
                "Facebook only",
                "Calls/SMS only",
                "Facebook + calls/SMS",
            ],
            default="Other",
        ),
        index=frame.index,
    )


def plot_physical_tie_probability(data: EDAData) -> plt.Figure:
    frame = data.dyads.assign(digital_group=_digital_groups(data.dyads))
    order = [
        "No digital edge",
        "Facebook only",
        "Calls/SMS only",
        "Facebook + calls/SMS",
    ]
    rates = (
        frame.groupby("digital_group", observed=True)["physical_tie"]
        .agg(["mean", "count"])
        .reindex(order)
        .reset_index()
    )
    rates["se"] = np.sqrt(rates["mean"] * (1 - rates["mean"]) / rates["count"])
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        rates["digital_group"],
        rates["mean"],
        color=["#BAB0AC", "#4C78A8", "#F58518", "#E45756"],
    )
    axis.errorbar(
        np.arange(4),
        rates["mean"],
        yerr=1.96 * rates["se"],
        fmt="none",
        color="black",
        capsize=4,
    )
    for i, row in rates.iterrows():
        axis.text(
            i,
            row["mean"] + 0.015,
            f"{row['mean']:.1%}\n(n={int(row['count']):,})",
            ha="center",
        )
    axis.set(
        title="Empirical probability of a repeated physical tie",
        xlabel="Observed digital relationship",
        ylabel="Physical-tie rate",
        ylim=(0, 0.9),
    )
    axis.tick_params(axis="x", rotation=15)
    return _save(fig, data, "09_physical_tie_probability.png")


def plot_class_vs_after_hours(data: EDAData) -> plt.Figure:
    ties = data.dyads.loc[data.dyads["physical_tie"] == 1]
    x = np.log1p(ties["close_class_hour_intervals"])
    y = np.log1p(ties["close_after_hours_intervals"])
    fig, axis = plt.subplots(figsize=(9, 7))
    image = axis.hexbin(x, y, gridsize=40, bins="log", mincnt=1, cmap="mako")
    maximum = max(float(x.max()), float(y.max()))
    axis.plot(
        [0, maximum],
        [0, maximum],
        "--",
        color="#E45756",
        label="Equal class-hour and after-hours counts",
    )
    axis.text(
        0.97, 0.05, "Class-hour\nconcentrated",
        transform=axis.transAxes, ha="right", color="#4C78A8", weight="bold"
    )
    axis.text(
        0.03, 0.95, "After-hours\nconcentrated",
        transform=axis.transAxes, va="top", color="#F58518", weight="bold"
    )
    axis.text(
        0.97, 0.95, "Active across\nboth periods",
        transform=axis.transAxes, ha="right", va="top",
        color="#54A24B", weight="bold"
    )
    fig.colorbar(image, ax=axis, label="log10(dyads per hexagon)")
    axis.set(
        title="When do repeated physical ties interact?",
        xlabel="log(1 + weekday 10 a.m.–2 p.m. intervals)",
        ylabel="log(1 + evening, overnight, or weekend intervals)",
    )
    axis.legend(loc="lower left")
    return _save(fig, data, "10_class_vs_after_hours.png")


def plot_schedule_by_digital_relationship(data: EDAData) -> plt.Figure:
    ties = data.dyads.loc[data.dyads["physical_tie"] == 1].copy()
    ties["digital_group"] = _digital_groups(ties)
    order = [
        "No digital edge",
        "Facebook only",
        "Calls/SMS only",
        "Facebook + calls/SMS",
    ]
    indicators = {
        "Any weekday 10 a.m.–2 p.m.": ties["close_class_hour_intervals"] > 0,
        "Any 9 p.m.–midnight": ties["close_late_evening_intervals"] > 0,
        "Any weekend": ties["close_weekend_intervals"] > 0,
    }
    records = []
    for label, indicator in indicators.items():
        rates = (
            ties.assign(indicator=indicator)
            .groupby("digital_group", observed=True)["indicator"]
            .mean()
            .reindex(order)
        )
        records.extend(
            {
                "digital_group": group,
                "time_window": label,
                "rate": value,
            }
            for group, value in rates.items()
        )
    fig, axis = plt.subplots(figsize=(13, 6))
    sns.barplot(
        data=pd.DataFrame(records),
        x="digital_group",
        y="rate",
        hue="time_window",
        order=order,
        ax=axis,
    )
    axis.set(
        title="Time-of-contact patterns within repeated physical ties",
        xlabel="Observed digital relationship",
        ylabel="Share of dyads with at least one observation",
        ylim=(0, 1),
    )
    axis.tick_params(axis="x", rotation=12)
    return _save(fig, data, "11_schedule_by_digital_relationship.png")


TEMPORAL_CLUSTER_FEATURES = [
    "close_intervals",
    "close_days",
    "class_hour_share",
    "evening_share",
    "late_evening_share",
    "weekend_share",
    "after_hours_share",
    "temporal_entropy",
    "active_hour_bins",
    "longest_session_intervals",
]


def _cluster_matrix(data: EDAData) -> tuple[pd.DataFrame, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    ties = data.dyads.loc[data.dyads["physical_tie"] == 1].copy()
    features = ties[TEMPORAL_CLUSTER_FEATURES].astype(float).copy()
    for column in [
        "close_intervals",
        "close_days",
        "active_hour_bins",
        "longest_session_intervals",
    ]:
        features[column] = np.log1p(features[column])
    return ties, StandardScaler().fit_transform(features)


def plot_temporal_cluster_diagnostics(data: EDAData) -> plt.Figure:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    _, scaled = _cluster_matrix(data)
    records = []
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(scaled)
        records.append(
            {"clusters": k, "silhouette": silhouette_score(scaled, labels)}
        )
    scores = pd.DataFrame(records)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(scores["clusters"], scores["silhouette"], marker="o", color="#4C78A8")
    for row in scores.itertuples(index=False):
        axis.text(row.clusters, row.silhouette + 0.005, f"{row.silhouette:.2f}", ha="center")
    axis.set(
        title="Do distinct temporal relationship patterns exist?",
        xlabel="Number of temporal clusters",
        ylabel="Silhouette score (higher means clearer separation)",
        xticks=range(2, 7),
        ylim=(0, 0.54),
    )
    return _save(fig, data, "12_temporal_cluster_diagnostics.png")


def _cluster_temporal_dyads(
    data: EDAData, n_clusters: int
) -> pd.DataFrame:
    from sklearn.cluster import KMeans

    ties, scaled = _cluster_matrix(data)
    ties["cluster_id"] = KMeans(
        n_clusters=n_clusters, random_state=42, n_init=20
    ).fit_predict(scaled)
    means = ties.groupby("cluster_id")[TEMPORAL_CLUSTER_FEATURES].mean()
    remaining = set(means.index)
    labels = {}
    selected = int(means.loc[list(remaining), "class_hour_share"].idxmax())
    labels[selected] = "Class-hour concentrated"
    remaining.remove(selected)
    selected = int(means.loc[list(remaining), "after_hours_share"].idxmax())
    labels[selected] = "After-hours concentrated"
    remaining.remove(selected)
    selected = int(means.loc[list(remaining), "temporal_entropy"].idxmax())
    labels[selected] = "Broad schedule"
    remaining.remove(selected)
    for selected in remaining:
        labels[int(selected)] = "Mixed / moderate"
    ties["temporal_archetype"] = ties["cluster_id"].map(labels)
    return ties


def plot_temporal_cluster_profiles(
    data: EDAData, n_clusters: int = 4
) -> tuple[plt.Figure, pd.DataFrame]:
    clustered = _cluster_temporal_dyads(data, n_clusters)
    assignments = clustered[["u", "v", "temporal_archetype"]]
    events = data.close_bluetooth.merge(assignments, on=["u", "v"], how="inner")
    profile = (
        events.groupby(
            ["u", "v", "temporal_archetype", "weekday", "hour"],
            observed=True,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    profile["share"] = profile["count"] / profile.groupby(
        ["u", "v"], observed=True
    )["count"].transform("sum")
    summary = (
        clustered.groupby("temporal_archetype", observed=True)
        .agg(
            dyads=("physical_tie", "size"),
            facebook_rate=("facebook", "mean"),
            direct_communication_rate=("digital_contact", "mean"),
            median_close_intervals=("close_intervals", "median"),
            median_close_days=("close_days", "median"),
            median_class_hour_share=("class_hour_share", "median"),
            median_after_hours_share=("after_hours_share", "median"),
        )
        .sort_values("median_after_hours_share")
    )
    matrices = {}
    for archetype in summary.index:
        matrices[archetype] = (
            profile.loc[profile["temporal_archetype"] == archetype]
            .groupby(["weekday", "hour"], observed=True)["share"]
            .mean()
            .unstack(fill_value=0)
            .reindex(index=range(7), columns=range(24), fill_value=0)
        )
    vmax = max(float(matrix.to_numpy().max()) for matrix in matrices.values())
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for axis, archetype in zip(axes.flat, summary.index):
        row = summary.loc[archetype]
        sns.heatmap(
            matrices[archetype],
            cmap="mako",
            vmin=0,
            vmax=vmax,
            yticklabels=weekdays,
            cbar_kws={"label": "Mean within-dyad activity share"},
            ax=axis,
        )
        axis.set(
            title=(
                f"{archetype}\n"
                f"n={int(row['dyads']):,}; Facebook={row['facebook_rate']:.0%}; "
                f"calls/SMS={row['direct_communication_rate']:.0%}"
            ),
            xlabel="Hour of day",
            ylabel="",
        )
    return _save(fig, data, "13_temporal_cluster_profiles.png"), summary
