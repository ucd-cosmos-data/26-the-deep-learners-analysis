from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
OUTPUT_FILE = PROJECT_DIR / "results" / "friendship_index_report.html"

BASELINE_WEIGHTS = {
    "investment_pillar": 0.35,
    "reciprocity_pillar": 0.35,
    "intimacy_pillar": 0.30,
}

CUSTOM_WEIGHTS = {
    "investment_pillar": 0.55,
    "reciprocity_pillar": 0.15,
    "intimacy_pillar": 0.30,
}


def positive_percentile(values: pd.Series) -> pd.Series:
    """Rank positive values from 0 to 1 while preserving zero as zero."""
    values = values.fillna(0).clip(lower=0)
    transformed = np.log1p(values)
    scores = pd.Series(0.0, index=values.index)
    positive = transformed.gt(0)
    scores.loc[positive] = transformed.loc[positive].rank(pct=True)
    return scores


def build_friend_pair_ids() -> set[str]:
    friends = pd.read_csv(INTERIM_DIR / "fb_friends.csv")
    user_a = friends[["user_a", "user_b"]].min(axis=1)
    user_b = friends[["user_a", "user_b"]].max(axis=1)
    return set(user_a.astype(str) + "-" + user_b.astype(str))


def build_index_data(friend_pair_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(INTERIM_DIR / "pairwise_features.csv")

    # Facebook-only rows leak the target into the candidate universe.
    data = data.loc[data["active_days"].notna()].copy()
    data["pair"] = data["user_a"].astype(str) + "-" + data["user_b"].astype(str)
    data["is_facebook_friend"] = data["pair"].isin(friend_pair_ids).astype(int)

    investment = pd.DataFrame(
        {
            "proximity_volume_score": positive_percentile(data["proximity_measurements"]),
            "text_volume_score": positive_percentile(data["texts_shared"]),
            "call_volume_score": positive_percentile(data["calls_shared"]),
            "call_time_score": positive_percentile(data["average_call_contact_time"]),
        }
    )
    reciprocity = pd.DataFrame(
        {
            "text_reciprocity_score": data["text_reciprocity"].fillna(0).clip(0, 1),
            "call_reciprocity_score": data["call_reciprocity"].fillna(0).clip(0, 1),
        }
    )
    intimacy = pd.DataFrame(
        {
            "weekend_score": data["weekend_interaction_fraction"].fillna(0).clip(0, 1),
            "active_days_score": positive_percentile(data["active_days"]),
            "consecutive_days_score": positive_percentile(
                data["longest_consecutive_days"]
            ),
        }
    )

    data["investment_pillar"] = investment.mean(axis=1)
    data["reciprocity_pillar"] = reciprocity.mean(axis=1)
    data["intimacy_pillar"] = intimacy.mean(axis=1)
    data["baseline_index"] = 100 * sum(
        weight * data[pillar] for pillar, weight in BASELINE_WEIGHTS.items()
    )
    data["custom_index"] = 100 * sum(
        weight * data[pillar] for pillar, weight in CUSTOM_WEIGHTS.items()
    )

    components = pd.concat([investment, reciprocity, intimacy], axis=1)
    component_std = StandardScaler().fit_transform(components)
    pca = PCA(n_components=1)
    pc1_raw = pca.fit_transform(component_std).ravel()
    orientation = 1
    if np.corrcoef(pc1_raw, data["custom_index"])[0, 1] < 0:
        orientation = -1
        pc1_raw *= -1
    data["pc1_score"] = (
        pd.Series(pc1_raw, index=data.index).rank(pct=True) * 100
    )

    loadings = pd.DataFrame(
        {
            "feature": components.columns,
            "PC1 loading": orientation * pca.components_[0],
        }
    ).sort_values("PC1 loading", key=abs, ascending=False)
    return data, loadings


def build_deciles(data: pd.DataFrame) -> pd.DataFrame:
    results = []
    for method, score_column in {
        "Custom 55/15/30": "custom_index",
        "Baseline 35/35/30": "baseline_index",
        "PCA PC1": "pc1_score",
    }.items():
        # Exact ties are broken deterministically to retain ten equal-sized groups.
        deciles = pd.qcut(
            data[score_column].rank(method="first"),
            q=10,
            labels=range(1, 11),
        )
        result = (
            data.assign(decile=deciles)
            .groupby("decile", observed=True)["is_facebook_friend"]
            .agg(facebook_friend_rate="mean", pair_count="size")
            .reset_index()
        )
        result["Method"] = method
        results.append(result)
    return pd.concat(results, ignore_index=True)


def build_lift(data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    overall_rate = data["is_facebook_friend"].mean()
    rows = []
    for method, score_column in {
        "Custom 55/15/30": "custom_index",
        "Baseline 35/35/30": "baseline_index",
        "PCA PC1": "pc1_score",
    }.items():
        for fraction in (0.01, 0.05, 0.10):
            pair_count = int(np.ceil(len(data) * fraction))
            top_pairs = data.nlargest(pair_count, score_column)
            friend_rate = top_pairs["is_facebook_friend"].mean()
            rows.append(
                {
                    "Method": method,
                    "Top segment": f"Top {fraction:.0%}",
                    "Facebook-friend rate": friend_rate,
                    "Lift": friend_rate / overall_rate,
                    "Pair count": pair_count,
                }
            )
    return pd.DataFrame(rows), overall_rate


def build_stability(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = {
        "Custom 55/15/30": (0.55, 0.15, 0.30),
        "Baseline 35/35/30": (0.35, 0.35, 0.30),
        "Equal weights": (1 / 3, 1 / 3, 1 / 3),
        "Investment-heavy": (0.50, 0.25, 0.25),
        "Reciprocity-heavy": (0.25, 0.50, 0.25),
        "Intimacy-heavy": (0.25, 0.25, 0.50),
    }
    scores = pd.DataFrame(index=data.index)
    for name, (investment, reciprocity, intimacy) in scenarios.items():
        scores[name] = 100 * (
            investment * data["investment_pillar"]
            + reciprocity * data["reciprocity_pillar"]
            + intimacy * data["intimacy_pillar"]
        )
    return scores, scores.corr(method="spearman")


def fit_logistic_benchmark(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    features = [
        "average_proximity_rssi",
        "proximity_measurements",
        "fraction_rssi_above_threshold",
        "fraction_rssi_below_threshold",
        "texts_shared",
        "calls_shared",
        "average_call_contact_time",
        "max_call_contact_time",
        "text_reciprocity",
        "call_reciprocity",
        "active_days",
        "longest_consecutive_days",
        "weekend_interaction_fraction",
        "weekday_interaction_fraction",
    ]
    X = data[features]
    y = data["is_facebook_friend"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=0,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1765,
        random_state=0,
        stratify=y_train_val,
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    l1_ratio=1.0,
                    solver="liblinear",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5_000,
                    random_state=0,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Recall",
                "Precision",
                "F1",
                "ROC AUC",
                "Naive majority accuracy",
            ],
            "Value": [
                accuracy_score(y_test, predictions),
                recall_score(y_test, predictions),
                precision_score(y_test, predictions),
                f1_score(y_test, predictions),
                roc_auc_score(y_test, probabilities),
                max(y_test.mean(), 1 - y_test.mean()),
            ],
        }
    )
    coefficients = pd.DataFrame(
        {
            "Feature": features,
            "Coefficient": pipeline.named_steps["model"].coef_[0],
        }
    ).sort_values("Coefficient", key=abs, ascending=False)
    return metrics, coefficients, y_test.to_numpy(), predictions, probabilities


def format_table(
    frame: pd.DataFrame,
    *,
    table_id: str,
    percent_columns: tuple[str, ...] = (),
    decimals: int = 3,
) -> str:
    formatted = frame.copy()
    for column in formatted.columns:
        if column in percent_columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2%}"
            )
        elif pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}"
            )
    return formatted.to_html(
        index=False,
        classes="data-table",
        table_id=table_id,
        border=0,
        escape=True,
    )


def figure_html(figure: go.Figure, *, include_plotly: bool = False) -> str:
    figure.update_layout(
        template="plotly_white",
        margin=dict(l=55, r=30, t=70, b=55),
        font=dict(family="Inter, system-ui, sans-serif", color="#17202a"),
        hoverlabel=dict(font_size=13),
    )
    return figure.to_html(
        full_html=False,
        include_plotlyjs=True if include_plotly else False,
        config={"responsive": True, "displaylogo": False},
    )


def main() -> None:
    friend_pair_ids = build_friend_pair_ids()
    data, pca_loadings = build_index_data(friend_pair_ids)
    deciles = build_deciles(data)
    lift, overall_friend_rate = build_lift(data)
    scenario_scores, stability = build_stability(data)
    model_metrics, coefficients, y_test, predictions, probabilities = (
        fit_logistic_benchmark(data)
    )

    custom_pearson_pc1 = data["custom_index"].corr(
        data["pc1_score"], method="pearson"
    )
    custom_spearman_pc1 = data["custom_index"].corr(
        data["pc1_score"], method="spearman"
    )
    baseline_pearson_pc1 = data["baseline_index"].corr(
        data["pc1_score"], method="pearson"
    )
    baseline_spearman_pc1 = data["baseline_index"].corr(
        data["pc1_score"], method="spearman"
    )
    custom_baseline_spearman = data["custom_index"].corr(
        data["baseline_index"], method="spearman"
    )
    minimum_stability = stability.where(
        ~np.eye(len(stability), dtype=bool)
    ).stack().min()

    inspection_columns = [
        "user_a",
        "user_b",
        "custom_index",
        "baseline_index",
        "pc1_score",
        "investment_pillar",
        "reciprocity_pillar",
        "intimacy_pillar",
        "proximity_measurements",
        "texts_shared",
        "calls_shared",
        "average_call_contact_time",
        "text_reciprocity",
        "call_reciprocity",
        "active_days",
        "longest_consecutive_days",
        "weekend_interaction_fraction",
        "is_facebook_friend",
    ]
    custom_top = data.nlargest(20, "custom_index")[inspection_columns]
    custom_bottom = data.nsmallest(20, "custom_index")[inspection_columns]
    baseline_top = data.nlargest(20, "baseline_index")[inspection_columns]
    pca_top = data.nlargest(20, "pc1_score")[inspection_columns]
    custom_pca_top_overlap = len(
        set(custom_top[["user_a", "user_b"]].itertuples(index=False, name=None))
        & set(pca_top[["user_a", "user_b"]].itertuples(index=False, name=None))
    )
    custom_baseline_top_overlap = len(
        set(custom_top[["user_a", "user_b"]].itertuples(index=False, name=None))
        & set(baseline_top[["user_a", "user_b"]].itertuples(index=False, name=None))
    )

    decile_figure = px.line(
        deciles,
        x="decile",
        y="facebook_friend_rate",
        color="Method",
        markers=True,
        title="Facebook-Friend Rate by Score Decile",
        labels={
            "decile": "Score decile (10 = highest)",
            "facebook_friend_rate": "Facebook-friend rate",
        },
    )
    decile_figure.update_yaxes(tickformat=".0%")

    lift_figure = px.bar(
        lift,
        x="Top segment",
        y="Lift",
        color="Method",
        barmode="group",
        title="Lift among Highest-Scoring Pairs",
        category_orders={"Top segment": ["Top 1%", "Top 5%", "Top 10%"]},
    )
    lift_figure.add_hline(
        y=1,
        line_dash="dash",
        line_color="grey",
        annotation_text="No lift",
    )

    plot_sample = data.sample(n=min(8_000, len(data)), random_state=42)
    correlation_figure = px.scatter(
        plot_sample,
        x="custom_index",
        y="pc1_score",
        color=plot_sample["is_facebook_friend"].map(
            {0: "Not Facebook friends", 1: "Facebook friends"}
        ),
        opacity=0.5,
        title="Custom 55/15/30 Index versus Oriented PCA PC1",
        labels={
            "custom_index": "Custom 55/15/30 friendship index (0–100)",
            "pc1_score": "PC1 percentile score (0–100)",
            "color": "Facebook status",
        },
        hover_data=["user_a", "user_b"],
    )

    stability_figure = go.Figure(
        data=go.Heatmap(
            z=stability.values,
            x=stability.columns,
            y=stability.index,
            zmin=0,
            zmax=1,
            colorscale="Viridis",
            text=np.round(stability.values, 3),
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>Spearman: %{z:.3f}<extra></extra>",
        )
    )
    stability_figure.update_layout(title="Rank Stability under Alternative Weights")

    score_distribution = px.histogram(
        data,
        x="custom_index",
        color=data["is_facebook_friend"].map(
            {0: "Not Facebook friends", 1: "Facebook friends"}
        ),
        histnorm="probability density",
        barmode="overlay",
        opacity=0.55,
        nbins=60,
        title="Custom 55/15/30 Index Distribution by Facebook Status",
        labels={
            "custom_index": "Custom 55/15/30 friendship index",
            "color": "Facebook status",
        },
    )

    coefficient_figure = px.bar(
        coefficients.sort_values("Coefficient"),
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="Balanced L1 Logistic-Regression Coefficients",
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    auc_value = model_metrics.loc[
        model_metrics["Metric"].eq("ROC AUC"), "Value"
    ].iloc[0]
    roc_figure = go.Figure()
    roc_figure.add_trace(
        go.Scatter(
            x=false_positive_rate,
            y=true_positive_rate,
            mode="lines",
            name=f"Model (AUC {auc_value:.3f})",
        )
    )
    roc_figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Random ranking",
        )
    )
    roc_figure.update_layout(
        title="Test-Set ROC Curve",
        xaxis_title="False-positive rate",
        yaxis_title="True-positive rate",
    )

    confusion = confusion_matrix(y_test, predictions)
    confusion_figure = go.Figure(
        data=go.Heatmap(
            z=confusion,
            x=["Predicted non-friend", "Predicted friend"],
            y=["Actual non-friend", "Actual friend"],
            colorscale="Blues",
            text=confusion,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>Pairs: %{z}<extra></extra>",
        )
    )
    confusion_figure.update_layout(title="Test-Set Confusion Matrix")

    custom_top_one = lift.loc[
        lift["Method"].eq("Custom 55/15/30")
        & lift["Top segment"].eq("Top 1%")
    ].iloc[0]
    custom_top_decile_rate = deciles.loc[
        deciles["Method"].eq("Custom 55/15/30") & deciles["decile"].eq(10),
        "facebook_friend_rate",
    ].iloc[0]
    baseline_top_decile_rate = deciles.loc[
        deciles["Method"].eq("Baseline 35/35/30") & deciles["decile"].eq(10),
        "facebook_friend_rate",
    ].iloc[0]
    pca_top_decile_rate = deciles.loc[
        deciles["Method"].eq("PCA PC1") & deciles["decile"].eq(10),
        "facebook_friend_rate",
    ].iloc[0]
    headline_metrics = pd.DataFrame(
        {
            "Quantity": [
                "Behavior-observed candidate pairs",
                "Overall Facebook-friend rate",
                "Custom 55/15/30 top-decile friend rate",
                "Baseline 35/35/30 top-decile friend rate",
                "PCA top-decile friend rate",
                "Custom–PC1 Pearson correlation",
                "Custom–PC1 Spearman correlation",
                "Baseline–PC1 Pearson correlation",
                "Baseline–PC1 Spearman correlation",
                "Custom–baseline Spearman correlation",
                "Minimum alternative-weight rank correlation",
                "Custom/PCA top-20 overlap",
                "Custom/baseline top-20 overlap",
            ],
            "Value": [
                f"{len(data):,}",
                f"{overall_friend_rate:.2%}",
                f"{custom_top_decile_rate:.2%}",
                f"{baseline_top_decile_rate:.2%}",
                f"{pca_top_decile_rate:.2%}",
                f"{custom_pearson_pc1:.3f}",
                f"{custom_spearman_pc1:.3f}",
                f"{baseline_pearson_pc1:.3f}",
                f"{baseline_spearman_pc1:.3f}",
                f"{custom_baseline_spearman:.3f}",
                f"{minimum_stability:.3f}",
                f"{custom_pca_top_overlap}/20",
                f"{custom_baseline_top_overlap}/20",
            ],
        }
    )

    tables = {
        "headline": format_table(headline_metrics, table_id="headline-table"),
        "model": format_table(model_metrics, table_id="model-table"),
        "lift": format_table(
            lift,
            table_id="lift-table",
            percent_columns=("Facebook-friend rate",),
        ),
        "deciles": format_table(
            deciles,
            table_id="decile-table",
            percent_columns=("facebook_friend_rate",),
        ),
        "coefficients": format_table(coefficients, table_id="coefficient-table"),
        "loadings": format_table(pca_loadings, table_id="loading-table"),
        "top": format_table(
            custom_top,
            table_id="top-table",
            percent_columns=("weekend_interaction_fraction",),
        ),
        "bottom": format_table(
            custom_bottom,
            table_id="bottom-table",
            percent_columns=("weekend_interaction_fraction",),
        ),
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Behavioral Friendship Index Report</title>
  <style>
    :root {{
      --ink: #17202a; --muted: #5f6b76; --paper: #f4f6f8; --card: #fff;
      --accent: #2667ff; --accent2: #14a38b; --line: #dce2e8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); background: var(--paper);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; line-height: 1.55; }}
    header {{ color: white; padding: 64px max(24px, calc((100vw - 1180px)/2));
      background: radial-gradient(circle at 80% 10%, #2f80ed 0, transparent 32%),
                  linear-gradient(135deg, #101a35, #163b67); }}
    header h1 {{ margin: 0 0 12px; font-size: clamp(2rem, 4vw, 3.7rem); line-height: 1.05; }}
    header p {{ max-width: 820px; margin: 0; color: #dce8ff; font-size: 1.12rem; }}
    nav {{ position: sticky; top: 0; z-index: 10; background: rgba(255,255,255,.96);
      border-bottom: 1px solid var(--line); padding: 10px max(20px, calc((100vw - 1180px)/2));
      overflow-x: auto; white-space: nowrap; }}
    nav a {{ color: #24415f; text-decoration: none; margin-right: 22px; font-size: .92rem; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 80px; }}
    section {{ scroll-margin-top: 68px; margin: 0 0 42px; }}
    h2 {{ margin-top: 0; font-size: 1.75rem; }} h3 {{ margin-bottom: 6px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px;
      padding: 22px; box-shadow: 0 7px 22px rgba(22,35,55,.06); margin: 16px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 18px; }}
    .callout {{ border-left: 5px solid var(--accent); background: #edf3ff; padding: 15px 18px;
      border-radius: 8px; margin: 16px 0; }}
    .warning {{ border-left-color: #d67a00; background: #fff5e8; }}
    .good {{ border-left-color: var(--accent2); background: #eaf9f5; }}
    code {{ background: #eef1f4; padding: 2px 6px; border-radius: 5px; }}
    .table-wrap {{ overflow: auto; max-height: 560px; border: 1px solid var(--line); border-radius: 10px; }}
    .table-search {{ width: min(420px, 100%); padding: 10px 12px; margin: 0 0 10px;
      border: 1px solid #b9c3cd; border-radius: 8px; }}
    table.data-table {{ border-collapse: collapse; width: 100%; font-size: .87rem; background: white; }}
    .data-table th {{ position: sticky; top: 0; background: #eaf0f7; cursor: pointer; z-index: 1; }}
    .data-table th, .data-table td {{ padding: 9px 10px; border-bottom: 1px solid #e5e9ed;
      text-align: right; white-space: nowrap; }}
    .data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
    .data-table tbody tr:hover {{ background: #f3f7ff; }}
    .metric {{ font-size: 2rem; font-weight: 750; color: var(--accent); }}
    .small {{ color: var(--muted); font-size: .9rem; }}
    footer {{ color: var(--muted); text-align: center; padding: 30px; }}
    @media (max-width: 650px) {{ .grid {{ grid-template-columns: 1fr; }} .card {{ padding: 15px; }} }}
  </style>
</head>
<body>
<header>
  <h1>Behavioral Friendship Index</h1>
  <p>An interactive robustness report comparing an interpretable three-pillar index with PCA,
  using Facebook friendship only as an external proxy—not as the definition of true friendship.</p>
</header>
<nav>
  <a href="#summary">Summary</a><a href="#method">Method</a><a href="#deciles">Deciles</a>
  <a href="#lift">Lift</a><a href="#extremes">Extremes</a><a href="#stability">Stability</a>
  <a href="#pca">PCA</a><a href="#model">Model</a><a href="#imputation">Imputation</a>
</nav>
<main>
<section id="summary">
  <h2>Executive summary</h2>
  <div class="grid">
    <div class="card"><div class="metric">{overall_friend_rate:.2%}</div>
      <b>baseline Facebook-friend rate</b><p class="small">Among behavior-observed candidate pairs.</p></div>
    <div class="card"><div class="metric">{custom_top_one['Lift']:.1f}×</div>
      <b>custom 55/15/30 top-1% lift</b><p class="small">{custom_top_one['Facebook-friend rate']:.1%} of the custom index’s top 1% are Facebook friends.</p></div>
    <div class="card"><div class="metric">{custom_spearman_pc1:.3f}</div>
      <b>custom rank correlation with PC1</b><p class="small">How closely the custom formula and PCA order the pairs.</p></div>
  </div>
  <div class="card">{tables['headline']}</div>
  <div class="callout good"><b>Main result.</b> The custom 55/15/30 score is evaluated alongside the
  original 35/35/30 benchmark and PCA. Upper-tail lift measures Facebook-friend enrichment, while
  rank correlations show how much the custom weighting changes pair ordering. These are robustness
  checks for a relative relationship-strength ranking, not proof of “true friendship.”</div>
</section>

<section id="method">
  <h2>What the index measures</h2>
  <div class="grid">
    <div class="card"><h3>Investment — 55%</h3><p>Positive-percentile ranks of proximity volume,
      texts, calls, and average completed-call duration. <code>log1p</code> limits domination by
      extreme activity.</p></div>
    <div class="card"><h3>Reciprocity — 15%</h3><p>Text and call balance. A value near one means
      similarly balanced traffic in both directions; zero means one-way or absent communication.</p></div>
    <div class="card"><h3>Intimacy/consistency — 30%</h3><p>Weekend interaction share, active days,
      and longest consecutive-day streak.</p></div>
  </div>
  <p class="small">The report retains 35/35/30 as the baseline comparison, but all cards and
  extreme-pair tables labeled “custom” use your 55/15/30 weights.</p>
  <div class="callout warning"><b>Scope:</b> The 0–100 score is a relative behavioral index, not a
  probability and not a clinical or social judgment. High scores can describe roommates, partners,
  classmates, or family; low reciprocity can have benign explanations.</div>
  {figure_html(score_distribution, include_plotly=True)}
</section>

<section id="deciles">
  <h2>Facebook-friend rate by score decile</h2>
  <p><b>How to interpret:</b> Decile 10 contains the highest-scoring 10% of pairs. A useful ranking
  should generally show increasing Facebook-friend rates toward higher deciles. Exact ties are broken
  deterministically so that each decile has approximately the same number of pairs.</p>
  <div class="card">{figure_html(decile_figure)}</div>
  <input class="table-search" data-target="decile-table" placeholder="Filter decile table…">
  <div class="table-wrap">{tables['deciles']}</div>
</section>

<section id="lift">
  <h2>Lift in the top 1%, 5%, and 10%</h2>
  <p><b>Lift = top-segment friend rate ÷ overall friend rate.</b> A lift of 10× means a pair in that
  segment is ten times as likely to be a Facebook friendship as a randomly selected behavior-observed
  candidate pair. Lift measures enrichment, not causal validity.</p>
  <div class="card">{figure_html(lift_figure)}</div>
  <div class="table-wrap">{tables['lift']}</div>
</section>

<section id="extremes">
  <h2>Extreme-pair inspection</h2>
  <p>The top table should show dense, persistent, and reciprocal behavior. The bottom table should
  show sparse or one-off behavior. Facebook status is included only as a proxy check.</p>
  <h3>Custom 55/15/30 top 20</h3>
  <input class="table-search" data-target="top-table" placeholder="Filter top pairs…">
  <div class="table-wrap">{tables['top']}</div>
  <h3>Custom 55/15/30 bottom 20</h3>
  <input class="table-search" data-target="bottom-table" placeholder="Filter bottom pairs…">
  <div class="table-wrap">{tables['bottom']}</div>
</section>

<section id="stability">
  <h2>Rank stability under different weights</h2>
  <p>Each cell is a Spearman rank correlation. Values near one mean the ordering barely changes when
  investment, reciprocity, or intimacy receives more weight. The minimum off-diagonal correlation is
  <b>{minimum_stability:.3f}</b>, indicating a robust ranking.</p>
  <div class="card">{figure_html(stability_figure)}</div>
</section>

<section id="pca">
  <h2>Comparison with PCA</h2>
  <p>PC1 is the direction explaining the most variance in the same nine normalized behavioral
  components. Its sign is arbitrary, so it is oriented to agree with the custom index before comparison.
  Pearson correlation measures linear agreement; Spearman measures ranking agreement.</p>
  <div class="grid">
    <div class="card"><div class="metric">{custom_pearson_pc1:.3f}</div><b>Custom–PC1 Pearson correlation</b></div>
    <div class="card"><div class="metric">{custom_spearman_pc1:.3f}</div><b>Custom–PC1 Spearman correlation</b></div>
    <div class="card"><div class="metric">{custom_baseline_spearman:.3f}</div><b>Custom–baseline Spearman correlation</b></div>
  </div>
  <div class="card">{figure_html(correlation_figure)}</div>
  <h3>PC1 loadings</h3>
  <p>Larger absolute loadings contribute more to PC1; signs describe direction after orientation.</p>
  <div class="table-wrap">{tables['loadings']}</div>
</section>

<section id="model">
  <h2>Facebook-friend logistic benchmark</h2>
  <p>This is a balanced L1 logistic regression on behavior-observed pairs. The imputer and scaler are
  fit on training data only. <code>mutual_friends</code> is intentionally excluded.</p>
  <div class="callout"><b>Metric interpretation:</b>
    <ul>
      <li><b>Accuracy:</b> total fraction classified correctly. Compare it with naive majority accuracy;
      imbalance makes accuracy alone misleading.</li>
      <li><b>Recall:</b> fraction of actual Facebook friendships detected.</li>
      <li><b>Precision:</b> fraction of predicted friendships that are actual Facebook edges.</li>
      <li><b>F1:</b> harmonic balance between precision and recall.</li>
      <li><b>ROC AUC:</b> threshold-independent ranking quality; 0.5 is random and 1.0 is perfect.</li>
    </ul>
  </div>
  <div class="grid">
    <div class="card"><div class="table-wrap">{tables['model']}</div></div>
    <div class="card">{figure_html(confusion_figure)}</div>
  </div>
  <div class="card">{figure_html(roc_figure)}</div>
  <div class="card">{figure_html(coefficient_figure)}</div>
  <p><b>Coefficients:</b> positive values increase the model’s estimated log-odds of a Facebook edge;
  negative values decrease them. Magnitude is comparable because inputs are standardized. Association
  is not causation.</p>
  <div class="table-wrap">{tables['coefficients']}</div>
</section>

<section id="imputation">
  <h2>Should mean imputation happen after scaling?</h2>
  <div class="callout warning"><b>No.</b> Split first, then fit the imputer on training data, then fit
  the scaler on the imputed training data. Apply those fitted transformations to validation and test
  sets. A pipeline enforces this order and prevents leakage.</div>
  <pre><code>Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced"))
])</code></pre>
  <p>Mean imputation is not universally appropriate here. A missing count often means zero observed
  interactions and should be encoded as zero by definition. Missing RSSI or call-duration summaries
  mean the measurement does not exist; consider a missingness indicator rather than inventing an
  average relationship.</p>
</section>

<section>
  <h2>Validity cautions</h2>
  <ul>
    <li>Facebook friendship is an imperfect proxy and does not define true friendship.</li>
    <li>Facebook-only pairs are excluded so the target does not leak into candidate selection.</li>
    <li>Bluetooth counts depend on device availability and sensing coverage.</li>
    <li>Weekend interaction can represent roommates, partners, family, work, or scheduling effects.</li>
    <li>All results are observational and should be described as association, enrichment, or ranking.</li>
  </ul>
</section>
</main>
<footer>Generated from the Copenhagen Networks Study interim feature tables.</footer>
<script>
document.querySelectorAll('.table-search').forEach(input => {{
  input.addEventListener('input', () => {{
    const table = document.getElementById(input.dataset.target);
    const query = input.value.toLowerCase();
    table.querySelectorAll('tbody tr').forEach(row => {{
      row.style.display = row.innerText.toLowerCase().includes(query) ? '' : 'none';
    }});
  }});
}});
document.querySelectorAll('table.data-table th').forEach(header => {{
  header.title = 'Click to sort';
  header.addEventListener('click', () => {{
    const table = header.closest('table');
    const body = table.querySelector('tbody');
    const index = Array.from(header.parentNode.children).indexOf(header);
    const ascending = header.dataset.order !== 'asc';
    const rows = Array.from(body.querySelectorAll('tr'));
    rows.sort((a, b) => {{
      const av = a.children[index].innerText.trim().replace(/[,%×]/g, '');
      const bv = b.children[index].innerText.trim().replace(/[,%×]/g, '');
      const an = Number(av), bn = Number(bv);
      const comparison = Number.isNaN(an) || Number.isNaN(bn)
        ? av.localeCompare(bv)
        : an - bn;
      return ascending ? comparison : -comparison;
    }});
    rows.forEach(row => body.appendChild(row));
    header.dataset.order = ascending ? 'asc' : 'desc';
  }});
}});
</script>
</body>
</html>"""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
