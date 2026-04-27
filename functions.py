import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

# ── Benchmarks ────────────────────────────────────────────────────────────────
BENCHMARKS = {
    "EAF Power":   {"value": 407.54, "unit": "kWh/ton",  "cost_usd_per_unit": 0.08},
    "Electrode":   {"value": 0.85,   "unit": "kg/ton",   "cost_usd_per_unit": 3.50},
    "Lime":        {"value": 22.0,   "unit": "kg/ton",   "cost_usd_per_unit": 0.04},
    "Dolomite":    {"value": 12.0,   "unit": "kg/ton",   "cost_usd_per_unit": 0.05},
    "Natural Gas": {"value": 32.0,   "unit": "Nm3/ton",  "cost_usd_per_unit": 0.35},
    "Yield":       {"value": 92.5,   "unit": "%",        "cost_usd_per_unit": 0},
    "Argon":       {"value": 0.22,   "unit": "Nm3/ton",  "cost_usd_per_unit": 0.80},
    "Nitrogen":    {"value": 32.0,   "unit": "Nm3/ton",  "cost_usd_per_unit": 0.12},
}

PARAMS_MAP = {
    "EAF":         "EAF Power",
    "Elec Cons":   "Electrode",
    "F/c Lime":    "Lime",
    "F/c Dolime":  "Dolomite",
    "Natural Gas": "Natural Gas",
    "Ch -> LM":    "Yield",
    "Argon":       "Argon",
    "Nitrogen":    "Nitrogen",
}


# ── Load pre-transformed file ─────────────────────────────────────────────────
def load_and_clean_data(file):
    try:
        df = pd.read_excel(file)
        df.columns = [str(c).strip() for c in df.columns]
        required = ["Parameters", "Month", "Metric_Value"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, (
                f"Missing columns: {', '.join(missing)}. "
                "Switch to 'Raw Plant Excel' mode or run transform.py first."
            )
        return df, "Success"
    except Exception as e:
        return None, str(e)


# ── Auto-transform raw plant Excel ────────────────────────────────────────────
def auto_transform(file) -> tuple:
    """
    Accepts a raw plant Excel file (e.g. Steel_Plant_Process_Master.xlsx)
    and returns a tidy DataFrame ready for audit.
    Works on any sheet that contains a 'Parameters' column and month columns.
    No manual transform.py needed.
    """
    try:
        xl = pd.ExcelFile(file)
        sheet_names = xl.sheet_names

        # Find best sheet — prefer sheets named with a year
        year_pattern = re.compile(r"20\d{2}")
        target_sheet = None
        for sheet in sheet_names:
            if year_pattern.search(str(sheet)):
                target_sheet = sheet
                break
        if target_sheet is None:
            target_sheet = sheet_names[0]

        # Read raw without assuming header position
        raw_df = pd.read_excel(file, sheet_name=target_sheet, header=None)

        # Find row containing "Parameters"
        header_row_index = None
        for i, row in raw_df.iterrows():
            if any("parameter" in str(v).lower() for v in row.values):
                header_row_index = i
                break

        if header_row_index is None:
            return None, (
                "Could not find a 'Parameters' column in your file. "
                "Make sure your Excel has a column named 'Parameters'."
            )

        # Re-read from the detected header row
        df = pd.read_excel(
            file, sheet_name=target_sheet, header=header_row_index
        )
        df.columns = [str(c).strip() for c in df.columns]

        # Find Parameters and UOM columns
        id_col = next(
            (c for c in df.columns if "parameter" in c.lower()), None
        )
        uom_col = next(
            (c for c in df.columns if "uom" in c.lower()), None
        )

        if not id_col:
            return None, "No 'Parameters' column found after header detection."

        if not uom_col:
            df["UOM"] = ""
            uom_col = "UOM"

        # Detect month columns — supports Jan'24, Feb-24, Jan 24, January 2024
        month_pattern = re.compile(
            r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"['\-\s]?\d{2,4}$",
            re.IGNORECASE,
        )
        available_months = [
            c for c in df.columns if month_pattern.match(str(c))
        ]

        if not available_months:
            return None, (
                f"No month columns detected in sheet '{target_sheet}'. "
                "Expected format: Jan'24, Feb'24, Jan-24, or Feb 24."
            )

        # Melt to tidy long format
        tidy_df = pd.melt(
            df,
            id_vars=[id_col, uom_col],
            value_vars=available_months,
            var_name="Month",
            value_name="Metric_Value",
        )
        tidy_df = tidy_df.rename(
            columns={id_col: "Parameters", uom_col: "UOM"}
        )
        tidy_df["Metric_Value"] = pd.to_numeric(
            tidy_df["Metric_Value"], errors="coerce"
        )
        tidy_df = tidy_df.dropna(subset=["Metric_Value"])

        return tidy_df, (
            f"Auto-transformed from sheet '{target_sheet}' — "
            f"{len(tidy_df)} rows, {len(available_months)} months detected"
        )

    except Exception as e:
        return None, f"Transform failed: {e}"


# ── Audit Logic ───────────────────────────────────────────────────────────────
def run_audit_logic(df, months_to_audit, annual_tonnage, exchange_rate):
    all_months = sorted(df["Month"].unique())
    selected_months = all_months[:months_to_audit]
    df = df[df["Month"].isin(selected_months)].copy()

    monthly_tonnage = annual_tonnage / 12
    summary_data = []

    for keyword, formal_name in PARAMS_MAP.items():
        rows = df[df["Parameters"].str.contains(keyword, case=False, na=False)]
        if rows.empty:
            continue

        bench_info         = BENCHMARKS.get(formal_name, {})
        bench_val          = bench_info.get("value", 0)
        unit               = bench_info.get("unit", "")
        cost_per_unit_usd  = bench_info.get("cost_usd_per_unit", 0)

        monthly_avg_total = rows.groupby("Month")["Metric_Value"].sum().mean()

        avg_val = (
            monthly_avg_total / monthly_tonnage
            if "/ton" in unit
            else monthly_avg_total
        )

        diff        = avg_val - bench_val
        perc_change = (diff / bench_val) * 100 if bench_val != 0 else 0

        extra_per_ton          = max(diff, 0) if formal_name != "Yield" else 0
        monthly_cost_impact_inr = (
            extra_per_ton * monthly_tonnage * cost_per_unit_usd * exchange_rate
        )

        status = (
            "✅ On Target"    if abs(perc_change) < 5  else
            "⚠️ Slightly Over" if abs(perc_change) < 15 else
            "🔴 High Variance"
        )

        summary_data.append({
            "Metric":                      formal_name,
            "Unit":                        unit,
            "Actual (per ton)":            round(avg_val, 2),
            "Benchmark":                   bench_val,
            "Deviation %":                 round(perc_change, 2),
            "Est. Extra Cost/Month (INR)": round(monthly_cost_impact_inr),
            "Status":                      status,
        })

    return summary_data, "Success"


# ── Deviation Chart ───────────────────────────────────────────────────────────
def create_deviation_chart(summary_data):
    df_plot = pd.DataFrame(summary_data)
    df_plot = df_plot.sort_values("Deviation %", ascending=True)

    colors = [
        "#2E7D32" if v <= 5 else ("#FF9800" if v <= 15 else "#D32F2F")
        for v in df_plot["Deviation %"]
    ]

    fig = go.Figure(go.Bar(
        x=df_plot["Deviation %"],
        y=df_plot["Metric"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in df_plot["Deviation %"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Deviation: %{x:.1f}%<extra></extra>",
    ))

    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="gray")
    fig.add_vrect(
        x0=-5, x1=5, fillcolor="green", opacity=0.07, line_width=0,
        annotation_text="Target zone", annotation_position="top right",
    )
    fig.update_layout(
        title="% Deviation from Benchmark (all metrics, same scale)",
        xaxis_title="% Above / Below Benchmark",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        height=350,
        margin=dict(l=20, r=60, t=50, b=40),
    )
    return fig


# ── Trend Chart ───────────────────────────────────────────────────────────────
def create_trend_chart(df, months_to_audit):
    all_months = sorted(df["Month"].unique())
    selected   = all_months[:months_to_audit]
    df         = df[df["Month"].isin(selected)].copy()

    colors = [
        "#42A5F5", "#EF5350", "#66BB6A", "#FFA726",
        "#AB47BC", "#26C6DA", "#EC407A", "#8D6E63",
    ]
    fig = go.Figure()

    for i, (keyword, formal_name) in enumerate(PARAMS_MAP.items()):
        rows = df[df["Parameters"].str.contains(keyword, case=False, na=False)]
        if rows.empty:
            continue
        monthly   = rows.groupby("Month")["Metric_Value"].sum().reindex(selected)
        bench_val = BENCHMARKS.get(formal_name, {}).get("value")
        if bench_val is None:
            continue
        pct_of_bench = (monthly / bench_val * 100).round(1)
        fig.add_trace(go.Scatter(
            x=list(selected), y=pct_of_bench,
            name=formal_name, mode="lines+markers",
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6),
        ))

    fig.add_hline(
        y=100, line_dash="dash", line_color="gray",
        annotation_text="Benchmark = 100%", annotation_position="right",
    )
    fig.update_layout(
        title="Monthly Performance as % of Benchmark",
        xaxis_title="Month",
        yaxis_title="% of Benchmark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        height=380,
        legend=dict(orientation="h", y=-0.25),
    )
    return fig
