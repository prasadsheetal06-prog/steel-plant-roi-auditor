import os
import streamlit as st
import pandas as pd
import io
import ai_assistant as ai
from functions import (
    load_and_clean_data,
    auto_transform,
    run_audit_logic,
    create_deviation_chart,
    create_trend_chart,
)

st.set_page_config(
    page_title="Steel ROI Auditor",
    layout="wide",
    page_icon="🏗️",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #aaa; }
div[data-testid="stSidebar"]  { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Audit Settings")
    time_frame   = st.radio("Audit Duration", ["6 Months", "12 Months"])
    months_count = 6 if time_frame == "6 Months" else 12
    st.divider()

    st.header("🏢 Plant Details")
    tonnage       = st.number_input("Annual Production (Tons)", value=120_000, step=1000)
    exchange_rate = st.number_input("Exchange Rate (USD → INR)", value=83.5, step=0.5)
    st.caption(f"Monthly tonnage: {tonnage/12:,.0f} tons")
    st.divider()

    st.header("📂 Upload Data")
    file_mode = st.radio(
        "File type",
        ["Raw Plant Excel", "Pre-transformed File"],
        help=(
            "Raw Plant Excel: your original master file "
            "(e.g. Steel_Plant_Process_Master.xlsx)\n\n"
            "Pre-transformed: already processed through transform.py"
        ),
    )
    uploaded_file = st.file_uploader("Upload File", type=["xlsx"])

    if file_mode == "Raw Plant Excel":
        st.caption(
            "App will auto-detect sheet, headers, and month columns. "
            "No manual steps needed."
        )
    else:
        st.caption("Must have columns: Parameters, UOM, Month, Metric_Value")

    # ── Step 4: Sample template download ─────────────────────────────────────
    # Place your Steel_Plant_Process_Master.xlsx in the same folder as app.py.
    # This lets any new user download it, fill in their data, and upload it back.
    st.divider()
    st.caption("New user? Download the sample template:")
    template_path = "Steel_Plant_Process_Master.xlsx"
    if os.path.exists(template_path):
        with open(template_path, "rb") as f:
            st.download_button(
                label="⬇️ Sample Template",
                data=f,
                file_name="Steel_Plant_Template.xlsx",
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
            )
    else:
        st.caption("_(template not found in project folder)_")

# ── Init AI (always before any st.stop()) ────────────────────────────────────
ai.init_chat_history()

st.title("🏗️ Steel Process Audit & Comparison")

# ── Guard: no file ────────────────────────────────────────────────────────────
if not uploaded_file or tonnage <= 0:
    st.info("Upload your plant Excel file and configure plant details in the sidebar.")

    st.markdown("""
    ### How to get started
    1. **Choose file type** in the sidebar — *Raw Plant Excel* for your master file,
       *Pre-transformed* if you already ran `transform.py`
    2. **Upload your `.xlsx` file**
    3. Set your **annual production** and **exchange rate**
    4. The audit runs automatically — no coding needed

    > New user? Download the **Sample Template** from the sidebar,
    > fill in your plant data, and upload it.
    """)
    st.stop()

# ── Load & transform ──────────────────────────────────────────────────────────
if file_mode == "Raw Plant Excel":
    df, load_status = auto_transform(uploaded_file)
    if df is not None:
        st.sidebar.success(f"✅ {load_status}")
else:
    df, load_status = load_and_clean_data(uploaded_file)

if df is None:
    st.error(load_status)
    with st.expander("What went wrong? — common fixes", expanded=True):
        st.markdown("""
        - Make sure your Excel has a column named **Parameters**
        - Month columns should be named like `Jan'24`, `Feb'24`, or `Jan-24`
        - Try switching between **Raw / Pre-transformed** mode in the sidebar
        - Check that the correct sheet in your Excel contains the data
        - If using Raw mode, make sure the year sheet (e.g. `2024`) exists
        """)
    st.stop()

# ── Run audit ─────────────────────────────────────────────────────────────────
summary, status = run_audit_logic(df, months_count, tonnage, exchange_rate)
if not summary:
    st.error("No matching parameters found in your file.")
    with st.expander("How to fix this", expanded=True):
        st.markdown("""
        The audit scans for these keywords in your **Parameters** column:

        | Keyword searched | Maps to metric |
        |---|---|
        | EAF | EAF Power |
        | Elec Cons | Electrode |
        | F/c Lime | Lime |
        | F/c Dolime | Dolomite |
        | Natural Gas | Natural Gas |
        | Ch -> LM | Yield |
        | Argon | Argon |
        | Nitrogen | Nitrogen |

        Make sure your Parameters column contains at least some of these terms.
        """)
    st.stop()

summary_df = pd.DataFrame(summary)
st.success(f"Audit complete — {time_frame} | {len(summary_df)} metrics analysed")

# ── Section 1: KPI Cards ──────────────────────────────────────────────────────
st.subheader("📌 Summary")
k1, k2, k3, k4 = st.columns(4)

n_bad             = (summary_df["Deviation %"] > 5).sum()
total_cost_impact = summary_df["Est. Extra Cost/Month (INR)"].sum()
worst_metric      = summary_df.loc[summary_df["Deviation %"].idxmax(), "Metric"]

k1.metric("Metrics Audited", len(summary_df))
k2.metric(
    "Above Benchmark",
    f"{n_bad} / {len(summary_df)}",
    delta=f"{n_bad} need attention",
    delta_color="inverse",
)
k3.metric(
    "Est. Extra Cost / Month",
    f"₹{total_cost_impact:,.0f}",
    delta="vs hitting all benchmarks",
    delta_color="inverse",
)
k4.metric("Worst Performing", worst_metric)

# ── Section 2: Deviation Chart + Cost Table ───────────────────────────────────
st.divider()
col_chart, col_table = st.columns([3, 2])

with col_chart:
    st.subheader("📉 Performance Gap (% from Benchmark)")
    st.plotly_chart(create_deviation_chart(summary), use_container_width=True)

with col_table:
    st.subheader("💰 Cost Impact by Metric")
    cost_df = summary_df[
        ["Metric", "Deviation %", "Est. Extra Cost/Month (INR)", "Status"]
    ].copy()
    cost_df = cost_df.sort_values("Est. Extra Cost/Month (INR)", ascending=False)
    cost_df["Est. Extra Cost/Month (INR)"] = cost_df[
        "Est. Extra Cost/Month (INR)"
    ].apply(lambda x: f"₹{x:,.0f}")
    cost_df["Deviation %"] = cost_df["Deviation %"].apply(lambda x: f"{x:+.1f}%")
    st.dataframe(cost_df, use_container_width=True, hide_index=True)

# ── Section 3: Trend Chart ────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Monthly Trend vs Benchmark")
st.plotly_chart(create_trend_chart(df, months_count), use_container_width=True)

# ── Section 4: Drill-down Table ───────────────────────────────────────────────
st.divider()
with st.expander("🔍 Full Parameter Breakdown", expanded=False):
    display_df = summary_df.copy()
    display_df["Deviation %"] = display_df["Deviation %"].apply(
        lambda x: f"{x:+.1f}%"
    )
    display_df["Est. Extra Cost/Month (INR)"] = display_df[
        "Est. Extra Cost/Month (INR)"
    ].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Section 5: Download ───────────────────────────────────────────────────────
st.divider()
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Audit Summary", index=False)
    df.to_excel(writer, sheet_name="Raw Data", index=False)
buf.seek(0)
st.download_button(
    label="⬇️ Download Audit Report (.xlsx)",
    data=buf,
    file_name=f"Steel_Audit_Report_{time_frame.replace(' ', '_')}.xlsx",
    mime=(
        "application/vnd.openxmlformats-"
        "officedocument.spreadsheetml.sheet"
    ),
)

# ── Section 6: SteelMind AI ───────────────────────────────────────────────────
st.divider()
st.subheader("🤖 SteelMind — AI Process Assistant")
st.caption("Ask anything about your audit: deviations, costs, root causes, custom charts.")

data_context = ai.build_data_context(
    df=df,
    summary_df=summary_df,
    tonnage=tonnage,
    exchange_rate=exchange_rate,
)

ai.render_chat_history()

if not st.session_state.chat_history:
    st.markdown("**Try asking:**")
    c1, c2, c3 = st.columns(3)
    if c1.button("Why is EAF Power high?"):
        st.session_state._prefill = (
            "Why might EAF Power be above benchmark? "
            "What should we investigate first?"
        )
    if c2.button("Cost breakdown chart"):
        st.session_state._prefill = (
            "Generate a plotly bar chart of Est. Extra Cost/Month (INR) "
            "per metric from summary_df, sorted descending"
        )
    if c3.button("Management summary"):
        st.session_state._prefill = (
            "Write a concise 5-line management summary of this audit "
            "suitable for a plant director"
        )

prefill    = st.session_state.pop("_prefill", "")
user_input = st.chat_input("Ask SteelMind about your plant data…") or prefill

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⚙️ Analysing your plant data…")
        response = ai.ask_steelmind(user_input, data_context)
        placeholder.empty()
        ai.render_response(response, df, summary_df)

# ── Footer + Clear ────────────────────────────────────────────────────────────
st.divider()
col_info, col_clear = st.columns([5, 1])
col_info.caption(
    f"SteelMind · Groq LLaMA 3.3 70B · "
    f"Free tier: 30 req/min · "
    f"History: {len(st.session_state.chat_history)} messages"
)
if st.session_state.chat_history:
    if col_clear.button("🗑️ Clear chat"):
        ai.clear_chat()
        st.rerun()
        