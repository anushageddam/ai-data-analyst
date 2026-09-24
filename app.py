"""
YOO PROJECT — AI-Powered Business Intelligence & Data Analytics
Automating Power BI Workflows: Profiling, Semantic Understanding, Automated KPIs,
Decision-Tree Visualizations, Evidence-Based Insights, Forecasting, and Grounded AI Chat.

Flagship Web Application available at: http://127.0.0.1:8000
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from backend.core.data_sources import load_sample_dataset, FileDataSource
from backend.core.profiler import DataProfiler
from backend.core.semantic import SemanticEngine
from backend.core.data_model import AnalyticalModel
from backend.core.kpi_engine import KPIEngine
from backend.core.vis_engine import VisDecisionEngine
from backend.core.insights import EvidenceInsightEngine
from backend.core.forecasting import ForecastEngine
from backend.core.chat_engine import AIChatEngine

st.set_page_config(
    page_title="YOO PROJECT — AI BI & Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #6366f1; margin-bottom: 0px; }
    .subtitle { font-size: 0.95rem; color: #94a3b8; margin-bottom: 20px; }
    .metric-box { background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; }
    .insight-card { background: rgba(30, 41, 59, 0.7); border-left: 4px solid #6366f1; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR: DATA INPUT & CONTROLS
st.sidebar.markdown("## 📊 YOO PROJECT")
st.sidebar.markdown("**Automated Power BI Experience**")

data_source_mode = st.sidebar.radio(
    "Data Source",
    ["🏢 SAP S/4HANA Procurement", "📈 Global Enterprise Sales", "📁 Upload Custom File (CSV/XLSX)"]
)

df = None
dataset_meta = {}

if data_source_mode == "🏢 SAP S/4HANA Procurement":
    df, dataset_meta = load_sample_dataset("sap_procurement")
elif data_source_mode == "📈 Global Enterprise Sales":
    df, dataset_meta = load_sample_dataset("enterprise_sales")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your data (.csv, .xlsx)", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        source = FileDataSource(file_bytes, uploaded_file.name)
        if source.is_excel:
            sheets = source.get_sheets()
            selected_sheet = st.sidebar.selectbox("Select Sheet", sheets)
            df = source.load(selected_sheet)
        else:
            df = source.load()
        dataset_meta = source.get_metadata()
    else:
        st.info("Please upload a CSV or Excel file, or choose a pre-loaded enterprise dataset.")
        st.stop()

# -------------------------------------------------------------
# RUN AUTOMATED ANALYTICS PIPELINE
# -------------------------------------------------------------
profiler = DataProfiler(df)
prof_summary = profiler.profile_dataset()

semantic_eng = SemanticEngine(df, prof_summary)
sem_summary = semantic_eng.get_understanding_summary()

# Shared analytical model used by future measures, visuals, filters and AI.
analytical_model = AnalyticalModel(df, prof_summary, sem_summary)
data_model = analytical_model.build()

# SIDEBAR FILTERS (AUTOMATICALLY DETECTED)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Global Cross-Filters")
active_filters = {}
for filter_dim in sem_summary.get("filter_dimensions", []):
    unique_vals = sorted([str(v) for v in df[filter_dim].dropna().unique()])
    selected = st.sidebar.multiselect(f"{filter_dim.replace('_', ' ')}", unique_vals)
    if selected:
        active_filters[filter_dim] = selected

# Apply active filters
filtered_df = df.copy()
for col, vals in active_filters.items():
    filtered_df = filtered_df[filtered_df[col].astype(str).isin(vals)]

# Recompute with filtered subset
kpi_eng = KPIEngine(filtered_df, sem_summary)
kpis = kpi_eng.generate_kpis()

vis_eng = VisDecisionEngine(filtered_df, sem_summary, prof_summary)
visuals = vis_eng.generate_dashboard_visuals()

insight_eng = EvidenceInsightEngine(filtered_df, sem_summary, prof_summary)
insights = insight_eng.generate_insights()

forecast_eng = ForecastEngine(filtered_df, sem_summary)
chat_eng = AIChatEngine(filtered_df, sem_summary, forecast_eng)

# HEADER DISPLAY
st.markdown('<div class="main-title">YOO PROJECT</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">Domain Detected: <strong>{sem_summary["detected_domain"]}</strong> • '
    f'Active Rows: <strong>{len(filtered_df):,} of {len(df):,}</strong> • '
    f'Data Health Score: <strong>{prof_summary["summary"]["data_health_score"]}% ({prof_summary["summary"]["data_health_status"]})</strong></div>',
    unsafe_allow_html=True
)

st.sidebar.info("💡 **Tip**: The flagship Single Page Application is also running at [http://127.0.0.1:8000](http://127.0.0.1:8000)")

# TABS INTERFACE
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 AI Dashboard",
    "🛡️ Data Profiling & Quality",
    "🧩 Data Model",
    "💡 Evidence Insights",
    "🔮 Predictive Forecast",
    "📋 Data Explorer",
    "💬 Ask AI Analyst"
])

# -------------------------------------------------------------
# TAB 1: AI DASHBOARD
# -------------------------------------------------------------
with tab1:
    st.subheader("Key Performance Indicators")
    kpi_cols = st.columns(len(kpis))
    for idx, kpi in enumerate(kpis):
        with kpi_cols[idx]:
            delta_str = f"{kpi['delta_percentage']:+0.1f}% vs prev" if kpi.get('delta_percentage') is not None else None
            st.metric(label=kpi["title"], value=kpi["value"], delta=delta_str)

    st.markdown("---")
    st.subheader("Automated BI Visualizations")
    for i in range(0, len(visuals), 2):
        row_cols = st.columns(2)
        for j in range(2):
            if i + j < len(visuals):
                v = visuals[i + j]
                with row_cols[j]:
                    st.markdown(f"**{v['title']}**")
                    st.caption(v.get("description", ""))
                    fig = go.Figure(data=v["plotly_spec"]["data"], layout=v["plotly_spec"]["layout"])
                    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: DATA PROFILING & QUALITY
# -------------------------------------------------------------
with tab2:
    st.subheader("Data Health & Schema Profiler")
    c1, c2, c3, c4 = st.columns(4)
    s = prof_summary["summary"]
    c1.metric("Data Health Score", f"{s['data_health_score']}%", s["data_health_status"])
    c2.metric("Missing Cells", f"{s['total_null_cells']} ({s['overall_null_percentage']}%)")
    c3.metric("Duplicate Rows", f"{s['duplicate_rows']} ({s['duplicate_percentage']}%)")
    c4.metric("Outliers Detected", s["total_outliers_detected"])

    st.markdown("#### Column Schema & Distributions")
    col_table = []
    for c in prof_summary["columns"]:
        col_table.append({
            "Column": c["name"],
            "Analytical Type": c["type"],
            "Null %": f"{c['null_percentage']}%",
            "Unique Count": c["unique_count"],
            "Min / Max": f"{c['min']} / {c['max']}" if c["min"] is not None else "--",
            "Mean / Median": f"{c['mean']} / {c['median']}" if c["mean"] is not None else "--",
            "Outliers": c["outlier_count"]
        })
    st.dataframe(pd.DataFrame(col_table), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: DATA MODEL
# -------------------------------------------------------------
with tab3:
    st.subheader("Analytical Data Model")
    st.caption("Shared model of the active dataset: tables, field roles, types and basic metadata.")

    model_table = data_model["tables"][0]
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Tables", len(data_model["tables"]))
    mc2.metric("Fields", len(model_table["fields"]))
    mc3.metric("Relationships", len(data_model["relationships"]))

    st.markdown(f"**Table:** {model_table["name"]} — {model_table["row_count"]:,} rows × {model_table["column_count"]} columns")

    model_rows = [
        {
            "Field": field["name"],
            "Role": field["role"],
            "Analytical Type": field["analytical_type"],
            "Unique Values": field["unique_count"],
            "Nullable": "Yes" if field["nullable"] else "No",
            "Sample Values": ", ".join(field["sample_values"]),
        }
        for field in model_table["fields"]
    ]
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True)

    if data_model["relationships"]:
        st.markdown("#### Relationships")
        st.dataframe(pd.DataFrame(data_model["relationships"]), use_container_width=True)
    else:
        st.caption("No relationships inferred for this single active dataset.")

# -------------------------------------------------------------
# TAB 4: EVIDENCE-BASED INSIGHTS
# -------------------------------------------------------------
with tab3:
    st.subheader("Evidence-Based Business Insights")
    st.caption("All findings are strictly backed by verifiable mathematical computations over the dataset.")
    for ins in insights:
        st.markdown(f"""
        <div class="insight-card">
            <h4>{ins['title']}</h4>
            <p>{ins['description']}</p>
            <small>Category: <strong>{ins['category']}</strong> | Evidence Metric: <strong>{ins['metric_value']}</strong></small>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 4: PREDICTIVE FORECASTING
# -------------------------------------------------------------
with tab4:
    st.subheader("Predictive Time-Series Forecast")
    measures = sem_summary.get("semantic_roles", {}).get("all_measures", [])
    fc_col1, fc_col2 = st.columns([2, 1])
    with fc_col1:
        target_m = st.selectbox("Select Target Measure", measures)
    with fc_col2:
        horizon = st.selectbox("Forecast Horizon", [3, 6, 12])

    fc_result = forecast_eng.run_forecast(target_measure=target_m, horizon=horizon)
    if fc_result["is_suitable"]:
        st.info(fc_result["narrative"])
        fig_fc = go.Figure(data=fc_result["plotly_spec"]["data"], layout=fc_result["plotly_spec"]["layout"])
        st.plotly_chart(fig_fc, use_container_width=True)

        m = fc_result["metrics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Historical Mean", f"{m['historical_mean']:,.2f}")
        m2.metric("Forecast Mean", f"{m['forecast_mean']:,.2f}")
        m3.metric("Model Error (MAE)", f"{m['mae']:,.2f}")
        m4.metric("Reliability (MAPE)", f"{m['mape']}%")
    else:
        st.warning(fc_result["reason"])

# -------------------------------------------------------------
# TAB 5: DATA EXPLORER
# -------------------------------------------------------------
with tab5:
    st.subheader("Underlying Data Explorer")
    search_q = st.text_input("Search records...")
    exp_df = filtered_df.copy()
    if search_q.strip():
        mask = exp_df.astype(str).apply(lambda row: row.str.lower().str.contains(search_q.lower(), na=False)).any(axis=1)
        exp_df = exp_df[mask]
    st.dataframe(exp_df, use_container_width=True)

# -------------------------------------------------------------
# TAB 6: ASK AI ANALYST (CHATBOT)
# -------------------------------------------------------------
with tab6:
    st.subheader("Conversational AI Data Analyst")
    st.caption("Ask questions about your data. The engine maintains conversational context and avoids hallucinated numbers.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart"):
                fig = go.Figure(data=msg["chart"]["data"], layout=msg["chart"]["layout"])
                st.plotly_chart(fig, use_container_width=True)

    user_query = st.chat_input("Ask your data anything (e.g. 'Which vendor has highest PO value?' or 'Show monthly trend')...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        res = chat_eng.process_query(user_query)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": res["answer"],
            "chart": res.get("chart")
        })
        with st.chat_message("assistant"):
            st.markdown(res["answer"])
            if res.get("chart"):
                fig = go.Figure(data=res["chart"]["data"], layout=res["chart"]["layout"])
                st.plotly_chart(fig, use_container_width=True)