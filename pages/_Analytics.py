import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database import get_accidents_df

st.set_page_config(page_title="📊 Analytics", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg,#0d0f18 0%,#141824 100%); color:#e8eaf0; }
h1, h2, h3 { font-family: 'Rajdhani', sans-serif; }
.metric-card {
    background:linear-gradient(135deg,#1a1d2e,#1f2340);
    border:1px solid #2a2f4a; border-radius:12px; padding:14px 18px; margin-bottom:10px;
}
.metric-label { color:#8892b0; font-size:10px; font-weight:600; letter-spacing:1.2px; text-transform:uppercase; }
.metric-value { color:#e8eaf0; font-size:26px; font-family:'Rajdhani',sans-serif; font-weight:700; }
div[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0a0c16 0%,#111422 100%);
    border-right:1px solid #1e2235;
}
div[data-testid="stSidebar"] * { color:#c8cee0 !important; }
</style>
""", unsafe_allow_html=True)

# Plotly dark theme for all charts
CHART_THEME = dict(
    paper_bgcolor="#0d0f18",
    plot_bgcolor="#1a1d2e",
    font=dict(color="#e8eaf0", family="Inter"),
    xaxis=dict(gridcolor="#2a2f4a", zeroline=False),
    yaxis=dict(gridcolor="#2a2f4a", zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

st.markdown("<h1>📊 ACCIDENT ANALYTICS DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8892b0;margin-top:-10px;'>Trends, patterns and vehicle breakdown from all detection sessions</p>",
            unsafe_allow_html=True)
st.markdown("---")

df = get_accidents_df()
if df.empty:
    st.info("No data yet. Run the detection system to generate logs.")
    st.stop()

df["date"]  = pd.to_datetime(df["date"], errors="coerce")
df["hour"]  = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce").dt.hour

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")
min_d = df["date"].min().date()
max_d = df["date"].max().date()
start = st.sidebar.date_input("Start Date", min_d, min_value=min_d, max_value=max_d)
end   = st.sidebar.date_input("End Date",   max_d, min_value=min_d, max_value=max_d)

fdf = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)].copy()
if fdf.empty:
    st.warning("No data in selected range.")
    st.stop()

# ── KPI row ──────────────────────────────────────────────────────────────────
total   = len(fdf)
avg_c   = fdf["confidence"].dropna().mean()
days    = max((end - start).days + 1, 1)
per_day = round(total / days, 1)

c1, c2, c3, c4 = st.columns(4)
for col, label, val, color in [
    (c1, "Total Accidents",  total,                             "#ff4444"),
    (c2, "Avg Confidence",   f"{avg_c:.1%}" if avg_c==avg_c else "—", "#00b894"),
    (c3, "Accidents / Day",  per_day,                           "#64b5f6"),
    (c4, "Days Covered",     days,                              "#ffb432"),
]:
    col.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value' style='color:{color};'>{val}</div></div>""",
        unsafe_allow_html=True)

st.markdown("---")

# ── Row 1: Daily trend + Weekly trend ────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    daily = fdf.groupby(fdf["date"].dt.date).size().reset_index(name="count")
    fig = px.bar(daily, x="date", y="count", title="Accidents Per Day",
                 text="count", color_discrete_sequence=["#ff4444"])
    fig.update_traces(textposition="outside", marker_line_color="#cc0000", marker_line_width=1)
    fig.update_layout(**CHART_THEME, title_font_size=14)
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    fdf["week"] = fdf["date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly = fdf.groupby("week").size().reset_index(name="count")
    fig2 = px.line(weekly, x="week", y="count", title="Accidents Per Week",
                   markers=True, color_discrete_sequence=["#3a7bd5"])
    fig2.update_traces(line_width=2.5, marker_size=8)
    fig2.update_layout(**CHART_THEME, title_font_size=14)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Hourly heatmap + Vehicle breakdown ─────────────────────────────────
col_l2, col_r2 = st.columns(2)

with col_l2:
    # Accidents by hour of day
    hourly = fdf.groupby("hour").size().reset_index(name="count")
    all_hours = pd.DataFrame({"hour": range(24)})
    hourly = all_hours.merge(hourly, on="hour", how="left").fillna(0)

    fig3 = px.bar(hourly, x="hour", y="count",
                  title="Accidents by Hour of Day",
                  color="count",
                  color_continuous_scale=["#1a1d2e", "#ff4444"],
                  labels={"hour": "Hour (0–23)", "count": "Accidents"})
    fig3.update_layout(**CHART_THEME, title_font_size=14,
                       coloraxis_showscale=False)
    fig3.update_traces(marker_line_width=0)
    st.plotly_chart(fig3, use_container_width=True)

with col_r2:
    # Vehicle type breakdown from 'vehicles' column
    if "vehicles" in fdf.columns and fdf["vehicles"].notna().any():
        vehicle_series = (
            fdf["vehicles"]
            .dropna()
            .str.split(",")
            .explode()
            .str.strip()
            .replace("", pd.NA)
            .dropna()
        )
        vc = vehicle_series.value_counts().reset_index()
        vc.columns = ["Vehicle", "Count"]

        COLORS = {
            "Car":           "#32ff82",
            "Truck":         "#3282ff",
            "Motorcycle":    "#ff8232",
            "Person":        "#ffb432",
            "Bus":           "#b432ff",
            "Auto Rickshaw": "#00c8ff",
        }
        vc["color"] = vc["Vehicle"].map(COLORS).fillna("#8892b0")

        fig4 = px.pie(
            vc, names="Vehicle", values="Count",
            title="Objects Involved in Accidents",
            color="Vehicle",
            color_discrete_map=COLORS,
            hole=0.45,
        )
        fig4.update_layout(**CHART_THEME, title_font_size=14,
                           legend=dict(font=dict(color="#e8eaf0")))
        fig4.update_traces(textfont_color="white", textinfo="percent+label")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Vehicle breakdown data will appear here once you run detection with the updated app.")

# ── Row 3: Confidence over time ───────────────────────────────────────────────
if "confidence" in fdf.columns and fdf["confidence"].notna().any():
    st.markdown("---")
    conf_df = fdf.dropna(subset=["confidence"]).copy()
    conf_df["datetime"] = pd.to_datetime(
        conf_df["date"].dt.strftime("%Y-%m-%d") + " " + conf_df["time"].astype(str),
        errors="coerce"
    )
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=conf_df["datetime"], y=conf_df["confidence"],
        mode="markers+lines",
        marker=dict(color="#ff4444", size=7, opacity=0.8),
        line=dict(color="#ff4444", width=1.5, dash="dot"),
        name="Confidence"
    ))
    fig5.add_hline(y=0.5, line_dash="dash", line_color="#8892b0",
                   annotation_text="0.5 threshold", annotation_font_color="#8892b0")
    fig5.update_layout(
        **CHART_THEME,
        title="Accident Detection Confidence Over Time",
        title_font_size=14,
        yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#2a2f4a"),
        showlegend=False,
    )
    st.plotly_chart(fig5, use_container_width=True)

# ── Location table ────────────────────────────────────────────────────────────
if "location_name" in fdf.columns and fdf["location_name"].notna().any():
    st.markdown("---")
    st.subheader("📍 Accidents by Location")
    loc_counts = (
        fdf["location_name"].dropna()
        .value_counts()
        .reset_index()
    )
    loc_counts.columns = ["Location", "Accidents"]
    st.dataframe(loc_counts, use_container_width=True, hide_index=True)

# ── Export ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.download_button(
    "⬇️ Download Analytics Data (CSV)",
    fdf.to_csv(index=False).encode("utf-8"),
    file_name=f"analytics_{start}_{end}.csv",
    mime="text/csv"
)