"""
Customer Signal Detector - operations dashboard.

Designed for a retention lead opening this at 9am. Within ten seconds they
should know: how many customers need attention today, who is worst, and what
to do first. Everything else is secondary.

Runs entirely from cached pipeline output - no API key required.

    streamlit run app/dashboard.py
"""

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROC = "data/processed"

# Risk colours are semantic and used identically everywhere - table, charts,
# gauge, badges. Consistency is what lets someone read the screen at a glance
# instead of consulting a legend.
RISK_COLOURS = {
    "Critical": "#B3261E",
    "High": "#C2610A",
    "Medium": "#9A7B00",
    "Low": "#2E6B45",
}
SEVERITY_COLOURS = {
    "critical": "#B3261E",
    "high": "#C2610A",
    "medium": "#9A7B00",
    "low": "#5A6672",
}
SOURCE_LABEL = {
    "behaviour": "Behaviour agent",
    "billing": "Billing agent",
    "conversation": "Sentiment agent",
}

st.set_page_config(page_title="Customer Signal Detector",
                   page_icon="◈", layout="wide")


# --------------------------------------------------------------- data ------

@st.cache_data
def load_pipeline():
    with open(f"{PROC}/pipeline_output.json") as f:
        results = json.load(f)

    rows = []
    for r in results:
        rec = r.get("recommendation") or {}
        rows.append({
            "customer_id": r["customer_id"],
            "name": r.get("name"),
            "segment": r.get("segment"),
            "monthly_revenue": r.get("monthly_revenue") or 0,
            "risk_score": r["risk_score"],
            "risk_level": r["risk_level"],
            "confidence": r["confidence"],
            "n_signals": len(r["signals"]),
            "action": rec.get("action", "MONITOR_ONLY"),
            "priority": rec.get("priority", "P4"),
            "owner_team": rec.get("owner_team", "None"),
            "sla_hours": rec.get("sla_hours"),
            "top_signals": ", ".join(s["name"] for s in r["signals"][:3]),
            "sources": ",".join(sorted({s["source"] for s in r["signals"]})),
        })

    df = pd.DataFrame(rows).sort_values("risk_score", ascending=False)
    df["annual_value"] = df.monthly_revenue * 12
    df["value_at_risk"] = (df.risk_score / 100 * df.annual_value).round(0)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df, {r["customer_id"]: r for r in results}


@st.cache_data
def load_conversations():
    path = f"{PROC}/conversations_clean.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def money(v):
    return f"${v:,.0f}"


# --------------------------------------------------------------- header ----

def header(df):
    critical = int((df.risk_level == "Critical").sum())
    high = int((df.risk_level == "High").sum())
    at_risk = df[df.risk_level.isin(["Critical", "High"])].value_at_risk.sum()
    urgent = int((df.priority == "P1").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers monitored", len(df))
    c2.metric("Critical", critical)
    c3.metric("High", high)
    c4.metric("Annual value at risk", money(at_risk))
    c5.metric("Needs contact today", urgent)


# ------------------------------------------------------- priority list -----

def priority_tab(df):
    st.subheader("Who to contact first")
    st.caption("Ranked by risk. Sort by value at risk to prioritise by revenue "
               "instead.")

    show = df[["rank", "customer_id", "name", "segment", "risk_score",
               "risk_level", "confidence", "monthly_revenue", "value_at_risk",
               "action", "owner_team", "top_signals"]].copy()

    def colour_level(val):
        return f"color: {RISK_COLOURS.get(val, '#333')}; font-weight: 600"

    def colour_score(val):
        for threshold, level in [(75, "Critical"), (55, "High"), (30, "Medium")]:
            if val >= threshold:
                return f"background-color: {RISK_COLOURS[level]}22"
        return f"background-color: {RISK_COLOURS['Low']}18"

    styled = (show.style
              .map(colour_level, subset=["risk_level"])
              .map(colour_score, subset=["risk_score"])
              .format({"monthly_revenue": "${:,.0f}",
                       "value_at_risk": "${:,.0f}",
                       "risk_score": "{:.1f}",
                       "confidence": "{:.2f}"}))

    st.dataframe(styled, use_container_width=True, height=520,
                 hide_index=True)

    st.download_button("Download this list as CSV",
                       show.to_csv(index=False).encode(),
                       "priority_list.csv", "text/csv")


# ---------------------------------------------------------- drill-down -----

def gauge(score, level):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": RISK_COLOURS[level], "thickness": 0.7},
            "steps": [
                {"range": [0, 30], "color": "#EEF3EF"},
                {"range": [30, 55], "color": "#FAF6E3"},
                {"range": [55, 75], "color": "#FBEEE2"},
                {"range": [75, 100], "color": "#F8E6E4"},
            ],
        },
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10))
    return fig


def detail_tab(df, results, conversations):
    options = df.apply(
        lambda r: f"#{r['rank']}  {r['customer_id']} · {r['name']} "
                  f"({r['risk_level']} {r['risk_score']:.0f})", axis=1).tolist()
    chosen = st.selectbox("Customer", options, index=0)
    cid = chosen.split()[1]
    r = results[cid]
    row = df[df.customer_id == cid].iloc[0]

    left, right = st.columns([1, 2])

    with left:
        st.plotly_chart(gauge(r["risk_score"], r["risk_level"]),
                        use_container_width=True)
        st.markdown(
            f"**{r['risk_level']} risk** · confidence {r['confidence']:.0%}  \n"
            f"{row['segment']} · {row['name']}  \n"
            f"{money(row.monthly_revenue)}/month · "
            f"{money(row.value_at_risk)} annual value at risk"
        )
        if r.get("degraded"):
            st.warning(f"Assessed without: {', '.join(r['degraded'])}. "
                       f"Confidence reduced accordingly.")

    with right:
        rec = r.get("recommendation") or {}
        if rec:
            sla = f"{rec['sla_hours']}h" if rec.get("sla_hours") else "—"
            st.markdown(f"### {rec['action'].replace('_', ' ').title()}")
            st.caption(f"{rec['priority']} · {rec['owner_team']} · "
                       f"respond within {sla} · "
                       f"briefing generated by {rec.get('generated_by', 'rules')}")
            st.info(rec["explanation"])
            st.markdown("**On the call**")
            for point in rec.get("talking_points", []):
                st.markdown(f"- {point}")

    st.divider()
    st.subheader("Why this customer was flagged")
    st.caption("Each signal shows the agent that raised it and the evidence "
               "behind it.")

    by_source = defaultdict(list)
    for s in r["signals"]:
        by_source[s["source"]].append(s)

    cols = st.columns(max(len(by_source), 1))
    for col, (source, signals) in zip(cols, by_source.items()):
        with col:
            st.markdown(f"**{SOURCE_LABEL.get(source, source)}** "
                        f"({len(signals)})")
            for s in signals:
                colour = SEVERITY_COLOURS[s["severity"]]
                st.markdown(
                    f"<div style='border-left:3px solid {colour};"
                    f"padding:2px 0 2px 10px;margin-bottom:8px'>"
                    f"<span style='color:{colour};font-size:11px;"
                    f"font-weight:700;letter-spacing:.4px'>"
                    f"{s['severity'].upper()} · {s['name']}</span><br>"
                    f"<span style='font-size:13px'>{s['evidence']}</span></div>",
                    unsafe_allow_html=True)

    with st.expander("Score breakdown — every point accounted for"):
        bd = pd.DataFrame(r["score_breakdown"])
        if not bd.empty:
            st.dataframe(
                bd[["signal", "source", "weight", "severity_multiplier",
                    "recency_weight", "contribution"]],
                use_container_width=True, hide_index=True)
            st.caption(f"Raw total {bd.contribution.sum():.1f} → "
                       f"saturated score {r['risk_score']}")

    with st.expander("Agent trace"):
        # Keep every column a single type - mixing "—" with integers makes
        # Arrow serialisation warn on every render.
        trace = pd.DataFrame([
            {"agent": k,
             "signals": str(v["signals"]) if "signals" in v else "n/a",
             "latency_ms": int(v.get("latency_ms", 0)),
             "status": v.get("error") or "ok"}
            for k, v in r["agent_trace"].items()])
        st.dataframe(trace, use_container_width=True, hide_index=True)

    if not conversations.empty:
        convs = conversations[conversations.customer_id == cid]
        with st.expander(f"Support conversations ({len(convs)})"):
            if convs.empty:
                st.write("No conversations on record. Silence is itself a "
                         "signal — see NO_CONTACT above.")
            for _, c in convs.iterrows():
                st.markdown(f"**{c['channel']}** · {c['days_ago']} days ago")
                st.text(c["transcript"])
                st.divider()


# ----------------------------------------------------------- analytics -----

def analytics_tab(df, results):
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk distribution")
        counts = df.risk_level.value_counts().reindex(
            ["Low", "Medium", "High", "Critical"]).fillna(0)
        fig = px.bar(x=counts.index, y=counts.values,
                     color=counts.index, color_discrete_map=RISK_COLOURS,
                     labels={"x": "", "y": "customers"})
        fig.update_layout(showlegend=False, height=320,
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Where the work lands")
        team = (df[df.risk_level.isin(["Critical", "High"])]
                .groupby("owner_team").size().sort_values())
        fig = px.bar(x=team.values, y=team.index, orientation="h",
                     labels={"x": "customers to action", "y": ""})
        fig.update_traces(marker_color="#4A5568")
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Signal frequency by segment")
    st.caption("Which problems cluster where. Useful for fixing causes rather "
               "than symptoms.")

    rows = []
    seg = df.set_index("customer_id").segment.to_dict()
    for cid, r in results.items():
        for s in r["signals"]:
            rows.append({"segment": seg.get(cid), "signal": s["name"]})
    heat = pd.DataFrame(rows)
    if not heat.empty:
        pivot = (heat.pivot_table(index="signal", columns="segment",
                                  aggfunc=len, fill_value=0)
                 .sort_values(by=heat.segment.mode()[0], ascending=False))
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Oranges",
                        labels=dict(color="customers"))
        fig.update_layout(height=520, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk against revenue")
    st.caption("Top right is where money and risk meet.")
    fig = px.scatter(df, x="risk_score", y="monthly_revenue",
                     color="risk_level", color_discrete_map=RISK_COLOURS,
                     hover_data=["customer_id", "name", "action"],
                     labels={"risk_score": "risk score",
                             "monthly_revenue": "monthly revenue"})
    fig.update_layout(height=420, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- main -----

def main():
    st.title("Customer Signal Detector")
    st.caption("Early warning for customer operations — behavioural, billing "
               "and conversational signals in one ranked view.")

    if not os.path.exists(f"{PROC}/pipeline_output.json"):
        st.error("No pipeline output found. Run "
                 "`python notebooks/08_run_pipeline.py` first.")
        st.stop()

    df, results = load_pipeline()
    conversations = load_conversations()

    with st.sidebar:
        st.header("Filters")
        levels = st.multiselect("Risk level",
                                ["Critical", "High", "Medium", "Low"],
                                default=["Critical", "High"])
        segments = st.multiselect("Segment", sorted(df.segment.dropna().unique()),
                                  default=list(df.segment.dropna().unique()))
        actions = st.multiselect("Action", sorted(df.action.unique()),
                                 default=list(df.action.unique()))
        min_score = st.slider("Minimum risk score", 0, 100, 0)
        search = st.text_input("Search name or ID").strip().lower()

        st.divider()
        st.caption(f"{len(results)} customers assessed  \n"
                   f"Replayed from cached analysis — no API calls")

    view = df[df.risk_level.isin(levels)
              & df.segment.isin(segments)
              & df.action.isin(actions)
              & (df.risk_score >= min_score)]
    if search:
        view = view[view.customer_id.str.lower().str.contains(search)
                    | view.name.str.lower().str.contains(search)]

    header(df)
    st.divider()

    tab1, tab2, tab3 = st.tabs(
        ["Priority list", "Customer detail", "Analytics"])

    with tab1:
        if view.empty:
            st.write("No customers match these filters. Widen the risk level "
                     "or lower the minimum score.")
        else:
            priority_tab(view)
    with tab2:
        detail_tab(df if view.empty else view, results, conversations)
    with tab3:
        analytics_tab(df, results)


if __name__ == "__main__":
    main()