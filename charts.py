import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np


def show():
    df = pd.read_csv("data/data.csv")
    freq_counts = df["Frequency"].value_counts().reset_index()
    freq_counts.columns = ["Frequency", "Count"]

# Order the categories (optional but recommended)
    category_order = ["Very often", "Often", "Sometimes", "Rarely"]

    freq_counts["Frequency"] = pd.Categorical(freq_counts["Frequency"],
                                          categories=category_order,
                                          ordered=True)
    freq_counts = freq_counts.sort_values("Frequency")

# Plotly horizontal bar chart
    fig = px.bar(
      freq_counts,
      x="Count",
      y="Frequency",
      orientation="h",
      color="Frequency",
      title="📊Frequency of AI Usage",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    
    #sunburst
    df_clean = df.dropna(subset=["age_group", "Frequency", "AI_Dependency_Avg"])

# Create sunburst chart
    fig = px.sunburst(
      df_clean,
      path=["age_group", "Frequency"],
      values="AI_Dependency_Avg",
      color="AI_Dependency_Avg",
      color_continuous_scale="RdBu",
      title="🌅 Sunburst Chart: Age Group → Frequency → AI Dependency"
    )

    fig.update_layout(
     font=dict(size=14),  # normal font size
    )

# Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    
    #heatmap
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    st.subheader("🔗 Correlation Matrix")

   
    method = st.radio("Correlation method", ["pearson", "spearman"], index=0)
    cols = st.multiselect(
            "Select numeric columns",
            options=numeric_cols,
            default=numeric_cols
    )
    annot = st.checkbox("Show values", value=True)

    if not cols:
        st.warning("Please select at least one numeric column.")
        return

    corr = df[cols].corr(method=method)
    
   
    # Heatmap (normal size, no styling inflation)
    fig = px.imshow(
        corr,
        text_auto=".2f" if annot else False,
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="",
        yaxis_title=""
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")


    # Highlight AI vs Critical Thinking if present
    if "AI_Dependency_Avg" in corr.index and "CriticalThinking_Avg" in corr.columns:
        val = corr.loc["AI_Dependency_Avg", "CriticalThinking_Avg"]
        st.write(
            f"**AI_Dependency_Avg ↔ CriticalThinking_Avg correlation:** `{val:.3f}` ({method})"
        )

    with st.expander("Show correlation table"):
        st.dataframe(corr)
    
    
    #
    df["Frequency"] = df["Frequency"].astype(str).str.strip().str.title()

    # Remove NaN, empty, or "Nan" values
    df = df[df["Frequency"].notna()]
    df = df[df["Frequency"].str.lower() != "nan"]
    df = df[df["Frequency"].str.strip() != ""]
    
    # Also remove NaN in dependency
    df_clean = df.dropna(subset=["AI_Dependency_Avg"]).copy()

    # Convert AI dependency to %
    df_clean["AI_Dependency_Percent"] = df_clean["AI_Dependency_Avg"] * 20

    # List all unique frequencies from cleaned data
    unique_freqs = sorted(df_clean["Frequency"].unique().tolist())

    # Logical ordering for known categories
    order_priority = {
        "Never": 1,
        "Rarely": 2,
        "Occasionally": 3,
        "Sometimes": 4,
        "Often": 5,
        "Very Often": 6,
        "Always": 7,
        "Almost Always": 7
    }

    # Assign ranks; unknown categories go to bottom
    df_clean["freq_rank"] = df_clean["Frequency"].map(order_priority).fillna(999)

    # Sort by rank
    df_clean = df_clean.sort_values("freq_rank")

    # Final sorted category order
    final_order = df_clean["Frequency"].unique().tolist()

    # Convert Category → Categorical dtype with sorted order
    df_clean["Frequency"] = pd.Categorical(
        df_clean["Frequency"],
        categories=final_order,
        ordered=True
    )

    # Add jitter to separate points visually
    df_clean["jitter"] = np.random.uniform(-0.2, 0.2, size=len(df_clean))
    df_clean["x_pos"] = df_clean["Frequency"].cat.codes + df_clean["jitter"]

    # Scatter plot
    fig = px.scatter(
        df_clean,
        x="x_pos",
        y="AI_Dependency_Percent",
        color="Frequency",
        color_discrete_sequence=px.colors.qualitative.Dark2,
        hover_data=["Frequency", "AI_Dependency_Percent"],
        labels={"AI_Dependency_Percent": "AI Dependency (%)"},
        title= "📈 Dependency on AI vs Frequency of AI use"
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(final_order))),
            ticktext=final_order,
            title="Frequency of AI Use"
        ),
        yaxis=dict(title="AI Dependency (%)"),
        font=dict(size=14),
        margin=dict(t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
