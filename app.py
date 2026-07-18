import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================================================================
# 1. PAGE CONFIGURATION & STATE INITIALIZATION
# ==============================================================================
st.set_page_config(
    page_title="FIFA Player Valuation Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# ==============================================================================
# 2. DESIGN SYSTEM & CUSTOM CSS
# ==============================================================================
# Define color tokens based on theme
bg = "#09090b" if IS_DARK else "#ffffff"
bg_subtle = "#0c0c0f" if IS_DARK else "#f9fafb"
card = "#0c0c0f" if IS_DARK else "#ffffff"
card_hover = "#131316" if IS_DARK else "#f4f4f5"
border = "#1e1e24" if IS_DARK else "#e4e4e7"
border_subtle = "#16161a" if IS_DARK else "#f0f0f2"
text = "#fafafa" if IS_DARK else "#09090b"
text_muted = "#71717a"
text_dim = "#52525b" if IS_DARK else "#a1a1aa"
accent = "#2563eb"  # Indigo-blue primary accent
accent_glow = "rgba(37, 99, 235, 0.18)"
emerald = "#10b981"
emerald_glow = "rgba(16, 185, 129, 0.15)"

css = f"""
<style>
:root {{
    --bg: {bg};
    --bg-subtle: {bg_subtle};
    --card: {card};
    --card-hover: {card_hover};
    --border: {border};
    --border-subtle: {border_subtle};
    --text: {text};
    --text-muted: {text_muted};
    --text-dim: {text_dim};
    --accent: {accent};
    --accent-glow: {accent_glow};
    --emerald: {emerald};
    --emerald-glow: {emerald_glow};
    --radius: 12px;
}}

/* Hide standard Streamlit header and footer */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

/* Global App Styling */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}
.block-container {{
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1360px !important;
}}

/* Custom Layout Gaps */
[data-testid="stHorizontalBlock"] {{
    gap: 1.25rem !important;
}}

/* Premium Card Styles */
.dashboard-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}

.dashboard-card:hover {{
    border-color: var(--accent);
    box-shadow: 0 0 20px var(--accent-glow);
    transform: translateY(-2px);
}}

/* Brand & Header */
.header-container {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.25rem;
    margin-bottom: 1.75rem;
}}

.brand-title {{
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, var(--accent) 0%, var(--emerald) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.brand-subtitle {{
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
}}

/* Prediction Visualization Card */
.prediction-card {{
    background: linear-gradient(135deg, rgba(37,99,235,0.06) 0%, rgba(16,185,129,0.06) 100%);
    border: 2px solid var(--accent);
    border-radius: var(--radius);
    padding: 2.25rem;
    text-align: center;
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px -10px var(--accent-glow);
    transition: all 0.3s ease;
}}

.prediction-card:hover {{
    box-shadow: 0 12px 35px -8px var(--accent-glow), 0 0 15px rgba(16,185,129,0.1);
    border-color: var(--emerald);
}}

.prediction-title {{
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    font-weight: 700;
}}

.prediction-value {{
    font-size: 3.8rem;
    font-weight: 900;
    color: var(--text);
    margin: 0.5rem 0;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, var(--text) 50%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.prediction-badge {{
    display: inline-block;
    padding: 0.4rem 1.1rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

.badge-elite {{
    color: var(--emerald);
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
}}

.badge-prospect {{
    color: var(--accent);
    background: rgba(37,99,235,0.15);
    border: 1px solid rgba(37,99,235,0.3);
}}

.badge-standard {{
    color: var(--text-muted);
    background: rgba(113,113,122,0.15);
    border: 1px solid rgba(113,113,122,0.3);
}}

/* Plotly Wrapper Cards */
.chart-wrap {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.25rem 0.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    margin-bottom: 1rem;
    transition: border-color 0.2s ease;
}}
.chart-wrap:hover {{
    border-color: var(--border-subtle);
}}
.chart-title {{
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.15rem;
}}
.chart-subtitle {{
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}}

/* Pill-Style Tabs */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.1rem !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--text) !important;
    background: var(--card) !important;
    border-color: var(--border) !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    display: none !important;
}}
[data-baseweb="tab-list"] {{
    gap: 4px !important;
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 4px;
    margin-bottom: 1.5rem !important;
}}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {{
    background-color: var(--bg-subtle) !important;
    border-right: 1px solid var(--border) !important;
}}

/* Custom Chat Bubbles */
.chat-container {{
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    padding: 0.5rem 0;
}}

.chat-bubble {{
    display: flex;
    gap: 1rem;
    padding: 1.25rem;
    border-radius: var(--radius);
    max-width: 88%;
    animation: fadeIn 0.3s ease;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.bubble-user {{
    background-color: var(--bg-subtle);
    border: 1px solid var(--border);
    align-self: flex-end;
}}

.bubble-assistant {{
    background-color: var(--card);
    border: 1px solid var(--border);
    align-self: flex-start;
}}

.chat-avatar {{
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.75rem;
    flex-shrink: 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

.avatar-user {{
    background: var(--accent);
    color: #ffffff;
}}

.avatar-assistant {{
    background: var(--emerald);
    color: #ffffff;
}}

.chat-text {{
    font-size: 0.875rem;
    line-height: 1.6;
    color: var(--text);
}}

.chat-text p:last-child {{
    margin-bottom: 0;
}}

/* Code blocks and tables in Chat */
.chat-text table {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.75rem 0;
    font-size: 0.8rem;
}}
.chat-text th {{
    border-bottom: 2px solid var(--border);
    padding: 0.4rem;
    text-align: left;
    color: var(--text-muted);
}}
.chat-text td {{
    border-bottom: 1px solid var(--border-subtle);
    padding: 0.4rem;
}}

</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ==============================================================================
# 3. MOCK DATA GENERATION
# ==============================================================================
@st.cache_data
def load_or_generate_dataset(n_samples=1000, seed=42):
    """
    Generates a realistic mock FIFA player dataset to train the regression model.
    """
    np.random.seed(seed)
    
    # Generate Age: 17 to 36
    age = np.random.randint(17, 36, n_samples)
    
    # Generate Overall Rating: 55 to 94
    overall = np.random.randint(55, 95, n_samples)
    
    # Generate Potential (must be at least equal to Overall)
    # Higher rating players have smaller room for growth
    max_growth = np.clip(100 - overall, 0, 15)
    growth = np.array([np.random.randint(0, int(g) + 1) if g > 0 else 0 for g in max_growth])
    potential = np.clip(overall + growth, overall, 99)
    
    # Base Valuation logic:
    # 1. Exponential relationship with Overall rating
    base_val = np.exp((overall - 50) / 9.5) * 0.4  # overall 90 = ~26M base, overall 70 = ~3.2M base
    
    # 2. Young prospects (high potential - overall) get a multiplier
    growth_room = potential - overall
    potential_multiplier = 1.0 + (growth_room * 0.08)
    
    # 3. Peak Age curves (peak value around 24-28)
    age_multiplier = np.ones(n_samples)
    for i in range(n_samples):
        a = age[i]
        if a < 24:
            # Young player rising multiplier
            age_multiplier[i] = 0.70 + (a - 17) * 0.04
        elif a > 28:
            # Older player declining multiplier
            age_multiplier[i] = max(0.08, 1.0 - (a - 28) * 0.09)
            
    # Combine factors and add some realistic market variance
    market_noise = np.random.normal(1.0, 0.12, n_samples)
    value_m = base_val * potential_multiplier * age_multiplier * market_noise
    
    # Clamp value between 0.1M (100k) and 180M
    value_m = np.clip(value_m, 0.1, 180.0)
    value_m = np.round(value_m, 1)
    
    df = pd.DataFrame({
        "Age": age,
        "Overall": overall,
        "Potential": potential,
        "Value_EUR_M": value_m
    })
    return df

df_players = load_or_generate_dataset()

# ==============================================================================
# 4. MODEL TRAINING & PIPELINE
# ==============================================================================
@st.cache_resource
def train_valuation_model(df):
    """
    Trains a Random Forest Regressor on the mock dataset and calculates validation metrics.
    """
    X = df[["Age", "Overall", "Potential"]]
    y = df["Value_EUR_M"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate feature importances
    importances = model.feature_importances_
    features = list(X.columns)
    feat_importances = dict(zip(features, importances))
    
    return model, mae, r2, feat_importances

model, model_mae, model_r2, feature_importances = train_valuation_model(df_players)

# ==============================================================================
# 5. SIDEBAR CONTROLS (PLAYER INPUTS)
# ==============================================================================
st.sidebar.markdown(f"""
<div style="text-align: center; padding-bottom: 1.5rem; border-bottom: 1px solid {border}; margin-bottom: 1.5rem;">
    <span style="font-size: 1.35rem; font-weight: 800; color: {text};">Attribute Controls</span>
</div>
""", unsafe_allow_html=True)

# Player sliders
input_overall = st.sidebar.slider(
    "Overall Rating",
    min_value=55,
    max_value=99,
    value=78,
    step=1,
    help="Current overall skill level of the player."
)

# Potential must be at least equal to Overall
input_potential = st.sidebar.slider(
    "Potential Rating",
    min_value=input_overall,
    max_value=99,
    value=max(input_overall, 83),
    step=1,
    help="Maximum potential skill level the player can reach. Must be >= Overall."
)

input_age = st.sidebar.slider(
    "Age",
    min_value=16,
    max_value=40,
    value=22,
    step=1,
    help="Age of the football player. Influences valuation trajectory."
)

# Player Status Card in Sidebar
growth_room = input_potential - input_overall
if input_age <= 21 and input_potential >= 85:
    status_label = "Wonderkid"
    status_color = emerald
elif input_overall >= 86:
    status_label = "World Class"
    status_color = "#f59e0b"  # Amber
elif input_overall >= 80:
    status_label = "Elite Peak"
    status_color = accent
elif input_age >= 32:
    status_label = "Experienced Veteran"
    status_color = "#8b5cf6"  # Purple
else:
    status_label = "First Team Regular"
    status_color = "#71717a"

st.sidebar.markdown(f"""
<div class="dashboard-card" style="margin-top: 1.5rem; text-align: center; border-left: 4px solid {status_color};">
    <div style="font-size: 0.72rem; text-transform: uppercase; color: var(--text-muted); font-weight: 700; letter-spacing: 0.05em;">Player Category</div>
    <div style="font-size: 1.15rem; font-weight: 800; color: {status_color}; margin-top: 0.25rem;">{status_label}</div>
    <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 0.35rem;">Growth room: +{growth_room} rating points</div>
</div>
""", unsafe_allow_html=True)

# API Configuration in Sidebar (for Tab 2 Chat)
st.sidebar.markdown(f"""
<div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid {border};">
    <span style="font-size: 0.9rem; font-weight: 700; color: {text};">AI Chat Configuration</span>
</div>
""", unsafe_allow_html=True)

# Retrieve key from environment or user input
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Enter API Key...",
        help="Input your Google Gemini API key to enable natural language queries on the dataset in the Chat tab."
    )

# Model Performance indicator in Sidebar
st.sidebar.markdown(f"""
<div class="dashboard-card" style="background: transparent; margin-top: 1.5rem; padding: 1rem; border-color: var(--border);">
    <div style="font-size: 0.72rem; text-transform: uppercase; color: var(--text-muted); font-weight: 700;">Model Quality (Random Forest)</div>
    <div style="display: flex; justify-content: space-between; margin-top: 0.4rem;">
        <span style="font-size: 0.75rem; color: var(--text-muted);">R² Score:</span>
        <span style="font-size: 0.75rem; font-weight: 700; color: var(--emerald);">{model_r2:.2f}</span>
    </div>
    <div style="display: flex; justify-content: space-between; margin-top: 0.2rem;">
        <span style="font-size: 0.75rem; color: var(--text-muted);">MAE:</span>
        <span style="font-size: 0.75rem; font-weight: 700; color: var(--text);">€{model_mae:.2f}M</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 6. HEADER LAYOUT
# ==============================================================================
head_left, head_right = st.columns([9, 1.5])
with head_left:
    st.markdown("""
    <div class="header-container">
        <div>
            <span class="brand-title">⚽ FIFA Player Valuation Predictor</span>
            <div class="brand-subtitle">An end-to-end Machine Learning dashboard & AI data analytics interface</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_btn_label = "☀️ Light" if IS_DARK else "🌙 Dark"
    st.button(
        theme_btn_label, 
        on_click=toggle_theme, 
        use_container_width=True,
        help="Switch between Dark and Light color aesthetics."
    )

# Establish two tabs
tab_pred, tab_chat = st.tabs(["📊 Valuation Predictor", "💬 AI Dataset Chat"])

# ==============================================================================
# 7. TAB 1: VALUATION PREDICTOR
# ==============================================================================
with tab_pred:
    # Perform prediction
    pred_input = pd.DataFrame({
        "Age": [input_age],
        "Overall": [input_overall],
        "Potential": [input_potential]
    })
    
    predicted_val = model.predict(pred_input)[0]
    # Format valuation nicely (e.g. €78.4M)
    formatted_val = f"€{predicted_val:.1f}M"
    
    # Prediction result card
    badge_class = "badge-standard"
    if predicted_val >= 40.0:
        badge_class = "badge-elite"
        badge_text = "Elite Tier Valuation"
    elif predicted_val >= 10.0:
        badge_class = "badge-prospect"
        badge_text = "High-Value Asset"
    else:
        badge_class = "badge-standard"
        badge_text = "Standard Market Asset"

    st.markdown(f"""
    <div class="prediction-card">
        <div class="prediction-title">Estimated Transfer Value</div>
        <div class="prediction-value">{formatted_val}</div>
        <div class="prediction-badge {badge_class}">{badge_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization Section
    col_scatter, col_bar = st.columns([6, 4])
    
    with col_scatter:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Market Valuation Distribution</div>
            <div class="chart-subtitle">Overall Rating vs. Valuation with current selection highlighted</div>
        """, unsafe_allow_html=True)
        
        # Plotly scatter plot
        fig_scatter = px.scatter(
            df_players,
            x="Overall",
            y="Value_EUR_M",
            color="Age",
            color_continuous_scale="Viridis",
            labels={"Overall": "Overall Rating", "Value_EUR_M": "Valuation (€M)", "Age": "Age"},
            hover_data=["Potential"],
            opacity=0.6
        )
        
        # Add a glowing gold star representing the predicted player
        fig_scatter.add_trace(
            go.Scatter(
                x=[input_overall],
                y=[predicted_val],
                mode="markers",
                marker=dict(
                    color="#f59e0b",
                    size=16,
                    line=dict(color="#ffffff", width=2),
                    symbol="star",
                ),
                name="Target Player",
                hoverinfo="text",
                hovertext=f"Selected Player<br>Age: {input_age}<br>Rating: {input_overall}<br>Potential: {input_potential}<br>Valuation: {formatted_val}"
            )
        )
        
        # Apply unified design system plot settings
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=10),
            margin=dict(l=0, r=0, t=15, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            coloraxis_colorbar=dict(
                title=dict(font=dict(size=10)),
                thickness=12,
                len=0.8,
                yanchor="middle",
                y=0.5
            ),
            xaxis=dict(
                gridcolor="rgba(0,0,0,0.05)" if not IS_DARK else "rgba(255,255,255,0.05)",
                zerolinecolor="rgba(0,0,0,0.05)" if not IS_DARK else "rgba(255,255,255,0.05)",
                tickfont=dict(size=10, color="#71717a"),
            ),
            yaxis=dict(
                gridcolor="rgba(0,0,0,0.05)" if not IS_DARK else "rgba(255,255,255,0.05)",
                zerolinecolor="rgba(0,0,0,0.05)" if not IS_DARK else "rgba(255,255,255,0.05)",
                tickfont=dict(size=10, color="#71717a"),
            ),
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_bar:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Attribute Comparison</div>
            <div class="chart-subtitle">Comparing player parameters vs. dataset average & maximum</div>
        """, unsafe_allow_html=True)
        
        # Calculate dataset statistics
        avg_age = df_players["Age"].mean()
        avg_overall = df_players["Overall"].mean()
        avg_potential = df_players["Potential"].mean()
        
        max_age = df_players["Age"].max()
        max_overall = df_players["Overall"].max()
        max_potential = df_players["Potential"].max()
        
        categories = ["Age", "Overall Rating", "Potential Rating"]
        
        fig_bar = go.Figure()
        # Selected Player
        fig_bar.add_trace(go.Bar(
            name="Selected Player",
            x=categories,
            y=[input_age, input_overall, input_potential],
            marker_color=accent,
            width=0.25
        ))
        # Dataset Average
        fig_bar.add_trace(go.Bar(
            name="Dataset Avg",
            x=categories,
            y=[avg_age, avg_overall, avg_potential],
            marker_color="#71717a" if IS_DARK else "#a1a1aa",
            width=0.25
        ))
        # Dataset Max
        fig_bar.add_trace(go.Bar(
            name="Dataset Max",
            x=categories,
            y=[max_age, max_overall, max_potential],
            marker_color=emerald,
            width=0.25
        ))
        
        fig_bar.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=10),
            margin=dict(l=0, r=0, t=15, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(
                gridcolor="rgba(0,0,0,0.05)" if not IS_DARK else "rgba(255,255,255,0.05)",
                zerolinecolor="rgba(0,0,0,0.05)" if not IS_DARK else "rgba(255,255,255,0.05)",
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                gridcolor="rgba(0,0,0,0.05)" if not IS_DARK else "rgba(255,255,255,0.05)",
                zerolinecolor="rgba(0,0,0,0.05)" if not IS_DARK else "rgba(255,255,255,0.05)",
                tickfont=dict(size=10),
            ),
        )
        
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Feature Importance Row
    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-title">Machine Learning Feature Significance</div>
        <div class="chart-subtitle">Contribution of each player feature to the valuation model</div>
    """, unsafe_allow_html=True)
    
    importance_df = pd.DataFrame({
        "Feature": list(feature_importances.keys()),
        "Importance": list(feature_importances.values())
    }).sort_values(by="Importance", ascending=True)
    
    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        labels={"Importance": "Relative Feature Weight", "Feature": "Player Attribute"},
        color="Importance",
        color_continuous_scale=[[0, "#3b82f6"], [1, "#10b981"]]
    )
    
    fig_importance.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=10),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.05)" if not IS_DARK else "rgba(255,255,255,0.05)",
            zerolinecolor="rgba(0,0,0,0.05)" if not IS_DARK else "rgba(255,255,255,0.05)",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.05)" if not IS_DARK else "rgba(255,255,255,0.05)",
            zerolinecolor="rgba(0,0,0,0.05)" if not IS_DARK else "rgba(255,255,255,0.05)",
            tickfont=dict(size=10),
        ),
    )
    
    st.plotly_chart(fig_importance, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 8. TAB 2: AI DATASET CHAT
# ==============================================================================
with tab_chat:
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <span style="font-size: 1.15rem; font-weight: 700; color: {text};">💬 Ask Gemini about the Dataset</span>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem;">
            Interact with the 1,000 player dataset in plain English. The AI knows the statistics, correlations, and individual records of all generated players.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not api_key:
        st.warning("⚠️ Please enter a Google Gemini API Key in the sidebar configuration to activate the AI Dataset Chat.")
    else:
        # Initialize Google Generative AI
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Quick suggestion buttons
            st.markdown(f"""
            <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase;">Suggested Queries</div>
            """, unsafe_allow_html=True)
            
            sq_col1, sq_col2, sq_col3 = st.columns(3)
            with sq_col1:
                if st.button("Who are the top 3 most valuable players?", use_container_width=True):
                    user_input_preset = "Who are the top 3 most valuable players? List their age, overall, potential and value."
                    st.session_state.chat_history.append({"role": "user", "content": user_input_preset})
                    st.rerun()
            with sq_col2:
                if st.button("What is the average age of players with overall >= 85?", use_container_width=True):
                    user_input_preset = "What is the average age of players with an overall rating of 85 or higher?"
                    st.session_state.chat_history.append({"role": "user", "content": user_input_preset})
                    st.rerun()
            with sq_col3:
                if st.button("Explain the correlation between Age and Value.", use_container_width=True):
                    user_input_preset = "Based on this dataset, explain the relationship and correlation between Age and market Value."
                    st.session_state.chat_history.append({"role": "user", "content": user_input_preset})
                    st.rerun()
            
            st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
            
            # Construct dataset CSV summary instruction for Gemini context
            @st.cache_data
            def get_dataset_context_prompt(df):
                csv_data = df.to_csv(index=False)
                return f"""You are an expert FIFA Player Valuation Data Analyst AI assistant.
You have full access to a dataset of 1,000 mock FIFA players with columns:
- Age: integer (17 to 35)
- Overall: integer (55 to 94)
- Potential: integer (55 to 99)
- Value_EUR_M: float in millions of Euros (0.1M to 180.0M)

Here is the complete dataset in CSV format:
```csv
{csv_data}
```

Your guidelines:
1. Provide accurate answers based solely on calculations on this dataset.
2. Formulate counts, averages, correlations, or lists directly using the provided CSV.
3. Be professional, clear, and structure tables using markdown where appropriate.
4. If a query is unrelated to the dataset or football player stats, politely steer the user back to the player data.
"""
            
            system_prompt = get_dataset_context_prompt(df_players)
            
            # Display chat history
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            for message in st.session_state.chat_history:
                role = message["role"]
                content = message["content"]
                
                avatar_class = "avatar-user" if role == "user" else "avatar-assistant"
                bubble_class = "bubble-user" if role == "user" else "bubble-assistant"
                avatar_char = "U" if role == "user" else "AI"
                
                st.markdown(f"""
                <div class="chat-bubble {bubble_class}">
                    <div class="chat-avatar {avatar_class}">{avatar_char}</div>
                    <div class="chat-text">{content}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Chat input text box
            user_msg = st.chat_input("Ask a question about the FIFA player dataset...")
            
            if user_msg:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_msg})
                st.rerun()
                
            # If the last message is from the user, generate the response
            if len(st.session_state.chat_history) > 0 and st.session_state.chat_history[-1]["role"] == "user":
                # Display current typing assistant bubble
                assistant_bubble_placeholder = st.empty()
                
                # Fetch response from Gemini
                with assistant_bubble_placeholder.container():
                    st.markdown(f"""
                    <div class="chat-bubble bubble-assistant">
                        <div class="chat-avatar avatar-assistant">AI</div>
                        <div class="chat-text"><i>Thinking...</i></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Setup Gemini model
                model_gemini = genai.GenerativeModel('gemini-2.0-flash', system_instruction=system_prompt)
                # Prepare history format for API call
                history_input = []
                for msg in st.session_state.chat_history[:-1]:
                    api_role = "user" if msg["role"] == "user" else "model"
                    history_input.append({"role": api_role, "parts": [msg["content"]]})
                
                chat_session = model_gemini.start_chat(history=history_input)
                
                try:
                    response_stream = chat_session.send_message(st.session_state.chat_history[-1]["content"], stream=True)
                    
                    full_response = ""
                    for chunk in response_stream:
                        if chunk.text:
                            full_response += chunk.text
                            # Update placeholder with streamed response
                            assistant_bubble_placeholder.markdown(f"""
                            <div class="chat-bubble bubble-assistant">
                                <div class="chat-avatar avatar-assistant">AI</div>
                                <div class="chat-text">{full_response}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    # Save response to history
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                    st.rerun()
                except Exception as ex:
                    err_msg = f"Error generating response: {str(ex)}. Please verify your API Key and internet connection."
                    assistant_bubble_placeholder.markdown(f"""
                    <div class="chat-bubble bubble-assistant">
                        <div class="chat-avatar avatar-assistant">AI</div>
                        <div class="chat-text" style="color: #ef4444;">{err_msg}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
            
            # Reset button
            if len(st.session_state.chat_history) > 0:
                if st.button("🧹 Clear Chat History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
                    
        except ImportError:
            st.error("Missing dependency: google-generativeai. Please verify requirements are fully installed.")
        except Exception as e:
            st.error(f"Failed to load AI Chat system: {str(e)}")
