import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import os

# Force Streamlit to not cache across runs
st.set_page_config(page_title="Mental Health Risk Dashboard", layout="wide")
if 'page_load' not in st.session_state:
    st.cache_data.clear()
    st.session_state.page_load = True
st.title("🧠 Mental Health Risk Prediction & Recommendations")
st.markdown("Welcome to the Group 18 Dashboard. Explore global trends or get a personalized risk assessment.")

# --- Caching Data and Models ---
@st.cache_data
def load_data():
    """Fetches the global dashboard-ready dataset from Postgres."""
    query = "SELECT * FROM mental_health_predictions"
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="airflow",
            user="airflow",
            password="airflow",
            port=5432
        )
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.warning(f"Could not connect to database. Please ensure Airflow has run and Postgres is up. Error: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model_and_explainer():
    """Loads the champion model and initializes the SHAP explainer."""
    model_path = '../models/champion_model.pkl' 
    try:
        with open(model_path, 'rb') as f:
            champion_model = pickle.load(f)
        
        model_to_explain = champion_model.named_steps['classifier'] if hasattr(champion_model, 'named_steps') else champion_model
        explainer = shap.Explainer(model_to_explain)
        return champion_model, explainer
    except FileNotFoundError:
        st.warning("Champion model not found. Please ensure the Airflow DAG has completed Phase 3.")
        return None, None

df = load_data()
champion_model, explainer = load_model_and_explainer()

# Define the expected ML columns
CATEGORICAL_COLS = ['gender', 'marital_status', 'education_level', 'employment_status']

# --- Sidebar: Interactive User Input ---
st.sidebar.header("User Data Input")
st.sidebar.markdown("Enter your details and click 'Run Prediction'.")

with st.sidebar.form("prediction_form"):
    user_data = {
        'age': st.slider("Age", 18, 60, 25),
        'gender': st.selectbox("Gender", ["Male", "Female", "Other"]),
        'marital_status': st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
        'education_level': st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"]),
        'employment_status': st.selectbox("Employment Status", ["Student", "Employed", "Self-Employed", "Unemployed"]),
        'sleep_hours': st.slider("Sleep Hours", 3.0, 10.0, 7.0, 0.5),
        'physical_activity_hours_per_week': st.slider("Physical Activity (Hrs/Wk)", 0.0, 15.0, 3.0, 0.5),
        'screen_time_hours_per_day': st.slider("Screen Time (Hrs/Day)", 1.0, 12.0, 6.0, 0.5),
        'social_support_score': st.slider("Social Support Score", 1, 10, 5),
        'work_stress_level': st.slider("Work/Study Stress Level", 1, 10, 5),
        'academic_pressure_level': st.slider("Academic Pressure", 1, 10, 5),
        'job_satisfaction_score': st.slider("Job/Study Satisfaction", 1, 10, 5),
        'financial_stress_level': st.slider("Financial Stress Level", 1, 10, 5),
        'working_hours_per_week': st.slider("Working Hours / Week", 20, 70, 40),
        'anxiety_score': st.slider("Anxiety Score", 1, 10, 5),
        'depression_score': st.slider("Depression Score", 1, 10, 5),
        'stress_level': st.slider("Overall Stress Level", 1, 10, 5),
        'mood_swings_frequency': st.slider("Mood Swings Frequency", 1, 10, 5),
        'concentration_difficulty_level': st.slider("Concentration Difficulty", 1, 10, 5),
        'panic_attack_history': st.selectbox("History of Panic Attacks", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"),
        'family_history_mental_illness': st.selectbox("Family History of Mental Illness", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"),
        'previous_mental_health_diagnosis': st.selectbox("Previous Diagnosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"),
        'therapy_history': st.selectbox("Therapy History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"),
        'substance_use': st.selectbox("Substance Use", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    }
    
    submit_button = st.form_submit_button(label="Run Prediction 🚀")

user_df = pd.DataFrame([user_data])

# --- Preprocessing User Input ---
user_df['composite_lifestyle_score'] = (user_df['sleep_hours'] / 10) + (user_df['physical_activity_hours_per_week'] / 14)

if champion_model is not None:
    if hasattr(champion_model, 'feature_names_in_'):
        expected_cols = champion_model.feature_names_in_
    elif hasattr(champion_model, 'named_steps') and hasattr(champion_model.named_steps['classifier'], 'feature_names_in_'):
        expected_cols = champion_model.named_steps['classifier'].feature_names_in_
    else:
        dummy_df = pd.get_dummies(df.drop(['predicted_risk', 'actionable_recommendation', 'mental_health_risk'], axis=1, errors='ignore'), columns=CATEGORICAL_COLS, drop_first=True)
        expected_cols = dummy_df.columns

    user_encoded = pd.get_dummies(user_df, columns=CATEGORICAL_COLS, drop_first=True)
    user_encoded = user_encoded.reindex(columns=expected_cols, fill_value=0)

# --- App Layout using Tabs ---
tab1, tab2 = st.tabs(["🌍 Part 1: Global Dashboards (Macro)", "🎯 Part 2: Interactive Prediction (Micro)"])

# Define preferred category order globally
risk_order = ["Low", "Moderate", "High"]

# ==========================================
# TAB 1: MACRO VIEW (GLOBAL DASHBOARDS)
# ==========================================
with tab1:
    st.header("Global Population Trends")
    if df.empty:
        st.error("No data available to display macro trends.")
    else:
        target_col = 'predicted_risk' if 'predicted_risk' in df.columns else 'mental_health_risk'
        risk_map = {0: 'Low', 1: 'Moderate', 2: 'High'}
        df['Risk Label'] = df[target_col].map(risk_map)
        color_map = {'Low': '#2ca02c', 'Moderate': '#ff7f0e', 'High': '#d62728'} 

        # --- 1. Demographics ---
        st.subheader("1. Demographic Risk Breakdown")
        st.markdown("Proportion of Low, Moderate, and High risk profiles across different demographics.")
        
        col1, col2, col3 = st.columns(3)
        demographics = ['employment_status', 'gender', 'education_level']
        columns = [col1, col2, col3]
        
        for ui_col, feature in zip(columns, demographics):
            demo_risk = df.groupby([feature, 'Risk Label']).size().reset_index(name='Count')
            demo_risk['Percentage'] = demo_risk.groupby(feature)['Count'].transform(lambda x: x / x.sum() * 100)
            
            # Create stacked bars using go.Bar for explicit control
            fig_bar = go.Figure()
            
            for risk_label in risk_order:
                data_subset = demo_risk[demo_risk['Risk Label'] == risk_label]
                fig_bar.add_trace(go.Bar(
                    x=data_subset[feature],
                    y=data_subset['Percentage'],
                    name=risk_label,
                    marker=dict(color=color_map[risk_label])
                ))
            
            fig_bar.update_layout(
                barmode='stack',
                title=dict(text=f"Risk by {feature.replace('_', ' ').title()}", x=0.5, xanchor='center', font=dict(size=15)),
                xaxis_title="", 
                yaxis_title="Percentage (%)",
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title_text=""),
                margin=dict(l=20, r=20, t=50, b=20),
                hovermode='x unified'
            )
            
            ui_col.plotly_chart(fig_bar, width='stretch')

        st.markdown("---")

        # --- 2. Lifestyle & Habits ---
        st.subheader("2. Lifestyle Habits vs. Mental Health Risk")
        col4, col5, col6 = st.columns(3)
        lifestyle_features = ['sleep_hours', 'screen_time_hours_per_day', 'physical_activity_hours_per_week']
        lifestyle_columns = [col4, col5, col6]
        
        for ui_col, feature in zip(lifestyle_columns, lifestyle_features):
            fig_box = px.box(df, x="Risk Label", y=feature, color="Risk Label",
                             color_discrete_map=color_map,
                             category_orders={"Risk Label": risk_order})
            
            fig_box.update_layout(
                title=dict(text=f"{feature.replace('_', ' ').title()}", x=0.5, xanchor='center', font=dict(size=15)),
                xaxis_title="", 
                showlegend=False,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            ui_col.plotly_chart(fig_box, width='stretch')

        st.markdown("---")

        # --- 3. Psychological Factors (Heatmap) ---
        st.subheader("3. Stress & Psychological Correlation")
        st.markdown("Identifies relationships between numerical stress and lifestyle factors.")
        
        heat_cols = ['work_stress_level', 'academic_pressure_level', 'financial_stress_level', 
                     'anxiety_score', 'depression_score', 'social_support_score']
        
        fig_heat, ax_heat = plt.subplots(figsize=(10, 5)) 
        sns.heatmap(df[heat_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_heat)
        st.pyplot(fig_heat)

        st.markdown("---")

        # --- 4. Average Scores by Risk Cluster (Radar Charts) ---
        st.subheader("4. Average Scores by Risk Cluster")
        st.markdown("Comparing the typical psychological profile of Low, Moderate, and High risk groups.")
        
        radar_vars = ['work_stress_level', 'financial_stress_level', 'anxiety_score', 'depression_score', 'social_support_score']
        radar_df = df.groupby('Risk Label')[radar_vars].mean().reset_index()
        radar_melted = radar_df.melt(id_vars=['Risk Label'], var_name='Metric', value_name='Average Score')
        
        radar_melted['Metric'] = radar_melted['Metric'].str.replace('_level', '').str.replace('_score', '').str.replace('_', '<br>').str.title()
        
        col_radar1, col_radar2, col_radar3 = st.columns(3)
        risk_levels = ['Low', 'Moderate', 'High']
        radar_columns = [col_radar1, col_radar2, col_radar3]
        
        for ui_col, risk_level in zip(radar_columns, risk_levels):
            subset = radar_melted[radar_melted['Risk Label'] == risk_level]
            
            fig_radar_macro = px.line_polar(subset, r='Average Score', theta='Metric', 
                                            line_close=True, color_discrete_sequence=[color_map[risk_level]])
            fig_radar_macro.update_traces(fill='toself')
            
            fig_radar_macro.update_layout(
                title=dict(text=f"<b>{risk_level} Risk Profile</b>", x=0.5, xanchor='center', y=1.0, font=dict(size=14)), 
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(color="black", size=9)),
                    angularaxis=dict(tickfont=dict(color="white", size=10)) 
                ),
                margin=dict(l=50, r=50, t=80, b=50), 
                height=300, 
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            ui_col.plotly_chart(fig_radar_macro, width='stretch')

        st.markdown("---")

        # --- 5. Medical & Clinical History ---
        st.subheader("5. Clinical History Indicators")
        st.markdown("Impact of baseline clinical risks (1 in X people with these traits fall into higher risk categories).")
        col9, col10, col11 = st.columns(3)
        clinical_features = ['substance_use', 'panic_attack_history', 'therapy_history']
        clinical_columns = [col9, col10, col11]

        for ui_col, feature in zip(clinical_columns, clinical_features):
            med_risk = df.groupby([feature, 'Risk Label']).size().reset_index(name='Count')
            med_risk['Percentage'] = med_risk.groupby(feature)['Count'].transform(lambda x: x / x.sum() * 100)
            med_risk[feature] = med_risk[feature].map({0: 'No', 1: 'Yes'}) 
            
            # Create stacked bars using go.Bar for explicit control
            fig_med = go.Figure()
            
            for risk_label in risk_order:
                data_subset = med_risk[med_risk['Risk Label'] == risk_label]
                fig_med.add_trace(go.Bar(
                    x=data_subset[feature],
                    y=data_subset['Percentage'],
                    name=risk_label,
                    marker=dict(color=color_map[risk_label])
                ))
            
            fig_med.update_layout(
                barmode='stack',
                title=dict(text=f"{feature.replace('_', ' ').title()}", x=0.5, xanchor='center', font=dict(size=15)),
                xaxis_title="History", 
                yaxis_title="Percentage (%)",
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title_text=""),
                margin=dict(l=20, r=20, t=50, b=20),
                hovermode='x unified'
            )
            
            ui_col.plotly_chart(fig_med, width='stretch')


# ==========================================
# TAB 2: MICRO VIEW (INTERACTIVE USER INPUT)
# ==========================================
with tab2:
    st.header("Personalized Risk Assessment")
    
    if champion_model is None:
        st.error("Model unavailable. Cannot generate predictions.")
    else:
        prediction = champion_model.predict(user_encoded)[0]
        
        if hasattr(champion_model, "predict_proba"):
            probs = champion_model.predict_proba(user_encoded)[0]
            risk_score = (probs[2] * 100) + (probs[1] * 50)
        else:
            risk_score = prediction * 50 

        risk_labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
        colors = {0: "green", 1: "orange", 2: "red"}

        # --- 4. Prediction Risk Meter ---
        st.subheader("4. Your Predicted Risk Level")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            title = {'text': f"<b>{risk_labels[prediction]}</b>"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "black", 'thickness': 0.2},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "navajowhite"},
                    {'range': [66, 100], 'color': "lightcoral"}],
            }
        ))
        
        
        st.plotly_chart(fig_gauge, width='stretch')

        st.info(f"**Actionable Recommendation:** \n" + 
                ("Maintain current healthy lifestyle habits." if prediction == 0 else 
                 "Consider discussing workload with HR or taking short breaks." if prediction == 1 and user_data['work_stress_level'] > 7 else
                 "Prioritize getting 7-8 hours of sleep." if prediction == 1 and user_data['sleep_hours'] < 6 else
                 "Monitor overall stress levels and practice daily self-care." if prediction == 1 else
                 "Significant risk indicators detected. Please consult a mental health professional."))

        st.markdown("---")

        # --- 5. Personalized Insights ---
        st.subheader("5. Personalized Insights")
        st.markdown("Based on your specific profile, here are the main factors driving your mental health risk score:")
        
        shap_values_user = explainer(user_encoded)
        
        if len(shap_values_user.shape) == 3:
            class_idx = prediction if prediction != 0 else 2 
            shap_vals = shap_values_user[0, :, class_idx].values
        else:
            shap_vals = shap_values_user[0].values
            
        importance_df = pd.DataFrame({
            'Feature': user_encoded.columns,
            'Impact': shap_vals,
            'User_Value': user_encoded.iloc[0].values
        })
        
        importance_df['Feature'] = importance_df['Feature'].str.replace('_', ' ').str.title()
        
        factors_increasing = importance_df[importance_df['Impact'] > 0].sort_values(by='Impact', ascending=False).head(3)
        factors_lowering = importance_df[importance_df['Impact'] < 0].sort_values(by='Impact', ascending=True).head(3)

        col_risk, col_safe = st.columns(2)
        
        with col_risk:
            st.error("📈 **Top 3 Factors Increasing Your Risk**")
            if factors_increasing.empty:
                st.write("*No significant risk factors detected.*")
            else:
                for _, row in factors_increasing.iterrows():
                    val = row['User_Value']
                    formatted_val = f"{val:.1f}" if isinstance(val, float) else val
                    st.write(f"• **{row['Feature']}** (Your Input: {formatted_val})")
                    
        with col_safe:
            st.success("📉 **Top 3 Factors Lowering Your Risk**")
            if factors_lowering.empty:
                st.write("*No significant protective factors detected.*")
            else:
                for _, row in factors_lowering.iterrows():
                    val = row['User_Value']
                    formatted_val = f"{val:.1f}" if isinstance(val, float) else val
                    st.write(f"• **{row['Feature']}** (Your Input: {formatted_val})")

        st.markdown("---")

        # --- 6. Personal vs. Demographic Baseline ---
        st.subheader("6. How Do You Compare? (Radar Chart)")
        st.markdown(f"Comparing your metrics against the average for **{user_data['employment_status']}s**.")
        
        if not df.empty:
            demo_baseline = df[df['employment_status'] == user_data['employment_status']]
            
            radar_cols = ['work_stress_level', 'anxiety_score', 'depression_score', 'social_support_score', 'job_satisfaction_score']
            
            if not demo_baseline.empty:
                avg_scores = demo_baseline[radar_cols].mean().to_dict()
                
                formatted_theta = [c.replace('_level', '').replace('_score', '').replace('_', '<br>').title() for c in radar_cols]
                
                radar_data = pd.DataFrame(dict(
                    r=[user_data[c] for c in radar_cols] + [avg_scores[c] for c in radar_cols],
                    theta=formatted_theta * 2,
                    Group=['You'] * len(radar_cols) + ['Demographic Average'] * len(radar_cols)
                ))

                fig_radar = px.line_polar(radar_data, r='r', theta='theta', color='Group', line_close=True)
                
                for trace in fig_radar.data:
                    trace.fill = 'toself'
                    if trace.name == 'You':
                        trace.fillcolor = 'rgba(30, 60, 150, 0.8)'
                        trace.line.color = 'rgba(30, 60, 150, 1.0)'
                    else:
                        trace.fillcolor = 'rgba(100, 200, 255, 0.15)'
                        trace.line.color = 'rgba(100, 200, 255, 0.8)'
                
                fig_radar.update_layout(
                    title=dict(x=0.5, xanchor='center'),
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(color="black")), 
                        angularaxis=dict(tickfont=dict(color="white"))
                    ),
                    margin=dict(l=60, r=60, t=50, b=50),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                st.plotly_chart(fig_radar, width='stretch')
            else:
                st.write("Not enough demographic data to form a baseline.")