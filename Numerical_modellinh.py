import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="SIR Epidemic Simulator",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for Styling
# -----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 0.5rem 0;
    }
    .intervention-card {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #BFDBFE;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #60A5FA;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #10B981;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# App Header
# -----------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🦠 SIR Epidemic Simulation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Explore disease dynamics, compare numerical methods, and analyze intervention strategies")

# -----------------------------
# Sidebar - Model Parameters
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    
    # Create tabs in sidebar for better organization
    tab_params, tab_method, tab_interv = st.tabs(["Parameters", "Methods", "Interventions"])
    
    with tab_params:
        st.markdown("#### Population Settings")
        N = st.number_input(
            "Total Population (N)", 
            min_value=100, 
            max_value=10000, 
            value=1000, 
            step=100,
            help="Total number of individuals in the population"
        )
        
        I0 = st.number_input(
            "Initial Infected (I₀)", 
            min_value=1, 
            max_value=min(100, N), 
            value=5,
            help="Number of initially infected individuals"
        )
        
        # Beta and Gamma with more intuitive explanations
        col_beta, col_gamma = st.columns(2)
        with col_beta:
            beta = st.slider(
                "β (Transmission Rate)", 
                0.05, 2.0, 0.3, 0.01,
                help="Average number of contacts per person per time unit"
            )
        with col_gamma:
            gamma = st.slider(
                "γ (Recovery Rate)", 
                0.05, 1.0, 0.1, 0.01,
                help="Recovery rate = 1/average infectious period"
            )
        
        # Calculate R0
        R0_basic = beta / gamma
        st.markdown(f"""
        <div class="metric-card">
            <b>Basic R₀:</b> {R0_basic:.2f}<br>
            <small>R₀ > 1: Epidemic spreads<br>
            R₀ < 1: Disease dies out</small>
        </div>
        """, unsafe_allow_html=True)
    
    with tab_method:
        st.markdown("#### Simulation Settings")
        
        # Time controls
        t_max = st.slider(
            "Simulation Time (days)", 
            30, 365, 160, 10,
            help="Total duration of the simulation"
        )
        
        # Time step selection with visual indicator
        dt_option = st.radio(
            "Time Step Method",
            ["Automatic (stable)", "Manual control"]
        )
        
        if dt_option == "Automatic (stable)":
            dt = 0.1  # Stable default
            st.info(f"Using stable dt = {dt}")
        else:
            dt = st.number_input(
                "Time Step (dt)", 
                min_value=0.01, 
                max_value=1.0, 
                value=0.1, 
                step=0.01,
                help="Smaller dt = more accurate but slower"
            )
            if dt > 0.5:
                st.warning("Large time step may cause instability")
        
        method = st.selectbox(
            "Numerical Method",
            ["Euler Method", "Runge-Kutta 4", "Compare Both"],
            help="Euler: Fast but less accurate. RK4: More accurate but slower"
        )
    
    with tab_interv:
        st.markdown("#### Intervention Strategies")
        
        # Vaccination
        st.markdown("##### 💉 Vaccination")
        V = st.slider(
            "Vaccinated Individuals", 
            0, N-I0, 0, 10,
            help="Number of individuals vaccinated before epidemic starts"
        )
        
        # Social distancing
        st.markdown("##### 🚶 Social Distancing")
        sd_strength = st.slider(
            "Effectiveness (%)", 
            0, 100, 0, 5,
            help="Reduction in transmission due to social distancing"
        )
        
        # Quarantine
        st.markdown("##### 🏠 Quarantine")
        quarantine_strength = st.slider(
            "Effectiveness (%)", 
            0, 100, 0, 5,
            help="Reduction in transmission due to quarantine/isolation"
        )
        
        # Lockdown option
        st.markdown("##### 🔒 Lockdown")
        lockdown = st.checkbox("Implement lockdown")
        if lockdown:
            lockdown_start = st.slider("Start Day", 0, t_max, 30)
            lockdown_duration = st.slider("Duration (days)", 0, t_max-lockdown_start, 30)
            lockdown_strength = st.slider("Transmission Reduction (%)", 0, 100, 70)

# -----------------------------
# Initial Calculations
# -----------------------------
S0 = N - I0 - V
I0_initial = I0
R0_initial = V

# Calculate effective beta with interventions
beta_eff = beta * (1 - sd_strength/100) * (1 - quarantine_strength/100)
R0_effective = beta_eff / gamma

# -----------------------------
# SIR Model Functions
# -----------------------------
def sir_rhs(S, I, R, beta_use, gamma_use, N):
    dS = -beta_use * S * I / N
    dI = beta_use * S * I / N - gamma_use * I
    dR = gamma_use * I
    return dS, dI, dR

def euler_method(S0, I0, R0, beta_use, gamma_use, N, dt, t_max, lockdown_params=None):
    t = np.arange(0, t_max, dt)
    S, I, R = [S0], [I0], [R0]
    
    for i, time in enumerate(t[1:], 1):
        current_S, current_I, current_R = S[-1], I[-1], R[-1]
        
        # Apply lockdown if active
        current_beta = beta_use
        if lockdown_params and lockdown_params['active']:
            lockdown_start = lockdown_params['start']
            lockdown_end = lockdown_start + lockdown_params['duration']
            if lockdown_start <= time <= lockdown_end:
                current_beta = beta_use * (1 - lockdown_params['strength']/100)
        
        dS, dI, dR = sir_rhs(current_S, current_I, current_R, current_beta, gamma_use, N)
        
        S.append(current_S + dt * dS)
        I.append(current_I + dt * dI)
        R.append(current_R + dt * dR)
    
    return np.array(S), np.array(I), np.array(R), t

def rk4_method(S0, I0, R0, beta_use, gamma_use, N, dt, t_max, lockdown_params=None):
    t = np.arange(0, t_max, dt)
    S, I, R = [S0], [I0], [R0]
    
    def sir_step(S, I, R, beta_local):
        return sir_rhs(S, I, R, beta_local, gamma_use, N)
    
    for i, time in enumerate(t[1:], 1):
        current_S, current_I, current_R = S[-1], I[-1], R[-1]
        
        # Apply lockdown if active
        current_beta = beta_use
        if lockdown_params and lockdown_params['active']:
            lockdown_start = lockdown_params['start']
            lockdown_end = lockdown_start + lockdown_params['duration']
            if lockdown_start <= time <= lockdown_end:
                current_beta = beta_use * (1 - lockdown_params['strength']/100)
        
        k1 = sir_step(current_S, current_I, current_R, current_beta)
        k2 = sir_step(current_S + dt*k1[0]/2, current_I + dt*k1[1]/2, 
                     current_R + dt*k1[2]/2, current_beta)
        k3 = sir_step(current_S + dt*k2[0]/2, current_I + dt*k2[1]/2, 
                     current_R + dt*k2[2]/2, current_beta)
        k4 = sir_step(current_S + dt*k3[0], current_I + dt*k3[1], 
                     current_R + dt*k3[2], current_beta)
        
        S.append(current_S + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6)
        I.append(current_I + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6)
        R.append(current_R + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6)
    
    return np.array(S), np.array(I), np.array(R), t

# -----------------------------
# Run Simulations
# -----------------------------
lockdown_params = None
if 'lockdown' in locals() and lockdown:
    lockdown_params = {
        'active': True,
        'start': lockdown_start,
        'duration': lockdown_duration,
        'strength': lockdown_strength
    }

if method == "Euler Method":
    S, I, R, t = euler_method(S0, I0_initial, R0_initial, beta_eff, gamma, N, dt, t_max, lockdown_params)
    method_name = "Euler"
elif method == "Runge-Kutta 4":
    S, I, R, t = rk4_method(S0, I0_initial, R0_initial, beta_eff, gamma, N, dt, t_max, lockdown_params)
    method_name = "RK4"
else:
    S_e, I_e, R_e, t = euler_method(S0, I0_initial, R0_initial, beta_eff, gamma, N, dt, t_max, lockdown_params)
    S_r, I_r, R_r, _ = rk4_method(S0, I0_initial, R0_initial, beta_eff, gamma, N, dt, t_max, lockdown_params)

# Baseline (no intervention) simulation
S_base, I_base, R_base, _ = euler_method(N-I0, I0, 0, beta, gamma, N, dt, t_max)

# -----------------------------
# Key Metrics Calculation
# -----------------------------
if method == "Compare Both":
    peak_I_e = np.max(I_e)
    peak_I_r = np.max(I_r)
    peak_day_e = t[np.argmax(I_e)]
    peak_day_r = t[np.argmax(I_r)]
    total_cases_e = R_e[-1]
    total_cases_r = R_r[-1]
else:
    peak_I = np.max(I)
    peak_day = t[np.argmax(I)]
    total_cases = R[-1]

# -----------------------------
# Main Dashboard Layout
# -----------------------------

# Row 1: Key Metrics
st.markdown("## 📊 Key Epidemic Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Basic Reproduction Number (R₀)",
        value=f"{R0_basic:.2f}",
        delta=f"Effective: {R0_effective:.2f}" if R0_effective != R0_basic else None,
        delta_color="inverse"
    )

with col2:
    if method == "Compare Both":
        st.metric(
            label="Peak Infections",
            value=f"{int(peak_I_e)}",
            delta=f"RK4: {int(peak_I_r)}",
            help="Maximum number of simultaneous infections"
        )
    else:
        st.metric(
            label="Peak Infections",
            value=f"{int(peak_I)}",
            help="Maximum number of simultaneous infections"
        )

with col3:
    if method == "Compare Both":
        st.metric(
            label="Total Cases",
            value=f"{int(total_cases_e)}",
            delta=f"RK4: {int(total_cases_r)}",
            help="Total individuals infected over entire epidemic"
        )
    else:
        st.metric(
            label="Total Cases",
            value=f"{int(total_cases)}",
            help="Total individuals infected over entire epidemic"
        )

with col4:
    if method != "Compare Both":
        st.metric(
            label="Peak Day",
            value=f"Day {int(peak_day)}",
            help="Day when infections reach their maximum"
        )
    else:
        st.metric(
            label="Peak Day Difference",
            value=f"{abs(int(peak_day_e - peak_day_r))} days",
            help="Difference in peak timing between methods"
        )

# Row 2: Main Visualization
st.markdown("## 📈 Epidemic Dynamics Visualization")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Chart", "🔄 Phase Plane", "📉 Comparison", "📊 Data Table"])

with tab1:
    # Create interactive Plotly chart
    fig = go.Figure()
    
    if method == "Compare Both":
        fig.add_trace(go.Scatter(x=t, y=S_e, mode='lines', name='Susceptible (Euler)', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=t, y=I_e, mode='lines', name='Infected (Euler)', line=dict(color='red', width=2)))
        fig.add_trace(go.Scatter(x=t, y=R_e, mode='lines', name='Recovered (Euler)', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=t, y=I_r, mode='lines', name='Infected (RK4)', line=dict(color='orange', width=2, dash='dash')))
    else:
        fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Susceptible', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Infected', line=dict(color='red', width=2)))
        fig.add_trace(go.Scatter(x=t, y=R, mode='lines', name='Recovered', line=dict(color='green', width=2)))
    
    # Add lockdown zone if applicable
    if lockdown_params and lockdown_params['active']:
        fig.add_vrect(
            x0=lockdown_params['start'],
            x1=lockdown_params['start'] + lockdown_params['duration'],
            fillcolor="gray",
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text="Lockdown",
            annotation_position="top left"
        )
    
    fig.update_layout(
        title="SIR Model Simulation",
        xaxis_title="Time (days)",
        yaxis_title="Population",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Phase plane plot
    fig_phase = go.Figure()
    
    # Add baseline trajectory
    fig_phase.add_trace(go.Scatter(
        x=S_base, y=I_base,
        mode='lines',
        name='No Interventions',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    # Add intervention trajectory
    if method == "Compare Both":
        fig_phase.add_trace(go.Scatter(
            x=S_e, y=I_e,
            mode='lines',
            name='With Interventions (Euler)',
            line=dict(color='blue', width=2)
        ))
        fig_phase.add_trace(go.Scatter(
            x=S_r, y=I_r,
            mode='lines',
            name='With Interventions (RK4)',
            line=dict(color='orange', width=2)
        ))
    else:
        fig_phase.add_trace(go.Scatter(
            x=S, y=I,
            mode='lines',
            name='With Interventions',
            line=dict(color='blue', width=2)
        ))
    
    fig_phase.update_layout(
        title="Phase Plane: Infected vs Susceptible",
        xaxis_title="Susceptible Population",
        yaxis_title="Infected Population",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_phase, use_container_width=True)

with tab3:
    # Intervention comparison
    fig_comp = go.Figure()
    
    fig_comp.add_trace(go.Scatter(
        x=t, y=I_base,
        mode='lines',
        name='No Interventions',
        line=dict(color='gray', width=3)
    ))
    
    if method == "Compare Both":
        fig_comp.add_trace(go.Scatter(
            x=t, y=I_e,
            mode='lines',
            name='With Interventions (Euler)',
            line=dict(color='blue', width=2)
        ))
        fig_comp.add_trace(go.Scatter(
            x=t, y=I_r,
            mode='lines',
            name='With Interventions (RK4)',
            line=dict(color='orange', width=2, dash='dash')
        ))
    else:
        fig_comp.add_trace(go.Scatter(
            x=t, y=I,
            mode='lines',
            name='With Interventions',
            line=dict(color='blue', width=2)
        ))
    
    fig_comp.update_layout(
        title="Intervention Impact Comparison",
        xaxis_title="Time (days)",
        yaxis_title="Infected Population",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)

with tab4:
    # Data table
    if method == "Compare Both":
        df = pd.DataFrame({
            'Day': t,
            'Susceptible_Euler': S_e,
            'Infected_Euler': I_e,
            'Recovered_Euler': R_e,
            'Susceptible_RK4': S_r,
            'Infected_RK4': I_r,
            'Recovered_RK4': R_r
        })
    else:
        df = pd.DataFrame({
            'Day': t,
            'Susceptible': S,
            'Infected': I,
            'Recovered': R
        })
    
    st.dataframe(df.style.background_gradient(subset=['Infected' if method != "Compare Both" else 'Infected_Euler'], 
                                             cmap='Reds'),
                height=300)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name="sir_simulation_data.csv",
        mime="text/csv"
    )

# Row 3: Numerical Method Analysis
if method == "Compare Both":
    st.markdown("## 🔬 Numerical Method Analysis")
    
    error_I = np.abs(I_e - I_r)
    rel_error = 100 * error_I / np.maximum(I_r, 1)  # Avoid division by zero
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Error Analysis")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Max Absolute Error', 'Mean Absolute Error', 'Max Relative Error', 'Mean Relative Error'],
            'Value': [f"{np.max(error_I):.2f}", f"{np.mean(error_I):.2f}", 
                     f"{np.max(rel_error):.2f}%", f"{np.mean(rel_error):.2f}%"]
        })
        
        st.table(metrics_df)
        
        # Error visualization
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(x=t, y=error_I, mode='lines', name='Absolute Error', line=dict(color='red')))
        fig_err.update_layout(
            title="Absolute Error Between Euler and RK4",
            xaxis_title="Time (days)",
            yaxis_title="Error (Population)",
            height=300
        )
        st.plotly_chart(fig_err, use_container_width=True)
    
    with col2:
        st.markdown("### Method Recommendation")
        
        st.markdown("""
        <div class="info-box">
        <h4>📊 Accuracy Assessment</h4>
        <ul>
            <li><b>Euler Method:</b> Simpler, faster computation, but larger errors with large time steps</li>
            <li><b>Runge-Kutta 4:</b> More accurate, stable with larger time steps, but slower computation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendation based on error
        if np.mean(rel_error) < 1:
            st.success("✅ Both methods provide similar results. Euler method is sufficient for this simulation.")
        elif np.mean(rel_error) < 5:
            st.warning("⚠️ Moderate differences detected. Consider using RK4 for better accuracy.")
        else:
            st.error("❌ Significant differences detected. RK4 is strongly recommended for accuracy.")
        
        # Performance comparison
        import time
        start = time.time()
        _ = euler_method(S0, I0_initial, R0_initial, beta_eff, gamma, N, dt, t_max, lockdown_params)
        euler_time = time.time() - start
        
        start = time.time()
        _ = rk4_method(S0, I0_initial, R0_initial, beta_eff, gamma, N, dt, t_max, lockdown_params)
        rk4_time = time.time() - start
        
        st.metric("Computation Time (Euler)", f"{euler_time:.4f}s")
        st.metric("Computation Time (RK4)", f"{rk4_time:.4f}s", 
                 delta=f"{(rk4_time/euler_time - 1)*100:.1f}% slower" if rk4_time > euler_time else f"{(1 - rk4_time/euler_time)*100:.1f}% faster")

# Row 4: Intervention Analysis
st.markdown("## 🛡️ Intervention Effectiveness")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Current Interventions")
    
    interventions = []
    if V > 0:
        interventions.append(f"💉 Vaccination: {V} individuals ({V/N*100:.1f}% of population)")
    if sd_strength > 0:
        interventions.append(f"🚶 Social Distancing: {sd_strength}% effectiveness")
    if quarantine_strength > 0:
        interventions.append(f"🏠 Quarantine: {quarantine_strength}% effectiveness")
    if lockdown_params and lockdown_params['active']:
        interventions.append(f"🔒 Lockdown: Day {lockdown_params['start']}-{lockdown_params['start']+lockdown_params['duration']}, {lockdown_params['strength']}% reduction")
    
    if interventions:
        for interv in interventions:
            st.markdown(f"<div class='intervention-card'>{interv}</div>", unsafe_allow_html=True)
    else:
        st.info("No interventions currently active. Try adding some to see their effects!")
    
    # Calculate effectiveness
    transmission_reduction = 100 * (1 - beta_eff/beta)
    if method == "Compare Both":
        peak_reduction_e = 100 * (1 - peak_I_e/np.max(I_base))
        peak_reduction_r = 100 * (1 - peak_I_r/np.max(I_base))
        peak_reduction = f"{peak_reduction_e:.1f}% (Euler) / {peak_reduction_r:.1f}% (RK4)"
    else:
        peak_reduction = f"{100 * (1 - peak_I/np.max(I_base)):.1f}%"
    
    st.metric("Transmission Reduction", f"{transmission_reduction:.1f}%")
    st.metric("Peak Infection Reduction", peak_reduction)

with col2:
    st.markdown("### Scenario Comparison")
    
    # Quick scenario buttons
    st.markdown("#### Try Preset Scenarios:")
    
    col_sc1, col_sc2, col_sc3 = st.columns(3)
    
    with col_sc1:
        if st.button("Mild Response", use_container_width=True):
            st.session_state.beta = 0.2
            st.session_state.sd_strength = 30
            st.rerun()
    
    with col_sc2:
        if st.button("Moderate Response", use_container_width=True):
            st.session_state.beta = 0.15
            st.session_state.sd_strength = 50
            st.session_state.quarantine_strength = 30
            st.rerun()
    
    with col_sc3:
        if st.button("Aggressive Response", use_container_width=True):
            st.session_state.beta = 0.1
            st.session_state.sd_strength = 70
            st.session_state.quarantine_strength = 50
            st.session_state.V = int(N * 0.2)
            st.rerun()
    
    # What-if analysis
    st.markdown("#### What-if Analysis")
    
    what_if_vaccine = st.slider("What if vaccination coverage was:", 0, 100, 50, 5)
    what_if_vaccine_effect = S0 * (what_if_vaccine/100)
    
    # Calculate hypothetical peak
    S_hypo = N - I0 - what_if_vaccine_effect
    beta_hypo = beta_eff
    # Simplified peak estimation
    if S_hypo * beta_hypo / gamma > N:
        peak_hypo = "N/A"
    else:
        peak_hypo_est = N - (gamma/beta_hypo) * (1 + np.log(N * beta_hypo/gamma))
        peak_hypo = f"{max(0, int(peak_hypo_est))}"
    
    st.metric(f"Hypothetical Peak with {what_if_vaccine}% vaccination", peak_hypo)

# Row 5: Additional Information and Export
st.markdown("---")
col_info, col_export = st.columns([2, 1])

with col_info:
    st.markdown("### 📚 About the SIR Model")
    st.markdown("""
    The SIR model divides the population into three compartments:
    - **Susceptible (S)**: Individuals who can contract the disease
    - **Infected (I)**: Individuals currently infected and contagious
    - **Recovered (R)**: Individuals who have recovered and are immune
    
    **Key Parameters:**
    - β (beta): Transmission rate - how easily the disease spreads
    - γ (gamma): Recovery rate - how quickly people recover (1/γ = average infectious period)
    - R₀ = β/γ: Basic reproduction number - average number of secondary cases
    """)

with col_export:
    st.markdown("### 📤 Export Results")
    
    # Export plot
    if st.button("Export Chart as PNG"):
        fig.write_image("sir_simulation.png")
        st.success("Chart exported as sir_simulation.png")
    
    # Export report
    if st.button("Generate Report"):
        report = f"""
        SIR Model Simulation Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Parameters:
        - Population (N): {N}
        - Initial Infected: {I0}
        - Transmission Rate (β): {beta}
        - Recovery Rate (γ): {gamma}
        - Basic R₀: {R0_basic:.2f}
        
        Interventions:
        - Vaccinated: {V}
        - Social Distancing: {sd_strength}%
        - Quarantine: {quarantine_strength}%
        - Effective R₀: {R0_effective:.2f}
        
        Results:
        - Peak Infections: {peak_I if method != "Compare Both" else f"{int(peak_I_e)} (Euler) / {int(peak_I_r)} (RK4)"}
        - Total Cases: {total_cases if method != "Compare Both" else f"{int(total_cases_e)} (Euler) / {int(total_cases_r)} (RK4)"}
        - Transmission Reduction: {transmission_reduction:.1f}%
        """
        
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name="sir_simulation_report.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🦠 SIR Epidemic Simulator | Created with Streamlit | "
    "Based on Kermack-McKendrick SIR Model"
    "</div>",
    unsafe_allow_html=True
) 
