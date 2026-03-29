# 🦠 SIR Epidemic Simulation Dashboard

An interactive web-based dashboard for simulating and analyzing disease epidemics using the SIR (Susceptible-Infected-Recovered) compartmental model.

## Overview

This project provides an educational tool to explore epidemic dynamics through numerical simulation. It allows users to configure population parameters, compare different numerical integration methods, and evaluate the impact of public health intervention strategies on disease spread.

## Features

### Core Functionality
- **SIR Model Implementation**: Classic epidemiological compartmental model for disease transmission
- **Numerical Methods**: Support for multiple integration techniques
  - Euler Method (first-order)
  - Runge-Kutta 4 (fourth-order)
- **Interactive Dashboard**: Built with Streamlit for easy parameter exploration
- **Real-time Visualization**: Dynamic plots using Matplotlib and Plotly

### Population Configuration
- Adjustable population size (N)
- Initial infected count (I₀)
- Initial recovered/vaccinated count (V)

### Model Parameters
- **β (Beta)**: Transmission rate - average contacts per person per time unit
- **γ (Gamma)**: Recovery rate - inverse of average infectious period
- **R₀**: Basic reproduction number - automatically calculated from β and γ

### Public Health Interventions
- **Social Distancing**: Reduces transmission rate
- **Quarantine**: Isolates infected individuals
- **Vaccination**: Pre-emptively recovers population members
- **Lockdown**: Time-specific transmission reduction with configurable:
  - Start day
  - Duration
  - Transmission reduction percentage

### Analysis Metrics
- Peak infection calculation
- Attack rate estimation
- Effective R₀ with interventions
- Time series data for S, I, R compartments

## Requirements

- Python 3.8+
- NumPy: Numerical computations
- Matplotlib: Static visualizations
- Plotly: Interactive visualizations
- Pandas: Data manipulation
- Streamlit: Web framework

## Installation

1. Clone or download the repository
2. Install required packages:
```bash
pip install numpy matplotlib streamlit pandas plotly
```

## Usage

Run the Streamlit application:
```bash
streamlit run Numerical_modellinh.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

### Quick Start

1. **Configure Parameters** in the sidebar:
   - Set population size and initial conditions
   - Adjust transmission (β) and recovery (γ) rates
   - Review calculated R₀ value

2. **Select Numerical Method**:
   - Choose between Euler or Runge-Kutta 4
   - Set simulation time and time step

3. **Apply Interventions** (optional):
   - Enable vaccination, social distancing, quarantine, or lockdown
   - Adjust intervention strength and timing

4. **View Results**:
   - Explore interactive plots of S, I, R populations over time
   - Compare results across different intervention scenarios
   - Analyze epidemic metrics

## Model Details

### SIR Equations

The model is defined by the following system of differential equations:

- **dS/dt** = -β·S·I / N
- **dI/dt** = β·S·I / N - γ·I  
- **dR/dt** = γ·I

Where:
- S = Susceptible population
- I = Infected population
- R = Recovered population
- N = Total population (S + I + R)

### Numerical Methods

**Euler Method**: First-order integration method for solving differential equations
- Simple and fast
- Less accurate for larger time steps
- Useful for quick exploratory simulations

**Runge-Kutta 4 (RK4)**: Fourth-order integration method
- More accurate than Euler
- Better stability for longer simulations
- Recommended for precise analysis

## Project Structure

```
Computational biology/
├── Numerical_modellinh.py  # Main Streamlit application
├── README.md               # This file
├── div                     # Output directory placeholder
└── euler_time              # Stored data placeholder
```

## Interpretation Guide

### Understanding R₀
- **R₀ > 1**: Disease spreads exponentially (epidemic)
- **R₀ < 1**: Disease dies out (disease eliminated)
- **R₀ ≈ 1**: Endemic state (stable circulation)

### Peak Infection
The maximum number of simultaneous infections during the epidemic curve. Flattening the curve (reducing peak) is achieved through interventions that reduce transmission.

### Attack Rate
Percentage of the population infected during the entire epidemic. Lower attack rates indicate more effective interventions.

## Educational Uses

This tool is suitable for:
- Understanding basic epidemiological principles
- Exploring the impact of disease parameters on spread
- Evaluating intervention effectiveness
- Comparing numerical solution accuracy
- Teaching computational biology and applied mathematics

## Notes

- The file `Numerical_modellinh.py` contains the complete implementation
- Placeholder files (`div`, `euler_time`) are included for future data storage
- All parameter ranges include helpful tooltips with explanations

## License

This project is provided for educational purposes.

## Authors

Developed as a computational biology educational tool.
