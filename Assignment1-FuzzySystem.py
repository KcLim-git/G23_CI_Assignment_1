import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# Fuzzy Flood Risk Prediction System (Final Version)
# ============================================================

# ------------------------------------------------------------
# PART 1 — Define Variables
# ------------------------------------------------------------
rain = ctrl.Antecedent(np.arange(0, 101, 1), 'rain')
drain = ctrl.Antecedent(np.arange(0, 101, 1), 'drain')
slope = ctrl.Antecedent(np.arange(0, 31, 1), 'slope')
risk  = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

# ------------------------------------------------------------
# PART 2 — Membership Functions (with overlap fixes)
# ------------------------------------------------------------
rain['low']      = mf.trapmf(rain.universe, [0, 0, 10, 35])
rain['medium']   = mf.trimf(rain.universe, [25, 50, 75])
rain['high']     = mf.trimf(rain.universe, [65, 80, 95])
rain['veryhigh'] = mf.trapmf(rain.universe, [85, 90, 100, 100])

drain['poor'] = mf.trapmf(drain.universe, [0, 0, 25, 45])
drain['fair'] = mf.trimf(drain.universe, [35, 55, 75])
drain['good'] = mf.trapmf(drain.universe, [65, 85, 100, 100])

slope['flat']   = mf.trapmf(slope.universe, [0, 0, 2, 6])
slope['gentle'] = mf.trimf(slope.universe, [4, 9, 16])
slope['steep']  = mf.trapmf(slope.universe, [13, 20, 30, 30])

risk['low']      = mf.trapmf(risk.universe, [0, 0, 15, 35])
risk['moderate'] = mf.trimf(risk.universe, [25, 45, 65])
risk['high']     = mf.trimf(risk.universe, [55, 75, 90])
risk['severe']   = mf.trapmf(risk.universe, [80, 90, 100, 100])

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

rain.view()
axs[0, 0].set_title("Rainfall Intensity (mm/hr)")

drain.view()
axs[0, 1].set_title("Drainage Capacity (%)")

slope.view()
axs[1, 0].set_title("Land Slope (°)")

risk.view()
axs[1, 1].set_title("Flood Risk Level (0–100)")

plt.tight_layout()
plt.show()




# ------------------------------------------------------------
# PART 3 — Rule Base (12 + fallback)
# ------------------------------------------------------------
rule1  = ctrl.Rule(rain['veryhigh'] & drain['poor'], risk['severe'])
rule2  = ctrl.Rule(rain['high'] & drain['poor'] & slope['flat'], risk['severe'])
rule3  = ctrl.Rule(rain['high'] & drain['fair'], risk['high'])
rule4  = ctrl.Rule(rain['medium'] & drain['poor'], risk['high'])
rule5  = ctrl.Rule(rain['medium'] & drain['good'], risk['moderate'])
rule6  = ctrl.Rule(rain['low'] & drain['good'], risk['low'])
rule7  = ctrl.Rule(rain['high'] & slope['steep'] & drain['good'], risk['moderate'])
rule8  = ctrl.Rule(rain['veryhigh'] & slope['flat'], risk['severe'])
rule9  = ctrl.Rule(rain['low'] & slope['steep'] & drain['fair'], risk['low'])
rule10 = ctrl.Rule(rain['medium'] & slope['flat'] & drain['fair'], risk['high'])
rule11 = ctrl.Rule(rain['high'] & slope['gentle'] & drain['fair'], risk['high'])
rule12 = ctrl.Rule(rain['medium'] & slope['gentle'] & drain['fair'], risk['moderate'])
rule13 = ctrl.Rule(rain['medium'] & drain['fair'] & slope['gentle'], risk['moderate'])  # fallback

rules = [
    rule1, rule2, rule3, rule4, rule5, rule6,
    rule7, rule8, rule9, rule10, rule11, rule12, rule13
]

# ------------------------------------------------------------
# PART 4 — Control System & Simulation
# ------------------------------------------------------------
flood_ctrl = ctrl.ControlSystem(rules)
flood_sim = ctrl.ControlSystemSimulation(flood_ctrl)

# Example case
flood_sim.input['rain']  = 72
flood_sim.input['drain'] = 30
flood_sim.input['slope'] = 6

flood_sim.compute()
print("Predicted Flood Risk (0–100):", round(flood_sim.output['risk'], 2))
risk.view(sim=flood_sim)
plt.show()

# ------------------------------------------------------------
# Helper: 3D Plot Function
# ------------------------------------------------------------
def plot3d(x, y, z, xlabel, ylabel, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                           cmap='viridis', linewidth=0.4, antialiased=True)
    ax.contourf(x, y, z, zdir='z', offset=-10, cmap='viridis', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Flood Risk (0–100)')
    ax.set_title(title)
    ax.view_init(30, 210)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8)
    plt.show()

# ------------------------------------------------------------
# PART 5 — 3D Visualization (Rain × Drain)
# ------------------------------------------------------------
rain_vals = np.linspace(0, 100, 40)
drain_vals = np.linspace(0, 100, 40)
x, y = np.meshgrid(rain_vals, drain_vals)
z_risk = np.zeros_like(x)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        sim = ctrl.ControlSystemSimulation(flood_ctrl)
        sim.input['rain'] = x[i, j]
        sim.input['drain'] = y[i, j]
        sim.input['slope'] = 6
        try:
            sim.compute()
            z_risk[i, j] = sim.output['risk']
        except KeyError:
            z_risk[i, j] = np.nan

plot3d(x, y, z_risk, "Rainfall (mm/hr)", "Drainage Capacity (%)",
       "Flood Risk Surface (Slope = 6°)")

# ------------------------------------------------------------
# PART 6 — Second 3D Visualization (Rain × Slope)
# ------------------------------------------------------------
rain_vals2 = np.linspace(0, 100, 40)
slope_vals = np.linspace(0, 30, 40)
x2, y2 = np.meshgrid(rain_vals2, slope_vals)
z_risk2 = np.zeros_like(x2)

for i in range(x2.shape[0]):
    for j in range(x2.shape[1]):
        sim = ctrl.ControlSystemSimulation(flood_ctrl)
        sim.input['rain'] = x2[i, j]
        sim.input['slope'] = y2[i, j]
        sim.input['drain'] = 40  # fixed drainage
        try:
            sim.compute()
            z_risk2[i, j] = sim.output['risk']
        except KeyError:
            z_risk2[i, j] = np.nan

plot3d(x2, y2, z_risk2, "Rainfall (mm/hr)", "Slope (°)",
       "Flood Risk Surface (Drainage = 40%)")
