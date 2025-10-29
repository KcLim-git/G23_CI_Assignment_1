
import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------------------------------------------------------
# PART 1 — Define Variables
# Inputs: Rainfall (0–100 mm/hr), Drainage (0–100%), Slope (0–30°)
# Output: Flood Risk (0–100)
# ------------------------------------------------------------

rain = ctrl.Antecedent(np.arange(0, 101, 1), 'rain')
drain = ctrl.Antecedent(np.arange(0, 101, 1), 'drain')
slope = ctrl.Antecedent(np.arange(0, 31, 1), 'slope')
risk  = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

# ------------------------------------------------------------
# PART 2 — Membership Functions (Triangular/Trapezoidal)
# ------------------------------------------------------------

# Rainfall Intensity (mm/hr)
rain['low']      = mf.trapmf(rain.universe, [0, 0, 10, 30])
rain['medium']   = mf.trimf(rain.universe, [20, 45, 70])
rain['high']     = mf.trimf(rain.universe, [60, 75, 90])
rain['veryhigh'] = mf.trapmf(rain.universe, [80, 90, 100, 100])

# Drainage Capacity (%)
drain['poor'] = mf.trapmf(drain.universe, [0, 0, 20, 40])
drain['fair'] = mf.trimf(drain.universe, [30, 50, 70])
drain['good'] = mf.trapmf(drain.universe, [60, 80, 100, 100])

# Land Slope (degrees)
slope['flat']   = mf.trapmf(slope.universe, [0, 0, 2, 5])
slope['gentle'] = mf.trimf(slope.universe, [3, 8, 15])
slope['steep']  = mf.trapmf(slope.universe, [12, 20, 30, 30])

# Flood Risk (0–100)
risk['low']      = mf.trapmf(risk.universe, [0, 0, 15, 35])
risk['moderate'] = mf.trimf(risk.universe, [25, 45, 65])
risk['high']     = mf.trimf(risk.universe, [55, 75, 90])
risk['severe']   = mf.trapmf(risk.universe, [80, 90, 100, 100])

# ------------------------------------------------------------
# Optional: Visualize the membership functions
# ------------------------------------------------------------
rain.view(); drain.view(); slope.view(); risk.view()
plt.show()

# ------------------------------------------------------------
# PART 3 — Define Rule Base (12 rules)
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

rules = [
    rule1, rule2, rule3, rule4, rule5, rule6,
    rule7, rule8, rule9, rule10, rule11, rule12
]

# ------------------------------------------------------------
# PART 4 — Build Control System & Simulation
# ------------------------------------------------------------

flood_ctrl = ctrl.ControlSystem(rules)
flood_sim = ctrl.ControlSystemSimulation(flood_ctrl)

# Example input case (hypothetical)
flood_sim.input['rain']  = 72   # mm/hr
flood_sim.input['drain'] = 30   # %
flood_sim.input['slope'] = 6    # degrees

# Compute inference
flood_sim.compute()

# Print numerical output (defuzzified centroid)
print("Predicted Flood Risk (0–100):", round(flood_sim.output['risk'], 2))
risk.view(sim=flood_sim)
plt.show()

# ------------------------------------------------------------
# PART 5 — 3D Visualization (Rain vs Drain vs Risk)
# ------------------------------------------------------------
def plot3d(x, y, z, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Main surface
    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    cmap='viridis', linewidth=0.4, antialiased=True)

    # Contour projections on each axis plane
    ax.contourf(x, y, z, zdir='z', offset=-10, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='x', offset=x.max()*1.2, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='y', offset=y.max()*1.2, cmap='viridis', alpha=0.5)

    ax.set_xlabel('Rainfall (mm/hr)')
    ax.set_ylabel('Drainage Capacity (%)')
    ax.set_zlabel('Flood Risk (0–100)')
    ax.set_title(title)
    ax.view_init(30, 210)
    plt.show()

# ------------------------------------------------------------
# Compute 3D surface values
# ------------------------------------------------------------
rain_vals = np.linspace(0, 100, 40)
drain_vals = np.linspace(0, 100, 40)
x, y = np.meshgrid(rain_vals, drain_vals)
z_risk = np.zeros_like(x)

# Generate risk values
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        sim = ctrl.ControlSystemSimulation(flood_ctrl)
        sim.input['rain'] = x[i, j]
        sim.input['drain'] = y[i, j]
        sim.input['slope'] = 6  # fixed slope angle
        try:
            sim.compute()
            z_risk[i, j] = sim.output['risk']
        except Exception:
            z_risk[i, j] = np.nan

# ------------------------------------------------------------
# Plot the surface
# ------------------------------------------------------------
plot3d(x, y, z_risk, "Flood Risk Surface (Slope = 6°)")