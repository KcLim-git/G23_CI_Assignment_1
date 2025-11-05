# ============================================================
#  Fuzzy Flood Risk Prediction System (Final Version)
#  Using Mamdani Inference with Defuzzification
#  ------------------------------------------------------------
#  Inputs : Rainfall Intensity (mm/hr), Drainage Capacity (%), Land Slope (°)
#  Output : Flood Risk Level (0–100)
#  Purpose: Predict flood risk based on environmental parameters
# ============================================================

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------
import numpy as np                        # For numerical computation and array operations
from skfuzzy import control as ctrl       # scikit-fuzzy control system framework
from skfuzzy import membership as mf      # For membership function definitions
import matplotlib.pyplot as plt           # For plotting graphs
from mpl_toolkits.mplot3d import Axes3D   # For 3D surface visualization

# ------------------------------------------------------------
# PART 1 — Define Input and Output Variables
# ------------------------------------------------------------
# Each fuzzy variable has a universe of discourse (range of values).
# These variables correspond to environmental factors affecting flood risk.

rain = ctrl.Antecedent(np.arange(0, 101, 1), 'rain')   # Input 1: Rainfall intensity (0–100 mm/hr)
drain = ctrl.Antecedent(np.arange(0, 101, 1), 'drain') # Input 2: Drainage capacity (0–100%)
slope = ctrl.Antecedent(np.arange(0, 31, 1), 'slope')  # Input 3: Land slope (0–30°)
risk  = ctrl.Consequent(np.arange(0, 101, 1), 'risk')  # Output : Flood risk level (0–100 scale)

# ------------------------------------------------------------
# PART 2 — Define Membership Functions (MFs)
# ------------------------------------------------------------
# Each input/output variable is represented by linguistic terms (Low, Medium, High, etc.).
# Triangular and trapezoidal shapes are chosen for their interpretability and simplicity.
# Overlapping ranges ensure smooth transitions between categories.

# --- Rainfall Intensity ---
rain['low']      = mf.trapmf(rain.universe, [0, 0, 10, 35])       # Low rain
rain['medium']   = mf.trimf(rain.universe, [25, 50, 75])          # Moderate rain
rain['high']     = mf.trimf(rain.universe, [65, 80, 95])          # High rain
rain['veryhigh'] = mf.trapmf(rain.universe, [85, 90, 100, 100])   # Very high rain

# --- Drainage Capacity ---
drain['poor'] = mf.trapmf(drain.universe, [0, 0, 25, 45])         # Poor drainage (bad infrastructure)
drain['fair'] = mf.trimf(drain.universe, [35, 55, 75])            # Moderate drainage
drain['good'] = mf.trapmf(drain.universe, [65, 85, 100, 100])     # Good drainage system

# --- Land Slope ---
slope['flat']   = mf.trapmf(slope.universe, [0, 0, 2, 6])         # Flat terrain
slope['gentle'] = mf.trimf(slope.universe, [4, 9, 16])            # Gentle slope
slope['steep']  = mf.trapmf(slope.universe, [13, 20, 30, 30])     # Steep terrain

# --- Flood Risk Output ---
risk['low']      = mf.trapmf(risk.universe, [0, 0, 15, 35])       # Low risk
risk['moderate'] = mf.trimf(risk.universe, [25, 45, 65])          # Moderate risk
risk['high']     = mf.trimf(risk.universe, [55, 75, 90])          # High risk
risk['severe']   = mf.trapmf(risk.universe, [80, 90, 100, 100])   # Severe flood risk

# ------------------------------------------------------------
# VISUALIZE MEMBERSHIP FUNCTIONS
# ------------------------------------------------------------
print("Displaying membership functions...")
# Each .view() method generates a separate plot window for clarity
rain.view()
drain.view()
slope.view()
risk.view()

# ------------------------------------------------------------
# PART 3 — Define Fuzzy Rule Base
# ------------------------------------------------------------
# Rules represent expert knowledge or logical reasoning.
# Each rule is written in IF–THEN form, linking environmental conditions to risk levels.

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
rule13 = ctrl.Rule(rain['medium'] | drain['fair'] | slope['gentle'], risk['moderate'])

# Combine all rules into a list
rules = [
    rule1, rule2, rule3, rule4, rule5, rule6,
    rule7, rule8, rule9, rule10, rule11, rule12,rule13
]

# ------------------------------------------------------------
# PART 4 — Build Control System and Simulate
# ------------------------------------------------------------
# The ControlSystem processes the rules and inputs to compute outputs.
# A new simulation object is used to calculate the defuzzified flood risk.

flood_ctrl = ctrl.ControlSystem(rules)         # Create fuzzy control system
flood_sim = ctrl.ControlSystemSimulation(flood_ctrl)  # Initialize simulation engine

# Example input scenario (case study)
flood_sim.input['rain']  = 72    # mm/hr rainfall
flood_sim.input['drain'] = 30    # % drainage capacity
flood_sim.input['slope'] = 6     # ° slope

# Perform fuzzy inference
flood_sim.compute()

# Display the crisp output (after defuzzification using centroid method)
print("Predicted Flood Risk (0–100):", round(flood_sim.output['risk'], 2))

# Visualize the aggregated output fuzzy set and defuzzified centroid
risk.view(sim=flood_sim)
plt.show()

# ------------------------------------------------------------
# Helper Function — 3D Surface Plot
# ------------------------------------------------------------
# This function plots 3D surfaces of flood risk versus two input parameters.
# It helps visualize how changes in input variables affect predicted risk.

def plot3d(x, y, z, xlabel, ylabel, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Create surface plot
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                           cmap='viridis', linewidth=0.4, antialiased=True)
    # Add contour shading for visual depth
    ax.contourf(x, y, z, zdir='z', offset=-10, cmap='viridis', alpha=0.5)
    # Label axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Flood Risk (0–100)')
    ax.set_title(title)
    # Adjust view angle for better visibility
    ax.view_init(30, 210)
    # Add color scale bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8)
    plt.show()

# ------------------------------------------------------------
# PART 5 — 3D Visualization 1: Rainfall × Drainage
# ------------------------------------------------------------
# This surface shows the relationship between rainfall and drainage at a fixed slope (6°).

rain_vals = np.linspace(0, 100, 40)
drain_vals = np.linspace(0, 100, 40)
x, y = np.meshgrid(rain_vals, drain_vals)
z_risk = np.zeros_like(x)

# Compute flood risk for each (rain, drain) pair
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        sim = ctrl.ControlSystemSimulation(flood_ctrl)
        sim.input['rain'] = x[i, j]
        sim.input['drain'] = y[i, j]
        sim.input['slope'] = 6              # keep slope constant
        try:
            sim.compute()
            z_risk[i, j] = sim.output['risk']
        except KeyError:
            z_risk[i, j] = np.nan           # handle undefined cases safely

# Plot the resulting surface
plot3d(x, y, z_risk, "Rainfall (mm/hr)", "Drainage Capacity (%)",
       "Flood Risk Surface (Slope = 6°)")

# ------------------------------------------------------------
# PART 6 — 3D Visualization 2: Rainfall × Slope
# ------------------------------------------------------------
# This surface shows flood risk for different slopes under fixed drainage (40%).

rain_vals2 = np.linspace(0, 100, 40)
slope_vals = np.linspace(0, 30, 40)
x2, y2 = np.meshgrid(rain_vals2, slope_vals)
z_risk2 = np.zeros_like(x2)

for i in range(x2.shape[0]):
    for j in range(x2.shape[1]):
        sim = ctrl.ControlSystemSimulation(flood_ctrl)
        sim.input['rain'] = x2[i, j]
        sim.input['slope'] = y2[i, j]
        sim.input['drain'] = 40             # fixed drainage level
        try:
            sim.compute()
            z_risk2[i, j] = sim.output['risk']
        except KeyError:
            z_risk2[i, j] = np.nan

plot3d(x2, y2, z_risk2, "Rainfall (mm/hr)", "Slope (°)",
       "Flood Risk Surface (Drainage = 40%)")

# ------------------------------------------------------------
# PART 7 — 3D Visualization 3: Drainage × Slope
# ------------------------------------------------------------
# This surface examines how drainage and slope interact under constant heavy rainfall (80 mm/hr).

drain_vals3 = np.linspace(0, 100, 40)
slope_vals3 = np.linspace(0, 30, 40)
x3, y3 = np.meshgrid(drain_vals3, slope_vals3)
z_risk3 = np.zeros_like(x3)

for i in range(x3.shape[0]):
    for j in range(x3.shape[1]):
        sim = ctrl.ControlSystemSimulation(flood_ctrl)
        sim.input['rain'] = 80               # fixed heavy rainfall
        sim.input['drain'] = x3[i, j]
        sim.input['slope'] = y3[i, j]
        try:
            sim.compute()
            z_risk3[i, j] = sim.output['risk']
        except KeyError:
            z_risk3[i, j] = np.nan

plot3d(x3, y3, z_risk3, "Drainage Capacity (%)", "Slope (°)",
       "Flood Risk Surface (Rainfall = 80 mm/hr)")

# ------------------------------------------------------------
# PART 8 — 3D Visualization 4: Multiple Slope Levels
# ------------------------------------------------------------
# This loop produces three separate 3D surfaces for different slope values (2°, 10°, 20°),
# showing how slope influences the rainfall–drainage relationship.

for slope_val in [2, 10, 20]:  # representing flat, gentle, and steep terrains
    rain_vals4 = np.linspace(0, 100, 40)
    drain_vals4 = np.linspace(0, 100, 40)
    x4, y4 = np.meshgrid(rain_vals4, drain_vals4)
    z_risk4 = np.zeros_like(x4)

    for i in range(x4.shape[0]):
        for j in range(x4.shape[1]):
            sim = ctrl.ControlSystemSimulation(flood_ctrl)
            sim.input['rain'] = x4[i, j]
            sim.input['drain'] = y4[i, j]
            sim.input['slope'] = slope_val
            try:
                sim.compute()
                z_risk4[i, j] = sim.output['risk']
            except KeyError:
                z_risk4[i, j] = np.nan

    plot3d(x4, y4, z_risk4,
           "Rainfall (mm/hr)", "Drainage Capacity (%)",
           f"Flood Risk Surface (Slope = {slope_val}°)")

