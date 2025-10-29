import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import matplotlib.pyplot as plt

# ------------------------------
# Define fuzzy input and output variables
# ------------------------------
rain = ctrl.Antecedent(np.arange(0, 101, 1), 'rain')
drain = ctrl.Antecedent(np.arange(0, 101, 1), 'drain')
slope = ctrl.Antecedent(np.arange(0, 31, 1), 'slope')
risk  = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

# ------------------------------
# Membership Functions
# ------------------------------
rain['low']      = fuzz.trapmf(rain.universe, [0, 0, 10, 30])
rain['medium']   = fuzz.trimf(rain.universe, [20, 45, 70])
rain['high']     = fuzz.trimf(rain.universe, [60, 75, 90])
rain['veryhigh'] = fuzz.trapmf(rain.universe, [80, 90, 100, 100])

drain['poor'] = fuzz.trapmf(drain.universe, [0, 0, 20, 40])
drain['fair'] = fuzz.trimf(drain.universe, [30, 50, 70])
drain['good'] = fuzz.trapmf(drain.universe, [60, 80, 100, 100])

slope['flat']   = fuzz.trapmf(slope.universe, [0, 0, 2, 5])
slope['gentle'] = fuzz.trimf(slope.universe, [3, 8, 15])
slope['steep']  = fuzz.trapmf(slope.universe, [12, 20, 30, 30])

risk['low']      = fuzz.trapmf(risk.universe, [0, 0, 15, 35])
risk['moderate'] = fuzz.trimf(risk.universe, [25, 45, 65])
risk['high']     = fuzz.trimf(risk.universe, [55, 75, 90])
risk['severe']   = fuzz.trapmf(risk.universe, [80, 90, 100, 100])

# ------------------------------
# Plot MFs for report/slide export
# ------------------------------
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

rain.view(ax=axs[0,0])
drain.view(ax=axs[0,1])
slope.view(ax=axs[1,0])
risk.view(ax=axs[1,1])

plt.tight_layout()
plt.show()

# ------------------------------
# Rule Base (12 example rules)
# ------------------------------
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

# ------------------------------
# System and Simulation
# ------------------------------
system = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6,
    rule7, rule8, rule9, rule10, rule11, rule12
])
sim = ctrl.ControlSystemSimulation(system)

# ------------------------------
# Example Case Study
# ------------------------------
sim.input['rain']  = 72   # mm/hr
sim.input['drain'] = 30   # %
sim.input['slope'] = 6    # degrees

sim.compute()

print("Predicted Flood Risk Level (0-100):", round(sim.output['risk'], 2))
risk.view(sim=sim)
plt.show()
