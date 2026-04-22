import os
import sys
import numpy as np
import pandas as pd
from mesa_reader import MesaData
import matplotlib.pyplot as plt

# ============== ARGUMENTS =====================================
path = f"{sys.argv[1]}/"
age = float(sys.argv[2])
percentile = float(sys.argv[3]) if len(sys.argv) > 3 else 75.0
# ==============================================================

md = np.array([])
file_list = os.listdir(path)
distances = np.array([])

for f in file_list:
    md = np.append(md, MesaData(path + f))

star_data = pd.DataFrame(columns=['luminosities', 'temperatures', 'ages', 'masses'])

for model in md:
    star_data.loc[len(star_data)] = {
        'luminosities': model.data('log_L').tolist(),
        'temperatures': model.data('log_Teff').tolist(),
        'ages': model.star_age.tolist(),
        'masses': model.star_mass[0]
    }

star_data.sort_values(by='masses', ascending=True, na_position='last')


def find_reinterpolation(star_data, desired_age):
    new_temps = np.array([])
    new_lums = np.array([])
    masses_used = np.array([])

    for i in range(len(star_data)):
        ages = star_data.loc[i, 'ages']

        idx1 = None
        idx2 = None

        for j in range(len(ages) - 1):
            if ages[j] <= desired_age <= ages[j + 1]:
                idx1 = j
                idx2 = j + 1
                break

        if idx1 is None:
            continue

        age0 = ages[idx1]
        age1 = ages[idx2]

        t1 = star_data.loc[i, 'temperatures'][idx1]
        t2 = star_data.loc[i, 'temperatures'][idx2]

        l1 = star_data.loc[i, 'luminosities'][idx1]
        l2 = star_data.loc[i, 'luminosities'][idx2]

        w = (desired_age - age0) / (age1 - age0)

        interp_temp = t1 + w * (t2 - t1)
        interp_lum  = l1 + w * (l2 - l1)

        new_temps = np.append(new_temps, interp_temp)
        new_lums = np.append(new_lums, interp_lum)
        masses_used = np.append(masses_used, star_data.loc[i, 'masses'])

    return new_temps, new_lums, masses_used


temps, lums, masses = find_reinterpolation(star_data, age)

new_star_data = pd.DataFrame({
    'temperatures': temps,
    'luminosities': lums,
    'masses': masses
})

new_star_data = new_star_data.sort_values(by='masses', ascending=True)
new_star_data = new_star_data.reset_index(drop=True)

for i in range(len(new_star_data["masses"]) - 1):
    x_diff = (new_star_data["temperatures"][i + 1] - new_star_data["temperatures"][i]) ** 2
    y_diff = (new_star_data["luminosities"][i + 1] - new_star_data["luminosities"][i]) ** 2
    d = np.sqrt(x_diff + y_diff)
    distances = np.append(distances, d)

threshold = np.percentile(distances, percentile)

masses_to_simulate = []

for i in range(len(distances)):
    if distances[i] > threshold:
        m1 = new_star_data.loc[i, 'masses']
        m2 = new_star_data.loc[i + 1, 'masses']

        n_insert = max(1, int(np.floor(distances[i] / threshold)))

        for k in range(1, n_insert + 1):
            new_mass = m1 + (m2 - m1) * k / (n_insert + 1)
            masses_to_simulate.append(new_mass)

masses_to_simulate = np.sort(np.array(masses_to_simulate))
print("(", end="")
for i in masses_to_simulate:
    print(f"{i:.4f}", end=" ")
print(")", end="")