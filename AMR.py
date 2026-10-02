import os
import sys
import numpy as np
import pandas as pd
from mesa_reader import MesaData

# ============== ARGUMENTS =====================================
path = f"{sys.argv[1]}/"
age = float(sys.argv[2])
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

masses = []          # masses that reach the age
masses_short = []    # masses that do NOT reach the age
count = 0

for track_ages, track_mass in zip(star_data.ages, star_data.masses):
    reached = False
    for a in track_ages:
        if a >= age:
            reached = True
            break
    if reached:
        masses.append(track_mass)
        count = count + 1
    else:
        masses_short.append(track_mass)

# print(f"number of files: {len(file_list)}")
# print(f"number of files with ages >= {age:.3g} yr: {count}")

if masses:
    max_reaching = max(masses) + 0.00000001
    # print(f"maximum mass value that reaches {age:.3g} yr: {max_reaching}")

    above = [m for m in masses_short if m > max_reaching]
    if above:
        min_out = min(above) - 0.00000001
        # print(f"next mass up that does NOT reach {age:.3g} yr: {min_out}")

masses_to_simulate = np.linspace(max_reaching, min_out, 150)
masses_to_simulate = masses_to_simulate.tolist()

star_data.sort_values(by='masses', ascending=True, na_position='last')

def __find_closest_age_index(self, age_array, desired_age):
    age_array = np.array(age_array)
    idx = (np.abs(age_array - desired_age)).argmin()
    percent_error = 100 * abs(age_array[idx] - desired_age) / desired_age
    return age_array[idx], idx, percent_error

temps = np.array([])
lums = np.array([])
masses = np.array([])

for i in range(len(star_data)):
    ages = star_data.loc[i, 'ages']

    closest_age, idx, percent_error = __find_closest_age_index(None, ages, age)
    if percent_error <= 0.1:
        temps = np.append(temps, star_data.loc[i, 'temperatures'][idx])
        lums = np.append(lums, star_data.loc[i, 'luminosities'][idx])
        masses = np.append(masses, star_data.loc[i, 'masses'])

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


threshold = 0.015

for i in range(len(distances)):
    if distances[i] > threshold:
        m1 = new_star_data.loc[i, 'masses']
        m2 = new_star_data.loc[i + 1, 'masses']

        n_insert = max(1, int(np.floor(distances[i] / threshold)))

        for k in range(1, n_insert + 1):
            new_mass = m1 + (m2 - m1) * k / (n_insert + 1)
            masses_to_simulate.append(new_mass)


masses_to_simulate = np.sort(np.array(masses_to_simulate))


print(" ".join(f"{m}" for m in masses_to_simulate))
