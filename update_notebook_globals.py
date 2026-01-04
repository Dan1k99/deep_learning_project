import json
import os

nb_path = r'c:\Users\dani9\.gemini\antigravity\scratch\deep_learning_project\main_experiment.ipynb'

if not os.path.exists(nb_path):
    print(f"Error: {nb_path} not found")
    exit(1)

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 2 (Configuration)
# Find the line "EPOCHS_B = 15" and replace it
cell2_source = nb['cells'][1]['source']
new_cell2_source = []
for line in cell2_source:
    if "EPOCHS_B = 15" in line:
        new_cell2_source.append("task_b_epochs_list = [5, 10, 15]\n")
    else:
        new_cell2_source.append(line)
nb['cells'][1]['source'] = new_cell2_source

# 2. Update Experiment Cells (4, 5, 6, 7, 8, 9)
# Remove the local definition "task_b_epochs_list = [5, 10, 15]\n"
experiment_indices = [3, 4, 5, 6, 7, 8] # 0-indexed in python list, matches cells 4,5,6,7,8,9

for idx in experiment_indices:
    source = nb['cells'][idx]['source']
    new_source = [line for line in source if "task_b_epochs_list = [5, 10, 15]" not in line]
    nb['cells'][idx]['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("Updated notebook global configuration.")
