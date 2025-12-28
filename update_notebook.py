import json
import os

NOTEBOOK_PATH = r"c:\Users\dani9\.gemini\antigravity\scratch\deep_learning_project\main_experiment.ipynb"

def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Notebook not found at {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cells = data.get('cells', [])
    updated = False

    for cell in cells:
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            # Convert list of strings to single string for easier checking, but keep structure for editing
            source_text = "".join(source)
            
            if "Experiment 5 - Randomized SVD" in source_text:
                print("Found Experiment 5 cell.")
                # We want to replace the RSVDProjector instantiation
                # Current: projector = RSVDProjector()
                # Target: projector = RSVDProjector(rank_fraction=0.5, p=10, q=2)
                
                new_source = []
                for line in source:
                    if "projector = RSVDProjector(" in line and "rank_fraction" not in line:
                        prefix = line.split("projector")[0]
                        new_line = prefix + "projector = RSVDProjector(rank_fraction=0.5, p=10, q=2)\n"
                        new_source.append(new_line)
                        print("Updated instantiation line.")
                    else:
                        new_source.append(line)
                
                cell['source'] = new_source
                updated = True
                break
    
    if updated:
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print("Notebook updated successfully.")
    else:
        print("Could not find Experiment 5 cell or it was already updated.")

if __name__ == "__main__":
    update_notebook()
