import json
import os

nb_path = r'c:\Users\dani9\.gemini\antigravity\scratch\deep_learning_project\main_experiment.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # New source code for Cell 10
    new_source = [
        "# --- CELL 10: Visualization ---\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "print(\"\\n--- Final Comparative Visualization ---\")\n",
        "\n",
        "# Robust check for results list in global scope\n",
        "if 'results' in globals() and results:\n",
        "    df = pd.DataFrame(results)\n",
        "    \n",
        "    # 1. Deduplication: Keep the LAST run for each method\n",
        "    df = df.drop_duplicates(subset=['Method'], keep='last').reset_index(drop=True)\n",
        "    \n",
        "    # 2. Filter Logic\n",
        "    # Exclude generic pruning, keep only 20% / 0.2\n",
        "    is_pruning = df['Method'].str.contains(\"Pruning\")\n",
        "    is_pruning_20 = is_pruning & (df['Method'].str.contains(\"20%\") | df['Method'].str.contains(\"0.2\"))\n",
        "    \n",
        "    df_plot = df[~is_pruning | is_pruning_20].copy().reset_index(drop=True)\n",
        "    \n",
        "    if df_plot.empty:\n",
        "        print(\"No results matched the filtering criteria. Available methods:\", df['Method'].unique())\n",
        "    else:\n",
        "        # Rename column for the legend if it exists\n",
        "        if 'Task A Acc (Retention)' in df_plot.columns:\n",
        "            df_plot.rename(columns={'Task A Acc (Retention)': 'Task A Final'}, inplace=True)\n",
        "        \n",
        "        plt.figure(figsize=(12, 6))\n",
        "        \n",
        "        # Basic plot parameters\n",
        "        y_cols = [\"Task B Acc (Plasticity)\", \"Task A Final\"]\n",
        "        # Verify columns exist\n",
        "        missing_cols = [c for c in y_cols if c not in df_plot.columns]\n",
        "        if missing_cols:\n",
        "            print(f\"Error: Missing columns in results: {missing_cols}\")\n",
        "        else:\n",
        "            ax = df_plot.plot(\n",
        "                x=\"Method\", \n",
        "                y=y_cols, \n",
        "                kind=\"bar\", \n",
        "                figsize=(12, 6),\n",
        "                color=['#1f77b4', '#ff7f0e'],\n",
        "                width=0.7,\n",
        "                rot=0\n",
        "            )\n",
        "            \n",
        "            plt.title(\"Method Comparison: Plasticity (Task B) vs. Retention (Task A Final)\")\n",
        "            plt.ylabel(\"Accuracy (%)\")\n",
        "            plt.ylim(0, 115)\n",
        "            plt.xticks(rotation=15, ha='right')\n",
        "            plt.grid(axis='y', alpha=0.3)\n",
        "            plt.legend(loc='lower right')\n",
        "            \n",
        "            # 3. Add Time Annotations\n",
        "            if 'Prep Time (s)' in df_plot.columns:\n",
        "                n_groups = len(df_plot)\n",
        "                # We iterate through the patches. \n",
        "                # Note: pandas plot creates 2 sets of bars (one for each Y column).\n",
        "                # The first n_groups patches are for Col 1, the next n_groups for Col 2.\n",
        "                patches = ax.patches\n",
        "                if len(patches) >= 2 * n_groups:\n",
        "                    for i in range(n_groups):\n",
        "                        rect_B = patches[i]\n",
        "                        rect_A = patches[i + n_groups]\n",
        "                        \n",
        "                        group_center = (rect_B.get_x() + rect_A.get_x() + rect_B.get_width() + rect_A.get_width()) / 2 - (rect_B.get_width() / 2)\n",
        "                        max_height = max(rect_B.get_height(), rect_A.get_height())\n",
        "                        \n",
        "                        time_val = df_plot.loc[i, 'Prep Time (s)']\n",
        "                        time_str = f\"{time_val:.2f}s\" if isinstance(time_val, (int, float)) else str(time_val)\n",
        "                        \n",
        "                        ax.text(\n",
        "                            group_center, \n",
        "                            max_height + 2, \n",
        "                            f\"Time:\\n{time_str}\", \n",
        "                            ha='center', \n",
        "                            va='bottom', \n",
        "                            fontsize=9, \n",
        "                            fontweight='bold',\n",
        "                            color='black'\n",
        "                        )\n",
        "            \n",
        "            plt.tight_layout()\n",
        "            plt.show()\n",
        "else:\n",
        "    print(\"No results to visualize. 'results' variable is empty or undefined.\")\n"
    ]

    # Find and update Cell 10
    updated = False
    for i, cell in enumerate(nb['cells']):
        source_str = "".join(cell.get('source', []))
        if "CELL 10" in source_str or "Final Comparative Visualization" in source_str:
            nb['cells'][i]['source'] = new_source
            updated = True
            print(f"Updated Cell index {i}")
            break
            
    if updated:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print("Successfully updated notebook.")
    else:
        print("Could not identify the target cell.")
        
except Exception as e:
    print(f"Error: {e}")
