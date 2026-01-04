import json
import re
import os

nb_path = r'c:\Users\dani9\.gemini\antigravity\scratch\deep_learning_project\main_experiment.ipynb'

if not os.path.exists(nb_path):
    print(f"Error: {nb_path} not found")
    exit(1)

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def inject_code(source_list):
    new_source = []
    
    # helper patterns
    train_pattern = re.compile(r"model_current\s*=\s*(train_baseline|train_constrained)\(")
    results_pattern = re.compile(r"\"Peak VRAM \(MB\)\":\s*(0|peak_vram)")
    print_pattern = re.compile(r"print\(f\"(.*?)Result:(.*?)Peak VRAM=")
    
    i = 0
    while i < len(source_list):
        line = source_list[i]
        
        # Get indentation
        indent = ""
        match = re.search(r"^(\s*)", line)
        if match:
            indent = match.group(1)

        # 1. Capture and Drift Calc
        if train_pattern.search(line):
            # Capture before
            new_source.append(f"{indent}# Save a snapshot of the weights after decomposition/pruning but before training\n")
            new_source.append(f"{indent}w_start = {{name: param.detach().cpu().clone() for name, param in model_current.named_parameters()}}\n")
            new_source.append("\n")
            
            # The training line
            new_source.append(line)
            
            # Drift calc after
            new_source.append("\n")
            new_source.append(f"{indent}drift_sq = 0.0\n")
            new_source.append(f"{indent}for name, param in model_current.named_parameters():\n")
            new_source.append(f"{indent}    if name in w_start:\n")
            new_source.append(f"{indent}        # Calculate squared difference for this layer\n")
            new_source.append(f"{indent}        diff = param.detach().cpu() - w_start[name]\n")
            new_source.append(f"{indent}        drift_sq += torch.sum(diff ** 2).item()\n")
            new_source.append("\n")
            new_source.append(f"{indent}weight_drift = drift_sq ** 0.5\n")
            
            i += 1
            continue
        
        # 2. Update Results Dictionary
        if results_pattern.search(line):
            # Fix comma on the matched line if missing
            # The line is usually '    "Peak VRAM (MB)": 0\n' or similar
            clean_str = line.rstrip('\n')
            if not clean_str.endswith(','):
                # Replace newline with comma+newline
                line = clean_str + ",\n"
            
            new_source.append(line)
            # Add new key
            new_source.append(f'{indent}"Weight Drift": weight_drift\n')
            
            i += 1
            continue

        # 3. Update Print Statement
        if print_pattern.search(line):
            # Insert Drift={weight_drift:.4f}, after Forgetting={forgetting:.2f}%,
            new_line = re.sub(
                r"(Forgetting=\{forgetting:\.2f\}%,)", 
                r"\1 Drift={weight_drift:.4f},", 
                line
            )
            new_source.append(new_line)
            i += 1
            continue
            
        new_source.append(line)
        i += 1
    
    return new_source

modified_count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        # Target experiment cells: contain train_baseline or train_constrained, excluding the expert training
        if ("train_baseline" in src or "train_constrained" in src) and "model_expert = train_baseline" not in src:
            print(f"Modifying cell with source length {len(cell['source'])}...")
            cell['source'] = inject_code(cell['source'])
            modified_count += 1

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4) # ipynb usually uses indentation 1 or 4? Colab is varying, lets use 1 or just default. Standard is often 1 space or 2 in some formats, but JSON dump default is valid. 
    # Actually, standard ipynb often uses space: 1. But let's check the original file content...
    # The original file View showed:
    # "cells": [
    #    {
    # Check indentation of source list... it was 4 spaces in the view.
    # JSON dump indent=4 is safe and readable.

print(f"Modified {modified_count} cells.")
