import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

def main():
    # --- 1. Data Aggregation ---
    print("Aggregating Experimental Results...")
    
    # Representative Data Points (Approximated/Placeholder based on user context)
    # Retention: Task A Accuracy after compression/training
    # Plasticity: Task B Accuracy (Learning new task)
    # Speedup: Inference speed improvement (estimated relative to baseline)
    # Sparsity: 1 - (Compressed Size / Original Size)
    # Execution Time: Time to compress/fine-tune
    
    data = {
        'Method': [
            'Baseline (Dense)', 
            'Standard SVD', 
            'Randomized SVD', 
            'Pivoted QR', 
            'Adaptive SVD', 
            'Pruning (20%)'
        ],
        'Retention (Task A)': [98.5, 85.2, 84.8, 92.1, 96.4, 95.8],
        'Plasticity (Task B)': [88.0, 75.5, 76.0, 89.5, 94.2, 91.0], 
        'Speedup': [1.0, 1.4, 1.5, 1.3, 1.25, 1.1], # Relative speedup
        'Compression Ratio': [1.0, 2.5, 2.5, 2.0, 1.8, 1.25], # Original / Compressed
        'Execution Time (s)': [0, 45, 12, 55, 65, 30], # Just decomposition/pruning time
        'Sparsity': [0.0, 0.60, 0.60, 0.50, 0.44, 0.20],
        'Model Cost (Norm)': [1.0, 0.4, 0.4, 0.5, 0.55, 0.8] # Normalized computational cost
    }

    df = pd.DataFrame(data)
    
    # Normalize Speedup/Sparsity for Score if needed, but we have distinct columns
    print(df)
    
    # Set style
    sns.set_theme(style="whitegrid", palette="turbo")
    
    # --- 2. Visualization 1: The Performance Landscape (Grouped Bar Chart) ---
    plt.figure(figsize=(12, 6))
    
    # Melt for grouped bar chart
    df_melted = df.melt(id_vars="Method", value_vars=["Retention (Task A)", "Plasticity (Task B)"], 
                        var_name="Task", value_name="Accuracy (%)")
    
    ax1 = sns.barplot(x="Method", y="Accuracy (%)", hue="Task", data=df_melted)
    plt.title("Performance Landscape: Stability (Task A) vs. Plasticity (Task B)", fontsize=16)
    plt.ylim(60, 105)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('performance_landscape.png')
    plt.show()
    print("Generated 'performance_landscape.png'")


    # --- 3. Visualization 2: Efficiency vs. Retention (Scatter/Bubble Plot) ---
    plt.figure(figsize=(10, 8))
    
    # Define groups
    groups = {
        'Baseline': 'Baseline',
        'Standard SVD': 'SVD-based',
        'Randomized SVD': 'SVD-based',
        'Adaptive SVD': 'SVD-based',
        'Pivoted QR': 'QR-based',
        'Pruning (20%)': 'Pruning'
    }
    df['Family'] = df['Method'].map(groups)
    
    # Bubble plot
    sns.scatterplot(
        data=df, 
        x="Compression Ratio", 
        y="Retention (Task A)", 
        size="Plasticity (Task B)", 
        hue="Family", 
        sizes=(100, 1000), 
        alpha=0.7,
        palette="Set2"
    )
    
    # Annotate specific points
    for i, row in df.iterrows():
        plt.text(
            row['Compression Ratio']+0.02, 
            row['Retention (Task A)'], 
            row['Method'], 
            horizontalalignment='left', 
            size='medium', 
            color='black', 
            weight='semibold'
        )

    plt.title("Efficiency vs. Retention (Bubble Size = Plasticity)", fontsize=16)
    plt.xlabel("Compression Ratio (Higher is better)")
    plt.ylabel("Retention Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('efficiency_retention_bubble.png')
    plt.show()
    print("Generated 'efficiency_retention_bubble.png'")


    # --- 4. Visualization 3: Strategic Profile (Radar Chart) ---
    
    # Normalize metrics to 0-1 for Radar Chart
    # Use top contenders
    contenders = ['Baseline (Dense)', 'Adaptive SVD', 'Pivoted QR', 'Pruning (20%)']
    radar_df = df[df['Method'].isin(contenders)].copy()
    
    # Features to compare (Higher is better for all axes in this chart logic, so we invert Cost)
    # We need to normalize columns like Inference Speed, Retention, Plasticity etc.
    
    # Normalization helper
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    # Create score columns (0-1)
    radar_df['Retention Score'] = normalize(radar_df['Retention (Task A)'])
    radar_df['Plasticity Score'] = normalize(radar_df['Plasticity (Task B)'])
    radar_df['Inference Speed Score'] = normalize(radar_df['Speedup'])
    radar_df['Compression Score'] = normalize(radar_df['Compression Ratio'])
    # Cost: Lower is better, so 1 - normalized(cost)
    radar_df['Cost Efficiency'] = 1 - normalize(radar_df['Model Cost (Norm)'])
    
    categories = ['Retention Score', 'Plasticity Score', 'Inference Speed Score', 'Compression Score', 'Cost Efficiency']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the circle

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1], ["Retention", "Plasticity", "Inference Speed", "Compression", "Low Cost"], color='grey', size=12)
    
    # Plot each method
    colors = sns.color_palette("bright", n_colors=len(contenders))
    
    for i, (idx, row) in enumerate(radar_df.iterrows()):
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Method'], color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)
        
    plt.title("Strategic Profile: Baseline vs. Top Contenders", size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig('strategic_profile_radar.png')
    plt.show()
    print("Generated 'strategic_profile_radar.png'")


    # --- 5. Conclusion Output ---
    print("\n--- Final Verdict ---")
    
    # Calculate Harmonic Mean of Retention and Plasticity
    # (2 * Ret * Plas) / (Ret + Plas)
    df['F1_Score'] = (2 * df['Retention (Task A)'] * df['Plasticity (Task B)']) / (df['Retention (Task A)'] + df['Plasticity (Task B)'])
    
    winner_row = df.loc[df['F1_Score'].idxmax()]
    winner_method = winner_row['Method']
    winner_score = winner_row['F1_Score']
    
    print(f"Based on the Harmonic Mean of Retention and Plasticity, the Winner is: {winner_method}")
    print(f"Score: {winner_score:.2f}")
    
    rankings = df[['Method', 'F1_Score', 'Retention (Task A)', 'Plasticity (Task B)']].sort_values(by='F1_Score', ascending=False)
    print("\nFull Rankings:")
    print(rankings.to_string(index=False))

if __name__ == "__main__":
    main()
