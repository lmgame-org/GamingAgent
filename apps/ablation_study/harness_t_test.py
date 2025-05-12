import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Create the data
data = {
    'Model': [
        'claude-3-5-sonnet-20241022', 'claude-3-5-sonnet-20241022',
        'claude-3-7-sonnet-20250219 (thinking)', 'claude-3-7-sonnet-20250219 (thinking)',
        'deepseek-r1', 'deepseek-r1',
        'gemini-2.5-flash-preview-04-17', 'gemini-2.5-flash-preview-04-17',
        'gemini-2.5-pro-preview-03-25', 'gemini-2.5-pro-preview-03-25',
        'grok-3-mini-beta (thinking)', 'grok-3-mini-beta (thinking)',
        'llama-4-maverick-17b-128e-instruct-fp8', 'llama-4-maverick-17b-128e-instruct-fp8',
        'gpt-4.1-2025-04-14', 'gpt-4.1-2025-04-14',
        'gpt-4o-2024-11-20', 'gpt-4o-2024-11-20',
        'o1-2024-12-17', 'o1-2024-12-17',
        'o3-2025-04-16', 'o3-2025-04-16',
        'o4-mini-2025-04-16', 'o4-mini-2025-04-16'
    ],
    'Harness': ['No', 'Yes'] * 12,
    'Sokoban': [0, 0, 0, 2, np.nan, 1, 0, 2, 0, 3, np.nan, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 5, 1, 2],
    'Super Mario Bros': [1540, 1267.67, 1430, 1683, np.nan, np.nan, 1540.67, 1671, 1700.67, np.nan, np.nan, np.nan, 786, 1485.33, 1991.33, 2126.33, 1028.33, 2047.33, 1434, 855, 1955, 3445, 1348.33, 1448],
    'Tetris': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 56, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    '2048': [84, 1914.67, 2953.33, 2624, np.nan, np.nan, 1738.67, np.nan, np.nan, 128, np.nan, 256, 382.67, 769.33, 1113.33, 1656, 176, 1656, 7176, 256, 7220, np.nan, 1882.67, 4432],
    'Candy Crush': [17, 106, np.nan, 35.33, np.nan, 91.67, 97.67, np.nan, np.nan, 51.33, np.nan, 106, 32.33, 128.67, 101, 182, 59, 147.33, 90, 159, 106, np.nan, 110.67, 487.33],
    'Ace Attorney': [np.nan, 2, np.nan, 7, np.nan, 0, 1, 4, np.nan, 7, np.nan, 0, np.nan, 0, 0, 2, 0, 0, np.nan, 14, np.nan, 14, 2, 4]
}

# Create DataFrame
df = pd.DataFrame(data)

def analyze_game_differences(game_name):
    # Create a pivot table with Model as index and Harness as columns
    pivot_df = df.pivot(index='Model', columns='Harness', values=game_name)
    
    # Convert values to numeric, removing any non-numeric characters
    for col in ['No', 'Yes']:
        pivot_df[col] = pd.to_numeric(pivot_df[col].astype(str).str.replace('*', ''), errors='coerce')
    
    # Remove rows where either No or Yes is NaN
    pivot_df = pivot_df.dropna()
    
    if len(pivot_df) < 2:
        print(f"\n{game_name}:")
        print("Not enough valid pairs for analysis")
        return None
    
    # Calculate differences
    differences = pivot_df['Yes'] - pivot_df['No']
    
    # Perform t-test
    t_stat, p_value = stats.ttest_rel(pivot_df['No'], pivot_df['Yes'])
    
    # Calculate statistics
    mean_diff = differences.mean()
    std_diff = differences.std()
    n = len(differences)
    sem = std_diff / np.sqrt(n)  # Standard error of the mean
    ci_95 = 1.96 * sem  # 95% confidence interval
    
    return {
        'game': game_name,
        'mean_diff': mean_diff,
        'ci_lower': mean_diff - ci_95,
        'ci_upper': mean_diff + ci_95,
        'n_pairs': n,
        'differences': differences,
        'p_value': p_value,
        't_stat': t_stat
    }

# Analyze all games
games = ['Sokoban', 'Super Mario Bros', 'Tetris', '2048', 'Candy Crush', 'Ace Attorney']
results = []

for game in games:
    result = analyze_game_differences(game)
    if result is not None:
        results.append(result)

# Create visualization
n_games = len(results)
n_cols = 2
n_rows = (n_games + 1) // 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten()

for idx, result in enumerate(results):
    ax = axes[idx]
    
    # Create box plot of differences
    sns.boxplot(y=result['differences'], ax=ax)
    
    # Add individual points
    sns.stripplot(y=result['differences'], ax=ax, color='black', alpha=0.5, size=4)
    
    # Add significance marker
    if result['p_value'] < 0.05:
        y_max = result['differences'].max()
        ax.text(0, y_max * 1.1, '*', ha='center', va='bottom', fontsize=20)
    
    # Add title with sample size
    significance = "Significant" if result['p_value'] < 0.05 else "Not Significant"
    ax.set_title(f"{result['game']}\n{significance} (n={result['n_pairs']}, p={result['p_value']:.3f})")
    
    # Add labels
    ax.set_ylabel('Score Difference\n(Harness - No Harness)')
    ax.set_xlabel('')
    ax.set_xticks([])  # Remove x-axis ticks since we only have one box

# Remove empty subplots if any
for idx in range(len(results), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('harness_differences_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# Print detailed results
print("\nDetailed Results:")
print("-" * 80)
for result in results:
    print(f"\n{result['game']}:")
    print(f"Number of valid pairs: {result['n_pairs']}")
    print(f"Mean difference: {result['mean_diff']:.2f}")
    print(f"95% Confidence Interval: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
    print(f"t-statistic: {result['t_stat']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Significant at p < 0.05: {'Yes' if result['p_value'] < 0.05 else 'No'}")
    print(f"Individual differences: {result['differences'].values}") 