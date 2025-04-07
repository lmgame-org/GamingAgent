from leaderboard_utils import get_combined_leaderboard, GAME_ORDER
from data_visualization import create_group_bar_chart
import matplotlib.pyplot as plt
import json

# Load the rank data
with open('rank_data_03_25_2025.json', 'r') as f:
    rank_data = json.load(f)

# Get combined leaderboard with all games
selected_games = {game: True for game in GAME_ORDER}
df = get_combined_leaderboard(rank_data, selected_games)

# Create the group bar chart using the visualization function
# To highlight a specific model, pass its name (or part of its name) to highlight_model
# For example, to highlight Llama-4:
fig = create_group_bar_chart(df, highlight_model="Llama-4")

# Save the figure
plt.savefig('ai_model_gaming_performance_group.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show() 