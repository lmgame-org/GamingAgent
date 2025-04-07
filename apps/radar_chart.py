from leaderboard_utils import get_combined_leaderboard, GAME_ORDER
from data_visualization import create_general_radar_chart
import matplotlib.pyplot as plt
import json

# Load the rank data
with open('rank_data_03_25_2025.json', 'r') as f:
    rank_data = json.load(f)

# Get combined leaderboard with all games
selected_games = {game: True for game in GAME_ORDER}
df = get_combined_leaderboard(rank_data, selected_games)

# Define selected games list for examples
selected_games_list = ["Super Mario Bros","Sokoban", "2048", "Candy Crash"]

# # Example 1: Create radar chart for all models and all games
# fig = create_general_radar_chart(df)
# plt.savefig('ai_model_gaming_performance_radar_all.png', bbox_inches='tight', dpi=300)
# plt.show()

# # Example 2: Create radar chart highlighting Llama-4
# fig = create_general_radar_chart(df, highlight_model="Llama-4")
# plt.savefig('ai_model_gaming_performance_radar_highlight.png', bbox_inches='tight', dpi=300)
# plt.show()

# # Example 3: Create radar chart for specific games
# fig = create_general_radar_chart(df, selected_games=selected_games_list)
# plt.savefig('ai_model_gaming_performance_radar_selected.png', bbox_inches='tight', dpi=300)
# plt.show()

# Example 4: Create radar chart for specific games with highlighted model
fig = create_general_radar_chart(df, highlight_model="Llama-4", selected_games=selected_games_list)
plt.savefig('ai_model_gaming_performance_radar_highlight_selected.png', bbox_inches='tight', dpi=300)
plt.show() 