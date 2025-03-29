import matplotlib
matplotlib.use('Agg')  # Use Agg backend for thread safety
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from leaderboard_utils import (
    get_organization,
    get_mario_leaderboard,
    get_sokoban_leaderboard,
    get_2048_leaderboard,
    get_candy_leaderboard,
    get_tetris_leaderboard,
    get_tetris_planning_leaderboard,
    get_combined_leaderboard,
    GAME_ORDER
)

# Define game score columns mapping
GAME_SCORE_COLUMNS = {
    "Super Mario Bros": "Score",
    "Sokoban": "Levels Cracked",
    "2048": "Score",
    "Candy Crash": "Average Score",
    "Tetris (complete)": "Score",
    "Tetris (planning only)": "Score"
}

def simplify_model_name(model_name):
    """
    Simplify model name by either taking first 11 chars or string before third '-'
    """
    hyphen_parts = model_name.split('-')
    return '-'.join(hyphen_parts[:3]) if len(hyphen_parts) >= 3 else model_name[:11]

def create_horizontal_bar_chart(df, game_name):
    """
    Create horizontal bar chart for detailed game view
    
    Args:
        df (pd.DataFrame): DataFrame containing game data
        game_name (str): Name of the game to display
        
    Returns:
        matplotlib.figure.Figure: The generated bar chart figure
    """
    # Close any existing figures to prevent memory leaks
    plt.close('all')
    
    # Set style
    plt.style.use('default')
    # Increase figure width to accommodate long model names
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by score
    if game_name == "Super Mario Bros":
        score_col = "Score"
        df_sorted = df.sort_values(by=score_col, ascending=True)
    elif game_name == "Sokoban":
        # Process Sokoban scores by splitting and getting max level
        def get_max_level(levels_str):
            try:
                # Split by semicolon, strip whitespace, filter empty strings, convert to integers
                levels = [int(x.strip()) for x in levels_str.split(";") if x.strip()]
                return max(levels) if levels else 0
            except:
                return 0
        
        # Create a temporary column with max levels
        df['Max Level'] = df['Levels Cracked'].apply(get_max_level)
        df_sorted = df.sort_values(by='Max Level', ascending=True)
        score_col = 'Max Level'
    elif game_name == "2048":
        score_col = "Score"
        df_sorted = df.sort_values(by=score_col, ascending=True)
    elif game_name == "Candy Crash":
        score_col = "Average Score"
        df_sorted = df.sort_values(by=score_col, ascending=True)
    elif game_name in ["Tetris (complete)", "Tetris (planning only)"]:
        score_col = "Score"
        df_sorted = df.sort_values(by=score_col, ascending=True)
    else:
        return None
    
    # Create color gradient
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_sorted)))
    
    # Create horizontal bars
    bars = ax.barh(range(len(df_sorted)), df_sorted[score_col], color=colors)
    
    # Add more space for labels on the left
    plt.subplots_adjust(left=0.3)
    
    # Customize the chart
    ax.set_yticks(range(len(df_sorted)))
    
    # Format player names: keep organization info and truncate the rest if too long
    def format_player_name(player, org):
        max_length = 40  # Maximum length for player name
        if len(player) > max_length:
            # Keep the first part and last part of the name
            parts = player.split('-')
            if len(parts) > 3:
                formatted = f"{parts[0]}-{parts[1]}-...{parts[-1]}"
            else:
                formatted = player[:max_length-3] + "..."
        else:
            formatted = player
        return f"{formatted} [{org}]"
    
    player_labels = [format_player_name(row['Player'], row['Organization']) 
                    for _, row in df_sorted.iterrows()]
    ax.set_yticklabels(player_labels, fontsize=9)
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if game_name == "Candy Crash":
            score_text = f'{width:.1f}'
        else:
            score_text = f'{width:.0f}'
            
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                score_text,
                ha='left', va='center',
                fontsize=10,
                fontweight='bold',
                color='white',
                bbox=dict(facecolor=(0, 0, 0, 0.3),
                         edgecolor='none', 
                         alpha=0.5,
                         pad=2))
    
    # Set title and labels
    ax.set_title(f"{game_name} Performance", 
                 pad=20, 
                 fontsize=14, 
                 fontweight='bold',
                 color='#2c3e50')
    
    if game_name == "Sokoban":
        ax.set_xlabel("Maximum Level Reached", 
                     fontsize=12, 
                     fontweight='bold',
                     color='#2c3e50',
                     labelpad=10)
    else:
        ax.set_xlabel(score_col, 
                     fontsize=12, 
                     fontweight='bold',
                     color='#2c3e50',
                     labelpad=10)
    
    # Add grid lines
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_radar_charts(df):
    """
    Create two radar charts with improved normalization using z-scores
    """
    # Close any existing figures to prevent memory leaks
    plt.close('all')
    
    # Define reasoning models
    reasoning_models = [
        'claude-3-7-sonnet-20250219(thinking)',
        'o1-2024-12-17',
        'gemini-2.0-flash-thinking-exp-1219',
        'o3-mini-2025-01-31(medium)',
        'gemini-2.5-pro-exp-03-25',
        'o1-mini-2024-09-12',
        'deepseek-r1'
    ]
    
    # Split dataframe into reasoning and non-reasoning models
    df_reasoning = df[df['Player'].isin(reasoning_models)]
    df_others = df[~df['Player'].isin(reasoning_models)]
    
    # Get game columns
    game_columns = [col for col in df.columns if col.endswith(' Score')]
    categories = [col.replace(' Score', '') for col in game_columns]
    
    # Create figure with two subplots - adjusted size for new layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('white')  # Set figure background to white
    
    def normalize_values(values, mean, std):
        """
        Normalize values using z-score and scale to 0-100 range
        """
        if std == 0:
            return [50 if v > 0 else 0 for v in values]  # Handle zero std case
        z_scores = [(v - mean) / std for v in values]
        # Scale z-scores to 0-100 range, with mean at 50
        scaled_values = [max(0, min(100, (z * 30) + 30)) for z in z_scores]
        return scaled_values

    def get_game_stats(df, game_col):
        """
        Get mean and std for a game column, handling missing values
        """
        values = []
        for val in df[game_col]:
            if isinstance(val, str) and val == '_':
                values.append(0)
            else:
                try:
                    values.append(float(val))
                except:
                    values.append(0)
        return np.mean(values), np.std(values)

    def setup_radar_plot(ax, data, title):
        ax.set_facecolor('white')  # Set subplot background to white
        
        num_vars = len(categories)
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        # Plot grid lines with darker color
        grid_values = [10, 30, 50, 70, 90]
        ax.set_rgrids(grid_values, 
                    labels=grid_values,
                    angle=45, 
                    fontsize=6, 
                    alpha=0.7,  # Increased alpha for better visibility
                    color='#404040')  # Darker color for grid labels
    
        # Make grid lines darker but still subtle
        ax.grid(True, color='#404040', alpha=0.3)  # Darker grid lines
        
        # Define darker, more vibrant colors for the radar plots
        colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
        
        # Calculate game statistics once
        game_stats = {col: get_game_stats(df, col) for col in game_columns}
        
        # Plot data with darker lines and higher opacity for fills
        for idx, (_, row) in enumerate(data.iterrows()):
            values = []
            for col in game_columns:
                val = row[col]
                if isinstance(val, str) and val == '_':
                    values.append(0)
                else:
                    try:
                        values.append(float(val))
                    except:
                        values.append(0)
            
            # Normalize values using game statistics
            normalized_values = []
            for i, v in enumerate(values):
                mean, std = game_stats[game_columns[i]]
                normalized_value = normalize_values([v], mean, std)[0]
                normalized_values.append(normalized_value)
            
            # Complete the circular plot
            normalized_values = np.concatenate((normalized_values, [normalized_values[0]]))
            
            model_name = simplify_model_name(row['Player'])
            ax.plot(angles, normalized_values, 'o-', linewidth=2.0,  # Increased line width
                   label=model_name,
                   color=colors[idx % len(colors)], 
                   markersize=4)  # Increased marker size
            ax.fill(angles, normalized_values, 
                   alpha=0.3,  # Increased fill opacity
                   color=colors[idx % len(colors)])
        
        # Format categories
        formatted_categories = []
        for game in categories:
            if game == "Tetris (planning only)":
                game = "Tetris\n(planning)"
            elif game == "Tetris (complete)":
                game = "Tetris\n(complete)"
            elif game == "Super Mario Bros":
                game = "Super\nMario"
            elif game == "Candy Crash":
                game = "Candy\nCrash"
            formatted_categories.append(game)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(formatted_categories, 
                          fontsize=8,  # Slightly larger font
                          color='#202020',  # Darker text
                          fontweight='bold')  # Bold text
        ax.tick_params(pad=10, colors='#202020')  # Darker tick colors
        
        ax.set_title(title, 
                    pad=20, 
                    fontsize=11,  # Slightly larger title
                    color='#202020',  # Darker title
                    fontweight='bold')  # Bold title
        
        legend = ax.legend(loc='upper right',
                          bbox_to_anchor=(1.3, 1.1),
                          fontsize=7,  # Slightly larger legend
                          framealpha=0.9,  # More opaque legend
                          edgecolor='#404040',  # Darker edge
                          ncol=1)
        
        ax.set_ylim(0, 105)
        ax.spines['polar'].set_color('#404040')  # Darker spine
        ax.spines['polar'].set_alpha(0.5)  # More visible spine
    
    # Setup both plots
    setup_radar_plot(ax1, df_reasoning, "Reasoning Models")
    setup_radar_plot(ax2, df_others, "Non-Reasoning Models")
    
    plt.subplots_adjust(right=0.85, wspace=0.3)
    
    return fig

def get_combined_leaderboard_with_radar(rank_data, selected_games):
    """
    Get combined leaderboard and create radar charts
    """
    df = get_combined_leaderboard(rank_data, selected_games)
    radar_fig = create_radar_charts(df)
    return df, radar_fig

def create_organization_radar_chart(rank_data):
    """
    Create radar chart comparing organizations
    """
    # Get combined leaderboard with all games
    df = get_combined_leaderboard(rank_data, {game: True for game in GAME_ORDER})
    
    # Group by organization and calculate average scores
    org_performance = {}
    for org in df["Organization"].unique():
        org_df = df[df["Organization"] == org]
        scores = {}
        for game in GAME_ORDER:
            game_scores = org_df[f"{game} Score"].apply(lambda x: float(x) if x != "_" else 0)
            scores[game] = game_scores.mean()
        org_performance[org] = scores
    
    # Create radar chart
    return create_radar_charts(pd.DataFrame([org_performance]))

def create_top_players_radar_chart(rank_data, n=5):
    """
    Create radar chart for top N players
    """
    # Get combined leaderboard with all games
    df = get_combined_leaderboard(rank_data, {game: True for game in GAME_ORDER})
    
    # Get top N players
    top_players = df["Player"].head(n).tolist()
    
    # Create radar chart for top players
    return create_radar_charts(df[df["Player"].isin(top_players)])

def create_player_radar_chart(rank_data, player_name):
    """
    Create radar chart for a specific player
    """
    # Get combined leaderboard with all games
    df = get_combined_leaderboard(rank_data, {game: True for game in GAME_ORDER})
    
    # Get player's data
    player_df = df[df["Player"] == player_name]
    
    if player_df.empty:
        return None
    
    # Create radar chart for the player
    return create_radar_charts(player_df)

def save_visualization(fig, filename):
    """
    Save visualization to file
    """
    fig.savefig(filename, bbox_inches='tight', dpi=300)
