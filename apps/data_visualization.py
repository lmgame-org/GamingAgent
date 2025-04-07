import matplotlib
matplotlib.use('Agg')  # Use Agg backend for thread safety
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import os
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

# Load model colors
with open('assets/model_color.json', 'r') as f:
    MODEL_COLORS = json.load(f)

# Define game score columns mapping
GAME_SCORE_COLUMNS = {
    "Super Mario Bros": "Score",
    "Sokoban": "Levels Cracked",
    "2048": "Score",
    "Candy Crash": "Average Score",
    "Tetris (complete)": "Score",
    "Tetris (planning only)": "Score"
}

def normalize_values(values, mean, std):
    """
    Normalize values using z-score and scale to 0-100 range
    
    Args:
        values (list): List of values to normalize
        mean (float): Mean value for normalization
        std (float): Standard deviation for normalization
        
    Returns:
        list: Normalized values scaled to 0-100 range
    """
    if std == 0:
        return [50 if v > 0 else 0 for v in values]  # Handle zero std case
    z_scores = [(v - mean) / std for v in values]
    # Scale z-scores to 0-100 range, with mean at 50
    scaled_values = [max(0, min(100, (z * 30) + 50)) for z in z_scores]
    return scaled_values

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
    fig, ax = plt.subplots(figsize=(20, 7))
    
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
    plt.subplots_adjust(left=0.3, top=0.85, bottom=0.3)
    
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
            
        # Get color for model from MODEL_COLORS, use default if not found
        model_name = df_sorted.iloc[i]['Player']
        color = MODEL_COLORS.get(model_name, '#808080')  # Default to gray if color not found
        bar.set_color(color)  # Set the bar color
        
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
                          bbox_to_anchor=(0.9, 1.1),
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

def create_group_bar_chart(df):
    """
    Create a grouped bar chart comparing AI model performance across different games
    
    Args:
        df (pd.DataFrame): DataFrame containing the combined leaderboard data
        
    Returns:
        matplotlib.figure.Figure: The generated group bar chart figure
    """
    # Close any existing figures to prevent memory leaks
    plt.close('all')
    
    # Create figure and axis with better styling
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(10, 7))
    
    # Create subplot with specific spacing
    ax = plt.subplot(111)
    
    # Adjust the subplot parameters
    plt.subplots_adjust(top=0.90,    # Add more space at the top
                       bottom=0.25,   # Increased from 0.15 to 0.25 to add more space at the bottom
                       right=0.70,   # Reduced from 0.75 to 0.70 to make more space for legend
                       left=0.1)     # Add space on the left

    # Get unique models
    models = df['Player'].unique()
    
    # Get active games (those that have score columns in the DataFrame)
    active_games = []
    for game in GAME_ORDER:
        score_col = f"{game} Score"  # Use the same column name for all games
        if score_col in df.columns:
            active_games.append(game)
    
    n_games = len(active_games)
    if n_games == 0:
        return fig  # Return empty figure if no games are selected

    # Keep track of which models have data in any game
    models_with_data = set()

    # Calculate normalized scores for each game
    for game_idx, game in enumerate(active_games):
        # Get all scores for this game
        game_scores = []
        
        # Use the same score column name for all games
        score_col = f"{game} Score"
            
        for model in models:
            try:
                score = df[df['Player'] == model][score_col].values[0]
                if score != '_' and float(score) > 0:  # Only include non-zero scores
                    game_scores.append((model, float(score)))
                    models_with_data.add(model)  # Add model to set if it has valid data
            except (IndexError, ValueError):
                continue
        
        if not game_scores:  # Skip if no valid scores for this game
            continue
            
        # Sort scores from highest to lowest
        game_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract sorted models and scores
        sorted_models = [x[0] for x in game_scores]
        scores = [x[1] for x in game_scores]
        
        # Calculate mean and std for normalization
        mean = np.mean(scores)
        std = np.std(scores)
        
        # Normalize scores
        normalized_scores = normalize_values(scores, mean, std)
        
        # Calculate bar width based on number of models in this game
        n_models_in_game = len(sorted_models)
        bar_width = 0.8 / n_models_in_game if n_models_in_game > 0 else 0.8
        
        # Plot bars for each model
        for i, (model, score) in enumerate(zip(sorted_models, normalized_scores)):
            # Only add to legend if first appearance and model has data
            should_label = model in models_with_data and model not in [l.get_text() for l in ax.get_legend().get_texts()] if ax.get_legend() else True
            
            # Get color from MODEL_COLORS, use a default if not found
            color = MODEL_COLORS.get(model, f"C{i % 10}")  # Use matplotlib default colors as fallback
            
            ax.bar(game_idx + i*bar_width, score, 
                  width=bar_width, 
                  label=model if should_label else "",
                  color=color,
                  alpha=0.8)

    # Customize the plot
    ax.set_xticks(np.arange(n_games))
    ax.set_xticklabels(active_games, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel('Normalized Performance Score', fontsize=12)
    ax.set_title('AI Model Performance Across Games', 
                 fontsize=14, pad=20, fontweight='bold')

    # Add grid lines
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Create legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Sort models by their first appearance in active games
    model_order = []
    for game in active_games:
        score_col = f"{game} Score"  # Use the same column name for all games
        for model in models:
            try:
                score = df[df['Player'] == model][score_col].values[0]
                if score != '_' and float(score) > 0 and model not in model_order:
                    model_order.append(model)
            except (IndexError, ValueError):
                continue
    
    # Create legend with sorted models
    sorted_handles = [by_label[model] for model in model_order if model in by_label]
    sorted_labels = [model for model in model_order if model in by_label]
    
    ax.legend(sorted_handles, sorted_labels, 
              bbox_to_anchor=(1.00, 1),
              loc='upper left',
              fontsize=9,
              title='AI Models',
              title_fontsize=10)  # Added bold font weight for model names

    # No need for tight_layout() as we're manually controlling the spacing
    
    return fig

def get_combined_leaderboard_with_group_bar(rank_data, selected_games):
    """
    Get combined leaderboard and create group bar chart
    
    Args:
        rank_data (dict): Dictionary containing rank data
        selected_games (dict): Dictionary of game names and their selection status
        
    Returns:
        tuple: (DataFrame, matplotlib.figure.Figure) containing the leaderboard data and group bar chart
    """
    df = get_combined_leaderboard(rank_data, selected_games)
    group_bar_fig = create_group_bar_chart(df)
    return df, group_bar_fig

def create_single_radar_chart(df, selected_games=None, highlight_models=None):
    """
    Create a single radar chart comparing AI model performance across selected games
    
    Args:
        df (pd.DataFrame): DataFrame containing the combined leaderboard data
        selected_games (list, optional): List of game names to include in the radar chart
        highlight_models (list, optional): List of model names to highlight in the chart
        
    Returns:
        matplotlib.figure.Figure: The generated radar chart figure
    """
    # Close any existing figures to prevent memory leaks
    plt.close('all')
    
    # Use provided selected_games or default to the four main games
    if selected_games is None:
        selected_games = ['Super Mario Bros', '2048', 'Candy Crash', 'Sokoban']
    
    game_columns = [f"{game} Score" for game in selected_games]
    categories = selected_games
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Compute number of variables
    num_vars = len(categories)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    # Set up the axes
    ax.set_xticks(angles[:-1])
    
    # Format categories with bold text
    formatted_categories = []
    for game in categories:
        if game == "Super Mario Bros":
            game = "Super\nMario"
        elif game == "Candy Crash":
            game = "Candy\nCrash"
        elif game == "Tetris (planning only)":
            game = "Tetris\n(planning)"
        elif game == "Tetris (complete)":
            game = "Tetris\n(complete)"
        formatted_categories.append(game)
    
    # Set bold labels for categories
    ax.set_xticklabels(formatted_categories, fontsize=10, fontweight='bold')
    
    # Draw grid lines
    ax.set_rgrids([20, 40, 60, 80, 100], 
                  labels=['20', '40', '60', '80', '100'],
                  angle=45,
                  fontsize=8)
    
    # Calculate game statistics for normalization
    def get_game_stats(df, game_col):
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
    
    game_stats = {col: get_game_stats(df, col) for col in game_columns}
    
    # Split the dataframe into highlighted and non-highlighted models
    if highlight_models:
        highlighted_df = df[df['Player'].isin(highlight_models)]
        non_highlighted_df = df[~df['Player'].isin(highlight_models)]
    else:
        highlighted_df = pd.DataFrame()
        non_highlighted_df = df
    
    # Plot non-highlighted models first
    for _, row in non_highlighted_df.iterrows():
        values = []
        for col in game_columns:
            val = row[col]
            if isinstance(val, str) and val == '_':
                values.append(0)
            else:
                try:
                    mean, std = game_stats[col]
                    if std == 0:
                        normalized = 50 if float(val) > 0 else 0
                    else:
                        z_score = (float(val) - mean) / std
                        normalized = max(0, min(100, (z_score * 30) + 50))
                    values.append(normalized)
                except:
                    values.append(0)
        
        # Complete the circular plot
        values = np.concatenate((values, [values[0]]))
        
        # Get color for model, use default if not found
        model_name = row['Player']
        color = MODEL_COLORS.get(model_name, '#808080')  # Default to gray if color not found
        
        # Plot with lines and markers
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    # Plot highlighted models last (so they appear on top)
    for _, row in highlighted_df.iterrows():
        values = []
        for col in game_columns:
            val = row[col]
            if isinstance(val, str) and val == '_':
                values.append(0)
            else:
                try:
                    mean, std = game_stats[col]
                    if std == 0:
                        normalized = 50 if float(val) > 0 else 0
                    else:
                        z_score = (float(val) - mean) / std
                        normalized = max(0, min(100, (z_score * 30) + 30))
                    values.append(normalized)
                except:
                    values.append(0)
        
        # Complete the circular plot
        values = np.concatenate((values, [values[0]]))
        
        # Plot with red color and thicker line
        model_name = row['Player']
        ax.plot(angles, values, 'o-', linewidth=6, label=model_name, color='red')
        ax.fill(angles, values, alpha=0.25, color='red')
    
    # Add title
    plt.title('AI Models Performance Across Games\n(Normalized Scores)',
              pad=20, fontsize=14, fontweight='bold')
    
    # Get handles and labels for legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Reorder legend to put highlighted models first
    if highlight_models:
        highlighted_handles = []
        highlighted_labels = []
        non_highlighted_handles = []
        non_highlighted_labels = []
        
        for handle, label in zip(handles, labels):
            if label in highlight_models:
                highlighted_handles.append(handle)
                highlighted_labels.append(label)
            else:
                non_highlighted_handles.append(handle)
                non_highlighted_labels.append(label)
        
        handles = highlighted_handles + non_highlighted_handles
        labels = highlighted_labels + non_highlighted_labels
    
    # Add legend with reordered handles and labels
    legend = plt.legend(handles, labels,
                       loc='center left',
                       bbox_to_anchor=(0.95, 1),
                       fontsize=8,
                       title='AI Models',
                       title_fontsize=10)  # Added bold font weight for model names
    
    # Adjust layout to prevent label cutoff
    plt.subplots_adjust(right=0.8)  # Added subplot adjustment to give more space on the right
    plt.tight_layout()
    
    return fig

def get_combined_leaderboard_with_single_radar(rank_data, selected_games, highlight_models=None):
    """
    Get combined leaderboard and create single radar chart
    
    Args:
        rank_data (dict): Dictionary containing rank data
        selected_games (dict): Dictionary of game names and their selection status
        highlight_models (list, optional): List of model names to highlight in the chart
        
    Returns:
        tuple: (DataFrame, matplotlib.figure.Figure) containing the leaderboard data and radar chart
    """
    df = get_combined_leaderboard(rank_data, selected_games)
    # Convert selected_games dict to list of selected game names
    selected_game_names = [game for game, selected in selected_games.items() if selected]
    radar_fig = create_single_radar_chart(df, selected_games=selected_game_names, highlight_models=highlight_models)
    return df, radar_fig

def save_visualization(fig, filename):
    """
    Save visualization to file
    """
    fig.savefig(filename, bbox_inches='tight', dpi=300)

