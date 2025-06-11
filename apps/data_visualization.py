import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from leaderboard_utils import (
    get_combined_leaderboard,
    GAME_ORDER
)

# Load model colors
with open('assets/model_color.json', 'r') as f:
    MODEL_COLORS = json.load(f)

GAME_SCORE_COLUMNS = {
    "Super Mario Bros": "Score",
    "Sokoban": "Levels Cracked",
    "2048": "Score",
    "Candy Crush": "Average Score",
    "Tetris (complete)": "Score",
    "Tetris (planning only)": "Score",
    "Ace Attorney": "Score"
}
def get_model_prefix(name):
    return name.split('-')[0]


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
    scaled_values = [max(0, min(100, (z * 30) + 35)) for z in z_scores]
    return scaled_values
def simplify_model_name(name):
    if name == "claude-3-7-sonnet-20250219(thinking)":
        name ="claude-3-7-thinking"
    parts = name.split('-')
    return '-'.join(parts[:4]) + '-...' if len(parts) > 4 else name

def create_horizontal_bar_chart(df, game_name):
    """Creates a horizontal bar chart for a given game's leaderboard data."""
    
    if df is None or df.empty:
        # Return a placeholder or an empty figure if there's no data
        fig = go.Figure()
        fig.update_layout(
            title=f"No data available for {game_name}",
            xaxis_title="Score",
            yaxis_title="Player",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        return fig

    score_col = "Score" # Standardized score column name

    if score_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(title=f"'{score_col}' column not found for {game_name}")
        return fig

    # Ensure the score column is numeric for sorting and plotting
    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
    df_cleaned = df.dropna(subset=[score_col]) # Remove rows where score is NaN after conversion

    if df_cleaned.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No valid score data to plot for {game_name}")
        return fig

    # Sort values for chart display (lowest score at the top of the chart)
    # The input df is already sorted descending by score from leaderboard_utils
    # Re-sorting ascending=True here means player with lowest score is at the top of the y-axis categories
    df_sorted = df_cleaned.sort_values(by=score_col, ascending=True)

    fig = go.Figure(
        go.Bar(
            y=df_sorted['Player'], 
            x=df_sorted[score_col], 
            orientation='h',
            marker=dict(
                color=df_sorted[score_col],
                colorscale='Viridis', # Example colorscale, can be changed
                line=dict(color='#2c3e50', width=1)
            ),
            hovertext=df_sorted[score_col].round(2).astype(str) + ' points',
            hoverinfo='y+text'
        )
    )

    fig.update_layout(
        title=dict(
            text=f'{game_name} Scores',
            x=0.5,
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis_title="Score",
        yaxis_title="Player",
        plot_bgcolor='rgba(0,0,0,0)', # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)', # Transparent paper background
        font=dict(color='#2c3e50'), # Dark text for better readability on light backgrounds
        margin=dict(l=150, r=20, t=50, b=50), # Adjust margins for player names
        yaxis=dict(
            automargin=True, 
            tickfont=dict(size=10)
        ),
        xaxis=dict(gridcolor='#e0e0e0') # Light gridlines for x-axis
    )
    
    return fig

def create_radar_charts(df):
    game_cols = [c for c in df.columns if c.endswith(" Score")]
    categories = [c.replace(" Score", "") for c in game_cols]

    for col in game_cols:
        vals = df[col].replace("n/a", 0).astype(float)
        mean, std = vals.mean(), vals.std()
        df[f"norm_{col}"] = normalize_values(vals, mean, std)

    fig = go.Figure()
    for _, row in df.iterrows():
        player = row["Player"]
        r = [row[f"norm_{c}"] for c in game_cols]

        color = MODEL_COLORS.get(player, '#808080')  # fallback to gray
        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]],
            theta=categories + [categories[0]],
            mode='lines+markers',
            fill='toself',
            name=player,
            line=dict(color=color, width=2),
            marker=dict(color=color),
            fillcolor=color + '33',  # add transparency to fill (33 = ~20% opacity)
            opacity=0.8
        ))


    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=80, r=150, t=20, b=20),
        title=dict(
            text="Radar Chart of AI Performance (Normalized)",
            pad=dict(t=10)
        ),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        legend=dict(
            font=dict(size=9),
            itemsizing='trace',
            x=1.4,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    return fig

def get_combined_leaderboard_with_radar(rank_data, selected_games):
    df = get_combined_leaderboard(rank_data, selected_games)
    # Create a copy for visualization to avoid modifying the original
    df_viz = df.copy()
    return df, create_radar_charts(df_viz)

def create_group_bar_chart(df, top_n=10):
    game_cols = {}
    for game in GAME_ORDER:
        col = f"{game} Score"
        if col in df.columns:
            # Replace "n/a" with np.nan and handle downcasting properly
            df[col] = df[col].replace("n/a", np.nan).infer_objects(copy=False).astype(float)
            if df[col].notna().any():
                game_cols[game] = col

    if not game_cols:
        return go.Figure().update_layout(title="No data available")

    # Drop players with no data
    df = df.dropna(subset=game_cols.values(), how='all')

    # Normalize scores per game
    for game, col in game_cols.items():
        valid = df[col].dropna()
        norm_col = f"norm_{col}"
        if valid.empty:
            df[norm_col] = np.nan
        else:
            mean, std = valid.mean(), valid.std()
            normalized = normalize_values(valid, mean, std)
            df[norm_col] = np.nan
            df.loc[valid.index, norm_col] = normalized

    # Build consistent game order (X-axis)
    sorted_games = [game for game in GAME_ORDER if f"norm_{game} Score" in df.columns]
    
    # Format game names with line breaks
    formatted_games = []
    for game in sorted_games:
        if len(game) > 10 and ' ' in game:
            parts = game.split(' ')
            midpoint = len(parts) // 2
            formatted_name = ' '.join(parts[:midpoint]) + '<br>' + ' '.join(parts[midpoint:])
            formatted_games.append(formatted_name)
        else:
            formatted_games.append(game)
    
    # Create mapping from original to formatted names
    game_display_map = dict(zip(sorted_games, formatted_games))
    
    # For each game, get top performers and create combined x-axis categories
    fig = go.Figure()
    all_x_categories = []
    all_players = set()
    unique_x_labels = []
    
    # First pass: collect all players and create x-axis categories
    game_rankings = {}
    for game in sorted_games:
        col = f"norm_{game} Score"
        # Get valid scores for this game and sort by score (highest first)
        game_data = df[df[col].notna()].copy()
        game_data = game_data.sort_values(by=col, ascending=False)
        
        # Store rankings for this game (limit to top_n)
        game_rankings[game] = []
        for i, (_, row) in enumerate(game_data.iterrows()):
            if i >= top_n:  # Limit to top_n performers
                break
                
            player = row["Player"]
            score = row[col]
            rank = i + 1
            x_category = f"{game_display_map[game]}<br>#{rank}"
            game_rankings[game].append({
                'player': player,
                'score': score,
                'x_category': x_category,
                'rank': rank
            })
            all_x_categories.append(x_category)
            all_players.add(player)
            
            # Show label at the middle position based on number of models
            middle_position = (top_n + 1) // 2
            if rank == middle_position:
                # Special case for Super Mario Bros (planning only)
                if game == "Super Mario Bros":
                    unique_x_labels.append("SMB")
                else:
                    unique_x_labels.append(game_display_map[game])  # Show just game name without rank
            else:
                unique_x_labels.append("")  # Empty string for other ranks
    
    # Second pass: create traces for each player
    for player in sorted(all_players):
        x_vals = []
        y_vals = []
        
        for game in sorted_games:
            # Find this player's data for this game
            player_data = None
            for data in game_rankings[game]:
                if data['player'] == player:
                    player_data = data
                    break
            
            if player_data:
                x_vals.append(player_data['x_category'])
                y_vals.append(player_data['score'])
        
        if x_vals:  # Only add trace if player has data
            fig.add_trace(go.Bar(
                name=player,
                x=x_vals,
                y=y_vals,
                marker_color=MODEL_COLORS.get(player, '#808080'),
                hovertemplate="<b>%{fullData.name}</b><br>Score: %{y:.1f}<extra></extra>"
            ))

    fig.update_layout(
        autosize=True,
        height=550,
        margin=dict(l=50, r=50, t=20, b=20),
        title=dict(text=f"Grouped Bar Chart - Top {top_n} Performers by Game", pad=dict(t=10)),
        xaxis_title="Games (Ranked by Performance)",
        yaxis_title="Normalized Score",
        xaxis=dict(
            categoryorder='array',
            categoryarray=all_x_categories,
            tickangle=0,  # Keep text horizontal since we're using line breaks
            ticktext=unique_x_labels,  # Show labels only for first occurrence
            tickvals=all_x_categories
        ),
        barmode='group',
        bargap=0.2,        # Gap between game categories
        bargroupgap=0.05,  # Gap between bars in a group
        uniformtext=dict(mode='hide', minsize=8),  # Hide text that doesn't fit
        legend=dict(
            font=dict(size=12),
            title="Choose your model 💡 (click / double-click)",
            itemsizing='trace',
            x=1.1,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor='gray',
            borderwidth=1
        )
    )

    return fig



def get_combined_leaderboard_with_group_bar(rank_data, selected_games, top_n=10, limit_to_top_n=None):
    df = get_combined_leaderboard(rank_data, selected_games, limit_to_top_n)
    # Create a copy for visualization to avoid modifying the original
    df_viz = df.copy()
    return df, create_group_bar_chart(df_viz, top_n)

def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'


def create_single_radar_chart(df, selected_games=None, highlight_models=None, chart_title=None, top_n=None, full_df=None):
    if selected_games is None:
        selected_games = ['Super Mario Bros', '2048', 'Candy Crush', 'Sokoban', 'Ace Attorney']

    # Format game names
    formatted_games = []
    for game in selected_games:
        if game == 'Super Mario Bros':
            formatted_games.append('SMB')  # Clean name without planning only
        else:
            formatted_games.append(game)  # Keep other names as is

    game_cols = [f"{game} Score" for game in selected_games]
    categories = formatted_games

    # Use full dataset for normalization to keep consistent scale
    # If full_df is not provided, use the current df (fallback for backward compatibility)
    normalization_df = full_df if full_df is not None else df
    
    # Normalize using the full dataset but apply to the limited df
    for col in game_cols:
        # Get normalization parameters from full dataset
        full_vals = normalization_df[col].replace("n/a", 0).infer_objects(copy=False).astype(float)
        mean, std = full_vals.mean(), full_vals.std()
        
        # Apply normalization to the limited df
        limited_vals = df[col].replace("n/a", 0).infer_objects(copy=False).astype(float)
        df[f"norm_{col}"] = normalize_values(limited_vals, mean, std)

    # Group players by prefix and sort alphabetically
    model_groups = {}
    for player in df["Player"]:
        prefix = get_model_prefix(player)
        model_groups.setdefault(prefix, []).append(player)
    
    # Sort each group alphabetically
    for prefix in model_groups:
        model_groups[prefix] = sorted(model_groups[prefix], key=str.lower)
    
    # Get sorted prefixes and create ordered player list
    sorted_prefixes = sorted(model_groups.keys(), key=str.lower)
    grouped_players = []
    for prefix in sorted_prefixes:
        grouped_players.extend(model_groups[prefix])

    fig = go.Figure()

    for player in grouped_players:
        row = df[df["Player"] == player]
        if row.empty:
            continue
        row = row.iloc[0]

        is_highlighted = highlight_models and player in highlight_models
        color = 'red' if is_highlighted else MODEL_COLORS.get(player, '#808080')
        fillcolor = 'rgba(255, 0, 0, 0.4)' if is_highlighted else hex_to_rgba(color, 0.2)

        r = [row[f"norm_{col}"] for col in game_cols]

        # Convert player name to lowercase for the legend
        display_name = player.lower()

        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]],
            theta=categories + [categories[0]],
            mode='lines+markers',
            fill='toself',
            name=display_name,  # Use lowercase name in legend
            line=dict(color=color, width=6 if is_highlighted else 2),
            marker=dict(color=color, size=10 if is_highlighted else 6),
            fillcolor=fillcolor,
            opacity=1.0 if is_highlighted else 0.7,
            hovertemplate='<b>%{fullData.name}</b><br>Game: %{theta}<br>Score: %{r:.1f}<extra></extra>'
        ))

    # Dynamic title based on the data source and top_n
    if chart_title is None:
        if top_n is not None:
            chart_title = f"Radar Chart - Top {top_n} Performers by Game"
        else:
            # Fallback title
            if len(df) <= 10:
                chart_title = "🎮 Agent Performance Across Games"
            else:
                chart_title = "🤖 Model Performance Across Games"
    
    fig.update_layout(
        autosize=True,
        height=550,  # Reduced height for better proportion with legend
        margin=dict(l=400, r=100, t=20, b=20),
        title=dict(
            text=chart_title,
            x=0.5,
            xanchor='center',
            yanchor='top',
            y=0.95,
            font=dict(size=20),
            pad=dict(b=20)
        ),
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 100],
                tickangle=45,
                tickfont=dict(size=12),
                gridcolor='lightgray',
                gridwidth=1,
                angle=45
            ),
            angularaxis=dict(
                tickfont=dict(size=14, weight='bold'),
                tickangle=0
            )
        ),
        legend=dict(
            font=dict(size=12),
            title="Choose your model 💡 (click / double-click)",
            itemsizing='trace',
            x=-1.4,  # Moved further left
            y=0.8,     # Moved to top
            yanchor='top',
            xanchor='left',
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor='gray',
            borderwidth=1
        )
    )

    fig.update_layout(
        legend=dict(
            itemclick="toggleothers",  # This will make clicked item the only visible one
            itemdoubleclick="toggle"   # Double click toggles visibility
        )
    )

    return fig

def get_combined_leaderboard_with_single_radar(rank_data, selected_games, highlight_models=None, limit_to_top_n=None, chart_title=None, top_n=None):
    # Get full dataset for normalization
    full_df = get_combined_leaderboard(rank_data, selected_games, limit_to_top_n=None)
    
    # Get limited dataset for display
    df = get_combined_leaderboard(rank_data, selected_games, limit_to_top_n)
    
    selected_game_names = [g for g, sel in selected_games.items() if sel]
    
    # Create copies for visualization to avoid modifying the original
    df_viz = df.copy()
    full_df_viz = full_df.copy()
    
    return df, create_single_radar_chart(df_viz, selected_game_names, highlight_models, chart_title, top_n, full_df_viz)

def create_organization_radar_chart(rank_data):
    df = get_combined_leaderboard(rank_data, {g: True for g in GAME_ORDER})
    orgs = df["Organization"].unique()
    game_cols = [f"{g} Score" for g in GAME_ORDER if f"{g} Score" in df.columns]
    categories = [g.replace(" Score", "") for g in game_cols]

    avg_df = pd.DataFrame([
        {
            **{col: df[df["Organization"] == org][col].replace("n/a", 0).infer_objects(copy=False).astype(float).mean() for col in game_cols},
            "Organization": org
        }
        for org in orgs
    ])

    for col in game_cols:
        vals = avg_df[col]
        mean, std = vals.mean(), vals.std()
        avg_df[f"norm_{col}"] = normalize_values(vals, mean, std)

    fig = go.Figure()
    for _, row in avg_df.iterrows():
        r = [row[f"norm_{col}"] for col in game_cols]
        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]],
            theta=categories + [categories[0]],
            mode='lines+markers',
            fill='toself',
            name=row["Organization"]
        ))

    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=80, r=150, t=20, b=20),
        title=dict(
            text="Radar Chart: Organization Performance (Normalized)",
            pad=dict(t=10)
        ),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        legend=dict(
            font=dict(size=9),
            itemsizing='trace',
            x=1.4,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    return fig

def create_top_players_radar_chart(rank_data, n=5):
    df = get_combined_leaderboard(rank_data, {g: True for g in GAME_ORDER})
    top_players = df.head(n)["Player"].tolist()
    top_df = df[df["Player"].isin(top_players)]

    game_cols = [f"{g} Score" for g in GAME_ORDER if f"{g} Score" in df.columns]
    categories = [g.replace(" Score", "") for g in game_cols]

    for col in game_cols:
        # Replace "n/a" with 0 and handle downcasting properly
        vals = top_df[col].replace("n/a", 0).infer_objects(copy=False).astype(float)
        mean, std = vals.mean(), vals.std()
        top_df[f"norm_{col}"] = normalize_values(vals, mean, std)

    fig = go.Figure()
    for _, row in top_df.iterrows():
        r = [row[f"norm_{col}"] for col in game_cols]
        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]],
            theta=categories + [categories[0]],
            mode='lines+markers',
            fill='toself',
            name=row["Player"]
        ))

    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=80, r=150, t=20, b=20),
        title=dict(
            text=f"Top {n} Players Radar Chart (Normalized)",
            pad=dict(t=10)
        ),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        legend=dict(
            font=dict(size=9),
            itemsizing='trace',
            x=1.4,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    return fig

def create_player_radar_chart(rank_data, player_name):
    df = get_combined_leaderboard(rank_data, {g: True for g in GAME_ORDER})
    player_df = df[df["Player"] == player_name]

    if player_df.empty:
        return go.Figure().update_layout(
            title=dict(text="Player not found", pad=dict(t=10)),
            autosize=False,
            width=800,
            height=400
        )

    game_cols = [f"{g} Score" for g in GAME_ORDER if f"{g} Score" in df.columns]
    categories = [g.replace(" Score", "") for g in game_cols]

    for col in game_cols:
        # Replace "n/a" with 0 and handle downcasting properly
        vals = player_df[col].replace("n/a", 0).infer_objects(copy=False).astype(float)
        mean, std = df[col].replace("n/a", 0).infer_objects(copy=False).astype(float).mean(), df[col].replace("n/a", 0).infer_objects(copy=False).astype(float).std()
        player_df[f"norm_{col}"] = normalize_values(vals, mean, std)

    fig = go.Figure()
    for _, row in player_df.iterrows():
        r = [row[f"norm_{col}"] for col in game_cols]
        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]],
            theta=categories + [categories[0]],
            mode='lines+markers',
            fill='toself',
            name=row["Player"]
        ))

    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=80, r=150, t=20, b=20),
        title=dict(
            text=f"{row['Player']} Radar Chart (Normalized)",
            pad=dict(t=10)
        ),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        legend=dict(
            font=dict(size=9),
            itemsizing='trace',
            x=1.4,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    return fig

def save_normalized_data(df, selected_games, filename="normalized_data.json"):
    """
    Save normalized data to a JSON file for caching
    
    Args:
        df (pd.DataFrame): DataFrame with raw scores
        selected_games (dict): Dictionary of selected games
        filename (str): Output filename
    """
    game_cols = [f"{game} Score" for game in GAME_ORDER if f"{game} Score" in df.columns]
    
    # Calculate normalization parameters and normalized values
    normalization_data = {
        "timestamp": datetime.now().isoformat(),
        "selected_games": selected_games,
        "games": {},
        "players": {}
    }
    
    # Store normalization parameters per game
    for col in game_cols:
        game_name = col.replace(" Score", "")
        vals = df[col].replace("n/a", 0).infer_objects(copy=False).astype(float)
        mean, std = vals.mean(), vals.std()
        
        normalization_data["games"][game_name] = {
            "mean": mean,
            "std": std,
            "raw_scores": vals.to_dict()
        }
    
    # Store normalized scores per player
    for _, row in df.iterrows():
        player = row["Player"]
        player_data = {"organization": row.get("Organization", "unknown")}
        
        for col in game_cols:
            game_name = col.replace(" Score", "")
            raw_score = row[col]
            
            if raw_score != "n/a":
                raw_score = float(raw_score)
                mean = normalization_data["games"][game_name]["mean"]
                std = normalization_data["games"][game_name]["std"]
                normalized = normalize_values([raw_score], mean, std)[0]
            else:
                raw_score = "n/a"
                normalized = 0
            
            player_data[f"{game_name}_raw"] = raw_score
            player_data[f"{game_name}_normalized"] = normalized
        
        normalization_data["players"][player] = player_data
    
    # Save to file
    os.makedirs("cache", exist_ok=True)
    filepath = os.path.join("cache", filename)
    
    with open(filepath, 'w') as f:
        json.dump(normalization_data, f, indent=2)
    
    print(f"Normalized data saved to {filepath}")
    return filepath

def load_normalized_data(filename="normalized_data.json"):
    """
    Load normalized data from a JSON file
    
    Args:
        filename (str): Input filename
        
    Returns:
        dict: Normalized data or None if file doesn't exist
    """
    filepath = os.path.join("cache", filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"Normalized data loaded from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading normalized data: {e}")
        return None

def get_normalized_scores_from_cache(players, games, cache_data):
    """
    Extract normalized scores from cached data
    
    Args:
        players (list): List of player names
        games (list): List of game names
        cache_data (dict): Cached normalization data
        
    Returns:
        pd.DataFrame: DataFrame with normalized scores
    """
    data = []
    
    for player in players:
        if player in cache_data["players"]:
            player_data = {"Player": player}
            player_cache = cache_data["players"][player]
            
            for game in games:
                raw_key = f"{game}_raw"
                norm_key = f"{game}_normalized"
                
                if raw_key in player_cache:
                    player_data[f"{game} Score"] = player_cache[raw_key]
                    player_data[f"norm_{game} Score"] = player_cache[norm_key]
                else:
                    player_data[f"{game} Score"] = "n/a"
                    player_data[f"norm_{game} Score"] = 0
            
            data.append(player_data)
    
    return pd.DataFrame(data)

def save_visualization(fig, filename):
    fig.write_image(filename)

def generate_and_save_normalized_data(rank_data, filename="normalized_data.json"):
    """
    Generate normalized data for all games and save to file
    
    Args:
        rank_data (dict): Raw rank data
        filename (str): Output filename
        
    Returns:
        str: Path to saved file
    """
    # Select all games
    all_games = {game: True for game in GAME_ORDER}
    
    # Get combined leaderboard
    df = get_combined_leaderboard(rank_data, all_games)
    
    # Save normalized data
    return save_normalized_data(df, all_games, filename)

def create_single_radar_chart_with_cache(df, selected_games=None, highlight_models=None, use_cache=True, cache_filename="normalized_data.json"):
    """
    Create radar chart with optional caching support
    """
    if selected_games is None:
        selected_games = ['Super Mario Bros', '2048', 'Candy Crush', 'Sokoban', 'Ace Attorney']

    # Try to load from cache first
    cached_data = None
    if use_cache:
        cached_data = load_normalized_data(cache_filename)
    
    if cached_data:
        # Use cached normalized data
        players = df["Player"].tolist()
        df_normalized = get_normalized_scores_from_cache(players, selected_games, cached_data)
        # Merge with original df to get Organization info
        df_normalized = df_normalized.merge(df[["Player", "Organization"]], on="Player", how="left")
    else:
        # Fall back to on-the-fly normalization
        df_normalized = df.copy()
        game_cols = [f"{game} Score" for game in selected_games]
        
        # Normalize
        for col in game_cols:
            vals = df_normalized[col].replace("n/a", 0).infer_objects(copy=False).astype(float)
            mean, std = vals.mean(), vals.std()
            df_normalized[f"norm_{col}"] = normalize_values(vals, mean, std)

    # Format game names
    formatted_games = []
    for game in selected_games:
        if game == 'Super Mario Bros':
            formatted_games.append('SMB')
        else:
            formatted_games.append(game)

    categories = formatted_games

    # Group players by prefix and sort alphabetically
    model_groups = {}
    for player in df_normalized["Player"]:
        prefix = get_model_prefix(player)
        model_groups.setdefault(prefix, []).append(player)
    
    # Sort each group alphabetically
    for prefix in model_groups:
        model_groups[prefix] = sorted(model_groups[prefix], key=str.lower)
    
    # Get sorted prefixes and create ordered player list
    sorted_prefixes = sorted(model_groups.keys(), key=str.lower)
    grouped_players = []
    for prefix in sorted_prefixes:
        grouped_players.extend(model_groups[prefix])

    fig = go.Figure()

    for player in grouped_players:
        row = df_normalized[df_normalized["Player"] == player]
        if row.empty:
            continue
        row = row.iloc[0]

        is_highlighted = highlight_models and player in highlight_models
        color = 'red' if is_highlighted else MODEL_COLORS.get(player, '#808080')
        fillcolor = 'rgba(255, 0, 0, 0.4)' if is_highlighted else hex_to_rgba(color, 0.2)

        # Get normalized values
        if cached_data:
            r = [row[f"norm_{game} Score"] for game in selected_games]
        else:
            r = [row[f"norm_{game} Score"] for game in selected_games]

        display_name = player.lower()

        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]],
            theta=categories + [categories[0]],
            mode='lines+markers',
            fill='toself',
            name=display_name,
            line=dict(color=color, width=6 if is_highlighted else 2),
            marker=dict(color=color, size=10 if is_highlighted else 6),
            fillcolor=fillcolor,
            opacity=1.0 if is_highlighted else 0.7,
            hovertemplate='<b>%{fullData.name}</b><br>Game: %{theta}<br>Score: %{r:.1f}<extra></extra>'
        ))

    fig.update_layout(
        autosize=True,
        height=550,
        margin=dict(l=400, r=100, t=20, b=20),
        title=dict(
            text="AI Normalized Performance Across Games",
            x=0.5,
            xanchor='center',
            yanchor='top',
            y=0.95,
            font=dict(size=20),
            pad=dict(b=20)
        ),
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 100],
                tickangle=45,
                tickfont=dict(size=12),
                gridcolor='lightgray',
                gridwidth=1,
                angle=45
            ),
            angularaxis=dict(
                tickfont=dict(size=14, weight='bold'),
                tickangle=0
            )
        ),
        legend=dict(
            font=dict(size=12),
            title="Choose your model 💡 (click / double-click)",
            itemsizing='trace',
            x=-1.4,
            y=0.8,
            yanchor='top',
            xanchor='left',
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor='gray',
            borderwidth=1,
            itemclick="toggleothers",
            itemdoubleclick="toggle"
        )
    )

    return fig