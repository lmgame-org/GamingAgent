import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
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

def create_group_bar_chart(df):
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
    
    # Group models by prefix, then sort alphabetically
    model_groups = {}
    for player in df["Player"].unique():
        prefix = player.split('-')[0]
        model_groups.setdefault(prefix, []).append(player)

    ordered_players = []
    for prefix in sorted(model_groups):
        ordered_players.extend(sorted(model_groups[prefix]))

    # Create one trace per player
    fig = go.Figure()
    for player in ordered_players:
        row = df[df["Player"] == player]
        if row.empty:
            continue
        row = row.iloc[0]

        y_vals = []
        has_data = False
        for game in sorted_games:
            col = f"norm_{game} Score"
            val = row.get(col, np.nan)
            if not np.isnan(val):
                has_data = True
            y_vals.append(val if not np.isnan(val) else 0)

        if not has_data:
            continue
            
        fig.add_trace(go.Bar(
            name=row["Player"],
            x=[game_display_map[game] for game in sorted_games],
            y=y_vals,
            marker_color=MODEL_COLORS.get(player, '#808080'),
            hovertemplate="<b>%{fullData.name}</b><br>Score: %{y:.1f}<extra></extra>"
        ))

    fig.update_layout(
        autosize=False,
        width=1000,
        height=800,
        margin=dict(l=200, r=200, t=20, b=20),
        title=dict(text="Grouped Bar Chart of AI Models (Consistent Trace Grouping)", pad=dict(t=10)),
        xaxis_title="Games",
        yaxis_title="Normalized Score",
        xaxis=dict(
            categoryorder='array',
            categoryarray=[game_display_map[g] for g in sorted_games],
            tickangle=0  # Keep text horizontal since we're using line breaks
        ),
        barmode='group',
        bargap=0.2,        # Gap between game categories
        bargroupgap=0.05,  # Gap between bars in a group
        uniformtext=dict(mode='hide', minsize=8),  # Hide text that doesn't fit
        legend=dict(
            font=dict(size=12),
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



def get_combined_leaderboard_with_group_bar(rank_data, selected_games):
    df = get_combined_leaderboard(rank_data, selected_games)
    # Create a copy for visualization to avoid modifying the original
    df_viz = df.copy()
    return df, create_group_bar_chart(df_viz)

def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'


def create_single_radar_chart(df, selected_games=None, highlight_models=None):
    if selected_games is None:
        selected_games = ['Super Mario Bros', '2048', 'Candy Crush', 'Sokoban', 'Ace Attorney']

    # Format game names
    formatted_games = []
    for game in selected_games:
        if game == 'Super Mario Bros (planning only)':
            formatted_games.append('Super Mario')  # Simplified name
        elif game == 'Tetris (planning only)':
            formatted_games.append('Tetris')
        else:
            formatted_games.append(game)  # Keep other names as is

    game_cols = [f"{game} Score" for game in selected_games]
    categories = formatted_games

    # Normalize
    for col in game_cols:
        vals = df[col].replace("n/a", 0).infer_objects(copy=False).astype(float)
        mean, std = vals.mean(), vals.std()
        df[f"norm_{col}"] = normalize_values(vals, mean, std)

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

    fig.update_layout(
        autosize=False,
        width=1000,
        height=700,  # Increased height to accommodate legend
        margin=dict(l=400, r=200, t=20, b=20),
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

def get_combined_leaderboard_with_single_radar(rank_data, selected_games, highlight_models=None):
    df = get_combined_leaderboard(rank_data, selected_games)
    selected_game_names = [g for g, sel in selected_games.items() if sel]
    # Create a copy for visualization to avoid modifying the original
    df_viz = df.copy()
    return df, create_single_radar_chart(df_viz, selected_game_names, highlight_models)

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


def save_visualization(fig, filename):
    fig.write_image(filename)