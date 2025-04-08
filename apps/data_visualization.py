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
    "Candy Crash": "Average Score",
    "Tetris (complete)": "Score",
    "Tetris (planning only)": "Score"
}
def get_model_prefix(name):
    return name.split('-')[0]


def normalize_values(values, mean=None, std=None):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [100 for _ in values]  # or 50
    return [(v - min_val) / (max_val - min_val) * 100 for v in values]

def simplify_model_name(name):
    if name == "claude-3-7-sonnet-20250219(thinking)":
        name ="claude-3-7-thinking"
    parts = name.split('-')
    return '-'.join(parts[:4]) + '-...' if len(parts) > 4 else name

def create_horizontal_bar_chart(df, game_name):


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



    x = df_sorted[score_col]
    y = [f"{simplify_model_name(row['Player'])} [{row['Organization']}]" for _, row in df_sorted.iterrows()]
    colors = [MODEL_COLORS.get(row['Player'], '#808080') for _, row in df_sorted.iterrows()]
    texts = [f"{v:.1f}" if game_name == "Candy Crash" else f"{int(v)}" for v in x]

    fig = go.Figure(go.Bar(
        x=x,
        y=y,
        orientation='h',
        marker_color=colors,
        text=texts,
        textposition='auto',
        hovertemplate='%{y}<br>Score: %{x}<extra></extra>'
    ))

    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=150, r=150, t=40, b=200),
        title=dict(
            text=f"{game_name} Performance",
            pad=dict(t=10)
        ),
        yaxis=dict(automargin=True),
        legend=dict(
            font=dict(size=9),
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

def create_radar_charts(df):
    game_cols = [c for c in df.columns if c.endswith(" Score")]
    categories = [c.replace(" Score", "") for c in game_cols]

    for col in game_cols:
        vals = df[col].replace("_", 0).astype(float)
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
        margin=dict(l=80, r=150, t=40, b=100),
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
            df[col] = df[col].replace("_", np.nan).astype(float)
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
            name=simplify_model_name(player),
            x=sorted_games,
            y=y_vals,
            marker_color=MODEL_COLORS.get(player, '#808080'),
            hovertemplate="%{x}<br>%{y:.1f}<extra></extra>"
        ))

    fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        margin=dict(l=80, r=150, t=40, b=200),
        title=dict(text="Grouped Bar Chart of AI Models (Consistent Trace Grouping)", pad=dict(t=10)),
        xaxis_title="Games",
        yaxis_title="Normalized Score",
        xaxis=dict(
            categoryorder='array',
            categoryarray=sorted_games
        ),
        barmode='group',
        legend=dict(
            font=dict(size=9),
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
        selected_games = ['Super Mario Bros', '2048', 'Candy Crash', 'Sokoban']

    game_cols = [f"{game} Score" for game in selected_games]
    categories = selected_games
    
    # Normalize
    for col in game_cols:
        vals = df[col].replace("_", 0).astype(float)
        mean, std = vals.mean(), vals.std()
        df[f"norm_{col}"] = normalize_values(vals, mean, std)

    # Group players by prefix
    model_groups = {}
    for player in df["Player"]:
        prefix = get_model_prefix(player)
        model_groups.setdefault(prefix, []).append(player)

    # Order: grouped by prefix, then alphabetically
    grouped_players = []
    for prefix in sorted(model_groups):
        grouped_players.extend(sorted(model_groups[prefix]))

    fig = go.Figure()

    for player in grouped_players:
        row = df[df["Player"] == player]
        if row.empty:
            continue
        row = row.iloc[0]

        is_highlighted = highlight_models and player in highlight_models
        color = 'red' if is_highlighted else MODEL_COLORS.get(player, '#808080')
        fillcolor = 'rgba(255, 0, 0, 0.3)' if is_highlighted else hex_to_rgba(color, 0.2)

        r = [row[f"norm_{col}"] for col in game_cols]

        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]],
            theta=categories + [categories[0]],
            mode='lines+markers',
            fill='toself',
            name=simplify_model_name(row["Player"]),
            line=dict(color=color, width=4 if is_highlighted else 2),
            marker=dict(color=color),
            fillcolor=fillcolor,
            opacity=1.0 if is_highlighted else 0.7
        ))

    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=80, r=150, t=40, b=100),
        title=dict(
            text="Single Radar Chart (Normalized Performance)",
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
            **{col: df[df["Organization"] == org][col].replace("_", 0).astype(float).mean() for col in game_cols},
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
        margin=dict(l=80, r=150, t=40, b=200),
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
        vals = top_df[col].replace("_", 0).astype(float)
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
            name=simplify_model_name(row["Player"])
        ))

    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=80, r=150, t=40, b=200),
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
        vals = player_df[col].replace("_", 0).astype(float)
        mean, std = df[col].replace("_", 0).astype(float).mean(), df[col].replace("_", 0).astype(float).std()
        player_df[f"norm_{col}"] = normalize_values(vals, mean, std)

    fig = go.Figure()
    for _, row in player_df.iterrows():
        r = [row[f"norm_{col}"] for col in game_cols]
        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]],
            theta=categories + [categories[0]],
            mode='lines+markers',
            fill='toself',
            name=simplify_model_name(row["Player"])
        ))

    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=80, r=150, t=40, b=200),
        title=dict(
            text=f"{simplify_model_name(player_name)} Radar Chart (Normalized)",
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
