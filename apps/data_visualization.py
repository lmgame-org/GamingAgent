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

def normalize_values(values, mean, std):
    if std == 0:
        return [50 if v > 0 else 0 for v in values]
    z_scores = [(v - mean) / std for v in values]
    return [max(0, min(100, (z * 30) + 50)) for z in z_scores]

def simplify_model_name(name):
    parts = name.split('-')
    return '-'.join(parts[:4]) + '-...' if len(parts) > 4 else name

def create_horizontal_bar_chart(df, game_name):
    score_col = "Average Score" if game_name == "Candy Crash" else "Score"
    if game_name == "Sokoban":
        def get_max_level(levels_str):
            try:
                levels = [int(x.strip()) for x in levels_str.split(";") if x.strip()]
                return max(levels) if levels else 0
            except:
                return 0
        df['Max Level'] = df['Levels Cracked'].apply(get_max_level)
        score_col = "Max Level"

    df_sorted = df.sort_values(by=score_col, ascending=True)
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
        r = [row[f"norm_{c}"] for c in game_cols]
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
    return df, create_radar_charts(df)

def create_group_bar_chart(df):
    active_games = [g for g in GAME_ORDER if f"{g} Score" in df.columns]
    game_cols = [f"{g} Score" for g in active_games]

    for col in game_cols:
        vals = df[col].replace("_", 0).astype(float)
        mean, std = vals.mean(), vals.std()
        df[f"norm_{col}"] = normalize_values(vals, mean, std)

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            name=simplify_model_name(row["Player"]),
            x=active_games,
            y=[row[f"norm_{g} Score"] for g in active_games]
        ))

    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=80, r=150, t=40, b=200),
        title=dict(
            text="Grouped Bar Chart of AI Models",
            pad=dict(t=10)
        ),
        xaxis_title="Games",
        yaxis_title="Normalized Score",
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
    return df, create_group_bar_chart(df)

def create_single_radar_chart(df, selected_games, highlight_models=None):
    game_cols = [f"{g} Score" for g in selected_games]
    categories = selected_games

    for col in game_cols:
        vals = df[col].replace("_", 0).astype(float)
        mean, std = vals.mean(), vals.std()
        df[f"norm_{col}"] = normalize_values(vals, mean, std)

    fig = go.Figure()
    for _, row in df.iterrows():
        highlight = highlight_models and row["Player"] in highlight_models
        r = [row[f"norm_{col}"] for col in game_cols]
        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]],
            theta=categories + [categories[0]],
            mode='lines+markers',
            fill='toself',
            name=simplify_model_name(row["Player"]),
            line=dict(width=4 if highlight else 2),
            opacity=1.0 if highlight else 0.4
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
    return df, create_single_radar_chart(df, selected_game_names, highlight_models)

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
