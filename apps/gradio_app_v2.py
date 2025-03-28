import gradio as gr
import os
import pandas as pd
import json
from PIL import Image, ImageSequence
import io
from functools import reduce
import numpy as np

# Load the JSON file with rank data
with open("rank_data_03_25_2025.json", "r") as f:
    rank_data = json.load(f)

# Define game order
GAME_ORDER = [
    "Super Mario Bros",
    "Sokoban",
    "2048",
    "Candy Crash",
    "Tetris (complete)",
    "Tetris (planning only)"
]

#######################################################
# Helper functions to build individual game leaderboards
#######################################################

def get_mario_leaderboard():
    data = rank_data.get("Super Mario Bros", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "progress": "Progress (current/total)", 
        "score": "Score", 
        "time_s": "Time (s)"
    })
    return df

def get_sokoban_leaderboard():
    data = rank_data.get("Sokoban", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "levels_cracked": "Levels Cracked", 
        "steps": "Steps"
    })
    return df

def get_2048_leaderboard():
    data = rank_data.get("2048", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "score": "Score", 
        "steps": "Steps", 
        "time": "Time"
    })
    return df

def get_candy_leaderboard():
    data = rank_data.get("Candy Crash", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "score_runs": "Score Runs", 
        "average_score": "Average Score", 
        "steps": "Steps"
    })
    return df

def get_tetris_leaderboard():
    data = rank_data.get("Tetris (complete)", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "score": "Score", 
        "steps_blocks": "Steps"
    })
    return df

def get_tetris_planning_leaderboard():
    data = rank_data.get("Tetris (planning only)", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "score": "Score", 
        "steps_blocks": "Steps"
    })
    return df

#######################################################
# Combined leaderboard with ranking system
#######################################################

def calculate_rank_and_completeness(selected_games):
    # Dictionary to store DataFrames for each game
    game_dfs = {}
    
    # Get DataFrames for selected games
    if selected_games.get("Super Mario Bros"):
        game_dfs["Super Mario Bros"] = get_mario_leaderboard()
    if selected_games.get("Sokoban"):
        game_dfs["Sokoban"] = get_sokoban_leaderboard()
    if selected_games.get("2048"):
        game_dfs["2048"] = get_2048_leaderboard()
    if selected_games.get("Candy Crash"):
        game_dfs["Candy Crash"] = get_candy_leaderboard()
    if selected_games.get("Tetris (complete)"):
        game_dfs["Tetris (complete)"] = get_tetris_leaderboard()
    if selected_games.get("Tetris (planning only)"):
        game_dfs["Tetris (planning only)"] = get_tetris_planning_leaderboard()

    # Get all unique players
    all_players = set()
    for df in game_dfs.values():
        all_players.update(df["Player"].unique())
    all_players = sorted(list(all_players))

    # Create results DataFrame
    results = []
    for player in all_players:
        player_data = {"Player": player}
        ranks = []
        games_played = 0

        # Calculate rank and completeness for each game
        for game in GAME_ORDER:
            if game in game_dfs:
                df = game_dfs[game]
                if player in df["Player"].values:
                    games_played += 1
                    # Get player's score based on game type
                    if game == "Super Mario Bros":
                        player_score = df[df["Player"] == player]["Score"].iloc[0]
                        rank = len(df[df["Score"] > player_score]) + 1
                    elif game == "Sokoban":
                        # Parse Sokoban score string and get maximum level
                        levels_str = df[df["Player"] == player]["Levels Cracked"].iloc[0]
                        try:
                            # Split by semicolon, strip whitespace, filter empty strings, convert to integers
                            levels = [int(x.strip()) for x in levels_str.split(";") if x.strip()]
                            player_score = max(levels) if levels else 0
                        except:
                            player_score = 0
                        # Calculate rank based on maximum level
                        rank = len(df[df["Levels Cracked"].apply(
                            lambda x: max([int(y.strip()) for y in x.split(";") if y.strip()]) > player_score
                        )]) + 1
                    elif game == "2048":
                        player_score = df[df["Player"] == player]["Score"].iloc[0]
                        rank = len(df[df["Score"] > player_score]) + 1
                    elif game == "Candy Crash":
                        player_score = df[df["Player"] == player]["Average Score"].iloc[0]
                        rank = len(df[df["Average Score"] > player_score]) + 1
                    elif game == "Tetris (complete)":
                        player_score = df[df["Player"] == player]["Score"].iloc[0]
                        rank = len(df[df["Score"] > player_score]) + 1
                    elif game == "Tetris (planning only)":
                        player_score = df[df["Player"] == player]["Score"].iloc[0]
                        rank = len(df[df["Score"] > player_score]) + 1

                    ranks.append(rank)
                    player_data[f"{game} Score"] = player_score
                else:
                    player_data[f"{game} Score"] = "_"

        # Calculate average rank and completeness for sorting only
        if ranks:
            player_data["Sort Rank"] = round(np.mean(ranks), 2)
            player_data["Games Played"] = games_played
        else:
            player_data["Sort Rank"] = float('inf')
            player_data["Games Played"] = 0

        results.append(player_data)

    # Create DataFrame and sort by average rank and completeness
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        # Sort by average rank (ascending) and completeness (descending)
        df_results = df_results.sort_values(
            by=["Sort Rank", "Games Played"],
            ascending=[True, False]
        )
        # Drop the sorting columns
        df_results = df_results.drop(["Sort Rank", "Games Played"], axis=1)

    return df_results

def get_combined_leaderboard(selected_games):
    return calculate_rank_and_completeness(selected_games)

#######################################################
# Update function for Gradio checkboxes and leaderboard
#######################################################

def update_leaderboard(mario_overall, mario_details,
                       sokoban_overall, sokoban_details,
                       _2048_overall, _2048_details,
                       candy_overall, candy_details,
                       tetris_overall, tetris_details,
                       tetris_plan_overall, tetris_plan_details):
    # Check if any detailed checkbox is selected
    if any([mario_details, sokoban_details, _2048_details, candy_details, tetris_details, tetris_plan_details]):
        # Use priority order
        if mario_details:
            chosen = "Super Mario Bros"
            df = get_mario_leaderboard()
        elif sokoban_details:
            chosen = "Sokoban"
            df = get_sokoban_leaderboard()
        elif _2048_details:
            chosen = "2048"
            df = get_2048_leaderboard()
        elif candy_details:
            chosen = "Candy Crash"
            df = get_candy_leaderboard()
        elif tetris_details:
            chosen = "Tetris (complete)"
            df = get_tetris_leaderboard()
        elif tetris_plan_details:
            chosen = "Tetris (planning only)"
            df = get_tetris_planning_leaderboard()
        
        # When details view is selected:
        # - Set all overall checkboxes to False except the chosen game
        # - Keep only the chosen game's details checkbox True
        return (df,
                chosen=="Super Mario Bros", mario_details,
                chosen=="Sokoban", sokoban_details,
                chosen=="2048", _2048_details,
                chosen=="Candy Crash", candy_details,
                chosen=="Tetris (complete)", tetris_details,
                chosen=="Tetris (planning only)", tetris_plan_details)
    else:
        # Build dictionary for selected games
        selected_games = {
            "Super Mario Bros": mario_overall,
            "Sokoban": sokoban_overall,
            "2048": _2048_overall,
            "Candy Crash": candy_overall,
            "Tetris (complete)": tetris_overall,
            "Tetris (planning only)": tetris_plan_overall
        }
        df_combined = get_combined_leaderboard(selected_games)
        # Keep overall checkboxes as they are, set all details to False
        return (df_combined,
                mario_overall, False,
                sokoban_overall, False,
                _2048_overall, False,
                candy_overall, False,
                tetris_overall, False,
                tetris_plan_overall, False)

#######################################################
# Build Gradio App
#######################################################

def build_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Game Arena: Gaming Agent")
        gr.Markdown("### Leaderboard")

        with gr.Row():
            # For each game, we have two checkboxes: one for overall and one for detailed view.
            with gr.Column():
                gr.Markdown("**Super Mario Bros**")
                mario_overall = gr.Checkbox(label="Super Mario Bros Score", value=True)
                mario_details = gr.Checkbox(label="Super Mario Bros Details", value=False)
            with gr.Column():
                gr.Markdown("**Sokoban**")
                sokoban_overall = gr.Checkbox(label="Sokoban Score", value=True)
                sokoban_details = gr.Checkbox(label="Sokoban Details", value=False)
            with gr.Column():
                gr.Markdown("**2048**")
                _2048_overall = gr.Checkbox(label="2048 Score", value=True)
                _2048_details = gr.Checkbox(label="2048 Details", value=False)
            with gr.Column():
                gr.Markdown("**Candy Crash**")
                candy_overall = gr.Checkbox(label="Candy Crash Score", value=True)
                candy_details = gr.Checkbox(label="Candy Crash Details", value=False)
            with gr.Column():
                gr.Markdown("**Tetris (complete)**")
                tetris_overall = gr.Checkbox(label="Tetris (complete) Score", value=True)
                tetris_details = gr.Checkbox(label="Tetris (complete) Details", value=False)
            with gr.Column():
                gr.Markdown("**Tetris (planning)**")
                tetris_plan_overall = gr.Checkbox(label="Tetris (planning) Score", value=True)
                tetris_plan_details = gr.Checkbox(label="Tetris (planning) Details", value=False)

        # Leaderboard display with initial value
        leaderboard_board = gr.DataFrame(
            value=get_combined_leaderboard({
                "Super Mario Bros": True,
                "Sokoban": True,
                "2048": True,
                "Candy Crash": True,
                "Tetris (complete)": True,
                "Tetris (planning only)": True
            }),
            interactive=False
        )

        # List of all checkboxes (in order)
        checkbox_list = [mario_overall, mario_details,
                        sokoban_overall, sokoban_details,
                        _2048_overall, _2048_details,
                        candy_overall, candy_details,
                        tetris_overall, tetris_details,
                        tetris_plan_overall, tetris_plan_details]

        # When any checkbox changes, update the leaderboard and the checkbox states
        for checkbox in checkbox_list:
            checkbox.change(
                fn=update_leaderboard,
                inputs=checkbox_list,
                outputs=[leaderboard_board] + checkbox_list
            )

    return demo

if __name__ == "__main__":
    demo_app = build_app()
    demo_app.launch()
