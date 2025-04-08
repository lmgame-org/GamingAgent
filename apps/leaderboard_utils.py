import pandas as pd
import json
import numpy as np

# Define game order
GAME_ORDER = [
    "Super Mario Bros",
    "Sokoban",
    "2048",
    "Candy Crash",
    "Tetris (complete)",
    "Tetris (planning only)"
]

def get_organization(model_name):
    m = model_name.lower()
    if "claude" in m:
        return "anthropic"
    elif "gemini" in m:
        return "google"
    elif "o1" in m or "gpt" in m or "o3" in m:
        return "openai"
    elif "deepseek" in m:
        return "deepseek"
    elif "llama" in m:
        return "meta"
    else:
        return "unknown"

def get_mario_leaderboard(rank_data):
    data = rank_data.get("Super Mario Bros", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "progress": "Progress (current/total)", 
        "score": "Score", 
        "time_s": "Time (s)"
    })
    df["Organization"] = df["Player"].apply(get_organization)
    df = df[["Player", "Organization", "Progress (current/total)", "Score", "Time (s)"]]
    return df

def get_sokoban_leaderboard(rank_data):
    data = rank_data.get("Sokoban", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "levels_cracked": "Levels Cracked", 
        "steps": "Steps"
    })
    df["Organization"] = df["Player"].apply(get_organization)
    df = df[["Player", "Organization", "Levels Cracked", "Steps"]]
    return df

def get_2048_leaderboard(rank_data):
    data = rank_data.get("2048", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "score": "Score", 
        "steps": "Steps", 
        "time": "Time"
    })
    df["Organization"] = df["Player"].apply(get_organization)
    df = df[["Player", "Organization", "Score", "Steps", "Time"]]
    return df

def get_candy_leaderboard(rank_data):
    data = rank_data.get("Candy Crash", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "score_runs": "Score Runs", 
        "average_score": "Average Score", 
        "steps": "Steps"
    })
    df["Organization"] = df["Player"].apply(get_organization)
    df = df[["Player", "Organization", "Score Runs", "Average Score", "Steps"]]
    return df

def get_tetris_leaderboard(rank_data):
    data = rank_data.get("Tetris (complete)", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "score": "Score", 
        "steps_blocks": "Steps"
    })
    df["Organization"] = df["Player"].apply(get_organization)
    df = df[["Player", "Organization", "Score", "Steps"]]
    return df

def get_tetris_planning_leaderboard(rank_data):
    data = rank_data.get("Tetris (planning only)", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "score": "Score", 
        "steps_blocks": "Steps"
    })
    df["Organization"] = df["Player"].apply(get_organization)
    df = df[["Player", "Organization", "Score", "Steps"]]
    return df

def calculate_rank_and_completeness(rank_data, selected_games):
    # Dictionary to store DataFrames for each game
    game_dfs = {}
    
    # Get DataFrames for selected games
    if selected_games.get("Super Mario Bros"):
        game_dfs["Super Mario Bros"] = get_mario_leaderboard(rank_data)
    if selected_games.get("Sokoban"):
        game_dfs["Sokoban"] = get_sokoban_leaderboard(rank_data)
    if selected_games.get("2048"):
        game_dfs["2048"] = get_2048_leaderboard(rank_data)
    if selected_games.get("Candy Crash"):
        game_dfs["Candy Crash"] = get_candy_leaderboard(rank_data)
    if selected_games.get("Tetris (complete)"):
        game_dfs["Tetris (complete)"] = get_tetris_leaderboard(rank_data)
    if selected_games.get("Tetris (planning only)"):
        game_dfs["Tetris (planning only)"] = get_tetris_planning_leaderboard(rank_data)

    # Get all unique players
    all_players = set()
    for df in game_dfs.values():
        all_players.update(df["Player"].unique())
    all_players = sorted(list(all_players))

    # Create results DataFrame
    results = []
    for player in all_players:
        player_data = {
            "Player": player,
            "Organization": get_organization(player)
        }
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
                    player_data[f"{game} Score"] = -1

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

def get_combined_leaderboard(rank_data, selected_games):
    """
    Get combined leaderboard for selected games
    
    Args:
        rank_data (dict): Dictionary containing rank data
        selected_games (dict): Dictionary of game names and their selection status
        
    Returns:
        pd.DataFrame: Combined leaderboard DataFrame
    """
    # Dictionary to store DataFrames for each game
    game_dfs = {}
    
    # Get DataFrames for selected games
    if selected_games.get("Super Mario Bros"):
        game_dfs["Super Mario Bros"] = get_mario_leaderboard(rank_data)
    if selected_games.get("Sokoban"):
        game_dfs["Sokoban"] = get_sokoban_leaderboard(rank_data)
    if selected_games.get("2048"):
        game_dfs["2048"] = get_2048_leaderboard(rank_data)
    if selected_games.get("Candy Crash"):
        game_dfs["Candy Crash"] = get_candy_leaderboard(rank_data)
    if selected_games.get("Tetris (complete)"):
        game_dfs["Tetris (complete)"] = get_tetris_leaderboard(rank_data)
    if selected_games.get("Tetris (planning only)"):
        game_dfs["Tetris (planning only)"] = get_tetris_planning_leaderboard(rank_data)

    # Get all unique players
    all_players = set()
    for df in game_dfs.values():
        all_players.update(df["Player"].unique())
    all_players = sorted(list(all_players))

    # Create results DataFrame
    results = []
    for player in all_players:
        player_data = {
            "Player": player,
            "Organization": get_organization(player)
        }

        # Add scores for each game
        for game in GAME_ORDER:
            if game in game_dfs:
                df = game_dfs[game]
                if player in df["Player"].values:
                    if game == "Super Mario Bros":
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                    elif game == "Sokoban":
                        # Parse Sokoban score string and get maximum level
                        levels_str = df[df["Player"] == player]["Levels Cracked"].iloc[0]
                        try:
                            levels = [int(x.strip()) for x in levels_str.split(";") if x.strip()]
                            player_data[f"{game} Score"] = max(levels) if levels else 0
                        except:
                            player_data[f"{game} Score"] = 0
                    elif game == "2048":
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                    elif game == "Candy Crash":
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Average Score"].iloc[0]
                    elif game in ["Tetris (complete)", "Tetris (planning only)"]:
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                else:
                    player_data[f"{game} Score"] = -1

        results.append(player_data)

    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by total score across all games
    if not df_results.empty:
        # Calculate total score for each player
        df_results["Total Score"] = 0
        for game in GAME_ORDER:
            if f"{game} Score" in df_results.columns:
                df_results["Total Score"] += df_results[f"{game} Score"].apply(
                    lambda x: float(x) if x != -1 else 0
                )
        
        # Sort by total score in descending order
        df_results = df_results.sort_values("Total Score", ascending=False)
        
        # Drop the temporary total score column
        df_results = df_results.drop("Total Score", axis=1)

    return df_results
