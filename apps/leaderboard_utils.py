import pandas as pd
import json
import numpy as np

# Define game order
GAME_ORDER = [
    # "Super Mario Bros", # Commented out
    "Super Mario Bros (planning only)",
    "Sokoban",
    "2048",
    "Candy Crush",
    # "Tetris (complete)", # Commented out
    "Tetris (planning only)",
    "Ace Attorney"
]

def get_organization(model_name):
    m = model_name.lower()
    if "claude" in m:
        return "anthropic"
    elif "gemini" in m:
        return "google"
    elif "o1" in m or "gpt" in m or "o3" in m or "o4" in m:
        return "openai"
    elif "deepseek" in m:
        return "deepseek"
    elif "llama" in m:
        return "meta"
    elif "grok" in m:
        return "xai"
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
    if "Score" in df.columns:
        df = df.sort_values("Score", ascending=False)
    return df

def get_sokoban_leaderboard(rank_data):
    data = rank_data.get("Sokoban", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player",
        "score": "Score",
        "steps": "Steps",
        "detail_box_on_target": "Detail Box On Target",
        "cracked_levels": "Levels Cracked"
    })
    df["Organization"] = df["Player"].apply(get_organization)
    
    # Define columns to keep, ensuring 'Score' is present
    columns_to_keep = ["Player", "Organization", "Score", "Levels Cracked", "Detail Box On Target", "Steps"]
    # Filter to only columns that actually exist in the DataFrame after renaming
    df_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[df_columns]

    if "Score" in df.columns:
        df["Score"] = pd.to_numeric(df["Score"], errors='coerce')
        df = df.sort_values("Score", ascending=False)
    return df

def get_2048_leaderboard(rank_data):
    data = rank_data.get("2048", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "score": "Score",       # From new JSON structure
        "details": "Details"    # From new JSON structure
        # Old fields like "steps", "time", "rank" are removed
    })
    df["Organization"] = df["Player"].apply(get_organization)
    
    columns_to_keep = ["Player", "Organization", "Score", "Details"]
    df_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[df_columns]

    if "Score" in df.columns:
        df["Score"] = pd.to_numeric(df["Score"], errors='coerce')
        df = df.sort_values("Score", ascending=False)
    return df

def get_candy_leaderboard(rank_data):
    data = rank_data.get("Candy Crush", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "score": "Score",
        "details": "Details"
    })
    df["Organization"] = df["Player"].apply(get_organization)
    
    columns_to_keep = ["Player", "Organization", "Score", "Details"]
    df_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[df_columns]

    if "Score" in df.columns:
        df["Score"] = pd.to_numeric(df["Score"], errors='coerce')
        df = df.sort_values("Score", ascending=False)
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
        "score": "Score",       # From new JSON structure
        "details": "Details"    # From new JSON structure
        # Old fields like "steps_blocks", "rank" are removed
    })
    df["Organization"] = df["Player"].apply(get_organization)
    
    columns_to_keep = ["Player", "Organization", "Score", "Details"]
    df_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[df_columns]

    if "Score" in df.columns:
        df["Score"] = pd.to_numeric(df["Score"], errors='coerce')
        df = df.sort_values("Score", ascending=False)
    return df

def get_ace_attorney_leaderboard(rank_data):
    data = rank_data.get("Ace Attorney", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player",
        "score": "Score",
        "progress": "Progress",
        "evaluator result": "Evaluator Result"
    })
    df["Organization"] = df["Player"].apply(get_organization)
    
    # Define columns to keep, including Evaluator Result
    columns_to_keep = ["Player", "Organization", "Score", "Progress", "Evaluator Result"]
    # Filter to only columns that actually exist in the DataFrame after renaming
    df_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[df_columns]

    if "Score" in df.columns:
        df["Score"] = pd.to_numeric(df["Score"], errors='coerce')
        df = df.sort_values("Score", ascending=False)  # Higher score is better
    return df

def get_mario_planning_leaderboard(rank_data):
    data = rank_data.get("Super Mario Bros (planning only)", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player", 
        "score": "Score", 
        "detail_data": "Detail Data",
        "progress": "Progress"
    })
    df["Organization"] = df["Player"].apply(get_organization)
    # Define columns to keep
    columns_to_keep = ["Player", "Organization", "Score", "Progress", "Detail Data"]
    df_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[df_columns]

    if "Score" in df.columns:
        df["Score"] = pd.to_numeric(df["Score"], errors='coerce')
        df = df.sort_values("Score", ascending=False)
    return df

def calculate_rank_and_completeness(rank_data, selected_games):
    # Dictionary to store DataFrames for each game
    game_dfs = {}
    
    # Get DataFrames for selected games
    # if selected_games.get("Super Mario Bros"): # Commented out
    #     game_dfs["Super Mario Bros"] = get_mario_leaderboard(rank_data)
    if selected_games.get("Super Mario Bros (planning only)"):
        game_dfs["Super Mario Bros (planning only)"] = get_mario_planning_leaderboard(rank_data)
    if selected_games.get("Sokoban"):
        game_dfs["Sokoban"] = get_sokoban_leaderboard(rank_data)
    if selected_games.get("2048"):
        game_dfs["2048"] = get_2048_leaderboard(rank_data)
    if selected_games.get("Candy Crush"):
        game_dfs["Candy Crush"] = get_candy_leaderboard(rank_data)
    # if selected_games.get("Tetris (complete)"): # Commented out
    #     game_dfs["Tetris (complete)"] = get_tetris_leaderboard(rank_data)
    if selected_games.get("Tetris (planning only)"):
        game_dfs["Tetris (planning only)"] = get_tetris_planning_leaderboard(rank_data)
    if selected_games.get("Ace Attorney"):
        game_dfs["Ace Attorney"] = get_ace_attorney_leaderboard(rank_data)

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
                    # if game == "Super Mario Bros": # Commented out
                    #     player_score = df[df["Player"] == player]["Score"].iloc[0]
                    #     rank = len(df[df["Score"] > player_score]) + 1
                    if game == "Super Mario Bros (planning only)":
                        player_score = df[df["Player"] == player]["Score"].iloc[0]
                        rank = len(df[df["Score"] > player_score]) + 1
                    elif game == "Sokoban":
                        player_score = df[df["Player"] == player]["Score"].iloc[0]
                        rank = len(df[df["Score"] > player_score]) + 1
                    elif game == "2048":
                        player_score = df[df["Player"] == player]["Score"].iloc[0]
                        rank = len(df[df["Score"] > player_score]) + 1
                    elif game == "Candy Crush":
                        player_score = df[df["Player"] == player]["Score"].iloc[0]
                        rank = len(df[df["Score"] > player_score]) + 1
                    elif game in ["Tetris (planning only)"]:
                        player_score = df[df["Player"] == player]["Score"].iloc[0]
                        rank = len(df[df["Score"] > player_score]) + 1
                    elif game == "Ace Attorney":
                        player_score = df[df["Player"] == player]["Score"].iloc[0]
                        rank = len(df[df["Score"] > player_score]) + 1

                    ranks.append(rank)
                    player_data[f"{game} Score"] = player_score
                else:
                    player_data[f"{game} Score"] = 'n/a'

        # Calculate average rank and completeness for sorting
        if ranks:
            player_data["Average Rank"] = round(np.mean(ranks), 2)
            player_data["Games Played"] = games_played
        else:
            player_data["Average Rank"] = float('inf')
            player_data["Games Played"] = 0

        results.append(player_data)

    # Create DataFrame and sort by average rank and completeness
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        # Sort by average rank (ascending) and games played (descending)
        df_results = df_results.sort_values(
            by=["Average Rank", "Games Played"],
            ascending=[True, False]
        )
        # Drop the sorting columns
        df_results = df_results.drop(["Average Rank", "Games Played"], axis=1)

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
    # if selected_games.get("Super Mario Bros"): # Commented out
    #     game_dfs["Super Mario Bros"] = get_mario_leaderboard(rank_data)
    if selected_games.get("Super Mario Bros (planning only)"):
        game_dfs["Super Mario Bros (planning only)"] = get_mario_planning_leaderboard(rank_data)
    if selected_games.get("Sokoban"):
        game_dfs["Sokoban"] = get_sokoban_leaderboard(rank_data)
    if selected_games.get("2048"):
        game_dfs["2048"] = get_2048_leaderboard(rank_data)
    if selected_games.get("Candy Crush"):
        game_dfs["Candy Crush"] = get_candy_leaderboard(rank_data)
    # if selected_games.get("Tetris (complete)"): # Commented out
    #     game_dfs["Tetris (complete)"] = get_tetris_leaderboard(rank_data)
    if selected_games.get("Tetris (planning only)"):
        game_dfs["Tetris (planning only)"] = get_tetris_planning_leaderboard(rank_data)
    if selected_games.get("Ace Attorney"):
        game_dfs["Ace Attorney"] = get_ace_attorney_leaderboard(rank_data)

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
                    # if game == "Super Mario Bros": # Commented out
                    #     player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                    if game == "Super Mario Bros (planning only)":
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                    elif game == "Sokoban":
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                    elif game == "2048":
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                    elif game == "Candy Crush":
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                    elif game in ["Tetris (planning only)"]:
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                    elif game == "Ace Attorney":
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                else:
                    player_data[f"{game} Score"] = 'n/a'

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
                    lambda x: float(x) if x != 'n/a' else 0
                )
        
        # Sort by total score in descending order
        df_results = df_results.sort_values("Total Score", ascending=False)
        
        # Drop the temporary total score column
        df_results = df_results.drop("Total Score", axis=1)

    return df_results
