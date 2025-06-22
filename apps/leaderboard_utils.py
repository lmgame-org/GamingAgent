import pandas as pd
import json
import numpy as np

# Define game order
GAME_ORDER = [
    # "Super Mario Bros", # Commented out
    "Super Mario Bros",
    "Sokoban",
    "2048",
    "Candy Crush",
    # "Tetris (complete)", # Commented out
    "Tetris",
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


def get_sokoban_leaderboard(rank_data, limit_to_top_n=None):
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
        # Apply limit if specified
        if limit_to_top_n is not None:
            df = df.head(limit_to_top_n)
    return df

def get_2048_leaderboard(rank_data, limit_to_top_n=None):
    data = rank_data.get("2048", {}).get("results", [])
    # --- Diagnostic Print Removed ---
    # if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
    #     print(f"DEBUG_UTILS: Keys in first item of raw data for 2048: {list(data[0].keys())}")
    # elif not data:
    #     print("DEBUG_UTILS: Raw data for 2048 is empty.")
    # else:
    #     print("DEBUG_UTILS: Raw data for 2048 is not in the expected list of dicts format.")
    # --- End Diagnostic Print Removed ---
    df = pd.DataFrame(data)
    # print(f"DEBUG_UTILS: Columns after pd.DataFrame(data): {df.columns.tolist()}") # REMOVED

    df = df.rename(columns={
        "model": "Player",
        "score": "Score",       # From new JSON structure
        "details": "Details",    # From new JSON structure
        "highest_tail": "Highest Tail" # Added new column
        # Old fields like "steps", "time", "rank" are removed
    })
    # print(f"DEBUG_UTILS: Columns after rename: {df.columns.tolist()}") # REMOVED

    # Ensure 'Player' column exists before applying get_organization
    if "Player" in df.columns:
        df["Organization"] = df["Player"].apply(get_organization)
    else:
        # Handle case where 'Player' column might be missing after rename (should not happen with current logic)
        # print("DEBUG_UTILS: 'Player' column not found after rename, skipping Organization.") # REMOVED
        df["Organization"] = "unknown" # Fallback

    columns_to_keep = ["Player", "Organization", "Score", "Highest Tail", "Details"] # Added "Highest Tail"
    
    # Defensive check for 'Highest Tail' before filtering - REMOVED
    # if 'highest_tail' in df.columns and 'Highest Tail' not in df.columns:
    #     print("DEBUG_UTILS: 'highest_tail' (lowercase) found, but 'Highest Tail' (capitalized) not. This indicates a rename issue.")
    # elif 'Highest Tail' not in df.columns and 'highest_tail' not in df.columns:
    #     print("DEBUG_UTILS: Neither 'Highest Tail' nor 'highest_tail' found in columns before filtering.")

    # df_columns = [col for col in columns_to_keep if col in df.columns] # REMOVED logic that used df_columns
    # print(f"DEBUG_UTILS: df_columns selected (columns that are in columns_to_keep AND in df.columns): {df_columns}") # REMOVED
    
    # Ensure all columns in columns_to_keep exist in df, fill with np.nan if not
    for col_k in columns_to_keep:
        if col_k not in df.columns:
            # print(f"DEBUG_UTILS: Column '{col_k}' from columns_to_keep not found in DataFrame. Adding it with NaN values.") # REMOVED
            df[col_k] = np.nan # Or some other default like 'n/a' if appropriate

    df = df[columns_to_keep] # Use columns_to_keep directly after ensuring they exist
    # print(f"DEBUG_UTILS: Columns after final selection: {df.columns.tolist()}") # REMOVED

    if "Score" in df.columns:
        df["Score"] = pd.to_numeric(df["Score"], errors='coerce')
        df = df.sort_values("Score", ascending=False)
        # Apply limit if specified
        if limit_to_top_n is not None:
            df = df.head(limit_to_top_n)
    return df

def get_candy_leaderboard(rank_data, limit_to_top_n=None):
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
        # Apply limit if specified
        if limit_to_top_n is not None:
            df = df.head(limit_to_top_n)
    return df

def get_tetris_planning_leaderboard(rank_data, limit_to_top_n=None):
    data = rank_data.get("Tetris", {}).get("results", [])
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
        # Apply limit if specified
        if limit_to_top_n is not None:
            df = df.head(limit_to_top_n)
    return df

def get_ace_attorney_leaderboard(rank_data, limit_to_top_n=None):
    data = rank_data.get("Ace Attorney", {}).get("results", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "model": "Player",
        "score": "Score",
        "progress": "Progress"
    })
    df["Organization"] = df["Player"].apply(get_organization)
    
    # Define columns to keep
    columns_to_keep = ["Player", "Organization", "Score", "Progress"]
    # Filter to only columns that actually exist in the DataFrame after renaming
    df_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[df_columns]

    if "Score" in df.columns:
        df["Score"] = pd.to_numeric(df["Score"], errors='coerce')
        df = df.sort_values("Score", ascending=False)  # Higher score is better
        # Apply limit if specified
        if limit_to_top_n is not None:
            df = df.head(limit_to_top_n)
    return df

def get_mario_planning_leaderboard(rank_data, limit_to_top_n=None):
    data = rank_data.get("Super Mario Bros", {}).get("results", [])
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
        # Apply limit if specified
        if limit_to_top_n is not None:
            df = df.head(limit_to_top_n)
    return df

def calculate_rank_and_completeness(rank_data, selected_games):
    # Dictionary to store DataFrames for each game
    game_dfs = {}
    
    # Get DataFrames for selected games
    # if selected_games.get("Super Mario Bros"): # Commented out
    #     game_dfs["Super Mario Bros"] = get_mario_leaderboard(rank_data)
    if selected_games.get("Super Mario Bros"):
        game_dfs["Super Mario Bros"] = get_mario_planning_leaderboard(rank_data)
    if selected_games.get("Sokoban"):
        game_dfs["Sokoban"] = get_sokoban_leaderboard(rank_data)
    if selected_games.get("2048"):
        game_dfs["2048"] = get_2048_leaderboard(rank_data)
    if selected_games.get("Candy Crush"):
        game_dfs["Candy Crush"] = get_candy_leaderboard(rank_data)
    # if selected_games.get("Tetris (complete)"): # Commented out
    #     game_dfs["Tetris (complete)"] = get_tetris_leaderboard(rank_data)
    if selected_games.get("Tetris"):
        game_dfs["Tetris"] = get_tetris_planning_leaderboard(rank_data)
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
                    if game == "Super Mario Bros":
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
                    elif game in ["Tetris"]:
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

def get_combined_leaderboard(rank_data, selected_games, limit_to_top_n=None):
    """
    Get combined leaderboard for selected games
    
    Args:
        rank_data (dict): Dictionary containing rank data
        selected_games (dict): Dictionary of game names and their selection status
        limit_to_top_n (int, optional): Limit results to top N entries. None means no limit.
        
    Returns:
        pd.DataFrame: Combined leaderboard DataFrame
    """
    # Dictionary to store DataFrames for each game
    game_dfs = {}
    
    # Get DataFrames for selected games
    # if selected_games.get("Super Mario Bros"): # Commented out
    #     game_dfs["Super Mario Bros"] = get_mario_leaderboard(rank_data)
    if selected_games.get("Super Mario Bros"):
        game_dfs["Super Mario Bros"] = get_mario_planning_leaderboard(rank_data)
    if selected_games.get("Sokoban"):
        game_dfs["Sokoban"] = get_sokoban_leaderboard(rank_data)
    if selected_games.get("2048"):
        game_dfs["2048"] = get_2048_leaderboard(rank_data)
    if selected_games.get("Candy Crush"):
        game_dfs["Candy Crush"] = get_candy_leaderboard(rank_data)
    # if selected_games.get("Tetris (complete)"): # Commented out
    #     game_dfs["Tetris (complete)"] = get_tetris_leaderboard(rank_data)
    if selected_games.get("Tetris"):
        game_dfs["Tetris"] = get_tetris_planning_leaderboard(rank_data)
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
                    if game == "Super Mario Bros":
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                    elif game == "Sokoban":
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                    elif game == "2048":
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                    elif game == "Candy Crush":
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                    elif game in ["Tetris"]:
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                    elif game == "Ace Attorney":
                        player_data[f"{game} Score"] = df[df["Player"] == player]["Score"].iloc[0]
                else:
                    player_data[f"{game} Score"] = 'n/a'

        results.append(player_data)

    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Calculate normalized scores and average normalized score
    if not df_results.empty:
        # Import the normalize_values function from data_visualization
        from data_visualization import normalize_values
        
        # Calculate normalized scores for each game
        game_score_columns = []
        for game in GAME_ORDER:
            score_col = f"{game} Score"
            if score_col in df_results.columns:
                game_score_columns.append(score_col)
                # Get numeric values, replacing 'n/a' with NaN
                # Use where() to avoid FutureWarning about downcasting in replace()
                series = df_results[score_col].copy()
                series = series.where(series != 'n/a', np.nan)
                numeric_scores = pd.to_numeric(series, errors='coerce')
                
                # Skip games where all scores are NaN or 0
                valid_scores = numeric_scores.dropna()
                if len(valid_scores) > 0 and valid_scores.sum() > 0:
                    mean = valid_scores.mean()
                    std = valid_scores.std() if len(valid_scores) > 1 else 0
                    
                    # Calculate normalized scores for all players
                    normalized_scores = []
                    for _, row in df_results.iterrows():
                        score = row[score_col]
                        if score == 'n/a' or pd.isna(score):
                            normalized_scores.append(0)
                        else:
                            normalized_scores.append(normalize_values([float(score)], mean, std)[0])
                    
                    df_results[f"norm_{score_col}"] = normalized_scores
                else:
                    # If no valid scores, set all normalized scores to 0
                    df_results[f"norm_{score_col}"] = 0
        
        # Calculate average normalized score across games
        normalized_columns = [f"norm_{col}" for col in game_score_columns if f"norm_{col}" in df_results.columns]
        if normalized_columns:
            df_results["Avg Normalized Score"] = df_results[normalized_columns].mean(axis=1).round(2)
        else:
            df_results["Avg Normalized Score"] = 0.0
        
        # Reorder columns to put Avg Normalized Score after Organization
        base_columns = ["Player", "Organization", "Avg Normalized Score"]
        game_columns = [col for col in df_results.columns if col.endswith(" Score") and not col.startswith("norm_") and col != "Avg Normalized Score"]
        other_columns = [col for col in df_results.columns if col not in base_columns + game_columns and not col.startswith("norm_")]
        
        # Create final column order
        final_columns = base_columns + game_columns + other_columns
        df_results = df_results[final_columns]
        
        # Sort by average normalized score in descending order
        df_results = df_results.sort_values("Avg Normalized Score", ascending=False)
        
        # Apply limit if specified
        if limit_to_top_n is not None:
            df_results = df_results.head(limit_to_top_n)

    return df_results
