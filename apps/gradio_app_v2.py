import gradio as gr
import os
import pandas as pd
import json
from PIL import Image, ImageSequence
import io
from functools import reduce
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from leaderboard_utils import (
    get_organization,
    get_mario_planning_leaderboard,
    get_sokoban_leaderboard,
    get_2048_leaderboard,
    get_candy_leaderboard,
    get_tetris_leaderboard,
    get_tetris_planning_leaderboard,
    get_ace_attorney_leaderboard,
    get_combined_leaderboard,
    GAME_ORDER
)
from data_visualization import (
    get_combined_leaderboard_with_group_bar,
    create_organization_radar_chart,
    create_top_players_radar_chart,
    create_player_radar_chart,
    create_horizontal_bar_chart,
    normalize_values,
    get_combined_leaderboard_with_single_radar
)
from gallery_tab import create_video_gallery



HAS_ENHANCED_LEADERBOARD = True


# Define time points and their corresponding data files
TIME_POINTS = {
    "03/25/2025": "rank_data_03_25_2025.json",
    # Add more time points here as they become available
}

# Load the initial JSON file with rank data
with open(TIME_POINTS["03/25/2025"], "r") as f:
    rank_data = json.load(f)

# Load the model leaderboard data
with open("rank_single_model_03_25_2025.json", "r") as f:
    model_rank_data = json.load(f)

# Add leaderboard state at the top level
leaderboard_state = {
    "current_game": None,
    "previous_overall": {
        # "Super Mario Bros": True, # Commented out
        "Super Mario Bros (planning only)": True,
        "Sokoban": True,
        "2048": True,
        "Candy Crush": True,
        # "Tetris (complete)", # Commented out
        "Tetris (planning only)": True,
        "Ace Attorney": True
    },
    "previous_details": {
        # "Super Mario Bros": False, # Commented out
        "Super Mario Bros (planning only)": False,
        "Sokoban": False,
        "2048": False,
        "Candy Crush": False,
        # "Tetris (complete)": False, # Commented out
        "Tetris (planning only)": False,
        "Ace Attorney": False
    }
}


# Load video links and news data
with open('assets/game_video_link.json', 'r') as f:
    VIDEO_LINKS = json.load(f)

with open('assets/news.json', 'r') as f:
    NEWS_DATA = json.load(f)

def load_rank_data(time_point):
    """Load rank data for a specific time point"""
    if time_point in TIME_POINTS:
        try:
            with open(TIME_POINTS[time_point], "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    return None

# Add a note about score values
def add_score_note():
    return gr.Markdown("*Note: 'n/a' in the table indicates no data point for that model.*", elem_classes="score-note")

# Function to prepare DataFrame for display
def prepare_dataframe_for_display(df, for_game=None):
    """Format DataFrame for better display in the UI"""
    # Clone the DataFrame to avoid modifying the original
    display_df = df.copy()
    
    # Filter out normalized score columns
    norm_columns = [col for col in display_df.columns if col.startswith('norm_')]
    if norm_columns:
        display_df = display_df.drop(columns=norm_columns)
    
    # Replace '_' with '-' for better display
    for col in display_df.columns:
        if col.endswith(' Score'):
            display_df[col] = display_df[col].apply(lambda x: '-' if x == '_' else x)
    
    # If we're in detailed view, sort by score
    if for_game:
        # Sort by relevant score column
        score_col = f"{for_game} Score"
        if score_col in display_df.columns:
            # Convert to numeric for sorting, treating '-' as NaN
            display_df[score_col] = pd.to_numeric(display_df[score_col], errors='coerce')
            # Sort by score in descending order
            display_df = display_df.sort_values(by=score_col, ascending=False)
            # Filter out models that didn't participate
            display_df = display_df[~display_df[score_col].isna()]
    else:
        # For overall view, sort by average of game scores (implicitly used for ranking)
        # but we won't add an explicit 'Rank' or 'Average Rank' column to the final display_df
        
        # Calculate an internal sorting key based on average scores, but don't add it to the display_df
        score_cols = [col for col in display_df.columns if col.endswith(' Score')]
        if score_cols:
            temp_sort_df = display_df.copy()
            for col in score_cols:
                temp_sort_df[col] = pd.to_numeric(temp_sort_df[col], errors='coerce')
            
            # Calculate average of the game scores (use mean of ranks from utils for actual ranking logic if different)
            # For display sorting, let's use a simple average of available scores.
            # The actual ranking for 'Average Rank' in leaderboard_utils uses mean of ranks, which is more robust.
            # Here we just need a consistent sort order.
            
            # Create a temporary column for sorting
            temp_sort_df['temp_avg_score_for_sort'] = temp_sort_df[score_cols].mean(axis=1)
            
            # Sort by this temporary average score (higher is better for scores)
            # and then by Player name as a tie-breaker
            display_df = display_df.loc[temp_sort_df.sort_values(by=['temp_avg_score_for_sort', 'Player'], ascending=[False, True]).index]
    
    # Add line breaks to column headers
    new_columns = {}
    for col in display_df.columns:
        if col.endswith(' Score'):
            # Replace 'Game Name Score' with 'Game Name\nScore'
            game_name = col.replace(' Score', '')
            new_col = f"{game_name}\nScore"
            new_columns[col] = new_col
    
    # Rename columns with new line breaks
    if new_columns:
        display_df = display_df.rename(columns=new_columns)
    
    return display_df

# Helper function to ensure leaderboard updates maintain consistent height
def update_df_with_height(df):
    """Update DataFrame with consistent height parameter."""
    # Create column widths array
    col_widths = ["40px"]  # Row number column width
    col_widths.append("230px")  # Player column - reduced by 20px
    col_widths.append("120px")  # Organization column
    # Add game score columns
    for _ in range(len(df.columns) - 2):
        col_widths.append("120px")
    
    return gr.update(value=df, 
                     show_row_numbers=True, 
                     show_fullscreen_button=True,
                     line_breaks=True,
                     show_search="search",
                     # max_height=None,  # Remove height limitation - COMMENTED OUT
                     column_widths=col_widths)

def update_leaderboard(# mario_overall, mario_details, # Commented out
                       mario_plan_overall, mario_plan_details, # Added
                       sokoban_overall, sokoban_details,
                       _2048_overall, _2048_details,
                       candy_overall, candy_details,
                       # tetris_overall, tetris_details, # Commented out
                       tetris_plan_overall, tetris_plan_details,
                       ace_attorney_overall, ace_attorney_details,
                       data_source=None):
    global leaderboard_state
    
    # Use provided data source or default to rank_data
    data = data_source if data_source is not None else rank_data
    
    # Convert current checkbox states to dictionary for easier comparison
    current_overall = {
        # "Super Mario Bros": mario_overall, # Commented out
        "Super Mario Bros (planning only)": mario_plan_overall,
        "Sokoban": sokoban_overall,
        "2048": _2048_overall,
        "Candy Crush": candy_overall,
        # "Tetris (complete)": tetris_overall, # Commented out
        "Tetris (planning only)": tetris_plan_overall,
        "Ace Attorney": ace_attorney_overall
    }
    
    current_details = {
        # "Super Mario Bros": mario_details, # Commented out
        "Super Mario Bros (planning only)": mario_plan_details,
        "Sokoban": sokoban_details,
        "2048": _2048_details,
        "Candy Crush": candy_details,
        # "Tetris (complete)": tetris_details, # Commented out
        "Tetris (planning only)": tetris_plan_details,
        "Ace Attorney": ace_attorney_details
    }
    
    # Find which game's state changed
    changed_game = None
    for game in current_overall.keys():
        if (current_overall[game] != leaderboard_state["previous_overall"][game] or 
            current_details[game] != leaderboard_state["previous_details"][game]):
            changed_game = game
            break
    
    if changed_game:
        # If a game's details checkbox was checked
        if current_details[changed_game] and not leaderboard_state["previous_details"][changed_game]:
            # Reset all other games' states
            for game in current_overall.keys():
                if game != changed_game:
                    current_overall[game] = False
                    current_details[game] = False
                    leaderboard_state["previous_overall"][game] = False
                    leaderboard_state["previous_details"][game] = False
            
            # Update state for the selected game
            leaderboard_state["current_game"] = changed_game
            leaderboard_state["previous_overall"][changed_game] = True
            leaderboard_state["previous_details"][changed_game] = True
            current_overall[changed_game] = True
        
        # If a game's overall checkbox was checked
        elif current_overall[changed_game] and not leaderboard_state["previous_overall"][changed_game]:
            # If we were in details view for another game, switch to overall view
            if leaderboard_state["current_game"] and leaderboard_state["previous_details"][leaderboard_state["current_game"]]:
                # Reset previous game's details
                leaderboard_state["previous_details"][leaderboard_state["current_game"]] = False
                current_details[leaderboard_state["current_game"]] = False
                leaderboard_state["current_game"] = None
            
            # Update state
            leaderboard_state["previous_overall"][changed_game] = True
            leaderboard_state["previous_details"][changed_game] = False
        
        # If a game's overall checkbox was unchecked
        elif not current_overall[changed_game] and leaderboard_state["previous_overall"][changed_game]:
            # If we're in details view, don't allow unchecking the overall checkbox
            if leaderboard_state["current_game"] == changed_game:
                current_overall[changed_game] = True
            else:
                leaderboard_state["previous_overall"][changed_game] = False
                if leaderboard_state["current_game"] == changed_game:
                    leaderboard_state["current_game"] = None
        
        # If a game's details checkbox was unchecked
        elif not current_details[changed_game] and leaderboard_state["previous_details"][changed_game]:
            leaderboard_state["previous_details"][changed_game] = False
            if leaderboard_state["current_game"] == changed_game:
                leaderboard_state["current_game"] = None
                # When exiting details view, only reset the current game's state
                current_overall[changed_game] = True
                current_details[changed_game] = False
                leaderboard_state["previous_overall"][changed_game] = True
                leaderboard_state["previous_details"][changed_game] = False
    
    # Special case: If all games are selected and we're trying to view details
    all_games_selected = all(current_overall.values()) and not any(current_details.values())
    if all_games_selected and changed_game and current_details[changed_game]:
        # Reset all other games' states
        for game in current_overall.keys():
            if game != changed_game:
                current_overall[game] = False
                current_details[game] = False
                leaderboard_state["previous_overall"][game] = False
                leaderboard_state["previous_details"][game] = False
        
        # Update state for the selected game
        leaderboard_state["current_game"] = changed_game
        leaderboard_state["previous_overall"][changed_game] = True
        leaderboard_state["previous_details"][changed_game] = True
        current_overall[changed_game] = True
    
    # Build dictionary for selected games
    selected_games = {
        # "Super Mario Bros": current_overall["Super Mario Bros"], # Commented out
        "Super Mario Bros (planning only)": current_overall["Super Mario Bros (planning only)"],
        "Sokoban": current_overall["Sokoban"],
        "2048": current_overall["2048"],
        "Candy Crush": current_overall["Candy Crush"],
        # "Tetris (complete)": current_overall["Tetris (complete)"], # Commented out
        "Tetris (planning only)": current_overall["Tetris (planning only)"],
        "Ace Attorney": current_overall["Ace Attorney"]
    }
    
    # Get the appropriate DataFrame and charts based on current state
    if leaderboard_state["current_game"]:
        # For detailed view
        # if leaderboard_state["current_game"] == "Super Mario Bros": # Commented out
        #     df = get_mario_leaderboard(data)
        if leaderboard_state["current_game"] == "Super Mario Bros (planning only)":
            df = get_mario_planning_leaderboard(data)
        elif leaderboard_state["current_game"] == "Sokoban":
            df = get_sokoban_leaderboard(data)
        elif leaderboard_state["current_game"] == "2048":
            df = get_2048_leaderboard(data)
        elif leaderboard_state["current_game"] == "Candy Crush":
            df = get_candy_leaderboard(data)
        elif leaderboard_state["current_game"] == "Tetris (planning only)":
            df = get_tetris_planning_leaderboard(data)
        elif leaderboard_state["current_game"] == "Ace Attorney":
            df = get_ace_attorney_leaderboard(data)
        else: # Should not happen if current_game is one of the known games
            df = pd.DataFrame() # Empty df
        
        display_df = prepare_dataframe_for_display(df, leaderboard_state["current_game"])
        chart = create_horizontal_bar_chart(df, leaderboard_state["current_game"])
        radar_chart = chart # In detailed view, radar and group bar can be the same as the main chart
        group_bar_chart = chart 
    else:
        # For overall view
        df, group_bar_chart = get_combined_leaderboard_with_group_bar(data, selected_games)
        display_df = prepare_dataframe_for_display(df)
        _, radar_chart = get_combined_leaderboard_with_single_radar(data, selected_games)
        chart = radar_chart # In overall view, the 'detailed' chart can be the radar chart
    
    # Return values, including all four plot placeholders
    return (update_df_with_height(display_df), chart, radar_chart, group_bar_chart,
            current_overall["Super Mario Bros (planning only)"], current_details["Super Mario Bros (planning only)"],
            current_overall["Sokoban"], current_details["Sokoban"],
            current_overall["2048"], current_details["2048"],
            current_overall["Candy Crush"], current_details["Candy Crush"],
            current_overall["Tetris (planning only)"], current_details["Tetris (planning only)"],
            current_overall["Ace Attorney"], current_details["Ace Attorney"])

def update_leaderboard_with_time(time_point, # mario_overall, mario_details, # Commented out
                               mario_plan_overall, mario_plan_details, # Added
                               sokoban_overall, sokoban_details,
                               _2048_overall, _2048_details,
                               candy_overall, candy_details,
                               # tetris_overall, tetris_details, # Commented out
                               tetris_plan_overall, tetris_plan_details,
                               ace_attorney_overall, ace_attorney_details):
    # Load rank data for the selected time point
    global rank_data
    new_rank_data = load_rank_data(time_point)
    if new_rank_data is not None:
        rank_data = new_rank_data
    
    # Use the existing update_leaderboard function, including Super Mario (planning only)
    return update_leaderboard(# mario_overall, mario_details, # Commented out
                            mario_plan_overall, mario_plan_details, # Added
                            sokoban_overall, sokoban_details,
                            _2048_overall, _2048_details,
                            candy_overall, candy_details,
                            # tetris_overall, tetris_details, # Commented out
                            tetris_plan_overall, tetris_plan_details,
                            ace_attorney_overall, ace_attorney_details)

def get_initial_state():
    """Get the initial state for the leaderboard"""
    return {
        "current_game": None,
        "previous_overall": {
            # "Super Mario Bros": True, # Commented out
            "Super Mario Bros (planning only)": True,
            "Sokoban": True,
            "2048": True,
            "Candy Crush": True,
            # "Tetris (complete)", # Commented out
            "Tetris (planning only)": True,
            "Ace Attorney": True
        },
        "previous_details": {
            # "Super Mario Bros": False, # Commented out
            "Super Mario Bros (planning only)": False,
            "Sokoban": False,
            "2048": False,
            "Candy Crush": False,
            # "Tetris (complete)": False, # Commented out
            "Tetris (planning only)": False,
            "Ace Attorney": False
        }
    }

def clear_filters(data_source=None):
    global leaderboard_state
    
    # Use provided data source or default to rank_data
    data = data_source if data_source is not None else rank_data
    
    selected_games = {
        "Super Mario Bros (planning only)": True,
        "Sokoban": True,
        "2048": True,
        "Candy Crush": True,
        "Tetris (planning only)": True,
        "Ace Attorney": True
    }
    
    df, group_bar_chart = get_combined_leaderboard_with_group_bar(data, selected_games)
    display_df = prepare_dataframe_for_display(df)
    _, radar_chart = get_combined_leaderboard_with_single_radar(data, selected_games)
    
    leaderboard_state = get_initial_state()
    
    # Return values, including all four plot placeholders
    return (update_df_with_height(display_df), radar_chart, radar_chart, group_bar_chart,
            True, False, # mario_plan
            True, False,  # sokoban
            True, False,  # 2048
            True, False,  # candy
            True, False,  # tetris plan
            True, False)  # ace attorney

def create_timeline_slider():
    """Create a custom timeline slider component"""
    timeline_html = """
    <div class="timeline-container">
        <style>
            .timeline-container {
                width: 85%;  /* Increased from 70% to 85% */
                padding: 8px;
                font-family: Arial, sans-serif;
                height: 40px;
                display: flex;
                align-items: center;
            }
            .timeline-track {
                position: relative;
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
                margin: 0;
                width: 100%;
            }
            .timeline-progress {
                position: absolute;
                height: 100%;
                background: #2196F3;
                border-radius: 3px;
                width: 100%;
            }
            .timeline-handle {
                position: absolute;
                right: 0;
                top: 50%;
                transform: translate(50%, -50%);
                width: 20px;
                height: 20px;
                background: #2196F3;
                border: 3px solid white;
                border-radius: 50%;
                cursor: pointer;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            }
            .timeline-date {
                position: absolute;
                top: -25px;
                transform: translateX(-50%);
                background: #2196F3;  /* Changed to match slider blue color */
                color: #ffffff !important;
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 12px;
                white-space: nowrap;
                font-weight: 600;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                letter-spacing: 0.5px;
                text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            }
        </style>
        <div class="timeline-track">
            <div class="timeline-progress"></div>
            <div class="timeline-handle">
                <div class="timeline-date">03/25/2025</div>
            </div>
        </div>
    </div>
    <script>
        (function() {
            const container = document.querySelector('.timeline-container');
            const track = container.querySelector('.timeline-track');
            const handle = container.querySelector('.timeline-handle');
            let isDragging = false;
            
            // For now, we only have one time point
            const timePoints = {
                "03/25/2025": 1.0
            };
            
            function updatePosition(e) {
                if (!isDragging) return;
                
                const rect = track.getBoundingClientRect();
                let x = (e.clientX - rect.left) / rect.width;
                x = Math.max(0, Math.min(1, x));
                
                // For now, snap to the only available time point
                x = 1.0;
                
                handle.style.right = `${(1 - x) * 100}%`;
            }
            
            handle.addEventListener('mousedown', (e) => {
                isDragging = true;
                e.preventDefault();
            });
            
            document.addEventListener('mousemove', updatePosition);
            document.addEventListener('mouseup', () => {
                isDragging = false;
            });
            
            // Prevent text selection while dragging
            container.addEventListener('selectstart', (e) => {
                if (isDragging) e.preventDefault();
            });
        })();
    </script>
    """
    return gr.HTML(timeline_html)

def build_app():
    with gr.Blocks(css="""
        /* Fix for scrolling issues */
        html, body {
            overflow-y: auto !important;
            overflow-x: hidden !important;
            width: 100% !important;
            height: 100% !important;
            max-height: none !important;
            position: relative !important;
        }
        .radar-tip {
            font-size: 14px;
            color: #555;
            margin-top: 5px;
            margin-bottom: 20px;
            font-style: italic;
        }

        
        /* Force scrolling to work on the main container */
        .gradio-container, #root, #app {
            width: 100% !important;
            max-width: 1200px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            min-height: auto !important;
            height: auto !important;
            overflow: visible !important;
            position: relative !important;
        }
        
        /* Remove ALL inner scrollbars - very important! */
        .gradio-container * {
            scrollbar-width: none !important;  /* Firefox */
        }
        
        /* Hide scrollbars for Chrome, Safari and Opera */
        .gradio-container *::-webkit-scrollbar {
            display: none !important;
        }
        
        /* Only allow scrollbar on body */
        body::-webkit-scrollbar {
            display: block !important;
            width: 10px !important;
        }
        
        body::-webkit-scrollbar-track {
            background: #f1f1f1 !important;
        }
        
        body::-webkit-scrollbar-thumb {
            background: #888 !important;
            border-radius: 5px !important;
        }
        
        body::-webkit-scrollbar-thumb:hover {
            background: #555 !important;
        }
        
        /* Clean up table styling */
        .table-container {
            width: 100% !important;
            overflow: hidden !important;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        /* Remove duplicate scrollbars */
        .gradio-dataframe [data-testid="table"],
        [data-testid="dataframe"] [data-testid="table"],
        .gradio-dataframe tbody,
        [data-testid="dataframe"] tbody,
        .table-container > div,
        .table-container > div > div {
            overflow: hidden !important;
            /* max-height: none !important; */ /* REMOVED */
        }
        
        /* Ensure table contents are visible without scrollbars */
        .gradio-dataframe,
        [data-testid="dataframe"] {
            overflow: visible !important;
            /* max-height: none !important; */ /* REMOVED */
            border: none !important;
        }
        
        /* Visualization styling */
        .visualization-container .js-plotly-plot {
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
            max-width: 1000px;
        }
        
        /* Section styling */
        .section-title {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
            text-align: center;
        }
        
        /* Fix table styling */
        .table-container table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            table-layout: fixed !important;
        }
        
        /* Column width customization - adjust for row numbers being first column */
        .table-container th:nth-child(2), 
        .table-container td:nth-child(2) {
            width: 230px !important;
            min-width: 200px !important;
            max-width: 280px !important;
            padding-left: 8px !important;
            padding-right: 8px !important;
        }
        
        .table-container th:nth-child(3), 
        .table-container td:nth-child(3) {
            width: 120px !important;
            min-width: 100px !important;
            max-width: 140px !important;
        }
        
        /* Game score columns */
        .table-container th:nth-child(n+4), 
        .table-container td:nth-child(n+4) {
            width: 120px !important;
            min-width: 100px !important;
            max-width: 140px !important;
            text-align: center !important;
        }
        
        /* Make headers sticky */
        .table-container th {
            position: sticky !important;
            top: 0 !important;
            background-color: var(--header-bg, #f8f9fa) !important;
            z-index: 10 !important;
            font-weight: bold;
            padding: 16px 10px !important;
            border-bottom: 2px solid var(--border-color, #e9ecef);
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            line-height: 1.2 !important;
            height: auto !important;
            min-height: 60px !important;
            vertical-align: middle !important;
            color: var(--header-text, #2c3e50) !important;
        }
        
        /* Dark mode specific styles */
        .dark .table-container th {
            --header-bg: #2d3748;
            --header-text: #e2e8f0;
            --border-color: #4a5568;
        }
        
        /* Light mode specific styles */
        .light .table-container th {
            --header-bg: #f8f9fa;
            --header-text: #2c3e50;
            --border-color: #e9ecef;
        }
        
        /* Simple cell styling */
        .table-container td {
            padding: 8px 8px;
            border-bottom: 1px solid var(--border-color, #e9ecef);
        }
        
        /* Row number column styling */
        .gradio-dataframe thead tr th[id="0"],
        .gradio-dataframe tbody tr td:nth-child(1),
        [data-testid="dataframe"] thead tr th[id="0"],
        [data-testid="dataframe"] tbody tr td:nth-child(1),
        .svelte-1gfkn6j thead tr th:first-child,
        .svelte-1gfkn6j tbody tr td:first-child {
            width: 40px !important;
            min-width: 40px !important;
            max-width: 40px !important;
            padding: 4px !important;
            text-align: center !important;
            font-size: 0.85em !important;
        }
        
        /* Fix for Gradio footer causing scroll issues */
        footer {
            position: relative !important;
            width: 100% !important;
            margin-top: 40px !important;
        }
    """) as demo:
        gr.Markdown("# 🎮 Lmgame Bench: Leaderboard 🎲")
        
        # Add custom JavaScript for table header line breaks
        gr.HTML("""
        
        <script>
        // Function to add line breaks to table headers
        function formatTableHeaders() {
            // Find all table headers in the document
            const headers = document.querySelectorAll('th');
            
            headers.forEach(header => {
                let text = header.textContent || '';
                
                // Skip if already processed
                if (header.getAttribute('data-processed') === 'true') {
                    return;
                }
                
                // Store original content for reference
                if (!header.getAttribute('data-original')) {
                    header.setAttribute('data-original', header.innerHTML);
                }
                
                let newContent = header.innerHTML;
                
                // Format Super Mario Bros header
                if (text.includes('Super Mario Bros')) {
                    newContent = newContent.replace(/Super\s+Mario\s+Bros/g, 'Super<br>Mario Bros');
                }
                
                // Format Tetris headers
                if (text.includes('Tetris (complete)')) {
                    newContent = newContent.replace(/Tetris\s+\(complete\)/g, 'Tetris<br>(complete)');
                }
                
                if (text.includes('Tetris (planning only)')) {
                    newContent = newContent.replace(/Tetris\s+\(planning\s+only\)/g, 'Tetris<br>(planning)');
                }
                
                // Format Candy Crush header
                if (text.includes('Candy Crush')) {
                    newContent = newContent.replace(/Candy\s+Crash/g, 'Candy<br>Crash');
                }
                
                // Make Organization header wider and fix its name
                if (text.includes('Organization') || text.includes('Organi-zation')) {
                    header.style.minWidth = '150px';
                    header.style.width = '150px';
                    
                    // Fix the Organization header name if it has a line break
                    if (text.includes('Organi-') || text.includes('zation')) {
                        newContent = newContent.replace(/Organi-<br>zation|Organi-zation/, 'Organization');
                    }
                }
                
                // Update content if changed
                if (newContent !== header.innerHTML) {
                    header.innerHTML = newContent;
                    header.setAttribute('data-processed', 'true');
                    
                    // Also ensure headers have proper styling
                    header.style.whiteSpace = 'normal';
                    header.style.lineHeight = '1.2';
                    header.style.verticalAlign = 'middle';
                    header.style.minHeight = '70px';
                    header.style.fontSize = '0.9em';
                }
            });
        }
        
        // Function to fix player name cells to prevent line breaking
        function fixPlayerCells() {
            // Find all table cells in the document
            const tables = document.querySelectorAll('table');
            
            tables.forEach(table => {
                // Process rows starting from index 1 (skip header)
                const rows = table.querySelectorAll('tr');
                
                rows.forEach((row, index) => {
                    // Skip header row
                    if (index === 0) return;
                    
                    // Get the player cell (typically 2nd cell)
                    const playerCell = row.querySelector('td:nth-child(2)');
                    const orgCell = row.querySelector('td:nth-child(3)');
                    
                    if (playerCell) {
                        playerCell.style.whiteSpace = 'nowrap';
                        playerCell.style.overflow = 'hidden';
                        playerCell.style.textOverflow = 'ellipsis';
                        playerCell.style.maxWidth = '230px';
                        playerCell.style.textAlign = 'left';
                    }
                    
                    if (orgCell) {
                        orgCell.style.whiteSpace = 'nowrap';
                        orgCell.style.overflow = 'hidden';
                        orgCell.style.textOverflow = 'ellipsis';
                        orgCell.style.minWidth = '150px';
                        orgCell.style.width = '150px';
                    }
                });
            });
        }
        
        // Function to run all formatting
        function formatTable() {
            formatTableHeaders();
            fixPlayerCells();
        }
        
        // Run on load and then periodically to catch any new tables
        setInterval(formatTable, 500);
        
        // Also run when the DOM content is loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', formatTable);
        } else {
            formatTable();
        }
        
        // Run when the page is fully loaded with resources
        window.addEventListener('load', formatTable);
        </script>
        """)
        
        with gr.Tabs():
            with gr.Tab("🏆 Overall Leaderboard"):
                # Visualization section
                with gr.Row():
                    gr.Markdown("### 📊 Data Visualization")
                
                # Detailed view visualization (single chart)
                detailed_visualization = gr.Plot(
                    label="Performance Visualization",
                    visible=False,
                    elem_classes="visualization-container"
                )
                
                with gr.Column(visible=True) as overall_visualizations:
                    with gr.Tabs():
                        with gr.Tab("📈 Radar Chart"):
                            
                            radar_visualization = gr.Plot(
                                label="Comparative Analysis (Radar Chart)",
                                elem_classes="visualization-container"
                            )
                            gr.Markdown(
                                    "*💡 Click a legend entry to isolate that model. Double-click additional ones to add them for comparison.*",
                                    elem_classes="radar-tip"
                                )
                        # Comment out the Group Bar Chart tab
                        with gr.Tab("📊 Group Bar Chart"):
                            group_bar_visualization = gr.Plot(
                                label="Comparative Analysis (Group Bar Chart)",
                                elem_classes="visualization-container"
                            )
                            

                # Hidden placeholder for group bar visualization (to maintain code references)
                # group_bar_visualization = gr.Plot(visible=False)

                # Game selection section
                with gr.Row():
                    gr.Markdown("### 🎮 Game Selection")
                with gr.Row():
                    # with gr.Column(): # Commented out Super Mario Bros UI
                    #     gr.Markdown("**🎮 Super Mario Bros**")
                    #     mario_overall = gr.Checkbox(label="Super Mario Bros Score", value=True)
                    #     mario_details = gr.Checkbox(label="Super Mario Bros Details", value=False)
                    with gr.Column(): # Added Super Mario Bros (planning only) UI
                        gr.Markdown("**📝 Super Mario Bros (planning only)**")
                        mario_plan_overall = gr.Checkbox(label="Super Mario Bros (planning only) Score", value=True)
                        mario_plan_details = gr.Checkbox(label="Super Mario Bros (planning only) Details", value=False)
                    with gr.Column(): # Sokoban is now after mario_plan
                        gr.Markdown("**📦 Sokoban**")
                        sokoban_overall = gr.Checkbox(label="Sokoban Score", value=True)
                        sokoban_details = gr.Checkbox(label="Sokoban Details", value=False)
                    with gr.Column():
                        gr.Markdown("**🔢 2048**")
                        _2048_overall = gr.Checkbox(label="2048 Score", value=True)
                        _2048_details = gr.Checkbox(label="2048 Details", value=False)
                    with gr.Column():
                        gr.Markdown("**🍬 Candy Crush**")
                        candy_overall = gr.Checkbox(label="Candy Crush Score", value=True)
                        candy_details = gr.Checkbox(label="Candy Crush Details", value=False)
                    # with gr.Column(): # Commented out Tetris (complete) UI
                    #     gr.Markdown("**🎯 Tetris (complete)**")
                    #     tetris_overall = gr.Checkbox(label="Tetris (complete) Score", value=True)
                    #     tetris_details = gr.Checkbox(label="Tetris (complete) Details", value=False)
                    with gr.Column():
                        gr.Markdown("**📋 Tetris (planning)**")
                        tetris_plan_overall = gr.Checkbox(label="Tetris (planning) Score", value=True)
                        tetris_plan_details = gr.Checkbox(label="Tetris (planning) Details", value=False)
                    with gr.Column():
                        gr.Markdown("**⚖️ Ace Attorney**")
                        ace_attorney_overall = gr.Checkbox(label="Ace Attorney Score", value=True)
                        ace_attorney_details = gr.Checkbox(label="Ace Attorney Details", value=False)
                
                # Controls
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("**⏰ Time Tracker**")
                        timeline = create_timeline_slider()
                    with gr.Column(scale=1):
                        gr.Markdown("**🔄 Controls**")
                        clear_btn = gr.Button("Reset Filters", variant="secondary")
                
                # Leaderboard table
                with gr.Row():
                    gr.Markdown("### 📋 Detailed Results")
                
                # Add reference to Jupyter notebook
                with gr.Row():
                    gr.Markdown("*All data analysis can be replicated by checking [this Jupyter notebook](https://colab.research.google.com/drive/1CYFiJGm3EoBXXI8vICPVR82J9qrmmRvc#scrollTo=qft1Oald-21J)*")
                
                # Get initial leaderboard dataframe
                initial_df = get_combined_leaderboard(rank_data, {
                    # "Super Mario Bros": True, # Commented out
                    "Super Mario Bros (planning only)": True,
                    "Sokoban": True,
                    "2048": True,
                    "Candy Crush": True,
                    # "Tetris (complete)": True, # Commented out
                    "Tetris (planning only)": True,
                    "Ace Attorney": True
                })
                
                # Format the DataFrame for display
                initial_display_df = prepare_dataframe_for_display(initial_df)
                
                # Custom column widths including row numbers
                col_widths = ["40px"]  # Row number column width
                col_widths.append("230px")  # Player column - reduced by 20px
                col_widths.append("120px")  # Organization column
                # Add game score columns
                for _ in range(len(initial_display_df.columns) - 2):
                    col_widths.append("120px")
                
                # Create a standard DataFrame component with enhanced styling
                with gr.Row():
                    leaderboard_df = gr.DataFrame(
                        value=initial_display_df,
                        interactive=True,
                        elem_id="leaderboard-table",
                        elem_classes="table-container",
                        wrap=True,
                        show_row_numbers=True,
                        show_fullscreen_button=True,
                        line_breaks=True,
                        max_height=1000,  # Set a larger fixed height
                        show_search="search",
                        column_widths=col_widths
                    )
                
                # Add the score note below the table
                with gr.Row():
                    score_note = add_score_note()
                
                # List of all checkboxes, including Super Mario Bros (planning only)
                checkbox_list = [
                    # mario_overall, mario_details, # Commented out
                    mario_plan_overall, mario_plan_details,
                    sokoban_overall, sokoban_details,
                    _2048_overall, _2048_details,
                    candy_overall, candy_details,
                    # tetris_overall, tetris_details, # Commented out
                    tetris_plan_overall, tetris_plan_details,
                    ace_attorney_overall, ace_attorney_details
                ]
                
                # Update visualizations when checkboxes change
                def update_visualizations(*checkbox_states):
                    # Check if any details checkbox is selected
                    # Adjusted indices due to addition of Super Mario (planning only)
                    is_details_view = any([
                        checkbox_states[1], # Mario Plan details
                        checkbox_states[3], # Sokoban details
                        checkbox_states[5], # 2048 details
                        checkbox_states[7], # Candy Crush details
                        checkbox_states[9], # Tetris (planning only) details
                        checkbox_states[11]  # Ace Attorney details
                    ])
                    
                    # Update visibility of visualization blocks
                    return {
                        detailed_visualization: gr.update(visible=is_details_view),
                        overall_visualizations: gr.update(visible=not is_details_view)
                    }
                
                # Add change event to all checkboxes
                for checkbox in checkbox_list:
                    checkbox.change(
                        update_visualizations,
                        inputs=checkbox_list,
                        outputs=[detailed_visualization, overall_visualizations]
                    )
                
                # Update leaderboard and visualizations when checkboxes change
                for checkbox in checkbox_list:
                    checkbox.change(
                        lambda *args: update_leaderboard(*args, data_source=rank_data),
                        inputs=checkbox_list,
                        outputs=[
                            leaderboard_df,
                            detailed_visualization,
                            radar_visualization,
                            group_bar_visualization
                        ] + checkbox_list
                    )
                
                # Update when clear button is clicked
                clear_btn.click(
                    lambda: clear_filters(data_source=rank_data),
                    inputs=[],
                    outputs=[
                        leaderboard_df,
                        detailed_visualization,
                        radar_visualization,
                        group_bar_visualization
                    ] + checkbox_list
                )
                
                # Initialize the app
                demo.load(
                    lambda: clear_filters(data_source=rank_data),
                    inputs=[],
                    outputs=[
                        leaderboard_df,
                        detailed_visualization,
                        radar_visualization,
                        group_bar_visualization
                    ] + checkbox_list
                )
            
            with gr.Tab("🤖 Model Leaderboard"):
                # Visualization section
                with gr.Row():
                    gr.Markdown("### 📊 Data Visualization")
                
                # Detailed view visualization (single chart)
                model_detailed_visualization = gr.Plot(
                    label="Performance Visualization",
                    visible=False,
                    elem_classes="visualization-container"
                )
                
                with gr.Column(visible=True) as model_overall_visualizations:
                    with gr.Tabs():
                        with gr.Tab("📈 Radar Chart"):
                            model_radar_visualization = gr.Plot(
                                label="Comparative Analysis (Radar Chart)",
                                elem_classes="visualization-container"
                            )
                            gr.Markdown(
                                    "*💡 Click a legend entry to isolate that model. Double-click additional ones to add them for comparison.*",
                                    elem_classes="radar-tip"
                                )
                        with gr.Tab("📊 Group Bar Chart"):
                            model_group_bar_visualization = gr.Plot(
                                label="Comparative Analysis (Group Bar Chart)",
                                elem_classes="visualization-container"
                            )

                # Game selection section
                with gr.Row():
                    gr.Markdown("### 🎮 Game Selection")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**📝 Super Mario Bros (planning only)**")
                        model_mario_plan_overall = gr.Checkbox(label="Super Mario Bros (planning only) Score", value=True)
                        model_mario_plan_details = gr.Checkbox(label="Super Mario Bros (planning only) Details", value=False)
                    with gr.Column():
                        gr.Markdown("**📦 Sokoban**")
                        model_sokoban_overall = gr.Checkbox(label="Sokoban Score", value=True)
                        model_sokoban_details = gr.Checkbox(label="Sokoban Details", value=False)
                    with gr.Column():
                        gr.Markdown("**🔢 2048**")
                        model_2048_overall = gr.Checkbox(label="2048 Score", value=True)
                        model_2048_details = gr.Checkbox(label="2048 Details", value=False)
                    with gr.Column():
                        gr.Markdown("**🍬 Candy Crush**")
                        model_candy_overall = gr.Checkbox(label="Candy Crush Score", value=True)
                        model_candy_details = gr.Checkbox(label="Candy Crush Details", value=False)
                    with gr.Column():
                        gr.Markdown("**📋 Tetris (planning)**")
                        model_tetris_plan_overall = gr.Checkbox(label="Tetris (planning) Score", value=True)
                        model_tetris_plan_details = gr.Checkbox(label="Tetris (planning) Details", value=False)
                    with gr.Column():
                        gr.Markdown("**⚖️ Ace Attorney**")
                        model_ace_attorney_overall = gr.Checkbox(label="Ace Attorney Score", value=True)
                        model_ace_attorney_details = gr.Checkbox(label="Ace Attorney Details", value=False)
                
                # Controls
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("**⏰ Time Tracker**")
                        model_timeline = create_timeline_slider()
                    with gr.Column(scale=1):
                        gr.Markdown("**🔄 Controls**")
                        model_clear_btn = gr.Button("Reset Filters", variant="secondary")
                
                # Leaderboard table
                with gr.Row():
                    gr.Markdown("### 📋 Detailed Results")
                
                # Get initial leaderboard dataframe
                model_initial_df = get_combined_leaderboard(model_rank_data, {
                    "Super Mario Bros (planning only)": True,
                    "Sokoban": True,
                    "2048": True,
                    "Candy Crush": True,
                    "Tetris (planning only)": True,
                    "Ace Attorney": True
                })
                
                # Format the DataFrame for display
                model_initial_display_df = prepare_dataframe_for_display(model_initial_df)
                
                # Create a standard DataFrame component with enhanced styling
                with gr.Row():
                    model_leaderboard_df = gr.DataFrame(
                        value=model_initial_display_df,
                        interactive=True,
                        elem_id="model-leaderboard-table",
                        elem_classes="table-container",
                        wrap=True,
                        show_row_numbers=True,
                        show_fullscreen_button=True,
                        line_breaks=True,
                        max_height=1000,
                        show_search="search",
                        column_widths=col_widths
                    )
                
                # Add the score note below the table
                with gr.Row():
                    model_score_note = add_score_note()
                
                # List of all checkboxes for model leaderboard
                model_checkbox_list = [
                    model_mario_plan_overall, model_mario_plan_details,
                    model_sokoban_overall, model_sokoban_details,
                    model_2048_overall, model_2048_details,
                    model_candy_overall, model_candy_details,
                    model_tetris_plan_overall, model_tetris_plan_details,
                    model_ace_attorney_overall, model_ace_attorney_details
                ]
                
                # Update visualizations when checkboxes change
                def update_model_visualizations(*checkbox_states):
                    # Check if any details checkbox is selected
                    is_details_view = any([
                        checkbox_states[1], # Mario Plan details
                        checkbox_states[3], # Sokoban details
                        checkbox_states[5], # 2048 details
                        checkbox_states[7], # Candy Crush details
                        checkbox_states[9], # Tetris (planning only) details
                        checkbox_states[11]  # Ace Attorney details
                    ])
                    
                    # Update visibility of visualization blocks
                    return {
                        model_detailed_visualization: gr.update(visible=is_details_view),
                        model_overall_visualizations: gr.update(visible=not is_details_view)
                    }
                
                # Add change event to all checkboxes
                for checkbox in model_checkbox_list:
                    checkbox.change(
                        update_model_visualizations,
                        inputs=model_checkbox_list,
                        outputs=[model_detailed_visualization, model_overall_visualizations]
                    )
                
                # Update leaderboard and visualizations when checkboxes change
                for checkbox in model_checkbox_list:
                    checkbox.change(
                        lambda *args: update_leaderboard(*args, data_source=model_rank_data),
                        inputs=model_checkbox_list,
                        outputs=[
                            model_leaderboard_df,
                            model_detailed_visualization,
                            model_radar_visualization,
                            model_group_bar_visualization
                        ] + model_checkbox_list
                    )
                
                # Update when clear button is clicked
                model_clear_btn.click(
                    lambda: clear_filters(data_source=model_rank_data),
                    inputs=[],
                    outputs=[
                        model_leaderboard_df,
                        model_detailed_visualization,
                        model_radar_visualization,
                        model_group_bar_visualization
                    ] + model_checkbox_list
                )
                
                # Initialize the model leaderboard
                demo.load(
                    lambda: clear_filters(data_source=model_rank_data),
                    inputs=[],
                    outputs=[
                        model_leaderboard_df,
                        model_detailed_visualization,
                        model_radar_visualization,
                        model_group_bar_visualization
                    ] + model_checkbox_list
                )
            
            with gr.Tab("🎥 Gallery"):
                video_gallery = create_video_gallery()
    
    return demo

if __name__ == "__main__":
    demo_app = build_app()
    # Add file serving configuration
    demo_app.launch(
        debug=True, 
        show_error=True, 
        share=True,
        height="100%",
        width="100%"
    )