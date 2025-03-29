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
    get_mario_leaderboard,
    get_sokoban_leaderboard,
    get_2048_leaderboard,
    get_candy_leaderboard,
    get_tetris_leaderboard,
    get_tetris_planning_leaderboard,
    get_combined_leaderboard,
    GAME_ORDER
)
from data_visualization import (
    get_combined_leaderboard_with_radar,
    create_organization_radar_chart,
    create_top_players_radar_chart,
    create_player_radar_chart,
    create_horizontal_bar_chart
)

# Define time points and their corresponding data files
TIME_POINTS = {
    "03/25/2025": "rank_data_03_25_2025.json",
    # Add more time points here as they become available
}

# Load the initial JSON file with rank data
with open(TIME_POINTS["03/25/2025"], "r") as f:
    rank_data = json.load(f)

# Add leaderboard state at the top level
leaderboard_state = {
    "current_game": None,
    "previous_overall": {
        "Super Mario Bros": True,
        "Sokoban": True,
        "2048": True,
        "Candy Crash": True,
        "Tetris (complete)": True,
        "Tetris (planning only)": True
    },
    "previous_details": {
        "Super Mario Bros": False,
        "Sokoban": False,
        "2048": False,
        "Candy Crash": False,
        "Tetris (complete)": False,
        "Tetris (planning only)": False
    }
}

# Define GIF paths for the carousel
GIF_PATHS = [
    "assets/sokoban/sokoban_sample.gif",
    "assets/super_mario_bros/super_mario_sample.gif"
]

# Print and verify GIF paths
print("\nChecking GIF paths:")
for gif_path in GIF_PATHS:
    if os.path.exists(gif_path):
        print(f"✓ Found: {gif_path}")
        # Print file size
        size = os.path.getsize(gif_path)
        print(f"  Size: {size / (1024*1024):.2f} MB")
    else:
        print(f"✗ Missing: {gif_path}")

def load_gif(gif_path):
    """Load a GIF file and return it as a PIL Image"""
    try:
        img = Image.open(gif_path)
        print(f"Successfully loaded GIF: {gif_path}")
        return img
    except Exception as e:
        print(f"Error loading GIF {gif_path}: {e}")
        return None

def create_gif_carousel():
    """Create a custom HTML/JS component for GIF carousel"""
    print("\nCreating GIF carousel with paths:", GIF_PATHS)
    html = f"""
    <div id="gif-carousel" style="width: 100%; height: 300px; position: relative; background-color: #f0f0f0;">
        <img id="current-gif" style="width: 100%; height: 100%; object-fit: contain;" onerror="console.error('Failed to load GIF:', this.src);">
    </div>
    <script>
        const gifs = {json.dumps(GIF_PATHS)};
        let currentIndex = 0;
        
        function updateGif() {{
            const img = document.getElementById('current-gif');
            console.log('Loading GIF:', gifs[currentIndex]);
            img.src = gifs[currentIndex];
            currentIndex = (currentIndex + 1) % gifs.length;
        }}
        
        // Update GIF every 5 seconds
        setInterval(updateGif, 5000);
        // Initial load
        updateGif();
    </script>
    """
    return gr.HTML(html)

def load_rank_data(time_point):
    """Load rank data for a specific time point"""
    if time_point in TIME_POINTS:
        try:
            with open(TIME_POINTS[time_point], "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    return None

def update_leaderboard(mario_overall, mario_details,
                       sokoban_overall, sokoban_details,
                       _2048_overall, _2048_details,
                       candy_overall, candy_details,
                       tetris_overall, tetris_details,
                       tetris_plan_overall, tetris_plan_details):
    global leaderboard_state
    
    # Convert current checkbox states to dictionary for easier comparison
    current_overall = {
        "Super Mario Bros": mario_overall,
        "Sokoban": sokoban_overall,
        "2048": _2048_overall,
        "Candy Crash": candy_overall,
        "Tetris (complete)": tetris_overall,
        "Tetris (planning only)": tetris_plan_overall
    }
    
    current_details = {
        "Super Mario Bros": mario_details,
        "Sokoban": sokoban_details,
        "2048": _2048_details,
        "Candy Crash": candy_details,
        "Tetris (complete)": tetris_details,
        "Tetris (planning only)": tetris_plan_details
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
            leaderboard_state["previous_overall"][changed_game] = True  # Set overall to True when details is checked
            leaderboard_state["previous_details"][changed_game] = True
            current_overall[changed_game] = True  # Ensure the overall checkbox is checked
        
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
    
    # Build dictionary for selected games
    selected_games = {
        "Super Mario Bros": current_overall["Super Mario Bros"],
        "Sokoban": current_overall["Sokoban"],
        "2048": current_overall["2048"],
        "Candy Crash": current_overall["Candy Crash"],
        "Tetris (complete)": current_overall["Tetris (complete)"],
        "Tetris (planning only)": current_overall["Tetris (planning only)"]
    }
    
    # Get the appropriate DataFrame and chart based on current state
    if leaderboard_state["current_game"]:
        # For detailed view
        if leaderboard_state["current_game"] == "Super Mario Bros":
            df = get_mario_leaderboard(rank_data)
        elif leaderboard_state["current_game"] == "Sokoban":
            df = get_sokoban_leaderboard(rank_data)
        elif leaderboard_state["current_game"] == "2048":
            df = get_2048_leaderboard(rank_data)
        elif leaderboard_state["current_game"] == "Candy Crash":
            df = get_candy_leaderboard(rank_data)
        elif leaderboard_state["current_game"] == "Tetris (complete)":
            df = get_tetris_leaderboard(rank_data)
        else:  # Tetris (planning only)
            df = get_tetris_planning_leaderboard(rank_data)
        
        # Always create a new chart for detailed view
        chart = create_horizontal_bar_chart(df, leaderboard_state["current_game"])
    else:
        # For overall view
        df = get_combined_leaderboard(rank_data, selected_games)
        _, chart = get_combined_leaderboard_with_radar(rank_data, selected_games)
    
    return (df, chart,
            current_overall["Super Mario Bros"], current_details["Super Mario Bros"],
            current_overall["Sokoban"], current_details["Sokoban"],
            current_overall["2048"], current_details["2048"],
            current_overall["Candy Crash"], current_details["Candy Crash"],
            current_overall["Tetris (complete)"], current_details["Tetris (complete)"],
            current_overall["Tetris (planning only)"], current_details["Tetris (planning only)"])

def update_leaderboard_with_time(time_point, mario_overall, mario_details,
                               sokoban_overall, sokoban_details,
                               _2048_overall, _2048_details,
                               candy_overall, candy_details,
                               tetris_overall, tetris_details,
                               tetris_plan_overall, tetris_plan_details):
    # Load rank data for the selected time point
    global rank_data
    new_rank_data = load_rank_data(time_point)
    if new_rank_data is not None:
        rank_data = new_rank_data
    
    # Use the existing update_leaderboard function
    return update_leaderboard(mario_overall, mario_details,
                            sokoban_overall, sokoban_details,
                            _2048_overall, _2048_details,
                            candy_overall, candy_details,
                            tetris_overall, tetris_details,
                            tetris_plan_overall, tetris_plan_details)

def clear_filters():
    global leaderboard_state
    
    # Reset all checkboxes to default state and get fresh data
    df = get_combined_leaderboard(rank_data, {
        "Super Mario Bros": True,
        "Sokoban": True,
        "2048": True,
        "Candy Crash": True,
        "Tetris (complete)": True,
        "Tetris (planning only)": True
    })
    
    # Get the radar chart visualization
    _, chart = get_combined_leaderboard_with_radar(rank_data, {
        "Super Mario Bros": True,
        "Sokoban": True,
        "2048": True,
        "Candy Crash": True,
        "Tetris (complete)": True,
        "Tetris (planning only)": True
    })
    
    # Reset the leaderboard state to match the default checkbox states
    leaderboard_state = {
        "current_game": None,
        "previous_overall": {
            "Super Mario Bros": True,
            "Sokoban": True,
            "2048": True,
            "Candy Crash": True,
            "Tetris (complete)": True,
            "Tetris (planning only)": True
        },
        "previous_details": {
            "Super Mario Bros": False,
            "Sokoban": False,
            "2048": False,
            "Candy Crash": False,
            "Tetris (complete)": False,
            "Tetris (planning only)": False
        }
    }
    
    # Return both the DataFrame and the visualization
    return (df, chart,
            True, False,  # mario
            True, False,  # sokoban
            True, False,  # 2048
            True, False,  # candy
            True, False,  # tetris
            True, False)  # tetris plan

def build_app():
    with gr.Blocks() as demo:
        gr.Markdown("# 🎮 Game Arena: Gaming Agent 🎲")
        
        with gr.Tabs():
            with gr.Tab("🏆 Leaderboard"):
                # Visualization section at the very top
                with gr.Row():
                    gr.Markdown("### 📊 Data Visualization")
                with gr.Row():
                    # Split into two columns
                    with gr.Column(scale=3):  # Changed from 1 to 3 to make demos larger
                        # GIF gallery on the left
                        gr.Gallery(
                            value=GIF_PATHS,
                            label="Game Demos",
                            show_label=True,
                            elem_id="gallery",
                            columns=1,
                            rows=1,
                            min_width=400,  # Control size through min_width
                            allow_preview=True,
                            object_fit='contain',  # Ensure proper scaling
                            show_download_button=False,  # Hide download button to keep it clean
                            show_share_button=False,  # Hide share button to keep it clean
                            show_fullscreen_button=True  # Allow fullscreen for better viewing
                        )
                    with gr.Column(scale=4):  # Changed from 2 to 4, but maintains a smaller ratio compared to demos
                        # Radar charts on the right
                        visualization = gr.Plot(
                            value=get_combined_leaderboard_with_radar(rank_data, {
                                "Super Mario Bros": True,
                                "Sokoban": True,
                                "2048": True,
                                "Candy Crash": True,
                                "Tetris (complete)": True,
                                "Tetris (planning only)": True
                            })[1],
                            label="Performance Visualization"
                        )

                # Time progression display and control buttons
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("**⏰ Time Tracker**")
                        time_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=1,
                            step=1,
                            label="Model Time Point",
                            info="Current Time: 03/25/2025"
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("**Controls**")
                        clear_btn = gr.Button("🔄 Reset Filters", variant="secondary")

                # Game selection section
                with gr.Row():
                    gr.Markdown("### 🎮 Game Selection")
                with gr.Row():
                    # For each game, we have two checkboxes: one for overall and one for detailed view.
                    with gr.Column():
                        gr.Markdown("**🎮 Super Mario Bros**")
                        mario_overall = gr.Checkbox(label="Super Mario Bros Score", value=True)
                        mario_details = gr.Checkbox(label="Super Mario Bros Details", value=False)
                    with gr.Column():
                        gr.Markdown("**📦 Sokoban**")
                        sokoban_overall = gr.Checkbox(label="Sokoban Score", value=True)
                        sokoban_details = gr.Checkbox(label="Sokoban Details", value=False)
                    with gr.Column():
                        gr.Markdown("**🔢 2048**")
                        _2048_overall = gr.Checkbox(label="2048 Score", value=True)
                        _2048_details = gr.Checkbox(label="2048 Details", value=False)
                    with gr.Column():
                        gr.Markdown("**🍬 Candy Crash**")
                        candy_overall = gr.Checkbox(label="Candy Crash Score", value=True)
                        candy_details = gr.Checkbox(label="Candy Crash Details", value=False)
                    with gr.Column():
                        gr.Markdown("**🎯 Tetris (complete)**")
                        tetris_overall = gr.Checkbox(label="Tetris (complete) Score", value=True)
                        tetris_details = gr.Checkbox(label="Tetris (complete) Details", value=False)
                    with gr.Column():
                        gr.Markdown("**📋 Tetris (planning)**")
                        tetris_plan_overall = gr.Checkbox(label="Tetris (planning) Score", value=True)
                        tetris_plan_details = gr.Checkbox(label="Tetris (planning) Details", value=False)

                # Leaderboard table section
                with gr.Row():
                    gr.Markdown("### 📋 Detailed Results")
                with gr.Row():
                    leaderboard_board = gr.DataFrame(
                        value=get_combined_leaderboard(rank_data, {
                            "Super Mario Bros": True,
                            "Sokoban": True,
                            "2048": True,
                            "Candy Crash": True,
                            "Tetris (complete)": True,
                            "Tetris (planning only)": True
                        }),
                        interactive=True,
                        wrap=True,
                        label="Leaderboard"
                    )

                # List of all checkboxes (in order)
                checkbox_list = [mario_overall, mario_details,
                                sokoban_overall, sokoban_details,
                                _2048_overall, _2048_details,
                                candy_overall, candy_details,
                                tetris_overall, tetris_details,
                                tetris_plan_overall, tetris_plan_details]

                # Initialize the leaderboard state when the app starts
                clear_filters()

                # Update both the leaderboard and visualization when checkboxes change
                for checkbox in checkbox_list:
                    checkbox.change(
                        fn=update_leaderboard,
                        inputs=checkbox_list,
                        outputs=[leaderboard_board, visualization] + checkbox_list
                    )

                # Update both when clear button is clicked
                clear_btn.click(
                    fn=clear_filters,
                    inputs=[],
                    outputs=[leaderboard_board, visualization] + checkbox_list
                )

    return demo

if __name__ == "__main__":
    demo_app = build_app()
    # Add file serving configuration
    demo_app.launch(debug=True, show_error=True, share=False)
