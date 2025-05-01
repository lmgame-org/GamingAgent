import gradio as gr
import json
from leaderboard_utils import (
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
    get_combined_leaderboard_with_group_bar,
    create_horizontal_bar_chart,
    get_combined_leaderboard_with_single_radar
)
import pandas as pd

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
                # When exiting details view, reset to show all games
                for game in current_overall.keys():
                    current_overall[game] = True
                    current_details[game] = False
                    leaderboard_state["previous_overall"][game] = True
                    leaderboard_state["previous_details"][game] = False
    
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
        "Super Mario Bros": current_overall["Super Mario Bros"],
        "Sokoban": current_overall["Sokoban"],
        "2048": current_overall["2048"],
        "Candy Crash": current_overall["Candy Crash"],
        "Tetris (complete)": current_overall["Tetris (complete)"],
        "Tetris (planning only)": current_overall["Tetris (planning only)"]
    }
    
    # Get the appropriate DataFrame and charts based on current state
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
        # For detailed view, we'll use the same chart for all visualizations
        radar_chart = chart
        group_bar_chart = chart
    else:
        # For overall view
        df, group_bar_chart = get_combined_leaderboard_with_group_bar(rank_data, selected_games)
        # Use the same selected_games for radar chart
        _, radar_chart = get_combined_leaderboard_with_single_radar(rank_data, selected_games)
        chart = group_bar_chart
    
    # Return exactly 16 values to match the expected outputs
    return (df, chart, radar_chart, group_bar_chart,
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

def get_initial_state():
    """Get the initial state for the leaderboard"""
    return {
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

def clear_filters():
    global leaderboard_state
    
    # Reset all checkboxes to default state
    selected_games = {
        "Super Mario Bros": True,
        "Sokoban": True,
        "2048": True,
        "Candy Crash": True,
        "Tetris (complete)": True,
        "Tetris (planning only)": True
    }
    
    # Get the combined leaderboard and group bar chart
    df, group_bar_chart = get_combined_leaderboard_with_group_bar(rank_data, selected_games)
    
    # Get the radar chart using the same selected games
    _, radar_chart = get_combined_leaderboard_with_single_radar(rank_data, selected_games)
    
    # Reset the leaderboard state to match the default checkbox states
    leaderboard_state = get_initial_state()
    
    # Return exactly 16 values to match the expected outputs
    return (df, group_bar_chart, radar_chart, group_bar_chart,
            True, False,  # mario
            True, False,  # sokoban
            True, False,  # 2048
            True, False,  # candy
            True, False,  # tetris
            True, False)  # tetris plan

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

def create_leaderboard_tab():
    """Create and return the leaderboard tab component"""
    with gr.Tab("🏆 Leaderboard") as leaderboard_tab:
        # Leaderboard header
        with gr.Row():
            gr.Markdown("### 📊 Leaderboard Overview")
        
        # Get initial data
        df = get_combined_leaderboard(rank_data, {game: True for game in GAME_ORDER})
        
        # Create interactive DataFrame component
        leaderboard_df = gr.DataFrame(
            value=df,
            label="Leaderboard",
            interactive=True,  # Enable sorting and filtering
            wrap=True,  # Enable text wrapping
            column_widths=["200px", "150px"] + ["100px"] * len(GAME_ORDER),  # Set column widths
            headers=["Model", "Organization"] + GAME_ORDER,  # Set column headers
            datatype=["str", "str"] + ["number"] * len(GAME_ORDER)  # Set column types
        )
        
        # Game selection section
        with gr.Row():
            gr.Markdown("### 🎮 Game Selection")
        with gr.Row():
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
        
        # Controls
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("**⏰ Time Tracker**")
                timeline = create_timeline_slider()
            with gr.Column(scale=1):
                gr.Markdown("**🔄 Controls**")
                clear_btn = gr.Button("Reset Filters", variant="secondary")
        
        # List of all checkboxes
        checkbox_list = [
            mario_overall, mario_details,
            sokoban_overall, sokoban_details,
            _2048_overall, _2048_details,
            candy_overall, candy_details,
            tetris_overall, tetris_details,
            tetris_plan_overall, tetris_plan_details
        ]
        
        def update_leaderboard(*checkbox_states):
            # Convert checkbox states to selected games dictionary
            selected_games = {
                "Super Mario Bros": checkbox_states[0],
                "Sokoban": checkbox_states[2],
                "2048": checkbox_states[4],
                "Candy Crash": checkbox_states[6],
                "Tetris (complete)": checkbox_states[8],
                "Tetris (planning only)": checkbox_states[10]
            }
            
            # Get updated DataFrame
            df = get_combined_leaderboard(rank_data, selected_games)
            
            # Format scores
            for game in GAME_ORDER:
                score_col = f"{game} Score"
                if score_col in df.columns:
                    df[score_col] = df[score_col].apply(lambda x: float(x) if x != '_' else 0)
            
            return df
        
        # Update leaderboard when checkboxes change
        for checkbox in checkbox_list:
            checkbox.change(
                update_leaderboard,
                inputs=checkbox_list,
                outputs=[leaderboard_df]
            )
        
        # Reset filters when clear button is clicked
        def reset_filters():
            # Reset all checkboxes to default state
            checkbox_states = [True, False] * len(GAME_ORDER)
            # Get DataFrame with all games selected
            df = get_combined_leaderboard(rank_data, {game: True for game in GAME_ORDER})
            return [df] + checkbox_states
        
        clear_btn.click(
            reset_filters,
            inputs=[],
            outputs=[leaderboard_df] + checkbox_list
        )
    
    return leaderboard_tab

def make_leaderboard_md(df, last_updated_time):
    """
    Create markdown for the gaming leaderboard
    """
    total_models = len(df)
    space = "&nbsp;&nbsp;&nbsp;"
    
    # Calculate total games played
    total_games = sum(1 for col in df.columns if col.endswith(' Score'))
    
    leaderboard_md = f"""
# 🎮 Gaming Performance Leaderboard
Total #models: **{total_models}**.{space} Total #games: **{total_games}**.{space} Last updated: {last_updated_time}.
"""
    return leaderboard_md

def make_category_leaderboard_md(df, game_name):
    """
    Create markdown for a specific game category
    """
    # Filter for models that participated in this game
    score_col = f"{game_name} Score"
    game_df = df[df[score_col] != '_']
    total_models = len(game_df)
    
    # Calculate average score
    avg_score = game_df[score_col].astype(float).mean()
    
    space = "&nbsp;&nbsp;&nbsp;"
    leaderboard_md = f"""
### {game_name}
#### {space} #models: **{total_models}** {space} Average Score: **{avg_score:.1f}**{space}
"""
    return leaderboard_md

def make_full_leaderboard_md():
    """
    Create markdown explaining the leaderboard metrics
    """
    leaderboard_md = """
The leaderboard displays performance across multiple games:
- **Super Mario Bros**: Platform game performance
- **Sokoban**: Puzzle-solving ability
- **2048**: Number puzzle game
- **Candy Crash**: Matching game
- **Tetris**: Classic block-stacking game

Scores are normalized within each game for fair comparison. Higher values indicate better performance.
"""
    return leaderboard_md

def create_leaderboard_table(df):
    """
    Create a formatted table of the leaderboard
    """
    # Select relevant columns
    columns = ['Player', 'Organization']
    for game in GAME_ORDER:
        columns.append(f"{game} Score")
    
    # Create table
    table = df[columns].copy()
    
    # Format scores
    for game in GAME_ORDER:
        score_col = f"{game} Score"
        table[score_col] = table[score_col].apply(lambda x: f"{float(x):.1f}" if x != '_' else '-')
    
    return table

def update_leaderboard(rank_data, selected_games):
    """
    Update the leaderboard with new data
    """
    # Get the combined leaderboard data
    df = get_combined_leaderboard(rank_data, selected_games)
    
    # Create markdown sections
    last_updated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    leaderboard_md = make_leaderboard_md(df, last_updated)
    
    # Add category sections
    for game in GAME_ORDER:
        if selected_games.get(game, False):
            leaderboard_md += make_category_leaderboard_md(df, game)
    
    # Add explanation
    leaderboard_md += make_full_leaderboard_md()
    
    # Create table
    table = create_leaderboard_table(df)
    
    return leaderboard_md, table 