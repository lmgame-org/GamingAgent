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
    get_combined_leaderboard_with_group_bar,
    create_organization_radar_chart,
    create_top_players_radar_chart,
    create_player_radar_chart,
    create_horizontal_bar_chart,
    normalize_values,
    get_combined_leaderboard_with_single_radar
)

# Try to import enhanced leaderboard, use standard DataFrame if not available

from gradio_leaderboard import Leaderboard, SelectColumns, ColumnFilter
from leaderboard_config import ON_LOAD_COLUMNS, TYPES
HAS_ENHANCED_LEADERBOARD = True


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
    "assets/super_mario_bros/super_mario.gif",
    "assets/sokoban/sokoban.gif",
    "assets/2048/2048.gif",
    "assets/candy/candy.gif",
    "assets/tetris/tetris.gif"
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

# Load video links and news data
with open('assets/game_video_link.json', 'r') as f:
    VIDEO_LINKS = json.load(f)

with open('assets/news.json', 'r') as f:
    NEWS_DATA = json.load(f)

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

# Function to prepare DataFrame for display
def prepare_dataframe_for_display(df, for_game=None):
    """Format DataFrame for better display in the UI"""
    # Clone the DataFrame to avoid modifying the original
    display_df = df.copy()
    
    # Replace '_' with '-' for better display
    for col in display_df.columns:
        if col.endswith(' Score'):
            display_df[col] = display_df[col].apply(lambda x: '-' if x == '_' else x)
    
    # If we're in detailed view, add a formatted rank column
    if for_game:
        # Sort by relevant score column
        score_col = f"{for_game} Score"
        if score_col in display_df.columns:
            # Convert to numeric for sorting, treating '-' as NaN
            display_df[score_col] = pd.to_numeric(display_df[score_col], errors='coerce')
            # Sort by score in descending order
            display_df = display_df.sort_values(by=score_col, ascending=False)
            # Add rank column based on the sort
            display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
            # Filter out models that didn't participate
            display_df = display_df[~display_df[score_col].isna()]
    
    return display_df

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
        
        # Format the DataFrame for display
        display_df = prepare_dataframe_for_display(df, leaderboard_state["current_game"])
        
        # Always create a new chart for detailed view
        chart = create_horizontal_bar_chart(df, leaderboard_state["current_game"])
        # For detailed view, we'll use the same chart for all visualizations
        radar_chart = chart
        group_bar_chart = chart
    else:
        # For overall view
        df, group_bar_chart = get_combined_leaderboard_with_group_bar(rank_data, selected_games)
        # Format the DataFrame for display
        display_df = prepare_dataframe_for_display(df)
        # Use the same selected_games for radar chart
        _, radar_chart = get_combined_leaderboard_with_single_radar(rank_data, selected_games)
        chart = group_bar_chart
    
    # Return exactly 16 values to match the expected outputs
    return (display_df, chart, radar_chart, group_bar_chart,
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
    
    # Format the DataFrame for display
    display_df = prepare_dataframe_for_display(df)
    
    # Get the radar chart using the same selected games
    _, radar_chart = get_combined_leaderboard_with_single_radar(rank_data, selected_games)
    
    # Reset the leaderboard state to match the default checkbox states
    leaderboard_state = get_initial_state()
    
    # Return exactly 16 values to match the expected outputs
    return (display_df, group_bar_chart, radar_chart, group_bar_chart,
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

def create_video_gallery():
    """Create a custom HTML/JS component for video gallery"""
    # Extract video IDs
    mario_id = VIDEO_LINKS["super_mario"].split("?v=")[1]
    sokoban_id = VIDEO_LINKS["sokoban"].split("?v=")[1]
    game_2048_id = VIDEO_LINKS["2048"].split("?v=")[1]
    candy_id = VIDEO_LINKS["candy"].split("?v=")[1]
    
    # Get the latest video from news data
    latest_news = NEWS_DATA["news"][0]  # First item is the latest
    latest_video_id = latest_news["video_link"].split("?v=")[1]
    latest_date = datetime.strptime(latest_news["date"], "%Y-%m-%d")
    formatted_latest_date = latest_date.strftime("%B %d, %Y")
    
    # Generate news HTML
    news_items = []
    for item in NEWS_DATA["news"]:
        video_id = item["video_link"].split("?v=")[1]
        date_obj = datetime.strptime(item["date"], "%Y-%m-%d")
        formatted_date = date_obj.strftime("%B %d, %Y")
        news_items.append(f'''
            <div class="news-item">
                <div class="news-date">{formatted_date}</div>
                <div class="news-content">
                    <div class="news-video">
                        <div class="video-wrapper">
                            <iframe src="https://www.youtube.com/embed/{video_id}"></iframe>
                        </div>
                    </div>
                    <div class="news-text">
                        <a href="{item["twitter_link"]}" target="_blank" class="twitter-link">
                            <span class="twitter-icon">📢</span>
                            {item["twitter_text"]}
                        </a>
                    </div>
                </div>
            </div>
        ''')
    
    news_html = '\n'.join(news_items)
    
    gallery_html = f'''
    <div class="video-gallery-container">
        <style>
            .video-gallery-container {{
                width: 100%;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            .highlight-section {{
                margin-bottom: 40px;
            }}
            .highlight-card {{
                background: #ffffff;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                overflow: hidden;
                transition: transform 0.3s;
                border: 2px solid #2196F3;
            }}
            .highlight-card:hover {{
                transform: translateY(-5px);
            }}
            .highlight-header {{
                background: #2196F3;
                color: white;
                padding: 15px 20px;
                font-size: 1.2em;
                font-weight: bold;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .highlight-date {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .highlight-content {{
                padding: 20px;
            }}
            .video-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-top: 20px;
                margin-bottom: 40px;
            }}
            .video-card {{
                background: #ffffff;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
                transition: transform 0.2s;
            }}
            .video-card:hover {{
                transform: translateY(-5px);
            }}
            .video-wrapper {{
                position: relative;
                padding-bottom: 56.25%;
                height: 0;
                overflow: hidden;
            }}
            .video-wrapper iframe {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                border: none;
            }}
            .video-title {{
                padding: 15px;
                font-size: 1.2em;
                font-weight: bold;
                color: #2c3e50;
                text-align: center;
                background: #f8f9fa;
                border-top: 1px solid #eee;
            }}
            .news-section {{
                margin-top: 40px;
                border-top: 2px solid #e9ecef;
                padding-top: 20px;
            }}
            .news-section-title {{
                font-size: 1.8em;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 20px;
                text-align: center;
            }}
            .news-item {{
                background: #ffffff;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                overflow: hidden;
            }}
            .news-date {{
                padding: 10px 20px;
                background: #f8f9fa;
                color: #666;
                font-size: 0.9em;
                border-bottom: 1px solid #eee;
            }}
            .news-content {{
                display: flex;
                padding: 20px;
                align-items: center;
                gap: 30px;
            }}
            .news-video {{
                flex: 0 0 300px;
            }}
            .news-text {{
                flex: 1;
                display: flex;
                align-items: center;
                min-height: 169px;
            }}
            .twitter-link {{
                color: #2c3e50;
                text-decoration: none;
                display: flex;
                align-items: center;
                gap: 15px;
                font-size: 1.4em;
                font-weight: 600;
                line-height: 1.4;
            }}
            .twitter-link:hover {{
                color: #1da1f2;
            }}
            .twitter-icon {{
                font-size: 1.5em;
                color: #1da1f2;
            }}
        </style>
        
        <!-- Highlight Section -->
        <div class="highlight-section">
            <div class="highlight-card">
                <div class="highlight-header">
                    <span>🌟 Latest Update</span>
                    <span class="highlight-date">{formatted_latest_date}</span>
                </div>
                <div class="highlight-content">
                    <div class="video-wrapper">
                        <iframe src="https://www.youtube.com/embed/{latest_video_id}"></iframe>
                    </div>
                    <div class="video-title">
                        <a href="{latest_news["twitter_link"]}" target="_blank" class="twitter-link">
                            <span class="twitter-icon">📢</span>
                            {latest_news["twitter_text"]}
                        </a>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Regular Video Grid -->
        <div class="video-grid">
            <div class="video-card">
                <div class="video-wrapper">
                    <iframe src="https://www.youtube.com/embed/{mario_id}"></iframe>
                </div>
                <div class="video-title">🎮 Super Mario Bros</div>
            </div>
            <div class="video-card">
                <div class="video-wrapper">
                    <iframe src="https://www.youtube.com/embed/{sokoban_id}"></iframe>
                </div>
                <div class="video-title">📦 Sokoban</div>
            </div>
            <div class="video-card">
                <div class="video-wrapper">
                    <iframe src="https://www.youtube.com/embed/{game_2048_id}"></iframe>
                </div>
                <div class="video-title">🔢 2048</div>
            </div>
            <div class="video-card">
                <div class="video-wrapper">
                    <iframe src="https://www.youtube.com/embed/{candy_id}"></iframe>
                </div>
                <div class="video-title">🍬 Candy Crash</div>
            </div>
        </div>
        
        <!-- News Section -->
        <div class="news-section">
            <div class="news-section-title">📰 Latest News</div>
            {news_html}
        </div>
    </div>
    '''
    return gr.HTML(gallery_html)

def build_app():
    with gr.Blocks(css="""
        .visualization-container {
            height: 70vh !important;  /* Reduced from 85vh to 70vh */
            max-height: 700px !important;  /* Reduced from 900px to 700px */
            min-height: 500px !important;  /* Reduced from 600px to 500px */
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;  /* Reduced padding from 25px to 20px */
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
            margin: 0 auto !important;
        }
        .visualization-container .plot {
            height: 100% !important;
            width: 100% !important;
            object-fit: contain !important;  /* Added to ensure proper scaling */
        }
        .section-title {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
            text-align: center;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        /* Enhanced table styling */
        .table-container {
            overflow-x: auto;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .table-container table {
            width: 100%;
            border-collapse: collapse;
        }
        .table-container th {
            background-color: #f8f9fa;
            position: sticky;
            top: 0;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            border-bottom: 2px solid #e9ecef;
        }
        .table-container td {
            padding: 10px 12px;
            border-bottom: 1px solid #e9ecef;
        }
        .table-container tr:hover {
            background-color: #f1f3f4;
        }
        /* Add a class for numeric columns to right-align them */
        .table-container .numeric {
            text-align: right;
        }
        /* Add alternating row colors */
        .table-container tr:nth-child(even) {
            background-color: #f8fafc;
        }
    """) as demo:
        gr.Markdown("# 🎮 Game Arena: Gaming Agent 🎲")
        
        with gr.Tabs():
            with gr.Tab("🏆 Leaderboard"):
                # Visualization section
                with gr.Row():
                    gr.Markdown("### 📊 Data Visualization")
                
                # Detailed view visualization (single chart)
                detailed_visualization = gr.Plot(
                    label="Performance Visualization",
                    visible=False,
                    elem_classes="visualization-container"
                )
                
                # Overall view visualizations (two charts)
                with gr.Row(visible=True) as overall_visualizations:
                    with gr.Column(scale=1):
                        radar_visualization = gr.Plot(
                            label="Comparative Analysis (Radar Chart)",
                            elem_classes="visualization-container"
                        )
                    with gr.Column(scale=1):
                        group_bar_visualization = gr.Plot(
                            label="Comparative Analysis (Group Bar Chart)",
                            elem_classes="visualization-container"
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
                
                # Leaderboard table
                with gr.Row():
                    gr.Markdown("### 📋 Detailed Results")

                with gr.Blocks():
                    # Add leaderboard search box
                    with gr.Row():
                        search_box = gr.Textbox(
                            label="🔍 Search by Player or Organization",
                            placeholder="Type to filter the table...",
                            show_label=True
                        )
                    
                    # Get initial leaderboard dataframe
                    initial_df = get_combined_leaderboard(rank_data, {
                        "Super Mario Bros": True,
                        "Sokoban": True,
                        "2048": True,
                        "Candy Crash": True,
                        "Tetris (complete)": True,
                        "Tetris (planning only)": True
                    })
                    
                    # Format the DataFrame for display
                    initial_display_df = prepare_dataframe_for_display(initial_df)
                    
                    # Create a standard DataFrame component with enhanced styling
                    leaderboard_df = gr.DataFrame(
                        value=initial_display_df,
                        interactive=True,
                        elem_id="leaderboard-table",
                        elem_classes="table-container",
                        wrap=True,
                        column_widths={"Player": "25%", "Organization": "20%"}
                    )
                    
                    # Add search functionality
                    def filter_table(search_term, current_df):
                        if not search_term:
                            return current_df
                        
                        # Filter the DataFrame by Player or Organization
                        filtered_df = current_df[
                            current_df["Player"].str.contains(search_term, case=False) | 
                            current_df["Organization"].str.contains(search_term, case=False)
                        ]
                        return filtered_df
                    
                    # Connect search box to the table
                    search_box.change(
                        filter_table,
                        inputs=[search_box, leaderboard_df],
                        outputs=[leaderboard_df]
                    )
                
                # List of all checkboxes
                checkbox_list = [
                    mario_overall, mario_details,
                    sokoban_overall, sokoban_details,
                    _2048_overall, _2048_details,
                    candy_overall, candy_details,
                    tetris_overall, tetris_details,
                    tetris_plan_overall, tetris_plan_details
                ]
                
                # Update visualizations when checkboxes change
                def update_visualizations(*checkbox_states):
                    # Check if any details checkbox is selected
                    is_details_view = any([
                        checkbox_states[1], checkbox_states[3], checkbox_states[5],
                        checkbox_states[7], checkbox_states[9], checkbox_states[11]
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
                        update_leaderboard,
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
                    clear_filters,
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
                    fn=clear_filters,
                    inputs=[],
                    outputs=[
                        leaderboard_df,
                        detailed_visualization,
                        radar_visualization,
                        group_bar_visualization
                    ] + checkbox_list
                )
            
            with gr.Tab("🎥 Gallery"):
                video_gallery = create_video_gallery()
    
    return demo

if __name__ == "__main__":
    demo_app = build_app()
    # Add file serving configuration
    demo_app.launch(debug=True, show_error=True, share=True)