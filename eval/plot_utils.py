import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont

# Game order for consistent plotting, can be customized or loaded
# For now, let's define a placeholder. This should ideally match the games in your rank data.
# You might want to extract this dynamically from the rank_data or have a more robust way to define it.
GAME_ORDER = [
    "sokoban", "super_mario_bros", "tetris", "twenty_forty_eight", "candy_crush", "ace_attorney" 
] # Default/example order

def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'

def prepare_dataframe_for_plots(
    rank_data_path: str, 
    selected_games: List[str], 
    game_specific_configs: Dict,
    harness_status_to_use: str = "harness_true"
) -> pd.DataFrame:
    """
    Loads model_perf_rank.json, calculates average scores for selected games and harness status.
    Scores are assumed to be already transformed by GameLogProcessor.

    Args:
        rank_data_path (str): Path to model_perf_rank.json.
        selected_games (List[str]): A list of game names to include.
        game_specific_configs (Dict): Configurations for each game, used for display_name.
        harness_status_to_use (str): "harness_true" or "harness_false".

    Returns:
        pd.DataFrame: DataFrame with 'Player' (model name) and 'Game X Score' columns.
                      Scores are averages. Returns empty DataFrame on error.
    """
    try:
        with open(rank_data_path, 'r') as f:
            rank_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Rank data file not found at {rank_data_path}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {rank_data_path}")
        return pd.DataFrame()

    plot_data_list = []
    all_model_names = list(rank_data.keys())

    for model_name in all_model_names:
        model_entry = rank_data.get(model_name, {})
        harness_entry = model_entry.get(harness_status_to_use, {})
        
        player_scores = {'Player': model_name}
        has_any_score_for_selected_games = False

        for game_name in selected_games: # game_name is the internal key
            game_scores_list = harness_entry.get(game_name, [])
            display_game_name = game_specific_configs.get(game_name, {}).get("display_name", game_name)
            score_col_name = f"{display_game_name} Score" 
            
            # Score transformation logic is removed from here.
            # Scores in game_scores_list are assumed to be final (already transformed).

            if game_scores_list and isinstance(game_scores_list, list):
                # Filter for numeric scores and convert to float, just in case
                numeric_scores = [float(s) for s in game_scores_list if isinstance(s, (int, float))]
                
                if numeric_scores:
                    avg_score = np.mean(numeric_scores)
                    player_scores[score_col_name] = avg_score
                    has_any_score_for_selected_games = True
                else:
                    # No numeric scores found for this game for this model
                    player_scores[score_col_name] = np.nan 
            else:
                # game_scores_list is empty or not a list (e.g., game not found for this model)
                player_scores[score_col_name] = np.nan 

        if has_any_score_for_selected_games:
            plot_data_list.append(player_scores)

    if not plot_data_list:
        print(f"Warning: No data found for harness_status='{harness_status_to_use}' and selected games: {selected_games}.")
        return pd.DataFrame()
            
    df = pd.DataFrame(plot_data_list)
    
    # Ensure all selected game score columns (using display names) exist, fill with NaN if missing
    for game_name_internal in selected_games:
        display_game_name = game_specific_configs.get(game_name_internal, {}).get("display_name", game_name_internal)
        score_col_to_check = f"{display_game_name} Score"
        if score_col_to_check not in df.columns:
            df[score_col_to_check] = np.nan
            
    # Reorder columns: Player first, then game scores in the order of selected_games (using display names)
    ordered_display_score_cols = [
        f"{game_specific_configs.get(g, {}).get('display_name', g)} Score"
        for g in selected_games # selected_games are internal keys
    ]
    # Filter out columns that might not exist if a game had no data across all models
    final_ordered_column_names = ['Player'] + [col for col in ordered_display_score_cols if col in df.columns]
    df = df[final_ordered_column_names]
    
    return df


# --- Radar Chart Function ---
def create_comparison_radar_chart(
    df: pd.DataFrame, 
    model_colors: Dict,
    selected_games_display_names: List[str], 
    harness_status: str,
    highlight_models: Optional[List[str]] = None
) -> go.Figure:
    """
    Creates a radar chart comparing models across selected games using their transformed scores.
    """
    df_plot = df.copy()
    if df_plot.empty:
        return go.Figure().update_layout(title_text=f"No data for Radar Chart ({harness_status})")

    game_score_cols = [f"{name} Score" for name in selected_games_display_names if f"{name} Score" in df_plot.columns]

    if not game_score_cols:
        game_score_cols = [col for col in df_plot.columns if col.endswith(" Score") and col != "Player Score"]
        if not game_score_cols:
            return go.Figure().update_layout(title_text=f"No game score columns found for Radar Chart ({harness_status})")

    categories = [col.replace(" Score", "") for col in game_score_cols]
    
    # Directly use the scores; fillna for safety before plotting
    for col in game_score_cols:
         df_plot[col] = df_plot[col].fillna(0)
    data_cols_for_radar = game_score_cols
    y_axis_title = "Average Transformed Score"

    # Determine radial axis range dynamically from the actual data
    all_scores_for_radar = []
    for col in data_cols_for_radar:
        all_scores_for_radar.extend(df_plot[col].dropna().tolist())
    
    min_val_all_games = 0
    max_val_all_games = 1 # Default if no scores
    if all_scores_for_radar:
        min_val_all_games = min(0, min(all_scores_for_radar)) # Ensure range includes 0 if scores can be negative or very low
        max_val_all_games = max(all_scores_for_radar)
        if min_val_all_games == max_val_all_games: # Handle case where all scores are same
            max_val_all_games = min_val_all_games + 1 # Add a small delta to avoid zero range
    
    radial_axis_range = [min_val_all_games, max_val_all_games]

    fig = go.Figure()
    sorted_players = sorted(df_plot['Player'].unique())

    for player in sorted_players:
        player_row = df_plot[df_plot['Player'] == player].iloc[0]
        r_values_raw = [player_row.get(data_col) for data_col in data_cols_for_radar]
        r_values = [val if pd.notna(val) else 0 for val in r_values_raw]

        is_highlighted = highlight_models and player in highlight_models
        
        model_color_hex = model_colors.get(player)
        if not model_color_hex or not isinstance(model_color_hex, str) or not model_color_hex.startswith('#'):
            model_color_hex = '#808080' # Default to grey

        if is_highlighted:
            line_props = dict(color='red', width=3)
            marker_props = dict(color='red', size=8, line=dict(color='darkred', width=2))
            current_fill_color_rgba = 'rgba(255, 0, 0, 0.4)'
            current_opacity = 0.9
        else:
            line_props = dict(color=model_color_hex, width=1.5)
            marker_props = dict(color=model_color_hex, size=4, line=dict(color='#B0B0B0', width=1))
            current_fill_color_rgba = hex_to_rgba(model_color_hex, 0.2)
            current_opacity = 0.7

        fig.add_trace(go.Scatterpolar(
            r=r_values + [r_values[0]],  
            theta=categories + [categories[0]], 
            mode='lines+markers',
            name=player,
            line=line_props,
            marker=marker_props,
            fill='toself',
            fillcolor=current_fill_color_rgba,
            opacity=current_opacity,
            hovertemplate=(
                f'<b>{player}</b><br>'
                f'Game: %{{theta}}<br>'
                f'{y_axis_title}: %{{r:.2f}}'
                '<extra></extra>'
            )
        ))

    fig.update_layout(
        title_text=f'Model Performance Radar ({harness_status}) - Transformed Scores',
        title_x=0.5,
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=radial_axis_range,
                gridcolor='lightgray',
                tickformat=".2f" 
            ),
            angularaxis=dict(
                tickfont=dict(size=10),
                gridcolor='lightgray'
            )
        ),
        legend=dict(
            title="Models (click to toggle)",
            orientation="v", 
            yanchor="top", y=1.02, xanchor="left", x=1.05 
        ),
        width=800,
        height=600,
        margin=dict(l=100, r=200, t=100, b=50) 
    )
    return fig


# --- Bar Chart Function ---
def create_comparison_bar_chart(
    df: pd.DataFrame, 
    model_colors: Dict,
    selected_games_display_names: List[str], 
    harness_status: str,
    highlight_models: Optional[List[str]] = None
) -> go.Figure:
    """
    Creates a grouped bar chart comparing models for each selected game using transformed scores.
    """
    df_plot = df.copy()
    if df_plot.empty:
        return go.Figure().update_layout(title_text=f"No data for Bar Chart ({harness_status})")

    y_axis_title = "Average Transformed Score"
    score_cols_to_use_in_plot = []
    x_axis_categories = [] 

    for display_name in selected_games_display_names:
        data_col = f"{display_name} Score"
        if data_col in df_plot.columns:
            df_plot[data_col] = df_plot[data_col].fillna(0) # Ensure NaNs are filled for plotting
            score_cols_to_use_in_plot.append(data_col)
            x_axis_categories.append(display_name)
    
    if not score_cols_to_use_in_plot: 
        return go.Figure().update_layout(title_text=f"No valid game columns for Bar Chart ({harness_status}) for games: {selected_games_display_names}")

    fig = go.Figure()
    sorted_players = sorted(df_plot['Player'].unique())
    
    for player_name in sorted_players:
        player_row = df_plot[df_plot['Player'] == player_name].iloc[0]
        y_values_raw = [player_row.get(col) for col in score_cols_to_use_in_plot]
        y_values = [val if pd.notna(val) else 0 for val in y_values_raw]
        
        model_color_hex = model_colors.get(player_name, '#808080') 
        if not model_color_hex or not isinstance(model_color_hex, str) or not model_color_hex.startswith('#'):
            model_color_hex = '#808080'

        is_highlighted = highlight_models and player_name in highlight_models
        bar_opacity = 1.0 if is_highlighted else 0.7
        line_width = 2 if is_highlighted else 0

        fig.add_trace(go.Bar(
            name=player_name,
            x=x_axis_categories, 
            y=y_values,
            marker_color=model_color_hex,
            opacity=bar_opacity,
            marker_line_width=line_width,
            marker_line_color='red' if is_highlighted else '#333333',
            hovertemplate=(
                f'<b>{player_name}</b><br>'
                f'Game: %{{x}}<br>'
                f'{y_axis_title}: %{{y:.2f}}'
                '<extra></extra>'
            )
        ))

    fig.update_layout(
        barmode='group',
        title_text=f'Model Performance Comparison ({harness_status}) - Transformed Scores',
        title_x=0.5,
        xaxis_title="Game",
        yaxis_title=y_axis_title,
        legend_title_text='Models (click to toggle)',
        legend=dict(orientation="v", yanchor="top", y=1.02, xanchor="left", x=1.02),
        width=max(800, 150 * len(x_axis_categories) * min(5, df_plot['Player'].nunique()) + 250 ),
        height=600,
        margin=dict(l=50, r=200, t=100, b=50) 
    )
    return fig 

def create_board_image_2048(board_powers: np.ndarray, save_path: str, size: int = 400, perf_score: Optional[float] = None) -> None:
    """Create a visualization of the 2048 board, incorporating new styling and perf_score display."""
    cell_size = size // 4
    padding = cell_size // 10

    img = Image.new('RGB', (size, size), (250, 248, 239)) 
    draw = ImageDraw.Draw(img)

    colors = {
        0: (205, 193, 180),
        2: (238, 228, 218),
        4: (237, 224, 200),
        8: (242, 177, 121),
        16: (245, 149, 99),
        32: (246, 124, 95),
        64: (246, 94, 59),
        128: (237, 207, 114),
        256: (237, 204, 97),
        512: (237, 200, 80),
        1024: (237, 197, 63),
        2048: (237, 194, 46),
        4096: (60, 58, 50),
        8192: (60, 58, 50)
    }
    
    dark_text_color = (119, 110, 101)
    light_text_color = (249, 246, 242)

    font = None
    perf_score_display_font = None
    base_font_size = cell_size // 3
    perf_score_font_size = max(15, size // 25)

    potential_fonts = [
        "arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    for font_name in potential_fonts:
        try:
            if font is None:
                font = ImageFont.truetype(font_name, base_font_size)
            if perf_score_display_font is None:
                 perf_score_display_font = ImageFont.truetype(font_name, perf_score_font_size)
            if font and perf_score_display_font:
                break 
        except (OSError, IOError):
            continue
    
    if font is None:
        font = ImageFont.load_default()
        print("[plot_utils.create_board_image_2048] Main font not found. Using PIL default.")
    if perf_score_display_font is None:
        perf_score_display_font = ImageFont.load_default(size=perf_score_font_size)
        print("[plot_utils.create_board_image_2048] Perf score font not found. Using PIL default.")

    draw.rectangle([0, 0, size, size], fill=(187, 173, 160))

    for r_idx in range(4):
        for c_idx in range(4):
            power = int(board_powers[r_idx, c_idx])
            value = 0 if power == 0 else 2**power
            
            x0 = c_idx * cell_size + padding
            y0 = r_idx * cell_size + padding
            x1 = (c_idx + 1) * cell_size - padding
            y1 = (r_idx + 1) * cell_size - padding
            
            cell_color = colors.get(value, (60, 58, 50)) 
            draw.rectangle([x0, y0, x1, y1], fill=cell_color)
            
            if value == 0:
                continue
            
            text_content = str(value)
            current_text_color = light_text_color if value > 4 else dark_text_color
            
            current_font_size = base_font_size
            if len(text_content) == 3:
                current_font_size = int(base_font_size * 0.8)
            elif len(text_content) >= 4:
                current_font_size = int(base_font_size * 0.65)
            
            final_font_for_tile = font
            if current_font_size != base_font_size:
                temp_font_found = False
                for font_name in potential_fonts:
                    try:
                        final_font_for_tile = ImageFont.truetype(font_name, current_font_size)
                        temp_font_found = True
                        break
                    except (OSError, IOError):
                        continue
                if not temp_font_found:
                    final_font_for_tile = ImageFont.load_default(size=current_font_size)
            
            text_width, text_height = 0, 0
            try:
                if hasattr(final_font_for_tile, 'getbbox'):
                    bbox = final_font_for_tile.getbbox(text_content)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                elif hasattr(final_font_for_tile, 'getsize'):
                    text_width, text_height = final_font_for_tile.getsize(text_content)
                else:
                    text_width = len(text_content) * current_font_size // 2
                    text_height = current_font_size
            except Exception as e:
                 print(f"[plot_utils.create_board_image_2048] Error getting text size: {e}. Using fallback.")
                 text_width = len(text_content) * current_font_size // 2
                 text_height = current_font_size
            
            cell_center_x = (x0 + x1) // 2
            cell_center_y = (y0 + y1) // 2
            text_x = cell_center_x - text_width // 2
            text_y = cell_center_y - text_height // 2 - (cell_size // 20)
            
            draw.text((text_x, text_y), text_content, fill=current_text_color, font=final_font_for_tile)
            if value >= 8:
                draw.text((text_x + 1, text_y), text_content, fill=current_text_color, font=final_font_for_tile)

    if perf_score is not None:
        score_text_content = f"Perf: {perf_score:.2f}"
        score_display_text_color = (10, 10, 10)
        score_pos_x = padding 
        score_pos_y = padding // 2 
        try:
            draw.text((score_pos_x, score_pos_y), score_text_content, fill=score_display_text_color, font=perf_score_display_font)
        except Exception as e:
            print(f"[plot_utils.create_board_image_2048] Error drawing perf_score on image: {e}")

    try:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        img.save(save_path)
    except Exception as e:
        print(f"[plot_utils.create_board_image_2048] Error saving 2048 board image to {save_path}: {e}") 