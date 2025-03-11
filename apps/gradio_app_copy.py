import gradio as gr
import os
import pandas as pd
from PIL import Image, ImageSequence
import io


#######################################################
# Dictionary of game -> directory paths
#######################################################
GAMES = {
    "Super Mario Bros": "assets/super_mario_bros",
    "Sokoban": "assets/sokoban",
    "Tetris": "assets/tetris",
    "2048": "assets/2048",
    "Candy Crash": "assets/candy"
}
# Ensure each directory exists
for path in GAMES.values():
    os.makedirs(path, exist_ok=True)

#######################################################
# Scoreboard data for each game
#######################################################

# TODO (lanxiang): read actual data here
mario_scores = [
    ["Alice", 9000, "3/5", "00:35"],
    ["Bob", 2500, "2/5", "00:12"],
    ["Carol", 500, "1/5", "00:05"]
]
sokoban_scores = [
    ["Alice", 100, 3],
    ["Bob", 350, 2],
    ["Carol", 500, 1]
]

tetris_scores = [
    ["Alice", 15000, 120],
    ["Bob", 8000, 60],
    ["Carol", 4000, 45]
]

candy_scores = [
    ["Alice", 12000, 10],
    ["Bob", 9500, 8],
    ["Carol", 3000, 5]
]

# 2048 columns: [Player, Scores, #Steps]
game_2048_scores = [
    ["Alice", 8192, 300],
    ["Bob", 4096, 200],
    ["Carol", 2048, 150]
]

#######################################################
# Functions to return the scoreboard as DataFrames
#######################################################
def get_mario_leaderboard():
    return pd.DataFrame(
        mario_scores, 
        columns=["Player", "Progress (current/total)", "Score", "Time"]
    )

def get_sokoban_leaderboard():
    return pd.DataFrame(
        sokoban_scores, 
        columns=["Player", "Levels Cracked", "Steps"]
    )

def get_tetris_leaderboard():
    return pd.DataFrame(
        tetris_scores, 
        columns=["Player", "Scores", "Steps"]
    )

def get_candy_leaderboard():
    return pd.DataFrame(
        candy_scores,
        columns=["Player", "Levels Cracked", "Scores"]
    )

def get_2048_leaderboard():
    return pd.DataFrame(
        game_2048_scores,
        columns=["Player", "Scores", "Steps"]
    )

#######################################################
# GIF Handling
#######################################################
def create_or_update_resized_gif(original_path, max_dim=600):
    base, ext = os.path.splitext(original_path)
    resized_path = f"{base}_resized{ext}"
    if os.path.exists(resized_path):
        return resized_path

    with Image.open(original_path) as im:
        w, h = im.size
        needs_resize = (w > max_dim or h > max_dim)

        frames = []
        for frame in ImageSequence.Iterator(im):
            frame_rgba = frame.convert("RGBA")
            if needs_resize:
                ratio = min(max_dim / w, max_dim / h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                frame_rgba = frame_rgba.resize((new_w, new_h), Image.LANCZOS)
            frames.append(frame_rgba.convert("P"))

        output_bytes = io.BytesIO()
        frames[0].save(
            output_bytes,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            loop=0,
            disposal=2,
            optimize=False
        )
        output_bytes.seek(0)

    with open(resized_path, "wb") as f_out:
        f_out.write(output_bytes.read())
    return resized_path

def list_gifs(game_name):
    gif_dir = GAMES[game_name]
    all_gifs = [
        os.path.join(gif_dir, f)
        for f in os.listdir(gif_dir)
        if f.lower().endswith(".gif") and not f.lower().endswith("_resized.gif")
    ]
    resized_paths = []
    for gif_path in all_gifs:
        resized_gif_path = create_or_update_resized_gif(gif_path, max_dim=600)
        resized_paths.append(resized_gif_path)

    previously_resized = [
        os.path.join(gif_dir, f)
        for f in os.listdir(gif_dir)
        if f.lower().endswith("_resized.gif")
    ]
    all_resized = list(set(resized_paths + previously_resized))
    return sorted(all_resized)

#######################################################
# Custom CSS
#######################################################
fancy_css = """
body {
    font-family: 'Trebuchet MS', sans-serif;
    background: #f0f8ff;
    color: #333333;
}
h1 {
    color: #4b9cd3;
    text-align: center;
    margin-top: 20px;
}
.gradio-container {
    max-width: 100%;
    margin: 0 auto;
    padding: 20px;
}
"""

#######################################################
# Build the App
#######################################################
def build_app():
    with gr.Blocks(css=fancy_css) as demo:
        gr.Markdown("# Game Arena: Gaming Agent")

        with gr.Tabs():
            # tab: "Gallery"
            with gr.Tab("Gallery"):
                with gr.Tabs():
                    for game_name in GAMES:
                        with gr.Tab(game_name):
                            gr.Markdown(f"### {game_name} Gallery")
                            gr.Gallery(
                                label="GIFs",
                                value=list_gifs(game_name)
                            )

            # tab: "Leaderboard"
            with gr.Tab("Leaderboard"):
                gr.Markdown("## Game Leaderboards")

                # Sub-tabs for each game
                with gr.Tabs():
                    with gr.Tab("Mario"):
                        gr.Markdown("### Mario Leaderboard")
                        gr.DataFrame(value=get_mario_leaderboard(), interactive=False)

                    with gr.Tab("Sokoban"):
                        gr.Markdown("### Sokoban Leaderboard")
                        gr.DataFrame(value=get_sokoban_leaderboard(), interactive=False)

                    with gr.Tab("Tetris"):
                        gr.Markdown("### Tetris Leaderboard")
                        gr.DataFrame(value=get_tetris_leaderboard(), interactive=False)

                    with gr.Tab("2048"):
                        gr.Markdown("### 2048 Leaderboard")
                        gr.DataFrame(value=get_2048_leaderboard(), interactive=False)
                    
                    with gr.Tab("Candy Crash"):
                        gr.Markdown("### Candy Crash Leaderboard")
                        gr.DataFrame(value=get_candy_leaderboard(), interactive=False)

            # Top-level tab: "Candy Crash Game"
            with gr.Tab("Candy Crash Game"):
                gr.Markdown("# Candy Crash Game 🎮")
                gr.HTML("""
                    <iframe src="http://127.0.0.1:5500/apps/web/index.html" width="50%" height="300px" style="border:none; border-radius:12px;"></iframe>
                """)
    return demo

if __name__ == "__main__":
    demo_app = build_app()
    demo_app.launch(server_name="0.0.0.0", server_port=7860)
