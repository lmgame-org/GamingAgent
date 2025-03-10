import gradio as gr
import os
import pandas as pd

GAMES = {
    "Super Mario Bros": "assets/super_mario_bros",
    "Sokoban": "assets/sokoban",
    "Tetris": "assets/tetris",
    "2048": "assets/2048",
}
# Ensure each directory exists
for path in GAMES.values():
    os.makedirs(path, exist_ok=True)

scores = {
    "Alice": 10,
    "Bob": 15,
    "Carol": 8
}

def get_leaderboard():
    """Return scoreboard as a sorted pandas DataFrame."""
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(sorted_scores, columns=["Player", "Score"])
    return df

def upload_gif(files, game_name):
    """
    Save uploaded GIF files to the directory corresponding
    to the given game_name. Return a message upon success or failure.
    """
    save_dir = GAMES[game_name]
    saved_paths = []
    for file in files:
        if file is not None and file.name.lower().endswith(".gif"):
            file_path = os.path.join(save_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            saved_paths.append(file_path)
    if saved_paths:
        return f"Uploaded {len(saved_paths)} GIF(s) to '{game_name}' successfully!"
    else:
        return "No valid GIFs uploaded."

def list_gifs(game_name):
    """Return a list of all GIFs in the directory for this game."""
    gif_dir = GAMES[game_name]
    return [
        os.path.join(gif_dir, f)
        for f in os.listdir(gif_dir)
        if f.lower().endswith(".gif")
    ]

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
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
"""

def build_app():
    with gr.Blocks(css=fancy_css) as demo:
        gr.Markdown("# Gaming Agent")

        with gr.Tabs():
            for game_name in GAMES:
                with gr.Tab(game_name):
                    gr.Markdown(f"### {game_name} Gallery")

                    # Gallery for each game
                    gallery = gr.Gallery(
                        label="GIFs",
                        value=list_gifs(game_name),
                        columns=3,  # Display 3 columns
                        rows=1      # Just 1 row for a smaller view
                    )

            with gr.Tab("Leaderboard"):
                gr.Markdown("### Game Scoreboard")
                leaderboard_df = gr.DataFrame(
                    value=get_leaderboard(),
                    label="Current Leaderboard",
                    headers=["Player", "Score"],
                    interactive=False
                )

                def refresh_leaderboard():
                    return get_leaderboard()

                refresh_leaderboard_btn = gr.Button("Refresh Leaderboard")
                refresh_leaderboard_btn.click(
                    refresh_leaderboard,
                    inputs=None,
                    outputs=leaderboard_df
                )

    return demo

if __name__ == "__main__":
    demo_app = build_app()
    demo_app.launch(server_name="0.0.0.0", server_port=7860)
