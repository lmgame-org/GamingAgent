import gradio as gr
import os
import pandas as pd
from PIL import Image, ImageSequence
import io

#######################################################
# Red Baron Game HTML
#######################################################
test_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Test Page</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f8ff;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #4169e1;
            text-align: center;
        }
        button {
            background-color: #4169e1;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 0;
        }
        button:hover {
            background-color: #3158d3;
        }
        #canvas {
            border: 1px solid #ddd;
            display: block;
            margin: 20px auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Gradio HTML Test</h1>
        <p>This is a simple HTML test page to verify that HTML rendering works in your Gradio application.</p>
        
        <h2>Interactive Elements</h2>
        <button id="testButton">Click Me!</button>
        <p id="result">Button not clicked yet.</p>
        
        <h2>Canvas Test</h2>
        <canvas id="canvas" width="300" height="200"></canvas>
        
        <h2>HTML Elements</h2>
        <ul>
            <li>List item 1</li>
            <li>List item 2</li>
            <li>List item 3</li>
        </ul>
    </div>

    <script>
        // Button click event
        document.getElementById('testButton').addEventListener('click', function() {
            document.getElementById('result').textContent = 'Button clicked at: ' + new Date().toLocaleTimeString();
        });
        
        // Canvas drawing
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        // Draw a simple scene
        ctx.fillStyle = '#e0f0ff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw a circle
        ctx.beginPath();
        ctx.arc(150, 100, 50, 0, Math.PI * 2);
        ctx.fillStyle = '#4169e1';
        ctx.fill();
        
        // Draw a rectangle
        ctx.fillStyle = '#ff6347';
        ctx.fillRect(50, 50, 40, 40);
    </script>
</body>
</html>
"""

game_html = """
<iframe id="gameFrame" style="width:660px; height:520px; border:none; display:block; margin:0 auto;" srcdoc='
<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      margin: 0;
      padding: 0;
      overflow: hidden;
      font-family: sans-serif;
      background: #1e1e1e;
    }
    #gameCanvas {
      background: #70c5ce;
      display: block;
      margin: 0 auto;
      border: 2px solid black;
    }
  </style>
</head>
<body>
  <canvas id="gameCanvas" width="640" height="480"></canvas>
  <script>
    // --- Simple "Red Baron"-style Game in Plain JS ---
    const canvas = document.getElementById("gameCanvas");
    const ctx = canvas.getContext("2d");
    let planeX = 50;
    let planeY = canvas.height / 2;
    const planeWidth = 40;
    const planeHeight = 20;
    const planeSpeed = 4;
    let bullets = [];
    const bulletSpeed = 6;
    const bulletWidth = 6;
    const bulletHeight = 2;
    let enemies = [];
    const enemyWidth = 40;
    const enemyHeight = 20;
    const enemySpeed = 2;
    const spawnInterval = 100;
    let spawnCounter = 0;
    let keys = { ArrowUp: false, ArrowDown: false, ArrowLeft: false, ArrowRight: false, Space: false };
    document.addEventListener("keydown", (e) => { if (keys.hasOwnProperty(e.code)) { keys[e.code] = true; } });
    document.addEventListener("keyup", (e) => { if (keys.hasOwnProperty(e.code)) { keys[e.code] = false; } });
    function update() {
      if (keys["ArrowUp"] && planeY > 0) { planeY -= planeSpeed; }
      if (keys["ArrowDown"] && planeY + planeHeight < canvas.height) { planeY += planeSpeed; }
      if (keys["ArrowLeft"] && planeX > 0) { planeX -= planeSpeed; }
      if (keys["ArrowRight"] && planeX + planeWidth < canvas.width) { planeX += planeSpeed; }
      if (keys["Space"]) {
        bullets.push({ x: planeX + planeWidth, y: planeY + planeHeight / 2 - bulletHeight / 2, w: bulletWidth, h: bulletHeight });
      }
      for (let i = 0; i < bullets.length; i++) { bullets[i].x += bulletSpeed; }
      bullets = bullets.filter((b) => b.x < canvas.width + 20);
      spawnCounter++;
      if (spawnCounter > spawnInterval) {
        spawnCounter = 0;
        enemies.push({ x: canvas.width, y: Math.random() * (canvas.height - enemyHeight), w: enemyWidth, h: enemyHeight });
      }
      for (let i = 0; i < enemies.length; i++) { enemies[i].x -= enemySpeed; }
      enemies = enemies.filter((e) => e.x > -enemyWidth);
      for (let e = enemies.length - 1; e >= 0; e--) {
        let enemy = enemies[e];
        for (let b = bullets.length - 1; b >= 0; b--) {
          let bullet = bullets[b];
          if (bullet.x < enemy.x + enemy.w && bullet.x + bullet.w > enemy.x &&
              bullet.y < enemy.y + enemy.h && bullet.y + bullet.h > enemy.y) {
            enemies.splice(e, 1);
            bullets.splice(b, 1);
            break;
          }
        }
      }
      draw();
      requestAnimationFrame(update);
    }
    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "red";
      ctx.fillRect(planeX, planeY, planeWidth, planeHeight);
      ctx.fillStyle = "black";
      bullets.forEach((b) => { ctx.fillRect(b.x, b.y, b.w, b.h); });
      ctx.fillStyle = "gray";
      enemies.forEach((en) => { ctx.fillRect(en.x, en.y, en.w, en.h); });
    }
    window.onload = function() { update(); };
  </script>
</body>
</html>
'></iframe>
<div style="text-align:center; margin-top:10px;">
  <p>Controls: Arrow keys to move, Space to shoot</p>
</div>
"""

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
for path in GAMES.values():
    os.makedirs(path, exist_ok=True)

#######################################################
# Scoreboard data for each game (Updated Rankings)
#######################################################

# SMB Ranking (Super Mario Bros)
mario_scores = [
    ["Claude 3.7", 710, "1-1", "64.2"],
    ["GPT 4o", 560, "1-1", "58.6"],
    ["Gemini 2.0 flash", 320, "1-1", "51.8"],
    ["GPT-4.5", 160, "1-1", "62.8"],
    ["Claude 3.5 haiku", 140, "1-1", "76.4"]
]

# 2048 Ranking
game_2048_scores = [
    ["o3-mini-medium", 256, 104, ">200"],
    ["Claude 3.7 thinking", 256, 114, ">200"],
    ["o1", 256, 116, ">200"],
    ["Claude 3.7", 256, 130, "∼20"],
    ["Gemini 2.0 flash", 128, 111, "∼18"],
    ["Gemini 2.0 flash thinking", 128, 132, ">100"],
    ["Claude 3.5 haiku", 128, 151, "∼1"],
    ["GPT-4.5", 34, 34, "∼8"],
    ["GPT 4o", 16, 21, "∼1"]
]

# Tetris Ranking (Complete (C) and Planning-only (P) Variants)
tetris_scores = [
    ["Claude 3.7", 95, 27, 110, 29],
    ["Claude 3.5 haiku", 90, 25, 92, 25],
    ["Gemini 2.0 flash", 82, 23, 87, 24],
    ["GPT 4o", 54, 19, 56, 20]
]

# Candy Crush Ranking
candy_scores = [
    ["o1", 97, 25],
    ["o3-mini-medium", 90, 25],
    ["Deepseek-R1", 91, 25],
    ["Claude 3.7", 35, 25],
    ["Gemini 2.0 flash thinking", 18, 25]
]

# Sokoban Ranking
sokoban_scores = [
    ["o3-mini-medium", "2 (3)", "[20,51,70,91]"],
    ["Claude 3.7 thinking", "1 (2)", "[16,38]"],
    ["Deepseek-R1", "1 (1)", "[17,39]"],
    ["Gemini 2.0 flash thinking", "0 (0)", "[17]"],
    ["Claude 3.7", "0", "[37]"],
    ["GPT 4o", "0 (0)", "[113]"]
]

#######################################################
# Functions to return the scoreboard as DataFrames
#######################################################
def get_mario_leaderboard():
    return pd.DataFrame(
        mario_scores, 
        columns=["Model", "Score", "Progress", "Time (s)"]
    )

def get_2048_leaderboard():
    return pd.DataFrame(
        game_2048_scores,
        columns=["Model", "Score", "Steps", "Time (mins)"]
    )

def get_tetris_leaderboard():
    return pd.DataFrame(
        tetris_scores, 
        columns=["Model", "Tetris (C) Score", "Tetris (C) Steps", "Tetris (P) Score", "Tetris (P) Steps"]
    )

def get_candy_leaderboard():
    return pd.DataFrame(
        candy_scores,
        columns=["Model", "Score", "Steps"]
    )

def get_sokoban_leaderboard():
    return pd.DataFrame(
        sokoban_scores, 
        columns=["Model", "Levels Cracked", "Steps"]
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
    max-width: 800px;
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
            # Gallery Tab
            with gr.Tab("Gallery"):
                with gr.Tabs():
                    for game_name in GAMES:
                        with gr.Tab(game_name):
                            gr.Markdown(f"### {game_name} Gallery")
                            gr.Gallery(
                                label="GIFs",
                                value=list_gifs(game_name)
                            )

            # Leaderboard Tab
            with gr.Tab("Leaderboard"):
                gr.Markdown("## Game Leaderboards")
                with gr.Tabs():
                    with gr.Tab("Mario"):
                        gr.Markdown("### SMB Leaderboard")
                        gr.DataFrame(value=get_mario_leaderboard(), interactive=False)
                    with gr.Tab("2048"):
                        gr.Markdown("### 2048 Leaderboard")
                        gr.DataFrame(value=get_2048_leaderboard(), interactive=False)
                    with gr.Tab("Tetris"):
                        gr.Markdown("### Tetris Leaderboard")
                        gr.DataFrame(value=get_tetris_leaderboard(), interactive=False)
                    with gr.Tab("Candy Crash"):
                        gr.Markdown("### Candy Crush Leaderboard")
                        gr.DataFrame(value=get_candy_leaderboard(), interactive=False)
                    with gr.Tab("Sokoban"):
                        gr.Markdown("### Sokoban Leaderboard")
                        gr.DataFrame(value=get_sokoban_leaderboard(), interactive=False)

            # Red Baron Game Demo Tab
            with gr.Tab("Red Baron"):
                gr.Markdown("## Red Baron Game Demo")
                gr.HTML(game_html)

    return demo

if __name__ == "__main__":
    demo_app = build_app()
    demo_app.launch(server_name="0.0.0.0", server_port=7860)
