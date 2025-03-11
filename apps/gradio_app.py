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
      background: #70c5ce; /* sky-like */
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

    // Grab the canvas and its context
    const canvas = document.getElementById("gameCanvas");
    const ctx = canvas.getContext("2d");

    // Plane properties
    let planeX = 50;
    let planeY = canvas.height / 2;
    const planeWidth = 40;
    const planeHeight = 20;
    const planeSpeed = 4;

    // Bullet properties
    let bullets = [];
    const bulletSpeed = 6;
    const bulletWidth = 6;
    const bulletHeight = 2;

    // Enemy properties
    let enemies = [];
    const enemyWidth = 40;
    const enemyHeight = 20;
    const enemySpeed = 2;
    const spawnInterval = 100; // frames between spawns
    let spawnCounter = 0;

    // Key states
    let keys = {
      ArrowUp: false,
      ArrowDown: false,
      ArrowLeft: false,
      ArrowRight: false,
      Space: false
    };

    // Listen for keydown / keyup
    document.addEventListener("keydown", (e) => {
      if (keys.hasOwnProperty(e.code)) {
        keys[e.code] = true;
      }
    });
    document.addEventListener("keyup", (e) => {
      if (keys.hasOwnProperty(e.code)) {
        keys[e.code] = false;
      }
    });

    // Main update loop
    function update() {
      // Move plane
      if (keys["ArrowUp"] && planeY > 0) {
        planeY -= planeSpeed;
      }
      if (keys["ArrowDown"] && planeY + planeHeight < canvas.height) {
        planeY += planeSpeed;
      }
      if (keys["ArrowLeft"] && planeX > 0) {
        planeX -= planeSpeed;
      }
      if (keys["ArrowRight"] && planeX + planeWidth < canvas.width) {
        planeX += planeSpeed;
      }

      // Fire bullet (on every frame while space is held)
      if (keys["Space"]) {
        bullets.push({
          x: planeX + planeWidth,
          y: planeY + planeHeight / 2 - bulletHeight / 2,
          w: bulletWidth,
          h: bulletHeight
        });
      }

      // Update bullets
      for (let i = 0; i < bullets.length; i++) {
        bullets[i].x += bulletSpeed;
      }
      // Remove bullets offscreen
      bullets = bullets.filter((b) => b.x < canvas.width + 20);

      // Spawn enemies periodically
      spawnCounter++;
      if (spawnCounter > spawnInterval) {
        spawnCounter = 0;
        enemies.push({
          x: canvas.width,
          y: Math.random() * (canvas.height - enemyHeight),
          w: enemyWidth,
          h: enemyHeight
        });
      }

      // Update enemies
      for (let i = 0; i < enemies.length; i++) {
        enemies[i].x -= enemySpeed;
      }
      // Remove enemies offscreen
      enemies = enemies.filter((e) => e.x > -enemyWidth);

      // Check collisions bullets vs enemies
      for (let e = enemies.length - 1; e >= 0; e--) {
        let enemy = enemies[e];
        for (let b = bullets.length - 1; b >= 0; b--) {
          let bullet = bullets[b];
          if (
            bullet.x < enemy.x + enemy.w &&
            bullet.x + bullet.w > enemy.x &&
            bullet.y < enemy.y + enemy.h &&
            bullet.y + bullet.h > enemy.y
          ) {
            // collision
            enemies.splice(e, 1);
            bullets.splice(b, 1);
            break;
          }
        }
      }

      // Draw everything
      draw();
      requestAnimationFrame(update);
    }

    // Render the scene
    function draw() {
      // Clear
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw plane (red rectangle for demonstration)
      ctx.fillStyle = "red";
      ctx.fillRect(planeX, planeY, planeWidth, planeHeight);

      // Draw bullets (small black rectangles)
      ctx.fillStyle = "black";
      bullets.forEach((b) => {
        ctx.fillRect(b.x, b.y, b.w, b.h);
      });

      // Draw enemies (simple gray rectangles)
      ctx.fillStyle = "gray";
      enemies.forEach((en) => {
        ctx.fillRect(en.x, en.y, en.w, en.h);
      });
    }

    // Start game loop
    window.onload = function() {
      // Make sure canvas is fully loaded
      update();
    };
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

            # Top-level tab: "Red Baron" game demo
            with gr.Tab("Red Baron"):
                gr.Markdown("## Red Baron Game Demo")
                gr.HTML(game_html)

    return demo

if __name__ == "__main__":
    demo_app = build_app()
    demo_app.launch(server_name="0.0.0.0", server_port=7860)
