import gradio as gr
from datetime import datetime
import json

# Load video links and news data
with open('assets/game_video_link.json', 'r') as f:
    VIDEO_LINKS = json.load(f)

with open('assets/news.json', 'r') as f:
    NEWS_DATA = json.load(f)

def create_video_gallery():
    """Create a custom HTML/JS component for video gallery"""
    # Extract video IDs
    mario_id = VIDEO_LINKS["super_mario_bros"].split("?v=")[1]
    sokoban_id = VIDEO_LINKS["sokoban"].split("?v=")[1]
    game_2048_id = VIDEO_LINKS["2048"].split("?v=")[1]
    candy_id = VIDEO_LINKS["candy"].split("?v=")[1]
    ace_attorney_id = VIDEO_LINKS["ace_attorney"].split("?v=")[1]
    tetris_id = VIDEO_LINKS["tetris"].split("?v=")[1]

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
                background: var(--card-bg, #ffffff);
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
                color: var(--title-text, #2c3e50);
                text-align: center;
                background: var(--title-bg, #f8f9fa);
                border-top: 1px solid var(--border-color, #eee);
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

            /* Dark mode specific styles */
            .dark .video-card {{
                --card-bg: #2d3748;
                --title-bg: #1a202c;
                --title-text: #e2e8f0;
                --border-color: #4a5568;
            }}

            /* Light mode specific styles */
            .light .video-card {{
                --card-bg: #ffffff;
                --title-bg: #f8f9fa;
                --title-text: #2c3e50;
                --border-color: #eee;
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
                <div class="video-title">🍬 Candy Crush</div>
            </div>
            <div class="video-card">
                <div class="video-wrapper">
                    <iframe src="https://www.youtube.com/embed/{ace_attorney_id}"></iframe>
                </div>
                <div class="video-title">⚖️ Ace Attorney</div>
            </div>
            <div class="video-card">
                <div class="video-wrapper">
                    <iframe src="https://www.youtube.com/embed/{tetris_id}"></iframe>
                </div>
                <div class="video-title">🧩 Tetris</div>
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

def create_gallery_tab():
    """Create and return the gallery tab component"""
    with gr.Tab("🎥 Gallery") as gallery_tab:
        video_gallery = create_video_gallery()
    return gallery_tab 