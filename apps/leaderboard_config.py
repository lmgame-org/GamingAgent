# Leaderboard configuration

# Default columns to display when the leaderboard is first loaded
ON_LOAD_COLUMNS = [
    "Player", 
    "Organization", 
    "Super Mario Bros Score", 
    "Sokoban Score", 
    "2048 Score", 
    "Candy Crash Score", 
    "Tetris (complete) Score", 
    "Tetris (planning only) Score"
]

# Data types for columns
TYPES = {
    "Player": "text",
    "Organization": "text",
    "Super Mario Bros Score": "numeric",
    "Sokoban Score": "numeric",
    "2048 Score": "numeric",
    "Candy Crash Score": "numeric",
    "Tetris (complete) Score": "numeric",
    "Tetris (planning only) Score": "numeric"
}

# Game order for consistent display
GAME_ORDER = [
    "Super Mario Bros",
    "Sokoban",
    "2048",
    "Candy Crash",
    "Tetris (complete)",
    "Tetris (planning only)"
] 