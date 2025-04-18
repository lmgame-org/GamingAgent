def get_2048_prompt(prev_response: str, board_text: str):
    path = "src/prompts/tile_2048/reasoning_prompt.txt"
    with open(path, "r") as f:
        template = f.read()
    return template.format(prev_response=prev_response, board_text=board_text)

def get_2048_read_prompt():
    path = "src/prompts/tile_2048/board_read_prompt.txt"
    with open(path, "r") as f:
        return f.read()