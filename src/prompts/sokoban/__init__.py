def get_sokoban_prompt(prev_response: str, board_text: str):
    with open("src/prompts/sokoban/reasoning_prompt.txt", "r") as f:
        template = f.read()
    return template.format(prev_response=prev_response, board_text=board_text)
