import time
import os
import pyautogui
import numpy as np

from tools.utils import encode_image, log_output, get_annotate_img, capture_game_window, log_request_cost
from tools.serving.api_providers import anthropic_completion, anthropic_text_completion, openai_completion, openai_text_reasoning_completion, gemini_completion, gemini_text_completion, deepseek_text_reasoning_completion, together_ai_completion
from tools.api_cost_calculator import calculate_all_costs_and_tokens, convert_string_to_messsage
import re
import json

# Default cache directory (can be overridden by passing cache_dir parameter)
DEFAULT_CACHE_DIR = "cache/ace_attorney"



def perform_move(move):
    """
    Directly performs the move using keyboard input without key mapping.
    """
    if move.lower() in ["up", "down", "left", "right"]:
        # For arrow keys, use the direct key name
        pyautogui.keyDown(move.lower())
        time.sleep(0.1)
        pyautogui.keyUp(move.lower())
    else:
        # For other keys, use the direct key press
        pyautogui.keyDown(move.lower())
        time.sleep(0.1)
        pyautogui.keyUp(move.lower())
    
    # print(f"Performed move: {move}")

def log_move_and_thought(move, thought, latency, cache_dir=None):
    """
    Logs the move and thought process into a log file inside the cache directory.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    log_file_path = os.path.join(cache_dir, "ace_attorney_moves.log")
    
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Move: {move}, Thought: {thought}, Latency: {latency:.2f} sec\n"
    
    try:
        with open(log_file_path, "a", encoding='utf-8') as log_file:
            log_file.write(log_entry)
    except Exception as e:
        print(f"[ERROR] Failed to write log entry: {e}")

def vision_evidence_worker(system_prompt, api_provider, model_name, modality, thinking, cache_dir=None):
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Capture game window screenshot
    screenshot_path = capture_game_window(
        image_name="ace_attorney_screenshot_evidence.png",
        window_name="Phoenix Wright: Ace Attorney Trilogy",
        cache_dir=cache_dir
    )
    if not screenshot_path:
        return {"error": "Failed to capture game window"}

    base64_image = encode_image(screenshot_path)
    
    # Construct prompt for LLM
    prompt = (
        "You are analyzing the evidence screen in Phoenix Wright: Ace Attorney.\n\n"
        "Describe ONLY the visual appearance and details of the evidence currently displayed.\n"
        "Do NOT restate the evidence name or text description from the Court Record.\n\n"
        "Focus on things like:\n"
        "- Shape, size, material, color\n"
        "- Any writing, symbols, damage, or special features\n"
        "- Context clues that might indicate how the item is used or related to the case\n\n"
        "Output format:\n"
        "Evidence Description: <your detailed visual description here>"
    )

    # print(f"Calling {model_name} API for vision analysis...")
    
    if api_provider == "anthropic" and modality=="text-only":
        response = anthropic_text_completion(system_prompt, model_name, prompt, thinking)
    elif api_provider == "anthropic":
        response = anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking)
    elif api_provider == "openai" and "o3" in model_name and modality=="text-only":
        response = openai_text_reasoning_completion(system_prompt, model_name, prompt)
    elif api_provider == "openai":
        response = openai_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "gemini" and modality=="text-only":
        response = gemini_text_completion(system_prompt, model_name, prompt)
    elif api_provider == "gemini":
        response = gemini_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "deepseek":
        response = deepseek_text_reasoning_completion(system_prompt, model_name, prompt)
    elif api_provider == "together_ai":
        response = together_ai_completion(system_prompt, model_name, prompt, base64_image=base64_image)
    else:
        raise NotImplementedError(f"API provider: {api_provider} is not supported.")
    if "claude" in model_name:
        prompt_message = convert_string_to_messsage(prompt)
    else:
        prompt_message = prompt
    # Update completion in cost data
    cost_data = calculate_all_costs_and_tokens(
        prompt=prompt_message,
        completion=response,
        model=model_name,
        image_path=screenshot_path if base64_image else None
    )
    
    # Log the request costs
    log_request_cost(
        num_input=cost_data["prompt_tokens"] + cost_data.get("image_tokens", 0),
        num_output=cost_data["completion_tokens"],
        input_cost=float(cost_data["prompt_cost"] + cost_data.get("image_cost", 0)),
        output_cost=float(cost_data["completion_cost"]),
        game_name="ace_attorney",
        input_image_tokens=cost_data.get("image_tokens", 0),
        model_name = model_name,
        cache_dir=cache_dir
    )
    
    return {
        "response": response,
        "screenshot_path": screenshot_path
    }

def vision_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    cache_dir=None
    ):
    """
    Captures and analyzes the current game screen.
    Returns scene analysis including game state, dialog text, and detailed scene description.
    """
    assert modality == "vision-text", "Vision worker requires vision-text modality"
    
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    # Capture game window screenshot
    screenshot_path = capture_game_window(
        image_name="ace_attorney_screenshot.png",
        window_name="Phoenix Wright: Ace Attorney Trilogy",
        cache_dir=cache_dir
    )
    if not screenshot_path:
        return {"error": "Failed to capture game window"}

    base64_image = encode_image(screenshot_path)
    
    prompt = (
            "You are now playing Ace Attorney. Analyze the current scene and provide the following information:\n\n"
            "Carefully analyze the game state. If any of the indicators are present, determine that it is Cross-Examination mode.\n\n"
            
            "1. Game State Detection Rules:\n"
            "   - Cross-Examination mode is indicated by ANY of these:\n"
            "     * A blue bar in the upper right corner\n"
            "     * Only green dialog text\n"
            "     * Two or more white-text options appearing in the **middle** of the screen (e.g., 'Yes' and 'No')\n"
            "     * EXACTLY three UI elements at the bottom-right corner: Options, Press, Present\n"
            "     * An evidence window visible in the middle of the screen\n"
            "   - If you see an evidence window, it is ALWAYS Cross-Examination mode\n"
            "   - Conversation mode is indicated by:\n"
            "     * EXACTLY two UI elements at the bottom-right corner: Options, Court Record\n"
            "     * Dialog text can be any color (most commonly white, but also blue, red, etc.)\n"
            "   - If none of the Cross-Examination indicators are present, it is Conversation mode\n\n"
            
            "2. Dialog Text Analysis:\n"
            "   - Look at the bottom-left area where dialog appears\n"
            "   - Note the color of the dialog text (green/white/blue/red)\n"
            "   - Extract the speaker's name and their dialog\n"
            "   - Format must be exactly: Dialog: NAME: dialog text\n\n"
            
            "3. Scene Analysis:\n"
            "   - Describe any visible characters and their expressions/poses\n"
            "   - Describe any other important visual elements or interactive UI components\n"
            "   - Describe any options with blue background appearing in the **middle** of the screen\n"
            "   - You MUST explicitly mention:\n"
            "     * The color of the dialog text (green/white/blue/red)\n"
            "     * Whether there is a blue bar in the upper right corner\n"
            "     * The exact UI elements present at the bottom-right corner (Options, Press, Present for Cross-Examination or Options, Court Record for Conversation)\n"
            "     * Whether there is an evidence window visible\n"
            "     * If options appear in the middle of the screen:\n"
            "       - List the text of each option in order from top to bottom\n"
            "       - Identify which one is currently selected\n"
            "       - Use the yellow or gold border around the option to determine selection\n"
            "       - Do NOT assume the bottom option is selected by default — selection depends entirely on the visual highlight\n"
            "     * If evidence window is visible:\n"
            "       - Name of the currently selected evidence\n"
            "       - Description of the evidence\n"
            "       - Position in the evidence list (if visible)\n"
            "       - Whether this is the evidence you intend to present\n\n"
            
            "Format your response EXACTLY as:\n"
            "Game State: <'Cross-Examination' or 'Conversation'>\n"
            "Dialog: NAME: dialog text\n"
            "Options: option1, selected; option2, not selected; option3, not selected\n"
            "Evidence: NAME: description\n"
            "Scene: <detailed description including dialog color, options text (if exsisit), blue bar presence, UI elements (corresponding keys, like r Present/Court Record or x Present), evidence window status and contents, and other visual elements>"
            #  "At the end of your Scene description, briefly summarize:\n"
            # "dialog text is <color>, evidence window is <open/closed>, options are <available/unavailable>"
            )



    # print(f"Calling {model_name} API for vision analysis...")
    
    if api_provider == "anthropic" and modality=="text-only":
        response = anthropic_text_completion(system_prompt, model_name, prompt, thinking)
    elif api_provider == "anthropic":
        response = anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking)
    elif api_provider == "openai" and "o3" in model_name and modality=="text-only":
        response = openai_text_reasoning_completion(system_prompt, model_name, prompt)
    elif api_provider == "openai":
        response = openai_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "gemini" and modality=="text-only":
        response = gemini_text_completion(system_prompt, model_name, prompt)
    elif api_provider == "gemini":
        response = gemini_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "deepseek":
        response = deepseek_text_reasoning_completion(system_prompt, model_name, prompt)
    elif api_provider == "together_ai":
        response = together_ai_completion(system_prompt, model_name, prompt, base64_image=base64_image)
    else:
        raise NotImplementedError(f"API provider: {api_provider} is not supported.")
    if "claude" in model_name:
        prompt_message = convert_string_to_messsage(prompt)
    else:
        prompt_message = prompt
    # Update completion in cost data
    cost_data = calculate_all_costs_and_tokens(
        prompt=prompt_message,
        completion=response,
        model=model_name,
        image_path=screenshot_path if base64_image else None
    )
    
    # Log the request costs
    log_request_cost(
        num_input=cost_data["prompt_tokens"] + cost_data.get("image_tokens", 0),
        num_output=cost_data["completion_tokens"],
        input_cost=float(cost_data["prompt_cost"] + cost_data.get("image_cost", 0)),
        output_cost=float(cost_data["completion_cost"]),
        game_name="ace_attorney",
        input_image_tokens=cost_data.get("image_tokens", 0),
        model_name=model_name,
        cache_dir=cache_dir
    )
    
    return {
        "response": response,
        "screenshot_path": screenshot_path
    }

def long_term_memory_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    episode_name="The First Turnabout",
    dialog=None,
    evidence=None,
    cache_dir=None
    ):
    """
    Maintains dialog history for the current episode.
    If evidence is provided, adds it to the evidences list.
    """
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Create cache directory for dialog history if it doesn't exist
    dialog_history_dir = os.path.join(cache_dir, "dialog_history")
    os.makedirs(dialog_history_dir, exist_ok=True)

    # Define the JSON file path based on the episode name
    json_file = os.path.join(dialog_history_dir, f"{episode_name.lower().replace(' ', '_')}.json")

    # Load existing dialog history or initialize new structure
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            dialog_history = json.load(f)
    else:
        dialog_history = {
            episode_name: {
                "Case_Transcript": [],
                "evidences": []
            }
        }

    # Update dialog history if dialog is provided
    if dialog and dialog["name"] and dialog["text"]:
        dialog_entry = f"{dialog['name']}: {dialog['text']}"
        if dialog_entry not in dialog_history[episode_name]["Case_Transcript"]:
            dialog_history[episode_name]["Case_Transcript"].append(dialog_entry)

    # Update evidence if provided
    if evidence and evidence["name"] and evidence["text"] and evidence["description"]:
        evidence_entry = f"{evidence['name']}: {evidence['text']}. UI description: {evidence['description']}"
        if evidence_entry not in dialog_history[episode_name]["evidences"]:
            dialog_history[episode_name]["evidences"].append(evidence_entry)

    # Save the updated dialog history
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(dialog_history, f, indent=2, ensure_ascii=False)

    return json_file

def short_term_memory_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    episode_name="The First Turnabout",
    cache_dir=None
    ):
    """
    Maintains a short-term memory of previous responses by storing the last 7 responses in the JSON file.
    Args:
        episode_name (str): Name of the current episode
        prev_response (str): The new response to add to the queue
        cache_dir (str, optional): Directory to save memory data
    """
    if not prev_response:
        return
    
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
        
    # Create cache directory if it doesn't exist
    dialog_history_dir = os.path.join(cache_dir, "dialog_history")
    os.makedirs(dialog_history_dir, exist_ok=True)
    
    # JSON file path for the episode
    json_file = os.path.join(dialog_history_dir, f"{episode_name.lower().replace(' ', '_')}.json")
    
    # Load existing dialog history or create new one
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            dialog_history = json.load(f)
    else:
        dialog_history = {
            episode_name: {
                "Case_Transcript": [],
                "prev_responses": [],
                "evidences": []
            }
        }
    
    # Initialize prev_responses if it doesn't exist
    if "prev_responses" not in dialog_history[episode_name]:
        dialog_history[episode_name]["prev_responses"] = []
    
    # Add new response and maintain only last 7 responses
    prev_responses = dialog_history[episode_name]["prev_responses"]
    prev_responses.append(prev_response)
    if len(prev_responses) > 14:
        prev_responses.pop(0)  # Remove oldest response
        
    # Save updated dialog history
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(dialog_history, f, indent=4, ensure_ascii=False)
            
    return json_file

def memory_retrieval_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    episode_name="The First Turnabout",
    cache_dir=None
    ):
    """
    Retrieves and composes complete memory context from long-term and short-term memory.
    """
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Load background conversation context
    background_file = "games/ace_attorney/ace_attorney_1.json"
    with open(background_file, 'r', encoding='utf-8') as f:
        background_data = json.load(f)
    background_context = background_data[episode_name]["Case_Transcript"]

    # Load current episode memory
    memory_file = os.path.join(cache_dir, "dialog_history", f"{episode_name.lower().replace(' ', '_')}.json")
    if os.path.exists(memory_file):
        with open(memory_file, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        current_episode = memory_data[episode_name]
        cross_examination_context = current_episode["Case_Transcript"]
        prev_responses = current_episode.get("prev_responses", [])
        collected_evidences = current_episode.get("evidences", [])
    else:
        cross_examination_context = []
        prev_responses = []
        collected_evidences = []

    # Compose complete memory context
    memory_context = f"""Background Conversation Context:
        {chr(10).join(background_context)}

        Cross-Examination Conversation Context:
        {chr(10).join(cross_examination_context)}

        Previous 7 manipulations:
        {chr(10).join(prev_responses)}

        Collected Evidences:
        {chr(10).join(collected_evidences)}"""

    return memory_context

def reasoning_worker(options, system_prompt, api_provider, model_name, game_state, c_statement, scene, memory_context, base64_image=None, modality="vision-text", thinking=True, screenshot_path=None, cache_dir=None):
    """
    Makes decisions about game moves based on current game state, scene description, and memory context.
    Uses API to generate thoughtful decisions.
    
    Args:
        system_prompt (str): System prompt for the API
        api_provider (str): API provider to use
        model_name (str): Model name to use
        game_state (str): Current game state (Cross-Examination, Conversation, or Evidence)
        scene (str): Description of the current scene
        memory_context (str): Complete memory context including dialog history and evidences
        base64_image (str, optional): Base64 encoded screenshot of the current game state
        modality (str): Modality to use (vision-text or text-only)
        thinking (bool): Whether to use deep thinking
        screenshot_path (str, optional): Path to the screenshot
        cache_dir (str, optional): Directory to save logs
    
    Returns:
        dict: Contains move and thought
    """
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Extract and format evidence information
    evidences_section = memory_context.split("Collected Evidences:")[1].strip()
    collected_evidences = [e for e in evidences_section.split("\n") if e.strip()]
    num_collected_evidences = len(collected_evidences)
    
    # Format evidence details for the prompt
    evidence_details = "\n".join([f"Evidence {i+1}: {e}" for i, e in enumerate(collected_evidences)])
    # print(scene)


    if game_state == "Cross-Examination":
        prompt = f"""You are Phoenix Wright, a defense attorney in Ace Attorney. Your goal is to prove your client's innocence by finding contradictions in witness testimonies and presenting the right evidence at the right time.

        CURRENT GAME STATE: {game_state}

        Your task is to evaluate the **current witness statement** and determine whether it contradicts any evidence in the Court Record.

        Current Statement: 
        "{c_statement}"

        Current options: (determine if there are options, if yes then use 'z' to continue or use 'down' to change)
        {options}

        Scene Description: (determine if the evidence window is already opened)
        {scene}

        Evidence Status:  
        - Total Evidence Collected: {num_collected_evidences}  
        {evidence_details}

        Memory Context:  
        {memory_context}

        Be patient. DO NOT rush to present evidence. Always wait until the **decisive contradiction** becomes clear.
        You only have 7 chances to make a mistake.  
        If you've already presented evidence but it wasn't successful, try going to the next statement or switching to a different piece of evidence.

        You may only present evidence if:
        - A clear and specific contradiction exists between the current statement and an item in the Court Record
        - The **correct** evidence item is currently selected
        - The **evidence window is open**, and you are on the exact item you want to present

        Never assume the correct evidence is selected. Always confirm it.

        Cross-Examination Mode (CURRENT STATE: {game_state}):
        - ALWAYS compare the witness's statement with the available evidence
        - For each statement, you have two options:
        * If you find a clear contradiction with evidence: (Three steps — one per turn)
            - Step 1: Use 'r' to open the evidence window
            - Step 2: Navigate through evidence using 'right'
                * Look at each item carefully
                * Keep navigating until the evidence that directly contradicts the statement is selected
            - Step 3: Use 'x' to present the contradicting evidence
                * Only present if the evidence is currently selected and the contradiction is clear
        * If you don't find a contradiction or need more context:
            - Use 'l' to press the witness for more details
            - Or use 'z' to move to the next statement
        - If there are on-screen decision options (like "Yes", "No", "Press", "Present"), you must:
            * Use `'down'` to navigate between them
            * Use `'z'` to confirm the currently highlighted option
        * If you don't find a contradiction but the evidence window is mistakely opened:
            - Use 'b' to close the evidence window

        Additional Rules:
        - The evidence window will auto-close after presenting
        - Do NOT use `'x'` or `'r'` unless you are certain
        - If the evidence window is NOT open, NEVER use `'x'` to present

        - Always loop through all Cross-Examination statements by using `'z'`.  
        After reaching the final statement, the game will automatically return to the first one.  
        This allows you to review all statements before taking action.

        Available moves:
        * `'l'`: Question the witness about their statement
        * `'z'`: Move to the next statement OR confirm a selected option
        * `'r'`: Open the evidence window (press `'b'` to cancel if unsure)
        * `'b'`: Close the evidence window or cancel a mistake
        * `'x'`: Present evidence (only after confirming it's correct)
        * `'right'`: Navigate through the evidence items
        * `'down'`: Navigate between options (like Yes/No or Press/Present) when visible

        Before using `'x'`, always ask:
        - "Is the currently selected evidence exactly the one I want to present?"

        If not:
        - Use `'right'` to select the correct evidence
        - DO NOT use `'x'` until it's confirmed

        Response Format (strict):
        move: <move>
        thought: <your internal reasoning>

        IMPORTANT:
        - If the evidence window is already open, do NOT use 'r' again
        - Check what evidence is selected (based on scene description)
        - Use 'right' to navigate if it's not the correct one
        - Only use 'x' when the right evidence is selected
        - If options are on screen, navigate with 'down', confirm with 'z'

        Example 1:
        Scene says: "The currently selected evidence is 'Attorney's Badge'."
        But I want to present: "Cindy's Autopsy Report"

        Turn 1:  
        move: right  
        thought: The Autopsy Report is not selected yet. I'll navigate to it.

        Turn 2:  
        move: x  
        thought: The Autopsy Report is now selected. I'll present it to contradict the witness.

        Example 2 - Clear Contradiction with No Evidence Window:
        Memory Context:
        Witness: "I was at home at 8 PM last night."
        Evidence: "Security Camera Footage: Shows the witness at the crime scene at 8 PM."

        Scene: "Dialog text is green. There is a blue bar in the upper right corner. There are exactly three UI elements at the bottom-right corner: Options, Press, Present. No evidence window is visible. The witness is sweating and looking nervous."

        Turn 1:
        move: r
        thought: I see a clear contradiction between the witness's statement and our security camera footage. We're in cross-examination mode, but the evidence window isn't open. I'll open it first.

        Turn 2:
        move: right
        thought: I need to navigate to the security camera footage.

        Turn 3:
        move: x
        thought: I've selected the right evidence. Presenting it now to contradict the witness.

        Example 3 - Using 'down' to select an option before confirming:

        Scene: "Two white-text options appear in the middle of the screen: 'Yes' and 'No'. 'No' is currently highlighted. Dialog text is white. This is Cross-Examination mode."

        I want to answer yes, so I need to switch to 'Yes' before confirming.

        Turn 1:
        move: down
        thought: 'No' is selected by default, but I want to choose 'Yes'. I'll navigate to it.

        Turn 2:
        move: z
        thought: 'Yes' is now selected. I'll confirm the choice.

        Stuck Situation Handling:
        - If no progress has been made in the last 5 responses with cross-examination game state (check prev_responses about whether they are the same.)
        - If the agent seems stuck in a loop or unable to advance
        - Use 'b' to break out of the loop
        - This helps the agent recover and move forward in the game
    """

        # Call the API
        if api_provider == "anthropic" and modality=="text-only":
            response = anthropic_text_completion(system_prompt, model_name, prompt, thinking)
        elif api_provider == "anthropic":
            response = anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking)
        elif api_provider == "openai" and "o3" in model_name and modality=="text-only":
            response = openai_text_reasoning_completion(system_prompt, model_name, prompt)
        elif api_provider == "openai":
            response = openai_completion(system_prompt, model_name, base64_image, prompt)
        elif api_provider == "gemini" and modality=="text-only":
            response = gemini_text_completion(system_prompt, model_name, prompt)
        elif api_provider == "gemini":
            response = gemini_completion(system_prompt, model_name, base64_image, prompt)
        elif api_provider == "deepseek":
            response = deepseek_text_reasoning_completion(system_prompt, model_name, prompt)
        elif api_provider == "together_ai":
            response = together_ai_completion(system_prompt, model_name, prompt, base64_image=base64_image)
        else:
            raise NotImplementedError(f"API provider: {api_provider} is not supported.")
        if "claude" in model_name:
            prompt_message = convert_string_to_messsage(prompt)
        else:
            prompt_message = prompt
        # Update completion in cost data
        cost_data = calculate_all_costs_and_tokens(
            prompt=prompt_message,
            completion=response,
            model=model_name,
            image_path=screenshot_path if base64_image else None
        )
        
        # Log the request costs
        log_request_cost(
            num_input=cost_data["prompt_tokens"] + cost_data.get("image_tokens", 0),
            num_output=cost_data["completion_tokens"],
            input_cost=float(cost_data["prompt_cost"] + cost_data.get("image_cost", 0)),
            output_cost=float(cost_data["completion_cost"]),
            game_name="ace_attorney",
            input_image_tokens=cost_data.get("image_tokens", 0),
            model_name=model_name,
            cache_dir=cache_dir
        )

        # Extract move and thought from response
        move_match = re.search(r"move:\s*(.+?)(?=\n|$)", response)
        thought_match = re.search(r"thought:\s*(.+?)(?=\n|$)", response)
        
        move = move_match.group(1).strip() if move_match else ""
        thought = thought_match.group(1).strip() if thought_match else ""

        return {
            "move": move,
            "thought": thought
        }
    
    else:
        time.sleep(1)
        return {
            "move": "z",
            "thought": "continue conversation"
        }

def ace_evidence_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    episode_name="The First Turnabout",
    cache_dir=None
    ):
    """
    Iterates through known evidences using vision, stores full evidence with name, text, and vision-based description.
    """
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    background_file = "games/ace_attorney/ace_attorney_1.json"
    with open(background_file, 'r') as f:
        background_data = json.load(f)
    evidence_lines = background_data[episode_name]["evidences"]

    PREDEFINED_EVIDENCES = {
        item.split(":")[0].strip().upper(): ":".join(item.split(":")[1:]).strip()
        for item in evidence_lines
    }

    print(PREDEFINED_EVIDENCES)

    evidence_names = list(PREDEFINED_EVIDENCES.keys())
    collected = []

    time.sleep(1)
    # Step 1: Open Court Record
    perform_move("r")
    time.sleep(1)

    for name in evidence_names:
        # Get visual description via LLM
        vision_result = vision_evidence_worker(
            system_prompt,
            api_provider,
            model_name,
            modality,
            thinking,
            cache_dir=cache_dir
        )
        if "error" in vision_result:
            return vision_result
        
        # Parse the vision result: look for line starting with "Evidence Description:"
        desc_match = re.search(r"Evidence Description:\s*(.+)", vision_result["response"])
        visual_description = desc_match.group(1).strip() if desc_match else "No description found."

        # Build evidence
        evidence = {
            "name": name,
            "text": PREDEFINED_EVIDENCES[name],
            "description": visual_description
        }

        # Save to memory
        long_term_memory_worker(
            system_prompt,
            api_provider,
            model_name,
            prev_response,
            thinking,
            modality,
            episode_name,
            evidence=evidence,
            cache_dir=cache_dir
        )

        print(f"[INFO] Collected evidence: {evidence}")
        collected.append(evidence)

        # Move to next item
        perform_move("right")
        time.sleep(1)

    # Step 2: Close Court Record
    perform_move("b")
    time.sleep(1)

    return {
        "game_state": "Evidence",
        "collected_evidences": collected
    }

def ace_attorney_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    episode_name="The First Turnabout",
    decision_state=None,
    cache_dir=None
    ):
    """
    1) Captures a screenshot of the current game state.
    2) Analyzes the scene using vision worker.
    3) Makes decisions based on the scene analysis.
    4) Maintains dialog history for the current episode.
    5) Makes decisions about game moves.
    
    Args:
        episode_name (str): Name of the current episode (default: "The First Turnabout")
        cache_dir (str, optional): Directory to save cache files
    """
    assert modality in ["text-only", "vision-text"], f"modality {modality} is not supported."
    
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    # -------------------- Vision Processing -------------------- #
    # First, analyze the current game state using vision
    vision_result = vision_worker(
        system_prompt,
        api_provider,
        model_name,
        prev_response,
        thinking,
        modality="vision-text",
        cache_dir=cache_dir
    )

    if "error" in vision_result:
        return vision_result

    # Extract the formatted outputs using regex
    response_text = vision_result["response"]
    
    # Extract Game State
    game_state_match = re.search(r"Game State:\s*(Cross-Examination|Conversation|Evidence)", response_text)
    game_state = game_state_match.group(1) if game_state_match else "Unknown"
    
    # Extract Dialog (with NAME: text format)
    dialog_match = re.search(r"Dialog:\s*([^:\n]+):\s*(.+?)(?=\n|$)", response_text)
    dialog = {
        "name": dialog_match.group(1) if dialog_match else "",
        "text": dialog_match.group(2).strip() if dialog_match else ""
    }
    
    # Extract Evidence
    evidence_match = re.search(r"Evidence:\s*([^:\n]+):\s*(.+?)(?=\n|$)", response_text)
    evidence = {
        "name": evidence_match.group(1) if evidence_match else "",
        "description": evidence_match.group(2).strip() if evidence_match else ""
    }
    
    ###------------ Extract Options ---------------###
    print(response_text)
    # Default options structure
    options = {
        "choices": [],
        "selected": ""
    }

    options_match = re.search(r"Options:\s*(.+)", response_text)
    if options_match:
        raw_options = options_match.group(1).strip()
        if raw_options.lower() != "none":
            # Extract individual option entries using comma-separated pairs
            option_entries = [opt.strip() for opt in raw_options.split(';') if opt.strip()]
            for entry in option_entries:
                match = re.match(r"(.+?),\s*(selected|not selected)", entry)
                if match:
                    text, state = match.groups()
                    options["choices"].append(text.strip())
                    if state == "selected":
                        options["selected"] = text.strip()

    if options["choices"]:
        game_state = "Cross-Examination"
        if decision_state is None:
            decision_state = {
                "has_options": True,
                "down_count": 0,
                "selection_index": 0,
                "selected_text": options["choices"][0],  # default to first option
                "decision_timestamp": None
            }
        options["selected"] = decision_state["selected_text"]
        
    # Extract Scene Description
    print("\n=== Vision Worker Output ===")
    print(response_text)
    scene_match = re.search(r"Scene:\s*((?:.|\n)+?)(?=\n(?:Game State:|Dialog:|Evidence:|Options:|$)|$)", response_text, re.DOTALL)
    scene = scene_match.group(1).strip() if scene_match else ""
    print("="*50 + "\n")


    # last_line = response_text.strip().split('\n')[-1]
    # print(game_state)
    # print(last_line)
    # # Check for keywords in the last line
    # if (
    #     "dialog text is green" in last_line 
    #     or "evidence window is open" in last_line 
    #     or "options are available" in last_line
    # ):
    #     game_state = "Cross-Examination"
    # else: 
    #     game_state = "Conversation"


    # -------------------- Memory Processing -------------------- #
    # Update long-term memory only
    if game_state == "Evidence":
        # If in Evidence mode, update evidence instead of dialog
        if evidence["name"] and evidence["description"]:
            evidence_file = long_term_memory_worker(
                system_prompt,
                api_provider,
                model_name,
                prev_response,
                thinking,
                modality,
                episode_name,
                evidence=evidence,
                cache_dir=cache_dir
            )
    else:
        # If in Conversation or Cross-Examination mode, update dialog
        if dialog["name"] and dialog["text"]:
            dialog_file = long_term_memory_worker(
                system_prompt,
                api_provider,
                model_name,
                prev_response,
                thinking,
                modality,
                episode_name,
                dialog=dialog,
                cache_dir=cache_dir
            )

    # Then, retrieve and compose complete memory context
    complete_memory = memory_retrieval_worker(
        system_prompt,
        api_provider,
        model_name,
        prev_response,
        thinking,
        modality,
        episode_name,
        cache_dir=cache_dir
    )

    c_statement = f"{dialog}"
    # -------------------- Reasoning -------------------- #
    # Make decisions about game moves
    reasoning_result = reasoning_worker(
        options,
        system_prompt,
        api_provider,
        model_name,
        game_state,
        c_statement,
        scene,
        complete_memory,
        base64_image=encode_image(vision_result["screenshot_path"]),
        modality='text-only',
        screenshot_path=vision_result["screenshot_path"],
        thinking=thinking,
        cache_dir=cache_dir
    )

    # In your reasoning loop, track moves:
    if decision_state:
        if reasoning_result["move"] == "down" and decision_state["has_options"]:
            decision_state["down_count"] += 1
            i = min(decision_state["down_count"], len(options["choices"]) - 1)
            decision_state["selection_index"] = i
            decision_state["selected_text"] = options["choices"][i]

        if reasoning_result["move"] == "z" and decision_state["has_options"]:
            decision_state["decision_timestamp"] = time.time()
            print(f"[Decision Made] Selected option: '{decision_state['selected_text']}' at index {decision_state['selection_index']} (via {decision_state['down_count']} down moves)")

    parsed_result = {
        "game_state": game_state,
        "dialog": dialog,
        "evidence": evidence,
        "scene": scene,
        "screenshot_path": vision_result["screenshot_path"],
        "memory_context": complete_memory,
        "move": reasoning_result["move"],
        "thought": reasoning_result["thought"],
        "options": options,
        "decision_state": decision_state
    }

    return parsed_result

def check_skip_conversation(dialog, episode_name):
    """
    Checks if the current dialog exists as a key in ace_attorney_1_skip_conversations.json.
    If found, returns the list of dialogs to skip through.
    Otherwise returns None.
    
    Args:
        dialog (dict or str): Dialog to check. Can be either a dict with 'name' and 'text' keys,
                            or a string in "name: text" format
        episode_name (str): Name of the current episode
    """
    try:
        with open("games/ace_attorney/ace_attorney_1_skip_conversations.json", 'r', encoding='utf-8') as f:
            skip_conversations = json.load(f)
        
        # Handle both dictionary and string formats for dialog
        if isinstance(dialog, dict) and 'name' in dialog and 'text' in dialog:
            dialog_entry = f"{dialog['name']}: {dialog['text']}"
        else:
            dialog_entry = str(dialog)
        
        print("\n=== Checking Skip Dialog ===")
        print(f"Current Dialog: {dialog_entry}")
        print(f"Episode: {episode_name}")
        
        # Check if dialog matches any key in the skip conversations
        episode_convs = skip_conversations.get(episode_name, {})
        if dialog_entry in episode_convs:
            print(f">>> MATCH FOUND! Dialog matches key in skip conversations")
            return episode_convs[dialog_entry]
            
        print("No matching key found in skip conversations")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to check skip conversation: {e}")
        return None

def handle_skip_conversation(system_prompt, api_provider, model_name, prev_response, thinking, modality, episode_name, dialog, skip_dialogs, cache_dir=None):
    """
    Handles skipping through a known conversation sequence.
    Updates long-term memory and performs the necessary moves.
    
    Args:
        dialog (dict): Current dialog that triggered the skip
        skip_dialogs (list): List of dialogs to skip through
        cache_dir (str, optional): Directory to save cache files
    """
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    if not skip_dialogs:
        return None
        
    print("\n" + "="*70)
    print("=== Starting Skip Conversation ===")
    print(f"├── Episode: {episode_name}")
    print(f"├── Number of dialogs to skip: {len(skip_dialogs) - 1}")
    print(f"└── Dialog sequence:")
    for i, skip_dialog in enumerate(skip_dialogs):
        print(f"    {i+1}. {skip_dialog}")
    print("="*70 + "\n")
        
    # Update long-term memory with all dialogs in the sequence
    for skip_dialog in skip_dialogs:
        # Extract name and text from the skip dialog
        name_text = skip_dialog.split(": ", 1)
        if len(name_text) == 2:
            name, text = name_text
            dialog_entry = {"name": name, "text": text}
            long_term_memory_worker(
                system_prompt,
                api_provider,
                model_name,
                prev_response,
                thinking,
                modality,
                episode_name,
                dialog=dialog_entry,
                cache_dir=cache_dir
            )
    
    # Perform 'z' moves for each dialog in the sequence (except the first one)
    for i in range(len(skip_dialogs) - 1):
        print(f"[Skip] Performing 'z' move {i+1} of {len(skip_dialogs) - 1}")
        perform_move("z")
        time.sleep(5)  # Reduced delay between moves since we're handling it centrally
        
        # Check if we've reached the end statement
        if check_end_statement(dialog, episode_name):
            print("\n=== End Statement Reached During Skip ===")
            break
    
    print("\n=== Skip Conversation Complete ===")
    print("="*70 + "\n")
    
    # Return parsed result with continue conversation
    return {
        "game_state": "Conversation",
        "dialog": dialog,
        "evidence": {},
        "scene": "",
        "move": "z",
        "thought": "continue conversation"
    }

def check_end_statement(dialog, episode_name):
    """
    Checks if the current dialog matches the end statement for the episode.
    Returns True if it's the end statement, False otherwise.
    """
    try:
        with open("games/ace_attorney/ace_attorney_1_skip_conversations.json", 'r', encoding='utf-8') as f:
            skip_conversations = json.load(f)
        
        # Handle both dictionary and string formats for dialog
        if isinstance(dialog, dict) and 'name' in dialog and 'text' in dialog:
            dialog_entry = f"{dialog['name']}: {dialog['text']}"
        else:
            dialog_entry = str(dialog)

        end_statements = skip_conversations.get(episode_name, {}).get("end_statements", [])
        
        return dialog_entry in end_statements
    except Exception as e:
        print(f"[ERROR] Failed to check end statement: {e}")
        return False

def vision_only_reasoning_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-only",
    episode_name="The First Turnabout",
    cache_dir=None
    ):
    """
    Combines vision analysis and reasoning in a single step.
    Captures the game screen, analyzes it, and makes decisions based on the analysis.
    Also updates long-term memory with new information.
    """
    assert modality == "vision-only", "This worker requires vision-only modality"
    
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    # Capture game window screenshot
    screenshot_path = capture_game_window(
        image_name="ace_attorney_screenshot.png",
        window_name="Phoenix Wright: Ace Attorney Trilogy",
        cache_dir=cache_dir
    )
    if not screenshot_path:
        return {"error": "Failed to capture game window"}

    base64_image = encode_image(screenshot_path)
    
    # Get memory context for reasoning
    memory_context = memory_retrieval_worker(
        system_prompt,
        api_provider,
        model_name,
        prev_response,
        thinking,
        modality,
        episode_name,
        cache_dir=cache_dir
    )

    # Extract and format evidence information from memory
    evidences_section = memory_context.split("Collected Evidences:")[1].strip()
    collected_evidences = [e for e in evidences_section.split("\n") if e.strip()]
    num_collected_evidences = len(collected_evidences)
    evidence_details = "\n".join([f"Evidence {i+1}: {e}" for i, e in enumerate(collected_evidences)])

    # Construct combined prompt for vision analysis and reasoning
    prompt = f"""You are Phoenix Wright, a defense attorney in Ace Attorney. Your goal is to prove your client's innocence by finding contradictions in witness testimonies and presenting the right evidence at the right time.

        First, analyze the current game screen and provide the following information:

        1. Game State Detection Rules:
           - Cross-Examination mode is indicated by ANY of these:
             * A blue bar in the upper right corner
             * Only Green dialog text
             * EXACTLY three UI elements at the right down corner: Options, Press, Present
             * An evidence window visible in the middle of the screen
           - If you see an evidence window, it is ALWAYS Cross-Examination mode
           - Conversation mode is indicated by:
             * EXACTLY two UI elements at the right down corner: Options, Court Record
             * Dialog text can be any color (most commonly white, but can also be blue, red, etc.)
           - If you don't see any Cross-Examination indicators, it's Conversation mode

        2. Dialog Text Analysis:
           - Look at the bottom-left area where dialog appears
           - Note the color of the dialog text (green/white/blue/red)
           - Extract the speaker's name and their dialog
           - Format must be exactly: Dialog: NAME: dialog text

        3. Scene Analysis:
           - Describe any visible characters and their expressions/poses
           - Describe any other important visual elements or interactive UI components
           - You MUST explicitly mention:
             * The color of the dialog text (green/white/blue/red)
             * Whether there is a blue bar in the upper right corner
             * The exact UI elements present at the right down corner
             * Whether there is an evidence window visible
             * If evidence window is visible:
               - Name of the currently selected evidence
               - Description of the evidence
               - Position in the evidence list (if visible)
               - Whether this is the evidence you intend to present

        Then, based on your analysis, make a decision about the next move:

        Evidence Status:  
        - Total Evidence Collected: {num_collected_evidences}  
        {evidence_details}

        Memory Context:  
        {memory_context}

        Be patient. DO NOT rush to present evidence. Always wait until the **decisive contradiction** becomes clear.

        You may only present evidence if:
        - A clear and specific contradiction exists between the current statement and an item in the Court Record
        - The **correct** evidence item is currently selected
        - The **evidence window is open**, and you are on the exact item you want to present

        Never assume the correct evidence is selected. Always confirm it.

        Available moves:
        * `'l'`: Question the witness about their statement
        * `'z'`: Move to the next statement
        * `'r'`: Open the evidence window (press `'b'` to cancel if unsure)
        * `'b'`: Close the evidence window (if opened unintentionally)
        * `'x'`: Present evidence (only after confirming it's correct)
        * `'right'`: Navigate through the evidence

        Format your response EXACTLY as:
        Game State: <'Cross-Examination' or 'Conversation'>
        Dialog: NAME: dialog text
        Evidence: NAME: description
        Scene: <detailed description>
        move: <move>
        thought: <your internal reasoning>
    """

    # Call the API
    if api_provider == "anthropic":
        response = anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking)
    elif api_provider == "openai":
        response = openai_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "gemini":
        response = gemini_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "together_ai":
        response = together_ai_completion(system_prompt, model_name, prompt, base64_image=base64_image)
    else:
        raise NotImplementedError(f"API provider: {api_provider} is not supported.")
    if "claude" in model_name:
        prompt_message = convert_string_to_messsage(prompt)
    else:
        prompt_message = prompt
    # Update completion in cost data
    cost_data = calculate_all_costs_and_tokens(
        prompt=prompt_message,
        completion=response,
        model=model_name,
        image_path=screenshot_path if base64_image else None
    )
    
    # Log the request costs
    log_request_cost(
        num_input=cost_data["prompt_tokens"] + cost_data.get("image_tokens", 0),
        num_output=cost_data["completion_tokens"],
        input_cost=float(cost_data["prompt_cost"] + cost_data.get("image_cost", 0)),
        output_cost=float(cost_data["completion_cost"]),
        game_name="ace_attorney",
        input_image_tokens=cost_data.get("image_tokens", 0),
        model_name=model_name,
        cache_dir=cache_dir
    )

    # Extract all information from response
    game_state_match = re.search(r"Game State:\s*(Cross-Examination|Conversation)", response)
    game_state = game_state_match.group(1) if game_state_match else "Unknown"
    
    dialog_match = re.search(r"Dialog:\s*([^:\n]+):\s*(.+?)(?=\n|$)", response)
    dialog = {
        "name": dialog_match.group(1) if dialog_match else "",
        "text": dialog_match.group(2).strip() if dialog_match else ""
    }
    
    evidence_match = re.search(r"Evidence:\s*([^:\n]+):\s*(.+?)(?=\n|$)", response)
    evidence = {
        "name": evidence_match.group(1) if evidence_match else "",
        "description": evidence_match.group(2).strip() if evidence_match else ""
    }
    
    scene_match = re.search(r"Scene:\s*((?:.|\n)+?)(?=\n(?:Game State:|Dialog:|Evidence:|Options:|move:|thought:|$)|$)", response, re.DOTALL)
    scene = scene_match.group(1).strip() if scene_match else ""
    
    move_match = re.search(r"move:\s*(.+?)(?=\n|$)", response)
    move = move_match.group(1).strip() if move_match else ""
    
    thought_match = re.search(r"thought:\s*(.+?)(?=\n|$)", response)
    thought = thought_match.group(1).strip() if thought_match else ""

    # Update long-term memory
    if game_state == "Evidence":
        if evidence["name"] and evidence["description"]:
            long_term_memory_worker(
                system_prompt,
                api_provider,
                model_name,
                prev_response,
                thinking,
                modality,
                episode_name,
                evidence=evidence,
                cache_dir=cache_dir
            )
    else:
        if dialog["name"] and dialog["text"]:
            long_term_memory_worker(
                system_prompt,
                api_provider,
                model_name,
                prev_response,
                thinking,
                modality,
                episode_name,
                dialog=dialog,
                cache_dir=cache_dir
            )

    return {
        "game_state": game_state,
        "dialog": dialog,
        "evidence": evidence,
        "scene": scene,
        "screenshot_path": screenshot_path,
        "memory_context": memory_context,
        "move": move,
        "thought": thought
    }

def vision_only_ace_attorney_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    episode_name="The First Turnabout",
    decision_state=None,
    cache_dir=None
    ):
    """
    1) Captures a screenshot of the current game state.
    2) Analyzes the scene using vision worker.
    3) Makes decisions based on the scene analysis.
    4) Maintains dialog history for the current episode.
    5) Makes decisions about game moves.
    
    Args:
        episode_name (str): Name of the current episode (default: "The First Turnabout")
        cache_dir (str, optional): Directory to save cache files
    """
    assert modality in ["text-only", "vision-text", "vision-only"], f"modality {modality} is not supported."
    
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    # Use vision_only_reasoning_worker which combines vision analysis and reasoning
    result = vision_only_reasoning_worker(
        system_prompt,
        api_provider,
        model_name,
        prev_response,
        thinking,
        modality="vision-only",
        episode_name=episode_name,
        cache_dir=cache_dir
    )
    
    if "error" in result:
        return result
    
    # Extract options if present in the scene description
    options = {
        "choices": [],
        "selected": ""
    }
    
    scene = result.get("scene", "")
    
    # Check if options are mentioned in the scene description
    if "option" in scene.lower() and "selected" in scene.lower():
        # Try to parse options from the scene description
        option_lines = [line for line in scene.split('\n') if "option" in line.lower() and ("selected" in line.lower() or "highlighted" in line.lower())]
        
        if option_lines:
            options["choices"] = []
            for line in option_lines:
                # Try to extract option text
                option_text = re.search(r'"([^"]+)"', line)
                if option_text:
                    option_choice = option_text.group(1).strip()
                    options["choices"].append(option_choice)
                    # If this option is selected
                    if "selected" in line.lower() or "highlighted" in line.lower():
                        options["selected"] = option_choice
    
    # Setup decision state for options if needed
    if options["choices"] and not decision_state:
        decision_state = {
            "has_options": True,
            "down_count": 0,
            "selection_index": 0,
            "selected_text": options["choices"][0],  # default to first option
            "decision_timestamp": None
        }
        
    # Update decision state based on move
    if decision_state and result.get("move"):
        if result["move"] == "down" and decision_state["has_options"]:
            decision_state["down_count"] += 1
            i = min(decision_state["down_count"], len(options["choices"]) - 1)
            decision_state["selection_index"] = i
            decision_state["selected_text"] = options["choices"][i]
            
        if result["move"] == "z" and decision_state["has_options"]:
            decision_state["decision_timestamp"] = time.time()
            print(f"[Decision Made] Selected option: '{decision_state['selected_text']}' at index {decision_state['selection_index']} (via {decision_state['down_count']} down moves)")
    
    # Add options and decision state to the result
    result["options"] = options
    result["decision_state"] = decision_state
    
    return result

# return move_thought_list