import time
import os
import pyautogui
import numpy as np
import json
import re
import datetime

from tools.utils import encode_image, log_output, get_annotate_img, capture_game_window, log_request_cost
from tools.serving.api_providers import anthropic_completion, anthropic_text_completion, openai_completion, openai_text_reasoning_completion, gemini_completion, gemini_text_completion, deepseek_text_reasoning_completion, together_ai_completion
from tools.api_cost_calculator import calculate_all_costs_and_tokens, convert_string_to_messsage

# Default cache directory (can be overridden by passing cache_dir parameter)
DEFAULT_CACHE_DIR = "cache/ace_attorney"

# Load prompts from JSON file
with open("games/ace_attorney/ace_attorney_prompts.json", 'r', encoding='utf-8') as f:
    PROMPTS = json.load(f)

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
    
    # Use prompt from JSON file
    prompt = PROMPTS["vision_evidence_prompt"]

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
    # cost_data = calculate_all_costs_and_tokens(
    #     prompt=prompt_message,
    #     completion=response,
    #     model=model_name,
    #     image_path=screenshot_path if base64_image else None
    # )
    
    # # Log the request costs
    # log_request_cost(
    #     num_input=cost_data["prompt_tokens"] + cost_data.get("image_tokens", 0),
    #     num_output=cost_data["completion_tokens"],
    #     input_cost=float(cost_data["prompt_cost"] + cost_data.get("image_cost", 0)),
    #     output_cost=float(cost_data["completion_cost"]),
    #     game_name="ace_attorney",
    #     input_image_tokens=cost_data.get("image_tokens", 0),
    #     model_name = model_name,
    #     cache_dir=cache_dir
    # )
    
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

    # Use prompt from JSON file
    prompt = PROMPTS["vision_worker_prompt"]

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
    # cost_data = calculate_all_costs_and_tokens(
    #     prompt=prompt_message,
    #     completion=response,
    #     model=model_name,
    #     image_path=screenshot_path if base64_image else None
    # )
    
    # # Log the request costs
    # log_request_cost(
    #     num_input=cost_data["prompt_tokens"] + cost_data.get("image_tokens", 0),
    #     num_output=cost_data["completion_tokens"],
    #     input_cost=float(cost_data["prompt_cost"] + cost_data.get("image_cost", 0)),
    #     output_cost=float(cost_data["completion_cost"]),
    #     game_name="ace_attorney",
    #     input_image_tokens=cost_data.get("image_tokens", 0),
    #     model_name=model_name,
    #     cache_dir=cache_dir
    # )
    
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
    
    # Add new response and maintain only last 30 responses
    prev_responses = dialog_history[episode_name]["prev_responses"]
    prev_responses.append(prev_response)
    if len(prev_responses) > 20:
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

def normalize_content(content, episode_name, cache_dir=None):
    """
    Normalizes content by replacing specific names with generic ones and evidence with symbols.
    
    Args:
        content (str): The content to normalize
        episode_name (str): Name of the current episode
        cache_dir (str, optional): Directory to save cache files
    
    Returns:
        str: Normalized content
    """
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    try:
        with open("games/ace_attorney/mapping.json", 'r', encoding='utf-8') as f:
            map_elements = json.load(f)
        
        episode_elems = map_elements.get(episode_name, {})
        name_mappings = episode_elems.get("name_mappings", {})
        evidence_mappings = episode_elems.get("evidence_mappings", {})
        dialog_mappings = episode_elems.get("dialog", {})
        
    except Exception as e:
        print(f"[ERROR] Failed to check skip conversation: {e}")
        return None
    
    # --- Swapped Order: Process Dialog First ---
    # Replace specific dialog phrases using case-insensitive word boundary matching
    for original_dialog, replacement_dialog in dialog_mappings.items():
        pattern = re.compile(r"^" + re.escape(original_dialog) + r"$", re.IGNORECASE)
        new_content = pattern.sub(replacement_dialog, content)
        if new_content != content:
            content = new_content
    # Replace evidence with case-insensitive boundaries
    for original, replacement in evidence_mappings.items():
        pattern = re.compile(r"\b" + re.escape(original) + r"\b", re.IGNORECASE)
        content = pattern.sub(replacement, content)

    # Replace names with case-insensitive boundaries
    for original, replacement in name_mappings.items():
        pattern = re.compile(r"\b" + re.escape(original) + r"\b", re.IGNORECASE)
        content = pattern.sub(replacement, content)

    return content

def update_state_file(game_state, c_statement, scene, memory_context, evidence_details, episode_name, cache_dir=None, is_repeated_statement=False):
    """
    Updates the state JSON file in the cache directory with current game state information.
    
    Args:
        game_state (str): Current game state
        c_statement (str): Current statement
        scene (str): Scene description
        memory_context (str): Memory context
        evidence_details (str): Evidence details
        episode_name (str): Name of the current episode
        cache_dir (str, optional): Directory to save cache files
        is_repeated_statement (bool): Whether the statement is repeated
    """
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Create state file path
    state_file = os.path.join(cache_dir, "game_state.json")
    
    # Load existing state or create new
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
    else:
        state = {
            "episode_name": episode_name,
            "history": []
        }
    
    # Add new state entry
    state_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "game_state": game_state,
        "current_statement": c_statement,
        "scene": scene,
        "memory_context": memory_context,
        "evidence_details": evidence_details,
        "is_repeated_statement": is_repeated_statement
    }
    
    state["history"].append(state_entry)
    
    # Save updated state
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def reasoning_worker(options, system_prompt, api_provider, model_name, game_state, c_statement, scene, memory_context, base64_image=None, modality="vision-text", thinking=True, screenshot_path=None, cache_dir=None, episode_name="The First Turnabout"):
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
        episode_name (str): Name of the current episode
    
    Returns:
        dict: Contains move and thought
    """
    # Use provided cache_dir or default
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    # set image as None to avoid data contamination
    base64_image = None

    # Extract and format evidence information
    evidences_section = memory_context.split("Collected Evidences:")[1].strip()
    collected_evidences = [e for e in evidences_section.split("\n") if e.strip()]
    num_collected_evidences = len(collected_evidences)
    # Normalize all content
    game_state = normalize_content(game_state, episode_name, cache_dir)
    c_statement = normalize_content(c_statement, episode_name, cache_dir)
    scene = normalize_content(scene, episode_name, cache_dir)
    memory_context = normalize_content(memory_context, episode_name, cache_dir)
    
    
    # Format evidence details for the prompt and normalize
    evidence_details = "\n".join([f"Evidence {i+1}: {e}" for i, e in enumerate(collected_evidences)])
    evidence_details = normalize_content(evidence_details, episode_name, cache_dir)

    if game_state == "Cross-Examination":
        # Check if statement exists in dialog history
        dialog_history_dir = os.path.join(cache_dir, "dialog_history")
        json_file = os.path.join(dialog_history_dir, f"{episode_name.lower().replace(' ', '_')}.json")
        
        is_repeated_statement = False
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                dialog_history = json.load(f)
                case_transcript = dialog_history[episode_name]["Case_Transcript"]
                # Check if current statement exists in transcript
                if isinstance(c_statement, dict):
                    statement_text = f"{c_statement.get('name', '')}: {c_statement.get('text', '')}"
                else:
                    statement_text = str(c_statement)
                is_repeated_statement = any(statement_text in entry for entry in case_transcript)

        # Update state file with repeated statement status
        update_state_file(game_state, c_statement, scene, memory_context, evidence_details, episode_name, cache_dir, is_repeated_statement)

        # Define prompt directly instead of loading from JSON
        prompt = """You are a diligent defense advocate.  
Forget anything you might recall from popular courtroom-game scripts; such memories may be unreliable here.
Focus on logical reasoning and evidence-based decision making in your analysis.

CURRENT GAME STATE: {game_state}

Your task is to evaluate the **current witness statement** and determine whether it contradicts any evidence in the Court Record.

Current Statement: 
"{c_statement}"

{repeated_statement_warning}

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
- When there are multiple options available:
    * Choose the option that best advances your case
    * Do NOT compare statements with evidence, try to make the best decision based on the options and statements
    * Focus on the strategic value of each option
- When the evidence window is visible:
    * Carefully examine the currently selected evidence
    * Compare it with the current statement for contradictions
    * If it's not the correct evidence, use 'right' to navigate to the evidence that contradicts the statement
    * Only use 'x' when you've found the evidence that directly contradicts the current statement
    * If no evidence contradicts the statement, use 'b' to close the evidence window
    * You may always try a new piece of evidence that you haven't presented before as an attempt, even if there's no clear contradiction
- When no options are present and no evidence window is visible:
    * ALWAYS compare the witness's statement with the available evidence
    * Look for contradictions or inconsistencies
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
    * If a previous option choice led to no progress or was wrong, try a different option
    * Keep track of which options you've tried and their outcomes in your prev_responses
    * Don't repeat options that didn't work unless you have new evidence or context
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

thought: Cause: <detailed explanation>; Evidence: <current state and target>; Effect: <expected outcome>; Reflection: <how this move relates to previous actions and maintains coherence>; Selected_Option: <the currently selected option if options are present>; Selected_Evidence: <the currently selected evidence if navigating evidence>; Presented_Evidence: <the evidence being presented when using 'x' move>

self_evaluation: <Yes / No>   # "Yes" if the Effect truly follows from the Cause + Evidence

IMPORTANT:
- If the evidence window is already open, do NOT use 'r' again
- Check what evidence is selected (based on scene description)
- Use 'right' to navigate if it's not the correct one
- Only use 'x' when the right evidence is selected
- If options are on screen, navigate with 'down', confirm with 'z'
- When in Conversation state with options, focus on choosing the best option to advance the story, NOT on evidence comparison
- If you see the same option again, it means your previous selection was wrong. You should try a different option instead
- Always include the currently selected option in your thought process when options are present
- If you find yourself selecting the same evidence multiple times, it likely means your previous evidence choice was wrong. Try a different piece of evidence instead
- Always include the currently selected evidence in your thought process when navigating evidence
- Always include the presented evidence in your thought process when using 'x' move

Example 1:
Scene says: "The currently selected evidence is E1."
But I want to present: "E2"

Turn 1:  
move: right  
thought: Cause: Need to present E2 to contradict witness's alibi; Evidence: Currently on E1, need to navigate to E2; Effect: Will be able to present the correct evidence that shows witness was at crime scene; Reflection: Making a strategic choice to present the most relevant evidence; Selected_Evidence: E2
self_evaluation: Yes

Turn 2:  
move: x  
thought: Cause: E2 is now selected and directly contradicts witness's statement; Evidence: E2 shows witness at crime scene at 8 PM; Effect: Will expose the contradiction in witness's alibi; Reflection: Confirming the evidence selection; Presented_Evidence: E2
self_evaluation: Yes

Example 2 - Clear Contradiction with No Evidence Window:
Memory Context:
Witness: "I was at home at 8 PM last night."
Evidence: "E1: Security Camera Footage shows the witness at the crime scene at 8 PM."

Scene: "Dialog text is green. There is a blue bar in the upper right corner. There are exactly three UI elements at the bottom-right corner: Options, Press, Present. No evidence window is visible. The witness is sweating and looking nervous."

Turn 1:
move: r
thought: Cause: Witness claims to be at home at 8 PM but E1 shows otherwise; Evidence: E1 (Security Camera) not yet accessible; Effect: Need to open evidence window to present E1
self_evaluation: Yes

Turn 2:
move: right
thought: Cause: Need to find E1 to contradict witness's alibi; Evidence: Currently on first evidence, must navigate to E1; Effect: Will be able to present the security camera footage
self_evaluation: Yes

Turn 3:
move: x
thought: Cause: E1 is now selected and directly contradicts witness's statement; Evidence: E1 shows witness at crime scene at 8 PM; Effect: Will expose the contradiction in witness's alibi
self_evaluation: Yes

Example 3 - Using 'down' to select an option before confirming:

Scene: "Two white-text options appear in the middle of the screen: 'Yes' and 'No'. 'No' is currently highlighted. Dialog text is white. This is Cross-Examination mode."

I want to answer yes, so I need to switch to 'Yes' before confirming.

Turn 1:
move: down
thought: Cause: Need to select 'Yes' to proceed with questioning; Evidence: Currently on 'No' option; Effect: Will be able to confirm the correct choice; Reflection: Making a strategic choice to advance the questioning; Selected_Option: 'Yes'
self_evaluation: Yes

Turn 2:
move: z
thought: Cause: 'Yes' is now selected and is the correct choice; Evidence: Option is highlighted; Effect: Will proceed with the questioning; Reflection: Confirming the strategic choice; Selected_Option: 'Yes'
self_evaluation: Yes


Stuck Situation Handling:
- If no progress has been made in the last 5 responses with cross-examination game state (check prev_responses about whether they are the same.)
- If the agent seems stuck in a loop or unable to advance
- Use 'b' to break out of the loop
- This helps the agent recover and move forward in the game""".format(
            game_state=game_state,
            c_statement=c_statement,
            repeated_statement_warning="WARNING: This statement has been repeated during cross-examination. This often indicates a potential contradiction point. If you don't find a clear contradiction, use 'l' to press for more details." if is_repeated_statement else "",
            options=options,
            scene=scene,
            num_collected_evidences=num_collected_evidences,
            evidence_details=evidence_details,
            memory_context=memory_context
        )

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
        # # Update completion in cost data
        # cost_data = calculate_all_costs_and_tokens(
        #     prompt=prompt_message,
        #     completion=response,
        #     model=model_name,
        #     image_path=screenshot_path if base64_image else None
        # )
        
        # # Log the request costs
        # log_request_cost(
        #     num_input=cost_data["prompt_tokens"] + cost_data.get("image_tokens", 0),
        #     num_output=cost_data["completion_tokens"],
        #     input_cost=float(cost_data["prompt_cost"] + cost_data.get("image_cost", 0)),
        #     output_cost=float(cost_data["completion_cost"]),
        #     game_name="ace_attorney",
        #     input_image_tokens=cost_data.get("image_tokens", 0),
        #     model_name=model_name,
        #     cache_dir=cache_dir
        # )

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
        # For non-cross-examination states, update state file with is_repeated_statement=False
        update_state_file(game_state, c_statement, scene, memory_context, evidence_details, episode_name, cache_dir, False)
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
        system_prompt (str): System prompt for the API
        api_provider (str): API provider to use
        model_name (str): Model name to use
        prev_response (str): Previous response from the API
        thinking (bool): Whether to use deep thinking
        modality (str): Modality to use (vision-text or text-only)
        episode_name (str): Name of the current episode
        decision_state (dict, optional): Current decision state
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
    print(dialog['text'])
    rewrite_match = re.search(r"Rewrite:\s*(.+?)(?=\n[A-Z]|$)", response_text, re.DOTALL)
    rewrite_text = rewrite_match.group(1).strip() if rewrite_match else ""
    if rewrite_text and game_state=="Cross-Examination":
        print('rewrite_text')
        dialog['text'] = rewrite_text
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

    # Format the dialog string properly before passing to reasoning_worker
    if dialog["name"] and dialog["text"]:
        formatted_dialog_statement = f"{dialog['name']}: {dialog['text']}"
    else:
        formatted_dialog_statement = ""

    # -------------------- Reasoning -------------------- #
    # Make decisions about game moves
    reasoning_result = reasoning_worker(
        options,
        system_prompt,
        api_provider,
        model_name,
        game_state,
        formatted_dialog_statement,
        scene,
        complete_memory,
        base64_image=encode_image(vision_result["screenshot_path"]),
        modality='text-only',
        screenshot_path=vision_result["screenshot_path"],
        thinking=thinking,
        cache_dir=cache_dir,
        episode_name=episode_name
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

    # Use prompt from JSON file and format it with the required variables
    prompt = PROMPTS["vision_only_reasoning_prompt"].format(
        num_collected_evidences=num_collected_evidences,
        evidence_details=evidence_details,
        memory_context=memory_context
    )

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
    # # Update completion in cost data
    # cost_data = calculate_all_costs_and_tokens(
    #     prompt=prompt_message,
    #     completion=response,
    #     model=model_name,
    #     image_path=screenshot_path if base64_image else None
    # )
    
    # # Log the request costs
    # log_request_cost(
    #     num_input=cost_data["prompt_tokens"] + cost_data.get("image_tokens", 0),
    #     num_output=cost_data["completion_tokens"],
    #     input_cost=float(cost_data["prompt_cost"] + cost_data.get("image_cost", 0)),
    #     output_cost=float(cost_data["completion_cost"]),
    #     game_name="ace_attorney",
    #     input_image_tokens=cost_data.get("image_tokens", 0),
    #     model_name=model_name,
    #     cache_dir=cache_dir
    # )

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