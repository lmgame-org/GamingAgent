import logging
from typing import Dict, List, Optional
import json
from datetime import datetime
import os
import requests

logger = logging.getLogger(__name__)

# Meta-critic prompts from the original repo
META_KNOWLEDGE_PROMPT = """
Examine the conversation history you have been provided, which is of an error-prone agent playing Pokemon Red.

Your job is to deduce the current state of the game from that conversation, as well as additional data you will be provided:
1. A screenshot of the game currently
2. An text_based collision map of the current location, based on exploration so far.
3. Information gathered from the RAM state of the game.
4. A list of checkpoints logged by the agent to track progress.
5. Labels for map locations assigned by the agent and other code.
6. A previous summary of the state of the game.

It is important to understand the grid system used on the text_based map and for the label list:

1. The top-left corner of the location is at 0, 0
2. The first number is the column number, increasing horizontally to the right.
3. The second number is the row number, increasing vertically downward.

Some example reasoning: If the top left of the text_based map is at (3, 38), then we are at least 38 units away from the top of the map. This is
relevant when looking for exits on the north or left of the map.

The numbers on the map indicate how far away any given tile is from the player character in terms of actual walking paths (not raw distance).

An important subgoal in every new location is to thoroughly explore the area. In mazes, it is often faster to find the exit by EXPLORING rather than
trying to go straight for the exit. Make sure to emphasize this when looking at your text_based map, and include it in your goals in large maps.

Please write down a list of FACTS about the current game state, organized into the following groups, sorted from most reliable to least reliable:

1. Data from RAM (100% accurate. This is provided directly by the developer and is not to be questioned.)
2. Information from your own knowledge about Pokemon Red (Mostly reliable, dependent on recollection)
3. Information from the checkpoints (Mostly reliable)
4. Information from the text_based map (Mostly reliable, dependent on accuracy reading the map)
5. Information from the previous game summary (Somewhat reliable, but outdated)
6. Labels for map locations assigned by the agent and other code. (Somewhat reliable)
7. Information from inspecting the screenshot (Not very reliable, due to mistakes in visual identification)
8. Information from the conversation history (Not very reliable; the agent is error-prone)

KEEP IN MIND: The MOST IMPORTANT thing you do is keep track of what the next step is to progress the game. If you encounter evidence that the game is
not in the expected state (a road is blocked, a HM is missing, etc.), you need to notice right away and include these observations.

Think VERY CAREFULLY about category 2. It is easy to accidentally leave out key steps that aren't very well known or are counterintuitive.
Pokemon Red is full of unexpected blocks to progress that require doing something unexpected to clear. A road may be blocked because of
a completely unrelated reason in the game logic. Please work hard to recall these details about the game.

Ensure that the information provided is grouped into these 4 groups, and that there is enough facts listed for another agent to continue
playing the game just by inspecting the list. Ensure that the following information is contained:

1. Key game events and milestones reached
2. Important decisions made
3. Current key objectives or goals

Make sure that each fact has a percentage next to it indicating how reliable you think it is (e.g. 0%, 25%, 50%, 100%)

Note: At times there will be long periods of nonactivity where another program is handling navigation between battles in an area. This is expected and normal.
"""

META_KNOWLEDGE_CLEANUP_PROMPT = """
Your job is to curate a list of assertions about the game state of a playthrough of Pokemon Red by an error-prone agent.

These will be provided to you in 4 groups, ranging from more to less reliable:

1. Data from RAM (100% accurate. This is provided directly by the developer and is not to be questioned.)
2. Information from your own knowledge about Pokemon Red (Mostly reliable, dependent on recollection)
3. Information from the checkpoints (Mostly reliable)
4. Information from the text_based map (Mostly reliable, dependent on accuracy reading the map)
5. Information from the previous game summary (Somewhat reliable, but outdated)
6. Labels for map locations assigned by the agent and other code. (Somewhat reliable)
7. Information from inspecting the screenshot (Not very reliable, due to mistakes in visual identification)
8. Information from the conversation history (Not very reliable; the agent is error-prone)

Next to each fact you will likely find a percentage indicating how reliable the fact is. Use this as a guide and avoid using unreliable facts.

Using the data from the _more_ reliable fact groups, please remove any inaccuracies from the data from the less reliable fact groups. Remove anything that doesn't make sense.

Examples:
1. The data from RAM says the current location is VIRIDIAN_CITY but the conversation history claims the current location is PALLET_TOWN
    1a. ANSWER: Delete the claim that the location is PALLET_TOWN, since the RAM data is far more reliable than conversation history.
2. The data from Knowledge about Pokemon Red asserts that after leaving the starting house, you have to go North of Pallet Town to trigger an encounter with Professor Oak. The previous game summary does not mention that this has happened yet.
   But on the screenshot it appears that Professor Oak is already standing inside Oak's Lab, and the conversation history mentions trying to talk with Professor Oak.
    2b. ANSWER: Delete any claims that Professor Oak is in the lab or needs to be talked to, and emphasize that you must go north of Pallet Town. Previous knowledge of Pokemon Red and the previous game summary is much more reliable than glasncing at the screenshot or the error-prone assertions in the conversation history.

In addition, delete facts from the less reliable sources (7, 8) if they are not very reliable, and also delete any coordinate information contained in these categories, as they are often wrong.

Output a corrected list of facts about the game state. Make sure that each fact has a percentage next to it indicating how reliable you think it is (e.g. 0%, 25%, 50%, 100%)

Ensure that the information provided is grouped into these 4 groups, and that there is enough facts listed for another agent to continue
playing the game just by inspecting the list. Ensure that the following information is contained:

1. Key game events and milestones reached
2. Important decisions made
3. Current key objectives or goals

Note: At times there will be long periods of nonactivity where another program is handling navigation between battles in an area. This is expected and normal.
"""

META_KNOWLEDGE_SUMMARIZER = """I need you to create a detailed summary of Pokemon Red game progress up to this point,
using a curated list of FACTS you will be provided. This information will be used to guide an agent to continue playing and progressing in the game.

Next to each fact you will likely find a percentage indicating how reliable the fact is. Use this as a guide and avoid using unreliable facts.

Ensure that the summary you provide contains the following information:

1. Key game events and milestones reached
2. Important decisions made
3. Current key objectives or goals

Make sure that each fact has a percentage next to it indicating how reliable you think it is (e.g. 0%, 25%, 50%, 100%)

Once this is done, inspect the conversation history and if the conversation shows signs of serious difficulty completing a task.
Append a section of IMPORTANT HINTS to help guide progress. 

PRIORITY ONE: If the conversation history shows gameplay that is in violation of the facts you have been provided, issue corrective guidance
about the CORRECT way to proceed.

PRIORITY TWO: If the conversation history shows signs of navigation problems, try to assist the agent with the following tips.
One big sign of navigation problems is if the model has been trying to navigate and area for more than 300 steps.

TIPS TO PROVIDE FOR NAVIGATION:
1. If a label is incorrect, STRONGLY ENCOURAGE stopping to edit the label to something else (potentially even " ").
2. Remind the agent to consult its text_based map.
3. Remember that "navigate_to_offscreen_coordinate" and the "detailed_navigator" tool are there to query for help.
4. If they seem to be stuck in a location, emphasize the importance of NOT revisiting EXPLORED tiles. It may even be PRIORITY ONE to stop stepping on EXPLORED tiles.
5. In mazes, it is MORE IMPORTANT to avoid EXPLORED tiles than to go in the correct direction.
    5a. Often in mazes, you have to go south first to eventually go north, for example. This can be very far -- 30 or more coordinate squaares away.
    5b. In Mazes, it is important to label dead-ends to avoid repeated visits, particularly if they are covered in EXPLORED tiles.
    5c. 0, 0 is the topmost-leftmost part of the map.
    5d. A DEPTH-FIRST SEARCH, using EXPLORED tiles as markers of previous locations, is a great way to get through mazes. Don't turn around unless you run into a dead end.
6. Remind about the BIG HINTS:
   6a. Doors and stairs are NEVER IMPASSABLE.
   6b. By extension, squares that are EXPLORED are NEVER Doors or stairs.
   6c. IMPASSABLE Squares are never the exit from an area UNLESS they are directly on top of the black void at the edge of the map. There must be a passable (non-red) path INTO the black area for this to work.
7. Pay attention to the text_based maps and whether the direction of travel is sensible. They may be pathing into a dead end!
   

OTHER NOTES:
1. If the wrong NPC is talked to frequently, remind yourself to label a wrong NPC's location (on the NPC's location)
2. If they are trying to reach a location on screen, remind them that the "navigate_to" tool may be able to get them there.

When hinting, AVOID repeating coordinates or locations you do not see on screen from the conversation history -- the conversation is often
mistaken about the exact location of objects or NPCs, and repeating it can reinforce the mistake.

HOWEVER coordinates you get from the summary are reliable.

Note: At times there will be long periods of nonactivity where another program is handling navigation between battles in an area. This is expected and normal.
"""

class MetaCritiqueSystem:
    def __init__(self, checkpoint_file: str = "checkpoints.json", model_name: str = "default", vllm_url: Optional[str] = None, modal_url: Optional[str] = None, runner_log_dir_base: Optional[str] = None):
        # Use runner_log_dir_base if provided, otherwise fallback to default cache directory
        if runner_log_dir_base is None:
            runner_log_dir_base = "."
            
        # Ensure the directory exists
        os.makedirs(runner_log_dir_base, exist_ok=True)
        
        self.checkpoint_file = os.path.join(runner_log_dir_base, checkpoint_file)
        
        self.checkpoints: List[Dict] = []
        self.current_state = {
            "location": None,
            "inventory": [],
            "pokemon": [],
            "quest_state": {},
            "recent_events": [],
            "game_progress": {},
            "conversation_history": [],
            "text_based_map": None,
            "screenshot": None
        }
        self.model_name = model_name
        self.vllm_url = vllm_url
        self.modal_url = modal_url
        self._load_checkpoints()
        logger.info(f"MetaCritiqueSystem initialized")
        logger.info(f"Checkpoints will be saved to: {self.checkpoint_file}")

    def _load_checkpoints(self):
        """Load existing checkpoints from file."""
        try:
            with open(self.checkpoint_file, 'r') as f:
                self.checkpoints = json.load(f)
            logger.info(f"Loaded {len(self.checkpoints)} existing checkpoints from {self.checkpoint_file}")
        except FileNotFoundError:
            self.checkpoints = []
            logger.info(f"No existing checkpoints found at {self.checkpoint_file}. Starting fresh.")

    def _save_checkpoints(self):
        """Save checkpoints to file."""
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoints, f, indent=2)
        logger.info(f"Saved {len(self.checkpoints)} checkpoints to {self.checkpoint_file}")

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API with the given prompt."""
        try:
            if self.vllm_url:
                logger.info(f"Calling vLLM API at {self.vllm_url}")
                response = requests.post(
                    self.vllm_url,
                    json={"prompt": prompt, "model": self.model_name}
                )
                return response.json()["response"]
            elif self.modal_url:
                logger.info(f"Calling Modal API at {self.modal_url}")
                response = requests.post(
                    self.modal_url,
                    json={"prompt": prompt, "model": self.model_name}
                )
                return response.json()["response"]
            else:
                logger.warning("No LLM API URL provided. Returning prompt as is.")
                return prompt
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return prompt

    def mark_checkpoint(self, 
                       location: Optional[str] = None,
                       inventory: Optional[List] = None,
                       pokemon: Optional[List] = None,
                       quest_state: Optional[Dict] = None,
                       event: Optional[str] = None,
                       game_progress: Optional[Dict] = None,
                       text_based_map: Optional[str] = None,
                       screenshot: Optional[str] = None):
        """
        Record a checkpoint with the current game state.
        """
        timestamp = datetime.now().isoformat()
        
        # Log state updates
        if location:
            logger.info(f"Updating location: {location}")
            self.current_state["location"] = location
        if inventory:
            logger.info(f"Updating inventory: {inventory}")
            self.current_state["inventory"] = inventory
        if pokemon:
            logger.info(f"Updating Pokemon party: {pokemon}")
            self.current_state["pokemon"] = pokemon
        if quest_state:
            logger.info(f"Updating quest state: {quest_state}")
            self.current_state["quest_state"].update(quest_state)
        if event:
            logger.info(f"Recording event: {event}")
            self.current_state["recent_events"].append({
                "timestamp": timestamp,
                "description": event
            })
            # Keep only last 10 events
            self.current_state["recent_events"] = self.current_state["recent_events"][-10:]
        if game_progress:
            logger.info(f"Updating game progress: {game_progress}")
            self.current_state["game_progress"].update(game_progress)
        if text_based_map:
            logger.info("Updating text-based map")
            self.current_state["text_based_map"] = text_based_map
        if screenshot:
            logger.info("Updating screenshot")
            self.current_state["screenshot"] = screenshot

        # Create checkpoint with all current state
        checkpoint = {
            "timestamp": timestamp,
            "state": self.current_state.copy()
        }
        
        self.checkpoints.append(checkpoint)
        self._save_checkpoints()
        
        logger.info(f"Checkpoint recorded at {timestamp} with full state information")

    def get_context_summary(self) -> str:
        """
        Generate a summary of the current game context using the meta-knowledge prompts.
        """
        logger.info("Generating context summary")
        
        # First, get the initial analysis using META_KNOWLEDGE_PROMPT
        logger.info("Getting initial analysis")
        initial_analysis = self._call_llm(META_KNOWLEDGE_PROMPT)
        
        # Then, clean up the analysis using META_KNOWLEDGE_CLEANUP_PROMPT
        logger.info("Cleaning up analysis")
        cleaned_analysis = self._call_llm(f"{META_KNOWLEDGE_CLEANUP_PROMPT}\n\n{initial_analysis}")
        
        # Finally, create a summary using META_KNOWLEDGE_SUMMARIZER
        logger.info("Creating final summary")
        summary = self._call_llm(f"{META_KNOWLEDGE_SUMMARIZER}\n\n{cleaned_analysis}")
        
        logger.info("Context summary generated successfully")
        return summary

    def get_meta_critique(self, action: str, observation: str) -> str:
        """
        Generate a meta-critique of the current action and observation using the LLM.
        """
        logger.info(f"Generating meta-critique for action: {action}")
        
        # Get current context
        context = self.get_context_summary()
        
        # Format recent events
        events = "\n".join([
            f"- {event['description']}"
            for event in self.current_state["recent_events"][-5:]
        ])
        
        # Format the prompt
        prompt = META_KNOWLEDGE_PROMPT.format(
            context=context,
            events=events
        )
        
        # Add action and observation
        prompt += f"\nAction: {action}\nObservation: {observation}\n"
        
        # Call LLM for analysis
        logger.info("Calling LLM for meta-critique")
        critique = self._call_llm(prompt)
        
        logger.info("Meta-critique generated successfully")
        return critique

    def verify_consistency(self, observation: str) -> bool:
        """
        Verify if the current observation is consistent with our recorded state.
        """
        # Basic consistency checks
        if not self.current_state["location"]:
            logger.info("No location recorded yet, skipping consistency check")
            return True
            
        # Check if location in observation matches recorded location
        if self.current_state["location"].lower() not in observation.lower():
            logger.warning(f"Location mismatch: Expected {self.current_state['location']}, got observation: {observation}")
            return False
            
        logger.info("Consistency check passed")
        return True 