import os
import time
from pyboy import PyBoy
import logging
import shutil
import sys
from datetime import datetime
import numpy as np
from PIL import Image

# Configure logging
log_dir = "gamingagent/envs/custom_05_pokemon_red/states"
frames_dir = os.path.join(log_dir, "frames")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"pokemon_red_state_creation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global frame counter
frame_counter = 0

def save_frame(pyboy, description):
    """Save a frame from the game with a description."""
    global frame_counter
    try:
        # Get the screen from PyBoy
        screen = pyboy.screen.ndarray
        if screen is None:
            logger.warning(f"Failed to get screen for frame {description}")
            return
            
        # Convert RGBA to RGB if needed
        if screen.shape[-1] == 4:
            screen = screen[:, :, :3]
            
        # Convert to PIL Image
        image = Image.fromarray(screen)
        
        # Save the frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_counter:04d}_{description}.png")
        image.save(frame_path)
        logger.info(f"Saved frame {frame_counter} to {frame_path}")
        
        # Increment frame counter
        frame_counter += 1
        
    except Exception as e:
        logger.error(f"Failed to save frame {frame_counter}: {str(e)}")

def verify_game_state(pyboy, expected_map_id=None, expected_name=None, expected_coords=None):
    """Verify the current game state matches expectations"""
    try:
        # Read map ID and coordinates
        map_id = pyboy.memory[0xD35E]
        x_coord = pyboy.memory[0xD362]
        y_coord = pyboy.memory[0xD361]
        name_bytes = pyboy.memory[0xD158:0xD163]
        name = "".join(chr(b) for b in name_bytes if b != 0x50)
        
        # Only log if we have meaningful values
        if map_id != 0 or name or (x_coord != 0 and y_coord != 0):
            logger.info(f"Current state:")
            logger.info(f"- Map ID: {map_id}")
            logger.info(f"- Coordinates: ({x_coord}, {y_coord})")
            if name:
                logger.info(f"- Player name: {name}")
        
        # Check expectations
        if expected_map_id is not None and map_id != expected_map_id:
            logger.warning(f"Expected map ID {expected_map_id}, got {map_id}")
            return False
            
        if expected_name is not None and name != expected_name:
            logger.warning(f"Expected name {expected_name}, got {name}")
            return False
            
        if expected_coords is not None and (x_coord, y_coord) != expected_coords:
            logger.warning(f"Expected coordinates {expected_coords}, got ({x_coord}, {y_coord})")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error verifying game state: {str(e)}")
        return False

def press_button(pyboy, button, frames=240, wait_time=1.0, capture_frame=True, description=""):
    """Press a button and optionally capture the frame"""
    pyboy.button_press(button)
    pyboy.tick(frames)
    pyboy.button_release(button)
    if wait_time > 0:
        time.sleep(wait_time)
    if capture_frame:
        save_frame(pyboy, f"after_{button}_{description}")

def wait_for_state(pyboy, expected_map_id=None, expected_name=None, expected_coords=None, 
                  max_attempts=50, check_interval=1.0, description="", 
                  frame_capture_interval=5):
    """Wait for a specific game state to be reached"""
    logger.info(f"Waiting for state: {description}")
    for i in range(max_attempts):
        if i % frame_capture_interval == 0:
            save_frame(pyboy, f"waiting_for_{description}_{i}")
        
        if verify_game_state(pyboy, expected_map_id, expected_name, expected_coords):
            save_frame(pyboy, f"reached_{description}")
            return True
            
        time.sleep(check_interval)
    
    logger.warning(f"Failed to reach state {description} after {max_attempts} attempts")
    return False

def handle_name_entry(pyboy):
    """Handle the player name entry screen by selecting 'RED' from the dropdown."""
    logger.info("Handling player name entry...")
    
    # Wait for name selection menu to appear
    time.sleep(2)
    save_frame(pyboy, "before_name_entry")
    
    # Press A multiple times to advance through dialogue
    for i in range(5):
        press_button(pyboy, "a", frames=240, wait_time=1.0, 
                    capture_frame=True, description=f"advance_dialogue_{i+1}")
        time.sleep(1)
    
    # Press down multiple times to ensure we're at the top of the list
    for i in range(3):
        press_button(pyboy, "down", frames=240, wait_time=0.5, 
                    capture_frame=True, description=f"scrolling_down_{i+1}")
    
    # Press down to select 'RED' from the dropdown
    press_button(pyboy, "down", frames=240, wait_time=1.0, 
                capture_frame=True, description="selected_RED")
    
    # Press A to confirm selection
    press_button(pyboy, "a", frames=240, wait_time=1.0, 
                capture_frame=True, description="confirmed_RED")
    
    # Wait a bit for the name to be set
    time.sleep(2)
    
    # Verify name was set
    name_bytes = pyboy.memory[0xD158:0xD163]
    name = "".join(chr(b) for b in name_bytes if b != 0x50)
    logger.info(f"Set player name to: {name}")
    
    # Save final frame
    save_frame(pyboy, "after_name_entry")
    
    # Additional verification
    if not name or name.strip() == "":
        logger.warning("Name appears to be empty, trying alternative memory locations")
        alt_locations = [
            (0xC4A4, 0xC4AF),
            (0xD158, 0xD163),
            (0xD35E, 0xD369)
        ]
        for start, end in alt_locations:
            try:
                alt_name_bytes = pyboy.memory[start:end]
                alt_name = "".join(chr(b) for b in alt_name_bytes if b != 0x50)
                if alt_name.strip():
                    logger.info(f"Found name at {hex(start)}: {alt_name}")
            except Exception as e:
                logger.error(f"Failed to read name at {hex(start)}: {str(e)}")

def handle_rival_name_entry(pyboy):
    """Handle the rival name entry screen by selecting 'BLUE' from the dropdown."""
    logger.info("Handling rival name entry...")
    
    # Wait for rival name selection menu to appear
    time.sleep(2)
    save_frame(pyboy, "before_rival_name_entry")
    
    # Press down multiple times to ensure we're at the top of the list
    for i in range(3):
        pyboy.button_press("down")
        pyboy.tick(240)
        pyboy.button_release("down")
        time.sleep(0.5)
        save_frame(pyboy, f"scrolling_down_rival_{i+1}")
    
    # Press down to select 'BLUE' from the dropdown
    pyboy.button_press("down")
    pyboy.tick(240)
    pyboy.button_release("down")
    time.sleep(1)
    save_frame(pyboy, "selected_BLUE")
    
    # Press A to confirm selection
    pyboy.button_press("a")
    pyboy.tick(240)
    pyboy.button_release("a")
    time.sleep(1)
    save_frame(pyboy, "confirmed_BLUE")
    
    # Wait a bit for the name to be set
    time.sleep(2)
    
    # Verify rival name was set
    name_bytes = pyboy.memory[0xD34A:0xD355]
    name = "".join(chr(b) for b in name_bytes if b != 0x50)
    logger.info(f"Set rival name to: {name}")
    
    # Save final frame
    save_frame(pyboy, "after_rival_name_entry")
    
    # Additional verification
    if not name or name.strip() == "":
        logger.warning("Rival name appears to be empty, trying alternative memory locations")
        alt_locations = [
            (0xC4A4, 0xC4AF),
            (0xD34A, 0xD355),
            (0xD35E, 0xD369)
        ]
        for start, end in alt_locations:
            try:
                alt_name_bytes = pyboy.memory[start:end]
                alt_name = "".join(chr(b) for b in alt_name_bytes if b != 0x50)
                if alt_name.strip():
                    logger.info(f"Found rival name at {hex(start)}: {alt_name}")
            except Exception as e:
                logger.error(f"Failed to read rival name at {hex(start)}: {str(e)}")

def create_initial_state(rom_path: str, state_path: str):
    """Create an initial state file with the game at the starting gameplay state."""
    try:
        # Clean up any existing cache directory
        cache_dir = "cache/pokemon_red/state_creation"
        if os.path.exists(cache_dir):
            logger.info(f"Cleaning up existing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
        logger.info(f"Creating cache directory: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Clean up frames directory
        if os.path.exists(frames_dir):
            logger.info(f"Cleaning up existing frames directory: {frames_dir}")
            shutil.rmtree(frames_dir)
        logger.info(f"Creating frames directory: {frames_dir}")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Initialize PyBoy
        logger.info(f"Initializing PyBoy with ROM: {rom_path}")
        pyboy = PyBoy(rom_path, window="SDL2", cgb=True)
        logger.info("PyBoy initialized successfully")
        save_frame(pyboy, "initial_startup")
        
        # Set frame skip to maximum for quick advancement
        pyboy.set_emulation_speed(0)  # 0 means unlimited speed
        logger.info("Set emulation speed to maximum")
        
        # Quickly advance through initial animation
        logger.info("Quickly advancing through initial animation...")
        for _ in range(1800):  # Advance 30 seconds worth of frames quickly
            pyboy.tick()
            if _ % 180 == 0:  # Log progress every 3 seconds
                logger.info(f"Advanced {_//60} seconds of frames")
        save_frame(pyboy, "after_initial_animation")
        
        # Reset emulation speed to normal for button presses
        pyboy.set_emulation_speed(1)  # 1 means normal speed
        logger.info("Reset emulation speed to normal")
        
        # Press Start to get past title screen
        logger.info("Pressing Start to get past title screen...")
        press_button(pyboy, "start", frames=720, wait_time=3.0, description="press_start")
        save_frame(pyboy, "after_start_press")
        
        # Wait for menu to appear and stabilize
        logger.info("Waiting for menu to appear...")
        time.sleep(3)
        save_frame(pyboy, "before_menu_select")
        
        # Press A to select NEW GAME
        logger.info("Selecting NEW GAME...")
        press_button(pyboy, "a", frames=720, wait_time=3.0, description="select_new_game")
        save_frame(pyboy, "after_new_game_select")
        
        # Wait for game to start
        logger.info("Waiting for game to start...")
        time.sleep(3)
        save_frame(pyboy, "before_dialogue")
        
        # Alternate between A and Start to advance dialogue
        logger.info("Alternating between A and Start to advance dialogue...")
        for i in range(120):  # Keep pressing buttons
            if i % 10 == 0:  # Only log every 10th press
                logger.info(f"Press {i+1}/120")
                save_frame(pyboy, f"press_{i+1}")
            
            # Alternate between A and Start
            if i % 2 == 0:
                press_button(pyboy, "a", frames=720, wait_time=3.0,
                           capture_frame=(i % 10 == 0), description=f"press_a_{i+1}")
            else:
                press_button(pyboy, "start", frames=720, wait_time=3.0,
                           capture_frame=(i % 10 == 0), description=f"press_start_{i+1}")
        
        # Handle player name entry
        # handle_name_entry(pyboy)
        
        # Handle rival name entry
        # handle_rival_name_entry(pyboy)
        
          
        # Wait a bit after all presses
        logger.info("Waiting for game to stabilize...")
        time.sleep(10)
        save_frame(pyboy, "final_state")
           
        # Save state
        logger.info(f"Saving state to {state_path}")
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "wb") as f:
            pyboy.save_state(f)
        logger.info(f"State saved successfully")
        
        # Close PyBoy
        logger.info("Closing PyBoy")
        pyboy.stop()
        
    except Exception as e:
        logger.error(f"Error creating initial state: {str(e)}")
        logger.exception("Full traceback:")
        raise

if __name__ == "__main__":
    rom_path = "gamingagent/configs/custom_05_pokemon_red/rom/pokemon.gb"
    state_path = "gamingagent/envs/custom_05_pokemon_red/states/initial.state"
    create_initial_state(rom_path, state_path) 