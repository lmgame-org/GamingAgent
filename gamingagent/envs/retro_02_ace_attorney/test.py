import retro
import os
import pygame # For keyboard input
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# GBA Button Mapping (keyboard key -> GBA button index)
# Actual GBA Buttons from env.buttons: ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', None, 'L', 'R']
# Indices:                                0     1       2        3       4      5       6        7     8    9     10   11
KEY_TO_GBA_BUTTON = {
    # User desired direct mapping
    pygame.K_a: 8,         # Keyboard 'A' -> GBA 'A' (index 8)
    pygame.K_b: 0,         # Keyboard 'B' -> GBA 'B' (index 0)
    pygame.K_l: 10,        # Keyboard 'L' -> GBA 'L' (index 10)
    pygame.K_r: 11,        # Keyboard 'R' -> GBA 'R' (index 11)

    # D-Pad
    pygame.K_UP: 4,        # Keyboard 'Up Arrow' -> GBA 'UP' (index 4)
    pygame.K_DOWN: 5,      # Keyboard 'Down Arrow' -> GBA 'DOWN' (index 5)
    pygame.K_LEFT: 6,      # Keyboard 'Left Arrow' -> GBA 'LEFT' (index 6)
    pygame.K_RIGHT: 7,     # Keyboard 'Right Arrow' -> GBA 'RIGHT' (index 7)

    # Start/Select
    pygame.K_RETURN: 3,    # Keyboard 'Enter' -> GBA 'START' (index 3)
    pygame.K_BACKSPACE: 2, # Keyboard 'Backspace' -> GBA 'SELECT' (index 2)
}
NUM_GBA_BUTTONS = 12 # Total number of buttons in the GBA array for this integration

def test_recording():
    """Test function to demonstrate recording functionality."""
    print("\nTesting recording functionality...")
    retro.data.Integrations.add_custom_path(SCRIPT_DIR)
    game_name = "AceAttorney-GbAdvance"  # Make sure this matches the directory name exactly
    
    # Create a directory for recordings
    record_dir = os.path.join(SCRIPT_DIR, "recordings")
    os.makedirs(record_dir, exist_ok=True)
    print(f"Recording will be saved to: {record_dir}")
    
    try:
        # Create environment with recording enabled
        env = retro.make(
            game=game_name,
            state="level1_1_5",
            record=record_dir,
            render_mode='human',
            use_restricted_actions=retro.Actions.FILTERED,
            inttype=retro.data.Integrations.ALL  # Add this to ensure all integrations are checked
        )
        
        print("Environment created successfully. Starting recording test...")
        env.reset()
        
        # Run for 1000 steps or until termination
        for step in range(1000):
            _, _, terminate, truncate, _ = env.step(env.action_space.sample())
            if terminate or truncate:
                print(f"Episode ended after {step + 1} steps")
                break
                
        print("Recording test completed.")
        env.close()
        
    except Exception as e:
        print(f"Error during recording test: {e}")
        print(f"Current directory contents: {os.listdir(SCRIPT_DIR)}")  # Add this to debug

def main():
    # Add command line argument parsing to choose between interactive and recording test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--record":
        test_recording()
        return

    retro.data.Integrations.add_custom_path(SCRIPT_DIR)
    game_name = "AceAttorney-GbAdvance"

    game_available = game_name in retro.data.list_games(inttype=retro.data.Integrations.ALL)
    print(f"Is '{game_name}' available? {game_available}")

    if not game_available:
        print(f"'{game_name}' not found. Please ensure:")
        print(f"1. The game folder is named '{game_name}'.")
        print(f"2. It is located at: {os.path.join(SCRIPT_DIR, game_name)}")
        print(f"3. It contains rom, data.json, scenario.json, etc.")
        return

    print(f"Attempting to make environment for '{game_name}'...")
    working_state_name = "level1_1_5" # Or your desired state
    print(f"Loading game with state: '{working_state_name}'")

    try:
        env = retro.make(game_name, state=working_state_name, inttype=retro.data.Integrations.ALL, render_mode='human', use_restricted_actions=retro.Actions.FILTERED)
    except Exception as e:
        print(f"Error creating Retro environment: {e}")
        return
        
    print(f"Successfully created environment: {env}")
    print(f"Environment's action space: {env.action_space}") # Print action space
    print(f"Environment's reported buttons (order of action array indices): {env.buttons}") 
    print(f"Environment's button combos (defined in integration): {env.button_combos}") # Print button combos

    obs, info = env.reset()
    env.render()

    pygame.init()
    screen = pygame.display.set_mode((200, 150), pygame.NOFRAME)
    pygame.display.set_caption("Keyboard Input Helper")

    print("\nControls (Keyboard -> GBA Button Index -> GBA Button Name based on env.buttons):")
    print(f"  Keyboard 'A' -> Index 8 (GBA 'A')")
    print(f"  Keyboard 'B' -> Index 0 (GBA 'B')")
    print(f"  Keyboard 'L' -> Index 10 (GBA 'L')")
    print(f"  Keyboard 'R' -> Index 11 (GBA 'R')")
    print(f"  Arrow Keys   -> Indices 4-7 (GBA D-Pad)")
    print(f"  Enter Key    -> Index 3 (GBA 'Start')")
    print(f"  Backspace    -> Index 2 (GBA 'Select')")
    print("Press ESC in Pygame window or Ctrl+C in terminal to quit.")

    running = True
    action_array = np.zeros(NUM_GBA_BUTTONS, dtype=bool)
    # Flag to indicate if a key was pressed in the current event processing cycle
    key_was_pressed_this_cycle = False 

    clock = pygame.time.Clock()

    while running:
        key_was_pressed_this_cycle = False # Reset for this iteration
        current_action_for_step = np.zeros(NUM_GBA_BUTTONS, dtype=bool) # Start with no-op for this step

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key in KEY_TO_GBA_BUTTON:
                    current_action_for_step[KEY_TO_GBA_BUTTON[event.key]] = True
                    key_was_pressed_this_cycle = True # A key was pressed
            # KEYUP events are now effectively ignored for constructing the action for the step,
            # as we send a momentary press.

        if not running:
            break
        
        # If a key was pressed, current_action_for_step contains that press.
        # If no key was pressed, current_action_for_step remains all zeros (no-op).
        obs, reward, terminated, truncated, info = env.step(current_action_for_step)
        env.render()

        if terminated or truncated:
            print("Episode finished. Resetting...")
            obs, info = env.reset()
        
        clock.tick(60)

    env.close()
    pygame.quit()
    print("Test script finished.")

if __name__ == "__main__":
    main()