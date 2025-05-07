import argparse
import retro
import numpy as np
import time
import sys
from gamingagent.envs.real_time_video_game_env import RealTimeVideoGameEnv

class Action:
    """Class to handle different actions in the game."""
    
    def __init__(self, env):
        """Initialize with environment to get button mapping."""
        self.env = env
        self.right_idx = env.buttons.index("RIGHT")
        self.a_idx = env.buttons.index("A")
        self.base_action = np.zeros(env.num_buttons, dtype=np.uint8)
        
    def get_action(self, action_type):
        """Get action array based on action type."""
        action = self.base_action.copy()
        
        if action_type == "run":
            action[self.right_idx] = 1
        elif action_type == "jump":
            action[self.right_idx] = 1
            action[self.a_idx] = 1
        elif action_type == "run_jump":
            action[self.right_idx] = 1
            action[self.a_idx] = 1
        elif action_type == "none":
            pass
            
        return action

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game",
        default="SuperMarioBros-Nes",
        help="the name or path for the game to run",
    )
    parser.add_argument(
        "--state",
        default=retro.State.DEFAULT,
        help="the initial state file to load, minus the extension",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        default="scenario",
        help="the scenario file to load, minus the extension",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="target frames per second for the game",
    )
    args = parser.parse_args()

    try:
        # Create real-time environment
        env = RealTimeVideoGameEnv(
            game=args.game,
            state=args.state,
            scenario=args.scenario,
            render_mode="human",
            target_fps=args.target_fps
        )

        # Reset environment
        observation = env.reset()
        print("Environment reset complete")
        
        # Initialize action handler
        action_handler = Action(env)
        
        # Print button mapping for debugging
        print("Button mapping:", env.buttons)
        print("Number of buttons:", env.num_buttons)
        print(f"Action indices - RIGHT: {action_handler.right_idx}, A: {action_handler.a_idx}")
        
        # For FPS monitoring
        last_fps_print = time.time()
        frame_count = 0
        action_counter = 0
        
        # Action sequence: run, jump, run, jump, etc.
        action_sequence = ["run", "run_jump", "run", "run_jump"]
        current_action_idx = 0
        
        # Execute action continuously
        while True:
            # Get current action
            current_action = action_sequence[current_action_idx]
            action = action_handler.get_action(current_action)
            
            # Update action sequence
            action_counter += 1
            if action_counter >= 30:  # Change action every second at 30 FPS
                action_counter = 0
                current_action_idx = (current_action_idx + 1) % len(action_sequence)
            
            # Take step
            step_result = env.step(action)
            frame_count += 1
            
            # Print FPS every second
            current_time = time.time()
            if current_time - last_fps_print >= 1.0:
                fps = frame_count / (current_time - last_fps_print)
                print(f"FPS: {fps:.1f}")
                if isinstance(step_result, tuple):
                    obs, reward, terminated, truncated, info = step_result
                    print(f"Step: {env.step_count}, Action: {current_action}")
                    print(f"Action state: {action}")
                    print(f"Action values - RIGHT: {action[action_handler.right_idx]}, A: {action[action_handler.a_idx]}")
                    if info:
                        print(f"Info: {info}")
                frame_count = 0
                last_fps_print = current_time
            
            # Check if episode is done
            if isinstance(step_result, tuple):
                _, _, terminated, truncated, _ = step_result
                if terminated or truncated:
                    print("Episode done, resetting...")
                    env.reset()
            
    except KeyboardInterrupt:
        print("\nStopping test...")
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        if 'env' in locals():
            env.close()
        print("Cleanup complete")

if __name__ == "__main__":
    main() 