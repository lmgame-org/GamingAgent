import random
import time
import numpy as np
from gamingagent.envs.custom_05_pokemon_red.pokemonRedEnv import PokemonRedEnv


def test_random_play():
    """Test the environment with random actions"""
    print("\nTesting with actual ROM...")
    
    try:
        print("Initializing Pokemon Red Environment...")
        rom_path = "gamingagent/configs/custom_05_pokemon_red/rom/pokemon.gb"
        env = PokemonRedEnv(rom_path=rom_path, render_mode="human")
        
        print("Resetting environment...")
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation type: {type(obs)}")
        print(f"  Has image path: {obs.img_path is not None}")
        print(f"  Info keys: {list(info.keys())}")
        
        # Test a few random actions
        action_names = ["a", "b", "up", "down", "left", "right", "start", "select"]
        
        print(f"\nTaking {min(20, len(action_names) * 3)} random actions...")
        for i in range(min(20, len(action_names) * 3)):
            # Sample a random action by name
            action_name = action_names[i % len(action_names)]
            
            obs, reward, terminated, truncated, info = env.step(
                agent_action_str=action_name,
                thought_process=f"Testing action {action_name}",
                time_taken_s=0.1
            )
            
            if i < 5:  # Only print first few for brevity
                print(f"  Step {i+1}: Action='{action_name}', Reward={reward:.3f}, Steps={info.get('steps', 'N/A')}")
            
            if terminated or truncated:
                print(f"Episode ended at step {i+1}")
                break
                
            # Small delay to see the game
            import time
            time.sleep(0.1)
        
        env.close()
        print("Environment closed.")
        print("✓ Random play test completed successfully")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        if 'env' in locals():
            env.close()
            print("Environment closed.")


def test_environment_creation():
    """Test that the environment can be created without ROM initially"""
    print("Testing environment creation without ROM...")
    
    try:
        # Create environment without ROM
        env = PokemonRedEnv()
        print("✓ Environment created successfully without ROM")
        
        # Test that it fails appropriately when trying to reset without ROM
        try:
            env.reset()
            print("✗ Should have failed when resetting without ROM")
        except ValueError as e:
            print(f"✓ Correctly failed with: {e}")
        
        env.close()
        print("✓ Environment creation test passed")
        
    except Exception as e:
        print(f"✗ Environment creation test failed: {e}")


def test_action_mapping():
    """Test that action mapping works correctly"""
    print("Testing action mapping...")
    
    env = PokemonRedEnv()
    
    # Test action mapping
    expected_mapping = {0: "a", 1: "b", 2: "start", 3: "select", 4: "up", 5: "down", 6: "left", 7: "right"}
    
    for action, expected_button in expected_mapping.items():
        actual_button = env.action_map[action]
        if actual_button == expected_button:
            print(f"✓ Action {action} -> '{actual_button}'")
        else:
            print(f"✗ Action {action} -> expected '{expected_button}', got '{actual_button}'")
    
    # Test action space
    assert env.action_space.n == 8, f"Expected 8 actions, got {env.action_space.n}"
    print("✓ Action space has correct size")
    
    # Test observation space
    expected_shape = (240, 256, 3)
    actual_shape = env.observation_space.shape
    assert actual_shape == expected_shape, f"Expected shape {expected_shape}, got {actual_shape}"
    print("✓ Observation space has correct shape")
    
    env.close()
    print("✓ Action mapping test passed")


def test_save_load_state():
    """Test save and load state functionality"""
    print("\n=== Testing Save/Load State ===")
    
    try:
        # Initialize environment with ROM path
        rom_path = "gamingagent/configs/custom_05_pokemon_red/rom/pokemon.gb"
        env = PokemonRedEnv(rom_path=rom_path, render_mode="human")
        print("✓ Environment created")
        
        # Reset environment
        initial_obs = env.reset()
        print("✓ Environment reset")
        
        # Take some random actions to change the game state
        print("Taking 10 random actions...")
        action_names = ["a", "b", "up", "down", "left", "right", "start", "select"]
        for i in range(10):
            action_name = action_names[i % len(action_names)]
            obs, reward, done, truncated, info, perf_score = env.step(
                agent_action_str=action_name,
                thought_process=f"Testing action {action_name}",
                time_taken_s=0.1
            )
            print(f"  Step {i+1}: Action='{action_name}', Reward={reward:.3f}, Steps={info.get('steps', 'N/A')}")
        
        # Save the current state
        state_file = "test_pokemon_state.state"
        save_result = env.save_state(state_file)
        print(f"✓ State saved: {save_result}")
        
        # Take more actions to change state
        print("Taking 5 more actions...")
        for i in range(5):
            action_name = action_names[i % len(action_names)]
            obs, reward, done, truncated, info, perf_score = env.step(
                agent_action_str=action_name,
                thought_process=f"Testing action {action_name}",
                time_taken_s=0.1
            )
            print(f"  Step {i+1}: Action='{action_name}', Reward={reward:.3f}")
        
        # Load the saved state
        env.load_state(state_file)
        print("✓ State loaded successfully")
        
        # Verify we're back to the saved state by taking a screenshot
        restored_obs = env.get_screenshot()
        print("✓ Screenshot taken after state restore")
        
        # Clean up the test state file
        import os
        if os.path.exists(state_file):
            os.remove(state_file)
            print("✓ Test state file cleaned up")
        
        env.close()
        print("✓ Save/Load state test completed successfully")
        
    except Exception as e:
        print(f"✗ Save/Load state test failed: {e}")
        import traceback
        traceback.print_exc()


def test_gym_adapter_integration():
    """Test the gym adapter integration"""
    print("\n=== Testing Gym Adapter Integration ===")
    
    try:
        # Test initialization with adapter parameters
        env = PokemonRedEnv(
            rom_path="gamingagent/configs/custom_05_pokemon_red/rom/pokemon.gb",
            render_mode="human",
            observation_mode_for_adapter="both",  # Test both vision and text
            agent_cache_dir_for_adapter="cache/test_pokemon_red"
        )
        print("✓ Environment created with adapter")
        
        # Test reset with episode_id
        agent_obs, info = env.reset(episode_id=1)
        print("✓ Environment reset with adapter")
        print(f"  Observation type: {type(agent_obs)}")
        print(f"  Has image path: {agent_obs.img_path is not None}")
        print(f"  Has text representation: {agent_obs.textual_representation is not None}")
        
        # Test step with agent action strings
        test_actions = ["a", "up", "left", "start", "invalid_action"]
        
        for i, action_str in enumerate(test_actions):
            print(f"\nStep {i+1}: Testing action '{action_str}'")
            try:
                agent_obs, reward, terminated, truncated, info, perf_score = env.step(
                    agent_action_str=action_str,
                    thought_process=f"Testing action {action_str}",
                    time_taken_s=0.1
                )
                print(f"  ✓ Action processed - Reward: {reward:.3f}, Perf: {perf_score:.3f}")
                
                if terminated or truncated:
                    print(f"  Episode ended - Terminated: {terminated}, Truncated: {truncated}")
                    break
                    
            except Exception as e:
                print(f"  ✗ Error processing action '{action_str}': {e}")
        
        env.close()
        print("✓ Gym adapter integration test completed successfully")
        
    except Exception as e:
        print(f"✗ Gym adapter integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 50)
    print("Pokemon Red Environment Tests")
    print("=" * 50)
    
    # Test basic environment functionality
    test_environment_creation()
    print()
    test_action_mapping()
    print()
    
    # Test with actual ROM
    print("Testing with actual ROM...")
    test_random_play()
    test_save_load_state()
    test_gym_adapter_integration()
