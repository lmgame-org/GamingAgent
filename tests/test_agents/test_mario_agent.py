import os
import pytest
from gamingagent.providers import APIProviderManager
from gamingagent.envs import RetroEnv
from gamingagent.agents import MarioAgent
from gamingagent.utils.logger import Logger
import time

def test_mario_agent():
    """Test the Mario agent with API provider manager."""
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    
    try:
        # Create environment
        env = RetroEnv(
            game_name='SuperMarioBros-Nes',
            render_mode="human",
            frame_skip=2,  # Skip 2 frames between actions
            fps=30        # Target 30 FPS
        )
        
        # Initialize API provider manager
        api_manager = APIProviderManager()
        api_manager.initialize_providers(
            anthropic_model="claude-3-opus-20240229",
            anthropic_max_tokens=1024,
            anthropic_temperature=0
        )
        
        # Create logger
        logger = Logger(log_dir="./logs")
        
        # Create agent with API manager
        agent = MarioAgent(
            env=env,
            provider=api_manager.anthropic,
            logger=logger,
            cache_dir="./cache",
            use_test_pattern=False,
            action_hold_times={
                'run': 0.5,
                'direction': 0.4,
                'jump': 0.3,
                'big_jump': 0.4
            },
            api_call_interval=60,  # Call API every 60 frames
            save_interval=30,      # Save observations every 30 frames
            frame_skip=2,          # Skip 2 frames between actions
            target_fps=30          # Target 30 FPS
        )
        
        # Run a test episode
        observation = env.reset()
        total_reward = 0
        done = False
        last_render_time = 0
        render_interval = 1.0 / 30  # 30 FPS
        
        for step in range(1000):  # Run for 1000 steps max
            # Control frame timing
            current_time = time.time()
            time_since_last_frame = current_time - last_render_time
            if time_since_last_frame < render_interval:
                time.sleep(render_interval - time_since_last_frame)
            
            # Get action from agent
            action = agent.step(observation)
            
            # Take step in environment
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Update render time
            last_render_time = time.time()
            
            # Optional: Save observation
            if step % 30 == 0:  # Save every 30 frames
                agent.save_observation(
                    observation=observation,
                    step=step,
                    reward=reward,
                    info=info,
                    action=action,
                    env=env,
                    episode_start_time=None
                )
            
            if terminated or truncated:
                break
                
        # Cleanup
        agent.close()
        env.close()
        
        # Basic assertions
        assert total_reward >= 0, "Agent should achieve non-negative reward"
        assert agent.total_actions > 0, "Agent should take actions"
        
    except Exception as e:
        pytest.fail(f"Test failed with error: {e}")

if __name__ == "__main__":
    # For manual testing/running
    test_mario_agent()