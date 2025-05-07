# filename: tests/test_agents/test_sokoban_agent.py
import unittest
import os
import sys
import numpy as np

# Add project root to path to allow importing gamingagent
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from gamingagent.agents.sokoban_agent import SokobanAgent
from gamingagent.envs.sokoban_env import CustomSokobanEnv

# Dummy levels file content for testing
DUMMY_LEVELS_CONTENT = """
Level 1
 #
#@$#
 #?#
  #

Level 2
#####
#@$.#
#####

"""
DUMMY_LEVELS_PATH = "tests/test_agents/dummy_sokoban_levels.txt"

class TestSokobanAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a dummy levels file for testing loading
        os.makedirs(os.path.dirname(DUMMY_LEVELS_PATH), exist_ok=True)
        with open(DUMMY_LEVELS_PATH, "w") as f:
            f.write(DUMMY_LEVELS_CONTENT)

    @classmethod
    def tearDownClass(cls):
        # Clean up the dummy file
        if os.path.exists(DUMMY_LEVELS_PATH):
            os.remove(DUMMY_LEVELS_PATH)

    def setUp(self):
        # Common setup for tests, create agent with rgb_array mode (no display)
        # Use the dummy level file
        self.agent = SokobanAgent(level_file=DUMMY_LEVELS_PATH, render_mode='rgb_array')

    def tearDown(self):
        # Ensure environment is closed after each test
        self.agent.close_env()

    def test_agent_initialization(self):
        self.assertIsNotNone(self.agent.env)
        self.assertIsInstance(self.agent.env, CustomSokobanEnv)
        self.assertIsNotNone(self.agent.action_space)

    def test_reset_specific_level(self):
        level_index = 1
        obs, info = self.agent.env.reset(options={'level_index': level_index})
        self.assertIsNotNone(obs)
        self.assertIsInstance(obs, np.ndarray)
        # Check shape based on expected dimensions of dummy level 1 (4x4 grid -> 64x64x3 pixels)
        # Env dim_room should be set correctly after parsing
        self.assertEqual(self.agent.env.dim_room, (4, 4))
        self.assertEqual(obs.shape, (64, 64, 3))
        self.assertEqual(self.agent.env.num_boxes, 1) # $
        self.assertEqual(self.agent.env.boxes_on_target, 0)

    def test_reset_random_level(self):
        # This relies on the underlying gym-sokoban random generation
        # We mostly just check that it runs without error and returns correctly
        try:
            obs, info = self.agent.env.reset(options={'level_index': None}) # Explicitly None
            self.assertIsNotNone(obs)
            self.assertIsInstance(obs, np.ndarray)
            # Random levels have default dims (e.g., 10x10 -> 160x160x3) unless specified otherwise
            # Check against the default or the dims passed during init if any
            expected_dims = self.agent.env._init_kwargs.get('dim_room', (10, 10))
            expected_shape = (expected_dims[0] * 16, expected_dims[1] * 16, 3)
            self.assertEqual(self.agent.env.dim_room, expected_dims)
            self.assertEqual(obs.shape, expected_shape)

        except ImportError:
             self.skipTest("Skipping random level test as gym-sokoban is not installed.")
        except Exception as e:
             # Catch potential generation errors from the underlying lib if they occur
             self.fail(f"Random reset failed with error: {e}")


    def test_select_action(self):
        # Reset to a known state first
        obs, _ = self.agent.env.reset(options={'level_index': 1})
        action = self.agent.select_action(obs)
        self.assertIn(action, range(self.agent.action_space.n)) # Action should be valid

    def test_run_episode_specific_level(self):
        # Run a short episode on a known level
        total_reward, steps = self.agent.run_episode(level_index=1, max_steps=10)
        self.assertIsInstance(total_reward, (float, int))
        self.assertGreaterEqual(steps, 0)
        self.assertLessEqual(steps, 10)

    def test_run_episode_random_level(self):
        # Run a short episode on a random level
        try:
             total_reward, steps = self.agent.run_episode(level_index=None, max_steps=10)
             self.assertIsInstance(total_reward, (float, int))
             self.assertGreaterEqual(steps, 0)
             self.assertLessEqual(steps, 10)
        except ImportError:
             self.skipTest("Skipping random episode test as gym-sokoban is not installed.")
        except Exception as e:
            self.fail(f"Random episode run failed with error: {e}")


    def test_level_load_fail_fallback(self):
         # Test fallback by requesting a non-existent level
         invalid_level_index = 999
         try:
             # Expect reset to fall back to random generation
             obs, info = self.agent.env.reset(options={'level_index': invalid_level_index})
             self.assertIsNotNone(obs)
             # Check if it looks like a random level (default dims)
             expected_dims = self.agent.env._init_kwargs.get('dim_room', (10, 10))
             expected_shape = (expected_dims[0] * 16, expected_dims[1] * 16, 3)
             self.assertEqual(self.agent.env.dim_room, expected_dims)

         except ImportError:
             self.skipTest("Skipping fallback test as gym-sokoban is not installed.")
         except Exception as e:
            self.fail(f"Fallback reset failed with error: {e}")

    
if __name__ == '__main__':
    unittest.main()