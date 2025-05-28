class DoomBaseModule:
    """
    A simplified module that directly processes observation images and returns actions for Doom.
    This module skips separate perception and memory stages used in the full pipeline.
    """
    def __init__(self):
        self.last_action = None
        self.available_actions = ["move_up", "move_down", "move_left", "move_right", "attack"]
        self.action_index = 0

    def process_observation(self, observation, info=None):
        """
        Process the observation and return an action.
        For Doom, we'll cycle through available actions.
        """
        # Cycle through available actions
        action = self.available_actions[self.action_index]
        self.action_index = (self.action_index + 1) % len(self.available_actions)
        self.last_action = action
        return action 