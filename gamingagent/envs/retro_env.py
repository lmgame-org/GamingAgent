import os
import retro
from typing import Any, Dict, Optional, Tuple
import numpy as np

class ClassicVideoGameEnv(retro.RetroEnv):
    """A wrapper around retro.RetroEnv that adds custom integration path support."""
    
    # ROM file extensions for different systems
    ROM_EXTENSIONS = {
        'Genesis': '.md',
        'SNES': '.sfc',
        'NES': '.nes',
        'Atari2600': '.a26',
        'GameBoy': '.gb',
        'GameBoyAdvance': '.gba',
        'GameBoyColor': '.gbc',
        'GameGear': '.gg',
        'TurboGrafx16': '.pce',
        'MasterSystem': '.sms'
    }
    
    def __init__(
        self,
        game,
        state=retro.State.DEFAULT,
        scenario=None,
        info=None,
        use_restricted_actions=retro.Actions.FILTERED,
        record=False,
        players=1,
        inttype=retro.data.Integrations.STABLE,
        obs_type=retro.Observations.IMAGE,
        render_mode="human",
    ):
        # Add custom integration path
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
        CUSTOM_INTEGRATIONS_PATH = os.path.join(PROJECT_ROOT, "custom_integrations", "roms")
        
        # Clear any existing custom paths and add our custom path
        retro.data.Integrations.clear_custom_paths()
        retro.data.add_custom_integration(CUSTOM_INTEGRATIONS_PATH)
        
        # Initialize parent class
        super().__init__(
            game=game,
            state=state,
            scenario=scenario,
            info=info,
            use_restricted_actions=use_restricted_actions,
            record=record,
            players=players,
            inttype=inttype | retro.data.Integrations.CUSTOM_ONLY,  # Include custom integrations
            obs_type=obs_type,
            render_mode=render_mode
        )