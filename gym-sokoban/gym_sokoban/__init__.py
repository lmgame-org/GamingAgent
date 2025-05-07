import logging
import pkg_resources
import json
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

resource_package = __name__
env_json = pkg_resources.resource_filename(resource_package, '/'.join(('envs', 'available_envs.json')))

with open(env_json) as f:
    envs = json.load(f)
    for env in envs:
        register(
            id=env["id"],
            entry_point=env["entry_point"],
            max_episode_steps=env.get("max_episode_steps", None)
        )
