import abc
import argparse
import ctypes
import sys
import time

import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key as keycodes

from gamingagent.envs.custom_05_doom.Doom_env import DoomEnvWrapper

# creating interactive test for doom, still working on it

class Interactive(abc.ABC):
    """
    Base class for making gym environments interactive for human use
    """

    def __init__(self, env, sync=True, tps=60, aspect_ratio=None):
        obs = env.reset()
        self._image = self.get_image(obs, env)
        assert (
            len(self._image.shape) == 3 and self._image.shape[2] == 3
        ), "must be an RGB image"
        image_height, image_width = self._image.shape[:2]

        if aspect_ratio is None:
            aspect_ratio = image_width / image_height

        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        max_win_width = screen.width * 0.9
        max_win_height = screen.height * 0.9
        win_width = image_width
        win_height = int(win_width / aspect_ratio)

        while win_width > max_win_width or win_height > max_win_height:
            win_width //= 2
            win_height //= 2
        while win_width < max_win_width / 2 and win_height < max_win_height / 2:
            win_width *= 2
            win_height *= 2

        win = pyglet.window.Window(width=win_width, height=win_height)

        self._key_handler = pyglet.window.key.KeyStateHandler()
        win.push_handlers(self._key_handler)
        win.on_close = self._on_close

        gl.glEnable(gl.GL_TEXTURE_2D)
        self._texture_id = gl.GLuint(0)
        gl.glGenTextures(1, ctypes.byref(self._texture_id))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            image_width,
            image_height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )

        self._env = env
        self._win = win

        self._key_previous_states = {}

        self._steps = 0
        self._episode_steps = 0
        self._episode_returns = 0
        self._prev_episode_returns = 0

        self._tps = tps
        self._sync = sync
        self._current_time = 0
        self._sim_time = 0
        self._max_sim_frames_per_update = 4

    def _update(self, dt):
        max_dt = self._max_sim_frames_per_update / self._tps
        if dt > max_dt:
            dt = max_dt

        self._current_time += dt
        while self._sim_time < self._current_time:
            self._sim_time += 1 / self._tps

            keys_clicked = set()
            keys_pressed = set()
            for key_code, pressed in self._key_handler.items():
                if pressed:
                    keys_pressed.add(key_code)

                if not self._key_previous_states.get(key_code, False) and pressed:
                    keys_clicked.add(key_code)
                self._key_previous_states[key_code] = pressed

            if keycodes.ESCAPE in keys_pressed:
                self._on_close()

            inputs = keys_pressed
            if self._sync:
                inputs = keys_clicked

            keys = []
            for keycode in inputs:
                for name in dir(keycodes):
                    if getattr(keycodes, name) == keycode:
                        keys.append(name)

            act = self.keys_to_act(keys)

            if not self._sync or act is not None:
                obs, rew, terminated, truncated, _info = self._env.step(act)
                done = terminated or truncated
                self._image = self.get_image(obs, self._env)
                self._episode_returns += rew
                self._steps += 1
                self._episode_steps += 1
                np.set_printoptions(precision=2)
                if self._sync:
                    done_int = int(done)
                    mess = f"steps={self._steps} episode_steps={self._episode_steps} rew={rew} episode_returns={self._episode_returns} done={done_int}"
                    print(mess)
                elif self._steps % self._tps == 0 or done:
                    episode_returns_delta = (
                        self._episode_returns - self._prev_episode_returns
                    )
                    self._prev_episode_returns = self._episode_returns
                    mess = f"steps={self._steps} episode_steps={self._episode_steps} episode_returns_delta={episode_returns_delta} episode_returns={self._episode_returns}"
                    print(mess)

                if done:
                    self._env.reset()
                    self._episode_steps = 0
                    self._episode_returns = 0
                    self._prev_episode_returns = 0

    def _draw(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        video_buffer = ctypes.cast(
            self._image.tobytes(),
            ctypes.POINTER(ctypes.c_short),
        )
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self._image.shape[1],
            self._image.shape[0],
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            video_buffer,
        )

        x = 0
        y = 0
        w = self._win.width
        h = self._win.height

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ("v2f", [x, y, x + w, y, x + w, y + h, x, y + h]),
            ("t2f", [0, 1, 1, 1, 1, 0, 0, 0]),
        )

    def _on_close(self):
        self._env.close()
        sys.exit(0)

    @abc.abstractmethod
    def get_image(self, obs, venv):
        pass

    @abc.abstractmethod
    def keys_to_act(self, keys):
        pass

    def run(self):
        prev_frame_time = time.time()
        while True:
            self._win.switch_to()
            self._win.dispatch_events()
            now = time.time()
            self._update(now - prev_frame_time)
            prev_frame_time = now
            self._draw()
            self._win.flip()


class DoomInteractive(Interactive):
    """
    Interactive setup for Doom environment
    """

    def __init__(self, game_config_path, observation_mode, base_log_dir, render_mode_human):
        env = DoomEnvWrapper(
            game_config_path=game_config_path,
            observation_mode=observation_mode,
            base_log_dir=base_log_dir,
            render_mode_human=render_mode_human
        )
        self._actions = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "SHOOT"]
        super().__init__(env=env, sync=False, tps=60, aspect_ratio=4 / 3)

    def get_image(self, _obs, env):
        return env.render()

    def keys_to_act(self, keys):
        inputs = {
            "MOVE_FORWARD": "UP" in keys,
            "TURN_LEFT": "LEFT" in keys,
            "TURN_RIGHT": "RIGHT" in keys,
            "SHOOT": "SPACE" in keys,
        }
        return [inputs[action] for action in self._actions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_config_path", default="configs/custom_05_doom/config.yaml")
    parser.add_argument("--observation_mode", default="vision")
    parser.add_argument("--base_log_dir", default="cache/doom/test_run")
    parser.add_argument("--render_mode_human", action="store_true")
    args = parser.parse_args()

    interactive_env = DoomInteractive(
        game_config_path=args.game_config_path,
        observation_mode=args.observation_mode,
        base_log_dir=args.base_log_dir,
        render_mode_human=args.render_mode_human
    )
    interactive_env.run()


if __name__ == "__main__":
    main()