import pyautogui
import json
import time

class Executor:
    def __init__(self, mode="pygame", game="sokoban"):
        self.mode = mode
        self.game = game.lower()
        self.key_map = self._load_key_map()

    def _load_key_map(self):
        if self.game == "sokoban":
            return {
                "up": "up",
                "down": "down",
                "left": "left",
                "right": "right",
                "restart": "r",
                "unmove": "d"
            }
        elif self.game == "mario":
            return {}  # LLM returns code directly, not move strings
        elif self.game == "2048":
            return {
                "up": "up",
                "down": "down",
                "left": "left",
                "right": "right"
            }
        elif self.game == "tetris":
            return {}  # Same as Mario, executes generated code
        elif self.game == "candy":
            return {}  # Uses drag based on ID positions
        else:
            return {}

    def execute(self, move):
        if self.game == "sokoban":
            self._exec_sokoban(move)
        elif self.game == "mario":
            self._exec_python_code(move)
        elif self.game == "tetris":
            self._exec_python_code(move)
        elif self.game == "candy":
            self._exec_candy_swap(move)
        elif self.game == "ace":
            self._exec_ace_attorney(move)
        else:
            print(f"[Executor] No executor defined for game '{self.game}'")


    def _exec_sokoban(self, move):
        move = move.strip().lower()
        if move not in self.key_map:
            print(f"[Executor] Invalid move: {move}")
            return
        pyautogui.press(self.key_map[move])
        print(f"[Executor] Performed move: {move}")

    def _exec_python_code(self, code_str):
        try:
            exec(code_str)
            print("[Executor] Executed Python code.")
        except Exception as e:
            print(f"[Executor] Error executing code: {e}")


    def _exec_ace_attorney(self, move):
        """
        Executes Ace Attorney moves directly using pyautogui.
        This supports both arrow keys and mapped special moves like 'z', 'x', 'r', 'b', etc.
        """
        move = move.strip().lower()
        valid_keys = ["up", "down", "left", "right", "z", "x", "r", "b", "l"]

        if move not in valid_keys:
            print(f"[Executor] WARNING: Invalid Ace Attorney move: {move}")
            return

        pyautogui.keyDown(move)
        time.sleep(0.1)
        pyautogui.keyUp(move)
        print(f"[Executor] Ace Attorney move performed: {move}")


    def _exec_candy_swap(self, move):
        """
        move: tuple like (id1, id2)
        Uses grid_annotation.json to find screen coordinates.
        """
        if not isinstance(move, tuple) or len(move) != 2:
            print("[Executor] Candy swap move must be a tuple (id1, id2)")
            return

        id1, id2 = move
        grid_path = "cache/candy_crush/grid_annotation.json"

        try:
            with open(grid_path, "r") as f:
                grid = json.load(f)
        except Exception as e:
            print(f"[Executor] Failed to read candy grid: {e}")
            return

        def find_coords(candy_id):
            entry = next((e for e in grid if e["id"] == candy_id), None)
            return (entry["x"], entry["y"]) if entry else None

        pos1 = find_coords(id1)
        pos2 = find_coords(id2)

        if not pos1 or not pos2:
            print(f"[Executor] IDs not found in grid: {id1}, {id2}")
            return

        x1, y1 = pos1
        x2, y2 = pos2

        pyautogui.moveTo(x1, y1, duration=0.2)
        pyautogui.mouseDown()
        pyautogui.moveTo(x2, y2, duration=0.2)
        pyautogui.mouseUp()
        print(f"[Executor] Swapped ({id1}) <-> ({id2}) on screen.")
