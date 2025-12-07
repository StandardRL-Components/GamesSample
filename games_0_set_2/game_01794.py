
# Generated: 2025-08-27T18:19:41.557819
# Source Brief: brief_01794.md
# Brief Index: 1794

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to select a column. Press space to place a note on the strike line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A musical rhythm game. Match the falling colored notes by placing them in the correct column as they cross the strike line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 8
    GRID_ROWS = 10
    NUM_PITCHES = 5
    MAX_MISSES = 3
    MAX_STEPS = 18000 # 10 minutes at 30fps
    SONG_LENGTH = 200

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_STRIKE_LINE = (100, 100, 120)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_MISS = (255, 50, 50)
    NOTE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_miss = pygame.font.SysFont("Consolas", 36, bold=True)
        
        # Grid dimensions
        self.grid_width = self.SCREEN_WIDTH * 0.8
        self.grid_height = self.SCREEN_HEIGHT * 0.8
        self.grid_x_offset = (self.SCREEN_WIDTH - self.grid_width) / 2
        self.grid_y_offset = self.SCREEN_HEIGHT * 0.15
        self.cell_width = self.grid_width / self.GRID_COLS
        self.cell_height = self.grid_height / self.GRID_ROWS
        self.strike_line_y = self.grid_y_offset + self.grid_height - self.cell_height
        self.match_tolerance = self.cell_height * 0.75

        # Initialize state variables
        self.game_state = {}
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game_state = {
            "steps": 0,
            "score": 0,
            "misses": 0,
            "combo": 0,
            "successful_placements": 0,
            "note_fall_speed": 2.0,
            "player_col": self.GRID_COLS // 2,
            "space_was_held": False,
            "grid": [[None] * self.GRID_ROWS for _ in range(self.GRID_COLS)], # None or {'pitch', 'state', 'timer'}
            "falling_notes": [], # list of {'pos': [x,y], 'pitch', 'target_col'}
            "particles": [],
            "song": self._generate_song(),
            "song_progress": 0,
            "game_over": False,
        }
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_song(self):
        song = []
        current_step = 60 # Start with a delay
        for _ in range(self.SONG_LENGTH):
            current_step += self.np_random.integers(30, 90)
            col = self.np_random.integers(0, self.GRID_COLS)
            pitch = self.np_random.integers(0, self.NUM_PITCHES)
            song.append({"spawn_step": current_step, "col": col, "pitch": pitch})
        return song
    
    def step(self, action):
        reward = 0
        terminated = False

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # --- 1. Handle Input ---
        # Movement: 3=left, 4=right
        if movement == 3:
            self.game_state["player_col"] = max(0, self.game_state["player_col"] - 1)
        elif movement == 4:
            self.game_state["player_col"] = min(self.GRID_COLS - 1, self.game_state["player_col"] + 1)

        # Place Note (on rising edge of space press)
        if space_held and not self.game_state["space_was_held"]:
            place_reward = self._place_note()
            reward += place_reward
        self.game_state["space_was_held"] = space_held

        # --- 2. Update Game Logic ---
        self.game_state["steps"] += 1
        
        # Spawn new notes from the song
        if self.game_state["song_progress"] < len(self.game_state["song"]):
            next_note_info = self.game_state["song"][self.game_state["song_progress"]]
            if self.game_state["steps"] >= next_note_info["spawn_step"]:
                col = next_note_info["col"]
                x_pos = self.grid_x_offset + col * self.cell_width + self.cell_width / 2
                new_note = {
                    "pos": [x_pos, self.grid_y_offset],
                    "pitch": next_note_info["pitch"],
                    "target_col": col,
                }
                self.game_state["falling_notes"].append(new_note)
                self.game_state["song_progress"] += 1

        # Update falling notes position and check for misses
        notes_to_remove = []
        for note in self.game_state["falling_notes"]:
            note["pos"][1] += self.game_state["note_fall_speed"]
            if note["pos"][1] > self.SCREEN_HEIGHT:
                notes_to_remove.append(note)
                self.game_state["misses"] += 1
                self.game_state["combo"] = 0
                # sfx: miss_sound
        self.game_state["falling_notes"] = [n for n in self.game_state["falling_notes"] if n not in notes_to_remove]
        
        # Update placed notes (for clearing animations)
        for col in range(self.GRID_COLS):
            for row in range(self.GRID_ROWS):
                cell = self.game_state["grid"][col][row]
                if cell and cell["state"] == "clearing":
                    cell["timer"] -= 1
                    if cell["timer"] <= 0:
                        self.game_state["grid"][col][row] = None

        # Update particles
        self._update_particles()

        # --- 3. Check Termination ---
        if self.game_state["misses"] >= self.MAX_MISSES:
            terminated = True
            reward += -50 # Failure penalty
        
        if self.game_state["song_progress"] >= len(self.game_state["song"]) and not self.game_state["falling_notes"]:
            terminated = True
            reward += 50 # Victory bonus
            
        if self.game_state["steps"] >= self.MAX_STEPS:
            terminated = True

        self.game_state["game_over"] = terminated
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _place_note(self):
        col = self.game_state["player_col"]
        
        # Find lowest empty cell in the column
        target_row = -1
        for r in range(self.GRID_ROWS - 1, -1, -1):
            if self.game_state["grid"][col][r] is None:
                target_row = r
                break
        
        if target_row == -1:
            return -0.5 # Column is full penalty

        # Find closest falling note in this column
        best_note = None
        min_dist = float('inf')
        for note in self.game_state["falling_notes"]:
            if note["target_col"] == col:
                dist = abs(note["pos"][1] - self.strike_line_y)
                if dist < min_dist:
                    min_dist = dist
                    best_note = note
        
        # Case 1: Successful match
        if best_note and min_dist <= self.match_tolerance:
            # sfx: match_success
            self.game_state["grid"][col][target_row] = {
                "pitch": best_note["pitch"],
                "state": "clearing",
                "timer": 15 # frames
            }
            self._create_particles(best_note["pos"], self.NOTE_COLORS[best_note["pitch"]])
            self.game_state["falling_notes"].remove(best_note)
            
            reward_val = 5.0 + self.game_state["combo"]
            self.game_state["score"] += int(reward_val)
            self.game_state["combo"] += 1
            
            self.game_state["successful_placements"] += 1
            if self.game_state["successful_placements"] > 0 and self.game_state["successful_placements"] % 200 == 0:
                self.game_state["note_fall_speed"] += 0.05
            
            return reward_val

        # Case 2: No note to match, but placement happens
        else:
            # sfx: placement_fail
            self.game_state["combo"] = 0
            is_correct_col = any(note["target_col"] == col for note in self.game_state["falling_notes"])
            if is_correct_col:
                return 1.0 # Reward for anticipating correctly
            else:
                return -0.1 # Penalty for wrong column

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_placed_notes()
        self._render_falling_notes()
        self._render_cursor()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        # Vertical lines
        for i in range(self.GRID_COLS + 1):
            x = self.grid_x_offset + i * self.cell_width
            pygame.draw.line(self.screen, self.COLOR_GRID, (int(x), int(self.grid_y_offset)), (int(x), int(self.grid_y_offset + self.grid_height)), 1)
        # Horizontal lines
        for i in range(self.GRID_ROWS + 1):
            y = self.grid_y_offset + i * self.cell_height
            pygame.draw.line(self.screen, self.COLOR_GRID, (int(self.grid_x_offset), int(y)), (int(self.grid_x_offset + self.grid_width), int(y)), 1)
        # Strike line
        pygame.draw.line(self.screen, self.COLOR_STRIKE_LINE, (int(self.grid_x_offset), int(self.strike_line_y)), (int(self.grid_x_offset + self.grid_width), int(self.strike_line_y)), 3)

    def _render_placed_notes(self):
        for col in range(self.GRID_COLS):
            for row in range(self.GRID_ROWS):
                cell = self.game_state["grid"][col][row]
                if cell:
                    color = self.NOTE_COLORS[cell["pitch"]]
                    x = self.grid_x_offset + col * self.cell_width
                    y = self.grid_y_offset + row * self.cell_height
                    rect = pygame.Rect(x, y, self.cell_width, self.cell_height)
                    
                    if cell["state"] == "clearing":
                        shrink_factor = cell["timer"] / 15.0
                        rect.inflate_ip(-self.cell_width * (1-shrink_factor), -self.cell_height * (1-shrink_factor))
                        pygame.draw.rect(self.screen, color, rect, border_radius=4)
                    else:
                        pygame.draw.rect(self.screen, color, rect, border_radius=4)

    def _render_falling_notes(self):
        note_size = self.cell_width * 0.8
        for note in self.game_state["falling_notes"]:
            color = self.NOTE_COLORS[note["pitch"]]
            x, y = note["pos"]
            rect = pygame.Rect(x - note_size / 2, y - note_size / 2, note_size, note_size)
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            pygame.draw.rect(self.screen, tuple(min(255, c + 60) for c in color), rect, width=2, border_radius=6)

    def _render_cursor(self):
        col = self.game_state["player_col"]
        x = self.grid_x_offset + col * self.cell_width
        y = self.strike_line_y - self.cell_height
        rect = pygame.Rect(x, y, self.cell_width, self.cell_height)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, width=3, border_radius=5)
        
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.game_state['score']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Combo
        if self.game_state["combo"] > 2:
            combo_text = self.font_ui.render(f"COMBO x{self.game_state['combo']}", True, self.COLOR_UI_TEXT)
            text_rect = combo_text.get_rect(center=(self.SCREEN_WIDTH / 2, 30))
            self.screen.blit(combo_text, text_rect)

        # Misses
        for i in range(self.game_state["misses"]):
            miss_text = self.font_miss.render("X", True, self.COLOR_MISS)
            self.screen.blit(miss_text, (self.SCREEN_WIDTH - 30 - i * 25, 5))

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                "pos": list(pos),
                "vel": vel,
                "life": random.randint(15, 30),
                "radius": random.uniform(2, 5),
                "color": color
            }
            self.game_state["particles"].append(particle)

    def _update_particles(self):
        particles_to_remove = []
        for p in self.game_state["particles"]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
            if p["life"] <= 0:
                particles_to_remove.append(p)
        self.game_state["particles"] = [p for p in self.game_state["particles"] if p not in particles_to_remove]
        
    def _render_particles(self):
        for p in self.game_state["particles"]:
            life_ratio = p["life"] / 30.0
            radius = p["radius"] * life_ratio
            if radius > 1:
                pos = (int(p["pos"][0]), int(p["pos"][1]))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius), p["color"])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius), p["color"])

    def _get_info(self):
        return {
            "score": self.game_state["score"],
            "steps": self.game_state["steps"],
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:", info)
    
    # Test a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Info={info}, Terminated={terminated}")
        if terminated:
            print("Episode finished.")
            break
    
    env.close()

    # To visualize the game, you would need a "human" render mode and a display.
    # The following is a conceptual example for interactive play.
    # It requires adding a "human" render_mode and modifying the main loop
    # to handle keyboard input and render to the screen.

    # print("\n--- Visual Example (Conceptual) ---")
    # try:
    #     del os.environ["SDL_VIDEODRIVER"]
    #     # A 'human' render mode would be needed in the class for this to work
    #     # env = GameEnv(render_mode="human") 
    #     # obs, info = env.reset()
    #     # running = True
    #     # for _ in range(1000):
    #     #     # In a real loop, you'd get keyboard state here
    #     #     action = env.action_space.sample() # Replace with keyboard mapping
    #     #     obs, reward, terminated, truncated, info = env.step(action)
    #     #     if terminated:
    #     #         print("Game Over! Final Score:", info['score'])
    #     #         env.reset()
    #     #     # The 'human' mode's step/render function would blit to screen
    #     #     # and call pygame.display.flip()
    #     # env.close()
    # except Exception as e:
    #     print(f"Could not run visual example: {e}")