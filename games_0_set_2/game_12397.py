import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Synchrony Puzzle Environment

    **Goal:** Activate all 5 switches within 15 turns.
    **Characters:**
    - Player 1 (Blue): Moves 2 units per turn.
    - Player 2 (Red): Moves 1 unit per turn.
    **Actions:** The action space is MultiDiscrete([5, 2, 2]).
    - `action[0]`: Determines the direction for Player 1 (0:None, 1:Up, 2:Down, 3:Left, 4:Right).
    - `action[1]`: If 1 (Space pressed), Player 2 attempts to move in the SAME direction as Player 1.
    - `action[2]`: If 1 (Shift pressed), Player 2 attempts to move in the OPPOSITE direction to Player 1.
    - If both Space and Shift are pressed, or neither are, Player 2 does not move.
    **Reward:**
    - +1 for each new switch activated.
    - +100 for activating all switches (winning).
    - -100 for running out of turns (losing).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "A cooperative puzzle game where you control two characters to activate all switches on the board within a limited number of turns."
    user_guide = "Use arrow keys (↑↓←→) to move the blue character. Hold space to make the red character move in the same direction, or hold shift to make it move in the opposite direction."
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_DIM = 10
    MAX_STEPS = 15

    CHAR1_SPEED = 2
    CHAR2_SPEED = 1

    # Colors (Clean, Minimalist, High Contrast)
    COLOR_BG = (20, 20, 30) # Dark blue-grey
    COLOR_GRID = (40, 40, 60)
    COLOR_TEXT = (220, 220, 240)
    
    COLOR_P1 = (0, 150, 255) # Bright Blue
    COLOR_P1_GLOW = (0, 150, 255, 50)
    
    COLOR_P2 = (255, 50, 50) # Bright Red
    COLOR_P2_GLOW = (255, 50, 50, 50)

    COLOR_SWITCH_OFF = (100, 100, 120)
    COLOR_SWITCH_OFF_BORDER = (140, 140, 160)
    COLOR_SWITCH_ON = (50, 255, 150) # Bright Green
    COLOR_SWITCH_ON_GLOW = (50, 255, 150, 70)
    COLOR_SWITCH_ON_BORDER = (150, 255, 200)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 16)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 22)
        
        # --- Game Board Layout ---
        self.board_pixel_size = min(self.SCREEN_WIDTH, self.SCREEN_HEIGHT) * 0.85
        self.cell_pixel_size = self.board_pixel_size / self.GRID_DIM
        self.board_offset_x = (self.SCREEN_WIDTH - self.board_pixel_size) / 2
        self.board_offset_y = (self.SCREEN_HEIGHT - self.board_pixel_size) / 2

        # --- Game Entities ---
        self.switches_locations = [
            np.array([2, 2]), np.array([2, 7]), np.array([5, 5]),
            np.array([7, 2]), np.array([7, 7])
        ]
        
        self.DIRECTIONS = {
            0: np.array([0, 0]),   # None
            1: np.array([0, -1]),  # Up
            2: np.array([0, 1]),   # Down
            3: np.array([-1, 0]),  # Left
            4: np.array([1, 0]),   # Right
        }
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.char1_pos = None
        self.char2_pos = None
        self.switches_status = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        
        # Initial positions
        self.char1_pos = np.array([0, 0])
        self.char2_pos = np.array([9, 9])
        
        # Switch status
        self.switches_status = [False] * len(self.switches_locations)
        self._update_switches() # Check initial state
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0.0
        
        # --- 1. Unpack Action & Determine Movement ---
        move_dir_idx = action[0]
        char2_move_same = action[1] == 1
        char2_move_opposite = action[2] == 1
        
        move_vec = self.DIRECTIONS[move_dir_idx]
        
        prev_switches_on = sum(self.switches_status)
        
        # --- 2. Update Character 1 Position ---
        new_pos1 = self.char1_pos + move_vec * self.CHAR1_SPEED
        self.char1_pos = np.clip(new_pos1, 0, self.GRID_DIM - 1).astype(int)

        # --- 3. Update Character 2 Position ---
        char2_move_vec = np.array([0, 0])
        # Exclusive OR: move if one button is pressed, but not both or neither.
        if char2_move_same ^ char2_move_opposite:
            if char2_move_same:
                char2_move_vec = move_vec
            else: # char2_move_opposite
                char2_move_vec = -move_vec
        
        new_pos2 = self.char2_pos + char2_move_vec * self.CHAR2_SPEED
        self.char2_pos = np.clip(new_pos2, 0, self.GRID_DIM - 1).astype(int)

        # --- 4. Update Switches and Calculate Reward ---
        self._update_switches()
        new_switches_on = sum(self.switches_status)
        
        newly_activated = new_switches_on - prev_switches_on
        if newly_activated > 0:
            reward += newly_activated * 1.0
            
        # --- 5. Check Termination and Terminal Rewards ---
        win = all(self.switches_status)
        loss = self.steps >= self.MAX_STEPS
        terminated = win or loss
        truncated = False # This game is not truncated
        
        if win:
            reward += 100.0
        elif loss and not win: # only apply loss penalty if not won on the last step
            reward -= 100.0
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_switches(self):
        for i, switch_pos in enumerate(self.switches_locations):
            on_by_p1 = np.array_equal(self.char1_pos, switch_pos)
            on_by_p2 = np.array_equal(self.char2_pos, switch_pos)
            if on_by_p1 or on_by_p2:
                if not self.switches_status[i]:
                    # This is the first time it's activated
                    self.switches_status[i] = True

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "switches_on": sum(self.switches_status),
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

    # --- Rendering Methods ---
    
    def _render_frame(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_switches()
        self._render_characters()
        self._render_ui()

    def _render_grid(self):
        for i in range(self.GRID_DIM + 1):
            # Vertical lines
            start_pos = (self.board_offset_x + i * self.cell_pixel_size, self.board_offset_y)
            end_pos = (self.board_offset_x + i * self.cell_pixel_size, self.board_offset_y + self.board_pixel_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.board_offset_x, self.board_offset_y + i * self.cell_pixel_size)
            end_pos = (self.board_offset_x + self.board_pixel_size, self.board_offset_y + i * self.cell_pixel_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_switches(self):
        radius = int(self.cell_pixel_size * 0.3)
        for i, pos in enumerate(self.switches_locations):
            pixel_pos = self._grid_to_pixel(pos)
            is_on = self.switches_status[i]
            
            if is_on:
                self._draw_glow_circle(self.screen, pixel_pos, int(radius * 1.8), self.COLOR_SWITCH_ON_GLOW)
                pygame.gfxdraw.aacircle(self.screen, pixel_pos[0], pixel_pos[1], radius, self.COLOR_SWITCH_ON_BORDER)
                pygame.gfxdraw.filled_circle(self.screen, pixel_pos[0], pixel_pos[1], radius, self.COLOR_SWITCH_ON)
            else:
                pygame.gfxdraw.aacircle(self.screen, pixel_pos[0], pixel_pos[1], radius, self.COLOR_SWITCH_OFF_BORDER)
                pygame.gfxdraw.filled_circle(self.screen, pixel_pos[0], pixel_pos[1], radius, self.COLOR_SWITCH_OFF)

    def _render_characters(self):
        char_radius = int(self.cell_pixel_size * 0.4)
        
        # Character 1 (Blue)
        p1_pixel_pos = self._grid_to_pixel(self.char1_pos)
        self._draw_glow_circle(self.screen, p1_pixel_pos, int(char_radius * 1.6), self.COLOR_P1_GLOW)
        pygame.gfxdraw.aacircle(self.screen, p1_pixel_pos[0], p1_pixel_pos[1], char_radius, self.COLOR_P1)
        pygame.gfxdraw.filled_circle(self.screen, p1_pixel_pos[0], p1_pixel_pos[1], char_radius, self.COLOR_P1)
        
        # Character 2 (Red)
        p2_pixel_pos = self._grid_to_pixel(self.char2_pos)
        self._draw_glow_circle(self.screen, p2_pixel_pos, int(char_radius * 1.6), self.COLOR_P2_GLOW)
        pygame.gfxdraw.aacircle(self.screen, p2_pixel_pos[0], p2_pixel_pos[1], char_radius, self.COLOR_P2)
        pygame.gfxdraw.filled_circle(self.screen, p2_pixel_pos[0], p2_pixel_pos[1], char_radius, self.COLOR_P2)

    def _render_ui(self):
        # Turns Left
        turns_left = self.MAX_STEPS - self.steps
        turns_text = self.font_main.render(f"TURNS LEFT: {turns_left}", True, self.COLOR_TEXT)
        self.screen.blit(turns_text, (20, 20))
        
        # Switches Activated
        switches_on = sum(self.switches_status)
        switches_text = self.font_main.render(f"SWITCHES: {switches_on} / {len(self.switches_locations)}", True, self.COLOR_TEXT)
        text_rect = switches_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(switches_text, text_rect)
        
        # Action Key hints
        action_text_p1 = self.font_small.render("P1 Move: ARROWS", True, self.COLOR_P1)
        self.screen.blit(action_text_p1, (20, self.SCREEN_HEIGHT - 50))
        
        action_text_p2a = self.font_small.render("P2 Move w/ P1: SPACE", True, self.COLOR_P2)
        self.screen.blit(action_text_p2a, (20, self.SCREEN_HEIGHT - 30))
        
        action_text_p2b = self.font_small.render("P2 Move vs P1: SHIFT", True, self.COLOR_P2)
        text_rect_p2b = action_text_p2b.get_rect(topright=(self.SCREEN_WIDTH - 20, self.SCREEN_HEIGHT - 30))
        self.screen.blit(action_text_p2b, text_rect_p2b)

    # --- Helper Utilities ---

    def _grid_to_pixel(self, grid_pos):
        # grid_pos is [col, row]
        col, row = grid_pos
        x = self.board_offset_x + (col + 0.5) * self.cell_pixel_size
        y = self.board_offset_y + (row + 0.5) * self.cell_pixel_size
        return int(x), int(y)

    def _draw_glow_circle(self, surface, center, radius, color):
        # color should be (r, g, b, a)
        temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color, (radius, radius), radius)
        surface.blit(temp_surf, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

# Example of how to run the environment for manual play
if __name__ == '__main__':
    # The following is not part of the required solution but is useful for testing
    try:
        env = GameEnv()
        
        obs, info = env.reset()
        terminated = False
        
        # For manual play, we need a real display
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        pygame.display.init()
        display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Synchrony Puzzle")
        clock = pygame.time.Clock()
        
        print("\n--- Manual Control ---")
        print(GameEnv.user_guide)
        print("R: Reset environment")
        print("Q: Quit")

        running = True
        while running:
            action = [0, 0, 0] # Default: no action
            should_step = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                        print("\n--- Environment Reset ---")
                    # Any key press that could lead to an action triggers a step
                    if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_RSHIFT]:
                        should_step = True
            
            if should_step and not terminated:
                keys = pygame.key.get_pressed()
                
                if keys[pygame.K_UP]: action[0] = 1
                elif keys[pygame.K_DOWN]: action[0] = 2
                elif keys[pygame.K_LEFT]: action[0] = 3
                elif keys[pygame.K_RIGHT]: action[0] = 4
                else: action[0] = 0

                if keys[pygame.K_SPACE]: action[1] = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
                
                # Only step if a move is attempted
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Step: {info['steps']}, Reward: {reward:.1f}, Score: {info['score']:.1f}, Terminated: {terminated}")

            # --- Rendering ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(10) # Lower tick rate for turn-based game

    finally:
        env.close()