
# Generated: 2025-08-27T23:23:17.446448
# Source Brief: brief_03442.md
# Brief Index: 3442

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor, Shift to cycle color, Space to paint."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target pixel art image by painting tiles before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Game Constants ---
    GRID_DIMS = (10, 10)
    MAX_STEPS = GRID_DIMS[0] * GRID_DIMS[1]
    TIME_LIMIT = 60.0
    TIME_PENALTY_PER_PAINT = 0.6

    # --- Visuals ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_GRID_LINE = (50, 50, 60)
    COLOR_BLANK_TILE = (35, 35, 45)

    PALETTE = [
        (255, 87, 87),    # Red
        (255, 195, 87),   # Orange
        (255, 244, 87),   # Yellow
        (134, 255, 87),   # Green
        (87, 169, 255),   # Blue
        (181, 87, 255),   # Purple
        (255, 130, 201)   # Pink
    ]

    # Layout
    PLAYER_GRID_TILE_SIZE = 36
    PLAYER_GRID_LINE_WIDTH = 2
    PLAYER_GRID_POS = (20, 20)

    UI_PANEL_X = 420
    TARGET_GRID_TILE_SIZE = 12
    TARGET_GRID_LINE_WIDTH = 1
    TARGET_GRID_POS = (UI_PANEL_X + 20, 60)
    
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
        
        # Fonts
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 20)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Initialize state variables
        self.target_image_indices = None
        self.current_grid_indices = None
        self.cursor_pos = None
        self.selected_color_idx = None
        self.time_remaining = None
        self.accuracy = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self._action_buffer = {"shift": False}

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT
        self.cursor_pos = (self.GRID_DIMS[0] // 2, self.GRID_DIMS[1] // 2)
        self.selected_color_idx = 0
        self._action_buffer["shift"] = False

        self._generate_target_image()
        self.current_grid_indices = np.full(self.GRID_DIMS, -1, dtype=int) # -1 for blank
        self._update_accuracy()
        
        return self._get_observation(), self._get_info()

    def _generate_target_image(self):
        self.target_image_indices = self.np_random.integers(
            0, len(self.PALETTE), size=self.GRID_DIMS, dtype=int
        )

    def step(self, action):
        reward = 0
        terminated = False
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Handle Actions ---
        
        # 1. Color Selection (Shift) - trigger on press, not hold
        if shift_held and not self._action_buffer["shift"]:
            self.selected_color_idx = (self.selected_color_idx + 1) % len(self.PALETTE)
            # sfx: color_select_chime.wav
        self._action_buffer["shift"] = shift_held

        # 2. Painting (Space)
        if space_held:
            cx, cy = self.cursor_pos
            target_color = self.target_image_indices[cy, cx]
            current_color = self.current_grid_indices[cy, cx]
            
            # Only apply penalty and reward if the tile is not already correctly painted
            if current_color != self.selected_color_idx:
                self.current_grid_indices[cy, cx] = self.selected_color_idx
                
                if self.selected_color_idx == target_color:
                    reward += 1.0  # Correct paint
                    # sfx: paint_correct.wav
                else:
                    reward -= 0.1 # Incorrect paint
                    # sfx: paint_wrong.wav

                self._update_accuracy()
            
            self.time_remaining -= self.TIME_PENALTY_PER_PAINT
            # sfx: paint_splash.wav
        
        # 3. Movement (Arrows)
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            new_x = (self.cursor_pos[0] + dx) % self.GRID_DIMS[0]
            new_y = (self.cursor_pos[1] + dy) % self.GRID_DIMS[1]
            self.cursor_pos = (new_x, new_y)
            # sfx: cursor_move.wav

        # --- Update Game State ---
        self.steps += 1
        self.score += reward
        
        # --- Check Termination ---
        if self.accuracy >= 1.0:
            terminated = True
            reward += 100  # Victory bonus
            # sfx: level_complete.wav
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 100  # Failure penalty
            # sfx: game_over.wav
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_accuracy(self):
        correct_tiles = np.sum(self.current_grid_indices == self.target_image_indices)
        total_tiles = self.GRID_DIMS[0] * self.GRID_DIMS[1]
        self.accuracy = correct_tiles / total_tiles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw player's painting grid
        self._render_grid(
            self.screen, 
            self.current_grid_indices, 
            self.PLAYER_GRID_POS, 
            self.PLAYER_GRID_TILE_SIZE, 
            self.PLAYER_GRID_LINE_WIDTH
        )
        # Draw cursor
        self._render_cursor()

    def _render_grid(self, surface, grid_indices, top_left, tile_size, line_width):
        grid_width = self.GRID_DIMS[0] * (tile_size + line_width)
        grid_height = self.GRID_DIMS[1] * (tile_size + line_width)
        
        # Draw grid background
        pygame.draw.rect(surface, self.COLOR_GRID_LINE, (*top_left, grid_width, grid_height))

        for y in range(self.GRID_DIMS[1]):
            for x in range(self.GRID_DIMS[0]):
                color_idx = grid_indices[y, x]
                color = self.PALETTE[color_idx] if color_idx != -1 else self.COLOR_BLANK_TILE
                
                px = top_left[0] + x * (tile_size + line_width)
                py = top_left[1] + y * (tile_size + line_width)
                
                pygame.draw.rect(surface, color, (px, py, tile_size, tile_size))

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        tile_size = self.PLAYER_GRID_TILE_SIZE
        line_width = self.PLAYER_GRID_LINE_WIDTH
        
        px = self.PLAYER_GRID_POS[0] + cx * (tile_size + line_width)
        py = self.PLAYER_GRID_POS[1] + cy * (tile_size + line_width)
        
        # Pulsing alpha for cursor glow
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        
        cursor_surface = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
        cursor_surface.fill((*self.COLOR_CURSOR, alpha))
        self.screen.blit(cursor_surface, (px, py))

    def _render_ui(self):
        # --- Helper for drawing text with shadow ---
        def draw_text(text, font, color, pos, shadow_color, shadow_offset=(2, 2)):
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # --- UI Panel ---
        # Target Image
        draw_text("TARGET", self.font_large, self.COLOR_TEXT, (self.UI_PANEL_X + 20, 20), self.COLOR_TEXT_SHADOW)
        self._render_grid(
            self.screen,
            self.target_image_indices,
            self.TARGET_GRID_POS,
            self.TARGET_GRID_TILE_SIZE,
            self.TARGET_GRID_LINE_WIDTH
        )

        # Time
        draw_text("TIME", self.font_medium, self.COLOR_TEXT, (self.UI_PANEL_X + 20, 210), self.COLOR_TEXT_SHADOW)
        time_str = f"{max(0, self.time_remaining):.1f}"
        draw_text(time_str, self.font_large, self.COLOR_TEXT, (self.UI_PANEL_X + 20, 235), self.COLOR_TEXT_SHADOW)

        # Accuracy
        draw_text("ACCURACY", self.font_medium, self.COLOR_TEXT, (self.UI_PANEL_X + 20, 270), self.COLOR_TEXT_SHADOW)
        acc_str = f"{self.accuracy * 100:.0f}%"
        draw_text(acc_str, self.font_large, self.COLOR_TEXT, (self.UI_PANEL_X + 20, 295), self.COLOR_TEXT_SHADOW)

        # Selected Color
        draw_text("COLOR", self.font_medium, self.COLOR_TEXT, (self.UI_PANEL_X + 20, 330), self.COLOR_TEXT_SHADOW)
        color_swatch_rect = pygame.Rect(self.UI_PANEL_X + 110, 290, 80, 80)
        pygame.draw.rect(self.screen, self.PALETTE[self.selected_color_idx], color_swatch_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, color_swatch_rect, 2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "accuracy": self.accuracy,
        }

    def close(self):
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


if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Pixel Painter")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Game loop
    while not terminated:
        # --- Action Mapping for Human Play ---
        movement = 0  # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # --- Handle Pygame Events ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            # The game advances state only on key presses for this turn-based game
            if event.type == pygame.KEYDOWN:
                action_taken = True
        
        # --- Step the Environment ---
        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Accuracy: {info['accuracy']:.2f}, Terminated: {terminated}")
        
        # --- Render to Screen ---
        # The observation is already a rendered frame
        # We need to get the latest obs if no action was taken
        current_obs = env._get_observation()
        # Pygame uses (width, height), numpy uses (height, width)
        # Transpose from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(current_obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()
    print("Game Over!")