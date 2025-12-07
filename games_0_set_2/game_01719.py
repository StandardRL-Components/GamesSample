
# Generated: 2025-08-28T02:29:55.105116
# Source Brief: brief_01719.md
# Brief Index: 1719

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Use arrow keys to push rows/columns. The selection cursor (white box) "
        "advances automatically with each action. Match the target image before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Recreate a target image by pushing rows and columns of pixels "
        "on a grid to match the goal. Plan your moves carefully before the timer expires!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    PIXEL_SIZE = 30
    GRID_W_H = GRID_SIZE * PIXEL_SIZE  # 300

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIMER_BAR = (70, 180, 255)
    COLOR_TIMER_BAR_BG = (40, 50, 70)
    COLOR_SELECTOR = (255, 255, 255)
    COLOR_PUSH_FLASH = (255, 255, 150)

    # Layout
    TARGET_GRID_POS = (20, 85)
    PLAYER_GRID_POS = (320, 85)

    # Game parameters
    TIME_LIMIT_SECONDS = 60.0
    MAX_STEPS = 600  # 60s / 0.1s per step

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self.font_large = pygame.font.Font(None, 36)
            self.font_medium = pygame.font.Font(None, 28)
            self.font_small = pygame.font.Font(None, 22)
        except IOError:
            # Fallback if default font is not found
            self.font_large = pygame.font.SysFont("sans", 36)
            self.font_medium = pygame.font.SysFont("sans", 28)
            self.font_small = pygame.font.SysFont("sans", 22)
            
        # --- Game State Variables ---
        self.np_random = None
        self.target_grid = None
        self.player_grid = None
        self.palette = None
        self.selector_pos = 0 # 0-99
        self.steps = 0
        self.score = 0.0 # Match percentage
        self.max_score_achieved = 0.0
        self.timer = self.TIME_LIMIT_SECONDS
        self.game_over = False
        self.last_push_info = None # For visual feedback

        # The validation is useful for development but should not be run in production.
        # It's called in the __main__ block for demonstration.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.max_score_achieved = 0.0
        self.timer = self.TIME_LIMIT_SECONDS
        self.game_over = False
        self.selector_pos = 0
        self.last_push_info = None

        # 1. Create a color palette
        self.palette = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (80, 255, 255),  # Cyan
            (255, 80, 255),  # Magenta
            (255, 160, 80),  # Orange
            (240, 240, 240)  # White
        ]

        # 2. Generate the target grid
        self.target_grid = self.np_random.choice(len(self.palette), size=(self.GRID_SIZE, self.GRID_SIZE))

        # 3. Generate the player grid by scrambling the target grid
        self.player_grid = self.target_grid.copy()
        num_scrambles = self.np_random.integers(15, 30)
        for _ in range(num_scrambles):
            axis = self.np_random.integers(0, 2)  # 0 for row, 1 for col
            index = self.np_random.integers(0, self.GRID_SIZE)
            shift = self.np_random.choice([-1, 1])
            if axis == 0: # Push row
                self.player_grid[index, :] = np.roll(self.player_grid[index, :], shift)
            else: # Push col
                self.player_grid[:, index] = np.roll(self.player_grid[:, index], shift)
        
        # Ensure start is not solved
        if np.array_equal(self.player_grid, self.target_grid):
             self.player_grid[0, :] = np.roll(self.player_grid[0, :], 1)

        self._update_score()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # --- 1. Unpack action and update game logic ---
        movement = action[0]
        self.last_push_info = None # Clear previous visual feedback

        sel_row = self.selector_pos // self.GRID_SIZE
        sel_col = self.selector_pos % self.GRID_SIZE

        # Actions 1-4 perform a push
        if movement == 1: # Push Up
            # Sound effect placeholder: # play_sound("push_vertical")
            self.player_grid[:, sel_col] = np.roll(self.player_grid[:, sel_col], -1)
            self.last_push_info = {"type": "col", "index": sel_col}
        elif movement == 2: # Push Down
            # Sound effect placeholder: # play_sound("push_vertical")
            self.player_grid[:, sel_col] = np.roll(self.player_grid[:, sel_col], 1)
            self.last_push_info = {"type": "col", "index": sel_col}
        elif movement == 3: # Push Left
            # Sound effect placeholder: # play_sound("push_horizontal")
            self.player_grid[sel_row, :] = np.roll(self.player_grid[sel_row, :], -1)
            self.last_push_info = {"type": "row", "index": sel_row}
        elif movement == 4: # Push Right
            # Sound effect placeholder: # play_sound("push_horizontal")
            self.player_grid[sel_row, :] = np.roll(self.player_grid[sel_row, :], 1)
            self.last_push_info = {"type": "row", "index": sel_row}
        
        # ANY action (including no-op) advances the selector
        self.selector_pos = (self.selector_pos + 1) % (self.GRID_SIZE * self.GRID_SIZE)
        
        self.steps += 1
        self.timer -= self.TIME_LIMIT_SECONDS / self.MAX_STEPS

        # --- 2. Calculate reward ---
        reward = 0
        
        matches = self._update_score()
        
        # Continuous reward: +0.1 for each correct pixel
        reward += matches * 0.1

        # Event reward: +5 for new high score
        if self.score > self.max_score_achieved:
            # Sound effect placeholder: # play_sound("new_highscore")
            reward += 5.0
            self.max_score_achieved = self.score

        # --- 3. Check for termination ---
        terminated = False
        if self.score >= 100.0:
            # Sound effect placeholder: # play_sound("win_game")
            reward += 100.0 # Goal reward
            terminated = True
            self.game_over = True
        elif self.timer <= 0 or self.steps >= self.MAX_STEPS:
            self.timer = 0
            # Sound effect placeholder: # play_sound("timeout")
            reward -= 100.0 # Timeout penalty
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_score(self):
        matches = np.sum(self.player_grid == self.target_grid)
        self.score = (matches / (self.GRID_SIZE * self.GRID_SIZE)) * 100
        return matches

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
        }

    def _render_game(self):
        # Render Target Grid
        self._render_grid(self.target_grid, self.TARGET_GRID_POS, is_player_grid=False)
        # Render Player Grid
        self._render_grid(self.player_grid, self.PLAYER_GRID_POS, is_player_grid=True)

        # Render selector on player grid
        sel_row = self.selector_pos // self.GRID_SIZE
        sel_col = self.selector_pos % self.GRID_SIZE
        selector_rect = pygame.Rect(
            self.PLAYER_GRID_POS[0] + sel_col * self.PIXEL_SIZE,
            self.PLAYER_GRID_POS[1] + sel_row * self.PIXEL_SIZE,
            self.PIXEL_SIZE, self.PIXEL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, 3)

        # Render push flash effect
        if self.last_push_info:
            flash_surface = pygame.Surface(
                (self.GRID_W_H if self.last_push_info["type"] == "row" else self.PIXEL_SIZE,
                 self.PIXEL_SIZE if self.last_push_info["type"] == "row" else self.GRID_W_H),
                pygame.SRCALPHA
            )
            flash_surface.fill((*self.COLOR_PUSH_FLASH, 70)) # Transparent yellow
            
            if self.last_push_info["type"] == "row":
                pos_x = self.PLAYER_GRID_POS[0]
                pos_y = self.PLAYER_GRID_POS[1] + self.last_push_info["index"] * self.PIXEL_SIZE
            else: # col
                pos_x = self.PLAYER_GRID_POS[0] + self.last_push_info["index"] * self.PIXEL_SIZE
                pos_y = self.PLAYER_GRID_POS[1]
            self.screen.blit(flash_surface, (pos_x, pos_y))


    def _render_grid(self, grid_data, pos, is_player_grid):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = grid_data[r, c]
                color = self.palette[color_index]
                
                if is_player_grid:
                    is_correct = grid_data[r, c] == self.target_grid[r, c]
                    if not is_correct:
                        # Hint at the correct color by desaturating it
                        target_color_index = self.target_grid[r,c]
                        color = self._desaturate(self.palette[target_color_index], 0.7)

                pixel_rect = pygame.Rect(
                    pos[0] + c * self.PIXEL_SIZE,
                    pos[1] + r * self.PIXEL_SIZE,
                    self.PIXEL_SIZE, self.PIXEL_SIZE
                )
                pygame.draw.rect(self.screen, color, pixel_rect)
        
        # Draw grid border
        border_rect = pygame.Rect(pos[0], pos[1], self.GRID_W_H, self.GRID_W_H)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, border_rect, 1)

    def _render_ui(self):
        # --- Titles ---
        target_text = self.font_medium.render("TARGET", True, self.COLOR_UI_TEXT)
        self.screen.blit(target_text, (self.TARGET_GRID_POS[0], self.TARGET_GRID_POS[1] - 30))

        player_text = self.font_medium.render("YOUR GRID", True, self.COLOR_UI_TEXT)
        self.screen.blit(player_text, (self.PLAYER_GRID_POS[0], self.PLAYER_GRID_POS[1] - 30))

        # --- Score / Match % ---
        score_str = f"MATCH: {self.score:.1f}%"
        score_text = self.font_large.render(score_str, True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, 35))
        self.screen.blit(score_text, score_rect)

        # --- Timer ---
        # Bar
        timer_bar_width = self.SCREEN_WIDTH - 40
        timer_bar_bg_rect = pygame.Rect(20, self.SCREEN_HEIGHT - 30, timer_bar_width, 20)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, timer_bar_bg_rect, border_radius=5)
        
        current_width = max(0, timer_bar_width * (self.timer / self.TIME_LIMIT_SECONDS))
        timer_bar_rect = pygame.Rect(20, self.SCREEN_HEIGHT - 30, current_width, 20)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, timer_bar_rect, border_radius=5)
        
        # Numeric
        timer_str = f"{max(0, self.timer):.1f}"
        timer_text = self.font_medium.render(timer_str, True, self.COLOR_UI_TEXT)
        timer_text_rect = timer_text.get_rect(center=timer_bar_bg_rect.center)
        self.screen.blit(timer_text, timer_text_rect)

    def _desaturate(self, color, amount):
        """Desaturates a color by a given amount (0.0 to 1.0)."""
        h, s, v, a = pygame.Color(*color).hsva
        s = max(0, s * (1 - amount))
        desaturated_color = pygame.Color(0)
        desaturated_color.hsva = (h, s, v, a)
        return (desaturated_color.r, desaturated_color.g, desaturated_color.b)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this to verify implementation.
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        
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
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Pixel Pusher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # Game loop
    running = True
    while running:
        action_taken_this_frame = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action = [0, 0, 0] # [movement, space, shift]
                
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                    continue
                if done: continue

                # Map keys to actions for human play
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                else:
                    # Allow any other key press to be a no-op that advances the selector
                    action[0] = 0
                
                action_taken_this_frame = True
                
                # A key was pressed, so we take a step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.1f}%, Done: {done}")

        # Render the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for human play

    env.close()