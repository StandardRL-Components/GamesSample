
# Generated: 2025-08-27T22:55:06.322695
# Source Brief: brief_03290.md
# Brief Index: 3290

        
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
    """
    FarmGrid: A grid-based farming simulation game.

    The player controls a cursor on a grid of farm plots. The objective is to
    plant seeds, wait for them to grow, and harvest the mature crops to earn
    coins. The game is won by accumulating 100 coins before the timer runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selector. "
        "Press space to plant a seed on an empty plot. "
        "Press shift to harvest a mature crop."
    )

    # Short, user-facing description of the game
    game_description = (
        "Cultivate a grid-based farm, strategically planting and harvesting "
        "crops to earn 100 coins before time runs out."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_ROWS = 6
    GRID_COLS = 10
    CELL_SIZE = 50
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_COLS * CELL_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_ROWS * CELL_SIZE) // 2 + 20

    # Colors
    COLOR_BG = (20, 30, 25)
    COLOR_GRID_LINES = (40, 60, 50)
    COLOR_PLOT_EMPTY = (94, 56, 34)
    COLOR_GROW_START = (34, 66, 44)
    COLOR_GROW_END = (100, 180, 80)
    COLOR_PLOT_MATURE = (255, 223, 0)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_COIN_PARTICLE = (255, 215, 0)
    COLOR_WIN = (173, 255, 47)
    COLOR_LOSE = (255, 69, 0)

    # Game Parameters
    MAX_TIME = 1800
    WIN_SCORE = 100
    CROP_GROW_TIME = 100  # in steps
    COIN_PER_HARVEST = 5

    # Plot States
    STATE_EMPTY = 0
    STATE_GROWING = 1
    STATE_MATURE = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)
        self.font_particle = pygame.font.SysFont("Consolas", 18, bold=True)

        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.time_left = None
        self.game_over = None
        self.win = None
        self.particles = None
        self.steps = None

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.grid = np.zeros((self.GRID_COLS, self.GRID_ROWS, 2), dtype=int)
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.score = 0
        self.time_left = self.MAX_TIME
        self.game_over = False
        self.win = False
        self.particles = []
        self.steps = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, no actions should change the state
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Player Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # 1. Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # 2. Plant (Space)
        if space_held:
            cx, cy = self.cursor_pos
            if self.grid[cx, cy, 0] == self.STATE_EMPTY:
                self.grid[cx, cy, 0] = self.STATE_GROWING
                self.grid[cx, cy, 1] = 0  # Reset growth timer
                reward += 0.1
                # SFX: Plant seed sound

        # 3. Harvest (Shift)
        if shift_held:
            cx, cy = self.cursor_pos
            if self.grid[cx, cy, 0] == self.STATE_MATURE:
                self.grid[cx, cy, 0] = self.STATE_EMPTY
                self.grid[cx, cy, 1] = 0
                self.score += self.COIN_PER_HARVEST
                reward += 1.0
                self._create_particle(cx, cy, f"+{self.COIN_PER_HARVEST}")
                # SFX: Harvest/Coin sound

        # --- Update Game State ---
        self.time_left -= 1

        # Update crops
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                if self.grid[x, y, 0] == self.STATE_GROWING:
                    self.grid[x, y, 1] += 1
                    reward += 0.002 # Scaled down from 0.2 to keep rewards reasonable
                    if self.grid[x, y, 1] >= self.CROP_GROW_TIME:
                        self.grid[x, y, 0] = self.STATE_MATURE
                        # SFX: Crop ready chime

        # Update particles
        self._update_particles()

        # --- Check Termination Conditions ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100
        elif self.time_left <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info(),
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": self.time_left}

    def _render_game(self):
        # Draw grid and plots
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                rect = pygame.Rect(
                    self.GRID_X_OFFSET + x * self.CELL_SIZE,
                    self.GRID_Y_OFFSET + y * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                
                state = self.grid[x, y, 0]
                growth = self.grid[x, y, 1]
                
                if state == self.STATE_EMPTY:
                    pygame.draw.rect(self.screen, self.COLOR_PLOT_EMPTY, rect)
                elif state == self.STATE_GROWING:
                    # Interpolate color based on growth
                    progress = min(1.0, growth / self.CROP_GROW_TIME)
                    color = [
                        int(s + (e - s) * progress) 
                        for s, e in zip(self.COLOR_GROW_START, self.COLOR_GROW_END)
                    ]
                    pygame.draw.rect(self.screen, color, rect)
                    # Draw a small "seed"
                    pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, int(3 + progress * 10), self.COLOR_GROW_START)
                elif state == self.STATE_MATURE:
                    pygame.draw.rect(self.screen, self.COLOR_PLOT_MATURE, rect)
                    # Draw a "wheat" icon
                    pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, 18, (255, 200, 0))
                    pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, 18, self.COLOR_BG)


                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect, 1)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_X_OFFSET + cx * self.CELL_SIZE,
            self.GRID_Y_OFFSET + cy * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        # Draw particles
        for p in self.particles:
            text_surf = self.font_particle.render(p["text"], True, p["color"])
            text_surf.set_alpha(p["alpha"])
            self.screen.blit(text_surf, (int(p["pos"][0]), int(p["pos"][1])))

    def _render_ui(self):
        # Score display
        score_text = f"COINS: {self.score:03d}/{self.WIN_SCORE}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Timer display
        time_text = f"TIME: {self.time_left:04d}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        time_rect = time_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(time_surf, time_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg_text = "YOU WIN!"
                msg_color = self.COLOR_WIN
            else:
                msg_text = "TIME UP!"
                msg_color = self.COLOR_LOSE
                
            msg_surf = self.font_game_over.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _create_particle(self, grid_x, grid_y, text):
        pos_x = self.GRID_X_OFFSET + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        pos_y = self.GRID_Y_OFFSET + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        self.particles.append({
            "pos": [pos_x, pos_y],
            "vel": [random.uniform(-0.5, 0.5), -2],
            "text": text,
            "color": self.COLOR_COIN_PARTICLE,
            "life": 45, # frames
            "alpha": 255
        })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.08  # Gravity
            p["life"] -= 1
            p["alpha"] = max(0, int(255 * (p["life"] / 45)))
            if p["life"] <= 0:
                self.particles.remove(p)

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert "score" in info and "steps" in info

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        assert "score" in info and "steps" in info

        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    # This requires a display. If running headlessly, this part will fail.
    try:
        import sys
        # To display the game, we need a screen and to handle events
        pygame.display.set_caption("FarmGrid")
        display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

        obs, info = env.reset()
        done = False
        
        print("\n--- Manual Control ---")
        print(GameEnv.user_guide)
        print("Press ESC or close the window to quit.")

        while not done:
            # Action defaults
            movement = 0 # none
            space = 0 # released
            shift = 0 # released

            # Pygame event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("--- Game Reset ---")


            # Key state handling for continuous actions
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render to the display
            # The observation is (H, W, C), but pygame blit needs (W, H) surface
            # So we re-transpose it back for display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Control frame rate
            env.clock.tick(30)
            
            if done:
                print(f"Game Over! Final Info: {info}")
                # Wait for a moment before auto-closing or resetting
                pygame.time.wait(3000)
                obs, info = env.reset()
                done = False # To continue playing after a game over

        pygame.quit()
        sys.exit()

    except ImportError:
        print("Pygame not found, skipping manual play example.")
    except pygame.error as e:
        print(f"Pygame display error: {e}. Can't run manual play example.")
        print("This is expected in a headless environment.")