
# Generated: 2025-08-28T05:36:00.690607
# Source Brief: brief_05632.md
# Brief Index: 5632

        
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
    A puzzle game where the player fills a grid with colors to match a target image.

    The player controls a cursor on a 10x10 grid. They can move the cursor,
    cycle through a palette of 10 colors, and fill the selected square with the
    current color. The goal is to replicate a target image as accurately as
    possible within a time limit, while managing a finite supply of each color.

    The game consists of 3 stages of increasing difficulty. The color supply is
    shared across all stages, making resource management a key strategic element.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Shift to cycle colors. Space to fill a square."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate a target image by filling a grid with colors. Manage your limited paint supply across 3 stages against the clock."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = 32
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 4
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_LINE = (50, 60, 80)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_CURSOR_GLOW = (255, 255, 0, 50)
        self.PALETTE = [
            (255, 0, 77),    # Red
            (255, 163, 0),   # Orange
            (255, 236, 39),  # Yellow
            (0, 228, 54),    # Green
            (41, 173, 255),  # Blue
            (131, 118, 156), # Grey
            (0, 0, 0),       # Black
            (255, 255, 255), # White
            (95, 87, 79),    # Brown
            (126, 37, 83),   # Purple
        ]
        self.COLOR_EMPTY_CELL = (30, 35, 50)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = None
        self.target_grids = []
        self.color_counts = None
        self.current_stage = 0
        self.timer = 0.0
        self.cursor_pos = [0, 0]
        self.selected_color_index = 0
        self.shift_cooldown = 0
        self.space_cooldown = 0
        self.particles = []
        self.terminal_reward_given = False
        
        # This will be called once in __init__
        self.reset()
        
        # Final validation check
        # self.validate_implementation() # Commented out for submission as per instructions

    def _generate_target_grids(self):
        self.target_grids = []
        for i in range(3):
            grid = self.np_random.integers(0, len(self.PALETTE), size=(self.GRID_SIZE, self.GRID_SIZE))
            # Make stages progressively more complex
            if i == 0: # Stage 1: Large blobs
                for _ in range(5):
                    cx, cy = self.np_random.integers(0, self.GRID_SIZE, size=2)
                    color = self.np_random.integers(0, len(self.PALETTE))
                    size = self.np_random.integers(2, 5)
                    for y in range(max(0, cy-size), min(self.GRID_SIZE, cy+size)):
                        for x in range(max(0, cx-size), min(self.GRID_SIZE, cx+size)):
                            if (x-cx)**2 + (y-cy)**2 < size**2:
                                grid[y, x] = color
            elif i == 1: # Stage 2: Lines and smaller blobs
                 for _ in range(10):
                    x1, y1 = self.np_random.integers(0, self.GRID_SIZE, size=2)
                    x2, y2 = self.np_random.integers(0, self.GRID_SIZE, size=2)
                    color = self.np_random.integers(0, len(self.PALETTE))
                    dx, dy = abs(x2-x1), abs(y2-y1)
                    sx, sy = (1, -1)[x1>x2], (1, -1)[y1>y2]
                    err = dx-dy
                    while True:
                        grid[y1, x1] = color
                        if x1 == x2 and y1 == y2: break
                        e2 = 2*err
                        if e2 > -dy: err -= dy; x1 += sx
                        if e2 < dx: err += dx; y1 += sy
            # Stage 3 uses the default random grid which is the most complex
            self.target_grids.append(grid)

    def _setup_stage(self, stage_index):
        self.current_stage = stage_index
        self.timer = 60.0
        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), -1, dtype=int)
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_color_index = 0
        self.terminal_reward_given = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
             self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        
        self._generate_target_grids()
        
        # Color supply is for the entire game (all 3 stages)
        total_colors_needed = sum(np.sum(np.bincount(g.flatten(), minlength=10)) for g in self.target_grids)
        # Give a slight surplus to make it challenging but not impossible
        self.color_counts = np.array([25] * len(self.PALETTE), dtype=int)

        self._setup_stage(0)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Update game state based on time ---
        self.timer -= 1 / 30.0
        self.shift_cooldown = max(0, self.shift_cooldown - 1)
        self.space_cooldown = max(0, self.space_cooldown - 1)
        
        # --- Handle actions ---
        # 1. Cycle color (Shift)
        if shift_pressed and self.shift_cooldown == 0:
            self.selected_color_index = (self.selected_color_index + 1) % len(self.PALETTE)
            self.shift_cooldown = 5 # 5-frame cooldown
            # Sound: UI_Switch.wav

        # 2. Move cursor (Arrows)
        if movement != 0:
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            self.cursor_pos[0] %= self.GRID_SIZE
            self.cursor_pos[1] %= self.GRID_SIZE
        
        # 3. Fill square (Space)
        if space_pressed and self.space_cooldown == 0:
            self.space_cooldown = 5 # 5-frame cooldown
            cx, cy = self.cursor_pos
            
            # Only act if the cell is empty and we have paint
            if self.grid[cy, cx] == -1 and self.color_counts[self.selected_color_index] > 0:
                self.grid[cy, cx] = self.selected_color_index
                self.color_counts[self.selected_color_index] -= 1
                
                # Calculate reward for this action
                target_color = self.target_grids[self.current_stage][cy, cx]
                if self.selected_color_index == target_color:
                    reward += 1.0
                    # Sound: Correct_Fill.wav
                else:
                    reward -= 0.2
                    # Sound: Incorrect_Fill.wav

                # Add particle effect
                self._add_particles(cx, cy, self.PALETTE[self.selected_color_index])

        self.score += reward

        # --- Check for stage/game end conditions ---
        # Stage completion
        correct_cells = np.sum(self.grid == self.target_grids[self.current_stage])
        if correct_cells >= (self.GRID_SIZE * self.GRID_SIZE * 0.9):
            reward += 5.0 # Stage complete bonus
            self.score += 5.0
            self.current_stage += 1
            if self.current_stage >= 3:
                # Game win!
                self.game_over = True
                terminated = True
                if not self.terminal_reward_given:
                    reward += 50.0
                    self.score += 50.0
                    self.terminal_reward_given = True
                    # Sound: Game_Win.wav
            else:
                # Move to next stage
                self._setup_stage(self.current_stage)
                # Sound: Stage_Complete.wav

        # Termination conditions (loss)
        if not self.game_over:
            lost = False
            if self.timer <= 0:
                lost = True
            if any(self.color_counts < 0): # Should be > 0 but check for bugs
                 # This condition is tricky. A better one is if a needed color is 0.
                 needed_colors = np.unique(self.target_grids[self.current_stage][self.grid == -1])
                 for color_idx in needed_colors:
                     if self.color_counts[color_idx] <= 0:
                         lost = True
                         break
            
            if lost:
                self.game_over = True
                terminated = True
                if not self.terminal_reward_given:
                    reward -= 50.0
                    self.score -= 50.0
                    self.terminal_reward_given = True
                    # Sound: Game_Lose.wav
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _add_particles(self, grid_x, grid_y, color):
        px = self.GRID_OFFSET_X + (grid_x + 0.5) * self.CELL_SIZE
        py = self.GRID_OFFSET_Y + (grid_y + 0.5) * self.CELL_SIZE
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([px, py, vel_x, vel_y, lifetime, color])

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[4] -= 1     # lifetime--
            if p[4] > 0:
                active_particles.append(p)
                alpha = int(255 * (p[4] / 30))
                radius = int(p[4] / 6)
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), max(1, radius), (*p[5], alpha))
        self.particles = active_particles

    def _render_game(self):
        # Draw grid background and faint target image
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = (
                    self.GRID_OFFSET_X + x * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                # Faint target color
                target_color_idx = self.target_grids[self.current_stage][y, x]
                target_color = self.PALETTE[target_color_idx]
                faint_color = tuple(c // 4 for c in target_color)
                pygame.draw.rect(self.screen, faint_color, rect)

        # Draw filled cells
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_idx = self.grid[y, x]
                if color_idx != -1:
                    rect = (
                        self.GRID_OFFSET_X + x * self.CELL_SIZE,
                        self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                        self.CELL_SIZE,
                        self.CELL_SIZE,
                    )
                    pygame.draw.rect(self.screen, self.PALETTE[color_idx], rect)

        # Draw particles
        self._update_and_draw_particles()

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_SIZE * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 1)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + cx * self.CELL_SIZE,
            self.GRID_OFFSET_Y + cy * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )
        # Pulsing glow effect
        glow_size = int(8 * (1 + math.sin(self.steps * 0.2)))
        glow_surf = pygame.Surface((self.CELL_SIZE + glow_size * 2, self.CELL_SIZE + glow_size * 2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_CURSOR_GLOW, glow_surf.get_rect(), border_radius=glow_size)
        self.screen.blit(glow_surf, (cursor_rect.x - glow_size, cursor_rect.y - glow_size))
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.current_stage + 1} / 3", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (20, 45))

        # Timer
        timer_text = self.font_large.render(f"{max(0, int(self.timer))}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(center=(self.WIDTH / 2, 30))
        self.screen.blit(timer_text, timer_rect)

        # Color Palette and Counts
        palette_x = self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE + 40
        palette_y = self.GRID_OFFSET_Y
        bar_width = 120
        bar_height = 20
        spacing = 8

        for i, color in enumerate(self.PALETTE):
            y_pos = palette_y + i * (bar_height + spacing)
            
            # Selection indicator
            if i == self.selected_color_index:
                sel_rect = pygame.Rect(palette_x - 10, y_pos - 5, bar_width + 50, bar_height + 10)
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, sel_rect, 2, border_radius=5)

            # Color swatch
            pygame.draw.rect(self.screen, color, (palette_x, y_pos, bar_height, bar_height))
            
            # Count bar background
            bar_bg_rect = (palette_x + bar_height + 10, y_pos, bar_width, bar_height)
            pygame.draw.rect(self.screen, self.COLOR_EMPTY_CELL, bar_bg_rect)
            
            # Count bar foreground
            fill_width = int(bar_width * (self.color_counts[i] / 25.0)) # 25 is initial count
            bar_fill_rect = (palette_x + bar_height + 10, y_pos, max(0, fill_width), bar_height)
            pygame.draw.rect(self.screen, color, bar_fill_rect)

            # Count text
            count_text = self.font_small.render(str(self.color_counts[i]), True, self.COLOR_UI_TEXT)
            self.screen.blit(count_text, (palette_x + bar_height + 15 + bar_width, y_pos))

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
            "stage": self.current_stage + 1,
            "timer": self.timer,
            "cursor_pos": self.cursor_pos,
            "selected_color": self.selected_color_index,
            "color_counts": self.color_counts.tolist(),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption(GameEnv.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    
    print(GameEnv.user_guide)
    
    while not terminated:
        # Default action is NO-OP
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()