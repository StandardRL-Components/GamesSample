
# Generated: 2025-08-27T21:13:41.880751
# Source Brief: brief_02718.md
# Brief Index: 2718

        
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
    """
    A Gymnasium environment for the puzzle game "Number Ninja".
    The player selects adjacent numbers in a grid that sum to 10 to clear them.
    The goal is to clear the entire grid.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a number. "
        "Select an adjacent number to combine. Press Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically combine adjacent numbers in a grid to sum to 10 and clear the board as the Number Ninja."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 10
    GRID_HEIGHT = 6
    CELL_SIZE = 50
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (60, 60, 80)
    COLOR_CURSOR = (255, 220, 0)
    COLOR_SELECTED = (0, 200, 255)
    COLOR_INVALID = (255, 50, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    
    NUMBER_COLORS = [
        (0, 0, 0),         # 0 (unused)
        (230, 57, 70),     # 1: Imperial Red
        (241, 150, 38),    # 2: Orange Peel
        (255, 215, 0),     # 3: Gold
        (144, 238, 144),   # 4: Light Green
        (66, 179, 245),    # 5: Dodger Blue
        (29, 122, 225),    # 6: Royal Blue
        (169, 122, 227),   # 7: Medium Purple
        (247, 102, 163),   # 8: Pink
        (168, 218, 220),   # 9: Light Cyan
    ]

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
        self.font_main = pygame.font.SysFont("Consolas", 28, bold=True)
        self.font_ui = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Impact", 80)

        self.grid_offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        
        # State variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.feedback_effect = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_pos = None
        self.particles = []
        self.feedback_effect = None
        
        self._generate_grid()

        return self._get_observation(), self._get_info()
    
    def _generate_grid(self):
        while True:
            self.grid = self.np_random.integers(1, 10, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if len(self._find_valid_combinations()) > 0:
                break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # 2. Handle deselect action (Shift)
        if shift_press and self.selected_pos is not None:
            self.selected_pos = None
            # sound: Deselect_SFX

        # 3. Handle select/combine action (Space)
        if space_press:
            cursor_x, cursor_y = self.cursor_pos
            
            if self.selected_pos is None:
                # Select the first number
                if self.grid[cursor_y, cursor_x] > 0:
                    self.selected_pos = tuple(self.cursor_pos)
                    # sound: Select_SFX
                # else: trying to select empty cell, no-op
            else:
                # A number is already selected, try to combine
                sel_x, sel_y = self.selected_pos
                
                if (cursor_x, cursor_y) == (sel_x, sel_y):
                    # Trying to select the same cell again
                    reward -= 0.1
                    self.selected_pos = None # Deselect on re-click
                elif self._is_adjacent(self.selected_pos, self.cursor_pos):
                    reward += 1.0
                    val1 = self.grid[sel_y, sel_x]
                    val2 = self.grid[cursor_y, cursor_x]
                    
                    if val1 + val2 == 10:
                        # Successful combination
                        reward += 10.0
                        self.score += 10
                        self.grid[sel_y, sel_x] = 0
                        self.grid[cursor_y, cursor_x] = 0
                        self._create_particles(self.selected_pos)
                        self._create_particles(self.cursor_pos)
                        self.selected_pos = None
                        # sound: Combo_Success_SFX
                    else:
                        # Failed combination (adjacent but not sum 10)
                        self.feedback_effect = {'type': 'invalid', 'pos1': self.selected_pos, 'pos2': self.cursor_pos, 'timer': 15}
                        self.selected_pos = None
                        # sound: Combo_Fail_SFX
                else:
                    # Not adjacent
                    reward -= 0.1
                    self.feedback_effect = {'type': 'invalid', 'pos1': self.selected_pos, 'pos2': self.cursor_pos, 'timer': 15}
                    self.selected_pos = None
                    # sound: Error_SFX

        self.steps += 1
        
        # 4. Check for termination
        terminated = self._check_termination()
        if terminated:
            if np.sum(self.grid) == 0: # Win
                reward += 100.0
                self.score += 100
            else: # Loss or max steps
                reward -= 50.0

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if np.sum(self.grid) == 0:
            self.game_over = True
            return True
        if not self._find_valid_combinations():
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _is_adjacent(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) + abs(y1 - y2) == 1

    def _find_valid_combinations(self):
        combos = []
        # Horizontal
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 1):
                if self.grid[r, c] > 0 and self.grid[r, c+1] > 0 and self.grid[r, c] + self.grid[r, c+1] == 10:
                    combos.append(((c, r), (c+1, r)))
        # Vertical
        for r in range(self.GRID_HEIGHT - 1):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0 and self.grid[r+1, c] > 0 and self.grid[r, c] + self.grid[r+1, c] == 10:
                    combos.append(((c, r), (c, r+1)))
        return combos

    def _grid_to_pixels(self, grid_pos, center=True):
        x, y = grid_pos
        px = self.grid_offset_x + x * self.CELL_SIZE
        py = self.grid_offset_y + y * self.CELL_SIZE
        if center:
            px += self.CELL_SIZE // 2
            py += self.CELL_SIZE // 2
        return int(px), int(py)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Update and draw particles
        self._update_and_draw_particles()
        
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_offset_x + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_offset_y), (x, self.grid_offset_y + self.GRID_HEIGHT * self.CELL_SIZE), 1)
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_offset_y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, y), (self.grid_offset_x + self.GRID_WIDTH * self.CELL_SIZE, y), 1)

        # Draw numbers and highlights
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                val = self.grid[r, c]
                px, py = self._grid_to_pixels((c, r))
                
                # Draw selected highlight (under number)
                if self.selected_pos == (c, r):
                    pygame.gfxdraw.filled_circle(self.screen, px, py, self.CELL_SIZE // 2 - 2, self.COLOR_SELECTED)
                    pygame.gfxdraw.aacircle(self.screen, px, py, self.CELL_SIZE // 2 - 2, self.COLOR_SELECTED)

                # Draw number tile and text
                if val > 0:
                    color = self.NUMBER_COLORS[val]
                    pygame.gfxdraw.filled_circle(self.screen, px, py, self.CELL_SIZE // 3, color)
                    pygame.gfxdraw.aacircle(self.screen, px, py, self.CELL_SIZE // 3, color)
                    
                    text_surf = self.font_main.render(str(val), True, (255,255,255))
                    text_rect = text_surf.get_rect(center=(px, py))
                    self.screen.blit(text_surf, text_rect)

        # Draw cursor
        cursor_px, cursor_py = self._grid_to_pixels(self.cursor_pos)
        cursor_rect = pygame.Rect(0, 0, self.CELL_SIZE-4, self.CELL_SIZE-4)
        cursor_rect.center = (cursor_px, cursor_py)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)
        
        # Draw feedback effects
        self._draw_feedback()

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Remaining combos
        combos_count = len(self._find_valid_combinations())
        combos_text = self.font_ui.render(f"MOVES: {combos_count}", True, self.COLOR_UI_TEXT)
        combos_rect = combos_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(combos_text, combos_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if np.sum(self.grid) == 0:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
                
            game_over_surf = self.font_game_over.render(msg, True, color)
            game_over_rect = game_over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_surf, game_over_rect)

    def _create_particles(self, grid_pos):
        px, py = self._grid_to_pixels(grid_pos)
        color = self.NUMBER_COLORS[self.np_random.integers(1,10)]
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.uniform(3, 8)
            life = self.np_random.integers(20, 40)
            self.particles.append({'pos': [px, py], 'vel': vel, 'size': size, 'life': life, 'color': color})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['size'] *= 0.97
            p['life'] -= 1
            if p['life'] > 0 and p['size'] > 1:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), p['color'])
                active_particles.append(p)
        self.particles = active_particles

    def _draw_feedback(self):
        if self.feedback_effect and self.feedback_effect['timer'] > 0:
            if self.feedback_effect['type'] == 'invalid':
                alpha = int(200 * (self.feedback_effect['timer'] / 15))
                color = self.COLOR_INVALID + (alpha,)
                
                p1 = self._grid_to_pixels(self.feedback_effect['pos1'])
                p2 = self._grid_to_pixels(self.feedback_effect['pos2'])
                
                pygame.gfxdraw.aacircle(self.screen, p1[0], p1[1], self.CELL_SIZE//2 - 4, color)
                pygame.gfxdraw.aacircle(self.screen, p2[0], p2[1], self.CELL_SIZE//2 - 4, color)
            self.feedback_effect['timer'] -= 1
        else:
            self.feedback_effect = None

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "selected_pos": self.selected_pos,
            "remaining_combos": len(self._find_valid_combinations())
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    pygame.display.set_caption("Number Ninja")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    action = [0, 0, 0] # No-op, no space, no shift
    
    print(env.user_guide)
    
    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_press = 1 if keys[pygame.K_SPACE] else 0
        shift_press = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_press, shift_press]
        
        # --- Step the environment ---
        # Since auto_advance is False, we step on every frame to process input
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render the observation to the screen ---
        # The observation is (H, W, C), but pygame wants (W, H)
        # and surfarray.make_surface expects (W, H, C)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30) # Limit frame rate for human play

    env.close()