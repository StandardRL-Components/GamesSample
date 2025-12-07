
# Generated: 2025-08-28T02:57:30.520734
# Source Brief: brief_01865.md
# Brief Index: 1865

        
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
        "Controls: ←→ to move the falling letter. Press Space to drop it instantly. Form full horizontal lines to clear them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based puzzle game. Strategically drop falling letters to match and clear lines, aiming for a high score and combos."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 20
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2 - 100
        self.GRID_Y = 0

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_UI_BG = (30, 35, 55)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_HEADER = (100, 180, 255)
        self.LETTER_DEFS = [
            {'char': 'A', 'color': (0, 200, 200)},  # Cyan
            {'char': 'B', 'color': (255, 200, 0)},  # Yellow
            {'char': 'C', 'color': (200, 0, 200)},  # Magenta
            {'char': 'D', 'color': (0, 200, 50)},   # Green
            {'char': 'E', 'color': (255, 100, 0)},  # Orange
        ]

        # Game settings
        self.WIN_CONDITION_LINES = 100
        self.MAX_STEPS = 10000
        self.INITIAL_FALL_TIME = 0.5  # seconds per cell
        self.MOVE_DELAY = 0.1 # seconds between horizontal moves

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_time = self.INITIAL_FALL_TIME
        self.fall_timer = 0.0
        self.move_timer = 0.0
        self.prev_space_state = False
        self.animation_effects = []
        self.np_random = None

        self.reset()
        
    def _create_new_piece(self):
        piece_id = self.np_random.integers(1, len(self.LETTER_DEFS) + 1)
        return {'id': piece_id, 'x': self.GRID_COLS // 2 - 1, 'y': 0}

    def _spawn_new_piece(self):
        self.current_piece = self.next_piece
        self.next_piece = self._create_new_piece()
        self.fall_timer = 0.0
        if self._check_collision(self.current_piece):
            self.game_over = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_time = self.INITIAL_FALL_TIME
        self.fall_timer = 0.0
        self.move_timer = 0.0
        self.prev_space_state = False
        self.animation_effects = []
        
        self.next_piece = self._create_new_piece()
        self._spawn_new_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        dt = self.clock.tick(30) / 1000.0  # Target 30 FPS
        self.steps += 1
        reward = 0
        
        if self.game_over:
            reward = -100 # Loss reward
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_state
        self.prev_space_state = space_held

        self.move_timer += dt
        if self.move_timer > self.MOVE_DELAY:
            if movement == 3:  # Left
                self._move_piece(-1)
                self.move_timer = 0.0
            elif movement == 4:  # Right
                self._move_piece(1)
                self.move_timer = 0.0
        
        if space_pressed:
            # Hard drop
            reward_from_drop = self._hard_drop()
            reward += reward_from_drop
            # Hard drop handles spawning new piece, so we can continue to next step
        else:
            # Normal fall
            self.fall_timer += dt
            if self.fall_timer >= self.fall_time:
                self.fall_timer = 0.0
                self.current_piece['y'] += 1
                if self._check_collision(self.current_piece):
                    self.current_piece['y'] -= 1
                    reward_from_lock = self._lock_piece()
                    reward += reward_from_lock
                    self._spawn_new_piece()

        # --- Update Game State ---
        # Difficulty scaling: Fall time decreases as more lines are cleared
        # This interprets "fall speed increases" as "fall time decreases"
        self.fall_time = max(0.05, self.INITIAL_FALL_TIME - (self.lines_cleared // 20) * 0.05)

        # --- Termination Conditions ---
        terminated = False
        if self.game_over:
            reward -= 100
            terminated = True
        elif self.lines_cleared >= self.WIN_CONDITION_LINES:
            reward += 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _move_piece(self, dx):
        if self.current_piece:
            self.current_piece['x'] += dx
            if self._check_collision(self.current_piece):
                self.current_piece['x'] -= dx

    def _hard_drop(self):
        if not self.current_piece: return 0
        
        # Find lowest valid position
        while not self._check_collision(self.current_piece, offset_y=1):
            self.current_piece['y'] += 1
        
        reward = self._lock_piece()
        self._spawn_new_piece()
        return reward

    def _lock_piece(self):
        if not self.current_piece: return 0
        px, py = self.current_piece['x'], self.current_piece['y']
        if 0 <= py < self.GRID_ROWS and 0 <= px < self.GRID_COLS:
            self.grid[py, px] = self.current_piece['id']
        
        self.current_piece = None
        return self._check_and_clear_lines()

    def _check_and_clear_lines(self):
        full_rows = []
        for r in range(self.GRID_ROWS):
            if np.all(self.grid[r, :] != 0):
                full_rows.append(r)
        
        num_cleared = len(full_rows)
        if num_cleared > 0:
            # Add line clear effect
            self.animation_effects.append({'type': 'line_clear', 'rows': full_rows, 'timer': 0.2})
            
            # Add combo effect
            if num_cleared > 1:
                y_pos = (full_rows[0] + 0.5) * self.CELL_SIZE
                self.animation_effects.append({
                    'type': 'combo_text', 
                    'text': f"COMBO x{num_cleared}!",
                    'pos': (self.GRID_X + self.GRID_WIDTH / 2, y_pos),
                    'timer': 1.0,
                    'color': (255, 255, 100)
                })

            # Update grid by removing cleared lines
            self.grid = np.delete(self.grid, full_rows, axis=0)
            new_rows = np.zeros((num_cleared, self.GRID_COLS), dtype=int)
            self.grid = np.vstack((new_rows, self.grid))
            
            # Update score and lines
            self.lines_cleared += num_cleared
            base_score = {1: 40, 2: 100, 3: 300, 4: 1200}.get(num_cleared, 0)
            self.score += base_score

            # Calculate reward based on brief
            reward = num_cleared * 0.1  # Per-line reward
            reward += max(0, num_cleared - 1) # Combo bonus
            return reward
        
        return 0

    def _check_collision(self, piece, offset_x=0, offset_y=0):
        if not piece: return True
        px = piece['x'] + offset_x
        py = piece['y'] + offset_y
        
        if not (0 <= px < self.GRID_COLS): return True
        if not (py < self.GRID_ROWS): return True
        if py < 0: return False # Above the grid is fine
        if self.grid[py, px] != 0: return True
        
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        self._update_and_render_effects(self.clock.get_time() / 1000.0)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_letter_block(self, surface, letter_id, x, y, size):
        if letter_id == 0: return
        
        letter_info = self.LETTER_DEFS[letter_id - 1]
        color = letter_info['color']
        char = letter_info['char']
        
        # 3D Bevel effect for visual polish
        dark_color = tuple(max(0, c - 40) for c in color)
        
        pygame.draw.rect(surface, dark_color, (x, y, size, size))
        pygame.draw.rect(surface, color, (x + 2, y + 2, size - 4, size - 4))
        
        # Render character
        font_surface = self.font_medium.render(char, True, (255, 255, 255))
        text_rect = font_surface.get_rect(center=(x + size / 2, y + size / 2))
        surface.blit(font_surface, text_rect)

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT))
        
        # Draw placed letters
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                letter_id = self.grid[r, c]
                if letter_id != 0:
                    self._render_letter_block(self.screen, letter_id, self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE)

        # Draw falling piece
        if self.current_piece:
            self._render_letter_block(self.screen, self.current_piece['id'], self.GRID_X + self.current_piece['x'] * self.CELL_SIZE, self.GRID_Y + self.current_piece['y'] * self.CELL_SIZE, self.CELL_SIZE)
            
        # Draw grid lines for clarity
        for r in range(self.GRID_ROWS + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_X, self.GRID_Y + r * self.CELL_SIZE), (self.GRID_X + self.GRID_WIDTH, self.GRID_Y + r * self.CELL_SIZE))
        for c in range(self.GRID_COLS + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_X + c * self.CELL_SIZE, self.GRID_Y), (self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT))

    def _render_ui(self):
        ui_x = self.GRID_X + self.GRID_WIDTH + 20
        ui_width = 180
        
        # UI Background panel
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (ui_x - 10, 10, ui_width, self.SCREEN_HEIGHT - 20), border_radius=10)
        
        # Score Display
        score_header = self.font_medium.render("SCORE", True, self.COLOR_UI_HEADER)
        self.screen.blit(score_header, (ui_x, 30))
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (ui_x, 60))

        # Lines Display
        lines_header = self.font_medium.render("LINES", True, self.COLOR_UI_HEADER)
        self.screen.blit(lines_header, (ui_x, 130))
        lines_text = self.font_large.render(f"{self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lines_text, (ui_x, 160))

        # Next Piece Display
        next_header = self.font_medium.render("NEXT", True, self.COLOR_UI_HEADER)
        self.screen.blit(next_header, (ui_x, 230))
        
        next_box_size = self.CELL_SIZE * 4
        pygame.draw.rect(self.screen, self.COLOR_BG, (ui_x, 260, next_box_size, next_box_size))
        if self.next_piece:
            self._render_letter_block(self.screen, self.next_piece['id'], ui_x + self.CELL_SIZE, 260 + self.CELL_SIZE, self.CELL_SIZE * 2)

    def _update_and_render_effects(self, dt):
        # Handle active effects like line clears and combo text
        next_effects = []
        for effect in self.animation_effects:
            effect['timer'] -= dt
            if effect['timer'] > 0:
                if effect['type'] == 'line_clear':
                    # Flash cleared rows white
                    alpha = int(255 * (effect['timer'] / 0.2)) # Fade out
                    flash_surface = pygame.Surface((self.GRID_WIDTH, self.CELL_SIZE), pygame.SRCALPHA)
                    flash_surface.fill((255, 255, 255, alpha))
                    for r in effect['rows']:
                        self.screen.blit(flash_surface, (self.GRID_X, self.GRID_Y + r * self.CELL_SIZE))
                
                elif effect['type'] == 'combo_text':
                    # Render floating combo text
                    alpha = 255
                    if effect['timer'] < 0.25: # Fade out at the end
                        alpha = int(255 * (effect['timer'] / 0.25))
                    
                    font_surface = self.font_large.render(effect['text'], True, effect['color'])
                    font_surface.set_alpha(alpha)
                    text_rect = font_surface.get_rect(center=effect['pos'])
                    self.screen.blit(font_surface, text_rect)

                next_effects.append(effect)
        self.animation_effects = next_effects

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

if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Re-initialize pygame for display
    pygame.display.set_caption("Letter Grid")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op

    while not done:
        # Action mapping from keyboard
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Lines: {info['lines_cleared']}, Reward: {reward:.2f}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over! Final Score: {info['score']}, Lines Cleared: {info['lines_cleared']}")
    env.close()