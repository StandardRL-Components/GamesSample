
# Generated: 2025-08-28T04:31:24.031447
# Source Brief: brief_05278.md
# Brief Index: 5278

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a square. "
        "Memorize the pattern, then replicate it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Memorize and replicate increasingly complex color "
        "patterns on the grid to test your spatial memory."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 7
        self.GRID_X_OFFSET, self.GRID_Y_OFFSET = 80, 80
        self.CELL_SIZE = 40
        self.MAX_STEPS = 2000
        self.WIN_SCORE = 15
        self.LOSE_SCORE = 3

        # Colors
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (60, 65, 75)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SUCCESS = (60, 220, 120)
        self.COLOR_FAIL = (220, 60, 60)
        self.PATTERN_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]

        # Gymnasium spaces
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
        self.font_large = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.np_random = None
        self.reset()

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        
        self.successful_matches = 0
        self.failed_matches = 0
        self.pattern_length = 3
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.grid_state = np.full((self.GRID_ROWS, self.GRID_COLS), -1, dtype=int)
        
        self.target_pattern = []
        self.player_sequence_indices = []
        self.current_attempt_failed = False
        self.particles = []

        self._generate_new_round()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # Handle timed phase transitions
        if self.phase_timer > 0:
            self.phase_timer -= 1
            if self.phase_timer == 0:
                if self.game_state == 'SHOWING_PATTERN':
                    self.game_state = 'AWAITING_INPUT'
                elif self.game_state == 'FEEDBACK':
                    if not self.game_over:
                        self._generate_new_round()
            # No other actions processed during timed phases
            return self._get_observation(), 0, False, False, self._get_info()

        # Unpack action
        movement, space_press, _ = action
        space_pressed = space_press == 1

        # Handle player input only during the input phase
        if self.game_state == 'AWAITING_INPUT':
            # --- Movement ---
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

            # --- Selection ---
            if space_pressed:
                current_step_index = len(self.player_sequence_indices)
                cx, cy = self.cursor_pos
                
                # Check if this square has already been selected in this attempt
                if any(self.target_pattern[i][0] == cx and self.target_pattern[i][1] == cy for i in self.player_sequence_indices):
                    # Penalize selecting an already-selected square, but don't fail the attempt
                    reward -= 0.1
                else:
                    # Find if the selected square is in the target pattern at all
                    target_index = -1
                    for i, (px, py, _) in enumerate(self.target_pattern):
                        if px == cx and py == cy:
                            target_index = i
                            break
                    
                    if target_index != -1:
                        # Correct square selected
                        self.player_sequence_indices.append(target_index)
                        self.grid_state[cy, cx] = self.target_pattern[target_index][2]
                        reward += 1
                        # sound: correct_click.wav
                        self._create_particles(cx, cy, self.PATTERN_COLORS[self.grid_state[cy, cx]])
                    else:
                        # Incorrect square selected (not in pattern at all)
                        self.current_attempt_failed = True
                        reward -= 1
                        # sound: wrong_click.wav
                        self._create_particles(cx, cy, self.COLOR_FAIL, is_error=True)

                # --- Check if pattern attempt is over ---
                is_attempt_over = len(self.player_sequence_indices) == self.pattern_length or self.current_attempt_failed

                if is_attempt_over:
                    # Check if sequence order is correct
                    is_sequence_correct = self.player_sequence_indices == sorted(self.player_sequence_indices)
                    
                    if not self.current_attempt_failed and is_sequence_correct:
                        # SUCCESS
                        self.successful_matches += 1
                        self.score += 10
                        reward += 10
                        # sound: pattern_success.wav
                        if self.successful_matches >= self.WIN_SCORE:
                            self.game_over = True
                            self.win_condition = True
                            reward += 100
                        elif self.successful_matches > 0 and self.successful_matches % 3 == 0:
                            self.pattern_length = min(self.pattern_length + 1, self.GRID_COLS * self.GRID_ROWS)
                    else:
                        # FAILURE
                        self.failed_matches += 1
                        self.score -= 5
                        reward -= 5
                        # sound: pattern_fail.wav
                        if self.failed_matches >= self.LOSE_SCORE:
                            self.game_over = True
                            self.win_condition = False
                            reward -= 100
                    
                    self.game_state = 'FEEDBACK'
                    self.phase_timer = 45  # 1.5 seconds at 30fps

        # Update particles
        self._update_particles()
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_new_round(self):
        self.game_state = 'SHOWING_PATTERN'
        self.phase_timer = 60 + self.pattern_length * 10
        self.player_sequence_indices = []
        self.current_attempt_failed = False
        self.grid_state.fill(-1)
        self.target_pattern = self._generate_pattern()

    def _generate_pattern(self):
        pattern = []
        available_positions = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        
        for _ in range(self.pattern_length):
            if not available_positions: break
            pos_idx = self.np_random.integers(0, len(available_positions))
            pos = available_positions.pop(pos_idx)
            color_idx = self.np_random.integers(0, len(self.PATTERN_COLORS))
            pattern.append((pos[0], pos[1], color_idx))
        
        return sorted(pattern, key=lambda p: (p[1], p[0])) # Sort for consistent display

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] * 0.95)

    def _create_particles(self, grid_x, grid_y, color, is_error=False):
        center_x = self.GRID_X_OFFSET + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        center_y = self.GRID_Y_OFFSET + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        
        for _ in range(15):
            if is_error:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 4)
                self.particles.append({
                    'x': center_x, 'y': center_y,
                    'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                    'radius': self.np_random.uniform(3, 6), 'life': 20, 'color': color
                })
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                self.particles.append({
                    'x': center_x, 'y': center_y,
                    'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                    'radius': self.np_random.uniform(5, 10), 'life': 25, 'color': color
                })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_COLS * self.CELL_SIZE, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_ROWS * self.CELL_SIZE))

        # Draw pattern/grid state
        if self.game_state == 'SHOWING_PATTERN':
            for x, y, color_idx in self.target_pattern:
                self._draw_cell(x, y, self.PATTERN_COLORS[color_idx])
        else: # AWAITING_INPUT or FEEDBACK
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    if self.grid_state[r, c] != -1:
                        self._draw_cell(c, r, self.PATTERN_COLORS[self.grid_state[r, c]])
            if self.game_state == 'FEEDBACK' and self.current_attempt_failed:
                # Show correct pattern on failure
                for x, y, color_idx in self.target_pattern:
                    is_player_choice = any(self.target_pattern[i][0] == x and self.target_pattern[i][1] == y for i in self.player_sequence_indices)
                    if not is_player_choice:
                        self._draw_cell(x, y, self.PATTERN_COLORS[color_idx], alpha=100)

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['radius']), p['color'] + (int(255 * (p['life'] / 25.0)),))

        # Draw cursor if not showing pattern
        if self.game_state == 'AWAITING_INPUT':
            self._draw_cursor()

    def _draw_cell(self, x, y, color, alpha=255):
        rect = pygame.Rect(
            self.GRID_X_OFFSET + x * self.CELL_SIZE + 1,
            self.GRID_Y_OFFSET + y * self.CELL_SIZE + 1,
            self.CELL_SIZE - 2,
            self.CELL_SIZE - 2
        )
        if alpha < 255:
            s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            s.fill(color + (alpha,))
            self.screen.blit(s, rect.topleft)
        else:
            pygame.draw.rect(self.screen, color, rect)

    def _draw_cursor(self):
        cx, cy = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_X_OFFSET + cx * self.CELL_SIZE,
            self.GRID_Y_OFFSET + cy * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        # Pulsing effect
        alpha = 128 + 127 * math.sin(self.steps * 0.2)
        color = (255, 255, 255)
        
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, color + (alpha,), (0, 0, self.CELL_SIZE, self.CELL_SIZE), border_radius=4)
        pygame.draw.rect(s, color + (alpha,), (2, 2, self.CELL_SIZE-4, self.CELL_SIZE-4), width=2, border_radius=4)
        self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        # Draw pattern display above grid
        pattern_display_y = 40
        total_width = self.pattern_length * 25
        start_x = self.WIDTH / 2 - total_width / 2
        for i, (_, _, color_idx) in enumerate(self.target_pattern):
            rect = pygame.Rect(start_x + i * 25, pattern_display_y, 20, 20)
            is_found = i in self.player_sequence_indices
            
            if self.game_state == 'SHOWING_PATTERN' or self.game_state == 'FEEDBACK':
                pygame.draw.rect(self.screen, self.PATTERN_COLORS[color_idx], rect, border_radius=4)
            else: # AWAITING_INPUT
                if is_found:
                    pygame.draw.rect(self.screen, self.PATTERN_COLORS[color_idx], rect, border_radius=4)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, border_radius=4)
        
        # Draw success/fail counts
        success_text = self.font_small.render(f"SUCCESS: {self.successful_matches}/{self.WIN_SCORE}", True, self.COLOR_SUCCESS)
        self.screen.blit(success_text, (10, 10))
        fail_text = self.font_small.render(f"FAILS: {self.failed_matches}/{self.LOSE_SCORE}", True, self.COLOR_FAIL)
        self.screen.blit(fail_text, (self.WIDTH - fail_text.get_width() - 10, 10))

        # Draw phase text
        phase_map = {
            'SHOWING_PATTERN': 'MEMORIZE',
            'AWAITING_INPUT': 'YOUR TURN',
            'FEEDBACK': 'RESULT'
        }
        if not self.game_over:
            phase_text = self.font_small.render(phase_map.get(self.game_state, ''), True, self.COLOR_TEXT)
            self.screen.blit(phase_text, (self.WIDTH/2 - phase_text.get_width()/2, 10))

        # Game Over / Win Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_condition:
                end_text = self.font_large.render("YOU WIN!", True, self.COLOR_SUCCESS)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_FAIL)
            
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "successful_matches": self.successful_matches,
            "failed_matches": self.failed_matches,
            "pattern_length": self.pattern_length,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To run the game manually
    # Use arrow keys for movement, space to select
    # The game only advances on a key press
    
    print(GameEnv.user_guide)
    
    running = True
    while running:
        action = [0, 0, 0] # Default no-op
        
        # This is just for human play, an agent would call env.step(action) directly
        event_occurred = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                event_occurred = True
            if event.type == pygame.KEYDOWN:
                event_occurred = True
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if event_occurred:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Create a display surface and show the rendered observation
            display_surf = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            # The observation is (H, W, C), but pygame blit needs (W, H, C)
            # surfarray.make_surface expects (W,H) array, so we need to transpose
            frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_surf.blit(frame_surface, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over. Final Info: {info}")
                # Wait for a moment before auto-resetting or closing
                pygame.time.wait(2000)
                obs, info = env.reset()

    env.close()