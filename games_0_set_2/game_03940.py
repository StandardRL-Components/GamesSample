
# Generated: 2025-08-28T00:54:41.372272
# Source Brief: brief_03940.md
# Brief Index: 3940

        
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
        "Controls: Arrow keys to move cursor. Space to select a number. Shift to restart the level."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race against the clock to select numbers on the grid that sum up to the target value."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Visuals & Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (50, 65, 80)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_POSITIVE = (60, 220, 180)
        self.COLOR_NEGATIVE = (250, 100, 100)
        self.COLOR_DISABLED = (80, 90, 100)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_TARGET = (100, 150, 255)
        self.COLOR_SUM = (200, 200, 200)

        # Game constants
        self.GRID_ROWS = 4
        self.GRID_COLS = 5
        self.CELL_SIZE = 80
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X = (640 - self.GRID_WIDTH) // 2
        self.GRID_Y = (400 - self.GRID_HEIGHT) // 2 + 20
        self.MAX_STEPS = 900  # 30 seconds at 30 FPS

        # Game state that persists across resets
        self.current_level_target_sum = 100
        
        # Initialize state variables
        self.grid = []
        self.selected_cells = set()
        self.cursor_pos = [0, 0]
        self.running_sum = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.last_space_held = False
        self.move_cooldown = 4 # Frames
        self.move_timer = 0
        self.particles = []
        
        # Initialize state for the first time
        self.reset()
        
        # Run validation
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.running_sum = 0
        self.target_sum = self.current_level_target_sum
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_cells = set()
        self.last_space_held = False
        self.move_timer = 0
        self.particles = []

        self._generate_grid()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        is_solvable = False
        while not is_solvable:
            self.grid = [[0 for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
            positive_sum = 0
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    val = self.np_random.integers(-25, 51)
                    if val == 0: val = 1 # Avoid zero
                    self.grid[r][c] = val
                    if val > 0:
                        positive_sum += val
            if positive_sum >= self.target_sum:
                is_solvable = True

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            # If the game is over, the only action that does anything is restarting
            # but the environment loop should handle calling reset().
            # We just return the final state.
            return (
                self._get_observation(),
                0,
                True,
                False,
                self._get_info()
            )

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Update game logic
        self.steps += 1
        if self.move_timer > 0:
            self.move_timer -= 1
            
        reward += self._handle_input(movement, space_held, shift_held)
        self._update_particles()
        
        terminated, term_reward = self._check_termination(shift_held)
        reward += term_reward

        if terminated:
            self.game_over = True
            if self.win_state:
                # Increase difficulty for the next round
                self.current_level_target_sum += 5
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        
        # --- Handle Movement ---
        if self.move_timer == 0:
            moved = False
            if movement == 1: # Up
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
                moved = True
            elif movement == 2: # Down
                self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
                moved = True
            elif movement == 3: # Left
                self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
                moved = True
            elif movement == 4: # Right
                self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)
                moved = True
            
            if moved:
                self.move_timer = self.move_cooldown

        # --- Handle Selection ---
        is_press = space_held and not self.last_space_held
        if is_press:
            r, c = self.cursor_pos
            if (r, c) not in self.selected_cells:
                # SFX: select_number.wav
                value = self.grid[r][c]
                
                # Calculate reward for getting closer/further
                old_dist = abs(self.target_sum - self.running_sum)
                new_dist = abs(self.target_sum - (self.running_sum + value))
                if new_dist < old_dist:
                    reward += 0.1
                else:
                    reward -= 0.1
                
                self.running_sum += value
                self.score += value
                self.selected_cells.add((r, c))
                
                # Add particle effect
                px = self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                py = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                color = self.COLOR_POSITIVE if value > 0 else self.COLOR_NEGATIVE
                for _ in range(20):
                    self._create_particle(px, py, color)

        self.last_space_held = space_held
        return reward
        
    def _check_termination(self, shift_held):
        reward = 0
        if self.running_sum == self.target_sum:
            # SFX: win_level.wav
            self.win_state = True
            reward += 15.0 # +5 for reaching, +10 for within time
            return True, reward
        if self.steps >= self.MAX_STEPS:
            # SFX: time_out.wav
            return True, -1.0 # Small penalty for timeout
        if shift_held:
            # Manual reset
            return True, -0.5
        return False, reward

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Grid and Numbers ---
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = pygame.Rect(
                    self.GRID_X + c * self.CELL_SIZE,
                    self.GRID_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                is_selected = (r, c) in self.selected_cells
                value = self.grid[r][c]
                
                color = self.COLOR_DISABLED if is_selected else \
                        self.COLOR_POSITIVE if value > 0 else self.COLOR_NEGATIVE
                
                num_text = str(value)
                text_surf = self.font_medium.render(num_text, True, color)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)

        # --- Draw Cursor ---
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_X + cursor_c * self.CELL_SIZE,
            self.GRID_Y + cursor_r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        # Pulsating effect for cursor
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        line_width = 2 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width, border_radius=4)
        
        # --- Draw Particles ---
        for p in self.particles:
            pos = [int(p['pos'][0]), int(p['pos'][1])]
            color_with_alpha = (*p['color'], p['alpha'])
            
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (pos[0] - p['radius'], pos[1] - p['radius']))


    def _render_ui(self):
        # --- Timer Bar ---
        time_ratio = (self.MAX_STEPS - self.steps) / self.MAX_STEPS
        time_bar_width = int(time_ratio * (self.GRID_WIDTH - 4))
        time_bar_rect = pygame.Rect(self.GRID_X + 2, 15, time_bar_width, 10)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, time_bar_rect, border_radius=5)
        
        # --- Target Sum ---
        target_text = self.font_small.render("TARGET", True, self.COLOR_TARGET)
        self.screen.blit(target_text, (20, 10))
        target_val_text = self.font_large.render(str(self.target_sum), True, self.COLOR_TARGET)
        self.screen.blit(target_val_text, (20, 35))
        
        # --- Running Sum ---
        sum_text = self.font_small.render("CURRENT SUM", True, self.COLOR_SUM)
        sum_text_rect = sum_text.get_rect(topright=(620, 10))
        self.screen.blit(sum_text, sum_text_rect)
        sum_val_text = self.font_large.render(str(self.running_sum), True, self.COLOR_SUM)
        sum_val_rect = sum_val_text.get_rect(topright=(620, 35))
        self.screen.blit(sum_val_text, sum_val_rect)

        # --- Game Over / Win Message ---
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "LEVEL COMPLETE!" if self.win_state else "TIME'S UP!"
            color = self.COLOR_POSITIVE if self.win_state else self.COLOR_NEGATIVE
            
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(320, 200))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "running_sum": self.running_sum,
            "target_sum": self.target_sum
        }
    
    def _create_particle(self, x, y, color):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        radius = self.np_random.uniform(3, 8)
        self.particles.append({
            'pos': [x, y],
            'vel': vel,
            'radius': radius,
            'alpha': 255,
            'color': color
        })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['alpha'] -= 10 # Fade out speed
            p['radius'] -= 0.1 # Shrink speed
            if p['alpha'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    # For headless execution, you might need this on some systems
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    env.validate_implementation()
    
    # To actually see the game, you'd need a different setup
    # For example, by creating a window and blitting the env.screen surface.
    # This block is for testing the gym interface.
    print("\n--- Testing a short episode ---")
    obs, info = env.reset()
    print(f"Initial state: {info}")
    terminated = False
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample() # Random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i+1) % 20 == 0:
            print(f"Step {i+1}: Info={info}, Reward={reward:.2f}")
        if terminated:
            print(f"Episode terminated at step {i+1}.")
            break
    
    print(f"Total reward after 100 steps or termination: {total_reward:.2f}")
    env.close()