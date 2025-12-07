import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:36:26.187829
# Source Brief: brief_00511.md
# Brief Index: 511
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A real-time puzzle game where the player merges falling numbers on a grid
    to create a 1024 tile before the timer runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = "Merge falling numbers on a grid to create a 1024 tile before the timer runs out."
    user_guide = "Use the ← and → arrow keys to move the falling number tile."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 8, 5
        self.CELL_SIZE = 60
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) + 10

        self.FPS = 30
        self.INITIAL_TIME_SECONDS = 60
        self.MAX_STEPS = 1000
        self.FALL_SPEED = 5 # Pixels per frame
        self.MOVE_COOLDOWN_FRAMES = 4
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Visuals ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TIMER = (70, 180, 255)
        self.COLOR_TIMER_BG = (50, 60, 70)
        self.COLOR_SCORE = (230, 230, 230)
        self.COLOR_GAMEOVER_TEXT = (255, 255, 255)
        self.VALUE_COLORS = {
            2: (120, 180, 255), 4: (120, 255, 180), 8: (255, 255, 120),
            16: (255, 180, 120), 32: (255, 120, 120), 64: (220, 120, 255),
            128: (255, 120, 220), 256: (120, 220, 255), 512: (120, 255, 120),
            1024: (255, 220, 80)
        }
        self.COLOR_1024_GLOW = (255, 255, 255)

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.tile_font = pygame.font.SysFont("Arial", 28, bold=True)
        self.ui_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.game_over_font = pygame.font.SysFont("Arial", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.falling_piece = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.termination_reason = ""
        self.initial_time_frames = 0
        self.time_left_frames = 0
        self.particles = []
        self.move_cooldown = 0
        self.step_reward = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        self.initial_time_frames = self.INITIAL_TIME_SECONDS * self.FPS
        self.time_left_frames = self.initial_time_frames
        self.particles = []
        self.move_cooldown = 0
        
        self._spawn_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.step_reward = 0

        self._handle_input(action)
        self._update_falling_piece()
        self._update_particles()
        self._update_timer()

        reward = self.step_reward
        terminated = self.game_over
        truncated = False

        if self.steps >= self.MAX_STEPS and not terminated:
            truncated = True
            terminated = True # For compatibility with older APIs that might not check truncated
            self.termination_reason = "max_steps"
            # No specific penalty for max steps, just ends the episode.

        if terminated and not truncated:
            if self.termination_reason == "win":
                reward += 100
            elif self.termination_reason == "time_out":
                reward += -100
            elif self.termination_reason == "grid_full":
                reward += -50
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0] # 0=none, 1=up, 2=down, 3=left, 4=right
        
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
            return
        
        if self.falling_piece is None:
            return

        new_col = self.falling_piece['col']
        if movement == 3: # Left
            new_col = max(0, new_col - 1)
        elif movement == 4: # Right
            new_col = min(self.GRID_COLS - 1, new_col + 1)
        
        if new_col != self.falling_piece['col']:
            self.falling_piece['col'] = new_col
            self.move_cooldown = self.MOVE_COOLDOWN_FRAMES

    def _update_falling_piece(self):
        if self.falling_piece is None:
            return

        self.falling_piece['y'] += self.FALL_SPEED
        
        col = self.falling_piece['col']
        
        # Find the highest occupied row in the current column
        target_row_idx = self.GRID_ROWS
        for r in range(self.GRID_ROWS):
            if self.grid[r, col] > 0:
                target_row_idx = r
                break
        
        landing_y = self.GRID_Y_OFFSET + target_row_idx * self.CELL_SIZE

        if self.falling_piece['y'] + self.CELL_SIZE >= landing_y:
            # --- LANDING LOGIC ---
            piece_to_place = self.falling_piece
            self.falling_piece = None # Stop it from falling further
            
            # Check for merge
            if target_row_idx < self.GRID_ROWS and self.grid[target_row_idx, col] == piece_to_place['value']:
                self.grid[target_row_idx, col] *= 2
                self.step_reward += 1.1 # +0.1 for merge, +1 for doubling
                
                # Visual effect
                px = self.GRID_X_OFFSET + col * self.CELL_SIZE + self.CELL_SIZE // 2
                py = self.GRID_Y_OFFSET + target_row_idx * self.CELL_SIZE + self.CELL_SIZE // 2
                color = self._get_color_for_value(self.grid[target_row_idx, col])
                self._create_merge_particles(px, py, color)

                # Time penalty for merging
                self.time_left_frames -= self.initial_time_frames * 0.01

                # Check for win condition
                if self.grid[target_row_idx, col] == 1024:
                    self.game_over = True
                    self.termination_reason = "win"
            else: # No merge, place on top
                place_row = target_row_idx - 1
                if place_row >= 0:
                    self.grid[place_row, col] = piece_to_place['value']
                else: # Column is full, but this is handled by spawn check
                    pass

            # Spawn next piece if game is not over
            if not self.game_over:
                self._spawn_piece()

    def _update_timer(self):
        if not self.game_over:
            self.time_left_frames -= 1
            if self.time_left_frames <= 0:
                self.time_left_frames = 0
                self.game_over = True
                self.termination_reason = "time_out"

    def _spawn_piece(self):
        valid_cols = [c for c in range(self.GRID_COLS) if self.grid[0, c] == 0]
        if not valid_cols:
            self.game_over = True
            self.termination_reason = "grid_full"
            self.falling_piece = None
            return

        spawn_col = self.np_random.choice(valid_cols)
        value = self.np_random.choice([2, 4])
        
        self.falling_piece = {
            'value': value,
            'col': spawn_col,
            'y': self.GRID_Y_OFFSET - self.CELL_SIZE,
            'color': self._get_color_for_value(value)
        }
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "termination_reason": self.termination_reason}

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=8)

        # Draw placed tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                value = self.grid[r, c]
                if value > 0:
                    self._draw_tile(c, r, value)

        # Draw falling piece
        if self.falling_piece:
            rect = pygame.Rect(
                self.GRID_X_OFFSET + self.falling_piece['col'] * self.CELL_SIZE + 2,
                int(self.falling_piece['y']) + 2,
                self.CELL_SIZE - 4, self.CELL_SIZE - 4
            )
            pygame.draw.rect(self.screen, self.falling_piece['color'], rect, border_radius=6)
            
            text_surf = self.tile_font.render(str(self.falling_piece['value']), True, (255,255,255))
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['x']) - p['size'], int(p['y']) - p['size']))

    def _draw_tile(self, col, row, value):
        rect = pygame.Rect(
            self.GRID_X_OFFSET + col * self.CELL_SIZE + 2,
            self.GRID_Y_OFFSET + row * self.CELL_SIZE + 2,
            self.CELL_SIZE - 4, self.CELL_SIZE - 4
        )
        color = self._get_color_for_value(value)
        
        # Glow for 1024
        if value == 1024:
            glow_size = int(self.CELL_SIZE * 0.8)
            glow_surf = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
            alpha = 100 + 50 * math.sin(pygame.time.get_ticks() * 0.005)
            pygame.gfxdraw.filled_circle(glow_surf, glow_size, glow_size, glow_size, (*self.COLOR_1024_GLOW, int(alpha)))
            self.screen.blit(glow_surf, glow_surf.get_rect(center=rect.center))

        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        text_surf = self.tile_font.render(str(value), True, (255,255,255))
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Draw timer bar
        timer_width = self.SCREEN_WIDTH - 40
        timer_rect_bg = pygame.Rect(20, 20, timer_width, 20)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BG, timer_rect_bg, border_radius=10)
        
        time_ratio = self.time_left_frames / self.initial_time_frames
        current_timer_width = int(timer_width * time_ratio)
        timer_rect_fg = pygame.Rect(20, 20, current_timer_width, 20)
        pygame.draw.rect(self.screen, self.COLOR_TIMER, timer_rect_fg, border_radius=10)

        # Draw score
        score_text = self.ui_font.render(f"Score: {int(self.score)}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (25, 50))

        # Draw game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = ""
            if self.termination_reason == "win":
                msg = "YOU WIN!"
            elif self.termination_reason == "time_out":
                msg = "TIME'S UP!"
            elif self.termination_reason == "grid_full":
                msg = "GRID FULL!"
            
            game_over_surf = self.game_over_font.render(msg, True, self.COLOR_GAMEOVER_TEXT)
            game_over_rect = game_over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_surf, game_over_rect)

    def _get_color_for_value(self, value):
        return self.VALUE_COLORS.get(value, (100, 100, 100))

    def _create_merge_particles(self, x, y, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': random.randint(15, 30), 'max_life': 30,
                'size': random.randint(2, 6),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for self-checking and is not part of the Gym API
        try:
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
            assert isinstance(trunc, bool)
            assert isinstance(info, dict)
            
            print("✓ Implementation validated successfully")
        except AssertionError as e:
            print(f"✗ Implementation validation failed: {e}")


if __name__ == '__main__':
    # --- Manual Play ---
    # This block will not run in the headless test environment
    # but is useful for local testing and debugging.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver for manual play
    
    env = GameEnv(render_mode="rgb_array")
    
    # Use a separate screen for rendering if playing manually
    manual_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("NumberFall 1024")
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Game loop for manual play
    while not (terminated or truncated):
        action = np.array([0, 0, 0]) # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Reason: {info['termination_reason']}")

        # Render the observation to the manual screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        manual_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    env.close()