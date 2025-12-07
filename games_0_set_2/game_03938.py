
# Generated: 2025-08-28T00:53:47.094245
# Source Brief: brief_03938.md
# Brief Index: 3938

        
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
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle colors and Space to paint."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate a target image on a pixel grid using limited paint. Earn points for speed and accuracy."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_STEPS = 300
        self.INITIAL_PAINT = 30
        self.NUM_COLORS = 3

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
        
        # Visuals
        self.FONT_S = pygame.font.SysFont("Consolas", 14)
        self.FONT_M = pygame.font.SysFont("Consolas", 18, bold=True)
        self.FONT_L = pygame.font.SysFont("Consolas", 24, bold=True)

        self.COLOR_BG = (32, 32, 48)
        self.COLOR_GRID_BG = (48, 48, 64)
        self.COLOR_GRID_LINE = (80, 80, 96)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        
        self.PAINT_COLORS = [
            (255, 64, 64),   # Red
            (64, 255, 64),   # Green
            (64, 64, 255),   # Blue
        ]
        self.BLANK_COLOR_IDX = -1 # Special index for blank canvas cells

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.selected_color_idx = 0
        self.paint_counts = []
        self.target_image = None
        self.canvas_indices = None
        self.particles = []
        self.completed_rows = set()
        self.completed_cols = set()
        self.np_random = None

        # Initialize state variables
        self.reset()
        
        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.selected_color_idx = 0
        self.paint_counts = [self.INITIAL_PAINT] * self.NUM_COLORS
        self.particles = []
        self.completed_rows = set()
        self.completed_cols = set()
        
        # Generate new target image
        self.target_image = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))
        
        # Reset canvas
        self.canvas_indices = np.full((self.GRID_SIZE, self.GRID_SIZE), self.BLANK_COLOR_IDX, dtype=int)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1
        
        reward = 0.0
        self.steps += 1

        # 1. Handle Movement
        if movement == 1: # Up
            self.cursor_pos[1] -= 1
        elif movement == 2: # Down
            self.cursor_pos[1] += 1
        elif movement == 3: # Left
            self.cursor_pos[0] -= 1
        elif movement == 4: # Right
            self.cursor_pos[0] += 1
        
        # Wrap cursor around grid
        self.cursor_pos[0] %= self.GRID_SIZE
        self.cursor_pos[1] %= self.GRID_SIZE

        # 2. Handle Primary Actions (Cycle > Paint)
        if shift_pressed:
            # SFX: color_cycle.wav
            self.selected_color_idx = (self.selected_color_idx + 1) % self.NUM_COLORS
            reward -= 0.001 # Small penalty for cycling to encourage planning
        
        elif space_pressed:
            x, y = self.cursor_pos
            
            if self.paint_counts[self.selected_color_idx] > 0:
                # Only apply changes and rewards if the pixel is not already the desired color
                if self.canvas_indices[y, x] != self.selected_color_idx:
                    # SFX: paint_splat.wav
                    target_color_idx = self.target_image[y, x]
                    
                    self.canvas_indices[y, x] = self.selected_color_idx
                    self.paint_counts[self.selected_color_idx] -= 1
                    
                    if self.selected_color_idx == target_color_idx:
                        reward += 0.1  # Correct placement
                    else:
                        reward -= 0.01 # Incorrect placement
                    
                    self._create_particles(x, y, self.PAINT_COLORS[self.selected_color_idx])
            else:
                # SFX: error_buzz.wav
                reward -= 0.05 # Penalty for trying to use empty paint

        # 3. Update particles
        self._update_particles()

        # 4. Calculate row/column completion rewards
        reward += self._check_line_completion()
        
        # 5. Check for termination
        terminated = False
        is_complete = np.array_equal(self.canvas_indices, self.target_image)
        paint_ran_out = any(p <= 0 for p in self.paint_counts)
        max_steps_reached = self.steps >= self.MAX_STEPS

        if is_complete:
            # SFX: victory_fanfare.wav
            reward += 100
            terminated = True
            self.game_over = True
        elif paint_ran_out and not is_complete:
            # SFX: failure_trombone.wav
            reward -= 50
            terminated = True
            self.game_over = True
        elif max_steps_reached:
            terminated = True
            self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_line_completion(self):
        reward = 0
        
        # Check rows
        newly_completed_rows = set()
        for i in range(self.GRID_SIZE):
            if i not in self.completed_rows:
                if np.array_equal(self.canvas_indices[i, :], self.target_image[i, :]):
                    newly_completed_rows.add(i)
        
        if newly_completed_rows:
            # SFX: bonus_chime.wav
            reward += len(newly_completed_rows) * 5
            self.completed_rows.update(newly_completed_rows)
            
        # Check columns
        newly_completed_cols = set()
        for j in range(self.GRID_SIZE):
            if j not in self.completed_cols:
                if np.array_equal(self.canvas_indices[:, j], self.target_image[:, j]):
                    newly_completed_cols.add(j)
        
        if newly_completed_cols:
            # SFX: bonus_chime.wav
            reward += len(newly_completed_cols) * 5
            self.completed_cols.update(newly_completed_cols)
            
        return reward

    def _create_particles(self, grid_x, grid_y, color):
        cell_size = 22
        offset_x = self.WIDTH // 2 + 15
        offset_y = (self.HEIGHT - self.GRID_SIZE * cell_size) // 2
        
        center_x = offset_x + grid_x * cell_size + cell_size // 2
        center_y = offset_y + grid_y * cell_size + cell_size // 2
        
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(15, 25),
                'color': color,
                'size': random.randint(2, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_grid(self, surface, grid_data, offset_x, offset_y, cell_size, is_target):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_idx = grid_data[y, x]
                rect = (offset_x + x * cell_size, offset_y + y * cell_size, cell_size, cell_size)
                
                # Draw cell background
                if color_idx == self.BLANK_COLOR_IDX:
                    pygame.draw.rect(surface, self.COLOR_GRID_BG, rect)
                else:
                    pygame.draw.rect(surface, self.PAINT_COLORS[color_idx], rect)

                # Draw grid lines
                pygame.draw.rect(surface, self.COLOR_GRID_LINE, rect, 1)

    def _render_game(self):
        cell_size = 22
        grid_width = grid_height = cell_size * self.GRID_SIZE
        
        # Target grid on the left
        target_offset_x = self.WIDTH // 4 - grid_width // 2
        grid_offset_y = (self.HEIGHT - grid_height) // 2
        self._render_grid(self.screen, self.target_image, target_offset_x, grid_offset_y, cell_size, True)

        # Player canvas on the right
        canvas_offset_x = self.WIDTH * 3 // 4 - grid_width // 2
        self._render_grid(self.screen, self.canvas_indices, canvas_offset_x, grid_offset_y, cell_size, False)

        # Draw cursor
        cursor_x = canvas_offset_x + self.cursor_pos[0] * cell_size
        cursor_y = grid_offset_y + self.cursor_pos[1] * cell_size
        cursor_rect = pygame.Rect(cursor_x, cursor_y, cell_size, cell_size)
        
        # Pulsating effect for cursor
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # Varies between 0 and 1
        line_width = int(2 + pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1]), p['size'], p['size']))

    def _render_ui(self):
        # Titles
        target_title = self.FONT_M.render("TARGET", True, self.COLOR_TEXT)
        canvas_title = self.FONT_M.render("YOUR CANVAS", True, self.COLOR_TEXT)
        self.screen.blit(target_title, (self.WIDTH // 4 - target_title.get_width() // 2, 40))
        self.screen.blit(canvas_title, (self.WIDTH * 3 // 4 - canvas_title.get_width() // 2, 40))

        # Score and Steps
        score_text = self.FONT_L.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        steps_text = self.FONT_S.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        # Paint Palette
        palette_w = 200
        palette_h = 50
        palette_x = (self.WIDTH - palette_w) // 2
        palette_y = self.HEIGHT - palette_h - 15
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (palette_x, palette_y, palette_w, palette_h), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, (palette_x, palette_y, palette_w, palette_h), 1, border_radius=5)
        
        swatch_size = 30
        spacing = (palette_w - self.NUM_COLORS * swatch_size) / (self.NUM_COLORS + 1)
        
        for i in range(self.NUM_COLORS):
            swatch_x = palette_x + spacing * (i + 1) + swatch_size * i
            swatch_y = palette_y + (palette_h - swatch_size) // 2
            
            # Draw color swatch
            pygame.draw.rect(self.screen, self.PAINT_COLORS[i], (swatch_x, swatch_y, swatch_size, swatch_size), border_radius=3)
            
            # Highlight selected color
            if i == self.selected_color_idx:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (swatch_x-2, swatch_y-2, swatch_size+4, swatch_size+4), 2, border_radius=5)

            # Draw paint count
            count_text = self.FONT_M.render(str(self.paint_counts[i]), True, self.COLOR_TEXT)
            self.screen.blit(count_text, (swatch_x + swatch_size // 2 - count_text.get_width() // 2, swatch_y + swatch_size // 2 - count_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "paint_left": self.paint_counts,
            "cursor_pos": self.cursor_pos,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Painter")
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")

    while not terminated:
        # --- Action mapping for human play ---
        movement = 0 # none
        space_pressed = 0
        shift_pressed = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: movement = 1
                elif keys[pygame.K_DOWN]: movement = 2
                elif keys[pygame.K_LEFT]: movement = 3
                elif keys[pygame.K_RIGHT]: movement = 4
                
                if keys[pygame.K_SPACE]: space_pressed = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_pressed = 1

        action = [movement, space_pressed, shift_pressed]

        # Only step if an action is taken (since auto_advance is False)
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.3f}")

        # --- Rendering ---
        # The observation is the rendered frame, so we just blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    print("\nGAME OVER!")
    print(f"Final Score: {info['score']:.2f} in {info['steps']} steps.")
    
    # Keep the window open for a few seconds to see the final state
    pygame.time.wait(3000)
    env.close()