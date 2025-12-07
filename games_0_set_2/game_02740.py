import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to paint. Shift to cycle color."
    )

    game_description = (
        "Recreate the target image by painting pixels before time runs out."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 32, 20
        self.MAX_STEPS = 600
        self.WIN_THRESHOLD = 0.95

        # --- Colors (ENDESGA 32 Palette) ---
        self.COLOR_BG = (34, 32, 52)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (52, 52, 78)
        self.COLOR_CURSOR = (255, 241, 232)
        self.PALETTE = [
            (172, 50, 50), (217, 87, 99), (217, 160, 102), (223, 248, 209),
            (126, 196, 193), (78, 128, 156), (92, 92, 117), (60, 41, 78)
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 20)
        self.font_gameover = pygame.font.Font(None, 60)

        # --- Layout Calculation ---
        self.TARGET_HEIGHT = 80
        self.PALETTE_HEIGHT = 40
        self.UI_HEIGHT = 30
        self.MARGIN = 10

        self.target_rect = pygame.Rect(self.MARGIN, self.MARGIN, self.WIDTH - 2 * self.MARGIN, self.TARGET_HEIGHT)
        
        canvas_top = self.target_rect.bottom + self.UI_HEIGHT + self.MARGIN
        canvas_height = self.HEIGHT - canvas_top - self.PALETTE_HEIGHT - self.MARGIN
        self.canvas_rect = pygame.Rect(self.MARGIN, canvas_top, self.WIDTH - 2 * self.MARGIN, canvas_height)
        
        self.pixel_size = min(self.canvas_rect.width // self.GRID_COLS, self.canvas_rect.height // self.GRID_ROWS)
        self.canvas_render_rect = pygame.Rect(0, 0, self.GRID_COLS * self.pixel_size, self.GRID_ROWS * self.pixel_size)
        self.canvas_render_rect.center = self.canvas_rect.center

        self.target_pixel_size = min(self.target_rect.width // self.GRID_COLS, self.target_rect.height // self.GRID_ROWS)
        self.target_render_rect = pygame.Rect(0, 0, self.GRID_COLS * self.target_pixel_size, self.GRID_ROWS * self.target_pixel_size)
        self.target_render_rect.center = self.target_rect.center

        # --- State Variables ---
        self.steps_remaining = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.cursor_pos = [0, 0]
        self.selected_color_index = 0
        self.target_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.uint8)
        self.canvas_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.uint8)
        self.similarity = 0.0
        self.particles = []
        self.shift_pressed_last_frame = False
        
        self.palette_map = np.array(self.PALETTE, dtype=np.uint8)

        # Initialize state
        # self.reset() is called here, which can cause the error if not fixed
        
        # Self-check is commented out during initial loading
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps_remaining = self.MAX_STEPS
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_color_index = 0
        self.particles = []
        self.shift_pressed_last_frame = False
        
        self._generate_target()
        self.canvas_grid = np.full((self.GRID_ROWS, self.GRID_COLS), 7, dtype=np.uint8) # Blank slate

        self.similarity = self._calculate_similarity()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

            # --- Action: Cycle Color (Shift) ---
            # Only triggers on the frame the key is pressed down
            if shift_held and not self.shift_pressed_last_frame:
                self.selected_color_index = (self.selected_color_index + 1) % len(self.PALETTE)
                # sfx: color_cycle.wav
            self.shift_pressed_last_frame = shift_held

            # --- Action: Move Cursor ---
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

            # --- Action: Paint Pixel (Space) ---
            if space_held:
                px, py = self.cursor_pos
                old_color_idx = self.canvas_grid[py, px]
                new_color_idx = self.selected_color_index
                
                if old_color_idx != new_color_idx:
                    target_color_idx = self.target_grid[py, px]
                    self.canvas_grid[py, px] = new_color_idx
                    
                    old_match = (old_color_idx == target_color_idx)
                    new_match = (new_color_idx == target_color_idx)
                    
                    if new_match and not old_match:
                        reward += 0.1
                        # sfx: paint_correct.wav
                        self._create_particles(px, py, self.PALETTE[new_color_idx], 'correct')
                    elif not new_match:
                        reward -= 0.01
                        # sfx: paint_wrong.wav
                        self._create_particles(px, py, self.PALETTE[new_color_idx], 'wrong')
                    else: # Painting over a correct pixel with the same correct color
                        # sfx: paint_noop.wav
                        pass
                    
                    self.similarity = self._calculate_similarity()

            self.steps_remaining -= 1
        
        self.score += reward
        
        # --- Termination Check ---
        terminated = False
        if self.similarity >= self.WIN_THRESHOLD:
            if not self.game_over: # First frame of winning
                reward += 100
                self.score += 100
                self.game_over_message = "SUCCESS!"
            terminated = True
        elif self.steps_remaining <= 0:
            if not self.game_over: # First frame of losing
                reward -= 10
                self.score -= 10
                self.game_over_message = "TIME UP"
            terminated = True
        
        self.game_over = terminated
        
        self._update_particles()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid(self.target_grid, self.target_render_rect, self.target_pixel_size)
        self._render_grid(self.canvas_grid, self.canvas_render_rect, self.pixel_size)
        
        self._render_cursor()
        self._render_palette()
        self._render_ui()
        self._render_particles()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps_remaining": self.steps_remaining,
            "similarity": self.similarity
        }

    def _generate_target(self):
        # Generate a symmetrical pattern for the target image
        self.target_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.uint8)
        half_cols = math.ceil(self.GRID_COLS / 2)
        half_rows = math.ceil(self.GRID_ROWS / 2)
        
        quadrant = self.np_random.integers(0, len(self.PALETTE), size=(half_rows, half_cols), dtype=np.uint8)
        
        self.target_grid[0:half_rows, 0:half_cols] = quadrant
        self.target_grid[0:half_rows, half_cols:] = np.fliplr(quadrant)[:,:self.GRID_COLS - half_cols]
        self.target_grid[half_rows:, :] = np.flipud(self.target_grid[0:half_rows, :])[:self.GRID_ROWS - half_rows, :]

    def _calculate_similarity(self):
        matches = np.sum(self.canvas_grid == self.target_grid)
        total_pixels = self.GRID_COLS * self.GRID_ROWS
        return matches / total_pixels

    def _render_grid(self, grid_data, rect, pixel_size):
        # Efficiently draw grid using a temporary surface and numpy
        grid_surface = pygame.Surface((self.GRID_COLS, self.GRID_ROWS))
        pixel_view = pygame.surfarray.pixels3d(grid_surface)
        
        # FIX: Transpose the color-mapped grid data.
        # Numpy arrays are (row, col) / (height, width) while Pygame surfaces are (width, height).
        # self.palette_map[grid_data] has shape (GRID_ROWS, GRID_COLS, 3) -> (20, 32, 3)
        # pixel_view expects shape (GRID_COLS, GRID_ROWS, 3) -> (32, 20, 3)
        # Transposing (1, 0, 2) swaps the first two axes to match.
        pixel_view[:] = np.transpose(self.palette_map[grid_data], (1, 0, 2))
        del pixel_view
        
        scaled_surface = pygame.transform.scale(grid_surface, (rect.width, rect.height))
        self.screen.blit(scaled_surface, rect.topleft)
        
        # Draw a border
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, rect, 2, border_radius=3)
        
    def _render_cursor(self):
        cursor_x = self.canvas_render_rect.left + self.cursor_pos[0] * self.pixel_size
        cursor_y = self.canvas_render_rect.top + self.cursor_pos[1] * self.pixel_size
        
        # Pulsing effect for visibility
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        alpha = 100 + pulse * 100
        
        cursor_surface = pygame.Surface((self.pixel_size, self.pixel_size), pygame.SRCALPHA)
        cursor_surface.fill((*self.COLOR_CURSOR, alpha))
        
        self.screen.blit(cursor_surface, (cursor_x, cursor_y))
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, self.pixel_size, self.pixel_size), 1)

    def _render_palette(self):
        swatch_size = 28
        total_width = len(self.PALETTE) * (swatch_size + 5) - 5
        start_x = (self.WIDTH - total_width) // 2
        y_pos = self.HEIGHT - self.PALETTE_HEIGHT

        for i, color in enumerate(self.PALETTE):
            rect = pygame.Rect(start_x + i * (swatch_size + 5), y_pos, swatch_size, swatch_size)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            if i == self.selected_color_index:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=3)

    def _render_ui(self):
        # Target Label
        target_label = self.font_title.render("TARGET", True, self.COLOR_UI_TEXT)
        self.screen.blit(target_label, (self.target_rect.left, self.target_rect.top - 20))
        
        # Canvas Label
        canvas_label = self.font_title.render("CANVAS", True, self.COLOR_UI_TEXT)
        self.screen.blit(canvas_label, (self.canvas_render_rect.left, self.canvas_render_rect.top - 20))

        # UI Bar
        ui_bar_rect = pygame.Rect(self.target_rect.left, self.target_rect.bottom + 5, self.target_rect.width, self.UI_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_bar_rect, border_radius=3)
        
        # Time
        time_text = f"TIME: {self.steps_remaining}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (ui_bar_rect.left + 10, ui_bar_rect.centery - time_surf.get_height() // 2))

        # Similarity
        sim_text = f"MATCH: {self.similarity:.1%}"
        sim_surf = self.font_main.render(sim_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(sim_surf, (ui_bar_rect.right - sim_surf.get_width() - 10, ui_bar_rect.centery - sim_surf.get_height() // 2))

    def _create_particles(self, grid_x, grid_y, color, p_type):
        screen_x = self.canvas_render_rect.left + (grid_x + 0.5) * self.pixel_size
        screen_y = self.canvas_render_rect.top + (grid_y + 0.5) * self.pixel_size
        
        num_particles = 10 if p_type == 'correct' else 5
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) if p_type == 'correct' else random.uniform(0.5, 1.5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(15, 30)
            
            p_color = (255, 50, 50) if p_type == 'wrong' else color
            
            self.particles.append({'pos': [screen_x, screen_y], 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': p_color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(life_ratio * 3)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, (*p['color'], int(life_ratio * 255)))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
        self.screen.blit(overlay, (0, 0))
        
        text_surf = self.font_gameover.render(self.game_over_message, True, self.COLOR_CURSOR)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play ---
    # Setup Pygame window for human interaction
    # Re-enable the default video driver for display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()

    while not done:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering to screen ---
        # The observation is already a rendered frame, just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        clock.tick(30) # Limit frame rate for playability

    print(f"Game Over. Final Score: {info['score']:.2f}, Similarity: {info['similarity']:.1%}")
    env.close()