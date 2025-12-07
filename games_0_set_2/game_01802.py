
# Generated: 2025-08-27T18:21:00.271120
# Source Brief: brief_01802.md
# Brief Index: 1802

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrows to move cursor. Hold Shift to cycle color. Hold Space to paint."
    )

    # Short, user-facing description of the game
    game_description = (
        "Recreate the target image on the canvas by painting pixels. Match it 100% before the 120-turn timer runs out!"
    )

    # Frames only advance when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.MAX_STEPS = 120
        self.CELL_SIZE = 32
        
        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID_BG = (30, 40, 65)
        self.COLOR_GRID_LINE = (50, 65, 95)
        self.COLOR_TEXT = (220, 230, 255)
        self.COLOR_TEXT_ACCENT = (100, 255, 255)
        self.COLOR_CURSOR = (255, 255, 100)
        self.PALETTE = [
            (0, 0, 0),         # 0: Black
            (255, 255, 255),   # 1: White
            (255, 216, 0),     # 2: Yellow
            (255, 0, 77),      # 3: Red
            (0, 135, 81),      # 4: Green
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
        self.font_main = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 22)
        
        # Initialize state variables
        self.canvas = None
        self.target_image = None
        self.cursor_pos = None
        self.selected_color_idx = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.completed_rows = None
        self.completed_cols = None
        self.particles = None
        self.rng = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.rng = np.random.default_rng(seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_color_idx = 0
        
        self.canvas = self.rng.choice([0], size=(self.GRID_SIZE, self.GRID_SIZE))
        self.target_image = self._generate_target_image()
        
        self.completed_rows = set()
        self.completed_cols = set()
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # 1. Cycle color if Shift is held
        if shift_held:
            self.selected_color_idx = (self.selected_color_idx + 1) % len(self.PALETTE)
            # Small penalty for cycling to discourage spamming
            reward -= 0.01

        # 2. Move cursor
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

        # 3. Paint pixel if Space is held
        if space_held:
            x, y = self.cursor_pos
            previous_color_idx = self.canvas[y, x]
            target_color_idx = self.target_image[y, x]
            
            if previous_color_idx != self.selected_color_idx:
                # Only apply changes and rewards if the color is different
                was_correct = (previous_color_idx == target_color_idx)
                
                self.canvas[y, x] = self.selected_color_idx
                
                is_now_correct = (self.selected_color_idx == target_color_idx)
                
                if not was_correct and is_now_correct:
                    reward += 1.0  # Corrected a wrong pixel
                elif was_correct and not is_now_correct:
                    reward -= 0.1 # Messed up a correct pixel
                else: # not was_correct and not is_now_correct
                    reward -= 0.1 # Changed a wrong pixel to another wrong pixel
                
                # Add particle effect for feedback
                # sfx: paint_pixel.wav
                self._add_particle(x, y, self.PALETTE[self.selected_color_idx])

                # Check for row/column completion bonus
                row_complete = np.array_equal(self.canvas[y, :], self.target_image[y, :])
                if row_complete and y not in self.completed_rows:
                    reward += 10.0
                    self.completed_rows.add(y)
                    # sfx: complete_row.wav
                
                col_complete = np.array_equal(self.canvas[:, x], self.target_image[:, x])
                if col_complete and x not in self.completed_cols:
                    reward += 10.0
                    self.completed_cols.add(x)
                    # sfx: complete_col.wav

        self.steps += 1
        self.score += reward
        
        # Check for termination
        is_complete = np.array_equal(self.canvas, self.target_image)
        is_timeout = self.steps >= self.MAX_STEPS
        terminated = is_complete or is_timeout
        
        if terminated:
            self.game_over = True
            if is_complete:
                reward += 100.0  # Victory bonus
                # sfx: victory.wav
            else: # Timeout
                reward -= 50.0   # Timeout penalty
                # sfx: game_over.wav

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game_area()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        completion = np.sum(self.canvas == self.target_image) / self.canvas.size
        return {
            "score": self.score,
            "steps": self.steps,
            "completion": completion,
        }

    def _generate_target_image(self):
        # Create a deterministic smiley face
        img = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        img[2:6, 1] = 2
        img[2:6, 6] = 2
        img[1, 2:6] = 2
        img[6, 2:6] = 2
        img[2, 2] = 0
        img[2, 5] = 0
        img[5, 2] = 0
        img[5, 5] = 0
        img[3:5, 3:5] = 1
        img[4, 2] = 2
        img[4, 5] = 2
        return img

    def _render_game_area(self):
        grid_width = self.GRID_SIZE * self.CELL_SIZE
        total_width = grid_width * 2 + 80
        start_x = (self.WIDTH - total_width) // 2
        top_y = 40
        
        # Render Target Grid
        target_x = start_x
        self._render_text("TARGET", (target_x + grid_width // 2, top_y - 15), self.font_title)
        self._render_grid(self.target_image, (target_x, top_y))

        # Render Player Canvas
        canvas_x = target_x + grid_width + 80
        self._render_text("CANVAS", (canvas_x + grid_width // 2, top_y - 15), self.font_title)
        self._render_grid(self.canvas, (canvas_x, top_y))
        
        # Render particles on player canvas
        self._update_and_render_particles((canvas_x, top_y))

        # Render Cursor
        cursor_screen_x = canvas_x + self.cursor_pos[0] * self.CELL_SIZE
        cursor_screen_y = top_y + self.cursor_pos[1] * self.CELL_SIZE
        
        # Pulsing effect for cursor
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
        line_width = int(2 + pulse * 2)
        pygame.draw.rect(
            self.screen, 
            self.COLOR_CURSOR, 
            (cursor_screen_x, cursor_screen_y, self.CELL_SIZE, self.CELL_SIZE), 
            line_width,
            border_radius=4
        )

    def _render_grid(self, grid_data, top_left):
        grid_pixel_size = self.GRID_SIZE * self.CELL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (*top_left, grid_pixel_size, grid_pixel_size), border_radius=5)
        
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_idx = grid_data[y, x]
                color = self.PALETTE[color_idx]
                rect = pygame.Rect(
                    top_left[0] + x * self.CELL_SIZE,
                    top_left[1] + y * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)

        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (top_left[0] + i * self.CELL_SIZE, top_left[1])
            end_pos = (top_left[0] + i * self.CELL_SIZE, top_left[1] + grid_pixel_size)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos)
            # Horizontal lines
            start_pos = (top_left[0], top_left[1] + i * self.CELL_SIZE)
            end_pos = (top_left[0] + grid_pixel_size, top_left[1] + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos)

    def _render_ui(self):
        ui_y = self.HEIGHT - 40

        # Timer
        time_left = self.MAX_STEPS - self.steps
        self._render_text(f"TIME: {time_left}", (self.WIDTH * 0.15, ui_y), self.font_main, self.COLOR_TEXT_ACCENT)
        
        # Completion Percentage
        completion = np.sum(self.canvas == self.target_image) / self.canvas.size * 100
        self._render_text(f"MATCH: {completion:.1f}%", (self.WIDTH * 0.85, ui_y), self.font_main, self.COLOR_TEXT_ACCENT)
        
        # Palette
        palette_width = len(self.PALETTE) * 30
        start_x = (self.WIDTH - palette_width) // 2
        self._render_text("COLOR", (start_x - 40, ui_y), self.font_main)
        
        for i, color in enumerate(self.PALETTE):
            rect = pygame.Rect(start_x + i * 30, ui_y - 12, 24, 24)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            if i == self.selected_color_idx:
                pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 2, border_radius=4)

    def _render_text(self, text, position, font, color=None):
        color = color or self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=position)
        self.screen.blit(text_surface, text_rect)

    def _add_particle(self, grid_x, grid_y, color):
        center_x = grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        center_y = grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        self.particles.append({
            "pos": [center_x, center_y],
            "radius": 1,
            "max_radius": self.CELL_SIZE * 0.75,
            "life": 15,
            "max_life": 15,
            "color": color
        })

    def _update_and_render_particles(self, grid_top_left):
        active_particles = []
        for p in self.particles:
            p['life'] -= 1
            if p['life'] > 0:
                progress = (p['max_life'] - p['life']) / p['max_life']
                p['radius'] = progress * p['max_radius']
                
                alpha = int(255 * (p['life'] / p['max_life']))
                
                pos_x = int(grid_top_left[0] + p['pos'][0])
                pos_y = int(grid_top_left[1] + p['pos'][1])
                
                # Use gfxdraw for anti-aliased circles
                pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, int(p['radius']), (*p['color'], alpha))
                pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, int(p['radius']), (*p['color'], alpha))
                
                active_particles.append(p)
        self.particles = active_particles

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To display the game, we need a different setup
    pygame.display.set_caption("Pixel Painter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    # Game loop for human play
    running = True
    while running:
        action = [0, 0, 0] # no-op, release space, release shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
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
        
        # Blit the observation to the display screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Completion: {info['completion']*100:.1f}%")
            pygame.time.wait(2000)
            env.reset()

    env.close()