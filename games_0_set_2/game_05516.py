
# Generated: 2025-08-28T05:15:23.201665
# Source Brief: brief_05516.md
# Brief Index: 5516

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to paint the selected "
        "pixel. Press Shift to cycle through available colors."
    )

    game_description = (
        "A fast-paced pixel art puzzle game. Fill the canvas with color against the clock. "
        "Each stage presents a larger canvas, but the time limit stays the same. Plan your "
        "path to paint every pixel before time runs out!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.W, self.H = 640, 400
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_EMPTY = (30, 35, 40)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_TIME_BAR_BG = (60, 60, 60)
        self.COLOR_TIME_BAR_FG = (100, 200, 255)
        self.COLOR_CURSOR = (255, 255, 255)
        self.PAINT_COLORS = [
            (255, 90, 90),   # Red
            (90, 255, 90),   # Green
            (90, 90, 255),   # Blue
            (255, 255, 90),  # Yellow
            (255, 90, 255),  # Magenta
            (90, 255, 255),  # Cyan
        ]

        # Game Constants
        self.MAX_EPISODE_STEPS = 1800  # 3 stages * 600 steps/stage
        self.STAGES_CONFIG = [
            {"size": 10, "time": 600},
            {"size": 12, "time": 600},
            {"size": 15, "time": 600},
        ]
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 0
        self.canvas = np.array([[]])
        self.cursor_pos = [0, 0]
        self.time_remaining = 0
        self.max_time = 0
        self.pixels_to_fill = 0
        self.pixels_filled = 0
        self.current_color_index = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []

        self.reset()
        self.validate_implementation()

    def _setup_stage(self, stage_index):
        config = self.STAGES_CONFIG[stage_index]
        grid_size = config["size"]

        self.canvas = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.cursor_pos = [grid_size // 2, grid_size // 2]
        self.max_time = config["time"]
        self.time_remaining = self.max_time
        self.pixels_to_fill = grid_size * grid_size
        self.pixels_filled = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.current_color_index = 0
        self.particles.clear()
        
        self._setup_stage(0)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1. Update time
        self.time_remaining -= 1

        # 2. Handle actions (rising edge detection)
        paint_action = space_held and not self.prev_space_held
        cycle_color_action = shift_held and not self.prev_shift_held

        # 3. Update game state based on actions
        grid_size = self.canvas.shape[0]
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        # Cursor wrap-around
        self.cursor_pos[0] %= grid_size
        self.cursor_pos[1] %= grid_size
        
        if cycle_color_action:
            self.current_color_index = (self.current_color_index + 1) % len(self.PAINT_COLORS)
            # SFX: color_swap.wav
        
        if paint_action and self.time_remaining > 0:
            cx, cy = self.cursor_pos
            if self.canvas[cy, cx] == 0:
                self.canvas[cy, cx] = self.current_color_index + 1
                self.pixels_filled += 1
                reward += 0.1
                self.score += 0.1
                self._spawn_particles(cx, cy, self.PAINT_COLORS[self.current_color_index])
                # SFX: paint_splat.wav

        # 4. Update particles
        self._update_particles()
        
        # 5. Check for stage/game completion
        if self.pixels_filled >= self.pixels_to_fill:
            reward += 10
            self.score += 10
            self.stage += 1
            if self.stage >= len(self.STAGES_CONFIG):
                self.game_over = True
                reward += 100
                self.score += 100
                # SFX: game_win.wav
            else:
                self._setup_stage(self.stage)
                # SFX: stage_clear.wav

        # 6. Check for termination
        terminated = self.game_over or self.time_remaining <= 0 or self.steps >= self.MAX_EPISODE_STEPS
        if self.time_remaining <= 0 and not self.game_over:
            reward -= 10  # Penalty for running out of time
            # SFX: game_over_timeout.wav

        # 7. Update internal step counter and previous action states
        self.steps += 1
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return (
            self._get_observation(),
            round(reward, 2),
            terminated,
            False,
            self._get_info()
        )

    def _spawn_particles(self, grid_x, grid_y, color):
        grid_size = self.canvas.shape[0]
        cell_size = min((self.W - 100) // grid_size, (self.H - 100) // grid_size)
        offset_x = (self.W - grid_size * cell_size) // 2
        offset_y = (self.H - grid_size * cell_size) // 2
        
        px = offset_x + grid_x * cell_size + cell_size / 2
        py = offset_y + grid_y * cell_size + cell_size / 2

        for _ in range(self.np_random.integers(8, 15)):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({"pos": [px, py], "vel": vel, "life": life, "max_life": life, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95  # friction
            p["vel"][1] *= 0.95
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_size = self.canvas.shape[0]
        
        # Calculate grid geometry to center it
        cell_size = min((self.W - 100) // grid_size, (self.H - 100) // grid_size)
        grid_pixel_width = grid_size * cell_size
        grid_pixel_height = grid_size * cell_size
        offset_x = (self.W - grid_pixel_width) // 2
        offset_y = (self.H - grid_pixel_height) // 2

        # Draw canvas cells
        for y in range(grid_size):
            for x in range(grid_size):
                color_index = self.canvas[y, x]
                color = self.PAINT_COLORS[color_index - 1] if color_index > 0 else self.COLOR_EMPTY
                rect = (offset_x + x * cell_size, offset_y + y * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, color, rect)

        # Draw grid lines
        for i in range(grid_size + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (offset_x + i * cell_size, offset_y), (offset_x + i * cell_size, offset_y + grid_pixel_height))
            pygame.draw.line(self.screen, self.COLOR_GRID, (offset_x, offset_y + i * cell_size), (offset_x + grid_pixel_width, offset_y + i * cell_size))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            size = int(3 * (p["life"] / p["max_life"]))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), size, (*p["color"], alpha))

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(offset_x + cx * cell_size, offset_y + cy * cell_size, cell_size, cell_size)
        
        # Pulsing glow effect
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        glow_alpha = 50 + pulse * 100
        glow_size = int(cell_size * 1.5)
        glow_color = (*self.COLOR_CURSOR, glow_alpha)
        
        glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, glow_color, (0, 0, glow_size, glow_size), border_radius=glow_size // 4)
        self.screen.blit(glow_surface, (cursor_rect.centerx - glow_size//2, cursor_rect.centery - glow_size//2), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=2)


    def _render_ui(self):
        # Time Bar
        time_bar_width = self.W - 40
        time_ratio = max(0, self.time_remaining / self.max_time)
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR_BG, (20, 20, time_bar_width, 20), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR_FG, (20, 20, time_bar_width * time_ratio, 20), border_radius=5)

        # Score Text
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (25, self.H - 35))
        
        # Stage Text
        stage_text = self.font_small.render(f"STAGE: {self.stage + 1}/{len(self.STAGES_CONFIG)}", True, self.COLOR_UI_TEXT)
        stage_rect = stage_text.get_rect(centerx=self.W // 2, y=self.H - 35)
        self.screen.blit(stage_text, stage_rect)

        # Fill Percentage
        fill_percent = (self.pixels_filled / self.pixels_to_fill) * 100 if self.pixels_to_fill > 0 else 0
        percent_text = self.font_small.render(f"{fill_percent:.1f}%", True, self.COLOR_UI_TEXT)
        percent_rect = percent_text.get_rect(right=self.W - 25, y=self.H - 35)
        self.screen.blit(percent_text, percent_rect)

        # Current Color Swatch
        swatch_size = 40
        swatch_rect = pygame.Rect(self.W - swatch_size - 20, 50, swatch_size, swatch_size)
        pygame.draw.rect(self.screen, self.PAINT_COLORS[self.current_color_index], swatch_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, swatch_rect, 2, border_radius=5)
        
        # Game Over / Win Text
        if self.time_remaining <= 0 and not self.game_over:
            text_surf = self.font_large.render("TIME'S UP!", True, self.PAINT_COLORS[0])
            text_rect = text_surf.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(text_surf, text_rect)
        elif self.game_over:
            text_surf = self.font_large.render("PERFECT!", True, self.PAINT_COLORS[3])
            text_rect = text_surf.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage + 1,
            "fill_percentage": (self.pixels_filled / self.pixels_to_fill) if self.pixels_to_fill > 0 else 0,
            "time_remaining": self.time_remaining,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Simple interactive loop
    # This demonstrates manual control
    pygame.display.set_caption("Pixel Painter")
    screen = pygame.display.set_mode((env.W, env.H))

    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward}, Score: {info['score']:.1f}, Terminated: {terminated}")
            
        env.clock.tick(30) # Limit frame rate for human play

    print("Game Over!")
    print(f"Final Score: {info['score']:.1f}, Steps: {info['steps']}")
    env.close()