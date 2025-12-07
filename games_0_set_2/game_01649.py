
# Generated: 2025-08-28T02:15:48.746180
# Source Brief: brief_01649.md
# Brief Index: 1649

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to swap the selected pixel with an adjacent one. "
        "Press Space for a risky swap with a random pixel on the grid."
    )

    game_description = (
        "Recreate the target image by swapping pixels on the grid before time runs out. "
        "Each move costs time, but correct placements will increase your score."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.PIXEL_SIZE = 24
        self.GAP = 2
        self.GRID_LINE_WIDTH = 1
        
        self.GRID_AREA_WIDTH = self.GRID_WIDTH * (self.PIXEL_SIZE + self.GAP) - self.GAP
        self.GRID_AREA_HEIGHT = self.GRID_HEIGHT * (self.PIXEL_SIZE + self.GAP) - self.GAP
        self.GRID_TOP_LEFT = (
            (self.SCREEN_WIDTH - self.GRID_AREA_WIDTH) // 2,
            (self.SCREEN_HEIGHT - self.GRID_AREA_HEIGHT) // 2
        )

        self.TARGET_PREVIEW_SIZE = 80
        self.TARGET_PREVIEW_POS = (20, self.SCREEN_HEIGHT - self.TARGET_PREVIEW_SIZE - 20)
        
        self.FPS = 30
        self.MAX_TIME = 60.0
        self.MAX_STEPS = int(self.MAX_TIME * self.FPS)

        # --- Colors ---
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID_BG = (25, 30, 50)
        self.COLOR_GRID_LINE = (40, 45, 70)
        self.COLOR_TEXT = (230, 230, 255)
        self.COLOR_UI_BG = (30, 35, 60, 200)
        self.COLOR_SELECTOR = (255, 255, 255)
        self.COLOR_CORRECT_HINT = (0, 255, 120, 100)
        self.COLOR_MOVED_HINT = (255, 200, 0)
        
        self.COLOR_PALETTE = [
            (255, 0, 95),    # Hot Pink
            (255, 175, 0),   # Orange
            (0, 204, 153),   # Teal
            (0, 153, 255),   # Blue
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        self.grid = None
        self.target_grid = None
        self.selector_pos = None
        self.timer = 0.0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.last_space_held = False
        self.particles = []
        self.last_moved_info = {"pos": None, "timer": 0}
        self.np_random = None

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.timer = self.MAX_TIME
        self.game_over = False
        self.last_space_held = False
        self.particles.clear()
        self.last_moved_info = {"pos": None, "timer": 0}

        self._generate_puzzle()
        self.selector_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.score = self._calculate_match_percentage()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.clock.tick(self.FPS)
        self.steps += 1
        
        reward = 0.0
        action_performed = False
        action_type = "none"

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        self.last_space_held = space_held

        old_correct_pixels = self._count_correct_pixels()

        if space_press:
            self._perform_risky_swap()
            action_performed = True
            action_type = "risky"
            # sfx: risky_swap.wav
        elif movement > 0:
            if self._perform_directional_swap(movement):
                action_performed = True
                action_type = "normal"
                # sfx: swap.wav
        
        # Update game state (timers, effects)
        self.timer -= 1.0 / self.FPS
        self._update_particles()
        if self.last_moved_info["timer"] > 0:
            self.last_moved_info["timer"] -= 1

        # Calculate reward
        if action_performed:
            new_correct_pixels = self._count_correct_pixels()
            delta_correct = new_correct_pixels - old_correct_pixels
            
            if action_type == "risky":
                if delta_correct > 0:
                    reward += 2.0
                elif delta_correct < 0:
                    reward -= 1.0
            elif action_type == "normal":
                reward += delta_correct * 0.5
        
        # Small time penalty to encourage efficiency
        reward -= 0.01

        # Update score and check for termination
        self.score = self._calculate_match_percentage()
        terminated = self._check_termination()
        
        if terminated:
            if self.score >= 100.0:
                reward = 100.0  # Win bonus
                # sfx: win_jingle.wav
            else:
                reward = -10.0 # Timeout penalty
                # sfx: lose_sound.wav
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_puzzle(self):
        self.target_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        # Create a simple, recognizable pattern (e.g., a cross)
        center_x, center_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        for i in range(self.GRID_HEIGHT):
            self.target_grid[i, center_x] = 1
        for j in range(self.GRID_WIDTH):
            self.target_grid[center_y, j] = 1
        self.target_grid[center_y, center_x] = 2
        self.target_grid[0, 0] = 3
        self.target_grid[0, -1] = 3
        self.target_grid[-1, 0] = 3
        self.target_grid[-1, -1] = 3

        # Shuffle the grid to create the starting puzzle state
        flat_grid = self.target_grid.flatten()
        self.np_random.shuffle(flat_grid)
        self.grid = flat_grid.reshape((self.GRID_HEIGHT, self.GRID_WIDTH))
        
        # Ensure it's not solved from the start
        if self._count_correct_pixels() == self.GRID_WIDTH * self.GRID_HEIGHT:
            self._generate_puzzle() # Recurse if already solved

    def _perform_directional_swap(self, movement):
        x, y = self.selector_pos
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        nx, ny = x + dx, y + dy

        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
            self._swap_pixels((x, y), (nx, ny))
            self.selector_pos = [nx, ny]
            return True
        return False

    def _perform_risky_swap(self):
        x1, y1 = self.selector_pos
        
        # Ensure we don't swap with the same pixel
        while True:
            x2 = self.np_random.integers(0, self.GRID_WIDTH)
            y2 = self.np_random.integers(0, self.GRID_HEIGHT)
            if (x1, y1) != (x2, y2):
                break
        
        self._swap_pixels((x1, y1), (x2, y2))

    def _swap_pixels(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        
        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
        
        self.last_moved_info = {"pos": pos1, "timer": self.FPS // 3}
        self._create_particles(pos1, self.COLOR_PALETTE[self.grid[y2, x2]])
        self._create_particles(pos2, self.COLOR_PALETTE[self.grid[y1, x1]])

    def _create_particles(self, pos, color):
        grid_x, grid_y = pos
        center_x = self.GRID_TOP_LEFT[0] + grid_x * (self.PIXEL_SIZE + self.GAP) + self.PIXEL_SIZE / 2
        center_y = self.GRID_TOP_LEFT[1] + grid_y * (self.PIXEL_SIZE + self.GAP) + self.PIXEL_SIZE / 2
        
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.uniform(0.3, 0.8)
            self.particles.append({
                "pos": [center_x, center_y],
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1.0 / self.FPS
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _check_termination(self):
        if self.timer <= 0 or self.score >= 100.0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _count_correct_pixels(self):
        return np.sum(self.grid == self.target_grid)

    def _calculate_match_percentage(self):
        total_pixels = self.GRID_WIDTH * self.GRID_HEIGHT
        correct_pixels = self._count_correct_pixels()
        return (correct_pixels / total_pixels) * 100.0

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
            "time_left": self.timer,
        }

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(
            self.screen, self.COLOR_GRID_BG,
            (self.GRID_TOP_LEFT[0] - self.GAP, self.GRID_TOP_LEFT[1] - self.GAP,
             self.GRID_AREA_WIDTH + self.GAP * 2, self.GRID_AREA_HEIGHT + self.GAP * 2),
            border_radius=4
        )
        
        # Draw grid pixels
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.grid[y, x]
                pixel_color = self.COLOR_PALETTE[color_idx]
                
                rect = pygame.Rect(
                    self.GRID_TOP_LEFT[0] + x * (self.PIXEL_SIZE + self.GAP),
                    self.GRID_TOP_LEFT[1] + y * (self.PIXEL_SIZE + self.GAP),
                    self.PIXEL_SIZE, self.PIXEL_SIZE
                )
                pygame.draw.rect(self.screen, pixel_color, rect, border_radius=2)

                # Hint for correctly placed pixels
                if self.grid[y, x] == self.target_grid[y, x]:
                    hint_surf = pygame.Surface((self.PIXEL_SIZE, self.PIXEL_SIZE), pygame.SRCALPHA)
                    pygame.draw.rect(hint_surf, self.COLOR_CORRECT_HINT, hint_surf.get_rect(), border_radius=3)
                    self.screen.blit(hint_surf, rect.topleft)

        # Draw selector
        sel_x, sel_y = self.selector_pos
        sel_rect = pygame.Rect(
            self.GRID_TOP_LEFT[0] + sel_x * (self.PIXEL_SIZE + self.GAP) - self.GAP,
            self.GRID_TOP_LEFT[1] + sel_y * (self.PIXEL_SIZE + self.GAP) - self.GAP,
            self.PIXEL_SIZE + self.GAP * 2, self.PIXEL_SIZE + self.GAP * 2
        )
        
        # Pulsating glow effect for selector
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        glow_size = int(pulse * 4)
        glow_alpha = int(100 + pulse * 100)
        glow_color = (*self.COLOR_SELECTOR, glow_alpha)
        
        glow_surf = pygame.Surface((sel_rect.width + glow_size*2, sel_rect.height + glow_size*2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=6)
        self.screen.blit(glow_surf, (sel_rect.left - glow_size, sel_rect.top - glow_size))
        
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, sel_rect, 2, border_radius=4)
        
        # Draw recently moved hint
        if self.last_moved_info["timer"] > 0:
            pos = self.last_moved_info["pos"]
            alpha = int(255 * (self.last_moved_info["timer"] / (self.FPS // 3)))
            color = (*self.COLOR_MOVED_HINT, alpha)
            rect = pygame.Rect(
                self.GRID_TOP_LEFT[0] + pos[0] * (self.PIXEL_SIZE + self.GAP),
                self.GRID_TOP_LEFT[1] + pos[1] * (self.PIXEL_SIZE + self.GAP),
                self.PIXEL_SIZE, self.PIXEL_SIZE
            )
            hint_surf = pygame.Surface((self.PIXEL_SIZE, self.PIXEL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(hint_surf, color, hint_surf.get_rect(), 2, border_radius=3)
            self.screen.blit(hint_surf, rect.topleft)

        # Draw particles
        for p in self.particles:
            size = int(p["life"] / p["max_life"] * 5)
            if size > 0:
                pygame.draw.rect(self.screen, p["color"], (p["pos"][0] - size/2, p["pos"][1] - size/2, size, size))

    def _render_ui(self):
        # UI Background Panel
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 60), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Render Timer
        time_text = f"TIME: {max(0, self.timer):.1f}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (20, 15))

        # Render Score
        score_text = f"MATCH: {self.score:.1f}%"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 20, 15))

        # Render Target Preview
        preview_bg_rect = pygame.Rect(self.TARGET_PREVIEW_POS[0], self.TARGET_PREVIEW_POS[1], self.TARGET_PREVIEW_SIZE, self.TARGET_PREVIEW_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, preview_bg_rect, border_radius=4)
        
        preview_title = self.font_small.render("TARGET", True, self.COLOR_TEXT)
        self.screen.blit(preview_title, (preview_bg_rect.centerx - preview_title.get_width()//2, preview_bg_rect.top - 22))

        pixel_size = (self.TARGET_PREVIEW_SIZE - (self.GRID_WIDTH - 1)) // self.GRID_WIDTH
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.target_grid[y, x]
                color = self.COLOR_PALETTE[color_idx]
                rect = pygame.Rect(
                    preview_bg_rect.left + x * (pixel_size + 1),
                    preview_bg_rect.top + y * (pixel_size + 1),
                    pixel_size, pixel_size
                )
                pygame.draw.rect(self.screen, color, rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Perfect")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert observation back to a Pygame surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.1f}%")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(env.FPS)

    env.close()