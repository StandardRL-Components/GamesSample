
# Generated: 2025-08-28T04:48:18.753168
# Source Brief: brief_05362.md
# Brief Index: 5362

        
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


# Helper class for particles. Defining it outside the main class for clarity,
# but it will be part of the single code block.
class _Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, rng):
        self.x = x
        self.y = y
        self.color = color
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = rng.integers(20, 40)
        self.radius = rng.integers(3, 7)

    def update(self):
        """Update particle position and lifespan."""
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius -= 0.15
        return self.lifespan > 0 and self.radius > 0

    def draw(self, surface):
        """Draw the particle on a surface."""
        if self.radius > 0:
            # Use a slightly transparent effect for particles
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(
                temp_surf, int(self.radius), int(self.radius), int(self.radius), (*self.color, 180)
            )
            surface.blit(temp_surf, (int(self.x - self.radius), int(self.y - self.radius)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Use arrow keys to move the cursor. Press an arrow key while holding Space to swap gems."
    )
    game_description = (
        "Swap gems to match 3 or more in a race against time! Create combos and clear the target number of gems to win."
    )
    auto_advance = True

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    NUM_GEM_TYPES = 6
    CELL_SIZE = 48
    GRID_LINE_WIDTH = 2
    BOARD_OFFSET_X = (640 - (GRID_WIDTH * CELL_SIZE)) // 2
    BOARD_OFFSET_Y = (400 - (GRID_HEIGHT * CELL_SIZE)) // 2 + 10

    TARGET_GEMS = 20
    MAX_TIME_SECONDS = 120
    FPS = 30
    MAX_STEPS = MAX_TIME_SECONDS * FPS

    # Animation Timings (in frames)
    SWAP_DURATION = 8
    REMOVE_DURATION = 12
    FALL_DURATION = 10

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (30, 45, 60)
    COLOR_GRID_LINE_LIGHT = (60, 75, 90)
    COLOR_GRID_LINE_DARK = (10, 20, 30)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SCORE = (255, 200, 0)
    COLOR_TIMER_GOOD = (0, 255, 128)
    COLOR_TIMER_WARN = (255, 128, 0)
    COLOR_TIMER_BAD = (255, 50, 50)

    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 120, 255),  # Blue
        (80, 255, 80),   # Green
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 140, 80),  # Orange
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.particle_class = _Particle

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.time_left = self.MAX_TIME_SECONDS

        self.grid = self._create_initial_grid()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.game_phase = "IDLE"
        self.animation_timer = 0
        self.swapping_gems = []
        self.removing_gems = []
        self.chain_level = 0
        self.step_reward = 0
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.step_reward = 0
        self.steps += 1
        self.time_left = max(0, self.time_left - 1.0 / self.FPS)

        self._update_game_state(action)
        self._update_particles()
        
        reward = self.step_reward
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.gems_collected >= self.TARGET_GEMS:
                reward += 100  # Win bonus
                self.game_phase = "WIN"
            else:
                reward -= 100  # Lose penalty
                self.game_phase = "LOSE"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_game_state(self, action):
        if self.game_phase == "IDLE":
            self._handle_input(action)
        elif self.game_phase == "SWAP":
            self._update_swap_animation()
        elif self.game_phase == "SWAP_BACK":
            self._update_swap_back_animation()
        elif self.game_phase == "REMOVE":
            self._update_remove_animation()
        elif self.game_phase == "FALL":
            self._update_fall_animation()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if space_held and (dx != 0 or dy != 0):
            x1, y1 = self.cursor_pos
            x2, y2 = x1 + dx, y1 + dy
            
            if 0 <= x2 < self.GRID_WIDTH and 0 <= y2 < self.GRID_HEIGHT:
                self.swapping_gems = [(x1, y1), (x2, y2)]
                self.game_phase = "SWAP"
                self.animation_timer = self.SWAP_DURATION
                # SFX: swap_attempt.wav
        elif not space_held and (dx != 0 or dy != 0):
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_HEIGHT - 1)

    def _update_swap_animation(self):
        self.animation_timer -= 1
        if self.animation_timer <= 0:
            (x1, y1), (x2, y2) = self.swapping_gems
            self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
            
            matches1 = self._find_matches_at(x1, y1)
            matches2 = self._find_matches_at(x2, y2)
            
            if not matches1 and not matches2:
                self.game_phase = "SWAP_BACK"
                self.animation_timer = self.SWAP_DURATION
                self.step_reward -= 0.1
                # SFX: invalid_swap.wav
            else:
                self.chain_level = 1
                self._find_and_process_matches()
                # SFX: match_found.wav

    def _update_swap_back_animation(self):
        self.animation_timer -= 1
        if self.animation_timer <= 0:
            (x1, y1), (x2, y2) = self.swapping_gems
            self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
            self.game_phase = "IDLE"
            self.swapping_gems = []

    def _find_and_process_matches(self):
        all_matches = self._find_all_matches()
        if all_matches:
            if self.chain_level > 1:
                self.step_reward += 5 # Chain reaction bonus
            
            self.step_reward += len(all_matches)
            self.score += len(all_matches) * self.chain_level
            self.gems_collected += len(all_matches)
            
            for (x, y) in all_matches:
                self._spawn_particles(x, y, self.GEM_COLORS[self.grid[y, x]])
            
            self.removing_gems = list(all_matches)
            self.game_phase = "REMOVE"
            self.animation_timer = self.REMOVE_DURATION
        else:
            self.game_phase = "IDLE"
            self.chain_level = 0
    
    def _update_remove_animation(self):
        self.animation_timer -= 1
        if self.animation_timer <= 0:
            for x, y in self.removing_gems:
                self.grid[y, x] = -1 # Mark as empty
            self.removing_gems = []
            self._apply_gravity_and_refill()
            self.game_phase = "FALL"
            self.animation_timer = self.FALL_DURATION
            # SFX: gems_fall.wav

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y, x] != -1:
                    if y != empty_row:
                        self.grid[empty_row, x] = self.grid[y, x]
                        self.grid[y, x] = -1
                    empty_row -= 1
            
            for y in range(empty_row, -1, -1):
                self.grid[y, x] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _update_fall_animation(self):
        self.animation_timer -= 1
        if self.animation_timer <= 0:
            self.chain_level += 1
            self._find_and_process_matches()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_bg()
        self._render_gems()
        self._render_cursor()
        self._render_particles_on_surface()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_bg(self):
        board_rect = pygame.Rect(self.BOARD_OFFSET_X, self.BOARD_OFFSET_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, board_rect)
        for i in range(self.GRID_WIDTH + 1):
            x = self.BOARD_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE_DARK, (x, self.BOARD_OFFSET_Y), (x, self.BOARD_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_HEIGHT + 1):
            y = self.BOARD_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE_DARK, (self.BOARD_OFFSET_X, y), (self.BOARD_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, y), self.GRID_LINE_WIDTH)

    def _render_gems(self):
        gem_radius = self.CELL_SIZE // 2 - 6
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[y, x]
                if gem_type == -1:
                    continue

                center_x = self.BOARD_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = self.BOARD_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
                
                draw_x, draw_y = center_x, center_y
                scale = 1.0

                if self.game_phase in ["SWAP", "SWAP_BACK"] and (x, y) in self.swapping_gems:
                    progress = self.animation_timer / self.SWAP_DURATION
                    (x1, y1), (x2, y2) = self.swapping_gems
                    
                    is_first_gem = (x, y) == (x1, y1)
                    other_gem_pos = (x2, y2) if is_first_gem else (x1, y1)

                    target_x = self.BOARD_OFFSET_X + other_gem_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
                    target_y = self.BOARD_OFFSET_Y + other_gem_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
                    
                    interp_progress = 1.0 - progress
                    draw_x = int(center_x + (target_x - center_x) * interp_progress)
                    draw_y = int(center_y + (target_y - center_y) * interp_progress)

                if self.game_phase == "REMOVE" and (x, y) in self.removing_gems:
                    progress = self.animation_timer / self.REMOVE_DURATION
                    scale = progress
                
                if self.game_phase == "FALL":
                    progress = 1.0 - (self.animation_timer / self.FALL_DURATION)
                    draw_y = int(center_y - (1 - progress) * self.CELL_SIZE / 2)
                    scale = progress * 0.5 + 0.5

                if scale <= 0: continue
                self._draw_gem(draw_x, draw_y, gem_radius * scale, self.GEM_COLORS[gem_type])

    def _draw_gem(self, x, y, r, color):
        x, y, r = int(x), int(y), int(r)
        if r <= 1: return
        
        pygame.gfxdraw.filled_circle(self.screen, x, y, r, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, r, tuple(min(255, c+30) for c in color))
        
        shine_color = (255, 255, 255, 128)
        shine_r = int(r * 0.4)
        shine_x = x - int(r * 0.3)
        shine_y = y - int(r * 0.3)
        if shine_r > 1:
            temp_surf = pygame.Surface((shine_r * 2, shine_r * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, shine_r, shine_r, shine_r, shine_color)
            self.screen.blit(temp_surf, (shine_x-shine_r, shine_y-shine_r))

    def _render_cursor(self):
        if self.game_phase == "IDLE":
            x, y = self.cursor_pos
            rect = pygame.Rect(
                self.BOARD_OFFSET_X + x * self.CELL_SIZE,
                self.BOARD_OFFSET_Y + y * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=4)

    def _render_ui(self):
        gem_icon_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        self._draw_gem(15, 15, 12, self.GEM_COLORS[0])
        self.screen.blit(gem_icon_surf, (10, 10))
        text = self.font_medium.render(f"{self.gems_collected} / {self.TARGET_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(text, (50, 12))

        time_ratio = self.time_left / self.MAX_TIME_SECONDS
        if time_ratio < 0.25: timer_color = self.COLOR_TIMER_BAD
        elif time_ratio < 0.5: timer_color = self.COLOR_TIMER_WARN
        else: timer_color = self.COLOR_TIMER_GOOD
        
        minutes = int(self.time_left) // 60
        seconds = int(self.time_left) % 60
        time_text = f"{minutes:02}:{seconds:02}"
        text = self.font_medium.render(time_text, True, timer_color)
        self.screen.blit(text, (640 - text.get_width() - 20, 12))

        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (640 // 2 - score_text.get_width() // 2, 400 - score_text.get_height() - 5))

        if self.game_phase in ["WIN", "LOSE"]:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            msg = "YOU WIN!" if self.game_phase == "WIN" else "TIME'S UP!"
            color = self.COLOR_TIMER_GOOD if self.game_phase == "WIN" else self.COLOR_TIMER_BAD
            text = self.font_large.render(msg, True, color)
            text_rect = text.get_rect(center=(320, 200))
            self.screen.blit(overlay, (0,0))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "gems_collected": self.gems_collected}

    def _check_termination(self):
        return self.time_left <= 0 or self.gems_collected >= self.TARGET_GEMS

    def _create_initial_grid(self):
        while True:
            grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if not self._find_all_matches(grid):
                return grid

    def _find_all_matches(self, grid=None):
        if grid is None:
            grid = self.grid
        
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if grid[r,c] == -1: continue
                if c < self.GRID_WIDTH - 2 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matches.update([(c, r), (c+1, r), (c+2, r)])
                if r < self.GRID_HEIGHT - 2 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matches.update([(c, r), (c, r+1), (c, r+2)])
        return matches

    def _find_matches_at(self, c, r):
        gem_type = self.grid[r, c]
        if gem_type == -1: return set()
        
        h_matches, v_matches = {(c, r)}, {(c, r)}
        for i in range(c - 1, -1, -1):
            if self.grid[r, i] == gem_type: h_matches.add((i, r))
            else: break
        for i in range(c + 1, self.GRID_WIDTH):
            if self.grid[r, i] == gem_type: h_matches.add((i, r))
            else: break
        
        for i in range(r - 1, -1, -1):
            if self.grid[i, c] == gem_type: v_matches.add((c, i))
            else: break
        for i in range(r + 1, self.GRID_HEIGHT):
            if self.grid[i, c] == gem_type: v_matches.add((c, i))
            else: break
        
        found_matches = set()
        if len(h_matches) >= 3: found_matches.update(h_matches)
        if len(v_matches) >= 3: found_matches.update(v_matches)
        return found_matches

    def _spawn_particles(self, x, y, color):
        center_x = self.BOARD_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.BOARD_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(self.np_random.integers(8, 15)):
            self.particles.append(self.particle_class(center_x, center_y, color, self.np_random))
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]
        
    def _render_particles_on_surface(self):
        for p in self.particles:
            p.draw(self.screen)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # To run and play the game
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Gem Swap")
    
    while running:
        move_action, space_action, shift_action = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1
            
        action = np.array([move_action, space_action, shift_action])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Gems: {info['gems_collected']}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            pygame.time.wait(3000)
            obs, info = env.reset()
        
        env.clock.tick(env.FPS)
        
    env.close()