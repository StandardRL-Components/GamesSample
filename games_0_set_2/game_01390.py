
# Generated: 2025-08-27T16:59:32.054342
# Source Brief: brief_01390.md
# Brief Index: 1390

        
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
        "Controls: Use arrow keys to move the selector. Press Space to collect the selected gem."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect all gems by triggering chain reactions. You have a limited number of moves. Plan carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_SIZE = 5
        self.NUM_GEM_TYPES = 5
        self.TOTAL_MOVES = 30
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.GEM_HIGHLIGHTS = [pygame.Color(c).lerp((255, 255, 255), 0.4) for c in self.GEM_COLORS]
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_MOVES_OK = (100, 220, 100)
        self.COLOR_MOVES_WARN = (220, 220, 100)
        self.COLOR_MOVES_DANGER = (220, 100, 100)
        
        # Isometric projection values
        self.tile_width = 72
        self.tile_height = self.tile_width / 2
        self.origin_x = self.SCREEN_WIDTH / 2
        self.origin_y = self.SCREEN_HEIGHT / 2 - self.GRID_SIZE * self.tile_height / 2 + 20

        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        self.moves_left = self.TOTAL_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1
        # shift_pressed = action[2] == 1 # Unused in this design

        reward = 0
        collected_this_step = False

        self._move_cursor(movement)

        if space_pressed:
            self.moves_left -= 1
            collected_gems_info = self._collect_gems_at_cursor()
            
            if collected_gems_info["count"] > 0:
                collected_this_step = True
                reward += collected_gems_info["reward"]
                self._create_particles(collected_gems_info["positions"], collected_gems_info["color_index"])
            else:
                # Penalty for trying to collect an empty spot
                reward = -0.2
        else: # Action was just moving the cursor
            # This is a no-op turn, penalize slightly to encourage action
            # The brief states "-0.2 for any action that does not collect a gem"
            # And "Each action consumes one move".
            # This is ambiguous. I'll interpret "action" as pressing space.
            # A turn without pressing space is a free cursor move.
            pass

        self.score += reward
        self.steps += 1
        
        # Update particle animations
        self._update_particles()

        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += terminal_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        gems_remaining = np.count_nonzero(self.grid)
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "gems_remaining": gems_remaining,
            "cursor_pos": list(self.cursor_pos),
        }

    def _check_termination(self):
        gems_remaining = np.count_nonzero(self.grid)
        
        if gems_remaining == 0:
            self.game_over = True
            # +10 board clear bonus, +100 win bonus
            return True, 110.0
        
        if self.moves_left <= 0:
            self.game_over = True
            # -50 lose penalty
            return True, -50.0
            
        return False, 0.0

    def _move_cursor(self, direction):
        if direction == 1:  # Up
            self.cursor_pos[1] -= 1
        elif direction == 2:  # Down
            self.cursor_pos[1] += 1
        elif direction == 3:  # Left
            self.cursor_pos[0] -= 1
        elif direction == 4:  # Right
            self.cursor_pos[0] += 1
        
        # Wrap around grid
        self.cursor_pos[0] %= self.GRID_SIZE
        self.cursor_pos[1] %= self.GRID_SIZE

    def _collect_gems_at_cursor(self):
        x, y = self.cursor_pos
        gem_type = self.grid[y, x]

        if gem_type == 0:  # Empty space
            return {"count": 0, "reward": 0, "positions": [], "color_index": -1}

        # Find all connected gems of the same type
        to_collect = self._flood_fill(x, y, gem_type, set())
        
        collected_count = len(to_collect)
        reward = collected_count  # +1 per gem
        if collected_count > 1:
            reward += 5  # Chain reaction bonus

        positions = []
        for gx, gy in to_collect:
            self.grid[gy, gx] = 0
            positions.append(self._grid_to_iso(gx, gy))

        return {"count": collected_count, "reward": reward, "positions": positions, "color_index": gem_type - 1}

    def _flood_fill(self, x, y, target_type, visited):
        if not (0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE):
            return set()
        if (x, y) in visited:
            return set()
        if self.grid[y, x] != target_type:
            return set()

        visited.add((x, y))
        collected = {(x, y)}
        
        collected.update(self._flood_fill(x + 1, y, target_type, visited))
        collected.update(self._flood_fill(x - 1, y, target_type, visited))
        collected.update(self._flood_fill(x, y + 1, target_type, visited))
        collected.update(self._flood_fill(x, y - 1, target_type, visited))
        
        return collected

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_SIZE + 1):
            start_pos = self._grid_to_iso(0, r)
            end_pos = self._grid_to_iso(self.GRID_SIZE, r)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for c in range(self.GRID_SIZE + 1):
            start_pos = self._grid_to_iso(c, 0)
            end_pos = self._grid_to_iso(c, self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw gems
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    color_index = gem_type - 1
                    self._draw_iso_gem(self.screen, c, r, self.GEM_COLORS[color_index], self.GEM_HIGHLIGHTS[color_index])
        
        # Draw selector
        self._draw_selector()
        
        # Draw particles
        self._update_and_render_particles()

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Moves display
        if self.moves_left > self.TOTAL_MOVES / 3 * 2:
            moves_color = self.COLOR_MOVES_OK
        elif self.moves_left > self.TOTAL_MOVES / 3:
            moves_color = self.COLOR_MOVES_WARN
        else:
            moves_color = self.COLOR_MOVES_DANGER
            
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, moves_color)
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(moves_text, moves_rect)

        if self.game_over:
            gems_remaining = np.count_nonzero(self.grid)
            end_text_str = "YOU WIN!" if gems_remaining == 0 else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 50))
            self.screen.blit(end_text, end_rect)


    def _grid_to_iso(self, x, y):
        iso_x = self.origin_x + (x - y) * (self.tile_width / 2)
        iso_y = self.origin_y + (x + y) * (self.tile_height / 2)
        return int(iso_x), int(iso_y)

    def _draw_iso_gem(self, surface, x, y, color, highlight_color):
        center_x, center_y = self._grid_to_iso(x, y)
        center_y += self.tile_height / 2 # Adjust to draw from top point
        
        points = [
            (center_x, center_y - self.tile_height / 2),
            (center_x + self.tile_width / 2, center_y),
            (center_x, center_y + self.tile_height / 2),
            (center_x - self.tile_width / 2, center_y)
        ]
        
        highlight_points = [
            (center_x, center_y - self.tile_height / 2),
            (center_x + self.tile_width / 2, center_y),
            (center_x, center_y),
            (center_x - self.tile_width / 2, center_y),
        ]
        
        # Draw main gem body
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)
        
        # Draw highlight
        pygame.gfxdraw.aapolygon(surface, highlight_points, highlight_color)
        pygame.gfxdraw.filled_polygon(surface, highlight_points, highlight_color)

    def _draw_selector(self):
        if self.game_over: return
        
        cx, cy = self.cursor_pos
        center_x, center_y = self._grid_to_iso(cx, cy)
        center_y += self.tile_height / 2
        
        # Pulsating effect
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 # a value between 0 and 1
        size_mod = 4 + pulse * 4
        
        points = [
            (center_x, center_y - self.tile_height / 2 - size_mod),
            (center_x + self.tile_width / 2 + size_mod, center_y),
            (center_x, center_y + self.tile_height / 2 + size_mod),
            (center_x - self.tile_width / 2 - size_mod, center_y)
        ]
        
        alpha = 150 + int(pulse * 105)
        color = (255, 255, 255, alpha)

        pygame.draw.lines(self.screen, color, True, points, 3)

    def _create_particles(self, positions, color_index):
        # sound: gem_shatter.wav
        color = self.GEM_COLORS[color_index]
        for pos in positions:
            for _ in range(15): # 15 particles per gem
                self.particles.append(Particle(pos[0], pos[1], color, self.np_random))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles:
            p.update()
            
    def _update_and_render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

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

class Particle:
    def __init__(self, x, y, color, np_random):
        self.np_random = np_random
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        self.pos = [x, y]
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.lifespan = self.np_random.uniform(20, 40)
        self.max_lifespan = self.lifespan
        self.color = color

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] += 0.1  # Gravity
        self.lifespan -= 1

    def is_alive(self):
        return self.lifespan > 0

    def draw(self, surface):
        if not self.is_alive():
            return
        
        life_ratio = self.lifespan / self.max_lifespan
        size = int(max(0, 5 * life_ratio))
        
        if size > 0:
            alpha = int(255 * life_ratio)
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, self.color + (alpha,), (size, size), size)
            surface.blit(temp_surf, (int(self.pos[0] - size), int(self.pos[1] - size)), special_flags=pygame.BLEND_RGBA_ADD)

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to False to play manually
    auto_play = True

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # For manual play
    if not auto_play:
        pygame.display.set_caption("Gem Collector")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    terminated = False
    total_reward = 0
    
    action = env.action_space.sample() # Start with a random action

    while True:
        if auto_play:
            if terminated:
                print(f"Episode finished. Total Reward: {total_reward}")
                total_reward = 0
                obs, info = env.reset()
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        else: # Manual play
            # Convert numpy array to pygame surface for display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Event handling
            movement = 0 # no-op
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: # Manual reset
                        obs, info = env.reset()
                        total_reward = 0
                        terminated = False
                    if terminated:
                        continue
                    
                    # This block only triggers one action per key press
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP]: movement = 1
                    elif keys[pygame.K_DOWN]: movement = 2
                    elif keys[pygame.K_LEFT]: movement = 3
                    elif keys[pygame.K_RIGHT]: movement = 4
                    
                    if keys[pygame.K_SPACE]: space = 1
                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

                    action = [movement, space, shift]
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}, Info: {info}")