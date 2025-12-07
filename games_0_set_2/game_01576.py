
# Generated: 2025-08-27T17:35:04.175439
# Source Brief: brief_01576.md
# Brief Index: 1576

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to collect a cluster of gems."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race against the clock to collect 50 gems. Clicking a gem collects it and all "
        "adjacent gems of the same color. New gems fall from above. Larger clusters give bonus points!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 8
        self.MAX_TIME = 60.0  # seconds
        self.WIN_GEMS = 50
        self.MAX_STEPS = 1800 # 60 seconds * 30 fps

        # --- Colors ---
        self.COLOR_BG = (20, 30, 50)
        self.COLOR_GRID = (40, 60, 90)
        self.COLOR_CURSOR = (255, 255, 0, 100) # Yellow, semi-transparent
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 20)

        self.GEM_COLORS = {
            1: (255, 50, 50),   # Red
            2: (50, 255, 50),   # Green
            3: (80, 80, 255),   # Blue
            4: (255, 220, 50),  # Yellow
        }
        self.GEM_POINTS = {1: 1, 2: 2, 3: 3, 4: 4}

        # --- Isometric Projection ---
        self.TILE_WIDTH_HALF = 24
        self.TILE_HEIGHT_HALF = 12
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Etc...        
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.gems_collected = None
        self.time_remaining = None
        self.steps = None
        self.game_over = None
        self.last_space_held = None
        self.particles = None
        self.floating_texts = None
        self.np_random = None
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.grid = self.np_random.integers(1, len(self.GEM_COLORS) + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.score = 0
        self.gems_collected = 0
        self.time_remaining = self.MAX_TIME
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.particles = []
        self.floating_texts = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1
        self.clock.tick(30)
        self.time_remaining = max(0, self.time_remaining - self.clock.get_time() / 1000.0)

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean (unused in this game)
        
        # Update game logic
        self._handle_input(movement, space_held)
        
        if space_held and not self.last_space_held:
            # Sound effect placeholder: # sfx_gem_collect()
            reward = self._collect_gems()
        self.last_space_held = space_held
        
        self._update_particles()
        self._update_floating_texts()

        terminated = self._check_termination()
        
        if terminated and not self.game_over: # First frame of termination
            if self.gems_collected >= self.WIN_GEMS:
                # Sound effect placeholder: # sfx_win_jingle()
                reward += 50 # Win bonus
                self._add_floating_text("YOU WIN!", (self.WIDTH // 2, self.HEIGHT // 2), self.font_large, (255, 255, 100), duration=90)
            elif self.time_remaining <= 0:
                # Sound effect placeholder: # sfx_lose_sound()
                self._add_floating_text("TIME'S UP!", (self.WIDTH // 2, self.HEIGHT // 2), self.font_large, (255, 100, 100), duration=90)
        
        self.game_over = terminated

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held):
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

    def _collect_gems(self):
        x, y = self.cursor_pos
        gem_type = self.grid[x, y]
        if gem_type == 0: return 0

        # Find connected gems using Breadth-First Search
        q = deque([(x, y)])
        visited = set([(x, y)])
        to_collect = []
        while q:
            cx, cy = q.popleft()
            to_collect.append((cx, cy))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited and self.grid[nx, ny] == gem_type:
                    visited.add((nx, ny))
                    q.append((nx, ny))

        if not to_collect: return 0
        
        collected_count = len(to_collect)
        base_points = sum(self.GEM_POINTS[self.grid[gx, gy]] for gx, gy in to_collect)
        bonus_reward = 5 if collected_count >= 4 else 0
        
        self.score += base_points
        self.gems_collected += collected_count

        avg_x = sum(c[0] for c in to_collect) / collected_count
        avg_y = sum(c[1] for c in to_collect) / collected_count
        center_pos = self._cart_to_iso(avg_x, avg_y)

        if bonus_reward > 0:
            self._add_floating_text(f"{collected_count} GEMS! +{bonus_reward}", center_pos, self.font_medium, (255, 255, 100))
        else:
            self._add_floating_text(f"{collected_count}", center_pos, self.font_small, (220, 220, 220))

        for gx, gy in to_collect:
            self._create_particles(gx, gy, self.GEM_COLORS[self.grid[gx, gy]])
            self.grid[gx, gy] = 0

        self._apply_gravity_and_refill()
        return collected_count + bonus_reward

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_slots = []
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == 0:
                    empty_slots.append(y)
                elif empty_slots:
                    dest_y = empty_slots.pop(0)
                    self.grid[x, dest_y] = self.grid[x, y]
                    self.grid[x, y] = 0
                    empty_slots.append(y)
            for y in empty_slots:
                self.grid[x, y] = self.np_random.integers(1, len(self.GEM_COLORS) + 1)

    def _check_termination(self):
        return self.gems_collected >= self.WIN_GEMS or self.time_remaining <= 0 or self.steps >= self.MAX_STEPS

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
            "gems_collected": self.gems_collected,
            "time_remaining": self.time_remaining,
        }

    def _cart_to_iso(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Draw grid lines
        for y in range(self.GRID_HEIGHT + 1):
            start = self._cart_to_iso(-0.5, y - 0.5)
            end = self._cart_to_iso(self.GRID_WIDTH - 0.5, y - 0.5)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_WIDTH + 1):
            start = self._cart_to_iso(x - 0.5, -0.5)
            end = self._cart_to_iso(x - 0.5, self.GRID_HEIGHT - 0.5)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # Draw gems
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[x, y]
                if gem_type > 0:
                    color = self.GEM_COLORS[gem_type]
                    dark_color = tuple(max(0, c * 0.7) for c in color)
                    pos = self._cart_to_iso(x, y)
                    points = [
                        (pos[0], pos[1] - self.TILE_HEIGHT_HALF),
                        (pos[0] + self.TILE_WIDTH_HALF, pos[1]),
                        (pos[0], pos[1] + self.TILE_HEIGHT_HALF),
                        (pos[0] - self.TILE_WIDTH_HALF, pos[1]),
                    ]
                    pygame.gfxdraw.aapolygon(self.screen, points, dark_color)
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_center = self._cart_to_iso(cursor_x, cursor_y)
        cursor_points = [
            (cursor_center[0], cursor_center[1] - self.TILE_HEIGHT_HALF - 2),
            (cursor_center[0] + self.TILE_WIDTH_HALF + 2, cursor_center[1]),
            (cursor_center[0], cursor_center[1] + self.TILE_HEIGHT_HALF + 2),
            (cursor_center[0] - self.TILE_WIDTH_HALF - 2, cursor_center[1]),
        ]
        pygame.gfxdraw.aapolygon(self.screen, cursor_points, self.COLOR_CURSOR[:3])
        pygame.gfxdraw.filled_polygon(self.screen, cursor_points, self.COLOR_CURSOR)

        # Draw particles and floating texts
        for p in self.particles:
            pos = (int(p['x']), int(p['y']))
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (pos[0] - p['size'], pos[1] - p['size']))
        for ft in self.floating_texts:
            self._render_text(ft['text'], ft['pos'], ft['font'], ft['color'], centered=True, alpha=ft['alpha'])

    def _render_ui(self):
        self._render_text(f"Score: {self.score}", (15, 10), self.font_medium, self.COLOR_TEXT)
        self._render_text(f"Gems: {self.gems_collected} / {self.WIN_GEMS}", (15, 40), self.font_small, self.COLOR_TEXT)
        time_str = f"{int(self.time_remaining // 60):01}:{int(self.time_remaining % 60):02}"
        self._render_text(time_str, (self.WIDTH - 15, 10), self.font_large, self.COLOR_TEXT, align="right")
        self._render_text(f"Steps: {self.steps}/{self.MAX_STEPS}", (self.WIDTH - 15, 50), self.font_small, self.COLOR_TEXT, align="right")

    def _render_text(self, text, pos, font, color, align="left", centered=False, alpha=255):
        shadow_color = self.COLOR_TEXT_SHADOW + (alpha,) if len(self.COLOR_TEXT_SHADOW) == 3 else self.COLOR_TEXT_SHADOW
        text_color = color + (alpha,) if len(color) == 3 else color
        text_surface = font.render(text, True, text_color)
        text_surface.set_alpha(alpha)
        shadow_surface = font.render(text, True, shadow_color)
        shadow_surface.set_alpha(alpha)
        text_rect, shadow_rect = text_surface.get_rect(), shadow_surface.get_rect()
        if centered:
            text_rect.center, shadow_rect.center = pos, (pos[0] + 2, pos[1] + 2)
        elif align == "right":
            text_rect.topright, shadow_rect.topright = pos, (pos[0] + 2, pos[1] + 2)
        else:
            text_rect.topleft, shadow_rect.topleft = pos, (pos[0] + 2, pos[1] + 2)
        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, grid_x, grid_y, color):
        pos = self._cart_to_iso(grid_x, grid_y)
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': pos[0], 'y': pos[1], 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30), 'max_life': 30,
                'color': color, 'size': self.np_random.integers(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']; p['y'] += p['vy']; p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _add_floating_text(self, text, pos, font, color, duration=30):
        self.floating_texts.append({
            'text': text, 'pos': pos, 'font': font, 'color': color,
            'life': duration, 'max_life': duration, 'alpha': 255
        })

    def _update_floating_texts(self):
        for ft in self.floating_texts:
            ft['life'] -= 1
            ft['pos'] = (ft['pos'][0], ft['pos'][1] - 1)
            ft['alpha'] = int(255 * (ft['life'] / ft['max_life']))
        self.floating_texts = [ft for ft in self.floating_texts if ft['life'] > 0]

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