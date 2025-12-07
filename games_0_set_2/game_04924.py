import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper class for crystals
class Crystal:
    def __init__(self, x, y, color_index):
        self.x = x
        self.y = y
        self.color_index = color_index

# Helper class for particles
class Particle:
    def __init__(self, x, y, color, lifetime=20, gravity=0.1):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 3)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 2 # Start with an upward pop
        self.color = color
        self.lifetime = lifetime
        self.life = lifetime
        self.gravity = gravity

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            size = int(max(1, 4 * (self.life / self.lifetime)))
            alpha = int(255 * (self.life / self.lifetime))
            # Create a temporary surface for the particle to handle alpha transparency
            particle_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, self.color + (alpha,), (size, size), size)
            surface.blit(particle_surf, (int(self.x) - size, int(self.y) - size))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→↑↓ to push selected crystal. Space/Shift to cycle selection."
    )

    game_description = (
        "Isometric puzzle game. Push crystals to align 5 of the same color horizontally or vertically before time runs out."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 10
    TILE_WIDTH, TILE_HEIGHT = 48, 24
    TILE_DEPTH = 30
    
    # Colors
    COLOR_BG = (15, 18, 23)
    COLOR_GRID = (40, 50, 60)
    CRYSTAL_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    COLOR_WHITE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HIGHLIGHT = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = 100

        self.grid = []
        self.crystals = []
        self.particles = []
        self.selected_crystal_idx = -1
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_match_info = []
        self.steps = 0
        self.score = 0
        self.max_steps = 1000
        self.game_over = False
        self.win_state = False
        self.np_random = None

        # self.reset() is called by gym.make, no need to call it here
        # self.validate_implementation() # Optional validation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.particles.clear()
        self.last_match_info.clear()

        # Procedurally generate initial board state without matches
        while True:
            self.grid = [[None for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
            for y in range(self.GRID_HEIGHT // 2, self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    if self.np_random.random() > 0.3:
                        color_idx = self.np_random.integers(0, len(self.CRYSTAL_COLORS))
                        self.grid[y][x] = Crystal(x, y, color_idx)
            
            self._apply_gravity()
            if not self._find_matches():
                break
        
        self.crystals = self._get_all_crystals()
        self.selected_crystal_idx = 0 if self.crystals else -1

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.last_match_info.clear()
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        action_taken = movement != 0 or space_pressed or shift_pressed
        if action_taken:
            reward -= 0.01

        # 1. Handle Selection
        if self.crystals and (space_pressed or shift_pressed):
            if space_pressed:
                self.selected_crystal_idx = (self.selected_crystal_idx + 1) % len(self.crystals)
            if shift_pressed:
                self.selected_crystal_idx = (self.selected_crystal_idx - 1 + len(self.crystals)) % len(self.crystals)
        
        # 2. Handle Push
        pushed = False
        if movement != 0 and self.crystals and self.selected_crystal_idx != -1:
            pushed = self._push_crystal(movement)
        
        # 3. Handle Gravity, Matches, and Spawning
        if pushed:
            cycle_rewards, is_win = self._run_game_cycle()
            reward += cycle_rewards
            if is_win:
                self.win_state = True
                self.game_over = True
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if not self.win_state: # Lost by running out of steps
                reward -= 50

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _push_crystal(self, movement):
        crystal = self.crystals[self.selected_crystal_idx]
        dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
        
        nx, ny = crystal.x + dx, crystal.y + dy
        
        if not (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT):
            return False # Out of bounds
        
        if self.grid[ny][nx] is not None:
            # Failed push feedback
            sx, sy = self._iso_to_screen(crystal.x, crystal.y)
            for _ in range(5): self.particles.append(Particle(sx, sy, self.COLOR_WHITE, lifetime=10, gravity=0.05))
            # Sound: *clink*
            return False

        # Execute push
        self.grid[ny][nx] = crystal
        self.grid[crystal.y][crystal.x] = None
        crystal.x, crystal.y = nx, ny
        return True

    def _run_game_cycle(self):
        total_reward = 0
        is_win = False
        
        while True:
            landed_crystals = self._apply_gravity()
            for x, y in landed_crystals:
                sx, sy = self._iso_to_screen(x, y)
                for _ in range(3): self.particles.append(Particle(sx, sy + self.TILE_DEPTH/2, self.COLOR_GRID, lifetime=15, gravity=0.2))
                # Sound: *thud*

            matches = self._find_matches()
            if not matches:
                break
            
            match_reward, is_win = self._process_matches(matches)
            total_reward += match_reward
            if is_win:
                break
            
            self._spawn_new_crystals()
        
        self.crystals = self._get_all_crystals()
        if self.crystals:
            self.selected_crystal_idx = min(self.selected_crystal_idx, len(self.crystals) - 1)
        else:
            self.selected_crystal_idx = -1
            
        return total_reward, is_win

    def _apply_gravity(self):
        landed_crystals = []
        moved = True
        while moved:
            moved = False
            for y in range(self.GRID_HEIGHT - 2, -1, -1):
                for x in range(self.GRID_WIDTH):
                    if self.grid[y][x] is not None and self.grid[y+1][x] is None:
                        crystal = self.grid[y][x]
                        self.grid[y+1][x] = crystal
                        self.grid[y][x] = None
                        crystal.y += 1
                        moved = True
                        if (y + 1 == self.GRID_HEIGHT - 1) or (y + 2 < self.GRID_HEIGHT and self.grid[y+2][x] is not None):
                             landed_crystals.append((crystal.x, crystal.y))
        return landed_crystals

    def _find_matches(self):
        matches = []
        # Horizontal
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                c1 = self.grid[y][x]
                if c1 is None: continue
                match = [(c1.x, c1.y)]
                for i in range(1, self.GRID_WIDTH - x):
                    c2 = self.grid[y][x+i]
                    if c2 and c2.color_index == c1.color_index:
                        match.append((c2.x, c2.y))
                    else:
                        break
                if len(match) >= 3: matches.append(match)
        # Vertical
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                c1 = self.grid[y][x]
                if c1 is None: continue
                match = [(c1.x, c1.y)]
                for i in range(1, self.GRID_HEIGHT - y):
                    c2 = self.grid[y+i][x]
                    if c2 and c2.color_index == c1.color_index:
                        match.append((c2.x, c2.y))
                    else:
                        break
                if len(match) >= 3: matches.append(match)
        
        # Deduplicate matches
        unique_matches = []
        seen_coords = set()
        for match in sorted(matches, key=len, reverse=True):
            is_new = False
            for coord in match:
                if coord not in seen_coords:
                    is_new = True
                    seen_coords.add(coord)
            if is_new:
                unique_matches.append(match)
        return unique_matches

    def _process_matches(self, matches):
        reward = 0
        is_win = False
        for match in matches:
            match_len = len(match)
            if match_len == 3: reward += 3
            elif match_len == 4: reward += 10
            elif match_len >= 5:
                reward += 100
                is_win = True
            
            self.last_match_info.append((match, self.CRYSTAL_COLORS[self.grid[match[0][1]][match[0][0]].color_index]))

            for x, y in match:
                crystal = self.grid[y][x]
                if crystal:
                    sx, sy = self._iso_to_screen(x, y)
                    color = self.CRYSTAL_COLORS[crystal.color_index]
                    for _ in range(15): self.particles.append(Particle(sx, sy, color, lifetime=25))
                    # Sound: *match success*
                    self.grid[y][x] = None
        
        self.score += reward
        return reward, is_win

    def _spawn_new_crystals(self):
        for x in range(self.GRID_WIDTH):
            if self.grid[0][x] is None:
                color_idx = self.np_random.integers(0, len(self.CRYSTAL_COLORS))
                self.grid[0][x] = Crystal(x, 0, color_idx)

    def _get_all_crystals(self):
        crystals = []
        for row in self.grid:
            for cell in row:
                if cell:
                    crystals.append(cell)
        # Sort top-to-bottom, then left-to-right for consistent selection order
        crystals.sort(key=lambda c: (c.y, c.x))
        return crystals

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _check_termination(self):
        return self.game_over or self.steps >= self.max_steps

    def _iso_to_screen(self, x, y):
        sx = self.origin_x + (x - y) * self.TILE_WIDTH / 2
        sy = self.origin_y + (x + y) * self.TILE_HEIGHT / 2
        return int(sx), int(sy)
    
    def _draw_iso_cube(self, surface, pos, color):
        x, y = pos
        top_color = color
        left_color = tuple(max(0, c - 40) for c in color)
        right_color = tuple(max(0, c - 20) for c in color)

        points_top = [
            (x, y - self.TILE_HEIGHT // 2),
            (x + self.TILE_WIDTH // 2, y),
            (x, y + self.TILE_HEIGHT // 2),
            (x - self.TILE_WIDTH // 2, y)
        ]
        points_left = [
            (x - self.TILE_WIDTH // 2, y),
            (x, y + self.TILE_HEIGHT // 2),
            (x, y + self.TILE_HEIGHT // 2 + self.TILE_DEPTH),
            (x - self.TILE_WIDTH // 2, y + self.TILE_DEPTH)
        ]
        points_right = [
            (x + self.TILE_WIDTH // 2, y),
            (x, y + self.TILE_HEIGHT // 2),
            (x, y + self.TILE_HEIGHT // 2 + self.TILE_DEPTH),
            (x + self.TILE_WIDTH // 2, y + self.TILE_DEPTH)
        ]
        
        pygame.gfxdraw.filled_polygon(surface, points_left, left_color)
        pygame.gfxdraw.aapolygon(surface, points_left, left_color)
        pygame.gfxdraw.filled_polygon(surface, points_right, right_color)
        pygame.gfxdraw.aapolygon(surface, points_right, right_color)
        pygame.gfxdraw.filled_polygon(surface, points_top, top_color)
        pygame.gfxdraw.aapolygon(surface, points_top, top_color)

    def _render_game(self):
        # Draw grid floor
        for y in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # Draw match highlights from previous step
        for match, color in self.last_match_info:
            for x, y in match:
                sx, sy = self._iso_to_screen(x, y)
                sy += self.TILE_DEPTH // 2
                pygame.draw.circle(self.screen, self.COLOR_WHITE, (sx, sy), self.TILE_WIDTH // 2, 2)

        # Draw crystals
        for crystal in self.crystals:
            sx, sy = self._iso_to_screen(crystal.x, crystal.y)
            color = self.CRYSTAL_COLORS[crystal.color_index]
            self._draw_iso_cube(self.screen, (sx, sy), color)
        
        # Draw selector
        if self.crystals and self.selected_crystal_idx != -1 and not self.game_over:
            crystal = self.crystals[self.selected_crystal_idx]
            sx, sy = self._iso_to_screen(crystal.x, crystal.y)
            
            pulse = (math.sin(self.steps * 0.3) + 1) / 2
            alpha = 100 + int(155 * pulse)
            
            points = [
                (sx, sy - self.TILE_HEIGHT // 2 - 4),
                (sx + self.TILE_WIDTH // 2 + 4, sy),
                (sx, sy + self.TILE_HEIGHT // 2 + 4),
                (sx - self.TILE_WIDTH // 2 - 4, sy)
            ]
            
            # Use a temporary surface for the alpha-blended polygon
            sel_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.gfxdraw.aapolygon(sel_surf, points, self.COLOR_HIGHLIGHT + (alpha,))
            pygame.gfxdraw.aapolygon(sel_surf, [(p[0], p[1]+1) for p in points], self.COLOR_HIGHLIGHT + (alpha,))
            self.screen.blit(sel_surf, (0,0))


        # Update and draw particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()
            p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time/Steps
        steps_left = self.max_steps - self.steps
        time_color = self.COLOR_TEXT if steps_left > 200 else (255, 100, 100)
        time_text = self.font_main.render(f"MOVES: {steps_left}", True, time_color)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win_state else "GAME OVER"
            color = (100, 255, 100) if self.win_state else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Need to reset to initialize np_random
        self.reset()
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")