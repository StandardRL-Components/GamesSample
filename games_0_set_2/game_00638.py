
# Generated: 2025-08-27T14:18:08.763341
# Source Brief: brief_00638.md
# Brief Index: 638

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to place a crystal. Shift to give up."
    )

    game_description = (
        "Navigate a crystalline cavern, placing crystals to refract light and illuminate all gems before time runs out."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_WALL = (40, 45, 60)
    COLOR_WALL_TOP = (60, 65, 80)
    COLOR_GRID = (25, 30, 45)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CRYSTAL = (0, 200, 255)
    COLOR_LIGHT_BEAM = (255, 255, 220)
    COLOR_LIGHT_GLOW = (255, 255, 180, 50)
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_UI_TEXT = (255, 255, 255)
    
    # Grid and Isometric Projection
    GRID_WIDTH = 20
    GRID_HEIGHT = 12
    TILE_W_HALF = 16
    TILE_H_HALF = 8
    
    # Game Parameters
    GAME_DURATION_SECONDS = 60
    FPS = 30
    
    # Grid Cell Types
    CELL_EMPTY = 0
    CELL_WALL = 1
    CELL_SOURCE = 2
    CELL_GEM = 3
    CELL_CRYSTAL = 4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width = 640
        self.screen_height = 400
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.origin_x = self.screen_width // 2
        self.origin_y = 80

        # These will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = None
        self.cursor_pos = None
        self.gems = None
        self.lit_gems = None
        self.total_gems = 0
        self.num_crystals = 0
        self.time_left = 0
        self.light_paths = []
        self.particles = []
        self.previous_space_held = False
        self.last_lit_gem_count = 0
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.GAME_DURATION_SECONDS * self.FPS
        self.previous_space_held = False
        self.last_lit_gem_count = 0
        self.particles = []

        self._generate_level()

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self._calculate_light_paths()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.grid = np.full((self.GRID_WIDTH, self.GRID_HEIGHT), self.CELL_EMPTY, dtype=int)
        
        # Create border walls
        self.grid[0, :] = self.CELL_WALL
        self.grid[-1, :] = self.CELL_WALL
        self.grid[:, 0] = self.CELL_WALL
        self.grid[:, -1] = self.CELL_WALL
        
        # Place light source
        source_y = self.np_random.integers(2, self.GRID_HEIGHT - 2)
        self.grid[0, source_y] = self.CELL_SOURCE
        self.source_pos = (0, source_y)
        self.source_dir = (1, 0)
        
        # Procedurally place gems and guarantee a solution path
        self.gems = []
        required_crystals = 0
        
        # Place a few internal walls
        for _ in range(self.np_random.integers(3, 6)):
            wx, wy = self.np_random.integers(2, self.GRID_WIDTH - 2), self.np_random.integers(2, self.GRID_HEIGHT - 2)
            if self.grid[wx, wy] == self.CELL_EMPTY:
                self.grid[wx, wy] = self.CELL_WALL

        # Place gems
        num_gems = self.np_random.integers(3, 6)
        for _ in range(num_gems):
            placed = False
            while not placed:
                gx, gy = self.np_random.integers(1, self.GRID_WIDTH - 1), self.np_random.integers(1, self.GRID_HEIGHT - 1)
                if self.grid[gx, gy] == self.CELL_EMPTY:
                    self.grid[gx, gy] = self.CELL_GEM
                    self.gems.append((gx, gy))
                    placed = True
        
        self.total_gems = len(self.gems)
        # A simple heuristic for crystal count
        self.num_crystals = self.total_gems + self.np_random.integers(1, 3)

    def _to_iso(self, x, y):
        iso_x = self.origin_x + (x - y) * self.TILE_W_HALF
        iso_y = self.origin_y + (x + y) * self.TILE_H_HALF
        return int(iso_x), int(iso_y)

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01  # Cost of time
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update Game Logic ---
        self.time_left -= 1
        
        # 1. Handle cursor movement
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        new_cursor_x = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_WIDTH - 1)
        new_cursor_y = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_HEIGHT - 1)
        self.cursor_pos = [new_cursor_x, new_cursor_y]
        
        # 2. Handle "give up" action
        if shift_held:
            terminated = True
            self.game_over = True
            reward -= 100 # Large penalty for giving up

        # 3. Handle "place crystal" action
        if space_held and not self.previous_space_held:
            cx, cy = self.cursor_pos
            if self.num_crystals > 0 and self.grid[cx, cy] == self.CELL_EMPTY:
                self.grid[cx, cy] = self.CELL_CRYSTAL
                self.num_crystals -= 1
                self._calculate_light_paths()
                self._create_particles(self.cursor_pos, self.COLOR_CRYSTAL, 20)
                # sfx: crystal_place.wav
            else:
                # sfx: action_fail.wav
                pass # Optional: add feedback for failed placement

        self.previous_space_held = space_held
        
        # 4. Update particles
        self._update_particles()
        
        # 5. Calculate rewards based on new state
        newly_lit_count = len(self.lit_gems) - self.last_lit_gem_count
        if newly_lit_count > 0:
            reward += newly_lit_count * 0.1
            self.last_lit_gem_count = len(self.lit_gems)
            # sfx: gem_lit.wav
        
        # 6. Check for termination conditions
        all_gems_lit = len(self.lit_gems) == self.total_gems
        
        if not terminated:
            if all_gems_lit and self.total_gems > 0:
                reward += 5 # Bonus for lighting the final gem
                reward += 100 # Win bonus
                terminated = True
                self.game_over = True
                # sfx: level_win.wav
            elif self.time_left <= 0:
                reward -= 100 # Loss penalty
                terminated = True
                self.game_over = True
                # sfx: level_lose.wav
            elif self.num_crystals == 0 and not all_gems_lit:
                # Check if light paths are stable and no more gems can be lit
                reward -= 100
                terminated = True
                self.game_over = True
                # sfx: level_lose.wav

        self.score += reward
        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_light_paths(self):
        self.light_paths = []
        self.lit_gems = set()
        
        q = collections.deque()
        
        # Initial beam from source
        start_pos = tuple(p + d for p, d in zip(self.source_pos, self.source_dir))
        q.append((start_pos, self.source_dir))

        visited_crystal_splits = set()

        while q:
            pos, direction = q.popleft()
            current_path = [self._to_iso(pos[0] - direction[0], pos[1] - direction[1])]
            
            for _ in range(max(self.GRID_WIDTH, self.GRID_HEIGHT) * 2):
                if not (0 <= pos[0] < self.GRID_WIDTH and 0 <= pos[1] < self.GRID_HEIGHT):
                    break
                
                cell_type = self.grid[pos[0], pos[1]]
                current_path.append(self._to_iso(pos[0], pos[1]))

                if cell_type == self.CELL_WALL:
                    break
                
                elif cell_type == self.CELL_GEM:
                    if pos not in self.lit_gems:
                        self.lit_gems.add(pos)
                        self._create_particles(pos, (255,255,255), 15)
                
                elif cell_type == self.CELL_CRYSTAL:
                    # Crystal refracts light 90 degrees, splitting it
                    # To prevent infinite loops, only split once per crystal per direction
                    h_dir = (direction[0], direction[1], 0) # Horizontal incoming
                    v_dir = (direction[0], direction[1], 1) # Vertical incoming

                    if direction[0] != 0: # Horizontal beam
                        if (pos, h_dir) not in visited_crystal_splits:
                            visited_crystal_splits.add((pos, h_dir))
                            q.append(((pos[0], pos[1] + 1), (0, 1))) # Down
                            q.append(((pos[0], pos[1] - 1), (0, -1))) # Up
                    else: # Vertical beam
                        if (pos, v_dir) not in visited_crystal_splits:
                            visited_crystal_splits.add((pos, v_dir))
                            q.append(((pos[0] + 1, pos[1]), (1, 0))) # Right
                            q.append(((pos[0] - 1, pos[1]), (-1, 0))) # Left
                    break
                
                pos = (pos[0] + direction[0], pos[1] + direction[1])
            
            if len(current_path) > 1:
                self.light_paths.append(current_path)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid floor
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._to_iso(x, y)
                p2 = self._to_iso(x + 1, y)
                p3 = self._to_iso(x + 1, y + 1)
                p4 = self._to_iso(x, y + 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

        # Render elements
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                cell_type = self.grid[x, y]
                if cell_type == self.CELL_WALL:
                    self._draw_wall_block(x, y)
                elif cell_type == self.CELL_GEM:
                    is_lit = (x, y) in self.lit_gems
                    self._draw_gem(x, y, is_lit)
                elif cell_type == self.CELL_CRYSTAL:
                    self._draw_crystal(x, y)
                elif cell_type == self.CELL_SOURCE:
                    self._draw_source(x, y)

        # Render light paths
        for path in self.light_paths:
            if len(path) > 1:
                pygame.draw.lines(self.screen, self.COLOR_LIGHT_GLOW, False, path, width=5)
                pygame.draw.lines(self.screen, self.COLOR_LIGHT_BEAM, False, path, width=1)
        
        # Render particles
        for p in self.particles:
            p_pos = self._to_iso(p['pos'][0], p['pos'][1])
            pygame.draw.circle(self.screen, p['color'], p_pos, int(p['size']))
        
        # Render cursor
        cx, cy = self.cursor_pos
        p1 = self._to_iso(cx, cy)
        p2 = self._to_iso(cx + 1, cy)
        p3 = self._to_iso(cx + 1, cy + 1)
        p4 = self._to_iso(cx, cy + 1)
        
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 * 155 + 100
        cursor_color = (pulse, pulse, 0)
        pygame.draw.polygon(self.screen, cursor_color, [p1, p2, p3, p4], 2)

    def _draw_wall_block(self, x, y):
        iso_x, iso_y = self._to_iso(x, y)
        top_points = [
            (iso_x, iso_y),
            (iso_x + self.TILE_W_HALF, iso_y + self.TILE_H_HALF),
            (iso_x, iso_y + self.TILE_H_HALF * 2),
            (iso_x - self.TILE_W_HALF, iso_y + self.TILE_H_HALF)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_WALL_TOP, top_points)
        
        side_points1 = [
            (iso_x, iso_y + self.TILE_H_HALF * 2),
            (iso_x - self.TILE_W_HALF, iso_y + self.TILE_H_HALF),
            (iso_x - self.TILE_W_HALF, iso_y + self.TILE_H_HALF + 20),
            (iso_x, iso_y + self.TILE_H_HALF * 2 + 20)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_WALL, side_points1)

        side_points2 = [
            (iso_x, iso_y + self.TILE_H_HALF * 2),
            (iso_x + self.TILE_W_HALF, iso_y + self.TILE_H_HALF),
            (iso_x + self.TILE_W_HALF, iso_y + self.TILE_H_HALF + 20),
            (iso_x, iso_y + self.TILE_H_HALF * 2 + 20)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_WALL, side_points2)

    def _draw_gem(self, x, y, is_lit):
        iso_x, iso_y = self._to_iso(x, y)
        pos = (iso_x, iso_y + self.TILE_H_HALF)
        
        if is_lit:
            color = (255, 80, 200)
            glow_color = (255, 120, 220, 100)
            size = 6
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 4, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)
        else:
            color = (100, 40, 90)
            size = 4
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)

    def _draw_crystal(self, x, y):
        iso_x, iso_y = self._to_iso(x, y)
        pos = (iso_x, iso_y + self.TILE_H_HALF)
        
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, (0, 150, 200, 60))
        
        # Crystal shape
        points = [
            (pos[0], pos[1] - 6),
            (pos[0] + 5, pos[1]),
            (pos[0], pos[1] + 6),
            (pos[0] - 5, pos[1]),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL)

    def _draw_source(self, x, y):
        iso_x, iso_y = self._to_iso(x, y)
        pos = (iso_x + self.TILE_W_HALF, iso_y + self.TILE_H_HALF)
        pygame.draw.rect(self.screen, self.COLOR_LIGHT_BEAM, (pos[0], pos[1]-3, 8, 6))

    def _create_particles(self, grid_pos, color, count):
        for _ in range(count):
            self.particles.append({
                'pos': [grid_pos[0] + 0.5, grid_pos[1] + 0.5],
                'vel': [self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.2, 0.2)],
                'size': self.np_random.uniform(1, 4),
                'life': self.np_random.integers(10, 20),
                'color': color,
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['size'] -= 0.1
            p['life'] -= 1
            if p['size'] > 0 and p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _render_ui(self):
        ui_panel = pygame.Surface((self.screen_width, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        time_str = f"TIME: {max(0, self.time_left / self.FPS):.1f}"
        crystals_str = f"CRYSTALS: {self.num_crystals}"
        gems_str = f"GEMS: {len(self.lit_gems)}/{self.total_gems}"
        score_str = f"SCORE: {int(self.score)}"
        
        time_surf = self.font_small.render(time_str, True, self.COLOR_UI_TEXT)
        crystals_surf = self.font_small.render(crystals_str, True, self.COLOR_UI_TEXT)
        gems_surf = self.font_small.render(gems_str, True, self.COLOR_UI_TEXT)
        score_surf = self.font_small.render(score_str, True, self.COLOR_UI_TEXT)

        self.screen.blit(time_surf, (10, 10))
        self.screen.blit(crystals_surf, (160, 10))
        self.screen.blit(gems_surf, (310, 10))
        self.screen.blit(score_surf, (460, 10))

        if self.game_over:
            win = len(self.lit_gems) == self.total_gems and self.total_gems > 0
            msg = "LEVEL COMPLETE" if win else "GAME OVER"
            color = (100, 255, 100) if win else (255, 100, 100)
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "crystals_left": self.num_crystals,
            "gems_lit": len(self.lit_gems),
            "total_gems": self.total_gems,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play ---
    # This block allows you to play the game manually.
    # Close the Pygame window to exit.
    
    # Re-initialize pygame for display
    pygame.display.init()
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        movement, space, shift = 0, 0, 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward}")
            print("Resetting in 3 seconds...")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

    env.close()