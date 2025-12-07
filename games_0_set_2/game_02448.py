
# Generated: 2025-08-27T20:23:56.363656
# Source Brief: brief_02448.md
# Brief Index: 2448

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to place a crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a crystal cavern by placing crystals to redirect lasers and activate all three exit receptors."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Grid
    GRID_WIDTH = 20
    GRID_HEIGHT = 14
    TILE_W_HALF = 20
    TILE_H_HALF = 10
    
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_WALL = (40, 30, 60)
    COLOR_WALL_TOP = (60, 50, 80)
    COLOR_FLOOR = (25, 20, 40)
    COLOR_CURSOR = (255, 255, 255, 100)
    COLOR_CRYSTAL = (0, 200, 255)
    COLOR_CRYSTAL_GLOW = (0, 200, 255, 50)
    COLOR_LASER = (255, 20, 90)
    COLOR_LASER_GLOW = (255, 20, 90, 80)
    COLOR_SOURCE = (255, 150, 0)
    COLOR_RECEPTOR = (0, 100, 50)
    COLOR_RECEPTOR_ACTIVE = (50, 255, 150)
    COLOR_TEXT = (220, 220, 240)
    
    # Tile Types
    EMPTY, WALL, SOURCE, RECEPTOR_OFF, RECEPTOR_ON, CRYSTAL = 0, 1, 2, 3, 4, 5
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        self.grid_offset_x = self.SCREEN_WIDTH // 2
        self.grid_offset_y = 80

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.cursor_pos = np.array([0, 0])
        self.laser_path = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.initial_crystals = 15
        self.crystals_left = self.initial_crystals
        self.receptors_activated_count = 0
        
        self._generate_level()
        self.cursor_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        self._calculate_laser_path()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01  # Small penalty for taking a step
        
        # --- Handle Movement ---
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        # --- Handle Crystal Placement ---
        if space_pressed:
            # sfx: crystal_place_try
            x, y = self.cursor_pos
            if self.grid[x, y] == self.EMPTY and self.crystals_left > 0:
                # sfx: crystal_place_success
                self.grid[x, y] = self.CRYSTAL
                self.crystals_left -= 1
                reward -= 1.0 # Penalty for using a resource
                self._calculate_laser_path()
                
                new_activations = self._count_active_receptors()
                if new_activations > self.receptors_activated_count:
                    # sfx: receptor_activate
                    reward += 10.0 * (new_activations - self.receptors_activated_count)
                    self.receptors_activated_count = new_activations

        self.steps += 1
        
        # --- Check Termination ---
        win = self.receptors_activated_count == 3
        loss_crystals = self.crystals_left <= 0 and not win
        loss_steps = self.steps >= 1000
        
        terminated = win or loss_crystals or loss_steps
        
        if win:
            # sfx: win_jingle
            reward += 100.0
        elif loss_crystals or loss_steps:
            # sfx: lose_sound
            reward -= 100.0

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        # Start with a solid grid of walls
        self.grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int) * self.WALL
        
        # Carve out a cavern using random walk
        start_x, start_y = self.rng.integers(1, self.GRID_WIDTH - 1), self.rng.integers(1, self.GRID_HEIGHT - 1)
        q = deque([(start_x, start_y)])
        self.grid[start_x, start_y] = self.EMPTY
        carved_tiles = 1
        max_tiles = int(self.GRID_WIDTH * self.GRID_HEIGHT * 0.6) # Carve about 60% of the area

        while q and carved_tiles < max_tiles:
            x, y = q.popleft()
            
            # Get neighbors
            neighbors = [(x + 2, y), (x - 2, y), (x, y + 2), (x, y - 2)]
            self.rng.shuffle(neighbors)
            
            for nx, ny in neighbors:
                if 0 < nx < self.GRID_WIDTH -1 and 0 < ny < self.GRID_HEIGHT -1 and self.grid[nx, ny] == self.WALL:
                    # Carve path to neighbor
                    self.grid[nx, ny] = self.EMPTY
                    self.grid[x + (nx - x) // 2, y + (ny - y) // 2] = self.EMPTY
                    q.append((nx, ny))
                    carved_tiles += 2
        
        # Get all valid empty spots
        empty_spots = list(zip(*np.where(self.grid == self.EMPTY)))
        self.rng.shuffle(empty_spots)
        
        # Place laser source
        source_pos = empty_spots.pop()
        self.grid[source_pos] = self.SOURCE
        self.source_pos = np.array(source_pos)
        
        # Place 3 receptors
        self.receptor_pos = []
        for _ in range(3):
            if not empty_spots: break
            receptor_pos = empty_spots.pop()
            self.grid[receptor_pos] = self.RECEPTOR_OFF
            self.receptor_pos.append(np.array(receptor_pos))

    def _calculate_laser_path(self):
        self.laser_path = []
        
        # Reset all receptors to OFF
        for r_pos in self.receptor_pos:
            self.grid[r_pos[0], r_pos[1]] = self.RECEPTOR_OFF

        # Initial laser state from source
        # Determine initial direction based on position to avoid immediate wall collision
        x, y = self.source_pos
        if x > 1 and self.grid[x - 1, y] != self.WALL:
            direction = np.array([-1, 0])
        elif x < self.GRID_WIDTH - 2 and self.grid[x + 1, y] != self.WALL:
            direction = np.array([1, 0])
        elif y > 1 and self.grid[x, y - 1] != self.WALL:
            direction = np.array([0, -1])
        else:
            direction = np.array([0, 1])

        pos = self.source_pos.copy()
        
        current_segment = [pos]
        for _ in range(self.GRID_WIDTH * self.GRID_HEIGHT): # safety break
            pos = pos + direction
            
            if not (0 <= pos[0] < self.GRID_WIDTH and 0 <= pos[1] < self.GRID_HEIGHT):
                break # Hit edge of map

            tile = self.grid[pos[0], pos[1]]
            current_segment.append(pos.copy())

            if tile == self.WALL:
                break
            
            elif tile == self.CRYSTAL:
                # Reflect 90 degrees. (dx, dy) -> (-dy, -dx)
                direction = np.array([-direction[1], -direction[0]])
                self.laser_path.append(np.array(current_segment))
                current_segment = [pos.copy()]
                # sfx: laser_reflect
            
            elif tile == self.RECEPTOR_OFF:
                self.grid[pos[0], pos[1]] = self.RECEPTOR_ON
                # Laser passes through receptor
        
        self.laser_path.append(np.array(current_segment))

    def _count_active_receptors(self):
        count = 0
        for r_pos in self.receptor_pos:
            if self.grid[r_pos[0], r_pos[1]] == self.RECEPTOR_ON:
                count += 1
        return count

    def _world_to_iso(self, x, y):
        iso_x = self.grid_offset_x + (x - y) * self.TILE_W_HALF
        iso_y = self.grid_offset_y + (x + y) * self.TILE_H_HALF
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, x, y, color_top, color_side):
        px, py = self._world_to_iso(x, y)
        
        points_top = [
            (px, py - self.TILE_H_HALF),
            (px + self.TILE_W_HALF, py),
            (px, py + self.TILE_H_HALF),
            (px - self.TILE_W_HALF, py)
        ]
        pygame.gfxdraw.filled_polygon(surface, points_top, color_top)
        pygame.gfxdraw.aapolygon(surface, points_top, color_top)

        points_left = [
            (px - self.TILE_W_HALF, py),
            (px, py + self.TILE_H_HALF),
            (px, py + self.TILE_H_HALF + self.TILE_H_HALF*2),
            (px - self.TILE_W_HALF, py + self.TILE_H_HALF*2)
        ]
        pygame.gfxdraw.filled_polygon(surface, points_left, color_side)
        pygame.gfxdraw.aapolygon(surface, points_left, color_side)

        points_right = [
            (px, py + self.TILE_H_HALF),
            (px + self.TILE_W_HALF, py),
            (px + self.TILE_W_HALF, py + self.TILE_H_HALF*2),
            (px, py + self.TILE_H_HALF + self.TILE_H_HALF*2)
        ]
        pygame.gfxdraw.filled_polygon(surface, points_right, color_side)
        pygame.gfxdraw.aapolygon(surface, points_right, color_side)

    def _render_game(self):
        # Render grid tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                tile = self.grid[x, y]
                px, py = self._world_to_iso(x, y)
                
                # Draw floor tile
                points_floor = [
                    (px, py + self.TILE_H_HALF),
                    (px + self.TILE_W_HALF, py + self.TILE_H_HALF*2),
                    (px, py + self.TILE_H_HALF*3),
                    (px - self.TILE_W_HALF, py + self.TILE_H_HALF*2)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points_floor, self.COLOR_FLOOR)
                
                if tile == self.WALL:
                    self._draw_iso_cube(self.screen, x, y, self.COLOR_WALL_TOP, self.COLOR_WALL)
        
        # Render objects after all floors/walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                tile = self.grid[x, y]
                px_c, py_c = self._world_to_iso(x, y)
                py_c += self.TILE_H_HALF * 2 # Center on floor tile

                if tile == self.SOURCE:
                    pygame.gfxdraw.filled_circle(self.screen, px_c, py_c, 10, self.COLOR_SOURCE)
                    pygame.gfxdraw.aacircle(self.screen, px_c, py_c, 10, self.COLOR_SOURCE)
                elif tile == self.RECEPTOR_OFF:
                    pygame.gfxdraw.filled_circle(self.screen, px_c, py_c, 8, self.COLOR_RECEPTOR)
                elif tile == self.RECEPTOR_ON:
                    # Glow effect
                    pygame.gfxdraw.filled_circle(self.screen, px_c, py_c, 12, (self.COLOR_RECEPTOR_ACTIVE[0], self.COLOR_RECEPTOR_ACTIVE[1], self.COLOR_RECEPTOR_ACTIVE[2], 50))
                    pygame.gfxdraw.filled_circle(self.screen, px_c, py_c, 8, self.COLOR_RECEPTOR_ACTIVE)
                elif tile == self.CRYSTAL:
                    # Draw glow
                    glow_surf = pygame.Surface((self.TILE_W_HALF*4, self.TILE_W_HALF*4), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, self.COLOR_CRYSTAL_GLOW, (self.TILE_W_HALF*2, self.TILE_W_HALF*2), self.TILE_W_HALF*1.5)
                    self.screen.blit(glow_surf, (px_c - self.TILE_W_HALF*2, py_c - self.TILE_W_HALF*2))
                    # Draw crystal
                    self._draw_iso_cube(self.screen, x, y, self.COLOR_CRYSTAL, tuple(c*0.7 for c in self.COLOR_CRYSTAL))

        # Render lasers
        for segment in self.laser_path:
            if len(segment) < 2: continue
            iso_points = []
            for p in segment:
                px, py = self._world_to_iso(p[0], p[1])
                iso_points.append((px, py + self.TILE_H_HALF * 2)) # Center on floor
            
            # Glow
            pygame.draw.lines(self.screen, self.COLOR_LASER_GLOW, False, iso_points, 7)
            # Core beam
            pygame.draw.lines(self.screen, self.COLOR_LASER, False, iso_points, 3)

        # Render cursor
        cx, cy = self.cursor_pos
        if self.grid[cx, cy] != self.WALL:
            px, py = self._world_to_iso(cx, cy)
            points_cursor = [
                (px, py + self.TILE_H_HALF),
                (px + self.TILE_W_HALF, py + self.TILE_H_HALF*2),
                (px, py + self.TILE_H_HALF*3),
                (px - self.TILE_W_HALF, py + self.TILE_H_HALF*2)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points_cursor, self.COLOR_CURSOR)
            pygame.gfxdraw.aapolygon(self.screen, points_cursor, (255,255,255))
    
    def _render_ui(self):
        # Crystals left
        crystal_text = self.font.render(f"Crystals: {self.crystals_left}", True, self.COLOR_TEXT)
        self.screen.blit(crystal_text, (10, 10))
        
        # Score
        score_text = self.font.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        # Receptors activated
        receptor_text = self.font.render(f"Receptors: {self.receptors_activated_count} / 3", True, self.COLOR_TEXT)
        receptor_rect = receptor_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 35))
        self.screen.blit(receptor_text, receptor_rect)

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
            "crystals_left": self.crystals_left,
            "receptors_activated": self.receptors_activated_count,
        }

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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Crystal Cavern")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0  # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Draw the observation to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        # Since auto_advance is False, we need a delay to make it playable by humans
        clock.tick(10) # Limit to 10 actions per second for human play
        
    env.close()