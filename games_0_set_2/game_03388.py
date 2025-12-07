
# Generated: 2025-08-27T23:13:31.360374
# Source Brief: brief_03388.md
# Brief Index: 3388

        
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
    """
    A Gymnasium environment for a procedurally generated grid maze game.

    The player must navigate a 10x10 maze to find the exit within a limited
    number of steps. The game is turn-based, with rewards encouraging efficient
    pathfinding while penalizing risky moves next to walls. The visual design
    is minimalist and geometric, with clear color-coding for different game
    elements and smooth animations for a polished user experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use Arrow Keys (↑, ↓, ←, →) to navigate the maze. "
        "Reach the green exit tile before you run out of steps."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze to reach the exit. Each move "
        "costs a step. Orange tiles are risky; entering them costs score. "
        "Find the exit for a big score bonus."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_SIZE = 10
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TURN_LIMIT = 25
    MIN_PATH_DISTANCE = 8

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_EMPTY = (40, 50, 75)
    COLOR_WALL = (10, 12, 20)
    COLOR_PLAYER = (255, 215, 0) # Gold
    COLOR_EXIT = (0, 255, 127) # Spring Green
    COLOR_RISKY = (255, 100, 0) # Orange
    COLOR_VISITED = (60, 75, 110)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    # Tile types
    TILE_EMPTY = 0
    TILE_WALL = 1

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
        
        # --- Rendering setup ---
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Centered grid rendering
        self.tile_size = self.SCREEN_HEIGHT // self.GRID_SIZE
        self.grid_width = self.tile_size * self.GRID_SIZE
        self.grid_height = self.tile_size * self.GRID_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2

        # --- Game State ---
        self.grid = None
        self.player_pos = None
        self.exit_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.visited_tiles = None
        self.particles = []
        self.last_move_time = 0

        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        
        self._generate_maze()
        
        empty_cells = np.argwhere(self.grid == self.TILE_EMPTY)
        
        while True:
            player_idx, exit_idx = self.np_random.choice(len(empty_cells), 2, replace=False)
            self.player_pos = tuple(empty_cells[player_idx])
            self.exit_pos = tuple(empty_cells[exit_idx])
            
            dist = self._manhattan_distance(self.player_pos, self.exit_pos)
            if dist >= self.MIN_PATH_DISTANCE:
                break
        
        self.visited_tiles = {self.player_pos}
        self.last_player_pos = self.player_pos
        self.prev_manhattan_dist = self._manhattan_distance(self.player_pos, self.exit_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        px, py = self.player_pos
        nx, ny = px, py

        if movement == 1: # Up
            nx -= 1
        elif movement == 2: # Down
            nx += 1
        elif movement == 3: # Left
            ny -= 1
        elif movement == 4: # Right
            ny += 1

        # Check for valid move
        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid[nx, ny] == self.TILE_EMPTY:
            self.last_player_pos = self.player_pos
            self.player_pos = (nx, ny)
            self.visited_tiles.add(self.player_pos)
            self.last_move_time = pygame.time.get_ticks()
            self._create_move_particles(self.last_player_pos)
            
            # --- Calculate Reward ---
            # 1. Distance-based reward
            new_dist = self._manhattan_distance(self.player_pos, self.exit_pos)
            if new_dist < self.prev_manhattan_dist:
                reward += 1.0
            else:
                reward -= 1.0
            self.prev_manhattan_dist = new_dist
            
            # 2. Risky tile penalty
            if self._is_risky(self.player_pos):
                reward -= 5.0

        self.steps += 1
        self.score += reward
        
        # --- Check Termination ---
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 50.0
            self.score += 50.0
            terminated = True
            # sfx: win_sound
        elif self.steps >= self.TURN_LIMIT:
            terminated = True
            # sfx: lose_sound
            
        if terminated:
            self.game_over = True

        self._update_particles()

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
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def _generate_maze(self):
        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), self.TILE_WALL, dtype=np.int8)
        
        # Use randomized DFS for maze generation
        start_x = self.np_random.integers(0, self.GRID_SIZE // 2) * 2
        start_y = self.np_random.integers(0, self.GRID_SIZE // 2) * 2
        
        stack = deque([(start_x, start_y)])
        self.grid[start_x, start_y] = self.TILE_EMPTY
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid[nx, ny] == self.TILE_WALL:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = neighbors[self.np_random.integers(0, len(neighbors))]
                # Carve path
                self.grid[(cx + nx) // 2, (cy + ny) // 2] = self.TILE_EMPTY
                self.grid[nx, ny] = self.TILE_EMPTY
                stack.append((nx, ny))
            else:
                stack.pop()
                
        # Cache risky tiles for rendering
        self.risky_cache = {
            (r, c) for r in range(self.GRID_SIZE) for c in range(self.GRID_SIZE) 
            if self.grid[r, c] == self.TILE_EMPTY and self._is_risky((r, c))
        }

    def _is_risky(self, pos):
        r, c = pos
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE) or self.grid[nr, nc] == self.TILE_WALL:
                return True
        return False

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _render_game(self):
        # --- Render Grid ---
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_offset_x + c * self.tile_size,
                    self.grid_offset_y + r * self.tile_size,
                    self.tile_size, self.tile_size
                )
                
                tile_type = self.grid[r, c]
                if tile_type == self.TILE_WALL:
                    color = self.COLOR_WALL
                else:
                    if (r, c) in self.risky_cache:
                        color = self.COLOR_RISKY
                    elif (r, c) in self.visited_tiles:
                        color = self.COLOR_VISITED
                    else:
                        color = self.COLOR_EMPTY
                pygame.draw.rect(self.screen, color, rect)

        # --- Render Particles ---
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

        # --- Render Exit ---
        pulse = (math.sin(pygame.time.get_ticks() * 0.002) + 1) / 2
        exit_r, exit_c = self.exit_pos
        exit_center_x = self.grid_offset_x + exit_c * self.tile_size + self.tile_size // 2
        exit_center_y = self.grid_offset_y + exit_r * self.tile_size + self.tile_size // 2
        radius = int(self.tile_size * 0.3 + pulse * 3)
        pygame.gfxdraw.filled_circle(self.screen, exit_center_x, exit_center_y, radius, self.COLOR_EXIT)
        pygame.gfxdraw.aacircle(self.screen, exit_center_x, exit_center_y, radius, self.COLOR_EXIT)

        # --- Render Player ---
        # Smooth interpolation for player movement
        time_since_move = pygame.time.get_ticks() - self.last_move_time
        interp_duration = 150 # ms
        interp_factor = min(1.0, time_since_move / interp_duration)
        
        start_r, start_c = self.last_player_pos
        end_r, end_c = self.player_pos
        
        player_draw_r = start_r + (end_r - start_r) * interp_factor
        player_draw_c = start_c + (end_c - start_c) * interp_factor

        player_rect = pygame.Rect(
            self.grid_offset_x + player_draw_c * self.tile_size + self.tile_size * 0.15,
            self.grid_offset_y + player_draw_r * self.tile_size + self.tile_size * 0.15,
            self.tile_size * 0.7, self.tile_size * 0.7
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        
        # Glow effect for player
        glow_color = (*self.COLOR_PLAYER, 60)
        glow_surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, glow_color, (self.tile_size//2, self.tile_size//2), self.tile_size//2 * (0.8 + pulse * 0.2))
        self.screen.blit(glow_surface, (player_rect.x - self.tile_size*0.15, player_rect.y - self.tile_size*0.15))

    def _render_ui(self):
        # Helper to render text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            main_text = font.render(text, True, color)
            self.screen.blit(main_text, pos)

        # Score
        draw_text("SCORE", self.font_small, self.COLOR_TEXT, (20, 20))
        draw_text(f"{int(self.score):,}", self.font_large, self.COLOR_PLAYER, (20, 40))

        # Steps
        steps_left = self.TURN_LIMIT - self.steps
        draw_text("STEPS LEFT", self.font_small, self.COLOR_TEXT, (20, 100))
        draw_text(f"{steps_left}", self.font_large, self.COLOR_EXIT, (20, 120))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.player_pos == self.exit_pos:
                msg = "EXIT REACHED!"
                color = self.COLOR_EXIT
            else:
                msg = "OUT OF STEPS!"
                color = self.COLOR_RISKY
                
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _create_move_particles(self, pos):
        # sfx: move_sound
        r, c = pos
        center_x = self.grid_offset_x + c * self.tile_size + self.tile_size // 2
        center_y = self.grid_offset_y + r * self.tile_size + self.tile_size // 2
        
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': random.uniform(2, 5),
                'lifetime': random.randint(15, 30),
                'color': self.COLOR_PLAYER
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['radius'] *= 0.95
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0 and p['radius'] > 0.5]

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
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
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Maze Navigator")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.user_guide)
    
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    
                if not terminated:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
        
        if not terminated and action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
        
        # In manual play, we need to get the observation again for animations
        # even if no action was taken.
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate to 30 FPS

    env.close()