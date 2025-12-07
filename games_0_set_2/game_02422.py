
# Generated: 2025-08-27T20:19:48.514939
# Source Brief: brief_02422.md
# Brief Index: 2422

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Arrow keys to navigate the maze. Collect all the gems before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze to collect gems before time runs out. Plan your route efficiently to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    MAZE_W = 31  # Must be odd
    MAZE_H = 19  # Must be odd
    TILE_SIZE = 20
    
    GAME_FPS = 30
    GAME_DURATION_SECONDS = 60
    
    TOTAL_GEMS = 25
    
    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_WALL = (40, 60, 120)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_GEMS = [(255, 80, 80), (80, 255, 80), (80, 80, 255)]
    COLOR_TEXT = (255, 255, 255)
    
    # Maze constants
    PATH = 0
    WALL = 1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Maze rendering offset to center it
        self.maze_render_offset_x = (self.SCREEN_WIDTH - self.MAZE_W * self.TILE_SIZE) // 2
        self.maze_render_offset_y = (self.SCREEN_HEIGHT - self.MAZE_H * self.TILE_SIZE) // 2

        # Initialize state variables
        self.player_pos = None
        self.gems = None
        self.maze = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_remaining = None
        self.last_dist_to_gem = None
        self.np_random = None

        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.GAME_FPS * self.GAME_DURATION_SECONDS
        
        self._generate_maze()
        self._place_entities()
        
        self.last_dist_to_gem = self._distance_to_nearest_gem(self.player_pos)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update game logic ---
        self.steps += 1
        self.time_remaining -= 1
        
        # 1. Handle player movement
        old_pos = self.player_pos
        px, py = self.player_pos
        
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        
        # Collision detection
        if self.maze[py, px] == self.PATH:
            self.player_pos = (px, py)

        # 2. Check for gem collection
        reward = 0
        gem_collected = False
        if self.player_pos in self.gems:
            self.gems.remove(self.player_pos)
            self.score += 1
            reward += 1.0
            gem_collected = True
            # SFX: Gem collect sound

        # 3. Calculate reward
        # Distance-based reward
        if self.gems:
            dist_to_gem = self._distance_to_nearest_gem(self.player_pos)
            if dist_to_gem < self.last_dist_to_gem:
                reward += 0.1
            else:
                reward -= 0.01
            self.last_dist_to_gem = dist_to_gem
        
        # 4. Check for termination
        terminated = False
        if self.score >= self.TOTAL_GEMS:
            terminated = True
            reward += 100.0  # Victory bonus
            self.game_over = True
        elif self.time_remaining <= 0:
            terminated = True
            reward -= 10.0  # Time out penalty
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_maze()
        self._render_gems()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "player_pos": self.player_pos,
            "gems_remaining": len(self.gems)
        }

    # --- Helper and Rendering Methods ---

    def _generate_maze(self):
        # Maze generation using iterative randomized Depth-First Search
        self.maze = np.ones((self.MAZE_H, self.MAZE_W), dtype=np.uint8)
        
        stack = deque()
        start_x, start_y = (1, 1)
        stack.append((start_x, start_y))
        self.maze[start_y, start_x] = self.PATH

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.MAZE_W -1 and 0 < ny < self.MAZE_H -1 and self.maze[ny, nx] == self.WALL:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                # Carve path to neighbor
                self.maze[ny, nx] = self.PATH
                self.maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = self.PATH
                stack.append((nx, ny))
            else:
                stack.pop()

    def _place_entities(self):
        # Get all valid path locations
        path_locations = []
        for y in range(self.MAZE_H):
            for x in range(self.MAZE_W):
                if self.maze[y, x] == self.PATH:
                    path_locations.append((x, y))

        self.np_random.shuffle(path_locations)

        # Place player
        self.player_pos = path_locations.pop()

        # Place gems
        self.gems = set()
        while len(self.gems) < self.TOTAL_GEMS and path_locations:
            self.gems.add(path_locations.pop())

    def _render_maze(self):
        for y in range(self.MAZE_H):
            for x in range(self.MAZE_W):
                if self.maze[y, x] == self.WALL:
                    rect = pygame.Rect(
                        self.maze_render_offset_x + x * self.TILE_SIZE,
                        self.maze_render_offset_y + y * self.TILE_SIZE,
                        self.TILE_SIZE,
                        self.TILE_SIZE
                    )
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

    def _render_gems(self):
        pulse = math.sin(self.steps * 0.2) * 2
        radius = int(self.TILE_SIZE * 0.3 + pulse)
        
        for i, (gx, gy) in enumerate(self.gems):
            color = self.COLOR_GEMS[i % len(self.COLOR_GEMS)]
            center_x = int(self.maze_render_offset_x + gx * self.TILE_SIZE + self.TILE_SIZE / 2)
            center_y = int(self.maze_render_offset_y + gy * self.TILE_SIZE + self.TILE_SIZE / 2)
            
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)

    def _render_player(self):
        px, py = self.player_pos
        player_rect = pygame.Rect(
            self.maze_render_offset_x + px * self.TILE_SIZE + self.TILE_SIZE * 0.15,
            self.maze_render_offset_y + py * self.TILE_SIZE + self.TILE_SIZE * 0.15,
            self.TILE_SIZE * 0.7,
            self.TILE_SIZE * 0.7
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        # Add a subtle glow
        glow_rect = player_rect.inflate(4, 4)
        glow_color = (*self.COLOR_PLAYER, 100) # (r,g,b,a)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, glow_rect.topleft)

    def _render_ui(self):
        # Render score
        score_text = f"Gems: {self.score} / {self.TOTAL_GEMS}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))

        # Render timer
        time_left_sec = max(0, self.time_remaining / self.GAME_FPS)
        timer_text = f"Time: {time_left_sec:.1f}"
        timer_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(timer_surf, timer_rect)
        
        if self.game_over:
            message = "YOU WIN!" if self.score >= self.TOTAL_GEMS else "TIME UP!"
            message_surf = self.font_large.render(message, True, self.COLOR_PLAYER)
            message_rect = message_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(message_surf, message_rect)

    def _distance_to_nearest_gem(self, pos):
        if not self.gems:
            return 0
        px, py = pos
        min_dist = float('inf')
        for gx, gy in self.gems:
            dist = abs(px - gx) + abs(py - gy)  # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a separate display for human play
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Collector")
    
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if done:
                print(f"Episode finished. Total Reward: {total_reward:.2f}")
                print(f"Final Info: {info}")

        # Render the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(GameEnv.GAME_FPS)

    env.close()