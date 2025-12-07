
# Generated: 2025-08-27T23:51:57.673617
# Source Brief: brief_03606.md
# Brief Index: 3606

        
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
        "Controls: Arrow keys to move your robot. Avoid the red lasers."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a robot through laser-filled mazes to reach the green exit. You can sustain 3 hits before failing."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 20
    GRID_W = SCREEN_WIDTH // TILE_SIZE  # 32
    GRID_H = SCREEN_HEIGHT // TILE_SIZE # 20
    
    MAX_STEPS = 1000
    MAX_HITS = 3
    
    # --- Colors ---
    COLOR_BG = (20, 25, 30)
    COLOR_WALL = (60, 70, 80)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_EXIT = (50, 255, 150)
    COLOR_LASER = (255, 50, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE_HIT = (255, 150, 150)

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
        self.font_ui = pygame.font.SysFont('Consolas', 20, bold=True)
        
        # Game state variables that persist across resets
        self.level = 0
        self.np_random = None
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def _generate_maze(self, width, height):
        # Maze grid (only odd dimensions for this algorithm)
        w, h = (width // 2), (height // 2)
        maze = np.ones((h, w), dtype=np.uint8)
        
        # Path tracking for finding the furthest point
        path_map = {}
        
        # Randomized DFS
        stack = []
        start_node = (0, 0)
        stack.append(start_node)
        maze[start_node[1], start_node[0]] = 0
        path_map[start_node] = 0 # distance from start
        
        farthest_node = start_node
        max_dist = 0

        while len(stack) > 0:
            current_node = stack[-1]
            x, y = current_node
            
            neighbors = []
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                next_node = self.np_random.choice(len(neighbors))
                nx, ny = neighbors[next_node]
                
                maze[ny, nx] = 0
                
                dist = path_map[current_node] + 1
                path_map[(nx, ny)] = dist
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = (nx, ny)

                stack.append((nx, ny))
            else:
                stack.pop()

        # Create a larger grid to represent walls and paths
        grid = np.ones((height, width), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                if maze[y, x] == 0:
                    grid[y * 2 + 1, x * 2 + 1] = 0
                    if y > 0 and maze[y - 1, x] == 0: grid[y * 2, x * 2 + 1] = 0
                    if x > 0 and maze[y, x - 1] == 0: grid[y * 2 + 1, x * 2] = 0
        
        start_pos = (1, 1)
        exit_pos = (farthest_node[0] * 2 + 1, farthest_node[1] * 2 + 1)
        
        return grid, start_pos, exit_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.hits_taken = 0
        self.particles = []

        # Generate maze and place objects
        self.maze, self.player_pos, self.exit_pos = self._generate_maze(self.GRID_W, self.GRID_H)
        
        # Place lasers
        path_tiles = np.argwhere(self.maze == 0).tolist()
        path_tiles = [tuple(pos) for pos in path_tiles]
        
        # Ensure lasers aren't on start/end
        if tuple(self.player_pos) in path_tiles:
            path_tiles.remove(tuple(self.player_pos))
        if tuple(self.exit_pos) in path_tiles:
            path_tiles.remove(tuple(self.exit_pos))
        
        num_lasers = 3 + int(self.level * 0.5)
        num_lasers = min(num_lasers, len(path_tiles)) # Can't have more lasers than spots
        
        laser_indices = self.np_random.choice(len(path_tiles), size=num_lasers, replace=False)
        self.lasers = [path_tiles[i] for i in laser_indices]

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = -0.1 # Cost of taking a step
        self.steps += 1
        
        # --- Update Game Logic ---
        px, py = self.player_pos
        if movement == 1: py -= 1 # Up
        elif movement == 2: py += 1 # Down
        elif movement == 3: px -= 1 # Left
        elif movement == 4: px += 1 # Right

        # Wall collision check
        if 0 <= px < self.GRID_W and 0 <= py < self.GRID_H and self.maze[py, px] == 0:
            self.player_pos = (px, py)
        
        # Event checks
        if self.player_pos in self.lasers:
            self.hits_taken += 1
            reward -= 10
            self._create_hit_particles()
            # SFX: Robot hit zapping sound

        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 100
            terminated = True
            self.game_over = True
            self.level += 1 # Difficulty progression
            # SFX: Level complete fanfare

        if self.hits_taken >= self.MAX_HITS:
            terminated = True
            self.game_over = True
            # SFX: Game over explosion sound

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.score += reward
        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _create_hit_particles(self):
        px, py = self.player_pos
        screen_x = (px + 0.5) * self.TILE_SIZE
        screen_y = (py + 0.5) * self.TILE_SIZE
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': [screen_x, screen_y], 'vel': vel, 'life': life, 'size': 3})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw maze walls
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self.maze[y, x] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        # Draw exit
        ex, ey = self.exit_pos
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex * self.TILE_SIZE, ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        # Draw lasers
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        laser_brightness = 150 + 105 * pulse
        laser_color = (int(laser_brightness), self.COLOR_LASER[1], self.COLOR_LASER[2])
        for lx, ly in self.lasers:
            center_x = int((lx + 0.5) * self.TILE_SIZE)
            center_y = int((ly + 0.5) * self.TILE_SIZE)
            radius = int(self.TILE_SIZE * 0.4 * (0.8 + 0.2 * pulse))
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, laser_color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, laser_color)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE_HIT, (int(p['pos'][0]), int(p['pos'][1])), int(p['size']))

        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(px * self.TILE_SIZE, py * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in self.COLOR_PLAYER), player_rect.inflate(-6, -6))


    def _render_ui(self):
        # Hits remaining
        hits_text = f"HITS: {self.MAX_HITS - self.hits_taken}/{self.MAX_HITS}"
        hits_surf = self.font_ui.render(hits_text, True, self.COLOR_LASER if self.hits_taken > 0 else self.COLOR_TEXT)
        self.screen.blit(hits_surf, (10, 10))
        
        # Steps taken
        steps_text = f"STEPS: {self.steps}"
        steps_surf = self.font_ui.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (self.SCREEN_WIDTH - steps_surf.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "hits_taken": self.hits_taken
        }

    def close(self):
        pygame.font.quit()
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
    
    # Create a window to display the game
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    print(env.user_guide)
    
    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    print("--- MAZE RESET ---")
                
                # Only step if a key was pressed
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated
                
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")

        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print("Game Over! Press 'R' to play again or close the window.")
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        done = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        done = False
                        wait_for_reset = False
                        print("\n--- NEW GAME ---")

    env.close()