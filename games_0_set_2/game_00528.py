
# Generated: 2025-08-27T13:55:02.198409
# Source Brief: brief_00528.md
# Brief Index: 528

        
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
        "Controls: Use arrow keys (↑↓←→) to navigate the maze. Collect all the gems!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect all the cyan gems in the procedurally generated maze while avoiding the red obstacles. Your score is based on gems collected and speed."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAZE_COLS = 31  # Must be odd
    MAZE_ROWS = 19  # Must be odd
    TILE_SIZE = min(SCREEN_WIDTH // (MAZE_COLS * 2 + 1), SCREEN_HEIGHT // (MAZE_ROWS * 2 + 1)) * 2
    
    RENDER_MAZE_WIDTH = (MAZE_COLS * 2 + 1) * (TILE_SIZE // 2)
    RENDER_MAZE_HEIGHT = (MAZE_ROWS * 2 + 1) * (TILE_SIZE // 2)
    OFFSET_X = (SCREEN_WIDTH - RENDER_MAZE_WIDTH) // 2
    OFFSET_Y = (SCREEN_HEIGHT - RENDER_MAZE_HEIGHT) // 2

    NUM_GEMS = 20
    NUM_OBSTACLES = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_WALL = (20, 30, 80)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_GEM = (0, 255, 255)
    COLOR_OBSTACLE = (255, 0, 60)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Initialize state variables
        self.maze = None
        self.player_pos = None
        self.gems = None
        self.obstacles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_dist_to_gem = 0

        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self._generate_maze_and_entities()
        self.last_dist_to_gem = self._get_dist_to_nearest_gem()
        
        return self._get_observation(), self._get_info()

    def _generate_maze_and_entities(self):
        # Maze represented by grid cells (MAZE_COLS x MAZE_ROWS)
        # Render grid is (2*C+1) x (2*R+1) to include walls
        self.maze = np.ones((self.MAZE_ROWS * 2 + 1, self.MAZE_COLS * 2 + 1), dtype=np.uint8)
        
        # Randomized DFS for maze generation
        stack = []
        visited = set()

        start_cell = (self.np_random.integers(0, self.MAZE_ROWS), self.np_random.integers(0, self.MAZE_COLS))
        stack.append(start_cell)
        visited.add(start_cell)
        self.maze[start_cell[0]*2+1, start_cell[1]*2+1] = 0

        while stack:
            cy, cx = stack[-1]
            neighbors = []
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.MAZE_ROWS and 0 <= nx < self.MAZE_COLS and (ny, nx) not in visited:
                    neighbors.append((ny, nx))
            
            if neighbors:
                ny, nx = neighbors[self.np_random.integers(0, len(neighbors))]
                
                # Remove wall between current and neighbor
                self.maze[cy*2+1 + (ny-cy), cx*2+1 + (nx-cx)] = 0
                self.maze[ny*2+1, nx*2+1] = 0

                visited.add((ny, nx))
                stack.append((ny, nx))
            else:
                stack.pop()

        # Get all valid floor tiles for entity placement
        floor_tiles = list(zip(*np.where(self.maze == 0)))
        
        # Shuffle and place entities
        self.np_random.shuffle(floor_tiles)
        
        self.player_pos = floor_tiles.pop(0)
        
        num_gems_to_place = min(self.NUM_GEMS, len(floor_tiles))
        self.gems = [floor_tiles.pop(0) for _ in range(num_gems_to_place)]
        
        num_obstacles_to_place = min(self.NUM_OBSTACLES, len(floor_tiles))
        self.obstacles = [floor_tiles.pop(0) for _ in range(num_obstacles_to_place)]


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        terminated = False
        
        # --- Player Movement ---
        py, px = self.player_pos
        ny, nx = py, px
        if movement == 1: # Up
            ny -= 1
        elif movement == 2: # Down
            ny += 1
        elif movement == 3: # Left
            nx -= 1
        elif movement == 4: # Right
            nx += 1

        # Check for wall collision
        if self.maze[ny, nx] == 0:
            self.player_pos = (ny, nx)

        # --- Reward for getting closer to a gem ---
        new_dist_to_gem = self._get_dist_to_nearest_gem()
        reward += (self.last_dist_to_gem - new_dist_to_gem) * 0.1
        self.last_dist_to_gem = new_dist_to_gem
        
        # --- Check for Interactions ---
        if self.player_pos in self.gems:
            # SFX: Gem collect
            self.gems.remove(self.player_pos)
            self.score += 1
            reward += 1.0
            self.last_dist_to_gem = self._get_dist_to_nearest_gem() # Recalculate dist
        
        if self.player_pos in self.obstacles:
            # SFX: Player hit
            self.game_over = True
            terminated = True
            reward = -10.0
            self.score = max(0, self.score - 10) # Penalty

        # --- Check Termination Conditions ---
        if not self.gems: # Victory
            self.game_over = True
            self.win = True
            terminated = True
            reward += 50.0
            self.score += 50
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_dist_to_nearest_gem(self):
        if not self.gems:
            return 0
        py, px = self.player_pos
        min_dist = float('inf')
        for gy, gx in self.gems:
            dist = abs(py - gy) + abs(px - gx) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        ts = self.TILE_SIZE // 2
        
        # Render maze walls
        for r in range(self.maze.shape[0]):
            for c in range(self.maze.shape[1]):
                if self.maze[r, c] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (self.OFFSET_X + c * ts, self.OFFSET_Y + r * ts, ts, ts))

        # Render obstacles
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
        obstacle_size = int(ts * (0.8 + pulse * 0.4))
        for r, c in self.obstacles:
            center_x = self.OFFSET_X + c * ts + ts // 2
            center_y = self.OFFSET_Y + r * ts + ts // 2
            # Pulsating triangle
            points = [
                (center_x, center_y - obstacle_size // 2),
                (center_x - obstacle_size // 2, center_y + obstacle_size // 2),
                (center_x + obstacle_size // 2, center_y + obstacle_size // 2),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_OBSTACLE)

        # Render gems
        sparkle = (math.sin(self.steps * 0.3) + 1) / 2
        gem_size = int(ts * 0.6)
        glow_size = int(gem_size * (1.5 + sparkle * 0.8))
        for r, c in self.gems:
            center_x = self.OFFSET_X + c * ts + ts // 2
            center_y = self.OFFSET_Y + r * ts + ts // 2
            # Glow effect
            glow_color = (*self.COLOR_GEM, int(80 * (1 - sparkle)))
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_size // 2, glow_color)
            # Main gem
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, gem_size // 2, self.COLOR_GEM)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, gem_size // 2, self.COLOR_GEM)

        # Render player
        player_r, player_c = self.player_pos
        player_size = int(ts * 0.9)
        player_x = self.OFFSET_X + player_c * ts + (ts - player_size) // 2
        player_y = self.OFFSET_Y + player_r * ts + (ts - player_size) // 2
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (player_x, player_y, player_size, player_size))
        pygame.draw.rect(self.screen, (255,255,200), (player_x, player_y, player_size, player_size), 1)

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Gems remaining display
        gems_text = self.font_ui.render(f"GEMS: {len(self.gems)}/{self.NUM_GEMS}", True, self.COLOR_UI_TEXT)
        gems_rect = gems_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(gems_text, gems_rect)

        # Game Over message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_remaining": len(self.gems),
            "player_pos": self.player_pos
        }

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

# Example usage for testing
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame window for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Episode Reset ---")
                if event.key == pygame.K_q:
                    running = False

        # --- Continuous key presses for movement ---
        # This part is for human play; the agent would just pass an action
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # For a turn-based game, we only want to step once per key press event.
        # The following logic is a simple way to achieve this for human testing.
        # An agent would simply call env.step() in its own loop.
        
        # To make it playable, we'll use a timer to allow holding keys
        # but only step at a controlled rate.
        
        if movement != 0:
            action[0] = movement
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            if terminated:
                print(f"--- Episode Finished --- Final Score: {info['score']}")
                # Optional: Auto-reset after a delay
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0
                print("--- Episode Reset ---")

        # --- Render the environment observation to the window ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Control human play speed

    env.close()