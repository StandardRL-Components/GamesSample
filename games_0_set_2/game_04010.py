
# Generated: 2025-08-28T01:07:49.764563
# Source Brief: brief_04010.md
# Brief Index: 4010

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. Reach the green exit before the timer runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze against the clock. Collect blue items for extra points and find the green exit to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (60, 60, 80)
        self.COLOR_PATH = (40, 40, 55)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 150)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_ITEM = (0, 150, 255)
        self.COLOR_TEXT = (240, 240, 240)
        
        # Game progression
        self.maze_level = 0
        self.max_maze_dim = 30
        
        # Initialize state variables
        self.maze = []
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.items = []
        self.player_trail = deque(maxlen=20)
        self.max_steps = 180
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Difficulty scaling
        maze_w = min(self.max_maze_dim, 15 + self.maze_level)
        maze_h = min(self.max_maze_dim, int((15 + self.maze_level) * (400/640)))
        
        # Ensure odd dimensions for maze generation
        self.maze_width = maze_w + 1 if maze_w % 2 == 0 else maze_w
        self.maze_height = maze_h + 1 if maze_h % 2 == 0 else maze_h
        
        self.player_pos = (1, 1)
        self.exit_pos = (self.maze_width - 2, self.maze_height - 2)
        
        self.maze = self._generate_maze(self.maze_width, self.maze_height)
        
        self._place_items()
        
        self.player_trail.clear()
        self.player_trail.append(self.player_pos)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        terminated = False
        
        prev_pos = self.player_pos
        px, py = self.player_pos
        
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
            
        # Check for wall collision
        if self.maze[py][px] == 0: # 0 is path, 1 is wall
            self.player_pos = (px, py)
        
        # Add to trail only on movement
        if prev_pos != self.player_pos:
            self.player_trail.append(self.player_pos)
        
        # Calculate distance-based reward
        dist_before = self._manhattan_distance(prev_pos, self.exit_pos)
        dist_after = self._manhattan_distance(self.player_pos, self.exit_pos)
        
        if dist_after < dist_before:
            reward += 1
        elif dist_after > dist_before and movement != 0:
            reward -= 1
        
        # Check for item collection
        if self.player_pos in self.items:
            self.items.remove(self.player_pos)
            self.score += 50
            reward += 50 # Brief specifies +50 event reward
            # Sound: Collect item
        
        self.steps += 1
        
        # Check for termination conditions
        if self.player_pos == self.exit_pos:
            terminated = True
            self.game_over = True
            self.score += 100
            reward += 100
            self.maze_level += 1 # Increase difficulty for next game
            # Sound: Win
        elif self.steps >= self.max_steps:
            terminated = True
            self.game_over = True
            self.score -= 50
            reward -= 50
            # Sound: Lose / Timeout
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _generate_maze(self, width, height):
        # 1 = wall, 0 = path
        maze = np.ones((height, width), dtype=np.int8)
        
        # Use numpy's random generator for reproducibility
        rng = np.random.default_rng(self.np_random)

        def is_valid(x, y):
            return 0 <= x < width and 0 <= y < height

        stack = [(1, 1)]
        maze[1, 1] = 0

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if is_valid(nx, ny) and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = neighbors[rng.integers(len(neighbors))]
                # Carve path
                maze[ny, nx] = 0
                maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        return maze.tolist()

    def _place_items(self):
        self.items = []
        num_items = self.maze_width // 5
        rng = np.random.default_rng(self.np_random)
        
        possible_locations = []
        for y in range(1, self.maze_height - 1):
            for x in range(1, self.maze_width - 1):
                if self.maze[y][x] == 0 and (x, y) != self.player_pos and (x, y) != self.exit_pos:
                    possible_locations.append((x, y))
        
        if len(possible_locations) > num_items:
            indices = rng.choice(len(possible_locations), size=num_items, replace=False)
            self.items = [possible_locations[i] for i in indices]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        tile_w = self.screen.get_width() / self.maze_width
        tile_h = self.screen.get_height() / self.maze_height
        
        # Draw maze paths and walls
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                rect = pygame.Rect(x * tile_w, y * tile_h, math.ceil(tile_w), math.ceil(tile_h))
                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_PATH, rect)

        # Draw player trail
        for i, pos in enumerate(self.player_trail):
            alpha = int(100 * (i / len(self.player_trail)))
            trail_color = self.COLOR_PLAYER + (alpha,)
            trail_surf = pygame.Surface((tile_w, tile_h), pygame.SRCALPHA)
            trail_surf.fill(trail_color)
            self.screen.blit(trail_surf, (pos[0] * tile_w, pos[1] * tile_h))
            
        # Draw items
        for ix, iy in self.items:
            item_rect = pygame.Rect(ix * tile_w, iy * tile_h, tile_w, tile_h)
            pygame.draw.rect(self.screen, self.COLOR_ITEM, item_rect.inflate(-tile_w*0.3, -tile_h*0.3))

        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(ex * tile_w, ey * tile_h, tile_w, tile_h)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Draw player
        px, py = self.player_pos
        player_center_x = int(px * tile_w + tile_w / 2)
        player_center_y = int(py * tile_h + tile_h / 2)
        player_radius = int(min(tile_w, tile_h) / 2.5)

        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER_GLOW)
        # Main body
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, int(player_radius * 0.8), self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, int(player_radius * 0.8), self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = self.max_steps - self.steps
        timer_color = (255, 100, 100) if time_left < 30 else self.COLOR_TEXT
        timer_text = self.font_ui.render(f"TIME: {time_left}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.screen.get_width() - 10, 10))
        self.screen.blit(timer_text, timer_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.max_steps - self.steps,
            "maze_level": self.maze_level,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Test Reset ---
    print("--- Testing Reset ---")
    obs, info = env.reset()
    print("Reset successful.")
    print(f"Initial Info: {info}")
    assert obs.shape == (400, 640, 3)
    
    # --- Test Step ---
    print("\n--- Testing Step ---")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step with action {action} successful.")
    print(f"Info after step: {info}")
    print(f"Reward: {reward}, Terminated: {terminated}")
    assert obs.shape == (400, 640, 3)
    
    # --- Run a short episode for visualization (if you have pygame display) ---
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        import time
        
        env_vis = GameEnv(render_mode="rgb_array")
        obs, info = env_vis.reset()
        
        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Maze Runner")
        
        terminated = False
        total_reward = 0
        
        print("\n--- Running visual demo. Close the window to exit. ---")
        
        running = True
        while running and not terminated:
            action = [0, 0, 0] # Default no-op
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Since auto_advance is False, we only step when a key is pressed
            if action[0] != 0:
                obs, reward, terminated, truncated, info = env_vis.step(action)
                total_reward += reward
            
            # Draw the observation to the screen
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Limit the loop speed to make it playable
            pygame.time.wait(30)
            
        print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward}")
        env_vis.close()
        
    except pygame.error as e:
        print(f"\nSkipping visual demo: Pygame display not available ({e})")

    env.close()