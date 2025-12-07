
# Generated: 2025-08-27T17:49:27.827524
# Source Brief: brief_01654.md
# Brief Index: 1654

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze and collect all the fruit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze, collecting all the fruit before the 60-second timer expires. Earn points for each fruit and a bonus for finishing quickly."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAZE_WIDTH = 31  # Must be odd
        self.MAZE_HEIGHT = 19 # Must be odd
        self.CELL_SIZE = self.SCREEN_HEIGHT // (self.MAZE_HEIGHT + 2) # Add padding
        self.MAZE_OFFSET_X = (self.SCREEN_WIDTH - self.MAZE_WIDTH * self.CELL_SIZE) // 2
        self.MAZE_OFFSET_Y = (self.SCREEN_HEIGHT - self.MAZE_HEIGHT * self.CELL_SIZE) // 2
        
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.NUM_FRUITS = 20

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (40, 40, 80)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 0, 50)
        self.FRUIT_COLORS = [
            (255, 0, 0), (0, 255, 0), (255, 165, 0), 
            (255, 0, 255), (0, 255, 255), (255, 105, 180)
        ]
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_WIN_TEXT = (100, 255, 100)
        self.COLOR_LOSE_TEXT = (255, 100, 100)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Game state variables are initialized in reset()
        self.maze = None
        self.player_pos = None
        self.fruits = None
        self.fruit_colors = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # Initialize state variables
        self.reset()
        
        # This check is for development and ensures the implementation is correct.
        # self.validate_implementation()

    def _generate_maze(self):
        # Using Randomized Prim's Algorithm
        maze = np.ones((self.MAZE_WIDTH, self.MAZE_HEIGHT), dtype=np.uint8)
        start_x, start_y = (self.np_random.integers(0, self.MAZE_WIDTH // 2) * 2 + 1,
                            self.np_random.integers(0, self.MAZE_HEIGHT // 2) * 2 + 1)
        maze[start_x, start_y] = 0
        
        walls = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = start_x + dx * 2, start_y + dy * 2
            if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT:
                walls.append((nx, ny, start_x + dx, start_y + dy))

        while walls:
            wall = walls.pop(self.np_random.integers(len(walls)))
            nx, ny, px, py = wall
            if maze[nx, ny] == 1:
                maze[nx, ny] = 0
                maze[px, py] = 0
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nnx, nny = nx + dx * 2, ny + dy * 2
                    if 0 <= nnx < self.MAZE_WIDTH and 0 <= nny < self.MAZE_HEIGHT and maze[nnx, nny] == 1:
                        walls.append((nnx, nny, nx + dx, ny + dy))
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.maze = self._generate_maze()
        
        possible_starts = np.argwhere(self.maze == 0)
        start_idx = self.np_random.integers(len(possible_starts))
        self.player_pos = possible_starts[start_idx].tolist()

        self.fruits = []
        self.fruit_colors = []
        possible_fruit_locs = [list(loc) for loc in possible_starts if list(loc) != self.player_pos]
        
        fruit_indices = self.np_random.choice(len(possible_fruit_locs), self.NUM_FRUITS, replace=False)
        for i in fruit_indices:
            self.fruits.append(possible_fruit_locs[i])
            self.fruit_colors.append(self.np_random.choice(self.FRUIT_COLORS))
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # Calculate distance to nearest fruit before moving
        dist_before = self._find_nearest_fruit_dist()
        
        # Unpack factorized action
        movement = action[0]
        
        # Update player position
        prev_pos = list(self.player_pos)
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
        
        # Collision detection
        if 0 <= new_pos[0] < self.MAZE_WIDTH and 0 <= new_pos[1] < self.MAZE_HEIGHT:
            if self.maze[new_pos[0], new_pos[1]] == 0:
                self.player_pos = new_pos
        
        # Calculate distance to nearest fruit after moving
        dist_after = self._find_nearest_fruit_dist()
        
        # Proximity reward
        if dist_after < dist_before:
            reward += 0.1
        elif dist_after > dist_before:
            reward -= 0.1

        # Fruit collection
        collected_fruit_index = -1
        for i, fruit_pos in enumerate(self.fruits):
            if self.player_pos == fruit_pos:
                collected_fruit_index = i
                break
        
        if collected_fruit_index != -1:
            self.fruits.pop(collected_fruit_index)
            self.fruit_colors.pop(collected_fruit_index)
            self.score += 10
            reward += 10
            # sfx: fruit_collect.wav

        self.steps += 1
        terminated = self._check_termination()

        if terminated and self.win:
            reward += 50 # Win bonus
            self.score += 50

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        if not self.fruits:
            self.game_over = True
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            return True
        return False

    def _find_nearest_fruit_dist(self):
        if not self.fruits:
            return 0
        min_dist = float('inf')
        for fruit_pos in self.fruits:
            dist = abs(self.player_pos[0] - fruit_pos[0]) + abs(self.player_pos[1] - fruit_pos[1])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _grid_to_pixel(self, grid_pos):
        px = self.MAZE_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.MAZE_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _render_game(self):
        # Draw maze
        for x in range(self.MAZE_WIDTH):
            for y in range(self.MAZE_HEIGHT):
                if self.maze[x, y] == 1:
                    rect = pygame.Rect(
                        self.MAZE_OFFSET_X + x * self.CELL_SIZE,
                        self.MAZE_OFFSET_Y + y * self.CELL_SIZE,
                        self.CELL_SIZE,
                        self.CELL_SIZE
                    )
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw fruits
        fruit_radius = self.CELL_SIZE // 3
        for i, fruit_pos in enumerate(self.fruits):
            px, py = self._grid_to_pixel(fruit_pos)
            color = self.fruit_colors[i]
            pygame.gfxdraw.filled_circle(self.screen, px, py, fruit_radius, color)
            pygame.gfxdraw.aacircle(self.screen, px, py, fruit_radius, color)

        # Draw player
        player_radius = self.CELL_SIZE // 2 - 2
        px, py = self._grid_to_pixel(self.player_pos)
        pygame.gfxdraw.filled_circle(self.screen, px, py, player_radius + 3, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, px, py, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, player_radius, self.COLOR_PLAYER)
        
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time
        remaining_time = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        time_text = self.font_ui.render(f"TIME: {remaining_time:.1f}", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)
        
        # Game Over Message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN_TEXT
            else:
                msg = "TIME'S UP!"
                color = self.COLOR_LOSE_TEXT
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


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
            "remaining_fruits": len(self.fruits),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    # You will need to install pygame for this to work: pip install pygame
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Maze Collector")
    
    terminated = False
    clock = pygame.time.Clock()
    
    print(env.game_description)
    print(env.user_guide)

    while not terminated:
        movement_action = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        # The environment expects a MultiDiscrete action
        action = [movement_action, 0, 0] # Space and Shift are not used

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS) # Control the frame rate

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000)
    env.close()