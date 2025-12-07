import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ↑↓←→ to move the robot one tile at a time. "
        "Reach the green exit portal to win."
    )

    game_description = (
        "Guide a robot through a procedurally generated isometric maze. "
        "Plan your moves to reach the exit while avoiding the dark pitfalls. "
        "Fewer steps lead to a higher score."
    )

    auto_advance = False

    # --- Constants ---
    MAZE_WIDTH = 21
    MAZE_HEIGHT = 15
    TILE_WIDTH_HALF = 20
    TILE_HEIGHT_HALF = 10
    WALL_HEIGHT = 20
    ROBOT_HEIGHT = 10
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (30, 35, 40)
    COLOR_GRID = (40, 45, 50)
    COLOR_FLOOR = (70, 80, 90)
    COLOR_WALL_TOP = (140, 150, 160)
    COLOR_WALL_SIDE = (110, 120, 130)
    COLOR_PIT = (10, 10, 15)
    COLOR_ROBOT_TOP = (60, 160, 220)
    COLOR_ROBOT_SIDE = (50, 130, 190)
    COLOR_EXIT_BASE = (20, 100, 60)
    COLOR_EXIT_GLOW1 = (50, 220, 120, 150)
    COLOR_EXIT_GLOW2 = (150, 255, 200, 100)
    COLOR_TEXT = (230, 230, 230)
    COLOR_SHADOW = (0, 0, 0, 50)

    # --- Maze Cell Types ---
    CELL_PATH = 0
    CELL_WALL = 1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        self.maze_offset_x = (640 - (self.MAZE_WIDTH - self.MAZE_HEIGHT) * self.TILE_WIDTH_HALF) / 2
        self.maze_offset_y = 60

        self.level = 1
        self.maze = []
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.pit_positions = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # self.reset() is called by the API, no need to call it here.
        # However, to pass validation, some attributes need to be initialized.
        # We can call a lightweight setup or just initialize maze here.
        self.np_random = np.random.default_rng()
        self._generate_maze()
        
        self.validate_implementation()

    def _generate_maze(self):
        # Initialize grid with walls
        self.maze = [[self.CELL_WALL for _ in range(self.MAZE_WIDTH)] for _ in range(self.MAZE_HEIGHT)]
        
        # All valid path cells, must be odd coordinates
        path_cells = []
        
        # Use randomized DFS to carve paths
        stack = []
        start_pos = (1, 1)
        self.maze[start_pos[1]][start_pos[0]] = self.CELL_PATH
        stack.append(start_pos)
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.MAZE_WIDTH -1 and 0 < ny < self.MAZE_HEIGHT -1 and self.maze[ny][nx] == self.CELL_WALL:
                    neighbors.append((nx, ny))
            
            if neighbors:
                random_index = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[random_index]
                
                # Carve path to neighbor
                self.maze[ny][nx] = self.CELL_PATH
                self.maze[(cy + ny) // 2][(cx + nx) // 2] = self.CELL_PATH
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Collect all possible path cells for placing items
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                if self.maze[y][x] == self.CELL_PATH:
                    path_cells.append((x, y))

        # Set player start and exit
        self.player_pos = (1, 1)
        self.exit_pos = (self.MAZE_WIDTH - 2, self.MAZE_HEIGHT - 2)
        if self.exit_pos not in path_cells: # Ensure exit is reachable
             self.exit_pos = path_cells[-1]

        # Place pits
        self.pit_positions = []
        num_pits = 1 + self.level # Starts with 2 pits on level 1
        
        possible_pit_locs = [p for p in path_cells if p not in [self.player_pos, self.exit_pos]]
        if possible_pit_locs:
            self.np_random.shuffle(possible_pit_locs)
            for i in range(min(num_pits, len(possible_pit_locs))):
                self.pit_positions.append(possible_pit_locs[i])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_maze()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        reward = -0.1
        self.steps += 1
        
        px, py = self.player_pos
        nx, ny = px, py

        if movement == 1: # Up
            ny -= 1
        elif movement == 2: # Down
            ny += 1
        elif movement == 3: # Left
            nx -= 1
        elif movement == 4: # Right
            nx += 1
        
        # Check boundaries and walls
        if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[ny][nx] == self.CELL_PATH:
            self.player_pos = (nx, ny)
        
        terminated = False
        truncated = False
        if self.player_pos == self.exit_pos:
            reward += 10
            self.score += 10
            terminated = True
            self.game_over = True
            self.level += 1
        
        if self.player_pos in self.pit_positions:
            reward += -10
            self.score -= 10
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            terminated = True # Per Gymnasium API, terminated should be True if truncated is True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _iso_transform(self, x, y):
        screen_x = self.maze_offset_x + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.maze_offset_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_poly(self, surface, points, color):
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_iso_block(self, surface, pos, top_color, side_color, height):
        px, py = pos
        hw, hh = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        
        top_points = [
            (px, py - height),
            (px + hw, py - height + hh),
            (px, py - height + hh * 2),
            (px - hw, py - height + hh)
        ]
        
        # Right Side
        right_points = [
            (px, py - height + hh * 2),
            (px + hw, py - height + hh),
            (px + hw, py + hh),
            (px, py + hh * 2)
        ]
        
        # Left Side
        left_points = [
            (px, py - height + hh * 2),
            (px - hw, py - height + hh),
            (px - hw, py + hh),
            (px, py + hh * 2)
        ]
        
        pygame.draw.polygon(surface, side_color, right_points)
        pygame.draw.polygon(surface, side_color, left_points)
        pygame.draw.polygon(surface, top_color, top_points)
        
        # Outlines for definition
        pygame.draw.aalines(surface, top_color, True, top_points)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_grid()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_grid(self):
        for i in range(30):
            # Horizontal-ish lines
            pygame.draw.aaline(self.screen, self.COLOR_GRID, (0, i * 2 * self.TILE_HEIGHT_HALF - self.TILE_HEIGHT_HALF * 10), (640, i * 2 * self.TILE_HEIGHT_HALF + 640 * 0.5 - self.TILE_HEIGHT_HALF * 10))
            # Vertical-ish lines
            pygame.draw.aaline(self.screen, self.COLOR_GRID, (i * 2 * self.TILE_WIDTH_HALF - self.TILE_WIDTH_HALF * 20, 0), (i * 2 * self.TILE_WIDTH_HALF - 640 * 0.5 - self.TILE_WIDTH_HALF * 20, 400))


    def _render_game(self):
        hw, hh = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        
        # Render maze tiles, pits, and exit
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                pos = self._iso_transform(x, y)
                px, py = pos
                
                floor_points = [
                    (px, py),
                    (px + hw, py + hh),
                    (px, py + hh * 2),
                    (px - hw, py + hh)
                ]

                # Draw floor tile
                if self.maze[y][x] == self.CELL_PATH:
                    if (x, y) in self.pit_positions:
                        self._draw_iso_poly(self.screen, floor_points, self.COLOR_PIT)
                    elif (x, y) == self.exit_pos:
                        self._draw_iso_poly(self.screen, floor_points, self.COLOR_EXIT_BASE)
                    else:
                        self._draw_iso_poly(self.screen, floor_points, self.COLOR_FLOOR)
                
                # Draw wall blocks
                elif self.maze[y][x] == self.CELL_WALL:
                    self._draw_iso_block(self.screen, pos, self.COLOR_WALL_TOP, self.COLOR_WALL_SIDE, self.WALL_HEIGHT)

        # Draw exit portal glow effect
        exit_screen_pos = self._iso_transform(self.exit_pos[0], self.exit_pos[1])
        glow_surface = pygame.Surface((hw * 4, hh * 4), pygame.SRCALPHA)
        glow_center = (hw * 2, hh * 2)
        pygame.draw.circle(glow_surface, self.COLOR_EXIT_GLOW1, glow_center, hw * 1.2)
        pygame.draw.circle(glow_surface, self.COLOR_EXIT_GLOW2, glow_center, hw * 0.7)
        self.screen.blit(glow_surface, (exit_screen_pos[0] - hw * 2, exit_screen_pos[1] - hh * 1.5))


        # Draw player robot
        player_screen_pos = self._iso_transform(self.player_pos[0], self.player_pos[1])
        
        # Shadow
        shadow_surface = pygame.Surface((hw * 2, hh * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, self.COLOR_SHADOW, [0, 0, hw * 2, hh * 2])
        self.screen.blit(shadow_surface, (player_screen_pos[0] - hw, player_screen_pos[1]))
        
        # Bobbing animation
        bob_offset = math.sin(self.steps * 0.3) * 3 if not self.game_over else 0
        
        self._draw_iso_block(self.screen, player_screen_pos, self.COLOR_ROBOT_TOP, self.COLOR_ROBOT_SIDE, self.ROBOT_HEIGHT + bob_offset)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_main.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (640 - steps_text.get_width() - 10, 10))
        
        level_text = self.font_small.render(f"MAZE LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 40))

        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if self.player_pos == self.exit_pos:
                end_text_str = "SUCCESS!"
            elif self.player_pos in self.pit_positions:
                end_text_str = "FELL INTO A PIT"
            else:
                end_text_str = "OUT OF STEPS"
                
            end_text = self.font_main.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(320, 200))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "player_pos": self.player_pos,
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # Example of a simple agent loop
    print("Running a short random agent test...")
    obs, info = env.reset(seed=42)
    terminated = False
    truncated = False
    total_reward = 0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward}, Info: {info}")
            total_reward = 0
            obs, info = env.reset(seed=43)
    
    env.close()
    print("Random agent test complete.")