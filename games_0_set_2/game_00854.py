
# Generated: 2025-08-27T14:59:10.750945
# Source Brief: brief_00854.md
# Brief Index: 854

        
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
    user_guide = "Controls: Use arrow keys (↑↓←→) to navigate the maze."

    # Must be a short, user-facing description of the game:
    game_description = "Navigate a procedurally generated maze to reach the green exit. Each win increases the maze size."

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and rendering
        self.screen_width = 640
        self.screen_height = 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255)
        self.COLOR_EXIT = (50, 255, 150)
        self.COLOR_UI_TEXT = (220, 220, 220)

        # Game state that persists across resets until a win
        self.maze_width = 5
        self.maze_height = 5
        self.max_maze_dim = 30
        self.max_steps = 1000

        # Initialize state variables
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.np_random = None
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def _generate_maze(self, width, height):
        maze = [[{'N': True, 'S': True, 'E': True, 'W': True} for _ in range(width)] for _ in range(height)]
        visited = [[False for _ in range(width)] for _ in range(height)]
        
        stack = [(0, 0)]
        visited[0][0] = True

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            if cy > 0 and not visited[cy - 1][cx]: neighbors.append((cx, cy - 1, 'N'))
            if cy < height - 1 and not visited[cy + 1][cx]: neighbors.append((cx, cy + 1, 'S'))
            if cx < width - 1 and not visited[cy][cx + 1]: neighbors.append((cx + 1, cy, 'E'))
            if cx > 0 and not visited[cy][cx - 1]: neighbors.append((cx - 1, cy, 'W'))

            if neighbors:
                # Use the environment's random number generator
                idx = self.np_random.integers(len(neighbors))
                nx, ny, direction = neighbors[idx]
                
                if direction == 'N':
                    maze[cy][cx]['N'] = False
                    maze[ny][nx]['S'] = False
                elif direction == 'S':
                    maze[cy][cx]['S'] = False
                    maze[ny][nx]['N'] = False
                elif direction == 'E':
                    maze[cy][cx]['E'] = False
                    maze[ny][nx]['W'] = False
                elif direction == 'W':
                    maze[cy][cx]['W'] = False
                    maze[ny][nx]['E'] = False
                
                visited[ny][nx] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        self.player_pos = [0, 0]
        self.exit_pos = [self.maze_width - 1, self.maze_height - 1]
        
        self.maze = self._generate_maze(self.maze_width, self.maze_height)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1  # Cost of taking a step

        px, py = self.player_pos
        next_px, next_py = px, py
        moved = False

        if movement == 1 and py > 0 and not self.maze[py][px]['N']:  # Up
            next_py -= 1
            moved = True
        elif movement == 2 and py < self.maze_height - 1 and not self.maze[py][px]['S']:  # Down
            next_py += 1
            moved = True
        elif movement == 3 and px > 0 and not self.maze[py][px]['W']:  # Left
            next_px -= 1
            moved = True
        elif movement == 4 and px < self.maze_width - 1 and not self.maze[py][px]['E']:  # Right
            next_px += 1
            moved = True
        
        if movement in [1, 2, 3, 4] and not moved:
            reward -= 5  # Penalty for bumping into a wall
            # SFX: *thud*

        self.player_pos = [next_px, next_py]
        self.steps += 1
        
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 60  # +10 for reaching exit, +50 for winning
            self.win_message = f"SOLVED! +{int(reward + 0.1)} SCORE"
            terminated = True
            self.game_over = True
            # SFX: *level complete fanfare*
            
            # Increase difficulty for the next round
            if self.maze_width < self.max_maze_dim:
                self.maze_width += 1
            if self.maze_height < self.max_maze_dim:
                self.maze_height += 1
        
        if self.steps >= self.max_steps:
            self.win_message = "TIME UP!"
            terminated = True
            self.game_over = True
            # SFX: *fail buzzer*

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_game(self):
        cell_w = self.screen_width / self.maze_width
        cell_h = self.screen_height / self.maze_height
        
        # Render exit
        exit_x = self.exit_pos[0] * cell_w
        exit_y = self.exit_pos[1] * cell_h
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (exit_x, exit_y, cell_w, cell_h))

        # Render player
        player_cx = int((self.player_pos[0] + 0.5) * cell_w)
        player_cy = int((self.player_pos[1] + 0.5) * cell_h)
        player_radius = int(min(cell_w, cell_h) * 0.3)
        
        # Glow effect
        glow_radius = int(player_radius * 1.5)
        if glow_radius > 0:
            pygame.gfxdraw.filled_circle(self.screen, player_cx, player_cy, glow_radius, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.aacircle(self.screen, player_cx, player_cy, glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Player circle
        if player_radius > 0:
            pygame.gfxdraw.filled_circle(self.screen, player_cx, player_cy, player_radius, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, player_cx, player_cy, player_radius, self.COLOR_PLAYER)
        
        # Render maze walls
        wall_thickness = max(1, int(min(cell_w, cell_h) * 0.05))
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                walls = self.maze[y][x]
                px1, py1 = x * cell_w, y * cell_h
                px2, py2 = (x + 1) * cell_w, (y + 1) * cell_h
                if walls['N']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px1, py1), (px2, py1), wall_thickness)
                if walls['S']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px1, py2), (px2, py2), wall_thickness)
                if walls['W']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px1, py1), (px1, py2), wall_thickness)
                if walls['E']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px2, py1), (px2, py2), wall_thickness)

    def _render_ui(self):
        moves_text = self.font_ui.render(f"Moves: {self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (10, 10))
        
        steps_left_text = self.font_ui.render(f"Steps Left: {self.max_steps - self.steps}", True, self.COLOR_UI_TEXT)
        text_rect = steps_left_text.get_rect(topright=(self.screen_width - 10, 10))
        self.screen.blit(steps_left_text, text_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_game_over.render(self.win_message, True, self.COLOR_UI_TEXT)
            text_rect = game_over_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(game_over_text, text_rect)
            
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
            "maze_size": (self.maze_width, self.maze_height)
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")