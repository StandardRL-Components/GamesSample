
# Generated: 2025-08-27T18:54:15.584028
# Source Brief: brief_01984.md
# Brief Index: 1984

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. "
        "Reach the green exit before you run out of moves."
    )

    # Short, user-facing description of the game
    game_description = (
        "Navigate a procedurally generated maze to reach the exit within a limited "
        "number of moves. Each step counts!"
    )

    # Frames only advance when an action is received
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = 32
        self.MAX_MOVES = 50
        self.MAX_STEPS = 500

        # Centering calculations
        self.MAZE_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.MAZE_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.X_OFFSET = (self.SCREEN_WIDTH - self.MAZE_WIDTH) // 2
        self.Y_OFFSET = (self.SCREEN_HEIGHT - self.MAZE_HEIGHT) // 2

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Consolas", 24, bold=True)
            self.end_font = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font = pygame.font.SysFont(None, 24)
            self.end_font = pygame.font.SysFont(None, 48)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (50, 50, 70)
        self.COLOR_PATH = (30, 30, 45)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_EXIT = (100, 255, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_FLASH = (255, 50, 50)
        
        # Initialize state variables to be populated in reset()
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.remaining_moves = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.show_flash = False
        
        self.reset()
        self.validate_implementation()

    def _generate_maze(self):
        """
        Generates a perfect maze using a randomized depth-first search (DFS) algorithm.
        A "perfect" maze has exactly one path between any two cells.
        Returns a 3D numpy array representing walls for each cell: (grid_size, grid_size, 4)
        The 4 values are booleans for [North, East, South, West] walls.
        """
        walls = np.ones((self.GRID_SIZE, self.GRID_SIZE, 4), dtype=bool)
        visited = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        stack = []
        
        start_x, start_y = self.np_random.integers(0, self.GRID_SIZE, size=2)
        stack.append((start_x, start_y))
        visited[start_y, start_x] = True
        
        while stack:
            x, y = stack[-1]
            
            unvisited_neighbors = []
            # North
            if y > 0 and not visited[y - 1, x]: unvisited_neighbors.append(('N', x, y - 1))
            # East
            if x < self.GRID_SIZE - 1 and not visited[y, x + 1]: unvisited_neighbors.append(('E', x + 1, y))
            # South
            if y < self.GRID_SIZE - 1 and not visited[y + 1, x]: unvisited_neighbors.append(('S', x, y + 1))
            # West
            if x > 0 and not visited[y, x - 1]: unvisited_neighbors.append(('W', x - 1, y))

            if unvisited_neighbors:
                idx = self.np_random.integers(0, len(unvisited_neighbors))
                direction, nx, ny = unvisited_neighbors[idx]
                
                if direction == 'N':
                    walls[y, x, 0] = False; walls[ny, nx, 2] = False
                elif direction == 'E':
                    walls[y, x, 1] = False; walls[ny, nx, 3] = False
                elif direction == 'S':
                    walls[y, x, 2] = False; walls[ny, nx, 0] = False
                elif direction == 'W':
                    walls[y, x, 3] = False; walls[ny, nx, 1] = False
                
                visited[ny, nx] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        
        return walls

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.maze = self._generate_maze()
        
        self.player_pos = [0, 0]
        self.exit_pos = [self.GRID_SIZE - 1, self.GRID_SIZE - 1]
        
        # Ensure player and exit are not at the same spot for small mazes
        if self.GRID_SIZE > 1 and self.player_pos == self.exit_pos:
            self.exit_pos = [0, self.GRID_SIZE - 1]

        self.remaining_moves = self.MAX_MOVES
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.show_flash = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.show_flash = False # Reset flash from previous invalid move
        reward = 0.0
        
        # Only process if a move is attempted (not a no-op)
        if movement > 0:
            self.remaining_moves -= 1
            
            px, py = self.player_pos
            tx, ty = px, py # Target position
            
            # Check for a valid move based on maze walls
            # 0=none, 1=up, 2=down, 3=left, 4=right
            # Wall indices: 0=N, 1=E, 2=S, 3=W
            can_move = False
            if movement == 1 and not self.maze[py, px, 0]: # UP (North wall)
                ty -= 1; can_move = True
            elif movement == 2 and not self.maze[py, px, 2]: # DOWN (South wall)
                ty += 1; can_move = True
            elif movement == 3 and not self.maze[py, px, 3]: # LEFT (West wall)
                tx -= 1; can_move = True
            elif movement == 4 and not self.maze[py, px, 1]: # RIGHT (East wall)
                tx += 1; can_move = True

            if can_move:
                self.player_pos = [tx, ty]
                reward = -0.1
            else:
                reward = -5.0
                self.show_flash = True
                # sfx_wall_thud()

        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 100.0
            terminated = True
            # sfx_win_jingle()
        
        if self.remaining_moves <= 0 and not terminated:
            terminated = True
            # sfx_lose_buzzer()

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_game(self):
        # Draw background path color for all cells
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                px = self.X_OFFSET + x * self.CELL_SIZE
                py = self.Y_OFFSET + y * self.CELL_SIZE
                pygame.draw.rect(self.screen, self.COLOR_PATH, (px, py, self.CELL_SIZE, self.CELL_SIZE))

        # Draw exit
        ex, ey = self.exit_pos
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (
            self.X_OFFSET + ex * self.CELL_SIZE, self.Y_OFFSET + ey * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        ))
        
        # Draw player as a smaller square to show the path underneath
        px, py = self.player_pos
        player_rect = pygame.Rect(
            self.X_OFFSET + px * self.CELL_SIZE + self.CELL_SIZE // 4,
            self.Y_OFFSET + py * self.CELL_SIZE + self.CELL_SIZE // 4,
            self.CELL_SIZE // 2, self.CELL_SIZE // 2
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in self.COLOR_PLAYER), player_rect, 2, border_radius=3)

        # Draw maze walls
        if self.maze is not None:
            for y in range(self.GRID_SIZE):
                for x in range(self.GRID_SIZE):
                    cell_walls = self.maze[y, x]
                    px1 = self.X_OFFSET + x * self.CELL_SIZE
                    py1 = self.Y_OFFSET + y * self.CELL_SIZE
                    px2 = px1 + self.CELL_SIZE
                    py2 = py1 + self.CELL_SIZE
                    
                    if cell_walls[0]: pygame.draw.line(self.screen, self.COLOR_WALL, (px1, py1), (px2, py1), 3)
                    if cell_walls[1]: pygame.draw.line(self.screen, self.COLOR_WALL, (px2, py1), (px2, py2), 3)
                    if cell_walls[2]: pygame.draw.line(self.screen, self.COLOR_WALL, (px1, py2), (px2, py2), 3)
                    if cell_walls[3]: pygame.draw.line(self.screen, self.COLOR_WALL, (px1, py1), (px1, py2), 3)
    
    def _render_ui(self):
        # Display remaining moves
        moves_text = self.font.render(f"Moves Left: {self.remaining_moves}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 15))

        # Display score
        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(score_text, score_rect)

        # Display invalid move flash
        if self.show_flash:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_FLASH, 70))
            self.screen.blit(flash_surface, (0, 0))

        # Display win/loss message on game over
        if self.game_over:
            message, color = ("YOU WIN!", self.COLOR_EXIT) if self.player_pos == self.exit_pos else ("OUT OF MOVES", self.COLOR_FLASH)
            end_text = self.end_font.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.Y_OFFSET // 2))
            
            # Add a subtle background for the text
            bg_rect = text_rect.inflate(40, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((10, 10, 20, 200))
            self.screen.blit(bg_surf, bg_rect)
            pygame.draw.rect(self.screen, color, bg_rect, 2, 5)

            self.screen.blit(end_text, text_rect)

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
            "remaining_moves": self.remaining_moves,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
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