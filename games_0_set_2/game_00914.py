
# Generated: 2025-08-27T15:10:57.906092
# Source Brief: brief_00914.md
# Brief Index: 914

        
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

    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. Reach the green exit before you run out of moves."
    )

    game_description = (
        "A top-down puzzle game where you navigate a procedurally generated maze. Each move costs a point. Reach the exit for a large bonus. Plan your path carefully to maximize your score."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CELL_SIZE = 16
        self.MAZE_CELL_WIDTH = 19
        self.MAZE_CELL_HEIGHT = 12
        self.GRID_WIDTH = self.MAZE_CELL_WIDTH * 2 + 1
        self.GRID_HEIGHT = self.MAZE_CELL_HEIGHT * 2 + 1
        
        self.maze_render_width = self.GRID_WIDTH * self.CELL_SIZE
        self.maze_render_height = self.GRID_HEIGHT * self.CELL_SIZE
        self.maze_offset_x = (self.SCREEN_WIDTH - self.maze_render_width) // 2
        self.maze_offset_y = (self.SCREEN_HEIGHT - self.maze_render_height) // 2

        self.MAX_MOVES = (self.MAZE_CELL_WIDTH * self.MAZE_CELL_HEIGHT) // 2
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (50, 50, 60)
        self.COLOR_PATH = (35, 35, 45)
        self.COLOR_TRAIL = (120, 40, 40)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_INNER = (200, 255, 255)
        self.COLOR_EXIT = (0, 255, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SUCCESS = (100, 255, 150)
        self.COLOR_FAIL = (255, 100, 100)

        # --- State Variables ---
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.path_taken = None
        self.moves_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False

        self.reset()
        self.validate_implementation()

    def _generate_maze(self):
        grid = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.uint8)
        
        # Use numpy's random generator
        start_x = self.np_random.integers(0, self.MAZE_CELL_WIDTH) * 2 + 1
        start_y = self.np_random.integers(0, self.MAZE_CELL_HEIGHT) * 2 + 1
        
        stack = [(start_x, start_y)]
        grid[start_y, start_x] = 0

        while stack:
            cx, cy = stack[-1]
            
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and grid[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                nx, ny = int(nx), int(ny) # choice returns array
                
                # Knock down wall
                grid[ny, nx] = 0
                grid[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                
                stack.append((nx, ny))
            else:
                stack.pop()
        
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.maze = self._generate_maze()
        self.player_pos = (1, 1)
        self.exit_pos = (self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2)
        
        # Ensure exit is not walled off
        self.maze[self.exit_pos[1], self.exit_pos[0]] = 0

        self.path_taken = [self.player_pos]
        self.moves_remaining = self.MAX_MOVES
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        terminated = False
        
        px, py = self.player_pos
        nx, ny = px, py

        moved = False
        if movement == 1: # Up
            ny -= 1
            moved = True
        elif movement == 2: # Down
            ny += 1
            moved = True
        elif movement == 3: # Left
            nx -= 1
            moved = True
        elif movement == 4: # Right
            nx += 1
            moved = True

        if moved:
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.maze[ny, nx] == 0:
                self.player_pos = (nx, ny)
                if self.player_pos not in self.path_taken:
                    self.path_taken.append(self.player_pos)
                reward = -1
                self.moves_remaining -= 1
            else:
                # Optional: penalty for hitting a wall
                # reward = -2 
                pass
        
        self.score += reward
        self.steps += 1

        if self.player_pos == self.exit_pos:
            reward = 100
            self.score += reward
            terminated = True
            self.game_over = True
            self.win_condition = True
        elif self.moves_remaining <= 0:
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
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

    def _render_game(self):
        # Render maze paths and trail
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_x = self.maze_offset_x + x * self.CELL_SIZE
                screen_y = self.maze_offset_y + y * self.CELL_SIZE
                rect = (screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE)
                
                if self.maze[y, x] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                elif (x, y) in self.path_taken:
                    pygame.draw.rect(self.screen, self.COLOR_TRAIL, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_PATH, rect)

        # Render Exit
        exit_x = self.maze_offset_x + self.exit_pos[0] * self.CELL_SIZE
        exit_y = self.maze_offset_y + self.exit_pos[1] * self.CELL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (exit_x, exit_y, self.CELL_SIZE, self.CELL_SIZE))

        # Render Player
        player_x = self.maze_offset_x + self.player_pos[0] * self.CELL_SIZE
        player_y = self.maze_offset_y + self.player_pos[1] * self.CELL_SIZE
        player_rect = (player_x, player_y, self.CELL_SIZE, self.CELL_SIZE)
        inner_padding = self.CELL_SIZE // 4
        inner_rect = (player_x + inner_padding, player_y + inner_padding, self.CELL_SIZE - 2 * inner_padding, self.CELL_SIZE - 2 * inner_padding)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_INNER, inner_rect)

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_ui.render(f"MOVES: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 10))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(score_text, score_rect)

        # Game Over Message
        if self.game_over:
            if self.win_condition:
                msg_text = self.font_msg.render("SUCCESS!", True, self.COLOR_SUCCESS)
            else:
                msg_text = self.font_msg.render("OUT OF MOVES", True, self.COLOR_FAIL)
            
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Add a semi-transparent background for the message
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default action is no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset the game
                    obs, info = env.reset()
                    done = False
                
                # Only register one-time key presses for turn-based movement
                if not done:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    
                    # If a move was made, step the environment
                    if action[0] != 0:
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated

        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    env.close()