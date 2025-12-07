import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to navigate the maze. Reach the green exit before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze to reach the exit within a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAZE_DIM = 10
        self.CELL_SIZE = 40
        self.MAZE_WIDTH = self.MAZE_DIM * self.CELL_SIZE
        self.MAZE_HEIGHT = self.MAZE_DIM * self.CELL_SIZE
        self.X_OFFSET = (self.WIDTH - self.MAZE_WIDTH) // 2
        self.Y_OFFSET = (self.HEIGHT - self.MAZE_HEIGHT) // 2
        self.MAX_MOVES = 100
        self.FONT_SIZE_UI = 24
        self.FONT_SIZE_MSG = 48

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_EXIT_GLOW = (0, 255, 120, 50) # RGBA for glow
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PARTICLE = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, self.FONT_SIZE_UI)
        self.font_msg = pygame.font.Font(None, self.FONT_SIZE_MSG)
        
        # --- Game State (initialized in reset) ---
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.remaining_moves = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.remaining_moves = self.MAX_MOVES
        self.particles = []

        self.maze = self._generate_maze(self.MAZE_DIM, self.MAZE_DIM)
        self.player_pos = [0, 0]
        self.exit_pos = [self.MAZE_DIM - 1, self.MAZE_DIM - 1]
        
        return self._get_observation(), self._get_info()

    def _generate_maze(self, width, height):
        grid = [[{'walls': [True, True, True, True], 'visited': False} for _ in range(width)] for _ in range(height)]

        def get_neighbors(x, y):
            neighbors = []
            if x > 0 and not grid[y][x-1]['visited']: neighbors.append((x-1, y, 'L'))
            if x < width - 1 and not grid[y][x+1]['visited']: neighbors.append((x+1, y, 'R'))
            if y > 0 and not grid[y-1][x]['visited']: neighbors.append((x, y-1, 'U'))
            if y < height - 1 and not grid[y+1][x]['visited']: neighbors.append((x, y+1, 'D'))
            return neighbors

        stack = []
        cx, cy = 0, 0
        grid[cy][cx]['visited'] = True

        while True:
            unvisited_neighbors = get_neighbors(cx, cy)
            if unvisited_neighbors:
                stack.append((cx, cy))
                
                # Use the environment's RNG
                neighbor_idx = self.np_random.integers(len(unvisited_neighbors))
                nx, ny, direction = unvisited_neighbors[neighbor_idx]
                
                if direction == 'L':
                    grid[cy][cx]['walls'][3] = False
                    grid[ny][nx]['walls'][1] = False
                elif direction == 'R':
                    grid[cy][cx]['walls'][1] = False
                    grid[ny][nx]['walls'][3] = False
                elif direction == 'U':
                    grid[cy][cx]['walls'][0] = False
                    grid[ny][nx]['walls'][2] = False
                elif direction == 'D':
                    grid[cy][cx]['walls'][2] = False
                    grid[ny][nx]['walls'][0] = False

                cx, cy = nx, ny
                grid[cy][cx]['visited'] = True
            elif stack:
                cx, cy = stack.pop()
            else:
                break
        return grid

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0=none, 1=up, 2=down, 3=left, 4=right
        reward = -0.1  # Cost for taking a turn

        px, py = self.player_pos
        
        if movement != 0:
            nx, ny = px, py
            wall_idx = -1
            # 1=up, 2=down, 3=left, 4=right
            if movement == 1: ny, wall_idx = py - 1, 0 # Up
            elif movement == 2: ny, wall_idx = py + 1, 2 # Down
            elif movement == 3: nx, wall_idx = px - 1, 3 # Left
            elif movement == 4: nx, wall_idx = px + 1, 1 # Right
            
            if 0 <= nx < self.MAZE_DIM and 0 <= ny < self.MAZE_DIM:
                if not self.maze[py][px]['walls'][wall_idx]:
                    self.player_pos = [nx, ny]
                    self._create_move_particles()

        self.remaining_moves -= 1
        self.steps += 1
        
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 10.0
            self.score += 1
            terminated = True
            self.game_over = True
            self.win = True
        elif self.remaining_moves <= 0:
            reward -= 5.0 # Penalty for running out of moves
            terminated = True
            self.game_over = True

        self._update_particles()
        
        # Truncated is always False as termination is part of the MDP
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _create_move_particles(self):
        px, py = self.player_pos
        center_x = self.X_OFFSET + px * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.Y_OFFSET + py * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 21) # .integers is exclusive on high end
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'max_life': life})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 1]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame array is (width, height, 3), but observation space is (height, width, 3)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        # Draw Exit Glow
        ex, ey = self.exit_pos
        glow_size = self.CELL_SIZE * 1.5
        glow_rect = pygame.Rect(
            self.X_OFFSET + ex * self.CELL_SIZE - (glow_size - self.CELL_SIZE) // 2,
            self.Y_OFFSET + ey * self.CELL_SIZE - (glow_size - self.CELL_SIZE) // 2,
            glow_size, glow_size
        )
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_EXIT_GLOW, (glow_size // 2, glow_size // 2), glow_size // 2)
        self.screen.blit(glow_surface, glow_rect.topleft)

        # Draw Exit
        exit_rect = pygame.Rect(
            self.X_OFFSET + ex * self.CELL_SIZE + 4, self.Y_OFFSET + ey * self.CELL_SIZE + 4,
            self.CELL_SIZE - 8, self.CELL_SIZE - 8
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=4)
        
        # Draw Maze Walls
        for r in range(self.MAZE_DIM):
            for c in range(self.MAZE_DIM):
                x1 = self.X_OFFSET + c * self.CELL_SIZE
                y1 = self.Y_OFFSET + r * self.CELL_SIZE
                x2 = x1 + self.CELL_SIZE
                y2 = y1 + self.CELL_SIZE
                
                if self.maze[r][c]['walls'][0]: pygame.draw.line(self.screen, self.COLOR_WALL, (x1, y1), (x2, y1), 3)
                if self.maze[r][c]['walls'][1]: pygame.draw.line(self.screen, self.COLOR_WALL, (x2, y1), (x2, y2), 3)
                if self.maze[r][c]['walls'][2]: pygame.draw.line(self.screen, self.COLOR_WALL, (x1, y2), (x2, y2), 3)
                if self.maze[r][c]['walls'][3]: pygame.draw.line(self.screen, self.COLOR_WALL, (x1, y1), (x1, y2), 3)

        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = max(0, int(5 * (p['life'] / p['max_life'])))
            if size > 0:
                s = pygame.Surface((size, size), pygame.SRCALPHA)
                s.fill((self.COLOR_PARTICLE[0], self.COLOR_PARTICLE[1], self.COLOR_PARTICLE[2], alpha))
                self.screen.blit(s, (int(p['pos'][0] - size / 2), int(p['pos'][1] - size / 2)))

        # Draw Player
        px, py = self.player_pos
        player_center_x = self.X_OFFSET + px * self.CELL_SIZE + self.CELL_SIZE // 2
        player_center_y = self.Y_OFFSET + py * self.CELL_SIZE + self.CELL_SIZE // 2
        
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        radius = int(self.CELL_SIZE * 0.3 + pulse * 2)
        
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (player_center_x, player_center_y), radius)

    def _render_ui(self):
        moves_text = f"Moves Left: {self.remaining_moves}"
        text_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.win else "OUT OF MOVES"
            color = self.COLOR_EXIT if self.win else self.COLOR_WALL
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            bg_rect = msg_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 180))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "remaining_moves": self.remaining_moves}
    
    def close(self):
        pygame.quit()