
# Generated: 2025-08-28T04:47:57.395770
# Source Brief: brief_02427.md
# Brief Index: 2427

        
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

    user_guide = (
        "Use arrow keys to select a direction, and press Space to move. Reach the green exit in 100 moves or less. Red tiles are risky shortcuts!"
    )

    game_description = (
        "Navigate a procedurally generated maze with limited moves. Plan your path, use risky shortcuts wisely, and try to reach the exit with the highest score."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()

        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 20
        self.CELL_SIZE = self.SCREEN_HEIGHT // self.GRID_SIZE
        self.MAZE_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.MAZE_OFFSET_Y = 0
        self.WALL_THICKNESS = 2

        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.COLOR_BG = (20, 25, 40)
        self.COLOR_WALL = (60, 70, 90)
        self.COLOR_PATH_FLOOR = (40, 45, 60)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_RISK = (200, 50, 50)
        self.COLOR_TRAIL = (100, 110, 130)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_INTENT = (255, 255, 255)

        self.font_ui = pygame.font.SysFont("Consolas", 24)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.np_random = None
        self.maze = None
        self.player_pos = None
        self.start_pos = None
        self.exit_pos = None
        self.distance_grid = None
        self.risky_tiles = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.remaining_moves = 0
        self.path_trail = []
        self.intended_direction = None

        self.reset()
        self.validate_implementation()

    def _generate_maze(self, width, height):
        maze = [[{'N': True, 'S': True, 'E': True, 'W': True, 'visited': False} for _ in range(width)] for _ in range(height)]
        dirs = {'N': (0, -1, 'S'), 'S': (0, 1, 'N'), 'E': (1, 0, 'W'), 'W': (-1, 0, 'E')}
        
        stack = [(0, 0)]
        maze[0][0]['visited'] = True
        
        while stack:
            cx, cy = stack[-1]
            
            shuffled_dirs = list(dirs.items())
            self.np_random.shuffle(shuffled_dirs)
            
            neighbors = []
            for d, (dx, dy, opposite_wall) in shuffled_dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and not maze[ny][nx]['visited']:
                    neighbors.append((nx, ny, d, opposite_wall))
            
            if neighbors:
                nx, ny, d, opposite_wall = neighbors[0]
                maze[cy][cx][d] = False
                maze[ny][nx][opposite_wall] = False
                maze[ny][nx]['visited'] = True
                stack.append((nx, ny))
            else:
                stack.pop()
                
        final_maze = [[{k: v for k, v in cell.items() if k != 'visited'} for cell in row] for row in maze]
        return final_maze

    def _calculate_distance_grid(self):
        grid = [[-1 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        q = deque([(self.exit_pos, 0)])
        grid[self.exit_pos[1]][self.exit_pos[0]] = 0
        dirs = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}
        
        while q:
            (cx, cy), dist = q.popleft()
            for d, (dx, dy) in dirs.items():
                if not self.maze[cy][cx][d]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and grid[ny][nx] == -1:
                        grid[ny][nx] = dist + 1
                        q.append(((nx, ny), dist + 1))
        return grid

    def _identify_risky_tiles(self):
        risky = set()
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                wall_count = sum(self.maze[y][x].values())
                if wall_count > 0 and (x, y) != self.exit_pos and (x, y) != self.start_pos:
                    risky.add((x, y))
        return risky

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        self.start_pos = (0, 0)
        self.exit_pos = (self.GRID_SIZE - 1, self.GRID_SIZE - 1)
        self.maze = self._generate_maze(self.GRID_SIZE, self.GRID_SIZE)
        self.distance_grid = self._calculate_distance_grid()
        self.risky_tiles = self._identify_risky_tiles()

        self.player_pos = self.start_pos
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.remaining_moves = 100
        self.path_trail = []
        self.intended_direction = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        if movement == 1: self.intended_direction = 0  # Up
        elif movement == 2: self.intended_direction = 1  # Down
        elif movement == 3: self.intended_direction = 2  # Left
        elif movement == 4: self.intended_direction = 3  # Right

        if space_held and self.intended_direction is not None:
            px, py = self.player_pos
            moves = [(0, -1, 'N'), (0, 1, 'S'), (-1, 0, 'W'), (1, 0, 'E')]
            dx, dy, wall_dir = moves[self.intended_direction]
            nx, ny = px + dx, py + dy

            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and not self.maze[py][px][wall_dir]:
                self.path_trail.append(self.player_pos)
                if len(self.path_trail) > 25: self.path_trail.pop(0)

                old_dist = self.distance_grid[py][px]
                self.player_pos = (nx, ny)
                new_dist = self.distance_grid[ny][nx]
                
                if new_dist < old_dist: reward += 1.0
                else: reward -= 0.2
                
                if self.player_pos in self.risky_tiles:
                    reward -= 1.0
                    # sound: risky_step.wav

                self.remaining_moves -= 1
                # sound: step.wav
            else:
                # sound: bump_wall.wav
                pass

        self.steps += 1
        self.score += reward

        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 50.0
            self.score += 50.0
            terminated = True
            self.game_over = True
            # sound: win.wav
        elif self.remaining_moves <= 0:
            terminated = True
            self.game_over = True
            # sound: lose.wav

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _grid_to_pixels(self, x, y):
        return (
            self.MAZE_OFFSET_X + x * self.CELL_SIZE,
            self.MAZE_OFFSET_Y + y * self.CELL_SIZE
        )

    def _render_game(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                px, py = self._grid_to_pixels(x, y)
                pygame.draw.rect(self.screen, self.COLOR_PATH_FLOOR, (px, py, self.CELL_SIZE, self.CELL_SIZE))

        for x, y in self.risky_tiles:
            px, py = self._grid_to_pixels(x, y)
            risk_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            risk_surface.fill((*self.COLOR_RISK, 50))
            self.screen.blit(risk_surface, (px, py))

        for i, (x, y) in enumerate(self.path_trail):
            px, py = self._grid_to_pixels(x, y)
            center_x, center_y = px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2
            radius = int(self.CELL_SIZE * 0.2 * (i / len(self.path_trail)))
            if radius > 1:
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, (*self.COLOR_TRAIL, 150))

        ex, ey = self._grid_to_pixels(*self.exit_pos)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex, ey, self.CELL_SIZE, self.CELL_SIZE))

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                px, py = self._grid_to_pixels(x, y)
                if self.maze[y][x]['N']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.CELL_SIZE, py), self.WALL_THICKNESS)
                if self.maze[y][x]['S']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.CELL_SIZE), (px + self.CELL_SIZE, py + self.CELL_SIZE), self.WALL_THICKNESS)
                if self.maze[y][x]['W']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + self.CELL_SIZE), self.WALL_THICKNESS)
                if self.maze[y][x]['E']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.CELL_SIZE, py), (px + self.CELL_SIZE, py + self.CELL_SIZE), self.WALL_THICKNESS)

        if self.intended_direction is not None and not self.game_over:
            px, py = self.player_pos
            moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dx, dy = moves[self.intended_direction]
            nx, ny = px + dx, py + dy
            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                ix, iy = self._grid_to_pixels(nx, ny)
                indicator_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(indicator_surface, (*self.COLOR_INTENT, 60), (2, 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4), border_radius=4)
                self.screen.blit(indicator_surface, (ix, iy))

        px, py = self._grid_to_pixels(*self.player_pos)
        center_x, center_y = px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2
        radius = int(self.CELL_SIZE * 0.35)
        glow_radius = radius + int(4 * (1 + math.sin(self.steps * 0.3)))
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, (*self.COLOR_PLAYER, 50))
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, glow_radius, (*self.COLOR_PLAYER, 70))
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)

    def _render_ui(self):
        moves_text = self.font_ui.render(f"Moves: {self.remaining_moves}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (15, 10))
        
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 15, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WON!" if self.player_pos == self.exit_pos else "OUT OF MOVES"
            color = self.COLOR_EXIT if self.player_pos == self.exit_pos else self.COLOR_RISK
            
            end_text = self.font_game_over.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
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
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    pygame.display.init()
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    action_held_timer = 0
    ACTION_DELAY = 5 # Number of frames an action is held

    while running:
        movement, space = 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False
                continue

        if not done:
            keys = pygame.key.get_pressed()
            
            # Allow continuous holding of action keys
            if action_held_timer == 0:
                if keys[pygame.K_UP]: movement = 1
                elif keys[pygame.K_DOWN]: movement = 2
                elif keys[pygame.K_LEFT]: movement = 3
                elif keys[pygame.K_RIGHT]: movement = 4
                
                if keys[pygame.K_SPACE]: space = 1

                if movement or space:
                    action_held_timer = ACTION_DELAY
                    action = np.array([movement, space, 0])
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                else: # No-op action
                    obs, reward, terminated, truncated, info = env.step(np.array([0,0,0]))

            else:
                action_held_timer -= 1
                # Send a no-op while waiting
                obs, reward, terminated, truncated, info = env.step(np.array([0,0,0]))

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) 

    env.close()