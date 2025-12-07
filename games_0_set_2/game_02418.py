
# Generated: 2025-08-27T20:18:35.878157
# Source Brief: brief_02418.md
# Brief Index: 2418

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
from collections import deque
import heapq
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move. Avoid the red patient, collect 3 yellow keycards, and reach the green exit."
    )

    game_description = (
        "A survival horror game. Escape a procedurally generated asylum by finding keycards while being hunted by a relentless patient."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.CELL_SIZE = self.WIDTH // self.GRID_WIDTH
        self.MAX_STEPS = 1000
        self.NUM_KEYCARDS = 3

        # Colors
        self.COLOR_BG = (10, 15, 20)
        self.COLOR_WALL = (40, 50, 60)
        self.COLOR_FLOOR = (20, 25, 30)
        self.COLOR_PLAYER = (220, 220, 255)
        self.COLOR_PATIENT = (255, 50, 50)
        self.COLOR_KEYCARD = (255, 220, 0)
        self.COLOR_EXIT = (50, 255, 100)
        self.COLOR_TEXT = (200, 200, 200)
        self.COLOR_TRAIL_START = (120, 0, 0)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)

        # Initialize state variables
        self.grid = None
        self.player_pos = None
        self.patient_pos = None
        self.exit_pos = None
        self.keycard_locs = None
        self.keycards_collected = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.patient_trail = deque(maxlen=15)
        self.last_known_player_pos = None
        self.patient_patrol_target = None

        self.reset()
        self.validate_implementation()

    def _generate_maze(self):
        grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.uint8)  # 1 for wall
        stack = [(1, 1)]
        grid[1, 1] = 0  # 0 for path

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.GRID_WIDTH - 1 and 0 < ny < self.GRID_HEIGHT - 1 and grid[nx, ny] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                grid[nx, ny] = 0
                grid[x + (nx - x) // 2, y + (dy - y) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return grid

    def _find_farthest_point(self, start_pos):
        q = deque([(start_pos, 0)])
        visited = {start_pos}
        farthest_point = start_pos
        max_dist = 0

        while q:
            (x, y), dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                farthest_point = (x, y)

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append(((nx, ny), dist + 1))
        return farthest_point

    def _get_path_a_star(self, start, end):
        open_set = [(0, start)]
        came_from = {}
        g_score = { (x,y): float('inf') for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT) }
        g_score[start] = 0
        f_score = { (x,y): float('inf') for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT) }
        f_score[start] = abs(start[0] - end[0]) + abs(start[1] - end[1])

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.GRID_WIDTH and 0 <= neighbor[1] < self.GRID_HEIGHT and self.grid[neighbor[0], neighbor[1]] == 0:
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def _has_line_of_sight(self, start, end):
        x0, y0 = start
        x1, y1 = end
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            if self.grid[x0, y0] == 1: return False
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._generate_maze()
        
        floor_tiles = np.argwhere(self.grid == 0).tolist()
        random.shuffle(floor_tiles)

        self.player_pos = tuple(floor_tiles.pop())
        self.exit_pos = self._find_farthest_point(self.player_pos)
        
        # Ensure exit is not a spawn point for other items
        if self.exit_pos in floor_tiles:
            floor_tiles.remove(list(self.exit_pos))

        self.keycard_locs = [tuple(floor_tiles.pop()) for _ in range(self.NUM_KEYCARDS)]
        
        # Place patient away from player
        while True:
            self.patient_pos = tuple(floor_tiles.pop())
            dist = abs(self.player_pos[0] - self.patient_pos[0]) + abs(self.player_pos[1] - self.patient_pos[1])
            if dist > 10:
                break
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.keycards_collected = 0
        self.patient_trail.clear()
        self.last_known_player_pos = None
        self.patient_patrol_target = None
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = -0.1  # Cost of living
        terminated = False
        self.steps += 1

        # --- Player Turn ---
        px, py = self.player_pos
        dist_before = abs(px - self.patient_pos[0]) + abs(py - self.patient_pos[1])
        
        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right

        if self.grid[px, py] == 0: # Check for wall
            self.player_pos = (px, py)
        
        # Check for interactions
        if self.player_pos in self.keycard_locs:
            self.keycard_locs.remove(self.player_pos)
            self.keycards_collected += 1
            reward += 10.0
            # Sfx: keycard_pickup.wav
        
        if self.player_pos == self.exit_pos and self.keycards_collected == self.NUM_KEYCARDS:
            reward += 100.0
            terminated = True
            self.game_over = True
            self.game_over_message = "ESCAPED!"
            # Sfx: win_sound.wav
        
        dist_after = abs(self.player_pos[0] - self.patient_pos[0]) + abs(self.player_pos[1] - self.patient_pos[1])
        if dist_after > dist_before:
            reward += 0.2

        # --- Patient Turn ---
        if not terminated:
            self.patient_trail.append(self.patient_pos)
            
            if self._has_line_of_sight(self.patient_pos, self.player_pos):
                self.last_known_player_pos = self.player_pos
                self.patient_patrol_target = None # Stop patrolling, start hunting
            
            target = self.last_known_player_pos if self.last_known_player_pos else self.patient_patrol_target

            if target:
                path = self._get_path_a_star(self.patient_pos, target)
                if path and len(path) > 0:
                    self.patient_pos = path[0]
                    if self.patient_pos == self.last_known_player_pos:
                        self.last_known_player_pos = None # Reached last known spot
                else: # Can't reach target, reset
                    self.last_known_player_pos = None
                    self.patient_patrol_target = None
            else: # Random patrol
                floor_tiles = np.argwhere(self.grid == 0)
                self.patient_patrol_target = tuple(random.choice(floor_tiles))

        # --- Check Termination Conditions ---
        if self.player_pos == self.patient_pos:
            reward = -100.0
            terminated = True
            self.game_over = True
            self.game_over_message = "CAUGHT!"
            # Sfx: player_death.wav
            
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.game_over_message = "TIME'S UP!"

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render game grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if self.grid[x, y] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)

        # Render patient trail
        for i, pos in enumerate(self.patient_trail):
            alpha = (i + 1) / len(self.patient_trail)
            trail_color = tuple(c * alpha for c in self.COLOR_TRAIL_START)
            rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, trail_color, rect)

        # Render exit
        exit_rect = pygame.Rect(self.exit_pos[0] * self.CELL_SIZE, self.exit_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Render keycards
        for kx, ky in self.keycard_locs:
            key_rect = pygame.Rect(kx * self.CELL_SIZE + self.CELL_SIZE // 4, ky * self.CELL_SIZE + self.CELL_SIZE // 4, self.CELL_SIZE // 2, self.CELL_SIZE // 2)
            pygame.draw.rect(self.screen, self.COLOR_KEYCARD, key_rect)
            
        # Render patient
        patient_rect = pygame.Rect(self.patient_pos[0] * self.CELL_SIZE, self.patient_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PATIENT, patient_rect)

        # Render player
        player_rect = pygame.Rect(self.player_pos[0] * self.CELL_SIZE, self.player_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        # Render UI
        keycard_text = self.font_small.render(f"Keycards: {self.keycards_collected} / {self.NUM_KEYCARDS}", True, self.COLOR_TEXT)
        self.screen.blit(keycard_text, (10, 10))
        
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.game_over_message, True, self.COLOR_PATIENT if "CAUGHT" in self.game_over_message else self.COLOR_EXIT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "keycards_collected": self.keycards_collected,
            "player_pos": self.player_pos,
            "patient_pos": self.patient_pos,
        }

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        assert self.grid[self.player_pos] == 0
        assert self.grid[self.patient_pos] == 0
        assert all(self.grid[pos] == 0 for pos in self.keycard_locs)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Asylum Escape")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    terminated = False
    
    print(env.user_guide)
    
    while running:
        action = np.array([0, 0, 0]) # Default action is no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                
                if not terminated:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    
                    # Only step if a movement key was pressed
                    if action[0] != 0:
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Step: {info['steps']}, Score: {info['score']:.1f}, Reward: {reward:.1f}, Terminated: {terminated}")

        # Draw the observation to the screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()