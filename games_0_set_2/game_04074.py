
# Generated: 2025-08-28T01:19:44.198153
# Source Brief: brief_04074.md
# Brief Index: 4074

        
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

    user_guide = "Controls: Use arrow keys to navigate the maze."
    game_description = "Navigate a procedurally generated maze, avoiding patrolling enemies, to reach the exit within the time limit."
    auto_advance = False
    
    # Class attribute for level progression across episodes
    _current_level = 1
    MAX_LEVEL = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Maze Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CELL_SIZE = 20
        self.MAZE_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE - 1 # 31
        self.MAZE_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE # 20 -> make it odd 19
        if self.MAZE_HEIGHT % 2 == 0: self.MAZE_HEIGHT -= 1

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_level = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_WALL = (20, 40, 80)
        self.COLOR_PLAYER = (255, 220, 0)
        self.COLOR_EXIT = (0, 255, 128)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_UI = (220, 220, 220)

        # Game State (initialized in reset)
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.enemies = []
        self.player_particles = []
        
        self.level = 1
        self.time_left = 0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.won = False
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()

    def _generate_maze(self):
        # Maze generation using recursive backtracking
        maze = np.ones((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.uint8)
        stack = deque()
        
        # Start at (1, 1)
        start_x, start_y = (1, 1)
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.MAZE_WIDTH -1 and 0 < ny < self.MAZE_HEIGHT - 1 and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                # Carve path
                maze[ny, nx] = 0
                maze[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _find_patrol_path(self, start_pos, length):
        # BFS to find a simple path for an enemy to patrol
        q = deque([(start_pos, [start_pos])])
        visited = {start_pos}
        while q:
            (x, y), path = q.popleft()
            if len(path) >= length:
                return path
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if self.maze[ny, nx] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = path + [(nx, ny)]
                    q.append(((nx, ny), new_path))
        return [start_pos] # Fallback

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Handle level progression
        if self.game_over and self.won:
             GameEnv._current_level = min(GameEnv._current_level + 1, self.MAX_LEVEL)
        # If the player fails, they retry the same level.
        self.level = GameEnv._current_level
        
        # Generate new maze and entities
        self.maze = self._generate_maze()
        self.start_pos = (1, 1)
        self.exit_pos = (self.MAZE_WIDTH - 2, self.MAZE_HEIGHT - 2)
        self.player_pos = list(self.start_pos)
        
        self.enemies = []
        num_enemies = self.level
        enemy_speed = 0.05 + (self.level - 1) * 0.015
        
        # Find valid spawn points for enemies
        possible_spawns = np.argwhere(self.maze == 0)
        self.np_random.shuffle(possible_spawns)
        spawn_idx = 0
        
        for _ in range(num_enemies):
            spawn_found = False
            while not spawn_found and spawn_idx < len(possible_spawns):
                y, x = possible_spawns[spawn_idx]
                spawn_idx += 1
                # Ensure enemy doesn't spawn too close to player start
                if math.dist((x,y), self.start_pos) > 10:
                    path = self._find_patrol_path((x, y), length=self.np_random.integers(5, 12))
                    self.enemies.append({
                        "pos": [float(x), float(y)],
                        "path": path,
                        "path_idx": 0,
                        "direction": 1,
                        "speed": enemy_speed
                    })
                    spawn_found = True
        
        # Reset state variables
        self.steps = 0
        self.time_left = 60 + (self.level - 1) * 5
        self.score = 0.0
        self.game_over = False
        self.won = False
        self.player_particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1  # Penalty for taking a step
        
        self._update_player(movement)
        self._update_enemies()
        self._update_particles()
        
        self.steps += 1
        self.time_left -= 1
        self.score += reward
        
        terminated = self._check_termination()
        
        if terminated:
            if self.won:
                reward += 100
            else: # Collision or timeout
                reward -= 10
            self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        px, py = self.player_pos
        
        # Add particle at old position
        self.player_particles.append({"pos": (px, py), "life": 15})

        if movement == 1 and self.maze[py - 1, px] == 0: # Up
            self.player_pos[1] -= 1
        elif movement == 2 and self.maze[py + 1, px] == 0: # Down
            self.player_pos[1] += 1
        elif movement == 3 and self.maze[py, px - 1] == 0: # Left
            self.player_pos[0] -= 1
        elif movement == 4 and self.maze[py, px + 1] == 0: # Right
            self.player_pos[0] += 1
        # No-op for movement == 0 or wall collision

    def _update_enemies(self):
        for enemy in self.enemies:
            if not enemy["path"] or len(enemy["path"]) <= 1:
                continue

            target_pos = enemy["path"][enemy["path_idx"]]
            direction_vec = (target_pos[0] - enemy["pos"][0], target_pos[1] - enemy["pos"][1])
            dist = math.hypot(*direction_vec)

            if dist < enemy["speed"]:
                enemy["pos"] = list(target_pos)
                enemy["path_idx"] += enemy["direction"]
                if not (0 <= enemy["path_idx"] < len(enemy["path"])):
                    enemy["direction"] *= -1
                    enemy["path_idx"] += 2 * enemy["direction"]
            else:
                norm_vec = (direction_vec[0] / dist, direction_vec[1] / dist)
                enemy["pos"][0] += norm_vec[0] * enemy["speed"]
                enemy["pos"][1] += norm_vec[1] * enemy["speed"]

    def _update_particles(self):
        self.player_particles = [p for p in self.player_particles if p["life"] > 0]
        for p in self.player_particles:
            p["life"] -= 1

    def _check_termination(self):
        # Win condition
        if tuple(self.player_pos) == self.exit_pos:
            self.game_over = True
            self.won = True
            return True
        
        # Loss conditions
        if self.time_left <= 0:
            self.game_over = True
            return True
        
        if self.steps >= 1800: # Hard step limit
            self.game_over = True
            return True

        for enemy in self.enemies:
            if tuple(self.player_pos) == (round(enemy["pos"][0]), round(enemy["pos"][1])):
                self.game_over = True
                return True
        
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_maze()
        self._render_entities()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_maze(self):
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                if self.maze[y, x] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, 
                                     (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

    def _render_entities(self):
        # Exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(ex * self.CELL_SIZE, ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        
        # Particles
        for p in self.player_particles:
            px, py = p["pos"]
            life_ratio = p["life"] / 15.0
            radius = int(self.CELL_SIZE * 0.2 * life_ratio)
            alpha = int(150 * life_ratio)
            color = (*self.COLOR_PLAYER, alpha)
            
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (radius, radius), radius)
            self.screen.blit(s, (px * self.CELL_SIZE + self.CELL_SIZE // 2 - radius, 
                                 py * self.CELL_SIZE + self.CELL_SIZE // 2 - radius))

        # Player
        px, py = self.player_pos
        center_x = int(px * self.CELL_SIZE + self.CELL_SIZE / 2)
        center_y = int(py * self.CELL_SIZE + self.CELL_SIZE / 2)
        radius = int(self.CELL_SIZE * 0.4)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)

        # Enemies
        for enemy in self.enemies:
            ex, ey = enemy["pos"]
            center_x = int(ex * self.CELL_SIZE + self.CELL_SIZE / 2)
            center_y = int(ey * self.CELL_SIZE + self.CELL_SIZE / 2)
            size = self.CELL_SIZE * 0.4
            
            points = [
                (center_x, center_y - size),
                (center_x - size, center_y + size * 0.7),
                (center_x + size, center_y + size * 0.7),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

    def _render_ui(self):
        level_text = self.font_level.render(f"Level: {self.level}/{self.MAX_LEVEL}", True, self.COLOR_UI)
        self.screen.blit(level_text, (10, 10))
        
        time_text = self.font_ui.render(f"Time: {self.time_left}", True, self.COLOR_UI)
        text_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time_left": self.time_left,
            "won": self.won
        }

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    total_score = 0
    
    while True:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_score += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Level: {info['level']}, Won: {info['won']}")
            obs, info = env.reset()
            total_score = 0
            pygame.time.wait(2000) # Pause before restarting
        
        # Since auto_advance is False, we need a small delay for human playability
        pygame.time.wait(100)