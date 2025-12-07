
# Generated: 2025-08-27T18:49:10.557277
# Source Brief: brief_01962.md
# Brief Index: 1962

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. "
        "Eat all the pellets to win while avoiding the ghosts!"
    )

    game_description = (
        "A classic arcade maze game. Navigate a procedurally generated maze, "
        "eat pellets for points, and grab power-ups to turn the tables on the ghosts that hunt you."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 31, 19 # Must be odd
        self.CELL_SIZE = min(self.WIDTH // (self.GRID_WIDTH + 1), self.HEIGHT // (self.GRID_HEIGHT + 1))
        self.MAZE_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.MAZE_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.OFFSET_X = (self.WIDTH - self.MAZE_WIDTH) // 2
        self.OFFSET_Y = (self.HEIGHT - self.MAZE_HEIGHT) // 2

        self.NUM_PELLETS = 100
        self.NUM_POWER_PELLETS = 4
        self.MAX_STEPS = 1000
        self.POWER_UP_DURATION = 20 # in steps

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_WALL = (20, 40, 150)
        self.COLOR_PELLET = (220, 220, 255)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_GHOST_VULNERABLE = (50, 50, 255)
        self.COLOR_GHOST_VULNERABLE_FLASH = (255, 255, 255)
        self.GHOST_COLORS = [(255, 0, 0), (255, 184, 255), (0, 255, 255), (255, 184, 82)]
        self.COLOR_UI_TEXT = (255, 255, 255)

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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)

        # Initialize state variables
        self.player = None
        self.ghosts = []
        self.maze = []
        self.pellets = set()
        self.power_pellets = set()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.power_up_timer = 0
        self.total_pellets = 0

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.power_up_timer = 0

        self._generate_maze()
        
        path_cells = [(x, y) for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH) if self.maze[y][x] == 0]
        self.np_random.shuffle(path_cells)

        player_pos = path_cells.pop()
        self.player = Player(player_pos[0], player_pos[1], self.CELL_SIZE)

        self.pellets = set(random.sample(path_cells, k=min(self.NUM_PELLETS, len(path_cells))))
        path_cells = [p for p in path_cells if p not in self.pellets]
        
        self.power_pellets = set(random.sample(path_cells, k=min(self.NUM_POWER_PELLETS, len(path_cells))))
        path_cells = [p for p in path_cells if p not in self.power_pellets]
        
        self.total_pellets = len(self.pellets) + len(self.power_pellets)

        self.ghosts = []
        patrol_points = random.sample(path_cells, k=min(len(self.GHOST_COLORS) * 2, len(path_cells)))
        for i, color in enumerate(self.GHOST_COLORS):
            if len(patrol_points) < 2: break
            start_pos = patrol_points.pop()
            end_pos = patrol_points.pop()
            patrol_path = self._find_path(start_pos, end_pos)
            if patrol_path:
                self.ghosts.append(Ghost(start_pos[0], start_pos[1], color, patrol_path, self.CELL_SIZE))

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = -0.2  # Cost for taking a step
        self.game_over = False

        if self.power_up_timer > 0:
            self.power_up_timer -= 1
            if self.power_up_timer == 0:
                for ghost in self.ghosts:
                    ghost.is_vulnerable = False
                    # SFX: Power-down sound

        # 1. Update Player
        prev_pos = (self.player.grid_x, self.player.grid_y)
        self.player.move(movement, self.maze)
        new_pos = (self.player.grid_x, self.player.grid_y)

        # 2. Update Ghosts
        for ghost in self.ghosts:
            ghost.update(self.maze, (self.player.grid_x, self.player.grid_y), self.power_up_timer > 0)

        # 3. Check Collisions & Collectibles
        if new_pos in self.pellets:
            self.pellets.remove(new_pos)
            self.score += 1
            reward += 1.0
            # SFX: Pellet munch sound

        if new_pos in self.power_pellets:
            self.power_pellets.remove(new_pos)
            self.score += 5
            reward += 10.0
            self.power_up_timer = self.POWER_UP_DURATION
            for ghost in self.ghosts:
                ghost.become_vulnerable()
            # SFX: Power-up activation sound

        for ghost in self.ghosts:
            if ghost.grid_x == self.player.grid_x and ghost.grid_y == self.player.grid_y:
                if ghost.is_vulnerable:
                    self.score += 10
                    reward += 5.0
                    ghost.respawn()
                    # SFX: Ghost eaten sound
                else:
                    self.game_over = True
                    reward += -50.0
                    # SFX: Player death sound
                    break

        # 4. Check Termination Conditions
        self.steps += 1
        terminated = self.game_over
        
        if not self.pellets and not self.power_pellets:
            terminated = True
            reward += 100.0 # Win bonus
            # SFX: Level complete sound
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_maze()
        self._render_collectibles()
        self._render_entities()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "power_up": self.power_up_timer > 0}

    def _to_screen_coords(self, grid_x, grid_y):
        return (
            self.OFFSET_X + grid_x * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.OFFSET_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE // 2,
        )

    def _render_maze(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.maze[y][x] == 1:
                    rect = pygame.Rect(
                        self.OFFSET_X + x * self.CELL_SIZE,
                        self.OFFSET_Y + y * self.CELL_SIZE,
                        self.CELL_SIZE,
                        self.CELL_SIZE,
                    )
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

    def _render_collectibles(self):
        pellet_radius = max(1, self.CELL_SIZE // 8)
        for x, y in self.pellets:
            sx, sy = self._to_screen_coords(x, y)
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, pellet_radius, self.COLOR_PELLET)

        power_pellet_radius = int(self.CELL_SIZE / 3.5)
        pulse = abs(math.sin(self.steps * 0.3)) * 0.2 + 0.8
        for x, y in self.power_pellets:
            sx, sy = self._to_screen_coords(x, y)
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, int(power_pellet_radius * pulse), self.COLOR_PELLET)
            pygame.gfxdraw.aacircle(self.screen, sx, sy, int(power_pellet_radius * pulse), self.COLOR_PELLET)

    def _render_entities(self):
        self.player.render(self.screen, self._to_screen_coords, self.COLOR_PLAYER, self.steps)
        for ghost in self.ghosts:
            ghost.render(self.screen, self._to_screen_coords, self.COLOR_GHOST_VULNERABLE, self.COLOR_GHOST_VULNERABLE_FLASH, self.steps, self.power_up_timer)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

    def _generate_maze(self):
        self.maze = [[1] * self.GRID_WIDTH for _ in range(self.GRID_HEIGHT)]
        start_x, start_y = (self.np_random.integers(0, self.GRID_WIDTH // 2) * 2,
                            self.np_random.integers(0, self.GRID_HEIGHT // 2) * 2)
        
        stack = [(start_x, start_y)]
        self.maze[start_y][start_x] = 0

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.maze[ny][nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                wall_x, wall_y = (x + nx) // 2, (y + ny) // 2
                self.maze[ny][nx] = 0
                self.maze[wall_y][wall_x] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _find_path(self, start, end):
        q = deque([(start, [start])])
        visited = {start}
        while q:
            (x, y), path = q.popleft()
            if (x, y) == end:
                return path
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and
                        self.maze[ny][nx] == 0 and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    new_path = list(path)
                    new_path.append((nx, ny))
                    q.append(((nx, ny), new_path))
        return None

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

class Player:
    def __init__(self, x, y, cell_size):
        self.grid_x, self.grid_y = x, y
        self.cell_size = cell_size
        self.direction = 0  # 0: none, 1: up, 2: down, 3: left, 4: right
        self.last_valid_direction = 4 # Start facing right

    def move(self, action, maze):
        dx, dy = 0, 0
        if action == 1: # Up
            dx, dy = 0, -1
            self.direction = 1
        elif action == 2: # Down
            dx, dy = 0, 1
            self.direction = 2
        elif action == 3: # Left
            dx, dy = -1, 0
            self.direction = 3
        elif action == 4: # Right
            dx, dy = 1, 0
            self.direction = 4
        
        if action != 0:
            self.last_valid_direction = self.direction

        nx, ny = self.grid_x + dx, self.grid_y + dy
        grid_height, grid_width = len(maze), len(maze[0])
        
        if 0 <= nx < grid_width and 0 <= ny < grid_height and maze[ny][nx] == 0:
            self.grid_x, self.grid_y = nx, ny
            # SFX: Player move (subtle)

    def render(self, surface, to_screen, color, steps):
        sx, sy = to_screen(self.grid_x, self.grid_y)
        radius = self.cell_size // 2 - 2
        
        # Animate mouth
        mouth_angle = (math.sin(steps * 0.8) * 25 + 35) * (math.pi / 180)
        
        p1 = (sx, sy)
        points = [p1]
        
        start_rad, end_rad = 0, 2 * math.pi
        if self.last_valid_direction == 1: # Up
            start_rad, end_rad = 0.75 * math.pi + mouth_angle, 1.25 * math.pi - mouth_angle
        elif self.last_valid_direction == 2: # Down
            start_rad, end_rad = -0.25 * math.pi + mouth_angle, 0.25 * math.pi - mouth_angle
        elif self.last_valid_direction == 3: # Left
            start_rad, end_rad = 0.5 * math.pi + mouth_angle, 1.5 * math.pi - mouth_angle
        elif self.last_valid_direction == 4: # Right
            start_rad, end_rad = mouth_angle, 2 * math.pi - mouth_angle
            
        for i in range(int(start_rad * 100), int(end_rad * 100)):
            angle = i / 100.0
            points.append((sx + radius * math.cos(angle), sy + radius * math.sin(angle)))
        
        if len(points) > 2:
            pygame.gfxdraw.aapolygon(surface, [(int(p[0]), int(p[1])) for p in points], color)
            pygame.gfxdraw.filled_polygon(surface, [(int(p[0]), int(p[1])) for p in points], color)

class Ghost:
    def __init__(self, x, y, color, patrol_path, cell_size):
        self.start_x, self.start_y = x, y
        self.grid_x, self.grid_y = x, y
        self.color = color
        self.cell_size = cell_size
        self.is_vulnerable = False
        self.respawn_timer = 0
        self.direction = (0, 0)
        
        self.patrol_path = patrol_path
        self.patrol_index = 0
        self.patrol_forward = True

    def become_vulnerable(self):
        self.is_vulnerable = True
        self.patrol_forward = not self.patrol_forward # Reverse direction

    def respawn(self):
        self.grid_x, self.grid_y = self.start_x, self.start_y
        self.is_vulnerable = False
        self.respawn_timer = 15 # Can't be eaten again immediately
        self.patrol_index = 0
        self.patrol_forward = True
    
    def update(self, maze, player_pos, is_power_up_active):
        if self.respawn_timer > 0:
            self.respawn_timer -= 1
            return

        self.is_vulnerable = is_power_up_active
        
        if self.is_vulnerable:
            self._flee(maze, player_pos)
        else:
            self._patrol()

    def _patrol(self):
        if not self.patrol_path: return
        
        if self.patrol_forward:
            self.patrol_index += 1
            if self.patrol_index >= len(self.patrol_path):
                self.patrol_index = len(self.patrol_path) - 2
                self.patrol_forward = False
        else:
            self.patrol_index -= 1
            if self.patrol_index < 0:
                self.patrol_index = 1
                self.patrol_forward = True
        
        next_pos = self.patrol_path[self.patrol_index]
        self.direction = (next_pos[0] - self.grid_x, next_pos[1] - self.grid_y)
        self.grid_x, self.grid_y = next_pos

    def _flee(self, maze, player_pos):
        best_move = (self.grid_x, self.grid_y)
        max_dist = -1

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = self.grid_x + dx, self.grid_y + dy
            if 0 <= nx < len(maze[0]) and 0 <= ny < len(maze) and maze[ny][nx] == 0:
                dist = abs(nx - player_pos[0]) + abs(ny - player_pos[1])
                if dist > max_dist:
                    max_dist = dist
                    best_move = (nx, ny)
        
        self.direction = (best_move[0] - self.grid_x, best_move[1] - self.grid_y)
        self.grid_x, self.grid_y = best_move

    def render(self, surface, to_screen, vulnerable_color, flash_color, steps, power_up_timer):
        if self.respawn_timer > 0: return # Invisible while respawning

        sx, sy = to_screen(self.grid_x, self.grid_y)
        w = self.cell_size * 0.8
        h = self.cell_size * 0.8
        r = w / 2
        
        body_color = self.color
        if self.is_vulnerable:
            # Flash when power-up is about to end
            if power_up_timer < 8 and steps % 2 == 0:
                body_color = flash_color
            else:
                body_color = vulnerable_color

        # Body
        body_rect = pygame.Rect(sx - w/2, sy - h/2, w, h)
        pygame.draw.rect(surface, body_color, body_rect, border_top_left_radius=int(r), border_top_right_radius=int(r))

        # Wavy bottom
        for i in range(4):
            bx = sx - w/2 + i * (w/4)
            pygame.draw.circle(surface, body_color, (int(bx + w/8), int(sy + h/2)), int(w/8))

        # Eyes
        eye_radius = int(w / 6)
        pupil_radius = int(eye_radius / 2)
        eye_lx, eye_rx = sx - w/4, sx + w/4
        eye_y = sy - h/8

        pygame.draw.circle(surface, (255, 255, 255), (int(eye_lx), int(eye_y)), eye_radius)
        pygame.draw.circle(surface, (255, 255, 255), (int(eye_rx), int(eye_y)), eye_radius)

        if self.is_vulnerable:
            pygame.draw.circle(surface, (0,0,0), (int(eye_lx), int(eye_y)), pupil_radius)
            pygame.draw.circle(surface, (0,0,0), (int(eye_rx), int(eye_y)), pupil_radius)
        else:
            pupil_dx = self.direction[0] * pupil_radius * 0.7
            pupil_dy = self.direction[1] * pupil_radius * 0.7
            pygame.draw.circle(surface, (0,0,0), (int(eye_lx + pupil_dx), int(eye_y + pupil_dy)), pupil_radius)
            pygame.draw.circle(surface, (0,0,0), (int(eye_rx + pupil_dx), int(eye_y + pupil_dy)), pupil_radius)