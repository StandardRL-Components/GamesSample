
# Generated: 2025-08-28T05:42:25.651367
# Source Brief: brief_02709.md
# Brief Index: 2709

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Ghost:
    """A helper class to manage the state of a single ghost."""
    def __init__(self, pos, color, patrol_path):
        self.start_pos = pos
        self.pos = pos
        self.color = color
        self.state = "PATROL"  # PATROL, CHASE, FLEE, EATEN
        self.direction = (0, 0)
        self.patrol_path = patrol_path
        self.patrol_index = 0
        self.respawn_timer = 0

    def reset(self):
        self.pos = self.start_pos
        self.state = "PATROL"
        self.direction = (0, 0)
        self.patrol_index = 0
        self.respawn_timer = 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to navigate the maze. Eat all the pellets to win!"
    )

    game_description = (
        "A retro arcade game. Navigate a maze, eat pellets, and avoid the ghosts. Grab power pellets to turn the tables and hunt the hunters!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 31, 19
        self.CELL_SIZE = 20
        self.SCREEN_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.SCREEN_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.MAX_STEPS = 1000
        self.POWERUP_DURATION = 150
        self.GHOST_RESPAWN_TIME = 200
        self.CHASE_RADIUS = 6

        # Colors
        self.COLOR_BG = (15, 15, 40)
        self.COLOR_WALL = (100, 100, 255)
        self.COLOR_PELLET = (255, 255, 0)
        self.COLOR_POWER_PELLET = (255, 255, 255)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_GHOST_FLEE = (50, 50, 255)
        self.COLOR_GHOST_EYES = (255, 255, 255)
        self.GHOST_COLORS = {
            "blinky": (255, 0, 0),
            "pinky": (255, 184, 255),
            "inky": (0, 255, 255),
            "clyde": (255, 184, 82),
        }

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.game_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.clock = pygame.time.Clock()

        self._action_to_direction = {
            0: (0, 0),  # none
            1: (0, -1), # up
            2: (0, 1),  # down
            3: (-1, 0), # left
            4: (1, 0),  # right
        }

        # Initialize state variables
        self.maze = None
        self.path_cells = None
        self.player_pos = None
        self.player_dir = (1, 0)
        self.pellets = None
        self.power_pellets = None
        self.total_pellets = 0
        self.ghosts = None
        self.powerup_timer = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.np_random = None # Will be initialized in reset
        
        # self.validate_implementation()

    def _generate_maze(self):
        w, h = self.GRID_WIDTH, self.GRID_HEIGHT
        maze = np.ones((h, w), dtype=np.uint8)
        
        stack = []
        start_x = self.np_random.integers(0, w // 2) * 2 + 1
        start_y = self.np_random.integers(0, h // 2) * 2 + 1
        
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < w - 1 and 0 < ny < h - 1 and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]
                maze[ny, nx] = 0
                maze[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

        return maze

    def _is_valid_pos(self, x, y):
        return 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT and self.maze[y, x] == 0

    def _find_next_step(self, start_pos, end_pos):
        if not self._is_valid_pos(*start_pos) or not self._is_valid_pos(*end_pos):
            return start_pos
            
        q = deque([(start_pos, [])])
        visited = {start_pos}

        while q:
            current_pos, path = q.popleft()
            if current_pos == end_pos:
                return path[0] if path else current_pos

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if next_pos not in visited and self._is_valid_pos(*next_pos):
                    visited.add(next_pos)
                    new_path = path + [next_pos]
                    q.append((next_pos, new_path))
        return start_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.maze = self._generate_maze()
        self.path_cells = set(zip(*np.where(self.maze == 0)))

        # Place player
        player_start_index = self.np_random.integers(len(self.path_cells))
        self.player_pos = tuple(list(self.path_cells)[player_start_index])
        self.player_dir = (1, 0)

        # Place pellets
        self.pellets = self.path_cells.copy()
        if self.player_pos in self.pellets:
            self.pellets.remove(self.player_pos)

        # Place power pellets
        self.power_pellets = set()
        num_power_pellets = 4
        if len(self.pellets) > num_power_pellets:
            power_pellet_indices = self.np_random.choice(len(self.pellets), size=num_power_pellets, replace=False)
            power_pellet_locs = [list(self.pellets)[i] for i in power_pellet_indices]
            for loc in power_pellet_locs:
                self.power_pellets.add(tuple(loc))
            self.pellets -= self.power_pellets
        self.total_pellets = len(self.pellets)

        # Place ghosts
        corners = [(1, 1), (self.GRID_WIDTH - 2, 1), (1, self.GRID_HEIGHT - 2), (self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2)]
        ghost_names = list(self.GHOST_COLORS.keys())
        self.ghosts = []
        for i, name in enumerate(ghost_names):
            start_pos = corners[i]
            if not self._is_valid_pos(*start_pos):
                for dx, dy in [(0,0), (1,0), (0,1), (-1,0), (0,-1)]:
                    if self._is_valid_pos(start_pos[0]+dx, start_pos[1]+dy):
                        start_pos = (start_pos[0]+dx, start_pos[1]+dy)
                        break
            self.ghosts.append(Ghost(start_pos, self.GHOST_COLORS[name], [start_pos]))
            if start_pos in self.pellets: self.pellets.remove(start_pos)
            if start_pos in self.power_pellets: self.power_pellets.remove(start_pos)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.powerup_timer = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.01

        # --- Update Player ---
        direction = self._action_to_direction[movement]
        if direction != (0, 0):
            next_pos = (self.player_pos[0] + direction[0], self.player_pos[1] + direction[1])
            if self._is_valid_pos(*next_pos):
                self.player_pos = next_pos
                self.player_dir = direction
                # Sfx: pacman_move

        # --- Handle Consumables ---
        if self.player_pos in self.pellets:
            self.pellets.remove(self.player_pos)
            self.score += 1
            reward += 1
            # Sfx: pellet_eat
        elif self.player_pos in self.power_pellets:
            self.power_pellets.remove(self.player_pos)
            self.score += 10
            reward += 10
            self.powerup_timer = self.POWERUP_DURATION
            for ghost in self.ghosts:
                if ghost.state != "EATEN":
                    ghost.state = "FLEE"
            # Sfx: power_pellet_eat

        # --- Update Timers ---
        if self.powerup_timer > 0:
            self.powerup_timer -= 1
            if self.powerup_timer == 0:
                for ghost in self.ghosts:
                    if ghost.state == "FLEE":
                        ghost.state = "PATROL"

        # --- Update Ghosts ---
        for ghost in self.ghosts:
            if ghost.state == "EATEN":
                ghost.respawn_timer -= 1
                if ghost.respawn_timer <= 0:
                    ghost.pos = ghost.start_pos
                    ghost.state = "PATROL"
                continue

            dist_to_player = math.hypot(ghost.pos[0] - self.player_pos[0], ghost.pos[1] - self.player_pos[1])
            if self.powerup_timer > 0:
                ghost.state = "FLEE"
            elif dist_to_player <= self.CHASE_RADIUS:
                ghost.state = "CHASE"
            else:
                ghost.state = "PATROL"

            target = None
            if ghost.state == "CHASE":
                target = self.player_pos
            elif ghost.state == "FLEE":
                corners = [(1, 1), (self.GRID_WIDTH - 2, 1), (1, self.GRID_HEIGHT - 2), (self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2)]
                farthest_corner = max(corners, key=lambda c: math.hypot(c[0] - self.player_pos[0], c[1] - self.player_pos[1]))
                target = farthest_corner
            elif ghost.state == "PATROL":
                valid_moves = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    next_pos = (ghost.pos[0] + dx, ghost.pos[1] + dy)
                    if self._is_valid_pos(*next_pos) and (dx, dy) != (-ghost.direction[0], -ghost.direction[1]):
                        valid_moves.append(next_pos)
                if not valid_moves: # If trapped, allow turning back
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        if self._is_valid_pos(ghost.pos[0] + dx, ghost.pos[1] + dy): valid_moves.append((ghost.pos[0] + dx, ghost.pos[1] + dy))
                if valid_moves:
                    target = valid_moves[self.np_random.integers(len(valid_moves))]

            if target:
                next_pos = self._find_next_step(ghost.pos, target)
                ghost.direction = (next_pos[0] - ghost.pos[0], next_pos[1] - ghost.pos[1])
                ghost.pos = next_pos
        
        # --- Handle Collisions ---
        terminated = False
        for ghost in self.ghosts:
            if ghost.pos == self.player_pos and ghost.state != "EATEN":
                if ghost.state == "FLEE":
                    self.score += 50
                    reward += 50
                    ghost.state = "EATEN"
                    ghost.respawn_timer = self.GHOST_RESPAWN_TIME
                    # Sfx: ghost_eat
                else:
                    self.game_over = True
                    terminated = True
                    reward = -50
                    # Sfx: death
                    break
        
        # --- Check for Win/Termination ---
        if not self.pellets and not self.power_pellets:
            self.game_over = True
            terminated = True
            reward += 100
            # Sfx: win

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _get_observation(self):
        self.game_surface.fill(self.COLOR_BG)
        self._render_game()
        
        x_offset = (640 - self.SCREEN_WIDTH) // 2
        y_offset = (400 - self.SCREEN_HEIGHT) // 2
        self.screen.fill(self.COLOR_BG)
        self.screen.blit(self.game_surface, (x_offset, y_offset))
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cs = self.CELL_SIZE
        
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.maze[y, x] == 1:
                    pygame.draw.rect(self.game_surface, self.COLOR_WALL, (x * cs, y * cs, cs, cs), border_radius=2)
        
        for x, y in self.pellets:
            pygame.gfxdraw.filled_circle(self.game_surface, x * cs + cs // 2, y * cs + cs // 2, cs // 6, self.COLOR_PELLET)
            
        flash_size = abs(math.sin(self.steps * 0.2)) * 3
        for x, y in self.power_pellets:
            pygame.gfxdraw.filled_circle(self.game_surface, x * cs + cs // 2, y * cs + cs // 2, int(cs / 3 + flash_size), self.COLOR_POWER_PELLET)

        for ghost in self.ghosts:
            if ghost.state == "EATEN":
                continue

            color = self.COLOR_GHOST_FLEE if ghost.state == "FLEE" else ghost.color
            if self.powerup_timer > 0 and self.powerup_timer < 40 and self.powerup_timer % 10 > 5:
                color = self.COLOR_POWER_PELLET

            gx, gy = ghost.pos
            g_rect = pygame.Rect(gx * cs, gy * cs, cs, cs)
            pygame.draw.rect(self.game_surface, color, g_rect, border_top_left_radius=5, border_top_right_radius=5)
            
            eye_off_x = ghost.direction[0] * cs // 5
            eye_off_y = ghost.direction[1] * cs // 5
            eye_size = cs // 6
            
            pupil_x_offset = eye_off_x * 1.5 if eye_off_x != 0 else ghost.direction[0] * eye_size / 2
            pupil_y_offset = eye_off_y * 1.5 if eye_off_y != 0 else ghost.direction[1] * eye_size / 2

            for i in [-1, 1]:
                eye_center_x = g_rect.centerx + i * cs//4 + eye_off_x
                eye_center_y = g_rect.centery - cs//4 + eye_off_y
                pupil_center_x = eye_center_x + pupil_x_offset
                pupil_center_y = eye_center_y + pupil_y_offset
                pygame.draw.circle(self.game_surface, self.COLOR_GHOST_EYES, (eye_center_x, eye_center_y), eye_size)
                pygame.draw.circle(self.game_surface, (0,0,0), (pupil_center_x, pupil_center_y), eye_size//2)

        px, py = self.player_pos
        center_x, center_y = int(px * cs + cs / 2), int(py * cs + cs / 2)
        radius = int(cs / 2 * 0.8)
        
        pygame.gfxdraw.aacircle(self.game_surface, center_x, center_y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.game_surface, center_x, center_y, radius, self.COLOR_PLAYER)
        
        mouth_angle = abs(math.sin(self.steps * 0.4)) * 40
        if mouth_angle > 5:
            dir_angle_rad = math.atan2(self.player_dir[1], self.player_dir[0])
            p1 = (center_x, center_y)
            p2 = (center_x + (radius+1) * math.cos(dir_angle_rad - math.radians(mouth_angle)), 
                  center_y + (radius+1) * math.sin(dir_angle_rad - math.radians(mouth_angle)))
            p3 = (center_x + (radius+1) * math.cos(dir_angle_rad + math.radians(mouth_angle)), 
                  center_y + (radius+1) * math.sin(dir_angle_rad + math.radians(mouth_angle)))
            pygame.gfxdraw.filled_polygon(self.game_surface, [p1, p2, p3], self.COLOR_BG)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_POWER_PELLET)
        self.screen.blit(score_text, (20, 10))

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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Pac-Man")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action[0] = 0

    while running:
        movement = 0 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    continue

        if movement == 0:
            # Since it's not auto-advancing, we need to step with no-op to see animations
            # or just not step if no key is pressed. Let's not step.
            keys = pygame.key.get_pressed()
            if not (keys[pygame.K_UP] or keys[pygame.K_DOWN] or keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]):
                # To keep animations running, we can step with no-op
                # obs, reward, terminated, truncated, info = env.step([0,0,0])
                # However, for turn-based, it's better to wait for input.
                # Let's compromise and only redraw
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                clock.tick(30)
                continue
        
        action = [movement, 0, 0] 

        obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            
        clock.tick(10)

    env.close()