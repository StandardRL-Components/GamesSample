
# Generated: 2025-08-28T05:52:31.171685
# Source Brief: brief_05720.md
# Brief Index: 5720

        
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
    user_guide = (
        "Controls: Use arrow keys to draw a path. Press Space to run the robot. "
        "Press Shift to clear the path."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide the robot to the green target by drawing a path. "
        "You have a limited time and path length to succeed."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.GRID_SIZE

        # Game constants
        self.MAX_TIME = 15  # seconds
        self.FPS = 30
        self.MAX_PATH_LENGTH = 20
        self.MAX_EPISODE_STEPS = 1000
        self.ROBOT_SPEED = 4 # pixels per frame

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_WALL = (30, 40, 50)
        self.COLOR_PATH = (80, 90, 110)
        self.COLOR_ROBOT = (50, 150, 255)
        self.COLOR_TARGET = (50, 255, 150)
        self.COLOR_DRAWN_PATH = (200, 200, 220)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_UI_BG = (40, 50, 70, 180)

        # Fonts
        try:
            self.font_large = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 24)
        except pygame.error:
            self.font_large = pygame.font.SysFont("sans", 36)
            self.font_small = pygame.font.SysFont("sans", 24)

        # Initialize state variables
        self.maze = None
        self.robot_start_pos = None
        self.robot_pos = None
        self.robot_exec_pos = None
        self.target_pos = None
        self.path = None
        self.path_length = None
        self.path_exec_index = None
        self.game_phase = None
        self.time_remaining = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME * self.FPS
        self.game_phase = "DRAWING"
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        
        self._generate_maze()
        
        open_cells = [
            (x, y) for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH) if self.maze[y][x] == 0
        ]
        
        # Ensure start and target are sufficiently far apart
        while True:
            start_idx, target_idx = self.np_random.choice(len(open_cells), 2, replace=False)
            self.robot_start_pos = open_cells[start_idx]
            self.target_pos = open_cells[target_idx]
            dist = abs(self.robot_start_pos[0] - self.target_pos[0]) + abs(self.robot_start_pos[1] - self.target_pos[1])
            if dist > (self.GRID_WIDTH + self.GRID_HEIGHT) / 4:
                break
        
        self.robot_pos = self.robot_start_pos
        self.path = [self.robot_pos]
        self.path_length = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01  # Small penalty for time passing
        self.steps += 1
        self.time_remaining = max(0, self.time_remaining - 1)
        
        self._update_particles()

        if self.game_phase == "DRAWING":
            space_press = space_held and not self.last_space_held
            shift_press = shift_held and not self.last_shift_held

            if shift_press:
                # sfx: path_clear
                self.path = [self.robot_pos]
                self.path_length = 0
            
            if space_press and len(self.path) > 1:
                # sfx: robot_start
                self.game_phase = "EXECUTION"
                self.robot_exec_pos = [p * self.GRID_SIZE + self.GRID_SIZE / 2 for p in self.path[0]]
                self.path_exec_index = 1
            
            if movement > 0:
                self._extend_path(movement)

        elif self.game_phase == "EXECUTION":
            if self.path_exec_index < len(self.path):
                target_node_pos = self.path[self.path_exec_index]
                target_pixel_pos = [p * self.GRID_SIZE + self.GRID_SIZE / 2 for p in target_node_pos]
                
                dx = target_pixel_pos[0] - self.robot_exec_pos[0]
                dy = target_pixel_pos[1] - self.robot_exec_pos[1]
                dist = math.hypot(dx, dy)
                
                if dist < self.ROBOT_SPEED:
                    self.robot_exec_pos = target_pixel_pos
                    self.path_exec_index += 1
                    # sfx: robot_step
                else:
                    self.robot_exec_pos[0] += (dx / dist) * self.ROBOT_SPEED
                    self.robot_exec_pos[1] += (dy / dist) * self.ROBOT_SPEED
            else: # Path execution finished
                final_pos = self.path[-1]
                if final_pos == self.target_pos:
                    if self.path_length <= self.MAX_PATH_LENGTH:
                        # sfx: win_perfect
                        reward += 50
                        self._create_particles(self.target_pos, 50, self.COLOR_TARGET)
                    else:
                        # sfx: win_simple
                        reward += 5
                        self._create_particles(self.target_pos, 20, (255, 255, 0))
                else:
                    # sfx: fail_miss
                    reward -= 10
                    self._create_particles(final_pos, 30, (255, 100, 50))
                self.game_over = True

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        if self.time_remaining <= 0 and not self.game_over:
            # sfx: fail_timeout
            reward -= 10
            self.game_over = True
        
        terminated = self.game_over or self.steps >= self.MAX_EPISODE_STEPS
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _extend_path(self, movement):
        if self.path_length >= self.MAX_PATH_LENGTH:
            return
        
        last_pos = self.path[-1]
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        new_pos = (last_pos[0] + dx, last_pos[1] + dy)

        # Boundary check
        if not (0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT):
            return
        
        # Wall collision check
        if self.maze[new_pos[1]][new_pos[0]] == 1:
            return
            
        # Prevent backtracking (e.g., L then R)
        if len(self.path) > 1 and new_pos == self.path[-2]:
            # sfx: path_retract
            self.path.pop()
            self.path_length -= 1
            return

        # Prevent adding the same point twice
        if new_pos == last_pos:
            return

        # sfx: path_draw
        self.path.append(new_pos)
        self.path_length += 1

    def _generate_maze(self):
        self.maze = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.uint8)
        start_x, start_y = (
            self.np_random.integers(0, self.GRID_WIDTH // 2) * 2,
            self.np_random.integers(0, self.GRID_HEIGHT // 2) * 2,
        )
        
        stack = [(start_x, start_y)]
        self.maze[start_y, start_x] = 0
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = neighbors[self.np_random.integers(0, len(neighbors))]
                self.maze[ny, nx] = 0
                self.maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw maze paths and walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                color = self.COLOR_PATH if self.maze[y][x] == 0 else self.COLOR_WALL
                pygame.draw.rect(self.screen, color, rect)

        # Draw drawn path
        if len(self.path) > 1:
            pixel_path = [[p[0] * self.GRID_SIZE + self.GRID_SIZE / 2, p[1] * self.GRID_SIZE + self.GRID_SIZE / 2] for p in self.path]
            pygame.draw.lines(self.screen, self.COLOR_DRAWN_PATH, False, pixel_path, 3)

        # Draw target with glow
        target_px, target_py = (
            int(self.target_pos[0] * self.GRID_SIZE + self.GRID_SIZE / 2),
            int(self.target_pos[1] * self.GRID_SIZE + self.GRID_SIZE / 2)
        )
        for i in range(5):
            alpha = 150 - i * 30
            radius = self.GRID_SIZE // 2 + i * 2
            pygame.gfxdraw.filled_circle(self.screen, target_px, target_py, radius, (*self.COLOR_TARGET, alpha))
        pygame.gfxdraw.filled_circle(self.screen, target_px, target_py, self.GRID_SIZE // 2 - 2, self.COLOR_TARGET)
        
        # Draw robot
        if self.game_phase == "DRAWING":
            robot_px, robot_py = (
                int(self.robot_pos[0] * self.GRID_SIZE),
                int(self.robot_pos[1] * self.GRID_SIZE)
            )
            robot_rect = pygame.Rect(robot_px + 2, robot_py + 2, self.GRID_SIZE - 4, self.GRID_SIZE - 4)
            pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect, border_radius=3)
        else: # EXECUTING
            robot_px, robot_py = int(self.robot_exec_pos[0]), int(self.robot_exec_pos[1])
            size = self.GRID_SIZE - 4
            robot_rect = pygame.Rect(robot_px - size // 2, robot_py - size // 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect, border_radius=3)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], p["pos"], int(p["radius"]))

    def _render_ui(self):
        ui_surface = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surface, (0, 0))

        # Time
        time_text = f"Time: {self.time_remaining / self.FPS:.1f}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Path Length
        path_color = self.COLOR_TEXT if self.path_length <= self.MAX_PATH_LENGTH else (255, 100, 100)
        path_text = f"Path: {self.path_length}/{self.MAX_PATH_LENGTH}"
        path_surf = self.font_large.render(path_text, True, path_color)
        self.screen.blit(path_surf, (self.SCREEN_WIDTH - path_surf.get_width() - 10, 10))

        # Game Phase
        phase_text = self.game_phase
        phase_surf = self.font_small.render(phase_text, True, self.COLOR_TEXT)
        self.screen.blit(phase_surf, (self.SCREEN_WIDTH / 2 - phase_surf.get_width() / 2, 15))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining / self.FPS,
            "path_length": self.path_length,
            "game_phase": self.game_phase,
        }
        
    def _create_particles(self, pos_grid, count, color):
        pos_pixel = [p * self.GRID_SIZE + self.GRID_SIZE / 2 for p in pos_grid]
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos_pixel),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "radius": self.np_random.uniform(2, 5),
                "life": self.np_random.integers(20, 40),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["radius"] -= 0.1
        self.particles = [p for p in self.particles if p["life"] > 0 and p["radius"] > 0]
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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