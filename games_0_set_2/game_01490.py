
# Generated: 2025-08-27T17:19:10.297503
# Source Brief: brief_01490.md
# Brief Index: 1490

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import collections
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move the robot. Reach the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a robot through a procedurally generated maze to the exit before running out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # Static class attributes for difficulty progression
    _successful_episodes = 0
    _initial_obstacles = 10
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Game Constants
        self.GRID_WIDTH = 32
        self.GRID_HEIGHT = 20
        self.CELL_SIZE = 20
        self.MAX_STEPS = 50

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_WALL = (80, 80, 90)
        self.COLOR_ROBOT = (50, 150, 255)
        self.COLOR_ROBOT_GLOW = (100, 200, 255)
        self.COLOR_EXIT = (50, 255, 150)
        self.COLOR_PATH = (30, 35, 45)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_WIN = (100, 255, 150)
        self.COLOR_LOSE = (255, 100, 100)
        
        # Initialize state variables
        self.robot_pos = None
        self.exit_pos = None
        self.walls = None
        self.path_trace = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = ""
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = ""
        
        num_obstacles = self._initial_obstacles + (GameEnv._successful_episodes // 50)
        self._generate_maze(num_obstacles)
        
        self.path_trace = [self.robot_pos]
        
        return self._get_observation(), self._get_info()

    def _generate_maze(self, num_obstacles):
        while True:
            self.walls = set()
            all_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
            
            start_idx, exit_idx = self.np_random.choice(len(all_cells), 2, replace=False)
            self.robot_pos = tuple(all_cells[start_idx])
            self.exit_pos = tuple(all_cells[exit_idx])

            possible_wall_pos = [p for p in all_cells if p not in {self.robot_pos, self.exit_pos}]
            self.np_random.shuffle(possible_wall_pos)
            
            temp_walls = set()
            placed_count = 0
            for pos in possible_wall_pos:
                if placed_count >= num_obstacles:
                    break
                
                temp_walls.add(pos)
                if self._bfs_path_exists(self.robot_pos, self.exit_pos, temp_walls):
                    placed_count += 1
                else:
                    temp_walls.remove(pos)
            
            self.walls = temp_walls
            
            if self._bfs_path_exists(self.robot_pos, self.exit_pos, self.walls):
                break

    def _bfs_path_exists(self, start, end, walls):
        q = collections.deque([start])
        visited = {start}
        while q:
            node = q.popleft()
            if node == end:
                return True
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                neighbor = (node[0] + dx, node[1] + dy)
                if (0 <= neighbor[0] < self.GRID_WIDTH and
                    0 <= neighbor[1] < self.GRID_HEIGHT and
                    neighbor not in walls and
                    neighbor not in visited):
                    visited.add(neighbor)
                    q.append(neighbor)
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = -0.1

        if movement > 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            new_pos = (self.robot_pos[0] + dx, self.robot_pos[1] + dy)

            if (0 <= new_pos[0] < self.GRID_WIDTH and
                0 <= new_pos[1] < self.GRID_HEIGHT and
                new_pos not in self.walls):
                self.robot_pos = new_pos
                if new_pos not in self.path_trace:
                    self.path_trace.append(new_pos)
                # Sound: Footstep sfx

        self.steps += 1
        
        terminated = False
        if self.robot_pos == self.exit_pos:
            reward += 10.0
            terminated = True
            self.game_over = True
            self.win_status = "SUCCESS"
            GameEnv._successful_episodes += 1
            # Sound: Win jingle
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_status = "OUT OF MOVES"
            # Sound: Fail buzzer
        elif self._is_trapped():
            reward -= 10.0
            terminated = True
            self.game_over = True
            self.win_status = "TRAPPED"
            # Sound: Fail buzzer

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _is_trapped(self):
        if self.robot_pos == self.exit_pos:
            return False
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor = (self.robot_pos[0] + dx, self.robot_pos[1] + dy)
            if (0 <= neighbor[0] < self.GRID_WIDTH and
                0 <= neighbor[1] < self.GRID_HEIGHT and
                neighbor not in self.walls):
                return False
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_pixel_width = self.GRID_WIDTH * self.CELL_SIZE
        grid_pixel_height = self.GRID_HEIGHT * self.CELL_SIZE
        offset_x = (640 - grid_pixel_width) // 2
        offset_y = (400 - grid_pixel_height) // 2

        for pos in self.path_trace:
            rect = pygame.Rect(offset_x + pos[0] * self.CELL_SIZE,
                               offset_y + pos[1] * self.CELL_SIZE,
                               self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PATH, rect)

        for wall_pos in self.walls:
            rect = pygame.Rect(offset_x + wall_pos[0] * self.CELL_SIZE,
                               offset_y + wall_pos[1] * self.CELL_SIZE,
                               self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(offset_x + x * self.CELL_SIZE,
                                   offset_y + y * self.CELL_SIZE,
                                   self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        exit_rect = pygame.Rect(offset_x + self.exit_pos[0] * self.CELL_SIZE,
                                offset_y + self.exit_pos[1] * self.CELL_SIZE,
                                self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        robot_center_x = int(offset_x + self.robot_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        robot_center_y = int(offset_y + self.robot_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        
        glow_radius = int(self.CELL_SIZE * 0.7)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_ROBOT_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (robot_center_x - glow_radius, robot_center_y - glow_radius))

        robot_rect = pygame.Rect(0, 0, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
        robot_rect.center = (robot_center_x, robot_center_y)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect, border_radius=3)

    def _render_ui(self):
        moves_left = max(0, self.MAX_STEPS - self.steps)
        text_surface = self.font_ui.render(f"Moves Left: {moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        score_text = f"Score: {self.score:.1f}"
        text_surface = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(topright=(630, 10))
        self.screen.blit(text_surface, text_rect)

        if self.game_over:
            color = self.COLOR_WIN if self.win_status == "SUCCESS" else self.COLOR_LOSE
            game_over_surface = self.font_game_over.render(self.win_status, True, color)
            game_over_rect = game_over_surface.get_rect(center=(640 // 2, 400 // 2))
            
            bg_rect = game_over_rect.inflate(40, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((20, 20, 30, 200))
            self.screen.blit(bg_surf, bg_rect)
            
            self.screen.blit(game_over_surface, game_over_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")