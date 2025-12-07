
# Generated: 2025-08-28T02:47:46.505575
# Source Brief: brief_04572.md
# Brief Index: 4572

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the robot. Reach the green exit before you run out of steps."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a robot through a procedurally generated maze to the exit within a limited number of steps."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # Class-level state for difficulty progression
    successful_episodes = 0
    base_num_obstacles = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 20
        self.MAX_STEPS = 1000
        self.INITIAL_MOVES = 500

        # Centered game area
        self.GAME_AREA_SIZE = 400
        self.CELL_SIZE = self.GAME_AREA_SIZE // self.GRID_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GAME_AREA_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GAME_AREA_SIZE) // 2

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_ROBOT = (50, 150, 255)
        self.COLOR_ROBOT_GLOW = (50, 150, 255, 64)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.COLOR_EXIT = (80, 255, 80)
        self.COLOR_TEXT = (240, 240, 240)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Initialize state variables
        self.robot_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.obstacle_positions = []
        self.steps = 0
        self.score = 0
        self.remaining_moves = 0
        self.game_over = False
        self.np_random = None
        
        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.remaining_moves = self.INITIAL_MOVES
        
        self._generate_maze()
        
        return self._get_observation(), self._get_info()

    def _generate_maze(self):
        """Generates a new maze layout, ensuring a path to the exit exists."""
        num_obstacles = self.base_num_obstacles + (self.successful_episodes // 50)
        
        is_valid = False
        while not is_valid:
            all_positions = set((x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE))
            
            # Place robot and exit
            self.robot_pos = tuple(self.np_random.integers(0, self.GRID_SIZE, size=2))
            all_positions.remove(self.robot_pos)
            
            self.exit_pos = tuple(self.np_random.choice(list(all_positions), size=1)[0])
            all_positions.remove(self.exit_pos)
            
            # Place obstacles
            obstacle_indices = self.np_random.choice(len(all_positions), size=min(num_obstacles, len(all_positions)), replace=False)
            self.obstacle_positions = [list(all_positions)[i] for i in obstacle_indices]
            
            # Validate path
            is_valid = self._is_path_valid()

    def _is_path_valid(self):
        """Checks if a path exists from robot to exit using Breadth-First Search."""
        q = deque([self.robot_pos])
        visited = {self.robot_pos}
        obstacles_set = set(map(tuple, self.obstacle_positions))

        while q:
            x, y = q.popleft()

            if (x, y) == self.exit_pos:
                return True

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                    neighbor = (nx, ny)
                    if neighbor not in visited and neighbor not in obstacles_set:
                        visited.add(neighbor)
                        q.append(neighbor)
        return False
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        # space_held and shift_held are ignored as per the brief.
        
        self.steps += 1
        reward = -1 # Cost for taking a step

        # --- Move Robot ---
        px, py = self.robot_pos
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        
        target_pos = (px, py)
        
        # Check boundaries and obstacles
        if (0 <= px < self.GRID_SIZE and 0 <= py < self.GRID_SIZE and
                target_pos not in self.obstacle_positions):
            self.robot_pos = target_pos
            # sfx: robot_move.wav

        self.remaining_moves -= 1
        
        # --- Check for Termination ---
        terminated = False
        if self.robot_pos == self.exit_pos:
            reward = 100
            terminated = True
            self.game_over = True
            type(self).successful_episodes += 1
            # sfx: win_fanfare.wav
        elif self.remaining_moves <= 0:
            terminated = True
            self.game_over = True
            # sfx: lose_buzzer.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GAME_AREA_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GAME_AREA_SIZE, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

        # Draw obstacles
        for x, y in self.obstacle_positions:
            rect = pygame.Rect(
                self.GRID_OFFSET_X + x * self.CELL_SIZE,
                self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)

        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(
            self.GRID_OFFSET_X + ex * self.CELL_SIZE,
            self.GRID_OFFSET_Y + ey * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Draw robot
        rx, ry = self.robot_pos
        robot_center_x = int(self.GRID_OFFSET_X + rx * self.CELL_SIZE + self.CELL_SIZE / 2)
        robot_center_y = int(self.GRID_OFFSET_Y + ry * self.CELL_SIZE + self.CELL_SIZE / 2)
        
        # Glow effect
        glow_radius = int(self.CELL_SIZE * 0.7)
        pygame.gfxdraw.filled_circle(self.screen, robot_center_x, robot_center_y, glow_radius, self.COLOR_ROBOT_GLOW)
        pygame.gfxdraw.aacircle(self.screen, robot_center_x, robot_center_y, glow_radius, self.COLOR_ROBOT_GLOW)
        
        # Main body
        robot_rect = pygame.Rect(
            self.GRID_OFFSET_X + rx * self.CELL_SIZE,
            self.GRID_OFFSET_Y + ry * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect)

    def _render_ui(self):
        # Display remaining moves
        moves_text = self.font_main.render(f"MOVES: {self.remaining_moves}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))
        
        # Display score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Display difficulty
        num_obstacles = self.base_num_obstacles + (self.successful_episodes // 50)
        difficulty_text = self.font_small.render(f"OBSTACLES: {num_obstacles}", True, self.COLOR_TEXT)
        self.screen.blit(difficulty_text, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_moves": self.remaining_moves,
            "robot_pos": self.robot_pos,
            "exit_pos": self.exit_pos,
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Robot Maze")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    
    action = np.array([0, 0, 0]) # No-op
    
    while not done:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                    action = np.array([0, 0, 0])
                    continue
        
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print("--- Episode Finished --- (Press 'r' to reset)")
            action[0] = 0 # Reset action to no-op after one step
            
        # --- Drawing ---
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(30) # Limit to 30 FPS
        
    env.close()