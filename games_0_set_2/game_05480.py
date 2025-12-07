
# Generated: 2025-08-28T05:08:16.759631
# Source Brief: brief_05480.md
# Brief Index: 5480

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use ↑↓←→ to move the robot. Reach the green exit tile."
    )

    # Short, user-facing description of the game
    game_description = (
        "Guide a robot through a procedurally generated obstacle course to the exit within a limited number of steps."
    )

    # Frames only advance when an action is received
    auto_advance = False

    # Static class variables for tracking difficulty across episodes
    _successful_episodes = 0
    _initial_obstacles = 10
    _obstacle_increment_rate = 5 # Add 1 obstacle every 5 successful episodes
    _max_obstacles = 50 # Cap at 50% of grid cells

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_STEPS_TOTAL = 50
        self.MAX_EPISODE_STEPS = 500 # Gymnasium standard

        # Define colors for visual clarity
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (50, 50, 60)
        self.COLOR_ROBOT = (255, 60, 60)
        self.COLOR_OBSTACLE = (60, 120, 220)
        self.COLOR_EXIT = (60, 255, 60)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_BORDER = (10, 10, 15)

        # EXACT spaces as per specification
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Calculate rendering geometry
        self.cell_size = min(self.WIDTH // (self.GRID_SIZE + 2), self.HEIGHT // (self.GRID_SIZE + 2))
        self.grid_width = self.cell_size * self.GRID_SIZE
        self.grid_height = self.cell_size * self.GRID_SIZE
        self.grid_offset_x = (self.WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.HEIGHT - self.grid_height) // 2

        # Initialize state variables (will be properly set in reset)
        self.robot_pos = None
        self.exit_pos = None
        self.obstacles = []
        self.steps_taken = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.num_obstacles = 0
        self.initial_dist = 0

        # Validate implementation against requirements
        # self.validate_implementation() # Commented out for submission, but useful for testing

    def _generate_level(self):
        """Generates a new level, ensuring a solvable path exists."""
        while True:
            self.num_obstacles = min(
                self._max_obstacles,
                self._initial_obstacles + (self._successful_episodes // self._obstacle_increment_rate)
            )

            all_cells = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
            
            # Use self.np_random for reproducibility
            spawn_points = self.np_random.choice(len(all_cells), size=2, replace=False)
            self.robot_pos = tuple(all_cells[spawn_points[0]])
            self.exit_pos = tuple(all_cells[spawn_points[1]])

            obstacle_candidates = [c for c in all_cells if c not in [self.robot_pos, self.exit_pos]]
            obstacle_indices = self.np_random.choice(len(obstacle_candidates), size=self.num_obstacles, replace=False)
            self.obstacles = {tuple(obstacle_candidates[i]) for i in obstacle_indices}

            if self._is_path_valid():
                break # Found a solvable level

    def _is_path_valid(self):
        """Checks for a path from robot to exit using Breadth-First Search."""
        q = collections.deque([self.robot_pos])
        visited = {self.robot_pos}
        
        while q:
            x, y = q.popleft()

            if (x, y) == self.exit_pos:
                return True

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in self.obstacles and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_level()
        
        self.steps_taken = 0
        self.score = 0.0
        self.game_over = False
        self.victory = False
        
        self.initial_dist = self._manhattan_distance(self.robot_pos, self.exit_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1  # Cost of taking a step

        old_dist = self._manhattan_distance(self.robot_pos, self.exit_pos)
        
        # --- Update game logic ---
        if movement > 0: # 0 is no-op
            # sound: robot_move.wav
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = (self.robot_pos[0] + dx, self.robot_pos[1] + dy)

            # Check for valid move (in bounds and not an obstacle)
            if (0 <= new_pos[0] < self.GRID_SIZE and
                0 <= new_pos[1] < self.GRID_SIZE and
                new_pos not in self.obstacles):
                self.robot_pos = new_pos
            else:
                # sound: bump_wall.wav
                pass # Robot hits a wall or obstacle

        self.steps_taken += 1
        
        # --- Calculate reward ---
        new_dist = self._manhattan_distance(self.robot_pos, self.exit_pos)
        if new_dist < old_dist:
            reward += 1.0 # Reward for getting closer
        elif new_dist > old_dist:
            reward -= 1.0 # Penalty for getting further

        # --- Check for termination ---
        terminated = False
        if self.robot_pos == self.exit_pos:
            # sound: level_win.wav
            reward += 100.0
            terminated = True
            self.game_over = True
            self.victory = True
            GameEnv._successful_episodes += 1
        elif self.steps_taken >= self.MAX_STEPS_TOTAL:
            # sound: level_fail.wav
            reward -= 10.0
            terminated = True
            self.game_over = True

        if self.steps_taken >= self.MAX_EPISODE_STEPS:
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

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps_taken,
            "remaining_steps": self.MAX_STEPS_TOTAL - self.steps_taken,
            "successful_episodes": self._successful_episodes,
        }

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _grid_to_pixels(self, grid_x, grid_y):
        px = self.grid_offset_x + grid_x * self.cell_size
        py = self.grid_offset_y + grid_y * self.cell_size
        return px, py

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            start_x = self.grid_offset_x + i * self.cell_size
            start_y = self.grid_offset_y
            end_x = start_x
            end_y = self.grid_offset_y + self.grid_height
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y))

            start_x = self.grid_offset_x
            start_y = self.grid_offset_y + i * self.cell_size
            end_x = self.grid_offset_x + self.grid_width
            end_y = start_y
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y))

        # Draw obstacles
        for ox, oy in self.obstacles:
            px, py = self._grid_to_pixels(ox, oy)
            pygame.draw.rect(self.screen, self.COLOR_BORDER, (px, py, self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (px + 2, py + 2, self.cell_size - 4, self.cell_size - 4))
            
        # Draw exit
        ex, ey = self._grid_to_pixels(*self.exit_pos)
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (ex, ey, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex + 2, ey + 2, self.cell_size - 4, self.cell_size - 4))

        # Draw robot
        rx, ry = self._grid_to_pixels(*self.robot_pos)
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (rx, ry, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, (rx + 2, ry + 2, self.cell_size - 4, self.cell_size - 4))

    def _render_ui(self):
        # Display steps remaining
        steps_left = max(0, self.MAX_STEPS_TOTAL - self.steps_taken)
        steps_text = self.font_ui.render(f"Steps Left: {steps_left}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (20, 20))

        # Display score
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 50))
        
        # Display difficulty
        difficulty_text = self.font_ui.render(f"Obstacles: {self.num_obstacles}", True, self.COLOR_TEXT)
        self.screen.blit(difficulty_text, (self.WIDTH - difficulty_text.get_width() - 20, 20))
        
        # Display game over message
        if self.game_over:
            message = "VICTORY!" if self.victory else "OUT OF STEPS"
            color = self.COLOR_EXIT if self.victory else self.COLOR_ROBOT
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_game_over.render(message, True, color)
            text_rect = game_over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    print(env.user_guide)

    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
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
                    done = False
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
        
        # Only step if an action was taken or if the game is over (to see final state)
        if action[0] != 0 or done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(15) # Limit frame rate for manual play

    env.close()