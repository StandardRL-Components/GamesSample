
# Generated: 2025-08-27T23:25:30.799077
# Source Brief: brief_03462.md
# Brief Index: 3462

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your robot. Reach the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a robot through a procedurally generated maze to reach the exit. "
        "Avoid black pitfalls. Each step costs a little, but moving away from the goal "
        "to find a better path is rewarded. Getting too close to pitfalls is penalized."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # Class-level state that persists across resets
    _successful_episodes = 0
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Visual Design ---
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_GRID = (50, 50, 70)
        self.COLOR_ROBOT = (255, 60, 60)
        self.COLOR_ROBOT_GLOW = (255, 150, 150)
        self.COLOR_EXIT = (60, 255, 60)
        self.COLOR_PITFALL = (0, 0, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game Mechanics ---
        self.grid_cols = 16
        self.grid_rows = 10
        self.cell_size = 40
        self.max_steps = 1000
        self.initial_pitfalls = 10
        self.max_pitfalls = 20
        self.difficulty_step = 5 # episodes per difficulty increase
        
        # --- State Variables (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.robot_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.pitfall_pos = set()
        self.num_pitfalls = 0

        # Initialize state variables
        self.reset()

        # --- CRITICAL: Validate implementation ---
        # self.validate_implementation() # Commented out for submission, but useful for testing
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize RNG
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # Reset game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # Update difficulty based on persistent success count
        self.num_pitfalls = min(
            self.max_pitfalls, 
            self.initial_pitfalls + (self._successful_episodes // self.difficulty_step)
        )
        
        # Generate a new solvable maze
        self._generate_maze()
        
        return self._get_observation(), self._get_info()

    def _generate_maze(self):
        """Generates a new maze with a guaranteed path from start to exit."""
        max_retries = 100
        for _ in range(max_retries):
            # 1. Place robot and exit
            all_cells = [(x, y) for x in range(self.grid_cols) for y in range(self.grid_rows)]
            start_idx, exit_idx = self.np_random.choice(len(all_cells), 2, replace=False)
            self.robot_pos = all_cells[start_idx]
            self.exit_pos = all_cells[exit_idx]

            # 2. Place pitfalls
            possible_pit_cells = [
                cell for cell in all_cells if cell != self.robot_pos and cell != self.exit_pos
            ]
            pit_indices = self.np_random.choice(len(possible_pit_cells), self.num_pitfalls, replace=False)
            self.pitfall_pos = {possible_pit_cells[i] for i in pit_indices}

            # 3. Verify path exists using BFS
            if self._has_path():
                return  # Maze is solvable, we're done

        raise RuntimeError(f"Failed to generate a solvable maze after {max_retries} attempts.")

    def _has_path(self):
        """Checks if a path exists from robot_pos to exit_pos using BFS."""
        q = deque([self.robot_pos])
        visited = {self.robot_pos}
        
        while q:
            x, y = q.popleft()

            if (x, y) == self.exit_pos:
                return True

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)

                if (0 <= nx < self.grid_cols and
                    0 <= ny < self.grid_rows and
                    neighbor not in visited and
                    neighbor not in self.pitfall_pos):
                    
                    visited.add(neighbor)
                    q.append(neighbor)
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self.steps += 1
        
        old_pos = self.robot_pos
        new_pos = self.robot_pos
        
        if movement == 1: # Up
            new_pos = (old_pos[0], old_pos[1] - 1)
        elif movement == 2: # Down
            new_pos = (old_pos[0], old_pos[1] + 1)
        elif movement == 3: # Left
            new_pos = (old_pos[0] - 1, old_pos[1])
        elif movement == 4: # Right
            new_pos = (old_pos[0] + 1, old_pos[1])

        # Check for boundary violations
        if (0 <= new_pos[0] < self.grid_cols and 0 <= new_pos[1] < self.grid_rows):
            self.robot_pos = new_pos
        else: # Hit a wall, no movement
            new_pos = old_pos

        # --- Calculate Reward ---
        reward = self._calculate_reward(old_pos, new_pos)
        self.score += reward

        # --- Check Termination ---
        terminated = False
        if self.robot_pos == self.exit_pos:
            # Win condition
            terminated = True
            self.__class__._successful_episodes += 1
        elif self.robot_pos in self.pitfall_pos:
            # Loss condition
            terminated = True
        elif self.steps >= self.max_steps:
            # Max steps reached
            terminated = True
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_reward(self, old_pos, new_pos):
        # Terminal rewards
        if new_pos == self.exit_pos:
            return 100.0
        if new_pos in self.pitfall_pos:
            return -100.0

        # Continuous/Event-based rewards
        reward = -0.1  # Cost per step

        # Reward for exploring (moving away from exit)
        dist_before = abs(old_pos[0] - self.exit_pos[0]) + abs(old_pos[1] - self.exit_pos[1])
        dist_after = abs(new_pos[0] - self.exit_pos[0]) + abs(new_pos[1] - self.exit_pos[1])
        if dist_after > dist_before:
            reward += 1.0

        # Penalty for being adjacent to a pitfall
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (new_pos[0] + dx, new_pos[1] + dy)
            if neighbor in self.pitfall_pos:
                reward -= 1.0
                break # Only apply penalty once

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.screen_width, self.cell_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, self.cell_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_width, y))

        # Draw pitfalls
        for x, y in self.pitfall_pos:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.COLOR_PITFALL, rect)

        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(ex * self.cell_size, ey * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Draw robot
        rx, ry = self.robot_pos
        robot_rect = pygame.Rect(rx * self.cell_size, ry * self.cell_size, self.cell_size, self.cell_size)
        
        # Add a glow effect for visual emphasis
        glow_rect = robot_rect.inflate(-self.cell_size * 0.4, -self.cell_size * 0.4)
        
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_GLOW, glow_rect)

    def _render_ui(self):
        # Display step count
        step_text = self.font_ui.render(f"Step: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(step_text, (10, 5))
        
        # Display score
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 30))
        
        # Display difficulty
        diff_text = self.font_ui.render(f"Pitfalls: {self.num_pitfalls}", True, self.COLOR_TEXT)
        self.screen.blit(diff_text, (self.screen_width - diff_text.get_width() - 10, 5))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "robot_pos": self.robot_pos,
            "exit_pos": self.exit_pos,
            "pitfalls": self.num_pitfalls,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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
    # This block allows you to play the game manually for testing
    # It will not be executed when the environment is used by Gymnasium runners
    
    # --- Manual Control Key Map ---
    #      ↑
    #  ←   ↓   →
    # Key: (movement_action)
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
        pygame.K_SPACE: 0, # No-op
    }

    env = GameEnv()
    env.reset()
    
    # Override screen for display
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption(env.game_description)
    
    running = True
    game_over_display = False

    while running:
        action = [0, 0, 0] # Default action is no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset on 'r'
                    env.reset()
                    game_over_display = False
                
                if not game_over_display and event.key in key_map:
                    action[0] = key_map[event.key]
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

                    if terminated:
                        game_over_display = True

        # Blit the environment's internal surface to the display screen
        # This is the correct way to render a headless env for human viewing
        frame = np.transpose(env._get_observation(), (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Limit frame rate

    env.close()