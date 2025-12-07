
# Generated: 2025-08-27T23:40:42.590232
# Source Brief: brief_03542.md
# Brief Index: 3542

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Adapted user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the robot. "
        "Reach the green exit while avoiding the red pits."
    )

    # Adapted user-facing description of the game
    game_description = (
        "Navigate a robot through obstacle-filled grids to reach the exit in the shortest time possible."
    )

    # Game state is static until an action is received
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 16
    GRID_ROWS = 10
    CELL_SIZE = 40
    MAX_STEPS = 1000

    # Cell types
    CELL_EMPTY = 0
    CELL_PIT = 1
    CELL_EXIT = 2

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_TEXT = (240, 240, 240)
    COLOR_ROBOT = (50, 150, 255)
    COLOR_EXIT = (50, 255, 150)
    COLOR_PIT = (255, 80, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables are initialized in reset()
        self.grid = None
        self.robot_pos = None
        self.start_pos = None
        self.exit_pos = None
        self.steps = 0
        self.score = 0
        self.level = 1
        self.level_cleared = False
        self.game_over = False

        self.reset()
        
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.level_cleared:
            self.level += 1
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level_cleared = False

        self._generate_level()
        self.robot_pos = list(self.start_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held are ignored as per the brief

        self.steps += 1
        reward = -0.1  # Per-step penalty

        # --- Update game logic ---
        prev_pos = list(self.robot_pos)
        
        if movement == 1:  # Up
            self.robot_pos[1] -= 1
        elif movement == 2:  # Down
            self.robot_pos[1] += 1
        elif movement == 3:  # Left
            self.robot_pos[0] -= 1
        elif movement == 4:  # Right
            self.robot_pos[0] += 1
        # movement == 0 is no-op

        # Boundary checks
        if not (0 <= self.robot_pos[0] < self.GRID_COLS and 0 <= self.robot_pos[1] < self.GRID_ROWS):
            self.robot_pos = prev_pos # Revert move if out of bounds
        
        # --- Check for terminal conditions ---
        terminated = False
        cell_type = self.grid[self.robot_pos[1], self.robot_pos[0]]

        if cell_type == self.CELL_EXIT:
            reward += 100
            terminated = True
            self.game_over = True
            self.level_cleared = True
            # sfx: win_sound
        elif cell_type == self.CELL_PIT:
            reward += -100
            terminated = True
            self.game_over = True
            # sfx: lose_sound
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        num_pits = self.level + 1
        
        while True:
            grid = np.full((self.GRID_ROWS, self.GRID_COLS), self.CELL_EMPTY, dtype=np.int8)
            
            # Place start and exit
            self.start_pos = (0, 0)
            self.exit_pos = (self.GRID_COLS - 1, self.GRID_ROWS - 1)
            grid[self.exit_pos[1], self.exit_pos[0]] = self.CELL_EXIT

            # Place pits
            possible_pit_coords = [
                (c, r) for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS)
                if (c, r) != self.start_pos and (c, r) != self.exit_pos
            ]
            
            # Ensure we don't try to sample more pits than available spaces
            num_pits_to_place = min(num_pits, len(possible_pit_coords))
            pit_indices = self.np_random.choice(len(possible_pit_coords), num_pits_to_place, replace=False)
            
            for index in pit_indices:
                px, py = possible_pit_coords[index]
                grid[py, px] = self.CELL_PIT
            
            # Check for a valid path
            if self._is_path_valid(grid, self.start_pos, self.exit_pos):
                self.grid = grid
                break

    def _is_path_valid(self, grid, start, end):
        q = deque([start])
        visited = {start}
        
        while q:
            x, y = q.popleft()
            
            if (x, y) == end:
                return True
                
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS and
                        grid[ny, nx] != self.CELL_PIT and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    q.append((nx, ny))
                    
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
            
        # Draw grid elements
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = pygame.Rect(c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                cell_type = self.grid[r, c]
                
                if cell_type == self.CELL_PIT:
                    pygame.draw.rect(self.screen, self.COLOR_PIT, rect.inflate(-4, -4))
                elif cell_type == self.CELL_EXIT:
                    pygame.draw.rect(self.screen, self.COLOR_EXIT, rect.inflate(-4, -4))

        # Draw robot
        robot_center_x = int(self.robot_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        robot_center_y = int(self.robot_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        
        # Glow effect
        glow_radius = int(self.CELL_SIZE * 0.6)
        # We need a surface with per-pixel alpha to draw the glow
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_ROBOT + (60,), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (robot_center_x - glow_radius, robot_center_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Main body
        robot_size = int(self.CELL_SIZE * 0.7)
        robot_rect = pygame.Rect(
            robot_center_x - robot_size // 2,
            robot_center_y - robot_size // 2,
            robot_size,
            robot_size
        )
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect)
        # Add a small highlight
        highlight_rect = robot_rect.copy()
        highlight_rect.width = highlight_rect.width // 4
        highlight_rect.height = highlight_rect.height // 4
        highlight_rect.topleft = robot_rect.topleft
        pygame.draw.rect(self.screen, (200, 220, 255), highlight_rect)

    def _render_ui(self):
        level_text = self.font_large.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        steps_text = self.font_large.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        steps_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(steps_text, steps_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "LEVEL CLEARED!" if self.level_cleared else "GAME OVER"
            message_text = self.font_large.render(message, True, self.COLOR_EXIT if self.level_cleared else self.COLOR_PIT)
            message_rect = message_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(message_text, message_rect)

            score_text = self.font_small.render(f"Final Score: {self.score:.1f}", True, self.COLOR_TEXT)
            score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level
        }

    def close(self):
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

# Example of how to run the environment
if __name__ == "__main__":
    import os
    # Set the SDL video driver to "dummy" to run headless
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    env.validate_implementation()
    
    # Test a few random steps
    obs, info = env.reset()
    print(f"Initial Info: {info}")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.1f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode finished. Resetting.")
            obs, info = env.reset()
    
    # Example of how a human might play (requires a display)
    try:
        del os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        pass # It was not set, which is fine
        
    print("\n--- Interactive Mode ---")
    print(GameEnv.user_guide)
    
    try:
        env_interactive = GameEnv(render_mode="rgb_array")
        obs, info = env_interactive.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Grid Robot")
        
        terminated = False
        running = True
        while running:
            # Convert observation back to a surface for display
            display_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(display_surf, (0, 0))
            pygame.display.flip()
            
            if terminated:
                print("Episode finished. Press R to reset.")
                while True:
                    wait_event = pygame.event.wait()
                    if wait_event.type == pygame.QUIT:
                        running = False
                        break
                    if wait_event.type == pygame.KEYDOWN and wait_event.key == pygame.K_r:
                        obs, info = env_interactive.reset()
                        terminated = False
                        break
                if not running:
                    break

            # Handle player input
            action = [0, 0, 0] # Default to no-op
            
            # This loop ensures we process all events in the queue
            # but only take the last key press as the action for this frame.
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
            
            # Since this is not auto-advancing, we only step if an action was taken
            # or if it's the very first frame after a reset.
            # A human player will always press a key to move.
            if action[0] != 0:
                obs, reward, terminated, truncated, info = env_interactive.step(action)
                print(f"Action: {action}, Reward: {reward:.1f}, Terminated: {terminated}, Info: {info}")

        env_interactive.close()

    except pygame.error as e:
        print(f"Could not start interactive mode. Pygame display not available? Error: {e}")
        print("This is expected if you are in a headless environment.")