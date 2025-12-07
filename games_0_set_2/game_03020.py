
# Generated: 2025-08-27T22:07:37.132264
# Source Brief: brief_03020.md
# Brief Index: 3020

        
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
    """
    A grid-based puzzle environment where a robot navigates through obstacles
    to reach an exit within a limited number of moves. The game is designed
    with a clean, minimalist aesthetic and clear visual feedback.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move the robot one square at a time."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a robot through obstacle-filled grids to reach the green exit. Each move costs a turn. Plan your path carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    MAX_LEVELS = 3
    INITIAL_MOVES = 50
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (50, 50, 60)
    COLOR_TEXT = (220, 220, 220)
    COLOR_ROBOT = (255, 80, 80)
    COLOR_ROBOT_GLOW = (255, 120, 120)
    COLOR_OBSTACLE = (80, 120, 255)
    COLOR_OBSTACLE_GLOW = (120, 160, 255)
    COLOR_EXIT = (80, 255, 120)
    COLOR_EXIT_GLOW = (120, 255, 160)


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # Game state variables
        self.level = 1
        self.robot_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.obstacles = []
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.np_random = None

        # Calculate grid rendering properties
        self.grid_area_size = min(self.SCREEN_WIDTH, self.SCREEN_HEIGHT) - 40
        self.cell_size = self.grid_area_size // self.GRID_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_area_size) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_area_size) // 2

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Use a numpy random generator for reproducible level generation
            self.np_random = np.random.default_rng(seed)
        
        # On first reset, or after a loss, or after completing all levels, start from level 1
        if self.game_over or (options and options.get("new_game", False)):
            self.level = 1
            self.score = 0

        self.steps = 0
        self.game_over = False
        self.moves_left = self.INITIAL_MOVES
        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Generates a new level layout, ensuring it's solvable."""
        num_obstacles = 2 + self.level # Starts at 3 for level 1
        
        while True:
            # Place robot and exit at fixed opposite corners for consistency
            self.robot_pos = (0, self.GRID_SIZE - 1)
            self.exit_pos = (self.GRID_SIZE - 1, 0)

            # Generate obstacle positions
            self.obstacles = []
            possible_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
            possible_coords.remove(self.robot_pos)
            possible_coords.remove(self.exit_pos)
            
            if self.np_random:
                obstacle_indices = self.np_random.choice(len(possible_coords), num_obstacles, replace=False)
                self.obstacles = [possible_coords[i] for i in obstacle_indices]
            else: # Fallback for initial reset without a seed
                random.shuffle(possible_coords)
                self.obstacles = possible_coords[:num_obstacles]

            # Check if a path exists
            if self._is_path_valid():
                break

    def _is_path_valid(self):
        """Checks for a valid path from robot to exit using Breadth-First Search."""
        q = deque([self.robot_pos])
        visited = {self.robot_pos}
        obstacle_set = set(self.obstacles)

        while q:
            x, y = q.popleft()

            if (x, y) == self.exit_pos:
                return True

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and \
                   (nx, ny) not in visited and (nx, ny) not in obstacle_set:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held are unused in this game
        
        reward = -0.1  # Cost of taking a step/turn
        
        # --- Update game logic ---
        self.steps += 1
        
        # Calculate new position
        x, y = self.robot_pos
        if movement == 1:  # Up
            y -= 1
        elif movement == 2:  # Down
            y += 1
        elif movement == 3:  # Left
            x -= 1
        elif movement == 4:  # Right
            x += 1
        
        # Check for valid move (within bounds and not an obstacle)
        is_valid_move = False
        if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE:
            if (x, y) not in self.obstacles:
                is_valid_move = True

        if is_valid_move and movement != 0:
            self.robot_pos = (x, y)
            # Placeholder for move sound effect: sfx.play('move')

        # Decrement moves only if a movement was attempted
        if movement != 0:
            self.moves_left -= 1

        # Check for win/loss conditions
        terminated = False
        if self.robot_pos == self.exit_pos:
            # Win condition
            win_reward = 10.0 * self.level
            reward += win_reward
            self.score += win_reward
            terminated = True
            self.game_over = True
            # Placeholder for win sound effect: sfx.play('win_level')
            if self.level < self.MAX_LEVELS:
                self.level += 1
            else:
                # Game completed, will reset to level 1 on next reset
                self.level = 1 
        elif self.moves_left <= 0:
            # Loss condition
            terminated = True
            self.game_over = True
            # Placeholder for lose sound effect: sfx.play('lose')
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

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
            start_x = self.grid_offset_x + i * self.cell_size
            start_y = self.grid_offset_y
            end_x = start_x
            end_y = self.grid_offset_y + self.grid_area_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y))

            start_x = self.grid_offset_x
            start_y = self.grid_offset_y + i * self.cell_size
            end_x = self.grid_offset_x + self.grid_area_size
            end_y = start_y
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y))

        # Draw obstacles
        for x, y in self.obstacles:
            self._draw_glowing_square(x, y, self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_GLOW)
        
        # Draw exit
        self._draw_glowing_square(self.exit_pos[0], self.exit_pos[1], self.COLOR_EXIT, self.COLOR_EXIT_GLOW)
        
        # Draw robot
        self._draw_glowing_square(self.robot_pos[0], self.robot_pos[1], self.COLOR_ROBOT, self.COLOR_ROBOT_GLOW)

    def _draw_glowing_square(self, grid_x, grid_y, color, glow_color):
        """Helper to draw a square with a soft glow effect for better visuals."""
        px = self.grid_offset_x + grid_x * self.cell_size
        py = self.grid_offset_y + grid_y * self.cell_size
        
        rect = pygame.Rect(px, py, self.cell_size, self.cell_size)
        
        # Draw soft glow by inflating the rect and drawing a transparent shape
        glow_rect = rect.inflate(self.cell_size * 0.5, self.cell_size * 0.5)
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, glow_color, shape_surf.get_rect(), border_radius=int(self.cell_size * 0.3))
        shape_surf.set_alpha(80)
        self.screen.blit(shape_surf, glow_rect.topleft)

        # Draw main square using gfxdraw for anti-aliasing
        inner_rect = rect.inflate(-4, -4)
        pygame.gfxdraw.box(self.screen, inner_rect, color)

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_large.render(f"Moves Left: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 10))

        # Level
        level_text = self.font_large.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 20, 10))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.robot_pos == self.exit_pos:
                msg = "LEVEL COMPLETE!"
            else:
                msg = "GAME OVER"
            
            game_over_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_left": self.moves_left,
            "robot_pos": self.robot_pos,
        }

    def close(self):
        pygame.font.quit()
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

# Example of how to run the environment for visualization and manual play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for display window
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Robot")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default to no-op
        
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
                elif event.key == pygame.K_r: # Reset key
                    obs, info = env.reset(options={"new_game": True})
                    terminated = False
                    print("Game Reset!")
                    continue # Skip step for this frame
                
                # Only step if a movement key was pressed and the game is not over
                if action[0] != 0 and not terminated:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
                
                if terminated:
                    print("Game Over! Press 'R' to restart.")

        # Update the display
        # The observation is the rendered frame, so we just need to display it.
        # We need to transpose it back for pygame's display format (W, H, C).
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate
        
    env.close()