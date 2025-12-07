
# Generated: 2025-08-27T20:06:24.306845
# Source Brief: brief_02350.md
# Brief Index: 2350

        
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
    """
    A puzzle game where the player pushes colored pixels on a grid to match a target image.
    The game is turn-based, with a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to push all pixels in a direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push colored pixels around the grid to match the target image in the top right. You have a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment.
        """
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 8
        self.CELL_SIZE = 32
        self.MAX_MOVES = 10
        self.NUM_PIXELS_PER_COLOR = 3
        self.SCRAMBLE_MOVES = 5

        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_GRID_BG = (40, 40, 60)
        self.COLOR_GRID_LINES = (60, 60, 80)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_CORRECT_OUTLINE = (255, 255, 255)
        self.PIXEL_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
        ]
        
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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Grid positioning
        self.grid_render_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.grid_render_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        
        # Initialize state variables
        self.current_grid = None
        self.target_grid = None
        self.total_pixels = 0
        self.moves_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment to a new initial state.
        """
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        """
        Generates a new solvable puzzle by creating a target and scrambling it.
        """
        # Create the target configuration
        self.target_grid = [[None for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        available_pos = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(available_pos)
        
        pixel_idx = 0
        self.total_pixels = 0
        for color_idx in range(len(self.PIXEL_COLORS)):
            for _ in range(self.NUM_PIXELS_PER_COLOR):
                if not available_pos: break
                x, y = available_pos.pop()
                self.target_grid[y][x] = color_idx
                self.total_pixels += 1
        
        # Scramble the target to create the starting grid
        self.current_grid = [row[:] for row in self.target_grid]
        for _ in range(self.SCRAMBLE_MOVES):
            # A push action is its own inverse, so we can just push randomly
            direction = self.np_random.integers(1, 5) # 1-4 for up/down/left/right
            self._push_pixels(direction, grid_to_modify=self.current_grid)


    def step(self, action):
        """
        Executes one time step within the environment.
        """
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        
        # Only push actions (1-4) consume a move
        if 1 <= movement <= 4:
            self.steps += 1
            self.moves_left -= 1
            # sfx: whoosh sound
            self._push_pixels(movement, grid_to_modify=self.current_grid)
            reward = self._calculate_reward()
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self._is_solved():
                # sfx: success chime
                reward += 50 # Completion bonus

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _push_pixels(self, direction, grid_to_modify):
        """
        Pushes all pixels in the given direction.
        1=up, 2=down, 3=left, 4=right
        """
        if direction == 1: # Up
            for x in range(self.GRID_WIDTH):
                col_pixels = [grid_to_modify[y][x] for y in range(self.GRID_HEIGHT) if grid_to_modify[y][x] is not None]
                for y in range(self.GRID_HEIGHT):
                    grid_to_modify[y][x] = col_pixels.pop(0) if col_pixels else None
        elif direction == 2: # Down
            for x in range(self.GRID_WIDTH):
                col_pixels = [grid_to_modify[y][x] for y in range(self.GRID_HEIGHT) if grid_to_modify[y][x] is not None]
                for y in range(self.GRID_HEIGHT - 1, -1, -1):
                    grid_to_modify[y][x] = col_pixels.pop() if col_pixels else None
        elif direction == 3: # Left
            for y in range(self.GRID_HEIGHT):
                row_pixels = [grid_to_modify[y][x] for x in range(self.GRID_WIDTH) if grid_to_modify[y][x] is not None]
                for x in range(self.GRID_WIDTH):
                    grid_to_modify[y][x] = row_pixels.pop(0) if row_pixels else None
        elif direction == 4: # Right
            for y in range(self.GRID_HEIGHT):
                row_pixels = [grid_to_modify[y][x] for x in range(self.GRID_WIDTH) if grid_to_modify[y][x] is not None]
                for x in range(self.GRID_WIDTH - 1, -1, -1):
                    grid_to_modify[y][x] = row_pixels.pop() if row_pixels else None

    def _calculate_reward(self):
        """Calculates the reward based on the number of correctly placed pixels."""
        correct_pixels = 0
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.current_grid[y][x] is not None and self.current_grid[y][x] == self.target_grid[y][x]:
                    correct_pixels += 1
        
        # The score is the number of correct pixels
        self.score = correct_pixels
        
        # Reward is 2 points per correct pixel
        return correct_pixels * 2

    def _is_solved(self):
        """Checks if the current grid matches the target grid."""
        return self.score == self.total_pixels

    def _check_termination(self):
        """Checks if the episode should terminate."""
        if self._is_solved():
            return True
        if self.moves_left <= 0:
            return True
        return False
        
    def _get_observation(self):
        """
        Renders the current game state to an RGB array.
        """
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the main grid, pixels, and target preview."""
        # Draw main grid background
        grid_rect = pygame.Rect(self.grid_render_x, self.grid_render_y, 
                                self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_render_x + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.grid_render_y), (x, self.grid_render_y + self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_render_y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.grid_render_x, y), (self.grid_render_x + self.GRID_WIDTH * self.CELL_SIZE, y))

        # Draw pixels on the main grid
        radius = int(self.CELL_SIZE * 0.4)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.current_grid[y][x]
                if color_idx is not None:
                    center_x = self.grid_render_x + x * self.CELL_SIZE + self.CELL_SIZE // 2
                    center_y = self.grid_render_y + y * self.CELL_SIZE + self.CELL_SIZE // 2
                    
                    # Draw pixel
                    color = self.PIXEL_COLORS[color_idx]
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)

                    # Highlight if correct
                    if color_idx == self.target_grid[y][x]:
                        # sfx: subtle correct placement 'ping'
                        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius + 2, self.COLOR_CORRECT_OUTLINE)
        
        # Draw target preview
        preview_cell_size = 10
        preview_x = self.SCREEN_WIDTH - self.GRID_WIDTH * preview_cell_size - 20
        preview_y = 40
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (preview_x - 5, preview_y - 5, self.GRID_WIDTH * preview_cell_size + 10, self.GRID_HEIGHT * preview_cell_size + 10), border_radius=5)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.target_grid[y][x]
                if color_idx is not None:
                    rect = pygame.Rect(preview_x + x * preview_cell_size, preview_y + y * preview_cell_size, preview_cell_size, preview_cell_size)
                    pygame.draw.rect(self.screen, self.PIXEL_COLORS[color_idx], rect)

    def _render_ui(self):
        """Renders the UI text elements."""
        # Moves Left
        moves_text = self.font_large.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}/{self.total_pixels}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, self.SCREEN_HEIGHT - 40))

        # Target Label
        target_label = self.font_small.render("TARGET", True, self.COLOR_TEXT)
        preview_x = self.SCREEN_WIDTH - self.GRID_WIDTH * 10 - 20
        self.screen.blit(target_label, (preview_x - 5, 20))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "PUZZLE COMPLETE!" if self._is_solved() else "OUT OF MOVES"
            end_text = self.font_large.render(message, True, (255, 255, 100))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        """Returns a dictionary with auxiliary diagnostic information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "is_solved": self._is_solved()
        }

    def close(self):
        """Closes the Pygame window."""
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Pixel Pusher")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    print("\n" + "="*30)
    print("Pixel Pusher - Manual Control")
    print(env.user_guide)
    print("Press R to reset, Q to quit.")
    print("="*30 + "\n")

    while not done:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                
                # Map keys to actions
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
        
        # If an action was taken, step the environment
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print("Game Over! Press R to play again or Q to quit.")
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()