
# Generated: 2025-08-27T17:46:27.860857
# Source Brief: brief_01634.md
# Brief Index: 1634

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to navigate the maze. Reach the highest number before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated number maze to reach the target number within a limited number of steps."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 16
        self.GRID_HEIGHT = 10
        self.CELL_SIZE = 40

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
        
        # Fonts
        try:
            self.number_font = pygame.font.SysFont("Consolas", 20, bold=True)
            self.ui_font = pygame.font.SysFont("Consolas", 24, bold=True)
            self.game_over_font = pygame.font.SysFont("Consolas", 70, bold=True)
        except pygame.error:
            self.number_font = pygame.font.Font(None, 24)
            self.ui_font = pygame.font.Font(None, 28)
            self.game_over_font = pygame.font.Font(None, 80)


        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (50, 50, 70)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_TARGET = (255, 50, 50)
        self.COLOR_START = (50, 200, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_LOW_NUM = (80, 80, 100)
        self.COLOR_HIGH_NUM = (200, 200, 220)

        # Game state variables are initialized in reset()
        self.grid = None
        self.player_pos = None
        self.start_pos = None
        self.target_pos = None
        self.target_number = 0
        self.max_steps = 0
        self.remaining_steps = 0
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        # Initialize state variables
        self.reset()
        
    def _generate_maze(self):
        """Generates a new solvable maze."""
        # 1. Create a shuffled grid of numbers
        grid_size = self.GRID_WIDTH * self.GRID_HEIGHT
        numbers = list(range(1, grid_size + 1))
        self.np_random.shuffle(numbers)
        self.grid = np.array(numbers).reshape((self.GRID_HEIGHT, self.GRID_WIDTH))
        
        self.target_number = grid_size

        # 2. Find start (1) and target (max number) positions
        start_indices = np.where(self.grid == 1)
        self.start_pos = (int(start_indices[1][0]), int(start_indices[0][0]))
        
        target_indices = np.where(self.grid == self.target_number)
        self.target_pos = (int(target_indices[1][0]), int(target_indices[0][0]))

        # 3. Ensure a path exists and set step limit
        path = self._find_shortest_path(self.start_pos, self.target_pos)
        if path is None:
            # This is very unlikely in an open grid, but as a fallback, we regenerate.
            self._generate_maze()
            return

        shortest_path_len = len(path) - 1
        self.max_steps = math.ceil(shortest_path_len * 1.5)
        if self.max_steps <= shortest_path_len:
             self.max_steps = shortest_path_len + 5 # Ensure some wiggle room
        
        self.remaining_steps = self.max_steps

    def _find_shortest_path(self, start, end):
        """BFS to find the shortest path in the grid."""
        q = deque([(start, [start])])
        visited = {start}
        
        while q:
            (vx, vy), path = q.popleft()

            if (vx, vy) == end:
                return path

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = vx + dx, vy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = list(path)
                    new_path.append((nx, ny))
                    q.append(((nx, ny), new_path))
        return None # No path found

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_maze()
        
        self.player_pos = self.start_pos
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0.0
        px, py = self.player_pos
        current_number = self.grid[py][px]

        # Process movement
        if movement != 0: # If not a no-op
            nx, ny = px, py
            if movement == 1: ny -= 1  # Up
            elif movement == 2: ny += 1 # Down
            elif movement == 3: nx -= 1 # Left
            elif movement == 4: nx += 1 # Right

            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                self.player_pos = (nx, ny)
                new_number = self.grid[ny][nx]
                
                if new_number > current_number:
                    reward += 0.1
                elif new_number < current_number:
                    reward += -0.2
                # No reward for moving to same-value cell
            else:
                # Penalty for hitting a wall
                reward += -0.5

            self.steps += 1
            self.remaining_steps -= 1

        self.score += reward
        
        terminated = self._check_termination()
        if terminated and self.player_pos == self.target_pos:
            win_reward = 100.0
            reward += win_reward
            self.score += win_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        if self.player_pos == self.target_pos:
            self.game_over = True
            return True
        if self.remaining_steps <= 0:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        max_val = self.GRID_WIDTH * self.GRID_HEIGHT
        
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                
                cell_value = self.grid[y][x]
                
                # Determine cell color
                is_start = (x, y) == self.start_pos
                is_target = (x, y) == self.target_pos

                if is_target:
                    cell_color = self.COLOR_TARGET
                elif is_start:
                    cell_color = self.COLOR_START
                else:
                    # Grayscale gradient for other numbers
                    normalized_value = (cell_value - 1) / max(1, max_val - 1)
                    r = self.COLOR_LOW_NUM[0] + normalized_value * (self.COLOR_HIGH_NUM[0] - self.COLOR_LOW_NUM[0])
                    g = self.COLOR_LOW_NUM[1] + normalized_value * (self.COLOR_HIGH_NUM[1] - self.COLOR_LOW_NUM[1])
                    b = self.COLOR_LOW_NUM[2] + normalized_value * (self.COLOR_HIGH_NUM[2] - self.COLOR_LOW_NUM[2])
                    cell_color = (int(r), int(g), int(b))

                pygame.draw.rect(self.screen, cell_color, rect)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                # Render number text
                text_surface = self.number_font.render(str(cell_value), True, self.COLOR_TEXT)
                text_rect = text_surface.get_rect(center=rect.center)
                self.screen.blit(text_surface, text_rect)
        
        # Highlight player position
        px, py = self.player_pos
        player_rect = pygame.Rect(px * self.CELL_SIZE, py * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, 4) # Thick border

    def _render_ui(self):
        # Moves remaining
        moves_text = f"Moves: {max(0, self.remaining_steps)}/{self.max_steps}"
        moves_surf = self.ui_font.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (10, 10))

        # Score
        score_text = f"Score: {self.score:.2f}"
        score_surf = self.ui_font.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            won = self.player_pos == self.target_pos
            message = "YOU WON!" if won else "OUT OF MOVES"
            msg_color = self.COLOR_PLAYER if won else self.COLOR_TARGET
            
            msg_surf = self.game_over_font.render(message, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_steps": self.remaining_steps,
            "player_pos": self.player_pos,
            "target_pos": self.target_pos,
        }

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # This part of the script is for validation and manual play.
    # To run headlessly (e.g., on a server), you might need to
    # set an environment variable:
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    # --- Validation ---
    env = GameEnv()
    env.validate_implementation()
    env.close()

    # --- Manual Play ---
    # To play manually, you need a display.
    # Make sure you have gymnasium[box2d] or other packages
    # that install the necessary pygame dependencies.
    print("\nStarting manual play...")
    print(GameEnv.user_guide)
    from gymnasium.utils.play import play
    
    # We need to create a new instance of the environment for play
    play_env = GameEnv(render_mode="rgb_array")
    
    # Define key mappings for the MultiDiscrete action space
    # action = [movement, space, shift]
    key_map = {
        "w": np.array([1, 0, 0]),      # Up
        "s": np.array([2, 0, 0]),      # Down
        "a": np.array([3, 0, 0]),      # Left
        "d": np.array([4, 0, 0]),      # Right
        "ArrowUp": np.array([1, 0, 0]),
        "ArrowDown": np.array([2, 0, 0]),
        "ArrowLeft": np.array([3, 0, 0]),
        "ArrowRight": np.array([4, 0, 0]),
    }

    play(play_env, keys_to_action=key_map, noop=np.array([0, 0, 0]))