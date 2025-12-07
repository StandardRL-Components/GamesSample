import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A minimalist puzzle game where the player fills a grid with color.

    The goal is to make the entire grid a single color by strategically clicking
    on cells. Clicking a cell of the 'initial' color that is adjacent to the
    'spread' color will convert that cell and all connected cells of the same
    initial color into the spread color. The player has a limited number of
    moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a cell."
    )

    # User-facing description of the game
    game_description = (
        "Fill the grid with a single color. Select cells adjacent to your color "
        "to expand your territory. You have a limited number of moves."
    )

    # The game is turn-based, so it only advances on action.
    auto_advance = False

    # --- Constants ---
    # Game world
    GRID_WIDTH = 20
    GRID_HEIGHT = 12
    CELL_SIZE = 30
    GRID_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
    GRID_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) // 2
    MAX_MOVES = 25
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_GRID_LINES = (10, 10, 20)
    COLOR_EMPTY = (50, 50, 70)
    COLOR_INITIAL = (160, 70, 70)
    COLOR_SPREAD = (0, 200, 200)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_HIGHLIGHT = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_WIN = (70, 220, 70)
    COLOR_LOSE = (220, 70, 70)

    # Cell states
    STATE_EMPTY = 0
    STATE_INITIAL = 1
    STATE_SPREAD = 2

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)

        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.moves_left = None
        self.valid_moves = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_state = None

        # self.validate_implementation() # Defer validation or ensure reset is called first

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self._generate_puzzle()
        self.valid_moves = self._find_valid_moves()

        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        self.grid = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), self.STATE_EMPTY, dtype=np.int8)
        
        # Create random blobs of the initial color
        num_blobs = self.np_random.integers(5, 10)
        for _ in range(num_blobs):
            start_x = self.np_random.integers(0, self.GRID_WIDTH)
            start_y = self.np_random.integers(0, self.GRID_HEIGHT)
            blob_size = self.np_random.integers(10, 30)
            
            q = [(start_y, start_x)]
            visited = set(q)
            if 0 <= start_y < self.GRID_HEIGHT and 0 <= start_x < self.GRID_WIDTH:
                self.grid[start_y, start_x] = self.STATE_INITIAL
            
            for _ in range(blob_size - 1):
                if not q: break
                cy, cx = random.choice(q)
                
                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < self.GRID_HEIGHT and 0 <= nx < self.GRID_WIDTH and (ny, nx) not in visited:
                        if self.np_random.random() < 0.6:
                            self.grid[ny, nx] = self.STATE_INITIAL
                            q.append((ny, nx))
                        visited.add((ny,nx))

        # Ensure there is at least one initial cell to start with
        initial_cells = list(zip(*np.where(self.grid == self.STATE_INITIAL)))
        if not initial_cells:
            # If generation failed, create a simple line
            for i in range(self.GRID_WIDTH // 2):
                self.grid[self.GRID_HEIGHT//2, i] = self.STATE_INITIAL
            initial_cells = list(zip(*np.where(self.grid == self.STATE_INITIAL)))

        # Pick a random initial cell and convert it to the spread color
        start_y, start_x = random.choice(initial_cells)
        self.grid[start_y, start_x] = self.STATE_SPREAD


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        reward = 0

        # 1. Handle cursor movement
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_WIDTH
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_HEIGHT
            # Sound effect: cursor move
        
        # 2. Handle cell selection
        if space_pressed:
            cursor_tuple = (self.cursor_pos[1], self.cursor_pos[0])
            if cursor_tuple in self.valid_moves:
                # Sound effect: valid click
                self.moves_left -= 1
                cells_filled = self._flood_fill(cursor_tuple)
                reward += cells_filled
                self.score += cells_filled
            else:
                # Sound effect: invalid buzz
                pass # No penalty for invalid clicks

        self.steps += 1
        self.valid_moves = self._find_valid_moves()
        terminated, win = self._check_termination()
        
        if terminated:
            self.game_over = True
            self.win_state = win
            if win:
                reward += 50
                self.score += 50

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _flood_fill(self, start_pos):
        """
        Fills a connected area of STATE_INITIAL with STATE_SPREAD.
        Returns the number of cells filled.
        """
        q = [start_pos]
        visited = {start_pos}
        
        # Check if start_pos is valid before accessing grid
        if not (0 <= start_pos[0] < self.GRID_HEIGHT and 0 <= start_pos[1] < self.GRID_WIDTH):
            return 0
        
        target_color = self.grid[start_pos]
        
        if target_color != self.STATE_INITIAL:
            return 0
            
        count = 0
        while q:
            y, x = q.pop(0)
            
            if self.grid[y, x] == self.STATE_INITIAL:
                self.grid[y, x] = self.STATE_SPREAD
                count += 1
                
                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.GRID_HEIGHT and 0 <= nx < self.GRID_WIDTH and \
                       (ny, nx) not in visited and self.grid[ny, nx] == self.STATE_INITIAL:
                        visited.add((ny, nx))
                        q.append((ny, nx))
        return count

    def _find_valid_moves(self):
        """Finds all initial cells adjacent to any spread cell."""
        valid_moves = set()
        if self.grid is None:
            return valid_moves
            
        spread_cells = np.where(self.grid == self.STATE_SPREAD)
        
        for y, x in zip(*spread_cells):
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.GRID_HEIGHT and 0 <= nx < self.GRID_WIDTH and \
                   self.grid[ny, nx] == self.STATE_INITIAL:
                    valid_moves.add((ny, nx))
        return valid_moves

    def _check_termination(self):
        """Checks for win, loss, or max steps conditions."""
        # Win condition: No initial cells are left
        if not np.any(self.grid == self.STATE_INITIAL):
            return True, True  # Terminated, Win

        # Loss condition: No moves left OR no valid moves possible
        if self.moves_left <= 0 or not self.valid_moves:
            return True, False  # Terminated, Loss

        # Max steps reached
        if self.steps >= self.MAX_STEPS:
            return True, False  # Terminated, Loss
        
        return False, False # Not terminated

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Create a surface for highlighting with per-pixel alpha
        highlight_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        highlight_surface.fill((*self.COLOR_HIGHLIGHT, 60))

        # Draw grid cells
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                cell_rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                
                state = self.grid[y, x]
                if state == self.STATE_EMPTY:
                    color = self.COLOR_EMPTY
                elif state == self.STATE_INITIAL:
                    color = self.COLOR_INITIAL
                else:  # self.STATE_SPREAD
                    color = self.COLOR_SPREAD
                
                pygame.draw.rect(self.screen, color, cell_rect)
                
                # Draw highlight for valid moves
                if (y, x) in self.valid_moves:
                    self.screen.blit(highlight_surface, cell_rect.topleft)

        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_AREA_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 1)
        for i in range(self.GRID_HEIGHT + 1):
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 1)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 15))

        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 15))
        self.screen.blit(score_text, score_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_state:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE
                
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

    def _validate_implementation(self):
        """
        Helper method to check if the environment implementation is valid.
        This is not part of the standard gym.Env API.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset and observation space
        # Calling reset() is crucial as it initializes the game state (e.g., self.grid)
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {obs.shape}"
        assert obs.dtype == np.uint8, f"Obs dtype is {obs.dtype}"
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls
    # It also serves as a validation test when run
    try:
        env = GameEnv(render_mode="rgb_array")
        env._validate_implementation() # Run validation
        
        # Setup Pygame window for human play
        # Re-initialize pygame with default video driver for display
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
        pygame.quit()
        pygame.init()
        pygame.display.set_caption("Color Grid Puzzle")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        
        obs, info = env.reset()
        running = True
        terminated = False

        while running:
            movement = 0  # No-op
            space_pressed = 0
            shift_pressed = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                    
                    if not terminated:
                        if event.key == pygame.K_UP:
                            movement = 1
                        elif event.key == pygame.K_DOWN:
                            movement = 2
                        elif event.key == pygame.K_LEFT:
                            movement = 3
                        elif event.key == pygame.K_RIGHT:
                            movement = 4
                        elif event.key == pygame.K_SPACE:
                            space_pressed = 1
            
            if not terminated:
                # For turn-based games, we only step when an action is taken
                if movement != 0 or space_pressed != 0:
                    action = np.array([movement, space_pressed, shift_pressed])
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
            
            # Render the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(30)  # Limit frame rate

        env.close()
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()