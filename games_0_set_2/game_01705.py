
# Generated: 2025-08-28T02:25:53.004955
# Source Brief: brief_01705.md
# Brief Index: 1705

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    GameEnv: A grid-based puzzle game where a robot collects parts in a maze.

    The player controls a robot with the goal of collecting all scattered parts
    within a limited number of moves. The maze is procedurally generated for
    each new episode.

    **State:**
    - Robot's (x, y) position on the grid.
    - A list of (x, y) positions for the remaining parts.
    - The number of moves remaining.

    **Actions (MultiDiscrete([5, 2, 2])):**
    - action[0]: Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - action[1]: Space button (unused)
    - action[2]: Shift button (unused)

    **Reward Structure:**
    - -0.1 per move.
    - +10 for each part collected.
    - +100 for collecting all parts (winning).
    - -50 for running out of moves (losing).

    **Termination:**
    - The episode ends when all parts are collected or when the move limit is reached.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move the robot one step at a time."
    )

    # User-facing description of the game
    game_description = (
        "Navigate a robot through a maze to collect all 5 green parts before you run out of moves. Each step costs."
    )

    # Frames advance only when an action is received.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE
        self.MAX_MOVES = 200
        self.NUM_PARTS = 5

        # --- Colors ---
        self.COLOR_BG = (220, 220, 220)  # Light gray
        self.COLOR_WALL = (60, 60, 60)  # Dark gray
        self.COLOR_ROBOT = (231, 76, 60)  # Red
        self.COLOR_ROBOT_GLOW = (255, 136, 120)
        self.COLOR_PART = (46, 204, 113)  # Green
        self.COLOR_PART_GLOW = (106, 255, 173)
        self.COLOR_TEXT = (20, 20, 20)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 28)
            self.font_game_over = pygame.font.SysFont(None, 52, bold=True)
        
        # --- Game State Variables (initialized in reset) ---
        self.grid = None
        self.robot_pos = None
        self.parts = None
        self.moves_remaining = 0
        self.parts_collected_this_ep = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        # --- Run validation ---
        self.validate_implementation()


    def _generate_maze(self):
        """Generates a maze using a randomized DFS algorithm."""
        grid = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.uint8)  # 1 = wall
        
        # Start DFS from a random odd-indexed cell
        start_x = self.np_random.integers(0, self.GRID_WIDTH // 2) * 2 + 1
        start_y = self.np_random.integers(0, self.GRID_HEIGHT // 2) * 2 + 1
        
        stack = [(start_x, start_y)]
        grid[start_y, start_x] = 0 # 0 = floor

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and grid[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                # Carve path
                grid[ny, nx] = 0
                grid[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Create some loops by removing a few walls
        num_walls_to_remove = int(self.GRID_WIDTH * self.GRID_HEIGHT * 0.05)
        for _ in range(num_walls_to_remove):
            rx = self.np_random.integers(1, self.GRID_WIDTH - 1)
            ry = self.np_random.integers(1, self.GRID_HEIGHT - 1)
            if grid[ry, rx] == 1:
                grid[ry, rx] = 0

        self.grid = grid


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_maze()
        
        # Get all valid floor cells for placement
        floor_cells = np.argwhere(self.grid == 0).tolist()
        self.np_random.shuffle(floor_cells)
        
        # Place robot and parts, ensuring no overlaps
        placements = floor_cells[:self.NUM_PARTS + 1]
        
        robot_y, robot_x = placements[0]
        self.robot_pos = (robot_x, robot_y)
        
        self.parts = []
        for part_y, part_x in placements[1:]:
            self.parts.append((part_x, part_y))
        
        # Initialize game state
        self.moves_remaining = self.MAX_MOVES
        self.parts_collected_this_ep = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = -0.1  # Cost for taking a step/turn
        
        self.moves_remaining -= 1
        
        # --- Handle Movement ---
        current_x, current_y = self.robot_pos
        target_x, target_y = current_x, current_y

        if movement == 1:  # Up
            target_y -= 1
        elif movement == 2:  # Down
            target_y += 1
        elif movement == 3:  # Left
            target_x -= 1
        elif movement == 4:  # Right
            target_x += 1
        
        # Check for valid move (within bounds and not a wall)
        if (0 <= target_x < self.GRID_WIDTH and
            0 <= target_y < self.GRID_HEIGHT and
            self.grid[target_y, target_x] == 0):
            self.robot_pos = (target_x, target_y)

        # --- Check for Part Collection ---
        if self.robot_pos in self.parts:
            self.parts.remove(self.robot_pos)
            reward += 10.0
            self.parts_collected_this_ep += 1
            # sfx: part_collect.wav

        # --- Check for Termination ---
        terminated = False
        if not self.parts: # Win condition
            self.game_over = True
            self.game_won = True
            terminated = True
            reward += 100.0
            # sfx: win_jingle.wav
        elif self.moves_remaining <= 0: # Lose condition
            self.game_over = True
            self.game_won = False
            terminated = True
            reward += -50.0
            # sfx: lose_buzzer.wav
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_remaining": self.moves_remaining,
            "parts_collected": self.parts_collected_this_ep,
            "parts_remaining": len(self.parts),
        }

    def _grid_to_pixels(self, grid_pos):
        """Converts grid coordinates to pixel coordinates for the center of the cell."""
        grid_x, grid_y = grid_pos
        return (
            int(grid_x * self.CELL_SIZE + self.CELL_SIZE / 2),
            int(grid_y * self.CELL_SIZE + self.CELL_SIZE / 2)
        )

    def _render_game(self):
        # Draw walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] == 1:
                    pygame.draw.rect(
                        self.screen,
                        self.COLOR_WALL,
                        (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    )

        # Draw parts
        part_radius = int(self.CELL_SIZE * 0.3)
        for part_pos in self.parts:
            px, py = self._grid_to_pixels(part_pos)
            pygame.gfxdraw.filled_circle(self.screen, px, py, part_radius, self.COLOR_PART)
            pygame.gfxdraw.aacircle(self.screen, px, py, part_radius, self.COLOR_PART)
        
        # Draw robot
        robot_px, robot_py = self._grid_to_pixels(self.robot_pos)
        robot_size = int(self.CELL_SIZE * 0.8)
        robot_rect = pygame.Rect(
            robot_px - robot_size // 2,
            robot_py - robot_size // 2,
            robot_size,
            robot_size
        )
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect, border_radius=2)
        
    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 5))
        
        # Parts remaining
        parts_text = self.font_ui.render(f"Parts: {len(self.parts)}/{self.NUM_PARTS}", True, self.COLOR_TEXT)
        self.screen.blit(parts_text, (self.WIDTH - parts_text.get_width() - 10, 5))

        # Game Over screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.game_won:
                end_text = self.font_game_over.render("ALL PARTS COLLECTED!", True, self.COLOR_PART)
            else:
                end_text = self.font_game_over.render("OUT OF MOVES", True, self.COLOR_ROBOT)
                
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
        # Note: reset must be called before _get_observation can work
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test reset
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a Pygame window to display the environment
    pygame.display.set_caption("Grid Collector")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    terminated = False
    
    print(env.user_guide)
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Press 'r' to reset
                    obs, info = env.reset()
                    terminated = False
                
                if not terminated:
                    # Map keys to actions
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    
                    # Since this is a turn-based game, we only step when a key is pressed.
                    # A no-op action (action[0] == 0) is not sent unless intended.
                    if action[0] != 0:
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    env.close()