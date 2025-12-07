import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your pixel."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a pixel through a shifting grid to reach the target destination within a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    # Colors
    COLOR_BG = (0, 0, 0) # Black
    COLOR_GRID = (40, 40, 40) # Dark Gray
    COLOR_PLAYER = (0, 255, 0) # Bright Green
    COLOR_PLAYER_GLOW = (0, 100, 0) # Dark Green
    COLOR_TARGET = (255, 0, 0) # Red
    COLOR_OBSTACLE = (0, 100, 200) # Dark Blue
    COLOR_TEXT = (255, 255, 255) # White
    COLOR_GAMEOVER = (255, 80, 80) # Light Red

    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    CELL_SIZE = 36
    GRID_WIDTH = GRID_HEIGHT = GRID_SIZE * CELL_SIZE
    GRID_ORIGIN_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_ORIGIN_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    # Game Mechanics
    INITIAL_MOVES = 20
    MAX_OBSTACLES = 3
    MAX_EPISODE_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_main = pygame.font.SysFont('monospace', 24, bold=True)
        self.font_gameover = pygame.font.SysFont('monospace', 48, bold=True)
        
        # Etc...        
        self.player_pos = (0, 0)
        self.target_pos = (0, 0)
        self.obstacles = []
        self.moves_remaining = 0
        self.level = 1
        self.score = 0
        self.total_steps = 0
        self.target_move_frequency = 0
        self.obstacle_move_frequency = 25
        self.game_over = False
        self.level_clear_msg_timer = 0
        
        # Initialize state variables
        # self.reset() is called after super().reset() in the reset method itself
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.level = 1
        self.moves_remaining = self.INITIAL_MOVES
        self.score = 0
        self.total_steps = 0
        self.target_move_frequency = 50
        self.game_over = False
        self.level_clear_msg_timer = 0

        # Place player and target
        self.player_pos = tuple(self.np_random.integers(0, self.GRID_SIZE, size=2))
        self.target_pos = self._get_random_empty_pos([self.player_pos])

        # Generate initial obstacles
        self._update_obstacles()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        if self.level_clear_msg_timer > 0:
            self.level_clear_msg_timer -= 1

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Boolean (unused)
        # shift_held = action[2] == 1  # Boolean (unused)
        
        reward = -0.1  # Cost of taking a step/turn
        old_dist = self._manhattan_distance(self.player_pos, self.target_pos)
        
        # Only process movement if it's not a no-op
        if movement != 0:
            self.moves_remaining -= 1
            
            next_pos = list(self.player_pos)
            if movement == 1:  # Up
                next_pos[1] -= 1
            elif movement == 2:  # Down
                next_pos[1] += 1
            elif movement == 3:  # Left
                next_pos[0] -= 1
            elif movement == 4:  # Right
                next_pos[0] += 1

            # Check boundaries and obstacles
            if (0 <= next_pos[0] < self.GRID_SIZE and
                0 <= next_pos[1] < self.GRID_SIZE and
                tuple(next_pos) not in self.obstacles):
                self.player_pos = tuple(next_pos)
                # SFX: Player move sound

        new_dist = self._manhattan_distance(self.player_pos, self.target_pos)

        # Distance-based reward
        if new_dist < old_dist:
            reward += 1.0
        elif new_dist > old_dist:
            reward -= 1.0

        self.total_steps += 1
        terminated = False

        # Check for reaching the target (level clear)
        if self.player_pos == self.target_pos:
            reward += 100.0
            self.score += 100
            self.level += 1
            self.moves_remaining = self.INITIAL_MOVES
            self.target_move_frequency = max(10, 50 - (self.level - 1) * 5)
            self.target_pos = self._get_random_empty_pos([self.player_pos] + self.obstacles)
            self.level_clear_msg_timer = 2 # Show message for 1 step after this one
            self._update_obstacles()
            # SFX: Level clear fanfare

        # Check for game over conditions
        if self.moves_remaining <= 0:
            reward -= 100.0
            self.game_over = True
            terminated = True
            # SFX: Game over sound

        if self.total_steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            terminated = True
            
        # Update dynamic elements
        if not terminated:
            if self.total_steps > 0 and self.total_steps % self.obstacle_move_frequency == 0:
                self._update_obstacles()
                # SFX: Obstacle shift sound
            if self.total_steps > 0 and self.total_steps % self.target_move_frequency == 0:
                self._update_target()
                # SFX: Target move sound
        
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.total_steps,
            "level": self.level,
            "moves_remaining": self.moves_remaining,
        }

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            x_start = self.GRID_ORIGIN_X + i * self.CELL_SIZE
            y_start = self.GRID_ORIGIN_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x_start, self.GRID_ORIGIN_Y), (x_start, self.GRID_ORIGIN_Y + self.GRID_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_ORIGIN_X, y_start), (self.GRID_ORIGIN_X + self.GRID_WIDTH, y_start))

        # Draw obstacles
        for obs_pos in self.obstacles:
            self._draw_grid_cell(obs_pos, self.COLOR_OBSTACLE)
        
        # Draw target
        self._draw_grid_cell(self.target_pos, self.COLOR_TARGET)
        
        # Draw player with glow
        self._draw_grid_cell(self.player_pos, self.COLOR_PLAYER_GLOW, scale=1.1)
        self._draw_grid_cell(self.player_pos, self.COLOR_PLAYER)

    def _render_ui(self):
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 15))

        level_text = self.font_main.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 20, 15))

        if self.game_over:
            gameover_surf = self.font_gameover.render("GAME OVER", True, self.COLOR_GAMEOVER)
            gameover_rect = gameover_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(gameover_surf, gameover_rect)
        elif self.level_clear_msg_timer > 0:
            level_clear_surf = self.font_gameover.render("LEVEL CLEAR!", True, self.COLOR_PLAYER)
            level_clear_rect = level_clear_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(level_clear_surf, level_clear_rect)

    def _grid_to_pixel(self, grid_pos):
        px = self.GRID_ORIGIN_X + grid_pos[0] * self.CELL_SIZE
        py = self.GRID_ORIGIN_Y + grid_pos[1] * self.CELL_SIZE
        return px, py

    def _draw_grid_cell(self, grid_pos, color, scale=0.8):
        px, py = self._grid_to_pixel(grid_pos)
        cell_offset = (1 - scale) * self.CELL_SIZE / 2
        
        rect = pygame.Rect(
            int(px + cell_offset),
            int(py + cell_offset),
            int(self.CELL_SIZE * scale),
            int(self.CELL_SIZE * scale)
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=3)

    def _get_random_empty_pos(self, occupied_pos):
        occupied_set = set(occupied_pos)
        while True:
            pos = tuple(self.np_random.integers(0, self.GRID_SIZE, size=2))
            if pos not in occupied_set:
                return pos

    def _update_obstacles(self):
        self.obstacles = []
        num_obstacles = self.np_random.integers(0, self.MAX_OBSTACLES + 1)
        
        # Try to generate a valid obstacle layout with a path
        for _ in range(100): # Safety break
            potential_obstacles = []
            occupied = {self.player_pos, self.target_pos}
            for _ in range(num_obstacles):
                obs_pos = self._get_random_empty_pos(list(occupied))
                potential_obstacles.append(obs_pos)
                occupied.add(obs_pos)
            
            if self._is_path_available(self.player_pos, self.target_pos, potential_obstacles):
                self.obstacles = potential_obstacles
                return
        
        # Fallback: if no valid layout is found after 100 tries, clear obstacles
        self.obstacles = []

    def _update_target(self):
        self.target_pos = self._get_random_empty_pos([self.player_pos] + self.obstacles)

    def _is_path_available(self, start, end, obstacles):
        obstacle_set = set(obstacles)
        q = deque([start])
        visited = {start}
        while q:
            current = q.popleft()
            if current == end:
                return True
            
            cx, cy = current
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)
                if (0 <= nx < self.GRID_SIZE and
                    0 <= ny < self.GRID_SIZE and
                    neighbor not in visited and
                    neighbor not in obstacle_set):
                    visited.add(neighbor)
                    q.append(neighbor)
        return False

    @staticmethod
    def _manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for debugging and not required by the final product.
        # It's good practice to have it during development.
        print("Attempting to validate implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space after reset
        test_obs_after_reset = self._get_observation()
        assert test_obs_after_reset.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs_after_reset.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: This will create a visible window.
    # To run headlessly, the main loop needs to be modified.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS" depending on your OS
    
    env = GameEnv()
    env.validate_implementation()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to action movements
    key_to_movement = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Since auto_advance is False, we need a display window
    # and a game loop that waits for player input.
    pygame.display.set_caption("Grid Runner")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    while running:
        movement_action = 0  # Default to no-op
        space_action = 0
        shift_action = 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_movement:
                    movement_action = key_to_movement[event.key]
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    done = False
        
        # Only step if an action was taken
        if movement_action != 0:
            if not done:
                # Construct the action for the step function
                action = [movement_action, space_action, shift_action]
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Rendering
        # The observation is already the rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()