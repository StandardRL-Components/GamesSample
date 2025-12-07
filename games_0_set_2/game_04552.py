
# Generated: 2025-08-28T02:45:01.540448
# Source Brief: brief_04552.md
# Brief Index: 4552

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to jump to an adjacent tile. Reach the goal in the fewest moves."
    )

    # Short, user-facing description of the game
    game_description = (
        "A strategic puzzle game. Navigate a grid of disappearing tiles to reach the goal. "
        "Blue tiles are safe, but light blue tiles are fragile and will disappear after you jump off them. "
        "Plan your path carefully to reach the red goal tile before you run out of moves or fall into the void."
    )

    # Frames only advance when an action is received
    auto_advance = False
    
    # --- Constants ---
    GRID_DIM = 10
    MAX_MOVES = 50
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Tile states
    TILE_INACTIVE = 0
    TILE_SAFE = 1
    TILE_FRAGILE = 2
    TILE_GOAL = 3

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (30, 45, 60)
    COLOR_INACTIVE = (40, 60, 80)
    COLOR_SAFE = (50, 100, 200)
    COLOR_FRAGILE = (100, 180, 255)
    COLOR_GOAL_BASE = (220, 50, 50)
    COLOR_START = (50, 200, 100)
    COLOR_PLAYER = (255, 200, 0)
    COLOR_PLAYER_OUTLINE = (255, 255, 255)
    COLOR_TEXT = (230, 230, 230)
    COLOR_WIN = (100, 255, 150)
    COLOR_LOSE = (255, 100, 100)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 24)
        self.font_large = pygame.font.SysFont("Consolas", 60, bold=True)
        
        # Game rendering parameters
        self.tile_size = 32
        self.tile_spacing = 4
        self.grid_total_size = self.GRID_DIM * self.tile_size + (self.GRID_DIM - 1) * self.tile_spacing
        self.offset_x = (self.SCREEN_WIDTH - self.grid_total_size) // 2
        self.offset_y = (self.SCREEN_HEIGHT - self.grid_total_size) // 2
        
        # Initialize state variables
        self.grid = None
        self.player_pos = None
        self.start_pos = None
        self.moves_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_status = ""
        self.player_fallen = False
        
        self.reset()

        # Self-check to ensure implementation correctness
        # self.validate_implementation() # Optional: uncomment for debugging
    
    def _generate_level(self):
        """Generates a new 10x10 grid with a guaranteed path to the goal."""
        self.grid = np.full((self.GRID_DIM, self.GRID_DIM), self.TILE_INACTIVE, dtype=np.int8)
        
        # 1. Place start and goal
        while True:
            self.start_pos = tuple(self.np_random.integers(0, self.GRID_DIM, size=2))
            goal_pos = tuple(self.np_random.integers(0, self.GRID_DIM, size=2))
            # Ensure start and goal are reasonably far apart
            if np.sum(np.abs(np.array(self.start_pos) - np.array(goal_pos))) > self.GRID_DIM:
                break
        
        # 2. Find a guaranteed path using BFS
        q = deque([(self.start_pos, [self.start_pos])])
        visited = {self.start_pos}
        path = []
        
        while q:
            curr, p = q.popleft()
            if curr == goal_pos:
                path = p
                break
            
            x, y = curr
            # Randomize neighbor order to get different paths
            neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
            self.np_random.shuffle(neighbors)
            
            for nx, ny in neighbors:
                if 0 <= nx < self.GRID_DIM and 0 <= ny < self.GRID_DIM and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append(((nx, ny), p + [(nx, ny)]))
        
        # 3. Populate grid based on path and randomness
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                pos = (c, r)
                if pos in path:
                    self.grid[r, c] = self.TILE_SAFE
                else:
                    # Fill other tiles randomly
                    rand_val = self.np_random.random()
                    if rand_val < 0.40:
                        self.grid[r, c] = self.TILE_SAFE
                    elif rand_val < 0.75:
                        self.grid[r, c] = self.TILE_FRAGILE
                    else:
                        self.grid[r, c] = self.TILE_INACTIVE

        self.grid[goal_pos[1], goal_pos[0]] = self.TILE_GOAL
        self.player_pos = self.start_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = ""
        self.player_fallen = False
        self.moves_remaining = self.MAX_MOVES
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        
        movement = action[0]
        
        # Only process non-noop actions
        if movement != 0:
            self.moves_remaining -= 1
            reward -= 0.1  # Cost for making a move
            
            px, py = self.player_pos
            old_pos = self.player_pos
            
            if movement == 1: py -= 1  # Up
            elif movement == 2: py += 1  # Down
            elif movement == 3: px -= 1  # Left
            elif movement == 4: px += 1  # Right
            
            new_pos = (px, py)
            
            # Check if move is valid
            if not (0 <= px < self.GRID_DIM and 0 <= py < self.GRID_DIM and self.grid[py, px] != self.TILE_INACTIVE):
                # Invalid move: fell off or into a hole
                reward = -10.0
                terminated = True
                self.game_over = True
                self.win_status = "FELL"
                self.player_fallen = True
            else:
                # Valid move
                self.player_pos = new_pos
                
                # Deactivate previous tile if it was fragile
                if self.grid[old_pos[1], old_pos[0]] == self.TILE_FRAGILE:
                    self.grid[old_pos[1], old_pos[0]] = self.TILE_INACTIVE
                    # sfx: tile_crack.wav
                
                # Check new tile state
                new_tile_type = self.grid[new_pos[1], new_pos[0]]
                if new_tile_type == self.TILE_GOAL:
                    reward = 10.0
                    terminated = True
                    self.game_over = True
                    self.win_status = "VICTORY!"
                    # sfx: win.wav
                elif new_tile_type == self.TILE_FRAGILE:
                    reward += 0.2 # Bonus for landing on a risky tile
            
            # Check for termination by running out of moves
            if self.moves_remaining <= 0 and not terminated:
                reward = -10.0
                terminated = True
                self.game_over = True
                self.win_status = "OUT OF MOVES"
                # sfx: lose.wav

            # Check if player is trapped
            if not terminated:
                is_trapped = True
                cx, cy = self.player_pos
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.GRID_DIM and 0 <= ny < self.GRID_DIM and self.grid[ny, nx] != self.TILE_INACTIVE:
                        is_trapped = False
                        break
                if is_trapped:
                    reward = -10.0
                    terminated = True
                    self.game_over = True
                    self.win_status = "TRAPPED"
                    
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
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
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
        }

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.offset_x, self.offset_y, self.grid_total_size, self.grid_total_size)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=5)

        # Draw tiles
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                tile_type = self.grid[r, c]
                if tile_type == self.TILE_INACTIVE:
                    continue

                tile_x = self.offset_x + c * (self.tile_size + self.tile_spacing)
                tile_y = self.offset_y + r * (self.tile_size + self.tile_spacing)
                rect = pygame.Rect(tile_x, tile_y, self.tile_size, self.tile_size)
                
                color = self.COLOR_INACTIVE
                if (c, r) == self.start_pos:
                    color = self.COLOR_START
                elif tile_type == self.TILE_SAFE:
                    color = self.COLOR_SAFE
                elif tile_type == self.TILE_FRAGILE:
                    color = self.COLOR_FRAGILE
                elif tile_type == self.TILE_GOAL:
                    # Pulsing effect for the goal
                    pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
                    color = tuple(int(c + (255 - c) * 0.2 * pulse) for c in self.COLOR_GOAL_BASE)
                
                pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # Draw player
        if not self.player_fallen:
            player_x, player_y = self.player_pos
            center_x = self.offset_x + player_x * (self.tile_size + self.tile_spacing) + self.tile_size // 2
            center_y = self.offset_y + player_y * (self.tile_size + self.tile_spacing) + self.tile_size // 2
            radius = self.tile_size // 3

            # Draw a simple "glow"
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius + 3, (*self.COLOR_PLAYER, 80))
            # Draw outline and filled circle for antialiasing effect
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER_OUTLINE)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Draw moves remaining
        moves_text = self.font_small.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Draw score
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)

        # Draw game over message
        if self.game_over:
            is_win = self.win_status == "VICTORY!"
            text_color = self.COLOR_WIN if is_win else self.COLOR_LOSE
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_large.render(self.win_status, True, text_color)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(game_over_text, text_rect)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert "score" in info and "steps" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        assert "score" in info and "steps" in info
        
        print("✓ Implementation validated successfully")

# This block allows running the environment directly for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_window = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Jumper")
    clock = pygame.time.Clock()

    print("\n" + "="*30)
    print("      GRID JUMPER")
    print("="*30)
    print(env.game_description)
    print("\n" + env.user_guide)
    print("Press 'R' to reset the level.")
    print("Press 'Q' or close the window to quit.")
    print("="*30 + "\n")

    while running:
        action = np.array([0, 0, 0])  # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                elif event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
        
        # Only step if a move was made
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Move: {action[0]}, Reward: {reward:.2f}, Terminated: {terminated}, Score: {info['score']:.2f}")

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_window.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    env.close()
    pygame.quit()