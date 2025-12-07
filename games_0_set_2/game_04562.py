
# Generated: 2025-08-28T02:47:02.236926
# Source Brief: brief_04562.md
# Brief Index: 4562

        
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
    A block-matching puzzle game where the player clears clusters of
    colored blocks to reach a target score within a move limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a block cluster."
    )

    # Short, user-facing description of the game
    game_description = (
        "Match 3 or more adjacent blocks of the same color to score points. "
        "Create larger matches for bonus points! Reach 1000 points in 100 moves to win."
    )

    # The game is turn-based, so it advances only on action.
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 6, 6
    MAX_MOVES = 100
    TARGET_SCORE = 1000
    MIN_CLUSTER_SIZE = 3

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (50, 60, 70)
    COLOR_CURSOR = (255, 255, 255, 100)
    COLOR_TEXT = (220, 220, 230)
    BLOCK_COLORS = [
        (239, 71, 111),  # Red
        (255, 209, 102), # Yellow
        (6, 214, 160),   # Green
        (17, 138, 178),  # Blue
        (111, 71, 239),  # Purple
        (255, 140, 70)   # Orange
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("sans-serif", 36, bold=True)
        self.font_ui = pygame.font.SysFont("sans-serif", 24)
        
        # --- Game Layout ---
        self.block_size = 50
        self.grid_width = self.GRID_COLS * self.block_size
        self.grid_height = self.GRID_ROWS * self.block_size
        self.grid_offset_x = (self.WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.HEIGHT - self.grid_height) // 2 + 20

        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.win = False
        self.last_action = None
        self.effects = [] # For single-frame visual effects
        
        self.reset()
        
        # This check is mandatory
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.effects = []
        self.last_action = self.action_space.sample() * 0 # Initialize with no-op

        # Generate initial grid
        self.grid = self.np_random.integers(0, len(self.BLOCK_COLORS), size=(self.GRID_ROWS, self.GRID_COLS))

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        # Clear single-frame effects from the previous step
        self.effects.clear()

        movement = action[0]
        space_pressed = action[1] == 1

        # --- Action Processing ---
        # Only process one action type per step to keep it turn-based
        if movement > 0 and self.last_action[0] != movement:
            # Move cursor
            if movement == 1:  # Up
                self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2:  # Down
                self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
            elif movement == 3:  # Left
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4:  # Right
                self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
        
        elif space_pressed and self.last_action[1] == 0:
            # "Click" action on space press
            self.moves_left -= 1
            
            cluster = self._find_cluster(self.cursor_pos[0], self.cursor_pos[1])
            
            if len(cluster) >= self.MIN_CLUSTER_SIZE:
                # Valid move
                # Sfx: Positive chime
                num_cleared = len(cluster)
                # Score is the reward for RL agent
                score_gain = 10 * (num_cleared ** 2)
                reward = score_gain
                self.score += score_gain
                
                self._remove_and_refill(cluster)
                # Add visual effects for cleared blocks
                for r, c in cluster:
                    self.effects.append(('flash', c, r, 20))
            else:
                # Invalid move
                # Sfx: Negative buzz
                reward = -0.1
                self.effects.append(('buzz', self.cursor_pos[0], self.cursor_pos[1], 15))

        # --- Check Termination ---
        if self.score >= self.TARGET_SCORE:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100  # Win bonus
        elif self.moves_left <= 0:
            terminated = True
            self.game_over = True
            self.win = False
            reward = -10  # Lose penalty

        self.last_action = action

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _find_cluster(self, start_c, start_r):
        """Finds all contiguous blocks of the same color using Breadth-First Search."""
        if not (0 <= start_r < self.GRID_ROWS and 0 <= start_c < self.GRID_COLS):
            return []
            
        target_color = self.grid[start_r, start_c]
        if target_color == -1: # Empty
            return []

        q = [(start_r, start_c)]
        visited = set(q)
        cluster = []

        while q:
            r, c = q.pop(0)
            cluster.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and
                        (nr, nc) not in visited and self.grid[nr, nc] == target_color):
                    visited.add((nr, nc))
                    q.append((nr, nc))
        
        return cluster

    def _remove_and_refill(self, cluster):
        """Removes blocks in the cluster, applies gravity, and fills empty spaces."""
        affected_cols = set()
        for r, c in cluster:
            self.grid[r, c] = -1  # Mark as empty
            affected_cols.add(c)

        for c in affected_cols:
            col_vals = self.grid[:, c]
            
            # Use a list to filter out -1 and then pad
            new_col = [val for val in col_vals if val != -1]
            
            # Generate new blocks to fill the top
            num_new_blocks = self.GRID_ROWS - len(new_col)
            new_blocks = self.np_random.integers(0, len(self.BLOCK_COLORS), size=num_new_blocks).tolist()
            
            # Combine new blocks with existing ones
            final_col = np.array(new_blocks + new_col)
            self.grid[:, c] = final_col

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "steps": self.MAX_MOVES - self.moves_left,
        }

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y, self.grid_width, self.grid_height)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)
        
        # Draw blocks
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_index = self.grid[r, c]
                if color_index != -1:
                    block_color = self.BLOCK_COLORS[color_index]
                    rect = pygame.Rect(
                        self.grid_offset_x + c * self.block_size + 2,
                        self.grid_offset_y + r * self.block_size + 2,
                        self.block_size - 4,
                        self.block_size - 4
                    )
                    pygame.draw.rect(self.screen, block_color, rect, border_radius=8)
                    # Inner highlight for 3D effect
                    highlight_color = tuple(min(255, val + 40) for val in block_color)
                    pygame.draw.rect(self.screen, highlight_color, rect.inflate(-8, -8), border_radius=6)

        # Draw cursor
        cursor_c, cursor_r = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cursor_c * self.block_size,
            self.grid_offset_y + cursor_r * self.block_size,
            self.block_size,
            self.block_size
        )
        pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, 3, border_radius=8)
        
        # Draw visual effects
        for effect in self.effects:
            etype, c, r, size = effect
            center_x = self.grid_offset_x + c * self.block_size + self.block_size // 2
            center_y = self.grid_offset_y + r * self.block_size + self.block_size // 2
            if etype == 'flash':
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, size, (255, 255, 255, 150))
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, size, (255, 255, 255, 200))
            elif etype == 'buzz':
                pygame.draw.line(self.screen, (255, 50, 50), (center_x - size, center_y - size), (center_x + size, center_y + size), 5)
                pygame.draw.line(self.screen, (255, 50, 50), (center_x - size, center_y + size), (center_x + size, center_y - size), 5)

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Moves left display
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 20))

        # Game Over / Win display
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                end_text_str = "YOU WIN!"
                end_color = self.BLOCK_COLORS[2] # Green
            else:
                end_text_str = "GAME OVER"
                end_color = self.BLOCK_COLORS[0] # Red
            
            end_text = self.font_main.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Matcher")
    
    running = True
    terminated = False
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        action = np.array([0, 0, 0]) # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not terminated:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
                    print("--- Game Reset ---")
                
                # Since auto_advance is False, we step on every key press
                obs, reward, terminated, _, info = env.step(action)
                
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")

                if terminated:
                    print("--- GAME OVER ---")
                    if info['score'] >= env.TARGET_SCORE:
                        print("Result: YOU WIN!")
                    else:
                        print("Result: YOU LOSE!")
                    print("Press 'R' to play again.")

        # Draw the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Limit FPS for the interactive loop

    env.close()