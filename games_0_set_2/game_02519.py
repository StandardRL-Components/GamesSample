import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a block. "
        "Select a second matching block to clear the pair. Press Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect matching colored blocks with a clear path of up to two turns. "
        "Clear 15 pairs before you run out of moves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame setup ---
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_msg = pygame.font.Font(None, 52)


        # --- Game Constants ---
        self.GRID_WIDTH, self.GRID_HEIGHT = 14, 10
        self.NUM_PAIRS = 15
        self.NUM_COLORS = 5
        self.MAX_MOVES = 25 
        self.MAX_STEPS = 1000

        # --- Visuals ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECT = (100, 255, 255)
        self.COLOR_CONNECT = (255, 255, 100)
        self.PALETTE = [
            (50, 200, 255),  # Cyan
            (255, 100, 200), # Pink
            (100, 255, 100), # Green
            (255, 150, 50),  # Orange
            (200, 100, 255), # Purple
        ]
        
        self.CELL_SIZE = 36
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2 + 20
        self.BLOCK_SIZE = int(self.CELL_SIZE * 0.8)
        self.BLOCK_OFFSET = (self.CELL_SIZE - self.BLOCK_SIZE) // 2

        # --- Game State ---
        self.np_random = None
        self.grid = None
        self.cursor_pos = [0, 0]
        self.selected_block = None  # Stores (pos, color_id)
        self.pairs_connected = 0
        self.moves_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.connection_effect = None # (p1_pixels, p2_pixels, lifetime)
        
        # The environment state must be initialized via reset() before any
        # methods that depend on it (like _get_observation) can be called.
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.pairs_connected = 0
        self.moves_remaining = self.MAX_MOVES
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_block = None
        self.connection_effect = None
        
        self._generate_grid()

        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        # We need NUM_PAIRS * 2 blocks total.
        # With NUM_COLORS, we have (NUM_PAIRS / NUM_COLORS) pairs of each color.
        pairs_per_color = self.NUM_PAIRS // self.NUM_COLORS
        blocks_to_place = []
        for i in range(self.NUM_COLORS):
            blocks_to_place.extend([i + 1] * pairs_per_color * 2)

        # Get all possible grid positions
        all_pos = [(r, c) for r in range(self.GRID_HEIGHT) for c in range(self.GRID_WIDTH)]
        self.np_random.shuffle(all_pos)
        
        # Place blocks
        for i, block_id in enumerate(blocks_to_place):
            r, c = all_pos[i]
            self.grid[r, c] = block_id

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.connection_effect = None # Clear transient effect
        reward = 0
        self.steps += 1

        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

        # --- 1. Handle cursor movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        if movement == 2: self.cursor_pos[1] += 1  # Down
        if movement == 3: self.cursor_pos[0] -= 1  # Left
        if movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- 2. Handle deselect (Shift) ---
        if shift_press and self.selected_block:
            self.selected_block = None

        # --- 3. Handle selection/match (Space) ---
        if space_press:
            cursor_tuple = tuple(self.cursor_pos)
            block_id = self.grid[cursor_tuple[1], cursor_tuple[0]]

            if block_id == 0: # Selected an empty space
                pass # No action, no penalty
            elif not self.selected_block: # First selection
                self.selected_block = (cursor_tuple, block_id)
            elif self.selected_block[0] == cursor_tuple: # Selected same block twice
                self.selected_block = None
            else: # Second selection - attempt to match
                self.moves_remaining -= 1
                
                p1_pos, p1_id = self.selected_block
                p2_pos, p2_id = cursor_tuple, block_id

                if p1_id == p2_id and self._is_valid_path(p1_pos, p2_pos):
                    # --- SUCCESSFUL MATCH ---
                    reward = 1.0
                    self.pairs_connected += 1
                    
                    # Milestone rewards
                    if self.pairs_connected in [5, 10]:
                        reward += {5: 5, 10: 10}[self.pairs_connected]

                    # Update grid
                    self.grid[p1_pos[1], p1_pos[0]] = 0
                    self.grid[p2_pos[1], p2_pos[0]] = 0
                    
                    # Add visual effect
                    p1_pixels = self._grid_to_pixels(p1_pos[0], p1_pos[1])
                    p2_pixels = self._grid_to_pixels(p2_pos[0], p2_pos[1])
                    self.connection_effect = (p1_pixels, p2_pixels, 15) # lifetime in frames
                    
                    self.selected_block = None
                else:
                    # --- FAILED MATCH ---
                    reward = -0.5
                    self.selected_block = None
        
        self.score += reward
        
        # --- 4. Check for termination ---
        terminated = self.pairs_connected >= self.NUM_PAIRS or self.moves_remaining <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.pairs_connected >= self.NUM_PAIRS:
                terminal_reward = 100
            else: # Ran out of moves or steps
                terminal_reward = -100
            reward += terminal_reward
            self.score += terminal_reward

        truncated = self.steps >= self.MAX_STEPS
        terminated = terminated or truncated

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _is_valid_path(self, p1, p2):
        # A path can have 0, 1, or 2 turns (up to 3 straight segments).
        # We temporarily treat p1 and p2 as empty for pathfinding.
        original_values = self.grid[p1[1], p1[0]], self.grid[p2[1], p2[0]]
        self.grid[p1[1], p1[0]], self.grid[p2[1], p2[0]] = 0, 0

        path_found = (self._check_line(p1, p2) or
                      self._check_one_corner(p1, p2) or
                      self._check_two_corners(p1, p2))
        
        # Restore grid
        self.grid[p1[1], p1[0]], self.grid[p2[1], p2[0]] = original_values
        return path_found

    def _check_line(self, p1, p2):
        if p1[0] != p2[0] and p1[1] != p2[1]: return False
        if p1[0] == p2[0]: # Vertical
            for y in range(min(p1[1], p2[1]) + 1, max(p1[1], p2[1])):
                if self.grid[y, p1[0]] != 0: return False
        else: # Horizontal
            for x in range(min(p1[0], p2[0]) + 1, max(p1[0], p2[0])):
                if self.grid[p1[1], x] != 0: return False
        return True

    def _check_one_corner(self, p1, p2):
        # Two potential corners
        c1, c2 = (p1[0], p2[1]), (p2[0], p1[1])
        if self.grid[c1[1], c1[0]] == 0:
            if self._check_line(p1, c1) and self._check_line(c1, p2): return True
        if self.grid[c2[1], c2[0]] == 0:
            if self._check_line(p1, c2) and self._check_line(c2, p2): return True
        return False
        
    def _check_two_corners(self, p1, p2):
        # Extend from p1 horizontally
        for x in range(self.GRID_WIDTH):
            if self.grid[p1[1], x] == 0 or (x, p1[1]) == p1:
                if self._check_line(p1, (x, p1[1])) and self._check_one_corner((x, p1[1]), p2):
                    return True
            else:
                continue
        # Extend from p1 vertically
        for y in range(self.GRID_HEIGHT):
            if self.grid[y, p1[0]] == 0 or (p1[0], y) == p1:
                if self._check_line(p1, (p1[0], y)) and self._check_one_corner((p1[0], y), p2):
                    return True
            else:
                continue
        return False

    def _grid_to_pixels(self, x, y):
        px = self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
        return px, py

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE))

        # Draw blocks
        if self.grid is not None:
            for r in range(self.GRID_HEIGHT):
                for c in range(self.GRID_WIDTH):
                    block_id = self.grid[r, c]
                    if block_id > 0:
                        rect = pygame.Rect(
                            self.GRID_OFFSET_X + c * self.CELL_SIZE + self.BLOCK_OFFSET,
                            self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.BLOCK_OFFSET,
                            self.BLOCK_SIZE, self.BLOCK_SIZE
                        )
                        color = self.PALETTE[block_id - 1]
                        pygame.draw.rect(self.screen, color, rect, border_radius=5)
                        pygame.draw.rect(self.screen, tuple(min(255, v+30) for v in color), rect, width=2, border_radius=5)
        
        # Draw selected block highlight
        if self.selected_block:
            pos, _ = self.selected_block
            rect = pygame.Rect(
                self.GRID_OFFSET_X + pos[0] * self.CELL_SIZE,
                self.GRID_OFFSET_Y + pos[1] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, width=3, border_radius=7)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=2, border_radius=7)

        # Draw connection effect
        if self.connection_effect:
            p1, p2, lifetime = self.connection_effect
            width = int(8 * math.sin(math.pi * (1 - lifetime / 15.0)))
            color = self.COLOR_CONNECT
            if width > 1:
                pygame.draw.line(self.screen, color, p1, p2, width)
            self.connection_effect = (p1, p2, lifetime - 1)
            if self.connection_effect[2] <= 0:
                self.connection_effect = None

        # Draw game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.pairs_connected >= self.NUM_PAIRS else "GAME OVER"
            text_surf = self.font_msg.render(msg, True, self.COLOR_CONNECT)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)


    def _render_ui(self):
        pairs_text = f"Pairs: {self.pairs_connected}/{self.NUM_PAIRS}"
        moves_text = f"Moves: {self.moves_remaining}"
        score_text = f"Score: {int(self.score)}"
        
        texts = [pairs_text, moves_text, score_text]
        for i, text in enumerate(texts):
            surf = self.font_ui.render(text, True, (200, 200, 220))
            rect = surf.get_rect(center=(self.WIDTH * (i + 1) // 4, 25))
            self.screen.blit(surf, rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "pairs_connected": self.pairs_connected,
            "moves_remaining": self.moves_remaining,
        }

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
        print("✓ Running implementation validation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")