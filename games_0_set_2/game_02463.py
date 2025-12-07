
# Generated: 2025-08-28T04:55:08.385246
# Source Brief: brief_02463.md
# Brief Index: 2463

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Use arrows to move the cursor. To swap, select a direction and the 'swap' action (Space) in the same turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent gems to match 3 or more in a row. Create combos and clear the board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.NUM_TILE_TYPES = 6
        self.TILE_SIZE = 40
        self.GRID_X = (self.WIDTH - self.GRID_COLS * self.TILE_SIZE) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_ROWS * self.TILE_SIZE) // 2
        self.MAX_MOVES = 20
        self.MAX_PARTICLES = 200

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.TILE_COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 150, 255),   # Blue
            (255, 255, 80),   # Yellow
            (255, 80, 255),   # Magenta
            (80, 255, 255),   # Cyan
        ]
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.moves_remaining = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.game_won = None
        self.particles = []

        # This will be seeded in reset()
        self.np_random = None
        
        # Initialize state variables
        self.reset()

        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.cursor_pos = [0, 0]
        self.moves_remaining = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.particles.clear()
        
        self._generate_grid()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = self.game_over

        if not terminated:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_held = action[1] == 1  # Boolean
            shift_held = action[2] == 1  # Boolean
            
            is_swap_action = space_held and movement != 0

            if is_swap_action:
                reward, terminated = self._process_swap_action(movement)
            else: # Cursor movement
                self._move_cursor(movement)
        
        # Check for game end conditions if not already triggered by a swap
        if not terminated and self.moves_remaining <= 0:
            terminated = True
            reward += -50.0
        
        if not terminated and not self.game_won and not self._find_possible_moves():
             terminated = True
             reward += -50.0 # No more moves left

        if terminated:
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _process_swap_action(self, movement):
        self.moves_remaining -= 1
        reward = -0.1  # Penalty for using a move

        dir_map = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        dr, dc = dir_map[movement]
        
        r1, c1 = self.cursor_pos
        r2, c2 = r1 + dr, c1 + dc

        if not (0 <= r2 < self.GRID_ROWS and 0 <= c2 < self.GRID_COLS):
            return reward, self.game_over # Invalid swap off-grid

        # Perform the swap
        self._swap_tiles((r1, c1), (r2, c2))

        # Process matches and chain reactions
        total_cleared_this_turn = 0
        is_first_match_in_chain = True
        
        while True:
            matches = self._find_matches()
            if not matches:
                # If no matches on the first try, it was a bad move. Swap back.
                if is_first_match_in_chain:
                    self._swap_tiles((r1, c1), (r2, c2))
                break

            # If we found a match, the move was productive
            if is_first_match_in_chain:
                reward += 0.1 # Cancel the penalty
                is_first_match_in_chain = False

            num_cleared = len(matches)
            total_cleared_this_turn += num_cleared
            
            # Base reward per tile
            reward += num_cleared

            # Bonus for larger matches
            if any(len(list(g)) >= 5 for k, g in self._group_matches(matches)): reward += 10
            elif any(len(list(g)) == 4 for k, g in self._group_matches(matches)): reward += 5
            
            self._clear_tiles(matches)
            # sfx: tile clear sound
            self._drop_tiles()
            self._fill_new_tiles()
            # sfx: tiles fall sound

        self.score += total_cleared_this_turn
        
        # Check for win condition
        if np.all(self.grid == -1):
            self.game_won = True
            reward += 100.0
        
        return reward, self.game_won

    def _move_cursor(self, movement):
        if movement == 0: return
        dir_map = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        dr, dc = dir_map[movement]
        self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dr, 0, self.GRID_ROWS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dc, 0, self.GRID_COLS - 1)
        # sfx: cursor move tick

    def _generate_grid(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_TILE_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))
            if not self._find_matches() and self._find_possible_moves():
                break
    
    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    for i in range(3): matches.add((r, c + i))
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    for i in range(3): matches.add((r + i, c))
        return matches

    def _group_matches(self, matches):
        from itertools import groupby
        # Group by row
        for r, group in groupby(sorted(matches, key=lambda p: (p[0], p[1])), key=lambda p: p[0]):
            yield f"row_{r}", group
        # Group by col
        for c, group in groupby(sorted(matches, key=lambda p: (p[1], p[0])), key=lambda p: p[1]):
            yield f"col_{c}", group

    def _find_possible_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                for dr, dc in [(0, 1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                        self._swap_tiles((r, c), (nr, nc))
                        has_match = len(self._find_matches()) > 0
                        self._swap_tiles((r, c), (nr, nc)) # Swap back
                        if has_match:
                            return True
        return False

    def _swap_tiles(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

    def _clear_tiles(self, matches):
        for r, c in matches:
            if self.grid[r, c] != -1:
                self._create_particles(r, c)
                self.grid[r, c] = -1 # -1 represents an empty space
    
    def _create_particles(self, r, c):
        tile_color = self.TILE_COLORS[self.grid[r,c]]
        center_x = self.GRID_X + c * self.TILE_SIZE + self.TILE_SIZE // 2
        center_y = self.GRID_Y + r * self.TILE_SIZE + self.TILE_SIZE // 2
        for _ in range(10):
            if len(self.particles) < self.MAX_PARTICLES:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel_x = math.cos(angle) * speed
                vel_y = math.sin(angle) * speed
                life = self.np_random.integers(20, 40)
                self.particles.append([center_x, center_y, vel_x, vel_y, life, tile_color])

    def _drop_tiles(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], -1
                    empty_row -= 1

    def _fill_new_tiles(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_COLS * self.TILE_SIZE, self.GRID_ROWS * self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)

        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[1]; p[1] += p[2]; p[4] -= 1; p[2] += 0.1
            radius = max(0, int(p[4] / 8))
            pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), radius)

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_type = self.grid[r, c]
                if tile_type != -1:
                    color = self.TILE_COLORS[tile_type]
                    rect = pygame.Rect(self.GRID_X + c * self.TILE_SIZE, self.GRID_Y + r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                    inner_rect = rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, color, inner_rect, border_radius=5)
                    highlight_color = tuple(min(255, x + 50) for x in color)
                    pygame.gfxdraw.arc(self.screen, inner_rect.centerx, inner_rect.centery, inner_rect.width//2 - 2, 120, 240, highlight_color)

        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_X + cursor_c * self.TILE_SIZE, self.GRID_Y + cursor_r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=3, border_radius=4)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 20))

        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (20, 20))

        if self.game_over:
            msg_text_str = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            msg_text = self.font_msg.render(msg_text_str, True, color)
            msg_rect = msg_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "cursor_pos": list(self.cursor_pos),
        }

    def close(self):
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