
# Generated: 2025-08-27T17:28:12.010526
# Source Brief: brief_01540.md
# Brief Index: 1540

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Arrow keys to move cursor. Space to reveal a tile. Shift to flag a tile."

    # Must be a short, user-facing description of the game:
    game_description = "A classic mine-sweeping puzzle game. Reveal all safe tiles without hitting a mine."

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Constants
        self.GRID_SIZE = 9
        self.NUM_MINES = 10
        self.MAX_STEPS = 1000
        self.WIDTH, self.HEIGHT = 640, 400

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_grid = pygame.font.SysFont("Arial", 24, bold=True)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Visual Style
        self.COLOR_BG = (40, 42, 54)
        self.COLOR_GRID = (70, 72, 84)
        self.COLOR_HIDDEN = (98, 114, 164)
        self.COLOR_REVEALED = (68, 71, 90)
        self.COLOR_FLAG = (255, 121, 198)
        self.COLOR_MINE = (255, 85, 85)
        self.COLOR_CURSOR = (80, 250, 123, 150)
        self.COLOR_CURSOR_BORDER = (80, 250, 123)
        self.COLOR_TEXT = (248, 248, 242)
        self.NUMBER_COLORS = {
            1: (139, 233, 253), 2: (80, 250, 123),  3: (255, 184, 108),
            4: (189, 147, 249), 5: (255, 85, 85),   6: (139, 233, 253),
            7: (241, 250, 140), 8: (255, 121, 198)
        }

        # Game state (initialized in reset)
        self.grid_state = None
        self.mine_map = None
        self.adjacency_map = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.flags_placed = 0
        self.revealed_count = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.tile_size = 0
        self.start_x = 0
        self.start_y = 0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.flags_placed = 0
        self.revealed_count = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        # Grid state: 0=hidden, 1=revealed, 2=flagged
        self.grid_state = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]

        self._generate_minefield()
        self._calculate_adjacency()

        return self._get_observation(), self._get_info()

    def _generate_minefield(self):
        self.mine_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        flat_indices = np.arange(self.GRID_SIZE * self.GRID_SIZE)
        mine_indices = self.np_random.choice(flat_indices, self.NUM_MINES, replace=False)
        rows, cols = np.unravel_index(mine_indices, (self.GRID_SIZE, self.GRID_SIZE))
        self.mine_map[rows, cols] = True

    def _calculate_adjacency(self):
        self.adjacency_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.mine_map[r, c]:
                    continue
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and self.mine_map[nr, nc]:
                            self.adjacency_map[r, c] += 1

    def step(self, action):
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        movement, space_held_int, shift_held_int = action
        space_held = space_held_int == 1
        shift_held = shift_held_int == 1

        # Handle Movement
        if movement == 1: self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE
        elif movement == 3: self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE

        # Handle Actions (on press)
        if space_held and not self.prev_space_held:
            # sound: click
            reward += self._reveal_tile(self.cursor_pos[0], self.cursor_pos[1])
        if shift_held and not self.prev_shift_held:
            # sound: flag_place
            reward += self._toggle_flag(self.cursor_pos[0], self.cursor_pos[1])

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        self.steps += 1
        
        # Check for win condition after action
        total_safe_tiles = self.GRID_SIZE * self.GRID_SIZE - self.NUM_MINES
        if not self.win and self.revealed_count >= total_safe_tiles:
            self.win = True

        terminated = self._check_termination()

        # Apply terminal rewards
        if terminated:
            if self.win:
                reward += 100
            elif self.game_over:
                reward -= 100
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _reveal_tile(self, r, c):
        if self.grid_state[r, c] != 0: return 0.0
        if self.mine_map[r, c]:
            self.game_over = True
            # sound: explosion
            for r_idx in range(self.GRID_SIZE):
                for c_idx in range(self.GRID_SIZE):
                    if self.mine_map[r_idx, c_idx]: self.grid_state[r_idx, c_idx] = 1
            return 0.0

        initial_revealed = self.revealed_count
        self._flood_fill(r, c)
        newly_revealed = self.revealed_count - initial_revealed
        return float(newly_revealed)

    def _flood_fill(self, r, c):
        stack = [(r, c)]
        visited = set()
        while stack:
            row, col = stack.pop()
            if not (0 <= row < self.GRID_SIZE and 0 <= col < self.GRID_SIZE): continue
            if (row, col) in visited: continue
            if self.grid_state[row, col] != 0: continue
            
            visited.add((row, col))
            self.grid_state[row, col] = 1
            self.revealed_count += 1
            
            if self.adjacency_map[row, col] > 0: continue
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    stack.append((row + dr, col + dc))

    def _toggle_flag(self, r, c):
        state = self.grid_state[r, c]
        if state == 1: return 0.0
        if state == 0:
            self.grid_state[r, c] = 2
            self.flags_placed += 1
            return -0.1
        elif state == 2:
            self.grid_state[r, c] = 0
            self.flags_placed -= 1
            return 0.1
        return 0.0

    def _check_termination(self):
        return self.game_over or self.win or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {
            "score": self.score, "steps": self.steps,
            "cursor_pos": self.cursor_pos, "flags_left": self.NUM_MINES - self.flags_placed,
            "win": self.win
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self.tile_size = 36
        grid_width = self.GRID_SIZE * self.tile_size
        grid_height = self.GRID_SIZE * self.tile_size
        self.start_x = (self.WIDTH - grid_width) // 2
        self.start_y = (self.HEIGHT - grid_height) // 2 + 20

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(self.start_x + c * self.tile_size, self.start_y + r * self.tile_size, self.tile_size, self.tile_size)
                state = self.grid_state[r, c]
                
                if state == 1: # Revealed
                    pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                    if self.mine_map[r, c]:
                        pygame.draw.circle(self.screen, self.COLOR_MINE, rect.center, self.tile_size // 3)
                    else:
                        num = self.adjacency_map[r, c]
                        if num > 0:
                            num_text = self.font_grid.render(str(num), True, self.NUMBER_COLORS.get(num, self.COLOR_TEXT))
                            text_rect = num_text.get_rect(center=rect.center)
                            self.screen.blit(num_text, text_rect)
                elif state == 2: # Flagged
                    pygame.draw.rect(self.screen, self.COLOR_HIDDEN, rect)
                    p1 = (rect.centerx, rect.top + 5)
                    p2 = (rect.left + 5, rect.centery)
                    p3 = (rect.centerx, rect.centery)
                    pygame.draw.polygon(self.screen, self.COLOR_FLAG, [p1, p2, p3])
                    pygame.draw.line(self.screen, self.COLOR_FLAG, (rect.centerx, rect.top + 5), (rect.centerx, rect.bottom - 5), 2)
                else: # Hidden
                    pygame.draw.rect(self.screen, self.COLOR_HIDDEN, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
        
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(self.start_x + cursor_c * self.tile_size, self.start_y + cursor_r * self.tile_size, self.tile_size, self.tile_size)
        s = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, cursor_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR_BORDER, cursor_rect, 2)

    def _render_ui(self):
        flags_left = self.NUM_MINES - self.flags_placed
        flags_text = self.font_small.render(f"MINES LEFT: {flags_left}", True, self.COLOR_TEXT)
        self.screen.blit(flags_text, (20, 15))

        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(score_text, score_rect)
        
        msg = ""
        color = self.COLOR_TEXT
        if self.game_over:
            msg = "GAME OVER"
            color = self.COLOR_MINE
        elif self.win:
            msg = "GRID CLEARED!"
            color = self.COLOR_CURSOR_BORDER
        
        if msg:
            msg_text = self.font_large.render(msg, True, color)
            msg_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.start_y / 2))
            
            bg_rect = msg_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 220))
            self.screen.blit(s, bg_rect.topleft)
            pygame.draw.rect(s, color, s.get_rect(), 2)
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(msg_text, msg_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")