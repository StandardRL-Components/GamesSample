
# Generated: 2025-08-28T06:58:09.934287
# Source Brief: brief_03067.md
# Brief Index: 3067

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to clear a highlighted group of blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the grid by matching adjacent colored blocks. Win by clearing the board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 12, 8
        self.BLOCK_SIZE = 40
        self.UI_HEIGHT = 60
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_COLS * self.BLOCK_SIZE) // 2
        self.GRID_Y_OFFSET = self.UI_HEIGHT + (self.SCREEN_HEIGHT - self.UI_HEIGHT - self.GRID_ROWS * self.BLOCK_SIZE) // 2

        # Colors
        self.COLORS = [
            (255, 87, 87),   # Red
            (87, 187, 255),  # Blue
            (87, 255, 150),  # Green
            (255, 255, 87),  # Yellow
            (200, 100, 255)  # Purple
        ]
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID_LINE = (40, 45, 50)
        self.COLOR_UI_BG = (30, 35, 40)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 255)

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.grid = []
        self.cursor_x = 0
        self.cursor_y = 0
        self.score = 0
        self.moves_left = 0
        self.particles = []
        self.game_over = False
        self.win_message = ""
        self.steps = 0
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = 60
        self.game_over = False
        self.win_message = ""
        self.particles = []
        self.cursor_x = self.GRID_COLS // 2
        self.cursor_y = self.GRID_ROWS // 2
        
        self._generate_board()
        
        return self._get_observation(), self._get_info()

    def _generate_board(self):
        # The board generation ensures at least one valid move is always available on reset.
        while True:
            self.grid = [
                [self.np_random.integers(0, len(self.COLORS)) for _ in range(self.GRID_COLS)]
                for _ in range(self.GRID_ROWS)
            ]
            if self._is_any_move_possible():
                break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1
        
        self.steps += 1
        reward = 0
        terminated = False

        self._update_particles()

        if shift_pressed:
            self.game_over = True
            terminated = True
            reward = -100  # Penalty for giving up
            self.win_message = "GAME RESET"
            return self._get_observation(), reward, terminated, False, self._get_info()

        if movement == 1: self.cursor_y = max(0, self.cursor_y - 1)
        elif movement == 2: self.cursor_y = min(self.GRID_ROWS - 1, self.cursor_y + 1)
        elif movement == 3: self.cursor_x = max(0, self.cursor_x - 1)
        elif movement == 4: self.cursor_x = min(self.GRID_COLS - 1, self.cursor_x + 1)

        if space_pressed:
            group = self._find_adjacent_group(self.cursor_x, self.cursor_y)
            if len(group) >= 2:
                self.moves_left -= 1
                num_cleared = len(group)
                self.score += num_cleared
                reward += num_cleared  # +1 reward per block

                # Clear blocks and create particles
                color_index = self.grid[self.cursor_y][self.cursor_x]
                for r, c in group:
                    self._create_particles(c, r, self.COLORS[color_index])
                    self.grid[r][c] = -1  # Mark for removal
                
                # NOTE: To make the "clear the board" win condition possible,
                # new blocks do not spawn from the top, differing from the brief's suggestion.
                # This aligns with the "Collapse" genre and the specified win condition.
                self._handle_gravity()

                if self._is_board_clear():
                    self.game_over = True
                    terminated = True
                    reward += 100  # Win bonus
                    self.win_message = "YOU WIN!"
                elif not self._is_any_move_possible():
                    self.game_over = True
                    terminated = True
                    reward = -100  # Loss penalty for soft-lock
                    self.win_message = "NO MOVES LEFT"

        if self.moves_left <= 0 and not terminated:
            self.game_over = True
            terminated = True
            reward = -100 # Loss penalty
            self.win_message = "OUT OF MOVES"

        if self.steps >= 1000 and not terminated:
            self.game_over = True
            terminated = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def _find_adjacent_group(self, x, y):
        if not (0 <= x < self.GRID_COLS and 0 <= y < self.GRID_ROWS): return []
        target_color = self.grid[y][x]
        if target_color == -1: return []

        q = [(y, x)]
        visited, group = set(q), []
        while q:
            r, c = q.pop(0)
            group.append((r, c))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and \
                   (nr, nc) not in visited and self.grid[nr][nc] == target_color:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return group

    def _handle_gravity(self):
        for c in range(self.GRID_COLS):
            write_ptr = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] != -1:
                    if write_ptr != r:
                        self.grid[write_ptr][c] = self.grid[r][c]
                        self.grid[r][c] = -1
                    write_ptr -= 1

    def _is_board_clear(self):
        return all(self.grid[r][c] == -1 for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS))

    def _is_any_move_possible(self):
        visited = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] != -1 and (r, c) not in visited:
                    group = self._find_adjacent_group(c, r)
                    if len(group) >= 2:
                        return True
                    visited.update(group)
        return False

    def _render_game(self):
        highlight_group = self._find_adjacent_group(self.cursor_x, self.cursor_y)
        is_valid_highlight = len(highlight_group) >= 2

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_index = self.grid[r][c]
                rect = pygame.Rect(self.GRID_X_OFFSET + c * self.BLOCK_SIZE, self.GRID_Y_OFFSET + r * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                
                if color_index != -1:
                    base_color = self.COLORS[color_index]
                    border_color = tuple(max(0, val - 40) for val in base_color)
                    
                    if is_valid_highlight and (r, c) in highlight_group:
                        highlight_color = tuple(min(255, val + 60) for val in base_color)
                        pygame.draw.rect(self.screen, highlight_color, rect)
                        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3)
                    else:
                        pygame.draw.rect(self.screen, base_color, rect)
                        pygame.draw.rect(self.screen, border_color, rect, 2)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)

        for p in self.particles:
            p_x, p_y, p_vx, p_vy, p_life, p_color = p
            alpha = max(0, min(255, int(255 * (p_life / 30.0))))
            radius = int(p_life / 6)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p_x), int(p_y), radius, (*p_color, alpha))

        cursor_rect = pygame.Rect(self.GRID_X_OFFSET + self.cursor_x * self.BLOCK_SIZE, self.GRID_Y_OFFSET + self.cursor_y * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=4)

    def _render_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (0, self.UI_HEIGHT - 1), (self.SCREEN_WIDTH, self.UI_HEIGHT - 1), 2)
        
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, self.UI_HEIGHT / 2 - score_text.get_height() / 2))
        
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        text_rect = moves_text.get_rect(right=self.SCREEN_WIDTH - 20, centery=self.UI_HEIGHT / 2)
        self.screen.blit(moves_text, text_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        game_over_text = self.font_main.render(self.win_message, True, (255, 255, 100))
        text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(game_over_text, text_rect)

    def _create_particles(self, c, r, color):
        center_x = self.GRID_X_OFFSET + c * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        center_y = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.uniform(20, 40)
            self.particles.append([center_x, center_y, vx, vy, life, color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 1]
        for p in self.particles:
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[4] -= 1     # life -= 1
            p[2] *= 0.98  # Drag
            p[3] *= 0.98