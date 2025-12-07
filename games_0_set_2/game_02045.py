
# Generated: 2025-08-28T03:31:19.491102
# Source Brief: brief_02045.md
# Brief Index: 2045

        
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

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to reveal a square and Shift to flag/unflag a square."
    )

    # User-facing description of the game
    game_description = (
        "A classic puzzle game. Reveal all the safe squares on the grid while avoiding the hidden mines. "
        "Numbers on revealed squares indicate how many mines are adjacent."
    )

    # Frames advance only on action
    auto_advance = False

    # --- Constants ---
    GRID_SIZE = 5
    NUM_MINES = 5
    MAX_STEPS = 1000

    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID_LINES = (50, 56, 72)
    COLOR_SQUARE_UNREVEALED = (70, 78, 97)
    COLOR_SQUARE_REVEALED = (90, 100, 120)
    COLOR_FLAGGED = (249, 226, 175)
    COLOR_MINE_BG = (235, 108, 111)
    COLOR_MINE_CIRCLE = (194, 69, 72)
    COLOR_CURSOR = (80, 250, 123)
    COLOR_CURSOR_GLOW = (80, 250, 123, 50)
    COLOR_TEXT = (248, 248, 242)
    COLOR_TEXT_GAMEOVER = (255, 85, 85)
    COLOR_TEXT_WIN = (80, 250, 123)

    # Number colors (classic minesweeper style)
    NUMBER_COLORS = {
        1: (139, 233, 253), 2: (80, 250, 123), 3: (255, 85, 85),
        4: (189, 147, 249), 5: (255, 121, 198), 6: (0, 255, 255),
        7: (255, 255, 0), 8: (255, 0, 0)
    }

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
        self.font_large = pygame.font.SysFont("consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 18)
        self.font_numbers = pygame.font.SysFont("consolas", 36, bold=True)

        # Grid layout
        self.cell_size = 70
        self.grid_width = self.GRID_SIZE * self.cell_size
        self.grid_height = self.GRID_SIZE * self.cell_size
        self.grid_origin_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_origin_y = (self.SCREEN_HEIGHT - self.grid_height) // 2 + 20

        # State variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.cursor_pos = np.array([0, 0])
        self.mine_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.revealed_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.flagged_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.number_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])

        # Generate new minefield
        self.mine_grid.fill(False)
        self.revealed_grid.fill(False)
        self.flagged_grid.fill(False)
        self.number_grid.fill(0)

        mine_indices = self.np_random.choice(self.GRID_SIZE * self.GRID_SIZE, self.NUM_MINES, replace=False)
        mine_coords = np.unravel_index(mine_indices, (self.GRID_SIZE, self.GRID_SIZE))
        self.mine_grid[mine_coords] = True

        # Pre-calculate adjacent mine counts
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if not self.mine_grid[r, c]:
                    self.number_grid[r, c] = np.sum(self.mine_grid[
                        max(0, r-1):min(self.GRID_SIZE, r+2),
                        max(0, c-1):min(self.GRID_SIZE, c+2)
                    ])

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        # --- Action Handling ---
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        
        # 1. Move Cursor
        if movement != 0:
            if movement == 1: self.cursor_pos[0] -= 1  # Up
            elif movement == 2: self.cursor_pos[0] += 1  # Down
            elif movement == 3: self.cursor_pos[1] -= 1  # Left
            elif movement == 4: self.cursor_pos[1] += 1  # Right
            # Wrap around grid
            self.cursor_pos[0] %= self.GRID_SIZE
            self.cursor_pos[1] %= self.GRID_SIZE

        r, c = self.cursor_pos

        # 2. Handle Primary Actions (Reveal > Flag)
        if space_press:
            # Action: Reveal square
            if not self.revealed_grid[r, c] and not self.flagged_grid[r, c]:
                if self.mine_grid[r, c]:
                    # Revealed a mine: GAME OVER
                    reward = -100
                    self.game_over = True
                    self.win = False
                    terminated = True
                    # SFX: Explosion
                else:
                    # Revealed a safe square
                    revealed_count = self._flood_fill(r, c)
                    reward += revealed_count  # +1 per revealed safe square
            else:
                reward -= 0.1 # Penalty for invalid action

        elif shift_press:
            # Action: Flag/unflag square
            if not self.revealed_grid[r, c]:
                is_flagging = not self.flagged_grid[r, c]
                self.flagged_grid[r, c] = is_flagging
                reward -= 0.2 # Cost for using flag action

                if is_flagging:
                    # SFX: Flag place
                    if self.mine_grid[r, c]:
                        reward += 5 # Correctly flagged a mine
                    else:
                        reward -= 5 # Incorrectly flagged a safe square
                # else: SFX: Flag remove
            else:
                reward -= 0.1 # Penalty for invalid action

        # --- Update State & Check for Win ---
        self.steps += 1
        self.score += reward
        
        if not terminated:
            safe_squares = self.GRID_SIZE * self.GRID_SIZE - self.NUM_MINES
            if np.sum(self.revealed_grid) == safe_squares:
                self.game_over = True
                self.win = True
                terminated = True
                reward += 100 # Win bonus
                self.score += 100 # Add to final score

        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _flood_fill(self, r, c):
        """Reveals a square and its neighbors if it's a '0'."""
        # SFX: Click reveal
        stack = [(r, c)]
        revealed_count = 0
        while stack:
            row, col = stack.pop()
            if not (0 <= row < self.GRID_SIZE and 0 <= col < self.GRID_SIZE):
                continue
            if self.revealed_grid[row, col] or self.flagged_grid[row, col]:
                continue
            
            self.revealed_grid[row, col] = True
            revealed_count += 1

            if self.number_grid[row, col] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        stack.append((row + dr, col + dc))
        return revealed_count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_cursor()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_grid(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(
                    self.grid_origin_x + c * self.cell_size,
                    self.grid_origin_y + r * self.cell_size,
                    self.cell_size, self.cell_size
                )

                # Determine cell state and color
                if self.revealed_grid[r, c]:
                    pygame.draw.rect(self.screen, self.COLOR_SQUARE_REVEALED, cell_rect)
                    num = self.number_grid[r, c]
                    if num > 0:
                        self._draw_text(
                            str(num), self.font_numbers, self.NUMBER_COLORS[num],
                            cell_rect.centerx, cell_rect.centery
                        )
                elif self.flagged_grid[r, c]:
                    pygame.draw.rect(self.screen, self.COLOR_FLAGGED, cell_rect)
                    self._draw_flag(cell_rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_SQUARE_UNREVEALED, cell_rect)

                # On game over, reveal all mines
                if self.game_over and self.mine_grid[r, c]:
                    pygame.draw.rect(self.screen, self.COLOR_MINE_BG, cell_rect)
                    pygame.gfxdraw.filled_circle(
                        self.screen, cell_rect.centerx, cell_rect.centery,
                        int(self.cell_size * 0.25), self.COLOR_MINE_CIRCLE
                    )
                    pygame.gfxdraw.aacircle(
                        self.screen, cell_rect.centerx, cell_rect.centery,
                        int(self.cell_size * 0.25), self.COLOR_MINE_CIRCLE
                    )

                # Draw grid lines over cells
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, cell_rect, 1)

    def _render_cursor(self):
        r, c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_origin_x + c * self.cell_size,
            self.grid_origin_y + r * self.cell_size,
            self.cell_size, self.cell_size
        )
        
        # Glow effect
        glow_rect = cursor_rect.inflate(8, 8)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_CURSOR_GLOW, s.get_rect(), border_radius=5)
        self.screen.blit(s, glow_rect.topleft)

        # Main cursor outline
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=3)

    def _render_ui(self):
        # Score display
        score_text = f"Score: {int(self.score)}"
        self._draw_text(score_text, self.font_medium, self.COLOR_TEXT, 10, 10, align="topleft")
        
        # Steps display
        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        self._draw_text(steps_text, self.font_medium, self.COLOR_TEXT, self.SCREEN_WIDTH - 10, 10, align="topright")

        # Game Over / Win message
        if self.game_over:
            if self.win:
                self._draw_text("YOU WIN!", self.font_large, self.COLOR_TEXT_WIN,
                                self.SCREEN_WIDTH // 2, self.grid_origin_y - 30)
            else:
                self._draw_text("GAME OVER", self.font_large, self.COLOR_TEXT_GAMEOVER,
                                self.SCREEN_WIDTH // 2, self.grid_origin_y - 30)

    def _draw_flag(self, rect):
        # Pole
        pole_start = (rect.centerx - 10, rect.centery + 20)
        pole_end = (rect.centerx - 10, rect.centery - 20)
        pygame.draw.line(self.screen, self.COLOR_BG, pole_start, pole_end, 3)
        # Flag triangle
        flag_points = [
            (rect.centerx - 10, rect.centery - 20),
            (rect.centerx + 15, rect.centery - 10),
            (rect.centerx - 10, rect.centery)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_MINE_BG)
        pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_MINE_BG)

    def _draw_text(self, text, font, color, x, y, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = (x, y)
        elif align == "topleft":
            text_rect.topleft = (x, y)
        elif align == "topright":
            text_rect.topright = (x, y)
        self.screen.blit(text_surface, text_rect)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Minesweeper Gym Environment")
    
    running = True
    while running:
        # --- Human Input to Action Conversion ---
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    done = False
                
                # Only step if an action was taken
                if movement or space or shift:
                    action = [movement, space, shift]
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
                    
                    if terminated:
                        print("--- GAME OVER ---")
                        print(f"Final Score: {info['score']}")
                        
        # --- Rendering ---
        # The observation is already the rendered screen
        # We just need to convert it back to a Pygame surface to display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    env.close()