
# Generated: 2025-08-28T01:05:57.138773
# Source Brief: brief_04004.md
# Brief Index: 4004

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to reveal a square. Shift to place/remove a flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic puzzle game. Use logic to clear the board without detonating any hidden mines."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = (9, 9)
        self.NUM_MINES = 10
        self.MAX_STEPS = 1000

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
        
        # Fonts
        try:
            self.font_main = pygame.font.Font(pygame.font.match_font('consolas,dejavusansmono'), 24)
            self.font_grid = pygame.font.Font(pygame.font.match_font('consolas,dejavusansmono'), 20)
            self.font_ui_header = pygame.font.Font(pygame.font.match_font('impact,arialblack'), 28)
            self.font_ui_value = pygame.font.Font(pygame.font.match_font('consolas,dejavusansmono'), 28)
        except:
            self.font_main = pygame.font.SysFont("monospace", 24)
            self.font_grid = pygame.font.SysFont("monospace", 20)
            self.font_ui_header = pygame.font.SysFont("monospace", 28)
            self.font_ui_value = pygame.font.SysFont("monospace", 28)

        # Colors
        self.COLOR_BG = (40, 42, 54)
        self.COLOR_GRID_LINES = (68, 71, 90)
        self.COLOR_UNREVEALED = (98, 114, 164)
        self.COLOR_REVEALED = (248, 248, 242)
        self.COLOR_CURSOR = (80, 250, 123)
        self.COLOR_FLAG = (255, 184, 108)
        self.COLOR_MINE = (255, 85, 85)
        self.COLOR_TEXT = (248, 248, 242)
        self.COLOR_HEADER = (189, 147, 249)

        self.NUMBER_COLORS = {
            1: (98, 114, 164), 2: (80, 250, 123), 3: (255, 85, 85),
            4: (139, 233, 253), 5: (255, 121, 198), 6: (255, 184, 108),
            7: (68, 71, 90), 8: (248, 248, 242)
        }
        
        # Grid rendering properties
        self.CELL_SIZE = 32
        self.GRID_WIDTH = self.GRID_SIZE[0] * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE[1] * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) / 2 - 100
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) / 2
        
        # Initialize state variables
        self.mines = np.zeros(self.GRID_SIZE, dtype=bool)
        self.revealed = np.zeros(self.GRID_SIZE, dtype=bool)
        self.flags = np.zeros(self.GRID_SIZE, dtype=bool)
        self.numbers = np.zeros(self.GRID_SIZE, dtype=int)
        self.cursor_pos = [0, 0]
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        self._create_minefield()
        
        return self._get_observation(), self._get_info()
    
    def _create_minefield(self):
        self.mines.fill(False)
        self.revealed.fill(False)
        self.flags.fill(False)
        self.numbers.fill(0)

        mine_indices = self.np_random.choice(
            self.GRID_SIZE[0] * self.GRID_SIZE[1], self.NUM_MINES, replace=False
        )
        self.mines.flat[mine_indices] = True
        
        for r in range(self.GRID_SIZE[1]):
            for c in range(self.GRID_SIZE[0]):
                if self.mines[r, c]:
                    self.numbers[r, c] = -1
                    continue
                
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.GRID_SIZE[1] and 0 <= nc < self.GRID_SIZE[0] and self.mines[nr, nc]:
                            count += 1
                self.numbers[r, c] = count

    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle cursor movement
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_SIZE[0]
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_SIZE[1]

        cx, cy = self.cursor_pos

        # 2. Handle flagging (Shift)
        if shift_pressed and not self.revealed[cy, cx]:
            is_placing_flag = not self.flags[cy, cx]
            self.flags[cy, cx] = not self.flags[cy, cx]
            
            if is_placing_flag:
                # sfx: flag_place
                reward -= 0.1
                if self.mines[cy, cx]:
                    reward += 10.0 # Correctly flagged a mine
                else:
                    reward -= 5.0 # Incorrectly flagged a safe square
            else:
                # sfx: flag_remove
                pass # No reward change for removing a flag

        # 3. Handle revealing (Space)
        elif space_pressed and not self.revealed[cy, cx] and not self.flags[cy, cx]:
            if self.mines[cy, cx]:
                # sfx: explosion
                self.game_over = True
                reward = -100.0
                self.revealed[cy, cx] = True # Reveal the stepped-on mine
            else:
                # sfx: reveal
                reward += self._reveal_squares(cx, cy)
        
        # 4. Check for win condition
        num_safe_squares = self.GRID_SIZE[0] * self.GRID_SIZE[1] - self.NUM_MINES
        if not self.game_over and np.sum(self.revealed) == num_safe_squares:
            # sfx: win_fanfare
            self.game_over = True
            reward += 100.0
            # Auto-flag remaining mines for visual satisfaction
            self.flags = np.logical_or(self.flags, self.mines)

        self.score += reward
        self.steps += 1
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reveal_squares(self, x, y):
        # Iterative flood fill to avoid recursion depth issues
        q = deque([(x, y)])
        revealed_count = 0
        
        while q:
            cx, cy = q.popleft()
            
            if not (0 <= cx < self.GRID_SIZE[0] and 0 <= cy < self.GRID_SIZE[1]):
                continue
            if self.revealed[cy, cx] or self.flags[cy, cx]:
                continue
            
            self.revealed[cy, cx] = True
            revealed_count += 1
            
            if self.numbers[cy, cx] == 0:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        q.append((cx + dx, cy + dy))
        
        return float(revealed_count) # +1 reward per revealed square

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        for r in range(self.GRID_SIZE[1]):
            for c in range(self.GRID_SIZE[0]):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                
                if self.revealed[r, c]:
                    pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                    if self.mines[r, c] and self.game_over:
                        # Draw mine
                        pygame.draw.circle(self.screen, self.COLOR_MINE, rect.center, self.CELL_SIZE * 0.35)
                        pygame.draw.circle(self.screen, self.COLOR_BG, rect.center, self.CELL_SIZE * 0.35, 2)
                    elif self.numbers[r, c] > 0:
                        # Draw number
                        num_text = str(self.numbers[r, c])
                        color = self.NUMBER_COLORS.get(self.numbers[r, c], self.COLOR_TEXT)
                        text_surf = self.font_grid.render(num_text, True, color)
                        text_rect = text_surf.get_rect(center=rect.center)
                        self.screen.blit(text_surf, text_rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)
                    if self.flags[r, c]:
                        # Draw flag
                        points = [
                            (rect.centerx, rect.top + 5),
                            (rect.right - 5, rect.centery - 2),
                            (rect.centerx, rect.centery + 2)
                        ]
                        pygame.draw.polygon(self.screen, self.COLOR_FLAG, points)
                        pygame.draw.line(self.screen, self.COLOR_FLAG, (rect.centerx, rect.top + 5), (rect.centerx, rect.bottom - 5), 2)

        # Draw grid lines
        for i in range(self.GRID_SIZE[0] + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT))
        for i in range(self.GRID_SIZE[1] + 1):
            y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH, y))

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + cx * self.CELL_SIZE,
            self.GRID_OFFSET_Y + cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        ui_x = self.GRID_OFFSET_X + self.GRID_WIDTH + 40
        ui_y = self.GRID_OFFSET_Y

        # Score
        score_header = self.font_ui_header.render("SCORE", True, self.COLOR_HEADER)
        self.screen.blit(score_header, (ui_x, ui_y))
        
        score_val = self.font_ui_value.render(f"{self.score:,.0f}", True, self.COLOR_TEXT)
        self.screen.blit(score_val, (ui_x, ui_y + 30))

        # Steps
        steps_header = self.font_ui_header.render("STEPS", True, self.COLOR_HEADER)
        self.screen.blit(steps_header, (ui_x, ui_y + 90))

        steps_val = self.font_ui_value.render(f"{self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_val, (ui_x, ui_y + 120))
        
        # Mines remaining
        flags_placed = np.sum(self.flags)
        mines_header = self.font_ui_header.render("MINES", True, self.COLOR_HEADER)
        self.screen.blit(mines_header, (ui_x, ui_y + 180))

        mines_val = self.font_ui_value.render(f"{self.NUM_MINES - flags_placed}", True, self.COLOR_TEXT)
        self.screen.blit(mines_val, (ui_x, ui_y + 210))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "mines_remaining": self.NUM_MINES - np.sum(self.flags)
        }

    def close(self):
        pygame.font.quit()
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
        
        # Test mine count
        self.reset()
        assert np.sum(self.mines) == self.NUM_MINES

        # Test loss condition
        self.reset()
        mine_pos = np.argwhere(self.mines)[0]
        self.cursor_pos = [mine_pos[1], mine_pos[0]] # [c, r]
        action = [0, 1, 0] # no-op move, press space, release shift
        obs, reward, term, trunc, info = self.step(action)
        assert reward == -100
        assert term == True

        # Test win condition
        self.reset()
        self.revealed = np.logical_not(self.mines)
        self.revealed[0, 0] = False # Leave one square unrevealed
        self.cursor_pos = [0, 0]
        action = [0, 1, 0] # Reveal last square
        obs, reward, term, trunc, info = self.step(action)
        assert reward > 100 # +1 for reveal, +100 for win
        assert term == True

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # You will need to install pygame: pip install pygame
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Set up the display window
    pygame.display.set_caption("Minesweeper Gym Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    action = [0, 0, 0] # No-op, no space, no shift
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        
        # Movement (mutually exclusive)
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # Buttons
        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_pressed, shift_pressed]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render the observation to the display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we need a delay for human playability
        pygame.time.wait(100) # 10 FPS for human input

    print(f"Game Over! Final Score: {info['score']}")
    env.close()