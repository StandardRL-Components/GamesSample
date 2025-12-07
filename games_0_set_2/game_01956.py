
# Generated: 2025-08-27T18:48:04.433171
# Source Brief: brief_01956.md
# Brief Index: 1956

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    """
    A procedurally generated Mine Sweeper clone where an RL agent learns to 
    navigate a minefield. The agent controls a cursor to reveal cells or 
    place flags, with the goal of revealing all non-mine cells.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to reveal a cell. Shift to place/remove a flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic puzzle game. Use logic to clear a grid of hidden mines. "
        "Numbers on revealed cells indicate how many mines are adjacent."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    NUM_MINES = 15
    MAX_STEPS = 1000

    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID_LINES = (60, 65, 80)
    COLOR_UNREVEALED = (80, 88, 107)
    COLOR_REVEALED_SAFE = (110, 118, 137)
    COLOR_FLAG = (249, 226, 175)
    COLOR_FLAG_POLE = (190, 170, 130)
    COLOR_MINE_BG = (224, 108, 117)
    COLOR_MINE_BODY = (40, 44, 52)
    COLOR_CURSOR = (97, 175, 239, 150) # RGBA for transparency
    COLOR_TEXT_UI = (220, 220, 220)
    COLOR_NUMBERS = {
        1: (97, 175, 239),
        2: (152, 195, 121),
        3: (224, 108, 117),
        4: (198, 120, 221),
        5: (209, 154, 102),
        6: (86, 182, 194),
        7: (171, 178, 191),
        8: (40, 44, 52),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_main = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_grid = pygame.font.SysFont("consolas", 20, bold=True)
        
        # Game state variables are initialized in reset()
        self.grid_state = None
        self.revealed_mask = None
        self.flag_mask = None
        self.cursor_pos = None
        self.flags_remaining = None
        self.game_over = None
        self.win = None
        self.score = None
        self.steps = None

        # Pre-calculate grid rendering values
        self.cell_size = 32
        self.grid_render_width = self.GRID_WIDTH * self.cell_size
        self.grid_render_height = self.GRID_HEIGHT * self.cell_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_render_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_render_height) // 2

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.flags_remaining = self.NUM_MINES
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]

        self._generate_minefield()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle cursor movement
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dx + self.GRID_WIDTH) % self.GRID_WIDTH
            self.cursor_pos[1] = (self.cursor_pos[1] + dy + self.GRID_HEIGHT) % self.GRID_HEIGHT

        # 2. Handle actions (Reveal > Flag)
        cx, cy = self.cursor_pos
        
        if space_pressed:
            # Reveal action
            if not self.flag_mask[cy, cx] and not self.revealed_mask[cy, cx]:
                reward += self._reveal_cell(cx, cy)
        elif shift_pressed:
            # Flag action
            if not self.revealed_mask[cy, cx]:
                reward += self._toggle_flag(cx, cy)

        # 3. Check for win/loss conditions
        if not self.game_over:
            revealed_count = np.sum(self.revealed_mask)
            if revealed_count == (self.GRID_WIDTH * self.GRID_HEIGHT) - self.NUM_MINES:
                self.win = True
                self.game_over = True
                reward += 100.0
                # Auto-flag remaining mines for visual satisfaction
                self.flag_mask = self.grid_state == -1
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        self.steps += 1
        self.score += reward
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_minefield(self):
        self.grid_state = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int8)
        self.revealed_mask = np.zeros_like(self.grid_state, dtype=bool)
        self.flag_mask = np.zeros_like(self.grid_state, dtype=bool)

        # Place mines
        mine_indices = self.np_random.choice(self.GRID_WIDTH * self.GRID_HEIGHT, self.NUM_MINES, replace=False)
        mine_coords = np.unravel_index(mine_indices, (self.GRID_HEIGHT, self.GRID_WIDTH))
        self.grid_state[mine_coords] = -1  # -1 represents a mine

        # Calculate numbers
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid_state[y, x] == -1:
                    continue
                mine_count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid_state[ny, nx] == -1:
                            mine_count += 1
                self.grid_state[y, x] = mine_count

    def _reveal_cell(self, x, y):
        """Reveals a cell and handles the consequences. Returns reward."""
        if self.grid_state[y, x] == -1:
            # Game over
            self.game_over = True
            # For final render, reveal all mines
            self.revealed_mask[self.grid_state == -1] = True
            return -100.0
        
        if self.grid_state[y, x] == 0:
            # Flood fill for empty cells
            return self._flood_fill(x, y)
        else:
            # Reveal a single numbered cell
            self.revealed_mask[y, x] = True
            return 1.0

    def _flood_fill(self, start_x, start_y):
        """Iterative flood fill. Returns reward for all revealed cells."""
        reward = 0
        q = deque([(start_x, start_y)])
        
        while q:
            x, y = q.popleft()
            
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                continue
            if self.revealed_mask[y, x]:
                continue
            
            self.revealed_mask[y, x] = True
            reward += 1.0
            
            # If we revealed a flag, return it to the pool
            if self.flag_mask[y, x]:
                self.flag_mask[y, x] = False
                self.flags_remaining += 1

            if self.grid_state[y, x] == 0:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        q.append((x + dx, y + dy))
        return reward

    def _toggle_flag(self, x, y):
        """Places or removes a flag. Returns reward/penalty."""
        reward = 0
        if not self.flag_mask[y, x]:
            if self.flags_remaining > 0:
                self.flag_mask[y, x] = True
                self.flags_remaining -= 1
                reward += -0.1
                if self.grid_state[y, x] == -1:
                    reward += 5.0 # Correctly flagged a mine
        else:
            self.flag_mask[y, x] = False
            self.flags_remaining += 1
            reward -= -0.1 # Undo penalty
            if self.grid_state[y, x] == -1:
                reward -= 5.0 # Undo bonus
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw cells
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                if self.revealed_mask[y, x]:
                    cell_value = self.grid_state[y, x]
                    if cell_value == -1:
                        # Revealed mine
                        pygame.draw.rect(self.screen, self.COLOR_MINE_BG, rect)
                        cx, cy = rect.center
                        pygame.gfxdraw.filled_circle(self.screen, cx, cy, self.cell_size // 4, self.COLOR_MINE_BODY)
                    else:
                        # Revealed safe cell
                        pygame.draw.rect(self.screen, self.COLOR_REVEALED_SAFE, rect)
                        if cell_value > 0:
                            num_surf = self.font_grid.render(str(cell_value), True, self.COLOR_NUMBERS[cell_value])
                            num_rect = num_surf.get_rect(center=rect.center)
                            self.screen.blit(num_surf, num_rect)
                elif self.flag_mask[y, x]:
                    # Flagged cell
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)
                    cx, cy = rect.center
                    flag_points = [
                        (cx - self.cell_size // 6, cy - self.cell_size // 4),
                        (cx + self.cell_size // 3, cy),
                        (cx - self.cell_size // 6, cy + self.cell_size // 4)
                    ]
                    pygame.draw.line(self.screen, self.COLOR_FLAG_POLE, (cx - self.cell_size // 6, cy - self.cell_size // 3), (cx - self.cell_size // 6, cy + self.cell_size // 3), 2)
                    pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_FLAG)
                    pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FLAG)

                else:
                    # Unrevealed cell
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)

        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            start_pos = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y + self.grid_render_height)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos)
        for i in range(self.GRID_HEIGHT + 1):
            start_pos = (self.grid_offset_x, self.grid_offset_y + i * self.cell_size)
            end_pos = (self.grid_offset_x + self.grid_render_width, self.grid_offset_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.cell_size,
            self.grid_offset_y + self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4)

    def _render_ui(self):
        # Flags remaining
        flags_text = f"FLAGS: {self.flags_remaining}"
        flags_surf = self.font_main.render(flags_text, True, self.COLOR_TEXT_UI)
        self.screen.blit(flags_surf, (20, 20))

        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT_UI)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_surf, score_rect)

        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER"
            if self.win:
                msg = "YOU WIN!"
            
            msg_surf = self.font_main.render(msg, True, self.COLOR_TEXT_UI)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(msg_surf, msg_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "flags_remaining": self.flags_remaining,
            "cursor_pos": self.cursor_pos,
            "win": self.win,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Minesweeper Gym Environment")
    
    terminated = False
    running = True
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
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
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    terminated = False

                # Since auto_advance is False, we only step when a key is pressed
                if not terminated:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
        
        # Draw the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate

    env.close()