
# Generated: 2025-08-28T02:25:50.726666
# Source Brief: brief_04442.md
# Brief Index: 4442

        
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

    # Corrected user guide and description based on the implemented game mechanics.
    user_guide = (
        "Use arrow keys to select a block and swap it with an adjacent one. "
        "Match 3 or more of the same color to clear them!"
    )

    game_description = (
        "A fast-paced color-matching puzzle. Swap blocks to create lines of 3 or more. "
        "Trigger chain reactions for huge scores and clear 5 lines to win!"
    )

    # The game is turn-based, so state advances only on action.
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 12
    GRID_HEIGHT = 8
    CELL_SIZE = 40
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2 + 20

    MAX_STEPS = 900  # Corresponds to 30s at 30fps if auto_advance=True
    WIN_CONDITION_LINES = 5

    COLOR_BG = (20, 25, 40)
    COLOR_GRID_LINES = (40, 50, 80)
    COLOR_TEXT = (220, 220, 240)
    COLOR_FLASH = (255, 255, 255)

    PALETTE = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_game_over = pygame.font.Font(None, 52)
        
        self.grid = None
        self.player_pos = None
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.last_cleared_cells = set()
        self.win_status = ""

        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.last_cleared_cells = set()
        self.win_status = ""

        self._create_initial_grid()
        self.player_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        
        return self._get_observation(), self._get_info()

    def _create_initial_grid(self):
        self.grid = self.np_random.integers(0, len(self.PALETTE), size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while True:
            matches = self._find_and_count_matches()
            if not matches['cells']:
                break
            for r, c in matches['cells']:
                self.grid[r, c] = self.np_random.integers(0, len(self.PALETTE))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held and shift_held are ignored as per the design brief
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        reward = 0
        self.last_cleared_cells.clear()

        moved = False
        dr, dc = 0, 0
        if movement != 0:
            dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
            r, c = self.player_pos
            nr, nc = r + dr, c + dc

            if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH:
                # Pre-move shaping reward
                current_color = self.grid[r, c]
                for dr_check, dc_check in [(-1,0), (1,0), (0,-1), (0,1)]:
                    check_r, check_c = nr + dr_check, nc + dc_check
                    # Don't check the cell we are swapping from
                    if (check_r, check_c) == (r, c): continue
                    if 0 <= check_r < self.GRID_HEIGHT and 0 <= check_c < self.GRID_WIDTH:
                        if self.grid[check_r, check_c] == current_color:
                            reward += 0.1

                # Swap blocks
                self.grid[r, c], self.grid[nr, nc] = self.grid[nr, nc], self.grid[r, c]
                self.player_pos = [nr, nc]
                moved = True
            else:
                # Invalid move (out of bounds)
                reward -= 0.01

        if not moved: # No-op or invalid move
             reward -= 0.01
        else: # A valid swap was made, check for matches
            total_lines_in_move = 0
            total_score_from_clear = 0
            is_first_pass = True
            
            while True:
                matches = self._find_and_count_matches()
                if not matches['cells']:
                    if is_first_pass: # If the swap resulted in no matches, it's a wasted move, swap back
                        r, c = self.player_pos
                        pr, pc = r - dr, c - dc # Previous position
                        self.grid[r, c], self.grid[pr, pc] = self.grid[pr, pc], self.grid[r, c]
                        self.player_pos = [pr, pc]
                    break
                
                is_first_pass = False
                num_lines = matches['count']
                total_lines_in_move += num_lines

                # Event-based reward for clearing lines
                line_reward = {1: 1, 2: 2, 3: 4, 4: 8}.get(num_lines, 16) if num_lines > 0 else 0
                total_score_from_clear += line_reward
                
                self.last_cleared_cells.update(matches['cells'])
                self._apply_gravity_and_refill(matches['cells'])
                # sfx: block clear sound

            self.score += total_score_from_clear
            self.lines_cleared += total_lines_in_move
            reward += total_score_from_clear
            # sfx: chain reaction sound if total_lines_in_move > 1
        
        self.steps += 1
        terminated = False
        
        if self.lines_cleared >= self.WIN_CONDITION_LINES:
            reward += 100
            terminated = True
            self.win_status = "YOU WIN!"
            # sfx: win fanfare
        elif self.steps >= self.MAX_STEPS:
            reward -= 10
            terminated = True
            self.win_status = "TIME'S UP!"
            # sfx: loss sound

        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _find_and_count_matches(self):
        matches = {'cells': set(), 'count': 0}
        
        # Horizontal
        for r in range(self.GRID_HEIGHT):
            c = 0
            while c < self.GRID_WIDTH:
                color = self.grid[r, c]
                if color == -1: c+=1; continue
                length = 1
                while c + length < self.GRID_WIDTH and self.grid[r, c + length] == color:
                    length += 1
                if length >= 3:
                    matches['count'] += 1
                    for i in range(length):
                        matches['cells'].add((r, c + i))
                c += length

        # Vertical
        for c in range(self.GRID_WIDTH):
            r = 0
            while r < self.GRID_HEIGHT:
                color = self.grid[r, c]
                if color == -1: r+=1; continue
                length = 1
                while r + length < self.GRID_HEIGHT and self.grid[r + length, c] == color:
                    length += 1
                if length >= 3:
                    matches['count'] += 1
                    for i in range(length):
                        matches['cells'].add((r + i, c))
                r += length
                
        return matches
    
    def _apply_gravity_and_refill(self, cleared_cells):
        for r, c in cleared_cells:
            self.grid[r, c] = -1 # Mark as empty

        for c in range(self.GRID_WIDTH):
            write_row = self.GRID_HEIGHT - 1
            for read_row in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[read_row, c] != -1:
                    if write_row != read_row:
                        self.grid[write_row, c] = self.grid[read_row, c]
                    write_row -= 1
            
            for r in range(write_row, -1, -1):
                self.grid[r, c] = self.np_random.integers(0, len(self.PALETTE))
                # sfx: block fall

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.GRID_X_OFFSET + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE))

        # Draw blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_index = self.grid[r, c]
                if color_index != -1:
                    color = self.PALETTE[color_index]
                    rect = pygame.Rect(
                        self.GRID_X_OFFSET + c * self.CELL_SIZE + 1,
                        self.GRID_Y_OFFSET + r * self.CELL_SIZE + 1,
                        self.CELL_SIZE - 2,
                        self.CELL_SIZE - 2
                    )
                    pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Draw flash effect for cleared cells
        if self.last_cleared_cells:
            for r, c in self.last_cleared_cells:
                rect = pygame.Rect(
                    self.GRID_X_OFFSET + c * self.CELL_SIZE,
                    self.GRID_Y_OFFSET + r * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE
                )
                flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                flash_surface.fill((*self.COLOR_FLASH, 180))
                self.screen.blit(flash_surface, rect.topleft)

        # Draw player cursor
        pr, pc = self.player_pos
        player_rect = pygame.Rect(
            self.GRID_X_OFFSET + pc * self.CELL_SIZE,
            self.GRID_Y_OFFSET + pr * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        # Pulsing glow effect
        glow_alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        glow_color = (255, 255, 255)
        
        pygame.gfxdraw.rectangle(self.screen, player_rect.inflate(1,1), (*glow_color, int(glow_alpha)))
        pygame.gfxdraw.rectangle(self.screen, player_rect.inflate(3,3), (*glow_color, int(glow_alpha/2)))
        pygame.gfxdraw.rectangle(self.screen, player_rect.inflate(5,5), (*glow_color, int(glow_alpha/4)))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lines cleared
        lines_text = self.font_ui.render(f"Lines: {self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (10, 35))

        # Timer bar
        timer_width = 200
        timer_height = 20
        time_left_ratio = max(0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
        
        bar_x = self.SCREEN_WIDTH - timer_width - 10
        bar_y = 10
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, (bar_x, bar_y, timer_width, timer_height))
        
        fill_color = (80, 255, 80) if time_left_ratio > 0.3 else (255, 80, 80)
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, timer_width * time_left_ratio, timer_height))
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_game_over.render(self.win_status, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Color Match Puzzle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if env.game_over:
                    if event.key == pygame.K_r: # Reset game
                        obs, info = env.reset()
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                else:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
        
        # Only step if an action was taken, because auto_advance is False
        if action[0] != 0 and not env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Lines: {info['lines_cleared']}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Press 'R' to restart or 'Q' to quit.")
        
        # Always get the latest observation to render
        obs = env._get_observation()
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    pygame.quit()