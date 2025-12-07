
# Generated: 2025-08-28T03:36:28.188776
# Source Brief: brief_04968.md
# Brief Index: 4968

        
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
    user_guide = "Controls: Arrows to move cursor. Space to start dragging. Arrows to extend drag. Shift to confirm connection."

    # Must be a short, user-facing description of the game:
    game_description = "Connect adjacent blocks of the same color to clear them. Clear 10 connections to win before the board fills up!"

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 10
    WIN_CONDITION_LINES = 10
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (50, 60, 80)
    COLOR_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_DRAG_LINE = (255, 255, 255, 200)

    BLOCK_COLORS = {
        1: (255, 80, 80),   # Red
        2: (80, 255, 80),   # Green
        3: (80, 120, 255),  # Blue
        4: (255, 255, 80),  # Yellow
        5: (200, 80, 255),  # Purple
        6: (100, 110, 120)  # Gray (Obstacle)
    }
    BLOCK_BORDERS = {k: tuple(max(0, c - 50) for c in v) for k, v in BLOCK_COLORS.items()}
    EMPTY_ID = 0
    GRAY_ID = 6
    NUM_COLORS = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Grid rendering properties
        self.grid_area_height = self.SCREEN_HEIGHT - 20
        self.cell_size = self.grid_area_height // self.GRID_HEIGHT
        self.grid_area_width = self.cell_size * self.GRID_WIDTH
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_area_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_area_height) // 2
        
        # Initialize state variables that are not reset every episode
        self.grid = None
        self.cursor_pos = None
        self.is_dragging = None
        self.drag_path = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.steps = None
        self.score = None
        self.lines_cleared = None
        self.game_over = None
        self.gray_block_prob = None
        self.rng = None
        self.last_cleared_info = None
        
        # Initialize state variables
        self.reset()
        
        # Self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.gray_block_prob = 0.05
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.is_dragging = False
        self.drag_path = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_cleared_info = {"coords": [], "timer": 0}

        self._initialize_board()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _initialize_board(self):
        self.grid = self.rng.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        
        num_gray = int(self.GRID_WIDTH * self.GRID_HEIGHT * self.gray_block_prob)
        flat_indices = self.rng.choice(self.GRID_WIDTH * self.GRID_HEIGHT, num_gray, replace=False)
        row_indices, col_indices = np.unravel_index(flat_indices, (self.GRID_HEIGHT, self.GRID_WIDTH))
        self.grid[row_indices, col_indices] = self.GRAY_ID
        
        while not self._has_valid_moves():
            self._initialize_board()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        invalid_action = False

        if self.is_dragging:
            if shift_pressed:
                # sound: release_drag_attempt.wav
                r, is_invalid = self._end_drag()
                reward += r
                if is_invalid:
                    invalid_action = True
            elif movement > 0:
                self._extend_drag(movement)
            elif space_pressed:
                invalid_action = True
        else:
            if space_pressed:
                # sound: start_drag.wav
                self._start_drag()
            elif movement > 0:
                self._move_cursor(movement)
            elif shift_pressed:
                invalid_action = True
        
        if invalid_action:
            # sound: invalid_action.wav
            reward -= 0.01

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        terminated = self._check_termination()
        if terminated:
            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                # sound: win_game.wav
                reward += 100
            else:
                # sound: lose_game.wav
                reward -= 100
            self.game_over = True
            
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

    def _start_drag(self):
        x, y = self.cursor_pos
        if self.grid[y, x] not in [self.GRAY_ID, self.EMPTY_ID]:
            self.is_dragging = True
            self.drag_path = [(x, y)]

    def _extend_drag(self, movement):
        if not self.drag_path: return
        last_x, last_y = self.drag_path[-1]
        next_pos = [last_x, last_y]

        if movement == 1: next_pos[1] -= 1
        elif movement == 2: next_pos[1] += 1
        elif movement == 3: next_pos[0] -= 1
        elif movement == 4: next_pos[0] += 1
        
        nx, ny = next_pos
        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
            if (nx, ny) not in self.drag_path:
                drag_color = self.grid[self.drag_path[0][1], self.drag_path[0][0]]
                if self.grid[ny, nx] == drag_color:
                    # sound: extend_drag_success.wav
                    self.drag_path.append((nx, ny))
            elif len(self.drag_path) > 1 and (nx, ny) == self.drag_path[-2]:
                self.drag_path.pop()

    def _end_drag(self):
        reward = 0
        is_invalid = True

        if len(self.drag_path) >= 2:
            is_invalid = False
            # sound: clear_blocks.wav
            num_cleared = len(self.drag_path)
            reward += num_cleared * 0.1 + 1.0
            self.score += num_cleared
            self.lines_cleared += 1

            self.last_cleared_info = {"coords": list(self.drag_path), "timer": 1}

            for x, y in self.drag_path:
                self.grid[y, x] = self.EMPTY_ID
            
            self._apply_gravity()
            self._spawn_new_blocks()
            self.gray_block_prob = min(0.5, 0.05 + 0.01 * (self.lines_cleared // 2))

        self.is_dragging = False
        self.drag_path = []
        return reward, is_invalid

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            col = self.grid[:, x]
            empty_cells = np.where(col == self.EMPTY_ID)[0]
            non_empty_cells = np.where(col != self.EMPTY_ID)[0]
            
            if len(empty_cells) > 0 and len(non_empty_cells) > 0:
                new_col = np.concatenate((np.full(len(empty_cells), self.EMPTY_ID), col[non_empty_cells]))
                self.grid[:, x] = new_col

    def _spawn_new_blocks(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] == self.EMPTY_ID:
                    if self.rng.random() < self.gray_block_prob:
                        self.grid[y, x] = self.GRAY_ID
                    else:
                        self.grid[y, x] = self.rng.integers(1, self.NUM_COLORS + 1)

    def _has_valid_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color = self.grid[y, x]
                if color not in [self.EMPTY_ID, self.GRAY_ID]:
                    if x + 1 < self.GRID_WIDTH and self.grid[y, x + 1] == color: return True
                    if y + 1 < self.GRID_HEIGHT and self.grid[y + 1, x] == color: return True
        return False

    def _check_termination(self):
        if self.lines_cleared >= self.WIN_CONDITION_LINES: return True
        if self.steps >= self.MAX_STEPS: return True
        if not self._has_valid_moves(): return True
        return False
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                block_id = self.grid[y, x]
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                if block_id != self.EMPTY_ID:
                    border_rect = rect.copy()
                    inner_rect = rect.inflate(-self.cell_size * 0.2, -self.cell_size * 0.2)
                    pygame.draw.rect(self.screen, self.BLOCK_BORDERS[block_id], border_rect, border_radius=4)
                    pygame.draw.rect(self.screen, self.BLOCK_COLORS[block_id], inner_rect, border_radius=3)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        if self.last_cleared_info["timer"] > 0:
            for x, y in self.last_cleared_info["coords"]:
                center_x = self.grid_offset_x + int((x + 0.5) * self.cell_size)
                center_y = self.grid_offset_y + int((y + 0.5) * self.cell_size)
                radius = int(self.cell_size * 0.6)
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, (255, 255, 255, 100))
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, (255, 255, 255, 150))
            self.last_cleared_info["timer"] -= 1

        if self.is_dragging and self.drag_path:
            points = [(self.grid_offset_x + int((x + 0.5) * self.cell_size), self.grid_offset_y + int((y + 0.5) * self.cell_size)) for x, y in self.drag_path]
            for px, py in points:
                pygame.gfxdraw.filled_circle(self.screen, px, py, 5, self.COLOR_DRAG_LINE)
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.COLOR_DRAG_LINE, False, points, 3)

        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.cell_size,
            self.grid_offset_y + self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        color = self.COLOR_CURSOR[:3] if not self.is_dragging else (255,100,0)
        alpha = self.COLOR_CURSOR[3]
        pygame.draw.rect(s, (*color, alpha), s.get_rect(), border_radius=5)
        pygame.draw.rect(s, color, s.get_rect(), width=3, border_radius=5)
        self.screen.blit(s, cursor_rect.topleft)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        lines_text = self.font_main.render(f"CLEARS: {self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (self.SCREEN_WIDTH - lines_text.get_width() - 20, 20))

        if self.game_over:
            outcome_str = "YOU WIN!" if self.lines_cleared >= self.WIN_CONDITION_LINES else "GAME OVER"
            outcome_text = self.font_main.render(outcome_str, True, (255, 255, 100))
            text_rect = outcome_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            bg_rect = text_rect.inflate(20, 20)
            pygame.draw.rect(self.screen, self.COLOR_BG, bg_rect, border_radius=10)
            pygame.draw.rect(self.screen, (255, 255, 100), bg_rect, width=2, border_radius=10)
            self.screen.blit(outcome_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Connect Blocks")
    
    action = env.action_space.sample()
    action.fill(0)

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            current_action = np.array([movement, space_held, shift_held])
            
            # Since auto_advance is False, we only step when an action is taken
            # For human play, we step on any key press/release
            if not np.array_equal(current_action, action) or movement > 0:
                obs, reward, terminated, truncated, info = env.step(current_action)

            action = current_action

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    env.close()