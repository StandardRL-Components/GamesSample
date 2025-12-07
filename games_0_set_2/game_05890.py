
# Generated: 2025-08-28T06:24:25.346389
# Source Brief: brief_05890.md
# Brief Index: 5890

        
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
        "Controls: ↑↓←→ to slide selected block. Space/Shift to cycle selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear lines of colored blocks in a strategic grid-based puzzle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    CELL_SIZE = 36
    GRID_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
    GRID_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) // 2
    MAX_STEPS = 1000
    INITIAL_BLOCKS = 15
    CLEAR_ANIMATION_DURATION = 15 # steps

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (50, 50, 60)
    COLOR_WHITE = (255, 255, 255)
    COLOR_HIGHLIGHT = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    
    BLOCK_COLORS = [
        (231, 76, 60),   # Red
        (46, 204, 113),  # Green
        (52, 152, 219),  # Blue
        (241, 196, 15),  # Yellow
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables are initialized in reset()
        self.grid = None
        self.block_positions = None
        self.selected_block_idx = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.clear_animation = None
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.block_positions = []
        
        # Place initial blocks
        empty_cells = [(r, c) for r in range(self.GRID_HEIGHT) for c in range(self.GRID_WIDTH)]
        initial_placements = self.np_random.choice(len(empty_cells), size=self.INITIAL_BLOCKS, replace=False)
        
        for idx in initial_placements:
            r, c = empty_cells[idx]
            color_idx = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)
            self.grid[r, c] = color_idx
        
        self._update_block_list()
        
        self.selected_block_idx = 0 if self.block_positions else -1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.clear_animation = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # Process animations first (clearing blocks from previous step)
        self._update_animations()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1
        
        action_taken = False
        cleared_count = 0

        # Prioritize movement over selection
        if movement > 0 and self.selected_block_idx != -1:
            action_taken = True
            dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
            
            start_r, start_c = self.block_positions[self.selected_block_idx]
            end_r, end_c = self._slide_block(start_r, start_c, dr, dc)
            
            if (start_r, start_c) != (end_r, end_c):
                # sfx: block_land.wav
                color = self.grid[start_r, start_c]
                self.grid[start_r, start_c] = 0
                self.grid[end_r, end_c] = color
                
                cleared_count = self._check_and_start_clears(end_r, end_c, color)
                if cleared_count > 0:
                    # sfx: line_clear.wav
                    pass

                self._spawn_new_block()
                self._update_block_list()
                self.selected_block_idx = 0 if self.block_positions else -1
            else:
                # sfx: block_bump.wav
                pass
        
        elif space_pressed and not action_taken:
            # sfx: select_tick.wav
            if self.block_positions:
                self.selected_block_idx = (self.selected_block_idx + 1) % len(self.block_positions)
        elif shift_pressed and not action_taken:
            # sfx: select_tick.wav
            if self.block_positions:
                self.selected_block_idx = (self.selected_block_idx - 1 + len(self.block_positions)) % len(self.block_positions)

        reward += self._calculate_reward(cleared_count)
        
        terminated = self._check_termination()
        if terminated:
            if self.game_won:
                reward += 100
                # sfx: game_win.wav
            else:
                reward -= 100
                # sfx: game_over.wav

        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _calculate_reward(self, cleared_count):
        reward = 0
        if cleared_count == 3: reward += 1
        elif cleared_count == 4: reward += 2
        elif cleared_count >= 5: reward += 3

        potential_lines = 0
        adjacencies = 0
        
        for r, c in self.block_positions:
            color = self.grid[r, c]
            if c + 1 < self.GRID_WIDTH and self.grid[r, c+1] == color: adjacencies += 1
            if r + 1 < self.GRID_HEIGHT and self.grid[r+1, c] == color: adjacencies += 1
            if c > 0 and c + 1 < self.GRID_WIDTH and self.grid[r, c-1] == 0 and self.grid[r, c+1] == color: potential_lines += 1
            if r > 0 and r + 1 < self.GRID_HEIGHT and self.grid[r-1, c] == 0 and self.grid[r+1, c] == color: potential_lines += 1
                
        reward += 0.1 * potential_lines
        reward -= 0.2 * adjacencies
        
        return reward

    def _check_termination(self):
        if not self.block_positions and self.steps > 0 and not self.clear_animation:
            self.game_over = True
            self.game_won = True
            return True

        if len(self.block_positions) == self.GRID_WIDTH * self.GRID_HEIGHT:
            self.game_over = True
            return True
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True

        return False

    def _update_animations(self):
        if not self.clear_animation: return
            
        new_anim_list = []
        cleared_this_frame = False
        for r, c, timer in self.clear_animation:
            timer -= 1
            if timer <= 0:
                self.grid[r, c] = 0
                cleared_this_frame = True
            else:
                new_anim_list.append((r, c, timer))
        
        self.clear_animation = new_anim_list
        
        if cleared_this_frame:
            self._update_block_list()
            if self.block_positions: self.selected_block_idx = 0
            else: self.selected_block_idx = -1

    def _update_block_list(self):
        self.block_positions = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0:
                    self.block_positions.append((r, c))

    def _slide_block(self, r, c, dr, dc):
        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH): break
            if self.grid[nr, nc] != 0: break
            r, c = nr, nc
        return r, c

    def _check_and_start_clears(self, r, c, color):
        to_clear = set()
        
        # Horizontal
        h_line = [(r, c)]
        for i in range(1, self.GRID_WIDTH):
            if c - i < 0 or self.grid[r, c - i] != color: break
            h_line.append((r, c - i))
        for i in range(1, self.GRID_WIDTH):
            if c + i >= self.GRID_WIDTH or self.grid[r, c + i] != color: break
            h_line.append((r, c + i))
        if len(h_line) >= 3: to_clear.update(h_line)
            
        # Vertical
        v_line = [(r, c)]
        for i in range(1, self.GRID_HEIGHT):
            if r - i < 0 or self.grid[r - i, c] != color: break
            v_line.append((r - i, c))
        for i in range(1, self.GRID_HEIGHT):
            if r + i >= self.GRID_HEIGHT or self.grid[r + i, c] != color: break
            v_line.append((r + i, c))
        if len(v_line) >= 3: to_clear.update(v_line)
        
        for pos_r, pos_c in to_clear:
            self.clear_animation.append((pos_r, pos_c, self.CLEAR_ANIMATION_DURATION))
            
        return len(to_clear)

    def _spawn_new_block(self):
        empty_cells = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                is_animating = any(ar == r and ac == c for ar, ac, _ in self.clear_animation)
                if self.grid[r, c] == 0 and not is_animating:
                    empty_cells.append((r, c))
        
        if not empty_cells: return

        spawn_pos_idx = self.np_random.choice(len(empty_cells))
        r, c = empty_cells[spawn_pos_idx]
        color_idx = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)
        self.grid[r, c] = color_idx

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_AREA_HEIGHT))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, y))

        # Draw blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.grid[r, c]
                if color_idx > 0:
                    color = self.BLOCK_COLORS[color_idx - 1]
                    rect = pygame.Rect(self.GRID_OFFSET_X + c * self.CELL_SIZE + 1, self.GRID_OFFSET_Y + r * self.CELL_SIZE + 1, self.CELL_SIZE - 2, self.CELL_SIZE - 2)
                    pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Draw selected block highlight
        if self.selected_block_idx != -1 and not self.game_over:
            r, c = self.block_positions[self.selected_block_idx]
            rect = pygame.Rect(self.GRID_OFFSET_X + c * self.CELL_SIZE, self.GRID_OFFSET_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, rect, width=3, border_radius=6)
            
        # Draw clear animations
        for r, c, timer in self.clear_animation:
            alpha = 255 * (math.sin((self.CLEAR_ANIMATION_DURATION - timer) / self.CLEAR_ANIMATION_DURATION * math.pi * 2) * 0.5 + 0.5)
            flash_surface = pygame.Surface((self.CELL_SIZE-2, self.CELL_SIZE-2), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (self.GRID_OFFSET_X + c * self.CELL_SIZE + 1, self.GRID_OFFSET_Y + r * self.CELL_SIZE + 1))

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_main.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        if self.game_over:
            msg = "YOU WON!" if self.game_won else "GAME OVER"
            color = (152, 251, 152) if self.game_won else (255, 100, 100)
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            shadow_surf = self.font_large.render(msg, True, (0,0,0))
            self.screen.blit(shadow_surf, (text_rect.x+3, text_rect.y+3))
            self.screen.blit(text_surf, text_rect)
            
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