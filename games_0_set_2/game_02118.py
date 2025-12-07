
# Generated: 2025-08-27T19:20:16.896263
# Source Brief: brief_02118.md
# Brief Index: 2118

        
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

    user_guide = (
        "Controls: Use arrow keys to slide the selected block. Selection cycles automatically after each move."
    )

    game_description = (
        "Slide colored blocks on a grid to form three target patterns before you run out of moves."
    )

    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 6, 6
    CELL_SIZE = 60
    GRID_X_OFFSET = 30
    GRID_Y_OFFSET = 20
    MAX_MOVES = 25
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (40, 50, 60)
    COLOR_GRID_LINE = (60, 70, 80)
    COLOR_WHITE = (230, 230, 230)
    COLOR_SELECTION = (255, 200, 0)
    COLOR_SUCCESS = (100, 255, 150)
    COLOR_FAILURE = (255, 100, 100)
    
    # --- Block Colors (Bright & Saturated) ---
    COLOR_BLOCK_A = (255, 80, 80)   # Red
    COLOR_BLOCK_B = (80, 255, 80)   # Green
    COLOR_BLOCK_C = (80, 150, 255)  # Blue
    COLOR_BLOCK_D = (255, 220, 80)  # Yellow

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.UI_FONT = pygame.font.SysFont("monospace", 18, bold=True)
        self.TITLE_FONT = pygame.font.SysFont("monospace", 48, bold=True)

        self._define_puzzle()
        
        self.reset()

        self.validate_implementation()
    
    def _define_puzzle(self):
        """Defines the starting block layout and target patterns for the puzzle."""
        self.start_blocks = [
            {'r': 1, 'c': 1, 'color': self.COLOR_BLOCK_A}, # Red
            {'r': 4, 'c': 1, 'color': self.COLOR_BLOCK_B}, # Green
            {'r': 1, 'c': 4, 'color': self.COLOR_BLOCK_C}, # Blue
            {'r': 4, 'c': 4, 'color': self.COLOR_BLOCK_D}, # Yellow
        ]
        
        self.target_patterns = [
            # Pattern 1: Red over Green (2x1 vertical)
            [
                {'r': 0, 'c': 0, 'color': self.COLOR_BLOCK_A},
                {'r': 1, 'c': 0, 'color': self.COLOR_BLOCK_B}
            ],
            # Pattern 2: Blue next to Yellow (1x2 horizontal)
            [
                {'r': 0, 'c': 0, 'color': self.COLOR_BLOCK_C},
                {'r': 0, 'c': 1, 'color': self.COLOR_BLOCK_D}
            ],
            # Pattern 3: Red and Yellow diagonal
            [
                {'r': 0, 'c': 0, 'color': self.COLOR_BLOCK_A},
                {'r': 1, 'c': 1, 'color': self.COLOR_BLOCK_D}
            ]
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.blocks = [b.copy() for b in self.start_blocks]
        self._update_grid()
        
        self.moves_remaining = self.MAX_MOVES
        self.patterns_completed = [False] * len(self.target_patterns)
        self.selected_block_idx = 0
        self.score = 0.0
        self.game_over = False
        self.steps = 0
        
        return self._get_observation(), self._get_info()
    
    def _update_grid(self):
        """Creates a 2D array representation from the list of blocks for fast lookups."""
        self.grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        for i, block in enumerate(self.blocks):
            self.grid[block['r']][block['c']] = i

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
            
        movement = action[0]
        reward = 0.0

        dist_before = self._calculate_total_pattern_distance()

        if movement != 0:  # 0 is no-op
            self.moves_remaining -= 1
            
            block_to_move_idx = self.selected_block_idx
            block = self.blocks[block_to_move_idx]
            original_pos = (block['r'], block['c'])

            dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
            
            r, c = block['r'], block['c']
            final_r, final_c = r, c
            while True:
                next_r, next_c = final_r + dr, final_c + dc
                if not (0 <= next_r < self.GRID_ROWS and 0 <= next_c < self.GRID_COLS):
                    break  # Hit wall
                if self.grid[next_r][next_c] is not None:
                    break  # Hit another block
                final_r, final_c = next_r, next_c

            if (final_r, final_c) != original_pos:
                # Update block position and grid representation
                self.grid[original_pos[0]][original_pos[1]] = None
                self.blocks[block_to_move_idx]['r'] = final_r
                self.blocks[block_to_move_idx]['c'] = final_c
                self.grid[final_r][final_c] = block_to_move_idx
                # sfx: block_slide.wav
            else:
                # sfx: block_bump.wav
                pass
            
            # Cycle selection after every attempted move
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.blocks)
        
        dist_after = self._calculate_total_pattern_distance()
        reward += (dist_before - dist_after) * 0.1
        
        newly_completed_indices = self._check_patterns()
        for _ in newly_completed_indices:
            reward += 5.0
            # sfx: pattern_complete.wav

        terminated = False
        if all(self.patterns_completed):
            reward += 50.0
            terminated = True
            self.game_over = True
            # sfx: win_jingle.wav
        elif self.moves_remaining <= 0:
            reward -= 10.0
            terminated = True
            self.game_over = True
            # sfx: lose_jingle.wav

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_total_pattern_distance(self):
        """Calculates a heuristic distance of current blocks to uncompleted patterns."""
        total_dist = 0
        # Since block colors are unique, we can map them
        block_by_color = {b['color']: b for b in self.blocks}

        for i, pattern in enumerate(self.target_patterns):
            if self.patterns_completed[i]:
                continue
            
            min_pattern_dist = float('inf')
            pattern_height = max(b['r'] for b in pattern) + 1
            pattern_width = max(b['c'] for b in pattern) + 1
            
            for r_offset in range(self.GRID_ROWS - pattern_height + 1):
                for c_offset in range(self.GRID_COLS - pattern_width + 1):
                    current_placement_dist = 0
                    for p_block in pattern:
                        target_r, target_c = r_offset + p_block['r'], c_offset + p_block['c']
                        actual_block = block_by_color[p_block['color']]
                        dist = abs(actual_block['r'] - target_r) + abs(actual_block['c'] - target_c)
                        current_placement_dist += dist
                    min_pattern_dist = min(min_pattern_dist, current_placement_dist)
            
            if min_pattern_dist != float('inf'):
                total_dist += min_pattern_dist
                
        return total_dist

    def _check_patterns(self):
        """Checks for pattern completion and updates state."""
        newly_completed_indices = []
        for i, pattern in enumerate(self.target_patterns):
            if self.patterns_completed[i]:
                continue
                
            pattern_height = max(b['r'] for b in pattern) + 1
            pattern_width = max(b['c'] for b in pattern) + 1
            
            is_complete = False
            for r_offset in range(self.GRID_ROWS - pattern_height + 1):
                for c_offset in range(self.GRID_COLS - pattern_width + 1):
                    match = True
                    for p_block in pattern:
                        grid_r, grid_c = r_offset + p_block['r'], c_offset + p_block['c']
                        block_idx_on_grid = self.grid[grid_r][grid_c]
                        
                        if block_idx_on_grid is None or self.blocks[block_idx_on_grid]['color'] != p_block['color']:
                            match = False
                            break
                    if match:
                        is_complete = True
                        break
                if is_complete:
                    break
            
            if is_complete:
                self.patterns_completed[i] = True
                newly_completed_indices.append(i)
        
        return newly_completed_indices

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_w = self.GRID_COLS * self.CELL_SIZE
        grid_h = self.GRID_ROWS * self.CELL_SIZE
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET, grid_w, grid_h))
        
        for i in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + grid_w, y), 2)
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + grid_h), 2)
            
        for i, block in enumerate(self.blocks):
            x = self.GRID_X_OFFSET + block['c'] * self.CELL_SIZE
            y = self.GRID_Y_OFFSET + block['r'] * self.CELL_SIZE
            rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
            
            # Draw block with inset effect for depth
            inner_rect = rect.inflate(-10, -10)
            pygame.draw.rect(self.screen, block['color'], inner_rect)
            pygame.draw.rect(self.screen, self.COLOR_WHITE, inner_rect, 1)

        if not self.game_over and self.selected_block_idx < len(self.blocks):
            selected_block = self.blocks[self.selected_block_idx]
            x = self.GRID_X_OFFSET + selected_block['c'] * self.CELL_SIZE
            y = self.GRID_Y_OFFSET + selected_block['r'] * self.CELL_SIZE
            rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
            
            # Pulsing selection highlight
            pulse = abs(math.sin(self.steps * 0.3))
            alpha = 100 + 155 * pulse
            radius = int(self.CELL_SIZE * 0.5 * (0.9 + 0.1 * pulse))
            
            pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, radius, (*self.COLOR_SELECTION, int(alpha)))
            pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, radius, (*self.COLOR_SELECTION, int(alpha/2)))
            pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, radius-1, (*self.COLOR_SELECTION, int(alpha)))


    def _render_ui(self):
        target_x_start = self.GRID_X_OFFSET + self.GRID_COLS * self.CELL_SIZE + 50
        target_y_start = self.GRID_Y_OFFSET + 30
        mini_cell_size = 20
        
        self._render_text("TARGETS", self.UI_FONT, target_x_start, target_y_start - 25, self.COLOR_WHITE)

        for i, pattern in enumerate(self.target_patterns):
            pattern_h_cells = (max(b['r'] for b in pattern) + 1) if pattern else 1
            y_offset = target_y_start + i * (pattern_h_cells * mini_cell_size + 40)
            
            for p_block in pattern:
                x = int(target_x_start + p_block['c'] * mini_cell_size)
                y = int(y_offset + p_block['r'] * mini_cell_size)
                pygame.draw.rect(self.screen, p_block['color'], (x, y, mini_cell_size, mini_cell_size))
                pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (x, y, mini_cell_size, mini_cell_size), 2)
            
            if self.patterns_completed[i]:
                check_x = int(target_x_start + 100)
                check_y = int(y_offset + (pattern_h_cells * mini_cell_size) / 2 - 5)
                pygame.draw.lines(self.screen, self.COLOR_SUCCESS, False, [(check_x, check_y), (check_x + 10, check_y + 10), (check_x + 25, check_y - 5)], 5)

        self._render_text(f"MOVES: {self.moves_remaining}", self.UI_FONT, self.GRID_X_OFFSET, self.HEIGHT - 35, self.COLOR_WHITE)
        self._render_text(f"SCORE: {self.score:.1f}", self.UI_FONT, self.GRID_X_OFFSET + 200, self.HEIGHT - 35, self.COLOR_WHITE)

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(s, (0, 0))
            message = "COMPLETE!" if all(self.patterns_completed) else "OUT OF MOVES"
            color = self.COLOR_SUCCESS if all(self.patterns_completed) else self.COLOR_FAILURE
            self._render_text(message, self.TITLE_FONT, self.WIDTH / 2, self.HEIGHT / 2, color, center=True)

    def _render_text(self, text, font, x, y, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (int(x), int(y))
        else:
            text_rect.topleft = (int(x), int(y))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def validate_implementation(self):
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

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Slidey Blocks")
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # Start with no-op
    
    print(env.user_guide)
    
    while not done:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    action = np.array([0, 0, 0])
                    continue
        
        # --- Step the environment ---
        # Only step if an action was taken (not a no-op from holding a key)
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")
            action = np.array([0, 0, 0]) # Reset action after processing

        # --- Render the game ---
        # The observation is the rendered frame, so we just need to display it
        frame = np.transpose(obs, (1, 0, 2)) # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS
        
    pygame.quit()