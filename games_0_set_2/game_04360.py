
# Generated: 2025-08-28T02:10:30.359342
# Source Brief: brief_04360.md
# Brief Index: 4360

        
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
        "Controls: Use arrow keys to move the selected pixel. "
        "Space and Shift cycle through pixels to select them."
    )

    game_description = (
        "A minimalist puzzle game. Swap colored pixels on the grid to match the target pattern "
        "on the right. Each move costs one point, so plan your swaps carefully to solve the "
        "puzzle before you run out of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 64)
        
        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (50, 60, 80)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_SELECT = (255, 255, 255)
        self.COLOR_WIN = (100, 255, 150)
        self.COLOR_LOSE = (255, 100, 100)
        self.PIXEL_COLORS = [
            (50, 150, 255),  # Blue
            (255, 200, 50),   # Yellow
            (255, 80, 120),   # Pink
        ]

        # --- Game Config ---
        self.max_stages = 3
        self.max_steps = 1000
        self.stage_configs = {
            1: {'moves': 25, 'num_pixels': 8, 'num_colors': 2, 'grid_dim': 5},
            2: {'moves': 35, 'num_pixels': 12, 'num_colors': 2, 'grid_dim': 6},
            3: {'moves': 50, 'num_pixels': 16, 'num_colors': 3, 'grid_dim': 7},
        }

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.current_stage = 1
        self.moves_left = 0
        self.pixels = []
        self.selected_pixel_index = 0
        self.grid_dim = 0
        
        # --- Layout ---
        self.main_grid_rect = pygame.Rect(40, 40, 320, 320)
        self.target_grid_rect = pygame.Rect(420, 150, 180, 180)

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.current_stage = 1
        
        self._setup_stage(self.current_stage)
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self, stage_num):
        config = self.stage_configs[stage_num]
        self.current_stage = stage_num
        self.moves_left = config['moves']
        self.grid_dim = config['grid_dim']
        num_pixels = config['num_pixels']
        num_colors = config['num_colors']

        # Generate unique grid positions for pixels
        all_positions = [(x, y) for x in range(self.grid_dim) for y in range(self.grid_dim)]
        target_positions = self.np_random.choice(
            np.array(all_positions, dtype=object), num_pixels, replace=False
        ).tolist()
        
        # Create a shuffled version for the initial state
        initial_positions = list(target_positions)
        self.np_random.shuffle(initial_positions)

        self.pixels = []
        for i in range(num_pixels):
            color_idx = self.np_random.integers(0, num_colors)
            self.pixels.append({
                'pos': tuple(initial_positions[i]),
                'target_pos': tuple(target_positions[i]),
                'color_idx': color_idx,
                'color': self.PIXEL_COLORS[color_idx]
            })
        
        self.selected_pixel_index = 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # 1. Handle selection change
        if space_held and not shift_held:
            self.selected_pixel_index = (self.selected_pixel_index + 1) % len(self.pixels)
        elif shift_held and not space_held:
            self.selected_pixel_index = (self.selected_pixel_index - 1 + len(self.pixels)) % len(self.pixels)

        # 2. Handle pixel movement
        if movement != 0:
            self.moves_left -= 1
            
            p_selected = self.pixels[self.selected_pixel_index]
            curr_pos = p_selected['pos']
            
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            dest_pos = (curr_pos[0] + dx, curr_pos[1] + dy)

            # Check bounds
            if 0 <= dest_pos[0] < self.grid_dim and 0 <= dest_pos[1] < self.grid_dim:
                # Find if another pixel is at the destination
                p_other = next((p for p in self.pixels if p['pos'] == dest_pos), None)
                
                if p_other:
                    # Swap positions
                    p_selected['pos'], p_other['pos'] = p_other['pos'], p_selected['pos']
                    # sound: swap_sfx
                else:
                    # Move to empty space
                    p_selected['pos'] = dest_pos
                    # sound: move_sfx
        
        # 3. Calculate rewards and check for termination
        is_match = self._check_pattern_match()
        
        if is_match:
            if self.current_stage < self.max_stages:
                # Stage complete
                reward += 5
                self.score += reward
                self._setup_stage(self.current_stage + 1)
                # sound: stage_clear_sfx
            else:
                # Game won
                reward += 100
                self.game_won = True
                self.game_over = True
                # sound: game_win_sfx
        elif self.moves_left <= 0:
            # Game over - ran out of moves
            reward += -50
            self.game_over = True
            # sound: game_lose_sfx
        else:
            # Continuous reward for partial progress
            correct, incorrect = 0, 0
            for p in self.pixels:
                if p['pos'] == p['target_pos']:
                    correct += 1
                else:
                    incorrect += 1
            reward += (correct * 1.0) + (incorrect * -0.1)

        self.score += reward
        self.steps += 1
        if self.steps >= self.max_steps:
            self.game_over = True
        
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_pattern_match(self):
        return all(p['pos'] == p['target_pos'] for p in self.pixels)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # --- Main Grid ---
        self._draw_grid(self.main_grid_rect, self.grid_dim, 2)
        
        # --- Target Grid ---
        self._draw_grid(self.target_grid_rect, self.grid_dim, 1)

        # --- Pixels on Main Grid ---
        cell_size = self.main_grid_rect.width / self.grid_dim
        for i, p in enumerate(self.pixels):
            px, py = self._grid_to_screen(p['pos'], self.main_grid_rect, self.grid_dim)
            pixel_rect = pygame.Rect(px, py, cell_size, cell_size)
            pygame.draw.rect(self.screen, p['color'], pixel_rect.inflate(-4, -4), border_radius=4)
            if i == self.selected_pixel_index:
                pygame.draw.rect(self.screen, self.COLOR_SELECT, pixel_rect, 2, border_radius=6)

        # --- Pixels on Target Grid ---
        for p in self.pixels:
            px, py = self._grid_to_screen(p['target_pos'], self.target_grid_rect, self.grid_dim)
            cell_size_target = self.target_grid_rect.width / self.grid_dim
            pixel_rect = pygame.Rect(px, py, cell_size_target, cell_size_target)
            pygame.draw.rect(self.screen, p['color'], pixel_rect.inflate(-2, -2), border_radius=2)
            
    def _draw_grid(self, rect, dim, line_width):
        cell_size = rect.width / dim
        for i in range(dim + 1):
            # Vertical lines
            start_pos = (rect.left + i * cell_size, rect.top)
            end_pos = (rect.left + i * cell_size, rect.bottom)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, line_width)
            # Horizontal lines
            start_pos = (rect.left, rect.top + i * cell_size)
            end_pos = (rect.right, rect.top + i * cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, line_width)

    def _grid_to_screen(self, grid_pos, rect, dim):
        cell_size = rect.width / dim
        screen_x = rect.left + grid_pos[0] * cell_size
        screen_y = rect.top + grid_pos[1] * cell_size
        return int(screen_x), int(screen_y)

    def _render_ui(self):
        # --- Static Text ---
        self._draw_text("MOVES", (420, 40), self.font_s, self.COLOR_TEXT)
        self._draw_text("STAGE", (530, 40), self.font_s, self.COLOR_TEXT)
        self._draw_text("TARGET", (420, 120), self.font_s, self.COLOR_TEXT)
        self._draw_text("SCORE", (420, 340), self.font_s, self.COLOR_TEXT)
        
        # --- Dynamic Values ---
        self._draw_text(f"{self.moves_left}", (420, 65), self.font_m, self.COLOR_TEXT)
        self. _draw_text(f"{self.current_stage}/{self.max_stages}", (530, 65), self.font_m, self.COLOR_TEXT)
        self._draw_text(f"{int(self.score)}", (420, 365), self.font_m, self.COLOR_TEXT)

        # --- Game Over / Win Message ---
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.game_won:
                self._draw_text("YOU WIN!", (self.screen_width // 2, self.screen_height // 2), self.font_l, self.COLOR_WIN, center=True)
            else:
                self._draw_text("GAME OVER", (self.screen_width // 2, self.screen_height // 2), self.font_l, self.COLOR_LOSE, center=True)

    def _draw_text(self, text, pos, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Pixel Match")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    running = True
    
    while running:
        # Default action is no-op
        action = [0, 0, 0] # move, space, shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                
                # For turn-based, we only step on a key press
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: action[0] = 1
                elif keys[pygame.K_DOWN]: action[0] = 2
                elif keys[pygame.K_LEFT]: action[0] = 3
                elif keys[pygame.K_RIGHT]: action[0] = 4
                
                if keys[pygame.K_SPACE]: action[1] = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

                obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
    
    pygame.quit()