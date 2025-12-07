
# Generated: 2025-08-28T02:36:12.701769
# Source Brief: brief_04502.md
# Brief Index: 4502

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import Counter
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press Space to select a block. "
        "Match two blocks of the same color to clear them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced grid-based puzzle game. Race against the clock to clear the board "
        "by matching pairs of colored blocks. Clear the board before time runs out to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.NUM_PAIRS = 20
        self.MAX_TIME = 45  # seconds
        self.MAX_STEPS = self.MAX_TIME * self.FPS

        # --- Colors ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (45, 50, 55)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TIMER_BG = (80, 20, 20)
        self.COLOR_TIMER_FG_GOOD = (40, 200, 40)
        self.COLOR_TIMER_FG_WARN = (255, 180, 0)
        self.COLOR_TIMER_FG_BAD = (220, 40, 40)
        
        # Block colors (index 0 is empty)
        self.BLOCK_COLORS = [
            (0, 0, 0),  # Empty
            (227, 68, 68),   # Red
            (68, 184, 227),  # Blue
            (111, 227, 68),  # Green
            (227, 147, 68),  # Orange
            (185, 68, 227),  # Purple
            (68, 227, 212),  # Cyan
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.cursor_pos = [0, 0]
        self.grid = []
        self.selected_block = None # Stores (pos, color_index)
        self.space_was_held = False
        self.particles = []
        self.np_random = None
        self.blocks_remaining = 0

        # Calculate grid layout
        self.grid_area_width = self.WIDTH * 0.8
        self.grid_area_height = self.HEIGHT * 0.9
        self.cell_width = self.grid_area_width / self.GRID_COLS
        self.cell_height = self.grid_area_height / self.GRID_ROWS
        self.grid_offset_x = (self.WIDTH - self.grid_area_width) / 2
        self.grid_offset_y = (self.HEIGHT - self.grid_area_height) / 2 + 20

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_TIME
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_block = None
        self.space_was_held = False
        self.particles = []
        
        self._generate_board()
        self.blocks_remaining = self.NUM_PAIRS * 2

        return self._get_observation(), self._get_info()
    
    def _generate_board(self):
        colors_to_place = []
        available_colors = list(range(1, len(self.BLOCK_COLORS)))
        for _ in range(self.NUM_PAIRS):
            color_idx = self.np_random.choice(available_colors)
            colors_to_place.extend([color_idx, color_idx])

        self.np_random.shuffle(colors_to_place)

        self.grid = [[0] * self.GRID_COLS for _ in range(self.GRID_ROWS)]
        
        all_coords = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        
        num_blocks = self.NUM_PAIRS * 2
        if num_blocks > len(all_coords):
            raise ValueError("Not enough grid cells for the number of blocks.")

        chosen_coords_indices = self.np_random.choice(len(all_coords), size=num_blocks, replace=False)
        chosen_coords = [all_coords[i] for i in chosen_coords_indices]

        for i, (x, y) in enumerate(chosen_coords):
            self.grid[y][x] = colors_to_place[i]


    def step(self, action):
        reward = 0
        self.game_over = self._check_termination()
        
        if self.game_over:
            # Apply terminal reward only once
            if self.blocks_remaining == 0:
                reward = 100
            else:
                reward = -50
            return self._get_observation(), reward, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        is_space_just_pressed = space_held and not self.space_was_held
        self.space_was_held = space_held

        self._move_cursor(movement)

        if is_space_just_pressed:
            selection_reward = self._handle_selection()
            reward += selection_reward
            self.score += selection_reward

        # Update game state
        self.steps += 1
        self.time_left -= 1.0 / self.FPS
        self._update_particles()
        
        terminated = self._check_termination()
        if terminated:
            if self.blocks_remaining == 0:
                reward += 100
            else:
                reward += -50
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, movement):
        x, y = self.cursor_pos
        if movement == 1: y -= 1  # Up
        elif movement == 2: y += 1  # Down
        elif movement == 3: x -= 1  # Left
        elif movement == 4: x += 1  # Right
        
        # Wrap around
        self.cursor_pos[0] = x % self.GRID_COLS
        self.cursor_pos[1] = y % self.GRID_ROWS

    def _handle_selection(self):
        cx, cy = self.cursor_pos
        color_index = self.grid[cy][cx]

        if color_index == 0: # Clicked on empty space
            return 0

        # Case 1: No block is selected
        if self.selected_block is None:
            self.selected_block = ((cx, cy), color_index)
            # Check for reward
            counts = Counter(c for row in self.grid for c in row if c != 0)
            if counts[color_index] >= 2:
                return 0.1 # Good selection
            else:
                return -0.1 # Bad selection (orphan block)

        # Case 2: A block is already selected
        else:
            sel_pos, sel_color_index = self.selected_block
            
            if (cx, cy) == sel_pos:
                self.selected_block = None
                return 0

            if color_index == sel_color_index:
                # sfx: match_success.wav
                self.grid[cy][cx] = 0
                self.grid[sel_pos[1]][sel_pos[0]] = 0
                self.blocks_remaining -= 2
                
                pos1 = self._get_pixel_pos(cx, cy)
                pos2 = self._get_pixel_pos(sel_pos[0], sel_pos[1])
                color = self.BLOCK_COLORS[sel_color_index]
                self._create_particles(pos1, color)
                self._create_particles(pos2, color)

                self.selected_block = None
                return 10
            
            else:
                # sfx: match_fail.wav
                self.selected_block = None
                return 0

    def _check_termination(self):
        if self.game_over:
            return True
        return self.time_left <= 0 or self.blocks_remaining == 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(self.GRID_COLS + 1):
            px = self.grid_offset_x + x * self.cell_width
            py_start = self.grid_offset_y
            py_end = self.grid_offset_y + self.grid_area_height
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, py_start), (px, py_end))
        for y in range(self.GRID_ROWS + 1):
            py = self.grid_offset_y + y * self.cell_height
            px_start = self.grid_offset_x
            px_end = self.grid_offset_x + self.grid_area_width
            pygame.draw.line(self.screen, self.COLOR_GRID, (px_start, py), (px_end, py))

        pulse = (math.sin(self.steps * 0.25) + 1) / 2
        
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                color_index = self.grid[y][x]
                if color_index != 0:
                    px, py, pw, ph = self._get_block_rect(x, y)
                    color = self.BLOCK_COLORS[color_index]
                    
                    shadow_color = tuple(max(0, c - 40) for c in color)
                    pygame.draw.rect(self.screen, shadow_color, (px + 2, py + 2, pw, ph), border_radius=4)
                    
                    pygame.draw.rect(self.screen, color, (px, py, pw, ph), border_radius=4)
                    
                    if self.selected_block and self.selected_block[0] == (x, y):
                        highlight_alpha = 100 + pulse * 120
                        highlight_color = (*self.COLOR_CURSOR, highlight_alpha)
                        s = pygame.Surface((pw, ph), pygame.SRCALPHA)
                        pygame.draw.rect(s, highlight_color, s.get_rect(), border_radius=6)
                        self.screen.blit(s, (px, py))

        cx, cy = self.cursor_pos
        px, py, pw, ph = self._get_block_rect(cx, cy)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (px, py, pw, ph), width=3, border_radius=6)

        self._render_particles()

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        timer_rect_bg = pygame.Rect(self.WIDTH - 215, 15, 200, 20)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BG, timer_rect_bg, border_radius=5)
        
        time_ratio = max(0, self.time_left / self.MAX_TIME)
        timer_width = 200 * time_ratio
        
        if time_ratio > 0.5:
            timer_color = self.COLOR_TIMER_FG_GOOD
        elif time_ratio > 0.2:
            timer_color = self.COLOR_TIMER_FG_WARN
        else:
            timer_color = self.COLOR_TIMER_FG_BAD
            
        timer_rect_fg = pygame.Rect(self.WIDTH - 215, 15, timer_width, 20)
        pygame.draw.rect(self.screen, timer_color, timer_rect_fg, border_radius=5)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.blocks_remaining == 0:
                msg = "BOARD CLEARED!"
                color = self.COLOR_TIMER_FG_GOOD
            else:
                msg = "TIME'S UP!"
                color = self.COLOR_TIMER_FG_BAD
            
            text_surf = self.font_msg.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_block_rect(self, grid_x, grid_y):
        padding = 4
        px = self.grid_offset_x + grid_x * self.cell_width + padding
        py = self.grid_offset_y + grid_y * self.cell_height + padding
        pw = self.cell_width - 2 * padding
        ph = self.cell_height - 2 * padding
        return int(px), int(py), int(pw), int(ph)

    def _get_pixel_pos(self, grid_x, grid_y):
        px, py, pw, ph = self._get_block_rect(grid_x, grid_y)
        return px + pw / 2, py + ph / 2

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = random.uniform(3, 7)
            life = random.randint(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'size': size, 'life': life, 'max_life': life, 'color': color})
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.98
            p['vel'][1] *= 0.98
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            current_size = int(p['size'] * life_ratio)
            if current_size > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                alpha = int(255 * life_ratio)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], current_size, (*p['color'], alpha))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], current_size, (*p['color'], alpha))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": round(self.time_left, 2),
            "blocks_remaining": self.blocks_remaining,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    running = True
    
    pygame.display.set_caption("Gridzapper")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    movement = 0
    space_held = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000)
            obs, info = env.reset()

        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))

        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()