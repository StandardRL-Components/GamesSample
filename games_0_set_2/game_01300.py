
# Generated: 2025-08-27T16:42:33.508279
# Source Brief: brief_01300.md
# Brief Index: 1300

        
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
        "Controls: Use arrow keys to move the cursor. Press SHIFT to cycle colors. Press SPACE to paint."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate a target pixel art image using a limited color palette before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_W, SCREEN_H = 640, 400
    CANVAS_W, CANVAS_H = 40, 30
    PIXEL_SIZE = 8
    
    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_UI_BG = (40, 40, 55)
    COLOR_UI_ACCENT = (80, 80, 100)
    COLOR_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    
    # Palette of 10 usable colors + 1 background color
    COLOR_PALETTE = [
        (255, 80, 80),   # 0: Red
        (80, 255, 80),   # 1: Green
        (80, 80, 255),   # 2: Blue
        (255, 255, 80),  # 3: Yellow
        (80, 255, 255),  # 4: Cyan
        (255, 80, 255),  # 5: Magenta
        (255, 160, 80),  # 6: Orange
        (160, 80, 255),  # 7: Purple
        (255, 255, 255), # 8: White
        (100, 100, 100), # 9: Gray
        (60, 60, 75)     # 10: BG Color (not paintable)
    ]
    BG_COLOR_INDEX = 10
    
    # Layout
    TARGET_RECT = pygame.Rect(40, 60, CANVAS_W * PIXEL_SIZE, CANVAS_H * PIXEL_SIZE)
    CANVAS_RECT = pygame.Rect(SCREEN_W - TARGET_RECT.right, 60, CANVAS_W * PIXEL_SIZE, CANVAS_H * PIXEL_SIZE)
    
    # Game settings
    TIME_PER_LEVEL = 60 * 30  # 60 seconds at 30fps
    NUM_LEVELS = 3
    WIN_THRESHOLD = 0.90 # 90% match

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 64)
        
        # Game state variables (initialized in reset)
        self.rng = None
        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.time_remaining = 0
        self.target_image = None
        self.player_canvas = None
        self.color_counts = []
        self.selected_color_index = 0
        self.cursor_pos = [0, 0]
        self.prev_space_state = False
        self.prev_shift_state = False
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self._load_level(self.level)
        
        return self._get_observation(), self._get_info()

    def _load_level(self, level_num):
        self.time_remaining = self.TIME_PER_LEVEL
        self.player_canvas = np.full((self.CANVAS_H, self.CANVAS_W), self.BG_COLOR_INDEX, dtype=int)
        self.cursor_pos = [self.CANVAS_W // 2, self.CANVAS_H // 2]
        self.selected_color_index = 0
        self.particles = []
        self.prev_space_state = False
        self.prev_shift_state = False

        # Procedurally generate target image
        target = np.full((self.CANVAS_H, self.CANVAS_W), self.BG_COLOR_INDEX, dtype=int)
        if level_num == 1: # Large blocks
            grid_w, grid_h = 4, 3
            colors = self.rng.choice(np.arange(4), size=grid_w*grid_h, replace=True)
            for y in range(self.CANVAS_H):
                for x in range(self.CANVAS_W):
                    grid_x, grid_y = x // (self.CANVAS_W // grid_w), y // (self.CANVAS_H // grid_h)
                    target[y, x] = colors[grid_y * grid_w + grid_x]
        elif level_num == 2: # Finer details, checkerboard
            grid_size = 4
            colors = self.rng.choice(np.arange(6), size=2, replace=False)
            for y in range(self.CANVAS_H):
                for x in range(self.CANVAS_W):
                    if (x // grid_size + y // grid_size) % 2 == 0:
                        target[y, x] = colors[0]
                    else:
                        target[y, x] = colors[1]
        elif level_num == 3: # Symmetric pattern with more colors
            colors = self.rng.choice(np.arange(10), size=4, replace=False)
            cx, cy = self.CANVAS_W // 2, self.CANVAS_H // 2
            for y in range(self.CANVAS_H):
                for x in range(self.CANVAS_W):
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < 4: target[y, x] = colors[0]
                    elif dist < 8: target[y, x] = colors[1]
                    elif dist < 12: target[y, x] = colors[2]
                    elif (x % 10 < 5) ^ (y % 10 < 5): target[y, x] = colors[3]
        
        self.target_image = target
        self._calculate_initial_colors()

    def _calculate_initial_colors(self):
        self.color_counts = [0] * len(self.COLOR_PALETTE)
        unique_colors, counts = np.unique(self.target_image, return_counts=True)
        for color_idx, count in zip(unique_colors, counts):
            if color_idx != self.BG_COLOR_INDEX:
                # Provide a 20% buffer + 5 extra for mistakes
                self.color_counts[color_idx] = int(count * 1.2) + 5
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        reward = 0
        terminated = False

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Actions ---
        # 1. Cursor Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] %= self.CANVAS_W
        self.cursor_pos[1] %= self.CANVAS_H

        # 2. Color Cycling (on key press, not hold)
        if shift_pressed and not self.prev_shift_state:
            self.selected_color_index = (self.selected_color_index + 1) % 10 # Only cycle paintable colors
            # Sound: color_cycle.wav
        
        # 3. Painting (on key press, not hold)
        if space_pressed and not self.prev_space_state:
            x, y = self.cursor_pos
            
            if self.color_counts[self.selected_color_index] > 0:
                # Paint the pixel
                prev_pixel_color = self.player_canvas[y, x]
                self.player_canvas[y, x] = self.selected_color_index
                self.color_counts[self.selected_color_index] -= 1
                # Sound: paint_pixel.wav
                self._create_particles(x, y, self.COLOR_PALETTE[self.selected_color_index])

                # Calculate reward
                target_pixel_color = self.target_image[y, x]
                if self.selected_color_index == target_pixel_color:
                    # Correct paint
                    reward += 0.1
                    # Bonus if it was previously wrong
                    if prev_pixel_color != target_pixel_color:
                        reward += 0.05
                else:
                    # Incorrect paint
                    reward -= 0.02
            else:
                # Tried to use an empty color -> Game Over
                # Sound: game_over.wav
                self.game_over = True
                terminated = True
                reward = -10
                self.win_message = "RAN OUT OF COLOR"

        self.prev_space_state = space_pressed
        self.prev_shift_state = shift_pressed

        # --- Update Game State ---
        self._update_particles()
        
        # --- Check for Termination/Level Completion ---
        if not terminated:
            match_percentage = self._calculate_match_percentage()
            if match_percentage >= self.WIN_THRESHOLD:
                reward += 5
                self.score += int(100 * match_percentage) + (self.time_remaining // 30)
                if self.level < self.NUM_LEVELS:
                    # Sound: level_complete.wav
                    self.level += 1
                    self._load_level(self.level)
                else:
                    # Sound: game_win.wav
                    self.game_over = True
                    terminated = True
                    reward += 50
                    self.win_message = "YOU WIN!"
            
            elif self.time_remaining <= 0:
                # Sound: game_over.wav
                self.game_over = True
                terminated = True
                reward = -10
                self.win_message = "TIME'S UP!"
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_match_percentage(self):
        if self.target_image is None or self.player_canvas is None:
            return 0.0
        
        total_pixels = self.CANVAS_W * self.CANVAS_H
        correct_pixels = np.sum(self.target_image == self.player_canvas)
        return correct_pixels / total_pixels

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
            "level": self.level,
            "time_remaining": self.time_remaining // 30,
            "match_percentage": self._calculate_match_percentage()
        }

    def _render_game(self):
        # Draw titles
        self._render_text("Target", self.TARGET_RECT.midtop[0], self.TARGET_RECT.top - 25, font=self.font_medium)
        self._render_text("Your Canvas", self.CANVAS_RECT.midtop[0], self.CANVAS_RECT.top - 25, font=self.font_medium)
        
        # Draw canvases
        self._draw_pixel_grid(self.target_image, self.TARGET_RECT.topleft)
        self._draw_pixel_grid(self.player_canvas, self.CANVAS_RECT.topleft)

        # Draw cursor
        if (self.steps // 10) % 2 == 0 and not self.game_over:
            cursor_screen_x = self.CANVAS_RECT.left + self.cursor_pos[0] * self.PIXEL_SIZE
            cursor_screen_y = self.CANVAS_RECT.top + self.cursor_pos[1] * self.PIXEL_SIZE
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_screen_x, cursor_screen_y, self.PIXEL_SIZE, self.PIXEL_SIZE), 2)
        
        # Draw particles
        for p in self.particles:
            size = int(p['life'] / p['max_life'] * (self.PIXEL_SIZE / 2))
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def _draw_pixel_grid(self, grid_data, top_left_pos):
        px, py = top_left_pos
        for y in range(self.CANVAS_H):
            for x in range(self.CANVAS_W):
                color_index = grid_data[y, x]
                color = self.COLOR_PALETTE[color_index]
                rect = (px + x * self.PIXEL_SIZE, py + y * self.PIXEL_SIZE, self.PIXEL_SIZE, self.PIXEL_SIZE)
                pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, (px, py, self.CANVAS_W * self.PIXEL_SIZE, self.CANVAS_H * self.PIXEL_SIZE), 1)

    def _render_ui(self):
        # Draw palette
        palette_w = 200
        palette_h = self.SCREEN_H - self.TARGET_RECT.bottom - 20
        palette_x = (self.SCREEN_W - palette_w) / 2
        palette_y = self.TARGET_RECT.bottom + 10
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (palette_x, palette_y, palette_w, palette_h), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, (palette_x, palette_y, palette_w, palette_h), 1, border_radius=5)
        
        swatch_size = 20
        for i in range(10):
            row = i // 5
            col = i % 5
            
            px = palette_x + 15 + col * (swatch_size + 15)
            py = palette_y + 15 + row * (swatch_size + 15)
            
            pygame.draw.rect(self.screen, self.COLOR_PALETTE[i], (px, py, swatch_size, swatch_size))
            if i == self.selected_color_index:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (px - 2, py - 2, swatch_size + 4, swatch_size + 4), 2)
            
            count_text = str(self.color_counts[i])
            self._render_text(count_text, px + swatch_size // 2, py + swatch_size + 8, font=self.font_small)

        # Draw stats
        match_pct = self._calculate_match_percentage() * 100
        self._render_text(f"Level: {self.level}/{self.NUM_LEVELS}", 80, 25, font=self.font_medium)
        self._render_text(f"Time: {max(0, self.time_remaining // 30)}s", self.SCREEN_W/2, 25, font=self.font_medium)
        self._render_text(f"Match: {match_pct:.1f}%", self.SCREEN_W - 80, 25, font=self.font_medium)

    def _render_text(self, text, x, y, color=COLOR_TEXT, font=None, center=True):
        if font is None: font = self.font_small
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        self.screen.blit(text_surf, text_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        msg_color = self.COLOR_WIN if "WIN" in self.win_message else self.COLOR_LOSE
        self._render_text(self.win_message, self.SCREEN_W / 2, self.SCREEN_H / 2, color=msg_color, font=self.font_large)

    def _create_particles(self, canvas_x, canvas_y, color):
        screen_x = self.CANVAS_RECT.left + (canvas_x + 0.5) * self.PIXEL_SIZE
        screen_y = self.CANVAS_RECT.top + (canvas_y + 0.5) * self.PIXEL_SIZE
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(10, 20)
            self.particles.append({'pos': [screen_x, screen_y], 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Pixel Painter")
    screen = pygame.display.set_mode((GameEnv.SCREEN_W, GameEnv.SCREEN_H))
    clock = pygame.time.Clock()
    
    terminated = False
    
    # --- Keyboard mapping for human play ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    action = [0, 0, 0] 
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Poll keyboard state ---
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # Default to no-op
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Buttons
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        # The observation is (H, W, C), but pygame wants (W, H) surface
        # The env's internal screen is already correct, so we just use that
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Reset if game over
        if terminated:
            print(f"Game Over! Final Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            terminated = False

        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()