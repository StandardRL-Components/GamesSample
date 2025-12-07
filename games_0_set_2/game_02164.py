
# Generated: 2025-08-27T19:28:14.248369
# Source Brief: brief_02164.md
# Brief Index: 2164

        
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
        "Controls: Arrow keys to move the cursor. Space to pick up or swap a pixel. Shift to cancel a pickup."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target image on the left by swapping pixels on your grid. Plan your moves carefully before you run out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_DIM = 8
        self.PIXEL_SIZE = 24
        self.GRID_BORDER = 2
        self.MAX_MOVES_START = 75
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_BG = (40, 50, 70)
        self.COLOR_GRID_LINE = (60, 70, 90)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 20)
        self.COLOR_CURSOR = (255, 255, 100)
        self.PIXEL_COLORS = [
            (255, 0, 77), (255, 163, 0), (255, 236, 39),
            (0, 228, 54), (41, 173, 255), (131, 118, 156),
            (255, 119, 168), (255, 255, 255), (95, 205, 228),
            (19, 21, 23), (126, 37, 83), (0, 135, 81)
        ]
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 22)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Grid layout
        self.grid_size_px = self.GRID_DIM * self.PIXEL_SIZE
        total_width = self.grid_size_px * 2 + 80
        start_x = (self.WIDTH - total_width) // 2
        self.target_grid_pos = (start_x, (self.HEIGHT - self.grid_size_px) // 2 + 20)
        self.player_grid_pos = (start_x + self.grid_size_px + 80, self.target_grid_pos[1])
        
        # State variables (initialized in reset)
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.level = 1
        self.max_moves = self.MAX_MOVES_START
        self.moves_left = 0
        self.game_over = False
        self.target_grid = None
        self.player_grid = None
        self.last_num_correct = 0
        self.cursor_pos = [0, 0]
        self.picked_up_pixel = None  # Dict: {'color': color_index, 'pos': (x, y)}
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        if options and "level" in options:
            self.level = options["level"]
        else:
            self.level = 1

        self.steps = 0
        self.score = 0
        self.game_over = False
        self._generate_puzzle()
        self.max_moves = self.MAX_MOVES_START - (self.level - 1) * 5
        self.moves_left = self.max_moves
        self.cursor_pos = [self.GRID_DIM // 2, self.GRID_DIM // 2]
        self.picked_up_pixel = None
        self.particles = []
        self.last_num_correct = np.sum(self.player_grid == self.target_grid)
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        num_colors = min(len(self.PIXEL_COLORS), 4 + (self.level - 1))
        total_pixels = self.GRID_DIM * self.GRID_DIM
        
        base_pixels_per_color = total_pixels // num_colors
        remainder = total_pixels % num_colors
        
        colors = []
        for i in range(num_colors):
            count = base_pixels_per_color + (1 if i < remainder else 0)
            colors.extend([i] * count)

        self.np_random.shuffle(colors)
        self.target_grid = np.array(colors).reshape((self.GRID_DIM, self.GRID_DIM))

        player_colors = list(colors)
        while np.array_equal(np.array(player_colors), np.array(colors)):
            self.np_random.shuffle(player_colors)
        
        self.player_grid = np.array(player_colors).reshape((self.GRID_DIM, self.GRID_DIM))

    def step(self, action):
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = -0.01  # Small cost for taking any action
        terminated = False
        
        # 1. Handle cursor movement
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_DIM - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_DIM - 1)
            # sfx: cursor_move.wav

        # 2. Handle Shift (Cancel Pickup)
        if shift_press and self.picked_up_pixel is not None:
            self.picked_up_pixel = None
            # sfx: cancel.wav
            reward -= 0.1 # Penalize cancelling

        # 3. Handle Space (Pickup/Drop)
        elif space_press:
            if self.picked_up_pixel is None:
                px, py = self.cursor_pos
                color_index = self.player_grid[py, px]
                self.picked_up_pixel = {'color': color_index, 'pos': (px, py)}
                # sfx: pickup.wav
            else:
                px_new, py_new = self.cursor_pos
                px_old, py_old = self.picked_up_pixel['pos']

                if (px_new, py_new) != (px_old, py_old):
                    val1 = self.player_grid[py_old, px_old]
                    val2 = self.player_grid[py_new, px_new]
                    self.player_grid[py_new, px_new] = val1
                    self.player_grid[py_old, px_old] = val2
                    
                    # sfx: swap.wav
                    self._create_particles((px_new, py_new), self.PIXEL_COLORS[val2])
                    self._create_particles((px_old, py_old), self.PIXEL_COLORS[val1])

                    self.moves_left -= 1
                    
                    new_correct_pixels = np.sum(self.player_grid == self.target_grid)
                    reward_delta = (new_correct_pixels - self.last_num_correct)
                    reward += reward_delta * 1.5
                    self.score += reward_delta
                    self.last_num_correct = new_correct_pixels

                self.picked_up_pixel = None

        self._update_particles()

        is_win = np.array_equal(self.player_grid, self.target_grid)
        is_loss = self.moves_left <= 0 and not is_win
        
        if is_win:
            terminated = True
            reward = 100.0
            self.score += 100
            self.game_over = True
            # sfx: win.wav
        elif is_loss:
            terminated = True
            reward = -50.0
            self.score -= 50
            self.game_over = True
            # sfx: lose.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grids()
        self._render_cursor()
        self._render_picked_up_pixel()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, font, pos, color, shadow_color=None):
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_grids(self):
        self._render_text("TARGET", self.font_m, (self.target_grid_pos[0] + self.grid_size_px // 2 - 45, self.target_grid_pos[1] - 40), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        self._render_grid(self.target_grid, self.target_grid_pos)
        
        self._render_text("YOUR GRID", self.font_m, (self.player_grid_pos[0] + self.grid_size_px // 2 - 60, self.player_grid_pos[1] - 40), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        self._render_grid(self.player_grid, self.player_grid_pos)

    def _render_grid(self, grid_data, top_left_pos):
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (*top_left_pos, self.grid_size_px, self.grid_size_px))
        for y in range(self.GRID_DIM):
            for x in range(self.GRID_DIM):
                color_index = grid_data[y, x]
                
                # Check if this pixel is currently picked up
                is_picked_up_source = False
                if self.picked_up_pixel and grid_data is self.player_grid:
                    if (x, y) == self.picked_up_pixel['pos']:
                        is_picked_up_source = True

                if not is_picked_up_source:
                    color = self.PIXEL_COLORS[color_index]
                    rect = (
                        top_left_pos[0] + x * self.PIXEL_SIZE,
                        top_left_pos[1] + y * self.PIXEL_SIZE,
                        self.PIXEL_SIZE, self.PIXEL_SIZE
                    )
                    pygame.gfxdraw.box(self.screen, rect, color)
        
        # Draw grid lines
        for i in range(self.GRID_DIM + 1):
            # Vertical
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, 
                (top_left_pos[0] + i * self.PIXEL_SIZE, top_left_pos[1]),
                (top_left_pos[0] + i * self.PIXEL_SIZE, top_left_pos[1] + self.grid_size_px))
            # Horizontal
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE,
                (top_left_pos[0], top_left_pos[1] + i * self.PIXEL_SIZE),
                (top_left_pos[0] + self.grid_size_px, top_left_pos[1] + i * self.PIXEL_SIZE))

    def _render_cursor(self):
        pulse = (math.sin(self.steps * 0.3) + 1) / 2  # 0 to 1
        thickness = 2 + int(pulse * 2)
        
        cx, cy = self.cursor_pos
        rect = pygame.Rect(
            self.player_grid_pos[0] + cx * self.PIXEL_SIZE,
            self.player_grid_pos[1] + cy * self.PIXEL_SIZE,
            self.PIXEL_SIZE, self.PIXEL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, thickness)

    def _render_picked_up_pixel(self):
        if self.picked_up_pixel:
            color = self.PIXEL_COLORS[self.picked_up_pixel['color']]
            
            # Draw floating pixel over cursor
            cx, cy = self.cursor_pos
            size_pulse = 1 + math.sin(self.steps * 0.2) * 0.1
            size = int(self.PIXEL_SIZE * 1.2 * size_pulse)
            
            center_x = self.player_grid_pos[0] + cx * self.PIXEL_SIZE + self.PIXEL_SIZE // 2
            center_y = self.player_grid_pos[1] + cy * self.PIXEL_SIZE + self.PIXEL_SIZE // 2
            
            rect = pygame.Rect(center_x - size//2, center_y - size//2, size, size)
            
            pygame.gfxdraw.box(self.screen, rect, color)
            pygame.draw.rect(self.screen, (255,255,255), rect, 2)
    
    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['life']))

    def _render_ui(self):
        # Top UI
        self._render_text(f"LEVEL: {self.level}", self.font_s, (20, 20), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        self._render_text(f"SCORE: {self.score:.0f}", self.font_s, (20, 45), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Moves remaining
        moves_text = f"MOVES: {self.moves_left}"
        moves_color = self.COLOR_TEXT if self.moves_left > 10 else (255, 100, 100)
        self._render_text(moves_text, self.font_m, (self.WIDTH - 160, 20), moves_color, self.COLOR_TEXT_SHADOW)
        
        # Game Over message
        if self.game_over:
            is_win = np.array_equal(self.player_grid, self.target_grid)
            message = "COMPLETE!" if is_win else "OUT OF MOVES"
            msg_color = (100, 255, 100) if is_win else (255, 100, 100)
            self._render_text(message, self.font_l, (self.WIDTH // 2 - self.font_l.size(message)[0] // 2, self.HEIGHT - 60), msg_color, self.COLOR_TEXT_SHADOW)

    def _create_particles(self, grid_pos, color):
        px, py = grid_pos
        center_x = self.player_grid_pos[0] + px * self.PIXEL_SIZE + self.PIXEL_SIZE // 2
        center_y = self.player_grid_pos[1] + py * self.PIXEL_SIZE + self.PIXEL_SIZE // 2
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.uniform(3, 6),
                'color': color
            })
    
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 0.5
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_left": self.moves_left,
        }

    def close(self):
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # This is a one-shot key press mapping for human play
                # RL agents will use the MultiDiscrete space directly
                keys = pygame.key.get_pressed()
                
                move_action = 0 # none
                if keys[pygame.K_UP]: move_action = 1
                elif keys[pygame.K_DOWN]: move_action = 2
                elif keys[pygame.K_LEFT]: move_action = 3
                elif keys[pygame.K_RIGHT]: move_action = 4

                space_action = 1 if keys[pygame.K_SPACE] else 0
                shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

                action = [move_action, space_action, shift_action]
                
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.0f}, Moves Left: {info['moves_left']}")

                if terminated:
                    print("Game Over!")
                    pygame.time.wait(2000)
                    obs, info = env.reset()

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        
        # Create a display window if one doesn't exist
        try:
            display_surf = pygame.display.get_surface()
            if display_surf is None:
                raise Exception
            display_surf.blit(surf, (0, 0))
        except Exception:
            display_surf = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            display_surf.blit(surf, (0, 0))

        pygame.display.flip()
        env.clock.tick(30) # Limit frame rate for human play

    env.close()