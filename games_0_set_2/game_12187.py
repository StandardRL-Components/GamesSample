import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:28:17.769669
# Source Brief: brief_02187.md
# Brief Index: 2187
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    NumberMerge Environment: A real-time puzzle game where the player merges numbers to sum to 10.
    The difficulty increases as the player's score rises, with both the numbers on screen and the
    player's cursor moving faster. Visual polish and satisfying game feel are prioritized.

    - Action Space: MultiDiscrete([5, 2, 2]) for movement, selection, and an unused action.
    - Observation Space: A 640x400 RGB image of the game state.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Move your cursor to select numbers. Select two numbers that sum to 10 to score points and clear them from the board."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a number, then press space over another to merge them."
    )
    auto_advance = True


    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 10000
    WIN_SCORE = 50
    MAX_FAILS = 3
    TARGET_SUM = 10
    FPS = 30

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_GRID = (30, 35, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_FAIL = (255, 80, 80)
    COLOR_SUCCESS = (80, 255, 80)

    NUMBER_COLORS = {
        1: (66, 135, 245),   # Blue
        2: (245, 66, 66),    # Red
        3: (66, 245, 114),   # Green
        4: (245, 239, 66),   # Yellow
        5: (168, 66, 245),   # Purple
        6: (245, 150, 66),   # Orange
        7: (66, 245, 239),   # Cyan
        8: (245, 66, 215),   # Magenta
        9: (200, 200, 200)   # White/Gray
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.failed_merges = 0
        self.cursor_pos = [0.0, 0.0]
        self.base_cursor_speed = 5.0
        self.cursor_speed = 0.0
        self.number_base_speed = 1.0
        self.number_current_speed = 0.0
        self.numbers = []
        self.selected_number_idx = None
        self.last_space_held = False
        self.particles = []
        self.floating_texts = []
        self.last_score_checkpoint = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.failed_merges = 0
        self.cursor_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2]
        self.cursor_speed = self.base_cursor_speed
        self.number_current_speed = self.number_base_speed
        self.numbers = []
        self.selected_number_idx = None
        self.last_space_held = False
        self.particles = []
        self.floating_texts = []
        self.last_score_checkpoint = 0

        for _ in range(5):
            self._spawn_number()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        reward += self._handle_input(action)
        self._update_game_state()
        reward += self._handle_progression()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if (terminated or truncated) and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
                self._create_floating_text("YOU WIN!", [self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2], self.COLOR_SUCCESS, size=2, duration=120)
            elif self.failed_merges >= self.MAX_FAILS:
                reward += -100
                self._create_floating_text("GAME OVER", [self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2], self.COLOR_FAIL, size=2, duration=120)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Cursor Movement ---
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        if movement in move_map:
            dx, dy = move_map[movement]
            self.cursor_pos[0] += dx * self.cursor_speed
            self.cursor_pos[1] += dy * self.cursor_speed
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # --- Number Selection (Spacebar Press) ---
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        if space_pressed:
            target_idx = self._get_number_under_cursor()
            
            if self.selected_number_idx is None:
                if target_idx is not None:
                    self.selected_number_idx = target_idx
                    # sfx_select
            else:
                if target_idx is not None and target_idx != self.selected_number_idx:
                    reward += self._attempt_merge(target_idx)
                else:
                    self.selected_number_idx = None # Deselect if clicking empty space or same number
                    # sfx_deselect
        return reward

    def _attempt_merge(self, target_idx):
        num1 = self.numbers[self.selected_number_idx]
        num2 = self.numbers[target_idx]

        if num1['value'] + num2['value'] == self.TARGET_SUM:
            # --- SUCCESSFUL MERGE ---
            self.score += 10
            self.cursor_speed *= 1.05
            
            mid_point = [(num1['pos'][0] + num2['pos'][0])/2, (num1['pos'][1] + num2['pos'][1])/2]
            self._create_particles(mid_point, self.COLOR_SUCCESS, 30, 5)
            self._create_floating_text("+10", mid_point, self.COLOR_SUCCESS)
            # sfx_merge_success
            
            indices_to_remove = sorted([self.selected_number_idx, target_idx], reverse=True)
            for idx in indices_to_remove:
                self.numbers.pop(idx)
            self.selected_number_idx = None
            return 0.1
        else:
            # --- FAILED MERGE ---
            self.failed_merges += 1
            self._create_particles(self.cursor_pos, self.COLOR_FAIL, 15, 2)
            # sfx_merge_fail
            self.selected_number_idx = None
            return 0

    def _update_game_state(self):
        # Move numbers
        for num in self.numbers:
            num['pos'][0] += num['vel'][0] * self.number_current_speed
            num['pos'][1] += num['vel'][1] * self.number_current_speed
            
            if not (num['radius'] <= num['pos'][0] <= self.SCREEN_WIDTH - num['radius']):
                num['vel'][0] *= -1
                num['pos'][0] = np.clip(num['pos'][0], num['radius'], self.SCREEN_WIDTH - num['radius'])
            if not (num['radius'] <= num['pos'][1] <= self.SCREEN_HEIGHT - num['radius']):
                num['vel'][1] *= -1
                num['pos'][1] = np.clip(num['pos'][1], num['radius'], self.SCREEN_HEIGHT - num['radius'])

        # Animate selected number glow
        if self.selected_number_idx is not None and self.selected_number_idx < len(self.numbers):
            self.numbers[self.selected_number_idx]['selected_glow'] = (self.steps * 0.2) % (2 * math.pi)
        
        self._update_particles()
        self._update_floating_texts()

    def _handle_progression(self):
        reward = 0
        score_level = self.score // 10
        last_level = self.last_score_checkpoint // 10
        if score_level > last_level:
            self.number_current_speed += 0.1
            reward += 1.0
            self._create_floating_text("SPEED UP!", [self.SCREEN_WIDTH/2, 50], self.NUMBER_COLORS[6], size=1.5, duration=90)
            # sfx_level_up
        
        score_spawn_tier = self.score // 5
        last_spawn_tier = self.last_score_checkpoint // 5
        if score_spawn_tier > last_spawn_tier:
            self._spawn_number()
        
        self.last_score_checkpoint = self.score
        return reward

    def _check_termination(self):
        return (self.score >= self.WIN_SCORE or self.failed_merges >= self.MAX_FAILS)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "failed_merges": self.failed_merges}

    def _spawn_number(self):
        max_val = 5 if self.score < 20 else 9
        value = self.np_random.integers(1, max_val + 1)
        radius = 15 + value * 1.5
        
        pos = [self.np_random.uniform(radius, self.SCREEN_WIDTH - radius), self.np_random.uniform(radius, self.SCREEN_HEIGHT - radius)]
        angle = self.np_random.uniform(0, 2 * math.pi)
        vel = [math.cos(angle), math.sin(angle)]
        
        self.numbers.append({'pos': pos, 'vel': vel, 'value': value, 'color': self.NUMBER_COLORS[value], 'radius': radius, 'selected_glow': 0.0})

    def _create_particles(self, pos, color, count, speed_scale):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_scale
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'color': color, 'life': self.np_random.integers(20, 40)})

    def _create_floating_text(self, text, pos, color, size=1.0, duration=60):
        self.floating_texts.append({'text': text, 'pos': list(pos), 'color': color, 'life': duration, 'max_life': duration, 'size': size})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_floating_texts(self):
        for ft in self.floating_texts:
            ft['pos'][1] -= 0.5
            ft['life'] -= 1
        self.floating_texts = [ft for ft in self.floating_texts if ft['life'] > 0]

    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

    def _render_game(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*p['color'], alpha))

        for i, num in enumerate(self.numbers):
            pos_int = (int(num['pos'][0]), int(num['pos'][1]))
            radius_int = int(num['radius'])
            
            if self.selected_number_idx == i:
                glow_factor = 1 + 0.2 * math.sin(num['selected_glow'])
                self._draw_glowing_circle(pos_int, radius_int * (1.2 + glow_factor * 0.2), num['color'])
            
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, num['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius_int, (0,0,0))
            
            text_surf = self.font_large.render(str(num['value']), True, self.COLOR_BG)
            self.screen.blit(text_surf, text_surf.get_rect(center=pos_int))

        cursor_pos_int = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 8, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 8, (0,0,0))
        pygame.gfxdraw.filled_circle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 2, (0,0,0))

        for ft in self.floating_texts:
            alpha = int(255 * (ft['life'] / ft['max_life']))
            font = pygame.font.SysFont("monospace", int(32 * ft['size']), bold=True) if ft['size'] != 1.0 else self.font_large
            text_surf = font.render(ft['text'], True, ft['color'])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, text_surf.get_rect(center=ft['pos']))

    def _draw_glowing_circle(self, pos, radius, color):
        for i in range(4):
            alpha = 60 - i * 15
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius + i * 2), (*color, alpha))

    def _render_ui(self):
        self.screen.blit(self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT), (10, 10))
        self.screen.blit(self.font_small.render(f"CURSOR SPD: {self.cursor_speed:.1f}", True, self.COLOR_TEXT), (10, 30))
        
        self.screen.blit(self.font_small.render("FAILS:", True, self.COLOR_TEXT), (self.SCREEN_WIDTH - 120, 10))
        for i in range(self.MAX_FAILS):
            color = self.COLOR_FAIL if i < self.failed_merges else self.COLOR_GRID
            pygame.draw.line(self.screen, color, (self.SCREEN_WIDTH - 60 + i*20, 10), (self.SCREEN_WIDTH - 50 + i*20, 25), 3)
            pygame.draw.line(self.screen, color, (self.SCREEN_WIDTH - 60 + i*20, 25), (self.SCREEN_WIDTH - 50 + i*20, 10), 3)

    def _get_number_under_cursor(self):
        for i, num in enumerate(self.numbers):
            dist = math.hypot(self.cursor_pos[0] - num['pos'][0], self.cursor_pos[1] - num['pos'][1])
            if dist < num['radius']:
                return i
        return None

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not used by the evaluation environment.
    # It is safe to modify or remove this block.
    # The validation logic has been removed as it is not part of the standard env.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Number Merge Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    movement, space_held, shift_held = 0, 0, 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    env.close()
    pygame.quit()