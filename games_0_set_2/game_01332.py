
# Generated: 2025-08-27T16:47:52.450451
# Source Brief: brief_01332.md
# Brief Index: 1332

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a location."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find all 20 hidden geometric shapes in the abstract scene before the 5-minute timer runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME = 300  # 5 minutes
        self.NUM_OBJECTS = 20
        self.CURSOR_SPEED = 15
        self.OBJECT_BASE_SIZE = 10
        self.CLICK_RADIUS = 15

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_CORRECT = (0, 255, 128)
        self.COLOR_INCORRECT = (255, 80, 80)
        self.BG_PALETTE = [
            (40, 50, 80), (50, 60, 90), (30, 40, 70)
        ]
        self.OBJECT_PALETTE = [
            (255, 0, 128), (0, 255, 255), (255, 255, 0),
            (0, 255, 0), (255, 128, 0), (128, 0, 255)
        ]
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_msg = pygame.font.SysFont('Consolas', 48, bold=True)
        
        # State variables
        self.cursor_pos = None
        self.objects = None
        self.found_count = None
        self.time_elapsed = None
        self.game_over = None
        self.steps = None
        self.score = None
        self.prev_space_held = None
        self.effects = None
        self.clicked_map = None
        self.background_surface = None
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.cursor_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.steps = 0
        self.score = 0
        self.found_count = 0
        self.time_elapsed = 0
        self.game_over = False
        self.prev_space_held = False
        self.effects = []
        self.clicked_map = np.zeros((self.WIDTH, self.HEIGHT), dtype=bool)

        self._generate_background()
        self._generate_objects()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.auto_advance:
            self.clock.tick(self.FPS)

        self.time_elapsed += 1 / self.FPS
        self.steps += 1
        reward = 0
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        self._handle_movement(movement)
        
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            # sfx: click_sound
            click_reward = self._handle_click()
            reward += click_reward
        self.prev_space_held = space_held
        
        self._update_effects()
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.found_count == self.NUM_OBJECTS:
                # sfx: win_jingle
                terminal_reward = 100
            else:
                # sfx: lose_buzzer
                terminal_reward = -50
            reward += terminal_reward

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED  # Up
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED  # Down
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED  # Left
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT - 1)

    def _handle_click(self):
        click_pos = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        
        for obj in self.objects:
            if not obj['found']:
                dist = math.hypot(click_pos[0] - obj['pos'][0], click_pos[1] - obj['pos'][1])
                if dist <= self.CLICK_RADIUS:
                    obj['found'] = True
                    self.found_count += 1
                    self.effects.append({'type': 'correct', 'pos': click_pos, 'radius': 0, 'max_radius': 30, 'life': 1})
                    # sfx: correct_ding
                    return 10.0

        # If no object was found
        if not self.clicked_map[click_pos]:
            self.clicked_map[click_pos] = True
            self.effects.append({'type': 'incorrect', 'pos': click_pos, 'radius': 0, 'max_radius': 20, 'life': 1})
            # sfx: incorrect_buzz
            return -1.0
        else:
            self.effects.append({'type': 'incorrect', 'pos': click_pos, 'radius': 0, 'max_radius': 15, 'life': 0.5})
            # sfx: incorrect_buzz_short
            return -0.01

    def _update_effects(self):
        for effect in self.effects[:]:
            effect['life'] -= 1 / self.FPS * 3
            effect['radius'] = effect['max_radius'] * (1 - effect['life'])
            if effect['life'] <= 0:
                self.effects.remove(effect)

    def _check_termination(self):
        time_up = self.time_elapsed >= self.MAX_TIME
        all_found = self.found_count == self.NUM_OBJECTS
        return time_up or all_found

    def _get_observation(self):
        self.screen.blit(self.background_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "objects_found": self.found_count,
            "time_remaining": self.MAX_TIME - self.time_elapsed,
        }

    def _render_game(self):
        # Draw objects
        for obj in self.objects:
            if obj['found']:
                # Draw found object clearly
                self._draw_shape(self.screen, obj['shape'], obj['pos'], obj['size'], obj['color'], rotation=obj['rotation'])
                pygame.gfxdraw.aacircle(self.screen, int(obj['pos'][0]), int(obj['pos'][1]), int(obj['size'] * 1.2), self.COLOR_CORRECT)
            else:
                # Draw hidden object with shimmer
                shimmer_alpha = 128 + 127 * math.sin(self.steps * 0.2 + obj['phase'])
                temp_surf = pygame.Surface((obj['size'] * 2.5, obj['size'] * 2.5), pygame.SRCALPHA)
                self._draw_shape(temp_surf, obj['shape'], (temp_surf.get_width()//2, temp_surf.get_height()//2), obj['size'], obj['color'], rotation=obj['rotation'])
                temp_surf.set_alpha(shimmer_alpha)
                self.screen.blit(temp_surf, (obj['pos'][0] - temp_surf.get_width()//2, obj['pos'][1] - temp_surf.get_height()//2))

        # Draw click feedback effects
        for effect in self.effects:
            color = self.COLOR_CORRECT if effect['type'] == 'correct' else self.COLOR_INCORRECT
            alpha = int(255 * effect['life'])
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, effect['pos'][0], effect['pos'][1], int(effect['radius']), (*color, alpha))

        # Draw cursor
        cx, cy = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx - 10, cy), (cx + 10, cy), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx, cy - 10), (cx, cy + 10), 2)
        pygame.gfxdraw.aacircle(self.screen, cx, cy, self.CLICK_RADIUS, self.COLOR_CURSOR)

    def _render_ui(self):
        # Found count
        found_text = self.font_ui.render(f"Found: {self.found_count}/{self.NUM_OBJECTS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(found_text, (10, 10))

        # Timer
        time_left = max(0, self.MAX_TIME - self.time_elapsed)
        minutes = int(time_left // 60)
        seconds = int(time_left % 60)
        timer_text = self.font_ui.render(f"Time: {minutes:02}:{seconds:02}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.found_count == self.NUM_OBJECTS:
                msg_text = self.font_msg.render("YOU WIN!", True, self.COLOR_CORRECT)
            else:
                msg_text = self.font_msg.render("TIME'S UP!", True, self.COLOR_INCORRECT)
            self.screen.blit(msg_text, (self.WIDTH // 2 - msg_text.get_width() // 2, self.HEIGHT // 2 - msg_text.get_height() // 2))

    def _generate_background(self):
        self.background_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.background_surface.fill(self.COLOR_BG)
        for _ in range(150):
            shape_type = self.np_random.choice(['rect', 'circle', 'line'])
            color = self.np_random.choice(self.BG_PALETTE)
            pos = (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT))
            if shape_type == 'rect':
                size = (self.np_random.integers(20, 100), self.np_random.integers(20, 100))
                pygame.draw.rect(self.background_surface, color, (*pos, *size))
            elif shape_type == 'circle':
                radius = self.np_random.integers(10, 60)
                pygame.gfxdraw.filled_circle(self.background_surface, pos[0], pos[1], radius, color)
            elif shape_type == 'line':
                end_pos = (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT))
                width = self.np_random.integers(1, 5)
                pygame.draw.line(self.background_surface, color, pos, end_pos, width)

    def _generate_objects(self):
        self.objects = []
        shapes = ['circle', 'square', 'triangle', 'star', 'cross']
        min_dist = 40
        attempts = 0
        while len(self.objects) < self.NUM_OBJECTS and attempts < 1000:
            attempts += 1
            size = self.OBJECT_BASE_SIZE * self.np_random.uniform(0.8, 1.5)
            pos = (
                self.np_random.integers(size, self.WIDTH - size),
                self.np_random.integers(size, self.HEIGHT - size)
            )
            
            # Check for overlap with existing objects
            is_overlapping = False
            for obj in self.objects:
                if math.hypot(pos[0] - obj['pos'][0], pos[1] - obj['pos'][1]) < min_dist:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                self.objects.append({
                    'pos': pos,
                    'shape': self.np_random.choice(shapes),
                    'color': self.np_random.choice(self.OBJECT_PALETTE),
                    'size': size,
                    'rotation': self.np_random.uniform(0, 360),
                    'phase': self.np_random.uniform(0, 2 * math.pi),
                    'found': False
                })
        
        if len(self.objects) < self.NUM_OBJECTS:
             print(f"Warning: Could only place {len(self.objects)}/{self.NUM_OBJECTS} objects.")


    def _draw_shape(self, surface, shape, pos, size, color, rotation=0):
        x, y = pos
        if shape == 'circle':
            pygame.gfxdraw.filled_circle(surface, int(x), int(y), int(size), color)
            pygame.gfxdraw.aacircle(surface, int(x), int(y), int(size), color)
        elif shape == 'square':
            points = [(-size, -size), (size, -size), (size, size), (-size, size)]
            rad = math.radians(rotation)
            rotated = [(p[0]*math.cos(rad) - p[1]*math.sin(rad) + x, p[0]*math.sin(rad) + p[1]*math.cos(rad) + y) for p in points]
            pygame.gfxdraw.filled_polygon(surface, rotated, color)
            pygame.gfxdraw.aapolygon(surface, rotated, color)
        elif shape == 'triangle':
            points = [(0, -size * 1.2), (-size, size * 0.8), (size, size * 0.8)]
            rad = math.radians(rotation)
            rotated = [(p[0]*math.cos(rad) - p[1]*math.sin(rad) + x, p[0]*math.sin(rad) + p[1]*math.cos(rad) + y) for p in points]
            pygame.gfxdraw.filled_polygon(surface, rotated, color)
            pygame.gfxdraw.aapolygon(surface, rotated, color)
        elif shape == 'cross':
            w = size / 4
            points1 = [(-size, -w), (size, -w), (size, w), (-size, w)]
            points2 = [(-w, -size), (w, -size), (w, size), (-w, size)]
            rad = math.radians(rotation)
            rotated1 = [(p[0]*math.cos(rad) - p[1]*math.sin(rad) + x, p[0]*math.sin(rad) + p[1]*math.cos(rad) + y) for p in points1]
            rotated2 = [(p[0]*math.cos(rad) - p[1]*math.sin(rad) + x, p[0]*math.sin(rad) + p[1]*math.cos(rad) + y) for p in points2]
            pygame.gfxdraw.filled_polygon(surface, rotated1, color)
            pygame.gfxdraw.filled_polygon(surface, rotated2, color)
        elif shape == 'star':
            points = []
            for i in range(10):
                r = size if i % 2 == 0 else size / 2
                angle = math.radians(rotation + i * 36)
                points.append((r * math.cos(angle) + x, r * math.sin(angle) + y))
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)

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

    def close(self):
        pygame.quit()