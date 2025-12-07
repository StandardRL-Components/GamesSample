import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move the cursor. Press space to slice."
    )

    game_description = (
        "Fast-paced arcade action. Slice the falling fruit and avoid the bombs!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Game constants
        self.CURSOR_SPEED = 10
        self.FRUITS_TO_WIN = 25
        self.MAX_BOMBS_HIT = 3
        self.MAX_STEPS = 2000
        self.INITIAL_FALL_SPEED = 1.5
        self.MAX_FALL_SPEED = 5.0
        self.SPAWN_INTERVAL = 30 # Ticks between potential spawns

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (30, 10, 20)
        self.COLOR_BOMB = (20, 20, 20)
        self.COLOR_BOMB_SKULL = (200, 200, 200)
        self.COLOR_SLICE_TRAIL = (255, 255, 255)
        self.FRUIT_COLORS = [(220, 50, 50), (50, 220, 50), (255, 190, 0), (220, 50, 220)]
        self.PARTICLE_COLORS = [(255, 80, 80), (255, 150, 80)]

        # Fonts
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24)

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.last_cursor_pos = [0, 0]
        self.slice_trail = []
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.bombs_hit = 0
        self.fruits_sliced_total = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.spawn_timer = 0
        
        # self.reset() is called by the environment wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [self.screen_width // 2, self.screen_height // 2]
        self.last_cursor_pos = list(self.cursor_pos)
        self.slice_trail.clear()

        self.fruits.clear()
        self.bombs.clear()
        self.particles.clear()

        self.bombs_hit = 0
        self.fruits_sliced_total = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.spawn_timer = self.SPAWN_INTERVAL

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        # --- 1. Unpack Action & Update Cursor ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.last_cursor_pos = list(self.cursor_pos)

        # Calculate distance-based rewards
        dist_fruit_before, _ = self._get_closest_object(self.fruits)
        dist_bomb_before, _ = self._get_closest_object(self.bombs)

        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.screen_width)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.screen_height)

        dist_fruit_after, _ = self._get_closest_object(self.fruits)
        dist_bomb_after, _ = self._get_closest_object(self.bombs)

        if dist_fruit_after < dist_fruit_before: reward += 0.1
        if dist_bomb_after > dist_bomb_before: reward += 0.05
        
        if not space_held:
            self.slice_trail.append(list(self.cursor_pos))
            if len(self.slice_trail) > 10:
                self.slice_trail.pop(0)
        else:
            self.slice_trail.clear()

        # --- 2. Handle Slicing Action ---
        if space_held:
            # Sfx: Whoosh!
            # Check for fruit slices
            for fruit in self.fruits[::-1]:
                if not fruit['sliced'] and self._check_line_circle_collision(self.last_cursor_pos, self.cursor_pos, fruit['pos'], fruit['radius']):
                    fruit['sliced'] = True
                    fruit['slice_angle'] = math.atan2(self.cursor_pos[1] - self.last_cursor_pos[1], self.cursor_pos[0] - self.last_cursor_pos[0])
                    self.score += 10
                    reward += 10
                    self.fruits_sliced_total += 1
                    self._create_slice_particles(fruit)
                    # Sfx: Splat!
            
            # Check for bomb slices
            for bomb in self.bombs[::-1]:
                if self._check_line_circle_collision(self.last_cursor_pos, self.cursor_pos, bomb['pos'], bomb['radius']):
                    self.bombs_hit += 1
                    self.score -= 20
                    reward -= 20
                    self._create_explosion(bomb['pos'])
                    self.bombs.remove(bomb)
                    # Sfx: Explosion!
                    break # Only hit one bomb per slice

        # --- 3. Update Game State ---
        self.steps += 1

        # Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.fall_speed = min(self.MAX_FALL_SPEED, self.fall_speed + 0.05)

        # Update fruits
        for fruit in self.fruits[::-1]:
            if not fruit['sliced']:
                fruit['pos'][0] += fruit['vel'][0]
                fruit['pos'][1] += fruit['vel'][1] * self.fall_speed
            else: # Sliced fruits fall apart
                fruit['slice_timer'] -= 1
                fruit['pos'][1] += fruit['vel'][1] * 0.5 # Fall slower when sliced
            if fruit['pos'][1] > self.screen_height + 50 or fruit['slice_timer'] <= 0:
                self.fruits.remove(fruit)

        # Update bombs
        for bomb in self.bombs[::-1]:
            bomb['pos'][0] += bomb['vel'][0]
            bomb['pos'][1] += bomb['vel'][1] * self.fall_speed
            if bomb['pos'][1] > self.screen_height + 50:
                self.bombs.remove(bomb)

        # Update particles
        for p in self.particles[::-1]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

        # --- 4. Spawning Logic ---
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_timer = self.SPAWN_INTERVAL - int(self.fall_speed * 2)
            
            if self.np_random.random() < 0.2 + self.steps / 8000:
                self._spawn_bomb()

            active_fruits = len([f for f in self.fruits if not f['sliced']])
            if self.fruits_sliced_total + active_fruits < self.FRUITS_TO_WIN:
                if self.np_random.random() < 0.8:
                    self._spawn_fruit()

        # --- 5. Check Termination Conditions ---
        truncated = False
        if self.fruits_sliced_total >= self.FRUITS_TO_WIN:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.bombs_hit >= self.MAX_BOMBS_HIT:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._draw_gradient_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "fruits_sliced": self.fruits_sliced_total, "bombs_hit": self.bombs_hit}

    def _spawn_fruit(self):
        self.fruits.append({
            'pos': [self.np_random.integers(50, self.screen_width - 50), -20.0],
            'vel': [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(0.8, 1.2)],
            'radius': self.np_random.integers(15, 25),
            'color': self.FRUIT_COLORS[self.np_random.integers(len(self.FRUIT_COLORS))],
            'sliced': False,
            'slice_angle': 0,
            'slice_timer': 60
        })

    def _spawn_bomb(self):
        self.bombs.append({
            'pos': [self.np_random.integers(50, self.screen_width - 50), -20.0],
            'vel': [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(0.8, 1.2)],
            'radius': 20,
        })

    def _create_explosion(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifetime': self.np_random.integers(20, 40),
                'radius': self.np_random.integers(3, 8),
                'color': self.PARTICLE_COLORS[self.np_random.integers(len(self.PARTICLE_COLORS))]
            })

    def _create_slice_particles(self, fruit):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(fruit['pos']),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifetime': self.np_random.integers(15, 30),
                'radius': self.np_random.integers(2, 5),
                'color': fruit['color']
            })

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            radius = int(p['radius'] * (p['lifetime']/40.0))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])

        # Draw bombs
        for bomb in self.bombs:
            pos_int = (int(bomb['pos'][0]), int(bomb['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], bomb['radius'], self.COLOR_BOMB)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], bomb['radius'], (50,50,50))
            cx, cy = pos_int
            r = bomb['radius']
            pygame.draw.rect(self.screen, self.COLOR_BOMB_SKULL, (cx - r*0.3, cy - r*0.4, r*0.6, r*0.5), border_radius=2)
            pygame.draw.circle(self.screen, self.COLOR_BOMB, (cx - r*0.15, cy - r*0.2), int(r*0.1))
            pygame.draw.circle(self.screen, self.COLOR_BOMB, (cx + r*0.15, cy - r*0.2), int(r*0.1))
            pygame.draw.rect(self.screen, self.COLOR_BOMB_SKULL, (cx - r*0.15, cy + r*0.2, r*0.3, r*0.1))

        # Draw fruits
        for fruit in self.fruits:
            pos_int = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            if not fruit['sliced']:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], fruit['radius'], fruit['color'])
                # FIX: A generator expression is not a valid color. Use a list comprehension.
                darker_color = [c // 2 for c in fruit['color']]
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], fruit['radius'], darker_color)
            else:
                self._draw_sliced_fruit(fruit)
        
        # Draw slice trail
        if len(self.slice_trail) > 1:
            for i in range(len(self.slice_trail) - 1):
                # FIX: Using a 4-component RGBA color with pygame.draw.line on a non-alpha surface can cause errors.
                # Instead, we create a 3-component color that fades to black to simulate fading.
                ratio = (i + 1) / len(self.slice_trail)
                color = [int(c * ratio) for c in self.COLOR_SLICE_TRAIL]
                start_pos = self.slice_trail[i]
                end_pos = self.slice_trail[i+1]
                width = max(1, i // 2)
                pygame.draw.line(self.screen, color, start_pos, end_pos, width)

    def _draw_sliced_fruit(self, fruit):
        center_x, center_y = fruit['pos']
        radius = fruit['radius']
        angle = fruit['slice_angle']
        
        offset_dist = (60 - fruit['slice_timer']) / 10.0
        
        perp_angle = angle + math.pi / 2
        dx = math.cos(perp_angle) * offset_dist
        dy = math.sin(perp_angle) * offset_dist

        # Pygame's arc drawing is tricky. We'll use filled polygons for robustness.
        def get_arc_poly(center, radius, start_angle, end_angle, num_segments=20):
            points = [center]
            for i in range(num_segments + 1):
                a = start_angle + (end_angle - start_angle) * i / num_segments
                points.append((center[0] + math.cos(a) * radius, center[1] + math.sin(a) * radius))
            return points

        # First half
        center1 = (center_x + dx, center_y + dy)
        poly1 = get_arc_poly(center1, radius, angle, angle + math.pi)
        pygame.draw.polygon(self.screen, fruit['color'], poly1)
        
        # Second half
        center2 = (center_x - dx, center_y - dy)
        poly2 = get_arc_poly(center2, radius, angle + math.pi, angle + 2 * math.pi)
        pygame.draw.polygon(self.screen, fruit['color'], poly2)

    def _render_ui(self):
        score_text = self.font_large.render(f"{self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 10))

        for i in range(self.MAX_BOMBS_HIT):
            color = (100, 0, 0) if i < self.bombs_hit else (80, 80, 80)
            pygame.gfxdraw.filled_circle(self.screen, self.screen_width - 30 - i*30, 30, 10, self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(self.screen, self.screen_width - 30 - i*30, 30, 8, color)

        fruit_prog_text = self.font_small.render(f"{self.fruits_sliced_total}/{self.FRUITS_TO_WIN}", True, (200, 200, 200))
        self.screen.blit(fruit_prog_text, (20, 50))

    def _draw_gradient_background(self):
        for y in range(self.screen_height):
            ratio = y / self.screen_height
            color = [int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(self.COLOR_BG_TOP, self.COLOR_BG_BOTTOM)]
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))

    def _get_closest_object(self, objects):
        min_dist = float('inf')
        closest_obj = None
        if not objects:
            return min_dist, closest_obj
        for obj in objects:
            if 'sliced' in obj and obj['sliced']: continue
            dist = math.hypot(self.cursor_pos[0] - obj['pos'][0], self.cursor_pos[1] - obj['pos'][1])
            if dist < min_dist:
                min_dist = dist
                closest_obj = obj
        return min_dist, closest_obj

    def _check_line_circle_collision(self, p1, p2, circle_center, r):
        if math.hypot(p1[0] - circle_center[0], p1[1] - circle_center[1]) < r: return True
        if math.hypot(p2[0] - circle_center[0], p2[1] - circle_center[1]) < r: return True

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = circle_center

        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0: return False

        t = ((cx - x1) * dx + (cy - y1) * dy) / (dx**2 + dy**2)
        t = np.clip(t, 0, 1)

        closest_x, closest_y = x1 + t * dx, y1 + t * dy
        dist = math.hypot(closest_x - cx, closest_y - cy)
        return dist < r

    def close(self):
        pygame.quit()