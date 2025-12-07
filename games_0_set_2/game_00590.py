
# Generated: 2025-08-27T14:07:25.494818
# Source Brief: brief_00590.md
# Brief Index: 590

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your slicer across the screen."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit to score points while avoiding the bombs in this fast-paced arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    # Colors
    COLOR_BG_TOP = (135, 206, 235)
    COLOR_BG_BOTTOM = (25, 25, 112)
    COLOR_TEXT = (255, 255, 255)
    COLOR_SLICER = (255, 255, 255)
    COLOR_BOMB = (30, 30, 30)
    COLOR_FUSE = (150, 75, 0)
    COLOR_SPARK = (255, 255, 0)
    COLOR_APPLE = (220, 20, 60)
    COLOR_BANANA = (255, 225, 53)
    COLOR_GRAPE = (106, 13, 173)
    
    # Game parameters
    SLICER_SPEED = 12
    SLICER_TRAIL_LENGTH = 8
    OBJECT_RADIUS = 20
    SPAWN_INTERVAL = 15
    DIFFICULTY_INTERVAL = 500
    INITIAL_FALL_SPEED = 2.0
    FALL_SPEED_INCREMENT = 0.1
    MAX_STEPS = 5000
    WIN_SCORE = 500
    MAX_BOMB_HITS = 3
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 64)
        self.font_small = pygame.font.Font(None, 32)
        
        self.slicer_pos = pygame.Vector2(0, 0)
        self.slicer_trail = []
        self.objects = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.bomb_hits = 0
        self.game_over = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.spawn_timer = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.slicer_pos = pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.slicer_trail.clear()
        self.objects.clear()
        self.particles.clear()
        
        self.steps = 0
        self.score = 0
        self.bomb_hits = 0
        self.game_over = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.spawn_timer = self.SPAWN_INTERVAL
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.clock.tick(self.FPS)
        
        reward = 0.0
        if not self.game_over:
            self.steps += 1
            
            prev_slicer_pos = self.slicer_pos.copy()
            self._handle_input(action)
            
            self._update_game_logic()
            
            reward = self._handle_slicing(prev_slicer_pos)
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE: reward += 100.0
            elif self.bomb_hits >= self.MAX_BOMB_HITS: reward -= 100.0
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        
        self.slicer_trail.append(self.slicer_pos.copy())
        if len(self.slicer_trail) > self.SLICER_TRAIL_LENGTH:
            self.slicer_trail.pop(0)

        if movement == 1: self.slicer_pos.y -= self.SLICER_SPEED
        elif movement == 2: self.slicer_pos.y += self.SLICER_SPEED
        elif movement == 3: self.slicer_pos.x -= self.SLICER_SPEED
        elif movement == 4: self.slicer_pos.x += self.SLICER_SPEED
        
        self.slicer_pos.x = np.clip(self.slicer_pos.x, 0, self.SCREEN_WIDTH)
        self.slicer_pos.y = np.clip(self.slicer_pos.y, 0, self.SCREEN_HEIGHT)

    def _update_game_logic(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.fall_speed += self.FALL_SPEED_INCREMENT
            
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self._spawn_object()
            self.spawn_timer = self.np_random.integers(self.SPAWN_INTERVAL - 5, self.SPAWN_INTERVAL + 5)
            
        for obj in self.objects:
            obj['pos'].y += self.fall_speed
            
        self.objects = [obj for obj in self.objects if obj['pos'].y < self.SCREEN_HEIGHT + self.OBJECT_RADIUS * 2]
        
        self._update_particles()

    def _handle_slicing(self, prev_pos):
        reward = 0.0
        sliced_indices = []
        if prev_pos.distance_to(self.slicer_pos) > 1:
            for i, obj in enumerate(self.objects):
                if self._line_segment_circle_collision(prev_pos, self.slicer_pos, obj['pos'], self.OBJECT_RADIUS):
                    if i not in sliced_indices:
                        sliced_indices.append(i)
        
        for i in sorted(sliced_indices, reverse=True):
            obj = self.objects.pop(i)
            if obj['type'] == 'fruit':
                # SFX: Fruit slice
                self.score += 10
                reward += 1.0
                self._create_fruit_particles(obj['pos'], obj['color'])
            elif obj['type'] == 'bomb':
                # SFX: Explosion
                self.score = max(0, self.score - 50)
                self.bomb_hits += 1
                reward -= 5.0
                self._create_explosion_particles(obj['pos'])
        return reward

    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE
            or self.bomb_hits >= self.MAX_BOMB_HITS
            or self.steps >= self.MAX_STEPS
        )

    def _spawn_object(self):
        obj_type = 'fruit' if self.np_random.random() > 0.25 else 'bomb'
        pos = pygame.Vector2(self.np_random.integers(self.OBJECT_RADIUS, self.SCREEN_WIDTH - self.OBJECT_RADIUS), -self.OBJECT_RADIUS)
        
        if obj_type == 'fruit':
            fruit_type = self.np_random.choice(['apple', 'banana', 'grape'])
            color = {'apple': self.COLOR_APPLE, 'banana': self.COLOR_BANANA, 'grape': self.COLOR_GRAPE}[fruit_type]
            self.objects.append({'type': 'fruit', 'pos': pos, 'variant': fruit_type, 'color': color})
        else:
            self.objects.append({'type': 'bomb', 'pos': pos, 'color': self.COLOR_BOMB})

    def _line_segment_circle_collision(self, p1, p2, circle_center, radius):
        num_checks = 5
        for i in range(num_checks + 1):
            t = i / num_checks
            point_on_line = p1.lerp(p2, t)
            if point_on_line.distance_to(circle_center) < radius:
                return True
        return False

    def _create_fruit_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'color': color,
                'lifespan': self.np_random.integers(15, 30),
                'type': 'fruit_splash'
            })

    def _create_explosion_particles(self, pos):
        self.particles.append({'type': 'flash', 'pos': pos.copy(), 'lifespan': 10})
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            self.particles.append({
                'type': 'smoke',
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'lifespan': self.np_random.integers(20, 40),
                'radius': self.np_random.integers(5, 10)
            })

    def _update_particles(self):
        for p in self.particles:
            p['lifespan'] -= 1
            if p['type'] != 'flash':
                p['pos'] += p['vel']
                p['vel'] *= 0.95
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        
    def _get_observation(self):
        self._render_background()
        self._render_objects()
        self._render_particles()
        self._render_slicer()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_TOP)
        height = self.SCREEN_HEIGHT
        for y in range(height):
            ratio = y / height
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y), 2)


    def _render_objects(self):
        for obj in self.objects:
            pos = (int(obj['pos'].x), int(obj['pos'].y))
            if obj['type'] == 'fruit':
                color = obj['color']
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.OBJECT_RADIUS, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.OBJECT_RADIUS, color)
            elif obj['type'] == 'bomb':
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.OBJECT_RADIUS, self.COLOR_BOMB)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.OBJECT_RADIUS, self.COLOR_BOMB)
                fuse_pos = (pos[0] + 15, pos[1] - 15)
                pygame.draw.line(self.screen, self.COLOR_FUSE, (pos[0]+10, pos[1]-10), fuse_pos, 3)
                spark_color = self.COLOR_SPARK if self.steps % 10 < 5 else (255, 165, 0)
                pygame.draw.circle(self.screen, spark_color, fuse_pos, 3)

    def _render_slicer(self):
        if len(self.slicer_trail) > 1:
            points = [(int(p.x), int(p.y)) for p in self.slicer_trail]
            for i in range(len(points) - 1):
                alpha = int(255 * (i / self.SLICER_TRAIL_LENGTH))
                color = (*self.COLOR_SLICER, alpha)
                start_point = points[i]
                end_point = points[i+1]
                # pygame.gfxdraw.line doesn't support alpha, so we draw on a temp surface
                line_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(line_surf, color, start_point, end_point, 3 + i//2)
                self.screen.blit(line_surf, (0,0))

    def _render_particles(self):
        surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for p in self.particles:
            alpha = max(0, int(255 * (p['lifespan'] / 30.0)))
            if p['type'] == 'flash':
                radius = int(80 * (1 - p['lifespan'] / 10.0))
                flash_alpha = max(0, int(200 * (p['lifespan'] / 10.0)))
                pygame.draw.circle(surf, (255, 255, 200, flash_alpha), (int(p['pos'].x), int(p['pos'].y)), radius)
            elif p['type'] == 'smoke':
                smoke_alpha = max(0, int(128 * (p['lifespan'] / 40.0)))
                pygame.draw.circle(surf, (*(80,80,80), smoke_alpha), p['pos'], p['radius'])
            elif p['type'] == 'fruit_splash':
                pygame.draw.rect(surf, (*p['color'], alpha), (*p['pos'], 4, 4))
        self.screen.blit(surf, (0,0))

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        for i in range(self.MAX_BOMB_HITS):
            pos = (self.SCREEN_WIDTH - 40 - i * 40, 10)
            pygame.gfxdraw.filled_circle(self.screen, pos[0] + 15, pos[1] + 15, 15, self.COLOR_BOMB)
            if i < self.bomb_hits:
                pygame.draw.line(self.screen, (255, 0, 0), (pos[0], pos[1]), (pos[0] + 30, pos[1] + 30), 4)
                pygame.draw.line(self.screen, (255, 0, 0), (pos[0] + 30, pos[1]), (pos[0], pos[1] + 30), 4)
                
    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        msg = "VICTORY!" if self.score >= self.WIN_SCORE else "GAME OVER"
        text = self.font_large.render(msg, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)
        
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
    
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