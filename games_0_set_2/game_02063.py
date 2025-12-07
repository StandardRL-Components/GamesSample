
# Generated: 2025-08-28T03:34:21.194640
# Source Brief: brief_02063.md
# Brief Index: 2063

        
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
        "Controls: Press space to jump. Time your jumps with the pulsating beat indicator for extra points and a higher jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced rhythm platformer. Jump over obstacles to the beat, aiming for a perfect run. Miss three times and it's game over."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.BPM = 120
        self.FRAMES_PER_BEAT = (60 / self.BPM) * self.FPS
        self.JUMP_WINDOW = 3  # Frames around the beat for a "good" jump

        self.COURSE_LENGTH = 8000
        self.MAX_STEPS = 2000
        self.MAX_MISSES = 3

        self.PLAYER_X_POS = 100
        self.GROUND_Y = self.HEIGHT - 50
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15

        # --- Colors ---
        self.COLOR_BG_TOP = (20, 30, 50)
        self.COLOR_BG_BOTTOM = (40, 60, 90)
        self.COLOR_GROUND = (10, 20, 30)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_OBSTACLE = (255, 80, 120)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_BEAT_GOOD = (0, 255, 128)
        self.COLOR_BEAT_OKAY = (255, 255, 0)
        self.COLOR_PARTICLE_HIT = (255, 100, 100)

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("sans-serif", 48, bold=True)
        self.font_small = pygame.font.SysFont("sans-serif", 20, bold=True)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_y = 0
        self.player_vy = 0
        self.on_ground = True
        self.misses = 0
        self.camera_x = 0
        self.obstacle_speed = 0
        self.beat_timer = 0
        self.obstacles = []
        self.particles = []
        self.prev_space_held = False
        self.next_obstacle_spawn_pos = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.on_ground = True
        self.misses = 0
        self.camera_x = 0
        self.obstacle_speed = 4.0
        self.beat_timer = 0
        self.obstacles = []
        self.particles = []
        self.prev_space_held = False
        self.next_obstacle_spawn_pos = self.WIDTH

        return self._get_observation(), self._get_info()

    def step(self, action):
        space_held = action[1] == 1
        space_pressed = space_held and not self.prev_space_held
        
        reward = -0.1
        terminated = False

        # 1. Update Beat Timer
        self.beat_timer = (self.beat_timer + 1) % self.FRAMES_PER_BEAT

        # 2. Player Physics and Input
        if not self.on_ground:
            self.player_vy += self.GRAVITY
        
        if space_pressed and self.on_ground:
            time_from_beat = min(self.beat_timer, self.FRAMES_PER_BEAT - self.beat_timer)
            if time_from_beat <= self.JUMP_WINDOW:
                self.player_vy = self.JUMP_STRENGTH
                self.on_ground = False
                reward += 1.0
                self._spawn_particles(self.PLAYER_X_POS, self.GROUND_Y, 20, self.COLOR_PLAYER, 'up')
                # sfx: good_jump.wav
            else:
                self.player_vy = self.JUMP_STRENGTH * 0.7
                self.on_ground = False
                # sfx: bad_jump.wav
        
        self.player_y += self.player_vy

        if self.player_y >= self.GROUND_Y:
            self.player_y = self.GROUND_Y
            self.player_vy = 0
            if not self.on_ground:
                self.on_ground = True
                self._spawn_particles(self.PLAYER_X_POS, self.GROUND_Y, 8, self.COLOR_GROUND, 'side')
                # sfx: land.wav
        
        # 3. World and Obstacle Update
        self.camera_x += self.obstacle_speed
        player_rect = pygame.Rect(self.PLAYER_X_POS - 10, self.player_y - 20, 20, 20)
        
        new_obstacles = []
        for obs in self.obstacles:
            obs['x'] -= self.obstacle_speed
            obs_rect = pygame.Rect(obs['x'], obs['y'], obs['w'], obs['h'])
            
            if not obs.get('collided', False) and player_rect.colliderect(obs_rect):
                self.misses += 1
                reward -= 5.0
                obs['collided'] = True
                self._spawn_particles(player_rect.centerx, player_rect.centery, 30, self.COLOR_PARTICLE_HIT, 'explosion')
                # sfx: hit.wav
            
            if obs['x'] + obs['w'] > 0:
                new_obstacles.append(obs)
        self.obstacles = new_obstacles

        # 4. Spawning
        if self.camera_x + self.WIDTH > self.next_obstacle_spawn_pos:
            self._spawn_obstacle()

        # 5. Effects and State
        self._update_particles()
        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_speed = min(8.0, self.obstacle_speed + 0.05)
        
        # 6. Termination
        if self.misses >= self.MAX_MISSES:
            terminated = True
            reward -= 100.0
        if self.camera_x >= self.COURSE_LENGTH:
            terminated = True
            reward += 50.0
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.steps += 1
        self.score += reward
        self.prev_space_held = space_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_obstacle(self):
        height = self.np_random.choice([20, 40, 60])
        width = 30
        x_pos = self.next_obstacle_spawn_pos
        y_pos = self.GROUND_Y - height
        
        self.obstacles.append({'x': x_pos, 'y': y_pos, 'w': width, 'h': height})
        
        gap = self.obstacle_speed * self.FRAMES_PER_BEAT * self.np_random.choice([2, 3])
        self.next_obstacle_spawn_pos += max(100, gap)

    def _spawn_particles(self, x, y, count, color, p_type):
        for _ in range(count):
            if p_type == 'explosion':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 8)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            elif p_type == 'up':
                angle = self.np_random.uniform(-math.pi * 0.8, -math.pi * 0.2)
                speed = self.np_random.uniform(1, 5)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            else: # 'side'
                angle = self.np_random.choice([self.np_random.uniform(math.pi*0.75, math.pi*1.25), self.np_random.uniform(-math.pi*0.25, math.pi*0.25)])
                speed = self.np_random.uniform(1, 3)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]

            self.particles.append({
                'pos': [x, y], 'vel': vel, 'radius': self.np_random.uniform(2, 5),
                'color': color, 'lifespan': self.np_random.integers(15, 30)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            p['radius'] -= 0.1

    def _get_observation(self):
        self._render_background()
        self._render_ground()
        self._render_obstacles()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        rect = pygame.Rect(0, 0, self.WIDTH, self.HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BG_TOP, rect)
        for y in range(self.HEIGHT):
            alpha = int(255 * (y / self.HEIGHT)**1.5)
            color = self.COLOR_BG_BOTTOM + (alpha,)
            s = pygame.Surface((self.WIDTH, 1), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (0, y))

    def _render_ground(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        line_height = self.HEIGHT - self.GROUND_Y
        for i in range(1, 6):
            y = self.GROUND_Y + (i / 6) * line_height
            offset = -(self.camera_x * (0.5 + i * 0.2)) % 100
            color = tuple(c * 1.5 for c in self.COLOR_GROUND[:3])
            for j in range(self.WIDTH // 100 + 2):
                start_pos = (j * 100 + offset, y)
                end_pos = (j * 100 + offset + 40, y)
                pygame.draw.line(self.screen, color, start_pos, end_pos, 2)

    def _render_player(self):
        player_rect = pygame.Rect(int(self.PLAYER_X_POS - 10), int(self.player_y - 20), 20, 20)
        points = [(player_rect.centerx, player_rect.top), (player_rect.right, player_rect.centery),
                  (player_rect.centerx, player_rect.bottom), (player_rect.left, player_rect.centery)]
        
        s = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER + (50,), (20, 20), 20)
        self.screen.blit(s, (player_rect.centerx - 20, player_rect.centery - 20), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_obstacles(self):
        for obs in self.obstacles:
            rect = (int(obs['x']), int(obs['y']), int(obs['w']), int(obs['h']))
            color = self.COLOR_OBSTACLE if not obs.get('collided') else (80, 20, 40)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 2)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(max(0, p['radius']))
            alpha = int(255 * (p['lifespan'] / 30))
            color = p['color'] + (alpha,)
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (radius, radius), radius)
            self.screen.blit(s, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        miss_str = "X " * self.misses + "O " * (self.MAX_MISSES - self.misses)
        miss_text = self.font_small.render(f"HITS: {miss_str.strip()}", True, self.COLOR_TEXT)
        self.screen.blit(miss_text, (self.WIDTH - miss_text.get_width() - 10, 10))

        progress = min(1.0, self.camera_x / self.COURSE_LENGTH)
        bar_width = self.WIDTH - 20
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (10, self.HEIGHT - 20, bar_width, 10), border_radius=5)
        if progress > 0:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, self.HEIGHT - 20, bar_width * progress, 10), border_radius=5)
        
        time_from_beat = min(self.beat_timer, self.FRAMES_PER_BEAT - self.beat_timer)
        pulse_factor = 1.0 - (time_from_beat / (self.FRAMES_PER_BEAT / 2))
        radius = 5 + 15 * (pulse_factor ** 2)
        color = self.COLOR_BEAT_GOOD if time_from_beat <= self.JUMP_WINDOW else self.COLOR_BEAT_OKAY
        center = (self.WIDTH // 2, self.GROUND_Y + 25)
        
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], int(radius), color)
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], int(radius * 0.7), color)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "COURSE COMPLETE!" if self.camera_x >= self.COURSE_LENGTH else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score, "steps": self.steps, "misses": self.misses,
            "progress": self.camera_x / self.COURSE_LENGTH, "obstacle_speed": self.obstacle_speed
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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