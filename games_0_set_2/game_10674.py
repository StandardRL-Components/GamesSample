import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:50:49.979628
# Source Brief: brief_00674.md
# Brief Index: 674
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
    
    game_description = (
        "Swing a pendulum to collect morphing orbs for points while avoiding hazardous obstacles. "
        "Adjust the swing's width to navigate the screen and maximize your score before time runs out."
    )
    user_guide = "Use the ← and → arrow keys to control the width of the pendulum's swing."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_SCORE = 50
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS

        # --- Colors ---
        self.COLOR_BG_TOP = (10, 15, 30)
        self.COLOR_BG_BOTTOM = (30, 10, 40)
        self.COLOR_PENDULUM = (255, 255, 255)
        self.COLOR_ORB_CIRCLE = (100, 150, 255)
        self.COLOR_ORB_SQUARE = (100, 255, 150)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)

        # --- Physics & Gameplay ---
        self.PENDULUM_PIVOT = (self.WIDTH // 2, 60)
        self.PENDULUM_LENGTH = 180
        self.PENDULUM_BOB_RADIUS = 12
        self.PENDULUM_FREQUENCY = 0.75  # Slower 0.75Hz swing is more manageable
        self.AMPLITUDE_CHANGE = 0.04
        self.MIN_AMPLITUDE = 0.1
        self.MAX_AMPLITUDE = math.pi / 2.1
        self.ORB_RADIUS = 15
        self.ORB_MORPH_TIME = 2 * self.FPS # 2 seconds

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
        self.font_large = pygame.font.SysFont('Consolas', 30, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 20)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.pendulum_amplitude = 0
        self.pendulum_angle = 0
        self.pendulum_bob_pos = (0, 0)
        self.orbs = []
        self.obstacles = []
        self.particles = []
        self.obstacle_spawn_flags = {'20': False, '40': False}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        
        self.pendulum_amplitude = math.pi / 4
        self.pendulum_angle = 0
        self.pendulum_bob_pos = self._calculate_bob_pos()
        
        self.orbs = []
        self.obstacles = []
        self.particles = []
        self.obstacle_spawn_flags = {'20': False, '40': False}

        for _ in range(3):
            self._spawn_orb()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_action(action)
        
        self.steps += 1
        self.time_remaining -= 1
        
        self._update_pendulum()
        self._update_orbs()
        self._update_obstacles()
        self._update_particles()
        
        reward, collected_orb = self._handle_collisions()
        
        if collected_orb:
            # sfx: orb_collect.wav
            self._spawn_orb()
        
        self._check_difficulty_scaling()
        
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        if terminated:
            self.game_over = True
            if self.score >= self.MAX_SCORE:
                # sfx: win_sound.wav
                pass
            else:
                # sfx: lose_sound.wav
                pass

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.pendulum_amplitude -= self.AMPLITUDE_CHANGE
        elif movement == 4:  # Right
            self.pendulum_amplitude += self.AMPLITUDE_CHANGE
        
        self.pendulum_amplitude = np.clip(
            self.pendulum_amplitude, self.MIN_AMPLITUDE, self.MAX_AMPLITUDE
        )

    def _update_pendulum(self):
        time = self.steps / self.FPS
        self.pendulum_angle = self.pendulum_amplitude * math.sin(
            2 * math.pi * self.PENDULUM_FREQUENCY * time
        )
        self.pendulum_bob_pos = self._calculate_bob_pos()

    def _calculate_bob_pos(self):
        px, py = self.PENDULUM_PIVOT
        x = px + self.PENDULUM_LENGTH * math.sin(self.pendulum_angle)
        y = py + self.PENDULUM_LENGTH * math.cos(self.pendulum_angle)
        return x, y

    def _update_orbs(self):
        for orb in self.orbs:
            orb['morph_timer'] = (orb['morph_timer'] + 1) % self.ORB_MORPH_TIME
            if orb['morph_timer'] == 0:
                if orb['type'] == 'circle':
                    orb['type'] = 'square'
                    orb['value'] = 10
                else:
                    orb['type'] = 'circle'
                    orb['value'] = 5

    def _update_obstacles(self):
        time = self.steps / self.FPS
        for obs in self.obstacles:
            oscillation = obs['range'] * math.sin(obs['speed'] * time)
            if obs['axis'] == 'h':
                obs['rect'].centerx = obs['center'][0] + oscillation
            else:
                obs['rect'].centery = obs['center'][1] + oscillation

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1

    def _handle_collisions(self):
        reward = 0
        collected_orb_this_step = False
        
        # Orb collisions
        for orb in self.orbs[:]:
            dist = math.hypot(
                self.pendulum_bob_pos[0] - orb['pos'][0],
                self.pendulum_bob_pos[1] - orb['pos'][1]
            )
            if dist < self.PENDULUM_BOB_RADIUS + self.ORB_RADIUS:
                self.score += orb['value']
                reward += 0.1 if orb['type'] == 'circle' else 0.2
                color = self.COLOR_ORB_CIRCLE if orb['type'] == 'circle' else self.COLOR_ORB_SQUARE
                self._spawn_particles(orb['pos'], color, 20)
                self.orbs.remove(orb)
                collected_orb_this_step = True
        
        # Obstacle collisions
        for obs in self.obstacles:
            px, py = self.pendulum_bob_pos
            if obs['rect'].collidepoint(px, py):
                self.game_over = True
                # sfx: obstacle_hit.wav
                self._spawn_particles(self.pendulum_bob_pos, self.COLOR_OBSTACLE, 40)
                return -100, collected_orb_this_step
        
        return reward, collected_orb_this_step

    def _check_difficulty_scaling(self):
        if self.score >= 20 and not self.obstacle_spawn_flags['20']:
            self._spawn_obstacle('h')
            self.obstacle_spawn_flags['20'] = True
        if self.score >= 40 and not self.obstacle_spawn_flags['40']:
            self._spawn_obstacle('v')
            self.obstacle_spawn_flags['40'] = True

    def _check_termination(self):
        if self.score >= self.MAX_SCORE:
            return True, 100.0
        if self.time_remaining <= 0:
            return True, -50.0
        if self.game_over: # from obstacle collision
            return True, -100.0
        return False, 0.0

    def _spawn_orb(self):
        while True:
            x = self.np_random.uniform(50, self.WIDTH - 50)
            y = self.np_random.uniform(100, self.HEIGHT - 50)
            
            # Ensure it's not too close to the pivot
            if math.hypot(x - self.PENDULUM_PIVOT[0], y - self.PENDULUM_PIVOT[1]) < 100:
                continue
            
            # Ensure it's not inside an obstacle's path
            valid = True
            for obs in self.obstacles:
                if obs['rect'].inflate(self.ORB_RADIUS*2, self.ORB_RADIUS*2).collidepoint(x, y):
                    valid = False
                    break
            if valid:
                break

        orb_type = 'circle' if self.np_random.random() > 0.5 else 'square'
        self.orbs.append({
            'pos': (x, y),
            'type': orb_type,
            'value': 5 if orb_type == 'circle' else 10,
            'morph_timer': self.np_random.integers(0, self.ORB_MORPH_TIME),
        })

    def _spawn_obstacle(self, axis):
        if axis == 'h':
            rect = pygame.Rect(0, 0, 150, 20)
            rect.centery = self.np_random.uniform(150, self.HEIGHT - 100)
            center_x = self.WIDTH / 2
            obs_range = self.np_random.uniform(50, 150)
            speed = self.np_random.uniform(1.5, 2.5)
            self.obstacles.append({
                'rect': rect, 'axis': 'h', 'speed': speed, 'range': obs_range, 'center': (center_x, rect.centery)
            })
        else: # 'v'
            rect = pygame.Rect(0, 0, 20, 150)
            rect.centerx = self.np_random.choice([self.WIDTH * 0.25, self.WIDTH * 0.75])
            center_y = self.HEIGHT / 2 + 20
            obs_range = self.np_random.uniform(50, 100)
            speed = self.np_random.uniform(1.5, 2.5)
            self.obstacles.append({
                'rect': rect, 'axis': 'v', 'speed': speed, 'range': obs_range, 'center': (rect.centerx, center_y)
            })

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        self._render_particles()
        self._render_obstacles()
        self._render_orbs()
        self._render_pendulum()

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 40))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

    def _render_obstacles(self):
        for obs in self.obstacles:
            self._draw_glow_rect(self.screen, obs['rect'], self.COLOR_OBSTACLE, 10)

    def _render_orbs(self):
        for orb in self.orbs:
            self._draw_morphing_orb(self.screen, orb)

    def _render_pendulum(self):
        px, py = self.PENDULUM_PIVOT
        bx, by = self.pendulum_bob_pos
        
        # Rod
        pygame.draw.aaline(self.screen, self.COLOR_PENDULUM, (px, py), (bx, by), 2)
        
        # Pivot
        self._draw_glow_circle(self.screen, (int(px), int(py)), 6, self.COLOR_PENDULUM, 10)
        
        # Bob
        self._draw_glow_circle(self.screen, (int(bx), int(by)), self.PENDULUM_BOB_RADIUS, self.COLOR_PENDULUM, 15)

    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        time_text = f"TIME: {self.time_remaining / self.FPS:.1f}"
        
        self._draw_text(score_text, (20, 15), self.font_large)
        self._draw_text(time_text, (self.WIDTH - 150, 15), self.font_large)

    def _draw_text(self, text, pos, font):
        shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
        surface = font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(surface, pos)

    def _draw_glow_circle(self, surface, pos, radius, color, glow_size):
        for i in range(glow_size, 0, -2):
            alpha = 100 * (1 - (i / glow_size))
            c = (*color, alpha)
            temp_surf = pygame.Surface((radius*2 + i*2, radius*2 + i*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, c, (radius + i, radius + i), radius + i)
            surface.blit(temp_surf, (pos[0] - radius - i, pos[1] - radius - i))
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)

    def _draw_glow_rect(self, surface, rect, color, glow_size):
        for i in range(glow_size, 0, -2):
            alpha = 80 * (1 - (i / glow_size))
            c = (*color, alpha)
            glow_rect = rect.inflate(i, i)
            temp_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, c, temp_surf.get_rect(), border_radius=5)
            surface.blit(temp_surf, glow_rect.topleft)
        pygame.draw.rect(surface, color, rect, border_radius=3)

    def _draw_morphing_orb(self, surface, orb):
        morph_progress = abs(orb['morph_timer'] - self.ORB_MORPH_TIME / 2) / (self.ORB_MORPH_TIME / 2)
        
        if orb['type'] == 'square':
            morph_progress = 1 - morph_progress
            
        color = tuple(
            int(c1 * morph_progress + c2 * (1 - morph_progress))
            for c1, c2 in zip(self.COLOR_ORB_CIRCLE, self.COLOR_ORB_SQUARE)
        )
        
        x, y = orb['pos']
        r = self.ORB_RADIUS
        
        # Calculate squircle points
        points = []
        for i in range(16):
            angle = i * (2 * math.pi / 16)
            
            # Circle point
            cx = x + r * math.cos(angle)
            cy = y + r * math.sin(angle)
            
            # Square point
            ca, sa = math.cos(angle), math.sin(angle)
            if ca == 0 or sa == 0:
                 sx, sy = x + r * ca, y + r * sa
            else:
                sx = x + r * (1 / max(abs(ca), abs(sa))) * ca
                sy = y + r * (1 / max(abs(ca), abs(sa))) * sa

            # Interpolate
            px = cx * morph_progress + sx * (1 - morph_progress)
            py = cy * morph_progress + sy * (1 - morph_progress)
            points.append((int(px), int(py)))

        # Draw glow
        for i in range(10, 0, -2):
            alpha = 100 * (1 - (i / 10))
            glow_color = (*color, alpha)
            glow_points = [(p[0] - x, p[1] - y) for p in points]
            
            scale = (r + i) / r
            glow_points = [(gp[0] * scale + x, gp[1] * scale + y) for gp in glow_points]
            
            pygame.gfxdraw.aapolygon(surface, glow_points, glow_color)
            pygame.gfxdraw.filled_polygon(surface, glow_points, glow_color)

        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": self.time_remaining / self.FPS,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual play and is not part of the gym environment
    # It will not be used in the evaluation.
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'mac', etc.
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pendulum Orb Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_r]:
            obs, info = env.reset()
            total_reward = 0
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()