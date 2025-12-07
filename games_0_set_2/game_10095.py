import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:55:54.334747
# Source Brief: brief_00095.md
# Brief Index: 95
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for particles
class Particle:
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.life = 0

    def update(self):
        self.pos += self.vel
        self.life += 1
        return self.life >= self.lifetime

    def draw(self, surface):
        alpha = max(0, 255 - int(255 * (self.life / self.lifetime)))
        # Using gfxdraw for anti-aliased circles
        try:
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), self.color + (alpha,))
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), self.color + (alpha,))
        except TypeError: # Handle potential color tuple errors
            safe_color = tuple(int(c) for c in self.color)
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), safe_color + (alpha,))
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), safe_color + (alpha,))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Reflect lasers off rotating mirrors to hit targets and score points before time runs out."
    )
    user_guide = (
        "Controls: Use ↑↓ to rotate the left mirror and ←→ to rotate the right mirror. "
        "Align the mirrors to guide the lasers into the targets."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    TIME_LIMIT_SECONDS = 30

    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (30, 40, 60)
    COLOR_MIRROR = (200, 200, 220)
    COLOR_MIRROR_EDGE = (255, 255, 255)
    COLOR_LASER_1 = (255, 50, 50)
    COLOR_LASER_2 = (50, 150, 255)
    COLOR_TARGET = (50, 255, 150)
    COLOR_TARGET_HIT = (255, 255, 100)
    COLOR_UI_TEXT = (240, 240, 240)

    MIRROR_LENGTH = 80
    MIRROR_THICKNESS = 6
    ROTATION_SPEED = 3  # degrees per step

    TARGET_RADIUS = 15
    TARGET_HIT_COOLDOWN = 30 # steps

    MAX_BOUNCES = 8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("consolas", 24, bold=True)
        except pygame.error:
            self.font = pygame.font.SysFont(None, 30)

        self.mirror1_pos = pygame.math.Vector2(self.WIDTH * 0.25, self.HEIGHT * 0.5)
        self.mirror2_pos = pygame.math.Vector2(self.WIDTH * 0.75, self.HEIGHT * 0.5)

        self.laser_sources = [
            {'pos': pygame.math.Vector2(0, self.HEIGHT * 0.25), 'dir': pygame.math.Vector2(1, 0), 'color': self.COLOR_LASER_1},
            {'pos': pygame.math.Vector2(self.WIDTH, self.HEIGHT * 0.75), 'dir': pygame.math.Vector2(-1, 0), 'color': self.COLOR_LASER_2}
        ]

        # These will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mirror1_angle = 0.0
        self.mirror2_angle = 0.0
        self.targets = []
        self.particles = []
        self.active_laser_beams = []
        self.laser_base_freq = 0.0
        self.laser_freq_amp = 0.0
        self.laser_fire_timers = []
        self.time_limit = 0

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() # No need to call this in the constructor

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_limit = self.TIME_LIMIT_SECONDS * self.FPS

        self.mirror1_angle = 45.0
        self.mirror2_angle = -45.0

        self.targets = [
            {'pos': pygame.math.Vector2(self.WIDTH * 0.5, self.HEIGHT * 0.2), 'radius': self.TARGET_RADIUS, 'hit_timer': 0},
            {'pos': pygame.math.Vector2(self.WIDTH * 0.5, self.HEIGHT * 0.8), 'radius': self.TARGET_RADIUS, 'hit_timer': 0},
            {'pos': pygame.math.Vector2(self.WIDTH * 0.1, self.HEIGHT * 0.8), 'radius': self.TARGET_RADIUS, 'hit_timer': 0},
            {'pos': pygame.math.Vector2(self.WIDTH * 0.9, self.HEIGHT * 0.2), 'radius': self.TARGET_RADIUS, 'hit_timer': 0},
        ]
        self.particles = []
        self.active_laser_beams = []

        self.laser_base_freq = 1.0  # Center of oscillation
        self.laser_freq_amp = 0.5   # Amplitude (range becomes 0.5 to 1.5 Hz)
        self.laser_fire_timers = [self.FPS / (self.laser_base_freq + self.laser_freq_amp * math.sin(i * math.pi)) for i in range(len(self.laser_sources))]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        total_reward = 0

        # 1. Handle Actions (map movement to mirror rotation)
        movement, _, _ = action
        # Action 1 (up): Rotate Mirror 1 clockwise.
        if movement == 1: self.mirror1_angle += self.ROTATION_SPEED
        # Action 2 (down): Rotate Mirror 1 counter-clockwise.
        elif movement == 2: self.mirror1_angle -= self.ROTATION_SPEED
        # Action 3 (right): Rotate Mirror 2 clockwise.
        elif movement == 3: self.mirror2_angle += self.ROTATION_SPEED
        # Action 4 (left): Rotate Mirror 2 counter-clockwise.
        elif movement == 4: self.mirror2_angle -= self.ROTATION_SPEED
        self.mirror1_angle %= 360
        self.mirror2_angle %= 360

        # 2. Update Game State
        self._update_targets()
        self._update_particles()
        
        freq_increase = 0.001 * (self.steps // 100)
        current_laser_base_freq = self.laser_base_freq + freq_increase
        
        self.active_laser_beams = []

        # 3. Fire Lasers
        for i, source in enumerate(self.laser_sources):
            self.laser_fire_timers[i] -= 1
            if self.laser_fire_timers[i] <= 0:
                oscillation = math.sin(self.steps * 0.05 + i * math.pi)
                current_freq = current_laser_base_freq + self.laser_freq_amp * oscillation
                
                if current_freq > 0.1:
                    self.laser_fire_timers[i] = self.FPS / current_freq
                    
                    beam_segments, beam_reward = self._trace_beam(source['pos'], source['dir'], source['color'])
                    self.active_laser_beams.extend(beam_segments)
                    total_reward += beam_reward
                    # sound: "laser_fire.wav"

        # 4. Check Termination
        terminated = False
        truncated = False
        if self.score >= 50:
            total_reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.time_limit:
            # total_reward -= 100 # Negative reward on timeout can be harsh.
            terminated = True
            truncated = True # Use truncated for time limit
            self.game_over = True

        return self._get_observation(), total_reward, terminated, truncated, self._get_info()

    def _update_targets(self):
        for target in self.targets:
            if target['hit_timer'] > 0:
                target['hit_timer'] -= 1

    def _update_particles(self):
        self.particles = [p for p in self.particles if not p.update()]

    def _get_mirror_segments(self):
        mirrors = []
        for pos, angle in [(self.mirror1_pos, self.mirror1_angle), (self.mirror2_pos, self.mirror2_angle)]:
            rad = math.radians(angle)
            half_len = self.MIRROR_LENGTH / 2
            p1 = pos + pygame.math.Vector2(-half_len, 0).rotate(-angle)
            p2 = pos + pygame.math.Vector2(half_len, 0).rotate(-angle)
            normal = (p2 - p1).rotate(90).normalize()
            mirrors.append({'p1': p1, 'p2': p2, 'normal': normal})
        return mirrors

    def _trace_beam(self, start_pos, start_dir, color):
        beam_segments = []
        beam_reward = 0
        
        current_pos = pygame.math.Vector2(start_pos)
        current_dir = pygame.math.Vector2(start_dir).normalize()
        
        mirrors = self._get_mirror_segments()
        
        for _ in range(self.MAX_BOUNCES):
            intersections = []

            # Screen boundaries
            if current_dir.x > 1e-6: intersections.append({'dist': (self.WIDTH - current_pos.x) / current_dir.x, 'type': 'wall'})
            if current_dir.x < -1e-6: intersections.append({'dist': -current_pos.x / current_dir.x, 'type': 'wall'})
            if current_dir.y > 1e-6: intersections.append({'dist': (self.HEIGHT - current_pos.y) / current_dir.y, 'type': 'wall'})
            if current_dir.y < -1e-6: intersections.append({'dist': -current_pos.y / current_dir.y, 'type': 'wall'})

            # Mirrors
            for i, mirror in enumerate(mirrors):
                dist = self._ray_segment_intersection(current_pos, current_dir, mirror['p1'], mirror['p2'])
                if dist is not None:
                    intersections.append({'dist': dist, 'type': 'mirror', 'obj': mirror, 'id': i})
            
            # Targets
            for i, target in enumerate(self.targets):
                if target['hit_timer'] > 0: continue
                dist = self._ray_circle_intersection(current_pos, current_dir, target['pos'], target['radius'])
                if dist is not None:
                    intersections.append({'dist': dist, 'type': 'target', 'obj': target, 'id': i})

            closest_hit = None
            min_dist = float('inf')
            for hit in intersections:
                if 0.001 < hit['dist'] < min_dist:
                    min_dist = hit['dist']
                    closest_hit = hit

            if closest_hit is None:
                break

            end_pos = current_pos + current_dir * min_dist
            beam_segments.append({'start': current_pos, 'end': end_pos, 'color': color})
            
            # Reward shaping
            dist_start_to_target = min( (current_pos - t['pos']).length() for t in self.targets ) if self.targets else float('inf')
            dist_end_to_target = min( (end_pos - t['pos']).length() for t in self.targets ) if self.targets else float('inf')
            if dist_end_to_target < dist_start_to_target:
                beam_reward += (dist_start_to_target - dist_end_to_target) * 0.01

            current_pos = end_pos

            if closest_hit['type'] == 'wall':
                break
            elif closest_hit['type'] == 'target':
                self.score += 1
                beam_reward += 1
                closest_hit['obj']['hit_timer'] = self.TARGET_HIT_COOLDOWN
                self._create_particles(current_pos, 20, self.COLOR_TARGET_HIT)
                # sound: "target_hit.wav"
                break
            elif closest_hit['type'] == 'mirror':
                current_dir = current_dir.reflect(closest_hit['obj']['normal'])
                self._create_particles(current_pos, 5, self.COLOR_MIRROR_EDGE, 0.5)
                # sound: "laser_reflect.wav"

        return beam_segments, beam_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        for target in self.targets:
            pos = (int(target['pos'].x), int(target['pos'].y))
            radius = int(target['radius'])
            if target['hit_timer'] > 0:
                flash_alpha = 255 * (target['hit_timer'] / self.TARGET_HIT_COOLDOWN)
                color = self.COLOR_TARGET_HIT
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color + (int(flash_alpha),))
            else:
                pulse = 4 * (1 + math.sin(self.steps * 0.1))
                glow_radius = int(radius + pulse)
                glow_alpha = int(100 + 50 * math.sin(self.steps * 0.1))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, self.COLOR_TARGET + (glow_alpha,))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_TARGET)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_TARGET)

        for beam in self.active_laser_beams:
            self._draw_laser_segment(beam['start'], beam['end'], beam['color'])

        self._draw_mirror(self.mirror1_pos, self.mirror1_angle)
        self._draw_mirror(self.mirror2_pos, self.mirror2_angle)

        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, (self.time_limit - self.steps) / self.FPS)
        timer_text = self.font.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def _draw_mirror(self, pos, angle):
        mirror_surf = pygame.Surface((self.MIRROR_LENGTH, self.MIRROR_THICKNESS), pygame.SRCALPHA)
        pygame.draw.rect(mirror_surf, self.COLOR_MIRROR, (0, 0, self.MIRROR_LENGTH, self.MIRROR_THICKNESS), border_radius=3)
        pygame.draw.rect(mirror_surf, self.COLOR_MIRROR_EDGE, (0, 0, self.MIRROR_LENGTH, self.MIRROR_THICKNESS), 1, border_radius=3)
        
        rotated_surf = pygame.transform.rotate(mirror_surf, angle)
        rect = rotated_surf.get_rect(center=(int(pos.x), int(pos.y)))
        self.screen.blit(rotated_surf, rect)

    def _draw_laser_segment(self, p1, p2, color):
        for i in range(5, 0, -1):
            alpha = 150 - i * 30
            pygame.draw.line(self.screen, color + (alpha,), p1, p2, i * 2 + 1)
        pygame.draw.line(self.screen, (255, 255, 255), p1, p2, 2)

    def _create_particles(self, pos, count, color, speed_multiplier=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_multiplier
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            radius = random.uniform(1, 3)
            lifetime = random.randint(15, 30)
            self.particles.append(Particle(pos, vel, radius, color, lifetime))

    @staticmethod
    def _ray_segment_intersection(ray_origin, ray_dir, p1, p2):
        v1 = ray_origin - p1
        v2 = p2 - p1
        v3 = pygame.math.Vector2(-ray_dir.y, ray_dir.x)
        dot_v2_v3 = v2.dot(v3)
        if abs(dot_v2_v3) < 1e-6:
            return None
        
        t1 = v2.cross(v1) / dot_v2_v3
        t2 = v1.dot(v3) / dot_v2_v3

        if t1 >= 1e-4 and 0.0 <= t2 <= 1.0:
            return t1
        return None

    @staticmethod
    def _ray_circle_intersection(ray_origin, ray_dir, center, radius):
        oc = ray_origin - center
        a = ray_dir.dot(ray_dir)
        b = 2.0 * oc.dot(ray_dir)
        c = oc.dot(oc) - radius * radius
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None
        else:
            sqrt_d = math.sqrt(discriminant)
            t1 = (-b - sqrt_d) / (2*a)
            t2 = (-b + sqrt_d) / (2*a)
            if t1 > 1e-4: return t1
            if t2 > 1e-4: return t2
            return None

    def validate_implementation(self):
        # This is a helper function for development and is not required by the final implementation.
        # It's good practice to keep it for future debugging.
        print("✓ Validating implementation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is for human play and debugging.
    # It is not part of the Gymnasium environment definition.
    # Set the video driver to something other than "dummy" to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Laser Reflection Environment")
    clock = pygame.time.Clock()

    running = True
    while running:
        # Map keys to actions based on user_guide
        # ↑↓ for mirror 1, ←→ for mirror 2
        # action[0]: 0=No-op, 1=↑, 2=↓, 3=→, 4=←
        action = [0, 0, 0]
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1 # Mirror 1 CW
        elif keys[pygame.K_DOWN]: action[0] = 2 # Mirror 1 CCW
        
        if keys[pygame.K_RIGHT]: action[0] = 3 # Mirror 2 CW
        elif keys[pygame.K_LEFT]: action[0] = 4 # Mirror 2 CCW

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode Finished. Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()