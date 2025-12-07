import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Navigate a gravity-flipping neon vortex, dodging hazards and using portals
    to survive and achieve higher speeds.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a gravity-flipping neon vortex, dodging hazards and using portals "
        "to survive and achieve higher speeds."
    )
    user_guide = (
        "Controls: ←→ to move along the ring, ↑↓ to flip between rings. "
        "Press space to use your shield."
    )
    auto_advance = True

    # Persistent state across resets
    persistent_player_speed_boost = 0.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000

        # Visuals
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PLAYER = (0, 192, 255)
        self.COLOR_HAZARD = (255, 50, 50)
        self.COLOR_PORTAL = (50, 255, 100)
        self.COLOR_TRACK = (150, 50, 255)
        self.COLOR_SHIELD = (255, 220, 0)
        self.COLOR_TEXT = (220, 220, 255)

        # Vortex track properties
        self.CENTER = (self.WIDTH // 2, self.HEIGHT // 2)
        self.OUTER_RADIUS = 160
        self.INNER_RADIUS = 120
        self.TRACK_WIDTH = self.OUTER_RADIUS - self.INNER_RADIUS

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_angle = 0.0
        self.player_on_outer_ring = True
        self.player_target_y = 0.0
        self.player_current_y = 0.0
        self.player_rotation = 0.0
        self.shields = 0
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown = 0
        self.gravity_flip_cooldown = 0
        self.last_space_held = False
        self.hazards = []
        self.portals = []
        self.particles = []
        self.base_player_speed = 2.0
        self.base_hazard_speed = 1.0
        self.hazard_spawn_rate = 500
        self.current_hazard_speed = 1.0
        self.current_hazard_count = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        # Player state
        self.player_angle = self.np_random.uniform(0, 360)
        self.player_on_outer_ring = True
        self.player_rotation = 0.0
        self.player_current_y = self.CENTER[1] - self.OUTER_RADIUS
        self.player_target_y = self.player_current_y

        # Abilities
        self.shields = 3
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown = 0

        # Cooldowns
        self.gravity_flip_cooldown = 0
        self.last_space_held = False

        # World state
        self.hazards = []
        self.portals = []
        self.particles = []
        self.current_hazard_speed = self.base_hazard_speed
        self.current_hazard_count = 2

        for _ in range(self.current_hazard_count):
            self._spawn_hazard(initial_spawn=True)
        for _ in range(2):
            self._spawn_portal()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Cooldowns
        if self.gravity_flip_cooldown > 0: self.gravity_flip_cooldown -= 1
        if self.shield_cooldown > 0: self.shield_cooldown -= 1

        # Movement
        if movement == 3: self.player_angle -= (self.base_player_speed + self.persistent_player_speed_boost)
        if movement == 4: self.player_angle += (self.base_player_speed + self.persistent_player_speed_boost)
        self.player_angle %= 360

        # Gravity Flip
        if self.gravity_flip_cooldown == 0:
            if movement == 1 and not self.player_on_outer_ring:
                self.player_on_outer_ring = True
                self.gravity_flip_cooldown = 15 # 0.5s cooldown
            elif movement == 2 and self.player_on_outer_ring:
                self.player_on_outer_ring = False
                self.gravity_flip_cooldown = 15 # 0.5s cooldown

        # Shield Activation (on button press)
        if space_held and not self.last_space_held and self.shields > 0 and not self.shield_active and self.shield_cooldown == 0:
            self.shield_active = True
            self.shield_timer = 90 # 3s duration
            self.shields -= 1
            for _ in range(50):
                self._spawn_particle(self._get_player_pos(), self.COLOR_SHIELD, 2.0, 30)

        self.last_space_held = space_held

        # --- Update Game State ---
        self._update_player()
        self._update_hazards()
        self._update_particles()

        # --- Collisions & Events ---
        collision_reward, terminated_by_collision = self._check_collisions()
        reward += collision_reward
        if terminated_by_collision:
            self.game_over = True

        # --- Difficulty Scaling & Progression ---
        if self.steps > 0:
            if self.steps % 200 == 0:
                self.current_hazard_speed += 0.05
            if self.steps % self.hazard_spawn_rate == 0:
                self.current_hazard_count += 1
                self._spawn_hazard()
            if self.steps % 500 == 0:
                GameEnv.persistent_player_speed_boost += 0.1

        # --- Termination Conditions ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False
        
        if self.game_over:
            reward = -100.0 # Death penalty
        elif self.steps >= self.MAX_STEPS:
            reward += 100.0 # Victory reward

        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_player(self):
        self.player_rotation = (self.player_rotation + 5) % 360
        target_radius = self.OUTER_RADIUS if self.player_on_outer_ring else self.INNER_RADIUS
        self.player_target_y = self.CENTER[1] - target_radius

        # Smooth interpolation for gravity flip
        self.player_current_y += (self.player_target_y - self.player_current_y) * 0.2

        if self.shield_active:
            self.shield_timer -= 1
            if self.shield_timer <= 0:
                self.shield_active = False
                self.shield_cooldown = 30 # 1s cooldown after use

    def _update_hazards(self):
        for hazard in self.hazards:
            hazard['angle'] = (hazard['angle'] + self.current_hazard_speed * hazard['dir']) % 360
            hazard['pulse'] = math.sin(self.steps * 0.1 + hazard['angle']) * 3 + 6

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _check_collisions(self):
        reward = 0
        terminated = False
        player_pos = self._get_player_pos()
        player_radius = 10

        # Player vs Hazards
        for hazard in self.hazards[:]:
            if hazard['on_outer_ring'] == self.player_on_outer_ring:
                h_pos = self._polar_to_cartesian(hazard['angle'], self._get_ring_radius(hazard['on_outer_ring']))
                dist = math.hypot(player_pos[0] - h_pos[0], player_pos[1] - h_pos[1])
                if dist < player_radius + hazard['pulse']:
                    if self.shield_active:
                        reward += 2.0 # Reward for using shield
                        self.hazards.remove(hazard)
                        for _ in range(30):
                            self._spawn_particle(h_pos, self.COLOR_HAZARD, 3.0, 40)
                        self._spawn_hazard() # Respawn one
                    else:
                        terminated = True
                        for _ in range(100):
                            self._spawn_particle(player_pos, self.COLOR_PLAYER, 4.0, 60)
                        return 0, True

        # Player vs Portals
        for portal in self.portals[:]:
            p_pos = self._polar_to_cartesian(portal['angle'], self._get_ring_radius(portal['on_outer_ring']))
            dist = math.hypot(player_pos[0] - p_pos[0], player_pos[1] - p_pos[1])
            if dist < player_radius + 15:
                reward += 5.0
                self.player_angle = self.np_random.uniform(0, 360)
                self.portals.remove(portal)
                for _ in range(50):
                    self._spawn_particle(p_pos, self.COLOR_PORTAL, 3.0, 50)
                self._spawn_portal()
                break # only one portal per frame
        
        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "shields": self.shields,
            "speed_boost": self.persistent_player_speed_boost
        }

    def _render_background(self):
        for i in range(18):
            angle = math.radians(i * 20 - self.player_angle)
            start_pos = self._polar_to_cartesian_center(angle, 50)
            end_pos = self._polar_to_cartesian_center(angle, self.WIDTH)
            pygame.draw.aaline(self.screen, (20, 15, 45), start_pos, end_pos)
        
        self._draw_neon_circle(self.CENTER, self.OUTER_RADIUS, self.COLOR_TRACK, 3)
        self._draw_neon_circle(self.CENTER, self.INNER_RADIUS, self.COLOR_TRACK, 3)

    def _render_game_elements(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            s = pygame.Surface((2, 2), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))

        for portal in self.portals:
            radius = self._get_ring_radius(portal['on_outer_ring'])
            pos = self._polar_to_cartesian(portal['angle'], radius)
            size = 15 + math.sin(self.steps * 0.05 + portal['angle']) * 3
            self._draw_neon_circle(pos, size, self.COLOR_PORTAL, 4)

        for hazard in self.hazards:
            radius = self._get_ring_radius(hazard['on_outer_ring'])
            pos = self._polar_to_cartesian(hazard['angle'], radius)
            size = hazard['pulse']
            angle_rad = math.radians(hazard['angle'] - self.player_angle + 90)
            points = [
                (pos[0] + size * math.cos(angle_rad), pos[1] + size * math.sin(angle_rad)),
                (pos[0] + size * math.cos(angle_rad + 2/3*math.pi), pos[1] + size * math.sin(angle_rad + 2/3*math.pi)),
                (pos[0] + size * math.cos(angle_rad - 2/3*math.pi), pos[1] + size * math.sin(angle_rad - 2/3*math.pi)),
            ]
            self._draw_neon_poly(points, self.COLOR_HAZARD, 2)

        if not self.game_over:
            self._render_player()

    def _render_player(self):
        pos = (self.CENTER[0], int(self.player_current_y))
        
        if self.shield_active:
            pulse = 1 + abs(math.sin(self.steps * 0.2)) * 0.2
            self._draw_neon_circle(pos, 20 * pulse, self.COLOR_SHIELD, 5)

        self._draw_neon_circle(pos, 15, self.COLOR_PLAYER, 8)

        size = 10
        angle_rad = math.radians(self.player_rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        points = []
        for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            x = pos[0] + (dx * cos_a - dy * sin_a) * size
            y = pos[1] + (dx * sin_a + dy * cos_a) * size
            points.append((x, y))
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_small.render(f"TIME: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 30))

        speed_text = self.font_small.render(f"SPEED MOD: +{self.persistent_player_speed_boost:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.WIDTH - speed_text.get_width() - 10, 10))

        for i in range(self.shields):
            pos = (self.WIDTH - 30 - i * 25, 40)
            self._draw_neon_circle(pos, 8, self.COLOR_SHIELD, 3)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 6, self.COLOR_SHIELD)

        if self.game_over:
            text = self.font_large.render("GAME OVER", True, self.COLOR_HAZARD)
            text_rect = text.get_rect(center=self.CENTER)
            self.screen.blit(text, text_rect)
        elif self.steps >= self.MAX_STEPS:
            text = self.font_large.render("VICTORY", True, self.COLOR_PORTAL)
            text_rect = text.get_rect(center=self.CENTER)
            self.screen.blit(text, text_rect)

    def _spawn_hazard(self, initial_spawn=False):
        angle = self.np_random.uniform(0, 360)
        if initial_spawn:
            player_angle_norm = self.player_angle % 360
            min_dist = 60 # degrees

            def angular_dist(a1, a2):
                d = abs(a1 - a2)
                return min(d, 360 - d)

            while angular_dist(angle, player_angle_norm) < min_dist:
                angle = self.np_random.uniform(0, 360)
        
        self.hazards.append({
            'angle': angle,
            'on_outer_ring': bool(self.np_random.choice([True, False])),
            'dir': self.np_random.choice([-1, 1]),
            'pulse': 6
        })

    def _spawn_portal(self):
        self.portals.append({
            'angle': self.np_random.uniform(0, 360),
            'on_outer_ring': bool(self.np_random.choice([True, False])),
        })
    
    def _spawn_particle(self, pos, color, speed_mult, max_life):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(0.5, 1.5) * speed_mult
        self.particles.append({
            'pos': list(pos),
            'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
            'life': self.np_random.integers(max_life // 2, max_life + 1),
            'max_life': max_life,
            'color': color
        })

    def _get_player_pos(self):
        return (self.CENTER[0], self.player_current_y)

    def _get_ring_radius(self, on_outer_ring):
        return self.OUTER_RADIUS if on_outer_ring else self.INNER_RADIUS

    def _polar_to_cartesian(self, angle, radius):
        rad = math.radians(angle - self.player_angle)
        x = self.CENTER[0] + radius * math.sin(rad)
        y = self.CENTER[1] - radius * math.cos(rad)
        return int(x), int(y)

    def _polar_to_cartesian_center(self, rad_angle, radius):
        x = self.CENTER[0] + radius * math.cos(rad_angle)
        y = self.CENTER[1] + radius * math.sin(rad_angle)
        return int(x), int(y)

    def _draw_neon_circle(self, pos, radius, color, glow_layers=5):
        x, y = int(pos[0]), int(pos[1])
        for i in range(glow_layers, 0, -1):
            alpha = 80 - i * (80 // glow_layers)
            pygame.gfxdraw.aacircle(self.screen, x, y, int(radius + i), (*color, alpha))
        pygame.gfxdraw.aacircle(self.screen, x, y, int(radius), color)

    def _draw_neon_poly(self, points, color, glow_layers=3):
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    
    # To play manually, you need a display.
    # The default SDL_VIDEODRIVER="dummy" will cause this to fail.
    # Comment out the os.environ line at the top of the file to run this.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Neon Vortex")
    except pygame.error as e:
        print("Could not create display. Did you comment out SDL_VIDEODRIVER='dummy'?")
        print(f"Pygame error: {e}")
        exit()

    obs, info = env.reset(seed=42)
    done = False
    
    while not done:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        env.clock.tick(env.FPS)
        
    env.close()