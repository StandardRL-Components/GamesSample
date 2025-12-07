import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a turn-based tactical space combat game.

    The player controls a stationary spaceship and must destroy incoming asteroids.
    The game is presented in a real-time fashion, but player actions (firing)
    are the primary drivers of game events, simulating a tactical decision loop.

    Visuals are prioritized, with clean vector graphics, particle effects, and
    smooth animations to create an engaging experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Defend your stationary turret from waves of incoming asteroids. Rotate your ship and use your laser and missile arsenal to survive."
    user_guide = "Controls: Use ←→ arrow keys to rotate your ship. Press space to fire your selected weapon and shift to switch between laser and missile."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # --- Visuals & Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_SHIP = (230, 80, 80)
        self.COLOR_SHIP_GLOW = (200, 50, 50)
        self.COLOR_ASTEROID_HEALTHY = (180, 180, 190)
        self.COLOR_ASTEROID_DAMAGED = (90, 90, 100)
        self.COLOR_LASER = (80, 255, 80)
        self.COLOR_MISSILE = (80, 150, 255)
        self.COLOR_EXPLOSION = [(255, 255, 100), (255, 200, 50), (255, 150, 0)]
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_HIGH = (80, 220, 80)
        self.COLOR_HEALTH_MED = (220, 220, 80)
        self.COLOR_HEALTH_LOW = (220, 80, 80)

        # --- Game Constants ---
        self.SHIP_SIZE = 15
        self.SHIP_ROTATION_SPEED = 4.0  # degrees per step
        self.MAX_STEPS = 5000
        
        self.WEAPON_COOLDOWNS = [10, 40]  # Laser, Missile
        self.WEAPON_NAMES = ["LASER", "MISSILE"]
        self.LASER_DAMAGE = 25
        self.MISSILE_DAMAGE = 75
        self.MISSILE_SPEED = 5.0

        self.BASE_ASTEROID_SPEED = 0.8
        self.BASE_SPAWN_INTERVAL = 60 # Steps between spawns at start

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ship_pos = (0, 0)
        self.ship_angle = 0.0
        self.asteroids = []
        self.projectiles = []
        self.particles = []
        self.starfield = []
        self.laser_render_info = None

        self.current_weapon = 0
        self.weapon_cooldown = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.switch_cooldown = 0

        self.asteroid_spawn_timer = 0
        self.current_spawn_interval = 0
        self.current_speed_multiplier = 0
        
        # This will be properly initialized in reset()
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.ship_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        self.ship_angle = -90.0  # Pointing up

        self.asteroids.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.laser_render_info = None
        
        self.current_weapon = 0
        self.weapon_cooldown = 0
        self.switch_cooldown = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.asteroid_spawn_timer = self.BASE_SPAWN_INTERVAL
        
        self._generate_starfield(200)
        self._spawn_asteroids(5)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0
        
        if not self.game_over:
            reward += self._handle_input(action)
            self._update_game_state()
            reward += self._handle_collisions()
            self._update_difficulty()
            self._spawn_new_asteroids()

        self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Rotation ---
        if movement == 3:  # Left
            self.ship_angle -= self.SHIP_ROTATION_SPEED
        elif movement == 4:  # Right
            self.ship_angle += self.SHIP_ROTATION_SPEED
        self.ship_angle %= 360

        # --- Weapon Switch (on press) ---
        switch_pressed = shift_held and not self.prev_shift_held
        if switch_pressed and self.switch_cooldown == 0:
            self.current_weapon = 1 - self.current_weapon
            self.switch_cooldown = 10 # Debounce
        
        # --- Fire Weapon (on press) ---
        fire_pressed = space_held and not self.prev_space_held
        if fire_pressed and self.weapon_cooldown == 0:
            reward += self._fire_weapon()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return reward

    def _fire_weapon(self):
        self.weapon_cooldown = self.WEAPON_COOLDOWNS[self.current_weapon]
        rad_angle = math.radians(self.ship_angle)
        cos_a, sin_a = math.cos(rad_angle), math.sin(rad_angle)
        
        if self.current_weapon == 0:  # Laser
            return self._fire_laser(rad_angle)
        else:  # Missile
            start_pos = (self.ship_pos[0] + cos_a * self.SHIP_SIZE, 
                         self.ship_pos[1] + sin_a * self.SHIP_SIZE)
            velocity = (cos_a * self.MISSILE_SPEED, sin_a * self.MISSILE_SPEED)
            self.projectiles.append({
                "pos": list(start_pos), "vel": velocity, "life": 150
            })
            return 0 # Reward is handled on impact

    def _fire_laser(self, rad_angle):
        hits = []
        for i, asteroid in enumerate(self.asteroids):
            dist = self._intersect_line_circle(self.ship_pos, rad_angle, asteroid['pos'], asteroid['radius'])
            if dist is not None:
                hits.append((dist, i))
        
        if not hits:
            # Miss
            end_point = self._get_edge_intersection(self.ship_pos, rad_angle)
            self.laser_render_info = {"start": self.ship_pos, "end": end_point, "life": 3}
            return -0.1
        
        # Hit
        dist, target_idx = min(hits, key=lambda x: x[0])
        target_asteroid = self.asteroids[target_idx]
        
        end_point_x = self.ship_pos[0] + math.cos(rad_angle) * dist
        end_point_y = self.ship_pos[1] + math.sin(rad_angle) * dist
        self.laser_render_info = {"start": self.ship_pos, "end": (end_point_x, end_point_y), "life": 5}
        
        target_asteroid['health'] -= self.LASER_DAMAGE
        self._create_particles(end_point_x, end_point_y, 5, self.COLOR_LASER, 1.5)

        if target_asteroid['health'] <= 0:
            self._destroy_asteroid(target_idx)
            return 10.0 # Destroy reward
        return 0.1 # Hit reward

    def _update_game_state(self):
        # Cooldowns
        if self.weapon_cooldown > 0: self.weapon_cooldown -= 1
        if self.switch_cooldown > 0: self.switch_cooldown -= 1
        if self.laser_render_info and self.laser_render_info['life'] > 0:
            self.laser_render_info['life'] -= 1

        # Asteroids
        for a in self.asteroids:
            a['pos'][0] += a['vel'][0] * self.current_speed_multiplier
            a['pos'][1] += a['vel'][1] * self.current_speed_multiplier
            a['angle'] += a['rot_speed']
            
            if not (a['radius'] < a['pos'][0] < self.WIDTH - a['radius']): a['vel'][0] *= -1
            if not (a['radius'] < a['pos'][1] < self.HEIGHT - a['radius']): a['vel'][1] *= -1

        # Projectiles (Missiles)
        for p in self.projectiles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] % 3 == 0:
                self._create_particles(p['pos'][0], p['pos'][1], 1, self.COLOR_MISSILE, 0.5, 5)

        self.projectiles = [p for p in self.projectiles if p['life'] > 0 and 0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT]
        
        # Particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Missile vs Asteroid
        for p_idx, p in reversed(list(enumerate(self.projectiles))):
            for a_idx, a in reversed(list(enumerate(self.asteroids))):
                dist_sq = (p['pos'][0] - a['pos'][0])**2 + (p['pos'][1] - a['pos'][1])**2
                if dist_sq < a['radius']**2:
                    a['health'] -= self.MISSILE_DAMAGE
                    self._create_particles(p['pos'][0], p['pos'][1], 10, self.COLOR_MISSILE, 2.5)
                    reward += 0.1 # Hit reward
                    if a['health'] <= 0:
                        reward += 10.0 - 0.1 # Destroy reward (avoid double counting hit)
                        self._destroy_asteroid(a_idx)
                    
                    self.projectiles.pop(p_idx)
                    break # Missile is consumed

        # Asteroid vs Ship
        for a in self.asteroids:
            dist_sq = (self.ship_pos[0] - a['pos'][0])**2 + (self.ship_pos[1] - a['pos'][1])**2
            if dist_sq < (a['radius'] + self.SHIP_SIZE * 0.5)**2:
                self.game_over = True
                self._create_particles(self.ship_pos[0], self.ship_pos[1], 50, self.COLOR_EXPLOSION, 4.0)
                break
        
        return reward

    def _destroy_asteroid(self, index):
        asteroid = self.asteroids.pop(index)
        self.score += 1
        self._create_particles(asteroid['pos'][0], asteroid['pos'][1], 30, self.COLOR_EXPLOSION, 3.0)

    def _update_difficulty(self):
        self.current_spawn_interval = self.BASE_SPAWN_INTERVAL - (self.steps // 100)
        self.current_spawn_interval = max(15, self.current_spawn_interval)
        self.current_speed_multiplier = self.BASE_ASTEROID_SPEED + 0.1 * (self.steps // 100)

    def _spawn_new_asteroids(self):
        self.asteroid_spawn_timer -= 1
        if self.asteroid_spawn_timer <= 0:
            self._spawn_asteroids(1)
            self.asteroid_spawn_timer = self.current_spawn_interval

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_objects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.starfield:
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['size'])

    def _render_game_objects(self):
        # Particles
        for p in self.particles:
            size = max(0, p['life'] / p['max_life'] * 3)
            pygame.draw.circle(self.screen, p['color'], p['pos'], size)
        
        # Missiles
        for p in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_MISSILE, p['pos'], 4)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 4, self.COLOR_MISSILE)

        # Asteroids
        for a in self.asteroids:
            health_ratio = max(0, a['health'] / a['max_health'])
            color = tuple(int(c1 + (c2 - c1) * health_ratio) for c1, c2 in zip(self.COLOR_ASTEROID_DAMAGED, self.COLOR_ASTEROID_HEALTHY))
            
            points = []
            for i in range(a['num_points']):
                angle = 2 * math.pi * i / a['num_points'] + math.radians(a['angle'])
                dist = a['shape'][i]
                x = a['pos'][0] + dist * math.cos(angle)
                y = a['pos'][1] + dist * math.sin(angle)
                points.append((int(x), int(y)))
            
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

        # Laser
        if self.laser_render_info and self.laser_render_info['life'] > 0:
            life_ratio = self.laser_render_info['life'] / 5.0
            alpha = int(255 * life_ratio)
            color = (*self.COLOR_LASER, alpha)
            
            line_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(line_surf, color, self.laser_render_info['start'], self.laser_render_info['end'], int(8 * life_ratio))
            pygame.draw.line(line_surf, (255, 255, 255, alpha), self.laser_render_info['start'], self.laser_render_info['end'], int(3 * life_ratio))
            self.screen.blit(line_surf, (0, 0))

        # Ship
        if not self.game_over:
            rad_angle = math.radians(self.ship_angle)
            p1 = (self.ship_pos[0] + math.cos(rad_angle) * self.SHIP_SIZE, self.ship_pos[1] + math.sin(rad_angle) * self.SHIP_SIZE)
            p2 = (self.ship_pos[0] + math.cos(rad_angle + 2.5) * self.SHIP_SIZE * 0.8, self.ship_pos[1] + math.sin(rad_angle + 2.5) * self.SHIP_SIZE * 0.8)
            p3 = (self.ship_pos[0] + math.cos(rad_angle - 2.5) * self.SHIP_SIZE * 0.8, self.ship_pos[1] + math.sin(rad_angle - 2.5) * self.SHIP_SIZE * 0.8)
            
            glow_surf = pygame.Surface((self.SHIP_SIZE*4, self.SHIP_SIZE*4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_SHIP_GLOW, 50), (self.SHIP_SIZE*2, self.SHIP_SIZE*2), self.SHIP_SIZE*1.5)
            pygame.draw.circle(glow_surf, (*self.COLOR_SHIP_GLOW, 25), (self.SHIP_SIZE*2, self.SHIP_SIZE*2), self.SHIP_SIZE*2.0)
            self.screen.blit(glow_surf, (self.ship_pos[0] - self.SHIP_SIZE*2, self.ship_pos[1] - self.SHIP_SIZE*2))

            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_SHIP)
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_SHIP)

    def _render_ui(self):
        # Score
        score_text = self.font_m.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Weapon
        weapon_text = self.font_m.render(f"WEAPON: {self.WEAPON_NAMES[self.current_weapon]}", True, self.COLOR_UI_TEXT)
        text_rect = weapon_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(weapon_text, text_rect)
        
        # Cooldown indicator
        if self.weapon_cooldown > 0:
            cooldown_ratio = self.weapon_cooldown / self.WEAPON_COOLDOWNS[self.current_weapon]
            bar_width = text_rect.width
            bar_height = 4
            bar_x = text_rect.left
            bar_y = text_rect.bottom + 5
            pygame.draw.rect(self.screen, self.COLOR_ASTEROID_DAMAGED, (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_MISSILE, (bar_x, bar_y, bar_width * cooldown_ratio, bar_height))

        # Game Over
        if self.game_over:
            over_text = self.font_l.render("GAME OVER", True, self.COLOR_SHIP)
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(over_text, text_rect)
            final_score_text = self.font_m.render(f"FINAL SCORE: {self.score}", True, self.COLOR_UI_TEXT)
            text_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, text_rect)
            
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Helper Methods ---
    
    def _spawn_asteroids(self, num):
        for _ in range(num):
            side = self.np_random.integers(4)
            if side == 0: pos = [self.np_random.uniform(-20, 0), self.np_random.uniform(0, self.HEIGHT)]
            elif side == 1: pos = [self.np_random.uniform(self.WIDTH, self.WIDTH + 20), self.np_random.uniform(0, self.HEIGHT)]
            elif side == 2: pos = [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(-20, 0)]
            else: pos = [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(self.HEIGHT, self.HEIGHT + 20)]
            
            angle = math.atan2(self.ship_pos[1] - pos[1], self.ship_pos[0] - pos[0])
            angle += self.np_random.uniform(-0.5, 0.5)
            vel = [math.cos(angle), math.sin(angle)]

            radius = self.np_random.uniform(15, 40)
            health = int(radius * 3)
            
            num_points = self.np_random.integers(7, 12)
            shape_mags = [self.np_random.uniform(radius * 0.8, radius * 1.2) for _ in range(num_points)]

            self.asteroids.append({
                "pos": pos, "vel": vel, "radius": radius, "health": health, "max_health": health,
                "angle": self.np_random.uniform(0, 360), "rot_speed": self.np_random.uniform(-1.0, 1.0),
                "num_points": num_points, "shape": shape_mags
            })

    def _generate_starfield(self, num_stars):
        self.starfield.clear()
        for _ in range(num_stars):
            brightness = self.np_random.choice([50, 100, 150], p=[0.6, 0.3, 0.1])
            self.starfield.append({
                "pos": (self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                "size": self.np_random.uniform(0.5, 1.5),
                "color": (brightness, brightness, brightness)
            })
            
    def _create_particles(self, x, y, num, color, speed_mult, max_life=20):
        colors = color if isinstance(color, list) else [color]
        for _ in range(num):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(max_life // 2, max_life)
            self.particles.append({
                "pos": [x, y], "vel": vel, "life": life, "max_life": life,
                "color": random.choice(colors)
            })

    def _intersect_line_circle(self, p1, angle, c, r):
        # Ray-circle intersection
        p2 = (p1[0] + math.cos(angle), p1[1] + math.sin(angle))
        d = (p2[0] - p1[0], p2[1] - p1[1])
        f = (p1[0] - c[0], p1[1] - c[1])

        a = d[0]**2 + d[1]**2
        b = 2 * (f[0]*d[0] + f[1]*d[1])
        c_ = f[0]**2 + f[1]**2 - r**2
        
        discriminant = b*b - 4*a*c_
        if discriminant >= 0:
            discriminant = math.sqrt(discriminant)
            t1 = (-b - discriminant) / (2*a)
            t2 = (-b + discriminant) / (2*a)
            
            if t1 >= 0: return t1
            if t2 >= 0: return t2
        return None

    def _get_edge_intersection(self, start_pos, angle):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        t_values = []
        if cos_a != 0:
            t_right = (self.WIDTH - start_pos[0]) / cos_a
            if t_right > 0: t_values.append(t_right)
            t_left = -start_pos[0] / cos_a
            if t_left > 0: t_values.append(t_left)
        if sin_a != 0:
            t_bottom = (self.HEIGHT - start_pos[1]) / sin_a
            if t_bottom > 0: t_values.append(t_bottom)
            t_top = -start_pos[1] / sin_a
            if t_top > 0: t_values.append(t_top)
            
        if not t_values:
            return (start_pos[0] + cos_a * 1000, start_pos[1] + sin_a * 1000)

        min_t = min(t_values)
        return (start_pos[0] + min_t * cos_a, start_pos[1] + min_t * sin_a)

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # This requires pygame to be installed with a display driver
    # To run this, you might need to unset the dummy videodriver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a window to display the game
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tactical Asteroids")
    
    # Game loop
    total_reward = 0
    while not done:
        # --- Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Render ---
        # The observation is already a rendered frame, we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # Limit to 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    pygame.quit()