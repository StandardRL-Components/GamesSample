import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move your ship. Hold Space to fire the mining laser. "
        "Collect 100 ore to win. Avoid colliding with asteroids!"
    )

    game_description = (
        "A fast-paced arcade shooter where you pilot a spaceship to mine valuable "
        "ore from asteroids. Collect 100 units of ore to win, but be careful! "
        "Colliding with an asteroid will cost you one of your three lives."
    )

    auto_advance = True

    # --- Game Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000
    WIN_ORE = 100
    INITIAL_LIVES = 3

    PLAYER_SIZE = 12
    PLAYER_ACCELERATION = 0.6
    PLAYER_FRICTION = 0.96
    PLAYER_MAX_SPEED = 6
    PLAYER_INVINCIBILITY_FRAMES = 90 # 3 seconds

    LASER_RANGE = 200
    LASER_WIDTH = 3
    LASER_DAMAGE = 1

    ASTEROID_SPAWN_INTERVAL = 60 # frames
    ASTEROID_DIFFICULTY_INTERVAL = 200 # steps to increase spawn rate

    # --- Color Palette ---
    COLOR_BG = (15, 18, 32)
    COLOR_PLAYER = (64, 180, 255)
    COLOR_PLAYER_SHIELD = (120, 220, 255, 100)
    COLOR_LASER = (255, 220, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_EXPLOSION = [(255, 60, 0), (255, 150, 0), (255, 200, 100)]
    
    ASTEROID_TYPES = {
        'grey':   {'color': (140, 140, 150), 'ore': 1,  'health': 20, 'value_tier': 0},
        'brown':  {'color': (180, 120, 80),  'ore': 2,  'health': 30, 'value_tier': 0},
        'gold':   {'color': (255, 215, 0),   'ore': 5,  'health': 50, 'value_tier': 0},
        'purple': {'color': (200, 80, 255),  'ore': 10, 'health': 70, 'value_tier': 1},
        'green':  {'color': (80, 255, 150),  'ore': 15, 'health': 90, 'value_tier': 1},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        self.render_mode = render_mode
        self.np_random = None # Will be seeded in reset

        # Initialize game state attributes to default values
        # to allow for validation before the first reset.
        self.player_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_angle = 0.0
        self.player_lives = 0
        self.player_invincible_timer = 0
        self.steps = 0
        self.ore_collected = 0
        self.game_over = True
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.laser_active = False
        self.laser_hit_pos = None
        self.current_spawn_interval = self.ASTEROID_SPAWN_INTERVAL
        self.asteroid_spawn_timer = 0

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_angle = -90.0
        self.player_lives = self.INITIAL_LIVES
        self.player_invincible_timer = self.PLAYER_INVINCIBILITY_FRAMES

        # Game state
        self.steps = 0
        self.ore_collected = 0
        self.game_over = False
        
        # Entities
        self.asteroids = []
        self.particles = []
        self.stars = self._create_stars(200)

        # Timers and difficulty
        self.asteroid_spawn_timer = 0
        self.current_spawn_interval = self.ASTEROID_SPAWN_INTERVAL
        
        # Action state
        self.laser_active = False
        self.laser_hit_pos = None

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.02 # Small penalty for every step to encourage action
        ore_mined_this_step = False

        self._update_player(movement)
        self.laser_active = space_held
        
        self._update_asteroids()
        
        # Mining and collisions
        if self.laser_active:
            mined_info = self._handle_laser()
            if mined_info:
                reward += 0.1 # Reward for any ore
                ore_mined_this_step = True
                if mined_info['value_tier'] == 1:
                    reward += 1.0 # Bonus for high-value ore
                self.ore_collected += mined_info['ore']

        if self.player_invincible_timer <= 0:
            if self._handle_player_collisions():
                reward -= 50 # Large penalty for losing a life
                self.player_lives -= 1
                self.player_invincible_timer = self.PLAYER_INVINCIBILITY_FRAMES
                self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
                self.player_vel = np.zeros(2, dtype=np.float32)
                # SFX: Player Explosion
                self._create_explosion(self.player_pos, 50, self.COLOR_PLAYER)

        self._update_particles()
        self._spawn_asteroids()

        self.steps += 1
        if self.steps % self.ASTEROID_DIFFICULTY_INTERVAL == 0:
            self.current_spawn_interval = max(15, self.current_spawn_interval * 0.95)

        terminated = self.ore_collected >= self.WIN_ORE or self.player_lives <= 0 or self.steps >= self.MAX_STEPS
        truncated = False
        
        if terminated and not self.game_over:
            if self.ore_collected >= self.WIN_ORE:
                reward += 100 # Large reward for winning
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_particles()
        self._render_asteroids()
        self._render_player()
        if self.laser_active:
            self._render_laser()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.ore_collected, "steps": self.steps, "lives": self.player_lives}

    # --- Update Logic ---

    def _update_player(self, movement):
        if self.player_invincible_timer > 0:
            self.player_invincible_timer -= 1

        acceleration = np.zeros(2, dtype=np.float32)
        if movement != 0:
            # 1=up, 2=down, 3=left, 4=right
            if movement == 1: acceleration[1] -= self.PLAYER_ACCELERATION
            if movement == 2: acceleration[1] += self.PLAYER_ACCELERATION
            if movement == 3: acceleration[0] -= self.PLAYER_ACCELERATION
            if movement == 4: acceleration[0] += self.PLAYER_ACCELERATION
            
            # Update angle to last movement direction
            if np.any(acceleration):
                self.player_angle = math.degrees(math.atan2(-acceleration[1], acceleration[0]))
            
            # Engine particles
            if self.np_random.random() < 0.7:
                angle_rad = math.radians(self.player_angle + 180)
                offset = np.array([math.cos(angle_rad), -math.sin(angle_rad)]) * self.PLAYER_SIZE
                self._create_particle(
                    pos=self.player_pos + offset,
                    vel=self.player_vel * 0.5,
                    lifespan=10,
                    size_range=(2, 4),
                    color=self.COLOR_LASER
                )

        self.player_vel += acceleration
        self.player_vel *= self.PLAYER_FRICTION
        
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED
            
        self.player_pos += self.player_vel
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _update_asteroids(self):
        for asteroid in self.asteroids[:]:
            asteroid['pos'] += asteroid['vel']
            if not (-50 < asteroid['pos'][0] < self.WIDTH + 50 and -50 < asteroid['pos'][1] < self.HEIGHT + 50):
                self.asteroids.remove(asteroid)

    def _handle_laser(self):
        self.laser_hit_pos = None
        laser_end = self.player_pos + np.array([math.cos(math.radians(self.player_angle)), -math.sin(math.radians(self.player_angle))]) * self.LASER_RANGE
        
        closest_asteroid = None
        min_dist = self.LASER_RANGE
        
        for asteroid in self.asteroids:
            # Simple line-circle intersection check
            p1 = self.player_pos
            p2 = laser_end
            c = asteroid['pos']
            r = asteroid['size']
            
            d = p2 - p1
            f = p1 - c
            
            a = d.dot(d)
            b = 2 * f.dot(d)
            c_ = f.dot(f) - r**2
            
            discriminant = b**2 - 4 * a * c_
            if discriminant >= 0:
                discriminant = math.sqrt(discriminant)
                t1 = (-b - discriminant) / (2 * a)
                t2 = (-b + discriminant) / (2 * a)
                
                if 0 <= t1 <= 1:
                    dist = t1 * np.linalg.norm(d)
                    if dist < min_dist:
                        min_dist = dist
                        closest_asteroid = asteroid
        
        if closest_asteroid:
            self.laser_hit_pos = self.player_pos + np.array([math.cos(math.radians(self.player_angle)), -math.sin(math.radians(self.player_angle))]) * min_dist
            closest_asteroid['health'] -= self.LASER_DAMAGE
            # SFX: Mining Beam Hit
            
            if self.np_random.random() < 0.5:
                self._create_particle(
                    pos=self.laser_hit_pos,
                    vel=self.np_random.uniform(-1, 1, 2),
                    lifespan=15,
                    size_range=(1, 3),
                    color=closest_asteroid['type']['color']
                )

            if closest_asteroid['health'] <= 0:
                mined_info = closest_asteroid['type']
                self._create_explosion(closest_asteroid['pos'], int(closest_asteroid['size']), closest_asteroid['type']['color'])
                # SFX: Asteroid Destroyed
                self.asteroids.remove(closest_asteroid)
                return mined_info
        return None

    def _handle_player_collisions(self):
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_SIZE + asteroid['size']:
                return True
        return False

    def _spawn_asteroids(self):
        self.asteroid_spawn_timer -= 1
        if self.asteroid_spawn_timer <= 0:
            self.asteroid_spawn_timer = self.current_spawn_interval
            self._create_asteroid()

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    # --- Creation Helpers ---

    def _create_stars(self, n):
        return [{'pos': (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                 'size': self.np_random.integers(1, 3),
                 'color': self.np_random.integers(50, 150)} for _ in range(n)]

    def _create_asteroid(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -40.0])
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 40.0])
        elif edge == 2: # Left
            pos = np.array([-40.0, self.np_random.uniform(0, self.HEIGHT)])
        else: # Right
            pos = np.array([self.WIDTH + 40.0, self.np_random.uniform(0, self.HEIGHT)])

        target = np.array([self.WIDTH / 2, self.HEIGHT / 2]) + self.np_random.uniform(-100, 100, 2)
        direction = (target - pos) / np.linalg.norm(target - pos)
        velocity = direction * self.np_random.uniform(0.5, 2.0)
        
        asteroid_key = self.np_random.choice(list(self.ASTEROID_TYPES.keys()), p=[0.35, 0.3, 0.2, 0.1, 0.05])
        asteroid_type = self.ASTEROID_TYPES[asteroid_key]
        size = self.np_random.uniform(15, 35)

        points = []
        num_points = self.np_random.integers(7, 12)
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            dist = size * self.np_random.uniform(0.7, 1.3)
            points.append((math.cos(angle) * dist, math.sin(angle) * dist))
        
        self.asteroids.append({
            'pos': pos,
            'vel': velocity,
            'size': size,
            'type': asteroid_type,
            'health': asteroid_type['health'],
            'points': points
        })

    def _create_particle(self, pos, vel, lifespan, size_range, color):
        self.particles.append({
            'pos': pos.copy(),
            'vel': vel.copy(),
            'lifespan': lifespan,
            'max_lifespan': lifespan,
            'size': self.np_random.uniform(size_range[0], size_range[1]),
            'color': color
        })

    def _create_explosion(self, pos, count, base_color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self._create_particle(
                pos=pos,
                vel=vel,
                lifespan=self.np_random.integers(15, 40),
                size_range=(1, 4),
                color=random.choice(self.COLOR_EXPLOSION)
            )

    # --- Render Methods ---

    def _render_stars(self):
        for star in self.stars:
            c = star['color']
            pygame.draw.circle(self.screen, (c, c, c), star['pos'], star['size'])

    def _render_player(self):
        angle_rad = math.radians(self.player_angle)
        p1 = (self.player_pos[0] + math.cos(angle_rad) * self.PLAYER_SIZE * 1.5, 
              self.player_pos[1] - math.sin(angle_rad) * self.PLAYER_SIZE * 1.5)
        p2 = (self.player_pos[0] + math.cos(angle_rad + 2.4) * self.PLAYER_SIZE, 
              self.player_pos[1] - math.sin(angle_rad + 2.4) * self.PLAYER_SIZE)
        p3 = (self.player_pos[0] + math.cos(angle_rad - 2.4) * self.PLAYER_SIZE, 
              self.player_pos[1] - math.sin(angle_rad - 2.4) * self.PLAYER_SIZE)
        
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

        if self.player_invincible_timer > 0 and self.steps % 10 < 5:
            shield_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(shield_surface, int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE + 5, self.COLOR_PLAYER_SHIELD)
            pygame.gfxdraw.filled_circle(shield_surface, int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE + 5, self.COLOR_PLAYER_SHIELD)
            self.screen.blit(shield_surface, (0,0))

    def _render_laser(self):
        end_pos = self.laser_hit_pos if self.laser_hit_pos is not None else \
                  self.player_pos + np.array([math.cos(math.radians(self.player_angle)), -math.sin(math.radians(self.player_angle))]) * self.LASER_RANGE
        
        pygame.draw.line(self.screen, self.COLOR_LASER, self.player_pos, end_pos, self.LASER_WIDTH)
        # Glow effect for laser
        pygame.draw.line(self.screen, (*self.COLOR_LASER, 50), self.player_pos, end_pos, self.LASER_WIDTH + 4)


    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = [(p[0] + asteroid['pos'][0], p[1] + asteroid['pos'][1]) for p in asteroid['points']]
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, asteroid['type']['color'])
                pygame.gfxdraw.filled_polygon(self.screen, points, asteroid['type']['color'])

    def _render_particles(self):
        for p in self.particles:
            alpha = p['lifespan'] / p['max_lifespan']
            size = p['size'] * alpha
            if size > 1:
                color = (*p['color'], int(255 * alpha))
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (p['pos'][0] - size, p['pos'][1] - size))

    def _render_ui(self):
        # Ore display
        ore_text = self.font_small.render(f"ORE: {self.ore_collected} / {self.WIN_ORE}", True, self.COLOR_TEXT)
        self.screen.blit(ore_text, (10, 10))

        # Lives display
        for i in range(self.player_lives):
            ship_icon_points = [
                (self.WIDTH - 20 - i*25, 25),
                (self.WIDTH - 35 - i*25, 15),
                (self.WIDTH - 35 - i*25, 35)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, ship_icon_points, self.COLOR_PLAYER)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        if self.ore_collected >= self.WIN_ORE:
            msg = "MISSION COMPLETE"
            color = (100, 255, 150)
        else:
            msg = "GAME OVER"
            color = (255, 100, 100)
            
        text = self.font_large.render(msg, True, color)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Temporarily seed RNG for validation if it's not already seeded
        if self.np_random is None:
            super().reset(seed=12345)
            # After seeding, create stars so _get_observation doesn't fail
            self.stars = self._create_stars(200)

        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Reset for a clean test of step and reset return values
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To run, you need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the game
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Ore: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()