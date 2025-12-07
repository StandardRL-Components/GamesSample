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


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move your ship. Press space to mine nearby asteroids."
    )

    game_description = (
        "Mine asteroids for minerals in a dangerous field while dodging deadly laser fire."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.WIN_SCORE = 100
        self.MAX_STEPS = 10000
        self.PLAYER_MAX_HEALTH = 3
        self.PLAYER_SPEED = 200
        self.PLAYER_SIZE = 15
        self.ASTEROID_COUNT = 10
        self.ASTEROID_MIN_SIZE = 20
        self.ASTEROID_MAX_SIZE = 40
        self.ASTEROID_MIN_SPEED = 10
        self.ASTEROID_MAX_SPEED = 30
        self.ASTEROID_MIN_VERTS = 7
        self.ASTEROID_MAX_VERTS = 12
        self.LASER_SPEED = 400
        self.MINING_RANGE = 60
        self.MINING_COOLDOWN_TIME = 0.25 # seconds
        self.HIT_FLASH_DURATION = 0.5 # seconds
        self.INITIAL_LASER_FREQ = 0.1 # lasers per second
        self.LASER_FREQ_INCREASE = 0.01 # per second

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_HIT = (255, 100, 100)
        self.COLOR_ASTEROID = (139, 125, 123)
        self.COLOR_LASER = (255, 0, 0)
        self.COLOR_LASER_GLOW = (255, 100, 100)
        self.COLOR_MINERAL = (255, 220, 0)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_EXPLOSION = [(255, 255, 255), (255, 200, 0), (255, 100, 0)]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_health = 0
        self.minerals_collected = 0
        self.asteroids = []
        self.lasers = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.game_over = False
        self.mining_cooldown = 0
        self.hit_flash_timer = 0
        self.laser_spawn_timer = 0
        self.current_laser_freq = 0
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.minerals_collected = 0
        
        self.asteroids = [self._create_asteroid() for _ in range(self.ASTEROID_COUNT)]
        self.lasers = []
        self.particles = []
        
        self.mining_cooldown = 0
        self.hit_flash_timer = 0
        self.current_laser_freq = self.INITIAL_LASER_FREQ
        self.laser_spawn_timer = 1.0 / self.current_laser_freq
        
        if not self.stars:
            self.stars = [
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
                for _ in range(150)
            ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        dt = self.clock.tick(self.FPS) / 1000.0
        
        movement = action[0]
        is_mining = action[1] == 1
        
        reward = -0.01  # Small penalty for every step

        self._handle_input(movement)
        self._update_player(dt)
        
        mine_reward = self._handle_mining(is_mining)
        reward += mine_reward

        self._update_asteroids(dt)
        self._update_lasers(dt)
        self._update_particles(dt)
        
        hit_reward = self._check_collisions()
        reward += hit_reward
        
        self._update_difficulty_and_spawns(dt)
        
        self.steps += 1
        terminated = False
        truncated = False
        
        if self.minerals_collected >= self.WIN_SCORE:
            terminated = True
            reward += 100.0
        elif self.player_health <= 0:
            terminated = True
            reward += -100.0
        elif self.steps >= self.MAX_STEPS:
            truncated = True

        if terminated or truncated:
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement):
        self.player_vel = pygame.Vector2(0, 0)
        if movement == 1: # Up
            self.player_vel.y = -1
        elif movement == 2: # Down
            self.player_vel.y = 1
        elif movement == 3: # Left
            self.player_vel.x = -1
        elif movement == 4: # Right
            self.player_vel.x = 1
        
        if self.player_vel.length() > 0:
            self.player_vel.normalize_ip()

    def _update_player(self, dt):
        self.player_pos += self.player_vel * self.PLAYER_SPEED * dt
        # World wrapping
        self.player_pos.x %= self.WIDTH
        self.player_pos.y %= self.HEIGHT
        
        if self.hit_flash_timer > 0:
            self.hit_flash_timer = max(0, self.hit_flash_timer - dt)
        if self.mining_cooldown > 0:
            self.mining_cooldown = max(0, self.mining_cooldown - dt)

    def _handle_mining(self, is_mining):
        if not is_mining or self.mining_cooldown > 0:
            return 0

        closest_asteroid = None
        min_dist = float('inf')
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
                closest_asteroid = asteroid

        if closest_asteroid and min_dist < self.MINING_RANGE:
            self.mining_cooldown = self.MINING_COOLDOWN_TIME
            self.minerals_collected += 1
            
            # Create mineral particle effect
            for _ in range(5):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(50, 100)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                self.particles.append({
                    'pos': pygame.Vector2(closest_asteroid['pos']),
                    'vel': vel,
                    'life': 0.5,
                    'max_life': 0.5,
                    'size': self.np_random.integers(2, 5),
                    'color': self.COLOR_MINERAL
                })
            
            closest_asteroid['minerals'] -= 1
            if closest_asteroid['minerals'] <= 0:
                self.asteroids.remove(closest_asteroid)
                self.asteroids.append(self._create_asteroid(on_edge=True))
            
            return 0.1
        return 0

    def _update_asteroids(self, dt):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel'] * dt
            asteroid['angle'] += asteroid['rot_speed'] * dt
            
            if (asteroid['pos'].x < -self.ASTEROID_MAX_SIZE or asteroid['pos'].x > self.WIDTH + self.ASTEROID_MAX_SIZE or
                asteroid['pos'].y < -self.ASTEROID_MAX_SIZE or asteroid['pos'].y > self.HEIGHT + self.ASTEROID_MAX_SIZE):
                self.asteroids.remove(asteroid)
                self.asteroids.append(self._create_asteroid(on_edge=True))

    def _update_lasers(self, dt):
        for laser in self.lasers[:]:
            laser['pos'] += laser['vel'] * self.LASER_SPEED * dt
            if not self.screen.get_rect().collidepoint(laser['pos']):
                self.lasers.remove(laser)

    def _update_particles(self, dt):
        for p in self.particles[:]:
            p['pos'] += p['vel'] * dt
            p['life'] -= dt
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_collisions(self):
        for laser in self.lasers[:]:
            if self.player_pos.distance_to(laser['pos']) < self.PLAYER_SIZE:
                self.lasers.remove(laser)
                self.player_health -= 1
                self.hit_flash_timer = self.HIT_FLASH_DURATION
                # Create explosion particle effect
                for _ in range(30):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(50, 250)
                    vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                    self.particles.append({
                        'pos': pygame.Vector2(self.player_pos),
                        'vel': vel,
                        'life': self.np_random.uniform(0.3, 0.8),
                        'max_life': 1.0,
                        'size': self.np_random.integers(2, 6),
                        'color': random.choice(self.COLOR_EXPLOSION)
                    })
                return -1.0
        return 0

    def _update_difficulty_and_spawns(self, dt):
        self.current_laser_freq += self.LASER_FREQ_INCREASE * dt
        self.laser_spawn_timer -= dt
        if self.laser_spawn_timer <= 0:
            self.lasers.append(self._create_laser())
            self.laser_spawn_timer = 1.0 / self.current_laser_freq

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_asteroids()
        self._render_particles()
        self._render_lasers()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_stars(self):
        for x, y, size in self.stars:
            color_val = 50 + size * 50
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (x, y), size // 2)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            cos_a = math.cos(asteroid['angle'])
            sin_a = math.sin(asteroid['angle'])
            points = []
            for p in asteroid['shape']:
                x = p.x * cos_a - p.y * sin_a + asteroid['pos'].x
                y = p.x * sin_a + p.y * cos_a + asteroid['pos'].y
                points.append((int(x), int(y)))
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = max(0, p['life'] / p['max_life'])
            alpha = int(255 * life_ratio)
            color = p['color']
            
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (color[0], color[1], color[2], alpha), (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

    def _render_lasers(self):
        for laser in self.lasers:
            start_pos = laser['pos']
            end_pos = laser['pos'] - laser['vel'] * 20
            # Glow
            pygame.draw.aaline(self.screen, self.COLOR_LASER_GLOW, start_pos, end_pos, 3)
            # Core
            pygame.draw.aaline(self.screen, (255,255,255), start_pos, end_pos, 1)

    def _render_player(self):
        color = self.COLOR_PLAYER
        if self.hit_flash_timer > 0:
            if int(self.hit_flash_timer * 10) % 2 == 0:
                color = self.COLOR_PLAYER_HIT
        
        p1 = pygame.Vector2(0, -self.PLAYER_SIZE)
        p2 = pygame.Vector2(-self.PLAYER_SIZE * 0.7, self.PLAYER_SIZE * 0.7)
        p3 = pygame.Vector2(self.PLAYER_SIZE * 0.7, self.PLAYER_SIZE * 0.7)
        
        points = [p1, p2, p3]
        
        if self.player_vel.length() > 0.1:
            angle = self.player_vel.angle_to(pygame.Vector2(0, -1))
            rads = math.radians(angle)
            cos_a = math.cos(rads)
            sin_a = math.sin(rads)
            
            rotated_points = []
            for p in points:
                x = p.x * cos_a - p.y * sin_a + self.player_pos.x
                y = p.x * sin_a + p.y * cos_a + self.player_pos.y
                rotated_points.append((int(x), int(y)))
        else:
             rotated_points = [(int(p.x + self.player_pos.x), int(p.y + self.player_pos.y)) for p in points]
        
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, color)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, color)
    
    def _render_ui(self):
        score_text = self.font_ui.render(f"MINERALS: {self.minerals_collected} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        health_text = self.font_ui.render("HITS: ", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.WIDTH - 120, 10))
        for i in range(self.PLAYER_MAX_HEALTH):
            color = self.COLOR_PLAYER if i < self.player_health else self.COLOR_PLAYER_HIT
            pygame.draw.rect(self.screen, color, (self.WIDTH - 60 + i * 20, 12, 15, 15))

        if self.game_over:
            if self.minerals_collected >= self.WIN_SCORE:
                end_text = self.font_game_over.render("YOU WIN!", True, self.COLOR_MINERAL)
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.COLOR_LASER)
            
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.minerals_collected,
            "steps": self.steps,
            "health": self.player_health,
        }

    def _create_asteroid(self, pos=None, on_edge=False):
        size = self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        if on_edge:
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -size)
            elif edge == 1: # Right
                pos = pygame.Vector2(self.WIDTH + size, self.np_random.uniform(0, self.HEIGHT))
            elif edge == 2: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + size)
            else: # Left
                pos = pygame.Vector2(-size, self.np_random.uniform(0, self.HEIGHT))
        elif pos is None:
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT))
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(self.ASTEROID_MIN_SPEED, self.ASTEROID_MAX_SPEED)
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        
        num_verts = self.np_random.integers(self.ASTEROID_MIN_VERTS, self.ASTEROID_MAX_VERTS + 1)
        shape_points = []
        for i in range(num_verts):
            angle = i * (2 * math.pi / num_verts)
            dist = self.np_random.uniform(size * 0.7, size)
            shape_points.append(pygame.Vector2(math.cos(angle) * dist, math.sin(angle) * dist))
            
        return {
            'pos': pos,
            'vel': vel,
            'size': size,
            'angle': 0,
            'rot_speed': self.np_random.uniform(-2, 2),
            'shape': shape_points,
            'minerals': self.np_random.integers(5, 15)
        }

    def _create_laser(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -10)
            target = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT)
        elif edge == 1: # Right
            pos = pygame.Vector2(self.WIDTH + 10, self.np_random.uniform(0, self.HEIGHT))
            target = pygame.Vector2(0, self.np_random.uniform(0, self.HEIGHT))
        elif edge == 2: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 10)
            target = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), 0)
        else: # Left
            pos = pygame.Vector2(-10, self.np_random.uniform(0, self.HEIGHT))
            target = pygame.Vector2(self.WIDTH, self.np_random.uniform(0, self.HEIGHT))
        
        vel = (target - pos).normalize()
        return {'pos': pos, 'vel': vel}

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to remove the headless mode for interactive play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    running = True
    game_window = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_window.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset(seed=random.randint(0, 1_000_000))
            total_reward = 0

    env.close()