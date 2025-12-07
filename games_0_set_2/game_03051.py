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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move your ship. Hold space to mine nearby asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade space miner. Mine asteroids for ore while dodging enemy ships. Get bonus ore for risky mining!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    WIN_SCORE = 100
    MAX_STEPS = 1500 # Increased to allow more time for skilled play

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ASTEROID = (150, 150, 150)
    COLOR_ORE = (255, 220, 0)
    COLOR_UI = (220, 220, 255)
    COLOR_EXPLOSION = [(255, 50, 50), (255, 150, 0), (255, 255, 0)]

    # Player settings
    PLAYER_ACCELERATION = 0.5
    PLAYER_FRICTION = 0.96
    PLAYER_MAX_SPEED = 6
    PLAYER_SIZE = 12
    PLAYER_COLLISION_RADIUS = 8

    # Enemy settings
    INITIAL_ENEMIES = 5
    ENEMY_SIZE = 10
    ENEMY_BASE_SPEED = 1.0
    ENEMY_COLLISION_RADIUS = 8
    DANGER_RADIUS = 100

    # Asteroid settings
    ASTEROID_COUNT = 8
    ASTEROID_MIN_SIZE = 15
    ASTEROID_MAX_SIZE = 35
    ASTEROID_MIN_ORE = 3
    ASTEROID_MAX_ORE = 8
    ASTEROID_ROTATION_SPEED = 0.5

    # Mining settings
    MINING_RANGE = 70
    MINING_COOLDOWN = 10 # frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.asteroids = []
        self.enemies = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.ore_collected = 0
        self.game_over = False
        self.mining_timer = 0
        self.enemy_speed_multiplier = 1.0
        self.np_random = None

        self.validate_implementation()

    def _get_dist(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _get_closest(self, pos, entity_list):
        closest_entity = None
        min_dist = float('inf')
        if not entity_list:
            return None, min_dist
        for entity in entity_list:
            dist = self._get_dist(pos, entity['pos'])
            if dist < min_dist:
                min_dist = dist
                closest_entity = entity
        return closest_entity, min_dist

    def _spawn_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)],
                'size': self.np_random.choice([1, 2, 3]),
            })

    def _spawn_asteroid(self):
        while True:
            pos = [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)]
            if self._get_dist(pos, self.player_pos) > 150: # Don't spawn on player
                break
        
        size = self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        num_vertices = self.np_random.integers(7, 12)
        
        self.asteroids.append({
            'pos': pos,
            'size': size,
            'ore': self.np_random.integers(self.ASTEROID_MIN_ORE, self.ASTEROID_MAX_ORE + 1),
            'angle': 0,
            'rotation_speed': self.np_random.uniform(-self.ASTEROID_ROTATION_SPEED, self.ASTEROID_ROTATION_SPEED),
            'vertices': [
                (math.cos(2 * math.pi * i / num_vertices) * self.np_random.uniform(0.8, 1.2),
                 math.sin(2 * math.pi * i / num_vertices) * self.np_random.uniform(0.8, 1.2))
                for i in range(num_vertices)
            ]
        })

    def _spawn_enemy(self):
        while True:
            pos = [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)]
            if self._get_dist(pos, self.player_pos) > 200: # Don't spawn on player
                break
        
        pattern = self.np_random.choice(['linear', 'circular'])
        if pattern == 'linear':
            vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)]
            vel_mag = math.sqrt(vel[0]**2 + vel[1]**2)
            if vel_mag > 0:
                vel = [v / vel_mag for v in vel]
            else:
                vel = [1, 0]
        else: # circular
            vel = {'center': [pos[0] + self.np_random.uniform(-50, 50), pos[1] + self.np_random.uniform(-50, 50)],
                   'radius': self.np_random.uniform(30, 80),
                   'angle': self.np_random.uniform(0, 2 * math.pi)}

        self.enemies.append({
            'pos': pos,
            'vel': vel,
            'pattern': pattern
        })
        
    def _create_explosion(self, pos, num_particles=50):
        # sfx: explosion
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'ttl': self.np_random.integers(15, 30),
                'color': random.choice(self.COLOR_EXPLOSION),
                'size': self.np_random.uniform(1, 4)
            })

    def _world_wrap(self, pos):
        pos[0] %= self.WIDTH
        pos[1] %= self.HEIGHT
        return pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        self.player_vel = [0, 0]
        
        self.steps = 0
        self.score = 0
        self.ore_collected = 0
        self.game_over = False
        self.mining_timer = 0
        self.enemy_speed_multiplier = 1.0

        self.asteroids = []
        self.enemies = []
        self.particles = []
        self._spawn_stars()
        
        for _ in range(self.ASTEROID_COUNT):
            self._spawn_asteroid()
        for _ in range(self.INITIAL_ENEMIES):
            self._spawn_enemy()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # --- Pre-move state for reward calculation ---
        _, dist_to_asteroid_before = self._get_closest(self.player_pos, self.asteroids)
        _, dist_to_enemy_before = self._get_closest(self.player_pos, self.enemies)
        
        # --- Handle Input ---
        movement = action[0]
        space_held = action[1] == 1
        
        accel = [0, 0]
        if movement == 1: accel[1] -= self.PLAYER_ACCELERATION # Up
        if movement == 2: accel[1] += self.PLAYER_ACCELERATION # Down
        if movement == 3: accel[0] -= self.PLAYER_ACCELERATION # Left
        if movement == 4: accel[0] += self.PLAYER_ACCELERATION # Right
        
        self.player_vel[0] += accel[0]
        self.player_vel[1] += accel[1]
        
        # --- Update Player ---
        self.player_vel[0] *= self.PLAYER_FRICTION
        self.player_vel[1] *= self.PLAYER_FRICTION
        
        speed = math.sqrt(self.player_vel[0]**2 + self.player_vel[1]**2)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel[0] = (self.player_vel[0] / speed) * self.PLAYER_MAX_SPEED
            self.player_vel[1] = (self.player_vel[1] / speed) * self.PLAYER_MAX_SPEED
            
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        self.player_pos = self._world_wrap(self.player_pos)
        
        # --- Update Enemies ---
        for enemy in self.enemies:
            speed = self.ENEMY_BASE_SPEED * self.enemy_speed_multiplier
            if enemy['pattern'] == 'linear':
                enemy['pos'][0] += enemy['vel'][0] * speed
                enemy['pos'][1] += enemy['vel'][1] * speed
                # Simple bounce off walls
                if not (0 < enemy['pos'][0] < self.WIDTH): enemy['vel'][0] *= -1
                if not (0 < enemy['pos'][1] < self.HEIGHT): enemy['vel'][1] *= -1
            else: # circular
                enemy['vel']['angle'] += 0.03 * speed
                enemy['pos'][0] = enemy['vel']['center'][0] + math.cos(enemy['vel']['angle']) * enemy['vel']['radius']
                enemy['pos'][1] = enemy['vel']['center'][1] + math.sin(enemy['vel']['angle']) * enemy['vel']['radius']
            enemy['pos'] = self._world_wrap(enemy['pos'])

        # --- Update Asteroids ---
        for asteroid in self.asteroids:
            asteroid['angle'] += asteroid['rotation_speed']

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['ttl'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['ttl'] -= 1
            p['vel'][0] *= 0.98 # friction
            p['vel'][1] *= 0.98

        # --- Handle Mining ---
        if self.mining_timer > 0: self.mining_timer -= 1
        
        if space_held and self.mining_timer == 0:
            closest_asteroid, dist = self._get_closest(self.player_pos, self.asteroids)
            if closest_asteroid and dist < self.MINING_RANGE:
                # sfx: mine_ore
                closest_asteroid['ore'] -= 1
                self.ore_collected += 1
                self.score += 1
                reward += 1
                self.mining_timer = self.MINING_COOLDOWN
                
                # Bonus for risky mining
                _, dist_to_enemy = self._get_closest(self.player_pos, self.enemies)
                if dist_to_enemy < self.DANGER_RADIUS:
                    reward += 2
                    self.score += 2

                # Ore particle effect
                for _ in range(5):
                    self.particles.append({
                        'pos': list(closest_asteroid['pos']),
                        'vel': [(self.player_pos[0] - closest_asteroid['pos'][0]) / 30 + self.np_random.uniform(-1,1), 
                                (self.player_pos[1] - closest_asteroid['pos'][1]) / 30 + self.np_random.uniform(-1,1)],
                        'ttl': 30,
                        'color': self.COLOR_ORE,
                        'size': self.np_random.uniform(2, 4)
                    })

                if closest_asteroid['ore'] <= 0:
                    self.asteroids.remove(closest_asteroid)
                    self._spawn_asteroid()

        # --- Post-move reward calculation ---
        _, dist_to_asteroid_after = self._get_closest(self.player_pos, self.asteroids)
        _, dist_to_enemy_after = self._get_closest(self.player_pos, self.enemies)
        
        if dist_to_asteroid_after < dist_to_asteroid_before:
            reward += 0.1
        if dist_to_enemy_after < dist_to_enemy_before:
            reward -= 0.02
        
        # --- Handle Collisions ---
        for enemy in self.enemies:
            if self._get_dist(self.player_pos, enemy['pos']) < self.PLAYER_COLLISION_RADIUS + self.ENEMY_COLLISION_RADIUS:
                self.game_over = True
                reward = -100
                self.score -= 100
                self._create_explosion(self.player_pos)
                break
        
        # --- Update Difficulty ---
        self.enemy_speed_multiplier = 1.0 + (self.ore_collected // 25) * 0.25

        # --- Termination ---
        terminated = self.game_over or self.ore_collected >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
        if self.ore_collected >= self.WIN_SCORE and not self.game_over:
            reward += 100
            self.score += 100
            self.game_over = True # End the game on win

        self.steps += 1
        
        return (
            self._get_observation(),
            np.clip(reward, -100, 100),
            terminated,
            False,
            self._get_info()
        )

    def _render_game(self):
        # Stars (parallax effect)
        for star in self.stars:
            # Slower stars appear further away
            star_pos_x = (star['pos'][0] - self.player_pos[0] / (star['size'] * 2)) % self.WIDTH
            star_pos_y = (star['pos'][1] - self.player_pos[1] / (star['size'] * 2)) % self.HEIGHT
            color_val = 50 * star['size']
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(star_pos_x), int(star_pos_y)), star['size'] // 2)

        # Asteroids
        for asteroid in self.asteroids:
            points = []
            for vx, vy in asteroid['vertices']:
                rotated_x = vx * math.cos(asteroid['angle']) - vy * math.sin(asteroid['angle'])
                rotated_y = vx * math.sin(asteroid['angle']) + vy * math.cos(asteroid['angle'])
                points.append((asteroid['pos'][0] + rotated_x * asteroid['size'],
                               asteroid['pos'][1] + rotated_y * asteroid['size']))
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_ASTEROID)

        # Enemies
        for enemy in self.enemies:
            pygame.gfxdraw.aacircle(self.screen, int(enemy['pos'][0]), int(enemy['pos'][1]), self.ENEMY_SIZE, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, int(enemy['pos'][0]), int(enemy['pos'][1]), self.ENEMY_SIZE, self.COLOR_ENEMY)

        # Player
        if not self.game_over:
            angle = math.atan2(self.player_vel[1], self.player_vel[0]) if self.player_vel[0] or self.player_vel[1] else -math.pi/2
            p1 = (self.player_pos[0] + math.cos(angle) * self.PLAYER_SIZE,
                  self.player_pos[1] + math.sin(angle) * self.PLAYER_SIZE)
            p2 = (self.player_pos[0] + math.cos(angle + 2.2) * self.PLAYER_SIZE * 0.8,
                  self.player_pos[1] + math.sin(angle + 2.2) * self.PLAYER_SIZE * 0.8)
            p3 = (self.player_pos[0] + math.cos(angle - 2.2) * self.PLAYER_SIZE * 0.8,
                  self.player_pos[1] + math.sin(angle - 2.2) * self.PLAYER_SIZE * 0.8)
            
            points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size'] * (p['ttl']/30)), p['color'])

    def _render_ui(self):
        ore_text = self.font_ui.render(f"ORE: {self.ore_collected}/{self.WIN_SCORE}", True, self.COLOR_UI)
        self.screen.blit(ore_text, (10, 10))

        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        if self.game_over:
            if self.ore_collected >= self.WIN_SCORE:
                msg = "MISSION COMPLETE"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ore_collected": self.ore_collected,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset and initial observation
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
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

if __name__ == "__main__":
    # This block allows you to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Ore: {info['ore_collected']}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()