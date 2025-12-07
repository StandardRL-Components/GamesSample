
# Generated: 2025-08-28T03:20:51.316584
# Source Brief: brief_04890.md
# Brief Index: 4890

        
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

    user_guide = (
        "Controls: ↑↓←→ to move. Hold Shift for a temporary shield. Press Space to fire."
    )

    game_description = (
        "Survive waves of increasingly difficult asteroid and laser barrages in this top-down space shooter."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_INVINCIBLE = (255, 255, 255)
    COLOR_ASTEROID = (120, 130, 150)
    COLOR_LASER_WARN = (255, 0, 0, 100)
    COLOR_LASER_BEAM = (255, 50, 50)
    COLOR_PROJECTILE = (200, 255, 255)
    COLOR_SHIELD = (100, 150, 255, 128)
    COLOR_EXPLOSION = (255, 180, 50)
    COLOR_TEXT = (220, 220, 220)
    
    # Player settings
    PLAYER_SPEED = 5
    PLAYER_SIZE = 12
    PLAYER_MAX_HEALTH = 3
    PLAYER_INVINCIBILITY_FRAMES = 90
    
    # Weapon settings
    FIRE_COOLDOWN = 8
    
    # Shield settings
    SHIELD_DURATION = 5
    SHIELD_COOLDOWN = 120

    # Game settings
    MAX_STEPS = 3000 # Increased for 10 waves
    WIN_WAVE = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        self.stars = []
        self.projectiles = []
        self.asteroids = []
        self.lasers = []
        self.particles = []

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.win = False

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_invincible_timer = 0
        
        self.fire_cooldown_timer = 0
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0
        
        self.projectiles.clear()
        self.asteroids.clear()
        self.lasers.clear()
        self.particles.clear()

        if not self.stars:
            self._generate_stars()
        
        self._start_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Base survival reward
        reward += 0.01

        self._handle_input(action)
        self._update_game_state()
        
        collision_reward, collision_termination = self._handle_collisions()
        reward += collision_reward
        
        self.steps += 1
        
        if not self.asteroids:
            self.wave += 1
            if self.wave > self.WIN_WAVE:
                self.win = True
                self.game_over = True
                reward += 100
            else:
                self._start_wave()
                # SFX: Wave Start
        
        if self.player_health <= 0:
            self.game_over = True
            reward -= 100
            self._create_explosion(self.player_pos, 50, self.COLOR_PLAYER)
            # SFX: Player Death Explosion

        if self.steps >= self.MAX_STEPS and not self.game_over:
             self.game_over = True # End due to time limit

        terminated = self.game_over or collision_termination

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

        # Firing
        if space_held and self.fire_cooldown_timer == 0:
            self.projectiles.append({'pos': self.player_pos.copy(), 'vel': np.array([0, -10])})
            self.fire_cooldown_timer = self.FIRE_COOLDOWN
            # SFX: Pew!

        # Shield
        if shift_held and self.shield_cooldown_timer == 0:
            self.shield_active = True
            self.shield_timer = self.SHIELD_DURATION
            self.shield_cooldown_timer = self.SHIELD_COOLDOWN
            # SFX: Shield Up!

    def _update_game_state(self):
        # Timers
        if self.fire_cooldown_timer > 0: self.fire_cooldown_timer -= 1
        if self.player_invincible_timer > 0: self.player_invincible_timer -= 1
        if self.shield_cooldown_timer > 0: self.shield_cooldown_timer -= 1
        if self.shield_timer > 0: self.shield_timer -= 1
        else: 
            if self.shield_active:
                # SFX: Shield Down
                self.shield_active = False

        # Projectiles
        self.projectiles = [p for p in self.projectiles if 0 < p['pos'][1] < self.SCREEN_HEIGHT]
        for p in self.projectiles:
            p['pos'] += p['vel']

        # Asteroids
        for a in self.asteroids:
            a['pos'] += a['vel']
            a['angle'] += a['rot_speed']
        self.asteroids = [a for a in self.asteroids if -a['size'] < a['pos'][1] < self.SCREEN_HEIGHT + a['size']]
        
        # Lasers
        laser_spawn_prob = 0.005 + self.wave * 0.005
        if self.np_random.random() < laser_spawn_prob:
            y_pos = self.np_random.integers(50, self.SCREEN_HEIGHT - 100)
            self.lasers.append({'y': y_pos, 'timer': 45}) # 1.5s warning
            # SFX: Laser Charge
        
        for l in self.lasers:
            l['timer'] -= 1
        self.lasers = [l for l in self.lasers if l['timer'] > -15]

        # Particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['type'] == 'explosion':
                p['radius'] *= 0.95
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Projectiles vs Asteroids
        for p in self.projectiles[:]:
            for a in self.asteroids[:]:
                if np.linalg.norm(p['pos'] - a['pos']) < a['size']:
                    self.projectiles.remove(p)
                    self.asteroids.remove(a)
                    self.score += 1
                    reward += 10
                    self._create_explosion(a['pos'], int(a['size']), self.COLOR_ASTEROID)
                    # SFX: Asteroid Explosion
                    break

        if self.player_invincible_timer > 0 or self.shield_active:
            return reward, False

        # Player vs Asteroids
        for a in self.asteroids:
            if np.linalg.norm(self.player_pos - a['pos']) < self.PLAYER_SIZE + a['size'] * 0.8:
                self.player_health -= 1
                reward -= 20
                self.player_invincible_timer = self.PLAYER_INVINCIBILITY_FRAMES
                self._create_explosion(self.player_pos, 20, self.COLOR_PLAYER)
                # SFX: Player Hit
                return reward, self.player_health <= 0

        # Player vs Lasers
        for l in self.lasers:
            if l['timer'] <= 0: # Only active lasers
                if abs(self.player_pos[1] - l['y']) < 5: # 5px laser thickness
                    self.player_health -= 1
                    reward -= 10
                    self.player_invincible_timer = self.PLAYER_INVINCIBILITY_FRAMES
                    self._create_explosion(self.player_pos, 20, self.COLOR_LASER_BEAM)
                    # SFX: Player Zapped
                    return reward, self.player_health <= 0
        
        return reward, False

    def _start_wave(self):
        num_asteroids = 4 + self.wave
        for _ in range(num_asteroids):
            pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(-100, -20)], dtype=np.float32)
            speed = 1.0 + self.wave * 0.2 + self.np_random.uniform(-0.2, 0.2)
            vel = np.array([self.np_random.uniform(-0.5, 0.5), speed], dtype=np.float32)
            size = self.np_random.uniform(15, 30)
            rot_speed = self.np_random.uniform(-0.05, 0.05)
            self.asteroids.append({'pos': pos, 'vel': vel, 'size': size, 'angle': 0, 'rot_speed': rot_speed, 'shape': self._create_asteroid_shape(size)})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "health": self.player_health}

    def _render_game(self):
        # Stars
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['size'])

        # Lasers
        for l in self.lasers:
            if l['timer'] > 0: # Warning
                s = pygame.Surface((self.SCREEN_WIDTH, 5), pygame.SRCALPHA)
                alpha = int(150 * (math.sin(l['timer'] * 0.2)**2))
                s.fill((255, 0, 0, alpha))
                self.screen.blit(s, (0, l['y'] - 2))
            else: # Firing
                pygame.draw.line(self.screen, self.COLOR_LASER_BEAM, (0, l['y']), (self.SCREEN_WIDTH, l['y']), 5)
                pygame.draw.line(self.screen, (255,255,255), (0, l['y']), (self.SCREEN_WIDTH, l['y']), 2)

        # Asteroids
        for a in self.asteroids:
            rotated_shape = [
                (x * math.cos(a['angle']) - y * math.sin(a['angle']) + a['pos'][0],
                 x * math.sin(a['angle']) + y * math.cos(a['angle']) + a['pos'][1])
                for x, y in a['shape']
            ]
            pygame.gfxdraw.aapolygon(self.screen, rotated_shape, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, rotated_shape, self.COLOR_ASTEROID)

        # Projectiles
        for p in self.projectiles:
            pos = p['pos'].astype(int)
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, (pos[0]-1, pos[1]-5, 3, 10))

        # Player
        if self.player_health > 0:
            player_color = self.COLOR_PLAYER
            if self.player_invincible_timer > 0 and self.steps % 10 < 5:
                player_color = self.COLOR_PLAYER_INVINCIBLE

            p = self.player_pos
            s = self.PLAYER_SIZE
            points = [(p[0], p[1] - s), (p[0] - s/2, p[1] + s/2), (p[0] + s/2, p[1] + s/2)]
            pygame.gfxdraw.aapolygon(self.screen, points, player_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, player_color)
        
        # Shield
        if self.shield_active:
            radius = int(self.PLAYER_SIZE * 2.5)
            temp_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            alpha = 100 + 50 * math.sin(self.steps * 0.3)
            pygame.gfxdraw.filled_circle(temp_surface, radius, radius, radius, (*self.COLOR_SHIELD[:3], int(alpha)))
            pygame.gfxdraw.aacircle(temp_surface, radius, radius, radius, (200, 220, 255))
            self.screen.blit(temp_surface, (int(self.player_pos[0]-radius), int(self.player_pos[1]-radius)))

        # Particles
        for p in self.particles:
            color = p['color']
            if p['type'] == 'explosion':
                alpha = int(255 * (p['life'] / p['max_life']))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), (*color, alpha))
            else: # trail, etc.
                pygame.draw.circle(self.screen, color, p['pos'].astype(int), int(p['life']/2))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.wave}/{self.WIN_WAVE}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        
        # Health Bar
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width = 150
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = self.SCREEN_HEIGHT - bar_height - 10
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x, bar_y, bar_width * health_pct, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # Game Over / Win Message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            pos = (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT))
            size = self.np_random.choice([1, 1, 1, 2, 2])
            brightness = self.np_random.integers(50, 150)
            color = (brightness, brightness, brightness)
            self.stars.append({'pos': pos, 'size': size, 'color': color})

    def _create_explosion(self, pos, num_particles, base_color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': base_color,
                'radius': self.np_random.uniform(2,5),
                'type': 'explosion'
            })

    def _create_asteroid_shape(self, base_size):
        points = []
        num_vertices = self.np_random.integers(7, 12)
        for i in range(num_vertices):
            angle = (2 * math.pi / num_vertices) * i
            dist = self.np_random.uniform(base_size * 0.7, base_size)
            points.append((dist * math.cos(angle), dist * math.sin(angle)))
        return points

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        # Test specific game logic
        self.reset()
        self.player_health = 3
        assert self.player_health <= self.PLAYER_MAX_HEALTH
        self.wave = 10
        assert self.wave <= self.WIN_WAVE
        self.score = 0
        self.asteroids = [{'pos': self.player_pos + np.array([0, -20]), 'size': 20, 'vel': np.array([0,1]), 'angle': 0, 'rot_speed': 0, 'shape': self._create_asteroid_shape(20)}]
        self.projectiles = [{'pos': self.player_pos + np.array([0, -21]), 'vel': np.array([0, -10])}]
        self._handle_collisions()
        assert self.score == 1
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Gymnasium's auto_advance is for RL loops. For human play, we use a manual loop.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # none
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
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
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()