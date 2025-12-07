
# Generated: 2025-08-27T21:03:07.991173
# Source Brief: brief_02662.md
# Brief Index: 2662

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game objects
class Particle:
    def __init__(self, x, y, vx, vy, life, color_start, color_end, size_start, size_end):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.color_start = color_start
        self.color_end = color_end
        self.size_start = size_start
        self.size_end = size_end

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.vx *= 0.98
        self.vy *= 0.98

    def draw(self, surface):
        if self.life > 0:
            life_ratio = self.life / self.max_life
            current_size = int(self.size_start * life_ratio + self.size_end * (1 - life_ratio))
            current_color = [
                int(self.color_start[i] * life_ratio + self.color_end[i] * (1 - life_ratio))
                for i in range(3)
            ]
            if current_size > 0:
                pygame.draw.circle(surface, current_color, (int(self.x), int(self.y)), current_size)

class Projectile:
    def __init__(self, x, y, vx, vy, is_player):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.is_player = is_player
        self.radius = 4 if is_player else 3
        self.has_near_missed = False
        self.has_hit = False

class Enemy:
    def __init__(self, x, y, pattern, pattern_params, np_random):
        self.x = x
        self.y = y
        self.start_x = x
        self.start_y = y
        self.radius = 12
        self.pattern = pattern
        self.pattern_params = pattern_params
        self.pattern_timer = 0
        self.fire_cooldown = np_random.uniform(2.0, 5.0) * 30  # in frames

    def update(self, fire_rate_multiplier):
        self.pattern_timer += 1
        self.fire_cooldown -= 1 * fire_rate_multiplier

        if self.pattern == 'sine':
            self.x += self.pattern_params['speed']
            self.y = self.start_y + math.sin(self.pattern_timer * self.pattern_params['freq']) * self.pattern_params['amp']
        elif self.pattern == 'linear':
            self.x += self.pattern_params['vx']
            self.y += self.pattern_params['vy']
        elif self.pattern == 'circle':
            self.x = self.start_x + math.cos(self.pattern_timer * self.pattern_params['speed']) * self.pattern_params['radius']
            self.y = self.start_y + math.sin(self.pattern_timer * self.pattern_params['speed']) * self.pattern_params['radius']

    def should_fire(self, np_random):
        if self.fire_cooldown <= 0:
            self.fire_cooldown = np_random.uniform(2.0, 5.0) * 30
            return True
        return False

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Hold space to fire your weapon."
    )

    game_description = (
        "A visually stunning side-view space shooter. Destroy waves of procedurally generated enemies. "
        "Survive as long as you can and achieve the highest score!"
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 3000
    PLAYER_SPEED = 5
    PLAYER_FIRE_COOLDOWN = 6  # frames
    PROJECTILE_SPEED = 10
    NEAR_MISS_RADIUS = 30

    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 75, 128)
    COLOR_PLAYER_PROJECTILE = (100, 255, 100)
    COLOR_ENEMY_PROJECTILE = (255, 255, 0)
    COLOR_SCORE = (255, 255, 255)
    COLOR_HEALTH_FG = (0, 255, 0)
    COLOR_HEALTH_BG = (100, 0, 0)
    ENEMY_COLORS = [(255, 50, 50), (255, 150, 50), (200, 50, 200)]
    
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

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([0.0, 0.0])
        self.player_health = 0
        self.player_max_health = 3
        self.player_fire_timer = 0
        self.last_space_held = False
        
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = []

        self.wave = 0
        self.enemies_to_spawn = 30
        self.enemy_fire_rate_multiplier = 1.0

        self.reset()
        
        # This is a critical self-check.
        # self.validate_implementation()

    def _generate_stars(self, n=100):
        self.stars = []
        for _ in range(n):
            self.stars.append({
                'x': self.np_random.uniform(0, self.WIDTH),
                'y': self.np_random.uniform(0, self.HEIGHT),
                'speed': self.np_random.uniform(0.1, 1.0),
                'size': self.np_random.uniform(1.0, 2.5)
            })

    def _spawn_wave(self):
        self.wave += 1
        self.enemies.clear()
        self.enemy_fire_rate_multiplier = 1.0 + (self.wave - 1) * 0.1
        
        for _ in range(self.enemies_to_spawn):
            pattern_choice = self.np_random.choice(['sine', 'linear', 'circle'])
            
            y_pos = self.np_random.uniform(30, self.HEIGHT - 30)
            x_pos = self.WIDTH + self.np_random.uniform(50, 400)
            
            params = {}
            if pattern_choice == 'sine':
                params = {
                    'speed': self.np_random.uniform(-3, -1),
                    'freq': self.np_random.uniform(0.01, 0.05),
                    'amp': self.np_random.uniform(20, 100)
                }
            elif pattern_choice == 'linear':
                params = {
                    'vx': self.np_random.uniform(-4, -1.5),
                    'vy': self.np_random.uniform(-1, 1)
                }
            elif pattern_choice == 'circle':
                params = {
                    'speed': self.np_random.uniform(0.01, 0.03),
                    'radius': self.np_random.uniform(30, 80)
                }
                x_pos = self.np_random.uniform(self.WIDTH * 0.6, self.WIDTH * 0.9)
                y_pos = self.np_random.uniform(self.HEIGHT * 0.2, self.HEIGHT * 0.8)

            self.enemies.append(Enemy(x_pos, y_pos, pattern_choice, params, self.np_random))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([100.0, self.HEIGHT / 2.0])
        self.player_health = self.player_max_health
        self.player_fire_timer = 0
        self.last_space_held = False
        
        self.enemies.clear()
        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        
        self.wave = 0
        self._spawn_wave()
        if not self.stars:
            self._generate_stars()
        
        return self._get_observation(), self._get_info()

    def _create_explosion(self, x, y, num_particles, color_start, color_end):
        # sfx: explosion_sound()
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(15, 40)
            size_start = self.np_random.uniform(5, 12)
            self.particles.append(Particle(x, y, vx, vy, life, color_start, color_end, size_start, 1))

    def step(self, action):
        reward = -0.01  # Penalty for existing
        self.steps += 1
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle player movement and find nearest enemy ---
        dist_before = float('inf')
        nearest_enemy = None
        if self.enemies:
            distances = [math.hypot(e.x - self.player_pos[0], e.y - self.player_pos[1]) for e in self.enemies]
            min_dist_idx = np.argmin(distances)
            dist_before = distances[min_dist_idx]
            nearest_enemy = self.enemies[min_dist_idx]
        
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED

        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)
        
        if nearest_enemy:
            dist_after = math.hypot(nearest_enemy.x - self.player_pos[0], nearest_enemy.y - self.player_pos[1])
            if dist_after < dist_before:
                reward += 0.1
            else:
                reward -= 0.2
        
        # --- Handle player firing ---
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
        
        if space_held and not self.last_space_held and self.player_fire_timer <= 0:
            # sfx: player_shoot_sound()
            self.player_projectiles.append(Projectile(self.player_pos[0] + 20, self.player_pos[1], self.PROJECTILE_SPEED, 0, True))
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN
        self.last_space_held = space_held

        # --- Update game objects ---
        # Player engine trail
        if self.steps % 2 == 0:
            self.particles.append(Particle(
                self.player_pos[0] - 15, self.player_pos[1], 
                -2, self.np_random.uniform(-0.5, 0.5), 
                20, self.COLOR_PLAYER, self.COLOR_BG, 5, 1
            ))

        # Player projectiles
        for p in self.player_projectiles[:]:
            p.x += p.vx
            if p.x > self.WIDTH:
                if not p.has_hit:
                    reward -= 1.0 # Missed shot penalty
                self.player_projectiles.remove(p)
                continue
            
            for enemy in self.enemies[:]:
                dist = math.hypot(p.x - enemy.x, p.y - enemy.y)
                if dist < enemy.radius + p.radius:
                    self._create_explosion(enemy.x, enemy.y, 30, (255, 255, 255), self.ENEMY_COLORS[0])
                    self.enemies.remove(enemy)
                    if p in self.player_projectiles: self.player_projectiles.remove(p)
                    self.score += 100
                    reward += 10.0
                    p.has_hit = True
                    break
                elif not p.has_near_missed and dist < self.NEAR_MISS_RADIUS:
                    reward += 2.0
                    p.has_near_missed = True

        # Enemies
        for enemy in self.enemies[:]:
            enemy.update(self.enemy_fire_rate_multiplier)
            if enemy.x < -enemy.radius or enemy.x > self.WIDTH + enemy.radius * 5: # Remove far-offscreen enemies
                self.enemies.remove(enemy)
                continue

            if enemy.should_fire(self.np_random):
                # sfx: enemy_shoot_sound()
                angle = math.atan2(self.player_pos[1] - enemy.y, self.player_pos[0] - enemy.x)
                speed = self.PROJECTILE_SPEED * 0.6
                self.enemy_projectiles.append(Projectile(enemy.x, enemy.y, math.cos(angle) * speed, math.sin(angle) * speed, False))
        
        # Enemy projectiles
        for p in self.enemy_projectiles[:]:
            p.x += p.vx
            p.y += p.vy
            if not (0 < p.x < self.WIDTH and 0 < p.y < self.HEIGHT):
                self.enemy_projectiles.remove(p)
                continue
            
            player_hitbox_radius = 10
            if math.hypot(p.x - self.player_pos[0], p.y - self.player_pos[1]) < player_hitbox_radius + p.radius:
                # sfx: player_hit_sound()
                self.enemy_projectiles.remove(p)
                self.player_health -= 1
                reward -= 5.0
                self._create_explosion(self.player_pos[0], self.player_pos[1], 15, (255, 255, 255), self.COLOR_PLAYER)
                if self.player_health <= 0:
                    self.game_over = True
                    reward -= 100.0

        # Particles
        for particle in self.particles[:]:
            particle.update()
            if particle.life <= 0:
                self.particles.remove(particle)

        # Stars
        for star in self.stars:
            star['x'] -= star['speed']
            if star['x'] < 0:
                star['x'] = self.WIDTH
                star['y'] = self.np_random.uniform(0, self.HEIGHT)
        
        # --- Check for wave clear ---
        if not self.enemies and not self.game_over:
            reward += 100
            self.score += 1000
            self._spawn_wave()
            # Add a brief pause effect or text
            
        # --- Termination ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Stars (background)
        for star in self.stars:
            size = int(star['size'])
            color_val = int(star['speed'] * 80) + 20
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(star['x']), int(star['y'])), max(1, size // 2))

        # Particles
        for p in self.particles:
            p.draw(self.screen)

        # Projectiles
        for p in self.player_projectiles:
            pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJECTILE, (p.x, p.y), (p.x-10, p.y), 3)
        for p in self.enemy_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_ENEMY_PROJECTILE, (int(p.x), int(p.y)), p.radius)

        # Enemies
        for enemy in self.enemies:
            color = self.ENEMY_COLORS[hash(enemy.pattern) % len(self.ENEMY_COLORS)]
            points = [
                (enemy.x + enemy.radius, enemy.y),
                (enemy.x - enemy.radius / 2, enemy.y - enemy.radius / 1.5),
                (enemy.x - enemy.radius, enemy.y),
                (enemy.x - enemy.radius / 2, enemy.y + enemy.radius / 1.5)
            ]
            pygame.gfxdraw.aapolygon(self.screen, [(int(px), int(py)) for px, py in points], color)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(px), int(py)) for px, py in points], color)

        # Player
        if self.player_health > 0:
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            player_points = [
                (px + 15, py),
                (px - 15, py - 10),
                (px - 10, py),
                (px - 15, py + 10)
            ]
            # Glow effect
            pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER_GLOW)
            # Main ship
            player_points_inner = [
                (px + 13, py),
                (px - 13, py - 8),
                (px - 8, py),
                (px - 13, py + 8)
            ]
            pygame.gfxdraw.aapolygon(self.screen, player_points_inner, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, player_points_inner, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Health Bar
        health_bar_width = 150
        health_bar_height = 15
        health_ratio = self.player_health / self.player_max_health
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, health_bar_width, health_bar_height))
        if health_ratio > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, int(health_bar_width * health_ratio), health_bar_height))

        # Game Over Text
        if self.game_over:
            over_text = self.font_large.render("GAME OVER", True, (255, 0, 0))
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_text, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health, "wave": self.wave}

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Space Shooter")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        movement = 0
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
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000)

        clock.tick(30)  # Run at 30 FPS

    env.close()