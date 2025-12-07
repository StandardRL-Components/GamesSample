
# Generated: 2025-08-28T01:52:53.370062
# Source Brief: brief_04262.md
# Brief Index: 4262

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game entities
class Particle:
    def __init__(self, pos, vel, size, color, lifespan, gravity_target=None):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.size = size
        self.color = color
        self.lifespan = lifespan
        self.gravity_target = gravity_target

    def update(self):
        if self.gravity_target:
            direction = self.gravity_target - self.pos
            if direction.length() > 1:
                # Apply gravity, stronger when closer
                accel = direction.normalize() * (100 / max(1, direction.length()))
                self.vel = self.vel * 0.95 + accel * 0.5
        self.pos += self.vel
        self.lifespan -= 1
        self.size = max(0, self.size - 0.05)
        return self.lifespan > 0 and self.size > 0

class Enemy:
    def __init__(self, pos, enemy_type, screen_dims, seed):
        self.pos = pygame.math.Vector2(pos)
        self.type = enemy_type
        self.size = 12
        self.screen_width, self.screen_height = screen_dims
        self.rng = np.random.default_rng(seed)
        self.fire_cooldown = self.rng.integers(30, 90)
        self.base_fire_rate = 120 # Lower is faster

        if self.type == 'red': # Linear patrol
            self.vel = pygame.math.Vector2(self.rng.uniform(-2, 2), self.rng.uniform(-2, 2))
            if self.vel.length() == 0: self.vel.x = 1
            self.vel.scale_to_length(self.rng.uniform(1.0, 1.8))
        elif self.type == 'blue': # Circular patrol
            self.center = pygame.math.Vector2(pos)
            self.radius = self.rng.uniform(40, 80)
            self.angle = self.rng.uniform(0, 2 * math.pi)
            self.angular_vel = self.rng.uniform(0.02, 0.04) * self.rng.choice([-1, 1])
        elif self.type == 'purple': # Sinusoidal patrol
            self.center_y = pos[1]
            self.amplitude = self.rng.uniform(50, 150)
            self.frequency = self.rng.uniform(0.02, 0.05)
            self.phase = 0
            self.vel_x = self.rng.choice([-1, 1]) * 1.5
            self.pos.x = self.rng.integers(self.amplitude, self.screen_width - self.amplitude)

    def update(self, player_pos):
        # Movement
        if self.type == 'red':
            self.pos += self.vel
            if self.pos.x < self.size or self.pos.x > self.screen_width - self.size: self.vel.x *= -1
            if self.pos.y < self.size or self.pos.y > self.screen_height - self.size: self.vel.y *= -1
        elif self.type == 'blue':
            self.angle += self.angular_vel
            self.pos.x = self.center.x + self.radius * math.cos(self.angle)
            self.pos.y = self.center.y + self.radius * math.sin(self.angle)
        elif self.type == 'purple':
            self.pos.x += self.vel_x
            if self.pos.x < self.amplitude or self.pos.x > self.screen_width - self.amplitude:
                self.vel_x *= -1
            self.phase += self.frequency
            self.pos.y = self.center_y + self.amplitude * math.sin(self.phase)

        self.pos.x = np.clip(self.pos.x, self.size, self.screen_width - self.size)
        self.pos.y = np.clip(self.pos.y, self.size, self.screen_height - self.size)

        # Firing
        self.fire_cooldown -= 1
        if self.fire_cooldown <= 0:
            self.fire_cooldown = self.base_fire_rate + self.rng.integers(-10, 10)
            direction = player_pos - self.pos
            if direction.length() > 0:
                # SFX: Enemy Shoot
                return Projectile(self.pos, direction.normalize() * 4)
        return None

class Projectile:
    def __init__(self, pos, vel):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.size = 4
        self.lifespan = 150

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        return self.lifespan > 0

class Asteroid:
    def __init__(self, pos, size, seed):
        self.pos = pygame.math.Vector2(pos)
        self.size = size
        self.angle = 0
        self.rng = np.random.default_rng(seed)
        self.rotation_speed = self.rng.uniform(-1.5, 1.5)
        num_points = self.rng.integers(7, 12)
        self.shape_points = []
        for i in range(num_points):
            a = 2 * math.pi * i / num_points
            r = self.rng.uniform(self.size * 0.8, self.size * 1.2)
            self.shape_points.append((math.cos(a) * r, math.sin(a) * r))

    def update(self):
        self.angle += self.rotation_speed

    def get_rotated_points(self):
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        points = []
        for x, y in self.shape_points:
            rx = x * cos_a - y * sin_a + self.pos.x
            ry = x * sin_a + y * cos_a + self.pos.y
            points.append((int(rx), int(ry)))
        return points

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Arrow keys to move. Hold space to mine asteroids. Avoid red enemies and their projectiles."
    game_description = "Pilot a mining ship, blast asteroids for ore, and evade cunning enemies in a visually vibrant asteroid field to collect 100 units of ore."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_SPEED = 4.0
    PLAYER_SIZE = 12
    WIN_ORE_COUNT = 100
    MAX_STEPS = 1800 # 60 seconds at 30fps
    NUM_ASTEROIDS = 8
    NUM_ENEMIES = 3
    BONUS_PROXIMITY = 120

    # --- Colors ---
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_ASTEROID = (120, 110, 100)
    COLOR_ASTEROID_OUTLINE = (160, 150, 140)
    COLOR_ORE = (255, 223, 0)
    COLOR_ENEMY_RED = (255, 50, 50)
    COLOR_ENEMY_BLUE = (50, 150, 255)
    COLOR_ENEMY_PURPLE = (200, 50, 255)
    COLOR_PROJECTILE = (255, 100, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BAR = (0, 180, 100)
    COLOR_UI_BAR_BG = (50, 50, 70)
    
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        self.render_mode = render_mode
        self.seed = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
        
        self.rng = np.random.default_rng(self.seed)
        
        self.steps = 0
        self.score = 0
        self.ore = 0
        self.game_over = False
        self.game_won = False
        self.bonus_timer = 0
        
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_angle = -90
        
        self.asteroids = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self._spawn_stars()
        self._spawn_initial_entities()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01 # Small time penalty

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_player_movement(movement)
        
        # --- Update Game State ---
        if space_held:
            mine_reward, bonus = self._handle_mining()
            reward += mine_reward
            if bonus: self.bonus_timer = 60 # 2 seconds
        
        self._update_enemies()
        self._update_projectiles()
        self._update_asteroids()
        self.particles = [p for p in self.particles if p.update()]
        
        ore_reward = self._collect_ore()
        reward += ore_reward

        # --- Check Collisions & Termination ---
        terminated = self._check_collisions()
        if terminated:
            reward -= 100
            self.game_over = True
            # SFX: Player Explosion
            self._create_explosion(self.player_pos, self.COLOR_PLAYER)
        
        if self.ore >= self.WIN_ORE_COUNT and not self.game_over:
            reward += 100
            terminated = True
            self.game_over = True
            self.game_won = True
        
        if self.steps >= self.MAX_STEPS and not self.game_over:
            terminated = True
            self.game_over = True

        self.steps += 1
        if self.bonus_timer > 0: self.bonus_timer -= 1
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _spawn_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                "pos": (self.rng.integers(0, self.WIDTH), self.rng.integers(0, self.HEIGHT)),
                "size": self.rng.choice([1, 1, 1, 2]),
                "color": self.rng.integers(50, 100)
            })

    def _spawn_initial_entities(self):
        # Spawn player
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        # Spawn asteroids
        for _ in range(self.NUM_ASTEROIDS):
            self._spawn_asteroid()

        # Spawn enemies
        enemy_types = ['red', 'blue', 'purple']
        for i in range(self.NUM_ENEMIES):
            pos = self._get_safe_spawn_pos(100)
            enemy_seed = self.rng.integers(0, 2**32-1)
            enemy = Enemy(pos, enemy_types[i % len(enemy_types)], (self.WIDTH, self.HEIGHT), enemy_seed)
            self.enemies.append(enemy)

    def _get_safe_spawn_pos(self, min_dist_from_player):
        while True:
            pos = pygame.math.Vector2(self.rng.integers(20, self.WIDTH - 20), self.rng.integers(20, self.HEIGHT - 20))
            if pos.distance_to(self.player_pos) > min_dist_from_player:
                return pos

    def _spawn_asteroid(self):
        pos = self._get_safe_spawn_pos(80)
        size = self.rng.integers(15, 25)
        asteroid_seed = self.rng.integers(0, 2**32-1)
        self.asteroids.append(Asteroid(pos, size, asteroid_seed))

    def _handle_player_movement(self, movement):
        vel = pygame.math.Vector2(0, 0)
        if movement == 1: vel.y = -1 # Up
        elif movement == 2: vel.y = 1  # Down
        elif movement == 3: vel.x = -1 # Left
        elif movement == 4: vel.x = 1  # Right
        
        if vel.length() > 0:
            vel.scale_to_length(self.PLAYER_SPEED)
            self.player_angle = vel.angle_to(pygame.math.Vector2(1, 0))
            # SFX: Ship Thrust
            self._create_thrust_particles()

        self.player_pos += vel
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _handle_mining(self):
        reward = 0
        is_bonus = False
        beam_end = self.player_pos + pygame.math.Vector2(1, 0).rotate(-self.player_angle) * 150
        
        mined_asteroid = None
        for asteroid in self.asteroids:
            if asteroid.pos.distance_to(self.player_pos) < 150 + asteroid.size:
                # Simple line-circle intersection check
                if self._line_circle_intersect(self.player_pos, beam_end, asteroid.pos, asteroid.size):
                    mined_asteroid = asteroid
                    break
        
        if mined_asteroid:
            # SFX: Asteroid Explode
            reward += 1.0
            
            # Check for bonus
            for enemy in self.enemies:
                if self.player_pos.distance_to(enemy.pos) < self.BONUS_PROXIMITY:
                    is_bonus = True
                    break
            
            num_ore = self.rng.integers(3, 6)
            for _ in range(num_ore):
                angle = self.rng.uniform(0, 360)
                speed = self.rng.uniform(2, 5)
                vel = pygame.math.Vector2(speed, 0).rotate(angle)
                p_color = (255, 255, 100) if is_bonus else self.COLOR_ORE
                self.particles.append(Particle(mined_asteroid.pos, vel, 5, p_color, 120, self.player_pos))
            
            self._create_explosion(mined_asteroid.pos, self.COLOR_ASTEROID_OUTLINE, count=20, max_speed=2)
            self.asteroids.remove(mined_asteroid)
            
        # Respawn asteroids to maintain count
        if len(self.asteroids) < self.NUM_ASTEROIDS:
            self._spawn_asteroid()
            
        return reward, is_bonus
        
    def _update_enemies(self):
        # Difficulty scaling
        ore_milestone = self.ore // 25
        fire_rate_mod = max(0.5, 1.0 - ore_milestone * 0.1)

        for enemy in self.enemies:
            enemy.base_fire_rate = 120 * fire_rate_mod
            new_projectile = enemy.update(self.player_pos)
            if new_projectile:
                self.projectiles.append(new_projectile)

    def _update_projectiles(self):
        self.projectiles = [p for p in self.projectiles if p.update() and 0 < p.pos.x < self.WIDTH and 0 < p.pos.y < self.HEIGHT]

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid.update()
            
    def _collect_ore(self):
        ore_reward = 0
        for p in self.particles[:]:
            if p.gravity_target and p.pos.distance_to(p.gravity_target) < self.PLAYER_SIZE:
                is_bonus = p.color != self.COLOR_ORE
                ore_reward += 0.5 if is_bonus else 0.1
                self.ore += 1
                # SFX: Ore Collect
                self.particles.remove(p)
        return ore_reward

    def _check_collisions(self):
        # Player vs Enemies
        for enemy in self.enemies:
            if self.player_pos.distance_to(enemy.pos) < self.PLAYER_SIZE + enemy.size:
                return True
        # Player vs Projectiles
        for proj in self.projectiles:
            if self.player_pos.distance_to(proj.pos) < self.PLAYER_SIZE + proj.size:
                return True
        return False
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "ore": self.ore}
        
    def _render_game(self):
        # Stars
        for star in self.stars:
            c = star["color"]
            pygame.draw.circle(self.screen, (c, c, c), star["pos"], star["size"])
        
        # Asteroids
        for asteroid in self.asteroids:
            points = asteroid.get_rotated_points()
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID_OUTLINE)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
        
        # Enemies
        for enemy in self.enemies:
            color = self.COLOR_ENEMY_RED
            if enemy.type == 'blue': color = self.COLOR_ENEMY_BLUE
            if enemy.type == 'purple': color = self.COLOR_ENEMY_PURPLE
            glow_color = (*color, 50)
            pygame.draw.circle(self.screen, glow_color, (int(enemy.pos.x), int(enemy.pos.y)), enemy.size + 5)
            pygame.draw.circle(self.screen, color, (int(enemy.pos.x), int(enemy.pos.y)), enemy.size)
            
        # Projectiles
        for proj in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(proj.pos.x), int(proj.pos.y)), proj.size)
        
        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p.color, (int(p.pos.x), int(p.pos.y)), int(p.size))

        # Player
        if not self.game_over or self.game_won:
            p1 = self.player_pos + pygame.math.Vector2(self.PLAYER_SIZE, 0).rotate(-self.player_angle)
            p2 = self.player_pos + pygame.math.Vector2(-self.PLAYER_SIZE, -self.PLAYER_SIZE*0.7).rotate(-self.player_angle)
            p3 = self.player_pos + pygame.math.Vector2(-self.PLAYER_SIZE, self.PLAYER_SIZE*0.7).rotate(-self.player_angle)
            points = [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y))]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
            
    def _render_ui(self):
        # Ore progress bar
        bar_width = self.WIDTH - 40
        progress = min(1.0, self.ore / self.WIN_ORE_COUNT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (20, 10, bar_width, 15))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (20, 10, int(bar_width * progress), 15))

        # Ore text
        ore_text = self.font_large.render(f"ORE: {self.ore}/{self.WIN_ORE_COUNT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ore_text, (20, 30))
        
        # Score text
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 30))
        
        # Bonus text
        if self.bonus_timer > 0:
            bonus_text = self.font_large.render("BONUS!", True, self.COLOR_ORE)
            alpha = min(255, int(255 * (self.bonus_timer / 30.0)))
            bonus_text.set_alpha(alpha)
            self.screen.blit(bonus_text, (self.WIDTH/2 - bonus_text.get_width()/2, 40))

        if self.game_over:
            msg = "MISSION COMPLETE" if self.game_won else "SHIP DESTROYED"
            color = self.COLOR_PLAYER if self.game_won else self.COLOR_ENEMY_RED
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))
            
    def _line_circle_intersect(self, p1, p2, circle_center, r):
        # Simplified check, good enough for game feel
        d = p2 - p1
        f = p1 - circle_center
        a = d.dot(d)
        b = 2 * f.dot(d)
        c = f.dot(f) - r * r
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return False
        else:
            discriminant = math.sqrt(discriminant)
            t1 = (-b - discriminant) / (2*a)
            t2 = (-b + discriminant) / (2*a)
            return (0 <= t1 <= 1) or (0 <= t2 <= 1)

    def _create_explosion(self, pos, color, count=50, max_speed=5):
        for _ in range(count):
            angle = self.rng.uniform(0, 360)
            speed = self.rng.uniform(1, max_speed)
            vel = pygame.math.Vector2(speed, 0).rotate(angle)
            size = self.rng.uniform(2, 5)
            lifespan = self.rng.integers(20, 50)
            self.particles.append(Particle(pos, vel, size, color, lifespan))

    def _create_thrust_particles(self):
        if self.steps % 2 == 0:
            angle = self.player_angle + 180 + self.rng.uniform(-15, 15)
            speed = self.rng.uniform(1, 3)
            vel = pygame.math.Vector2(speed, 0).rotate(-angle)
            pos = self.player_pos + pygame.math.Vector2(-self.PLAYER_SIZE * 0.8, 0).rotate(-self.player_angle)
            self.particles.append(Particle(pos, vel, 3, (200, 220, 255), 15))

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Must be set for pygame to run headlessly
    
    env = GameEnv(render_mode="rgb_array")
    
    # To run with manual controls:
    # 1. Comment out the os.environ line above.
    # 2. Add `pygame.display.set_mode((env.WIDTH, env.HEIGHT))` in __init__.
    # 3. Add `pygame.display.flip()` in _get_observation.
    # 4. Use the code below.
    
    # Example of running the environment for a few steps
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    
    # --- For manual play ---
    # To enable manual play, you need to modify the __init__ slightly.
    # 1. After `self.screen = ...`, add `self.display_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))`
    # 2. In `_get_observation`, after `self._render_ui()`, add `self.display_screen.blit(self.screen, (0, 0))` and `pygame.display.flip()`
    # 3. Uncomment the os.environ line above.
    # Then run this script.
    
    # play = True
    # if play:
    #     obs, info = env.reset()
    #     terminated = False
    #     total_reward = 0
    #     while not terminated:
    #         keys = pygame.key.get_pressed()
    #         movement = 0 # none
    #         if keys[pygame.K_UP]: movement = 1
    #         elif keys[pygame.K_DOWN]: movement = 2
    #         elif keys[pygame.K_LEFT]: movement = 3
    #         elif keys[pygame.K_RIGHT]: movement = 4
            
    #         space_held = 1 if keys[pygame.K_SPACE] else 0
    #         shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
    #         action = [movement, space_held, shift_held]
            
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         total_reward += reward
            
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 terminated = True
            
    #         env.clock.tick(30)
            
    #     print(f"Game Over! Final Score: {total_reward}, Info: {info}")
    #     pygame.quit()
    # else: # Default headless run
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"Episode finished. Total reward: {total_reward}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0