
# Generated: 2025-08-27T20:01:48.335180
# Source Brief: brief_02326.md
# Brief Index: 2326

        
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
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.initial_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.radius = max(0, self.radius * (self.lifespan / self.initial_lifespan))

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.initial_lifespan))
            r, g, b = self.color
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (r, g, b, alpha), (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)), special_flags=pygame.BLEND_RGBA_ADD)

class Projectile:
    def __init__(self, pos, vel):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = 3
        self.color = (255, 255, 0) # Bright Yellow

    def update(self):
        self.pos += self.vel

    def draw(self, surface):
        start_pos = self.pos - self.vel.normalize() * 10
        pygame.draw.line(surface, self.color, (int(start_pos.x), int(start_pos.y)), (int(self.pos.x), int(self.pos.y)), 2)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)

class PowerUp:
    TYPE_COLORS = {
        "rapid_fire": (255, 60, 60),
        "shield": (60, 255, 60),
        "score_multiplier": (255, 255, 60),
    }
    TYPE_LETTERS = {
        "rapid_fire": "R",
        "shield": "S",
        "score_multiplier": "X2",
    }

    def __init__(self, pos, type, font):
        self.pos = pygame.Vector2(pos)
        self.type = type
        self.radius = 12
        self.color = self.TYPE_COLORS[type]
        self.pulse_phase = random.uniform(0, 2 * math.pi)
        self.font = font
        self.letter = self.TYPE_LETTERS[type]
        self.text_surf = self.font.render(self.letter, True, (0, 0, 0))
        self.text_rect = self.text_surf.get_rect()

    def update(self):
        self.pulse_phase += 0.1

    def draw(self, surface):
        pulse_radius = self.radius + 2 * math.sin(self.pulse_phase)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(pulse_radius), self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(pulse_radius), self.color)
        
        self.text_rect.center = (int(self.pos.x), int(self.pos.y))
        surface.blit(self.text_surf, self.text_rect)


class Asteroid:
    def __init__(self, pos, vel, size, np_random):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = size * 10
        self.color = (random.randint(100, 150), random.randint(100, 150), random.randint(100, 150))
        self.angle = 0
        self.rot_speed = np_random.uniform(-2, 2)
        
        num_points = np_random.integers(7, 12)
        self.shape_points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dist = np_random.uniform(0.7, 1.0) * self.radius
            self.shape_points.append((math.cos(angle) * dist, math.sin(angle) * dist))

    def update(self):
        self.pos += self.vel
        self.angle += self.rot_speed

    def draw(self, surface):
        rotated_points = []
        rad_angle = math.radians(self.angle)
        cos_a = math.cos(rad_angle)
        sin_a = math.sin(rad_angle)
        for x, y in self.shape_points:
            new_x = self.pos.x + x * cos_a - y * sin_a
            new_y = self.pos.y + x * sin_a + y * cos_a
            rotated_points.append((int(new_x), int(new_y)))
        
        if len(rotated_points) > 2:
            pygame.gfxdraw.aapolygon(surface, rotated_points, self.color)
            pygame.gfxdraw.filled_polygon(surface, rotated_points, self.color)

class Player:
    def __init__(self, pos, screen_width, screen_height):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.radius = 15
        self.color = (0, 200, 255)
        self.glow_color = (0, 100, 150)
        self.health = 3
        self.max_health = 3
        self.fire_cooldown = 0
        self.base_fire_rate = 15 # frames
        self.powerup_timers = {"rapid_fire": 0, "shield": 0, "score_multiplier": 0}
        self.invulnerable_timer = 0
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Ship shape
        self.shape_points = [pygame.Vector2(0, -self.radius), pygame.Vector2(-self.radius*0.7, self.radius*0.7), pygame.Vector2(self.radius*0.7, self.radius*0.7)]

    def update(self, movement_action, particles):
        # Movement
        accel = 0.8
        damping = 0.92
        if movement_action == 1: self.vel.y -= accel # Up
        if movement_action == 2: self.vel.y += accel # Down
        if movement_action == 3: self.vel.x -= accel # Left
        if movement_action == 4: self.vel.x += accel # Right
        
        self.vel *= damping
        self.pos += self.vel

        # Boundaries
        self.pos.x = np.clip(self.pos.x, self.radius, self.screen_width - self.radius)
        self.pos.y = np.clip(self.pos.y, self.radius, self.screen_height - self.radius)

        # Timers
        self.fire_cooldown = max(0, self.fire_cooldown - 1)
        self.invulnerable_timer = max(0, self.invulnerable_timer - 1)
        for p_type in self.powerup_timers:
            self.powerup_timers[p_type] = max(0, self.powerup_timers[p_type] - 1)

        # Thruster particles
        if self.vel.length() > 0.5:
            for _ in range(2):
                p_pos = self.pos - self.vel.normalize() * self.radius
                p_vel = -self.vel.normalize() * random.uniform(1, 3) + pygame.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
                p_radius = random.uniform(2, 5)
                p_lifespan = 20
                p_color = random.choice([(255, 100, 0), (255, 200, 0)])
                particles.append(Particle(p_pos, p_vel, p_radius, p_color, p_lifespan))
    
    def shoot(self, projectiles):
        fire_rate = self.base_fire_rate / 2 if self.powerup_timers["rapid_fire"] > 0 else self.base_fire_rate
        if self.fire_cooldown == 0:
            proj_vel = pygame.Vector2(15, 0)
            projectiles.append(Projectile(self.pos, proj_vel))
            self.fire_cooldown = fire_rate
            return True
        return False

    def take_damage(self):
        if self.invulnerable_timer == 0 and self.powerup_timers["shield"] == 0:
            self.health -= 1
            self.invulnerable_timer = 120 # 2 seconds of invulnerability
            return True
        return False
        
    def draw(self, surface):
        # Glow
        if self.invulnerable_timer > 0 and (self.invulnerable_timer // 6) % 2 == 0:
            return # Flicker when invulnerable
            
        glow_radius = int(self.radius * (1.5 + 0.2 * math.sin(pygame.time.get_ticks() / 200)))
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*self.glow_color, 80), (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (int(self.pos.x - glow_radius), int(self.pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Ship body
        rotated_points = [p.rotate(90) + self.pos for p in self.shape_points]
        pygame.gfxdraw.aapolygon(surface, rotated_points, self.color)
        pygame.gfxdraw.filled_polygon(surface, rotated_points, self.color)

        # Shield
        if self.powerup_timers["shield"] > 0:
            shield_color = (60, 255, 60)
            alpha = 100 + int(50 * math.sin(pygame.time.get_ticks() / 150))
            if self.powerup_timers["shield"] < 180 and (self.powerup_timers["shield"] // 10) % 2 == 0:
                alpha = 20 # Flicker when about to expire
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.radius * 1.8), (*shield_color, alpha))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Press space to fire your weapon. Survive for 60 seconds!"
    )
    game_description = (
        "Fast-paced arcade space shooter. Survive waves of asteroids, grab power-ups, and blast your way to a high score."
    )
    auto_advance = True
    
    # Game Constants
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 60 * FPS # 60 seconds
    COLOR_BG = (10, 10, 25)

    # Reward Constants
    REWARD_SURVIVAL_FRAME = 0.0016 # ~0.1 per second
    REWARD_ASTEROID_DESTROYED = 1.0
    REWARD_POWERUP_COLLECTED = 5.0
    REWARD_WIN = 100.0
    REWARD_PENALTY_MOVE_TOWARDS = -0.2
    REWARD_BONUS_MOVE_AWAY = 0.5
    REWARD_PENALTY_DAMAGE = -10.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_powerup = pygame.font.Font(None, 20)
        
        self.np_random = None
        self.player = None
        self.asteroids = []
        self.projectiles = []
        self.powerups = []
        self.particles = []
        self.stars = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.asteroid_spawn_timer = 0
        self.powerup_spawn_timer = 0
        self.base_asteroid_spawn_rate = 1.5 # seconds
        self.asteroid_spawn_rate = 0
        self.max_asteroid_speed = 0
        self.last_closest_dist = float('inf')
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player = Player((100, self.HEIGHT // 2), self.WIDTH, self.HEIGHT)
        self.asteroids = []
        self.projectiles = []
        self.powerups = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.asteroid_spawn_rate = self.base_asteroid_spawn_rate * self.FPS
        self.asteroid_spawn_timer = self.asteroid_spawn_rate
        self.max_asteroid_speed = 1.5
        self.powerup_spawn_timer = self.np_random.integers(10, 20) * self.FPS

        self.last_closest_dist = float('inf')

        # Starfield
        self.stars = []
        for _ in range(150):
            self.stars.append({
                "pos": pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                "speed": self.np_random.uniform(0.1, 0.5),
                "radius": self.np_random.uniform(0.5, 1.5)
            })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        step_reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Handle player actions
        self.player.update(movement, self.particles)
        if space_held:
            if self.player.shoot(self.projectiles):
                pass # sfx: laser_shoot.wav
        
        # Update game objects
        for p in self.projectiles: p.update()
        for a in self.asteroids: a.update()
        for p in self.powerups: p.update()
        for p in self.particles: p.update()
        
        # Spawning
        self._spawn_entities()

        # Collisions and reward events
        step_reward += self._handle_collisions()

        # Cleanup
        self._cleanup_objects()
        
        # Difficulty progression
        self._update_difficulty()
        
        # Calculate final reward for the step
        reward = self._calculate_reward(step_reward)
        
        self.steps += 1
        terminated = self._check_termination()
        if terminated and self.steps >= self.MAX_STEPS and self.player.health > 0:
            reward += self.REWARD_WIN # Win bonus
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_entities(self):
        # Asteroids
        self.asteroid_spawn_timer -= 1
        if self.asteroid_spawn_timer <= 0:
            side = self.np_random.choice(['top', 'bottom', 'right'])
            if side == 'right':
                pos = (self.WIDTH + 50, self.np_random.uniform(0, self.HEIGHT))
                vel = (-self.np_random.uniform(0.5, self.max_asteroid_speed), self.np_random.uniform(-0.5, 0.5))
            elif side == 'top':
                pos = (self.np_random.uniform(0, self.WIDTH), -50)
                vel = (self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(0.5, self.max_asteroid_speed))
            else: # bottom
                pos = (self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 50)
                vel = (self.np_random.uniform(-0.5, 0.5), -self.np_random.uniform(0.5, self.max_asteroid_speed))
            
            size = self.np_random.uniform(1.5, 4.0)
            self.asteroids.append(Asteroid(pos, vel, size, self.np_random))
            self.asteroid_spawn_timer = self.asteroid_spawn_rate

        # Power-ups
        self.powerup_spawn_timer -= 1
        if self.powerup_spawn_timer <= 0 and len(self.powerups) < 2:
            pos = (self.np_random.uniform(100, self.WIDTH - 100), self.np_random.uniform(100, self.HEIGHT - 100))
            p_type = self.np_random.choice(["rapid_fire", "shield", "score_multiplier"])
            self.powerups.append(PowerUp(pos, p_type, self.font_powerup))
            self.powerup_spawn_timer = self.np_random.integers(15, 25) * self.FPS
    
    def _handle_collisions(self):
        reward = 0
        
        # Projectiles vs Asteroids
        for proj in self.projectiles[:]:
            for ast in self.asteroids[:]:
                if proj.pos.distance_to(ast.pos) < ast.radius:
                    self.projectiles.remove(proj)
                    self.asteroids.remove(ast)
                    
                    score_multiplier = 2 if self.player.powerup_timers["score_multiplier"] > 0 else 1
                    self.score += 1 * score_multiplier
                    reward += self.REWARD_ASTEROID_DESTROYED
                    
                    self._create_explosion(ast.pos, ast.radius)
                    # sfx: explosion.wav
                    
                    # Chance to spawn power-up
                    if self.np_random.random() < 0.1 and len(self.powerups) < 2:
                        p_type = self.np_random.choice(["rapid_fire", "shield", "score_multiplier"])
                        self.powerups.append(PowerUp(ast.pos, p_type, self.font_powerup))

                    break
        
        # Player vs Asteroids
        for ast in self.asteroids[:]:
            if self.player.pos.distance_to(ast.pos) < self.player.radius + ast.radius:
                if self.player.take_damage():
                    reward += self.REWARD_PENALTY_DAMAGE
                    self._create_explosion(self.player.pos, self.player.radius * 2)
                    # sfx: player_hit.wav
                self.asteroids.remove(ast)
                self._create_explosion(ast.pos, ast.radius)
                break
                
        # Player vs Power-ups
        for power in self.powerups[:]:
            if self.player.pos.distance_to(power.pos) < self.player.radius + power.radius:
                self.player.powerup_timers[power.type] = 10 * self.FPS # 10 seconds duration
                if power.type == "shield": self.player.health = min(self.player.max_health, self.player.health + 1)
                self.powerups.remove(power)
                reward += self.REWARD_POWERUP_COLLECTED
                # sfx: powerup.wav
                break
                
        return reward
        
    def _cleanup_objects(self):
        self.projectiles = [p for p in self.projectiles if 0 < p.pos.x < self.WIDTH and 0 < p.pos.y < self.HEIGHT]
        self.asteroids = [a for a in self.asteroids if -100 < a.pos.x < self.WIDTH + 100 and -100 < a.pos.y < self.HEIGHT + 100]
        self.particles = [p for p in self.particles if p.lifespan > 0]

    def _update_difficulty(self):
        # Increase spawn rate by 0.001 per second (0.001/60 per frame)
        self.asteroid_spawn_rate = max(20, self.asteroid_spawn_rate - (0.001 / self.FPS) * self.FPS)
        # Increase speed every 10 seconds
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.max_asteroid_speed += 0.05
    
    def _calculate_reward(self, event_reward):
        reward = self.REWARD_SURVIVAL_FRAME + event_reward
        
        min_dist = float('inf')
        if self.asteroids:
            for ast in self.asteroids:
                min_dist = min(min_dist, self.player.pos.distance_to(ast.pos) - ast.radius)
            
            if self.last_closest_dist != float('inf'):
                if min_dist < self.last_closest_dist:
                    reward += self.REWARD_PENALTY_MOVE_TOWARDS
                elif min_dist > self.last_closest_dist:
                    reward += self.REWARD_BONUS_MOVE_AWAY
            self.last_closest_dist = min_dist
        else:
            self.last_closest_dist = float('inf')
            
        return reward
        
    def _check_termination(self):
        if self.player.health <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
        
    def _get_observation(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        
        # Stars
        for star in self.stars:
            star["pos"].x -= star["speed"]
            if star["pos"].x < 0:
                star["pos"].x = self.WIDTH
            pygame.draw.circle(self.screen, (200, 200, 255), star["pos"], star["radius"])

        # Game elements
        for ast in self.asteroids: ast.draw(self.screen)
        for power in self.powerups: power.draw(self.screen)
        for proj in self.projectiles: proj.draw(self.screen)
        if self.player.health > 0: self.player.draw(self.screen)
        for part in self.particles: part.draw(self.screen)
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
        
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, (255, 255, 255))
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Health
        for i in range(self.player.max_health):
            color = self.player.color if i < self.player.health else (50, 50, 80)
            pos = (20 + i * 25, 45)
            points = [p.rotate(90) + pos for p in self.player.shape_points]
            points = [(p.x*0.5, p.y*0.5) for p in points]
            pygame.draw.polygon(self.screen, color, points)

        # Game Over
        if self.game_over:
            font_large = pygame.font.Font(None, 72)
            msg = "MISSION COMPLETE" if self.player.health > 0 else "GAME OVER"
            color = (0, 255, 0) if self.player.health > 0 else (255, 0, 0)
            text_surf = font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _create_explosion(self, pos, radius):
        num_particles = int(radius * 1.5)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            p_radius = self.np_random.uniform(2, radius / 4)
            lifespan = self.np_random.integers(20, 50)
            color = self.np_random.choice([(255,100,0), (255,200,0), (255,255,255)], p=[0.5, 0.4, 0.1])
            self.particles.append(Particle(pos, vel, p_radius, color, lifespan))

    def close(self):
        pygame.quit()

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Requires pygame to be installed with display support
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Astro Survival")
    clock = pygame.time.Clock()

    print(GameEnv.user_guide)

    while running:
        # Construct action from keyboard input
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    env.close()