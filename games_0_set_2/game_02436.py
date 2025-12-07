
# Generated: 2025-08-27T20:22:24.033372
# Source Brief: brief_02436.md
# Brief Index: 2436

        
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
    """
    An arcade-style, top-down spaceship shooter set in an asteroid field.
    The player must survive for 60 seconds while destroying asteroids for score.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold space to fire your laser cannon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot your ship through a dangerous asteroid field. Dodge and destroy "
        "asteroids to score points. Survive for 60 seconds to win!"
    )

    # Frames auto-advance at a rate of 60fps.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_THRUST = (255, 200, 50)
    COLOR_ASTEROID = (200, 200, 200)
    COLOR_PROJECTILE = (255, 80, 80)
    COLOR_EXPLOSION = [(255, 200, 50), (255, 120, 0), (200, 50, 0)]
    COLOR_TEXT = (240, 240, 240)
    COLOR_STAR = (100, 100, 120)

    # Player settings
    PLAYER_SIZE = 10
    PLAYER_ACCELERATION = 0.4
    PLAYER_FRICTION = 0.96
    PLAYER_TURN_SPEED = 0.1
    PLAYER_MAX_SPEED = 5

    # Projectile settings
    PROJECTILE_SPEED = 8
    PROJECTILE_COOLDOWN = 8  # frames

    # Asteroid settings
    ASTEROID_BASE_SPEED = 0.8
    ASTEROID_SPAWN_INTERVAL_START = 120 # frames (2 seconds)
    ASTEROID_SPAWN_INTERVAL_END = 30 # frames (0.5 seconds)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('monospace', 24, bold=True)
        self.font_small = pygame.font.SysFont('monospace', 16)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.asteroids = None
        self.projectiles = None
        self.particles = None
        self.stars = None
        self.shoot_cooldown = None
        self.asteroid_spawn_timer = None
        self.asteroid_spawn_interval = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.np_random = None

        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = -math.pi / 2  # Pointing up

        self.asteroids = []
        self.projectiles = []
        self.particles = []
        
        if self.stars is None:
            self.stars = [
                (self.np_random.integers(0, self.SCREEN_WIDTH), 
                 self.np_random.integers(0, self.SCREEN_HEIGHT), 
                 self.np_random.integers(1, 3))
                for _ in range(150)
            ]

        self.shoot_cooldown = 0
        self.asteroid_spawn_interval = self.ASTEROID_SPAWN_INTERVAL_START
        self.asteroid_spawn_timer = self.asteroid_spawn_interval

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.01  # Survival reward

        self._handle_input(movement, space_held)
        self._update_player()
        self._update_projectiles()
        self._update_asteroids()
        self._update_particles()
        
        collision_reward, termination_reason = self._handle_collisions()
        reward += collision_reward

        self._spawn_asteroids()
        self._update_difficulty()

        self.steps += 1
        
        terminated = self.game_over
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
            self.win = True
            reward = 100.0
        elif termination_reason == "collision":
            reward = -100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        target_vel = pygame.math.Vector2(0, 0)
        if movement == 1: # Up
            target_vel.y = -1
        elif movement == 2: # Down
            target_vel.y = 1
        elif movement == 3: # Left
            target_vel.x = -1
        elif movement == 4: # Right
            target_vel.x = 1
        
        if target_vel.length() > 0:
            target_vel.normalize_ip()
            # Smoothly turn ship towards movement direction
            target_angle = target_vel.angle_to(pygame.math.Vector2(1, 0))
            self.player_angle = self._lerp_angle(self.player_angle, math.radians(-target_angle), self.PLAYER_TURN_SPEED)

            # Accelerate
            self.player_vel += target_vel * self.PLAYER_ACCELERATION
            if self.player_vel.length() > self.PLAYER_MAX_SPEED:
                self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)

        # Firing
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        
        if space_held and self.shoot_cooldown == 0:
            self._fire_projectile()
            self.shoot_cooldown = self.PROJECTILE_COOLDOWN

    def _fire_projectile(self):
        # SFX: Laser fire
        direction = pygame.math.Vector2(math.cos(self.player_angle), math.sin(self.player_angle))
        start_pos = self.player_pos + direction * self.PLAYER_SIZE
        velocity = direction * self.PROJECTILE_SPEED
        self.projectiles.append({'pos': start_pos, 'vel': velocity, 'life': 60})

    def _update_player(self):
        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION
        if self.player_vel.length() < 0.1:
            self.player_vel.update(0, 0)
        
        self.player_pos += self.player_vel

        # Screen wrapping
        self.player_pos.x %= self.SCREEN_WIDTH
        self.player_pos.y %= self.SCREEN_HEIGHT

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0 or not (0 < p['pos'].x < self.SCREEN_WIDTH and 0 < p['pos'].y < self.SCREEN_HEIGHT):
                self.projectiles.remove(p)

    def _update_asteroids(self):
        for a in self.asteroids:
            a['pos'] += a['vel']
            a['angle'] += a['rot_speed']
            # Screen wrapping
            a['pos'].x %= self.SCREEN_WIDTH
            a['pos'].y %= self.SCREEN_HEIGHT

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_asteroids(self):
        self.asteroid_spawn_timer -= 1
        if self.asteroid_spawn_timer <= 0:
            self.asteroid_spawn_timer = self.asteroid_spawn_interval
            
            # Spawn off-screen
            edge = self.np_random.integers(0, 4)
            if edge == 0: # Top
                pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -30)
            elif edge == 1: # Bottom
                pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 30)
            elif edge == 2: # Left
                pos = pygame.math.Vector2(-30, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            else: # Right
                pos = pygame.math.Vector2(self.SCREEN_WIDTH + 30, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            
            target = pygame.math.Vector2(
                self.np_random.uniform(self.SCREEN_WIDTH * 0.2, self.SCREEN_WIDTH * 0.8),
                self.np_random.uniform(self.SCREEN_HEIGHT * 0.2, self.SCREEN_HEIGHT * 0.8)
            )
            vel = (target - pos).normalize() * self.ASTEROID_BASE_SPEED * self.np_random.uniform(0.8, 1.5)
            
            size = self.np_random.integers(1, 4) # 1=small, 2=medium, 3=large
            self._create_asteroid(pos, vel, size)

    def _create_asteroid(self, pos, vel, size):
        hp = size
        radius = size * 8
        num_vertices = self.np_random.integers(7, 13)
        shape = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = radius * self.np_random.uniform(0.75, 1.25)
            shape.append(pygame.math.Vector2(math.cos(angle), math.sin(angle)) * dist)
        
        self.asteroids.append({
            'pos': pos, 'vel': vel, 'hp': hp, 'size': size,
            'radius': radius, 'shape': shape, 'angle': 0,
            'rot_speed': self.np_random.uniform(-0.03, 0.03)
        })

    def _update_difficulty(self):
        # Decrease spawn interval over time
        progress = self.steps / self.MAX_STEPS
        self.asteroid_spawn_interval = self._lerp(self.ASTEROID_SPAWN_INTERVAL_START, self.ASTEROID_SPAWN_INTERVAL_END, progress)

    def _handle_collisions(self):
        reward = 0
        termination_reason = None

        # Projectile-Asteroid
        for p in self.projectiles[:]:
            for a in self.asteroids[:]:
                if p['pos'].distance_to(a['pos']) < a['radius']:
                    # SFX: Asteroid hit
                    self.projectiles.remove(p)
                    a['hp'] -= 1
                    self._create_explosion(a['pos'], 3, a['size'])
                    if a['hp'] <= 0:
                        # SFX: Asteroid explosion
                        reward += 1.0
                        self.score += 10 * a['size']
                        self._create_explosion(a['pos'], 15, a['size'] * 2)
                        self.asteroids.remove(a)
                        # Split asteroid
                        if a['size'] > 1:
                            for _ in range(2):
                                new_vel = a['vel'].rotate(self.np_random.uniform(-45, 45)) * 1.2
                                self._create_asteroid(a['pos'].copy(), new_vel, a['size'] - 1)
                    break 

        # Player-Asteroid
        if not self.game_over:
            for a in self.asteroids:
                if self.player_pos.distance_to(a['pos']) < a['radius'] + self.PLAYER_SIZE * 0.8:
                    # SFX: Player explosion
                    self.game_over = True
                    termination_reason = "collision"
                    self._create_explosion(self.player_pos, 40, 5)
                    break
        
        return reward, termination_reason

    def _create_explosion(self, pos, num_particles, scale):
        for _ in range(num_particles):
            vel = pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            vel.scale_to_length(self.np_random.uniform(0.5, 3) * scale * 0.3)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 40),
                'color': random.choice(self.COLOR_EXPLOSION)
            })

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render all elements
        self._render_stars()
        self._render_particles()
        self._render_projectiles()
        self._render_asteroids()
        if not self.game_over:
            self._render_player()
        
        # Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))

    def _render_player(self):
        # Calculate vertices for the triangular ship
        p1 = self.player_pos + pygame.math.Vector2(math.cos(self.player_angle), math.sin(self.player_angle)) * self.PLAYER_SIZE
        p2 = self.player_pos + pygame.math.Vector2(math.cos(self.player_angle + 2.5), math.sin(self.player_angle + 2.5)) * self.PLAYER_SIZE * 0.8
        p3 = self.player_pos + pygame.math.Vector2(math.cos(self.player_angle - 2.5), math.sin(self.player_angle - 2.5)) * self.PLAYER_SIZE * 0.8
        points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
        
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_PLAYER)

        # Render thrust
        if self.player_vel.length() > 1:
            thrust_len = self.player_vel.length() * 1.5
            p_thrust = self.player_pos - self.player_vel.normalize() * (self.PLAYER_SIZE * 0.8 + thrust_len/2)
            pygame.draw.line(self.screen, self.COLOR_PLAYER_THRUST, 
                             (int(self.player_pos.x), int(self.player_pos.y)),
                             (int(p_thrust.x), int(p_thrust.y)),
                             max(1, int(thrust_len/3)))


    def _render_asteroids(self):
        for a in self.asteroids:
            rotated_shape = []
            for point in a['shape']:
                rotated_point = point.rotate(math.degrees(a['angle']))
                rotated_shape.append((a['pos'].x + rotated_point.x, a['pos'].y + rotated_point.y))
            
            if len(rotated_shape) > 2:
                int_points = [(int(p[0]), int(p[1])) for p in rotated_shape]
                pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_ASTEROID)

    def _render_projectiles(self):
        for p in self.projectiles:
            start_pos = (int(p['pos'].x), int(p['pos'].y))
            end_pos = (int(p['pos'].x - p['vel'].x * 1.5), int(p['pos'].y - p['vel'].y * 1.5))
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = p['color']
            radius = int(p['life'] * 0.2)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), radius, (*color, alpha))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over or self.win:
            msg = "MISSION FAILED"
            if self.win:
                msg = "MISSION COMPLETE"
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        }
    
    def _lerp(self, a, b, t):
        return a + (b - a) * t

    def _lerp_angle(self, a, b, t):
        diff = (b - a + math.pi) % (2 * math.pi) - math.pi
        return a + diff * t

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Game loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            # --- Map keyboard inputs to the MultiDiscrete action space ---
            keys = pygame.key.get_pressed()
            
            movement = 0 # No-op
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
                
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step the environment ---
            obs, reward, terminated, truncated, info = env.step(action)

        # --- Render the observation to a display window ---
        # Note: The environment's observation is (H, W, C). Pygame screen expects (W, H).
        # We need to get the surface from the env *before* it's converted to a numpy array.
        display_surface = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Asteroid Shooter")
        
        # The env._get_observation() already rendered to env.screen, so we just blit it.
        display_surface.blit(env.screen, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.FPS)

    env.close()