import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:56:45.623965
# Source Brief: brief_02060.md
# Brief Index: 2060
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player navigates a ship through an asteroid field.
    The goal is to build momentum and reach an exit gate while avoiding collisions.
    
    Visuals: Retro vector graphics with glow effects and particles.
    Gameplay: Skill-based arcade action with momentum physics.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a spaceship through a dangerous asteroid field. "
        "Build momentum to reach the exit gate while avoiding collisions."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to apply thrust and navigate the asteroid field."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60 # For physics simulation, not rendering speed
        self.MAX_STEPS = 5000

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_SHIP = (255, 255, 255)
        self.COLOR_ASTEROID = (255, 80, 80)
        self.COLOR_ASTEROID_OUTLINE = (255, 120, 120)
        self.COLOR_EXIT = (100, 255, 100)
        self.COLOR_EXIT_GLOW = (100, 255, 100, 50)
        self.COLOR_HEALTH_BAR = (40, 220, 90)
        self.COLOR_MOMENTUM_BAR = (40, 150, 255)
        self.COLOR_UI_BG = (255, 255, 255, 20)
        self.COLOR_UI_TEXT = (220, 220, 240)

        # Ship properties
        self.SHIP_ACCELERATION = 0.1
        self.SHIP_MAX_SPEED = 5.0
        self.SHIP_DRAG = 0.995 # slight velocity decay
        self.SHIP_RADIUS = 12
        self.SHIP_HEALTH_ON_COLLISION = 25

        # Asteroid properties
        self.INITIAL_ASTEROID_SPAWN_INTERVAL = 2.0 # seconds
        self.ASTEROID_SPAWN_RATE_INCREASE = 0.001 # per second
        self.ASTEROID_MIN_SPEED = 0.5
        self.ASTEROID_MAX_SPEED = 2.0
        self.ASTEROID_MIN_RADIUS = 15
        self.ASTEROID_MAX_RADIUS = 40

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(None, 22)
            self.font_ui_label = pygame.font.Font(None, 16)
        except IOError:
            self.font_ui = pygame.font.SysFont("monospace", 20)
            self.font_ui_label = pygame.font.SysFont("monospace", 14)


        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.ship_pos = pygame.math.Vector2(0, 0)
        self.ship_vel = pygame.math.Vector2(0, 0)
        self.ship_health = 100.0
        self.momentum = 0.0
        self.last_momentum_tier = 0

        self.asteroids = []
        self.particles = []
        self.stars = []
        
        self.asteroid_spawn_timer = 0.0
        self.current_asteroid_spawn_interval = self.INITIAL_ASTEROID_SPAWN_INTERVAL
        
        self.exit_rect = pygame.Rect(self.WIDTH - 20, self.HEIGHT/2 - 50, 20, 100)

        # Initialize state by calling reset
        # self.reset() # reset() is called by the environment runner
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.ship_pos = pygame.math.Vector2(self.WIDTH / 4, self.HEIGHT / 2)
        self.ship_vel = pygame.math.Vector2(0, 0)
        self.ship_health = 100.0
        self.momentum = 0.0
        self.last_momentum_tier = 0
        
        self.asteroids = []
        self.particles = []
        self.stars = self._create_stars(200)

        self.asteroid_spawn_timer = self.INITIAL_ASTEROID_SPAWN_INTERVAL
        self.current_asteroid_spawn_interval = self.INITIAL_ASTEROID_SPAWN_INTERVAL

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        self._handle_input(action)
        self._update_game_state()
        
        reward += self._calculate_reward()
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        self.game_over = terminated or truncated
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement = action[0]
        
        if movement == 1: # Up
            self.ship_vel.y -= self.SHIP_ACCELERATION
        elif movement == 2: # Down
            self.ship_vel.y += self.SHIP_ACCELERATION
        elif movement == 3: # Left
            self.ship_vel.x -= self.SHIP_ACCELERATION
        elif movement == 4: # Right
            self.ship_vel.x += self.SHIP_ACCELERATION
            
        # Add engine particles if moving
        if movement != 0:
            self._create_engine_particles()

    def _update_game_state(self):
        # Update ship
        if self.ship_vel.magnitude() > self.SHIP_MAX_SPEED:
            self.ship_vel.scale_to_length(self.SHIP_MAX_SPEED)
        
        self.ship_pos += self.ship_vel
        self.ship_vel *= self.SHIP_DRAG

        # Screen wrap (toroidal world)
        self.ship_pos.x %= self.WIDTH
        self.ship_pos.y %= self.HEIGHT

        # Update momentum
        self.momentum = (self.ship_vel.magnitude() / self.SHIP_MAX_SPEED) * 100
        self.momentum = max(0, min(100, self.momentum))

        # Update asteroids
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] += asteroid['rot_speed']
            asteroid['angle'] %= 360
            
            # Screen wrap asteroids
            if asteroid['pos'].x < -asteroid['radius']: asteroid['pos'].x = self.WIDTH + asteroid['radius']
            if asteroid['pos'].x > self.WIDTH + asteroid['radius']: asteroid['pos'].x = -asteroid['radius']
            if asteroid['pos'].y < -asteroid['radius']: asteroid['pos'].y = self.HEIGHT + asteroid['radius']
            if asteroid['pos'].y > self.HEIGHT + asteroid['radius']: asteroid['pos'].y = -asteroid['radius']

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)

        # Asteroid spawning
        self.asteroid_spawn_timer -= 1 / self.FPS
        if self.asteroid_spawn_timer <= 0:
            self._spawn_asteroid()
            self.current_asteroid_spawn_interval = max(0.5, self.current_asteroid_spawn_interval - (self.ASTEROID_SPAWN_RATE_INCREASE / self.FPS))
            self.asteroid_spawn_timer = self.current_asteroid_spawn_interval
            
        # Collision detection
        self._handle_collisions()

    def _handle_collisions(self):
        asteroids_to_remove = []
        for i, asteroid in enumerate(self.asteroids):
            dist = self.ship_pos.distance_to(asteroid['pos'])
            if dist < self.SHIP_RADIUS + asteroid['radius']:
                asteroids_to_remove.append(i)
                self.ship_health -= self.SHIP_HEALTH_ON_COLLISION
                self.momentum = 0
                self.ship_vel *= 0.2 # Dampen velocity on hit
                self._create_explosion(asteroid['pos'], asteroid['radius'])
                # sfx: explosion

        # Remove collided asteroids (in reverse to avoid index errors)
        for i in sorted(asteroids_to_remove, reverse=True):
            del self.asteroids[i]

    def _calculate_reward(self):
        reward = 0.0
        
        # Continuous reward for survival
        reward += 0.01 # Scaled down from brief for better balance with other rewards

        # Reward for gaining momentum
        current_momentum_tier = int(self.momentum / 10)
        if current_momentum_tier > self.last_momentum_tier:
            reward += 1.0 * (current_momentum_tier - self.last_momentum_tier)
        self.last_momentum_tier = current_momentum_tier
        
        return reward

    def _check_termination(self):
        # Health below zero
        if self.ship_health <= 0:
            self.ship_health = 0
            self.score -= 100 # Apply terminal reward directly to score
            return True
        
        # Reached exit
        ship_rect = pygame.Rect(self.ship_pos.x - self.SHIP_RADIUS, self.ship_pos.y - self.SHIP_RADIUS, self.SHIP_RADIUS*2, self.SHIP_RADIUS*2)
        if self.exit_rect.colliderect(ship_rect):
            if self.ship_health >= 50:
                self.score += 100 # Victory
            else:
                self.score -= 50 # Reached exit but failed health condition
            return True
            
        return False

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
            "health": self.ship_health,
            "momentum": self.momentum,
            "asteroids": len(self.asteroids)
        }

    def _render_game(self):
        # Render stars with parallax
        self._render_stars()
        
        # Render exit gate
        self._render_exit_gate()
        
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['initial_life']))))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

        # Render asteroids
        for asteroid in self.asteroids:
            self._render_asteroid(asteroid)

        # Render ship
        self._render_ship()

    def _render_ui(self):
        # --- Health Bar ---
        bar_width = 200
        bar_height = 20
        health_percent = self.ship_health / 100.0
        
        # Background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, bar_width, bar_height))
        # Fill
        fill_width = max(0, int(bar_width * health_percent))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, fill_width, bar_height))
        # Label
        health_text = self.font_ui_label.render("HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # --- Momentum Bar ---
        momentum_percent = self.momentum / 100.0
        # Background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 35, bar_width, bar_height))
        # Fill
        fill_width = max(0, int(bar_width * momentum_percent))
        pygame.draw.rect(self.screen, self.COLOR_MOMENTUM_BAR, (10, 35, fill_width, bar_height))
        # Label
        momentum_text = self.font_ui_label.render("MOMENTUM", True, self.COLOR_UI_TEXT)
        self.screen.blit(momentum_text, (15, 37))

    def _render_ship(self):
        # Ship's direction for pointing
        if self.ship_vel.magnitude() > 0.1:
            angle = self.ship_vel.angle_to(pygame.math.Vector2(1, 0))
        else:
            angle = 0
        
        # Define ship points relative to center
        p1 = pygame.math.Vector2(self.SHIP_RADIUS, 0).rotate(-angle)
        p2 = pygame.math.Vector2(-self.SHIP_RADIUS/2, self.SHIP_RADIUS * 0.8).rotate(-angle)
        p3 = pygame.math.Vector2(-self.SHIP_RADIUS/2, -self.SHIP_RADIUS * 0.8).rotate(-angle)

        # Translate points to ship's position
        points = [
            (self.ship_pos.x + p1.x, self.ship_pos.y + p1.y),
            (self.ship_pos.x + p2.x, self.ship_pos.y + p2.y),
            (self.ship_pos.x + p3.x, self.ship_pos.y + p3.y),
        ]
        
        # Draw with antialiasing
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_SHIP)

    def _render_asteroid(self, asteroid):
        # Rotate and translate points
        rotated_points = []
        for point in asteroid['points']:
            rotated_point = point.rotate(asteroid['angle'])
            rotated_points.append((int(asteroid['pos'].x + rotated_point.x), int(asteroid['pos'].y + rotated_point.y)))
        
        # Draw with antialiasing
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_ASTEROID_OUTLINE)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_ASTEROID)

    def _render_exit_gate(self):
        # Pulsating glow effect
        glow_alpha = 50 + 30 * math.sin(pygame.time.get_ticks() * 0.005)
        glow_color = (self.COLOR_EXIT[0], self.COLOR_EXIT[1], self.COLOR_EXIT[2], int(glow_alpha))
        
        glow_rect = self.exit_rect.inflate(20, 20)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, glow_color, glow_surface.get_rect(), border_radius=10)
        self.screen.blit(glow_surface, glow_rect.topleft)
        
        # Main gate rectangle
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect, 3, border_radius=5)

    def _render_stars(self):
        # Parallax effect based on ship velocity
        parallax_factor = self.ship_vel * 0.05
        for star in self.stars:
            pos_x = (star['pos'].x - parallax_factor.x * star['depth']) % self.WIDTH
            pos_y = (star['pos'].y - parallax_factor.y * star['depth']) % self.HEIGHT
            pygame.gfxdraw.pixel(self.screen, int(pos_x), int(pos_y), star['color'])

    def _create_stars(self, count):
        stars = []
        for _ in range(count):
            depth = random.uniform(0.1, 1.0)
            brightness = int(50 + 150 * depth)
            stars.append({
                'pos': pygame.math.Vector2(random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)),
                'depth': depth,
                'color': (brightness, brightness, brightness)
            })
        return stars

    def _spawn_asteroid(self):
        # Spawn off-screen
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        radius = random.uniform(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS)

        if edge == 'top':
            pos = pygame.math.Vector2(random.uniform(0, self.WIDTH), -radius)
        elif edge == 'bottom':
            pos = pygame.math.Vector2(random.uniform(0, self.WIDTH), self.HEIGHT + radius)
        elif edge == 'left':
            pos = pygame.math.Vector2(-radius, random.uniform(0, self.HEIGHT))
        else: # right
            pos = pygame.math.Vector2(self.WIDTH + radius, random.uniform(0, self.HEIGHT))
        
        # Aim towards the center of the screen with some variance
        target = pygame.math.Vector2(self.WIDTH/2 + random.uniform(-100, 100), self.HEIGHT/2 + random.uniform(-100, 100))
        speed = random.uniform(self.ASTEROID_MIN_SPEED, self.ASTEROID_MAX_SPEED)
        vel = (target - pos).normalize() * speed
        
        # Create polygon points
        num_vertices = random.randint(5, 9)
        points = []
        for i in range(num_vertices):
            angle = i * (360 / num_vertices)
            dist = random.uniform(radius * 0.7, radius)
            points.append(pygame.math.Vector2(dist, 0).rotate(angle))
            
        self.asteroids.append({
            'pos': pos,
            'vel': vel,
            'radius': radius,
            'angle': 0,
            'rot_speed': random.uniform(-2.0, 2.0),
            'points': points
        })

    def _create_engine_particles(self):
        # Create a burst of particles for the ship's engine trail
        if self.ship_vel.magnitude_squared() == 0: return
        
        direction = -self.ship_vel.normalize()
        for _ in range(2):
            vel = direction.rotate(random.uniform(-15, 15)) * random.uniform(1, 3)
            self.particles.append({
                'pos': self.ship_pos.copy(),
                'vel': vel,
                'radius': random.uniform(2, 4),
                'life': random.randint(15, 30),
                'initial_life': 30,
                'color': (200, 220, 255)
            })

    def _create_explosion(self, pos, radius):
        # sfx: small_explosion
        num_particles = int(radius)
        for _ in range(num_particles):
            angle = random.uniform(0, 360)
            speed = random.uniform(1, 4)
            vel = pygame.math.Vector2(1, 0).rotate(angle) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': random.uniform(1, 4),
                'life': random.randint(20, 40),
                'initial_life': 40,
                'color': random.choice([(255, 80, 80), (255, 150, 50), (200, 200, 200)])
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    truncated = False
    
    # Use a real pygame screen for manual play
    manual_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("GameEnv Manual Test")
    clock = pygame.time.Clock()
    
    total_reward = 0

    while running:
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0
            terminated = False
            truncated = False

        # --- Action Mapping for Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                terminated = True # Force reset
        
        # --- Rendering for Manual Play ---
        # The observation is already a rendered frame, just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        manual_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()