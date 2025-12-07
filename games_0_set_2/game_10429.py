import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:22:51.004779
# Source Brief: brief_00429.md
# Brief Index: 429
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player pilots a spaceship through an asteroid field.
    The goal is to collect power-ups and survive as long as possible by using thrusters
    to navigate and counteract the gravitational pull of asteroids.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a spaceship through a dangerous asteroid field, using thrusters to counteract "
        "gravity and collect power-ups for a speed boost."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to apply thrust. Press space to use your boost "
        "when a power-up is active."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_STAR = (200, 200, 220)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (100, 150, 255)
    COLOR_ASTEROID = (120, 130, 140)
    COLOR_POWERUP = (255, 215, 0) # Gold
    COLOR_POWERUP_GLOW = (255, 165, 0)
    COLOR_THRUSTER = (0, 191, 255) # Deep Sky Blue
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_UI_BAR_BG = (50, 50, 70)
    COLOR_UI_BAR_FG = (0, 191, 255)

    # Physics
    PLAYER_THRUST = 0.15
    PLAYER_BOOST_MULTIPLIER = 2.5
    PLAYER_DRAG = 0.985
    PLAYER_ROTATION_SPEED = 5
    GRAVITY_CONSTANT = 200.0
    MAX_VELOCITY = 10.0
    
    # Game Parameters
    FPS = 30
    MAX_EPISODE_STEPS = 5000
    INITIAL_ASTEROIDS = 5
    MAX_ASTEROIDS = 40
    POWERUP_DURATION = 300 # steps (10 seconds at 30 FPS)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # --- Game State Variables ---
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.asteroids = []
        self.powerups = []
        self.particles = []
        self.stars = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.thrust_powerup_timer = 0
        self.asteroid_spawn_counter = 0.0
        self.max_asteroid_speed = 1.0
        
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Player ---
        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = -90 # Pointing up
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.thrust_powerup_timer = 0
        self.max_asteroid_speed = 1.0
        self.asteroid_spawn_rate = 0.01 # Initial spawn rate per step

        # --- Reset Game Objects ---
        self.particles.clear()
        
        # Create a static starfield
        self.stars = [
            (
                self.np_random.integers(0, self.SCREEN_WIDTH), 
                self.np_random.integers(0, self.SCREEN_HEIGHT), 
                self.np_random.choice([1, 1, 1, 2])
            ) for _ in range(150)
        ]

        # Create initial asteroids
        self.asteroids = [self._create_asteroid(is_initial=True) for _ in range(self.INITIAL_ASTEROIDS)]

        # Create initial power-up
        self.powerups = [self._create_powerup()]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0.1 # Survival reward
        
        # --- Handle Actions ---
        movement, space_held, _ = action
        is_boosting = space_held == 1 and self.thrust_powerup_timer > 0

        thrust_vec = pygame.math.Vector2(0, 0)
        if movement != 0:
            # Determine direction from action
            if movement == 1: thrust_vec.y = -1 # Up
            elif movement == 2: thrust_vec.y = 1 # Down
            elif movement == 3: thrust_vec.x = -1 # Left
            elif movement == 4: thrust_vec.x = 1 # Right
            
            thrust_magnitude = self.PLAYER_THRUST * (self.PLAYER_BOOST_MULTIPLIER if is_boosting else 1)
            self.player_vel += thrust_vec * thrust_magnitude
            
            # Create thruster particles
            self._create_particles(thrust_vec, is_boosting)
            
            # Smoothly rotate player towards movement direction
            target_angle = thrust_vec.angle_to(pygame.math.Vector2(1, 0))
            angle_diff = (target_angle - self.player_angle + 180) % 360 - 180
            self.player_angle += np.clip(angle_diff, -self.PLAYER_ROTATION_SPEED, self.PLAYER_ROTATION_SPEED)

        # --- Update Game Logic ---
        self._update_player_movement()
        self._update_asteroids()
        self._update_particles()
        self._update_progression()
        
        # --- Handle Collisions & Interactions ---
        # Player-Asteroid collision
        for asteroid in self.asteroids:
            if self.player_pos.distance_to(asteroid['pos']) < asteroid['radius'] + 10:
                self.game_over = True
                reward = -100 # Terminal penalty
                break
        
        # Player-Powerup collection
        collected_powerups = []
        for i, powerup in enumerate(self.powerups):
            if self.player_pos.distance_to(powerup['pos']) < powerup['radius'] + 10:
                self.score += 1
                reward += 10
                self.thrust_powerup_timer = self.POWERUP_DURATION
                collected_powerups.append(i)
        
        if collected_powerups:
            self.powerups = [p for i, p in enumerate(self.powerups) if i not in collected_powerups]
            self.powerups.append(self._create_powerup())

        # --- Check Termination ---
        terminated = self.game_over
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _update_player_movement(self):
        # Apply gravity from asteroids
        gravity_force = pygame.math.Vector2(0, 0)
        for asteroid in self.asteroids:
            dist_vec = asteroid['pos'] - self.player_pos
            distance = dist_vec.length()
            if distance > 1:
                force_magnitude = self.GRAVITY_CONSTANT / (distance * distance)
                gravity_force += dist_vec.normalize() * force_magnitude
        
        self.player_vel += gravity_force
        
        # Apply drag
        self.player_vel *= self.PLAYER_DRAG
        
        # Clamp velocity
        if self.player_vel.length() > self.MAX_VELOCITY:
            self.player_vel.scale_to_length(self.MAX_VELOCITY)
            
        # Update position
        self.player_pos += self.player_vel
        
        # World wrapping
        self.player_pos.x %= self.SCREEN_WIDTH
        self.player_pos.y %= self.SCREEN_HEIGHT

        # Update powerup timer
        if self.thrust_powerup_timer > 0:
            self.thrust_powerup_timer -= 1
            
    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] = (asteroid['angle'] + asteroid['rot_speed']) % 360
            
            # World wrapping
            asteroid['pos'].x %= self.SCREEN_WIDTH
            asteroid['pos'].y %= self.SCREEN_HEIGHT
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['initial_radius'] * (p['life'] / p['initial_life']))

    def _update_progression(self):
        # Increase difficulty over time
        if self.steps > 0 and self.steps % 1000 == 0:
            self.max_asteroid_speed = min(3.0, self.max_asteroid_speed + 0.1)
        
        # Increase spawn rate (0.001 per second -> 0.001/30 per step)
        self.asteroid_spawn_rate += (0.001 / self.FPS)

        # Spawn new asteroids
        if len(self.asteroids) < self.MAX_ASTEROIDS and self.np_random.random() < self.asteroid_spawn_rate:
            self.asteroids.append(self._create_asteroid())

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_asteroids()
        self._render_powerups()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Helper Functions for Creation ---
    
    def _create_asteroid(self, is_initial=False):
        # Avoid spawning on top of the player initially
        if is_initial:
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT))
            while pos.distance_to(self.player_pos) < 150:
                pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT))
        else: # Spawn off-screen
            edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top': pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -30)
            elif edge == 'bottom': pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 30)
            elif edge == 'left': pos = pygame.math.Vector2(-30, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            else: pos = pygame.math.Vector2(self.SCREEN_WIDTH + 30, self.np_random.uniform(0, self.SCREEN_HEIGHT))

        speed = self.np_random.uniform(0.5, self.max_asteroid_speed)
        angle = self.np_random.uniform(0, 360)
        vel = pygame.math.Vector2(speed, 0).rotate(angle)
        
        radius = self.np_random.integers(15, 36)
        num_points = self.np_random.integers(7, 13)
        shape_points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dist = self.np_random.uniform(radius * 0.8, radius * 1.2)
            shape_points.append((dist * math.cos(angle), dist * math.sin(angle)))

        return {
            'pos': pos, 'vel': vel, 'radius': radius,
            'angle': self.np_random.uniform(0, 360), 'rot_speed': self.np_random.uniform(-1, 1),
            'shape_points': shape_points
        }

    def _create_powerup(self):
        pos = pygame.math.Vector2(self.np_random.uniform(50, self.SCREEN_WIDTH - 50), self.np_random.uniform(50, self.SCREEN_HEIGHT - 50))
        while self.player_pos.distance_to(pos) < 100:
            pos = pygame.math.Vector2(self.np_random.uniform(50, self.SCREEN_WIDTH - 50), self.np_random.uniform(50, self.SCREEN_HEIGHT - 50))
        return {'pos': pos, 'radius': 12}

    def _create_particles(self, thrust_vec, is_boosting):
        # Create particles shooting out opposite to thrust
        particle_vel_base = -thrust_vec.normalize() * 3
        num_particles = 8 if is_boosting else 4
        for _ in range(num_particles):
            # Offset from player center to back of ship
            offset = pygame.math.Vector2(-12, 0).rotate(-self.player_angle)
            pos = self.player_pos + offset
            
            vel_spread = self.np_random.uniform(-30, 30)
            life = self.np_random.integers(15, 31) if is_boosting else self.np_random.integers(10, 21)
            self.particles.append({
                'pos': pos,
                'vel': particle_vel_base.rotate(vel_spread) * self.np_random.uniform(0.8, 1.2),
                'life': life,
                'initial_life': life,
                'initial_radius': self.np_random.uniform(2, 4) if is_boosting else self.np_random.uniform(1, 3)
            })

    # --- Helper Functions for Rendering ---
    
    def _render_background(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size / 2)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = []
            for p in asteroid['shape_points']:
                rotated_p = pygame.math.Vector2(p).rotate(asteroid['angle'])
                points.append(asteroid['pos'] + rotated_p)
            
            if len(points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in points], self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in points], self.COLOR_ASTEROID)

    def _render_powerups(self):
        for powerup in self.powerups:
            # Pulsating effect
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            radius = powerup['radius'] + pulse * 4
            glow_alpha = int(100 + pulse * 100)
            
            # Draw glow
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_POWERUP_GLOW + (glow_alpha,), (radius, radius), radius)
            self.screen.blit(s, (powerup['pos'].x - radius, powerup['pos'].y - radius))

            # Draw core
            pygame.gfxdraw.filled_circle(self.screen, int(powerup['pos'].x), int(powerup['pos'].y), powerup['radius'], self.COLOR_POWERUP)
            pygame.gfxdraw.aacircle(self.screen, int(powerup['pos'].x), int(powerup['pos'].y), powerup['radius'], self.COLOR_POWERUP)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['initial_life']))
            color = self.COLOR_THRUSTER
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color + (alpha,), (p['radius'], p['radius']), p['radius'])
            self.screen.blit(s, (p['pos'].x - p['radius'], p['pos'].y - p['radius']))

    def _render_player(self):
        # Draw glow
        if self.thrust_powerup_timer > 0:
            glow_radius = int(20 + 5 * (self.thrust_powerup_timer / self.POWERUP_DURATION))
            glow_alpha = int(50 + 100 * (self.thrust_powerup_timer / self.POWERUP_DURATION))
            s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_PLAYER_GLOW + (glow_alpha,), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (self.player_pos.x - glow_radius, self.player_pos.y - glow_radius))

        # Define ship points relative to center (0,0)
        p1 = pygame.math.Vector2(15, 0) # Nose
        p2 = pygame.math.Vector2(-10, 10) # Left wing
        p3 = pygame.math.Vector2(-10, -10) # Right wing
        
        # Rotate points
        angle_rad = math.radians(self.player_angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        p1_rot = pygame.math.Vector2(p1.x * cos_a - p1.y * sin_a, p1.x * sin_a + p1.y * cos_a)
        p2_rot = pygame.math.Vector2(p2.x * cos_a - p2.y * sin_a, p2.x * sin_a + p2.y * cos_a)
        p3_rot = pygame.math.Vector2(p3.x * cos_a - p3.y * sin_a, p3.x * sin_a + p3.y * cos_a)

        # Translate to player position
        points = [
            self.player_pos + p1_rot,
            self.player_pos + p2_rot,
            self.player_pos + p3_rot
        ]
        
        # Draw ship
        int_points = [(int(p.x), int(p.y)) for p in points]
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Thrust Power-up Bar
        bar_width = 150
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        
        power_ratio = self.thrust_powerup_timer / self.POWERUP_DURATION
        fill_width = int(bar_width * power_ratio)

        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FG, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 2)
        
        boost_text = self.font_small.render("BOOST", True, self.COLOR_UI_TEXT)
        self.screen.blit(boost_text, (bar_x + bar_width/2 - boost_text.get_width()/2, bar_y + bar_height/2 - boost_text.get_height()/2))

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It will not run when the environment is used by an agent
    
    # Un-set the headless environment variable to allow display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Gravity Pilot")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("Manual Control:")
    print("  - Arrows: Move")
    print("  - Space: Boost (when power-up is active)")
    print("  - Q: Quit")

    while not terminated:
        movement_action = 0 # No-op
        space_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        
        action = [movement_action, space_action, 0] # Shift is not used
        
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if term or trunc:
            print(f"Game Over! Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Reset after a short delay
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            if trunc:
                terminated = True # End loop if truncated
        
        clock.tick(GameEnv.FPS)
        
    env.close()