import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:38:15.372723
# Source Brief: brief_01111.md
# Brief Index: 1111
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for Asteroids
class Asteroid:
    def __init__(self, world_size, speed_multiplier, pattern_id=None):
        self.world_width, self.world_height = world_size
        self.speed_multiplier = speed_multiplier

        # Visual properties
        self.size = random.uniform(15, 35)
        self.color = (220, 40, 40)
        self.rotation_angle = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-1.5, 1.5)

        # State properties
        self.was_on_screen = False
        self.collided = False

        # Pattern Initialization
        if pattern_id is None:
            self.pattern_id = random.choice([0, 1])
        else:
            self.pattern_id = pattern_id

        self.time = 0
        spawn_edge = random.randint(0, 3)
        if spawn_edge == 0: # Top
            x, y = random.uniform(0, self.world_width), -self.size
        elif spawn_edge == 1: # Bottom
            x, y = random.uniform(0, self.world_width), self.world_height + self.size
        elif spawn_edge == 2: # Left
            x, y = -self.size, random.uniform(0, self.world_height)
        else: # Right
            x, y = self.world_width + self.size, random.uniform(0, self.world_height)
        
        self.pos = np.array([x, y], dtype=np.float32)

        # Pattern 0: Spiral
        if self.pattern_id == 0:
            self.spiral_center = self.pos.copy()
            self.spiral_radius = 0
            self.spiral_angle = random.uniform(0, 2 * math.pi)
            self.spiral_radial_vel = random.uniform(0.5, 1.0)
            self.spiral_angular_vel = random.uniform(0.02, 0.04)
        
        # Pattern 1: Oscillating Line
        elif self.pattern_id == 1:
            self.osc_axis = random.choice([0, 1]) # 0 for x-axis, 1 for y-axis
            self.osc_amplitude = random.uniform(50, 150)
            self.osc_frequency = random.uniform(0.02, 0.05)
            self.osc_base_velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)], dtype=np.float32)
            self.osc_base_velocity /= (np.linalg.norm(self.osc_base_velocity) + 1e-6)
            self.osc_base_pos = self.pos.copy()


    def update(self):
        self.time += 1
        self.rotation_angle = (self.rotation_angle + self.rotation_speed) % 360

        if self.pattern_id == 0: # Spiral
            self.spiral_radius += self.spiral_radial_vel * self.speed_multiplier
            self.spiral_angle += self.spiral_angular_vel
            offset_x = self.spiral_radius * math.cos(self.spiral_angle)
            offset_y = self.spiral_radius * math.sin(self.spiral_angle)
            self.pos[0] = self.spiral_center[0] + offset_x
            self.pos[1] = self.spiral_center[1] + offset_y
        
        elif self.pattern_id == 1: # Oscillating Line
            self.osc_base_pos += self.osc_base_velocity * 1.5 * self.speed_multiplier
            oscillation = self.osc_amplitude * math.sin(self.osc_frequency * self.time)
            if self.osc_axis == 0: # Oscillate on y-axis, move on x
                self.pos[0] = self.osc_base_pos[0]
                self.pos[1] = self.osc_base_pos[1] + oscillation
            else: # Oscillate on x-axis, move on y
                self.pos[0] = self.osc_base_pos[0] + oscillation
                self.pos[1] = self.osc_base_pos[1]

# Helper class for visual effect particles
class Particle:
    def __init__(self, pos, color, min_vel, max_vel, duration):
        self.pos = np.array(pos, dtype=np.float32)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(min_vel, max_vel)
        self.vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
        self.duration = duration
        self.max_duration = duration
        self.color = color
        self.size = random.uniform(2, 5)

    def update(self):
        self.pos += self.vel
        self.duration -= 1
        return self.duration > 0

    def draw(self, surface, camera_offset, screen_size):
        screen_pos = self.pos - camera_offset + np.array(screen_size) / 2
        alpha = int(255 * (self.duration / self.max_duration))
        if alpha > 0:
            pygame.gfxdraw.filled_circle(
                surface, int(screen_pos[0]), int(screen_pos[1]),
                int(self.size * (self.duration / self.max_duration)),
                (*self.color, alpha)
            )

# Main Gymnasium Environment
class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Dodge asteroids in a recursive, fractal space. Survive as long as you can by avoiding incoming threats."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move your ship and dodge the asteroids."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_WIDTH, WORLD_HEIGHT = 1280, 800 # Toroidal world
    
    COLOR_BG = (10, 10, 25)
    COLOR_SHIP_HIGH_HEALTH = (50, 255, 50)
    COLOR_SHIP_LOW_HEALTH = (255, 50, 50)
    COLOR_ASTEROID = (220, 40, 40)
    COLOR_UI_TEXT = (50, 255, 50)
    COLOR_STAR = (200, 200, 220)

    SHIP_SPEED = 4.0
    SHIP_SIZE = 12.0
    INITIAL_SCORE = 100
    MAX_SCORE = 200
    
    MAX_ASTEROIDS = 15
    WIN_CONDITION_DODGES = 100
    MAX_EPISODE_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.render_mode = render_mode

        # Initialize attributes to be defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ship_pos = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.dodged_asteroids_total = 0
        self.asteroid_speed_multiplier = 1.0
        self.step_reward = 0.0
        
        # self.reset() # No need to call reset in init
        # self.validate_implementation() # Validation is for testing, not production
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = self.INITIAL_SCORE
        self.game_over = False
        
        self.ship_pos = np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2], dtype=np.float32)
        
        self.asteroids = []
        self.particles = []
        self.dodged_asteroids_total = 0
        self.asteroid_speed_multiplier = 1.0

        self.stars = [
            (self.np_random.integers(0, self.WORLD_WIDTH), self.np_random.integers(0, self.WORLD_HEIGHT), self.np_random.uniform(0.5, 1.5))
            for _ in range(200)
        ]
        
        for _ in range(5):
            self._spawn_asteroid()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.step_reward = 0.0

        self._update_ship(action)
        self._update_asteroids()
        self._update_particles()
        self._check_collisions()

        if len(self.asteroids) < self.MAX_ASTEROIDS:
            if self.np_random.random() < 0.05:
                 self._spawn_asteroid()

        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        
        if terminated or truncated:
            if self.dodged_asteroids_total >= self.WIN_CONDITION_DODGES and not truncated:
                reward += 50.0 # Goal-oriented reward
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_ship(self, action):
        movement = action[0]
        
        vel = np.array([0.0, 0.0])
        if movement == 1: vel[1] = -1.0 # UP
        elif movement == 2: vel[1] = 1.0 # DOWN
        elif movement == 3: vel[0] = -1.0 # LEFT
        elif movement == 4: vel[0] = 1.0 # RIGHT
        
        if np.linalg.norm(vel) > 0:
            vel = vel / np.linalg.norm(vel) * self.SHIP_SPEED
        
        self.ship_pos += vel
        self.ship_pos[0] %= self.WORLD_WIDTH
        self.ship_pos[1] %= self.WORLD_HEIGHT

    def _update_asteroids(self):
        asteroids_to_keep = []
        for asteroid in self.asteroids:
            asteroid.update()
            
            screen_pos = asteroid.pos - self.ship_pos + np.array([self.SCREEN_WIDTH, self.SCREEN_HEIGHT]) / 2
            on_screen = -asteroid.size < screen_pos[0] < self.SCREEN_WIDTH + asteroid.size and \
                        -asteroid.size < screen_pos[1] < self.SCREEN_HEIGHT + asteroid.size
            
            if on_screen:
                asteroid.was_on_screen = True
            
            # Keep if not collided and within a larger boundary of the world (prevents instant despawn)
            if not asteroid.collided and \
               -self.SCREEN_WIDTH < asteroid.pos[0] < self.WORLD_WIDTH + self.SCREEN_WIDTH and \
               -self.SCREEN_HEIGHT < asteroid.pos[1] < self.WORLD_HEIGHT + self.SCREEN_HEIGHT:
                asteroids_to_keep.append(asteroid)
            elif asteroid.was_on_screen and not asteroid.collided:
                # Successfully dodged
                self.step_reward += 5.0
                self.dodged_asteroids_total += 1
                if self.dodged_asteroids_total > 0 and self.dodged_asteroids_total % 20 == 0:
                    self.asteroid_speed_multiplier += 0.05
        self.asteroids = asteroids_to_keep

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _spawn_asteroid(self):
        self.asteroids.append(Asteroid((self.WORLD_WIDTH, self.WORLD_HEIGHT), self.asteroid_speed_multiplier))

    def _check_collisions(self):
        ship_radius = self.SHIP_SIZE
        for asteroid in self.asteroids:
            if asteroid.collided:
                continue
            
            dist = np.linalg.norm(self.ship_pos - asteroid.pos)
            if dist < ship_radius + asteroid.size * 0.7: # Asteroid hitbox is slightly smaller than visual
                asteroid.collided = True
                
                # Score penalty
                size_normalized = asteroid.size / 35.0
                penalty = 10.0 * size_normalized
                self.score = max(0, self.score - penalty)
                self.step_reward -= penalty

                # SFX Placeholder: # pygame.mixer.Sound('collision.wav').play()
                # Particle burst effect
                for _ in range(20):
                    self.particles.append(Particle(self.ship_pos, self.COLOR_ASTEROID, 1, 4, 20))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_background()
        self._render_asteroids()
        self._render_particles()
        self._render_ship()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        camera_offset = self.ship_pos
        for x, y, size in self.stars:
            # Apply parallax effect
            px = (x - camera_offset[0] * 0.5) % self.WORLD_WIDTH
            py = (y - camera_offset[1] * 0.5) % self.WORLD_HEIGHT
            
            # Map world to screen
            screen_x = px - self.WORLD_WIDTH/2 + self.SCREEN_WIDTH/2
            screen_y = py - self.WORLD_HEIGHT/2 + self.SCREEN_HEIGHT/2

            if 0 <= screen_x < self.SCREEN_WIDTH and 0 <= screen_y < self.SCREEN_HEIGHT:
                pygame.draw.circle(self.screen, self.COLOR_STAR, (int(screen_x), int(screen_y)), size)

    def _render_ship(self):
        center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        
        score_ratio = np.clip(self.score / self.INITIAL_SCORE, 0, 1)
        ship_color = (
            int(self.COLOR_SHIP_LOW_HEALTH[0] + (self.COLOR_SHIP_HIGH_HEALTH[0] - self.COLOR_SHIP_LOW_HEALTH[0]) * score_ratio),
            int(self.COLOR_SHIP_LOW_HEALTH[1] + (self.COLOR_SHIP_HIGH_HEALTH[1] - self.COLOR_SHIP_LOW_HEALTH[1]) * score_ratio),
            int(self.COLOR_SHIP_LOW_HEALTH[2] + (self.COLOR_SHIP_HIGH_HEALTH[2] - self.COLOR_SHIP_LOW_HEALTH[2]) * score_ratio)
        )

        points = [
            (center_x, center_y - self.SHIP_SIZE),
            (center_x - self.SHIP_SIZE / 1.5, center_y + self.SHIP_SIZE / 2),
            (center_x + self.SHIP_SIZE / 1.5, center_y + self.SHIP_SIZE / 2)
        ]
        
        self._draw_glowing_polygon(self.screen, points, ship_color, glow_radius=15, steps=5)

    def _render_asteroids(self):
        camera_offset = self.ship_pos
        screen_center = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        
        for asteroid in self.asteroids:
            screen_pos = asteroid.pos - camera_offset + screen_center
            
            if -asteroid.size < screen_pos[0] < self.SCREEN_WIDTH + asteroid.size and \
               -asteroid.size < screen_pos[1] < self.SCREEN_HEIGHT + asteroid.size:
                
                points = []
                for i in range(4):
                    angle = math.radians(asteroid.rotation_angle + 90 * i + 45)
                    x = screen_pos[0] + asteroid.size * math.cos(angle)
                    y = screen_pos[1] + asteroid.size * math.sin(angle)
                    points.append((x, y))
                
                self._draw_glowing_polygon(self.screen, points, self.COLOR_ASTEROID, glow_radius=10, steps=4)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen, self.ship_pos, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        dodged_text = self.font.render(f"DODGED: {self.dodged_asteroids_total}/{self.WIN_CONDITION_DODGES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(dodged_text, (10, 40))

    def _draw_glowing_polygon(self, surface, points, color, glow_radius, steps):
        for i in range(steps, 0, -1):
            glow_alpha = 80 * (1 - (i / steps))**2
            glow_color = (*color, glow_alpha)
            
            # This requires a temporary surface for proper alpha blending
            temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            
            inflated_points = []
            center = np.mean(points, axis=0)
            for p in points:
                vec = np.array(p) - center
                norm_vec = vec / (np.linalg.norm(vec) + 1e-6)
                new_p = center + vec + norm_vec * (i * glow_radius / steps)
                inflated_points.append(tuple(new_p))

            pygame.gfxdraw.aapolygon(temp_surf, [(int(p[0]), int(p[1])) for p in inflated_points], glow_color)
            pygame.gfxdraw.filled_polygon(temp_surf, [(int(p[0]), int(p[1])) for p in inflated_points], glow_color)
            surface.blit(temp_surf, (0, 0))

        pygame.gfxdraw.aapolygon(surface, [(int(p[0]), int(p[1])) for p in points], color)
        pygame.gfxdraw.filled_polygon(surface, [(int(p[0]), int(p[1])) for p in points], color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "dodged_asteroids": self.dodged_asteroids_total,
        }

    def _calculate_reward(self):
        # Base survival reward
        reward = 1.0
        # Add rewards from events during the step
        reward += self.step_reward
        return reward

    def _check_termination(self):
        if self.score <= 0:
            return True
        if self.dodged_asteroids_total >= self.WIN_CONDITION_DODGES:
            return True
        return False

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For this to work, you must unset the dummy videodriver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Astro-Recursion")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement_action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement_action = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement_action = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement_action = 4
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Match the intended FPS for smooth gameplay feel

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}, Steps: {info['steps']}")
            # Reset and play again after a pause
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0

    env.close()