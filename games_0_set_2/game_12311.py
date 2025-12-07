import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:40:44.585591
# Source Brief: brief_02311.md
# Brief Index: 2311
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player pilots a rocket through a procedurally
    generated asteroid field. The goal is to reach the finish line while collecting
    fuel to extend the timer and avoiding asteroids.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a rocket through a dangerous asteroid field. Collect fuel to extend your time and navigate to the finish line to win."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to steer the rocket and dodge asteroids."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 10, 25)
    COLOR_ROCKET = (224, 66, 66)
    COLOR_ROCKET_GLOW = (255, 100, 100)
    COLOR_FLAME_START = (255, 204, 0)
    COLOR_FLAME_END = (255, 77, 0)
    COLOR_ASTEROID = (120, 120, 130)
    COLOR_ASTEROID_GLOW = (150, 150, 160)
    COLOR_FUEL = (66, 224, 101)
    COLOR_FUEL_GLOW = (100, 255, 130)
    COLOR_FINISH_LINE = (66, 135, 224)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_TIMER_BAR = (66, 135, 224)
    COLOR_TIMER_BAR_LOW = (224, 66, 66)

    # Rocket Physics
    ROCKET_THRUST = 0.25
    ROCKET_DRAG = 0.95
    ROCKET_MAX_VEL = 5.0
    ROCKET_RADIUS = 12

    # Game Parameters
    LEVEL_LENGTH = 20000  # Total distance to scroll in pixels
    INITIAL_TIMER = 30.0  # seconds
    MAX_EPISODE_STEPS = 5000
    FPS = 30.0
    
    # Spawning
    ASTEROID_SPAWN_CHANCE = 0.05
    FUEL_SPAWN_CHANCE = 0.015

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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        self.render_mode = render_mode
        
        # Initialize state variables to prevent uninitialized attribute errors
        self.steps = 0
        self.score = 0
        self.timer = 0.0
        self.world_y = 0.0
        self.rocket_vel_x = 0.0
        self.rocket_pos = pygame.Vector2(0, 0)
        self.asteroids = []
        self.fuel_canisters = []
        self.particles = []
        self.stars = []
        self.current_asteroid_speed = 0.0
        
        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.INITIAL_TIMER
        
        self.world_y = 0.0
        self.rocket_vel_x = 0.0
        self.rocket_pos = pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT * 0.8)

        self.asteroids = []
        self.fuel_canisters = []
        self.particles = []
        
        self.base_asteroid_speed = 1.0
        self.current_asteroid_speed = self.base_asteroid_speed
        
        if not self.stars:
            self.stars = [
                (
                    self.np_random.uniform(0, self.SCREEN_WIDTH),
                    self.np_random.uniform(0, self.SCREEN_HEIGHT),
                    self.np_random.uniform(0.1, 1.0), # scroll speed factor
                    self.np_random.integers(1, 3) # size
                )
                for _ in range(200)
            ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # --- Update Game Logic ---
        if self.timer > 0:
            self._handle_input(movement)
            self._update_world()
            self._spawn_objects()
            
            collision_reward, collision_detected = self._handle_collisions()
            reward += collision_reward
            
            if collision_detected:
                terminated = True
        
        self._update_particles()

        # --- Update Timer & Score ---
        self.timer -= 1.0 / self.FPS
        if not terminated:
            reward += 0.1 # Survival reward
            
        # --- Check Termination Conditions ---
        if self.world_y >= self.LEVEL_LENGTH:
            terminated = True
            reward += 100.0  # Victory reward
            self._create_particle_burst(self.rocket_pos, self.COLOR_FINISH_LINE, 50)
        elif self.timer <= 0:
            terminated = True
            # No specific penalty for running out of time, the lack of victory is the penalty.
        elif self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True # Use truncated for time/step limits
            
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.rocket_vel_x -= self.ROCKET_THRUST
            self._create_thruster_particles(side='right')
        elif movement == 4:  # Right
            self.rocket_vel_x += self.ROCKET_THRUST
            self._create_thruster_particles(side='left')

    def _update_world(self):
        # Update rocket horizontal position with drag and clamping
        self.rocket_vel_x *= self.ROCKET_DRAG
        self.rocket_vel_x = np.clip(self.rocket_vel_x, -self.ROCKET_MAX_VEL, self.ROCKET_MAX_VEL)
        self.rocket_pos.x += self.rocket_vel_x
        self.rocket_pos.x = np.clip(self.rocket_pos.x, self.ROCKET_RADIUS, self.SCREEN_WIDTH - self.ROCKET_RADIUS)

        # Update world vertical scroll
        scroll_speed = 2.0 + (self.world_y / self.LEVEL_LENGTH) * 4.0 # Speed up as we progress
        self.world_y += scroll_speed

        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.base_asteroid_speed += 0.05
        self.current_asteroid_speed = self.base_asteroid_speed + (self.world_y / self.LEVEL_LENGTH) * 2.0

        # Update objects
        for obj_list in [self.asteroids, self.fuel_canisters]:
            for obj in obj_list:
                obj['pos'].y += self.current_asteroid_speed
                obj['angle'] = (obj['angle'] + obj['rot_speed']) % 360
        
        # Prune off-screen objects
        self.asteroids = [a for a in self.asteroids if a['pos'].y < self.world_y + self.SCREEN_HEIGHT + 50]
        self.fuel_canisters = [f for f in self.fuel_canisters if f['pos'].y < self.world_y + self.SCREEN_HEIGHT + 50]

    def _spawn_objects(self):
        # Spawn asteroids
        if self.np_random.random() < self.ASTEROID_SPAWN_CHANCE:
            radius = self.np_random.integers(15, 40)
            self.asteroids.append({
                'pos': pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.world_y - 50),
                'radius': radius,
                'angle': self.np_random.uniform(0, 360),
                'rot_speed': self.np_random.uniform(-1.5, 1.5),
                'shape': self._create_asteroid_shape(radius, self.np_random.integers(7, 12))
            })

        # Spawn fuel
        if self.np_random.random() < self.FUEL_SPAWN_CHANCE:
            self.fuel_canisters.append({
                'pos': pygame.Vector2(self.np_random.uniform(50, self.SCREEN_WIDTH - 50), self.world_y - 50),
                'radius': 12,
                'angle': 0,
                'rot_speed': 2.0
            })

    def _handle_collisions(self):
        reward = 0.0
        collision_detected = False

        # Rocket vs Asteroids
        for asteroid in self.asteroids[:]:
            dist = self.rocket_pos.distance_to(self._get_screen_pos(asteroid['pos']))
            if dist < self.ROCKET_RADIUS + asteroid['radius']:
                reward = -100.0  # Collision penalty
                collision_detected = True
                self.asteroids.remove(asteroid)
                self._create_particle_burst(self.rocket_pos, self.COLOR_ROCKET, 100)
                # Sound: Explosion
                break
        
        if collision_detected:
            return reward, True

        # Rocket vs Fuel
        for fuel in self.fuel_canisters[:]:
            dist = self.rocket_pos.distance_to(self._get_screen_pos(fuel['pos']))
            if dist < self.ROCKET_RADIUS + fuel['radius']:
                reward += 1.0  # Fuel collection reward
                self.score += 10
                self.timer = min(self.INITIAL_TIMER, self.timer + 5.0)
                self.fuel_canisters.remove(fuel)
                self._create_particle_burst(self._get_screen_pos(fuel['pos']), self.COLOR_FUEL, 30)
                # Sound: Fuel pickup
        
        return reward, False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_finish_line()
        self._render_objects()
        if self.timer > 0 or self.world_y >= self.LEVEL_LENGTH:
            self._render_rocket()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "progress": self.world_y / self.LEVEL_LENGTH,
        }
    
    def _get_screen_pos(self, world_pos):
        return pygame.Vector2(world_pos.x, world_pos.y - self.world_y)

    # --- Rendering Methods ---

    def _render_background(self):
        for x, y, speed, size in self.stars:
            screen_y = (y - self.world_y * speed) % self.SCREEN_HEIGHT
            pygame.draw.rect(self.screen, (255, 255, 255), (x, screen_y, size, size))
    
    def _render_finish_line(self):
        finish_screen_y = self.LEVEL_LENGTH - self.world_y
        if 0 < finish_screen_y < self.SCREEN_HEIGHT:
            for i in range(20):
                alpha = 100 + (math.sin(self.steps * 0.1 + i) * 50)
                color = (*self.COLOR_FINISH_LINE, alpha)
                height = 20 - i
                pygame.draw.rect(self.screen, color, (0, finish_screen_y + i * 2, self.SCREEN_WIDTH, height), 0)

    def _render_objects(self):
        # Render Asteroids
        for asteroid in self.asteroids:
            screen_pos = self._get_screen_pos(asteroid['pos'])
            if -50 < screen_pos.y < self.SCREEN_HEIGHT + 50:
                # Glow
                pygame.gfxdraw.filled_circle(
                    self.screen, int(screen_pos.x), int(screen_pos.y),
                    int(asteroid['radius'] * 1.2), (*self.COLOR_ASTEROID_GLOW, 50)
                )
                # Body
                rotated_shape = self._rotate_points(asteroid['shape'], screen_pos, asteroid['angle'])
                pygame.gfxdraw.filled_polygon(self.screen, rotated_shape, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, rotated_shape, self.COLOR_ASTEROID)
        
        # Render Fuel
        for fuel in self.fuel_canisters:
            screen_pos = self._get_screen_pos(fuel['pos'])
            if -20 < screen_pos.y < self.SCREEN_HEIGHT + 20:
                radius = fuel['radius']
                # Glow
                pygame.gfxdraw.filled_circle(
                    self.screen, int(screen_pos.x), int(screen_pos.y),
                    int(radius * 1.8), (*self.COLOR_FUEL_GLOW, 80)
                )
                # Body (rotating cross)
                points = [(-radius, 0), (radius, 0), (0, 0), (0, -radius), (0, radius)]
                rotated_points = self._rotate_points(points, screen_pos, fuel['angle'])
                pygame.draw.line(self.screen, self.COLOR_FUEL, rotated_points[0], rotated_points[1], 4)
                pygame.draw.line(self.screen, self.COLOR_FUEL, rotated_points[3], rotated_points[4], 4)

    def _render_rocket(self):
        p = self.rocket_pos
        r = self.ROCKET_RADIUS
        
        # Glow effect
        glow_radius = int(r * 2.5 + abs(math.sin(self.steps * 0.2)) * 5)
        glow_alpha = int(60 + abs(math.sin(self.steps * 0.2)) * 20)
        pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), glow_radius, (*self.COLOR_ROCKET_GLOW, glow_alpha))

        # Rocket body points
        points = [
            (p.x, p.y - r),
            (p.x - r * 0.7, p.y + r * 0.7),
            (p.x + r * 0.7, p.y + r * 0.7)
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ROCKET)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ROCKET)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            p_color = (
                int(p['start_color'][0] * life_ratio + p['end_color'][0] * (1 - life_ratio)),
                int(p['start_color'][1] * life_ratio + p['end_color'][1] * (1 - life_ratio)),
                int(p['start_color'][2] * life_ratio + p['end_color'][2] * (1 - life_ratio))
            )
            p_radius = int(p['radius'] * life_ratio)
            if p_radius > 0:
                pygame.draw.circle(self.screen, p_color, p['pos'], p_radius)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Progress
        progress_percent = min(100, int((self.world_y / self.LEVEL_LENGTH) * 100))
        progress_text = self.font_small.render(f"PROGRESS: {progress_percent}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(progress_text, (self.SCREEN_WIDTH - progress_text.get_width() - 10, 10))

        # Timer bar
        timer_ratio = max(0, self.timer / self.INITIAL_TIMER)
        bar_width = (self.SCREEN_WIDTH - 20) * timer_ratio
        bar_color = self.COLOR_TIMER_BAR if timer_ratio > 0.25 else self.COLOR_TIMER_BAR_LOW
        pygame.draw.rect(self.screen, (40,40,60), (10, self.SCREEN_HEIGHT - 20, self.SCREEN_WIDTH - 20, 10))
        pygame.draw.rect(self.screen, bar_color, (10, self.SCREEN_HEIGHT - 20, bar_width, 10))

        # Game Over / Victory Message
        if self.timer <= 0 or self.world_y >= self.LEVEL_LENGTH:
            if self.world_y >= self.LEVEL_LENGTH:
                msg = "VICTORY"
                color = self.COLOR_FINISH_LINE
            else:
                msg = "GAME OVER"
                color = self.COLOR_ROCKET
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    # --- Particle and Shape Helpers ---

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particle_burst(self, pos, color, count):
        # Sound: Burst/Pop
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'radius': self.np_random.integers(3, 7),
                'start_color': color, 'end_color': self.COLOR_BG,
                'life': self.np_random.integers(15, 30), 'max_life': 30
            })

    def _create_thruster_particles(self, side):
        # Sound: Thruster hiss
        for _ in range(2):
            if side == 'left':
                pos = self.rocket_pos + pygame.Vector2(-self.ROCKET_RADIUS * 0.6, self.ROCKET_RADIUS * 0.5)
                vel = pygame.Vector2(-self.np_random.uniform(1, 3), self.np_random.uniform(-0.5, 0.5))
            else: # right
                pos = self.rocket_pos + pygame.Vector2(self.ROCKET_RADIUS * 0.6, self.ROCKET_RADIUS * 0.5)
                vel = pygame.Vector2(self.np_random.uniform(1, 3), self.np_random.uniform(-0.5, 0.5))
            
            self.particles.append({
                'pos': pos, 'vel': vel + pygame.Vector2(self.rocket_vel_x, 0),
                'radius': self.np_random.integers(2, 5),
                'start_color': self.COLOR_FLAME_START, 'end_color': self.COLOR_FLAME_END,
                'life': self.np_random.integers(10, 20), 'max_life': 20
            })

    def _create_asteroid_shape(self, radius, num_points):
        points = []
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points)
            dist = self.np_random.uniform(0.7, 1.0) * radius
            points.append((math.cos(angle) * dist, math.sin(angle) * dist))
        return points

    def _rotate_points(self, points, center, angle_degrees):
        angle_rad = math.radians(angle_degrees)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rotated = []
        for x, y in points:
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            rotated.append((center.x + x_rot, center.y + y_rot))
        return rotated
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # The main script now handles window creation
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.set_caption("Rocket Asteroid Dodger")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not terminated and not truncated:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        action = [movement, 0, 0] # Movement, space, shift

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause before quitting

    env.close()