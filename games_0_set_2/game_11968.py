import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:48:50.667917
# Source Brief: brief_01968.md
# Brief Index: 1968
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for piloting a spaceship through an asteroid field.
    The goal is to collect 7 power-ups to activate a hyperspace jump,
    while avoiding deadly asteroids and a time limit.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    game_description = (
        "Pilot a spaceship through a dangerous asteroid field. "
        "Collect power-ups to enable a hyperspace jump before time runs out."
    )
    user_guide = "Controls: Use arrow keys (↑↓←→) to apply thrust and navigate your ship."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (100, 180, 255)
    COLOR_POWERUP = (50, 255, 150)
    COLOR_POWERUP_GLOW = (150, 255, 200)
    COLOR_ASTEROID_SAFE = (150, 150, 150)
    COLOR_ASTEROID_DEADLY = (255, 80, 80)
    COLOR_ASTEROID_DEADLY_GLOW = (255, 120, 120)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_METER_BG = (40, 40, 60)
    COLOR_METER_FILL = (255, 220, 0)
    
    # Player
    PLAYER_RADIUS = 12
    PLAYER_ACCELERATION = 0.4
    PLAYER_MAX_SPEED = 6.0
    PLAYER_DRAG = 0.98
    
    # Game
    GAME_DURATION_SECONDS = 120
    POWERUPS_TO_WIN = 7
    INITIAL_ASTEROIDS = 5
    ASTEROID_SPAWN_INTERVAL_SECONDS = 10
    DEADLY_ASTEROID_CHANCE = 0.15 # Increased from 0.1 for more challenge

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        if self.render_mode == "human":
            pygame.display.set_caption("Asteroid Jumper")
            self.display_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0.0
        
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = 0.0
        
        self.asteroids = []
        self.powerups = []
        self.particles = []
        self.stars = []
        
        self.powerup_count = 0
        self.next_asteroid_spawn_time = 0
        
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.GAME_DURATION_SECONDS
        
        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = -90
        
        self.powerup_count = 0
        
        self.asteroids.clear()
        self.powerups.clear()
        self.particles.clear()
        self.stars.clear()

        self._create_stars(200)
        for _ in range(self.INITIAL_ASTEROIDS):
            self._spawn_asteroid()
        
        self._spawn_powerup()

        self.next_asteroid_spawn_time = self.GAME_DURATION_SECONDS - self.ASTEROID_SPAWN_INTERVAL_SECONDS
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.01 # Small reward for surviving a step
        
        # --- Update Game Logic ---
        self._update_player(movement)
        self._update_asteroids()
        self._update_particles()
        
        self.timer -= 1.0 / self.metadata["render_fps"]
        
        # Difficulty scaling
        if self.timer < self.next_asteroid_spawn_time:
            self._spawn_asteroid()
            self.next_asteroid_spawn_time -= self.ASTEROID_SPAWN_INTERVAL_SECONDS

        # Collision detection and rewards
        reward += self._handle_collisions()
        
        self.steps += 1
        terminated = self._check_termination()
        truncated = False # This environment does not truncate based on time limits

        # Terminal rewards
        if terminated:
            if self.powerup_count >= self.POWERUPS_TO_WIN:
                reward += 100.0 # Victory
            elif self.timer <= 0:
                reward += -50.0 # Timeout
            else:
                reward += -100.0 # Crashed

        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def render(self):
        self._render_all()
        if self.render_mode == "human":
            self.display_screen.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def close(self):
        pygame.quit()

    # --- Private Helper Methods ---

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer, "powerups": self.powerup_count}

    def _update_player(self, movement):
        acceleration = pygame.math.Vector2(0, 0)
        thruster_on = False
        if movement == 1: # Up
            acceleration.y = -self.PLAYER_ACCELERATION
            thruster_on = True
        elif movement == 2: # Down
            acceleration.y = self.PLAYER_ACCELERATION
            thruster_on = True
        elif movement == 3: # Left
            acceleration.x = -self.PLAYER_ACCELERATION
            thruster_on = True
        elif movement == 4: # Right
            acceleration.x = self.PLAYER_ACCELERATION
            thruster_on = True

        self.player_vel += acceleration
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)
        
        self.player_vel *= self.PLAYER_DRAG
        self.player_pos += self.player_vel

        # Screen wrap
        self.player_pos.x %= self.SCREEN_WIDTH
        self.player_pos.y %= self.SCREEN_HEIGHT

        # Update angle to face velocity direction
        if self.player_vel.length() > 0.1:
            self.player_angle = self.player_vel.angle_to(pygame.math.Vector2(1, 0))

        # Thruster particles
        if thruster_on:
            self._create_particles(
                self.player_pos, 5, (255, 150, 50), 0.5, 1.5,
                angle=-self.player_angle, cone_spread=15
            )

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] += asteroid['rot_speed']
            
            # Screen wrap
            if asteroid['pos'].x < -asteroid['radius']: asteroid['pos'].x = self.SCREEN_WIDTH + asteroid['radius']
            if asteroid['pos'].x > self.SCREEN_WIDTH + asteroid['radius']: asteroid['pos'].x = -asteroid['radius']
            if asteroid['pos'].y < -asteroid['radius']: asteroid['pos'].y = self.SCREEN_HEIGHT + asteroid['radius']
            if asteroid['pos'].y > self.SCREEN_HEIGHT + asteroid['radius']: asteroid['pos'].y = -asteroid['radius']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.98

    def _handle_collisions(self):
        reward = 0
        
        # Player vs Powerups
        for powerup in self.powerups[:]:
            if self.player_pos.distance_to(powerup['pos']) < self.PLAYER_RADIUS + powerup['radius']:
                self.powerups.remove(powerup)
                self.powerup_count += 1
                reward += 1.0
                self._create_particles(powerup['pos'], 30, self.COLOR_POWERUP, 2, 5)
                self._spawn_powerup()
                if self.powerup_count >= self.POWERUPS_TO_WIN:
                    self.game_over = True # Win condition met
                    self._create_particles(self.player_pos, 100, (255, 255, 255), 3, 8, cone_spread=360)
        
        # Player vs Asteroids
        for asteroid in self.asteroids:
            if self.player_pos.distance_to(asteroid['pos']) < self.PLAYER_RADIUS + asteroid['radius']:
                if asteroid['deadly']:
                    self.game_over = True
                    self._create_particles(self.player_pos, 100, self.COLOR_ASTEROID_DEADLY, 2, 7)
                else: # Safe asteroid collision
                    reward += 0.5
                    self.powerup_count = min(self.powerup_count + 0.1, self.POWERUPS_TO_WIN) # Minor meter increase
                    self._create_particles(asteroid['pos'], 20, self.COLOR_ASTEROID_SAFE, 1, 3)
                    # Push away other asteroids
                    for other in self.asteroids:
                        if other is not asteroid:
                            dist = asteroid['pos'].distance_to(other['pos'])
                            if dist < 100 and dist > 0:
                                push_vec = (other['pos'] - asteroid['pos']).normalize()
                                other['vel'] += push_vec * (100 - dist) / 50.0
                    
                    asteroid['vel'] = (asteroid['pos'] - self.player_pos).normalize() * 2 # Bounce asteroid away
                    self.player_vel *= -0.5 # Bounce player back

        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.timer <= 0:
            self.game_over = True
            return True
        return False

    def _spawn_asteroid(self):
        edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
        radius = self.np_random.uniform(15, 35)
        pos = pygame.math.Vector2(0, 0)

        if edge == 'top':
            pos.x = self.np_random.uniform(0, self.SCREEN_WIDTH)
            pos.y = -radius
        elif edge == 'bottom':
            pos.x = self.np_random.uniform(0, self.SCREEN_WIDTH)
            pos.y = self.SCREEN_HEIGHT + radius
        elif edge == 'left':
            pos.x = -radius
            pos.y = self.np_random.uniform(0, self.SCREEN_HEIGHT)
        else: # right
            pos.x = self.SCREEN_WIDTH + radius
            pos.y = self.np_random.uniform(0, self.SCREEN_HEIGHT)

        angle = self.np_random.uniform(0, 360)
        target = pygame.math.Vector2(
            self.SCREEN_WIDTH * self.np_random.uniform(0.2, 0.8),
            self.SCREEN_HEIGHT * self.np_random.uniform(0.2, 0.8)
        )
        vel = (target - pos).normalize() * self.np_random.uniform(0.5, 2.0)

        points = []
        num_points = self.np_random.integers(7, 13)
        for i in range(num_points):
            angle_rad = (i / num_points) * 2 * math.pi
            dist = self.np_random.uniform(0.7, 1.1) * radius
            points.append((math.cos(angle_rad) * dist, math.sin(angle_rad) * dist))

        self.asteroids.append({
            'pos': pos, 'vel': vel, 'radius': radius,
            'deadly': self.np_random.random() < self.DEADLY_ASTEROID_CHANCE,
            'angle': 0, 'rot_speed': self.np_random.uniform(-2, 2),
            'points': points
        })

    def _spawn_powerup(self):
        # Avoid spawning on asteroids
        while True:
            pos = pygame.math.Vector2(
                self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
                self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
            )
            too_close = False
            for asteroid in self.asteroids:
                if pos.distance_to(asteroid['pos']) < asteroid['radius'] + 50:
                    too_close = True
                    break
            if not too_close:
                break
        
        self.powerups.append({'pos': pos, 'radius': 10})

    def _create_particles(self, pos, count, color, min_speed, max_speed, cone_spread=360, angle=0):
        for _ in range(count):
            particle_angle = math.radians(self.np_random.uniform(-cone_spread / 2, cone_spread / 2) + angle)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.math.Vector2(math.cos(particle_angle), -math.sin(particle_angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(20, 41),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })
    
    def _create_stars(self, count):
        for _ in range(count):
            self.stars.append({
                'pos': pygame.math.Vector2(self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                'brightness': self.np_random.integers(50, 151),
                'size': self.np_random.uniform(0.5, 1.5)
            })

    # --- Rendering Methods ---

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_asteroids()
        self._render_powerups()
        self._render_player()
        self._render_ui()

    def _render_background(self):
        for star in self.stars:
            b = star['brightness']
            pygame.draw.circle(self.screen, (b, b, b), (int(star['pos'].x), int(star['pos'].y)), star['size'])

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40.0))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            
            radius = int(max(0, p['radius']))
            if radius == 0: continue
            
            surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (radius, radius), radius)
            self.screen.blit(surf, (int(p['pos'].x - radius), int(p['pos'].y - radius)))

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pos = (int(asteroid['pos'].x), int(asteroid['pos'].y))
            color = self.COLOR_ASTEROID_DEADLY if asteroid['deadly'] else self.COLOR_ASTEROID_SAFE
            
            if asteroid['deadly']:
                glow_radius = int(asteroid['radius'] * 1.5)
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*self.COLOR_ASTEROID_DEADLY_GLOW, 50), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

            rotated_points = []
            angle_rad = math.radians(asteroid['angle'])
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            for p in asteroid['points']:
                x = p[0] * cos_a - p[1] * sin_a + pos[0]
                y = p[0] * sin_a + p[1] * cos_a + pos[1]
                rotated_points.append((x, y))
            
            pygame.gfxdraw.aapolygon(self.screen, rotated_points, color)
            pygame.gfxdraw.filled_polygon(self.screen, rotated_points, color)

    def _render_powerups(self):
        for powerup in self.powerups:
            pos = (int(powerup['pos'].x), int(powerup['pos'].y))
            
            pulse = math.sin(self.steps * 0.1) * 3
            radius = int(powerup['radius'] + pulse)
            
            glow_radius = int(radius * 2)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_POWERUP_GLOW, 70), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_POWERUP)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_POWERUP)

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        
        glow_radius = int(self.PLAYER_RADIUS * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 60), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

        angle_rad = math.radians(self.player_angle)
        p1 = (
            pos[0] + math.cos(angle_rad) * self.PLAYER_RADIUS,
            pos[1] - math.sin(angle_rad) * self.PLAYER_RADIUS
        )
        p2 = (
            pos[0] + math.cos(angle_rad + 2.4) * self.PLAYER_RADIUS,
            pos[1] - math.sin(angle_rad + 2.4) * self.PLAYER_RADIUS
        )
        p3 = (
            pos[0] + math.cos(angle_rad - 2.4) * self.PLAYER_RADIUS,
            pos[1] - math.sin(angle_rad - 2.4) * self.PLAYER_RADIUS
        )
        points = [p1, p2, p3]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        timer_text = f"TIME: {max(0, int(self.timer))}"
        timer_surf = self.font_small.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 10, 10))

        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        meter_width = self.SCREEN_WIDTH - 40
        meter_height = 20
        meter_x = 20
        meter_y = self.SCREEN_HEIGHT - 30
        
        fill_ratio = min(1.0, self.powerup_count / self.POWERUPS_TO_WIN)
        fill_width = int(meter_width * fill_ratio)
        
        pygame.draw.rect(self.screen, self.COLOR_METER_BG, (meter_x, meter_y, meter_width, meter_height), border_radius=5)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_METER_FILL, (meter_x, meter_y, fill_width, meter_height), border_radius=5)
        
        meter_text = f"HYPERSPACE: {int(fill_ratio * 100)}%"
        meter_surf = self.font_small.render(meter_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(meter_surf, (meter_x + (meter_width - meter_surf.get_width()) / 2, meter_y))

        if self.game_over:
            if self.powerup_count >= self.POWERUPS_TO_WIN:
                msg = "HYPERSPACE JUMP SUCCESSFUL!"
            elif self.timer <= 0:
                msg = "TIME'S UP!"
            else:
                msg = "GAME OVER"
            
            msg_surf = self.font_large.render(msg, True, (255, 255, 255))
            self.screen.blit(msg_surf, (
                (self.SCREEN_WIDTH - msg_surf.get_width()) / 2,
                (self.SCREEN_HEIGHT - msg_surf.get_height()) / 2
            ))

if __name__ == "__main__":
    env = GameEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    
    terminated = False
    total_reward = 0
    
    print("Controls: Arrow keys to move. Close window to quit.")
    
    clock = pygame.time.Clock()
    while not terminated:
        action = [0, 0, 0] 
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset(seed=random.randint(0, 10000))
            terminated = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.metadata["render_fps"])

    env.close()