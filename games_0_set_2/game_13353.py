import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:56:51.149382
# Source Brief: brief_03353.md
# Brief Index: 3353
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player pilots a rocket through an asteroid field.
    
    The goal is to travel a target distance while collecting fuel and avoiding asteroids.
    The rocket is always centered, and the world moves around it.
    
    Visuals are prioritized, with procedural generation for asteroids, a starfield background,
    and particle effects for the rocket's thruster.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Pilot a rocket through a dangerous asteroid field. Collect fuel for speed boosts "
        "and try to travel the target distance without crashing."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to apply thrust to your rocket."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TARGET_DISTANCE = 1000.0
        self.MAX_STEPS = 1000
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_ROCKET = (230, 230, 240)
        self.COLOR_FLAME_OUTER = (255, 150, 0)
        self.COLOR_FLAME_INNER = (255, 220, 150)
        self.COLOR_FUEL = (50, 255, 50)
        self.COLOR_FUEL_GLOW = (50, 255, 50, 50) # RGBA
        self.COLOR_ASTEROID = (139, 115, 85)
        self.COLOR_ASTEROID_OUTLINE = (90, 70, 50)
        self.COLOR_UI_TEXT = (255, 220, 100)
        
        # Physics & Gameplay
        self.INITIAL_GRAVITY = 0.03
        self.THRUST = 0.25
        self.MAX_SPEED = 6.0
        self.VELOCITY_DECAY = 0.995
        self.FUEL_SPEED_BOOST = 1.2 # 20% boost, more impactful than 10%
        self.DISTANCE_PER_ASTEROID = 100.0
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 72)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_vel = [0.0, 0.0]
        self.distance_traveled = 0.0
        self.fuel_collected = 0
        self.consecutive_fuel = 0
        self.gravity = 0.0
        self.asteroids = []
        self.fuel_orbs = []
        self.stars = []
        self.particles = []
        self.last_asteroid_spawn_dist = 0.0

        # self.reset() is called to initialize np_random
        # self.validate_implementation() is called after to check spaces

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.player_vel = [0.0, 0.0]
        self.distance_traveled = 0.0
        self.fuel_collected = 0
        self.consecutive_fuel = 0
        self.gravity = self.INITIAL_GRAVITY
        
        self.last_asteroid_spawn_dist = 0.0
        
        self.asteroids = []
        for _ in range(5): # Start with 5 asteroids
            self._spawn_asteroid(random_pos=True)
            
        self.fuel_orbs = []
        self._spawn_fuel_orb()
        
        stars_x = self.np_random.integers(0, self.WIDTH, size=150, endpoint=True)
        stars_y = self.np_random.integers(0, self.HEIGHT, size=150, endpoint=True)
        stars_size = self.np_random.integers(1, 2, size=150, endpoint=True)
        self.stars = list(zip(stars_x, stars_y, stars_size))

        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        # --- Action Handling ---
        movement = action[0]
        # space_held (action[1]) and shift_held (action[2]) are ignored
        self._apply_player_movement(movement)

        # --- Physics Update ---
        self._update_physics()

        # --- Update World Entities ---
        self._update_entities()

        # --- Collision & Event Handling ---
        collected_fuel_this_step = self._handle_fuel_collection()
        if collected_fuel_this_step:
            reward += 1.0
        else:
            self.consecutive_fuel = 0
        
        # --- Reward for distance ---
        distance_this_step = max(0, -self.player_vel[1] * 0.1)
        self.distance_traveled += distance_this_step
        reward += distance_this_step * 0.1

        # --- Spawn New Asteroids ---
        if self.distance_traveled > self.last_asteroid_spawn_dist + self.DISTANCE_PER_ASTEROID:
            self._spawn_asteroid()
            self.last_asteroid_spawn_dist += self.DISTANCE_PER_ASTEROID

        # --- Termination Checks ---
        terminated = False
        truncated = False
        if self._check_asteroid_collision():
            reward = -100.0
            terminated = True
            self.game_over = True
        elif self.distance_traveled >= self.TARGET_DISTANCE:
            reward = 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _apply_player_movement(self, movement):
        if movement == 1: # Up
            self.player_vel[1] -= self.THRUST
        elif movement == 2: # Down
            self.player_vel[1] += self.THRUST
        elif movement == 3: # Left
            self.player_vel[0] -= self.THRUST
        elif movement == 4: # Right
            self.player_vel[0] += self.THRUST

    def _update_physics(self):
        # Apply gravity
        self.player_vel[1] += self.gravity
        
        # Apply velocity decay (drag)
        self.player_vel[0] *= self.VELOCITY_DECAY
        self.player_vel[1] *= self.VELOCITY_DECAY
        
        # Cap speed
        speed = math.hypot(*self.player_vel)
        if speed > self.MAX_SPEED:
            self.player_vel[0] = (self.player_vel[0] / speed) * self.MAX_SPEED
            self.player_vel[1] = (self.player_vel[1] / speed) * self.MAX_SPEED

    def _update_entities(self):
        # Move asteroids
        for asteroid in self.asteroids:
            asteroid['pos'][0] -= self.player_vel[0]
            asteroid['pos'][1] -= self.player_vel[1]
            asteroid['angle'] = (asteroid['angle'] + asteroid['rot_speed']) % 360
            if asteroid['pos'][0] < -50: asteroid['pos'][0] = self.WIDTH + 50
            if asteroid['pos'][0] > self.WIDTH + 50: asteroid['pos'][0] = -50
            if asteroid['pos'][1] < -50: asteroid['pos'][1] = self.HEIGHT + 50
            if asteroid['pos'][1] > self.HEIGHT + 50: asteroid['pos'][1] = -50

        # Move fuel orbs
        for orb in self.fuel_orbs:
            orb['pos'][0] -= self.player_vel[0]
            orb['pos'][1] -= self.player_vel[1]
            if orb['pos'][0] < -20: orb['pos'][0] = self.WIDTH + 20
            if orb['pos'][0] > self.WIDTH + 20: orb['pos'][0] = -20
            if orb['pos'][1] < -20: orb['pos'][1] = self.HEIGHT + 20
            if orb['pos'][1] > self.HEIGHT + 20: orb['pos'][1] = -20
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1

    def _handle_fuel_collection(self):
        player_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        player_radius = 10
        collected_any = False
        
        for orb in self.fuel_orbs[:]:
            dist = math.hypot(player_pos[0] - orb['pos'][0], player_pos[1] - orb['pos'][1])
            if dist < player_radius + orb['radius']:
                self.fuel_orbs.remove(orb)
                self._spawn_fuel_orb()
                
                self.fuel_collected += 1
                self.consecutive_fuel += 1
                collected_any = True
                
                # Boost speed
                speed = math.hypot(*self.player_vel)
                if speed > 0.1:
                    new_speed = speed * self.FUEL_SPEED_BOOST
                    self.player_vel = [(v / speed) * new_speed for v in self.player_vel]
                else: # Give a small push upwards if stationary
                    self.player_vel[1] -= 1.0

                # Check for gravity increase
                if self.consecutive_fuel >= 3:
                    self.gravity *= 2.0
                    self.consecutive_fuel = 0
        return collected_any

    def _check_asteroid_collision(self):
        player_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        player_radius = 8 # Smaller hitbox for fairness
        for asteroid in self.asteroids:
            dist = math.hypot(player_pos[0] - asteroid['pos'][0], player_pos[1] - asteroid['pos'][1])
            if dist < player_radius + asteroid['radius'] * 0.8: # Use 80% of visual radius for hitbox
                return True
        return False

    def _spawn_asteroid(self, random_pos=False):
        radius = self.np_random.integers(15, 40, endpoint=True)
        if random_pos:
             # Spawn anywhere but the center for the initial set
            while True:
                pos = self.np_random.uniform(low=[0,0], high=[self.WIDTH, self.HEIGHT]).tolist()
                if math.hypot(pos[0] - self.WIDTH/2, pos[1] - self.HEIGHT/2) > 200:
                    break
        else:
            # Spawn off-screen
            side = self.np_random.choice(['top', 'bottom', 'left', 'right'])
            if side == 'top':
                pos = [self.np_random.uniform(0, self.WIDTH), -radius]
            elif side == 'bottom':
                pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + radius]
            elif side == 'left':
                pos = [-radius, self.np_random.uniform(0, self.HEIGHT)]
            else: # right
                pos = [self.WIDTH + radius, self.np_random.uniform(0, self.HEIGHT)]

        num_points = self.np_random.integers(7, 12, endpoint=True)
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            dist = self.np_random.uniform(radius * 0.7, radius)
            points.append((math.cos(angle) * dist, math.sin(angle) * dist))
            
        self.asteroids.append({
            'pos': pos,
            'radius': radius,
            'points': points,
            'angle': self.np_random.uniform(0, 360),
            'rot_speed': self.np_random.uniform(-0.5, 0.5)
        })

    def _spawn_fuel_orb(self):
        while True:
            pos = self.np_random.uniform(low=50, high=[self.WIDTH - 50, self.HEIGHT - 50]).tolist()
            # Ensure it doesn't spawn inside an asteroid
            too_close = False
            for asteroid in self.asteroids:
                if math.hypot(pos[0] - asteroid['pos'][0], pos[1] - asteroid['pos'][1]) < asteroid['radius'] + 30:
                    too_close = True
                    break
            if not too_close:
                break
        self.fuel_orbs.append({'pos': pos, 'radius': 10})
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_asteroids()
        self._render_fuel_orbs()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": self.distance_traveled,
            "fuel": self.fuel_collected,
            "speed": math.hypot(*self.player_vel),
            "gravity": self.gravity
        }

    def _render_background(self):
        for x, y, size in self.stars:
            # Move stars opposite to player velocity for parallax effect
            star_x = (x - self.player_vel[0] * 0.1 * size) % self.WIDTH
            star_y = (y - self.player_vel[1] * 0.1 * size) % self.HEIGHT
            color_val = 50 * size
            pygame.draw.rect(self.screen, (color_val, color_val, color_val + 20), (int(star_x), int(star_y), size, size))

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            x, y = asteroid['pos']
            angle_rad = math.radians(asteroid['angle'])
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            
            rotated_points = []
            for px, py in asteroid['points']:
                rx = px * cos_a - py * sin_a + x
                ry = px * sin_a + py * cos_a + y
                rotated_points.append((int(rx), int(ry)))
            
            if len(rotated_points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_ASTEROID_OUTLINE)
    
    def _render_fuel_orbs(self):
        for orb in self.fuel_orbs:
            x, y = int(orb['pos'][0]), int(orb['pos'][1])
            # Pulsating glow effect
            pulse = (math.sin(self.steps * 0.1) + 1) / 2 # Varies between 0 and 1
            glow_radius = int(orb['radius'] * (1.5 + pulse * 0.5))
            pygame.gfxdraw.filled_circle(self.screen, x, y, glow_radius, self.COLOR_FUEL_GLOW)
            # Main orb
            pygame.gfxdraw.filled_circle(self.screen, x, y, orb['radius'], self.COLOR_FUEL)
            pygame.gfxdraw.aacircle(self.screen, x, y, orb['radius'], self.COLOR_FUEL)

    def _render_player(self):
        x, y = self.WIDTH // 2, self.HEIGHT // 2
        
        # Rocket body
        angle = math.atan2(self.player_vel[0], -self.player_vel[1]) # Point in direction of velocity
        
        p1 = (x + math.sin(angle) * 15, y - math.cos(angle) * 15)
        p2 = (x + math.sin(angle + 2.2) * 10, y - math.cos(angle + 2.2) * 10)
        p3 = (x + math.sin(angle - 2.2) * 10, y - math.cos(angle - 2.2) * 10)
        
        pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_ROCKET)
        pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_ROCKET)

        # Rocket flame
        speed = math.hypot(*self.player_vel)
        if speed > 0.1:
            flame_length = min(25, 5 + speed * 3)
            flame_width = min(10, 3 + speed)
            
            # Base of flame is opposite to p1
            base_x = (p2[0] + p3[0]) / 2
            base_y = (p2[1] + p3[1]) / 2
            
            # Tip of flame extends from base
            tip_x = base_x - math.sin(angle) * flame_length * (1 + self.np_random.uniform(-0.1, 0.1))
            tip_y = base_y + math.cos(angle) * flame_length * (1 + self.np_random.uniform(-0.1, 0.1))
            
            flame_p2 = (base_x + math.sin(angle + 1.57) * flame_width, base_y - math.cos(angle + 1.57) * flame_width)
            flame_p3 = (base_x + math.sin(angle - 1.57) * flame_width, base_y - math.cos(angle - 1.57) * flame_width)
            
            # Outer flame
            pygame.gfxdraw.filled_trigon(self.screen, int(tip_x), int(tip_y), int(flame_p2[0]), int(flame_p2[1]), int(flame_p3[0]), int(flame_p3[1]), self.COLOR_FLAME_OUTER)
            
            # Inner flame
            pygame.gfxdraw.filled_trigon(self.screen, int(tip_x), int(tip_y), int(flame_p2[0]*0.5 + base_x*0.5), int(flame_p2[1]*0.5 + base_y*0.5), int(flame_p3[0]*0.5 + base_x*0.5), int(flame_p3[1]*0.5 + base_y*0.5), self.COLOR_FLAME_INNER)
            
            # Emit particles
            if self.steps % 2 == 0:
                p_vel_x = -math.sin(angle) * 2 + self.np_random.uniform(-0.5, 0.5)
                p_vel_y = math.cos(angle) * 2 + self.np_random.uniform(-0.5, 0.5)
                self.particles.append({
                    'pos': [base_x, base_y],
                    'vel': [p_vel_x, p_vel_y],
                    'life': 20,
                    'radius': self.np_random.uniform(2, 4),
                    'color': self.np_random.choice([self.COLOR_FLAME_OUTER, self.COLOR_FLAME_INNER])
                })

    def _render_particles(self):
        for p in self.particles:
            if p['life'] > 0:
                radius = int(p['radius'] * (p['life'] / 20.0))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])

    def _render_ui(self):
        dist_text = self.font.render(f"DISTANCE: {int(self.distance_traveled)} / {int(self.TARGET_DISTANCE)}", True, self.COLOR_UI_TEXT)
        fuel_text = self.font.render(f"FUEL: {self.fuel_collected}", True, self.COLOR_UI_TEXT)
        speed = math.hypot(*self.player_vel)
        speed_text = self.font.render(f"SPEED: {speed:.1f}", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(dist_text, (10, 10))
        self.screen.blit(fuel_text, (10, 40))
        self.screen.blit(speed_text, (10, 70))
        
        if self.game_over:
            if self.distance_traveled >= self.TARGET_DISTANCE:
                end_text = self.font_large.render("VICTORY!", True, self.COLOR_FUEL)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_FLAME_OUTER)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # Use arrow keys to control the rocket
    
    obs, info = env.reset(seed=42)
    done = False
    
    # Pygame window for human interaction
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"] # Allow display for manual play
    pygame.display.set_caption("Rocket Asteroid Field")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no movement
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
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
            
        action = [movement, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset(seed=42)

        clock.tick(30) # Run at 30 FPS

    env.close()