import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:33:55.133538
# Source Brief: brief_00525.md
# Brief Index: 525
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle class for visual effects."""
    def __init__(self, x, y, vx, vy, life, color, radius, gravity=0):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.array([vx, vy], dtype=np.float32)
        self.life = life
        self.max_life = life
        self.color = color
        self.radius = radius
        self.gravity = gravity

    def update(self):
        self.pos += self.vel
        self.vel[1] += self.gravity
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            current_radius = int(self.radius * (self.life / self.max_life))
            if current_radius > 0:
                # Use a temporary surface for transparency
                temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, self.color + (alpha,), (current_radius, current_radius), current_radius)
                surface.blit(temp_surf, (int(self.pos[0] - current_radius), int(self.pos[1] - current_radius)), special_flags=pygame.BLEND_RGBA_ADD)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate your ship through a hazardous nebula, avoiding asteroids and managing fuel. "
        "Flip gravity to your advantage and warp to reach the target portal."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to thrust. Press space to flip gravity and hold shift to warp."
    )
    auto_advance = True

    # Class-level variable for difficulty progression
    target_distance_base = 300.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.W, self.H = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 200, 0)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (200, 0, 0)
        self.COLOR_FUEL = (0, 255, 150)
        self.COLOR_FUEL_GLOW = (0, 200, 100)
        self.COLOR_TARGET = (255, 255, 255)
        self.COLOR_TARGET_GLOW = (200, 200, 255)
        self.COLOR_WARP_TRAIL = (180, 0, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_BAR = (0, 255, 150)
        self.COLOR_UI_BAR_BG = (50, 50, 50)

        # Physics
        self.GRAVITY_STRENGTH = 0.1
        self.THRUST_MAGNITUDE = 0.2
        self.DRAG = 0.99
        self.MAX_VEL = 7.0
        self.WARP_MULTIPLIER = 2.5
        self.FUEL_PER_THRUST = 0.05
        self.FUEL_PER_WARP = 0.5

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.fuel = 100.0
        self.gravity_dir = 1  # 1 for down, -1 for up
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.asteroids = []
        self.fuel_cells = []
        self.particles = []
        self.nebula_layers = []

        self.prev_space_held = False
        self.is_warping = False
        self.last_dist_to_target = 0.0
        self.last_min_dist_to_obstacle = float('inf')

        # This is not part of the official API, but good for internal checks
        # self.reset()
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array([self.W / 2, self.H / 2], dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.fuel = 100.0
        self.gravity_dir = 1
        self.prev_space_held = False
        self.is_warping = False

        self.particles.clear()
        
        # Spawn Target
        angle = self.np_random.uniform(0, 2 * math.pi)
        dist = self.__class__.target_distance_base + self.np_random.uniform(-50, 50)
        self.target_pos = self.player_pos + np.array([math.cos(angle) * dist, math.sin(angle) * dist])

        # Spawn Game Entities
        self.initial_obstacle_density = 0.01
        self._spawn_asteroids()
        self._spawn_fuel_cells()
        
        # Spawn Nebula Background
        self._spawn_nebula()

        self.last_dist_to_target = np.linalg.norm(self.player_pos - self.target_pos)
        self.last_min_dist_to_obstacle = self._get_min_dist_to_obstacle()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        terminated = False
        truncated = False
        
        self._handle_input(action)
        self._update_physics()
        self._update_particles()
        
        event_reward, termination_reason = self._update_game_state()
        reward += event_reward
        
        # Calculate continuous rewards
        dist_to_target = np.linalg.norm(self.player_pos - self.target_pos)
        reward += (self.last_dist_to_target - dist_to_target) / 10.0
        self.last_dist_to_target = dist_to_target

        min_dist_to_obstacle = self._get_min_dist_to_obstacle()
        if min_dist_to_obstacle < self.last_min_dist_to_obstacle:
            reward -= 0.01
        self.last_min_dist_to_obstacle = min_dist_to_obstacle

        self.steps += 1
        self.score += reward

        # Check for termination conditions
        if termination_reason:
            terminated = True
            self.game_over = True
            if termination_reason == "win":
                reward += 10.0
                self.__class__.target_distance_base += 10.0 # Difficulty progression
            elif termination_reason == "crash":
                reward -= 10.0
            elif termination_reason == "no_fuel":
                reward -= 5.0
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held_int, shift_held_int = action
        space_held = space_held_int == 1
        shift_held = shift_held_int == 1

        # Movement
        thrust_vec = np.zeros(2, dtype=np.float32)
        if self.fuel > 0:
            if movement == 1: thrust_vec[1] = -1 # Up
            elif movement == 2: thrust_vec[1] = 1  # Down
            elif movement == 3: thrust_vec[0] = -1 # Left
            elif movement == 4: thrust_vec[0] = 1  # Right
        
        if np.linalg.norm(thrust_vec) > 0:
            self.player_vel += thrust_vec * self.THRUST_MAGNITUDE
            self.fuel = max(0, self.fuel - self.FUEL_PER_THRUST)
            # Thrust particles
            for _ in range(2):
                p_vel = -thrust_vec * self.np_random.uniform(1, 3) + self.np_random.uniform(-0.5, 0.5, 2)
                self.particles.append(Particle(self.player_pos[0], self.player_pos[1], p_vel[0], p_vel[1], 20, (255, 150, 0), 3))

        # Minimal thrust at zero fuel
        if self.fuel <= 0 and np.linalg.norm(thrust_vec) > 0:
             self.player_vel += thrust_vec * self.THRUST_MAGNITUDE * 0.1

        # Gravity Flip (on press)
        if space_held and not self.prev_space_held:
            self.gravity_dir *= -1
            # Gravity flip particle effect
            for i in range(50):
                angle = i * (2 * math.pi / 50)
                speed = self.np_random.uniform(2, 4)
                p_vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
                color = (150, 100, 255) if self.gravity_dir == 1 else (255, 100, 150)
                self.particles.append(Particle(self.player_pos[0], self.player_pos[1], p_vel[0], p_vel[1], 25, color, 2))
        self.prev_space_held = space_held

        # Warp Drive (on hold)
        self.is_warping = shift_held and self.fuel > self.FUEL_PER_WARP

    def _update_physics(self):
        # Apply gravity
        self.player_vel[1] += self.GRAVITY_STRENGTH * self.gravity_dir

        # Apply warp
        if self.is_warping:
            self.fuel = max(0, self.fuel - self.FUEL_PER_WARP)
            current_max_vel = self.MAX_VEL * self.WARP_MULTIPLIER
            # Warp trail particles
            p_vel = -self.player_vel * 0.5 + self.np_random.uniform(-1, 1, 2)
            self.particles.append(Particle(self.player_pos[0], self.player_pos[1], p_vel[0], p_vel[1], 30, self.COLOR_WARP_TRAIL, 5))
        else:
            current_max_vel = self.MAX_VEL

        # Apply drag and clamp velocity
        self.player_vel *= self.DRAG
        speed = np.linalg.norm(self.player_vel)
        if speed > current_max_vel:
            self.player_vel = self.player_vel * (current_max_vel / speed)

        # Update position
        self.player_pos += self.player_vel

        # Toroidal world wrapping
        self.player_pos[0] %= self.W
        self.player_pos[1] %= self.H
        
        # Update nebula for parallax effect
        for layer in self.nebula_layers:
            for nebula_cloud in layer['clouds']:
                nebula_cloud['pos'] = (nebula_cloud['pos'] + nebula_cloud['vel'])
                if nebula_cloud['pos'][0] < -nebula_cloud['radius']: nebula_cloud['pos'][0] = self.W + nebula_cloud['radius']
                if nebula_cloud['pos'][0] > self.W + nebula_cloud['radius']: nebula_cloud['pos'][0] = -nebula_cloud['radius']
                if nebula_cloud['pos'][1] < -nebula_cloud['radius']: nebula_cloud['pos'][1] = self.H + nebula_cloud['radius']
                if nebula_cloud['pos'][1] > self.H + nebula_cloud['radius']: nebula_cloud['pos'][1] = -nebula_cloud['radius']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _update_game_state(self):
        event_reward = 0.0
        termination_reason = None

        # Player-Asteroid collision
        for asteroid in self.asteroids:
            if np.linalg.norm(self.player_pos - asteroid['pos']) < asteroid['radius'] + 5:
                termination_reason = "crash"
                self._create_explosion(self.player_pos, self.COLOR_PLAYER)
                break
        if termination_reason: return event_reward, termination_reason

        # Player-Fuel Cell collection
        for cell in self.fuel_cells[:]:
            if np.linalg.norm(self.player_pos - cell['pos']) < cell['radius'] + 5:
                self.fuel = min(100.0, self.fuel + 25.0)
                event_reward += 0.1
                self.fuel_cells.remove(cell)
                self._create_explosion(cell['pos'], self.COLOR_FUEL, 20)
                break

        # Player-Target collision
        if np.linalg.norm(self.player_pos - self.target_pos) < 20 + 5:
            termination_reason = "win"
            self._create_explosion(self.player_pos, self.COLOR_TARGET, 80)

        # Fuel check
        if self.fuel <= 0:
            termination_reason = "no_fuel"

        # Update obstacle density
        if self.steps > 0 and self.steps % 100 == 0:
            self.initial_obstacle_density += 0.001
            self._spawn_asteroids(1)

        return event_reward, termination_reason

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Nebula
        for layer in self.nebula_layers:
            alpha = layer['alpha']
            for cloud in layer['clouds']:
                temp_surf = pygame.Surface((cloud['radius']*2, cloud['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, cloud['color'] + (alpha,), (cloud['radius'], cloud['radius']), cloud['radius'])
                self.screen.blit(temp_surf, (int(cloud['pos'][0] - cloud['radius']), int(cloud['pos'][1] - cloud['radius'])), special_flags=pygame.BLEND_RGBA_ADD)

        # Render Target (draw all 9 versions for toroidal wrapping)
        for dx in [-self.W, 0, self.W]:
            for dy in [-self.H, 0, self.H]:
                pos = (int(self.target_pos[0] + dx), int(self.target_pos[1] + dy))
                self._draw_glowing_circle(self.screen, pos, 20, self.COLOR_TARGET, self.COLOR_TARGET_GLOW)

        # Render Fuel Cells
        for cell in self.fuel_cells:
            for dx in [-self.W, 0, self.W]:
                for dy in [-self.H, 0, self.H]:
                    pos = (int(cell['pos'][0] + dx), int(cell['pos'][1] + dy))
                    self._draw_glowing_circle(self.screen, pos, cell['radius'], self.COLOR_FUEL, self.COLOR_FUEL_GLOW)
        
        # Render Asteroids
        for asteroid in self.asteroids:
            for dx in [-self.W, 0, self.W]:
                for dy in [-self.H, 0, self.H]:
                    points = [(p[0] + dx, p[1] + dy) for p in asteroid['points']]
                    self._draw_glowing_polygon(self.screen, points, self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_GLOW)

        # Render Particles
        for p in self.particles:
            p.draw(self.screen)

        # Render Player
        info = self._get_info()
        termination_reason = info.get("termination_reason") if hasattr(self, 'game_over') and self.game_over else None
        if not (self.game_over and termination_reason == "crash"):
            self._draw_player()

    def _render_ui(self):
        # Fuel bar
        fuel_bar_width = 150
        fuel_bar_height = 15
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 10, fuel_bar_width, fuel_bar_height))
        current_fuel_width = int(fuel_bar_width * (self.fuel / 100.0))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (10, 10, max(0, current_fuel_width), fuel_bar_height))
        fuel_text = self.font_small.render("FUEL", True, self.COLOR_UI_TEXT)
        self.screen.blit(fuel_text, (15 + fuel_bar_width, 10))

        # Distance to target
        dist = np.linalg.norm(self.player_pos - self.target_pos)
        dist_text = self.font_small.render(f"DISTANCE: {int(dist)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(dist_text, (10, 30))

        # Gravity indicator
        grav_text = "GRAVITY: DOWN" if self.gravity_dir == 1 else "GRAVITY: UP"
        grav_color = (150, 100, 255) if self.gravity_dir == 1 else (255, 100, 150)
        grav_surf = self.font_small.render(grav_text, True, grav_color)
        self.screen.blit(grav_surf, (self.W - grav_surf.get_width() - 10, 10))

    def _get_info(self):
        info = {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "distance_to_target": np.linalg.norm(self.player_pos - self.target_pos),
        }
        return info

    # --- Helper Methods ---
    
    def _draw_player(self):
        pos = self.player_pos
        vel = self.player_vel
        angle = math.atan2(vel[1], vel[0]) if np.linalg.norm(vel) > 0.1 else -math.pi / 2

        # Glow
        glow_radius = 20 if self.is_warping else 15
        self._draw_glowing_circle(self.screen, (int(pos[0]), int(pos[1])), glow_radius, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, 0.5)

        # Ship body
        size = 10
        p1 = (pos[0] + math.cos(angle) * size, pos[1] + math.sin(angle) * size)
        p2 = (pos[0] + math.cos(angle + 2.2) * size * 0.8, pos[1] + math.sin(angle + 2.2) * size * 0.8)
        p3 = (pos[0] + math.cos(angle - 2.2) * size * 0.8, pos[1] + math.sin(angle - 2.2) * size * 0.8)
        points = [p1, p2, p3]
        
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)

    def _spawn_asteroids(self, num_to_spawn=None):
        if num_to_spawn is None:
            self.asteroids.clear()
            num_asteroids = int(self.initial_obstacle_density * (self.W * self.H) / (25*25))
        else:
            num_asteroids = num_to_spawn
            
        for _ in range(num_asteroids):
            while True:
                pos = self.np_random.uniform([0, 0], [self.W, self.H])
                if np.linalg.norm(pos - self.player_pos) > 100: # Don't spawn on player
                    break
            radius = self.np_random.uniform(15, 35)
            shape_points = self._create_asteroid_shape(pos, radius)
            self.asteroids.append({'pos': pos, 'radius': radius, 'points': shape_points})

    def _spawn_fuel_cells(self):
        self.fuel_cells.clear()
        num_fuel_cells = 5
        for _ in range(num_fuel_cells):
            while True:
                pos = self.np_random.uniform([0, 0], [self.W, self.H])
                if np.linalg.norm(pos - self.player_pos) > 100:
                    break
            self.fuel_cells.append({'pos': pos, 'radius': 8})
            
    def _spawn_nebula(self):
        self.nebula_layers.clear()
        for i in range(3): # 3 layers
            layer = {'clouds': [], 'alpha': 20 + i*15, 'vel_scale': 0.05 + i*0.05}
            colors = [(100, 20, 180), (20, 80, 150), (180, 40, 90)]
            for _ in range(10): # 10 clouds per layer
                layer['clouds'].append({
                    'pos': self.np_random.uniform([0,0], [self.W, self.H]),
                    'radius': self.np_random.uniform(100, 250),
                    'color': random.choice(colors),
                    'vel': self.np_random.uniform(-1, 1, 2) * layer['vel_scale']
                })
            self.nebula_layers.append(layer)

    def _create_asteroid_shape(self, center, avg_radius):
        points = []
        num_vertices = self.np_random.integers(7, 12)
        for i in range(num_vertices):
            angle = i * (2 * math.pi / num_vertices)
            radius = self.np_random.uniform(avg_radius * 0.7, avg_radius * 1.3)
            p = (center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius)
            points.append(p)
        return points

    def _create_explosion(self, pos, color, num_particles=50):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            p_vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(20, 40)
            radius = self.np_random.uniform(2, 5)
            self.particles.append(Particle(pos[0], pos[1], p_vel[0], p_vel[1], life, color, radius))

    def _get_min_dist_to_obstacle(self):
        if not self.asteroids:
            return float('inf')
        min_dist = float('inf')
        for asteroid in self.asteroids:
            # Check all 9 toroidal instances
            for dx in [-self.W, 0, self.W]:
                for dy in [-self.H, 0, self.H]:
                    ast_pos = asteroid['pos'] + np.array([dx, dy])
                    dist = np.linalg.norm(self.player_pos - ast_pos) - asteroid['radius']
                    if dist < min_dist:
                        min_dist = dist
        return min_dist

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_color, glow_scale=2.0):
        pos = (int(pos[0]), int(pos[1]))
        # Glow
        glow_radius = int(radius * glow_scale)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color + (80,), (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        # Core
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius), color)

    def _draw_glowing_polygon(self, surface, points, color, glow_color):
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(surface, int_points, glow_color)
        pygame.draw.polygon(surface, color, int_points)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # The original code had a validate_implementation method, but it's not
    # part of the standard gym API and can cause issues if called in __init__.
    # It's better to run validation separately, if needed.
    
    # To run validation:
    # env = GameEnv()
    # env.reset()
    # print("Action space:", env.action_space)
    # print("Observation space:", env.observation_space)
    # test_action = env.action_space.sample()
    # obs, reward, term, trunc, info = env.step(test_action)
    # print("Step successful.")
    # print("Validation complete.")

    # --- Manual Play ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    # This will fail if `SDL_VIDEODRIVER` is "dummy" unless you unset it.
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
    
    pygame.display.init()
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Gravity Nebula")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # Start with no-op

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        keys = pygame.key.get_pressed()
        
        # Movement
        mov = 0
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        # Actions
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()