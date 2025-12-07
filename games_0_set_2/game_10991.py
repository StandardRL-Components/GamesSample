import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:26:56.176550
# Source Brief: brief_00991.md
# Brief Index: 991
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Navigate a vibrant, gravity-shifting cosmos by matching spectral energies 
    to power portals and repair your damaged ship.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Navigate a damaged ship through a cosmic nebula. Collect resources for repairs and "
        "manipulate gravity by charging and activating colored portals to guide your vessel."
    )
    user_guide = (
        "Controls: Use ↑, ↓, and ← arrow keys to charge the red, green, and blue portal energies. "
        "Press → to activate the dominant portal and change the central planet's gravity."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CONSTANTS ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.REPAIR_GOAL = 100
        
        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_SHIP = (220, 255, 255)
        self.COLOR_SHIP_GLOW = (100, 200, 255)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.COLOR_OBSTACLE_GLOW = (150, 0, 0)
        self.COLOR_RESOURCE = (255, 255, 0)
        self.COLOR_GRAVITY_NEUTRAL = (100, 100, 100)
        self.COLOR_GRAVITY_RED = (255, 50, 50)     # Attraction
        self.COLOR_GRAVITY_GREEN = (50, 255, 50)   # Repulsion
        self.COLOR_GRAVITY_BLUE = (50, 50, 255)    # Orbital
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_GREY = (150, 150, 150)

        # Physics & Gameplay
        self.GRAVITY_STRENGTH = 0.05
        self.DRAG_FACTOR = 0.99
        self.ENERGY_INCREMENT = 32
        self.SHIP_SIZE = 12
        self.OBSTACLE_COUNT = 5
        self.RESOURCE_COUNT = 5
        self.PARTICLE_LIFESPAN = 40
        
        # --- GYM SPACES ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- STATE VARIABLES ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.ship_pos = pygame.Vector2(0, 0)
        self.ship_vel = pygame.Vector2(0, 0)
        self.repair_progress = 0
        
        self.planet_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.planet_radius = 50
        self.active_gravity_type = 'NEUTRAL' # 'RED', 'GREEN', 'BLUE'
        
        self.portal_energy = {'R': 0, 'G': 0, 'B': 0}
        
        self.resources = []
        self.obstacles = []
        self.particles = []
        self.nebula_stars = []

        self._generate_nebula()
        # self.reset() is called by the wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Ship state
        self.ship_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT * 0.2)
        self.ship_vel = pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5))
        self.repair_progress = 0
        
        # Environment state
        self.active_gravity_type = 'NEUTRAL'
        self.portal_energy = {'R': 0, 'G': 0, 'B': 0}
        
        # Particles
        self.particles.clear()
        
        # Generate obstacles and resources
        self.obstacles = self._generate_game_objects(self.OBSTACLE_COUNT, 20)
        self.resources = self._generate_game_objects(self.RESOURCE_COUNT, 15)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # 1. Handle player actions
        self._handle_actions(action)
        
        # 2. Update game physics
        self._update_ship_physics()

        # 3. Check for collisions and events
        reward += self._check_collisions()
        
        # 4. Update particles
        self._update_particles()
        
        # 5. Calculate rewards
        reward += self._calculate_reward()
        self.score += reward
        
        # 6. Check termination conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated:
            self.game_over = True
            if self.repair_progress >= self.REPAIR_GOAL:
                reward += 100  # Victory bonus
            else:
                reward -= 100 # Crashing penalty
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        """Maps MultiDiscrete action to game logic."""
        action_type = action[0]
        
        # Action 0: no-op
        # Action 1 (Up): Increase Red energy
        if action_type == 1:
            self.portal_energy['R'] = min(255, self.portal_energy['R'] + self.ENERGY_INCREMENT)
            # Sound: energy_charge_r.wav
        # Action 2 (Down): Increase Green energy
        elif action_type == 2:
            self.portal_energy['G'] = min(255, self.portal_energy['G'] + self.ENERGY_INCREMENT)
            # Sound: energy_charge_g.wav
        # Action 3 (Left): Increase Blue energy
        elif action_type == 3:
            self.portal_energy['B'] = min(255, self.portal_energy['B'] + self.ENERGY_INCREMENT)
            # Sound: energy_charge_b.wav
        # Action 4 (Right): Activate portal
        elif action_type == 4:
            if any(v > 0 for v in self.portal_energy.values()):
                dominant_energy = max(self.portal_energy, key=self.portal_energy.get)
                if self.portal_energy[dominant_energy] > 0:
                    if dominant_energy == 'R':
                        self.active_gravity_type = 'RED'
                        color = self.COLOR_GRAVITY_RED
                    elif dominant_energy == 'G':
                        self.active_gravity_type = 'GREEN'
                        color = self.COLOR_GRAVITY_GREEN
                    else: # 'B'
                        self.active_gravity_type = 'BLUE'
                        color = self.COLOR_GRAVITY_BLUE
                    
                    self._create_particle_burst(self.planet_pos, 50, color)
                    self.portal_energy = {'R': 0, 'G': 0, 'B': 0}
                    # Sound: portal_activate.wav

    def _update_ship_physics(self):
        """Updates ship velocity and position based on gravity."""
        to_planet = self.planet_pos - self.ship_pos
        distance = to_planet.length()
        
        if distance > 1:
            direction = to_planet.normalize()
            
            if self.active_gravity_type == 'RED': # Attraction
                force = direction * self.GRAVITY_STRENGTH
            elif self.active_gravity_type == 'GREEN': # Repulsion
                force = -direction * self.GRAVITY_STRENGTH * 1.5 # Stronger push
            elif self.active_gravity_type == 'BLUE': # Orbital
                # Perpendicular force for orbit
                force = pygame.Vector2(-direction.y, direction.x) * self.GRAVITY_STRENGTH
            else: # Neutral
                force = pygame.Vector2(0, 0)
                
            self.ship_vel += force
        
        # Drag
        self.ship_vel *= self.DRAG_FACTOR
        
        # Update position
        self.ship_pos += self.ship_vel
        
        # Ship trail particles
        if self.ship_vel.length() > 0.5:
            self._create_particle_burst(self.ship_pos, 1, self.COLOR_SHIP_GLOW, particle_speed=0.5, lifespan_mod=0.5)

    def _check_collisions(self):
        """Checks for collisions with resources, obstacles, and planet."""
        reward = 0
        ship_rect = pygame.Rect(self.ship_pos.x - self.SHIP_SIZE / 2, self.ship_pos.y - self.SHIP_SIZE / 2, self.SHIP_SIZE, self.SHIP_SIZE)
        
        # Resources
        for res in self.resources[:]:
            if ship_rect.colliderect(res):
                self.resources.remove(res)
                self.repair_progress = min(self.REPAIR_GOAL, self.repair_progress + 20)
                reward += 5
                self.score += 5 # Add to score immediately for display
                self._create_particle_burst(self.ship_pos, 20, self.COLOR_RESOURCE)
                # Sound: resource_collect.wav
                
        # Obstacles
        for obs in self.obstacles:
            if ship_rect.colliderect(obs):
                self.game_over = True
                self._create_particle_burst(self.ship_pos, 100, self.COLOR_OBSTACLE)
                # Sound: ship_explosion.wav
        
        # Planet
        if self.ship_pos.distance_to(self.planet_pos) < self.planet_radius + self.SHIP_SIZE / 2:
            self.game_over = True
            self._create_particle_burst(self.ship_pos, 100, self.COLOR_OBSTACLE)
            # Sound: ship_explosion.wav

        return reward

    def _calculate_reward(self):
        """Calculates non-event-based rewards."""
        reward = 0.01 # Small reward for surviving
        
        # Penalty for being near edges
        margin = 50
        if not (margin < self.ship_pos.x < self.WIDTH - margin and margin < self.ship_pos.y < self.HEIGHT - margin):
            reward -= 0.1
            
        return reward

    def _check_termination(self):
        """Checks for all termination conditions."""
        if self.game_over:
            return True
        if self.repair_progress >= self.REPAIR_GOAL:
            return True
        if not (0 < self.ship_pos.x < self.WIDTH and 0 < self.ship_pos.y < self.HEIGHT):
            return True
        return False

    def _get_observation(self):
        """Renders the game state to the screen and returns it as a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_planet_and_portal()
        self._render_entities()
        self._render_particles()
        self._render_ship()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "repair_progress": self.repair_progress,
            "ship_pos": (self.ship_pos.x, self.ship_pos.y),
            "ship_vel": (self.ship_vel.x, self.ship_vel.y),
            "active_gravity": self.active_gravity_type
        }

    # --- RENDERING HELPERS ---

    def _render_background(self):
        for star in self.nebula_stars:
            pygame.gfxdraw.filled_circle(self.screen, star['x'], star['y'], star['r'], star['c'])

    def _render_planet_and_portal(self):
        # Gravity color
        grav_color = self.COLOR_GRAVITY_NEUTRAL
        if self.active_gravity_type == 'RED': grav_color = self.COLOR_GRAVITY_RED
        elif self.active_gravity_type == 'GREEN': grav_color = self.COLOR_GRAVITY_GREEN
        elif self.active_gravity_type == 'BLUE': grav_color = self.COLOR_GRAVITY_BLUE

        # Planet Core Glow
        for i in range(self.planet_radius, 0, -5):
            alpha = 50 * (1 - i / self.planet_radius)
            pygame.gfxdraw.filled_circle(self.screen, int(self.planet_pos.x), int(self.planet_pos.y), i, (*grav_color, alpha))
        
        # Planet Body
        pygame.gfxdraw.filled_circle(self.screen, int(self.planet_pos.x), int(self.planet_pos.y), self.planet_radius, self.COLOR_BG)
        pygame.gfxdraw.aacircle(self.screen, int(self.planet_pos.x), int(self.planet_pos.y), self.planet_radius, grav_color)

        # Portal Energy Rings
        self._render_energy_arc(self.portal_energy['R'], self.COLOR_GRAVITY_RED, 0, 120, 1)
        self._render_energy_arc(self.portal_energy['G'], self.COLOR_GRAVITY_GREEN, 120, 240, 2)
        self._render_energy_arc(self.portal_energy['B'], self.COLOR_GRAVITY_BLUE, 240, 360, 3)

    def _render_energy_arc(self, energy, color, start_angle, end_angle, offset):
        if energy > 0:
            alpha = int(100 + 155 * (energy / 255))
            radius = self.planet_radius + 10 + offset * 5
            rect = pygame.Rect(self.planet_pos.x - radius, self.planet_pos.y - radius, radius * 2, radius * 2)
            pygame.draw.arc(self.screen, (*color, alpha), rect, math.radians(start_angle), math.radians(end_angle), 3)

    def _render_ship(self):
        # Ship shape (triangle)
        p1 = self.ship_pos + pygame.Vector2(0, -self.SHIP_SIZE).rotate(-math.degrees(math.atan2(self.ship_vel.y, self.ship_vel.x)) - 90)
        p2 = self.ship_pos + pygame.Vector2(-self.SHIP_SIZE / 2, self.SHIP_SIZE / 2).rotate(-math.degrees(math.atan2(self.ship_vel.y, self.ship_vel.x)) - 90)
        p3 = self.ship_pos + pygame.Vector2(self.SHIP_SIZE / 2, self.SHIP_SIZE / 2).rotate(-math.degrees(math.atan2(self.ship_vel.y, self.ship_vel.x)) - 90)
        points = [(int(p.x), int(p.y)) for p in [p1, p2, p3]]
        
        # Glow
        glow_size = int(self.SHIP_SIZE * (1.5 + self.repair_progress / 200.0))
        glow_alpha = 50 + int(100 * (self.repair_progress / self.REPAIR_GOAL))
        pygame.gfxdraw.filled_trigon(self.screen, *points[0], *points[1], *points[2], (*self.COLOR_SHIP_GLOW, glow_alpha))
        
        # Main body
        pygame.gfxdraw.filled_trigon(self.screen, *points[0], *points[1], *points[2], self.COLOR_BG) # Inner fill
        pygame.gfxdraw.aatrigon(self.screen, *points[0], *points[1], *points[2], self.COLOR_SHIP) # Outline
        
        # Repair progress indicator on ship
        if self.repair_progress > 0:
            progress_height = self.SHIP_SIZE * (self.repair_progress / self.REPAIR_GOAL)
            p_repair_2 = p2 + (p1 - p2) * (self.repair_progress / self.REPAIR_GOAL)
            p_repair_3 = p3 + (p1 - p3) * (self.repair_progress / self.REPAIR_GOAL)
            repair_points = [(int(p.x), int(p.y)) for p in [p1, p_repair_2, p_repair_3]]
            pygame.gfxdraw.filled_trigon(self.screen, *repair_points[0], *repair_points[1], *repair_points[2], self.COLOR_SHIP)

    def _render_entities(self):
        # Obstacles
        for obs in self.obstacles:
            pygame.gfxdraw.filled_circle(self.screen, obs.centerx, obs.centery, obs.width // 2 + 3, (*self.COLOR_OBSTACLE_GLOW, 100))
            pygame.gfxdraw.filled_circle(self.screen, obs.centerx, obs.centery, obs.width // 2, self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, obs.centerx, obs.centery, obs.width // 2, self.COLOR_WHITE)
            
        # Resources
        for res in self.resources:
            pygame.gfxdraw.filled_circle(self.screen, res.centerx, res.centery, res.width // 2 + 5, (*self.COLOR_RESOURCE, 100))
            pygame.gfxdraw.filled_circle(self.screen, res.centerx, res.centery, res.width // 2, self.COLOR_RESOURCE)

    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), size)

    def _render_ui(self):
        # Repair Progress Bar
        bar_width = 200
        bar_height = 15
        bar_x = (self.WIDTH - bar_width) // 2
        bar_y = 10
        
        fill_width = int(bar_width * (self.repair_progress / self.REPAIR_GOAL))
        pygame.draw.rect(self.screen, self.COLOR_GREY, (bar_x, bar_y, bar_width, bar_height), 2)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_SHIP_GLOW, (bar_x, bar_y, fill_width, bar_height))
        
        repair_text = self.font_ui.render(f"REPAIR: {int(self.repair_progress)}%", True, self.COLOR_WHITE)
        self.screen.blit(repair_text, (bar_x + bar_width + 10, bar_y))
        
        # Portal Energy UI
        energy_y = self.HEIGHT - 25
        texts = [f"R:{self.portal_energy['R']}", f"G:{self.portal_energy['G']}", f"B:{self.portal_energy['B']}"]
        colors = [self.COLOR_GRAVITY_RED, self.COLOR_GRAVITY_GREEN, self.COLOR_GRAVITY_BLUE]
        for i, (text, color) in enumerate(zip(texts, colors)):
            rendered_text = self.font_ui.render(text, True, color)
            self.screen.blit(rendered_text, (10 + i * 100, energy_y))

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "SHIP REPAIRED" if self.repair_progress >= self.REPAIR_GOAL else "SHIP DESTROYED"
        color = self.COLOR_SHIP_GLOW if self.repair_progress >= self.REPAIR_GOAL else self.COLOR_OBSTACLE
        
        text_surf = self.font_game_over.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    # --- UTILITY HELPERS ---

    def _generate_nebula(self):
        """Generates a static starfield for the background."""
        self.nebula_stars.clear()
        for _ in range(150):
            color = random.choice([
                (20, 10, 40), (40, 10, 20), (10, 20, 40), (30, 30, 50)
            ])
            self.nebula_stars.append({
                'x': random.randint(0, self.WIDTH),
                'y': random.randint(0, self.HEIGHT),
                'r': random.randint(1, 4),
                'c': color
            })

    def _generate_game_objects(self, count, size):
        """Generates non-overlapping resources or obstacles."""
        objects = []
        for _ in range(count):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(50, self.WIDTH - 50),
                    self.np_random.uniform(50, self.HEIGHT - 50)
                )
                # Avoid spawning inside planet or too close to center
                if pos.distance_to(self.planet_pos) < self.planet_radius + 50:
                    continue
                
                new_obj = pygame.Rect(pos.x - size/2, pos.y - size/2, size, size)
                
                # Check for overlap with existing objects
                if not any(new_obj.colliderect(obj) for obj in objects):
                    objects.append(new_obj)
                    break
        return objects

    def _create_particle_burst(self, position, count, color, particle_speed=2.0, lifespan_mod=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * particle_speed
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = int(self.np_random.uniform(20, self.PARTICLE_LIFESPAN) * lifespan_mod)
            self.particles.append({
                'pos': position.copy(),
                'vel': vel,
                'life': lifespan,
                'max_life': lifespan,
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set a real video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gravity Portal")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # Mapping keys to MultiDiscrete action components
    key_map = {
        pygame.K_UP: 1,      # Increase Red
        pygame.K_DOWN: 2,    # Increase Green
        pygame.K_LEFT: 3,    # Increase Blue
        pygame.K_RIGHT: 4,   # Activate Portal
    }
    
    action = [0, 0, 0] # [movement, space, shift]

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in key_map:
                    action[0] = key_map[event.key]
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
            if event.type == pygame.KEYUP:
                if event.key in key_map and action[0] == key_map[event.key]:
                    action[0] = 0 # No-op

        # For manual play, we only trigger actions on keydown, then reset to no-op
        # For an agent, it would hold the action.
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Reset movement action after one step to simulate a key press
        action[0] = 0
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Keep displaying the final frame until quit
            final_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        done = True
                screen.blit(final_surf, (0,0))
                pygame.display.flip()
                clock.tick(env.metadata["render_fps"])

        # Render the observation from the environment
        if not done:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(env.metadata["render_fps"])

    env.close()