import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:30:17.036134
# Source Brief: brief_00458.md
# Brief Index: 458
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent controls three gravity wells to capture asteroids.
    
    The goal is to attract asteroids to a central capture zone using three movable
    gravity wells (Red, Green, Blue). The agent selects which well to move using
    the 'space' and 'shift' actions, and directs its movement with the 'movement' action.
    Points are scored for each captured asteroid, proportional to its mass. The game
    ends if 10 asteroids are missed or the time limit is reached.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Use three movable gravity wells to pull asteroids into a central capture zone. "
        "Score points for each captured asteroid before time runs out or too many are missed."
    )
    user_guide = (
        "Controls: Use ↑↓←→ to move the selected gravity well. Hold 'space' to control the Green "
        "well or 'shift' to control the Blue well. Releasing both defaults to the Red well."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.MAX_MISSES = 10
        self.WELL_SPEED = 8
        self.WELL_MASS = 8000
        self.EDGE_GRAVITY_MASS = 4000
        self.CAPTURE_ZONE_RADIUS = 25
        
        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_STAR = (200, 200, 220)
        self.COLOR_ASTEROID = (160, 160, 160)
        self.COLOR_CAPTURE_ZONE = (255, 255, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.WELL_COLORS = [(255, 80, 80), (80, 255, 80), (80, 80, 255)] # Red, Green, Blue
        self.WELL_NAMES = ["RED", "GREEN", "BLUE"]

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.missed_asteroids = 0
        self.asteroids = []
        self.wells = []
        self.particles = []
        self.stars = []
        self.spawn_timer = 0.0
        self.spawn_period = 0.0
        self.asteroid_speed_multiplier = 1.0
        self.selected_well_index = 0
        
        # Define static edge gravity sources to keep asteroids on screen
        self.edge_gravity_sources = [
            pygame.Vector2(self.WIDTH / 2, -self.HEIGHT),
            pygame.Vector2(self.WIDTH / 2, self.HEIGHT * 2),
            pygame.Vector2(-self.WIDTH, self.HEIGHT / 2),
            pygame.Vector2(self.WIDTH * 2, self.HEIGHT / 2),
        ]

        # Call reset to initialize the state for the first time
        # self.reset() # No need to call reset in init, handled by gym.make

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.missed_asteroids = 0
        
        # Initialize wells
        self.wells = [
            {'pos': pygame.Vector2(self.WIDTH * 0.25, self.HEIGHT / 2), 'color': self.WELL_COLORS[0], 'mass': self.WELL_MASS, 'radius': 15},
            {'pos': pygame.Vector2(self.WIDTH * 0.50, self.HEIGHT * 0.25), 'color': self.WELL_COLORS[1], 'mass': self.WELL_MASS, 'radius': 15},
            {'pos': pygame.Vector2(self.WIDTH * 0.75, self.HEIGHT / 2), 'color': self.WELL_COLORS[2], 'mass': self.WELL_MASS, 'radius': 15},
        ]
        
        self.asteroids.clear()
        self.particles.clear()
        
        # Generate a static starfield for the background
        if not self.stars:
            self.stars = [
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.uniform(0.5, 1.5))
                for _ in range(150)
            ]
        
        # Reset difficulty scaling
        self.spawn_period = 2.0  # Time in seconds between spawns
        self.spawn_timer = self.spawn_period
        self.asteroid_speed_multiplier = 1.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- 1. Handle Player Action ---
        if space_held and not shift_held:       # Space only -> Green Well
            self.selected_well_index = 1
        elif not space_held and shift_held:     # Shift only -> Blue Well
            self.selected_well_index = 2
        else:                                   # Neither or Both -> Red Well (default)
            self.selected_well_index = 0
        
        self._move_well(self.wells[self.selected_well_index], movement)
        
        # --- 2. Update Game Logic ---
        self._update_spawner()
        capture_reward = self._update_asteroids_and_get_reward()
        self._update_particles()
        
        self.steps += 1
        
        # --- 3. Calculate Reward and Termination ---
        terminated = self._check_termination()
        reward = capture_reward
        
        if terminated:
            if self.missed_asteroids >= self.MAX_MISSES:
                reward = -100.0 # Terminal penalty for losing
            elif self.steps >= self.MAX_STEPS:
                reward += 10.0 * (self.score / 100.0) # Bonus for surviving
        
        truncated = False # This environment does not truncate
        
        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _move_well(self, well, movement):
        """Moves a gravity well based on the movement action."""
        if movement == 1: # Up
            well['pos'].y -= self.WELL_SPEED
        elif movement == 2: # Down
            well['pos'].y += self.WELL_SPEED
        elif movement == 3: # Left
            well['pos'].x -= self.WELL_SPEED
        elif movement == 4: # Right
            well['pos'].x += self.WELL_SPEED
        
        # Clamp position to screen bounds
        well['pos'].x = max(well['radius'], min(self.WIDTH - well['radius'], well['pos'].x))
        well['pos'].y = max(well['radius'], min(self.HEIGHT - well['radius'], well['pos'].y))

    def _update_spawner(self):
        """Handles asteroid spawning and difficulty scaling."""
        # Update timer (assuming 30 FPS)
        self.spawn_timer += 1 / self.metadata["render_fps"]
        if self.spawn_timer >= self.spawn_period:
            self.spawn_timer = 0
            self._spawn_asteroid()

        # Increase difficulty over time
        # Spawn rate increases by 0.001 per second (period decreases)
        self.spawn_period = max(0.5, self.spawn_period - (0.001 / self.metadata["render_fps"]))
        # Asteroid speed increases every 500 steps
        if self.steps > 0 and self.steps % 500 == 0:
            self.asteroid_speed_multiplier = min(2.0, self.asteroid_speed_multiplier + 0.01)

    def _spawn_asteroid(self):
        """Creates a new asteroid at a random screen edge."""
        edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
        radius = self.np_random.uniform(4, 12)
        mass = radius ** 2
        
        if edge == 'top':
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -radius)
        elif edge == 'bottom':
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + radius)
        elif edge == 'left':
            pos = pygame.Vector2(-radius, self.np_random.uniform(0, self.HEIGHT))
        else: # right
            pos = pygame.Vector2(self.WIDTH + radius, self.np_random.uniform(0, self.HEIGHT))
            
        # Aim towards the center with some randomness
        target = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        direction = (target - pos).normalize()
        direction.rotate_ip(self.np_random.uniform(-15, 15))
        
        speed = self.np_random.uniform(0.8, 1.2) * self.asteroid_speed_multiplier
        vel = direction * speed
        
        self.asteroids.append({'pos': pos, 'vel': vel, 'radius': radius, 'mass': mass})

    def _update_asteroids_and_get_reward(self):
        """Updates physics for all asteroids and handles captures/misses."""
        capture_reward = 0.0
        center = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        for asteroid in self.asteroids[:]:
            # Calculate total gravitational force
            total_force = pygame.Vector2(0, 0)
            sources = self.wells + [{'pos': s, 'mass': self.EDGE_GRAVITY_MASS} for s in self.edge_gravity_sources]
            
            for source in sources:
                vec_to_source = source['pos'] - asteroid['pos']
                dist_sq = vec_to_source.length_squared()
                if dist_sq > 1: # Avoid division by zero and extreme forces
                    force_magnitude = source['mass'] / dist_sq
                    total_force += vec_to_source.normalize() * force_magnitude
            
            # Update velocity and position (F=ma, so a=F/m)
            acceleration = total_force / asteroid['mass']
            asteroid['vel'] += acceleration
            # Cap speed to prevent runaways
            if asteroid['vel'].length_squared() > 25: # Max speed of 5
                asteroid['vel'].scale_to_length(5)
            asteroid['pos'] += asteroid['vel']
            
            # Check for capture
            if asteroid['pos'].distance_to(center) < self.CAPTURE_ZONE_RADIUS + asteroid['radius']:
                self.score += asteroid['mass']
                capture_reward += 0.1 * asteroid['mass']
                self._create_capture_particles(asteroid['pos'], asteroid['mass'])
                self.asteroids.remove(asteroid)
                continue

            # Check for miss (out of bounds)
            if not (-50 < asteroid['pos'].x < self.WIDTH + 50 and -50 < asteroid['pos'].y < self.HEIGHT + 50):
                self.missed_asteroids += 1
                self.asteroids.remove(asteroid)
        
        return capture_reward

    def _update_particles(self):
        """Updates position and lifetime of all particles."""
        for particle in self.particles[:]:
            particle['pos'] += particle['vel']
            particle['life'] -= 1
            if particle['life'] <= 0:
                self.particles.remove(particle)

    def _create_capture_particles(self, position, mass):
        """Spawns a burst of particles upon asteroid capture."""
        num_particles = int(math.sqrt(mass)) * 2
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            life = self.np_random.integers(15, 31)
            self.particles.append({'pos': position.copy(), 'vel': vel, 'life': life})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_capture_zone()
        self._render_wells()
        self._render_asteroids()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "missed": self.missed_asteroids}
    
    def _check_termination(self):
        return self.missed_asteroids >= self.MAX_MISSES or self.steps >= self.MAX_STEPS

    # --- Rendering Methods ---
    def _render_background(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size)
    
    def _render_capture_zone(self):
        center = (self.WIDTH // 2, self.HEIGHT // 2)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.CAPTURE_ZONE_RADIUS, self.COLOR_CAPTURE_ZONE)
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.CAPTURE_ZONE_RADIUS, (*self.COLOR_CAPTURE_ZONE, 30))

    def _render_wells(self):
        for i, well in enumerate(self.wells):
            pos = (int(well['pos'].x), int(well['pos'].y))
            # Glow effect
            for j in range(well['radius'], 0, -2):
                alpha = 80 * (1 - j / well['radius'])
                color = (*well['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], j + 5, color)
            # Core circle
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], well['radius'], well['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], well['radius'], well['color'])
            # Selection indicator
            if i == self.selected_well_index:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], well['radius'] + 5, well['color'])

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pos = (int(asteroid['pos'].x), int(asteroid['pos'].y))
            radius = int(asteroid['radius'])
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)

    def _render_particles(self):
        for particle in self.particles:
            pos = (int(particle['pos'].x), int(particle['pos'].y))
            size = max(1, int(particle['life'] / 10))
            pygame.draw.circle(self.screen, self.COLOR_CAPTURE_ZONE, pos, size)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Misses
        miss_text = self.font_large.render(f"MISSES: {self.missed_asteroids}/{self.MAX_MISSES}", True, self.COLOR_TEXT)
        miss_rect = miss_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(miss_text, miss_rect)
        
        # Selected Well Indicator
        well_name = self.WELL_NAMES[self.selected_well_index]
        well_color = self.WELL_COLORS[self.selected_well_index]
        control_text = self.font_small.render(f"CONTROL: {well_name}", True, well_color)
        self.screen.blit(control_text, (10, 45))

    def close(self):
        pygame.font.quit()
        pygame.quit()

# --- Example Usage ---
if __name__ == '__main__':
    # This block will not run in the test environment, but is useful for local testing.
    # It requires a display. To run, comment out the `os.environ` line at the top.
    # For example: # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    # Check if we are in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Running in headless mode. Skipping interactive example.")
    else:
        env = GameEnv(render_mode="rgb_array")
        
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Gravity Well")
        
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            # --- Human Input Mapping ---
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- Rendering ---
            # The observation is already the rendered frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # --- Event Handling & Clock ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            env.clock.tick(env.metadata["render_fps"])

        print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
        env.close()