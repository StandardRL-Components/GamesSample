import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:54:15.205302
# Source Brief: brief_00087.md
# Brief Index: 87
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls three orbiting gravity wells
    to funnel asteroids into a central collector. The goal is to fill the
    collector to 100% capacity within a 60-second time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control three orbiting gravity wells to funnel asteroids into a central collector. "
        "Fill the collector to 100% capacity before time runs out."
    )
    user_guide = (
        "Controls: Use ↑, ↓, and ← arrow keys to toggle the corresponding gravity wells on and off."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.NUM_ASTEROIDS = 20
        self.COLLECTOR_CAPACITY = 100.0

        # --- Colors (Futuristic, High Contrast) ---
        self.COLOR_BG = (15, 18, 23)
        self.COLOR_STAR = (100, 100, 120)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_TIMER_NORMAL = (180, 180, 200)
        self.COLOR_TIMER_WARN = (255, 100, 100)
        self.COLOR_COLLECTOR_RING = (60, 80, 120)
        self.COLOR_COLLECTOR_FILL = (0, 255, 150)
        self.COLOR_SATELLITE_INACTIVE = (70, 70, 90)
        self.COLOR_SATELLITE_ACTIVE = (0, 150, 255)
        self.COLOR_SATELLITE_CHAIN = (255, 255, 0)
        self.COLOR_WELL_RADIUS = (*self.COLOR_SATELLITE_ACTIVE, 30)
        self.COLOR_ASTEROID = (255, 255, 255)
        self.COLOR_PARTICLE = (255, 200, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 18)
            self.font_title = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 28)
            self.font_small = pygame.font.Font(None, 22)
            self.font_title = pygame.font.Font(None, 52)
        
        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = 0.0
        self.collector_fill = 0.0
        self.last_fill_reported = 0.0
        self.asteroids = []
        self.satellites = []
        self.particles = []
        self.stars = []
        self.last_action = self.action_space.sample() * 0 # Initialize with no-op

        self.collector_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.collector_radius = 30

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = float(self.TIME_LIMIT_SECONDS)
        self.collector_fill = 0.0
        self.last_fill_reported = 0.0
        
        self.asteroids = []
        self.particles = []
        self.last_action = self.action_space.sample() * 0

        # --- Initialize Satellites ---
        self.satellites = []
        orbit_radius = 150
        for i in range(3):
            angle = (2 * math.pi / 3) * i + (math.pi / 2)
            pos = self.collector_pos + np.array([math.cos(angle), math.sin(angle)]) * orbit_radius
            self.satellites.append({
                "pos": pos,
                "is_active": False,
                "chain_reaction_timer": 0.0,
                "radius": 100
            })

        # --- Initialize Asteroids ---
        for _ in range(self.NUM_ASTEROIDS):
            while True:
                pos = np.array([
                    self.np_random.uniform(20, self.WIDTH - 20),
                    self.np_random.uniform(20, self.HEIGHT - 20)
                ])
                if np.linalg.norm(pos - self.collector_pos) > self.collector_radius * 2:
                    break
            self.asteroids.append({
                "pos": pos,
                "vel": self.np_random.uniform(-1, 1, size=2) * 0.5,
                "radius": self.np_random.uniform(2, 4)
            })
            
        # --- Initialize Starfield ---
        self.stars = []
        for _ in range(150):
            self.stars.append((
                random.randint(0, self.WIDTH),
                random.randint(0, self.HEIGHT),
                random.uniform(0.5, 1.5)
            ))

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        dt = 1.0 / self.FPS

        # --- Handle Action (Toggle on press, not hold) ---
        movement, _, _ = action
        last_movement, _, _ = self.last_action
        
        if movement in [1, 2, 3] and movement != last_movement:
            well_index = movement - 1
            self.satellites[well_index]["is_active"] = not self.satellites[well_index]["is_active"]
            # Visual feedback is handled by rendering
            # Sound effect placeholder: // sfx_toggle_on or // sfx_toggle_off
        self.last_action = action

        # --- Update Game State ---
        self.time_remaining = max(0, self.time_remaining - dt)
        
        # Update satellite chain reaction timers
        for sat in self.satellites:
            sat["chain_reaction_timer"] = max(0, sat["chain_reaction_timer"] - dt)

        # Update asteroid physics
        for asteroid in self.asteroids:
            total_force = np.zeros(2, dtype=float)
            for sat in self.satellites:
                if sat["is_active"]:
                    vec_to_sat = sat["pos"] - asteroid["pos"]
                    dist = np.linalg.norm(vec_to_sat)
                    if 0 < dist < sat["radius"]:
                        force_magnitude = 1500 / max(dist, 10) # Inverse square-like, with min dist
                        if sat["chain_reaction_timer"] > 0:
                            force_magnitude *= 2.0 # Chain reaction doubles pull strength
                        
                        force_dir = vec_to_sat / dist
                        total_force += force_dir * force_magnitude
            
            asteroid["vel"] += total_force * (dt**2)
            asteroid["vel"] *= 0.98 # Damping for game feel
            asteroid["pos"] += asteroid["vel"]

            # Boundary collision (bounce)
            if not (0 < asteroid["pos"][0] < self.WIDTH): asteroid["vel"][0] *= -0.8
            if not (0 < asteroid["pos"][1] < self.HEIGHT): asteroid["vel"][1] *= -0.8
            asteroid["pos"][0] = np.clip(asteroid["pos"][0], 0, self.WIDTH)
            asteroid["pos"][1] = np.clip(asteroid["pos"][1], 0, self.HEIGHT)

        # --- Collision Detection ---
        # Asteroid-Collector
        asteroids_to_remove = []
        for i, asteroid in enumerate(self.asteroids):
            if np.linalg.norm(asteroid["pos"] - self.collector_pos) < self.collector_radius:
                self.collector_fill += self.COLLECTOR_CAPACITY / self.NUM_ASTEROIDS
                asteroids_to_remove.append(i)
                # Sound effect placeholder: // sfx_collect_asteroid

        if asteroids_to_remove:
            self.asteroids = [a for i, a in enumerate(self.asteroids) if i not in asteroids_to_remove]

        # Asteroid-Asteroid (Chain Reaction Trigger)
        for i in range(len(self.asteroids)):
            for j in range(i + 1, len(self.asteroids)):
                a1 = self.asteroids[i]
                a2 = self.asteroids[j]
                dist = np.linalg.norm(a1["pos"] - a2["pos"])
                if dist < a1["radius"] + a2["radius"]:
                    # Trigger chain reaction
                    reward += 1.0
                    collision_point = (a1["pos"] + a2["pos"]) / 2
                    self._create_particles(collision_point, 10)
                    # Sound effect placeholder: // sfx_chain_reaction

                    for sat in self.satellites:
                        if sat["is_active"] and np.linalg.norm(collision_point - sat["pos"]) < sat["radius"]:
                            sat["chain_reaction_timer"] = 2.0 # 2-second boost

                    # Simple bounce for game feel
                    midpoint = (a1['pos'] + a2['pos']) / 2
                    normal = (a1['pos'] - a2['pos'])
                    if np.linalg.norm(normal) > 0:
                        normal /= np.linalg.norm(normal)
                        a1['pos'] = midpoint + normal * (a1['radius'] + a2['radius']) / 2
                        a2['pos'] = midpoint - normal * (a1['radius'] + a2['radius']) / 2
                        v_rel = a1['vel'] - a2['vel']
                        v_normal = np.dot(v_rel, normal)
                        if v_normal < 0:
                            a1['vel'] -= v_normal * normal
                            a2['vel'] += v_normal * normal
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= dt
            p['vel'] *= 0.95

        # --- Calculate Reward ---
        fill_increase = self.collector_fill - self.last_fill_reported
        if fill_increase > 0:
            reward += (fill_increase / 0.1) * 0.1 # Per brief: +0.1 per 0.1% increase
            self.last_fill_reported = self.collector_fill

        # --- Check Termination ---
        terminated = False
        self.collector_fill = min(self.collector_fill, self.COLLECTOR_CAPACITY)
        
        if self.collector_fill >= self.COLLECTOR_CAPACITY:
            terminated = True
            self.game_over = True
            reward += 100.0 # Win bonus
        elif self.time_remaining <= 0:
            terminated = True
            self.game_over = True
            reward -= 100.0 # Loss penalty

        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, truncated can also imply termination
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

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
            "collector_fill": self.collector_fill,
            "time_remaining": self.time_remaining,
        }

    def _render_game(self):
        # Draw stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))

        # Draw satellites and their influence radii
        for sat in self.satellites:
            pos_int = sat["pos"].astype(int)
            if sat["is_active"]:
                self._draw_glowing_circle(self.screen, self.COLOR_WELL_RADIUS, pos_int, int(sat["radius"]), 5)
                color = self.COLOR_SATELLITE_CHAIN if sat["chain_reaction_timer"] > 0 else self.COLOR_SATELLITE_ACTIVE
                if sat["chain_reaction_timer"] > 0:
                    pulse_radius = 12 + 3 * math.sin(self.steps * 0.5)
                    self._draw_glowing_circle(self.screen, (*self.COLOR_SATELLITE_CHAIN, 100), pos_int, int(pulse_radius), 3)
            else:
                color = self.COLOR_SATELLITE_INACTIVE
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 8, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 8, color)

        # Draw collector
        collector_pos_int = self.collector_pos.astype(int)
        fill_radius = int(self.collector_radius * math.sqrt(self.collector_fill / self.COLLECTOR_CAPACITY))
        pygame.gfxdraw.filled_circle(self.screen, collector_pos_int[0], collector_pos_int[1], self.collector_radius, self.COLOR_COLLECTOR_RING)
        if fill_radius > 0:
            self._draw_glowing_circle(self.screen, (*self.COLOR_COLLECTOR_FILL, 150), collector_pos_int, fill_radius, 5)
            pygame.gfxdraw.filled_circle(self.screen, collector_pos_int[0], collector_pos_int[1], fill_radius, self.COLOR_COLLECTOR_FILL)
        pygame.gfxdraw.aacircle(self.screen, collector_pos_int[0], collector_pos_int[1], self.collector_radius, self.COLOR_TEXT)

        # Draw asteroids
        for asteroid in self.asteroids:
            pos_int = asteroid["pos"].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(asteroid["radius"]), self.COLOR_ASTEROID)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(asteroid["radius"]), self.COLOR_ASTEROID)
            
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*self.COLOR_PARTICLE, alpha)
            # Use a surface with SRCALPHA for proper alpha blending
            particle_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(particle_surf, (p['pos'] - p['size']).astype(int))


    def _render_ui(self):
        # Render Collector Fill %
        fill_text = f"{self.collector_fill:.1f}%"
        text_surface = self.font_main.render(fill_text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 50))
        self.screen.blit(text_surface, text_rect)

        # Render Timer
        timer_color = self.COLOR_TIMER_WARN if self.time_remaining < 10 else self.COLOR_TIMER_NORMAL
        timer_text = f"TIME: {self.time_remaining:.2f}"
        text_surface = self.font_main.render(timer_text, True, timer_color)
        text_rect = text_surface.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(text_surface, text_rect)
        
        # Render Satellite labels
        labels = ["UP", "DOWN", "LEFT"]
        for i, sat in enumerate(self.satellites):
            label_text = f"{labels[i]}"
            text_surface = self.font_small.render(label_text, True, self.COLOR_SATELLITE_ACTIVE if sat["is_active"] else self.COLOR_SATELLITE_INACTIVE)
            text_rect = text_surface.get_rect(center=(sat["pos"][0], sat["pos"][1] + 20))
            self.screen.blit(text_surface, text_rect)

        # Render Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.collector_fill >= self.COLLECTOR_CAPACITY else "TIME UP"
            text_surface = self.font_title.render(msg, True, self.COLOR_COLLECTOR_FILL if msg == "VICTORY!" else self.COLOR_TIMER_WARN)
            text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _draw_glowing_circle(self, surface, color, center, radius, glow_size):
        """Helper to draw a circle with a soft glow effect."""
        r, g, b, a = color
        for i in range(glow_size):
            alpha = a * (1.0 - (i / glow_size)**2)
            pygame.gfxdraw.aacircle(
                surface,
                center[0],
                center[1],
                radius + i,
                (r, g, b, int(alpha))
            )

    def _create_particles(self, pos, count):
        """Helper to spawn particles for effects."""
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.uniform(0.5, 1.0)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": life,
                "max_life": life,
                "size": self.np_random.uniform(1, 3)
            })

    def close(self):
        pygame.quit()
        super().close()

# Example of how to run the environment
if __name__ == "__main__":
    # This block will not run in the testing environment, but is useful for development.
    # To run, you'll need to `pip install pygame`.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Gravity Well Synchronizer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    # Action mapping for human player
    # 0=none, 1=up, 2=down, 3=left, 4=right
    key_to_movement = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4, # Note: K_RIGHT is mapped but has no effect in game logic
    }
    
    last_pressed_key = None
    
    while running:
        movement_action = 0 # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_movement:
                    # Toggle logic requires a press, not hold.
                    # This logic differs slightly from the agent's `last_action`
                    # but provides a better human play experience.
                    if event.key != last_pressed_key:
                         movement_action = key_to_movement[event.key]
                    last_pressed_key = event.key
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0.0
                if event.key == pygame.K_ESCAPE:
                    running = False
            if event.type == pygame.KEYUP:
                if event.key == last_pressed_key:
                    last_pressed_key = None

        # Construct the MultiDiscrete action
        # For human play, we only care about the movement part
        action = [movement_action, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart.")
        
        clock.tick(env.FPS)
        
    env.close()