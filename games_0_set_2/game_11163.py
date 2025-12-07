import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:43:48.203932
# Source Brief: brief_01163.md
# Brief Index: 1163
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent controls gravity wells to collect asteroids.

    **Core Concept:**
    Manipulate dual gravity wells to trigger asteroid chain reactions and
    collect them in a central collector for points within a time limit.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - actions[0]: Gravity Well Activation (0=none, 1=TL, 2=TR, 3=BL, 4=BR)
    - actions[1]: Space button (unused)
    - actions[2]: Shift button (unused)

    **Observation Space:**
    A 640x400x3 RGB image of the game state.

    **Rewards:**
    - +0.1 for each asteroid pulled closer to an active well.
    - +0.5 for each asteroid merger.
    - +1.0 for each asteroid collected.
    - +100 for winning (reaching score goal).
    - -100 for losing (time out).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Activate gravity wells to pull in and merge asteroids, guiding them into a central collector to score points before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys to activate the corresponding gravity well (↑ for Top-Left, → for Top-Right, ← for Bottom-Left, ↓ for Bottom-Right). Release keys to deactivate."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds
        self.WIN_SCORE = 1000
        self.MAX_ASTEROIDS = 100

        # --- Colors (Neon on Dark) ---
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_WELL_INACTIVE = (50, 50, 80)
        self.COLOR_WELL_ACTIVE = (0, 255, 255) # Cyan
        self.COLOR_COLLECTOR = (255, 255, 0) # Yellow
        self.COLOR_PARTICLE = (255, 100, 255) # Magenta
        self.ASTEROID_COLORS = [
            (0, 150, 255),  # Blue (Small)
            (0, 255, 150),  # Green
            (255, 255, 0),  # Yellow
            (255, 50, 50)   # Red (Large)
        ]

        # --- Game Object Specs ---
        self.COLLECTOR_POS = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.COLLECTOR_RADIUS = 20
        self.GRAVITY_PULL = 3000
        self.MIN_ASTEROID_MASS = 5
        self.MAX_ASTEROID_MASS = 40

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.Font(None, 48)
            self.font_small = pygame.font.Font(None, 24)
        except pygame.error:
            # Fallback if default font is not found (e.g., in minimal environments)
            self.font_large = pygame.font.SysFont("sans-serif", 48)
            self.font_small = pygame.font.SysFont("sans-serif", 24)

        # --- Well Positions ---
        self.well_positions = [
            pygame.Vector2(self.WIDTH * 0.25, self.HEIGHT * 0.25), # 0: Top-Left
            pygame.Vector2(self.WIDTH * 0.75, self.HEIGHT * 0.25), # 1: Top-Right
            pygame.Vector2(self.WIDTH * 0.25, self.HEIGHT * 0.75), # 2: Bottom-Left
            pygame.Vector2(self.WIDTH * 0.75, self.HEIGHT * 0.75)  # 3: Bottom-Right
        ]

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.asteroids = []
        self.particles = []
        self.active_well_index = -1
        self.current_spawn_rate = 0
        self.spawn_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_STEPS

        self.asteroids = []
        self.particles = []

        self.active_well_index = -1

        self.initial_spawn_rate = 2 * self.FPS
        self.current_spawn_rate = self.initial_spawn_rate
        self.spawn_timer = self.initial_spawn_rate

        for _ in range(5):
            self._spawn_asteroid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0

        # 1. Process Action
        movement = action[0]
        # Action mapping from brief: 1=TL, 2=TR, 3=BL, 4=BR
        if movement in [1, 2, 3, 4]:
            self.active_well_index = movement - 1
        else: # movement == 0 (no-op)
            self.active_well_index = -1

        # 2. Update Spawner
        self._update_spawner()

        # 3. Update Asteroids
        reward += self._update_asteroids()

        # 4. Handle Collisions
        reward += self._handle_collisions()

        # 5. Update Particles
        self._update_particles()

        # 6. Update Game State
        self.steps += 1
        self.time_remaining -= 1

        # 7. Check Termination
        terminated = self.score >= self.WIN_SCORE or self.time_remaining <= 0
        truncated = False # No truncation condition other than time limit, which is termination.
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100.0  # Win reward
            else:
                reward -= 100.0  # Lose penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Helper Methods ---

    def _spawn_asteroid(self):
        if len(self.asteroids) >= self.MAX_ASTEROIDS:
            return

        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -20)
        elif edge == 1: # Right
            pos = pygame.Vector2(self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT))
        elif edge == 2: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20)
        else: # Left
            pos = pygame.Vector2(-20, self.np_random.uniform(0, self.HEIGHT))

        # Aim towards the center with some variance
        direction = self.COLLECTOR_POS - pos
        direction.rotate_ip(self.np_random.uniform(-30, 30))
        vel = direction.normalize() * self.np_random.uniform(0.5, 1.5)

        mass = self.np_random.uniform(self.MIN_ASTEROID_MASS, self.MIN_ASTEROID_MASS + 10)

        self.asteroids.append({
            'pos': pos, 'vel': vel, 'mass': mass,
            'radius': math.sqrt(mass), 'trail': deque(maxlen=10)
        })

    def _update_spawner(self):
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self._spawn_asteroid()
            # Spawn rate increases by 0.01 per second (decrease interval by 0.01*FPS frames)
            self.current_spawn_rate -= 0.01 * self.FPS
            self.current_spawn_rate = max(0.5 * self.FPS, self.current_spawn_rate)
            self.spawn_timer = int(self.current_spawn_rate)

    def _update_asteroids(self):
        reward = 0.0
        active_well_pos = self.well_positions[self.active_well_index] if self.active_well_index != -1 else None

        for a in self.asteroids:
            # Apply gravity from active well
            if active_well_pos:
                dist_vec = active_well_pos - a['pos']
                dist_sq = dist_vec.length_squared()

                if dist_sq > 1:
                    dist_before = math.sqrt(dist_sq)
                    force_mag = self.GRAVITY_PULL / dist_sq
                    a['vel'] += dist_vec.normalize() * force_mag

                    # Reward for being pulled closer
                    dist_after = (active_well_pos - (a['pos'] + a['vel'])).length()
                    if dist_after < dist_before:
                        reward += 0.1

            # Update position and trail
            a['trail'].append(a['pos'].copy())
            a['pos'] += a['vel']

            # Wall bouncing
            if a['pos'].x < a['radius'] or a['pos'].x > self.WIDTH - a['radius']:
                a['vel'].x *= -0.8
                a['pos'].x = max(a['radius'], min(self.WIDTH - a['radius'], a['pos'].x))
            if a['pos'].y < a['radius'] or a['pos'].y > self.HEIGHT - a['radius']:
                a['vel'].y *= -0.8
                a['pos'].y = max(a['radius'], min(self.HEIGHT - a['radius'], a['pos'].y))

        return reward

    def _handle_collisions(self):
        reward = 0.0
        collected_indices = set()

        # Asteroid-Collector collision
        for i, a in enumerate(self.asteroids):
            if (a['pos'] - self.COLLECTOR_POS).length() < a['radius'] + self.COLLECTOR_RADIUS:
                collected_indices.add(i)
                self.score += int(a['mass'])
                reward += 1.0
                self._create_explosion(a['pos'], int(a['mass'] * 2), self.COLOR_PARTICLE)
                # Sound: sfx_collect.wav

        # Asteroid-Asteroid collision
        merged_indices = set()
        new_asteroids = []
        for i in range(len(self.asteroids)):
            if i in collected_indices or i in merged_indices: continue
            for j in range(i + 1, len(self.asteroids)):
                if j in collected_indices or j in merged_indices: continue

                a1 = self.asteroids[i]
                a2 = self.asteroids[j]
                dist_vec = a1['pos'] - a2['pos']

                if dist_vec.length() < a1['radius'] + a2['radius']:
                    # Merge
                    total_mass = a1['mass'] + a2['mass']
                    new_pos = (a1['pos'] * a1['mass'] + a2['pos'] * a2['mass']) / total_mass
                    new_vel = (a1['vel'] * a1['mass'] + a2['vel'] * a2['mass']) / total_mass

                    new_asteroids.append({
                        'pos': new_pos, 'vel': new_vel, 'mass': total_mass,
                        'radius': math.sqrt(total_mass), 'trail': deque(maxlen=10)
                    })

                    merged_indices.add(i)
                    merged_indices.add(j)
                    reward += 0.5
                    self._create_explosion(new_pos, int(total_mass), self._get_asteroid_color(total_mass))
                    # Sound: sfx_merge.wav
                    break # Move to next i

        # Rebuild asteroid list
        all_removed_indices = collected_indices.union(merged_indices)
        if all_removed_indices:
            self.asteroids = [a for i, a in enumerate(self.asteroids) if i not in all_removed_indices]
        self.asteroids.extend(new_asteroids)

        # Clamp asteroid count
        if len(self.asteroids) > self.MAX_ASTEROIDS:
            self.asteroids.sort(key=lambda a: a['mass'], reverse=True)
            self.asteroids = self.asteroids[:self.MAX_ASTEROIDS]

        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': self.np_random.integers(15, 30),
                'color': color, 'radius': self.np_random.uniform(1, 3)
            })

    def _get_asteroid_color(self, mass):
        mass_t = np.clip((mass - self.MIN_ASTEROID_MASS) / (self.MAX_ASTEROID_MASS - self.MIN_ASTEROID_MASS), 0, 1)
        if mass_t < 0.33:
            return self._lerp_color(self.ASTEROID_COLORS[0], self.ASTEROID_COLORS[1], mass_t / 0.33)
        elif mass_t < 0.66:
            return self._lerp_color(self.ASTEROID_COLORS[1], self.ASTEROID_COLORS[2], (mass_t - 0.33) / 0.33)
        else:
            return self._lerp_color(self.ASTEROID_COLORS[2], self.ASTEROID_COLORS[3], (mass_t - 0.66) / 0.34)

    def _lerp_color(self, c1, c2, t):
        return (c1[0] + (c2[0] - c1[0]) * t, c1[1] + (c2[1] - c1[1]) * t, c1[2] + (c2[2] - c1[2]) * t)

    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_wells()
        self._render_collector()
        self._render_particles()
        self._render_asteroids()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_wells(self):
        for i, pos in enumerate(self.well_positions):
            if i == self.active_well_index:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                radius = 80 + pulse * 20
                alpha = 50 + pulse * 30

                # Draw pulsating glow
                glow_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(glow_surf, int(radius), int(radius), int(radius), (*self.COLOR_WELL_ACTIVE, int(alpha)))
                self.screen.blit(glow_surf, (int(pos.x - radius), int(pos.y - radius)))

                # Draw core circle
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 15, self.COLOR_WELL_ACTIVE)
            else:
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 15, self.COLOR_WELL_INACTIVE)

    def _render_collector(self):
        pos_int = (int(self.COLLECTOR_POS.x), int(self.COLLECTOR_POS.y))

        # Pulsating glow
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        glow_radius = self.COLLECTOR_RADIUS + 5 + pulse * 5
        glow_alpha = 100 + pulse * 50

        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, int(glow_radius), int(glow_radius), int(glow_radius), (*self.COLOR_COLLECTOR, int(glow_alpha)))
        self.screen.blit(glow_surf, (pos_int[0] - int(glow_radius), pos_int[1] - int(glow_radius)))

        # Core
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.COLLECTOR_RADIUS, self.COLOR_COLLECTOR)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.COLLECTOR_RADIUS, (0,0,0))


    def _render_asteroids(self):
        for a in self.asteroids:
            pos_int = (int(a['pos'].x), int(a['pos'].y))
            radius_int = int(a['radius'])
            color = self._get_asteroid_color(a['mass'])

            # Trail
            for i, p in enumerate(a['trail']):
                if i % 2 == 0: # Draw every other point for a dashed look
                    alpha = (i / len(a['trail'])) * 100
                    pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), int(radius_int * 0.5), (*color, alpha))

            # Asteroid body
            if radius_int > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, color)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius_int, (0,0,0))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, (p['life'] / 30.0) * 255)
            color_with_alpha = (*p['color'], int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color_with_alpha)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_sec = self.time_remaining / self.FPS
        time_text = self.font_large.render(f"{time_sec:.1f}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)

    # --- Gym Interface Methods ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining / self.FPS,
            "asteroid_count": len(self.asteroids)
        }

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == '__main__':
    # The original code had a validation function that is not part of the Gym API.
    # It's good for development but can be removed from the final version.
    # We will run the manual play example instead.
    
    # Set a real video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv()

    # --- Manual Play Controls ---
    # Arrow keys to activate wells
    # Q to quit
    # R to reset

    key_map = {
        pygame.K_UP: 1,      # Top-Left
        pygame.K_RIGHT: 2,   # Top-Right
        pygame.K_LEFT: 3,    # Bottom-Left
        pygame.K_DOWN: 4,    # Bottom-Right
    }

    obs, info = env.reset()
    done = False

    # Create a display window for manual play
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gravity Well Asteroid Collector")
    clock = pygame.time.Clock()

    running = True
    while running:
        action_movement = 0 # Default to no-op

        # Pygame event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # Get pressed keys for continuous action
        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                action_movement = move_action
                break

        # Construct the MultiDiscrete action
        action = [action_movement, 0, 0] # Space and Shift are not used

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()

        clock.tick(env.FPS)

    env.close()