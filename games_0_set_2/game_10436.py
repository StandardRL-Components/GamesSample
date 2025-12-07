import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:22:51.536572
# Source Brief: brief_00436.md
# Brief Index: 436
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a neutron's spin to navigate a gravity well.
    The goal is to survive as long as possible and trigger chain reactions by colliding with protons.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    game_description = (
        "Control a neutron's spin to orbit a gravity well. Survive as long as possible and collide with protons to score points."
    )
    user_guide = (
        "Controls: Use ↑ and ↓ to set spin direction (clockwise/counter-clockwise). Press space to toggle spin influence for sharper turns."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Core Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = self.metadata["render_fps"]
        self.MAX_STEPS = 120 * self.FPS  # 120 seconds

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Visuals ---
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_GRID = (25, 35, 55)
        self.COLOR_PROTON = (100, 255, 100)
        self.COLOR_SPIN_UP = (100, 170, 255)
        self.COLOR_SPIN_DOWN = (255, 130, 100)
        self.COLOR_TRAJECTORY = (255, 255, 255)
        self.COLOR_UI = (220, 220, 240)
        self.COLOR_INFLUENCE_GLOW = (255, 255, 150)

        # --- Physics & Gameplay ---
        self.INITIAL_GRAVITY = 0.02
        self.GRAVITY_INCREASE_RATE_PER_SECOND = 1.05
        self.GRAVITY_INCREASE_RATE_PER_FRAME = self.GRAVITY_INCREASE_RATE_PER_SECOND**(1/self.FPS)
        self.SPIN_FORCE_MAGNITUDE = 0.045
        self.SPIN_INFLUENCE_MULTIPLIER = 3.0
        self.VELOCITY_DAMPING = 0.997
        self.NEUTRON_RADIUS = 12
        self.PROTON_RADIUS = 6
        self.INITIAL_PROTON_COUNT = 5
        self.PROTON_SPAWN_INTERVAL = 10 * self.FPS  # 600 steps

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_status = pygame.font.SysFont("monospace", 18, bold=True)

        # --- State Variables ---
        # These are initialized in reset() to ensure a clean slate for each episode
        self.steps = None
        self.score = None
        self.terminated = None
        self.screen_center = None
        self.neutron_pos = None
        self.neutron_vel = None
        self.neutron_spin = None
        self.spin_influence_on = None
        self.last_space_state = None
        self.gravity_amplitude = None
        self.protons = None
        self.particles = None
        self.proton_spawn_timer = None
        self.proton_hits = None

        # --- Initialize state ---
        # self.reset() # No need to call reset in init, will be called by wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.proton_hits = 0
        self.terminated = False

        self.screen_center = np.array([self.WIDTH / 2.0, self.HEIGHT / 2.0])
        
        # Start neutron at the center with a random initial velocity
        self.neutron_pos = self.screen_center.copy()
        angle = self.np_random.uniform(0, 2 * math.pi)
        self.neutron_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * 1.5
        
        self.neutron_spin = 1  # 1 for "up" (clockwise), -1 for "down" (counter-clockwise)
        self.spin_influence_on = False
        self.last_space_state = False
        self.gravity_amplitude = self.INITIAL_GRAVITY

        self.protons = []
        self.particles = deque()
        self.proton_spawn_timer = 0
        
        for _ in range(self.INITIAL_PROTON_COUNT):
            self._spawn_proton()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0

        self._handle_input(action)
        self._update_physics()
        step_reward = self._handle_collisions()
        self._update_game_state()
        
        # Calculate survival reward
        reward += 0.1
        reward += step_reward

        self.steps += 1
        
        # Check for termination conditions
        if self.neutron_pos[0] < 0 or self.neutron_pos[0] > self.WIDTH or \
           self.neutron_pos[1] < 0 or self.neutron_pos[1] > self.HEIGHT:
            self.terminated = True
        
        if self.steps >= self.MAX_STEPS:
            self.terminated = True
            reward += 100.0  # Victory reward

        truncated = False # This env does not truncate based on time limit, it terminates.

        return (
            self._get_observation(),
            reward,
            self.terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1:  # Up
            self.neutron_spin = 1
        elif movement == 2:  # Down
            self.neutron_spin = -1
        
        # Toggle spin influence on the rising edge of the spacebar press
        if space_pressed and not self.last_space_state:
            self.spin_influence_on = not self.spin_influence_on
            # Sound effect placeholder: # sfx_toggle_on.play() or sfx_toggle_off.play()
        self.last_space_state = space_pressed

    def _update_physics(self):
        # Gravity force pulls towards the center
        vec_to_center = self.screen_center - self.neutron_pos
        dist_to_center = np.linalg.norm(vec_to_center)
        
        if dist_to_center > 1e-6:
            gravity_force = (vec_to_center / dist_to_center) * self.gravity_amplitude
        else:
            gravity_force = np.zeros(2)

        # Spin force is perpendicular to the gravity vector, creating orbital motion
        if dist_to_center > 1e-6:
            perp_vec = np.array([-vec_to_center[1], vec_to_center[0]]) / dist_to_center
            spin_force = perp_vec * self.neutron_spin * self.SPIN_FORCE_MAGNITUDE
        else:
            spin_force = np.zeros(2)

        # Amplify spin force if influence is active
        if self.spin_influence_on:
            spin_force *= self.SPIN_INFLUENCE_MULTIPLIER

        # Update velocity and position
        total_force = gravity_force + spin_force
        self.neutron_vel += total_force
        self.neutron_vel *= self.VELOCITY_DAMPING
        self.neutron_pos += self.neutron_vel

    def _handle_collisions(self):
        reward = 0
        collided_protons = []
        for i, proton_pos in enumerate(self.protons):
            dist = np.linalg.norm(self.neutron_pos - proton_pos)
            if dist < self.NEUTRON_RADIUS + self.PROTON_RADIUS:
                collided_protons.append(i)
                reward += 1.0
                self.proton_hits += 1
                self._create_collision_particles(proton_pos)
                # Sound effect placeholder: # sfx_proton_hit.play()

        # Remove collided protons (in reverse order to avoid index errors)
        for i in sorted(collided_protons, reverse=True):
            del self.protons[i]
            
        return reward

    def _update_game_state(self):
        # Increase gravity over time
        self.gravity_amplitude *= self.GRAVITY_INCREASE_RATE_PER_FRAME

        # Spawn new protons periodically
        self.proton_spawn_timer += 1
        if self.proton_spawn_timer >= self.PROTON_SPAWN_INTERVAL:
            self.proton_spawn_timer = 0
            self._spawn_proton()
        
        # Update particles
        for _ in range(len(self.particles)):
            p = self.particles.popleft()
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                self.particles.append(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_protons()
        self._render_particles()
        self._render_trajectory()
        self._render_neutron()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.proton_hits,
            "steps": self.steps,
            "time_seconds": self.steps / self.FPS,
            "gravity": self.gravity_amplitude
        }

    # --- Rendering Methods ---

    def _render_background(self):
        # Faint grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Gravity well visualization (concentric circles with decreasing alpha)
        center_int = (int(self.screen_center[0]), int(self.screen_center[1]))
        max_radius = int(math.sqrt(self.WIDTH**2 + self.HEIGHT**2) / 2)
        for i in range(20):
            radius = int(max_radius * (1 - i / 20.0))
            alpha = int(15 * (self.gravity_amplitude / self.INITIAL_GRAVITY) * (1 - i / 20.0))
            alpha = min(255, max(0, alpha))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, center_int[0], center_int[1], radius, (*self.COLOR_GRID, alpha))

    def _render_protons(self):
        for pos in self.protons:
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.PROTON_RADIUS, self.COLOR_PROTON)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.PROTON_RADIUS, self.COLOR_PROTON)

    def _render_neutron(self):
        pos_int = (int(self.neutron_pos[0]), int(self.neutron_pos[1]))
        color = self.COLOR_SPIN_UP if self.neutron_spin == 1 else self.COLOR_SPIN_DOWN

        # Glow effect
        for i in range(4):
            radius = self.NEUTRON_RADIUS + i * 4
            alpha = 60 - i * 15
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, (*color, alpha))
        
        # Influence indicator
        if self.spin_influence_on:
            for i in range(3):
                radius = self.NEUTRON_RADIUS + i * 2
                alpha = 100 - i * 30
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, (*self.COLOR_INFLUENCE_GLOW, alpha))


        # Core neutron
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.NEUTRON_RADIUS, color)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.NEUTRON_RADIUS, color)

    def _render_trajectory(self):
        # Simulate future path for visualization
        temp_pos = self.neutron_pos.copy()
        temp_vel = self.neutron_vel.copy()
        points = []
        
        for _ in range(30):
            vec_to_center = self.screen_center - temp_pos
            dist = np.linalg.norm(vec_to_center)
            if dist < 1e-6: break

            gravity_force = (vec_to_center / dist) * self.gravity_amplitude
            perp_vec = np.array([-vec_to_center[1], vec_to_center[0]]) / dist
            spin_force = perp_vec * self.neutron_spin * self.SPIN_FORCE_MAGNITUDE
            if self.spin_influence_on:
                spin_force *= self.SPIN_INFLUENCE_MULTIPLIER

            temp_vel += gravity_force + spin_force
            temp_vel *= self.VELOCITY_DAMPING
            temp_pos += temp_vel
            points.append(tuple(map(int, temp_pos)))

        if len(points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_TRAJECTORY, False, points, 1)

    def _render_particles(self):
        for p in self.particles:
            alpha = p['lifespan'] / p['max_lifespan']
            size = int(p['size'] * alpha)
            if size > 0:
                color = (*p['color'], int(255 * alpha))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

    def _render_ui(self):
        # Top-left: Timer
        time_text = f"TIME: {self.steps / self.FPS:.1f}s / 120.0s"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI)
        self.screen.blit(time_surf, (10, 10))

        # Top-right: Gravity and Score
        grav_text = f"GRAVITY: {self.gravity_amplitude:.3f}"
        grav_surf = self.font_ui.render(grav_text, True, self.COLOR_UI)
        self.screen.blit(grav_surf, (self.WIDTH - grav_surf.get_width() - 10, 10))
        
        score_text = f"PROTONS: {self.proton_hits}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 30))

        # Bottom-left: Spin status
        spin_str = "CLOCKWISE" if self.neutron_spin == 1 else "COUNTER-CW"
        spin_color = self.COLOR_SPIN_UP if self.neutron_spin == 1 else self.COLOR_SPIN_DOWN
        spin_surf = self.font_status.render(f"SPIN: {spin_str}", True, spin_color)
        self.screen.blit(spin_surf, (10, self.HEIGHT - 50))

        influence_str = "ON" if self.spin_influence_on else "OFF"
        influence_color = self.COLOR_INFLUENCE_GLOW if self.spin_influence_on else self.COLOR_UI
        influence_surf = self.font_status.render(f"INFLUENCE: {influence_str}", True, influence_color)
        self.screen.blit(influence_surf, (10, self.HEIGHT - 28))

    # --- Helper Methods ---

    def _spawn_proton(self):
        # Spawn a proton in a valid location (not too close to center or edges)
        padding = 20
        while True:
            pos = np.array([
                self.np_random.uniform(padding, self.WIDTH - padding),
                self.np_random.uniform(padding, self.HEIGHT - padding)
            ], dtype=np.float64)
            if np.linalg.norm(pos - self.screen_center) > 100:
                self.protons.append(pos)
                break

    def _create_collision_particles(self, position):
        num_particles = 20
        neutron_color = self.COLOR_SPIN_UP if self.neutron_spin == 1 else self.COLOR_SPIN_DOWN
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(20, 40)
            
            # Mix neutron and proton colors for particles
            mix_ratio = self.np_random.uniform(0.3, 0.7)
            p_color = tuple(int(c1 * mix_ratio + c2 * (1-mix_ratio)) for c1, c2 in zip(neutron_color, self.COLOR_PROTON))

            self.particles.append({
                'pos': position.copy(),
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'size': self.np_random.integers(2, 5),
                'color': p_color
            })

    def close(self):
        pygame.quit()

    def render(self):
        # The main render method for Gymnasium is to return the observation.
        # This is handled by _get_observation().
        return self._get_observation()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires pygame to be installed with a display driver.
    # If you are running in a headless environment, this block will fail.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Pygame setup for manual play
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Neutron Spin")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # [movement, space, shift]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Reset action
        action = [0, 0, 0] # 0 = no movement
        
        # Map keys to actions
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1 # up
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[0] = 2 # down
        
        if keys[pygame.K_SPACE]:
            action[1] = 1 # space held
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1 # shift held

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            # You can add a "Game Over" screen or just reset
            print(f"Episode finished. Final Score: {info['score']}, Time: {info['time_seconds']:.2f}s")
            obs, info = env.reset()
            terminated = False

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()