import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:56:55.365309
# Source Brief: brief_01381.md
# Brief Index: 1381
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper function for angular distance, handles wrapping around 360 degrees
def angular_diff(a1, a2):
    """Calculates the shortest angle between a1 and a2 in degrees."""
    diff = (a2 - a1 + 180) % 360 - 180
    return diff

class Landmass:
    """A class to represent an orbiting landmass."""
    def __init__(self, angle, radius, mass, color, ang_vel=0):
        self.angle = angle  # In degrees
        self.radius = radius
        self.mass = mass
        self.color = color
        self.ang_vel = ang_vel  # In degrees per step
        self.active = True

    def get_pos(self, center):
        """Calculates cartesian coordinates from polar coordinates."""
        rad_angle = math.radians(self.angle)
        x = center[0] + self.radius * math.cos(rad_angle)
        y = center[1] + self.radius * math.sin(rad_angle)
        return int(x), int(y)

    def get_size(self):
        """Calculates visual size based on mass."""
        return 5 * (self.mass ** (1/3))

    def update(self, gravity_angle, gravity_strength, damping):
        """Updates the landmass's physics state."""
        if not self.active:
            return

        angle_to_gravity = angular_diff(self.angle, gravity_angle)
        force = gravity_strength * math.cos(math.radians(angle_to_gravity))
        acceleration = force / self.mass
        
        self.ang_vel += acceleration
        self.ang_vel *= damping
        self.angle = (self.angle + self.ang_vel) % 360

    def draw(self, surface, center):
        """Renders the landmass with a glow effect."""
        if not self.active:
            return

        pos = self.get_pos(center)
        size = self.get_size()
        
        # Glow effect: draw multiple semi-transparent circles
        for i in range(int(size // 2), 0, -2):
            glow_radius = int(size + i)
            glow_alpha = 40 - int(i * (40 / (size // 2)))
            glow_color = (*self.color, glow_alpha)
            temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            surface.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Main body using anti-aliased drawing
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(size), self.color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(size), self.color)

class Particle:
    """A class for simple explosion particles on merger."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = random.randint(20, 40)
        self.initial_lifetime = self.lifetime

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            size = int(3 * (self.lifetime / self.initial_lifetime))
            if size > 0:
                # Use a temporary surface for alpha blending
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.color, alpha), (size, size), size)
                surface.blit(temp_surf, (int(self.x) - size, int(self.y) - size))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Use a central gravity beam to attract and merge orbiting celestial bodies. "
        "Combine all mass into a single super-body to win before time runs out."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to rotate the central gravity beam and pull the landmasses together."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    CENTER = (WIDTH // 2, HEIGHT // 2)
    PLANET_ORBIT_RADIUS = 150
    MAX_STEPS = 1800  # 180 seconds at 10 steps/sec
    VICTORY_CONSOLIDATION = 0.9
    NUM_LANDMASSES = 5
    
    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_PLANET = (20, 40, 60)
    COLOR_MANIPULATOR = (255, 255, 255)
    LANDMASS_COLORS = [
        (255, 87, 87),   # Red
        (87, 255, 87),   # Green
        (87, 87, 255),   # Blue
        (255, 255, 87),  # Yellow
        (255, 87, 255),  # Magenta
    ]
    COLOR_UI_TEXT = (220, 220, 240)
    
    # Physics
    GRAVITY_STRENGTH = 0.05
    DAMPING_FACTOR = 0.98
    MERGE_THRESHOLD_FACTOR = 1.2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        
        # Initialize state variables to be populated in reset()
        self.landmasses = []
        self.particles = []
        self.manipulator_angle = 0.0
        self.steps = 0
        self.score = 0
        self.consolidation = 0.0
        self.total_mass = 0
        self.last_spread = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.manipulator_angle = self.np_random.uniform(0, 360)
        self.particles.clear()
        
        self.total_mass = self.NUM_LANDMASSES
        self.landmasses = []
        start_angles = [i * (360 / self.NUM_LANDMASSES) for i in range(self.NUM_LANDMASSES)]
        self.np_random.shuffle(start_angles)

        for i in range(self.NUM_LANDMASSES):
            angle = start_angles[i] + self.np_random.uniform(-10, 10)
            ang_vel = self.np_random.uniform(-0.1, 0.1)
            self.landmasses.append(
                Landmass(angle, self.PLANET_ORBIT_RADIUS, 1, self.LANDMASS_COLORS[i], ang_vel)
            )
        
        self._calculate_consolidation()
        self.last_spread = self._calculate_angular_spread()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, _, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.steps += 1
        
        self._apply_action(movement)
        self._update_physics()
        
        merger_reward = self._handle_mergers()
        reward += merger_reward
        if merger_reward > 0:
            self._calculate_consolidation()

        spread_reward = self._calculate_spread_reward()
        reward += spread_reward
        
        terminated = self.consolidation >= self.VICTORY_CONSOLIDATION or self.steps >= self.MAX_STEPS
        truncated = False
        
        if terminated:
            if self.consolidation >= self.VICTORY_CONSOLIDATION:
                reward += 100
            else:
                reward -= 100
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _apply_action(self, movement):
        if movement == 1: self.manipulator_angle = (self.manipulator_angle + 1) % 360
        elif movement == 2: self.manipulator_angle = (self.manipulator_angle - 1 + 360) % 360
        elif movement == 3: self.manipulator_angle = (self.manipulator_angle + 5) % 360
        elif movement == 4: self.manipulator_angle = (self.manipulator_angle - 5 + 360) % 360

    def _update_physics(self):
        for lm in self.landmasses:
            lm.update(self.manipulator_angle, self.GRAVITY_STRENGTH, self.DAMPING_FACTOR)
        
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifetime > 0]

    def _handle_mergers(self):
        merger_reward = 0
        active_landmasses = [lm for lm in self.landmasses if lm.active]
        
        for i in range(len(active_landmasses)):
            for j in range(i + 1, len(active_landmasses)):
                lm1 = active_landmasses[i]
                lm2 = active_landmasses[j]

                size1_rad = math.radians(lm1.get_size() / (2 * math.pi * lm1.radius) * 360)
                size2_rad = math.radians(lm2.get_size() / (2 * math.pi * lm2.radius) * 360)
                angular_size_sum = math.degrees(size1_rad + size2_rad)

                if abs(angular_diff(lm1.angle, lm2.angle)) < angular_size_sum * self.MERGE_THRESHOLD_FACTOR:
                    # Merge lm2 into lm1 (the larger one)
                    if lm1.mass < lm2.mass: lm1, lm2 = lm2, lm1

                    total_mass = lm1.mass + lm2.mass
                    lm1.ang_vel = (lm1.mass * lm1.ang_vel + lm2.mass * lm2.ang_vel) / total_mass
                    
                    d = angular_diff(lm1.angle, lm2.angle)
                    lm1.angle = (lm1.angle + d * lm2.mass / total_mass) % 360
                    
                    lm1.mass = total_mass
                    lm2.active = False
                    
                    merger_reward += 5
                    # SFX: MERGE_SUCCESS

                    pos = lm1.get_pos(self.CENTER)
                    for _ in range(30):
                        self.particles.append(Particle(pos[0], pos[1], lm1.color))

                    return merger_reward + self._handle_mergers()
        return merger_reward

    def _calculate_consolidation(self):
        if not self.landmasses:
            self.consolidation = 0
            return
        max_mass = max(lm.mass for lm in self.landmasses)
        self.consolidation = max_mass / self.total_mass
    
    def _calculate_angular_spread(self):
        active_lms = [lm for lm in self.landmasses if lm.active]
        if len(active_lms) < 2:
            return 0
        
        angles = sorted([lm.angle for lm in active_lms])
        max_gap = 0
        for i in range(len(angles) - 1):
            max_gap = max(max_gap, angles[i+1] - angles[i])
        max_gap = max(max_gap, (360 - angles[-1]) + angles[0])
        return 360 - max_gap

    def _calculate_spread_reward(self):
        current_spread = self._calculate_angular_spread()
        spread_reduction = self.last_spread - current_spread
        self.last_spread = current_spread
        return spread_reduction * 0.05

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER[0], self.CENTER[1], self.PLANET_ORBIT_RADIUS + 15, self.COLOR_PLANET)
        pygame.gfxdraw.aacircle(self.screen, self.CENTER[0], self.CENTER[1], self.PLANET_ORBIT_RADIUS + 15, self.COLOR_PLANET)
        pygame.gfxdraw.aacircle(self.screen, self.CENTER[0], self.CENTER[1], self.PLANET_ORBIT_RADIUS, (50, 70, 100))

        rad_angle = math.radians(self.manipulator_angle)
        end_x = self.CENTER[0] + (self.PLANET_ORBIT_RADIUS + 25) * math.cos(rad_angle)
        end_y = self.CENTER[1] + (self.PLANET_ORBIT_RADIUS + 25) * math.sin(rad_angle)
        pygame.draw.aaline(self.screen, self.COLOR_MANIPULATOR, self.CENTER, (end_x, end_y), 2)
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER[0], self.CENTER[1], 10, self.COLOR_MANIPULATOR)
        
        for lm in sorted(self.landmasses, key=lambda l: l.mass):
            lm.draw(self.screen, self.CENTER)
        
        particle_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for p in self.particles:
            p.draw(particle_surface)
        self.screen.blit(particle_surface, (0, 0))

    def _render_ui(self):
        consol_text = f"CONSOLIDATION: {self.consolidation:.1%}"
        text_surf = self.font_large.render(consol_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (20, 10))

        time_left = (self.MAX_STEPS - self.steps) / 10.0
        time_text = f"TIME: {max(0, time_left):.1f}s"
        text_surf = self.font_large.render(time_text, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "consolidation": self.consolidation,
            "active_landmasses": sum(1 for lm in self.landmasses if lm.active)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not part of the Gymnasium environment
    # It will not be checked by the tests, but is useful for debugging.
    # To run, you will need to remove the os.environ call that sets the video driver to "dummy"
    # and ensure you have a display environment.
    
    # To make this runnable, comment out or remove this line at the top of the file:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    try:
        env = GameEnv()
        obs, info = env.reset()
        terminated = False
        
        render_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Gravity Manipulator")
        clock = pygame.time.Clock()
        
        print("--- Human Play Controls ---")
        print("W/Up Arrow: Rotate CCW (small)")
        print("S/Down Arrow: Rotate CW (small)")
        print("A/Left Arrow: Rotate CCW (large)")
        print("D/Right Arrow: Rotate CW (large)")
        print("Q: Quit")
        
        while not terminated:
            action = [0, 0, 0]
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: action[0] = 4
            if keys[pygame.K_q]: terminated = True
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(render_surface, (0, 0))
            pygame.display.flip()
            
            clock.tick(30)
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Consolidation: {info['consolidation']:.1%}, Steps: {info['steps']}")
                pygame.time.wait(2000)

        env.close()
    except pygame.error as e:
        print("\nPygame error detected. This is expected if you are running in a headless environment.")
        print("The `if __name__ == '__main__':` block is for local human play and requires a display.")
        print("The environment itself is running headlessly as required.\n")