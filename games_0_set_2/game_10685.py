import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:49:05.369833
# Source Brief: brief_00685.md
# Brief Index: 685
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An expert-crafted Gymnasium environment for "Orbital Ballet".

    The agent controls four satellites orbiting a central hub. The goal is to
    adjust their orbital radii to collect 30 data points without any of the
    satellites colliding.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Select satellite (0=none, 1=Sat1, 2=Sat2, 3=Sat3, 4=Sat4)
    - action[1]: Increase radius (0=no, 1=yes)
    - action[2]: Decrease radius (0=no, 1=yes)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Rewards:
    - +1.0 for each data point collected.
    - +0.01 for each step survived.
    - +50.0 for winning (collecting all data points).
    - -50.0 for a satellite collision.
    - -10.0 for running out of time.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    game_description = (
        "Control four orbiting satellites to collect data points. "
        "Adjust their orbits to gather all points without any collisions."
    )
    user_guide = (
        "Controls: Use number keys 1-4 to select a satellite. "
        "Use ↑ to increase its orbit and ↓ to decrease its orbit."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

    # Game parameters
    NUM_SATELLITES = 4
    NUM_DATA_POINTS = 30
    MAX_STEPS = 2250  # 75 seconds at 30 FPS
    WIN_SCORE = 30

    # Satellite parameters
    SATELLITE_RADIUS = 8
    MIN_ORBIT_RADIUS = 60
    MAX_ORBIT_RADIUS = 180
    ORBIT_ADJUST_SPEED = 0.5
    SATELLITE_SPEED_CONSTANT = 200 # Larger is slower

    # Data point parameters
    DATA_POINT_RADIUS = 5
    DATA_COLLECTION_RANGE = SATELLITE_RADIUS + DATA_POINT_RADIUS

    # Visuals
    COLOR_BG = (10, 15, 30)
    COLOR_HUB = (50, 60, 80)
    COLOR_ORBIT_LINE = (40, 50, 70)
    COLOR_SELECTED_ORBIT = (200, 200, 200)
    COLOR_DATA_POINT = (255, 255, 255)
    COLOR_DATA_GLOW = (180, 220, 255)
    COLOR_COLLISION = (255, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    SATELLITE_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
        (255, 128, 0),  # Orange
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.screen_width = self.SCREEN_WIDTH
        self.screen_height = self.SCREEN_HEIGHT

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()

        if self.render_mode == "human":
            pygame.display.set_caption("Orbital Ballet")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("sans-serif", 50, bold=True)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        self.satellites = []
        self.data_points = []
        self.particles = []
        self.starfield = []
        self.selected_satellite_idx = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        self.selected_satellite_idx = -1
        self.particles.clear()

        self._initialize_starfield()
        self._initialize_satellites()
        self._initialize_data_points()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.01  # Small survival reward

        self._handle_input(action)
        self._update_game_state()

        collected_this_step, collision_detected = self._check_events()
        reward += collected_this_step

        terminated = self._check_termination(collision_detected)
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated:
            if self.termination_reason == "win":
                reward += 50.0
            elif self.termination_reason == "collision":
                reward -= 50.0
        elif truncated:
            self.termination_reason = "timeout"
            reward -= 10.0


        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, increase_radius, decrease_radius = action
        
        # Movement action selects the satellite
        if 1 <= movement <= self.NUM_SATELLITES:
            self.selected_satellite_idx = movement - 1
        # No else, selection persists if movement is 0

        # Adjust radius of the selected satellite
        if self.selected_satellite_idx != -1:
            radius_change = (increase_radius - decrease_radius) * self.ORBIT_ADJUST_SPEED
            sat = self.satellites[self.selected_satellite_idx]
            sat['orbit_radius'] = np.clip(
                sat['orbit_radius'] + radius_change,
                self.MIN_ORBIT_RADIUS,
                self.MAX_ORBIT_RADIUS
            )

    def _update_game_state(self):
        # Update satellites
        for sat in self.satellites:
            # Slower speed for larger orbits (Kepler-like feel)
            angular_velocity = self.SATELLITE_SPEED_CONSTANT / (sat['orbit_radius']**1.5 + 1e-6)
            sat['angle'] += angular_velocity
            sat['pos'] = (
                self.CENTER[0] + sat['orbit_radius'] * math.cos(sat['angle']),
                self.CENTER[1] + sat['orbit_radius'] * math.sin(sat['angle'])
            )
            # Emit trail particles
            if self.steps % 2 == 0:
                self._create_particles(sat['pos'], sat['color'], 1, 0.5, 15)

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['life'] -= 1

    def _check_events(self):
        # Data collection
        collected_reward = 0
        remaining_data = []
        for dp_pos in self.data_points:
            collected = False
            for sat in self.satellites:
                dist = math.hypot(sat['pos'][0] - dp_pos[0], sat['pos'][1] - dp_pos[1])
                if dist < self.DATA_COLLECTION_RANGE:
                    collected = True
                    break
            if collected:
                self.score += 1
                collected_reward += 1.0
                # sfx: data_collect.wav
                self._create_particles(dp_pos, self.COLOR_DATA_GLOW, 20, 2.0, 25)
            else:
                remaining_data.append(dp_pos)
        self.data_points = remaining_data

        # Satellite collisions
        collision = False
        for i in range(self.NUM_SATELLITES):
            for j in range(i + 1, self.NUM_SATELLITES):
                sat1 = self.satellites[i]
                sat2 = self.satellites[j]
                dist = math.hypot(sat1['pos'][0] - sat2['pos'][0], sat1['pos'][1] - sat2['pos'][1])
                if dist < self.SATELLITE_RADIUS * 2:
                    collision = True
                    self.game_over = True
                    # sfx: explosion.wav
                    self._create_particles(
                        ((sat1['pos'][0] + sat2['pos'][0])/2, (sat1['pos'][1] + sat2['pos'][1])/2),
                        self.COLOR_COLLISION, 50, 4.0, 40
                    )
                    break
            if collision:
                break
        
        return collected_reward, collision

    def _check_termination(self, collision_detected):
        if collision_detected:
            self.termination_reason = "collision"
            self.game_over = True
            return True
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.termination_reason = "win"
            # sfx: win_fanfare.wav
            return True
        # Timeout is handled by truncation, not termination
        return False

    def _get_observation(self):
        # This method is used for the RL agent, it renders to a buffer
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "data_remaining": len(self.data_points)}

    def render(self):
        # This method is for human consumption
        if self.render_mode == "human":
            self._render_frame()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _render_frame(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for star in self.starfield:
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['radius'])

        # --- Game Elements ---
        self._render_particles()
        self._render_orbits()
        self._render_hub()
        self._render_data_points()
        self._render_satellites()
        
        # --- UI ---
        self._render_ui()

        # --- Game Over Message ---
        if self.game_over or self.steps >= self.MAX_STEPS:
            self._render_game_over_message()

    def _render_hub(self):
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER[0], self.CENTER[1], 20, self.COLOR_HUB)
        pygame.gfxdraw.aacircle(self.screen, self.CENTER[0], self.CENTER[1], 20, self.COLOR_HUB)

    def _render_orbits(self):
        for i, sat in enumerate(self.satellites):
            radius = int(sat['orbit_radius'])
            color = self.COLOR_ORBIT_LINE
            if i == self.selected_satellite_idx:
                color = self.COLOR_SELECTED_ORBIT
                # Pulsating effect for selection
                pulse = abs(math.sin(self.steps * 0.1))
                pulse_color = tuple(min(255, c + int(pulse * 50)) for c in color)
                pygame.gfxdraw.aacircle(self.screen, self.CENTER[0], self.CENTER[1], radius, pulse_color)
            
            pygame.gfxdraw.aacircle(self.screen, self.CENTER[0], self.CENTER[1], radius, color)

    def _render_satellites(self):
        for i, sat in enumerate(self.satellites):
            pos = (int(sat['pos'][0]), int(sat['pos'][1]))
            
            # Glow effect
            self._draw_glowing_circle(self.screen, sat['color'], pos, self.SATELLITE_RADIUS, 1.5)

            # Selection indicator
            if i == self.selected_satellite_idx:
                pulse_radius = self.SATELLITE_RADIUS + 4 + int(abs(math.sin(self.steps * 0.1) * 3))
                pulse_alpha = 100 + int(abs(math.sin(self.steps * 0.1) * 50))
                # Create a surface for alpha blending
                indicator_surf = pygame.Surface((pulse_radius*2, pulse_radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.aacircle(indicator_surf, pulse_radius, pulse_radius, pulse_radius-1, (255, 255, 255, pulse_alpha))
                self.screen.blit(indicator_surf, (pos[0]-pulse_radius, pos[1]-pulse_radius))


    def _render_data_points(self):
        for pos in self.data_points:
            self._draw_glowing_circle(self.screen, self.COLOR_DATA_POINT, (int(pos[0]), int(pos[1])), self.DATA_POINT_RADIUS, 2.0)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                # Create a surface for alpha blending
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, (*p['color'], alpha), (size, size), size)
                self.screen.blit(particle_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))


    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"DATA: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.metadata["render_fps"])
        time_text = self.font_main.render(f"TIME: {time_left:.1f}s", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.screen_width - time_text.get_width() - 10, 10))

        # Progress Bar
        bar_y = self.screen_height - 15
        bar_height = 10
        progress = self.score / self.WIN_SCORE
        bar_width = (self.screen_width - 20) * progress
        
        pygame.draw.rect(self.screen, self.COLOR_HUB, (10, bar_y, self.screen_width - 20, bar_height))
        if bar_width > 0:
            pygame.draw.rect(self.screen, self.SATELLITE_COLORS[0], (10, bar_y, bar_width, bar_height))

    def _render_game_over_message(self):
        if self.termination_reason == "win":
            msg = "MISSION COMPLETE"
            color = self.SATELLITE_COLORS[0]
        elif self.termination_reason == "collision":
            msg = "CATASTROPHIC FAILURE"
            color = self.COLOR_COLLISION
        elif self.termination_reason == "timeout" or self.steps >= self.MAX_STEPS:
            msg = "TIME UP"
            color = self.SATELLITE_COLORS[3]
        else:
            return

        text = self.font_msg.render(msg, True, color)
        text_rect = text.get_rect(center=self.CENTER)
        
        # Add a semi-transparent background for readability
        bg_surf = pygame.Surface(text_rect.size, pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 150))
        self.screen.blit(bg_surf, text_rect.topleft)
        self.screen.blit(text, text_rect)

    # --- Initialization Helpers ---
    def _initialize_satellites(self):
        self.satellites.clear()
        initial_radii = np.linspace(self.MIN_ORBIT_RADIUS + 10, self.MAX_ORBIT_RADIUS - 10, self.NUM_SATELLITES)
        for i in range(self.NUM_SATELLITES):
            angle = self.np_random.uniform(0, 2 * math.pi)
            self.satellites.append({
                'orbit_radius': initial_radii[i],
                'angle': angle,
                'pos': (0, 0), # Will be updated in first step
                'color': self.SATELLITE_COLORS[i]
            })

    def _initialize_data_points(self):
        self.data_points.clear()
        while len(self.data_points) < self.NUM_DATA_POINTS:
            angle = self.np_random.uniform(0, 2 * math.pi)
            radius = self.np_random.uniform(self.MIN_ORBIT_RADIUS - 10, self.MAX_ORBIT_RADIUS + 10)
            pos = (
                self.CENTER[0] + radius * math.cos(angle),
                self.CENTER[1] + radius * math.sin(angle)
            )
            # Ensure points aren't too close to each other
            if all(math.hypot(pos[0] - p[0], pos[1] - p[1]) > 20 for p in self.data_points):
                self.data_points.append(pos)
    
    def _initialize_starfield(self):
        self.starfield.clear()
        for _ in range(150):
            self.starfield.append({
                'pos': (self.np_random.integers(0, self.screen_width), self.np_random.integers(0, self.screen_height)),
                'radius': self.np_random.uniform(0.5, 1.5),
                'color': random.choice([(100, 100, 100), (150, 150, 150), (200, 200, 220)])
            })

    # --- Effect Helpers ---
    def _create_particles(self, pos, color, count, speed_scale, life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * speed_scale
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': self.np_random.integers(life // 2, life),
                'max_life': life,
                'size': self.np_random.uniform(1, 3)
            })

    def _draw_glowing_circle(self, surface, color, center, radius, glow_factor):
        glow_radius = int(radius * glow_factor)
        
        # Create a temporary surface for the glow
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_alpha = 60
        pygame.draw.circle(temp_surf, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius)
        
        # Scale it down for a softer falloff - this is a fast blur approximation
        scaled_size = (int(radius * 2.5), int(radius * 2.5))
        if scaled_size[0] > 0 and scaled_size[1] > 0:
            temp_surf = pygame.transform.smoothscale(temp_surf, scaled_size)
        
        surface.blit(temp_surf, (center[0] - temp_surf.get_width() / 2, center[1] - temp_surf.get_height() / 2))
        
        # Draw the main circle on top
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], int(radius), color)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], int(radius), color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("----------------------\n")

    while not done:
        # Default action: no selection change, no radius change
        action = [0, 0, 0] 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Satellite selection (takes priority)
        if keys[pygame.K_1]: action[0] = 1
        elif keys[pygame.K_2]: action[0] = 2
        elif keys[pygame.K_3]: action[0] = 3
        elif keys[pygame.K_4]: action[0] = 4
        
        # Radius adjustment
        if keys[pygame.K_w] or keys[pygame.K_UP]: action[1] = 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        env.render()
        
        if terminated or truncated:
            print(f"Game Over! Reason: {env.termination_reason}")
            print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            done = True
            pygame.time.wait(3000) # Pause for 3 seconds to show the final screen

    env.close()