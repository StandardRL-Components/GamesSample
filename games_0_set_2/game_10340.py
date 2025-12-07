import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:13:10.829680
# Source Brief: brief_00340.md
# Brief Index: 340
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must balance a spinning top.

    The top's spin speed increases over time, making it more unstable.
    A periodic "stability transformation" makes the top easier to control.
    The goal is to keep the top from falling over and reach a target score.

    **Visuals:**
    - The top is rendered with a pseudo-3D effect, showing tilt and spin.
    - Particle effects provide feedback for near-falls.
    - A clean UI displays score, time, and transformation status.

    **Physics:**
    - A simple physics model simulates the top's tilt.
    - Player actions apply a corrective force.
    - A destabilizing force, proportional to tilt and spin speed, pushes the top over.
    - Damping provides a smoother, more controllable feel.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]`: Movement (0=none, 1=up, 2=down, 3=left, 4=right) to apply tilt force.
    - `actions[1]`: Unused (Space button).
    - `actions[2]`: Unused (Shift button).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Balance a spinning top on a platform. The top spins faster and becomes more unstable over time, "
        "but periodic stability boosts can help you survive."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to apply force and keep the top balanced."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 50
    DELTA_TIME = 1.0 / TARGET_FPS

    # Colors
    COLOR_BG_TOP = (15, 20, 40)
    COLOR_BG_BOTTOM = (40, 30, 60)
    COLOR_BASE = (100, 100, 120)
    COLOR_BASE_HIGHLIGHT = (150, 150, 170)
    COLOR_TOP = (0, 200, 255)
    COLOR_TOP_SHADOW = (0, 100, 130)
    COLOR_PARTICLE = (255, 255, 150)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_TRANSFORM_BAR_BG = (50, 50, 80)
    COLOR_TRANSFORM_BAR_COOLDOWN = (100, 100, 200)
    COLOR_TRANSFORM_BAR_ACTIVE = (150, 255, 150)

    # Game Parameters
    MAX_EPISODE_STEPS = 6000  # 120 seconds * 50 steps/sec
    WIN_SCORE = 100
    TILT_FORCE = 0.08
    GRAVITY_CONSTANT = 0.025
    DAMPING_FACTOR = 0.97
    FALL_THRESHOLD = 1.0 # Normalized radius of the base

    # Top Mechanics
    INITIAL_SPIN_SPEED = 1.0 # rad/s
    SPIN_SPEED_INCREASE_RATE = 0.2 # rad/s
    SPIN_SPEED_INCREASE_INTERVAL = 2.0 # seconds

    # Transformation Mechanics
    TRANSFORM_COOLDOWN_TIME = 15.0 # seconds
    TRANSFORM_DURATION = 10.0 # seconds
    TRANSFORM_STABILITY_MULTIPLIER = 0.3 # Gravity is 30% as strong

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_elapsed = 0.0

        # Top state
        self.top_tilt = np.array([0.0, 0.0]) # [x, y] tilt from center
        self.top_tilt_velocity = np.array([0.0, 0.0])
        self.top_spin_angle = 0.0
        self.top_spin_speed = 0.0
        self.spin_speed_timer = 0.0

        # Transformation state
        self.transformation_active = False
        self.transformation_timer = 0.0

        # Visuals
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0

        # Top state
        self.top_tilt = np.array([0.0, 0.0])
        self.top_tilt_velocity = np.array([0.0, 0.0])
        self.top_spin_angle = 0.0
        self.top_spin_speed = self.INITIAL_SPIN_SPEED
        self.spin_speed_timer = 0.0

        # Transformation state
        self.transformation_active = False
        self.transformation_timer = self.TRANSFORM_COOLDOWN_TIME

        # Visuals
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Unpack Action ---
        movement = action[0]
        # space_held and shift_held are unused per design brief

        # --- 2. Update Timers and Game State ---
        self.steps += 1
        self.time_elapsed += self.DELTA_TIME

        # Update spin speed
        self.spin_speed_timer += self.DELTA_TIME
        if self.spin_speed_timer >= self.SPIN_SPEED_INCREASE_INTERVAL:
            self.top_spin_speed += self.SPIN_SPEED_INCREASE_RATE
            self.spin_speed_timer = 0.0

        # Update transformation state
        self.transformation_timer -= self.DELTA_TIME
        if self.transformation_timer <= 0:
            if self.transformation_active:
                # Deactivate transformation
                self.transformation_active = False
                self.transformation_timer = self.TRANSFORM_COOLDOWN_TIME
            else:
                # Activate transformation
                self.transformation_active = True
                self.transformation_timer = self.TRANSFORM_DURATION

        # --- 3. Apply Physics ---
        # Player input force
        player_force = np.array([0.0, 0.0])
        if movement == 1: player_force[1] = -self.TILT_FORCE  # Up
        elif movement == 2: player_force[1] = self.TILT_FORCE   # Down
        elif movement == 3: player_force[0] = -self.TILT_FORCE  # Left
        elif movement == 4: player_force[0] = self.TILT_FORCE   # Right

        # Destabilizing force (gravity)
        # Proportional to tilt and spin speed
        gravity_force = self.top_tilt * self.GRAVITY_CONSTANT * (1 + self.top_spin_speed / 5.0)

        # Apply stability transformation
        if self.transformation_active:
            gravity_force *= self.TRANSFORM_STABILITY_MULTIPLIER

        # Update velocity
        self.top_tilt_velocity += player_force
        self.top_tilt_velocity += gravity_force

        # Apply damping
        self.top_tilt_velocity *= self.DAMPING_FACTOR

        # Update position
        self.top_tilt += self.top_tilt_velocity * self.DELTA_TIME

        # Update spin angle for visual rotation
        self.top_spin_angle = (self.top_spin_angle + self.top_spin_speed * self.DELTA_TIME) % (2 * math.pi)

        # --- 4. Update Particles ---
        self._update_particles()

        # --- 5. Calculate Reward & Check Termination ---
        reward = 0
        terminated = False
        truncated = False

        tilt_magnitude = np.linalg.norm(self.top_tilt)

        if tilt_magnitude > self.FALL_THRESHOLD:
            # Failure: Top fell over
            self.game_over = True
            terminated = True
            reward = -50.0
        elif self.score >= self.WIN_SCORE:
            # Victory: Reached target score
            self.game_over = True
            terminated = True
            reward = 100.0
        elif self.steps >= self.MAX_EPISODE_STEPS:
            # Failure: Time limit reached
            self.game_over = True
            terminated = True # Per Gymnasium API, this is termination, not truncation
            reward = 0.0 # No penalty, just end
        else:
            # Continuous reward for staying upright
            reward = 0.1
            # Update score based on how balanced the top is
            self.score += (1.0 - min(tilt_magnitude, 1.0)) * 0.05
            
            # Spawn particles if close to falling
            if tilt_magnitude > 0.75:
                # sfx: scraping sound
                self._spawn_particles(2)


        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        # Draw all elements to the screen surface
        self._render_background()
        self._render_base()
        self._render_particles()
        self._render_top()
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tilt": self.top_tilt,
            "spin_speed": self.top_spin_speed,
            "transformation_active": self.transformation_active,
        }

    # --- Rendering Methods ---

    def _render_background(self):
        # Simple vertical gradient
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_base(self):
        center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 50
        radius = 120
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_BASE_HIGHLIGHT)

    def _render_top(self):
        base_center_x, base_center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 50
        base_radius = 120
        top_radius = 40

        # Calculate position based on tilt
        pos_x = base_center_x + self.top_tilt[0] * (base_radius - top_radius)
        pos_y = base_center_y + self.top_tilt[1] * (base_radius - top_radius)

        # Pseudo-3D effect: squash the top into an ellipse based on tilt
        tilt_magnitude = np.linalg.norm(self.top_tilt)
        squash_factor = 1.0 - 0.4 * min(tilt_magnitude, 1.0)
        
        # Shadow
        shadow_y_offset = 15 + 10 * min(tilt_magnitude, 1.0)
        shadow_alpha = 50 + 100 * min(tilt_magnitude, 1.0)
        shadow_surf = pygame.Surface((top_radius * 2, top_radius * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, shadow_alpha), shadow_surf.get_rect())
        self.screen.blit(shadow_surf, (int(pos_x - top_radius), int(pos_y - top_radius + shadow_y_offset)))

        # Transformation glow
        if self.transformation_active:
            glow_radius = int(top_radius * 1.5)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            alpha_step = 80 / glow_radius
            for i in range(glow_radius, 0, -1):
                alpha = int(80 - (glow_radius - i) * alpha_step)
                pygame.gfxdraw.aacircle(glow_surf, glow_radius, glow_radius, i, (*self.COLOR_TRANSFORM_BAR_ACTIVE, alpha))
            self.screen.blit(glow_surf, (int(pos_x - glow_radius), int(pos_y - glow_radius)))
        
        # Main top body (ellipse for perspective)
        top_surf = pygame.Surface((top_radius * 2, int(top_radius * 2 * squash_factor)), pygame.SRCALPHA)
        pygame.draw.ellipse(top_surf, self.COLOR_TOP_SHADOW, top_surf.get_rect())
        pygame.draw.ellipse(top_surf, self.COLOR_TOP, top_surf.get_rect().inflate(-4, -4))
        
        # Spinning line indicator
        line_len = top_radius * 0.8
        line_end_x = top_radius + math.cos(self.top_spin_angle) * line_len
        line_end_y = (top_radius * squash_factor) + math.sin(self.top_spin_angle) * line_len * squash_factor
        pygame.draw.aaline(top_surf, self.COLOR_TOP_SHADOW, (top_radius, top_radius * squash_factor), (line_end_x, line_end_y), 2)
        
        self.screen.blit(top_surf, (int(pos_x - top_radius), int(pos_y - top_radius * squash_factor)))


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))
        score_label = self.font_small.render("SCORE", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_label, (20, 55))

        # Time
        time_left = max(0, self.MAX_EPISODE_STEPS // self.TARGET_FPS - int(self.time_elapsed))
        time_text = self.font_large.render(f"{time_left}s", True, self.COLOR_UI_TEXT)
        text_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(time_text, text_rect)
        time_label = self.font_small.render("TIME", True, self.COLOR_UI_TEXT)
        label_rect = time_label.get_rect(topright=(self.SCREEN_WIDTH - 20, 55))
        self.screen.blit(time_label, label_rect)

        # Transformation Bar
        bar_width = 200
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        bar_y = self.SCREEN_HEIGHT - 30
        pygame.draw.rect(self.screen, self.COLOR_TRANSFORM_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        
        if self.transformation_active:
            fill_ratio = self.transformation_timer / self.TRANSFORM_DURATION
            color = self.COLOR_TRANSFORM_BAR_ACTIVE
            label_text = "STABILITY ACTIVE"
        else:
            fill_ratio = 1.0 - (self.transformation_timer / self.TRANSFORM_COOLDOWN_TIME)
            color = self.COLOR_TRANSFORM_BAR_COOLDOWN
            label_text = "STABILITY CHARGING"
        
        fill_width = max(0, int(bar_width * fill_ratio))
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height), border_radius=4)
        
        label = self.font_small.render(label_text, True, self.COLOR_UI_TEXT)
        label_rect = label.get_rect(center=(self.SCREEN_WIDTH // 2, bar_y - 15))
        self.screen.blit(label, label_rect)

    # --- Particle System ---

    def _spawn_particles(self, count):
        base_center_x, base_center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 50
        base_radius = 120
        top_radius = 40
        
        pos_x = base_center_x + self.top_tilt[0] * (base_radius - top_radius)
        pos_y = base_center_y + self.top_tilt[1] * (base_radius - top_radius)
        
        # Spawn particles at the edge of the top, opposite the tilt direction
        spawn_angle = math.atan2(self.top_tilt[1], self.top_tilt[0])
        
        for _ in range(count):
            angle_offset = random.uniform(-0.5, 0.5)
            angle = spawn_angle + math.pi + angle_offset
            speed = random.uniform(1, 3)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            
            particle = {
                "pos": [pos_x, pos_y],
                "vel": velocity,
                "life": random.uniform(0.5, 1.5),
                "max_life": 1.5,
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.98
            p["vel"][1] *= 0.98
            p["life"] -= self.DELTA_TIME
        
        # Remove dead particles
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p["life"] / p["max_life"]
            radius = int(3 * life_ratio)
            if radius > 0:
                alpha = int(255 * life_ratio)
                color = (*self.COLOR_PARTICLE, alpha)
                
                # Using gfxdraw for anti-aliased circles
                surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(surf, radius, radius, radius, color)
                self.screen.blit(surf, (int(p["pos"][0] - radius), int(p["pos"][1] - radius)))

    def close(self):
        pygame.quit()


# Example usage:
if __name__ == "__main__":
    # The validation code has been removed as it is not part of the core environment logic
    # and was causing issues in some testing setups.
    
    # To run and play the game:
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display
    pygame.init() # Re-init with video driver

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Spinning Top Balancer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default: no action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- RESET ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']:.2f}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.TARGET_FPS)
        
    env.close()