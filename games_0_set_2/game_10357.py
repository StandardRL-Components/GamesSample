import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:15:04.561979
# Source Brief: brief_00357.md
# Brief Index: 357
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Swing a pendulum using magnetic pulses to collect gems. Grab all the gems to win, but don't let the pendulum stop swinging!"
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to apply magnetic pulses. Hold space for a stronger pulse."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    VICTORY_GEMS = 15
    MAX_STEPS = 5000

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (20, 40, 80)
    COLOR_PLATFORM = (100, 180, 255)
    COLOR_GEM = (255, 220, 0)
    COLOR_PENDULUM_ARM = (150, 150, 170)
    COLOR_PENDULUM_BOB = (255, 255, 255)
    COLOR_PULSE = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_UI_BAR_BG = (50, 50, 90)
    COLOR_UI_BAR_FG = (100, 200, 255)

    # Physics
    GRAVITY = 0.002
    INITIAL_FRICTION_MULTIPLIER = 0.995
    PENDULUM_LENGTH = 180
    PENDULUM_BOB_RADIUS = 12
    GEM_RADIUS = 8
    PLATFORM_HEIGHT = 10
    
    PULSE_WEAK = 0.004
    PULSE_STRONG = 0.008

    # Difficulty Scaling
    INITIAL_PLATFORM_SPEED = 1.0
    PLATFORM_SPEED_INCREASE_INTERVAL = 250
    PLATFORM_SPEED_INCREASE_AMOUNT = 0.05
    FRICTION_INCREASE_INTERVAL = 500
    FRICTION_DECREASE_AMOUNT = 0.0005 # Multiplier decreases, so friction increases
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        self.pivot_pos = (self.SCREEN_WIDTH // 2, 80)
        self.bg_surface = self._create_gradient_background()

        # These will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gems_collected = 0
        self.platform_speed = 0.0
        self.friction_multiplier = 0.0
        
        self.pendulum_angle = 0.0
        self.pendulum_ang_vel = 0.0
        
        self.platforms = []
        self.particles = []
        self.pulse_effect_timer = 0
        
        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # validation is not needed in the final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gems_collected = 0
        
        self.platform_speed = self.INITIAL_PLATFORM_SPEED
        self.friction_multiplier = self.INITIAL_FRICTION_MULTIPLIER
        
        self.pendulum_angle = math.pi * 0.75
        self.pendulum_ang_vel = 0.0
        
        self.platforms = self._initialize_platforms()
        self.particles = []
        self.pulse_effect_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Get previous state for reward calculation ---
        prev_bob_pos = self._get_bob_position()

        # --- 2. Apply action and update physics ---
        self._update_physics(action)
        self._update_platforms()
        
        # --- 3. Handle interactions and state changes ---
        collision_reward = self._check_collisions()
        self.score += collision_reward

        # --- 4. Update difficulty ---
        self._update_difficulty()
        
        # --- 5. Calculate reward ---
        reward = self._calculate_reward(prev_bob_pos, collision_reward)

        # --- 6. Check for termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.gems_collected >= self.VICTORY_GEMS:
                reward += 100
                self.score += 100
            else: # Pendulum stopped or max steps
                reward -= 10
                self.score -= 10
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_physics(self, action):
        movement, space_held, _ = action
        
        # Apply magnetic pulse
        pulse_direction = 0
        if movement == 3:  # Left
            pulse_direction = -1
        elif movement == 4:  # Right
            pulse_direction = 1
            
        if pulse_direction != 0:
            pulse_strength = self.PULSE_STRONG if space_held == 1 else self.PULSE_WEAK
            impulse = pulse_direction * pulse_strength
            self.pendulum_ang_vel += impulse
            self.pulse_effect_timer = 10 # frames
            # sfx: magnetic_pulse.wav

        # Apply gravity
        angular_acceleration = -self.GRAVITY * math.sin(self.pendulum_angle) / (self.PENDULUM_LENGTH / 100)
        self.pendulum_ang_vel += angular_acceleration
        
        # Apply friction
        self.pendulum_ang_vel *= self.friction_multiplier
        
        # Update angle
        self.pendulum_angle += self.pendulum_ang_vel

    def _update_platforms(self):
        for i, p in enumerate(self.platforms):
            # Two sets of platforms alternate visibility
            is_set_a = i % 2 == 0
            is_visible_now = (self.steps // 60) % 2 == 0 if is_set_a else (self.steps // 60) % 2 != 0
            p['is_visible'] = is_visible_now
            
            # Oscillate
            p['pos'][0] = p['center_x'] + p['amplitude'] * math.sin(self.steps * 0.01 * self.platform_speed + p['phase'])
            
            # Update gem position if it exists
            if not p['gem_collected']:
                p['gem_pos'] = (p['pos'][0], p['pos'][1] - self.PLATFORM_HEIGHT / 2 - self.GEM_RADIUS)

    def _check_collisions(self):
        bob_pos = self._get_bob_position()
        reward = 0
        
        for p in self.platforms:
            if p['is_visible'] and not p['gem_collected']:
                dist = math.hypot(bob_pos[0] - p['gem_pos'][0], bob_pos[1] - p['gem_pos'][1])
                if dist < self.PENDULUM_BOB_RADIUS + self.GEM_RADIUS:
                    p['gem_collected'] = True
                    self.gems_collected += 1
                    reward += 1.0
                    self._create_particles(p['gem_pos'])
                    # sfx: gem_collect.wav
        return reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.PLATFORM_SPEED_INCREASE_INTERVAL == 0:
            self.platform_speed += self.PLATFORM_SPEED_INCREASE_AMOUNT
        if self.steps > 0 and self.steps % self.FRICTION_INCREASE_INTERVAL == 0:
            self.friction_multiplier = max(0.95, self.friction_multiplier - self.FRICTION_DECREASE_AMOUNT)

    def _calculate_reward(self, prev_bob_pos, collision_reward):
        # Continuous reward for moving towards the nearest gem
        continuous_reward = 0
        
        bob_pos = self._get_bob_position()
        active_gems = [p['gem_pos'] for p in self.platforms if p['is_visible'] and not p['gem_collected']]
        
        if active_gems:
            # Find closest gem
            distances = [math.hypot(bob_pos[0] - g[0], bob_pos[1] - g[1]) for g in active_gems]
            min_dist_idx = np.argmin(distances)
            closest_gem_pos = active_gems[min_dist_idx]
            
            # Calculate distance change
            prev_dist_to_gem = math.hypot(prev_bob_pos[0] - closest_gem_pos[0], prev_bob_pos[1] - closest_gem_pos[1])
            current_dist_to_gem = distances[min_dist_idx]
            
            if prev_dist_to_gem > current_dist_to_gem: # Moved closer
                continuous_reward = 0.1
            else: # Moved away or stayed same
                continuous_reward = -0.01

        return collision_reward + continuous_reward

    def _check_termination(self):
        if self.gems_collected >= self.VICTORY_GEMS:
            # sfx: victory.wav
            return True
        
        # Check if pendulum has stopped near the bottom
        is_stopped = abs(self.pendulum_ang_vel) < 0.0005
        is_at_bottom = abs(math.sin(self.pendulum_angle)) < 0.01
        if is_stopped and is_at_bottom:
            # sfx: failure.wav
            return True
            
        return False

    def _get_observation(self):
        # --- Render all game elements to self.screen ---
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
        }

    def _render_game(self):
        self._draw_platforms_and_gems()
        self._draw_pulse_effect()
        self._draw_pendulum()
        self._update_and_draw_particles()

    def _render_ui(self):
        # Gem Count
        gem_text = self.font_main.render(f"GEMS: {self.gems_collected}/{self.VICTORY_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (10, 10))
        
        # Swing Strength
        strength_label = self.font_small.render("SWING", True, self.COLOR_TEXT)
        self.screen.blit(strength_label, (10, 45))
        
        bar_width = 150
        bar_height = 15
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 65, bar_width, bar_height))
        
        # Velocity is max around +/- 0.15 in practice
        swing_strength = min(1.0, abs(self.pendulum_ang_vel) / 0.15)
        filled_width = int(bar_width * swing_strength)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FG, (10, 65, filled_width, bar_height))

    def _get_bob_position(self):
        x = self.pivot_pos[0] + self.PENDULUM_LENGTH * math.sin(self.pendulum_angle)
        y = self.pivot_pos[1] + self.PENDULUM_LENGTH * math.cos(self.pendulum_angle)
        return (x, y)

    def _draw_pendulum(self):
        bob_pos = self._get_bob_position()
        
        # Arm
        pygame.draw.aaline(self.screen, self.COLOR_PENDULUM_ARM, self.pivot_pos, bob_pos, 2)
        
        # Bob Glow
        for i in range(self.PENDULUM_BOB_RADIUS, 0, -2):
            alpha = 80 * (1 - i / self.PENDULUM_BOB_RADIUS)
            pygame.gfxdraw.filled_circle(
                self.screen, int(bob_pos[0]), int(bob_pos[1]), i + 5,
                (*self.COLOR_PENDULUM_BOB, int(alpha))
            )
        
        # Bob Core
        pygame.gfxdraw.filled_circle(self.screen, int(bob_pos[0]), int(bob_pos[1]), self.PENDULUM_BOB_RADIUS, self.COLOR_PENDULUM_BOB)
        pygame.gfxdraw.aacircle(self.screen, int(bob_pos[0]), int(bob_pos[1]), self.PENDULUM_BOB_RADIUS, self.COLOR_PENDULUM_BOB)

    def _draw_platforms_and_gems(self):
        for p in self.platforms:
            if p['is_visible']:
                # Platform
                rect = pygame.Rect(p['pos'][0] - p['width'] / 2, p['pos'][1] - self.PLATFORM_HEIGHT / 2, p['width'], self.PLATFORM_HEIGHT)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, rect, border_radius=3)
                
                # Gem
                if not p['gem_collected']:
                    gem_pos_int = (int(p['gem_pos'][0]), int(p['gem_pos'][1]))
                    # Gem body
                    pygame.gfxdraw.filled_circle(self.screen, *gem_pos_int, self.GEM_RADIUS, self.COLOR_GEM)
                    pygame.gfxdraw.aacircle(self.screen, *gem_pos_int, self.GEM_RADIUS, self.COLOR_GEM)
                    # Sparkle effect
                    sparkle_x = gem_pos_int[0] + self.np_random.integers(-3, 4)
                    sparkle_y = gem_pos_int[1] + self.np_random.integers(-3, 4)
                    pygame.draw.circle(self.screen, (255, 255, 255), (sparkle_x, sparkle_y), 1)

    def _draw_pulse_effect(self):
        if self.pulse_effect_timer > 0:
            max_radius = 40
            radius = int(max_radius * (1 - self.pulse_effect_timer / 10))
            alpha = int(200 * (self.pulse_effect_timer / 10))
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, int(self.pivot_pos[0]), int(self.pivot_pos[1]), radius, (*self.COLOR_PULSE, alpha))
            self.pulse_effect_timer -= 1
            
    def _create_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 31)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifespan, 'max_life': lifespan})
    
    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # particle gravity
            p['life'] -= 1
            
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = 255 * (p['life'] / p['max_life'])
                color = (*self.COLOR_GEM, alpha)
                pos_int = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, *pos_int, 2, color)

    def _initialize_platforms(self):
        platforms = []
        num_platforms = 6
        y_positions = [200, 250, 300, 350]
        for i in range(num_platforms):
            y_pos = self.np_random.choice(y_positions) + self.np_random.integers(-10, 11)
            center_x = self.SCREEN_WIDTH / (num_platforms + 1) * (i + 1)
            amplitude = self.np_random.integers(50, 151)
            phase = self.np_random.uniform(0, 2 * math.pi)
            width = self.np_random.integers(60, 101)
            
            p = {
                'pos': [center_x, y_pos],
                'center_x': center_x,
                'amplitude': amplitude,
                'phase': phase,
                'width': width,
                'is_visible': False,
                'gem_collected': False,
                'gem_pos': (0, 0)
            }
            platforms.append(p)
        return platforms

    def _create_gradient_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            r = int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio)
            g = int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio)
            b = int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            pygame.draw.line(bg, (r, g, b), (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not be run by the autograder, but is useful for testing
    
    # Un-comment the following line to run in a window
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pendulum Gems")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # --- Main Game Loop ---
    while not terminated and not truncated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]
        
        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        truncated = trunc
        total_reward += reward
        
        # --- Display the observation from the environment ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # and surfarray.make_surface expects (W, H, C)
        obs_transposed = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(obs_transposed)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")

    env.close()