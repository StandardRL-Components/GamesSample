import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:29:33.500676
# Source Brief: brief_01729.md
# Brief Index: 1729
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Balance a tilting platform by adjusting the speeds of three constantly oscillating gears. "
        "Survive for 45 seconds to win."
    )
    user_guide = (
        "Controls: Use ↑↓ for the left gear, ←→ for the middle gear, and space/shift for the right gear. "
        "Keep the platform level!"
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (30, 40, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_STABLE = (0, 255, 150)
    COLOR_WARN = (255, 255, 0)
    COLOR_UNSTABLE = (255, 50, 50)
    GEAR_COLORS = [(0, 200, 255), (255, 100, 255), (255, 200, 0)]

    # Game Parameters
    MAX_TILT_ANGLE = 20.0  # degrees
    WIN_TIME_SECONDS = 45.0
    MAX_STEPS = int(WIN_TIME_SECONDS * FPS) + 300 # 50 seconds total

    # Physics
    PLATFORM_TILT_CONSTANT = 3.0
    PLATFORM_DAMPING = 0.1
    USER_ADJUSTMENT_STEP = 0.1
    USER_ADJUSTMENT_DECAY = 0.98
    USER_ADJUSTMENT_MAX = 2.0
    GEAR_SPEED_MULTIPLIER = 5.0 # Visual rotation speed

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        self.gears = []
        self.platform_angle = 0.0
        self.target_platform_angle = 0.0
        self.time_elapsed_seconds = 0.0
        self.oscillation_amplitude = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Action feedback indicators
        self.action_feedback = []

        # The initial reset is now handled by the environment runner
        # self.reset()
        # self.validate_implementation() # This is for dev, not production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed_seconds = 0.0

        self.platform_angle = 0.0
        self.target_platform_angle = 0.0
        self.oscillation_amplitude = 0.20

        self.gears = []
        base_speeds_rpm = [1.0, 1.5, 1.0] # Modified for better balance gameplay
        osc_freqs = [0.015, 0.02, 0.025] # Different frequencies
        positions = [(160, 280), (320, 280), (480, 280)]
        radii = [60, 70, 60]

        for i in range(3):
            self.gears.append({
                "pos": positions[i],
                "radius": radii[i],
                "base_speed_rpm": base_speeds_rpm[i],
                "user_adjustment": 0.0,
                "current_speed": base_speeds_rpm[i],
                "current_angle": self.np_random.uniform(0, 360),
                "color": self.GEAR_COLORS[i],
                "osc_freq": osc_freqs[i] + self.np_random.uniform(-0.002, 0.002)
            })
            
        self.action_feedback = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_elapsed_seconds = self.steps / self.FPS
        
        self.action_feedback = [] # Clear feedback each step

        # --- 1. Apply Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Decay user adjustments for dynamic control
        for gear in self.gears:
            gear["user_adjustment"] *= self.USER_ADJUSTMENT_DECAY

        # Action Mapping: Movement for Gears 1 & 2, Space/Shift for Gear 3
        adjustment_step = self.USER_ADJUSTMENT_STEP
        
        # Gear 1 (index 0)
        if movement == 1: # Up
            self.gears[0]["user_adjustment"] += adjustment_step * self.gears[0]["base_speed_rpm"]
            self.action_feedback.append({'gear': 0, 'type': 'increase'})
            # sfx: gear_increase_short
        elif movement == 2: # Down
            self.gears[0]["user_adjustment"] -= adjustment_step * self.gears[0]["base_speed_rpm"]
            self.action_feedback.append({'gear': 0, 'type': 'decrease'})
            # sfx: gear_decrease_short

        # Gear 2 (index 1)
        if movement == 3: # Left
            self.gears[1]["user_adjustment"] -= adjustment_step * self.gears[1]["base_speed_rpm"]
            self.action_feedback.append({'gear': 1, 'type': 'decrease'})
            # sfx: gear_decrease_short
        elif movement == 4: # Right
            self.gears[1]["user_adjustment"] += adjustment_step * self.gears[1]["base_speed_rpm"]
            self.action_feedback.append({'gear': 1, 'type': 'increase'})
            # sfx: gear_increase_short

        # Gear 3 (index 2)
        if space_held:
            self.gears[2]["user_adjustment"] += adjustment_step * self.gears[2]["base_speed_rpm"]
            self.action_feedback.append({'gear': 2, 'type': 'increase'})
            # sfx: gear_increase_sustained
        if shift_held:
            self.gears[2]["user_adjustment"] -= adjustment_step * self.gears[2]["base_speed_rpm"]
            self.action_feedback.append({'gear': 2, 'type': 'decrease'})
            # sfx: gear_decrease_sustained
            
        for gear in self.gears:
            gear["user_adjustment"] = np.clip(gear["user_adjustment"], -self.USER_ADJUSTMENT_MAX, self.USER_ADJUSTMENT_MAX)

        # --- 2. Update Game State ---
        # Increase oscillation amplitude over time for difficulty scaling
        self.oscillation_amplitude = 0.20 + 0.02 * (self.time_elapsed_seconds // 10)

        # Update gear speeds and angles
        for gear in self.gears:
            oscillation = self.oscillation_amplitude * math.sin(self.steps * gear["osc_freq"])
            effective_speed = gear["base_speed_rpm"] * (1 + oscillation) + gear["user_adjustment"]
            gear["current_speed"] = max(0.1, effective_speed) # Prevent stalling/reversing
            gear["current_angle"] = (gear["current_angle"] + gear["current_speed"] * self.GEAR_SPEED_MULTIPLIER) % 360

        # Update platform tilt based on speed difference of outer gears
        speed_diff = self.gears[2]["current_speed"] - self.gears[0]["current_speed"]
        self.target_platform_angle = np.clip(speed_diff * self.PLATFORM_TILT_CONSTANT, -self.MAX_TILT_ANGLE * 1.5, self.MAX_TILT_ANGLE * 1.5)
        self.platform_angle += (self.target_platform_angle - self.platform_angle) * self.PLATFORM_DAMPING

        # --- 3. Calculate Reward & Termination ---
        reward = -0.1 * abs(self.platform_angle)
        terminated = False
        truncated = False

        win = self.time_elapsed_seconds >= self.WIN_TIME_SECONDS
        loss = abs(self.platform_angle) > self.MAX_TILT_ANGLE
        timeout = self.steps >= self.MAX_STEPS

        if win:
            terminated = True
            reward += 100.0
            self.score = 100
            # sfx: win_chime
        elif loss:
            terminated = True
            reward -= 100.0
            self.score = -100
            # sfx: fail_buzzer
        elif timeout:
            truncated = True # Use truncated for timeout
            # sfx: timeout_sound
        
        self.game_over = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        self._render_platform()
        for i, gear in enumerate(self.gears):
            self._draw_gear(
                self.screen,
                gear["pos"],
                gear["radius"],
                gear["current_angle"],
                gear["color"]
            )
        self._render_action_feedback()

    def _render_platform(self):
        # Determine platform color based on tilt
        tilt_ratio = min(1.0, abs(self.platform_angle) / self.MAX_TILT_ANGLE)
        if tilt_ratio < 0.5:
            color = self._lerp_color(self.COLOR_STABLE, self.COLOR_WARN, tilt_ratio * 2)
        else:
            color = self._lerp_color(self.COLOR_WARN, self.COLOR_UNSTABLE, (tilt_ratio - 0.5) * 2)

        # Create platform surface and rotate it
        platform_width, platform_height = 500, 12
        platform_surf = pygame.Surface((platform_width, platform_height), pygame.SRCALPHA)
        pygame.draw.rect(platform_surf, color, (0, 0, platform_width, platform_height), border_radius=6)
        pygame.draw.rect(platform_surf, self._scale_color(color, 1.2), (0, 0, platform_width, platform_height), 2, border_radius=6)

        pivot_pos = (self.SCREEN_WIDTH // 2, 100)
        rotated_surf = pygame.transform.rotate(platform_surf, self.platform_angle)
        rotated_rect = rotated_surf.get_rect(center=pivot_pos)

        self.screen.blit(rotated_surf, rotated_rect)
        
        # Draw pivot
        pygame.gfxdraw.filled_circle(self.screen, pivot_pos[0], pivot_pos[1], 8, self.COLOR_UI_TEXT)
        pygame.gfxdraw.aacircle(self.screen, pivot_pos[0], pivot_pos[1], 8, self.COLOR_UI_TEXT)

    def _draw_gear(self, surface, pos, radius, angle, color):
        # Glow effect
        for i in range(10, 0, -2):
            glow_color = color + (i * 5,)
            pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), radius + i, glow_color)

        # Main body
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), radius, color)
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), radius, self._scale_color(color, 1.2))

        # Inner details
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius * 0.8), self.COLOR_BG)
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), int(radius * 0.8), color)
        
        # Spokes
        num_spokes = 6
        for i in range(num_spokes):
            rad = math.radians(angle + i * (360 / num_spokes))
            start_pos = (
                pos[0] + math.cos(rad) * radius * 0.2,
                pos[1] + math.sin(rad) * radius * 0.2
            )
            end_pos = (
                pos[0] + math.cos(rad) * radius * 0.75,
                pos[1] + math.sin(rad) * radius * 0.75
            )
            pygame.draw.line(surface, color, start_pos, end_pos, 4)

        # Center pin
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius * 0.2), self._scale_color(color, 1.2))
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), int(radius * 0.2), self.COLOR_BG)
        
    def _render_action_feedback(self):
        for fb in self.action_feedback:
            gear = self.gears[fb['gear']]
            pos = gear['pos']
            color = (255, 255, 255, 150)
            if fb['type'] == 'increase':
                points = [(pos[0], pos[1] - gear['radius'] - 20),
                          (pos[0] - 10, pos[1] - gear['radius'] - 10),
                          (pos[0] + 10, pos[1] - gear['radius'] - 10)]
            else: # decrease
                points = [(pos[0], pos[1] + gear['radius'] + 20),
                          (pos[0] - 10, pos[1] + gear['radius'] + 10),
                          (pos[0] + 10, pos[1] + gear['radius'] + 10)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer bar
        timer_ratio = min(1.0, self.time_elapsed_seconds / self.WIN_TIME_SECONDS)
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 15
        bar_x, bar_y = 10, self.SCREEN_HEIGHT - bar_height - 10
        
        # Background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        # Foreground
        fill_width = bar_width * timer_ratio
        if fill_width > 0:
             pygame.draw.rect(self.screen, self.COLOR_STABLE, (bar_x, bar_y, fill_width, bar_height), border_radius=4)
        
        # Timer text
        timer_text = self.font_small.render(f"{self.time_elapsed_seconds:.1f}s / {self.WIN_TIME_SECONDS:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (bar_x + bar_width - timer_text.get_width() - 5, bar_y - 20))
        
        # Platform tilt angle
        tilt_text = self.font_main.render(f"{self.platform_angle:.2f}°", True, self.COLOR_UI_TEXT)
        text_rect = tilt_text.get_rect(center=(self.SCREEN_WIDTH // 2, 50))
        self.screen.blit(tilt_text, text_rect)
        
        # Gear speeds
        for gear in self.gears:
            speed_text = self.font_small.render(f"{gear['current_speed']:.2f} RPM", True, gear['color'])
            text_rect = speed_text.get_rect(center=(gear['pos'][0], gear['pos'][1] + gear['radius'] + 20))
            self.screen.blit(speed_text, text_rect)

    def close(self):
        pygame.quit()

    # --- Helper Functions ---
    def _lerp_color(self, c1, c2, t):
        t = np.clip(t, 0.0, 1.0)
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )

    def _scale_color(self, color, factor):
        return tuple(min(255, int(c * factor)) for c in color[:3])

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    # For manual play, we need a real display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Gear Balancer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    terminated = False
    truncated = False
    
    while running:
        # --- Human Input ---
        movement = 0 # None
        space = 0 # Released
        shift = 0 # Released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                truncated = False

        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        if terminated or truncated:
            font = pygame.font.SysFont("Consolas", 50, bold=True)
            if terminated:
                msg = "YOU WON!" if info['score'] > 0 else "YOU LOST"
                color = GameEnv.COLOR_STABLE if info['score'] > 0 else GameEnv.COLOR_UNSTABLE
            else: # truncated
                msg = "TIMEOUT"
                color = GameEnv.COLOR_WARN

            text = font.render(msg, True, color)
            text_rect = text.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 - 30))
            screen.blit(text, text_rect)
            
            font_small = pygame.font.SysFont("Consolas", 20)
            text_restart = font_small.render("Press 'R' to restart", True, GameEnv.COLOR_UI_TEXT)
            text_restart_rect = text_restart.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 + 30))
            screen.blit(text_restart, text_restart_rect)

        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()