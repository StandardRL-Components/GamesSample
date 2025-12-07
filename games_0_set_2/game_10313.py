import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:10:51.916549
# Source Brief: brief_00313.md
# Brief Index: 313
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent must maintain the size of four platforms.
    Each platform constantly shrinks. The agent can use actions to grow or shrink
    specific platforms to keep them all above a failure threshold.

    - Visuals: Minimalist geometric style with smooth animations and clear status indicators.
    - Gameplay: Real-time resource management under time pressure.
    - RL Challenge: Balancing actions across four independent but interconnected state variables.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Maintain the size of four constantly shrinking platforms. Use your actions to grow the platforms and prevent them from collapsing under time pressure."
    )
    user_guide = (
        "Controls: Use ↑ to grow the top platform and ← to grow the left. Be careful, as other actions (↓, →) will shrink the other platforms."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 90
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 45, 60)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_GOOD = (70, 220, 120)
    COLOR_WARN = (240, 190, 80)
    COLOR_FAIL = (220, 70, 90)
    COLOR_PULSE = (255, 255, 255)

    # Game Mechanics
    PLATFORM_MAX_SIZE_PX = 120
    SHRINK_RATE_PER_SEC = 0.01
    GROWTH_AMOUNT = 0.05
    SHRINK_PENALTY = 0.15
    FAILURE_THRESHOLD = 0.5
    WARNING_THRESHOLD = 0.75
    LERP_FACTOR = 0.2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # [Top, Bottom, Left, Right]
        self.platform_target_sizes = [1.0] * 4
        self.platform_visual_sizes = [1.0] * 4
        self.platform_positions = [
            (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.28),
            (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.72),
            (self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT / 2),
            (self.SCREEN_WIDTH * 0.75, self.SCREEN_HEIGHT / 2)
        ]
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.platform_target_sizes = [1.0, 1.0, 1.0, 1.0]
        self.platform_visual_sizes = [1.0, 1.0, 1.0, 1.0]
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]

        # --- Store pre-action state for reward calculation ---
        old_sizes = self.platform_target_sizes[:]
        
        # --- Update Game Logic ---
        # 1. Natural Shrinking
        shrink_per_frame = self.SHRINK_RATE_PER_SEC / self.FPS
        for i in range(4):
            self.platform_target_sizes[i] -= shrink_per_frame

        # 2. Player Action
        action_taken = True
        if movement == 1: # Up -> Increase Top
            self.platform_target_sizes[0] += self.GROWTH_AMOUNT
            self._create_pulse_effect(0)
        elif movement == 2: # Down -> Decrease Bottom
            self.platform_target_sizes[1] -= self.GROWTH_AMOUNT
        elif movement == 3: # Left -> Increase Left
            self.platform_target_sizes[2] += self.GROWTH_AMOUNT
            self._create_pulse_effect(2)
        elif movement == 4: # Right -> Decrease Right
            self.platform_target_sizes[3] -= self.GROWTH_AMOUNT
        else: # No-op
            action_taken = False
        
        # 3. Clamp sizes
        for i in range(4):
            self.platform_target_sizes[i] = min(1.0, max(0.0, self.platform_target_sizes[i]))

        # --- Termination and Penalty Check ---
        terminated = False
        failure = False
        penalty_applied = False
        for i in range(4):
            if self.platform_target_sizes[i] < self.FAILURE_THRESHOLD:
                failure = True
                if not penalty_applied:
                    # Sound: Failure_Penalty.wav
                    for j in range(4):
                        if i != j:
                            self.platform_target_sizes[j] -= self.SHRINK_PENALTY
                    penalty_applied = True
        
        if failure:
            terminated = True
        
        truncated = False
        if self.steps >= self.MAX_STEPS - 1:
            terminated = True # Game ends due to time limit, but it's a win condition if not failed.
            truncated = False # Not truncated, but terminated.

        # --- Calculate Reward ---
        reward = 0.0
        
        # Living reward
        if not failure:
            reward += 1.0
        
        # Event-based rewards
        if movement == 1 and old_sizes[0] < self.WARNING_THRESHOLD:
            reward += 5.0 # Rewarded for saving the Top platform
        if movement == 3 and old_sizes[2] < self.WARNING_THRESHOLD:
            reward += 5.0 # Rewarded for saving the Left platform

        # Goal-oriented rewards
        if terminated:
            if failure:
                reward -= 100.0 # Penalty for any platform failing
            else: # Reached time limit successfully
                reward += 100.0 # Bonus for winning
        
        # --- Update State ---
        self.score += reward
        self.steps += 1
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        # --- Smooth animations ---
        for i in range(4):
            self.platform_visual_sizes[i] += (self.platform_target_sizes[i] - self.platform_visual_sizes[i]) * self.LERP_FACTOR

        # --- Rendering ---
        self._render_background()
        self._render_particles()
        self._render_platforms()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_platforms(self):
        for i in range(4):
            size_perc = self.platform_visual_sizes[i]
            px_size = max(0, int(size_perc * self.PLATFORM_MAX_SIZE_PX))
            pos = self.platform_positions[i]
            
            # Determine color
            if self.platform_target_sizes[i] < self.FAILURE_THRESHOLD:
                color = self.COLOR_FAIL
            elif self.platform_target_sizes[i] < self.WARNING_THRESHOLD:
                color = self.COLOR_WARN
            else:
                color = self.COLOR_GOOD

            # Draw platform
            rect = pygame.Rect(0, 0, px_size, px_size)
            rect.center = (int(pos[0]), int(pos[1]))
            
            # Outline for depth
            outline_rect = rect.inflate(6, 6)
            pygame.draw.rect(self.screen, (0,0,0,50), outline_rect, border_radius=12)
            
            pygame.draw.rect(self.screen, color, rect, border_radius=10)
            pygame.gfxdraw.rectangle(self.screen, rect, (255,255,255,50)) # subtle highlight

            # Draw text
            text = f"{int(self.platform_target_sizes[i] * 100)}%"
            self._draw_text(text, rect.center, self.font_medium, self.COLOR_TEXT)

    def _render_ui(self):
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        self._draw_text(timer_text, (self.SCREEN_WIDTH - 10, 10), self.font_large, self.COLOR_TEXT, align="topright")
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (10, 10), self.font_large, self.COLOR_TEXT, align="topleft")
        
        # Platform labels
        labels = ["UP", "DOWN", "LEFT", "RIGHT"]
        for i, pos in enumerate(self.platform_positions):
            offset = (0, self.PLATFORM_MAX_SIZE_PX / 2 + 20)
            label_pos = (pos[0] + offset[0], pos[1] + offset[1])
            self._draw_text(labels[i], label_pos, self.font_small, self.COLOR_GRID)

    def _render_particles(self):
        for p in self.particles[:]:
            p[0] -= 1 # lifetime
            if p[0] <= 0:
                self.particles.remove(p)
                continue
            
            p[2] += p[3] # radius update
            alpha = int(255 * (p[0] / p[1]))
            if alpha <= 0: continue

            color = (*p[4], alpha)
            pos = self.platform_positions[p[5]]
            
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(p[2]), color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "platform_sizes": self.platform_target_sizes,
        }

    def _create_pulse_effect(self, platform_index):
        # Sound: Grow_Pulse.wav
        # [lifetime, max_lifetime, radius, radius_vel, color, platform_idx]
        lifetime = 15 # frames
        self.particles.append([lifetime, lifetime, self.PLATFORM_MAX_SIZE_PX/2, 2, self.COLOR_PULSE, platform_index])

    def _draw_text(self, text, pos, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surface.get_rect()
        
        if align == "center":
            text_rect.center = pos
        elif align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        
        self.screen.blit(text_shadow, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    # Set a real video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Platform Maintenance")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("W: Grow Top    | S: Shrink Bottom")
    print("A: Grow Left   | D: Shrink Right")
    print("Space: No-op")
    print("Q: Quit")
    
    while not done:
        # --- Action Mapping for Human ---
        movement_action = 0 # no-op
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True
                continue
        
        if done:
            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: movement_action = 1
        elif keys[pygame.K_s]: movement_action = 2
        elif keys[pygame.K_a]: movement_action = 3
        elif keys[pygame.K_d]: movement_action = 4
        
        action = [movement_action, 0, 0] # space/shift not used

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Pygame Rendering ---
        # Convert observation back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print(f"\nGame Over!")
    print(f"Final Score: {total_reward:.2f}")
    print(f"Info: {info}")
    
    env.close()