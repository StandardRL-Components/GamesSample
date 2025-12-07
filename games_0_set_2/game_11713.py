import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:27:41.226224
# Source Brief: brief_01713.md
# Brief Index: 1713
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a color-changing chameleon.
    The goal is to match the chameleon's color to the shifting background to avoid
    detection by a scanner. Survival for a set duration leads to victory.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (Unused)
    - action[1]: Space Button (0=released, 1=held) -> Change to RED
    - action[2]: Shift Button (0=released, 1=held) -> Change to GREEN
    - (Space + Shift held) -> Change to BLUE

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +1 per step for being camouflaged (mismatch < 70%).
    - -1 per step for being exposed (mismatch >= 70%).
    - +5 for a perfect color match.
    - +100 for winning (surviving 900 steps).
    - -100 for losing (being detected).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a chameleon and change its color to match the shifting background. "
        "Evade the sweeping scanner to survive and win."
    )
    user_guide = (
        "Controls: Press space to turn RED, shift to turn GREEN, and both space+shift to turn BLUE. "
        "Match the background color to avoid detection!"
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30 # Affects visual smoothness, not game logic speed

    # Game parameters
    MAX_STEPS = 1000
    WIN_STEPS = 900
    BG_CHANGE_INTERVAL = 225  # 15 seconds at 15 steps/sec, as per brief
    SCANNER_SWEEP_STEPS = 50
    DETECTION_THRESHOLD = 0.70

    # Colors
    COLOR_RED = (220, 38, 38)
    COLOR_GREEN = (5, 150, 105)
    COLOR_BLUE = (37, 99, 235)
    PALETTE = [COLOR_RED, COLOR_GREEN, COLOR_BLUE]

    COLOR_BG = (15, 23, 42) # Dark blue-gray background
    COLOR_WHITE = (245, 245, 245)
    COLOR_SCANNER = (239, 68, 68)
    COLOR_SCANNER_GLOW = (239, 68, 68, 30)
    COLOR_AURA_SAFE = (200, 200, 255)
    COLOR_AURA_DANGER = (255, 50, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20, 200)
    COLOR_UI_BG = (10, 10, 10, 150)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        self.render_mode = render_mode

        # Initialize state variables
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_condition_met = False
        self.chameleon_pos = (0,0)
        self.chameleon_radius = 0
        self.chameleon_color = (0,0,0)
        self.chameleon_target_color = (0,0,0)
        self.background_color = (0,0,0)
        self.scanner_y = 0
        self.color_mismatch = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_condition_met = False

        self.chameleon_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        self.chameleon_radius = 30
        
        # Start as blue, as per brief
        self.chameleon_target_color = self.COLOR_BLUE
        self.chameleon_color = self.chameleon_target_color

        self.background_color = random.choice(self.PALETTE)
        self.scanner_y = 0
        self.color_mismatch = self._calculate_mismatch()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_action(action)
        self._update_game_state()

        terminated = self._check_termination()
        reward = self._calculate_reward(terminated)
        self.score += reward
        
        if terminated:
            self.game_over = True
            # Sound: Game Over or Victory sound

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        # movement = action[0] # Unused as per brief
        space_held = action[1] == 1
        shift_held = action[2] == 1

        color_changed = False
        if space_held and shift_held:
            if self.chameleon_target_color != self.COLOR_BLUE:
                self.chameleon_target_color = self.COLOR_BLUE
                color_changed = True
        elif space_held:
            if self.chameleon_target_color != self.COLOR_RED:
                self.chameleon_target_color = self.COLOR_RED
                color_changed = True
        elif shift_held:
            if self.chameleon_target_color != self.COLOR_GREEN:
                self.chameleon_target_color = self.COLOR_GREEN
                color_changed = True
        
        if color_changed:
            pass # Sound: Color change swoosh

    def _update_game_state(self):
        self.steps += 1
        
        # Smoothly interpolate chameleon color
        self.chameleon_color = self._lerp_color(
            self.chameleon_color, self.chameleon_target_color, 0.25
        )

        # Update background color at interval
        if self.steps > 0 and self.steps % self.BG_CHANGE_INTERVAL == 0:
            current_bg_idx = self.PALETTE.index(self.background_color)
            next_bg_idx = (current_bg_idx + self.np_random.integers(1, len(self.PALETTE))) % len(self.PALETTE)
            self.background_color = self.PALETTE[next_bg_idx]
            # Sound: Background shift chime

        # Update scanner position
        self.scanner_y = (self.steps % self.SCANNER_SWEEP_STEPS) / (self.SCANNER_SWEEP_STEPS -1) * self.HEIGHT

        # Update color mismatch
        self.color_mismatch = self._calculate_mismatch()

    def _calculate_mismatch(self):
        c1 = self.chameleon_color
        c2 = self.background_color
        diff_r = abs(c1[0] - c2[0])
        diff_g = abs(c1[1] - c2[1])
        diff_b = abs(c1[2] - c2[2])
        total_diff = diff_r + diff_g + diff_b
        max_diff = 255 * 3
        return total_diff / max_diff if max_diff > 0 else 0

    def _check_termination(self):
        if self.color_mismatch > self.DETECTION_THRESHOLD:
            self.win_condition_met = False
            return True
        if self.steps >= self.WIN_STEPS:
            self.win_condition_met = True
            return True
        return False

    def _calculate_reward(self, terminated):
        reward = 0.0

        if self.color_mismatch < self.DETECTION_THRESHOLD:
            reward += 1.0
        else:
            reward -= 1.0
        
        if self.color_mismatch < 0.01: # Perfect match
            reward += 5.0

        if terminated:
            if self.win_condition_met:
                reward += 100.0
            else: # Detected or timed out
                reward -= 100.0
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.background_color)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render scanner
        scan_y_int = int(self.scanner_y)
        pygame.draw.line(self.screen, self.COLOR_SCANNER, (0, scan_y_int), (self.WIDTH, scan_y_int), 2)
        glow_rect = pygame.Rect(0, scan_y_int - 5, self.WIDTH, 10)
        glow_surf = pygame.Surface((self.WIDTH, 10), pygame.SRCALPHA)
        glow_surf.fill(self.COLOR_SCANNER_GLOW)
        self.screen.blit(glow_surf, (0, scan_y_int - 5))
        
        # Render detection aura
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        base_aura_radius = self.chameleon_radius + 10
        aura_radius_extension = self.color_mismatch * 150 # Max extension of 150px
        current_aura_radius = base_aura_radius + aura_radius_extension + pulse * 10

        aura_color = self._lerp_color(self.COLOR_AURA_SAFE, self.COLOR_AURA_DANGER, self.color_mismatch)
        
        # Draw multiple layers for a soft glow effect
        for i in range(5):
            alpha = 100 - i * 20
            radius = current_aura_radius * (1 - i * 0.1)
            if radius > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(self.chameleon_pos[0]), int(self.chameleon_pos[1]),
                    int(radius),
                    (*aura_color, alpha)
                )

        # Render chameleon
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.chameleon_pos[0]), int(self.chameleon_pos[1]),
            self.chameleon_radius, self.chameleon_color
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(self.chameleon_pos[0]), int(self.chameleon_pos[1]),
            self.chameleon_radius, self.chameleon_color
        )

    def _render_ui(self):
        # Render time bar
        time_bar_width = self.WIDTH - 20
        progress = min(self.steps / self.WIN_STEPS, 1.0)
        
        bg_rect = pygame.Rect(10, 10, time_bar_width, 20)
        pygame.draw.rect(self.screen, (50, 50, 50), bg_rect, border_radius=5)
        
        fill_width = int(time_bar_width * progress)
        fill_rect = pygame.Rect(10, 10, fill_width, 20)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, fill_rect, border_radius=5)

        # Render score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (self.WIDTH - 10, self.HEIGHT - 10), self.font_main, "bottomright")

        # Render game over/win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_UI_BG)
            self.screen.blit(overlay, (0, 0))

            if self.win_condition_met:
                message = "ESCAPED"
                color = self.COLOR_GREEN
            else:
                message = "DETECTED"
                color = self.COLOR_RED
            
            self._draw_text(message, (self.WIDTH // 2, self.HEIGHT // 2), self.font_large, "center", color)

    def _draw_text(self, text, pos, font, align="topleft", color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        shadow_surf = font.render(text, True, shadow_color)

        setattr(text_rect, align, pos)
        
        shadow_pos = (text_rect.x + 2, text_rect.y + 2)
        self.screen.blit(shadow_surf, shadow_pos)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "color_mismatch": round(self.color_mismatch, 3),
            "win": self.win_condition_met,
        }

    def _lerp_color(self, c1, c2, t):
        t = max(0, min(1, t))
        r = c1[0] + (c2[0] - c1[0]) * t
        g = c1[1] + (c2[1] - c1[1]) * t
        b = c1[2] + (c2[2] - c1[2]) * t
        return (int(r), int(g), int(b))
    
    def close(self):
        pygame.quit()

# Example of how to run the environment for interactive play
if __name__ == "__main__":
    # The validation function was removed from __init__ as it's not standard practice.
    # It's better to run validation separately if needed.
    
    # Check if a display is available for 'human' render mode
    try:
        pygame.display.init()
        # If we are in a headless environment, this will fail.
        has_display = pygame.display.get_driver() is not None
    except pygame.error:
        has_display = False

    render_mode = "human" if has_display else "rgb_array"
    if render_mode == "human":
        print("Running in human mode.")
        os.environ["SDL_VIDEODRIVER"] = "" # Use the default video driver
    else:
        print("Running in headless (rgb_array) mode.")

    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset()
    
    if render_mode == "human":
        # Setup for human play
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Chameleon Stealth")
        clock = pygame.time.Clock()
        
        terminated = False
        truncated = False
        total_reward = 0

        # Game loop
        while not terminated and not truncated:
            # Action mapping for human play
            keys = pygame.key.get_pressed()
            space_held = keys[pygame.K_SPACE]
            shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            
            action = [0, 0, 0] # [movement, space, shift]
            if space_held:
                action[1] = 1
            if shift_held:
                action[2] = 1

            # Process Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(GameEnv.FPS)
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}, Win: {info['win']}")
                # Wait a bit before closing
                pygame.time.wait(3000)

    env.close()