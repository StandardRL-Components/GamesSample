import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:40:10.209023
# Source Brief: brief_02927.md
# Brief Index: 2927
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must flip quantum bits to match a target sequence.
    This is an arcade-style puzzle game with a real-time countdown timer. Successfully
    matching a sequence grants points, bonus time, and a speed increase for subsequent flips.
    The episode ends when the timer runs out.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=flip bit 1, 2=flip bit 2, 3=flip bit 3, 4=flip bit 4)
    - actions[1]: Space button (unused)
    - actions[2]: Shift button (unused)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Flip quantum bits to match a target sequence against a real-time countdown timer. "
        "Successful matches grant points, bonus time, and increase your flip speed."
    )
    user_guide = "Controls: Use keys 1, 2, 3, and 4 to flip the corresponding quantum bit."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # --- Colors (Futuristic/Minimalist Theme) ---
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_BIT_0 = (255, 80, 80)
    COLOR_BIT_1 = (80, 255, 80)
    COLOR_TARGET = (100, 150, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TIMER_BAR_FULL = (255, 200, 0)
    COLOR_TIMER_BAR_MID = (255, 120, 0)
    COLOR_TIMER_BAR_LOW = (255, 50, 50)
    COLOR_FLIP_COOLDOWN = (200, 200, 255, 150)
    COLOR_SUCCESS_FLASH = (255, 255, 255)

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
        self.font_huge = pygame.font.Font(None, 80)
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_bits = np.zeros(4, dtype=int)
        self.target_bits = np.zeros(4, dtype=int)
        self.timer = 0.0
        self.max_time = 15.0
        self.initial_flip_cooldown = 1.0  # 1 second
        self.flip_cooldown_max = self.initial_flip_cooldown
        self.flip_cooldown_timer = 0.0
        self.animations = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.max_time
        self.flip_cooldown_max = self.initial_flip_cooldown
        self.flip_cooldown_timer = 0.0
        self.animations = []

        self._generate_sequences()

        return self._get_observation(), self._get_info()

    def _generate_sequences(self):
        self.target_bits = self.np_random.integers(0, 2, size=4)
        while True:
            self.current_bits = self.np_random.integers(0, 2, size=4)
            if not np.array_equal(self.current_bits, self.target_bits):
                break

    def step(self, action):
        if self.game_over:
            # On subsequent steps after termination, just return the final state
            # This is useful for some training loops.
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        terminated = False
        truncated = False

        # --- Update Timers ---
        time_delta = 1 / self.FPS
        self.timer -= time_delta
        if self.flip_cooldown_timer > 0:
            self.flip_cooldown_timer = max(0, self.flip_cooldown_timer - time_delta)

        # --- Process Action ---
        if movement in [1, 2, 3, 4] and self.flip_cooldown_timer <= 0:
            bit_index = movement - 1
            # sfx: bit_flip.wav
            self.current_bits[bit_index] = 1 - self.current_bits[bit_index]
            self.flip_cooldown_timer = self.flip_cooldown_max
            self._add_flip_animation(bit_index)

        # --- Calculate Rewards & Check for Success ---
        num_matches = np.sum(self.current_bits == self.target_bits)
        reward += num_matches * 0.01 # Small continuous reward for matching bits

        if np.array_equal(self.current_bits, self.target_bits):
            # sfx: success.wav
            reward += 50
            self.score += 1
            self.timer = min(self.max_time, self.timer + 2.0) # Bonus time
            self.flip_cooldown_max = max(0.2, self.flip_cooldown_max * 0.8) # 20% faster
            self._add_success_flash()
            self._generate_sequences() # New puzzle

        # --- Check for Termination ---
        self.steps += 1
        if self.timer <= 0:
            # sfx: failure.wav
            reward = -10 # Terminal penalty
            terminated = True
            self.game_over = True
        elif self.steps >= 1000:
            # Using truncated for step limit
            truncated = True
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
        self._render_grid()
        self._render_game()
        self._update_and_render_animations()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        self._render_bits()
        self._render_target_sequence()

    def _render_ui(self):
        # Timer Bar
        timer_ratio = self.timer / self.max_time
        bar_width = int(timer_ratio * self.SCREEN_WIDTH)
        if timer_ratio > 0.5:
            bar_color = self.COLOR_TIMER_BAR_FULL
        elif timer_ratio > 0.2:
            bar_color = self.COLOR_TIMER_BAR_MID
        else:
            bar_color = self.COLOR_TIMER_BAR_LOW
        pygame.draw.rect(self.screen, bar_color, (0, 0, bar_width, 10))

        # Score Text
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Flip Speed
        speed_text = self.font_small.render(f"FLIP SPEED: {1/self.flip_cooldown_max:.1f}x", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (self.SCREEN_WIDTH - speed_text.get_width() - 20, 20))

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_bits(self):
        bit_radius = 40
        spacing = 150
        start_x = (self.SCREEN_WIDTH - (4 * spacing) + spacing) / 2
        y_pos = self.SCREEN_HEIGHT / 2 - 30

        for i, bit_val in enumerate(self.current_bits):
            x_pos = int(start_x + i * spacing)
            color = self.COLOR_BIT_1 if bit_val == 1 else self.COLOR_BIT_0
            self._draw_glowing_circle(self.screen, color, (x_pos, int(y_pos)), bit_radius)
            
            # Bit value text
            bit_text = self.font_huge.render(str(bit_val), True, self.COLOR_BG)
            text_rect = bit_text.get_rect(center=(x_pos, int(y_pos)))
            self.screen.blit(bit_text, text_rect)

            # Bit label
            label_text = self.font_medium.render(f"Q{i+1}", True, self.COLOR_UI_TEXT)
            label_rect = label_text.get_rect(center=(x_pos, int(y_pos) - bit_radius - 20))
            self.screen.blit(label_text, label_rect)

            # Cooldown indicator
            if self.flip_cooldown_timer > 0:
                cooldown_ratio = self.flip_cooldown_timer / self.flip_cooldown_max
                bar_height = int(cooldown_ratio * (bit_radius * 2))
                bar_rect = pygame.Rect(x_pos - bit_radius - 15, y_pos + bit_radius - bar_height, 10, bar_height)
                pygame.draw.rect(self.screen, self.COLOR_FLIP_COOLDOWN, bar_rect)

    def _render_target_sequence(self):
        target_radius = 15
        spacing = 50
        total_width = (4 * spacing) - (spacing - target_radius * 2)
        start_x = (self.SCREEN_WIDTH - total_width) / 2
        y_pos = self.SCREEN_HEIGHT - 60

        # Label
        label_text = self.font_medium.render("TARGET", True, self.COLOR_TARGET)
        label_rect = label_text.get_rect(center=(self.SCREEN_WIDTH / 2, y_pos - 40))
        self.screen.blit(label_text, label_rect)
        
        for i, bit_val in enumerate(self.target_bits):
            x_pos = int(start_x + i * spacing)
            color = self.COLOR_BIT_1 if bit_val == 1 else self.COLOR_BIT_0
            self._draw_glowing_circle(self.screen, color, (x_pos, int(y_pos)), target_radius, glow_alpha=40)
            
            bit_text = self.font_medium.render(str(bit_val), True, self.COLOR_BG)
            text_rect = bit_text.get_rect(center=(x_pos, int(y_pos)))
            self.screen.blit(bit_text, text_rect)

    def _draw_glowing_circle(self, surface, color, center, radius, glow_factor=1.5, glow_alpha=60):
        glow_color = (*color, glow_alpha)
        glow_radius = int(radius * glow_factor)
        
        # Draw the glow using a larger, transparent circle
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (center[0] - glow_radius, center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Draw the main, solid circle on top
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)

    def _update_and_render_animations(self):
        active_animations = []
        for anim in self.animations:
            anim['life'] -= 1 / self.FPS
            if anim['life'] > 0:
                if anim['type'] == 'ring':
                    anim['radius'] += anim['expand_speed'] * (1 / self.FPS)
                    alpha = int(255 * (anim['life'] / anim['max_life']))
                    color = (*anim['color'], alpha)
                    
                    # Create a temporary surface for the ring
                    ring_surf = pygame.Surface((int(anim['radius']*2), int(anim['radius']*2)), pygame.SRCALPHA)
                    pygame.draw.circle(ring_surf, color, (int(anim['radius']), int(anim['radius'])), int(anim['radius']), width=anim['width'])
                    self.screen.blit(ring_surf, (anim['pos'][0] - int(anim['radius']), anim['pos'][1] - int(anim['radius'])))

                if anim['type'] == 'flash':
                    alpha = int(255 * (anim['life'] / anim['max_life'])**2) # Fast fade out
                    flash_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                    flash_surface.fill((*self.COLOR_SUCCESS_FLASH, alpha))
                    self.screen.blit(flash_surface, (0, 0))

                active_animations.append(anim)
        self.animations = active_animations

    def _add_flip_animation(self, bit_index):
        bit_radius = 40
        spacing = 150
        start_x = (self.SCREEN_WIDTH - (4 * spacing) + spacing) / 2
        y_pos = self.SCREEN_HEIGHT / 2 - 30
        x_pos = int(start_x + bit_index * spacing)
        
        self.animations.append({
            'type': 'ring',
            'pos': (x_pos, int(y_pos)),
            'radius': float(bit_radius),
            'expand_speed': 200.0,
            'color': self.COLOR_UI_TEXT,
            'width': 4,
            'life': 0.5,
            'max_life': 0.5
        })

    def _add_success_flash(self):
        self.animations.append({
            'type': 'flash',
            'life': 0.3,
            'max_life': 0.3
        })

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # --- Manual Play Example ---
    # To run this, you need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Quantum Bit Flipper")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("1, 2, 3, 4: Flip corresponding bit")
    print("R: Reset environment")
    print("Q: Quit")

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"--- Env Reset --- Score: {info['score']}")
                
                # Map keys to actions
                if event.key == pygame.K_1: action[0] = 1
                if event.key == pygame.K_2: action[0] = 2
                if event.key == pygame.K_3: action[0] = 3
                if event.key == pygame.K_4: action[0] = 4
        
        # In a real-time game, we always step, even if no key is pressed
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
    env.close()