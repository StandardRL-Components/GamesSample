import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:23:29.353764
# Source Brief: brief_02143.md
# Brief Index: 2143
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Wave Matcher'.

    The agent controls the frequency and amplitude of a sine wave to match a
    procedurally generated target wave. The goal is to maintain a close match
    over time to progress through levels.

    **Visuals:**
    - Player Wave: Bright blue, with a glow effect.
    - Target Wave: Bright red.
    - Background: Dark grid for a retro-tech feel.
    - UI: Clean, readable score and level display.
    - Sliders: Visual representation of frequency and amplitude on the right.

    **Gameplay:**
    - Match your blue wave to the moving red target wave.
    - Use Up/Down arrow keys (actions[0]=1,2) to control Frequency.
    - Use Left/Right arrow keys (actions[0]=3,4) to control Amplitude.
    - A precise match for a continuous duration completes a level.
    - Pushing the amplitude too high will cause a level failure.
    - The target wave moves faster on higher levels.

    **State:**
    The observation is the rendered game screen as a 640x400x3 RGB array.

    **Actions (MultiDiscrete([5, 2, 2])):**
    - actions[0]: Movement
        - 0: No-op
        - 1: Up (Increase Frequency)
        - 2: Down (Decrease Frequency)
        - 3: Left (Decrease Amplitude)
        - 4: Right (Increase Amplitude)
    - actions[1]: Space button (unused)
    - actions[2]: Shift button (unused)

    **Reward:**
    - +100 for completing all 5 levels.
    - -10 for exceeding maximum amplitude (terminates episode).
    - +25 bonus for completing a level.
    - +5 for a precise match on a single frame.
    - +1 for a close match.
    - +0.1 for a decent match.
    - 0 otherwise.

    **Termination:**
    - Episode ends if max amplitude is exceeded.
    - Episode ends if all 5 levels are completed.
    - Episode ends if the step limit (5000) is reached.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Match your wave's frequency and amplitude to the moving target wave. "
        "Maintain a precise match to progress through levels, but be careful not to push the amplitude too high."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to adjust frequency and ←→ arrow keys to adjust amplitude."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WAVE_AREA_WIDTH = 500
    UI_AREA_WIDTH = SCREEN_WIDTH - WAVE_AREA_WIDTH

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_PLAYER_WAVE = (0, 170, 255)
    COLOR_PLAYER_GLOW = (0, 170, 255, 50)
    COLOR_TARGET_WAVE = (255, 68, 68)
    COLOR_SLIDER_BG = (50, 50, 70)
    COLOR_SLIDER_FG = (204, 204, 204)
    COLOR_UI_TEXT = (255, 255, 255)

    # Game Parameters
    MAX_LEVELS = 5
    MAX_EPISODE_STEPS = 5000
    WAVE_CENTER_Y = SCREEN_HEIGHT // 2
    STEPS_FOR_LEVEL_WIN = 60 # 2 seconds at 30 FPS

    # Wave Parameters
    FREQ_MIN, FREQ_MAX, FREQ_STEP = 0.01, 0.1, 0.001
    AMP_MIN, AMP_MAX, AMP_STEP = 0, 150, 2.0
    MAX_ALLOWED_AMP = 180 # Boundary for failure

    # Reward Thresholds (based on Mean Squared Error)
    PRECISE_MATCH_THRESHOLD = 50
    CLOSE_MATCH_THRESHOLD = 200
    WIDE_MATCH_THRESHOLD = 800

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables (to be defined in reset)
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False

        self.player_freq = 0.0
        self.player_amp = 0.0
        self.target_freq = 0.0
        self.target_amp = 0.0
        self.target_phase_speed = 0.0
        self.wave_phase = 0.0

        self.precise_match_counter = 0
        self.level_up_flash = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.wave_phase = 0
        self.precise_match_counter = 0
        self.level_up_flash = 0

        self._setup_level()

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes the parameters for the current level."""
        # Reset player sliders to a neutral position
        self.player_freq = (self.FREQ_MIN + self.FREQ_MAX) / 2
        self.player_amp = self.AMP_MIN

        # Procedurally generate target wave for the level
        self.target_freq = self.np_random.uniform(self.FREQ_MIN * 2, self.FREQ_MAX * 0.8)
        self.target_amp = self.np_random.uniform(self.AMP_MAX * 0.3, self.AMP_MAX * 0.9)

        # Difficulty scaling: wave moves faster on higher levels
        self.target_phase_speed = 0.05 + (self.level - 1) * 0.05

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Unpack Action & Update Player Controls ---
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        if movement == 1:  # Up: Increase Frequency
            self.player_freq += self.FREQ_STEP
        elif movement == 2:  # Down: Decrease Frequency
            self.player_freq -= self.FREQ_STEP
        elif movement == 3:  # Left: Decrease Amplitude
            self.player_amp -= self.AMP_STEP
        elif movement == 4:  # Right: Increase Amplitude
            self.player_amp += self.AMP_STEP

        # Clamp slider values
        self.player_freq = np.clip(self.player_freq, self.FREQ_MIN, self.FREQ_MAX)
        self.player_amp = np.clip(self.player_amp, self.AMP_MIN, self.MAX_ALLOWED_AMP)

        # --- 2. Update Game State ---
        self.steps += 1
        self.wave_phase += self.target_phase_speed
        if self.level_up_flash > 0:
            self.level_up_flash -= 1

        # --- 3. Calculate Reward & Check Termination ---
        reward = 0
        terminated = False

        # Check for amplitude failure
        if self.player_amp >= self.MAX_ALLOWED_AMP:
            # Sound: Failure buzz
            reward = -10
            terminated = True
            self.game_over = True
        
        if not terminated:
            # Calculate match quality
            error = self._calculate_wave_error()

            if error < self.PRECISE_MATCH_THRESHOLD:
                # Sound: High-pitched 'ding'
                reward = 5
                self.score += 5
                self.precise_match_counter += 1
            elif error < self.CLOSE_MATCH_THRESHOLD:
                # Sound: Mid-pitched 'click'
                reward = 1
                self.score += 1
                self.precise_match_counter = 0 # Reset counter if match is not precise
            elif error < self.WIDE_MATCH_THRESHOLD:
                reward = 0.1
                self.score += 0.1
                self.precise_match_counter = 0
            else:
                self.precise_match_counter = 0

            # Check for level completion
            if self.precise_match_counter >= self.STEPS_FOR_LEVEL_WIN:
                # Sound: Level up fanfare
                self.level += 1
                self.precise_match_counter = 0
                self.level_up_flash = 15 # Flash for 0.5s

                if self.level > self.MAX_LEVELS:
                    # Game Won!
                    reward += 100
                    terminated = True
                    self.game_over = True
                else:
                    # Progress to next level
                    reward += 25 # Level completion bonus
                    self._setup_level()

        # Check for max steps
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            self.game_over = True

        terminated = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _calculate_wave_error(self):
        """Calculates the Mean Squared Error between the player and target waves."""
        player_ys = []
        target_ys = []
        # Sample points across the wave display area
        for x in range(0, self.WAVE_AREA_WIDTH, 5):
            angle = self.wave_phase + x
            player_y = self.WAVE_CENTER_Y - self.player_amp * math.sin(self.player_freq * angle)
            target_y = self.WAVE_CENTER_Y - self.target_amp * math.sin(self.target_freq * angle)
            player_ys.append(player_y)
            target_ys.append(target_y)
        
        if not player_ys:
            return float('inf')

        mse = sum((p - t) ** 2 for p, t in zip(player_ys, target_ys)) / len(player_ys)
        return mse

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "level": self.level,
            "steps": self.steps,
            "player_freq": self.player_freq,
            "player_amp": self.player_amp,
            "target_freq": self.target_freq,
            "target_amp": self.target_amp,
        }

    def _render_game(self):
        self._render_grid()
        
        # Render target wave first (in the background)
        self._render_wave(self.target_freq, self.target_amp, self.COLOR_TARGET_WAVE, 2, False)
        
        # Render player wave on top, with glow
        self._render_wave(self.player_freq, self.player_amp, self.COLOR_PLAYER_WAVE, 3, True)

        self._render_sliders()

        # Render level up flash effect
        if self.level_up_flash > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(150 * (self.level_up_flash / 15.0))
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_grid(self):
        for x in range(0, self.WAVE_AREA_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WAVE_AREA_WIDTH, y))

    def _render_wave(self, freq, amp, color, width, glow):
        points = []
        for x in range(self.WAVE_AREA_WIDTH + 1):
            angle = self.wave_phase + x
            y = self.WAVE_CENTER_Y - amp * math.sin(freq * angle)
            points.append((x, int(y)))

        if len(points) > 1:
            # Note: pygame.draw.aalines doesn't support width, so we ignore it.
            # The 'glow' effect is a semi-transparent version of the line.
            if glow:
                # This is a simple approximation for a glow. A more complex one might use blurs.
                # We draw a slightly thicker, transparent line.
                # Since aalines doesn't support width, we can't just make it thicker.
                # We will just draw the same line with an alpha color for a subtle effect.
                # Create a temporary surface for the glow
                temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                pygame.draw.aalines(temp_surf, self.COLOR_PLAYER_GLOW, False, points)
                self.screen.blit(temp_surf, (0,0))
            pygame.draw.aalines(self.screen, color, False, points)


    def _render_sliders(self):
        slider_x = self.WAVE_AREA_WIDTH + self.UI_AREA_WIDTH / 2
        slider_height = 150
        slider_y_start = 80

        # Frequency Slider
        freq_y = slider_y_start
        pygame.draw.rect(self.screen, self.COLOR_SLIDER_BG, (slider_x - 5, freq_y, 10, slider_height), border_radius=5)
        freq_ratio = (self.player_freq - self.FREQ_MIN) / (self.FREQ_MAX - self.FREQ_MIN)
        knob_y = freq_y + (1 - freq_ratio) * slider_height
        pygame.draw.circle(self.screen, self.COLOR_SLIDER_FG, (int(slider_x), int(knob_y)), 10)
        label = self.font_small.render("FREQ", True, self.COLOR_UI_TEXT)
        self.screen.blit(label, (slider_x - label.get_width() / 2, freq_y - 30))

        # Amplitude Slider
        amp_y = slider_y_start + slider_height + 60
        pygame.draw.rect(self.screen, self.COLOR_SLIDER_BG, (slider_x - 5, amp_y, 10, slider_height), border_radius=5)
        amp_ratio = (self.player_amp - self.AMP_MIN) / (self.AMP_MAX - self.AMP_MIN)
        knob_y = amp_y + (1 - amp_ratio) * slider_height
        pygame.draw.circle(self.screen, self.COLOR_SLIDER_FG, (int(slider_x), int(knob_y)), 10)
        label = self.font_small.render("AMP", True, self.COLOR_UI_TEXT)
        self.screen.blit(label, (slider_x - label.get_width() / 2, amp_y - 30))
        
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Level
        level_text = self.font_large.render(f"LEVEL: {self.level}/{self.MAX_LEVELS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.WAVE_AREA_WIDTH - level_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            if self.level > self.MAX_LEVELS:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WAVE_AREA_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            pygame.draw.rect(self.screen, self.COLOR_BG, text_rect.inflate(20, 20))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # This block is for human play and will not run in the test environment
    # but is useful for debugging.
    
    # Re-enable video driver for local play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Wave Matcher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    movement_action = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get keyboard state for continuous actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        else:
            movement_action = 0

        # Construct the MultiDiscrete action
        action = [movement_action, 0, 0] # space and shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Level: {info['level']}")
            # Add a small delay before resetting to show the final screen
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    env.close()