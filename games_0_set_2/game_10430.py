import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:27:18.391833
# Source Brief: brief_00430.md
# Brief Index: 430
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An expert-crafted Gymnasium environment for a minimalist puzzle game.
    The goal is to align three oscillating values within a step limit.
    This environment prioritizes visual quality, smooth animations, and satisfying "game feel".
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Align three oscillating values to the same number before the step limit is reached. "
        "Nudge values up or down, but be quick as all values automatically shift over time."
    )
    user_guide = (
        "Controls: Use ↑, ↓, and ← to select an oscillator. "
        "Press space to increase its value and shift to decrease it."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 100
    OSCILLATOR_COUNT = 3
    OSCILLATOR_MAX_VALUE = 9

    # --- Visuals ---
    COLOR_BG = (15, 18, 26)
    COLOR_GRID = (30, 35, 50)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_DIM = (100, 100, 110)
    COLOR_WIN = (180, 255, 180)
    COLOR_LOSE = (255, 180, 180)
    
    OSC_COLORS = [
        (255, 80, 120),   # Red/Pink
        (80, 255, 150),  # Green/Mint
        (80, 150, 255)   # Blue
    ]
    
    LERP_FACTOR = 0.25  # For smooth value animation

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
        self.font_large = pygame.font.SysFont('Consolas', 48, bold=True)
        self.font_medium = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 18)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.oscillators = []
        self.selected_osc_idx = 0
        self.particles = []

        # This will be called again in reset(), but it's good practice
        # to define all instance attributes in __init__.
        self._initialize_game_state()
        
        # --- Critical Self-Check ---
        # self.validate_implementation() # Commented out for submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_game_state()
        return self._get_observation(), self._get_info()
    
    def _initialize_game_state(self):
        """Initializes or resets all game state variables."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_osc_idx = 0
        self.particles = []

        # Create oscillators with random, non-winning start values
        while True:
            values = self.np_random.integers(0, self.OSCILLATOR_MAX_VALUE + 1, size=self.OSCILLATOR_COUNT).tolist()
            if len(set(values)) > 1:
                break
        
        self.oscillators = []
        for i in range(self.OSCILLATOR_COUNT):
            self.oscillators.append({
                "value": values[i],
                "display_value": float(values[i]), # For smooth animation
                "color": self.OSC_COLORS[i],
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Unpack Action ---
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        
        # --- 2. Update Game Logic ---
        self.steps += 1
        reward = 0
        
        # Store pre-action state for reward calculation
        prev_values = [osc['value'] for osc in self.oscillators]
        
        # --- 3. Handle Player Input ---
        # Selection
        if movement == 1: self.selected_osc_idx = 0  # Up
        elif movement == 2: self.selected_osc_idx = 1  # Down
        elif movement == 3: self.selected_osc_idx = 2  # Left
        
        # Modification (Space/Shift)
        osc_to_modify = self.oscillators[self.selected_osc_idx]
        value_changed = False
        if space_press:
            # Player increments value
            osc_to_modify['value'] = (osc_to_modify['value'] + 1) % (self.OSCILLATOR_MAX_VALUE + 1)
            value_changed = True
            # Sound: 'increment.wav'
            self._spawn_particles(self.selected_osc_idx, 1)
        elif shift_press:
            # Player decrements value
            osc_to_modify['value'] = (osc_to_modify['value'] - 1) % (self.OSCILLATOR_MAX_VALUE + 1)
            value_changed = True
            # Sound: 'decrement.wav'
            self._spawn_particles(self.selected_osc_idx, -1)

        # --- 4. Automatic Oscillation ---
        total_value = sum(osc['value'] for osc in self.oscillators)
        speed = (total_value % 4) + 1
        
        if self.steps > 0 and self.steps % speed == 0:
            for osc in self.oscillators:
                osc['value'] = (osc['value'] + 1) % (self.OSCILLATOR_MAX_VALUE + 1)
            # Sound: 'oscillate.wav'

        # --- 5. Calculate Reward ---
        current_values = [osc['value'] for osc in self.oscillators]
        
        # Continuous reward for converging
        median_val = np.median(current_values)
        prev_dist = sum(abs(v - np.median(prev_values)) for v in prev_values)
        current_dist = sum(abs(v - median_val) for v in current_values)
        reward += (prev_dist - current_dist)

        # Event-based reward for pairs
        if len(set(current_values)) == 2:
            reward += 5
            
        # --- 6. Check Termination Conditions ---
        win_condition = len(set(current_values)) == 1
        lose_condition = self.steps >= self.MAX_STEPS
        terminated = win_condition or lose_condition

        if terminated:
            self.game_over = True
            if win_condition:
                reward += 100
                # Sound: 'win.wav'
            else: # lose_condition
                reward -= 100
                # Sound: 'lose.wav'
        
        self.score += reward

        # --- 7. Return Gymnasium 5-tuple ---
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _get_observation(self):
        # Update animations
        self._update_animations()
        
        # Render all elements
        self._render_background()
        self._render_oscillators()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        # Convert to numpy array in the required format
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "values": [osc['value'] for osc in self.oscillators],
            "win": len(set(osc['value'] for osc in self.oscillators)) == 1 if self.game_over else False
        }
    
    def _update_animations(self):
        # Smoothly interpolate display values towards actual values
        for osc in self.oscillators:
            osc['display_value'] += (osc['value'] - osc['display_value']) * self.LERP_FACTOR
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        grid_spacing = 40
        for x in range(0, self.SCREEN_WIDTH, grid_spacing):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, grid_spacing):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_oscillators(self):
        bar_width = 80
        max_bar_height = 200
        total_width = self.OSCILLATOR_COUNT * bar_width + (self.OSCILLATOR_COUNT - 1) * 60
        start_x = (self.SCREEN_WIDTH - total_width) / 2
        bottom_y = self.SCREEN_HEIGHT - 80

        for i, osc in enumerate(self.oscillators):
            center_x = start_x + i * (bar_width + 60) + bar_width / 2
            
            # --- Render Glow for selected oscillator ---
            if i == self.selected_osc_idx and not self.game_over:
                glow_color = osc['color']
                for j in range(15, 0, -2):
                    alpha = 60 * (1 - j / 15)
                    s = pygame.Surface((bar_width + j*2, max_bar_height + j*2), pygame.SRCALPHA)
                    pygame.draw.rect(s, (*glow_color, alpha), s.get_rect(), border_radius=12)
                    self.screen.blit(s, (center_x - s.get_width()/2, bottom_y - max_bar_height - j))

            # --- Render Bar ---
            bar_height = (osc['display_value'] / self.OSCILLATOR_MAX_VALUE) * max_bar_height
            bar_rect = pygame.Rect(0, 0, bar_width, max(1, bar_height))
            bar_rect.bottomleft = (center_x - bar_width / 2, bottom_y)
            pygame.draw.rect(self.screen, osc['color'], bar_rect, border_radius=8)

            # --- Render Value Text ---
            value_text = self.font_medium.render(str(osc['value']), True, self.COLOR_TEXT)
            text_rect = value_text.get_rect(center=(center_x, bottom_y - max_bar_height - 30))
            self.screen.blit(value_text, text_rect)

            # --- Render Label Text ---
            label_text = self.font_medium.render(str(i + 1), True, self.COLOR_TEXT_DIM)
            label_rect = label_text.get_rect(center=(center_x, bottom_y + 30))
            self.screen.blit(label_text, label_rect)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, p['life'] * 5))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['life'] / 5)
            if radius > 1:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def _render_ui(self):
        # Steps counter
        steps_text = self.font_small.render(f"STEP: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (20, 20))
        
        # Score display
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
        self.screen.blit(overlay, (0, 0))

        win = len(set(osc['value'] for osc in self.oscillators)) == 1
        message = "ALIGNED" if win else "TIME UP"
        color = self.COLOR_WIN if win else self.COLOR_LOSE

        text = self.font_large.render(message, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def _spawn_particles(self, osc_idx, direction):
        osc = self.oscillators[osc_idx]
        bar_width = 80
        max_bar_height = 200
        total_width = self.OSCILLATOR_COUNT * bar_width + (self.OSCILLATOR_COUNT - 1) * 60
        start_x = (self.SCREEN_WIDTH - total_width) / 2
        bottom_y = self.SCREEN_HEIGHT - 80
        center_x = start_x + osc_idx * (bar_width + 60) + bar_width / 2
        
        bar_height = (osc['display_value'] / self.OSCILLATOR_MAX_VALUE) * max_bar_height
        
        for _ in range(15):
            self.particles.append({
                'pos': [center_x + self.np_random.uniform(-bar_width/3, bar_width/3), bottom_y - bar_height],
                'vel': [self.np_random.uniform(-1, 1), self.np_random.uniform(-2, -0.5) * direction],
                'life': self.np_random.integers(20, 40),
                'color': osc['color']
            })

    def close(self):
        pygame.quit()
        
    def render(self):
        # This method is not used by the agent but can be useful for human play.
        # It's not part of the standard Gym API loop but is good practice to include.
        # We will just return the observation array.
        return self._get_observation()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This requires a display. If you are running in a headless environment,
    # you might need to use a virtual display buffer like Xvfb.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame window for human interaction
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Oscillator Alignment Puzzle")
    clock = pygame.time.Clock()

    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        # The action space is [5, 2, 2]
        # Movement: 0=None, 1=Up, 2=Down, 3=Left, 4=Right
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        # No mapping for right arrow in this game's logic
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

        if done:
            print(f"Game Over! Final Score: {info['score']}, Win: {info['win']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            done = False

    env.close()