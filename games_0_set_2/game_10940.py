import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:11:31.768666
# Source Brief: brief_00940.md
# Brief Index: 940
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player nurtures a procedurally generated plant.
    The goal is to grow the plant to a target height by watering it, while managing
    a fluctuating water supply and a depleting timer.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Nurture a procedurally generated plant by watering it to reach a target height. "
        "Manage a fluctuating water supply and race against a depleting timer."
    )
    user_guide = (
        "Controls: Press space to water the plant and help it grow."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_DARK = (25, 28, 36)
    COLOR_BG_LIGHT = (40, 44, 52)
    COLOR_PLANT = (152, 195, 121)
    COLOR_PLANT_PULSE = (198, 232, 173, 100)
    COLOR_WATER = (97, 175, 239)
    COLOR_TIMER = (224, 108, 117)
    COLOR_UI_FRAME = (60, 65, 75)
    COLOR_TEXT = (210, 210, 210)

    # Game Parameters
    INITIAL_TIMER = 500
    MAX_STEPS = 1000
    WIN_HEIGHT = 20.0
    MAX_PLANT_HEIGHT = 40.0
    MAX_WATER_LEVEL = 10.0
    
    # UI Layout
    BAR_WIDTH = 50
    PLANT_BAR_X = SCREEN_WIDTH // 2 - BAR_WIDTH // 2
    WATER_BAR_X = PLANT_BAR_X - BAR_WIDTH - 20
    BAR_Y = 80
    BAR_MAX_HEIGHT = 280
    TIMER_BAR_Y = 20
    TIMER_BAR_HEIGHT = 20
    TIMER_BAR_WIDTH = SCREEN_WIDTH - 40
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.plant_height = None
        self.water_level = None
        self.timer = None
        self.growth_rate = None
        
        # --- Visual State Variables (for smooth interpolation) ---
        self.visual_plant_height = 0.0
        self.visual_water_level = 0.0
        self.water_pulse_effect = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.plant_height = 0.0
        self.water_level = 5.0
        self.timer = self.INITIAL_TIMER
        self.growth_rate = 1.0
        
        # Reset visual state
        self.visual_plant_height = 0.0
        self.visual_water_level = self.water_level
        self.water_pulse_effect = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self.reset()

        reward = 0.0
        self.steps += 1
        
        # --- Action Handling ---
        # movement = action[0] # Unused
        water_action = action[1] == 1
        # shift_action = action[2] == 1 # Unused

        # --- Game Logic Update ---
        # 1. Update timer
        self.timer -= 1
        
        # 2. Update water level (oscillates)
        water_change = self.np_random.uniform(-2.0, 2.0)
        self.water_level = np.clip(self.water_level + water_change, 0, self.MAX_WATER_LEVEL)

        # 3. Process player's watering action
        if water_action:
            self.water_pulse_effect = 15  # Trigger visual pulse effect
            water_cost = 1.0
            
            if self.water_level >= water_cost:
                # --- Successful Watering ---
                # sfx: water_splash_positive.wav
                self.water_level -= water_cost
                growth = self.np_random.uniform(0.5, 1.5) * self.growth_rate
                
                # Check for growth rate bonus trigger
                if self.plant_height < self.WIN_HEIGHT and (self.plant_height + growth) >= self.WIN_HEIGHT:
                    # sfx: bonus_achieved.wav
                    reward += 5.0
                    self.score += 5.0
                    self.growth_rate = 2.0
                
                self.plant_height += growth
                reward += 0.1 * growth
                self.score += 0.1 * growth
            else:
                # --- Failed Watering (Overwatering attempt) ---
                # sfx: error_buzz.wav
                damage = self.np_random.uniform(0.1, 0.5)
                self.plant_height -= damage
                reward -= 0.2 * damage
                self.score -= 0.2 * damage

        # 4. Clamp state values
        self.plant_height = np.clip(self.plant_height, 0, self.MAX_PLANT_HEIGHT)
        
        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.plant_height >= self.WIN_HEIGHT and self.growth_rate > 1.0: # Win condition is met and bonus applied
            # sfx: victory_fanfare.wav
            reward += 100.0
            self.score += 100.0
            terminated = True
            self.game_over = True
        elif self.timer <= 0:
            # sfx: loss_sound.wav
            reward -= 10.0
            self.score -= 10.0
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
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
        # --- Smooth visual state via interpolation ---
        lerp_factor = 0.15
        self.visual_plant_height += (self.plant_height - self.visual_plant_height) * lerp_factor
        self.visual_water_level += (self.water_level - self.visual_water_level) * lerp_factor
        
        # --- Render Background ---
        self.screen.fill(self.COLOR_BG_DARK)
        bg_light_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT * 0.7)
        pygame.draw.rect(self.screen, self.COLOR_BG_LIGHT, bg_light_rect)

        # --- Render Game Elements ---
        self._render_bars()
        
        # --- Render UI Overlay ---
        self._render_ui()
        
        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_bars(self):
        # --- Water Bar ---
        water_h = (self.visual_water_level / self.MAX_WATER_LEVEL) * self.BAR_MAX_HEIGHT
        water_rect = pygame.Rect(self.WATER_BAR_X, self.BAR_Y + self.BAR_MAX_HEIGHT - water_h, self.BAR_WIDTH, water_h)
        pygame.draw.rect(self.screen, self.COLOR_WATER, water_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)
        
        # --- Plant Bar ---
        plant_h_ratio = min(self.visual_plant_height, self.WIN_HEIGHT) / self.WIN_HEIGHT
        plant_h = plant_h_ratio * self.BAR_MAX_HEIGHT
        plant_rect = pygame.Rect(self.PLANT_BAR_X, self.BAR_Y + self.BAR_MAX_HEIGHT - plant_h, self.BAR_WIDTH, plant_h)
        
        # Bonus growth indicator
        if self.visual_plant_height > self.WIN_HEIGHT:
            bonus_h_ratio = (self.visual_plant_height - self.WIN_HEIGHT) / (self.MAX_PLANT_HEIGHT - self.WIN_HEIGHT)
            bonus_h = bonus_h_ratio * self.BAR_MAX_HEIGHT
            bonus_rect = pygame.Rect(self.PLANT_BAR_X, self.BAR_Y + self.BAR_MAX_HEIGHT - bonus_h, self.BAR_WIDTH, bonus_h)
            
            # Draw bonus part with a slightly different color
            bonus_color = tuple(min(255, c + 20) for c in self.COLOR_PLANT)
            pygame.draw.rect(self.screen, bonus_color, bonus_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)
            # Draw main part up to win height
            main_plant_rect = pygame.Rect(self.PLANT_BAR_X, self.BAR_Y, self.BAR_WIDTH, self.BAR_MAX_HEIGHT)
            pygame.draw.rect(self.screen, self.COLOR_PLANT, main_plant_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)
        else:
            pygame.draw.rect(self.screen, self.COLOR_PLANT, plant_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)

        # --- Watering Pulse Effect ---
        if self.water_pulse_effect > 0:
            pulse_alpha = (self.water_pulse_effect / 15.0) * 150
            pulse_size_increase = (1.0 - (self.water_pulse_effect / 15.0)) * 20
            
            pulse_surf = pygame.Surface((self.BAR_WIDTH + pulse_size_increase, self.BAR_WIDTH + pulse_size_increase), pygame.SRCALPHA)
            pygame.draw.circle(
                pulse_surf, 
                (*self.COLOR_PLANT_PULSE[:3], int(pulse_alpha)), 
                (pulse_surf.get_width()//2, pulse_surf.get_height()//2), 
                pulse_surf.get_width()//2
            )
            
            self.screen.blit(pulse_surf, (int(plant_rect.centerx - pulse_surf.get_width()//2), int(plant_rect.centery - pulse_surf.get_height()//2)))
            self.water_pulse_effect -= 1

        # --- Bar Frames ---
        water_frame = pygame.Rect(self.WATER_BAR_X, self.BAR_Y, self.BAR_WIDTH, self.BAR_MAX_HEIGHT)
        plant_frame = pygame.Rect(self.PLANT_BAR_X, self.BAR_Y, self.BAR_WIDTH, self.BAR_MAX_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_FRAME, water_frame, 2, 5)
        pygame.draw.rect(self.screen, self.COLOR_UI_FRAME, plant_frame, 2, 5)

        # --- Win Height Marker ---
        win_marker_y = self.BAR_Y + self.BAR_MAX_HEIGHT - (self.WIN_HEIGHT / self.WIN_HEIGHT * self.BAR_MAX_HEIGHT)
        pygame.draw.line(self.screen, self.COLOR_TEXT, (self.PLANT_BAR_X - 5, win_marker_y), (self.PLANT_BAR_X + self.BAR_WIDTH + 5, win_marker_y), 2)

    def _render_ui(self):
        # --- Timer Bar ---
        timer_x = (self.SCREEN_WIDTH - self.TIMER_BAR_WIDTH) / 2
        timer_frame_rect = pygame.Rect(timer_x, self.TIMER_BAR_Y, self.TIMER_BAR_WIDTH, self.TIMER_BAR_HEIGHT)
        
        time_ratio = self.timer / self.INITIAL_TIMER
        fill_width = self.TIMER_BAR_WIDTH * time_ratio
        timer_fill_rect = pygame.Rect(timer_x, self.TIMER_BAR_Y, fill_width, self.TIMER_BAR_HEIGHT)
        
        pygame.draw.rect(self.screen, self.COLOR_TIMER, timer_fill_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_FRAME, timer_frame_rect, 2, 5)
        
        # --- Text Rendering ---
        # Plant Height
        height_text_str = f"{self.plant_height:.1f} / {self.WIN_HEIGHT:.0f}"
        height_surf = self.font_small.render(height_text_str, True, self.COLOR_TEXT)
        height_rect = height_surf.get_rect(center=(self.PLANT_BAR_X + self.BAR_WIDTH / 2, self.BAR_Y - 20))
        self.screen.blit(height_surf, height_rect)
        
        # Timer Value
        timer_text_str = f"Time: {self.timer}"
        timer_surf = self.font_small.render(timer_text_str, True, self.COLOR_TEXT)
        timer_rect = timer_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.TIMER_BAR_Y + self.TIMER_BAR_HEIGHT / 2))
        self.screen.blit(timer_surf, timer_rect)

        # Score
        score_text_str = f"Score: {self.score:.1f}"
        score_surf = self.font_large.render(score_text_str, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(bottomleft=(20, self.SCREEN_HEIGHT - 10))
        self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "plant_height": self.plant_height,
            "timer": self.timer
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {self.action_space.nvec.tolist()}, expected [5, 2, 2]"
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}, expected {(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)}"
        assert test_obs.dtype == np.uint8, f"Obs dtype is {test_obs.dtype}, expected uint8"
        
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
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Plant Nurturer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("Q or Escape: Quit")
    print("----------------\n")
    
    while running:
        # --- Action Mapping for Manual Play ---
        # Default action is to do nothing
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1 # Spacebar is pressed

        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to Screen ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished!")
            print(f"Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause for 2 seconds before restarting

        clock.tick(30) # Run at 30 FPS

    env.close()