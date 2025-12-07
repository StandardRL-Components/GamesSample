
# Generated: 2025-08-27T17:45:17.943310
# Source Brief: brief_01632.md
# Brief Index: 1632

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    A farm management simulation where the player plants, harvests, and sells crops.

    The goal is to earn 1000 coins within a time limit of 180 turns. Each action,
    including waiting, consumes one turn. The game is played from a top-down
    perspective, with the player selecting plots or the market to perform actions.

    Visuals are designed to be clear and engaging, with particle effects and smooth
    UI feedback providing a polished experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to select a plot, ↓ to select the market. Press Space to interact (plant/harvest/sell)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage a small farm. Plant, harvest, and sell crops to earn 1000 coins before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WIN_COINS = 1000
        self.START_TIME = 180
        self.NUM_PLOTS = 10
        self.CROP_GROW_TIME = 20  # steps to become ready
        self.COIN_PER_CROP = 10

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_large = pygame.font.SysFont("Arial Black", 48)
            self.font_medium = pygame.font.SysFont("Arial", 24)
            self.font_small = pygame.font.SysFont("Arial", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 60)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (86, 138, 69)  # Muted green
        self.COLOR_PLOT_EMPTY = (139, 69, 19)  # Brown
        self.COLOR_PLOT_GROWING = (60, 100, 60) # Darker green
        self.COLOR_PLOT_READY = (255, 215, 0)  # Gold
        self.COLOR_MARKET = (205, 92, 92) # Indian Red
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HIGHLIGHT = (255, 255, 255)
        self.COLOR_PARTICLE_DIRT = (101, 67, 33)
        self.COLOR_PARTICLE_SPARKLE = (255, 255, 102)
        self.COLOR_PARTICLE_COIN = (255, 223, 0)

        # Game state variables (initialized in reset)
        self.plots = []
        self.harvested_crops = 0
        self.coins = 0
        self.time_remaining = 0
        self.selected_target = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.last_message = ""
        self.message_timer = 0
        self.particles = []
        
        # Initialize state
        self.reset()

        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Plot states: 0=empty, 1=growing, 2=ready
        # Each plot is a tuple: (state, growth_progress)
        self.plots = [(0, 0)] * self.NUM_PLOTS
        self.harvested_crops = 0
        self.coins = 0
        self.time_remaining = self.START_TIME
        self.selected_target = 0  # 0-9 for plots, 10 for market
        self.game_over = False
        self.win = False
        self.steps = 0
        self.last_message = ""
        self.message_timer = 0
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.1  # Cost for taking a turn/time passing

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        # --- Game Logic Update ---
        self._update_crop_growth()
        reward_bonus, message = self._handle_input(movement, space_pressed)
        reward += reward_bonus
        if message:
            self.last_message = message
            self.message_timer = 60 # frames

        self._update_particles()
        self.time_remaining -= 1

        # --- Check for Termination ---
        terminated = False
        if self.coins >= self.WIN_COINS:
            reward += 100
            terminated = True
            self.game_over = True
            self.win = True
            self.last_message = "YOU WIN!"
            self.message_timer = 180
        elif self.time_remaining <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            self.win = False
            self.last_message = "TIME'S UP!"
            self.message_timer = 180
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_crop_growth(self):
        new_plots = []
        for i, (state, growth) in enumerate(self.plots):
            if state == 1: # Growing
                growth += 1
                if growth >= self.CROP_GROW_TIME:
                    new_plots.append((2, self.CROP_GROW_TIME)) # Ready
                    # Sound: crop_ready.wav
                else:
                    new_plots.append((1, growth))
            else:
                new_plots.append((state, growth))
        self.plots = new_plots

    def _handle_input(self, movement, space_pressed):
        reward_bonus = 0
        message = ""

        # Handle movement to change selected target
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: # Up
            if self.selected_target == 10: self.selected_target = 4 # From market to middle plot
        elif movement == 2: # Down
            if self.selected_target < self.NUM_PLOTS: self.selected_target = 10 # From plots to market
        elif movement == 3: # Left
            if self.selected_target < self.NUM_PLOTS: self.selected_target = max(0, self.selected_target - 1)
        elif movement == 4: # Right
            if self.selected_target < self.NUM_PLOTS: self.selected_target = min(self.NUM_PLOTS - 1, self.selected_target + 1)
        
        if space_pressed:
            if self.selected_target < self.NUM_PLOTS: # Interacting with a plot
                plot_idx = self.selected_target
                state, _ = self.plots[plot_idx]
                plot_x, plot_y = self._get_plot_pos(plot_idx)

                if state == 0: # Empty -> Plant
                    self.plots[plot_idx] = (1, 0) # Set to growing
                    message = "Planted!"
                    self._create_particles(plot_x, plot_y, self.COLOR_PARTICLE_DIRT, 20)
                    # Sound: plant.wav
                elif state == 2: # Ready -> Harvest
                    self.plots[plot_idx] = (0, 0) # Set to empty
                    self.harvested_crops += 1
                    reward_bonus += 1.0
                    message = "Harvested!"
                    self._create_particles(plot_x, plot_y, self.COLOR_PARTICLE_SPARKLE, 30)
                    # Sound: harvest.wav
            
            elif self.selected_target == 10: # Interacting with market
                if self.harvested_crops > 0:
                    coins_earned = self.harvested_crops * self.COIN_PER_CROP
                    self.coins += coins_earned
                    reward_bonus += 10.0
                    message = f"+{coins_earned} Coins!"
                    self.harvested_crops = 0
                    self._create_particles(self.SCREEN_WIDTH // 2, 20, self.COLOR_PARTICLE_COIN, 50, "rain")
                    # Sound: cash_register.wav
                else:
                    message = "Nothing to sell!"
                    # Sound: error.wav
        
        return reward_bonus, message

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_plots_and_market()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_screen()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_plot_pos(self, i):
        plot_size = 40
        plot_spacing = 10
        total_width = self.NUM_PLOTS * plot_size + (self.NUM_PLOTS - 1) * plot_spacing
        start_x = (self.SCREEN_WIDTH - total_width) // 2
        x = start_x + i * (plot_size + plot_spacing) + plot_size // 2
        y = self.SCREEN_HEIGHT // 2
        return x, y

    def _render_plots_and_market(self):
        plot_size = 40
        # Render plots
        for i in range(self.NUM_PLOTS):
            x, y = self._get_plot_pos(i)
            rect = pygame.Rect(x - plot_size // 2, y - plot_size // 2, plot_size, plot_size)
            
            state, growth = self.plots[i]
            if state == 0: color = self.COLOR_PLOT_EMPTY
            elif state == 1: color = self.COLOR_PLOT_GROWING
            else: color = self.COLOR_PLOT_READY
            
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            
            if state == 1: # Draw growth indicator
                growth_ratio = growth / self.CROP_GROW_TIME
                pygame.draw.rect(self.screen, self.COLOR_PLOT_READY, (rect.x, rect.bottom - 5, rect.width * growth_ratio, 5), border_radius=2)
            elif state == 2: # Draw sparkle indicator
                spark_pos_x = rect.centerx + math.sin(self.steps * 0.2) * 5
                spark_pos_y = rect.centery + math.cos(self.steps * 0.2) * 5
                pygame.draw.circle(self.screen, (255,255,255), (int(spark_pos_x), int(spark_pos_y)), 2)

            if i == self.selected_target:
                pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, rect, 3, border_radius=5)

        # Render market
        market_w, market_h = 100, 50
        market_x = self.SCREEN_WIDTH // 2
        market_y = self.SCREEN_HEIGHT - 75
        market_rect = pygame.Rect(market_x - market_w // 2, market_y - market_h // 2, market_w, market_h)
        pygame.draw.rect(self.screen, self.COLOR_MARKET, market_rect, border_radius=8)
        
        market_text = self.font_small.render("SELL", True, self.COLOR_TEXT)
        self.screen.blit(market_text, market_text.get_rect(center=market_rect.center))

        if self.selected_target == 10:
            pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, market_rect, 3, border_radius=8)

    def _render_ui(self):
        # Coins display
        coin_text = self.font_medium.render(f"Coins: {self.coins}", True, self.COLOR_TEXT)
        self.screen.blit(coin_text, (20, 20))

        # Time display
        time_text = self.font_medium.render(f"Time: {self.time_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, time_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20)))

        # Harvested crops display
        harvest_text = self.font_medium.render(f"Harvested: {self.harvested_crops}", True, self.COLOR_TEXT)
        self.screen.blit(harvest_text, harvest_text.get_rect(midtop=(self.SCREEN_WIDTH // 2, 20)))

        # Action message display
        if self.message_timer > 0:
            self.message_timer -= 1
            alpha = min(255, self.message_timer * 5)
            message_surf = self.font_medium.render(self.last_message, True, self.COLOR_TEXT)
            message_surf.set_alpha(alpha)
            self.screen.blit(message_surf, message_surf.get_rect(center=(self.SCREEN_WIDTH // 2, 100)))

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        text = "YOU WIN!" if self.win else "TIME'S UP"
        color = self.COLOR_PLOT_READY if self.win else self.COLOR_MARKET
        
        end_text_surf = self.font_large.render(text, True, color)
        self.screen.blit(end_text_surf, end_text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)))

    def _create_particles(self, x, y, color, count, p_type="burst"):
        for _ in range(count):
            if p_type == "burst":
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                life = self.np_random.integers(20, 40)
                self.particles.append([x, y, vx, vy, life, color])
            elif p_type == "rain":
                px = self.np_random.uniform(0, self.SCREEN_WIDTH)
                py = self.np_random.uniform(-50, 0)
                vx = 0
                vy = self.np_random.uniform(2, 5)
                life = self.np_random.integers(60, 100)
                self.particles.append([px, py, vx, vy, life, color])


    def _update_particles(self):
        self.particles = [
            [p[0] + p[2], p[1] + p[3], p[2] * 0.98, p[3] * 0.98 + 0.1, p[4] - 1, p[5]]
            for p in self.particles if p[4] > 0
        ]

    def _render_particles(self):
        for x, y, _, _, life, color in self.particles:
            radius = max(1, int(life * 0.15))
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), radius, color)

    def _get_info(self):
        return {
            "score": self.coins,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "harvested_crops": self.harvested_crops,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example usage to test the environment
if __name__ == '__main__':
    # Set Pygame to use a visible display driver
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'windows' or 'x11' or 'dummy'

    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Re-initialize pygame for display
    pygame.display.init()
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Farm Manager")
    
    running = True
    while running:
        # Default action is NO-OP
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # In a turn-based game, we only step when there's an input.
        # For a smoother manual play experience, we can step on every frame.
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        env.clock.tick(30) # Limit manual play speed

    env.close()