import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:46:52.965325
# Source Brief: brief_02379.md
# Brief Index: 2379
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
        "Manage water and sunlight to balance an ecosystem of producers, consumers, and decomposers. "
        "Maintain equilibrium to achieve a high score and prevent collapse."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to adjust water levels and ←→ arrow keys to adjust sunlight. "
        "Try to keep all populations within their stable ranges."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    UI_WIDTH = 120
    GAME_WIDTH = SCREEN_WIDTH - UI_WIDTH * 2

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_PRODUCER = (46, 204, 113)
    COLOR_CONSUMER = (230, 126, 34)
    COLOR_DECOMPOSER = (149, 113, 80)
    COLOR_WATER = (52, 152, 219)
    COLOR_SUN = (241, 196, 15)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BG = (25, 30, 45)
    COLOR_UI_BAR_BG = (40, 50, 70)
    COLOR_SUCCESS = (46, 204, 113)
    COLOR_FAILURE = (231, 76, 60)
    
    # Game Parameters
    MAX_STEPS = 1000
    WIN_STREAK = 100
    RESOURCE_MAX = 100.0
    RESOURCE_CHANGE_RATE = 4.0
    PARTICLE_VISUAL_SCALE = 10  # 1 particle represents this many population
    PARTICLE_SPEED = 0.5

    # Population Dynamics (tuned for interesting interactions)
    INITIAL_PRODUCERS = 500
    INITIAL_CONSUMERS = 100
    INITIAL_DECOMPOSERS = 75
    
    MAX_PRODUCERS = 2000
    MAX_CONSUMERS = 500
    MAX_DECOMPOSERS = 500
    
    EQUILIBRIUM_PRODUCER = (450, 650)
    EQUILIBRIUM_CONSUMER = (80, 150)
    EQUILIBRIUM_DECOMPOSER = (60, 110)

    # Simulation Rates
    PRODUCER_GROWTH_RATE = 0.1
    PRODUCER_DEATH_RATE = 0.01
    CONSUMPTION_RATE = 0.0002
    CONSUMER_EFFICIENCY = 0.4
    CONSUMER_DEATH_RATE = 0.04
    DECOMPOSER_EFFICIENCY = 0.8
    DECOMPOSER_DEATH_RATE = 0.02
    
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
        
        self.font_main = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_status = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_end = pygame.font.SysFont("Consolas", 48, bold=True)

        # Initialize state variables to be populated in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.victory = False
        self.equilibrium_streak = 0
        
        self.resource_water = 0.0
        self.resource_sunlight = 0.0
        
        self.pop_producers = 0.0
        self.pop_consumers = 0.0
        self.pop_decomposers = 0.0
        
        self.producer_particles = []
        self.consumer_particles = []
        self.decomposer_particles = []
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.victory = False
        self.equilibrium_streak = 0
        
        self.resource_water = self.RESOURCE_MAX * 0.6
        self.resource_sunlight = self.RESOURCE_MAX * 0.6
        
        self.pop_producers = float(self.INITIAL_PRODUCERS)
        self.pop_consumers = float(self.INITIAL_CONSUMERS)
        self.pop_decomposers = float(self.INITIAL_DECOMPOSERS)

        self._update_all_particles(initial=True)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack and apply action
        movement = action[0]
        self._update_resources(movement)

        # Update game logic
        self._update_ecosystem()
        self.steps += 1
        
        # Calculate reward and termination
        in_equilibrium = self._is_in_equilibrium()
        reward = self._calculate_reward(in_equilibrium)
        self.score += reward
        
        if in_equilibrium:
            self.equilibrium_streak += 1
        else:
            self.equilibrium_streak = 0
            
        terminated = self._check_termination()
        self.game_over = terminated
        
        # Update visuals
        self._update_all_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_resources(self, movement):
        if movement == 1: # Up
            self.resource_water = min(self.RESOURCE_MAX, self.resource_water + self.RESOURCE_CHANGE_RATE)
        elif movement == 2: # Down
            self.resource_water = max(0, self.resource_water - self.RESOURCE_CHANGE_RATE)
        elif movement == 4: # Right
            self.resource_sunlight = min(self.RESOURCE_MAX, self.resource_sunlight + self.RESOURCE_CHANGE_RATE)
        elif movement == 3: # Left
            self.resource_sunlight = max(0, self.resource_sunlight - self.RESOURCE_CHANGE_RATE)

    def _update_ecosystem(self):
        # Calculate changes based on the current state
        producers_growth = self.pop_producers * self.PRODUCER_GROWTH_RATE * \
                           (self.resource_water / self.RESOURCE_MAX) * \
                           (self.resource_sunlight / self.RESOURCE_MAX) * \
                           (1 - self.pop_producers / self.MAX_PRODUCERS)
        
        producers_eaten = self.pop_producers * self.pop_consumers * self.CONSUMPTION_RATE
        producers_death = self.pop_producers * self.PRODUCER_DEATH_RATE

        consumers_growth = producers_eaten * self.CONSUMER_EFFICIENCY
        consumers_death = self.pop_consumers * self.CONSUMER_DEATH_RATE

        dead_biomass = (producers_death + consumers_death)
        decomposers_growth = dead_biomass * self.DECOMPOSER_EFFICIENCY * (1 - self.pop_decomposers / self.MAX_DECOMPOSERS)
        decomposers_death = self.pop_decomposers * self.DECOMPOSER_DEATH_RATE

        # Apply changes
        self.pop_producers += producers_growth - producers_eaten - producers_death
        self.pop_consumers += consumers_growth - consumers_death
        self.pop_decomposers += decomposers_growth - decomposers_death

        # Clamp populations to be non-negative
        self.pop_producers = max(0, self.pop_producers)
        self.pop_consumers = max(0, self.pop_consumers)
        self.pop_decomposers = max(0, self.pop_decomposers)
        
    def _is_in_equilibrium(self):
        p_ok = self.EQUILIBRIUM_PRODUCER[0] <= self.pop_producers <= self.EQUILIBRIUM_PRODUCER[1]
        c_ok = self.EQUILIBRIUM_CONSUMER[0] <= self.pop_consumers <= self.EQUILIBRIUM_CONSUMER[1]
        d_ok = self.EQUILIBRIUM_DECOMPOSER[0] <= self.pop_decomposers <= self.EQUILIBRIUM_DECOMPOSER[1]
        return p_ok and c_ok and d_ok

    def _calculate_reward(self, in_equilibrium):
        if self.pop_producers <= 1 or self.pop_consumers <= 1 or self.pop_decomposers <= 1:
            return -100.0 # Extinction
        if self.equilibrium_streak >= self.WIN_STREAK:
            return 100.0 # Victory
        
        reward = 1.0 # Survival reward
        if in_equilibrium:
            reward += 5.0 # Equilibrium bonus
        return reward

    def _check_termination(self):
        extinction = self.pop_producers <= 1 or self.pop_consumers <= 1 or self.pop_decomposers <= 1
        self.victory = self.equilibrium_streak >= self.WIN_STREAK
        timeout = self.steps >= self.MAX_STEPS
        return extinction or self.victory or timeout

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        game_rect = pygame.Rect(self.UI_WIDTH, 0, self.GAME_WIDTH, self.SCREEN_HEIGHT)
        
        # Draw grid
        for x in range(self.UI_WIDTH, self.SCREEN_WIDTH - self.UI_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.UI_WIDTH, y), (self.SCREEN_WIDTH - self.UI_WIDTH, y))
            
        # Draw particles
        self._render_particles(self.producer_particles, self.COLOR_PRODUCER, game_rect)
        self._render_particles(self.consumer_particles, self.COLOR_CONSUMER, game_rect)
        self._render_particles(self.decomposer_particles, self.COLOR_DECOMPOSER, game_rect)

    def _render_particles(self, particles, color, bounds):
        for p in particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy

            # Bounce off walls
            if not bounds.left < p[0] < bounds.right - 4:
                p[2] *= -1
                p[0] = max(bounds.left, min(p[0], bounds.right - 4))
            if not bounds.top < p[1] < bounds.bottom - 4:
                p[3] *= -1
                p[1] = max(bounds.top, min(p[1], bounds.bottom - 4))
            
            # Draw glow
            glow_color = (*color, 30)
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 4, glow_color)
            # Draw particle
            pygame.gfxdraw.aacircle(self.screen, int(p[0]), int(p[1]), 2, color)
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 2, color)

    def _update_all_particles(self, initial=False):
        self._update_particle_list(self.producer_particles, self.pop_producers, initial)
        self._update_particle_list(self.consumer_particles, self.pop_consumers, initial)
        self._update_particle_list(self.decomposer_particles, self.pop_decomposers, initial)

    def _update_particle_list(self, particle_list, population, initial=False):
        target_count = int(population / self.PARTICLE_VISUAL_SCALE)
        
        if initial:
            particle_list.clear()
        
        # Add new particles if needed
        while len(particle_list) < target_count:
            x = random.uniform(self.UI_WIDTH, self.SCREEN_WIDTH - self.UI_WIDTH)
            y = random.uniform(0, self.SCREEN_HEIGHT)
            angle = random.uniform(0, 2 * math.pi)
            vx = math.cos(angle) * self.PARTICLE_SPEED
            vy = math.sin(angle) * self.PARTICLE_SPEED
            particle_list.append([x, y, vx, vy])
            
        # Remove excess particles
        while len(particle_list) > target_count:
            particle_list.pop(random.randrange(len(particle_list)))

    def _render_ui(self):
        # Left Panel (Resources)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.UI_WIDTH, self.SCREEN_HEIGHT))
        self._render_bar("WATER", self.resource_water, self.RESOURCE_MAX, self.COLOR_WATER, 20)
        self._render_bar("SUNLIGHT", self.resource_sunlight, self.RESOURCE_MAX, self.COLOR_SUN, 180)

        # Right Panel (Stats)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.SCREEN_WIDTH - self.UI_WIDTH, 0, self.UI_WIDTH, self.SCREEN_HEIGHT))
        y_pos = 20
        self._render_text(f"CYCLE: {self.steps}/{self.MAX_STEPS}", self.SCREEN_WIDTH - self.UI_WIDTH / 2, y_pos, self.font_main, self.COLOR_UI_TEXT)
        y_pos += 30
        self._render_text(f"SCORE: {self.score:.0f}", self.SCREEN_WIDTH - self.UI_WIDTH / 2, y_pos, self.font_main, self.COLOR_UI_TEXT)
        
        y_pos += 50
        self._render_text("POPULATIONS", self.SCREEN_WIDTH - self.UI_WIDTH / 2, y_pos, self.font_title, self.COLOR_UI_TEXT)
        y_pos += 30
        self._render_text(f"{int(self.pop_producers)}", self.SCREEN_WIDTH - self.UI_WIDTH / 2, y_pos, self.font_main, self.COLOR_PRODUCER)
        y_pos += 20
        self._render_text(f"{int(self.pop_consumers)}", self.SCREEN_WIDTH - self.UI_WIDTH / 2, y_pos, self.font_main, self.COLOR_CONSUMER)
        y_pos += 20
        self._render_text(f"{int(self.pop_decomposers)}", self.SCREEN_WIDTH - self.UI_WIDTH / 2, y_pos, self.font_main, self.COLOR_DECOMPOSER)

        # Equilibrium Status
        y_pos += 60
        if self._is_in_equilibrium():
            self._render_text("EQUILIBRIUM", self.SCREEN_WIDTH - self.UI_WIDTH / 2, y_pos, self.font_main, self.COLOR_SUCCESS)
            y_pos += 20
            self._render_text(f"STREAK: {self.equilibrium_streak}", self.SCREEN_WIDTH - self.UI_WIDTH / 2, y_pos, self.font_main, self.COLOR_SUCCESS)
        else:
            self._render_text("UNSTABLE", self.SCREEN_WIDTH - self.UI_WIDTH / 2, y_pos, self.font_main, self.COLOR_FAILURE)

        # Game Over / Victory Message
        if self.game_over:
            overlay = pygame.Surface((self.GAME_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (self.UI_WIDTH, 0))
            
            if self.victory:
                msg = "EQUILIBRIUM ACHIEVED"
                color = self.COLOR_SUCCESS
            else:
                msg = "ECOSYSTEM COLLAPSED"
                color = self.COLOR_FAILURE
            self._render_text(msg, self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2, self.font_end, color)

    def _render_bar(self, label, value, max_value, color, y_pos):
        bar_width = self.UI_WIDTH - 40
        bar_height = 120
        # Title
        self._render_text(label, self.UI_WIDTH / 2, y_pos, self.font_title, self.COLOR_UI_TEXT)
        # Bar BG
        bg_rect = pygame.Rect(20, y_pos + 20, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, bg_rect, border_radius=4)
        # Bar Fill
        fill_height = max(0, (value / max_value) * (bar_height - 8))
        fill_rect = pygame.Rect(24, y_pos + 24 + (bar_height - 8 - fill_height), bar_width - 8, fill_height)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=4)

    def _render_text(self, text, x, y, font, color):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(x, y))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "pop_producers": int(self.pop_producers),
            "pop_consumers": int(self.pop_consumers),
            "pop_decomposers": int(self.pop_decomposers),
            "equilibrium_streak": self.equilibrium_streak,
        }

    def close(self):
        pygame.quit()

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This block is not run in the test environment, but is useful for local development.
    # It requires a display to be available.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a display window
    pygame.display.set_caption("Ecosystem Simulator")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    print("\n--- Manual Control ---")
    print("W/A/S/D or Arrow Keys to control resources.")
    print("W/Up: Water+ | S/Down: Water-")
    print("D/Right: Sun+ | A/Left: Sun-")
    print("Q to quit.")
    
    while running:
        # Default action is no-op
        movement_action = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                movement_action = 1
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                movement_action = 2
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                movement_action = 3
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                movement_action = 4
            
            # Action is [movement, space, shift]
            action = [movement_action, 0, 0]
            obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for manual play

    env.close()
    print("Game window closed.")