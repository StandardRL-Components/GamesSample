
# Generated: 2025-08-28T00:03:20.677891
# Source Brief: brief_03668.md
# Brief Index: 3668

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, vx, vy, life, color, radius, gravity=0.1):
        self.pos = [x, y]
        self.vel = [vx, vy]
        self.life = life
        self.color = color
        self.radius = radius
        self.gravity = gravity

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] += self.gravity
        self.life -= 1
        if self.radius > 0.1:
            self.radius -= 0.05

    def draw(self, surface):
        if self.life > 0:
            pos = (int(self.pos[0]), int(self.pos[1]))
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(self.radius), self.color)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(self.radius), self.color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to select plot/market. Space to plant, harvest, or sell."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage a small farm to earn 1000 gold. Plant seeds, harvest crops, and sell them at the market before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    WIN_GOLD = 1000
    STARTING_GOLD = 10
    PLANT_COST = 1
    PRODUCE_VALUE = 5
    CROP_GROW_TIME = 80
    GRID_ROWS = 2
    GRID_COLS = 2

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_SOIL = (85, 60, 40)
    COLOR_PLOT_BORDER = (65, 40, 20)
    COLOR_SPROUT = (120, 200, 80)
    COLOR_READY = (255, 220, 50)
    COLOR_READY_GLOW = (255, 220, 50, 60)
    COLOR_MARKET = (130, 90, 60)
    COLOR_MARKET_AWNING = [(220, 220, 220), (200, 50, 50)]
    COLOR_CURSOR = (255, 255, 255, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_GOLD = (255, 215, 0)
    COLOR_PRODUCE = (230, 180, 40)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.gold = 0
        self.game_over = False
        self.plots = []
        self.plot_timers = []
        self.produce_collected = 0
        self.cursor_pos = [0, 0]
        self.particles = []
        self.rng = None
        
        # Initialize state variables
        self.reset()

        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()
        
        self.steps = 0
        self.gold = self.STARTING_GOLD
        self.game_over = False
        self.plots = [[0] * self.GRID_COLS for _ in range(self.GRID_ROWS)]
        self.plot_timers = [[0] * self.GRID_COLS for _ in range(self.GRID_ROWS)]
        self.produce_collected = 0
        self.cursor_pos = [0, 0]
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1 # Unused in this game
        
        reward = 0
        self.game_over = self.steps >= self.MAX_STEPS or self.gold >= self.WIN_GOLD
        if self.game_over:
            # If game is over, do nothing but allow observation
            if self.gold >= self.WIN_GOLD:
                reward = 100.0
            else: # Timeout
                reward = -10.0
            return self._get_observation(), reward, True, False, self._get_info()

        # 1. Handle cursor movement
        self._handle_movement(movement)
        
        # 2. Handle interaction
        if space_pressed:
            interaction_reward = self._handle_interaction()
            reward += interaction_reward
        
        # 3. Update game state
        self._update_plots()
        self._update_particles()
        
        # 4. Calculate ongoing penalties
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.plots[r][c] == 2: # Ready to harvest but not harvested
                    reward -= 0.01

        self.steps += 1
        
        # 5. Check for termination
        terminated = self.steps >= self.MAX_STEPS or self.gold >= self.WIN_GOLD
        if terminated and not self.game_over:
            if self.gold >= self.WIN_GOLD:
                reward += 100.0
            else: # Timeout
                reward -= -10.0
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        r, c = self.cursor_pos
        if movement == 1: # Up
            r = max(0, r - 1)
        elif movement == 2: # Down
            r = min(self.GRID_ROWS, r + 1)
        elif movement == 3: # Left
            c = max(0, c - 1)
        elif movement == 4: # Right
            c = min(self.GRID_COLS - 1, c + 1)
        
        # The market is on row `GRID_ROWS`, but only at column 0
        if r == self.GRID_ROWS:
            c = 0
        
        self.cursor_pos = [r, c]

    def _handle_interaction(self):
        r, c = self.cursor_pos
        # Market interaction
        if r == self.GRID_ROWS:
            if self.produce_collected > 0:
                # SFX: Cha-ching!
                amount_sold = self.produce_collected
                gold_earned = amount_sold * self.PRODUCE_VALUE
                self.gold += gold_earned
                self.produce_collected = 0
                self._create_coin_particles(20, self.SCREEN_WIDTH - 120, 40)
                return 1.0 + (amount_sold / 10.0)
        # Plot interaction
        else:
            plot_state = self.plots[r][c]
            # Plant
            if plot_state == 0 and self.gold >= self.PLANT_COST:
                # SFX: Plant seed
                self.plots[r][c] = 1
                self.plot_timers[r][c] = 0
                self.gold -= self.PLANT_COST
                px, py = self._get_plot_center(r, c)
                self._create_dust_particles(10, px, py)
                return 0.0
            # Harvest
            elif plot_state == 2:
                # SFX: Pop!
                self.plots[r][c] = 0
                self.produce_collected += 1
                px, py = self._get_plot_center(r, c)
                self._create_harvest_particles(15, px, py)
                return 0.1
        return 0.0 # No valid action taken

    def _update_plots(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.plots[r][c] == 1: # Planted
                    self.plot_timers[r][c] += 1
                    if self.plot_timers[r][c] >= self.CROP_GROW_TIME:
                        # SFX: Ding!
                        self.plots[r][c] = 2 # Ready to harvest

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render plots
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                plot_rect = self._get_plot_rect(r, c)
                pygame.draw.rect(self.screen, self.COLOR_SOIL, plot_rect)
                pygame.draw.rect(self.screen, self.COLOR_PLOT_BORDER, plot_rect, 3)

                state = self.plots[r][c]
                center_x, center_y = plot_rect.center
                if state == 1: # Growing
                    progress = self.plot_timers[r][c] / self.CROP_GROW_TIME
                    radius = int(5 + 25 * progress)
                    color = self.COLOR_SPROUT
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                elif state == 2: # Ready
                    glow_radius = 40
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, self.COLOR_READY_GLOW)
                    radius = 30
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_READY)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_READY)

        # Render market
        market_rect = self._get_market_rect()
        pygame.draw.rect(self.screen, self.COLOR_MARKET, market_rect)
        for i in range(10):
            stripe_color = self.COLOR_MARKET_AWNING[i % 2]
            stripe_rect = pygame.Rect(market_rect.left + i * (market_rect.width/10), market_rect.top, market_rect.width/10, 15)
            pygame.draw.rect(self.screen, stripe_color, stripe_rect)
        pygame.draw.rect(self.screen, self.COLOR_PLOT_BORDER, market_rect, 3)
        
        produce_text = self.font_medium.render(f"{self.produce_collected}", True, self.COLOR_PRODUCE)
        produce_rect = produce_text.get_rect(center=(market_rect.centerx, market_rect.centery + 10))
        self.screen.blit(produce_text, produce_rect)

        # Render cursor
        cursor_r, cursor_c = self.cursor_pos
        if cursor_r == self.GRID_ROWS:
            highlight_rect = self._get_market_rect().inflate(10, 10)
        else:
            highlight_rect = self._get_plot_rect(cursor_r, cursor_c).inflate(10, 10)
        
        # Use a surface for transparency
        s = pygame.Surface(highlight_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), border_radius=8)
        self.screen.blit(s, highlight_rect.topleft)

        # Render particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Gold display
        gold_text = self.font_large.render(f"GOLD: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (20, 20))

        # Time display
        time_left = self.MAX_STEPS - self.steps
        time_color = self.COLOR_TEXT if time_left > 100 else (255, 80, 80)
        time_text = self.font_large.render(f"TIME: {time_left}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(time_text, time_rect)

        # User guide
        guide_text = self.font_small.render(self.user_guide, True, self.COLOR_TEXT)
        guide_rect = guide_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20))
        self.screen.blit(guide_text, guide_rect)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _create_particles(self, count, x, y, color, speed, life, gravity, radius):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            s = self.rng.uniform(0.5, speed)
            vx = math.cos(angle) * s
            vy = math.sin(angle) * s
            p_life = life + self.rng.randint(-life//4, life//4)
            self.particles.append(Particle(x, y, vx, vy, p_life, color, radius, gravity))

    def _create_dust_particles(self, count, x, y):
        self._create_particles(count, x, y, self.COLOR_SOIL, 1.5, 20, 0.05, 4)
    
    def _create_harvest_particles(self, count, x, y):
        self._create_particles(count, x, y, self.COLOR_PRODUCE, 3, 30, 0.1, 5)

    def _create_coin_particles(self, count, x, y):
        self._create_particles(count, x, y, self.COLOR_GOLD, 4, 40, 0.15, 6)

    def _get_plot_rect(self, r, c):
        plot_size = 100
        spacing = 20
        grid_width = self.GRID_COLS * plot_size + (self.GRID_COLS - 1) * spacing
        grid_height = self.GRID_ROWS * plot_size + (self.GRID_ROWS - 1) * spacing
        start_x = (self.SCREEN_WIDTH - grid_width) / 2
        start_y = 80
        
        x = start_x + c * (plot_size + spacing)
        y = start_y + r * (plot_size + spacing)
        return pygame.Rect(int(x), int(y), plot_size, plot_size)

    def _get_plot_center(self, r, c):
        rect = self._get_plot_rect(r, c)
        return rect.centerx, rect.centery

    def _get_market_rect(self):
        plot_rect = self._get_plot_rect(self.GRID_ROWS - 1, 0)
        market_y = plot_rect.bottom + 20
        market_width = self.GRID_COLS * 100 + (self.GRID_COLS - 1) * 20
        market_x = (self.SCREEN_WIDTH - market_width) / 2
        return pygame.Rect(int(market_x), int(market_y), int(market_width), 60)

    def _get_info(self):
        return {
            "score": self.gold,
            "steps": self.steps,
            "produce": self.produce_collected,
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Key mapping
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Farm Manager")
    clock = pygame.time.Clock()

    running = True
    while running:
        movement_action = 0
        space_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # This simple polling is fine for a turn-based game
        # For a real-time game, you'd want to handle KEYDOWN events
        for key, move in key_to_action.items():
            if keys[key]:
                movement_action = move
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space_action = 1
            
        action = [movement_action, space_action, 0] # Shift is not used
        
        # We only step if an action is taken in a non-auto_advance game
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Gold: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")
            if terminated:
                print("Game Over!")
                # Wait a bit before resetting
                pygame.time.wait(2000)
                obs, info = env.reset()

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(10) # Limit frame rate for human play

    env.close()