
# Generated: 2025-08-28T02:54:06.700616
# Source Brief: brief_01849.md
# Brief Index: 1849

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to plant/harvest. Shift at the barn to sell."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage your farm by planting, harvesting, and selling crops. Earn 1000 coins in 5 minutes to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Game parameters
    WIN_COINS = 1000
    MAX_TIME_SECONDS = 300
    MAX_STEPS = MAX_TIME_SECONDS * FPS
    CROP_PRICE = 5

    # Grid parameters
    GRID_COLS = 12
    GRID_ROWS = 8
    CELL_SIZE = 40
    GRID_OFFSET_X = 20
    GRID_OFFSET_Y = 60

    # Crop states
    STATE_EMPTY = 0
    STATE_PLANTED = 1
    STATE_SPROUT = 2
    STATE_RIPE = 3
    GROWTH_TIME = [0, 5 * FPS, 7 * FPS, 0] # Time in frames for each stage

    # Colors
    COLOR_SKY = (135, 206, 235)
    COLOR_GRASS = (34, 139, 34)
    COLOR_SOIL = (139, 69, 19)
    COLOR_BARN_ROOF = (178, 34, 34)
    COLOR_BARN_WALL = (210, 180, 140)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BACK = (0, 0, 0, 128)

    CROP_COLORS = {
        STATE_PLANTED: (0, 100, 0),
        STATE_SPROUT: (124, 252, 0),
        STATE_RIPE: (255, 215, 0),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
        
        # State variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = None
        self.field_data = None
        self.inventory = 0
        self.timer = 0
        self.win_message = ""
        self.particles = []

        # Barn position
        self.barn_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.GRID_COLS * self.CELL_SIZE + 30,
            self.GRID_OFFSET_Y + self.GRID_ROWS * self.CELL_SIZE / 2 - 50,
            100, 100
        )

        # Initialize state
        self.reset()

        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.timer = self.MAX_STEPS
        self.inventory = 0
        self.particles = []

        # Cursor starts in the middle of the grid
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        # Field data: [state, growth_timer]
        self.field_data = np.zeros((self.GRID_ROWS, self.GRID_COLS, 2), dtype=int)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Using pressed, not held, for single actions
        shift_pressed = action[2] == 1  # Using pressed, not held

        reward = 0
        
        # 1. Handle player actions
        self._handle_movement(movement)
        
        if space_pressed:
            reward += self._handle_plant_harvest()

        if shift_pressed:
            reward += self._handle_sell()

        # 2. Update game state (time-based)
        self._update_crops()
        self._update_particles()
        
        self.timer -= 1
        self.steps += 1
        
        # 3. Check for termination
        terminated = False
        if self.score >= self.WIN_COINS:
            reward += 100
            self.game_over = True
            self.win_message = "YOU WIN!"
            terminated = True
        elif self.timer <= 0:
            reward -= 10
            self.game_over = True
            self.win_message = "TIME'S UP!"
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] -= 1
        elif movement == 2: # Down
            self.cursor_pos[1] += 1
        elif movement == 3: # Left
            self.cursor_pos[0] -= 1
        elif movement == 4: # Right
            self.cursor_pos[0] += 1
        
        # Clamp cursor to grid boundaries
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

    def _handle_plant_harvest(self):
        cy, cx = self.cursor_pos[1], self.cursor_pos[0]
        plot_state, _ = self.field_data[cy, cx]
        
        if plot_state == self.STATE_EMPTY:
            # Plant a seed
            self.field_data[cy, cx, 0] = self.STATE_PLANTED
            self.field_data[cy, cx, 1] = self.GROWTH_TIME[self.STATE_PLANTED]
            self._create_particles(self._get_cursor_screen_pos(), 10, self.CROP_COLORS[self.STATE_SPROUT])
            # sfx: plant_seed.wav
            return 0.1
        
        if plot_state == self.STATE_RIPE:
            # Harvest a crop
            self.field_data[cy, cx, 0] = self.STATE_EMPTY
            self.field_data[cy, cx, 1] = 0
            self.inventory += 1
            self._create_particles(self._get_cursor_screen_pos(), 20, self.CROP_COLORS[self.STATE_RIPE])
            # sfx: harvest.wav
            return 0.2
        
        return 0

    def _handle_sell(self):
        cursor_screen_pos = self._get_cursor_screen_pos(center=True)
        if self.barn_rect.collidepoint(cursor_screen_pos) and self.inventory > 0:
            coins_earned = self.inventory * self.CROP_PRICE
            reward_earned = self.inventory * 1.0 # As per brief
            self.score += coins_earned
            self.inventory = 0
            self._create_particles(self.barn_rect.center, 30, (255, 215, 0), count_mult=self.inventory)
            # sfx: cash_register.wav
            return reward_earned
        return 0

    def _update_crops(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                state, timer = self.field_data[r, c]
                if self.STATE_PLANTED <= state < self.STATE_RIPE:
                    timer -= 1
                    if timer <= 0:
                        state += 1
                        timer = self.GROWTH_TIME[state]
                        self.field_data[r, c, 0] = state
                    self.field_data[r, c, 1] = timer

    def _get_cursor_screen_pos(self, center=False):
        x = self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE
        if center:
            x += self.CELL_SIZE // 2
            y += self.CELL_SIZE // 2
        return (x, y)
    
    def _render_game(self):
        # Render farm plots
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                plot_rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_SOIL, plot_rect)
                pygame.draw.rect(self.screen, self.COLOR_GRASS, plot_rect, 1)

                state, _ = self.field_data[r, c]
                if state > self.STATE_EMPTY:
                    self._render_crop(plot_rect.center, state)

        # Render barn
        pygame.draw.rect(self.screen, self.COLOR_BARN_WALL, self.barn_rect)
        roof_points = [
            (self.barn_rect.left, self.barn_rect.top),
            (self.barn_rect.centerx, self.barn_rect.top - 30),
            (self.barn_rect.right, self.barn_rect.top)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_BARN_ROOF, roof_points)
        pygame.draw.rect(self.screen, (0,0,0), self.barn_rect, 2)

        # Render cursor
        cursor_pos = self._get_cursor_screen_pos()
        cursor_rect = pygame.Rect(cursor_pos[0], cursor_pos[1], self.CELL_SIZE, self.CELL_SIZE)
        # Pulsing alpha for cursor
        alpha = 128 + 127 * math.sin(self.steps * 0.2)
        cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 3)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

        # Render particles
        self._render_particles()

    def _render_crop(self, center, state):
        x, y = center
        if state == self.STATE_PLANTED:
            pygame.gfxdraw.filled_circle(self.screen, x, y, 4, self.CROP_COLORS[state])
        elif state == self.STATE_SPROUT:
            pygame.draw.line(self.screen, self.CROP_COLORS[state], (x, y + 5), (x - 5, y - 5), 3)
            pygame.draw.line(self.screen, self.CROP_COLORS[state], (x, y + 5), (x + 5, y - 5), 3)
        elif state == self.STATE_RIPE:
            pygame.gfxdraw.filled_circle(self.screen, x, y, 12, self.CROP_COLORS[state])
            pygame.gfxdraw.aacircle(self.screen, x, y, 12, (255, 255, 255))


    def _render_ui(self):
        # UI Background panels
        s = pygame.Surface((160, 40), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BACK)
        self.screen.blit(s, (10, 10))
        self.screen.blit(s, (self.SCREEN_WIDTH - 170, 10))

        s_inv = pygame.Surface((200, 40), pygame.SRCALPHA)
        s_inv.fill(self.COLOR_UI_BACK)
        self.screen.blit(s_inv, (self.SCREEN_WIDTH / 2 - 100, self.SCREEN_HEIGHT - 50))

        # Coins display
        coin_text = self.font_small.render(f"COINS: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(coin_text, (20, 20))

        # Timer display
        time_left = max(0, self.timer / self.FPS)
        timer_text = self.font_small.render(f"TIME: {int(time_left)}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - 160, 20))

        # Inventory display
        inv_text = self.font_small.render(f"HARVESTED: {self.inventory}", True, self.COLOR_UI_TEXT)
        text_rect = inv_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 30))
        self.screen.blit(inv_text, text_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_SKY)
        pygame.draw.rect(self.screen, self.COLOR_GRASS, (0, self.GRID_OFFSET_Y - 10, self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "inventory": self.inventory,
            "time_left": max(0, self.timer / self.FPS),
        }

    # --- Particle System ---
    def _create_particles(self, pos, amount, color, count_mult=1):
        for _ in range(amount * int(count_mult)):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            particle = {
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(15, 30), # frames
                "color": color,
                "radius": random.uniform(2, 5)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (p["pos"][0] - p["radius"], p["pos"][1] - p["radius"]))

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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_window = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Farm Manager")
    clock = pygame.time.Clock()

    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_window.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            env.reset()

        clock.tick(env.FPS)

    env.close()