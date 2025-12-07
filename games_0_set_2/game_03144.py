import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to plant or harvest. Press Shift to sell crops."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage your isometric farm. Plant seeds, wait for them to grow, harvest the crops, and sell them at the barn to earn coins. Reach 1000 coins in 60 seconds to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 7
    TILE_WIDTH_ISO, TILE_HEIGHT_ISO = 64, 32
    ORIGIN_X, ORIGIN_Y = SCREEN_WIDTH // 2, 80

    MAX_TIME = 60 * 60  # 60 seconds at 60 FPS
    GROWTH_DURATION = 5 * 60  # 5 seconds
    WIN_SCORE = 1000
    CROP_SELL_VALUE = 10

    REWARD_HARVEST = 0.1
    REWARD_SELL_PER_CROP = 1.0
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0

    # --- Colors ---
    COLOR_BG = (10, 25, 35)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_BG = (30, 55, 75, 180)
    COLOR_TIME_BAR = (50, 205, 50)
    COLOR_TIME_BAR_BG = (70, 70, 70)
    COLOR_PLOT_EMPTY = (139, 69, 19)
    COLOR_PLOT_GROWING = (0, 100, 0)
    COLOR_PLOT_READY = (255, 215, 0)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_BARN_ROOF = (178, 34, 34)
    COLOR_BARN_WALL = (210, 180, 140)

    # --- Plot States ---
    STATE_EMPTY = 0
    STATE_GROWING = 1
    STATE_READY = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # State variables (will be properly initialized in reset)
        self.plots = []
        self.cursor_pos = [0, 0]
        self.score = 0
        self.harvested_crops = 0
        self.time_remaining = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []

        # Initialize state
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.harvested_crops = 0
        self.time_remaining = self.MAX_TIME
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles.clear()

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.plots = [
            [{"state": self.STATE_EMPTY, "growth": 0} for _ in range(self.GRID_HEIGHT)]
            for _ in range(self.GRID_WIDTH)
        ]

        return self._get_observation(), self._get_info()

    def _handle_input_and_get_reward(self, movement, space_pressed, shift_pressed):
        reward = 0.0
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Actions ---
        cx, cy = self.cursor_pos
        plot = self.plots[cx][cy]
        screen_pos = self._iso_to_screen(cx, cy)

        if space_pressed:
            if plot["state"] == self.STATE_EMPTY:
                plot["state"] = self.STATE_GROWING
                plot["growth"] = 0
                self._create_particles(screen_pos[0], screen_pos[1] - 10, 20, (124, 252, 0))
            elif plot["state"] == self.STATE_READY:
                plot["state"] = self.STATE_EMPTY
                self.harvested_crops += 1
                reward += self.REWARD_HARVEST
                self._create_particles(screen_pos[0], screen_pos[1] - 10, 20, self.COLOR_PLOT_READY)

        if shift_pressed and self.harvested_crops > 0:
            reward += self.harvested_crops * self.REWARD_SELL_PER_CROP
            self.score += self.harvested_crops * self.CROP_SELL_VALUE
            self._create_particles(60, 60, self.harvested_crops * 5, (255, 223, 0))
            self.harvested_crops = 0
        
        return reward

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        reward = 0.0
        if not self.game_over:
            reward += self._handle_input_and_get_reward(movement, space_pressed, shift_pressed)
            self._update_crops()
            self.steps += 1
            self.time_remaining -= 1

        self._update_particles()

        terminated = self.score >= self.WIN_SCORE or self.time_remaining <= 0
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += self.REWARD_WIN
                self._create_particles(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, 100, (255, 215, 0))
            else:
                reward += self.REWARD_LOSS

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _update_crops(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                plot = self.plots[x][y]
                if plot["state"] == self.STATE_GROWING:
                    plot["growth"] += 1
                    if plot["growth"] >= self.GROWTH_DURATION:
                        plot["state"] = self.STATE_READY
                        plot["growth"] = 0

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * (self.TILE_WIDTH_ISO / 2)
        screen_y = self.ORIGIN_Y + (x + y) * (self.TILE_HEIGHT_ISO / 2)
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw barn
        barn_base_x, barn_base_y = self._iso_to_screen(-1.5, self.GRID_HEIGHT -1)
        pygame.draw.rect(self.screen, self.COLOR_BARN_WALL, (barn_base_x, barn_base_y, 80, 60))
        roof_points = [(barn_base_x-10, barn_base_y), (barn_base_x+90, barn_base_y), (barn_base_x+40, barn_base_y-40)]
        pygame.draw.polygon(self.screen, self.COLOR_BARN_ROOF, roof_points)

        # Draw plots
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                plot = self.plots[x][y]
                screen_pos = self._iso_to_screen(x, y)
                
                # Plot polygon points
                points = [
                    (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT_ISO / 2),
                    (screen_pos[0] + self.TILE_WIDTH_ISO / 2, screen_pos[1]),
                    (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT_ISO / 2),
                    (screen_pos[0] - self.TILE_WIDTH_ISO / 2, screen_pos[1]),
                ]

                # Determine color
                if plot["state"] == self.STATE_EMPTY:
                    color = self.COLOR_PLOT_EMPTY
                elif plot["state"] == self.STATE_GROWING:
                    color = self.COLOR_PLOT_GROWING
                else: # READY
                    color = self.COLOR_PLOT_READY
                
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                # FIX: Convert the generator to a tuple/list of integers for the color argument.
                border_color = tuple(int(c * 0.8) for c in color)
                pygame.gfxdraw.aapolygon(self.screen, points, border_color)

                # Growth indicator
                if plot["state"] == self.STATE_GROWING:
                    progress = plot["growth"] / self.GROWTH_DURATION
                    radius = int(progress * (self.TILE_HEIGHT_ISO / 3))
                    pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, (144, 238, 144))
                    pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius, (144, 238, 144))

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        screen_pos = self._iso_to_screen(cursor_x, cursor_y)
        points = [
            (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT_ISO / 2),
            (screen_pos[0] + self.TILE_WIDTH_ISO / 2, screen_pos[1]),
            (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT_ISO / 2),
            (screen_pos[0] - self.TILE_WIDTH_ISO / 2, screen_pos[1]),
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, points, 3)

        self._draw_particles()

    def _render_ui(self):
        # UI Background Panel
        panel_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 40)
        s = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (0,0))
        
        # Score
        score_text = self.font_ui.render(f"Coins: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Harvested Crops
        crop_text = self.font_ui.render(f"Harvested: {self.harvested_crops}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crop_text, (150, 10))

        # Time Bar
        time_bar_width = 200
        time_ratio = self.time_remaining / self.MAX_TIME
        current_width = int(time_bar_width * time_ratio)
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR_BG, (self.SCREEN_WIDTH - time_bar_width - 10, 10, time_bar_width, 20))
        if current_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_TIME_BAR, (self.SCREEN_WIDTH - time_bar_width - 10, 10, current_width, 20))

        # Game Over Text
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = (255, 215, 0)
            else:
                msg = "TIME'S UP!"
                color = (200, 0, 0)
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            s = pygame.Surface(text_rect.inflate(20,20).size, pygame.SRCALPHA)
            s.fill((0,0,0,150))
            self.screen.blit(s, text_rect.inflate(20,20).topleft)
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "harvested_crops": self.harvested_crops,
            "cursor_pos": self.cursor_pos
        }
        
    def _create_particles(self, x, y, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            lifespan = random.randint(20, 40)
            self.particles.append({
                "pos": [x, y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": lifespan,
                "max_life": lifespan,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_life"]))
            color = (*p["color"], alpha)
            size = int(5 * (p["lifespan"] / p["max_life"]))
            if size > 0:
                rect = pygame.Rect(p["pos"][0] - size // 2, p["pos"][1] - size // 2, size, size)
                # Use a surface for transparency
                s = pygame.Surface((size, size), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size // 2, size // 2), size // 2)
                self.screen.blit(s, rect.topleft)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the game directly for testing
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Isometric Farm Manager")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Get human input ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}")

        if done:
            print("Game Over!")
            print(f"Final Score: {info['score']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        # --- Render the observation ---
        # The observation is already a numpy array, convert it back to a Pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Cap the frame rate ---
        clock.tick(60)

    pygame.quit()