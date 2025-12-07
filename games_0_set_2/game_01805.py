
# Generated: 2025-08-28T02:45:58.243094
# Source Brief: brief_01805.md
# Brief Index: 1805

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to plant wheat on empty soil or harvest a ripe crop. "
        "Move to the market (bottom right) and press Shift to sell all harvested crops."
    )

    # User-facing game description
    game_description = (
        "Fast-paced farming fun! Plant, grow, and harvest crops against the clock. "
        "Sell your produce at the market to reach the target earnings before time runs out."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 60
        self.MAX_FRAMES = self.GAME_DURATION_SECONDS * self.FPS
        self.WIN_SCORE = 500

        # Grid layout
        self.GRID_COLS, self.GRID_ROWS = 10, 6
        self.CELL_SIZE = 36
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X_START = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_START = 80

        # Crop data: {name: {time_frames, value, color}}
        self.CROP_DATA = {
            "wheat": {"time": 5 * self.FPS, "value": 10, "color": (245, 222, 179)},
        }
        self.PLANT_TYPE = "wheat"

        # State enums for grid plots
        self.PLOT_EMPTY = 0
        self.PLOT_PLANTED = 1
        self.PLOT_RIPE = 2

        # Colors
        self.COLOR_BG = (40, 50, 45)
        self.COLOR_GRID = (60, 80, 70)
        self.COLOR_SOIL = (92, 64, 51)
        self.COLOR_CURSOR = (0, 255, 255)
        self.COLOR_TEXT = (255, 255, 230)
        self.COLOR_RIPE_GLOW = (255, 255, 0)
        self.COLOR_SPROUT = (144, 238, 144)
        self.COLOR_MARKET_BG = (139, 69, 19)
        self.COLOR_MARKET_ROOF = (165, 42, 42)

        # Market Stall
        self.MARKET_RECT = pygame.Rect(self.SCREEN_WIDTH - 120, self.SCREEN_HEIGHT - 100, 100, 80)

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
        self.font_large = pygame.font.SysFont('monospace', 24, bold=True)
        self.font_medium = pygame.font.SysFont('monospace', 18)
        self.font_small = pygame.font.SysFont('monospace', 14)

        # State variables (initialized in reset)
        self.grid = []
        self.cursor_pos = [0, 0]
        self.cursor_cooldown = 0
        self.score = 0
        self.frames_elapsed = 0
        self.steps = 0
        self.game_over = False
        self.harvested_inventory = {}
        self.floating_texts = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        # Initialize game state
        self.grid = [[{"state": self.PLOT_EMPTY} for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.cursor_cooldown = 0
        self.score = 0
        self.frames_elapsed = 0
        self.steps = 0
        self.game_over = False
        self.harvested_inventory = {crop: 0 for crop in self.CROP_DATA}
        self.floating_texts = []
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True # Prevent action on first frame

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.frames_elapsed += 1
        self.steps += 1

        self._handle_actions(action)
        reward += self._update_game_state()
        self._update_particles()

        terminated = self._check_termination()
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Cooldown for smoother cursor movement
        if self.cursor_cooldown > 0:
            self.cursor_cooldown -= 1
        
        if self.cursor_cooldown == 0:
            if movement == 1:  # Up
                self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
                self.cursor_cooldown = 5
            elif movement == 2:  # Down
                self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
                self.cursor_cooldown = 5
            elif movement == 3:  # Left
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
                self.cursor_cooldown = 5
            elif movement == 4:  # Right
                self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
                self.cursor_cooldown = 5

        # Check for key presses (rising edge)
        is_space_press = space_held and not self.prev_space_held
        is_shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # Plant/Harvest action
        if is_space_press:
            cx, cy = self.cursor_pos
            plot = self.grid[cy][cx]
            if plot["state"] == self.PLOT_EMPTY:
                # Sfx: Plant seed
                plot["state"] = self.PLOT_PLANTED
                plot["type"] = self.PLANT_TYPE
                plot["growth"] = 0
            elif plot["state"] == self.PLOT_RIPE:
                # Sfx: Harvest
                crop_type = plot["type"]
                self.harvested_inventory[crop_type] += 1
                plot["state"] = self.PLOT_EMPTY
        
        # Sell action
        cursor_screen_x = self.GRID_X_START + self.cursor_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        cursor_screen_y = self.GRID_Y_START + self.cursor_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        cursor_rect = pygame.Rect(cursor_screen_x - 5, cursor_screen_y - 5, 10, 10)

        if is_shift_press and self.MARKET_RECT.colliderect(cursor_rect):
            money_earned = 0
            for crop_type, count in self.harvested_inventory.items():
                if count > 0:
                    money_earned += count * self.CROP_DATA[crop_type]["value"]
                    self.harvested_inventory[crop_type] = 0
            
            if money_earned > 0:
                # Sfx: Cha-ching!
                self.score += money_earned
                self._create_floating_text(f"+${money_earned}", (self.MARKET_RECT.centerx, self.MARKET_RECT.top), (255, 215, 0))
                return money_earned # Return as part of reward

    def _update_game_state(self):
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                plot = self.grid[y][x]
                if plot["state"] == self.PLOT_PLANTED:
                    plot["growth"] += 1
                    max_growth = self.CROP_DATA[plot["type"]]["time"]
                    if plot["growth"] >= max_growth:
                        # Sfx: Crop ripe!
                        plot["state"] = self.PLOT_RIPE
        return 0

    def _update_particles(self):
        for text in self.floating_texts[:]:
            text['y'] -= text['vy']
            text['life'] -= 1
            text['alpha'] = max(0, 255 * (text['life'] / text['max_life']))
            if text['life'] <= 0:
                self.floating_texts.remove(text)

    def _check_termination(self):
        if self.score >= self.WIN_SCORE or self.frames_elapsed >= self.MAX_FRAMES:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and plots
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                rect = pygame.Rect(
                    self.GRID_X_START + x * self.CELL_SIZE,
                    self.GRID_Y_START + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_SOIL, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                plot = self.grid[y][x]
                center_x, center_y = rect.center
                
                if plot["state"] == self.PLOT_PLANTED:
                    max_growth = self.CROP_DATA[plot["type"]]["time"]
                    growth_ratio = plot["growth"] / max_growth
                    radius = int(2 + (self.CELL_SIZE // 2 - 4) * growth_ratio)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_SPROUT)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_SPROUT)

                elif plot["state"] == self.PLOT_RIPE:
                    base_radius = self.CELL_SIZE // 2 - 4
                    # Pulsing glow effect
                    pulse = (math.sin(self.frames_elapsed * 0.2) + 1) / 2
                    glow_radius = base_radius + int(3 * pulse)
                    glow_color = (*self.COLOR_RIPE_GLOW, int(50 + 100 * pulse))
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, glow_color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, glow_radius, glow_color)
                    
                    # Main crop body
                    crop_color = self.CROP_DATA[plot["type"]]["color"]
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, base_radius, crop_color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, base_radius, crop_color)

        # Draw market
        pygame.draw.rect(self.screen, self.COLOR_MARKET_BG, self.MARKET_RECT)
        roof_points = [
            (self.MARKET_RECT.left, self.MARKET_RECT.top),
            (self.MARKET_RECT.right, self.MARKET_RECT.top),
            (self.MARKET_RECT.centerx, self.MARKET_RECT.top - 20)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_MARKET_ROOF, roof_points)
        market_text = self.font_medium.render("$", True, self.COLOR_TEXT)
        self.screen.blit(market_text, market_text.get_rect(center=self.MARKET_RECT.center))

        # Draw cursor
        cursor_x = self.GRID_X_START + self.cursor_pos[0] * self.CELL_SIZE
        cursor_y = self.GRID_Y_START + self.cursor_pos[1] * self.CELL_SIZE
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Money: ${self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.GAME_DURATION_SECONDS - self.frames_elapsed / self.FPS)
        time_color = (255, 100, 100) if time_left < 10 else self.COLOR_TEXT
        time_text = self.font_large.render(f"Time: {int(time_left):02d}", True, time_color)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Goal
        goal_text = self.font_medium.render(f"Goal: ${self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(goal_text, (10, 40))

        # Harvested inventory
        inv_y = self.SCREEN_HEIGHT - 30
        total_harvested = sum(self.harvested_inventory.values())
        inv_text_str = f"Harvested: {total_harvested}"
        inv_text = self.font_medium.render(inv_text_str, True, self.COLOR_TEXT)
        self.screen.blit(inv_text, (10, inv_y))

        # Floating text particles
        for text in self.floating_texts:
            rendered_text = self.font_medium.render(text['text'], True, text['color'])
            rendered_text.set_alpha(text['alpha'])
            text_rect = rendered_text.get_rect(center=(text['x'], text['y']))
            self.screen.blit(rendered_text, text_rect)

        # Game Over/Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                msg = "GOAL REACHED!"
                color = (100, 255, 100)
            else:
                msg = "TIME'S UP!"
                color = (255, 100, 100)
                
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.GAME_DURATION_SECONDS - self.frames_elapsed / self.FPS),
        }

    def _create_floating_text(self, text, pos, color):
        self.floating_texts.append({
            'text': text,
            'x': pos[0],
            'y': pos[1],
            'vy': 1.5,
            'life': self.FPS, # Lasts 1 second
            'max_life': self.FPS,
            'alpha': 255,
            'color': color
        })

    def close(self):
        pygame.font.quit()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run pygame in a window
    import os
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Farming Simulator")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print(env.user_guide)

    while not done:
        # --- Action mapping for human keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to screen ---
        # The observation is already a rendered frame, so we just need to display it
        # Pygame uses (width, height), numpy uses (height, width)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()