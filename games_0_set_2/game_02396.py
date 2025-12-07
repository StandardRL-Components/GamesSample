
# Generated: 2025-08-27T20:14:43.213531
# Source Brief: brief_02396.md
# Brief Index: 2396

        
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
        "Controls: Arrow keys to move cursor. Space to plant/harvest. Shift to sell all harvested crops."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Isometric farm simulator. Plant, grow, and harvest crops. Sell them to reach 1000 gold before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GOLD_GOAL = 1000
    MAX_STEPS = 180 * 30  # 180 seconds at 30 FPS

    GRID_W, GRID_H = 5, 5
    TILE_W, TILE_H = 80, 40

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_SOIL = (85, 65, 50)
    COLOR_SOIL_DARK = (65, 45, 30)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_UI_BG = (10, 20, 30, 200)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_GOLD = (255, 215, 0)
    COLOR_TIME_BAR = (70, 130, 180)
    COLOR_TIME_BAR_BG = (40, 60, 80)
    COLOR_SELL_POINT = (200, 180, 150)
    COLOR_SELL_POINT_ROOF = (180, 80, 80)

    # Crop Definitions
    CROP_DEFS = [
        {
            "name": "Carrot", "grow_time": 240, "price": 10,
            "growing_color": (255, 165, 0), "ready_color": (255, 100, 0)
        },
        {
            "name": "Cabbage", "grow_time": 450, "price": 25,
            "growing_color": (152, 251, 152), "ready_color": (0, 200, 0)
        }
    ]
    # "Plant most valuable" rule means we always plant Cabbage (index 1)
    PLANT_CROP_ID = 1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_small = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0  # Gold
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.farm_grid = []
        self.inventory = {}
        self.particles = []
        self.last_movement_action = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.rng = None
        
        self.grid_origin_x = self.SCREEN_WIDTH // 2
        self.grid_origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_H * self.TILE_H) // 4 + 40
        self.sell_point_pos = self._iso_to_screen(self.GRID_W, self.GRID_H // 2)

        # Initial reset to populate state
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.inventory = {i: 0 for i in range(len(self.CROP_DEFS))}
        self.particles = []

        self.farm_grid = [
            [
                {"state": "empty", "crop_id": None, "growth": 0}
                for _ in range(self.GRID_W)
            ]
            for _ in range(self.GRID_H)
        ]

        self.last_movement_action = 0
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        action_taken = False

        # --- Action Handling ---
        # Movement (on change of direction key)
        if movement != 0 and movement != self.last_movement_action:
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_H - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)

        # Plant / Harvest (on space press)
        if space_pressed:
            r, c = self.cursor_pos
            plot = self.farm_grid[r][c]
            if plot["state"] == "empty":
                # Plant
                plot["state"] = "growing"
                plot["crop_id"] = self.PLANT_CROP_ID
                plot["growth"] = 0
                action_taken = True
                # sfx: plant_seed.wav
            elif plot["state"] == "ready":
                # Harvest
                crop_id = plot["crop_id"]
                self.inventory[crop_id] += 1
                plot["state"] = "empty"
                plot["crop_id"] = None
                plot["growth"] = 0
                reward += 0.1  # Continuous feedback for harvesting
                action_taken = True
                self._create_harvest_particles(r, c, crop_id)
                # sfx: harvest.wav
        
        # Sell (on shift press)
        if shift_pressed:
            sell_value = 0
            for crop_id, count in self.inventory.items():
                if count > 0:
                    sell_value += count * self.CROP_DEFS[crop_id]["price"]
                    self.inventory[crop_id] = 0
            
            if sell_value > 0:
                self.score += sell_value
                reward += 1.0 + (sell_value * 0.01) # Event-based reward for selling
                action_taken = True
                self._create_sell_particles()
                # sfx: cash_register.wav

        if not action_taken:
            reward -= 0.01 # Small penalty for inaction

        self.last_movement_action = movement
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Game Logic Update ---
        # Grow crops
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                plot = self.farm_grid[r][c]
                if plot["state"] == "growing":
                    plot["growth"] += 1
                    if plot["growth"] >= self.CROP_DEFS[plot["crop_id"]]["grow_time"]:
                        plot["state"] = "ready"
                        # sfx: crop_ready.wav

        # Update particles
        self._update_particles()
        
        # --- Termination Check ---
        if self.score >= self.GOLD_GOAL:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward -= 100
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _iso_to_screen(self, row, col):
        x = self.grid_origin_x + (col - row) * (self.TILE_W / 2)
        y = self.grid_origin_y + (col + row) * (self.TILE_H / 2)
        return int(x), int(y)

    def _draw_iso_poly(self, surface, x, y, color, dark_color):
        points = [
            (x, y - self.TILE_H / 2),
            (x + self.TILE_W / 2, y),
            (x, y + self.TILE_H / 2),
            (x - self.TILE_W / 2, y),
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, dark_color)
        
    def _render_game(self):
        # Draw sell point first (behind grid)
        sx, sy = self.sell_point_pos
        pygame.draw.rect(self.screen, self.COLOR_SELL_POINT, (sx - 20, sy - 40, 40, 40))
        pygame.gfxdraw.filled_polygon(self.screen, [(sx - 25, sy-40), (sx, sy-60), (sx+25, sy-40)], self.COLOR_SELL_POINT_ROOF)
        
        # Draw grid and crops
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                x, y = self._iso_to_screen(r, c)
                self._draw_iso_poly(self.screen, x, y, self.COLOR_SOIL, self.COLOR_SOIL_DARK)
                
                plot = self.farm_grid[r][c]
                if plot["state"] != "empty":
                    crop_def = self.CROP_DEFS[plot["crop_id"]]
                    
                    if plot["state"] == "growing":
                        progress = plot["growth"] / crop_def["grow_time"]
                        radius = int(max(2, progress * self.TILE_H * 0.35))
                        color = crop_def["growing_color"]
                        pygame.gfxdraw.aacircle(self.screen, x, y - int(self.TILE_H * 0.3), radius, color)
                        pygame.gfxdraw.filled_circle(self.screen, x, y - int(self.TILE_H * 0.3), radius, color)
                    elif plot["state"] == "ready":
                        radius = int(self.TILE_H * 0.4)
                        color = crop_def["ready_color"]
                        # Pulsing effect for ready crops
                        pulse = abs(math.sin(self.steps * 0.1)) * 5
                        pygame.gfxdraw.aacircle(self.screen, x, y - int(self.TILE_H * 0.3), radius, color)
                        pygame.gfxdraw.filled_circle(self.screen, x, y - int(self.TILE_H * 0.3), radius, color)
                        pygame.gfxdraw.aacircle(self.screen, x, y - int(self.TILE_H * 0.3), radius + int(pulse), (*color, 50))

        # Draw cursor
        cx, cy = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 100 + 155
        cursor_color = (self.COLOR_CURSOR[0], self.COLOR_CURSOR[1], self.COLOR_CURSOR[2], int(pulse))
        points = [
            (cx, cy - self.TILE_H / 2), (cx + self.TILE_W / 2, cy),
            (cx, cy + self.TILE_H / 2), (cx - self.TILE_W / 2, cy)
        ]
        pygame.draw.lines(self.screen, cursor_color, True, points, 3)

        # Draw particles
        for p in self.particles:
            p_color = (*p['color'], p['alpha'])
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), p_color)

    def _render_ui(self):
        # UI Panel
        panel_surf = pygame.Surface((self.SCREEN_WIDTH, 80), pygame.SRCALPHA)
        panel_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(panel_surf, (0, 0))
        
        # Gold Display
        gold_text = self.font_medium.render(f"{self.score}", True, self.COLOR_UI_GOLD)
        gold_icon = self.font_medium.render("G", True, self.COLOR_UI_GOLD)
        self.screen.blit(gold_icon, (20, 10))
        self.screen.blit(gold_text, (45, 10))
        
        # Time Bar
        time_ratio = self.steps / self.MAX_STEPS
        bar_width = self.SCREEN_WIDTH - 40
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR_BG, (20, self.SCREEN_HEIGHT - 30, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR, (20, self.SCREEN_HEIGHT - 30, bar_width * (1 - time_ratio), 20))
        
        # Inventory Display
        inv_x = self.SCREEN_WIDTH - 200
        for i, crop_def in enumerate(self.CROP_DEFS):
            count = self.inventory[i]
            pygame.gfxdraw.filled_circle(self.screen, inv_x, 25, 10, crop_def["ready_color"])
            count_text = self.font_small.render(f"x{count}", True, self.COLOR_UI_TEXT)
            self.screen.blit(count_text, (inv_x + 15, 15))
            inv_x += 70

        # Game Over/Win Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            if self.score >= self.GOLD_GOAL:
                msg = "YOU WIN!"
                color = self.COLOR_UI_GOLD
            else:
                msg = "TIME UP!"
                color = self.COLOR_TIME_BAR
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text_surf, text_rect)

    def _create_harvest_particles(self, r, c, crop_id):
        x, y = self._iso_to_screen(r, c)
        color = self.CROP_DEFS[crop_id]["ready_color"]
        for _ in range(5):
            self.particles.append({
                "pos": [x, y - 20],
                "vel": [self.rng.uniform(-2, 2), self.rng.uniform(-3, -1)],
                "size": self.rng.integers(3, 6),
                "life": 30, "max_life": 30,
                "color": color, "alpha": 255
            })

    def _create_sell_particles(self):
        for _ in range(20):
            self.particles.append({
                "pos": [self.sell_point_pos[0], self.sell_point_pos[1] - 20],
                "vel": [self.rng.uniform(-3, 3), self.rng.uniform(-4, 0)],
                "size": self.rng.integers(4, 8),
                "life": 40, "max_life": 40,
                "color": self.COLOR_UI_GOLD, "alpha": 255
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            p['alpha'] = int(255 * (p['life'] / p['max_life']))
        self.particles = [p for p in self.particles if p['life'] > 0]

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Setup a window to display the game
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Farming Sim")
    
    # Game loop
    running = True
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get key presses
        keys = pygame.key.get_pressed()
        
        # Reset actions
        movement = 0
        space_held = 0
        shift_held = 0

        # Map keys to actions
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause before reset
            obs, info = env.reset()
            total_reward = 0

        # Control the frame rate
        env.clock.tick(30)
        
    env.close()