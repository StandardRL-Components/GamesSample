
# Generated: 2025-08-27T23:28:59.582746
# Source Brief: brief_03474.md
# Brief Index: 3474

        
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

    user_guide = (
        "Controls: Arrows to move cursor. Press Space to plant/harvest/sell. Press Shift to cycle crop type."
    )

    game_description = (
        "Manage a small isometric farm to maximize profit within a time limit by planting, harvesting, and selling crops."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame setup ---
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 15)
        self.font_m = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 30, bold=True)

        # --- Game constants ---
        self.FPS = 30
        self.MAX_STEPS = 180 * self.FPS
        self.STAGE_DURATION = 60 * self.FPS
        self.WIN_INCOME = 1000
        self.GRID_SIZE = 10
        self.MOVE_COOLDOWN_FRAMES = 4

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID_LINE = (40, 50, 60)
        self.COLOR_SOIL = (87, 56, 40)
        self.COLOR_SOIL_DARK = (69, 45, 32)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_BARN_ROOF = (200, 50, 50)
        self.COLOR_BARN_WALL = (220, 200, 180)

        # --- Crop Definitions ---
        self.CROPS = {
            "wheat": {"name": "Wheat", "growth_time": 10 * self.FPS, "value": 10, "color": (245, 222, 179), "ripe_color": (255, 215, 0)},
            "carrot": {"name": "Carrot", "growth_time": 15 * self.FPS, "value": 25, "color": (255, 165, 0), "ripe_color": (230, 100, 20)},
            "strawberry": {"name": "Strawberry", "growth_time": 20 * self.FPS, "value": 50, "color": (220, 20, 60), "ripe_color": (180, 0, 30)},
        }
        self.CROP_TYPES = list(self.CROPS.keys())

        # --- Isometric projection values ---
        self.tile_w = 40
        self.tile_h = 20
        self.origin_x = self.WIDTH // 2 - self.tile_w / 2
        self.origin_y = 120

        # --- State variables (will be initialized in reset) ---
        self.steps = 0
        self.total_income = 0
        self.game_over = False
        self.stage = 1
        self.farm_grid = []
        self.cursor_pos = [0, 0]
        self.barn_pos = [self.GRID_SIZE // 2, -2]
        self.available_crops = []
        self.selected_crop_index = 0
        self.harvested_produce = {}
        self.floating_texts = []
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.move_cooldown = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.total_income = 0
        self.game_over = False
        self.stage = 0 # Will be set to 1 in _init_stage
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_crop_index = 0
        
        self.floating_texts = []
        self.particles = []
        
        self.last_space_held = False
        self.last_shift_held = False
        self.move_cooldown = 0
        
        self._init_stage()

        return self._get_observation(), self._get_info()
    
    def _init_stage(self):
        self.stage += 1
        self.farm_grid = [
            [{"state": "empty", "crop_type": None, "growth": 0} for _ in range(self.GRID_SIZE)]
            for _ in range(self.GRID_SIZE)
        ]
        self.harvested_produce = {crop: 0 for crop in self.CROP_TYPES}
        
        if self.stage == 1:
            self.available_crops = [self.CROP_TYPES[0]]
        elif self.stage == 2:
            self.available_crops = self.CROP_TYPES[:2]
        else: # Stage 3 and beyond
            self.available_crops = self.CROP_TYPES
        
        self.selected_crop_index = 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        step_reward = -0.01  # Small penalty for time passing

        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        reward_from_action = self._handle_input(movement, space_pressed, shift_pressed)
        step_reward += reward_from_action

        terminal_reward = self._update_game_state()
        step_reward += terminal_reward

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        terminated = self.game_over

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        reward = 0
        
        # --- Movement ---
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        
        if movement != 0 and self.move_cooldown == 0:
            new_pos = list(self.cursor_pos)
            if movement == 1: new_pos[1] -= 1  # Up
            elif movement == 2: new_pos[1] += 1 # Down
            elif movement == 3: new_pos[0] -= 1 # Left
            elif movement == 4: new_pos[0] += 1 # Right
            
            # Clamp to grid or allow moving to barn
            is_barn = new_pos[0] == self.barn_pos[0] and new_pos[1] == self.barn_pos[1]
            if (0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE) or is_barn:
                self.cursor_pos = new_pos
                self.move_cooldown = self.MOVE_COOLDOWN_FRAMES

        # --- Shift Press: Cycle Crop ---
        if shift_pressed:
            self.selected_crop_index = (self.selected_crop_index + 1) % len(self.available_crops)
            # sfx: menu_click

        # --- Space Press: Contextual Action ---
        if space_pressed:
            cx, cy = self.cursor_pos
            # Action: Sell at barn
            if cx == self.barn_pos[0] and cy == self.barn_pos[1]:
                reward += self._sell_produce()
            # Action: Plant or Harvest on grid
            elif 0 <= cx < self.GRID_SIZE and 0 <= cy < self.GRID_SIZE:
                plot = self.farm_grid[cy][cx]
                if plot["state"] == "empty":
                    self._plant_crop(cx, cy)
                elif plot["state"] == "ripe":
                    reward += self._harvest_crop(cx, cy)
        
        return reward

    def _update_game_state(self):
        self.steps += 1
        terminal_reward = 0

        # --- Update crop growth ---
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                plot = self.farm_grid[y][x]
                if plot["state"] == "growing":
                    plot["growth"] += 1
                    crop_info = self.CROPS[plot["crop_type"]]
                    if plot["growth"] >= crop_info["growth_time"]:
                        plot["state"] = "ripe"

        # --- Update animations ---
        self.floating_texts = [t for t in self.floating_texts if t.update()]
        self.particles = [p for p in self.particles if p.update()]

        # --- Check for stage transition ---
        if self.steps in [self.STAGE_DURATION, self.STAGE_DURATION * 2]:
            self._init_stage()

        # --- Check for game over conditions ---
        if self.total_income >= self.WIN_INCOME:
            self.game_over = True
            terminal_reward = 100
            self._create_floating_text("YOU WIN!", self.WIDTH//2, self.HEIGHT//2, 120, self.font_l, (100, 255, 100))
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminal_reward = -10
            self._create_floating_text("TIME UP", self.WIDTH//2, self.HEIGHT//2, 120, self.font_l, (255, 100, 100))
        
        return terminal_reward
    
    def _plant_crop(self, x, y):
        plot = self.farm_grid[y][x]
        selected_crop_type = self.available_crops[self.selected_crop_index]
        plot["state"] = "growing"
        plot["crop_type"] = selected_crop_type
        plot["growth"] = 0
        # sfx: plant_seed
        screen_pos = self._iso_to_screen(x, y)
        self._create_particles(screen_pos[0], screen_pos[1], 10, self.COLOR_SOIL_DARK)

    def _harvest_crop(self, x, y):
        plot = self.farm_grid[y][x]
        crop_type = plot["crop_type"]
        self.harvested_produce[crop_type] += 1
        plot["state"] = "empty"
        plot["crop_type"] = None
        plot["growth"] = 0
        # sfx: harvest_pop
        screen_pos = self._iso_to_screen(x, y)
        crop_info = self.CROPS[crop_type]
        self._create_particles(screen_pos[0], screen_pos[1], 20, crop_info["ripe_color"])
        return 0.1 # Reward for harvesting

    def _sell_produce(self):
        income_this_sale = 0
        has_produce = False
        for crop_type, count in self.harvested_produce.items():
            if count > 0:
                has_produce = True
                income_this_sale += count * self.CROPS[crop_type]["value"]
                self.harvested_produce[crop_type] = 0
        
        if has_produce:
            self.total_income += income_this_sale
            # sfx: cash_register
            screen_pos = self._iso_to_screen(self.barn_pos[0], self.barn_pos[1])
            self._create_floating_text(f"+${income_this_sale}", screen_pos[0], screen_pos[1] - 30, 60, self.font_m, (100, 255, 100))
            return 1.0 # Reward for selling
        return 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_farm()
        self._render_barn()
        self._render_cursor()
        self._render_harvested_produce()
        self._render_particles()
        self._render_ui()
        self._render_floating_texts()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.total_income,
            "steps": self.steps,
        }
        
    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * self.tile_w / 2
        screen_y = self.origin_y + (x + y) * self.tile_h / 2
        return int(screen_x), int(screen_y)

    def _render_farm(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                screen_pos = self._iso_to_screen(x, y)
                self._draw_iso_tile(screen_pos, self.COLOR_SOIL, self.COLOR_SOIL_DARK)
                
                plot = self.farm_grid[y][x]
                if plot["state"] in ["growing", "ripe"]:
                    self._render_crop(plot, screen_pos)

    def _render_crop(self, plot, pos):
        crop_info = self.CROPS[plot["crop_type"]]
        progress = min(1.0, plot["growth"] / crop_info["growth_time"])
        
        base_color = crop_info["color"]
        ripe_color = crop_info["ripe_color"]
        current_color = (
            int(base_color[0] + (ripe_color[0] - base_color[0]) * progress),
            int(base_color[1] + (ripe_color[1] - base_color[1]) * progress),
            int(base_color[2] + (ripe_color[2] - base_color[2]) * progress),
        )

        if plot["state"] == "ripe":
            current_color = ripe_color
        
        # Simple visual representation: growing circle
        max_radius = self.tile_h * 0.6
        radius = int(max_radius * math.sqrt(progress)) # sqrt for non-linear growth feel
        
        if radius > 0:
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1] - 10, radius, current_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1] - 10, radius, current_color)

    def _render_barn(self):
        pos = self._iso_to_screen(self.barn_pos[0], self.barn_pos[1])
        # Draw base
        self._draw_iso_tile((pos[0], pos[1] + 10), self.COLOR_BARN_WALL, (180, 160, 140))
        
        # Draw walls
        wall_points = [
            (pos[0], pos[1]),
            (pos[0] - self.tile_w // 2, pos[1] - self.tile_h // 2),
            (pos[0] - self.tile_w // 2, pos[1] - self.tile_h // 2 - 30),
            (pos[0], pos[1] - 30),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, wall_points, self.COLOR_BARN_WALL)
        pygame.gfxdraw.aapolygon(self.screen, wall_points, self.COLOR_BARN_WALL)
        
        wall_points_2 = [
            (pos[0], pos[1]),
            (pos[0] + self.tile_w // 2, pos[1] - self.tile_h // 2),
            (pos[0] + self.tile_w // 2, pos[1] - self.tile_h // 2 - 30),
            (pos[0], pos[1] - 30),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, wall_points_2, (200, 180, 160))
        pygame.gfxdraw.aapolygon(self.screen, wall_points_2, (200, 180, 160))

        # Draw roof
        roof_points = [
            (pos[0], pos[1] - 50),
            (pos[0] - self.tile_w // 2, pos[1] - self.tile_h // 2 - 30),
            (pos[0], pos[1] - 30),
            (pos[0] + self.tile_w // 2, pos[1] - self.tile_h // 2 - 30),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, roof_points, self.COLOR_BARN_ROOF)
        pygame.gfxdraw.aapolygon(self.screen, roof_points, self.COLOR_BARN_ROOF)
        
    def _render_cursor(self):
        pos = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        points = [
            (pos[0], pos[1] - self.tile_h / 2),
            (pos[0] + self.tile_w / 2, pos[1]),
            (pos[0], pos[1] + self.tile_h / 2),
            (pos[0] - self.tile_w / 2, pos[1]),
        ]
        # Draw a thicker line by drawing multiple times
        for i in range(2):
            pygame.draw.aalines(self.screen, self.COLOR_CURSOR, True, [(p[0], p[1]-i) for p in points])
            pygame.draw.aalines(self.screen, self.COLOR_CURSOR, True, [(p[0]-i, p[1]) for p in points])


    def _draw_iso_tile(self, pos, top_color, side_color):
        x, y = pos
        top_points = [
            (x, y - self.tile_h / 2),
            (x + self.tile_w / 2, y),
            (x, y + self.tile_h / 2),
            (x - self.tile_w / 2, y),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, top_points, top_color)
        
        side_height = 10
        side_points = [
            (x - self.tile_w / 2, y),
            (x + self.tile_w / 2, y),
            (x + self.tile_w / 2, y + side_height),
            (x, y + self.tile_h/2 + side_height),
            (x - self.tile_w / 2, y + side_height),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, side_points, side_color)
        pygame.gfxdraw.aapolygon(self.screen, top_points, self.COLOR_GRID_LINE)

    def _render_harvested_produce(self):
        start_x, start_y = 20, 350
        for i, crop_type in enumerate(self.CROP_TYPES):
            count = self.harvested_produce[crop_type]
            if count > 0:
                crop_info = self.CROPS[crop_type]
                pos_x = start_x + i * 70
                pygame.gfxdraw.filled_circle(self.screen, pos_x, start_y, 15, crop_info["ripe_color"])
                pygame.gfxdraw.aacircle(self.screen, pos_x, start_y, 15, crop_info["ripe_color"])
                
                count_text = self.font_m.render(f"x{count}", True, self.COLOR_TEXT)
                self.screen.blit(count_text, (pos_x + 20, start_y - 10))

    def _render_ui(self):
        # --- Income ---
        income_text = self.font_m.render(f"Income: ${self.total_income} / ${self.WIN_INCOME}", True, self.COLOR_TEXT)
        self.screen.blit(income_text, (10, 10))

        # --- Time ---
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_m.render(f"Time: {int(time_left)}s", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)
        
        # --- Stage ---
        stage_text = self.font_m.render(f"Stage: {self.stage}/3", True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(topright=(self.WIDTH - 10, 35))
        self.screen.blit(stage_text, stage_rect)

        # --- Selected Crop Panel ---
        panel_rect = pygame.Rect(self.WIDTH // 2 - 150, self.HEIGHT - 50, 300, 45)
        pygame.draw.rect(self.screen, (40, 50, 60), panel_rect, border_radius=5)
        
        selected_crop_type = self.available_crops[self.selected_crop_index]
        crop_info = self.CROPS[selected_crop_type]
        
        crop_name_text = self.font_m.render(f"Planting: {crop_info['name']}", True, self.COLOR_TEXT)
        self.screen.blit(crop_name_text, (panel_rect.x + 10, panel_rect.y + 12))
        
        pygame.gfxdraw.filled_circle(self.screen, panel_rect.right - 30, panel_rect.centery, 12, crop_info["color"])
        pygame.gfxdraw.aacircle(self.screen, panel_rect.right - 30, panel_rect.centery, 12, crop_info["color"])
        
    def _create_floating_text(self, text, x, y, duration, font, color):
        self.floating_texts.append(FloatingText(text, x, y, duration, font, color))
        
    def _render_floating_texts(self):
        for t in self.floating_texts:
            t.draw(self.screen)

    def _create_particles(self, x, y, count, color):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

# Helper classes for visual effects
class FloatingText:
    def __init__(self, text, x, y, duration, font, color):
        self.x, self.y = x, y
        self.duration = duration
        self.initial_duration = duration
        self.text_surface = font.render(text, True, color)
        self.alpha = 255

    def update(self):
        self.duration -= 1
        self.y -= 0.5
        self.alpha = max(0, 255 * (self.duration / self.initial_duration))
        return self.duration > 0

    def draw(self, screen):
        self.text_surface.set_alpha(self.alpha)
        text_rect = self.text_surface.get_rect(center=(self.x, self.y))
        screen.blit(self.text_surface, text_rect)

class Particle:
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        self.color = color
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-2.5, -0.5)
        self.lifespan = random.randint(15, 30)
        self.radius = random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1 # Gravity
        self.lifespan -= 1
        self.radius -= 0.1
        return self.lifespan > 0 and self.radius > 0

    def draw(self, screen):
        if self.radius > 1:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.radius))