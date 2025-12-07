import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:50:40.194339
# Source Brief: brief_00058.md
# Brief Index: 58
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple

# Helper classes for game entities
Ingredient = namedtuple("Ingredient", ["name", "color", "glow_color", "base_potency", "synergy", "clash"])
Particle = namedtuple("Particle", ["pos", "vel", "color", "radius", "lifespan"])
FloatingText = namedtuple("FloatingText", ["text", "pos", "color", "age", "lifespan"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Craft magical artifacts by placing enchanted ingredients on a grid. "
        "Create powerful synergies and avoid clashes to reach the target potency before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor between the grid and the ingredient list. "
        "Press space to place an ingredient. Hold shift to speed up time."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Game world
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_DIMS = (6, 5)
    CELL_SIZE = 60
    GRID_TOP_LEFT = (40, 50)
    MAX_STEPS = 2000
    FPS = 30
    
    # UI Panel
    UI_PANEL_X = GRID_TOP_LEFT[0] + GRID_DIMS[0] * CELL_SIZE + 20
    UI_PANEL_WIDTH = SCREEN_WIDTH - UI_PANEL_X - 20

    # Colors
    COLOR_BG = (15, 10, 30)
    COLOR_GRID = (50, 40, 80)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_INVALID = (255, 0, 0)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TEXT_SHADOW = (10, 5, 20)
    COLOR_POTENCY_BAR = (50, 200, 255)
    COLOR_POTENCY_BAR_BG = (30, 30, 60)
    COLOR_TIMER_BAR = (255, 180, 50)
    
    # --- INITIALIZATION ---
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
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        # Persistent state (survives resets)
        self._initialize_ingredients()
        self.unlocked_ingredient_indices = [0, 1, 2]
        self.target_potency = 100
        self.time_limit = 90 # seconds
        self.successful_crafts = 0

        # Episode state (reset each time)
        self.grid = None
        self.grid_potency = None
        self.cursor_pos = None
        self.selected_ingredient_idx = None
        self.timer = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = []
        self.floating_texts = []
        
        self.reset()

    def _initialize_ingredients(self):
        self.INGREDIENTS = [
            Ingredient("Aura Quartz", (180, 180, 255), (100, 100, 220), 10, "Moonpetal", None),
            Ingredient("Moonpetal", (250, 250, 240), (200, 200, 180), 15, "Aura Quartz", "Fire Ruby"),
            Ingredient("Sunstone", (255, 200, 80), (220, 150, 50), 15, "Fire Ruby", None),
            Ingredient("Fire Ruby", (255, 80, 80), (220, 50, 50), 20, "Sunstone", "Moonpetal"),
            Ingredient("Shadow Crystal", (150, 80, 255), (100, 50, 200), -5, None, "Sunstone"),
        ]

    # --- GYM API ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = [[None for _ in range(self.GRID_DIMS[1])] for _ in range(self.GRID_DIMS[0])]
        self.grid_potency = np.zeros(self.GRID_DIMS)
        self.cursor_pos = [self.GRID_DIMS[0] // 2, self.GRID_DIMS[1] // 2]
        self.selected_ingredient_idx = 0
        self.timer = self.time_limit
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles.clear()
        self.floating_texts.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self._update_time(shift_held)
        reward -= 0.05 # Small penalty for time passing

        self._handle_input(movement, space_held)
        
        potency_change, reaction_reward = self._update_potency()
        self.score += potency_change
        reward += reaction_reward
        reward += potency_change * 0.1 # Reward for potency gain
        
        self._update_effects()

        terminated = self._check_termination()
        truncated = False
        if terminated:
            if self.score >= self.target_potency:
                reward += 100
                # // SFX: Win
                self._level_up()
            else:
                reward += -10
                # // SFX: Lose

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True # Let's use terminated for both win/loss and step limit

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- GAME LOGIC ---
    def _update_time(self, shift_held):
        time_multiplier = 2.0 if shift_held else 1.0
        self.timer -= (1 / self.FPS) * time_multiplier

    def _handle_input(self, movement, space_held):
        # Movement
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            # If cursor is on the selection panel
            if self.cursor_pos[0] == -1:
                if dx == 1: # Move right to grid
                    self.cursor_pos[0] = 0
                else: # Move up/down in list
                    self.selected_ingredient_idx = (self.selected_ingredient_idx - dy + len(self.unlocked_ingredient_indices)) % len(self.unlocked_ingredient_indices)
            # If cursor is on the grid
            else:
                new_x, new_y = self.cursor_pos[0] + dx, self.cursor_pos[1] + dy
                if new_x < 0: # Move left to selection panel
                    self.cursor_pos[0] = -1
                else:
                    self.cursor_pos[0] = max(0, min(self.GRID_DIMS[0] - 1, new_x))
                    self.cursor_pos[1] = max(0, min(self.GRID_DIMS[1] - 1, new_y))
        
        # Placement
        if space_held and self.cursor_pos[0] != -1:
            x, y = self.cursor_pos
            if self.grid[x][y] is None:
                self._place_ingredient(x, y)

    def _place_ingredient(self, x, y):
        # // SFX: place_ingredient
        ing_idx = self.unlocked_ingredient_indices[self.selected_ingredient_idx]
        self.grid[x][y] = ing_idx
        ingredient = self.INGREDIENTS[ing_idx]
        
        # Spawn placement particles
        for _ in range(20):
            self._spawn_particle(self._grid_to_pixel(x, y), ingredient.color, 1.5)

        # Trigger one-time reactions with neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.GRID_DIMS[0] and 0 <= ny < self.GRID_DIMS[1] and self.grid[nx][ny] is not None:
                neighbor_idx = self.grid[nx][ny]
                neighbor = self.INGREDIENTS[neighbor_idx]
                
                # Synergy
                if ingredient.synergy == neighbor.name:
                    self.grid_potency[x, y] += 25
                    self.grid_potency[nx, ny] += 25
                    self._create_floating_text("+25", self._grid_to_pixel(x, y), (0, 255, 0))
                    self._create_floating_text("+25", self._grid_to_pixel(nx, ny), (0, 255, 0))
                
                # Clash
                if ingredient.clash == neighbor.name:
                    self.grid_potency[x, y] -= 15
                    self.grid_potency[nx, ny] -= 15
                    self._create_floating_text("-15", self._grid_to_pixel(x, y), (255, 0, 0))
                    self._create_floating_text("-15", self._grid_to_pixel(nx, ny), (255, 0, 0))

    def _update_potency(self):
        previous_potency = np.sum(self.grid_potency)
        current_potency_grid = np.zeros(self.GRID_DIMS)
        reaction_reward = 0

        for x in range(self.GRID_DIMS[0]):
            for y in range(self.GRID_DIMS[1]):
                if self.grid[x][y] is not None:
                    ing_idx = self.grid[x][y]
                    ingredient = self.INGREDIENTS[ing_idx]
                    
                    # Base potency + triggered potency
                    current_potency_grid[x, y] += ingredient.base_potency + self.grid_potency[x, y]
                    
                    # Continuous effects (e.g., Shadow Crystal drain)
                    if ingredient.name == "Shadow Crystal":
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.GRID_DIMS[0] and 0 <= ny < self.GRID_DIMS[1] and self.grid[nx][ny] is not None:
                                current_potency_grid[nx, ny] -= 0.1 # Drain per step
                                reaction_reward -= 0.01

        total_potency = np.sum(current_potency_grid)
        potency_change = total_potency - self.score
        return potency_change, reaction_reward

    def _update_effects(self):
        # Update particles
        new_particles = []
        for p in self.particles:
            new_pos = (p.pos[0] + p.vel[0], p.pos[1] + p.vel[1])
            new_lifespan = p.lifespan - 1
            new_radius = p.radius * 0.95
            if new_lifespan > 0 and new_radius > 0.5:
                new_particles.append(p._replace(pos=new_pos, lifespan=new_lifespan, radius=new_radius))
        self.particles = new_particles

        # Update floating texts
        new_texts = []
        for t in self.floating_texts:
            new_age = t.age + 1
            if new_age < t.lifespan:
                new_pos = (t.pos[0], t.pos[1] - 0.5)
                new_texts.append(t._replace(age=new_age, pos=new_pos))
        self.floating_texts = new_texts

    def _check_termination(self):
        if self.timer <= 0 or self.score >= self.target_potency:
            self.game_over = True
            return True
        return False

    def _level_up(self):
        self.successful_crafts += 1
        self.target_potency = int(self.target_potency * 1.05)
        if self.successful_crafts % 5 == 0:
            self.time_limit = max(30, self.time_limit - 2)
        if self.successful_crafts % 10 == 0 and len(self.unlocked_ingredient_indices) < len(self.INGREDIENTS):
            self.unlocked_ingredient_indices.append(len(self.unlocked_ingredient_indices))
            # // SFX: unlock
            
    # --- RENDERING ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_grid()
        self._render_ingredients()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        self._render_floating_texts()

        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_effects(self):
        for _ in range(3): # Add a few random sparks
            if random.random() < 0.1:
                pos = (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT))
                self._spawn_particle(pos, (255, 255, 255, 50), 0.5, lifespan=15)

    def _render_grid(self):
        gx, gy = self.GRID_TOP_LEFT
        gw, gh = self.GRID_DIMS[0] * self.CELL_SIZE, self.GRID_DIMS[1] * self.CELL_SIZE
        for i in range(self.GRID_DIMS[0] + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (gx + i * self.CELL_SIZE, gy), (gx + i * self.CELL_SIZE, gy + gh))
        for i in range(self.GRID_DIMS[1] + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (gx, gy + i * self.CELL_SIZE), (gx + gw, gy + i * self.CELL_SIZE))

    def _render_ingredients(self):
        for x in range(self.GRID_DIMS[0]):
            for y in range(self.GRID_DIMS[1]):
                if self.grid[x][y] is not None:
                    ing_idx = self.grid[x][y]
                    ingredient = self.INGREDIENTS[ing_idx]
                    px, py = self._grid_to_pixel(x, y)
                    self._draw_glowing_circle(self.screen, (px, py), ingredient.color, ingredient.glow_color, 20)

    def _render_cursor(self):
        if self.cursor_pos[0] != -1: # Grid cursor
            x, y = self.cursor_pos
            is_valid = self.grid[x][y] is None
            color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID
            rect = pygame.Rect(
                self.GRID_TOP_LEFT[0] + x * self.CELL_SIZE,
                self.GRID_TOP_LEFT[1] + y * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, color, rect, 2, border_radius=5)
        # Selection panel cursor is handled in _render_ui

    def _render_ui(self):
        # Panel Background
        pygame.draw.rect(self.screen, (25, 20, 45), (self.UI_PANEL_X, 0, self.UI_PANEL_WIDTH, self.SCREEN_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_GRID, (self.UI_PANEL_X, 0), (self.UI_PANEL_X, self.SCREEN_HEIGHT), 2)
        
        y_offset = 20
        
        # Score
        self._draw_text("Potency", (self.UI_PANEL_X + 15, y_offset), self.font_medium)
        y_offset += 25
        self._draw_text(f"{int(self.score)} / {self.target_potency}", (self.UI_PANEL_X + 15, y_offset), self.font_large)
        y_offset += 40
        
        # Potency Bar
        bar_x = self.UI_PANEL_X + 15
        bar_w = self.UI_PANEL_WIDTH - 30
        progress = min(1, self.score / self.target_potency if self.target_potency > 0 else 0)
        pygame.draw.rect(self.screen, self.COLOR_POTENCY_BAR_BG, (bar_x, y_offset, bar_w, 15))
        pygame.draw.rect(self.screen, self.COLOR_POTENCY_BAR, (bar_x, y_offset, bar_w * progress, 15))
        y_offset += 35
        
        # Timer
        self._draw_text("Time", (self.UI_PANEL_X + 15, y_offset), self.font_medium)
        y_offset += 25
        timer_progress = max(0, self.timer / self.time_limit)
        pygame.draw.rect(self.screen, self.COLOR_POTENCY_BAR_BG, (bar_x, y_offset, bar_w, 15))
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (bar_x, y_offset, bar_w * timer_progress, 15))
        y_offset += 35
        
        # Ingredient Selection
        self._draw_text("Ingredients", (self.UI_PANEL_X + 15, y_offset), self.font_medium)
        y_offset += 30
        
        for i, ing_game_idx in enumerate(self.unlocked_ingredient_indices):
            ingredient = self.INGREDIENTS[ing_game_idx]
            item_y = y_offset + i * 40
            
            # Highlight if selected
            if self.cursor_pos[0] == -1 and self.selected_ingredient_idx == i:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (self.UI_PANEL_X + 5, item_y - 5, self.UI_PANEL_WIDTH - 10, 35), 2, border_radius=5)
            
            self._draw_glowing_circle(self.screen, (self.UI_PANEL_X + 30, item_y + 10), ingredient.color, ingredient.glow_color, 12)
            self._draw_text(ingredient.name, (self.UI_PANEL_X + 55, item_y), self.font_small)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p.lifespan / 30))
            color = p.color + (alpha,) if len(p.color) == 3 else p.color
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), int(p.radius), color)

    def _render_floating_texts(self):
        for t in self.floating_texts:
            alpha = 255 * (1 - (t.age / t.lifespan))
            text_surf = self.font_medium.render(t.text, True, t.color)
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (t.pos[0] - text_surf.get_width()//2, t.pos[1] - text_surf.get_height()//2))

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        
        message = "Artifact Complete!" if self.score >= self.target_potency else "Time's Up!"
        color = (100, 255, 150) if self.score >= self.target_potency else (255, 100, 100)
        
        text_surf = self.font_large.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(text_surf, text_rect)
        
    # --- HELPERS ---
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "target": self.target_potency}

    def _grid_to_pixel(self, x, y):
        return (
            self.GRID_TOP_LEFT[0] + x * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.GRID_TOP_LEFT[1] + y * self.CELL_SIZE + self.CELL_SIZE // 2,
        )

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        shadow_surf = font.render(text, True, shadow_color)
        self.screen.blit(shadow_surf, (pos[0] + 1, pos[1] + 1))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)
        
    def _draw_glowing_circle(self, surface, pos, color, glow_color, radius):
        for i in range(4):
            alpha = 60 - i * 15
            pygame.gfxdraw.filled_circle(
                surface, int(pos[0]), int(pos[1]),
                int(radius + i * 2),
                (glow_color[0], glow_color[1], glow_color[2], alpha)
            )
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), int(radius), color)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius), color)
        
    def _spawn_particle(self, pos, color, speed_mult=1.0, lifespan=30):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 3) * speed_mult
        vel = (math.cos(angle) * speed, math.sin(angle) * speed)
        radius = random.uniform(2, 5)
        self.particles.append(Particle(pos, vel, color, radius, lifespan))
        
    def _create_floating_text(self, text, pos, color, lifespan=45):
        self.floating_texts.append(FloatingText(text, pos, color, 0, lifespan))

if __name__ == '__main__':
    # The original __main__ block has been removed as it is not part of the
    # required GameEnv class definition and relies on a non-headless display.
    # A simple validation check is included instead.
    
    print("Validating GameEnv...")
    env = GameEnv()
    
    # Test reset
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (400, 640, 3)
    assert obs.dtype == np.uint8
    assert isinstance(info, dict)
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (400, 640, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    print("GameEnv validation successful.")