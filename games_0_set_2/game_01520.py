
# Generated: 2025-08-27T17:24:15.709926
# Source Brief: brief_01520.md
# Brief Index: 1520

        
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
        "Use ←→ to select an ingredient, SPACE to pick it up. "
        "Use ↑↓←→ to position the ingredient on the grid, and SPACE again to place it. "
        "Create 5 perfect dishes to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A mystical cooking challenge! Combine enchanted ingredients on your isometric cooking station. "
        "Discover secret recipes, create perfect dishes, and manage your limited supplies to become a legendary chef."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.NUM_INGREDIENTS = 10
        self.INITIAL_INGREDIENT_COUNT = 8
        self.GRID_SIZE = 4
        self.DISH_CAPACITY = 4
        self.PERFECT_SCORE = 10
        self.WIN_CONDITION = 5

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_COUNTER = (50, 60, 80)
        self.COLOR_GRID_LINE = (70, 80, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 20)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_POSITIVE = (0, 255, 128)
        self.COLOR_NEGATIVE = (255, 80, 80)
        self.COLOR_NEUTRAL = (100, 150, 255)

        # Fonts
        self.font_s = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_m = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_xl = pygame.font.SysFont("monospace", 48, bold=True)

        # Isometric projection settings
        self.tile_width = 48
        self.tile_height = 24
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = 120
        
        # Pre-define ingredients
        self._define_ingredients()

        # Initialize state variables
        self.reset()
        
        # Run validation
        self.validate_implementation()
    
    def _define_ingredients(self):
        self.ingredients = [
            {"name": "Gloom Bloom", "color": (150, 100, 255)},
            {"name": "Sun Drop", "color": (255, 220, 50)},
            {"name": "Earth Root", "color": (140, 80, 50)},
            {"name": "Sky Leaf", "color": (100, 200, 255)},
            {"name": "Heart Berry", "color": (255, 50, 100)},
            {"name": "Glimmer Moss", "color": (50, 255, 150)},
            {"name": "Shadow Pearl", "color": (80, 80, 100)},
            {"name": "Spice Spark", "color": (255, 120, 0)},
            {"name": "Aqua Gem", "color": (0, 180, 180)},
            {"name": "Void Salt", "color": (200, 200, 220)},
        ]

    def _generate_interaction_matrix(self):
        rng = np.random.default_rng(self.seed_value)
        self.interaction_matrix = rng.integers(-3, 4, size=(self.NUM_INGREDIENTS, self.NUM_INGREDIENTS))
        
        # Ensure some strong positive and negative interactions exist for interesting gameplay
        for i in range(self.NUM_INGREDIENTS):
            # Each ingredient has a "best friend"
            friend = (i + rng.integers(1, self.NUM_INGREDIENTS)) % self.NUM_INGREDIENTS
            self.interaction_matrix[i][friend] = rng.integers(3, 5)
            # And an "enemy"
            enemy = (i + rng.integers(1, self.NUM_INGREDIENTS)) % self.NUM_INGREDIENTS
            if enemy != friend:
                self.interaction_matrix[i][enemy] = rng.integers(-5, -3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = seed
        else:
            # Fallback if seed is not provided, though Gym usually provides one.
            self.seed_value = random.randint(0, 1_000_000)
        
        self._generate_interaction_matrix()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_status = "" # "win", "lose", "time_up"

        self.ingredient_counts = [self.INITIAL_INGREDIENT_COUNT] * self.NUM_INGREDIENTS
        self.cooking_grid = [[-1 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.placed_in_dish_count = 0
        self.current_dish_score = 0
        self.perfect_dishes_served = 0

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_ingredient_idx = 0
        self.held_ingredient_idx = None

        self.prev_space_held = False
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle player input and actions
        if self.held_ingredient_idx is not None: # Placing mode
            dx = (1 if movement == 4 else -1 if movement == 3 else 0)
            dy = (1 if movement == 2 else -1 if movement == 1 else 0)
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_SIZE - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_SIZE - 1)

            if space_pressed:
                x, y = self.cursor_pos
                if self.cooking_grid[y][x] == -1:
                    # Place ingredient
                    self.cooking_grid[y][x] = self.held_ingredient_idx
                    self.ingredient_counts[self.held_ingredient_idx] -= 1
                    
                    placement_reward = self._calculate_placement_score(x, y)
                    reward += placement_reward
                    self.current_dish_score += placement_reward
                    
                    self.placed_in_dish_count += 1
                    self.held_ingredient_idx = None
                    # sfx: place_item.wav

        else: # Browsing mode
            dx = (1 if movement == 4 or movement == 2 else -1 if movement == 3 or movement == 1 else 0)
            if dx != 0:
                self.selected_ingredient_idx = (self.selected_ingredient_idx + dx + self.NUM_INGREDIENTS) % self.NUM_INGREDIENTS
            
            if space_pressed:
                if self.ingredient_counts[self.selected_ingredient_idx] > 0:
                    self.held_ingredient_idx = self.selected_ingredient_idx
                    # sfx: pickup_item.wav
                else:
                    # sfx: error.wav
                    pass

        # 2. Update game state
        if self.placed_in_dish_count >= self.DISH_CAPACITY:
            # Serve dish
            if self.current_dish_score == self.PERFECT_SCORE:
                self.perfect_dishes_served += 1
                reward += 10 # Event reward for perfect dish
                # sfx: perfect_dish.wav
                self._create_particles(self.SCREEN_WIDTH // 2, 50, 50, self.COLOR_GOLD)
            else:
                # sfx: serve_dish.wav
                pass
            
            self.score += self.current_dish_score
            self.current_dish_score = 0
            self.placed_in_dish_count = 0
            self.cooking_grid = [[-1 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        self._update_particles()
        self.steps += 1

        # 3. Check for termination
        if self.perfect_dishes_served >= self.WIN_CONDITION:
            terminated = True
            reward += 100
            self.game_over = True
            self.game_over_status = "WIN"
            # sfx: victory.wav
        elif any(c <= 0 for c in self.ingredient_counts):
            terminated = True
            reward -= 100
            self.game_over = True
            self.game_over_status = "SUPPLIES DEPLETED"
            # sfx: failure.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.game_over_status = "TIME UP"
            # sfx: failure.wav
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_placement_score(self, x, y):
        placed_idx = self.cooking_grid[y][x]
        score_change = 0
        
        # Check 4 neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                neighbor_idx = self.cooking_grid[ny][nx]
                if neighbor_idx != -1:
                    # Interaction is two-way
                    score_change += self.interaction_matrix[placed_idx][neighbor_idx]
                    score_change += self.interaction_matrix[neighbor_idx][placed_idx]

        screen_x, screen_y = self._iso_to_screen(x, y)
        if score_change > 0:
            self._create_particles(screen_x, screen_y, 15, self.COLOR_POSITIVE)
        elif score_change < 0:
            self._create_particles(screen_x, screen_y, 15, self.COLOR_NEGATIVE)
        else:
            self._create_particles(screen_x, screen_y, 10, self.COLOR_NEUTRAL, life=20)
            
        return score_change

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "perfect_dishes": self.perfect_dishes_served,
            "current_dish_score": self.current_dish_score,
        }

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * self.tile_width
        screen_y = self.origin_y + (x + y) * self.tile_height
        return int(screen_x), int(screen_y)

    def _render_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_game(self):
        # Render counter
        counter_rect = pygame.Rect(0, 280, self.SCREEN_WIDTH, 120)
        pygame.draw.rect(self.screen, self.COLOR_COUNTER, counter_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (0, 280), (self.SCREEN_WIDTH, 280), 2)

        # Render cooking grid
        for y in range(self.GRID_SIZE + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_SIZE, y)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start, end, 1)
        for x in range(self.GRID_SIZE + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start, end, 1)

        # Render placed ingredients
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                ing_idx = self.cooking_grid[y][x]
                if ing_idx != -1:
                    sx, sy = self._iso_to_screen(x, y)
                    color = self.ingredients[ing_idx]["color"]
                    pygame.draw.circle(self.screen, (0,0,0), (sx, sy), 16)
                    pygame.draw.circle(self.screen, color, (sx, sy), 14)
        
        # Render cursor and held ingredient
        if self.held_ingredient_idx is not None:
            cx, cy = self.cursor_pos
            sx, sy = self._iso_to_screen(cx, cy)
            # Cursor
            points = [
                self._iso_to_screen(cx, cy), self._iso_to_screen(cx+1, cy),
                self._iso_to_screen(cx+1, cy+1), self._iso_to_screen(cx, cy+1)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_SELECTOR, points, 2)
            # Held ingredient
            ing_color = self.ingredients[self.held_ingredient_idx]["color"]
            pygame.draw.circle(self.screen, (0,0,0), (sx, sy), 18)
            pygame.draw.circle(self.screen, ing_color, (sx, sy), 16)
            pygame.gfxdraw.aacircle(self.screen, sx, sy, 16, ing_color)

        # Render particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Render ingredient shelves
        shelf_y = 340
        for i in range(self.NUM_INGREDIENTS):
            ing = self.ingredients[i]
            x_pos = 40 + i * 58
            
            # Selection highlight
            if self.held_ingredient_idx is None and self.selected_ingredient_idx == i:
                pygame.draw.rect(self.screen, self.COLOR_SELECTOR, (x_pos-20, shelf_y-30, 40, 60), 2, border_radius=5)

            # Ingredient icon
            count = self.ingredient_counts[i]
            color = ing["color"] if count > 0 else (80, 80, 80)
            pygame.draw.circle(self.screen, (0,0,0), (x_pos, shelf_y - 10), 16)
            pygame.draw.circle(self.screen, color, (x_pos, shelf_y - 10), 14)
            
            # Count text
            count_str = str(count)
            text_surf = self.font_m.render(count_str, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(x_pos, shelf_y + 25))
            self.screen.blit(text_surf, text_rect)

        # Render dish score
        score_str = f"Dish Score: {self.current_dish_score}"
        self._render_text(score_str, self.font_l, self.COLOR_TEXT, (10, 80))

        # Render perfect dish counter
        self._render_text("Perfect Dishes:", self.font_m, self.COLOR_TEXT, (10, 10))
        for i in range(self.WIN_CONDITION):
            color = self.COLOR_GOLD if i < self.perfect_dishes_served else self.COLOR_COUNTER
            star_points = self._get_star_points((180 + i * 30, 22), 5, 12, 6)
            pygame.draw.polygon(self.screen, color, star_points)
        
        # Render total score
        total_score_str = f"Total: {self.score}"
        text_surf = self.font_m.render(total_score_str, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)

        # Render game over screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            self._render_text(self.game_over_status, self.font_xl, self.COLOR_GOLD, (self.SCREEN_WIDTH // 2 - self.font_xl.size(self.game_over_status)[0] // 2, 150))
            final_score_str = f"Final Score: {self.score}"
            self._render_text(final_score_str, self.font_l, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2 - self.font_l.size(final_score_str)[0] // 2, 210))

    def _get_star_points(self, center, num_points, outer_radius, inner_radius):
        points = []
        angle_step = 360 / (num_points * 2)
        for i in range(num_points * 2):
            angle = math.radians(i * angle_step - 90)
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        return points

    def _create_particles(self, x, y, count, color, life=40):
        for _ in range(count):
            self.particles.append(Particle(x, y, color, life, self.np_random))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

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
        obs, info = self.reset(seed=123)
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

class Particle:
    def __init__(self, x, y, color, life, rng):
        self.x = x
        self.y = y
        self.color = color
        self.max_life = life
        self.life = self.max_life
        self.vx = rng.uniform(-1.5, 1.5)
        self.vy = rng.uniform(-2.0, -0.5)
        self.g = 0.05

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.g
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            color = (*self.color, alpha)
            temp_surf = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (3, 3), 3)
            surface.blit(temp_surf, (int(self.x) - 3, int(self.y) - 3), special_flags=pygame.BLEND_RGBA_ADD)