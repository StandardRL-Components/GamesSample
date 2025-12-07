
# Generated: 2025-08-27T23:03:16.653428
# Source Brief: brief_03330.md
# Brief Index: 3330

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select an "
        "ingredient. Move to an adjacent ingredient and press Space again to "
        "combine. Press Shift to deselect."
    )

    game_description = (
        "A puzzle game where you combine ingredients on a grid to brew potions. "
        "Discover recipes and manage your limited resources to create all 5 target "
        "potions before you run out of supplies."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Visuals & Layout ---
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.CELL_SIZE = self.HEIGHT // self.GRID_ROWS
        self.GRID_WIDTH = self.CELL_SIZE * self.GRID_COLS
        self.UI_WIDTH = self.WIDTH - self.GRID_WIDTH
        
        self.COLOR_BG = (21, 30, 39)
        self.COLOR_GRID = (41, 50, 60)
        self.COLOR_UI_BG = (31, 40, 50)
        self.COLOR_TEXT = (238, 238, 238)
        self.COLOR_TEXT_DIM = (150, 150, 150)
        self.COLOR_CURSOR = (255, 204, 0)
        self.COLOR_SELECTED = (0, 173, 181, 100)
        self.COLOR_SUCCESS = (152, 251, 152)
        self.COLOR_FAIL = (255, 102, 102)
        self.COLOR_WIN = (173, 255, 47)
        self.COLOR_LOSS = (220, 20, 60)

        self.font_l = pygame.font.Font(None, 36)
        self.font_m = pygame.font.Font(None, 24)
        self.font_s = pygame.font.Font(None, 18)
        self.font_xs = pygame.font.Font(None, 16)
        
        # --- Game Data ---
        self._define_game_data()

        # --- State Variables (initialized in reset) ---
        self.np_random = None
        self.grid = None
        self.ingredient_counts = None
        self.potions_brewed = None
        self.cursor_pos = None
        self.selected_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.space_was_held = False
        self.feedback_message = ""
        self.feedback_timer = 0
        self.particles = []

        self.reset()
        self.validate_implementation()

    def _define_game_data(self):
        self.INGREDIENTS = [
            {"name": "Ruby Dust", "color": (255, 64, 64)},
            {"name": "Sapphire Dew", "color": (64, 128, 255)},
            {"name": "Emerald Leaf", "color": (64, 255, 128)},
            {"name": "Sunstone", "color": (255, 192, 64)},
            {"name": "Moonpetal", "color": (192, 192, 255)},
            {"name": "Shadow Root", "color": (128, 64, 128)},
            {"name": "Iron Fungi", "color": (160, 160, 160)},
            {"name": "Glow Moss", "color": (224, 255, 96)},
            {"name": "Sky Blossom", "color": (135, 206, 250)},
            {"name": "Ember Crystal", "color": (255, 140, 0)},
        ]
        
        self.RECIPES = [
            {"name": "Potion of Strength", "ingredients": {0, 6}}, # Ruby + Iron
            {"name": "Elixir of Wisdom", "ingredients": {1, 4}}, # Sapphire + Moonpetal
            {"name": "Draught of Agility", "ingredients": {2, 8}}, # Emerald + Sky
            {"name": "Tonic of Fortitude", "ingredients": {3, 9}}, # Sunstone + Ember
            {"name": "Philter of Night", "ingredients": {5, 7}}, # Shadow + Glow
        ]
        # Generate potion colors by averaging ingredient colors
        for recipe in self.RECIPES:
            c1 = self.INGREDIENTS[list(recipe["ingredients"])[0]]["color"]
            c2 = self.INGREDIENTS[list(recipe["ingredients"])[1]]["color"]
            recipe["color"] = tuple(int((x+y)/2) for x, y in zip(c1, c2))
        
        self.required_ingredients = set()
        for recipe in self.RECIPES:
            self.required_ingredients.update(recipe["ingredients"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.space_was_held = True # Prevent action on first frame
        self.feedback_message = ""
        self.feedback_timer = 0
        self.particles = []

        self.ingredient_counts = self.np_random.integers(15, 25, size=len(self.INGREDIENTS))
        self.grid = self.np_random.integers(0, len(self.INGREDIENTS), size=(self.GRID_COLS, self.GRID_ROWS))
        
        self.potions_brewed = [False] * len(self.RECIPES)
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_pos = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1
        
        space_pressed = space_held and not self.space_was_held
        self.space_was_held = space_held

        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        else:
            self.feedback_message = ""

        # --- Action Handling ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        if shift_held:
            if self.selected_pos:
                self.selected_pos = None
                self._set_feedback("Selection cleared.", 30)

        if space_pressed:
            if self.selected_pos is None:
                self.selected_pos = list(self.cursor_pos)
            else:
                if self._is_adjacent(self.selected_pos, self.cursor_pos):
                    reward += self._attempt_combination()
                else:
                    self.selected_pos = list(self.cursor_pos)

        self._update_particles()
        
        # --- Termination Check ---
        terminated = False
        if all(self.potions_brewed):
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100
            self.score += 100
            self._set_feedback("All potions brewed! YOU WIN!", 1000)
        elif self._is_any_required_ingredient_depleted():
            self.game_over = True
            terminated = True
            reward -= 100
            self.score -= 100
            self._set_feedback("Ingredients depleted! GAME OVER!", 1000)
        elif self.steps >= 1000:
            self.game_over = True
            terminated = True
            self._set_feedback("Time's up! GAME OVER!", 1000)
        
        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _attempt_combination(self):
        reward = 0
        pos1, pos2 = self.selected_pos, self.cursor_pos
        ing1_idx = self.grid[pos1[0], pos1[1]]
        ing2_idx = self.grid[pos2[0], pos2[1]]

        if self.ingredient_counts[ing1_idx] > 0: self.ingredient_counts[ing1_idx] -= 1
        if self.ingredient_counts[ing2_idx] > 0: self.ingredient_counts[ing2_idx] -= 1

        recipe_idx = self._check_recipe(ing1_idx, ing2_idx)
        
        if recipe_idx is not None:
            # Potion Sfx: Brew success
            reward += 1
            self._create_particles(pos1, pos2, self.RECIPES[recipe_idx]["color"])
            if not self.potions_brewed[recipe_idx]:
                self.potions_brewed[recipe_idx] = True
                reward += 5
                self._set_feedback(f"New Potion: {self.RECIPES[recipe_idx]['name']}!", 90, self.COLOR_SUCCESS)
            else:
                self._set_feedback("Potion Brewed!", 60, self.COLOR_SUCCESS)
        else:
            # Potion Sfx: Brew fail
            reward -= 0.1
            self._create_particles(pos1, pos2, self.COLOR_FAIL, 20, 1.5)
            self._set_feedback("Ingredients Wasted!", 60, self.COLOR_FAIL)

        self._refill_grid_cell(pos1)
        self._refill_grid_cell(pos2)
        self.selected_pos = None
        return reward

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _check_recipe(self, ing1_idx, ing2_idx):
        combo = {ing1_idx, ing2_idx}
        for i, recipe in enumerate(self.RECIPES):
            if recipe["ingredients"] == combo:
                return i
        return None

    def _is_any_required_ingredient_depleted(self):
        for ing_idx in self.required_ingredients:
            if self.ingredient_counts[ing_idx] <= 0:
                # Check if this ingredient is still needed for an unbrewed potion
                for i, recipe in enumerate(self.RECIPES):
                    if not self.potions_brewed[i] and ing_idx in recipe["ingredients"]:
                        return True
        return False

    def _refill_grid_cell(self, pos):
        self.grid[pos[0], pos[1]] = self.np_random.integers(0, len(self.INGREDIENTS))

    def _set_feedback(self, message, duration, color=None):
        self.feedback_message = message
        self.feedback_timer = duration
        self.feedback_color = color if color is not None else self.COLOR_TEXT

    def _create_particles(self, pos1, pos2, color, count=40, speed=2.5):
        # Potion Sfx: Fizzing
        mid_x = (pos1[0] + 0.5 + pos2[0] + 0.5) * self.CELL_SIZE / 2
        mid_y = (pos1[1] + 0.5 + pos2[1] + 0.5) * self.CELL_SIZE / 2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = self.np_random.uniform(0.5, 1.0) * speed
            vel_x = math.cos(angle) * vel
            vel_y = math.sin(angle) * vel
            lifespan = self.np_random.integers(20, 40)
            self.particles.append([mid_x, mid_y, vel_x, vel_y, lifespan, color])

    def _update_particles(self):
        self.particles = [
            [p[0] + p[2], p[1] + p[3], p[2]*0.95, p[3]*0.95, p[4] - 1, p[5]]
            for p in self.particles if p[4] > 0
        ]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, r * self.CELL_SIZE), (self.GRID_WIDTH, r * self.CELL_SIZE))
        for c in range(self.GRID_COLS + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (c * self.CELL_SIZE, 0), (c * self.CELL_SIZE, self.HEIGHT))

        # Draw ingredients
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                ing_idx = self.grid[c, r]
                color = self.INGREDIENTS[ing_idx]["color"]
                rect = pygame.Rect(c * self.CELL_SIZE + 2, r * self.CELL_SIZE + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
                pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Draw selection highlight
        if self.selected_pos:
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill(self.COLOR_SELECTED)
            self.screen.blit(s, (self.selected_pos[0] * self.CELL_SIZE, self.selected_pos[1] * self.CELL_SIZE))

        # Draw cursor
        cursor_rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p[4] / 40.0))
            color = (*p[5], alpha)
            size = max(1, int(p[4] / 8))
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), size, color)

    def _render_ui(self):
        ui_rect = pygame.Rect(self.GRID_WIDTH, 0, self.UI_WIDTH, self.HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_WIDTH, 0), (self.GRID_WIDTH, self.HEIGHT), 2)
        
        y_offset = 15

        # Title
        title_surf = self.font_m.render("TARGET POTIONS", True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (self.GRID_WIDTH + (self.UI_WIDTH - title_surf.get_width()) // 2, y_offset))
        y_offset += 30

        # Target Potions
        for i, recipe in enumerate(self.RECIPES):
            color = self.COLOR_TEXT if self.potions_brewed[i] else self.COLOR_TEXT_DIM
            text = f"✓ {recipe['name']}" if self.potions_brewed[i] else f"  {recipe['name']}"
            potion_surf = self.font_s.render(text, True, color)
            pygame.draw.rect(self.screen, recipe['color'], (self.GRID_WIDTH + 10, y_offset, 10, 10), border_radius=2)
            self.screen.blit(potion_surf, (self.GRID_WIDTH + 25, y_offset - 3))
            y_offset += 20
        
        y_offset += 15
        
        # Inventory Title
        inv_surf = self.font_m.render("INGREDIENTS", True, self.COLOR_TEXT)
        self.screen.blit(inv_surf, (self.GRID_WIDTH + (self.UI_WIDTH - inv_surf.get_width()) // 2, y_offset))
        y_offset += 25

        # Ingredient Inventory
        for i, ing in enumerate(self.INGREDIENTS):
            count = self.ingredient_counts[i]
            is_req = i in self.required_ingredients
            color = self.COLOR_FAIL if is_req and count == 0 else self.COLOR_TEXT
            
            count_surf = self.font_xs.render(f"{count:02d}", True, color)
            self.screen.blit(count_surf, (self.GRID_WIDTH + 15, y_offset))
            
            pygame.draw.rect(self.screen, ing['color'], (self.GRID_WIDTH + 40, y_offset+2, 8, 8), border_radius=2)
            
            name_surf = self.font_xs.render(ing['name'], True, self.COLOR_TEXT_DIM)
            self.screen.blit(name_surf, (self.GRID_WIDTH + 55, y_offset))
            
            y_offset += 16
            if i == 4: # Split into two columns
                y_offset -= 16 * 5
                self.GRID_WIDTH += self.UI_WIDTH / 2
        self.GRID_WIDTH = self.CELL_SIZE * self.GRID_COLS # Reset for next frame
        y_offset += 16 * 5 + 10

        # Score & Steps
        score_surf = self.font_s.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.GRID_WIDTH + 15, self.HEIGHT - 55))
        steps_surf = self.font_s.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (self.GRID_WIDTH + 15, self.HEIGHT - 35))

        # Feedback Message
        if self.feedback_timer > 0:
            alpha = min(255, self.feedback_timer * 10)
            feedback_surf = self.font_m.render(self.feedback_message, True, self.feedback_color)
            feedback_surf.set_alpha(alpha)
            pos_x = (self.GRID_WIDTH - feedback_surf.get_width()) // 2
            pos_y = self.HEIGHT - 40
            self.screen.blit(feedback_surf, (pos_x, pos_y))

        # Game Over Overlay
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_WIN if self.win else self.COLOR_LOSS
            
            end_surf = self.font_l.render(msg, True, color)
            self.screen.blit(end_surf, ((self.WIDTH - end_surf.get_width()) // 2, self.HEIGHT // 2 - 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "potions_brewed": sum(self.potions_brewed),
            "win": self.win,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Potion Brewer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True

    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        if keys[pygame.K_r]:
            obs, info = env.reset()
            done = False
            continue

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Draw to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if done:
            print(f"Game Over! Final Score: {info['score']}, Win: {info['win']}")
        
        clock.tick(30) # Limit frame rate for human play

    env.close()