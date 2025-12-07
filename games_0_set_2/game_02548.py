
# Generated: 2025-08-27T20:44:05.891448
# Source Brief: brief_02548.md
# Brief Index: 2548

        
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
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle ingredients and Space to place them."
    )

    game_description = (
        "A magical puzzle game. Combine ingredients on the grid, discover powerful reactions, and cook five perfect dishes to master the cursed kitchen."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.TILE_WIDTH_HALF = 28
        self.TILE_HEIGHT_HALF = 14
        self.GRID_OFFSET_X = self.WIDTH // 2
        self.GRID_OFFSET_Y = 100
        self.MAX_STEPS = 2000
        self.WIN_DISHES = 5
        self.INITIAL_INGREDIENTS = 40

        # Colors
        self.COLOR_BG = (20, 15, 30)
        self.COLOR_GRID = (50, 40, 70)
        self.COLOR_GRID_LINE = (80, 70, 110)
        self.COLOR_CURSOR = (255, 255, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.INGREDIENT_COLORS = [
            (0, 0, 0), # 0: Empty
            (50, 255, 150), # 1: Green
            (255, 80, 200), # 2: Magenta
            (80, 180, 255)  # 3: Cyan
        ]
        self.REACTION_COLORS = {
            "good": (100, 255, 100),
            "bad": (255, 100, 100),
            "neutral": (150, 150, 255)
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        # Game state variables are initialized in reset()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.dishes_completed = 0
        
        self.ingredient_counts = [self.INITIAL_INGREDIENTS] * 3
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_ingredient = 1 # 1, 2, or 3
        
        self.particles = []
        self.last_action = np.array([0, 0, 0])

        self.dish_quality_meter = 0.0
        self.dish_quality_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        _, last_space_held, last_shift_held = self.last_action[0], self.last_action[1] == 1, self.last_action[2] == 1
        
        space_pressed = space_held and not last_space_held
        shift_pressed = shift_held and not last_shift_held

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Handle cursor movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
            
        # Handle ingredient cycling (Shift)
        if shift_pressed:
            self.selected_ingredient = (self.selected_ingredient % 3) + 1
            # sfx: cycle_ingredient.wav
            
        # Handle ingredient placement (Space)
        if space_pressed:
            x, y = self.cursor_pos
            if self.grid[y][x] == 0: # If cell is empty
                if self.ingredient_counts[self.selected_ingredient - 1] > 0:
                    self.ingredient_counts[self.selected_ingredient - 1] -= 1
                    self.grid[y][x] = self.selected_ingredient
                    reaction_reward = self._calculate_reactions(x, y)
                    reward += reaction_reward
                    self.score += reaction_reward
                    # sfx: place_ingredient.wav
                    
                    if np.count_nonzero(self.grid) == self.GRID_SIZE * self.GRID_SIZE:
                        dish_reward, quality = self._cook_dish()
                        reward += dish_reward
                        self.score += dish_reward
                        self.dish_quality_meter = quality
                        self.dish_quality_timer = 90 # 3 seconds at 30fps
                else:
                    # sfx: error.wav
                    pass

        self.steps += 1
        self._update_particles()
        if self.dish_quality_timer > 0:
            self.dish_quality_timer -= 1

        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            if self.dishes_completed >= self.WIN_DISHES:
                reward += 50 # Win bonus
                self.score += 50
                # sfx: win_game.wav
            else:
                # sfx: lose_game.wav
                pass
            self.game_over = True
            
        self.last_action = action
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_reactions(self, x, y):
        reaction_reward = 0
        placed_type = self.grid[y][x]
        
        # Good combos: (1,2), (2,3), (3,1) and vice-versa
        good_combos = {(1, 2), (2, 1), (2, 3), (3, 2), (3, 1), (1, 3)}
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                neighbor_type = self.grid[ny][nx]
                if neighbor_type != 0:
                    combo = (placed_type, neighbor_type)
                    px, py = self._iso_to_screen(x, y)
                    npx, npy = self._iso_to_screen(nx, ny)
                    mid_x, mid_y = (px + npx) // 2, (py + npy) // 2
                    
                    if combo in good_combos:
                        reaction_reward += 1
                        self._create_particles(mid_x, mid_y, "good")
                    elif placed_type == neighbor_type: # Bad combo
                        reaction_reward -= 1
                        self._create_particles(mid_x, mid_y, "bad")
                    else: # Neutral combo
                        self._create_particles(mid_x, mid_y, "neutral")
        return reaction_reward
    
    def _cook_dish(self):
        # sfx: cook_dish.wav
        self.dishes_completed += 1
        
        total_score = 0
        max_possible_score = 0
        good_combos = {(1, 2), (2, 1), (2, 3), (3, 2), (3, 1), (1, 3)}

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                type1 = self.grid[y][x]
                # Check right and down neighbors to avoid double counting
                for dx, dy in [(1, 0), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                        type2 = self.grid[ny][nx]
                        combo = (type1, type2)
                        max_possible_score += 1
                        if combo in good_combos:
                            total_score += 1
                        elif type1 == type2:
                            total_score -= 1
        
        # Normalize quality to [0, 1]
        quality = 0.0
        if max_possible_score > 0:
            quality = (total_score + max_possible_score) / (2 * max_possible_score)
        
        # Reset grid for next dish
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        # Visual effect for cooking
        for _ in range(100):
            self.particles.append({
                'pos': [random.randint(200, self.WIDTH - 200), random.randint(100, self.HEIGHT-100)],
                'vel': [random.uniform(-2, 2), random.uniform(-5, -2)],
                'life': random.randint(30, 60),
                'color': (255, 220, 100),
                'radius': random.uniform(2, 5)
            })

        return 10 * quality, quality

    def _check_termination(self):
        if self.dishes_completed >= self.WIN_DISHES:
            return True
        if any(count <= 0 for count in self.ingredient_counts):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False
        
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
            "dishes_completed": self.dishes_completed,
            "ingredient_counts": list(self.ingredient_counts),
            "cursor_pos": list(self.cursor_pos),
        }

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.GRID_OFFSET_X + (grid_x - grid_y) * self.TILE_WIDTH_HALF
        screen_y = self.GRID_OFFSET_Y + (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Render grid tiles
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                cx, cy = self._iso_to_screen(x, y)
                points = [
                    (cx, cy - self.TILE_HEIGHT_HALF),
                    (cx + self.TILE_WIDTH_HALF, cy),
                    (cx, cy + self.TILE_HEIGHT_HALF),
                    (cx - self.TILE_WIDTH_HALF, cy)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GRID)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID_LINE)

        # Render ingredients and cursor
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                # Render cursor
                if x == self.cursor_pos[0] and y == self.cursor_pos[1]:
                    cx, cy = self._iso_to_screen(x, y)
                    points = [
                        (cx, cy - self.TILE_HEIGHT_HALF),
                        (cx + self.TILE_WIDTH_HALF, cy),
                        (cx, cy + self.TILE_HEIGHT_HALF),
                        (cx - self.TILE_WIDTH_HALF, cy)
                    ]
                    pulse = (math.sin(self.steps * 0.2) + 1) / 2
                    color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in self.COLOR_CURSOR)
                    pygame.draw.polygon(self.screen, color, points, 3)

                # Render ingredients
                ingredient_type = self.grid[y][x]
                if ingredient_type != 0:
                    cx, cy = self._iso_to_screen(x, y)
                    color = self.INGREDIENT_COLORS[ingredient_type]
                    bob = math.sin(self.steps * 0.1 + x + y) * 2
                    radius = int(self.TILE_HEIGHT_HALF * 0.8)
                    pygame.gfxdraw.filled_circle(self.screen, cx, int(cy - bob), radius, color)
                    pygame.gfxdraw.aacircle(self.screen, cx, int(cy - bob), radius, color)

        self._render_particles()

    def _render_ui(self):
        # Score and Dishes
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        dishes_text = self.font_small.render(f"DISHES: {self.dishes_completed} / {self.WIN_DISHES}", True, self.COLOR_TEXT)
        self.screen.blit(dishes_text, (10, 35))
        
        # Ingredient Dispensers (UI)
        dispenser_y = 350
        for i in range(3):
            dispenser_x = self.WIDTH // 4 * (i + 1)
            color = self.INGREDIENT_COLORS[i+1]
            count = self.ingredient_counts[i]
            
            # Highlight selected ingredient
            if self.selected_ingredient == i + 1:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (dispenser_x - 25, dispenser_y - 25, 50, 50), 3, 5)

            pygame.gfxdraw.filled_circle(self.screen, dispenser_x, dispenser_y, 15, color)
            pygame.gfxdraw.aacircle(self.screen, dispenser_x, dispenser_y, 15, color)
            
            count_color = self.COLOR_TEXT if count > 5 else self.REACTION_COLORS["bad"]
            count_text = self.font_small.render(str(count), True, count_color)
            self.screen.blit(count_text, (dispenser_x - count_text.get_width()//2, dispenser_y + 20))

        # Dish Quality Meter
        if self.dish_quality_timer > 0:
            alpha = min(255, self.dish_quality_timer * 4)
            quality_text = self.font_large.render("DISH COMPLETE!", True, self.COLOR_TEXT)
            quality_text.set_alpha(alpha)
            self.screen.blit(quality_text, (self.WIDTH//2 - quality_text.get_width()//2, 150))
            
            bar_width = 200
            bar_height = 20
            bar_x = self.WIDTH//2 - bar_width//2
            bar_y = 200
            
            fill_width = int(bar_width * self.dish_quality_meter)
            quality_color = pygame.Color(0,0,0)
            quality_color.lerp(self.REACTION_COLORS["bad"], self.dish_quality_meter)
            quality_color.lerp(self.REACTION_COLORS["good"], self.dish_quality_meter)


            bg_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
            fill_rect = pygame.Rect(bar_x, bar_y, fill_width, bar_height)
            
            pygame.draw.rect(self.screen, self.COLOR_GRID, bg_rect, border_radius=5)
            pygame.draw.rect(self.screen, quality_color, fill_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, bg_rect, 2, border_radius=5)


        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.dishes_completed >= self.WIN_DISHES:
                end_text = self.font_large.render("YOU WIN!", True, self.REACTION_COLORS["good"])
            else:
                end_text = self.font_large.render("GAME OVER", True, self.REACTION_COLORS["bad"])
            
            self.screen.blit(end_text, (self.WIDTH//2 - end_text.get_width()//2, self.HEIGHT//2 - end_text.get_height()//2))

    def _create_particles(self, x, y, p_type):
        color = self.REACTION_COLORS[p_type]
        for _ in range(15):
            self.particles.append({
                'pos': [x, y],
                'vel': [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)],
                'life': random.randint(15, 30),
                'color': color,
                'radius': random.uniform(1, 3)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.05
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color_with_alpha = p['color'] + (alpha,)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color_with_alpha, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0]-radius, pos[1]-radius))

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    while not done:
        # Map keyboard keys to actions
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the game to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # Create a display if one doesn't exist
        try:
            screen = pygame.display.get_surface()
            if screen is None:
                screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
                pygame.display.set_caption("Cursed Kitchen")
        except pygame.error:
            screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            pygame.display.set_caption("Cursed Kitchen")

        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()