
# Generated: 2025-08-28T04:21:56.136138
# Source Brief: brief_02277.md
# Brief Index: 2277

        
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
        "Controls: ↑/↓ to select an ingredient. ←/→ to select a destination. Space to move the ingredient."
    )

    # Must be a user-facing description of the game:
    game_description = (
        "A spooky culinary puzzle! Place ingredients in their correct spots to prepare dishes, but avoid the cursed zones that haunt the kitchen counter."
    )

    # Should frames auto-advance or wait for user input?
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
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- Colors ---
        self.COLOR_BG = (28, 24, 36)
        self.COLOR_COUNTER = (60, 50, 70)
        self.COLOR_COUNTER_EDGE = (45, 38, 53)
        self.COLOR_GRID = (75, 65, 85)
        self.COLOR_SAFE = (80, 200, 120)
        self.COLOR_CURSE = (180, 40, 90)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_SELECT_INGREDIENT = (255, 255, 100)
        self.COLOR_SELECT_TARGET = (100, 255, 255)

        self.INGREDIENT_COLORS = [
            (220, 50, 50),   # Tomato (Red)
            (255, 200, 60),  # Cheese (Yellow)
            (80, 180, 80),   # Lettuce (Green)
            (150, 90, 60),   # Mushroom (Brown)
            (230, 150, 200)  # Onion (Pink/Purple)
        ]

        # --- Isometric Grid ---
        self.grid_w, self.grid_h = 14, 10
        self.tile_w, self.tile_h = 60, 30
        self.grid_offset_x = self.screen_width / 2
        self.grid_offset_y = 120

        # --- Game Constants ---
        self.MAX_STEPS = 1000
        self.WIN_CONDITION = 5
        self.LOSE_CONDITION = 3
        self.CURSE_RADIUS = 1.5  # In grid units

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.dishes_prepared = 0
        self.curses_triggered = 0
        
        self.ingredients = []
        self.dishes = []
        self.curse_zones = []
        
        self.selected_ingredient_idx = 0
        self.selected_dish_idx = 0
        
        self.prev_space_held = False
        self.particles = []
        
        self.rng = None # Initialized in reset

        # Initialize state variables
        self.reset()
        
        # --- Validation ---
        self.validate_implementation()

    def _to_iso(self, grid_x, grid_y):
        screen_x = self.grid_offset_x + (grid_x - grid_y) * self.tile_w / 2
        screen_y = self.grid_offset_y + (grid_x + grid_y) * self.tile_h / 2
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.dishes_prepared = 0
        self.curses_triggered = 0
        self.prev_space_held = False
        self.particles.clear()

        # --- Define Game Elements ---
        ingredient_names = ["Tomato", "Cheese", "Lettuce", "Mushroom", "Onion"]
        self.ingredients = [
            {
                "name": name,
                "color": self.INGREDIENT_COLORS[i],
                "pos": [1, 2 + i],
                "id": i,
                "locked": False,
            }
            for i, name in enumerate(ingredient_names)
        ]

        dish_positions = [[5, 2], [7, 2], [9, 4], [5, 6], [7, 6]]
        self.rng.shuffle(dish_positions)
        self.dishes = [
            {
                "requires": ingredient_names[i],
                "target_pos": dish_positions[i],
                "complete": False,
            }
            for i in range(len(ingredient_names))
        ]

        self.curse_zones = [
            {"pos": [6, 4], "pulse": 0.0},
            {"pos": [8, 3], "pulse": math.pi / 2},
            {"pos": [6, 7], "pulse": math.pi},
        ]

        self.selected_ingredient_idx = 0
        self.selected_dish_idx = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Handle Input ---
        # Cycle through unlocked ingredients
        unlocked_indices = [i for i, ing in enumerate(self.ingredients) if not ing["locked"]]
        if unlocked_indices:
            current_selection_pos = unlocked_indices.index(self.selected_ingredient_idx) if self.selected_ingredient_idx in unlocked_indices else 0
            
            if movement == 1: # Up
                current_selection_pos = (current_selection_pos - 1 + len(unlocked_indices)) % len(unlocked_indices)
                self.selected_ingredient_idx = unlocked_indices[current_selection_pos]
            elif movement == 2: # Down
                current_selection_pos = (current_selection_pos + 1) % len(unlocked_indices)
                self.selected_ingredient_idx = unlocked_indices[current_selection_pos]

        # Cycle through dishes/targets
        if movement == 3: # Left
            self.selected_dish_idx = (self.selected_dish_idx - 1 + len(self.dishes)) % len(self.dishes)
        elif movement == 4: # Right
            self.selected_dish_idx = (self.selected_dish_idx + 1) % len(self.dishes)

        # --- Handle Action Confirmation ---
        space_press = space_held and not self.prev_space_held
        if space_press:
            self.steps += 1
            
            source_ingredient = self.ingredients[self.selected_ingredient_idx]
            target_dish = self.dishes[self.selected_dish_idx]

            if not source_ingredient["locked"]:
                old_pos = list(source_ingredient["pos"])
                new_pos = list(target_dish["target_pos"])
                
                # Check for an ingredient already at the target location to swap with
                swapped_ingredient = None
                for ing in self.ingredients:
                    if ing["pos"] == new_pos and not ing["locked"]:
                        swapped_ingredient = ing
                        break

                # Move ingredients
                source_ingredient["pos"] = new_pos
                if swapped_ingredient:
                    swapped_ingredient["pos"] = old_pos
                
                # --- Calculate Rewards & Check Game State ---
                # 1. Distance-to-curse reward
                dist_before = min(math.dist(old_pos, cz["pos"]) for cz in self.curse_zones)
                dist_after = min(math.dist(new_pos, cz["pos"]) for cz in self.curse_zones)
                if dist_after > dist_before: reward += 0.1
                elif dist_after < dist_before: reward -= 0.1

                # 2. Curse trigger check
                if dist_after < self.CURSE_RADIUS:
                    # sfx: curse_trigger.wav
                    self.curses_triggered += 1
                    reward -= 20
                    self._create_particles(self._to_iso(*new_pos), "curse")
                
                # 3. Dish completion check
                for i, dish in enumerate(self.dishes):
                    if not dish["complete"]:
                        ingredient_for_dish = next(ing for ing in self.ingredients if ing["name"] == dish["requires"])
                        if ingredient_for_dish["pos"] == dish["target_pos"]:
                            # sfx: dish_complete.wav
                            dish["complete"] = True
                            ingredient_for_dish["locked"] = True
                            self.dishes_prepared += 1
                            reward += 10
                            self._create_particles(self._to_iso(*ingredient_for_dish["pos"]), "success")
                            
                            # Auto-select next available ingredient
                            unlocked_indices = [i for i, ing in enumerate(self.ingredients) if not ing["locked"]]
                            if unlocked_indices:
                                self.selected_ingredient_idx = unlocked_indices[0]
                            break # Only one dish can be completed per move
        
        self.prev_space_held = space_held
        self.score += reward

        # --- Check Termination ---
        terminated = False
        if self.dishes_prepared >= self.WIN_CONDITION:
            # sfx: win_jingle.wav
            reward += 100
            self.score += 100
            terminated = True
        elif self.curses_triggered >= self.LOSE_CONDITION:
            # sfx: lose_sound.wav
            reward -= 100
            self.score -= 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "dishes_prepared": self.dishes_prepared,
            "curses_triggered": self.curses_triggered,
        }

    def _render_game(self):
        # --- Draw Counter and Grid ---
        counter_points = [self._to_iso(3, 0), self._to_iso(12, 0), self._to_iso(12, 9), self._to_iso(3, 9)]
        pygame.gfxdraw.filled_polygon(self.screen, counter_points, self.COLOR_COUNTER)
        pygame.gfxdraw.aapolygon(self.screen, counter_points, self.COLOR_COUNTER_EDGE)

        # --- Collect and Sort all drawable entities for correct isometric order ---
        drawable_entities = []
        
        # Curse zones (drawn on the ground)
        for cz in self.curse_zones:
            cz["pulse"] = (cz["pulse"] + 0.05) % (2 * math.pi)
            radius = 20 + 5 * math.sin(cz["pulse"])
            drawable_entities.append({"type": "curse_zone", "pos": cz["pos"], "radius": radius, "sort_key": cz["pos"][1] + cz["pos"][0] * 0.01})
            
        # Dish target zones
        for i, dish in enumerate(self.dishes):
            is_selected = (i == self.selected_dish_idx)
            drawable_entities.append({"type": "dish_zone", "dish": dish, "is_selected": is_selected, "sort_key": dish["target_pos"][1] + dish["target_pos"][0] * 0.01})

        # Ingredients
        for i, ing in enumerate(self.ingredients):
            is_selected = (i == self.selected_ingredient_idx and not ing["locked"])
            drawable_entities.append({"type": "ingredient", "ing": ing, "is_selected": is_selected, "sort_key": ing["pos"][1] + ing["pos"][0] * 0.01 + 0.5})

        drawable_entities.sort(key=lambda x: x["sort_key"])

        # --- Render sorted entities ---
        for entity in drawable_entities:
            if entity["type"] == "curse_zone":
                center = self._to_iso(*entity["pos"])
                color = (*self.COLOR_CURSE, 70 + int(30 * math.sin(entity["radius"] * 0.1)))
                pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], int(entity["radius"]), color)
            
            elif entity["type"] == "dish_zone":
                dish = entity["dish"]
                pos = dish["target_pos"]
                center = self._to_iso(*pos)
                color = self.COLOR_SAFE if dish["complete"] else self.COLOR_GRID
                
                # Draw dish slot
                points = [self._to_iso(pos[0], pos[1]), self._to_iso(pos[0] + 1, pos[1]), self._to_iso(pos[0] + 1, pos[1] + 1), self._to_iso(pos[0], pos[1] + 1)]
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_COUNTER_EDGE)

                # Highlight if selected target
                if entity["is_selected"] and not dish["complete"]:
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SELECT_TARGET)
                    pygame.gfxdraw.aapolygon(self.screen, [(p[0]+1, p[1]) for p in points], self.COLOR_SELECT_TARGET)
                    pygame.gfxdraw.aapolygon(self.screen, [(p[0]-1, p[1]) for p in points], self.COLOR_SELECT_TARGET)

            elif entity["type"] == "ingredient":
                ing = entity["ing"]
                center_x, center_y = self._to_iso(*ing["pos"])
                center_y -= 10 # Elevate ingredient slightly
                
                # Shadow
                shadow_center_y = center_y + 12
                pygame.gfxdraw.filled_ellipse(self.screen, center_x, shadow_center_y, 12, 6, (0,0,0,90))

                # Ingredient Body
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 10, ing["color"])
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 10, tuple(max(0, c-40) for c in ing["color"]))

                # Highlight if selected ingredient
                if entity["is_selected"]:
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 14, self.COLOR_SELECT_INGREDIENT)
                
                # Lock icon if part of a completed dish
                if ing["locked"]:
                    pygame.gfxdraw.filled_circle(self.screen, center_x + 8, center_y - 8, 4, self.COLOR_SAFE)
                    pygame.gfxdraw.aacircle(self.screen, center_x + 8, center_y - 8, 4, (200,255,220))


        self._update_and_render_particles()

    def _render_ui(self):
        # Dishes prepared
        dishes_text = self.font_small.render(f"Dishes: {self.dishes_prepared} / {self.WIN_CONDITION}", True, self.COLOR_TEXT)
        self.screen.blit(dishes_text, (10, 10))

        # Curses triggered
        curses_text = self.font_small.render(f"Curses: {self.curses_triggered} / {self.LOSE_CONDITION}", True, self.COLOR_TEXT)
        self.screen.blit(curses_text, (self.screen_width - curses_text.get_width() - 10, 10))

        # Score
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.screen_width / 2 - score_text.get_width() / 2, self.screen_height - score_text.get_height() - 5))

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.dishes_prepared >= self.WIN_CONDITION:
                end_text = self.font_large.render("Dishes Served!", True, self.COLOR_SAFE)
            else:
                end_text = self.font_large.render("Kitchen Cursed!", True, self.COLOR_CURSE)
            
            self.screen.blit(end_text, (self.screen_width / 2 - end_text.get_width() / 2, self.screen_height / 2 - end_text.get_height() / 2))

    def _create_particles(self, pos, p_type):
        if p_type == "success":
            for _ in range(30):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed - 1]
                life = random.randint(20, 40)
                color = random.choice([(255, 255, 100), (255, 255, 255), (200, 255, 200)])
                self.particles.append({"pos": list(pos), "vel": vel, "life": life, "color": color, "type": "spark"})
        elif p_type == "curse":
            for _ in range(40):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(0.5, 2)
                vel = [math.cos(angle) * speed, -random.uniform(0.5, 1.5)]
                life = random.randint(40, 70)
                color = random.choice([(100, 20, 60), (50, 10, 30), (120, 40, 80)])
                self.particles.append({"pos": list(pos), "vel": vel, "life": life, "color": color, "type": "smoke"})

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                continue
            
            alpha = int(255 * (p["life"] / 50))
            color = (*p["color"], max(0, min(255, alpha)))
            
            if p["type"] == "spark":
                pygame.draw.line(self.screen, color, p["pos"], (p["pos"][0]-p["vel"][0]*2, p["pos"][1]-p["vel"][1]*2), 1)
            elif p["type"] == "smoke":
                p["vel"][0] += random.uniform(-0.1, 0.1) # Wiggle
                radius = int(10 * (1 - p["life"] / 70))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, color)

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for interactive testing ---
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Cursed Kitchen")
    clock = pygame.time.Clock()
    running = True
    
    while running:
        movement = 0
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space = 1
                elif event.key == pygame.K_r:
                    env.reset()
                elif event.key == pygame.K_q:
                    running = False

        action = [movement, space, 0] # Movement, Space, Shift
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # A small delay before auto-resetting in interactive mode
            pygame.time.wait(2000)
            env.reset()
        
        # --- Display the observation ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
        
    env.close()