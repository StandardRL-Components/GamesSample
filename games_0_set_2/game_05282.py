
# Generated: 2025-08-28T04:32:59.161204
# Source Brief: brief_05282.md
# Brief Index: 5282

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move cursor between ingredients and potions. "
        "Press Space to select an item. Select an ingredient then a potion to combine them. "
        "Press Shift to cancel selections."
    )

    game_description = (
        "Brew perfect potions by strategically combining ingredients in this isometric puzzle game. "
        "Discover the correct recipes for the three potions before you run out of ingredients."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and rendering setup
        self.width, self.height = 640, 400
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 18)

        # Colors
        self.COLOR_BG = (28, 22, 38)
        self.COLOR_GRID = (40, 32, 52)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_CURSOR = (255, 220, 0)
        self.COLOR_SELECTION = (0, 255, 150)
        self.COLOR_WATER = (135, 206, 235)
        self.COLOR_RUINED = (80, 80, 70)

        # Isometric projection settings
        self.tile_w, self.tile_h = 64, 32
        self.origin_x, self.origin_y = self.width // 2, 100

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        self.ingredients_data = []
        self.recipes = []
        self.potion_targets = []
        
        self.ingredients_used = []
        self.potions = []
        
        self.cursor_pos = 0
        self.selected_ingredient = None
        self.selected_potion = None
        
        self.particles = []
        
        self.reset()
        
        # self.validate_implementation() # Optional: Call to self-check during development

    def _define_game_elements(self):
        # Define ingredients with their properties
        self.ingredients_data = [
            {'name': 'Ruby Dust', 'color': (231, 76, 60), 'shape': 'circle'},
            {'name': 'Sun Crystal', 'color': (241, 196, 15), 'shape': 'diamond'},
            {'name': 'Sky Blossom', 'color': (52, 152, 219), 'shape': 'triangle'},
            {'name': 'Forest Heart', 'color': (46, 204, 113), 'shape': 'square'},
            {'name': 'Moon Petal', 'color': (142, 68, 173), 'shape': 'star'},
            {'name': 'Obsidian Shard', 'color': (52, 73, 94), 'shape': 'hex'},
            {'name': 'Grumbleweed', 'color': (120, 90, 40), 'shape': 'circle'}, # Dud
            {'name': 'Gloom Fungus', 'color': (149, 165, 166), 'shape': 'square'}, # Dud
        ]
        
        # Define recipes and target potion appearances
        self.recipes = [
            {'name': 'Potion of Vigor', 'ids': {0, 1}, 'target_color': (236, 136, 37)},
            {'name': 'Potion of Clarity', 'ids': {2, 4}, 'target_color': (97, 110, 196)},
            {'name': 'Potion of Luck', 'ids': {3, 5}, 'target_color': (49, 138, 103)},
        ]
        
        # Grid positions for all items
        self.ingredient_grid_pos = [(x, y) for y in range(2) for x in range(4)]
        self.potion_grid_pos = [(0.5, 3), (1.5, 3), (2.5, 3)]

    def _iso_to_screen(self, iso_x, iso_y):
        screen_x = self.origin_x + (iso_x - iso_y) * self.tile_w / 2
        screen_y = self.origin_y + (iso_x + iso_y) * self.tile_h / 2
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._define_game_elements()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        self.ingredients_used = [False] * 8
        self.potions = [
            {
                'ingredients': set(),
                'color': self.COLOR_WATER,
                'state': 'empty', # empty, brewing, perfect, ruined
                'pos': self.potion_grid_pos[i]
            } for i in range(3)
        ]
        
        self.cursor_pos = 0 # 0-7 for ingredients, 8-10 for potions
        self.selected_ingredient = None
        self.selected_potion = None
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle Input and State Changes
        if shift_press:
            self.selected_ingredient = None
            self.selected_potion = None
            # Sound: Cancel/Deselect
        
        self._handle_movement(movement)

        if space_press:
            reward += self._handle_selection()
        
        # 2. Update Particles
        self._update_particles()
        
        # 3. Check for Termination
        num_perfect = sum(1 for p in self.potions if p['state'] == 'perfect')
        num_used_ingredients = sum(self.ingredients_used)

        if num_perfect == 3:
            self.game_over = True
            terminated = True
            reward += 50
            self.score += 50
            self.win_message = "PERFECT ALCHEMY!"
            # Sound: Victory Fanfare
        elif num_used_ingredients >= 8:
            self.game_over = True
            terminated = True
            self.win_message = "OUT OF INGREDIENTS"
            # Sound: Game Over
        
        if self.steps >= 1000:
            self.game_over = True
            terminated = True
            self.win_message = "TIME'S UP"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 0: return
        
        x, y = self.cursor_pos % 4, self.cursor_pos // 4
        
        if self.cursor_pos >= 8: # Potion row
            potion_idx = self.cursor_pos - 8
            if movement == 1: # Up
                self.cursor_pos = 4 + potion_idx + (1 if potion_idx > 1 else 0)
            elif movement == 3: # Left
                self.cursor_pos = 8 + max(0, potion_idx - 1)
            elif movement == 4: # Right
                self.cursor_pos = 8 + min(2, potion_idx + 1)
        else: # Ingredient grid
            if movement == 1: # Up
                self.cursor_pos = max(0, self.cursor_pos - 4)
            elif movement == 2: # Down
                if self.cursor_pos < 4:
                    self.cursor_pos += 4
                else: # Move to potions
                    self.cursor_pos = 8 + min(self.cursor_pos - 4, 2)
            elif movement == 3: # Left
                if x > 0: self.cursor_pos -= 1
            elif movement == 4: # Right
                if x < 3: self.cursor_pos += 1

    def _handle_selection(self):
        # Sound: Select/Confirm
        if self.cursor_pos < 8: # Selected an ingredient
            if not self.ingredients_used[self.cursor_pos]:
                self.selected_ingredient = self.cursor_pos
            else: # Tried to select used ingredient
                # Sound: Error/Buzz
                return 0
        else: # Selected a potion
            self.selected_potion = self.cursor_pos - 8

        # Check for combination
        if self.selected_ingredient is not None and self.selected_potion is not None:
            return self._combine_items()
        return 0

    def _combine_items(self):
        ing_idx = self.selected_ingredient
        pot_idx = self.selected_potion
        potion = self.potions[pot_idx]

        # Reset selections for next action
        self.selected_ingredient = None
        self.selected_potion = None

        if potion['state'] in ['perfect', 'ruined']:
            # Sound: Error/Cannot combine
            return 0
        
        # Mark ingredient as used
        self.ingredients_used[ing_idx] = True
        potion['ingredients'].add(ing_idx)
        potion['state'] = 'brewing'

        # Spawn particles
        ing_pos = self._iso_to_screen(*self.ingredient_grid_pos[ing_idx])
        pot_pos = self._iso_to_screen(*self.potion_grid_pos[pot_idx])
        ing_color = self.ingredients_data[ing_idx]['color']
        self._spawn_particles(ing_pos, pot_pos, ing_color)
        
        # Sound: Potion brewing/fizz
        
        # Check for recipe completion or ruin
        is_dud = ing_idx >= 6
        is_overfilled = len(potion['ingredients']) > 2
        
        if is_dud or is_overfilled:
            potion['state'] = 'ruined'
            potion['color'] = self.COLOR_RUINED
            self.score -= 2
            return -2
        
        for recipe in self.recipes:
            if potion['ingredients'] == recipe['ids']:
                potion['state'] = 'perfect'
                potion['color'] = recipe['target_color']
                # Sound: Success/Chime
                self.score += 10
                return 10
        
        # If not perfect or ruined, just a brewing step
        self.score -= 0.1
        # Blend colors for visual feedback
        old_r, old_g, old_b = potion['color']
        add_r, add_g, add_b = self.ingredients_data[ing_idx]['color']
        num_ings = len(potion['ingredients'])
        potion['color'] = (
            (old_r * (num_ings -1) + add_r) / num_ings,
            (old_g * (num_ings -1) + add_g) / num_ings,
            (old_b * (num_ings -1) + add_b) / num_ings,
        )
        return -0.1

    def _spawn_particles(self, start_pos, end_pos, color):
        for _ in range(30):
            # Travel particles
            self.particles.append({
                'pos': list(start_pos),
                'vel': [(end_pos[0] - start_pos[0]) / 30 + self.np_random.uniform(-1, 1), 
                        (end_pos[1] - start_pos[1]) / 30 + self.np_random.uniform(-1, 1)],
                'color': color,
                'lifetime': 30,
                'type': 'travel'
            })
        # Splash particles
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(end_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 2], # Move upwards
                'color': color,
                'lifetime': self.np_random.integers(20, 40),
                'type': 'splash'
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            if p['type'] == 'splash':
                p['vel'][1] += 0.15 # Gravity
            if p['lifetime'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        self._render_grid()
        self._render_ingredients()
        self._render_potions()
        self._render_cursor_and_selections()
        self._render_particles()

    def _render_grid(self):
        for i in range(6):
            for j in range(5):
                p1 = self._iso_to_screen(i, j)
                p2 = self._iso_to_screen(i + 1, j)
                p3 = self._iso_to_screen(i, j + 1)
                pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p3)

    def _render_ingredients(self):
        for i, pos in enumerate(self.ingredient_grid_pos):
            screen_pos = self._iso_to_screen(*pos)
            color = self.ingredients_data[i]['color']
            shape = self.ingredients_data[i]['shape']
            
            if self.ingredients_used[i]:
                color = tuple(c // 4 for c in color)

            size = 12
            if shape == 'circle':
                pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1] - size, size, color)
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1] - size, size, color)
            elif shape == 'square':
                rect = pygame.Rect(screen_pos[0] - size, screen_pos[1] - size * 1.5, size * 2, size * 2)
                pygame.draw.rect(self.screen, color, rect)
            elif shape == 'triangle':
                points = [(screen_pos[0], screen_pos[1] - size*1.8), (screen_pos[0]-size, screen_pos[1]-size*0.2), (screen_pos[0]+size, screen_pos[1]-size*0.2)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            else: # Default to diamond/hex-like shape for others
                points = [(screen_pos[0], screen_pos[1] - size*1.5), (screen_pos[0]-size, screen_pos[1]-size*0.75), (screen_pos[0], screen_pos[1]), (screen_pos[0]+size, screen_pos[1]-size*0.75)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_potions(self):
        for i, potion in enumerate(self.potions):
            screen_pos = self._iso_to_screen(*potion['pos'])
            
            # Draw liquid
            liquid_level = 0.7 if potion['state'] != 'empty' else 0.1
            if potion['state'] == 'perfect': liquid_level = 0.8
            
            fill_points = [
                (screen_pos[0] - 18, screen_pos[1] + 10),
                (screen_pos[0] + 18, screen_pos[1] + 10),
                (screen_pos[0] + 15, screen_pos[1] - 30 * liquid_level),
                (screen_pos[0] - 15, screen_pos[1] - 30 * liquid_level)
            ]
            
            color = tuple(int(c) for c in potion['color'])
            pygame.gfxdraw.filled_polygon(self.screen, fill_points, color)
            
            # Draw bottle outline
            bottle_points = [
                (screen_pos[0] - 20, screen_pos[1] + 12), (screen_pos[0] + 20, screen_pos[1] + 12),
                (screen_pos[0] + 15, screen_pos[1] - 25), (screen_pos[0] + 8, screen_pos[1] - 35),
                (screen_pos[0] + 10, screen_pos[1] - 40), (screen_pos[0] - 10, screen_pos[1] - 40),
                (screen_pos[0] - 8, screen_pos[1] - 35), (screen_pos[0] - 15, screen_pos[1] - 25),
            ]
            pygame.gfxdraw.aapolygon(self.screen, bottle_points, self.COLOR_TEXT)

            # Shine effect
            shine_rect = pygame.Rect(screen_pos[0] - 12, screen_pos[1] - 20, 8, 25)
            pygame.draw.arc(self.screen, (255, 255, 255, 100), shine_rect, math.radians(90), math.radians(180), 2)

            if potion['state'] == 'perfect':
                self._render_star(screen_pos[0], screen_pos[1] - 45, 10, self.COLOR_CURSOR)

    def _render_star(self, x, y, size, color):
        points = []
        for i in range(10):
            radius = size if i % 2 == 0 else size / 2
            angle = i * math.pi / 5
            points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_cursor_and_selections(self):
        # Draw selected items first
        if self.selected_ingredient is not None:
            pos = self.ingredient_grid_pos[self.selected_ingredient]
            self._render_selection_highlight(pos, self.COLOR_SELECTION)
        if self.selected_potion is not None:
            pos = self.potion_grid_pos[self.selected_potion]
            self._render_selection_highlight(pos, self.COLOR_SELECTION)

        # Draw cursor
        if self.cursor_pos < 8:
            pos = self.ingredient_grid_pos[self.cursor_pos]
        else:
            pos = self.potion_grid_pos[self.cursor_pos - 8]
        self._render_selection_highlight(pos, self.COLOR_CURSOR)

    def _render_selection_highlight(self, iso_pos, color):
        screen_pos = self._iso_to_screen(*iso_pos)
        points = [
            self._iso_to_screen(iso_pos[0] - 0.4, iso_pos[1] - 0.4),
            self._iso_to_screen(iso_pos[0] + 0.4, iso_pos[1] - 0.4),
            self._iso_to_screen(iso_pos[0] + 0.4, iso_pos[1] + 0.4),
            self._iso_to_screen(iso_pos[0] - 0.4, iso_pos[1] + 0.4),
        ]
        pygame.draw.aalines(self.screen, color, True, points, 2)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifetime'] / 30.0))))
            color = p['color'] + (alpha,)
            size = 3 if p['type'] == 'travel' else 2
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)
            except TypeError: # Color might not have alpha
                try:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, p['color'])
                except: # Catch any other drawing errors
                    pass


    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Ingredients remaining
        rem_text = self.font_medium.render(f"Ingredients Left: {8 - sum(self.ingredients_used)}", True, self.COLOR_TEXT)
        rem_rect = rem_text.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(rem_text, rem_rect)
        
        # Controls guide
        guide_text = self.font_small.render(self.user_guide, True, self.COLOR_GRID)
        guide_rect = guide_text.get_rect(center=(self.width / 2, self.height - 15))
        self.screen.blit(guide_text, guide_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_CURSOR)
            end_rect = end_text.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(end_text, end_rect)

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
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a dictionary to track key presses for smooth human play
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Potion Brewer")
    clock = pygame.time.Clock()

    while not done:
        # Construct action from keyboard input
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement = move_val
                break # only one movement at a time
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we need to control the speed of human input
        clock.tick(10) # Limit to 10 actions per second

    env.close()