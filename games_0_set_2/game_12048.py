import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:55:48.692024
# Source Brief: brief_02048.md
# Brief Index: 2048
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Catch falling ingredients to brew powerful potions in your cauldron. "
        "Discover new recipes and manage your health to achieve a high score."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to move your catcher and catch ingredients."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500
        self.WIN_SCORE = 1000

        # Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Colors
        self.COLOR_BG_START = (10, 20, 30)
        self.COLOR_BG_END = (30, 10, 20)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_TEXT_SHADOW = (20, 30, 40)
        self.COLOR_CATCHER = (255, 255, 100)
        self.COLOR_CATCHER_GLOW = (255, 255, 100, 50)
        self.COLOR_CAULDRON = (40, 40, 50)
        self.COLOR_CAULDRON_LIP = (60, 60, 70)
        self.COLOR_HEALTH_FG = (50, 220, 50)
        self.COLOR_HEALTH_BG = (220, 50, 50)

        # Game assets (no external files)
        self._define_game_assets()

        # Persistent state (survives reset)
        self.unlocked_recipe_indices = {0} 

        # Initialize state variables
        self.catcher_x = 0
        self.health = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.falling_ingredients = []
        self.cauldron_contents = []
        self.particles = []
        self.fireflies = []
        self.bubbles = []
        self.fall_speed = 0
        self.bg_lerp_factor = 0.0

        # The reset method will be called to set initial state
        # self.reset() is not needed here as it's part of the Gym API lifecycle
        
    def _define_game_assets(self):
        self.INGREDIENT_TYPES = [
            {'name': 'Glowshroom', 'color': (255, 80, 80), 'shape': 'circle'},
            {'name': 'Moonpetal', 'color': (80, 80, 255), 'shape': 'triangle'},
            {'name': 'Sunfruit', 'color': (255, 180, 50), 'shape': 'square'},
            {'name': 'Starbloom', 'color': (255, 255, 80), 'shape': 'star'},
        ]
        self.RECIPES = [
            {'name': 'Minor Heal', 'ingredients': [0, 0], 'reward': 1, 'score': 10, 'heal': 20},
            {'name': 'Lesser Potion', 'ingredients': [0, 1], 'reward': 2, 'score': 25, 'heal': 30},
            {'name': 'Steady Brew', 'ingredients': [2, 2, 2], 'reward': 3, 'score': 50, 'heal': 50},
            {'name': 'Lunar Elixir', 'ingredients': [1, 1, 3], 'reward': 5, 'score': 75, 'heal': 70},
            {'name': 'Solar Draught', 'ingredients': [0, 2, 3], 'reward': 7, 'score': 100, 'heal': 80},
            {'name': 'Celestial Panacea', 'ingredients': [0, 1, 2, 3], 'reward': 10, 'score': 150, 'heal': 100},
        ]
        # Pre-sort recipe ingredients for easy comparison
        for recipe in self.RECIPES:
            recipe['ingredients'].sort()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.catcher_x = self.WIDTH / 2
        self.health = 100.0
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        # Entity lists
        self.falling_ingredients = []
        self.cauldron_contents = []
        self.particles = []
        
        # Difficulty
        self.fall_speed = 2.0
        self.bg_lerp_factor = 0.0
        self._last_score_milestone_speed = 0
        self._last_score_milestone_recipe = 0
        self._last_score_milestone_bg = 0
        
        # Ambient effects
        self.fireflies = [{'pos': [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)],
                           'radius': self.np_random.uniform(1, 2.5),
                           'speed': self.np_random.uniform(0.1, 0.3),
                           'angle': self.np_random.uniform(0, 2 * math.pi),
                           'blink_offset': self.np_random.uniform(0, 5)} for _ in range(20)]
        self.bubbles = [{'pos': [self.np_random.uniform(self.WIDTH/2 - 40, self.WIDTH/2 + 40), self.HEIGHT - 20],
                         'radius': self.np_random.uniform(1, 4),
                         'speed': self.np_random.uniform(0.2, 0.8)} for _ in range(15)]

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, _, _ = action
        reward = 0
        self.steps += 1

        # 1. Handle player input
        self._handle_input(movement)
        
        # 2. Update game state
        self.health -= 0.05  # Constant health drain
        self._update_particles()
        self._update_ambient_effects()
        
        # 3. Spawn and update ingredients
        self._maybe_spawn_ingredient()
        reward += self._update_falling_ingredients()
        
        # 4. Check for potion brewing
        brew_reward, brew_score, heal_amount = self._check_brewing()
        if brew_score > 0:
            reward += brew_reward
            self.score += brew_score
            self.health = min(100.0, self.health + heal_amount)
            # # Play potion brew sound
            self._create_particles((self.WIDTH/2, self.HEIGHT-60), (150, 255, 150), 100, 
                                   life=60, speed_range=(1, 5), gravity=0.05)

        # 5. Update progression
        self._update_progression()

        # 6. Check for termination
        terminated = self.health <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True
        
        if terminated:
            self.game_over = True
            if self.health <= 0:
                reward -= 10  # Lose penalty
            if self.score >= self.WIN_SCORE:
                reward += 100  # Win bonus
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement):
        catcher_speed = 12
        if movement == 3:  # Left
            self.catcher_x -= catcher_speed
        elif movement == 4:  # Right
            self.catcher_x += catcher_speed
        self.catcher_x = np.clip(self.catcher_x, 30, self.WIDTH - 30)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += p['gravity']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_ambient_effects(self):
        # Fireflies
        for ff in self.fireflies:
            ff['pos'][0] += math.cos(ff['angle']) * ff['speed']
            ff['pos'][1] += math.sin(ff['angle']) * ff['speed']
            ff['angle'] += self.np_random.uniform(-0.1, 0.1)
            if not (0 < ff['pos'][0] < self.WIDTH and 0 < ff['pos'][1] < self.HEIGHT):
                ff['pos'] = [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)]
        # Bubbles
        for b in self.bubbles:
            b['pos'][1] -= b['speed']
            if b['pos'][1] < self.HEIGHT - 80:
                b['pos'] = [self.np_random.uniform(self.WIDTH/2 - 40, self.WIDTH/2 + 40), self.HEIGHT - 20]

    def _maybe_spawn_ingredient(self):
        spawn_chance = 0.02 + self.score / 5000  # Increases with score
        if self.np_random.random() < spawn_chance and len(self.falling_ingredients) < 10:
            unlocked_types = min(4, 1 + self.score // 150)
            ing_type = self.np_random.integers(0, unlocked_types)
            self.falling_ingredients.append({
                'type': ing_type,
                'pos': [self.np_random.uniform(50, self.WIDTH - 50), -20],
                'size': 12
            })

    def _update_falling_ingredients(self):
        reward = 0
        catcher_rect = pygame.Rect(self.catcher_x - 30, self.HEIGHT - 85, 60, 20)
        
        for ing in self.falling_ingredients[:]:
            ing['pos'][1] += self.fall_speed
            
            ing_rect = pygame.Rect(ing['pos'][0] - ing['size'], ing['pos'][1] - ing['size'], ing['size']*2, ing['size']*2)

            if catcher_rect.colliderect(ing_rect):
                self.cauldron_contents.append(ing['type'])
                self.cauldron_contents.sort()
                reward += 0.1  # Catch reward
                # # Play catch sound
                self._create_particles(ing['pos'], self.INGREDIENT_TYPES[ing['type']]['color'], 20, life=20)
                self.falling_ingredients.remove(ing)
            elif ing['pos'][1] > self.HEIGHT:
                reward -= 0.1  # Miss penalty
                self.falling_ingredients.remove(ing)
        return reward

    def _check_brewing(self):
        for i in range(len(self.RECIPES) - 1, -1, -1):
            if i in self.unlocked_recipe_indices:
                recipe = self.RECIPES[i]
                if recipe['ingredients'] == self.cauldron_contents:
                    self.cauldron_contents = []
                    return recipe['reward'], recipe['score'], recipe['heal']
        
        # If no match and cauldron is full, clear it
        if len(self.cauldron_contents) >= 4:
            self._create_particles((self.WIDTH/2, self.HEIGHT-60), (100, 100, 100), 50, life=30)
            self.cauldron_contents = []
            # # Play fizzle sound
            
        return 0, 0, 0

    def _update_progression(self):
        # Increase fall speed
        score_milestone_speed = self.score // 100
        if score_milestone_speed > self._last_score_milestone_speed:
            self.fall_speed += (score_milestone_speed - self._last_score_milestone_speed) * 0.05
            self._last_score_milestone_speed = score_milestone_speed

        # Unlock recipes
        score_milestone_recipe = self.score // 200
        if score_milestone_recipe > self._last_score_milestone_recipe:
            num_to_unlock = score_milestone_recipe - self._last_score_milestone_recipe
            for _ in range(num_to_unlock):
                if len(self.unlocked_recipe_indices) < len(self.RECIPES):
                    self.unlocked_recipe_indices.add(len(self.unlocked_recipe_indices))
            self._last_score_milestone_recipe = score_milestone_recipe

        # Update background
        score_milestone_bg = self.score // 500
        if score_milestone_bg > self._last_score_milestone_bg:
            self.bg_lerp_factor = min(1.0, self.bg_lerp_factor + 0.25)
            self._last_score_milestone_bg = score_milestone_bg
    
    def _get_observation(self):
        # Background
        bg_color = [int(s + (e - s) * self.bg_lerp_factor) for s, e in zip(self.COLOR_BG_START, self.COLOR_BG_END)]
        self.screen.fill(bg_color)
        
        # Render all elements
        self._render_ambient_effects()
        self._render_cauldron()
        self._render_game_elements()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ambient_effects(self):
        # Fireflies
        for ff in self.fireflies:
            blink = 0.5 + 0.5 * math.sin(self.steps / 10.0 + ff['blink_offset'])
            radius = int(ff['radius'] * blink)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(ff['pos'][0]), int(ff['pos'][1]), radius, (255, 255, 100, 150))
        # Bubbles
        for b in self.bubbles:
            pygame.gfxdraw.aacircle(self.screen, int(b['pos'][0]), int(b['pos'][1]), int(b['radius']), (100, 150, 200, 100))

    def _render_cauldron(self):
        cauldron_rect = pygame.Rect(self.WIDTH/2 - 60, self.HEIGHT - 80, 120, 80)
        pygame.draw.ellipse(self.screen, self.COLOR_CAULDRON, cauldron_rect)
        pygame.draw.ellipse(self.screen, self.COLOR_CAULDRON_LIP, cauldron_rect, width=5)
        pygame.draw.rect(self.screen, self.COLOR_CAULDRON_LIP, (cauldron_rect.left, cauldron_rect.top, cauldron_rect.width, 20))

    def _render_game_elements(self):
        # Catcher (represents the player's hands over the cauldron)
        catcher_y = self.HEIGHT - 80
        pygame.gfxdraw.filled_circle(self.screen, int(self.catcher_x), int(catcher_y), 22, self.COLOR_CATCHER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(self.catcher_x), int(catcher_y), 15, self.COLOR_CATCHER)
        pygame.gfxdraw.aacircle(self.screen, int(self.catcher_x), int(catcher_y), 15, (0,0,0))

        # Falling ingredients
        for ing in self.falling_ingredients:
            self._draw_ingredient_icon(self.screen, ing['type'], ing['pos'], ing['size'])

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (12, 12))
        
        # Health Bar
        health_bar_width = 200
        health_pct = max(0, self.health / 100.0)
        health_color = [int(s + (e - s) * health_pct) for s, e in zip(self.COLOR_HEALTH_BG, self.COLOR_HEALTH_FG)]
        
        pulse = 0
        if health_pct < 0.3:
            pulse = int(5 * math.sin(self.steps * 0.5))

        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (self.WIDTH - health_bar_width - 10, 10, health_bar_width, 25))
        if health_pct > 0:
            pygame.draw.rect(self.screen, health_color, (self.WIDTH - health_bar_width - 10 + pulse/2, 10, health_pct * health_bar_width - pulse, 25))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.WIDTH - health_bar_width - 10, 10, health_bar_width, 25), 2)
        
        # Recipe Display (most complex available)
        if self.unlocked_recipe_indices:
            recipe_to_show_idx = max(self.unlocked_recipe_indices)
            recipe_to_show = self.RECIPES[recipe_to_show_idx]
            recipe_text = self.font_small.render(f"Next Brew: {recipe_to_show['name']}", True, self.COLOR_TEXT)
            self.screen.blit(recipe_text, (self.WIDTH/2 - recipe_text.get_width()/2, self.HEIGHT - 140))
            
            total_recipe_width = len(recipe_to_show['ingredients']) * 30
            start_x = self.WIDTH/2 - total_recipe_width/2
            for i, ing_idx in enumerate(recipe_to_show['ingredients']):
                self._draw_ingredient_icon(self.screen, ing_idx, (start_x + i * 30, self.HEIGHT - 115), 8, alpha=150)

        # Cauldron contents
        total_contents_width = len(self.cauldron_contents) * 35
        start_x = self.WIDTH/2 - total_contents_width/2
        for i, ing_idx in enumerate(self.cauldron_contents):
            self._draw_ingredient_icon(self.screen, ing_idx, (start_x + i * 35, self.HEIGHT - 55), 10)

    def _draw_ingredient_icon(self, surface, ing_idx, pos, size, alpha=255):
        details = self.INGREDIENT_TYPES[ing_idx]
        color = details['color']
        glow_color = (*color, int(alpha*0.3))
        main_color = (*color, alpha)
        
        x, y = int(pos[0]), int(pos[1])
        
        # Glow
        pygame.gfxdraw.filled_circle(surface, x, y, int(size * 1.5), glow_color)

        if details['shape'] == 'circle':
            pygame.gfxdraw.filled_circle(surface, x, y, size, main_color)
            pygame.gfxdraw.aacircle(surface, x, y, size, (0,0,0,alpha))
        elif details['shape'] == 'square':
            rect = pygame.Rect(x - size, y - size, size * 2, size * 2)
            pygame.draw.rect(surface, main_color, rect)
            pygame.draw.rect(surface, (0,0,0,alpha), rect, 1)
        elif details['shape'] == 'triangle':
            points = [(x, y - size), (x - size, y + size//2), (x + size, y + size//2)]
            pygame.gfxdraw.filled_polygon(surface, points, main_color)
            pygame.gfxdraw.aapolygon(surface, points, (0,0,0,alpha))
        elif details['shape'] == 'star':
            num_points = 5
            points = []
            for i in range(num_points * 2):
                r = size if i % 2 == 0 else size / 2
                angle = i * math.pi / num_points - math.pi/2
                points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
            pygame.gfxdraw.filled_polygon(surface, points, main_color)
            pygame.gfxdraw.aapolygon(surface, points, (0,0,0,alpha))

    def _create_particles(self, pos, color, count, life=40, speed_range=(0.5, 2.5), gravity=0.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(*speed_range)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(life // 2, life),
                'max_life': life,
                'color': color,
                'radius': self.np_random.uniform(1, 4),
                'gravity': gravity
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.health,
            "unlocked_recipes": len(self.unlocked_recipe_indices)
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and testing, and will not be run by the evaluation server.
    # The evaluation server will instantiate the class and call reset() and step() directly.
    
    # Un-comment the line below to run with display
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    pygame.display.set_caption("Potion Brewer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not terminated:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # space/shift not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()