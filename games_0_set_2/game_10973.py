import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:17:48.996790
# Source Brief: brief_00973.md
# Brief Index: 973
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your potion shop from waves of monsters. Brew powerful potions from ingredients and use them to "
        "defeat enemies before they destroy your shop."
    )
    user_guide = (
        "Use ←→ arrow keys to switch between brewing and using potions. Use ↑↓ to select an item. "
        "Press space to brew or use the selected potion."
    )
    auto_advance = False

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    SHOP_Y = 320
    ENEMY_SPAWN_Y = 50
    MAX_WAVES = 20
    MAX_STEPS = 1000
    SHOP_MAX_HEALTH = 100

    # --- COLORS ---
    COLOR_BG = (18, 12, 24)
    COLOR_UI_BG = (30, 22, 40)
    COLOR_UI_BORDER = (60, 52, 70)
    COLOR_TEXT = (230, 230, 240)
    COLOR_TEXT_DIM = (150, 150, 160)
    COLOR_HIGHLIGHT = (255, 220, 100)
    COLOR_HIGHLIGHT_GLOW = (255, 220, 100, 50)
    COLOR_HEALTH_BAR = (40, 200, 80)
    COLOR_HEALTH_BAR_BG = (200, 40, 80)
    
    # Ingredient & Potion Colors
    INGREDIENT_COLORS = {
        "Fiery Bloom": (255, 80, 0),
        "Ice Petal": (100, 200, 255),
        "Glow Mushroom": (220, 180, 255),
        "Sunstone Shard": (255, 200, 0)
    }
    POTION_COLORS = {
        "Fire Bomb": (230, 40, 40),       # Red: Damage
        "Frost Flask": (60, 150, 255),    # Blue: Defensive/Slow
        "Healing Draught": (50, 220, 90), # Green: Healing
        "Thunder Potion": (200, 50, 255)  # Purple: Special/AoE
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 14)
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- Game Data ---
        self.ingredients_data = list(self.INGREDIENT_COLORS.keys())
        self.potions_data = {
            "Fire Bomb": {
                "recipe": {"Fiery Bloom": 1, "Glow Mushroom": 1},
                "unlock_wave": 1,
                "effect": "damage", "power": 30
            },
            "Frost Flask": {
                "recipe": {"Ice Petal": 1, "Glow Mushroom": 1},
                "unlock_wave": 1,
                "effect": "slow", "power": 15
            },
            "Healing Draught": {
                "recipe": {"Sunstone Shard": 1, "Glow Mushroom": 1},
                "unlock_wave": 5,
                "effect": "heal", "power": 25
            },
            "Thunder Potion": {
                "recipe": {"Fiery Bloom": 1, "Ice Petal": 1, "Sunstone Shard": 1},
                "unlock_wave": 10,
                "effect": "aoe_damage", "power": 20
            }
        }
        
        # --- Initialize State Variables ---
        # These are reset in self.reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.shop_health = 0
        self.current_wave = 0
        self.enemies = []
        self.ingredients = {}
        self.potions = {}
        self.unlocked_recipes = []
        self.particles = []
        self.ui_messages = deque(maxlen=3)
        
        # UI State
        self.ui_focus = "brew"  # "brew" or "use"
        self.brew_selection_idx = 0
        self.use_selection_idx = 0
        
        # Input handling
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.shop_health = self.SHOP_MAX_HEALTH
        self.current_wave = 1
        
        self.ingredients = {name: 5 for name in self.ingredients_data}
        self.potions = {name: 0 for name in self.potions_data}
        self.potions["Fire Bomb"] = 2
        
        self.enemies = []
        self._spawn_wave()
        
        self._update_unlocked_recipes()
        
        self.particles = []
        self.ui_messages.clear()
        
        self.ui_focus = "brew"
        self.brew_selection_idx = 0
        self.use_selection_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        
        reward = 0
        player_took_turn = False
        
        # --- 1. Handle Player Input & Action ---
        self._handle_ui_navigation(movement)
        
        if space_pressed:
            if self.ui_focus == "brew":
                action_reward, success = self._try_brew_potion()
                if success:
                    player_took_turn = True
                    reward += action_reward
            elif self.ui_focus == "use":
                action_reward, success = self._try_use_potion()
                if success:
                    player_took_turn = True
                    reward += action_reward
        
        # --- 2. If Turn Occurred, Advance Game State ---
        if player_took_turn:
            self.steps += 1
            
            # Enemy turn
            damage_taken = self._update_enemies()
            reward -= damage_taken * 0.1

            # Check for wave completion
            if not self.enemies:
                reward += 5  # Wave clear bonus
                self.current_wave += 1
                
                # Replenish basic ingredients
                for ing in self.ingredients: self.ingredients[ing] += 2
                
                if self.current_wave > self.MAX_WAVES:
                    self.game_over = True # VICTORY
                else:
                    new_unlocks = self._update_unlocked_recipes()
                    if new_unlocks:
                        reward += 10 # Unlock bonus
                        self.ui_messages.appendleft("New recipes unlocked!")
                    self._spawn_wave()

        # --- 3. Update Visuals & Check Termination ---
        self._update_particles()
        
        terminated = self._check_termination()
        if terminated:
            if self.shop_health <= 0:
                reward = -100
                self.ui_messages.appendleft("Your shop has been destroyed!")
            elif self.current_wave > self.MAX_WAVES:
                reward = 100
                self.ui_messages.appendleft("You survived all waves! Victory!")
            self.game_over = True
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        obs = self._get_observation()
        info = self._get_info()
        truncated = False # This environment does not truncate based on time limit in the same way, MAX_STEPS is a termination condition
        
        return obs, reward, terminated, truncated, info

    # --- Game Logic Helpers ---

    def _handle_ui_navigation(self, movement):
        # movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 3: self.ui_focus = "brew"
        elif movement == 4: self.ui_focus = "use"
        
        if self.ui_focus == "brew":
            if not self.unlocked_recipes: return
            if movement == 1: self.brew_selection_idx = max(0, self.brew_selection_idx - 1)
            if movement == 2: self.brew_selection_idx = min(len(self.unlocked_recipes) - 1, self.brew_selection_idx + 1)
        elif self.ui_focus == "use":
            available_potions = [p for p, c in self.potions.items() if c > 0]
            if not available_potions: return
            if movement == 1: self.use_selection_idx = max(0, self.use_selection_idx - 1)
            if movement == 2: self.use_selection_idx = min(len(available_potions) - 1, self.use_selection_idx + 1)

    def _try_brew_potion(self):
        if not self.unlocked_recipes: return 0, False
        potion_name = self.unlocked_recipes[self.brew_selection_idx]
        recipe = self.potions_data[potion_name]["recipe"]
        
        can_brew = all(self.ingredients[ing] >= count for ing, count in recipe.items())
        
        if can_brew:
            for ing, count in recipe.items():
                self.ingredients[ing] -= count
            self.potions[potion_name] += 1
            self.ui_messages.appendleft(f"Brewed {potion_name}!")
            # Sound: *bubble pop*
            self._create_particles(
                (140, 250), 20, self.POTION_COLORS[potion_name], 
                min_vel=1, max_vel=3, min_life=20, max_life=40
            )
            return 0.5, True # Small reward for brewing
        else:
            self.ui_messages.appendleft("Not enough ingredients!")
            # Sound: *error buzz*
            return 0, False

    def _try_use_potion(self):
        available_potions = [p for p, c in self.potions.items() if c > 0]
        if not available_potions:
            self.ui_messages.appendleft("No potions to use!")
            return 0, False
        
        # Ensure selection index is valid
        if self.use_selection_idx >= len(available_potions):
            self.use_selection_idx = len(available_potions) - 1

        potion_name = available_potions[self.use_selection_idx]
        self.potions[potion_name] -= 1
        
        data = self.potions_data[potion_name]
        effect, power = data["effect"], data["power"]
        total_damage_dealt = 0
        
        self.ui_messages.appendleft(f"Used {potion_name}!")
        # Sound: *potion splash*
        
        if effect == "damage":
            if self.enemies:
                target = min(self.enemies, key=lambda e: e['pos'].y)
                damage_dealt = min(power, target['hp'])
                target['hp'] -= power
                total_damage_dealt += damage_dealt
                self.score += damage_dealt
                self._create_particles(target['pos'], 30, self.POTION_COLORS[potion_name], min_life=15, max_life=30)
        elif effect == "aoe_damage":
            for e in self.enemies:
                damage_dealt = min(power, e['hp'])
                e['hp'] -= power
                total_damage_dealt += damage_dealt
                self.score += damage_dealt
            if self.enemies:
                 self._create_particles((self.SCREEN_WIDTH/2, 180), 80, self.POTION_COLORS[potion_name], min_vel=2, max_vel=5, min_life=30, max_life=50)
        elif effect == "slow":
            for e in self.enemies:
                damage_dealt = min(power, e['hp'])
                e['hp'] -= power
                total_damage_dealt += damage_dealt
                self.score += damage_dealt
                e['slowed'] = 2 # Slowed for this turn and the next
            if self.enemies:
                 self._create_particles((self.SCREEN_WIDTH/2, 180), 50, self.POTION_COLORS[potion_name], min_vel=1, max_vel=2, min_life=40, max_life=60)
        elif effect == "heal":
            healed_amount = min(power, self.SHOP_MAX_HEALTH - self.shop_health)
            self.shop_health += healed_amount
            self.shop_health = min(self.shop_health, self.SHOP_MAX_HEALTH)
            self._create_particles((self.SCREEN_WIDTH/2, self.SHOP_Y), 50, self.POTION_COLORS[potion_name], min_vel=1, max_vel=3, min_life=30, max_life=50)

        self.enemies = [e for e in self.enemies if e['hp'] > 0]
        return total_damage_dealt * 0.1, True

    def _update_enemies(self):
        total_damage_taken = 0
        for enemy in self.enemies:
            if enemy['slowed'] > 0:
                enemy['slowed'] -= 1
                continue
            
            enemy['pos'].y += enemy['speed']
            if enemy['pos'].y >= self.SHOP_Y:
                enemy['pos'].y = self.SHOP_Y
                total_damage_taken += enemy['attack']
                self.shop_health -= enemy['attack']
                # Sound: *crunch*
                self._create_particles(enemy['pos'], 10, self.COLOR_HEALTH_BAR_BG, min_life=10, max_life=20)
        return total_damage_taken

    def _spawn_wave(self):
        self.enemies = []
        num_enemies = 3 + self.current_wave
        base_hp = 10 * (1.1 ** (self.current_wave - 1))
        base_attack = 1 * (1.05 ** (self.current_wave - 1))
        
        for i in range(num_enemies):
            x_pos = self.SCREEN_WIDTH / (num_enemies + 1) * (i + 1)
            self.enemies.append({
                'pos': pygame.Vector2(x_pos, self.ENEMY_SPAWN_Y + random.uniform(-20, 20)),
                'hp': base_hp,
                'max_hp': base_hp,
                'attack': base_attack,
                'speed': random.uniform(2, 4),
                'slowed': 0
            })

    def _update_unlocked_recipes(self):
        newly_unlocked = False
        new_list = []
        for name, data in self.potions_data.items():
            if self.current_wave >= data["unlock_wave"]:
                if name not in self.unlocked_recipes:
                    newly_unlocked = True
                new_list.append(name)
        self.unlocked_recipes = sorted(new_list)
        return newly_unlocked

    def _check_termination(self):
        return self.shop_health <= 0 or self.current_wave > self.MAX_WAVES or self.steps >= self.MAX_STEPS

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_elements()
        self._render_enemies()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_elements(self):
        # Shop counter
        pygame.draw.rect(self.screen, (41, 31, 51), (0, self.SHOP_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.SHOP_Y))
        pygame.draw.rect(self.screen, (61, 51, 71), (0, self.SHOP_Y, self.SCREEN_WIDTH, 5))
        # Brewing cauldron
        pygame.draw.circle(self.screen, (30, 30, 30), (140, 290), 45)
        pygame.draw.circle(self.screen, (50, 50, 50), (140, 290), 45, 5)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_x, pos_y = int(enemy['pos'].x), int(enemy['pos'].y)
            color = (180, 20, 20) if enemy['slowed'] <= 0 else (100, 100, 220)
            
            # Simple triangle shape for enemy
            points = [(pos_x, pos_y - 12), (pos_x - 10, pos_y + 8), (pos_x + 10, pos_y + 8)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            
            # Health bar
            hp_ratio = max(0, enemy['hp'] / enemy['max_hp'])
            bar_w = 30
            bar_h = 4
            bar_x = pos_x - bar_w / 2
            bar_y = pos_y - 25
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, bar_w * hp_ratio, bar_h))

    def _render_ui(self):
        # --- Panels ---
        # Brew Panel
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, 250, 180), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, (10, 10, 250, 180), 2, border_radius=5)
        brew_title = self.font_m.render("Brewing", True, self.COLOR_TEXT)
        self.screen.blit(brew_title, (20, 18))

        # Use Panel
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.SCREEN_WIDTH - 260, 10, 250, 180), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, (self.SCREEN_WIDTH - 260, 10, 250, 180), 2, border_radius=5)
        use_title = self.font_m.render("Potions", True, self.COLOR_TEXT)
        self.screen.blit(use_title, (self.SCREEN_WIDTH - 250, 18))
        
        # Ingredients Panel
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (270, 10, 100, 180), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, (270, 10, 100, 180), 2, border_radius=5)
        ing_title = self.font_m.render("Stock", True, self.COLOR_TEXT)
        self.screen.blit(ing_title, (280, 18))

        # --- Content ---
        # Ingredients
        for i, (name, count) in enumerate(self.ingredients.items()):
            y_pos = 50 + i * 35
            pygame.draw.circle(self.screen, self.INGREDIENT_COLORS[name], (290, y_pos), 7)
            text = self.font_s.render(f"{count}", True, self.COLOR_TEXT)
            self.screen.blit(text, (305, y_pos - 8))
        
        # Brewable Recipes
        for i, name in enumerate(self.unlocked_recipes):
            y_pos = 50 + i * 25
            is_selected = self.ui_focus == "brew" and i == self.brew_selection_idx
            
            if is_selected:
                pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, (15, y_pos - 4, 240, 22), 1, border_radius=3)
            
            color = self.POTION_COLORS[name]
            text = self.font_s.render(name, True, color if not is_selected else self.COLOR_HIGHLIGHT)
            self.screen.blit(text, (25, y_pos))
            
            recipe = self.potions_data[name]["recipe"]
            recipe_str = ", ".join([f"{c} {n.split(' ')[0]}" for n, c in recipe.items()])
            recipe_text = self.font_s.render(recipe_str, True, self.COLOR_TEXT_DIM)
            self.screen.blit(recipe_text, (130, y_pos))

        # Usable Potions
        available_potions = {p: c for p, c in self.potions.items() if c > 0}
        for i, (name, count) in enumerate(available_potions.items()):
            y_pos = 50 + i * 25
            is_selected = self.ui_focus == "use" and i == self.use_selection_idx
            
            if is_selected:
                pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, (self.SCREEN_WIDTH - 255, y_pos - 4, 240, 22), 1, border_radius=3)
            
            color = self.POTION_COLORS[name]
            text = self.font_s.render(f"{name} (x{count})", True, color if not is_selected else self.COLOR_HIGHLIGHT)
            self.screen.blit(text, (self.SCREEN_WIDTH - 245, y_pos))
            
        # --- Top Bar ---
        # Shop Health
        hp_ratio = max(0, self.shop_health / self.SHOP_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (self.SCREEN_WIDTH / 2 - 100, self.SHOP_Y + 20, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (self.SCREEN_WIDTH / 2 - 100, self.SHOP_Y + 20, 200 * hp_ratio, 20))
        hp_text = self.font_m.render(f"Shop Health: {int(self.shop_health)}/{self.SHOP_MAX_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(hp_text, (self.SCREEN_WIDTH / 2 - hp_text.get_width() / 2, self.SHOP_Y + 21))
        
        # Wave & Score
        wave_text = self.font_l.render(f"Wave: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH/2 - wave_text.get_width()/2, self.SCREEN_HEIGHT - 35))
        
        # Messages
        for i, msg in enumerate(self.ui_messages):
            msg_surf = self.font_s.render(msg, True, self.COLOR_TEXT)
            msg_surf.set_alpha(255 - i * 85)
            self.screen.blit(msg_surf, (self.SCREEN_WIDTH/2 - msg_surf.get_width()/2, 200 + i * 20))

    # --- Particles & Effects ---

    def _create_particles(self, pos, count, color, min_vel=0.5, max_vel=2.5, min_life=20, max_life=40):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_vel, max_vel)
            velocity = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': velocity,
                'radius': random.uniform(2, 5),
                'lifespan': random.randint(min_life, max_life),
                'color': color
            })
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['lifespan'] -= 1
            p['radius'] -= 0.05
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 20))))
            # Create a new color tuple with alpha if it's not already there
            if len(p['color']) == 3:
                color = p['color'] + (alpha,)
            else:
                color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'].x), int(p['pos'].y), 
                max(0, int(p['radius'])), color
            )

    # --- Gymnasium Interface Helpers ---

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "shop_health": self.shop_health}

    def render(self):
        # This is not used by the agent but can be for human playing
        pass

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # Unset the dummy video driver if running locally for display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup for human play
    pygame.display.set_caption("Potion Shop Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input to Action ---
        movement = 0 # none
        space = 0
        shift = 0
        
        # Check for key presses once per frame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4

        # Handle held keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait for 'R' to reset or QUIT
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
        
        clock.tick(10) # Run at a slower pace for turn-based game
        
    env.close()