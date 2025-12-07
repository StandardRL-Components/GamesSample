import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:02:09.590057
# Source Brief: brief_00167.md
# Brief Index: 167
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Engage in a magical duel, collecting elemental reagents to craft powerful spells. "
        "Survive waves of enemies in a fast-paced arena shooter."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to fire the selected spell "
        "and shift to cycle through available recipes."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 3000 # Increased from 1000 to allow for longer games
    MAX_WAVES = 10

    # Colors
    COLOR_BG = (25, 20, 35)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_RETICLE = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR_BG = (70, 0, 0)
    COLOR_HEALTH_BAR_FG = (255, 0, 0)

    ELEMENTS = {
        'fire': {'color': (255, 80, 0), 'weak_against': 'water'},
        'water': {'color': (0, 150, 255), 'weak_against': 'lightning'},
        'earth': {'color': (100, 220, 50), 'weak_against': 'fire'},
        'lightning': {'color': (255, 255, 0), 'weak_against': 'earth'},
    }
    
    # Player
    PLAYER_SPEED = 5.0
    PLAYER_RADIUS = 12
    PLAYER_MAX_HEALTH = 100
    
    # Enemies
    ENEMY_RADIUS = 15
    ENEMY_BASE_HEALTH = 20
    ENEMY_BASE_SPEED = 1.0
    ENEMY_ATTACK_COOLDOWN = 60 # 2 seconds at 30fps
    
    # Projectiles
    PROJECTILE_SPEED = 8.0
    PROJECTILE_RADIUS = 5
    PROJECTILE_BASE_DAMAGE = 10
    
    # Reagents
    REAGENT_RADIUS = 8
    REAGENT_SPAWN_POINTS = {
        'fire': pygame.Vector2(50, 50),
        'water': pygame.Vector2(SCREEN_WIDTH - 50, 50),
        'earth': pygame.Vector2(50, SCREEN_HEIGHT - 50),
        'lightning': pygame.Vector2(SCREEN_WIDTH - 50, SCREEN_HEIGHT - 50),
    }
    REAGENT_RESPAWN_TIME = 10 * FPS # 10 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 1
        
        self.player_pos = None
        self.player_health = None
        self.player_inventory = None
        self.aim_direction = None
        
        self.enemies = []
        self.projectiles = []
        self.reagents = []
        self.particles = []
        
        self.unlocked_recipes = []
        self.selected_recipe_index = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.all_recipes = [
            {'name': 'Fire', 'cost': {'fire': 1}, 'elements': ['fire']},
            {'name': 'Water', 'cost': {'water': 1}, 'elements': ['water']},
            {'name': 'Earth', 'cost': {'earth': 1}, 'elements': ['earth']},
            {'name': 'Lightning', 'cost': {'lightning': 1}, 'elements': ['lightning']},
            # Unlockable recipes
            {'name': 'Magma', 'cost': {'fire': 1, 'earth': 1}, 'elements': ['fire', 'earth']},
            {'name': 'Storm', 'cost': {'water': 1, 'lightning': 1}, 'elements': ['water', 'lightning']},
            {'name': 'Steam', 'cost': {'fire': 1, 'water': 1}, 'elements': ['fire', 'water']},
        ]
        
        # self.reset() # This line is removed to avoid initialization before it's needed.
        # self.validate_implementation() # This line is removed as it's not part of the standard API.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 1
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_inventory = {el: 0 for el in self.ELEMENTS}
        self.aim_direction = pygame.Vector2(1, 0)
        
        self.enemies = []
        self.projectiles = []
        self.reagents = []
        self.particles = []
        
        self.unlocked_recipes = self.all_recipes[:4]
        self.selected_recipe_index = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self._spawn_reagents()
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0.0
        
        # Handle input and state transitions
        self._handle_input(action)
        
        # Update game objects
        self._update_player()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        self._update_reagents()
        
        # Check collisions and handle game events
        reward += self._handle_collisions()
        
        # Check for wave completion
        if not self.enemies and not self.game_over:
            reward += 100
            self.wave_number += 1
            if self.wave_number > self.MAX_WAVES:
                self.game_over = True # Victory
            else:
                self._spawn_wave()
                if self.wave_number % 3 == 0 and len(self.unlocked_recipes) < len(self.all_recipes):
                    self.unlocked_recipes.append(self.all_recipes[len(self.unlocked_recipes)])

        # Check termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS

        if self.player_health <= 0 and not self.game_over:
            reward = -100.0
            self.game_over = True
            terminated = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
            self.aim_direction = move_vec
        
        # Action: Cycle recipe (on press)
        if shift_action and not self.prev_shift_held:
            self.selected_recipe_index = (self.selected_recipe_index + 1) % len(self.unlocked_recipes)
        
        # Action: Fire projectile (on press)
        if space_action and not self.prev_space_held:
            self._fire_projectile()
            
        self.prev_space_held = space_action
        self.prev_shift_held = shift_action
        
    def _fire_projectile(self):
        recipe = self.unlocked_recipes[self.selected_recipe_index]
        
        # Check if player has enough reagents
        can_craft = all(self.player_inventory[el] >= recipe['cost'][el] for el in recipe['cost'])
        
        if can_craft:
            # Consume reagents
            for el, amount in recipe['cost'].items():
                self.player_inventory[el] -= amount
            
            # Create projectile
            start_pos = self.player_pos + self.aim_direction * (self.PLAYER_RADIUS + 5)
            self.projectiles.append({
                'pos': start_pos.copy(),
                'vel': self.aim_direction.copy() * self.PROJECTILE_SPEED,
                'elements': recipe['elements'],
                'radius': self.PROJECTILE_RADIUS + len(recipe['elements']), # Bigger for combos
                'owner': 'player',
                'trail': []
            })

    def _update_player(self):
        # Clamp player position to screen bounds
        self.player_pos.x = max(self.PLAYER_RADIUS, min(self.player_pos.x, self.SCREEN_WIDTH - self.PLAYER_RADIUS))
        self.player_pos.y = max(self.PLAYER_RADIUS, min(self.player_pos.y, self.SCREEN_HEIGHT - self.PLAYER_RADIUS))

    def _update_enemies(self):
        for enemy in self.enemies:
            # Movement
            direction = (self.player_pos - enemy['pos'])
            if direction.length() > 0:
                direction.normalize_ip()
                enemy['pos'] += direction * enemy['speed']
            
            # Attack
            enemy['attack_cooldown'] -= 1
            if enemy['attack_cooldown'] <= 0:
                enemy['attack_cooldown'] = self.ENEMY_ATTACK_COOLDOWN * (1 + self.np_random.uniform(-0.2, 0.2))
                proj_dir = (self.player_pos - enemy['pos']).normalize()
                start_pos = enemy['pos'] + proj_dir * (enemy['radius'] + 5)
                self.projectiles.append({
                    'pos': start_pos.copy(),
                    'vel': proj_dir * self.PROJECTILE_SPEED * 0.75,
                    'elements': [enemy['element']],
                    'radius': self.PROJECTILE_RADIUS,
                    'owner': 'enemy',
                    'trail': []
                })

    def _update_projectiles(self):
        for p in self.projectiles:
            p['trail'].append(p['pos'].copy())
            if len(p['trail']) > 5:
                p['trail'].pop(0)
            p['pos'] += p['vel']
        
        # Remove off-screen projectiles
        self.projectiles = [p for p in self.projectiles if self.screen.get_rect().collidepoint(p['pos'])]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _update_reagents(self):
        for r in self.reagents:
            if not r['active'] and r['timer'] > 0:
                r['timer'] -= 1
                if r['timer'] <= 0:
                    r['active'] = True

    def _handle_collisions(self):
        reward = 0.0
        
        # Player projectiles vs Enemies
        projectiles_to_remove = set()
        enemies_to_remove = set()

        for p_idx, p in enumerate(self.projectiles):
            if p['owner'] != 'player' or p_idx in projectiles_to_remove: continue
            for e_idx, enemy in enumerate(self.enemies):
                if e_idx in enemies_to_remove: continue
                if p['pos'].distance_to(enemy['pos']) < p['radius'] + enemy['radius']:
                    damage = 0
                    for proj_element in p['elements']:
                        current_damage = self.PROJECTILE_BASE_DAMAGE
                        if self.ELEMENTS[enemy['element']]['weak_against'] == proj_element:
                            current_damage *= 2.0 # Weakness
                        elif self.ELEMENTS[proj_element]['weak_against'] == enemy['element']:
                            current_damage *= 0.5 # Resistance
                        damage += current_damage
                    
                    enemy['health'] -= damage
                    reward += 1.0 # Reward for hitting
                    self._create_explosion(p['pos'], p['elements'])
                    
                    projectiles_to_remove.add(p_idx)
                    
                    if enemy['health'] <= 0:
                        reward += 5.0 # Reward for defeating
                        self._create_explosion(enemy['pos'], [enemy['element']], 30, 2.0)
                        enemies_to_remove.add(e_idx)
                    break
        
        self.enemies = [e for i, e in enumerate(self.enemies) if i not in enemies_to_remove]

        # Enemy projectiles vs Player
        for p_idx, p in enumerate(self.projectiles):
            if p['owner'] != 'enemy' or p_idx in projectiles_to_remove: continue
            if p['pos'].distance_to(self.player_pos) < p['radius'] + self.PLAYER_RADIUS:
                self.player_health -= self.PROJECTILE_BASE_DAMAGE
                self.player_health = max(0, self.player_health)
                reward -= 0.1 # Small penalty for getting hit
                self._create_explosion(p['pos'], p['elements'], 15, 0.5, (255,255,255))
                projectiles_to_remove.add(p_idx)
        
        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]

        # Player vs Reagents
        for r in self.reagents:
            if r['active'] and self.player_pos.distance_to(r['pos']) < self.PLAYER_RADIUS + r['radius']:
                self.player_inventory[r['element']] += 1
                r['active'] = False
                r['timer'] = self.REAGENT_RESPAWN_TIME
                reward += 0.1 # Reward for collecting
                
        return reward

    def _spawn_wave(self):
        num_enemies = min(self.wave_number, 8)
        enemy_health = self.ENEMY_BASE_HEALTH * (1 + (self.wave_number - 1) * 0.1)
        enemy_speed = self.ENEMY_BASE_SPEED * (1 + (self.wave_number - 1) * 0.05)
        
        for _ in range(num_enemies):
            side = self.np_random.integers(4)
            if side == 0: # top
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ENEMY_RADIUS)
            elif side == 1: # bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ENEMY_RADIUS)
            elif side == 2: # left
                pos = pygame.Vector2(-self.ENEMY_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            else: # right
                pos = pygame.Vector2(self.SCREEN_WIDTH + self.ENEMY_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            
            element = self.np_random.choice(list(self.ELEMENTS.keys()))
            
            self.enemies.append({
                'pos': pos,
                'health': enemy_health,
                'max_health': enemy_health,
                'speed': enemy_speed,
                'element': element,
                'radius': self.ENEMY_RADIUS,
                'attack_cooldown': self.np_random.integers(30, self.ENEMY_ATTACK_COOLDOWN)
            })

    def _spawn_reagents(self):
        self.reagents = []
        for el, pos in self.REAGENT_SPAWN_POINTS.items():
            self.reagents.append({
                'pos': pos.copy(),
                'element': el,
                'radius': self.REAGENT_RADIUS,
                'active': True,
                'timer': 0
            })

    def _create_explosion(self, pos, elements, num_particles=20, scale=1.0, force_color=None):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * scale
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            
            if force_color:
                color = force_color
            else:
                element_color = self.ELEMENTS[self.np_random.choice(elements)]['color']
                color = pygame.Color(element_color)
                color.lerp_to((255,255,255), self.np_random.uniform(0, 0.5))

            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.uniform(1, 4) * scale
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number}

    def _render_game(self):
        # Render reagent spawn points (as dark circles)
        for el, pos in self.REAGENT_SPAWN_POINTS.items():
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.REAGENT_RADIUS + 4, (40, 35, 55))
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.REAGENT_RADIUS + 4, self.ELEMENTS[el]['color'])

        # Render active reagents
        for r in self.reagents:
            if r['active']:
                glow_radius = int(r['radius'] * (1.5 + 0.5 * math.sin(self.steps * 0.1)))
                color = self.ELEMENTS[r['element']]['color']
                glow_color = (*color, 50)
                pygame.gfxdraw.filled_circle(self.screen, int(r['pos'].x), int(r['pos'].y), glow_radius, glow_color)
                pygame.gfxdraw.filled_circle(self.screen, int(r['pos'].x), int(r['pos'].y), r['radius'], color)
                pygame.gfxdraw.aacircle(self.screen, int(r['pos'].x), int(r['pos'].y), r['radius'], (255,255,255))
    
        # Render player
        if self.player_health > 0:
            # Glow
            glow_radius = int(self.PLAYER_RADIUS * 2.0)
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), glow_radius, (*self.COLOR_PLAYER_GLOW, 50))
            # Body
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_RADIUS, (200, 255, 255))
            # Aiming reticle
            reticle_pos = self.player_pos + self.aim_direction * 40
            pygame.draw.line(self.screen, self.COLOR_RETICLE, self.player_pos, reticle_pos, 1)
            pygame.gfxdraw.aacircle(self.screen, int(reticle_pos.x), int(reticle_pos.y), 5, self.COLOR_RETICLE)

        # Render enemies
        for enemy in self.enemies:
            color = self.ELEMENTS[enemy['element']]['color']
            # Health bar
            bar_width = int(enemy['radius'] * 2)
            bar_height = 5
            health_pct = max(0, enemy['health'] / enemy['max_health'])
            pygame.draw.rect(self.screen, (50,50,50), (enemy['pos'].x - bar_width/2, enemy['pos'].y - enemy['radius'] - 10, bar_width, bar_height))
            pygame.draw.rect(self.screen, color, (enemy['pos'].x - bar_width/2, enemy['pos'].y - enemy['radius'] - 10, bar_width * health_pct, bar_height))
            # Body
            pygame.gfxdraw.filled_circle(self.screen, int(enemy['pos'].x), int(enemy['pos'].y), enemy['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, int(enemy['pos'].x), int(enemy['pos'].y), enemy['radius'], (255,255,255))

        # Render projectiles
        for p in self.projectiles:
            # Trail
            for i, pos in enumerate(p['trail']):
                alpha = int(255 * (i / len(p['trail'])))
                color = (*self.ELEMENTS[p['elements'][0]]['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(p['radius'] * (i/len(p['trail']))), color)
            # Main body
            main_color = self.ELEMENTS[p['elements'][0]]['color']
            if len(p['elements']) > 1: # Combo projectile visuals
                main_color = (255,255,255) # White core for combos
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), p['radius'], main_color)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), p['radius'], (255,255,255))
            
        # Render particles
        for p in self.particles:
            alpha = p['lifespan'] / 30.0
            color_with_alpha = (*p['color'][:3], int(alpha * 255))
            if p['size'] > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), color_with_alpha)

    def _render_ui(self):
        # Health Bar
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, self.SCREEN_HEIGHT - 30, bar_width, 20))
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, self.SCREEN_HEIGHT - 30, bar_width * health_pct, 20))
        health_text = self.font_small.render(f"{int(self.player_health)} / {self.PLAYER_MAX_HEALTH}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, self.SCREEN_HEIGHT - 28))

        # Reagent Inventory
        inventory_start_x = 230
        for i, (el, color_data) in enumerate(self.ELEMENTS.items()):
            pos_x = inventory_start_x + i * 50
            pygame.gfxdraw.filled_circle(self.screen, pos_x, self.SCREEN_HEIGHT - 20, 12, color_data['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_x, self.SCREEN_HEIGHT - 20, 12, (255,255,255))
            count_text = self.font_large.render(f"{self.player_inventory[el]}", True, self.COLOR_UI_TEXT)
            self.screen.blit(count_text, (pos_x + 18, self.SCREEN_HEIGHT - 30))

        # Selected Recipe
        recipe_start_x = self.SCREEN_WIDTH - 150
        recipe = self.unlocked_recipes[self.selected_recipe_index]
        recipe_text = self.font_small.render(f"Craft: {recipe['name']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(recipe_text, (recipe_start_x, self.SCREEN_HEIGHT - 35))
        for i, el in enumerate(recipe['elements']):
            color = self.ELEMENTS[el]['color']
            pygame.gfxdraw.filled_circle(self.screen, recipe_start_x + 15 + i * 25, self.SCREEN_HEIGHT - 15, 8, color)
            pygame.gfxdraw.aacircle(self.screen, recipe_start_x + 15 + i * 25, self.SCREEN_HEIGHT - 15, 8, (255,255,255))

        # Score and Wave
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        wave_text = self.font_large.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 40))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual play and debugging.
    # It will not be executed by the test suite.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Manual Play Controls ---
    # Arrow Keys: Move
    # Space: Shoot
    # Left Shift: Cycle Recipe
    # Q: Quit
    
    action = [0, 0, 0] # No-op
    
    display_surf = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Elemental Alchemy Duel")

    while not terminated:
        # --- Pygame event handling for manual play ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # none
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
            
        # Space
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift
        action[2] = 1 if keys[pygame.K_LSHIFT] else 0

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render for human viewing ---
        frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(display_surf, frame)
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(GameEnv.FPS)

    env.close()
    print("Game Over. Final Score:", info['score'])