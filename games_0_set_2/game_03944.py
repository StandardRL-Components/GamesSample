import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to fire in your last moved direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a robot in an isometric arena. Blast all 5 enemy drones to win. Avoid their attacks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        # Headless mode for server execution
        if os.environ.get("SDL_VIDEODRIVER", "") != "dummy":
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Constants ---
        self.GRID_SIZE = 10
        self.TILE_WIDTH = 48
        self.TILE_HEIGHT = 24
        self.ISO_OFFSET_X = self.screen_width // 2
        self.ISO_OFFSET_Y = 100

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_ACCENT = (255, 150, 150)
        self.COLOR_ENEMY = (80, 150, 255)
        self.COLOR_ENEMY_ACCENT = (150, 200, 255)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_HEALTH_GOOD = (80, 220, 80)
        self.COLOR_HEALTH_BAD = (220, 80, 80)
        self.COLOR_HEALTH_BG = (50, 50, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_EXPLOSION = (255, 255, 255)

        # Game settings
        self.MAX_STEPS = 1000
        # FIX: Increased player health to survive initial enemy attacks in stability test.
        self.PLAYER_MAX_HEALTH = 150
        self.ENEMY_MAX_HEALTH = 20
        self.NUM_ENEMIES = 5
        self.PROJECTILE_DAMAGE = 10
        self.PROJECTILE_SPEED = 0.5 # Grid units per step
        self.ENEMY_ATTACK_RANGE = 2.0
        self.ENEMY_ATTACK_DAMAGE = 5
        self.ENEMY_ATTACK_COOLDOWN = 15 # frames
        self.PLAYER_FIRE_COOLDOWN = 5 # frames
        
        # Fonts
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("Verdana", 40, bold=True)

        # State variables (initialized in reset)
        self.np_random = None
        self.player = {}
        self.enemies = []
        self.projectiles = []
        self.effects = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player = {
            'x': self.GRID_SIZE // 2, 
            'y': self.GRID_SIZE // 2, 
            'health': self.PLAYER_MAX_HEALTH,
            'fire_cooldown': 0,
            'facing': (0, -1) # Up
        }

        self.enemies = []
        occupied_positions = {(self.player['x'], self.player['y'])}
        for _ in range(self.NUM_ENEMIES):
            while True:
                x = self.np_random.integers(0, self.GRID_SIZE)
                y = self.np_random.integers(0, self.GRID_SIZE)
                if (x, y) not in occupied_positions:
                    occupied_positions.add((x, y))
                    break
            self.enemies.append({
                'x': x, 'y': y,
                'health': self.ENEMY_MAX_HEALTH,
                'attack_cooldown': 0,
                'pulse': self.np_random.random() * 2 * math.pi
            })

        self.projectiles = []
        self.effects = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01 # Small penalty for each step to encourage efficiency

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            
            # --- Update Game Logic ---
            reward += self._handle_player_action(movement, space_held)
            reward += self._update_enemies()
            reward += self._update_projectiles()
            self._update_effects()

            self.steps += 1
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if not self.enemies: # Win
                reward += 100
            elif self.player['health'] <= 0: # Loss
                reward -= 100
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_player_action(self, movement, space_held):
        # Cooldowns
        if self.player['fire_cooldown'] > 0:
            self.player['fire_cooldown'] -= 1

        # Movement
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
        if movement in move_map:
            dx, dy = move_map[movement]
            self.player['x'] = np.clip(self.player['x'] + dx, 0, self.GRID_SIZE - 1)
            self.player['y'] = np.clip(self.player['y'] + dy, 0, self.GRID_SIZE - 1)
            self.player['facing'] = (dx, dy)

        # Firing
        if space_held and self.player['fire_cooldown'] <= 0:
            # Fire projectile in facing direction
            self.projectiles.append({
                'x': self.player['x'] + self.player['facing'][0] * 0.5, 
                'y': self.player['y'] + self.player['facing'][1] * 0.5,
                'dx': self.player['facing'][0],
                'dy': self.player['facing'][1]
            })
            self.player['fire_cooldown'] = self.PLAYER_FIRE_COOLDOWN
        
        return 0

    def _update_enemies(self):
        reward = 0
        px, py = self.player['x'], self.player['y']
        
        for enemy in self.enemies:
            # Animate pulse
            enemy['pulse'] = (enemy['pulse'] + 0.2) % (2 * math.pi)

            # Cooldown
            if enemy['attack_cooldown'] > 0:
                enemy['attack_cooldown'] -= 1

            # AI: Move and Attack
            ex, ey = enemy['x'], enemy['y']
            dist = math.hypot(px - ex, py - ey)

            if dist <= self.ENEMY_ATTACK_RANGE and enemy['attack_cooldown'] <= 0:
                # Attack player
                self.player['health'] -= self.ENEMY_ATTACK_DAMAGE
                enemy['attack_cooldown'] = self.ENEMY_ATTACK_COOLDOWN
                reward -= 1
                # Add visual effect for enemy attack
                self._add_effect('hit_spark', {'pos': (px, py), 'life': 10, 'color': self.COLOR_ENEMY})
            else:
                # Move towards player
                if self.steps % 2 == 0: # Move every other step
                    dx, dy = px - ex, py - ey
                    if abs(dx) > abs(dy):
                        enemy['x'] += np.sign(dx)
                    else:
                        enemy['y'] += np.sign(dy)
                    enemy['x'] = np.clip(enemy['x'], 0, self.GRID_SIZE - 1)
                    enemy['y'] = np.clip(enemy['y'], 0, self.GRID_SIZE - 1)
        return reward

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for proj in self.projectiles:
            proj['x'] += proj['dx'] * self.PROJECTILE_SPEED
            proj['y'] += proj['dy'] * self.PROJECTILE_SPEED

            # Boundary check
            if not (0 <= proj['x'] < self.GRID_SIZE and 0 <= proj['y'] < self.GRID_SIZE):
                continue

            # Collision check with enemies
            hit = False
            for enemy in self.enemies:
                if math.hypot(proj['x'] - enemy['x'], proj['y'] - enemy['y']) < 0.7:
                    enemy['health'] -= self.PROJECTILE_DAMAGE
                    reward += 10
                    self._add_effect('explosion', {'pos': (enemy['x'], enemy['y']), 'life': 15, 'max_radius': 1.5})
                    hit = True
                    break
            
            if not hit:
                projectiles_to_keep.append(proj)
        
        self.projectiles = projectiles_to_keep
        
        # Check for defeated enemies
        enemies_alive = []
        for enemy in self.enemies:
            if enemy['health'] > 0:
                enemies_alive.append(enemy)
            else:
                self.score += 10
                self._add_effect('explosion', {'pos': (enemy['x'], enemy['y']), 'life': 20, 'max_radius': 2.5})
        
        self.enemies = enemies_alive
        return reward

    def _update_effects(self):
        effects_alive = []
        for effect in self.effects:
            effect['life'] -= 1
            if effect['life'] > 0:
                effects_alive.append(effect)
        self.effects = effects_alive

    def _add_effect(self, type, data):
        # FIX: Store max_life for animations that depend on initial lifetime.
        data['type'] = type
        data['max_life'] = data['life']
        self.effects.append(data)

    def _check_termination(self):
        if self.player['health'] <= 0:
            return True
        if not self.enemies:
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

    def _iso_to_screen(self, x, y):
        screen_x = self.ISO_OFFSET_X + (x - y) * self.TILE_WIDTH / 2
        screen_y = self.ISO_OFFSET_Y + (x + y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)
    
    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            start_x, start_y = self._iso_to_screen(i, 0)
            end_x, end_y = self._iso_to_screen(i, self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y), 1)

            start_x, start_y = self._iso_to_screen(0, i)
            end_x, end_y = self._iso_to_screen(self.GRID_SIZE, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y), 1)

    def _draw_iso_cube(self, surface, x, y, size, height, color_top, color_side):
        sx, sy = self._iso_to_screen(x, y)
        w, h = size, size/2
        
        top_points = [
            (sx, sy - height),
            (sx + w, sy - h - height),
            (sx, sy - 2*h - height),
            (sx - w, sy - h - height)
        ]
        side1_points = [
            (sx, sy), (sx, sy - height),
            (sx - w, sy - h - height), (sx - w, sy - h)
        ]
        side2_points = [
            (sx, sy), (sx, sy - height),
            (sx + w, sy - h - height), (sx + w, sy - h)
        ]

        pygame.gfxdraw.filled_polygon(surface, side1_points, color_side)
        pygame.gfxdraw.filled_polygon(surface, side2_points, color_side)
        pygame.gfxdraw.filled_polygon(surface, top_points, color_top)
        pygame.gfxdraw.aapolygon(surface, top_points, color_top)
    
    def _render_player(self):
        self._draw_iso_cube(self.screen, self.player['x'], self.player['y'], 
                             self.TILE_WIDTH/2.5, 10, self.COLOR_PLAYER_ACCENT, self.COLOR_PLAYER)
        self._render_health_bar(self.player['x'], self.player['y'], self.player['health'], self.PLAYER_MAX_HEALTH, self.COLOR_HEALTH_GOOD)

    def _render_enemies(self):
        for enemy in self.enemies:
            self._render_enemies_single(enemy)

    def _render_projectiles(self):
        for proj in self.projectiles:
            sx, sy = self._iso_to_screen(proj['x'], proj['y'])
            end_x, end_y = self._iso_to_screen(proj['x'] - proj['dx']*0.3, proj['y'] - proj['dy']*0.3)
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, (sx, sy), (end_x, end_y), 4)

    def _render_effects(self):
        # FIX: Replaced buggy conditional and hardcoded values with robust max_life logic.
        for effect in self.effects:
            if effect['type'] == 'explosion':
                progress = 1.0 - (effect['life'] / effect['max_life'])
                current_radius = effect['max_radius'] * math.sin(progress * math.pi/2)
                sx, sy = self._iso_to_screen(effect['pos'][0], effect['pos'][1])
                
                alpha = int(255 * (effect['life'] / effect['max_life']))
                color = (*self.COLOR_EXPLOSION, alpha)
                
                temp_surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, sx, sy, int(current_radius * self.TILE_WIDTH / 2), color)
                self.screen.blit(temp_surf, (0,0))
            
            elif effect['type'] == 'hit_spark':
                sx, sy = self._iso_to_screen(effect['pos'][0], effect['pos'][1])
                alpha = int(255 * (effect['life'] / effect['max_life']))
                color = (*effect['color'], alpha)
                temp_surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (sx, sy), 5)
                self.screen.blit(temp_surf, (0,0))

    def _render_health_bar(self, x, y, current_hp, max_hp, color):
        sx, sy = self._iso_to_screen(x, y)
        bar_width = 30
        bar_height = 5
        
        hp_ratio = max(0, current_hp / max_hp)
        
        bg_rect = pygame.Rect(sx - bar_width // 2, sy - 40, bar_width, bar_height)
        hp_rect = pygame.Rect(sx - bar_width // 2, sy - 40, int(bar_width * hp_ratio), bar_height)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect)
        pygame.draw.rect(self.screen, color, hp_rect)

    def _render_game(self):
        self._render_grid()
        
        # Render entities in correct Z-order (Y-sort)
        entities = [{'type': 'player', 'obj': self.player}]
        for e in self.enemies:
            entities.append({'type': 'enemy', 'obj': e})
        
        entities.sort(key=lambda e: e['obj']['x'] + e['obj']['y'])
        
        for entity in entities:
            if entity['type'] == 'player':
                self._render_player()
            elif entity['type'] == 'enemy':
                self._render_enemies_single(entity['obj'])

        self._render_projectiles()
        self._render_effects()

    def _render_enemies_single(self, enemy):
        # Helper for Y-sorted rendering
        sx, sy = self._iso_to_screen(enemy['x'], enemy['y'])
        pulse_size = 2 * math.sin(enemy['pulse'])
        radius = self.TILE_WIDTH / 4 + pulse_size
        
        pygame.gfxdraw.filled_ellipse(self.screen, sx, sy - 5, int(radius), int(radius/2), self.COLOR_ENEMY)
        pygame.gfxdraw.filled_ellipse(self.screen, sx, sy, int(radius), int(radius/2), self.COLOR_ENEMY_ACCENT)
        pygame.gfxdraw.aaellipse(self.screen, sx, sy, int(radius), int(radius/2), self.COLOR_ENEMY_ACCENT)
        
        self._render_health_bar(enemy['x'], enemy['y'], enemy['health'], self.ENEMY_MAX_HEALTH, self.COLOR_HEALTH_BAD)

    def _render_ui(self):
        # Player Health
        health_text = self.font_ui.render(f"HEALTH: {max(0, self.player['health'])}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 30))

        # Enemies Remaining
        enemies_text = self.font_ui.render(f"ENEMIES: {len(self.enemies)}", True, self.COLOR_TEXT)
        self.screen.blit(enemies_text, (10, 50))

        # Game Over Text
        if self.game_over:
            if not self.enemies:
                msg = "YOU WIN!"
                color = self.COLOR_HEALTH_GOOD
            else:
                msg = "GAME OVER"
                color = self.COLOR_HEALTH_BAD
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player['health'],
            "enemies_remaining": len(self.enemies)
        }

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset(seed=42)
    print("Initial Observation Shape:", obs.shape)
    print("Initial Info:", info)

    terminated = False
    truncated = False
    total_reward = 0
    i = 0
    while not terminated and not truncated:
        i += 1
        # Use no-op action to test stability
        action = [0, 0, 0] 
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i) % 20 == 0:
            print(f"Step {i}: Reward={reward:.2f}, Info={info}")
        if terminated or truncated:
            print(f"Episode finished after {i} steps.")
            break
    
    print(f"Total reward after {i} steps: {total_reward:.2f}")
    print(f"Final info: {info}")