import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move placement cursor. Space to place Melee Unit. Shift to place Ranged Unit."
    )

    game_description = (
        "Defend your castle from waves of enemies by strategically placing defensive units in an isometric world."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 50, 60)
    COLOR_CURSOR = (255, 255, 0, 150)
    
    COLOR_PLAYER_CASTLE = (120, 140, 160)
    COLOR_PLAYER_UNIT_MELEE = (0, 200, 100)
    COLOR_PLAYER_UNIT_RANGED = (0, 150, 255)
    COLOR_PLAYER_PROJECTILE = (100, 200, 255)
    
    COLOR_ENEMY_CASTLE = (80, 70, 70)
    COLOR_ENEMY_UNIT = (255, 50, 50)
    
    COLOR_HEALTH_BAR_BG = (80, 80, 80)
    COLOR_HEALTH_BAR_PLAYER = (0, 255, 0)
    COLOR_HEALTH_BAR_ENEMY = (255, 0, 0)
    
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_INSUFFICIENT = (255, 100, 100)

    # Screen and Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE_X, GRID_SIZE_Y = 12, 12
    TILE_WIDTH, TILE_HEIGHT = 24, 12
    ISO_OFFSET_X = SCREEN_WIDTH // 2
    ISO_OFFSET_Y = 100

    # Game Mechanics
    MAX_STEPS = 3000 # Increased for longer games
    INITIAL_RESOURCES = 200
    RESOURCE_GAIN_RATE = 0.2  # Resources per step
    WAVE_INTERVAL = 600  # Steps between waves

    PLAYER_CASTLE_HEALTH = 1000
    ENEMY_CASTLE_HEALTH = 1000

    UNIT_SPECS = {
        "melee": {"cost": 50, "hp": 100, "damage": 8, "range": 1.5, "atk_speed": 30, "color": COLOR_PLAYER_UNIT_MELEE},
        "ranged": {"cost": 80, "hp": 60, "damage": 10, "range": 5, "atk_speed": 45, "color": COLOR_PLAYER_UNIT_RANGED},
    }
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_castle_health = 0
        self.enemy_castle_health = 0
        self.resources = 0
        self.wave_number = 0
        self.wave_timer = 0
        self.cursor_pos = (0, 0)
        self.player_units = []
        self.enemy_units = []
        self.projectiles = []
        self.particles = []
        self.occupied_cells = set()
        self.insufficient_funds_timer = 0
        
        # self.reset() # Not needed as it's called by wrappers

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_castle_health = self.PLAYER_CASTLE_HEALTH
        self.enemy_castle_health = self.ENEMY_CASTLE_HEALTH
        
        self.resources = self.INITIAL_RESOURCES
        self.wave_number = 0
        self.wave_timer = self.WAVE_INTERVAL // 2 # First wave comes sooner
        
        self.cursor_pos = (self.GRID_SIZE_X // 2, self.GRID_SIZE_Y - 2)
        
        self.player_units = []
        self.enemy_units = []
        self.projectiles = []
        self.particles = []
        self.occupied_cells = set()
        
        self.insufficient_funds_timer = 0

        # Player castle occupies the bottom center, enemy castle top center
        self.player_castle_pos = (self.GRID_SIZE_X // 2, self.GRID_SIZE_Y - 1)
        self.enemy_castle_pos = (self.GRID_SIZE_X // 2, 0)
        self.occupied_cells.add(self.player_castle_pos)
        self.occupied_cells.add(self.enemy_castle_pos)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        step_reward = 0
        self.steps += 1
        self.resources += self.RESOURCE_GAIN_RATE
        if self.insufficient_funds_timer > 0:
            self.insufficient_funds_timer -= 1

        # 1. Handle Player Input
        self._handle_input(movement, space_held, shift_held)

        # 2. Update Wave Logic
        self._update_waves()

        # 3. Update Game Entities
        damage_events = self._update_units_and_combat()
        self._update_projectiles()
        self._update_particles()

        # 4. Process Damage and Rewards
        for event in damage_events:
            if event['target_type'] == 'enemy_unit':
                step_reward += 0.1
            elif event['target_type'] == 'player_castle':
                step_reward -= 0.01 * event['damage']
        
        # 5. Clean up dead units and apply rewards
        destroyed_enemy_reward = self._cleanup_dead_units()
        step_reward += destroyed_enemy_reward
        self.score += step_reward

        # 6. Check Termination Conditions
        terminated = False
        if self.player_castle_health <= 0:
            terminated = True
            step_reward -= 50
            self.score -= 50
            self._create_explosion(self.player_castle_pos, 50, self.COLOR_PLAYER_CASTLE)
        elif self.enemy_castle_health <= 0:
            terminated = True
            step_reward += 100
            self.score += 100
            self._create_explosion(self.enemy_castle_pos, 50, self.COLOR_ENEMY_CASTLE)
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        
        return self._get_observation(), step_reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        cx, cy = self.cursor_pos
        if movement == 1: cy -= 1  # Up
        if movement == 2: cy += 1  # Down
        if movement == 3: cx -= 1  # Left
        if movement == 4: cx += 1  # Right
        self.cursor_pos = (cx % self.GRID_SIZE_X, cy % self.GRID_SIZE_Y)

        # Place units (prioritize shift)
        if shift_held:
            self._place_unit("ranged")
        elif space_held:
            self._place_unit("melee")

    def _place_unit(self, unit_type):
        spec = self.UNIT_SPECS[unit_type]
        if self.resources >= spec["cost"] and self.cursor_pos not in self.occupied_cells:
            self.resources -= spec["cost"]
            self.occupied_cells.add(self.cursor_pos)
            new_unit = {
                "pos": self.cursor_pos,
                "type": unit_type,
                "hp": spec["hp"],
                "max_hp": spec["hp"],
                "attack_cooldown": 0,
                "target": None,
                "id": self.np_random.integers(1, 1e9)
            }
            self.player_units.append(new_unit)
            self._create_explosion(self.cursor_pos, 10, spec["color"])
        elif self.resources < spec["cost"]:
            self.insufficient_funds_timer = 15 # Flash for 0.5s at 30fps

    def _update_waves(self):
        self.wave_timer -= 1
        if self.wave_timer <= 0:
            self.wave_timer = self.WAVE_INTERVAL
            self.wave_number += 1
            
            num_enemies = 2 + self.wave_number
            enemy_hp = 50 * (1.05 ** (self.wave_number - 1))
            enemy_damage = 5 * (1.05 ** (self.wave_number - 1))
            
            for _ in range(num_enemies):
                spawn_x = self.np_random.integers(0, self.GRID_SIZE_X)
                spawn_pos = (spawn_x, 0)
                
                # Avoid spawning on top of each other
                while spawn_pos in [u['pos'] for u in self.enemy_units]:
                    spawn_x = self.np_random.integers(0, self.GRID_SIZE_X)
                    spawn_pos = (spawn_x, 0)

                self.enemy_units.append({
                    "pos": spawn_pos,
                    "hp": enemy_hp,
                    "max_hp": enemy_hp,
                    "damage": enemy_damage,
                    "attack_cooldown": 0,
                    "target": None,
                    "id": self.np_random.integers(1, 1e9)
                })

    def _update_units_and_combat(self):
        damage_events = []

        # Player units
        for unit in self.player_units:
            spec = self.UNIT_SPECS[unit['type']]
            unit['attack_cooldown'] = max(0, unit['attack_cooldown'] - 1)
            
            # Find target
            unit['target'] = self._find_closest_target(unit['pos'], self.enemy_units, spec['range'])
            if unit['target'] is None and self._distance(unit['pos'], self.enemy_castle_pos) <= spec['range']:
                 unit['target'] = {'type': 'castle', 'pos': self.enemy_castle_pos}

            # Attack
            if unit['attack_cooldown'] == 0 and unit['target'] is not None:
                unit['attack_cooldown'] = spec['atk_speed']
                if unit['type'] == 'ranged':
                    self._create_projectile(unit['pos'], unit['target'], spec['damage'], 'player')
                else: # Melee
                    if 'type' in unit['target'] and unit['target']['type'] == 'castle':
                        self.enemy_castle_health -= spec['damage']
                        damage_events.append({'target_type': 'enemy_castle', 'damage': spec['damage']})
                        self._create_hit_effect(unit['target']['pos'], self.COLOR_ENEMY_CASTLE)
                    else:
                        unit['target']['hp'] -= spec['damage']
                        damage_events.append({'target_type': 'enemy_unit', 'damage': spec['damage']})
                        self._create_hit_effect(unit['target']['pos'], self.COLOR_ENEMY_UNIT)

        # Enemy units
        for unit in self.enemy_units:
            unit['attack_cooldown'] = max(0, unit['attack_cooldown'] - 1)
            
            # Find target
            target = self._find_closest_target(unit['pos'], self.player_units, 1.5)
            if target is None:
                target = {'type': 'castle', 'pos': self.player_castle_pos}

            # Move or Attack
            dist_to_target = self._distance(unit['pos'], target['pos'])
            if dist_to_target > 1.1: # Move towards target
                dx, dy = target['pos'][0] - unit['pos'][0], target['pos'][1] - unit['pos'][1]
                if abs(dx) > abs(dy):
                    new_pos = (unit['pos'][0] + np.sign(dx), unit['pos'][1])
                else:
                    new_pos = (unit['pos'][0], unit['pos'][1] + np.sign(dy))
                unit['pos'] = new_pos # Simple grid movement
            elif unit['attack_cooldown'] == 0: # Attack
                unit['attack_cooldown'] = 45 # Enemy attack speed
                if 'type' in target and target['type'] == 'castle':
                    damage = unit['damage']
                    self.player_castle_health -= damage
                    damage_events.append({'target_type': 'player_castle', 'damage': damage})
                    self._create_hit_effect(self.player_castle_pos, self.COLOR_PLAYER_CASTLE)
                else:
                    target['hp'] -= unit['damage']
                    damage_events.append({'target_type': 'player_unit', 'damage': unit['damage']})
                    spec = self.UNIT_SPECS[target['type']]
                    self._create_hit_effect(target['pos'], spec['color'])

        return damage_events

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            target_pos = self._iso_to_screen(*p['target']['pos']) if p['target'] else p['end_pos']
            
            p['pos'] = (
                p['pos'][0] + p['vel'][0],
                p['pos'][1] + p['vel'][1]
            )
            p['life'] -= 1
            
            dist_to_target = math.hypot(p['pos'][0] - target_pos[0], p['pos'][1] - target_pos[1])

            if dist_to_target < 10 or p['life'] <= 0:
                if p['owner'] == 'player':
                    if 'type' in p['target'] and p['target']['type'] == 'castle':
                        self.enemy_castle_health -= p['damage']
                        self._create_hit_effect(p['target']['pos'], self.COLOR_ENEMY_CASTLE)
                    elif p['target'] and p['target']['hp'] > 0:
                        p['target']['hp'] -= p['damage']
                        self._create_hit_effect(p['target']['pos'], self.COLOR_ENEMY_UNIT)
                self.projectiles.remove(p)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _cleanup_dead_units(self):
        reward = 0
        
        for unit in self.player_units[:]:
            if unit['hp'] <= 0:
                self.occupied_cells.remove(unit['pos'])
                self.player_units.remove(unit)
                spec = self.UNIT_SPECS[unit['type']]
                self._create_explosion(unit['pos'], 20, spec['color'])
        
        for unit in self.enemy_units[:]:
            if unit['hp'] <= 0:
                self.enemy_units.remove(unit)
                self._create_explosion(unit['pos'], 20, self.COLOR_ENEMY_UNIT)
                reward += 1 # Reward for destroying an enemy
        
        return reward

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
            "player_castle_health": self.player_castle_health,
            "enemy_castle_health": self.enemy_castle_health,
            "resources": self.resources,
            "wave": self.wave_number,
        }

    def _render_game(self):
        self._draw_grid()
        self._draw_cursor()
        
        # Draw castles
        self._draw_castle(self.player_castle_pos, self.player_castle_health, self.PLAYER_CASTLE_HEALTH, self.COLOR_PLAYER_CASTLE)
        self._draw_castle(self.enemy_castle_pos, self.enemy_castle_health, self.ENEMY_CASTLE_HEALTH, self.COLOR_ENEMY_CASTLE)

        # Draw units (sorted by y-pos for correct layering)
        all_units = self.player_units + self.enemy_units
        sorted_units = sorted(all_units, key=lambda u: u['pos'][0] + u['pos'][1])

        for unit in sorted_units:
            if 'type' in unit: # Player unit
                spec = self.UNIT_SPECS[unit['type']]
                self._draw_unit(unit, spec['color'], unit['type'])
            else: # Enemy unit
                self._draw_unit(unit, self.COLOR_ENEMY_UNIT, 'enemy')

        # Draw projectiles and particles
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, p['color'])
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, p['color'])

        for p in self.particles:
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_ui(self):
        # Resources
        res_text = f"Resources: {int(self.resources)}"
        res_color = self.COLOR_UI_INSUFFICIENT if self.insufficient_funds_timer > 0 else self.COLOR_UI_TEXT
        res_surf = self.font_small.render(res_text, True, res_color)
        self.screen.blit(res_surf, (10, 10))
        
        # Wave
        wave_text = f"Wave: {self.wave_number}"
        wave_surf = self.font_small.render(wave_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_surf, (self.SCREEN_WIDTH - wave_surf.get_width() - 10, 10))

        # Score
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH // 2 - score_surf.get_width() // 2, 10))

        # Game Over
        if self.game_over:
            result_text = "VICTORY" if self.enemy_castle_health <= 0 else "DEFEAT"
            result_color = self.COLOR_HEALTH_BAR_PLAYER if self.enemy_castle_health <= 0 else self.COLOR_ENEMY_UNIT
            result_surf = self.font_large.render(result_text, True, result_color)
            self.screen.blit(result_surf, (self.SCREEN_WIDTH // 2 - result_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - result_surf.get_height() // 2))

    # --- Drawing Helpers ---

    def _iso_to_screen(self, iso_x, iso_y):
        screen_x = self.ISO_OFFSET_X + (iso_x - iso_y) * self.TILE_WIDTH
        screen_y = self.ISO_OFFSET_Y + (iso_x + iso_y) * self.TILE_HEIGHT
        return int(screen_x), int(screen_y)

    def _draw_grid(self):
        for y in range(self.GRID_SIZE_Y + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_SIZE_X, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.GRID_SIZE_X + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_SIZE_Y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

    def _draw_cursor(self):
        x, y = self.cursor_pos
        points = [
            self._iso_to_screen(x, y),
            self._iso_to_screen(x + 1, y),
            self._iso_to_screen(x + 1, y + 1),
            self._iso_to_screen(x, y + 1),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CURSOR)

    def _draw_health_bar(self, screen_pos, hp, max_hp, color):
        bar_width = 30
        bar_height = 4
        x, y = screen_pos[0] - bar_width // 2, screen_pos[1] - self.TILE_HEIGHT * 2.5
        
        fill_ratio = max(0, hp / max_hp)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (x, y, bar_width, bar_height))
        pygame.draw.rect(self.screen, color, (x, y, int(bar_width * fill_ratio), bar_height))

    def _draw_unit(self, unit, color, unit_type):
        x, y = self._iso_to_screen(*unit['pos'])
        
        if unit_type == 'melee':
            size = 8
            points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        elif unit_type == 'ranged':
            size = 7
            pygame.gfxdraw.filled_circle(self.screen, x, y, size, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, size, color)
        elif unit_type == 'enemy':
            size = 9
            points = [(x, y - size), (x + size // 2, y + size // 2), (x - size // 2, y + size // 2)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            
        health_color = self.COLOR_HEALTH_BAR_PLAYER if 'type' in unit else self.COLOR_HEALTH_BAR_ENEMY
        self._draw_health_bar((x, y), unit['hp'], unit['max_hp'], health_color)

    def _draw_castle(self, pos, hp, max_hp, color):
        x, y = self._iso_to_screen(*pos)
        h = self.TILE_HEIGHT * 2
        w = self.TILE_WIDTH
        
        top_points = [(x, y - h), (x + w, y - h + self.TILE_HEIGHT), (x, y - h + self.TILE_HEIGHT * 2), (x - w, y - h + self.TILE_HEIGHT)]
        left_face = [(x - w, y - h + self.TILE_HEIGHT), (x, y - h + self.TILE_HEIGHT*2), (x, y + self.TILE_HEIGHT), (x - w, y)]
        right_face = [(x + w, y - h + self.TILE_HEIGHT), (x, y - h + self.TILE_HEIGHT*2), (x, y + self.TILE_HEIGHT), (x + w, y)]
        
        pygame.gfxdraw.filled_polygon(self.screen, top_points, color)
        pygame.gfxdraw.filled_polygon(self.screen, left_face, tuple(max(0, c-30) for c in color))
        pygame.gfxdraw.filled_polygon(self.screen, right_face, tuple(max(0, c-50) for c in color))

        self._draw_health_bar((x, y), hp, max_hp, self.COLOR_HEALTH_BAR_PLAYER if color == self.COLOR_PLAYER_CASTLE else self.COLOR_HEALTH_BAR_ENEMY)

    # --- Game Logic Helpers ---

    def _distance(self, pos1, pos2):
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

    def _find_closest_target(self, pos, target_list, max_range):
        closest_target = None
        min_dist = float('inf')
        for target in target_list:
            if target['hp'] > 0:
                dist = self._distance(pos, target['pos'])
                if dist <= max_range and dist < min_dist:
                    min_dist = dist
                    closest_target = target
        return closest_target

    def _create_projectile(self, start_pos, target_obj, damage, owner):
        start_screen = self._iso_to_screen(*start_pos)
        end_screen = self._iso_to_screen(*target_obj['pos'])
        
        angle = math.atan2(end_screen[1] - start_screen[1], end_screen[0] - start_screen[0])
        speed = 8
        self.projectiles.append({
            'pos': start_screen,
            'vel': (math.cos(angle) * speed, math.sin(angle) * speed),
            'end_pos': end_screen,
            'target': target_obj,
            'damage': damage,
            'owner': owner,
            'life': 100,
            'color': self.COLOR_PLAYER_PROJECTILE
        })

    def _create_explosion(self, pos, num_particles, color):
        screen_pos = self._iso_to_screen(*pos)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(screen_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _create_hit_effect(self, pos, color):
        screen_pos = self._iso_to_screen(*pos)
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5)
            self.particles.append({
                'pos': list(screen_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(5, 10),
                'max_life': 10,
                'color': (255, 255, 255),
                'size': self.np_random.uniform(1, 3)
            })

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to play the game manually
    # To use, you might need to comment out `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")`
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
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
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause before resetting
            obs, info = env.reset()
            total_reward = 0

    env.close()