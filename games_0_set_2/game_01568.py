
# Generated: 2025-08-27T17:33:02.635863
# Source Brief: brief_01568.md
# Brief Index: 1568

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Shift to cycle tower types. Press Space to build a tower."
    )

    game_description = (
        "Defend your base from waves of enemies in this isometric tower defense game. "
        "Place towers strategically to survive all 20 waves."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 12
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 24, 12
    ISO_OFFSET_X = SCREEN_WIDTH // 2
    ISO_OFFSET_Y = 80

    MAX_STEPS = 5000
    MAX_WAVES = 20
    INITIAL_BASE_HEALTH = 100
    INITIAL_GOLD = 150

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 44, 52)
    COLOR_PATH = (60, 65, 75)
    COLOR_PATH_BORDER = (80, 85, 95)
    COLOR_BASE = (0, 150, 255)
    COLOR_BASE_DAMAGED = (255, 100, 100)
    COLOR_ENEMY = (255, 70, 70)
    COLOR_ENEMY_OUTLINE = (255, 150, 150)
    COLOR_CURSOR_VALID = (80, 250, 123, 150)
    COLOR_CURSOR_INVALID = (255, 80, 80, 150)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BG = (35, 38, 46, 200)
    COLOR_GOLD = (255, 215, 0)
    
    TOWER_SPECS = {
        "Cannon": {
            "cost": 50, "range": 3.5, "damage": 10, "fire_rate": 45, 
            "color": (52, 152, 219), "proj_color": (100, 200, 255), "proj_speed": 8
        },
        "Missile": {
            "cost": 100, "range": 5.0, "damage": 35, "fire_rate": 90, 
            "color": (231, 76, 60), "proj_color": (255, 150, 100), "proj_speed": 6
        }
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        self.path_grid_coords = self._generate_path()
        self.buildable_tiles = self._get_buildable_tiles()
        self.tower_types = list(self.TOWER_SPECS.keys())

        self.reset()
        
        # This can be commented out for performance but is good for development
        self.validate_implementation()

    def _generate_path(self):
        # A fixed, winding path for enemies
        path = []
        for x in range(-1, 8): path.append((x, 3))
        for y in range(4, 8): path.append((7, y))
        for x in range(6, 15): path.append((x, 7))
        for y in range(6, 2, -1): path.append((14, y))
        for x in range(15, self.GRID_WIDTH + 1): path.append((x, 3))
        return path

    def _get_buildable_tiles(self):
        buildable = set()
        path_set = set(self.path_grid_coords)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) not in path_set:
                    buildable.add((x, y))
        return buildable

    def _grid_to_iso(self, x, y):
        iso_x = self.ISO_OFFSET_X + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.ISO_OFFSET_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.base_health = self.INITIAL_BASE_HEALTH
        self.gold = self.INITIAL_GOLD
        self.wave_number = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.wave_in_progress = False
        self.time_to_next_spawn = 0
        self.enemies_to_spawn_in_wave = 0
        self.wave_timer = 150 # Time between waves

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Player Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Cursor movement
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Cycle tower type (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.tower_types)
            # sfx: UI_cycle

        # Place tower (on press)
        if space_held and not self.last_space_held:
            reward += self._place_tower()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Update Game Logic ---
        self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        # --- Wave Management ---
        if not self.wave_in_progress:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_new_wave()
        else:
            self._update_wave_spawning()
            if self.enemies_to_spawn_in_wave == 0 and not self.enemies:
                self.wave_in_progress = False
                self.wave_timer = 240 # Longer pause after a wave
                reward += 1.0
                self.score += 10
                # sfx: wave_complete

        # --- Termination Checks ---
        terminated = False
        if self.base_health <= 0:
            self.game_over = True
            terminated = True
            reward -= 100.0
        elif self.wave_number > self.MAX_WAVES:
            self.game_won = True
            terminated = True
            reward += 100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_tower(self):
        pos_tuple = tuple(self.cursor_pos)
        spec = self.TOWER_SPECS[self.tower_types[self.selected_tower_type_idx]]
        
        is_buildable = pos_tuple in self.buildable_tiles
        is_occupied = any(t['pos'] == pos_tuple for t in self.towers)
        can_afford = self.gold >= spec['cost']

        if is_buildable and not is_occupied and can_afford:
            self.gold -= spec['cost']
            self.towers.append({
                "pos": pos_tuple,
                "type": self.tower_types[self.selected_tower_type_idx],
                "cooldown": 0,
                "spec": spec
            })
            # sfx: build_tower
            return 0.05 # Small reward for building
        else:
            # sfx: build_fail
            return -0.01 # Small penalty for failed attempt

    def _start_new_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return

        self.wave_in_progress = True
        self.enemies_to_spawn_in_wave = 2 + self.wave_number
        self.time_to_next_spawn = 0
        
    def _update_wave_spawning(self):
        if self.enemies_to_spawn_in_wave > 0:
            self.time_to_next_spawn -= 1
            if self.time_to_next_spawn <= 0:
                self.time_to_next_spawn = 30 # Spawn an enemy every second at 30fps
                self.enemies_to_spawn_in_wave -= 1
                
                start_pos = self.path_grid_coords[0]
                start_iso = self._grid_to_iso(start_pos[0] - 0.5, start_pos[1] - 0.5)

                scale_factor = 1 + (self.wave_number - 1) * 0.05
                self.enemies.append({
                    "pos": list(start_iso),
                    "path_idx": 1,
                    "health": 20 * scale_factor,
                    "max_health": 20 * scale_factor,
                    "speed": 0.7 * scale_factor,
                    "value": 5,
                    "id": self.np_random.integers(1, 1e9)
                })
                # sfx: enemy_spawn

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            tower_iso_pos = self._grid_to_iso(*tower['pos'])
            target = None
            min_dist = float('inf')

            for enemy in self.enemies:
                dist = math.hypot(enemy['pos'][0] - tower_iso_pos[0], enemy['pos'][1] - tower_iso_pos[1])
                if dist < tower['spec']['range'] * self.TILE_WIDTH_HALF * 1.5:
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy
            
            if target:
                tower['cooldown'] = tower['spec']['fire_rate']
                self.projectiles.append({
                    "start_pos": list(tower_iso_pos),
                    "pos": list(tower_iso_pos),
                    "target_id": target['id'],
                    "spec": tower['spec']
                })
                # sfx: tower_fire

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for proj in self.projectiles:
            target = next((e for e in self.enemies if e['id'] == proj['target_id']), None)
            
            if not target:
                continue # Target is gone, projectile fizzles

            target_pos = target['pos']
            dx = target_pos[0] - proj['pos'][0]
            dy = target_pos[1] - proj['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < proj['spec']['proj_speed']:
                # Hit
                target['health'] -= proj['spec']['damage']
                self._create_particles(target['pos'], self.COLOR_ENEMY_OUTLINE, 5)
                # sfx: projectile_hit
                if target['health'] <= 0:
                    reward += 0.1
                    self.gold += target['value']
                    self.score += target['value']
                    self._create_particles(target['pos'], self.COLOR_GOLD, 15)
                    self.enemies = [e for e in self.enemies if e['id'] != target['id']]
                    # sfx: enemy_die
                continue # Projectile is consumed

            # Move projectile towards target
            proj['pos'][0] += (dx / dist) * proj['spec']['proj_speed']
            proj['pos'][1] += (dy / dist) * proj['spec']['proj_speed']
            projectiles_to_keep.append(proj)
        
        self.projectiles = projectiles_to_keep
        return reward

    def _update_enemies(self):
        reward = 0
        enemies_to_keep = []
        for enemy in self.enemies:
            if enemy['path_idx'] >= len(self.path_grid_coords):
                # Reached the base
                self.base_health -= 10
                reward -= 0.1 * 10
                self.score -= 10
                self._create_particles(enemy['pos'], self.COLOR_BASE_DAMAGED, 20)
                # sfx: base_damage
                continue # Enemy is removed

            target_grid_pos = self.path_grid_coords[enemy['path_idx']]
            target_iso_pos = self._grid_to_iso(target_grid_pos[0], target_grid_pos[1])
            
            dx = target_iso_pos[0] - enemy['pos'][0]
            dy = target_iso_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < enemy['speed']:
                enemy['path_idx'] += 1
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']
            
            enemies_to_keep.append(enemy)
        self.enemies = enemies_to_keep
        return reward

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": self.np_random.integers(15, 30),
                "color": color,
                "size": self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and path
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                is_path = (x, y) in self.path_grid_coords
                color = self.COLOR_PATH if is_path else self.COLOR_GRID
                border_color = self.COLOR_PATH_BORDER if is_path else self.COLOR_GRID
                
                p1 = self._grid_to_iso(x, y)
                p2 = self._grid_to_iso(x + 1, y)
                p3 = self._grid_to_iso(x + 1, y + 1)
                p4 = self._grid_to_iso(x, y + 1)
                
                pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3, p4), color)
                pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3, p4), border_color)
        
        # Draw base at the end of the path
        base_pos = self._grid_to_iso(*self.path_grid_coords[-1])
        base_health_ratio = max(0, self.base_health / self.INITIAL_BASE_HEALTH)
        base_color = tuple(np.add(np.multiply(self.COLOR_BASE, base_health_ratio), np.multiply(self.COLOR_BASE_DAMAGED, 1 - base_health_ratio)))
        pygame.gfxdraw.filled_circle(self.screen, base_pos[0], base_pos[1], 12, base_color)
        pygame.gfxdraw.aacircle(self.screen, base_pos[0], base_pos[1], 12, tuple(c*0.8 for c in base_color))


        # Draw towers
        for tower in self.towers:
            iso_pos = self._grid_to_iso(*tower['pos'])
            spec = tower['spec']
            pygame.draw.circle(self.screen, spec['color'], (iso_pos[0], iso_pos[1] - 8), 8)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in spec['color']), (iso_pos[0]-8, iso_pos[1]-8, 16, 10))

        # Draw projectiles
        for proj in self.projectiles:
            pygame.draw.circle(self.screen, proj['spec']['proj_color'], (int(proj['pos'][0]), int(proj['pos'][1])), 3)

        # Draw enemies
        for enemy in sorted(self.enemies, key=lambda e: e['pos'][1]):
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, self.COLOR_ENEMY_OUTLINE)
            # Health bar
            health_ratio = max(0, enemy['health'] / enemy['max_health'])
            bar_width = 12
            bar_y = pos[1] - 12
            pygame.draw.rect(self.screen, (50, 50, 50), (pos[0] - bar_width//2, bar_y, bar_width, 3))
            pygame.draw.rect(self.screen, (100, 255, 100), (pos[0] - bar_width//2, bar_y, int(bar_width * health_ratio), 3))

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['size'] * (p['lifespan']/30.0)))

        # Draw cursor
        cursor_iso_pos = self._grid_to_iso(*self.cursor_pos)
        spec = self.TOWER_SPECS[self.tower_types[self.selected_tower_type_idx]]
        is_buildable = tuple(self.cursor_pos) in self.buildable_tiles
        is_occupied = any(t['pos'] == tuple(self.cursor_pos) for t in self.towers)
        can_afford = self.gold >= spec['cost']
        cursor_color = self.COLOR_CURSOR_VALID if (is_buildable and not is_occupied and can_afford) else self.COLOR_CURSOR_INVALID
        
        # Draw tower footprint and range
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(s, cursor_color, (cursor_iso_pos[0], cursor_iso_pos[1] - 8), 8)
        range_px = spec['range'] * self.TILE_WIDTH_HALF * 1.5
        pygame.gfxdraw.aacircle(s, cursor_iso_pos[0], cursor_iso_pos[1], int(range_px), cursor_color)
        self.screen.blit(s, (0, 0))

    def _render_ui(self):
        # UI Panel
        panel_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 40)
        s = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (0, 0))

        # Base Health
        health_text = self.font_large.render(f"Base: {max(0, self.base_health)}/100", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Gold
        gold_text = self.font_large.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (180, 10))

        # Wave
        wave_str = f"Wave: {self.wave_number}/{self.MAX_WAVES}" if self.wave_number <= self.MAX_WAVES else "All Waves Survived!"
        wave_text = self.font_large.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (320, 10))
        
        if not self.wave_in_progress and not self.game_won and self.wave_number < self.MAX_WAVES:
            next_wave_text = self.font_small.render(f"Next wave in {self.wave_timer // 30 + 1}", True, self.COLOR_UI_TEXT)
            self.screen.blit(next_wave_text, (320, 28))

        # Selected Tower Info
        selected_type = self.tower_types[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[selected_type]
        tower_text = self.font_small.render(
            f"Build: {selected_type} (Cost: {spec['cost']})", True, self.COLOR_UI_TEXT
        )
        self.screen.blit(tower_text, (self.SCREEN_WIDTH - tower_text.get_width() - 10, 5))
        stats_text = self.font_small.render(
            f"Dmg: {spec['damage']} Rng: {spec['range']:.1f}", True, self.COLOR_UI_TEXT
        )
        self.screen.blit(stats_text, (self.SCREEN_WIDTH - stats_text.get_width() - 10, 22))

        # Game Over / Win Text
        if self.game_over:
            text = self.font_game_over.render("GAME OVER", True, self.COLOR_BASE_DAMAGED)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)
        elif self.game_won:
            text = self.font_game_over.render("VICTORY!", True, self.COLOR_GOLD)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.wave_number,
            "base_health": self.base_health,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy' or 'windib' depending on your system

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This loop allows a human to play the game
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    action = env.action_space.sample()
    action = [0, 0, 0] # No-op start
    
    print("\n" + "="*30)
    print("MANUAL PLAY INSTRUCTIONS")
    print(env.user_guide)
    print("="*30 + "\n")
    
    while not done:
        # Get user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # None
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
            
        # Space and Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Gold: {info['gold']}, Wave: {info['wave']}")
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control frame rate
        env.clock.tick(30)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()