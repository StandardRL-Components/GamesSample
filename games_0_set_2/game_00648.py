
# Generated: 2025-08-27T14:19:49.286387
# Source Brief: brief_00648.md
# Brief Index: 648

        
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
        "Controls: Arrow keys to move cursor. Space to place selected tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from increasingly difficult waves of enemies in this isometric tower defense game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Game Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PATH = (50, 60, 80)
    COLOR_GRID = (30, 35, 50)
    COLOR_BASE = (0, 100, 200)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_GOLD = (255, 215, 0)
    COLOR_UI_BG = (40, 50, 70, 180)
    COLOR_HEALTH_BAR = (40, 200, 40)
    COLOR_HEALTH_BAR_BG = (200, 40, 40)
    
    # Screen & Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 22, 14
    TILE_W_HALF, TILE_H_HALF = 16, 8
    
    # Game Parameters
    BASE_START_HEALTH = 1000
    STARTING_RESOURCES = 300
    MAX_WAVES = 10
    WAVE_PREP_TIME = 300 # 10 seconds at 30fps
    MAX_STEPS = 15000 # 500 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Game path definition (grid coordinates)
        self.path = [
            (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7),
            (5, 6), (5, 5), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4),
            (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (10, 9), (11, 9),
            (12, 9), (13, 9), (14, 9), (15, 9), (16, 9),
            (16, 8), (16, 7), (16, 6), (17, 6), (18, 6), (19, 6),
            (20, 6), (21, 6)
        ]
        self.buildable_tiles = self._get_buildable_tiles()
        self.base_pos = self.path[-1]

        # Tower definitions
        self.TOWER_TYPES = [
            {'name': 'Gun Turret', 'cost': 75, 'range': 100, 'damage': 15, 'fire_rate': 20, 'color': (0, 150, 255), 'proj_speed': 8},
            {'name': 'Cannon', 'cost': 200, 'range': 80, 'damage': 80, 'fire_rate': 80, 'color': (255, 100, 0), 'proj_speed': 5}
        ]

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.base_health = self.BASE_START_HEALTH
        self.resources = self.STARTING_RESOURCES
        
        self.wave_number = 0
        self.wave_in_progress = False
        self.time_until_next_wave = self.WAVE_PREP_TIME

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        step_reward = -0.01 # Small penalty for time passing

        self._handle_input(movement, space_held, shift_held)
        
        step_reward += self._update_game_state()
        
        self.score += step_reward
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Cycle tower type (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
        
        # Place tower (on press)
        if space_held and not self.last_space_held:
            self._place_tower()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_game_state(self):
        reward = 0
        # Wave management
        if not self.wave_in_progress and not self.game_won:
            self.time_until_next_wave -= 1
            if self.time_until_next_wave <= 0:
                self._spawn_wave()
        
        # Update towers
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                reward += self._tower_attack(tower)

        # Update projectiles
        reward += self._update_projectiles()

        # Update enemies
        reward += self._update_enemies()

        # Update particles
        self._update_particles()

        # Check for wave clear
        if self.wave_in_progress and not self.enemies:
            self.wave_in_progress = False
            reward += 10
            if self.wave_number >= self.MAX_WAVES:
                self.game_won = True
                reward += 100
            else:
                self.time_until_next_wave = self.WAVE_PREP_TIME
        
        return reward
    
    def _spawn_wave(self):
        self.wave_number += 1
        self.wave_in_progress = True
        num_enemies = 5 + self.wave_number * 2
        base_health = 20 + self.wave_number * 10
        base_speed = 0.5 + self.wave_number * 0.05
        
        for i in range(num_enemies):
            enemy_health = base_health * (1 + self.np_random.uniform(-0.1, 0.1))
            enemy_speed = base_speed * (1 + self.np_random.uniform(-0.1, 0.1))
            self.enemies.append({
                'id': self.np_random.integers(1, 1e9),
                'pos': list(self.path[0]),
                'pixel_offset': [-i * 20, 0], # Stagger spawn
                'path_index': 0,
                'max_hp': enemy_health,
                'hp': enemy_health,
                'speed': enemy_speed,
            })

    def _update_enemies(self):
        reward = 0
        for enemy in reversed(self.enemies):
            if enemy['path_index'] >= len(self.path) - 1:
                self.base_health -= enemy['hp']
                self.enemies.remove(enemy)
                self._create_particles(self._grid_to_screen(*self.base_pos), 20, self.COLOR_ENEMY, 2.0)
                continue

            target_node = self.path[enemy['path_index'] + 1]
            current_pos_grid = self.path[enemy['path_index']]
            
            direction = (target_node[0] - current_pos_grid[0], target_node[1] - current_pos_grid[1])
            
            enemy['pixel_offset'][0] += direction[0] * enemy['speed']
            enemy['pixel_offset'][1] += direction[1] * enemy['speed']
            
            # Check if we passed the center of the tile
            if abs(enemy['pixel_offset'][0]) >= self.TILE_W_HALF or abs(enemy['pixel_offset'][1]) >= self.TILE_H_HALF:
                enemy['path_index'] += 1
                enemy['pos'] = list(self.path[enemy['path_index']])
                enemy['pixel_offset'] = [0, 0]
        return reward

    def _tower_attack(self, tower):
        tower_spec = self.TOWER_TYPES[tower['type']]
        tower_pos_screen = self._grid_to_screen(*tower['pos'])
        
        target = None
        min_dist_sq = (tower_spec['range'])**2
        
        # Find closest enemy in range
        for enemy in self.enemies:
            enemy_pos_screen = self._get_enemy_screen_pos(enemy)
            dist_sq = (tower_pos_screen[0] - enemy_pos_screen[0])**2 + (tower_pos_screen[1] - enemy_pos_screen[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                target = enemy
        
        if target:
            # Fire projectile
            self.projectiles.append({
                'start_pos': list(tower_pos_screen),
                'pos': list(tower_pos_screen),
                'target_id': target['id'],
                'speed': tower_spec['proj_speed'],
                'damage': tower_spec['damage'],
                'color': tower_spec['color']
            })
            tower['cooldown'] = tower_spec['fire_rate']
            # sfx: fire_weapon
        return 0

    def _update_projectiles(self):
        reward = 0
        for proj in reversed(self.projectiles):
            target_enemy = next((e for e in self.enemies if e['id'] == proj['target_id']), None)
            
            if not target_enemy:
                self.projectiles.remove(proj)
                continue

            target_pos = self._get_enemy_screen_pos(target_enemy)
            direction = (target_pos[0] - proj['pos'][0], target_pos[1] - proj['pos'][1])
            dist = math.hypot(*direction)
            
            if dist < proj['speed']:
                # Hit
                target_enemy['hp'] -= proj['damage']
                reward += 0.1 # Reward for damaging
                self._create_particles(target_pos, 5, proj['color'], 1.0)
                if target_enemy['hp'] <= 0:
                    reward += 1 # Reward for kill
                    self.resources += 10 + self.wave_number
                    self.enemies.remove(target_enemy)
                    # sfx: enemy_destroyed
                self.projectiles.remove(proj)
            else:
                # Move
                proj['pos'][0] += (direction[0] / dist) * proj['speed']
                proj['pos'][1] += (direction[1] / dist) * proj['speed']
        return reward
    
    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _place_tower(self):
        spec = self.TOWER_TYPES[self.selected_tower_type]
        gx, gy = self.cursor_pos
        
        is_buildable = (gx, gy) in self.buildable_tiles
        is_occupied = any(t['pos'] == [gx, gy] for t in self.towers)
        
        if is_buildable and not is_occupied and self.resources >= spec['cost']:
            self.resources -= spec['cost']
            self.towers.append({
                'pos': [gx, gy],
                'type': self.selected_tower_type,
                'cooldown': 0
            })
            # sfx: place_tower

    def _check_termination(self):
        if self.game_over: return True
        if self.base_health <= 0:
            self.game_over = True
            self.score -= 100 # Penalty for losing
            return True
        if self.game_won:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

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
                screen_pos = self._grid_to_screen(x, y)
                color = self.COLOR_GRID
                if (x, y) in self.path: color = self.COLOR_PATH
                elif (x, y) in self.buildable_tiles: color = (45, 55, 75)
                self._draw_iso_rect(screen_pos, color)
        
        # Draw base
        base_screen_pos = self._grid_to_screen(*self.base_pos)
        self._draw_iso_rect(base_screen_pos, self.COLOR_BASE, border_color=(100, 180, 255))
        
        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_TYPES[tower['type']]
            pos = self._grid_to_screen(*tower['pos'])
            self._draw_iso_rect(pos, spec['color'], border_color=tuple(min(255, c + 80) for c in spec['color']))
            # Draw range indicator when selected
            if tower['pos'] == self.cursor_pos:
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), spec['range'], (255, 255, 255, 80))


        # Draw enemies
        for enemy in self.enemies:
            pos = self._get_enemy_screen_pos(enemy)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 5, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 5, self.COLOR_ENEMY)
            # Health bar
            hp_ratio = enemy['hp'] / enemy['max_hp']
            bar_w = 12
            pygame.draw.rect(self.screen, (100,0,0), (pos[0] - bar_w/2, pos[1] - 12, bar_w, 3))
            pygame.draw.rect(self.screen, (0,200,0), (pos[0] - bar_w/2, pos[1] - 12, bar_w * hp_ratio, 3))

        # Draw projectiles
        for proj in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 3, proj['color'])
            pygame.gfxdraw.aacircle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 3, proj['color'])
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
            color = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)

        # Draw cursor
        self._render_cursor()

    def _render_cursor(self):
        gx, gy = self.cursor_pos
        pos = self._grid_to_screen(gx, gy)
        
        is_buildable = (gx, gy) in self.buildable_tiles
        is_occupied = any(t['pos'] == [gx, gy] for t in self.towers)
        can_afford = self.resources >= self.TOWER_TYPES[self.selected_tower_type]['cost']

        color = (0, 255, 0, 100) # Valid
        if not is_buildable or is_occupied:
            color = (255, 0, 0, 100) # Invalid position
        elif not can_afford:
            color = (255, 255, 0, 100) # Cannot afford

        self._draw_iso_rect(pos, color, border_only=True, border_width=2)


    def _render_ui(self):
        # Top bar
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Base Health
        health_text = self.font_small.render("BASE HP", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 12))
        hp_ratio = max(0, self.base_health / self.BASE_START_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (80, 10, 120, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (80, 10, 120 * hp_ratio, 20))
        
        # Resources
        gold_text = self.font_small.render(f"GOLD: {self.resources}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (220, 12))
        
        # Wave info
        wave_str = f"WAVE: {self.wave_number}/{self.MAX_WAVES}"
        if not self.wave_in_progress and not self.game_won:
            wave_str += f" (Next in {self.time_until_next_wave / 30:.1f}s)"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (340, 12))

        # Selected Tower Info
        spec = self.TOWER_TYPES[self.selected_tower_type]
        tower_info_text = self.font_small.render(
            f"Tower: {spec['name']} | Cost: {spec['cost']}", True, self.COLOR_TEXT
        )
        self.screen.blit(tower_info_text, (10, self.SCREEN_HEIGHT - 25))

        if self.game_over:
            msg = "YOU WON!" if self.game_won else "GAME OVER"
            color = (0, 255, 0) if self.game_won else (255, 0, 0)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_number,
        }
        
    # --- Helper & Drawing Functions ---
    def _grid_to_screen(self, x, y):
        screen_x = (self.SCREEN_WIDTH / 2) + (x - y) * self.TILE_W_HALF
        screen_y = 60 + (x + y) * self.TILE_H_HALF # Offset for UI
        return screen_x, screen_y

    def _get_enemy_screen_pos(self, enemy):
        base_pos = self._grid_to_screen(*enemy['pos'])
        iso_offset_x = (enemy['pixel_offset'][0] - enemy['pixel_offset'][1])
        iso_offset_y = (enemy['pixel_offset'][0] + enemy['pixel_offset'][1]) / 2
        return base_pos[0] + iso_offset_x, base_pos[1] + iso_offset_y

    def _draw_iso_rect(self, pos, color, border_color=None, border_only=False, border_width=1):
        x, y = pos
        points = [
            (x, y - self.TILE_H_HALF),
            (x + self.TILE_W_HALF, y),
            (x, y + self.TILE_H_HALF),
            (x - self.TILE_W_HALF, y)
        ]
        int_points = [(int(px), int(py)) for px, py in points]
        if not border_only:
            pygame.gfxdraw.filled_polygon(self.screen, int_points, color)
        if border_only or border_color:
            final_border_color = border_color if border_color else color
            if border_width > 1:
                pygame.draw.polygon(self.screen, final_border_color, int_points, border_width)
            else:
                pygame.gfxdraw.aapolygon(self.screen, int_points, final_border_color)
    
    def _create_particles(self, pos, count, color, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'size': self.np_random.uniform(1, 4)
            })

    def _get_buildable_tiles(self):
        buildable = set()
        path_set = set(self.path)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) not in path_set:
                    # Check if adjacent to path
                    is_adjacent = False
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        if (x+dx, y+dy) in path_set:
                            is_adjacent = True
                            break
                    if is_adjacent:
                        buildable.add((x,y))
        return buildable

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

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part requires a display. It will not run in a headless environment.
    try:
        # Re-initialize pygame with a display
        pygame.display.init()
        game_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Isometric Tower Defense")
        
        obs, info = env.reset()
        terminated = False
        
        print(env.user_guide)
        
        while not terminated:
            # Action mapping from keyboard
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            game_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Control frame rate
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

    except pygame.error as e:
        print(f"Pygame display error: {e}")
        print("Manual play requires a display. Running in headless mode might cause this.")
    finally:
        env.close()