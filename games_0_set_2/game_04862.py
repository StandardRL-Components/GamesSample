
# Generated: 2025-08-28T03:14:13.412277
# Source Brief: brief_04862.md
# Brief Index: 4862

        
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
        "Controls: Arrow keys to move the cursor. Space to place the selected tower. "
        "Shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers on a grid "
        "in this minimalist, procedurally generated tower defense game."
    )

    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 50)
    COLOR_PATH = (60, 60, 70)
    COLOR_BASE = (0, 200, 200)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TOWER_1 = (50, 150, 255)
    COLOR_TOWER_2 = (200, 100, 255)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH_GREEN = (50, 255, 50)
    COLOR_HEALTH_RED = (255, 50, 50)

    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 12
    CELL_SIZE = 32
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2

    # Game Parameters
    MAX_STEPS = 30 * 60 * 2 # 2 minutes at 30fps
    TOTAL_WAVES = 10
    INITIAL_GOLD = 150
    INITIAL_BASE_HEALTH = 20
    WAVE_INTERMISSION_TIME = 30 * 5 # 5 seconds
    ENEMY_SPAWN_INTERVAL = 15 # frames

    # Tower Types
    TOWER_SPECS = [
        {"name": "Cannon", "cost": 50, "range": 3.5, "fire_rate": 30, "damage": 10, "color": COLOR_TOWER_1},
        {"name": "Sniper", "cost": 80, "range": 6.0, "fire_rate": 75, "damage": 35, "color": COLOR_TOWER_2},
    ]

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.render_mode = render_mode
        self.np_random = None
        
        # Action state tracking
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.path, self.base_pos = self._generate_path()
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        
        self.gold = self.INITIAL_GOLD
        self.base_health = self.INITIAL_BASE_HEALTH
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.current_wave = 0
        self.wave_timer = self.WAVE_INTERMISSION_TIME
        self.enemies_to_spawn = []
        self.spawn_timer = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.001 # Small penalty for existing
        self.steps += 1
        
        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)

        # --- Game Logic Update ---
        self._update_wave_system()
        
        new_reward = self._update_towers()
        reward += new_reward

        new_reward = self._update_enemies()
        reward += new_reward

        new_reward = self._update_projectiles()
        reward += new_reward

        self._update_particles()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.game_won:
                reward += 100
            else: # Game Lost
                reward += -100
        
        self.score += reward

        if self.auto_advance:
            self.clock.tick(30)
            
        return (
            self._get_observation(),
            reward,
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

        # Place tower on key press
        if space_held and not self.prev_space_held:
            self._place_tower()
        
        # Cycle tower type on key press
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_cycle_sound

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _place_tower(self):
        x, y = self.cursor_pos
        spec = self.TOWER_SPECS[self.selected_tower_type]
        
        if self.grid[x, y] == 0 and self.gold >= spec["cost"]:
            self.gold -= spec["cost"]
            self.grid[x, y] = 2 # Mark as tower
            self.towers.append({
                "pos": [x, y],
                "cooldown": 0,
                "type": self.selected_tower_type
            })
            # sfx: place_tower_sound
            self._create_particles(self._grid_to_pixel(x, y), 20, spec['color'], 2.0, 15)

    def _update_wave_system(self):
        if self.current_wave == 0 or (len(self.enemies) == 0 and len(self.enemies_to_spawn) == 0):
            self.wave_timer -= 1
            if self.wave_timer <= 0 and self.current_wave < self.TOTAL_WAVES:
                self.current_wave += 1
                self.enemies_to_spawn = self._generate_wave(self.current_wave)
                self.wave_timer = self.WAVE_INTERMISSION_TIME
                self.score += 1 # Reward for starting a new wave
        
        if len(self.enemies_to_spawn) > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self.enemies.append(self.enemies_to_spawn.pop(0))
                self.spawn_timer = self.ENEMY_SPAWN_INTERVAL
    
    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                spec = self.TOWER_SPECS[tower['type']]
                target = self._find_target(tower)
                if target:
                    # sfx: tower_shoot_sound
                    self._fire_projectile(tower, target)
                    tower['cooldown'] = spec['fire_rate']
        return reward
        
    def _find_target(self, tower):
        tower_pos = self._grid_to_pixel(tower['pos'][0], tower['pos'][1])
        spec = self.TOWER_SPECS[tower['type']]
        range_sq = (spec['range'] * self.CELL_SIZE) ** 2
        
        best_target = None
        max_dist = -1

        for enemy in self.enemies:
            dist_sq = (tower_pos[0] - enemy['pos'][0])**2 + (tower_pos[1] - enemy['pos'][1])**2
            if dist_sq <= range_sq:
                # Target enemy furthest along the path
                if enemy['path_progress'] > max_dist:
                    max_dist = enemy['path_progress']
                    best_target = enemy
        return best_target
        
    def _fire_projectile(self, tower, target):
        start_pos = self._grid_to_pixel(tower['pos'][0], tower['pos'][1])
        spec = self.TOWER_SPECS[tower['type']]
        self.projectiles.append({
            "pos": list(start_pos),
            "target": target,
            "damage": spec['damage'],
            "speed": 8,
        })
        self._create_particles(start_pos, 5, self.COLOR_PROJECTILE, 2, 5)

    def _update_enemies(self):
        reward = 0
        base_damage = 0
        for enemy in reversed(self.enemies):
            path_node_idx = enemy['path_index']
            
            if path_node_idx + 1 >= len(self.path):
                # Reached the base
                base_damage += 1
                self.enemies.remove(enemy)
                base_pixel_pos = self._grid_to_pixel(self.base_pos[0], self.base_pos[1])
                self._create_particles(base_pixel_pos, 30, self.COLOR_ENEMY, 3.0, 20)
                # sfx: base_hit_sound
                continue

            target_node = self.path[path_node_idx + 1]
            target_pos = self._grid_to_pixel(target_node[0], target_node[1])
            
            direction = [target_pos[0] - enemy['pos'][0], target_pos[1] - enemy['pos'][1]]
            dist = math.hypot(*direction)
            
            if dist < enemy['speed']:
                enemy['pos'] = list(target_pos)
                enemy['path_index'] += 1
            else:
                norm_dir = [d / dist for d in direction]
                enemy['pos'][0] += norm_dir[0] * enemy['speed']
                enemy['pos'][1] += norm_dir[1] * enemy['speed']
            
            enemy['path_progress'] = enemy['path_index'] + (1 - dist / self.CELL_SIZE)

        if base_damage > 0:
            self.base_health -= base_damage
            reward -= base_damage * 0.5 # Penalty for each enemy reaching base

        return reward

    def _update_projectiles(self):
        reward = 0
        for p in reversed(self.projectiles):
            if p['target'] not in self.enemies:
                self.projectiles.remove(p)
                continue

            target_pos = p['target']['pos']
            direction = [target_pos[0] - p['pos'][0], target_pos[1] - p['pos'][1]]
            dist = math.hypot(*direction)

            if dist < p['speed']:
                # Hit target
                p['target']['health'] -= p['damage']
                self._create_particles(p['pos'], 10, self.COLOR_ENEMY, 1.5, 10)
                # sfx: enemy_hit_sound
                if p['target']['health'] <= 0:
                    self.gold += p['target']['gold_value']
                    self._create_particles(p['target']['pos'], 40, self.COLOR_ENEMY, 4.0, 25)
                    self.enemies.remove(p['target'])
                    # sfx: enemy_death_sound
                    reward += 0.1 # Reward for killing enemy
                self.projectiles.remove(p)
            else:
                norm_dir = [d / dist for d in direction]
                p['pos'][0] += norm_dir[0] * p['speed']
                p['pos'][1] += norm_dir[1] * p['speed']
        return reward
        
    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
        elif self.current_wave >= self.TOTAL_WAVES and len(self.enemies) == 0 and len(self.enemies_to_spawn) == 0:
            self.game_over = True
            self.game_won = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._draw_grid()
        self._draw_path()
        self._draw_base()
        
        for tower in self.towers:
            self._draw_tower(tower)
            
        for p in self.projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

        for enemy in self.enemies:
            self._draw_enemy(enemy)
            
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(self.screen, color, pos, int(p['size']))

        self._draw_cursor()

    def _render_ui(self):
        # Gold
        gold_text = self.font_small.render(f"GOLD: {self.gold}", True, self.COLOR_TEXT)
        self.screen.blit(gold_text, (10, 10))
        
        # Base Health
        health_text = self.font_small.render(f"BASE HP: {self.base_health}/{self.INITIAL_BASE_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 30))

        # Wave Info
        wave_str = f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}"
        if self.wave_timer > 0 and self.current_wave < self.TOTAL_WAVES:
            wave_str = f"NEXT WAVE IN {self.wave_timer / 30:.1f}s"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        
        # Tower Selection
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_info = f"TOWER: {spec['name']} | COST: {spec['cost']}"
        tower_text = self.font_small.render(tower_info, True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (10, self.SCREEN_HEIGHT - 25))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "VICTORY!" if self.game_won else "DEFEAT"
        color = self.COLOR_BASE if self.game_won else self.COLOR_ENEMY
        
        text = self.font_large.render(message, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.current_wave,
            "base_health": self.base_health,
        }

    # --- Drawing Helpers ---
    def _draw_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

    def _draw_path(self):
        for x, y in self.path:
            rect = pygame.Rect(self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PATH, rect)

    def _draw_base(self):
        x, y = self.base_pos
        center = self._grid_to_pixel(x, y)
        size = self.CELL_SIZE // 2
        pygame.draw.rect(self.screen, self.COLOR_BASE, (center[0] - size, center[1] - size, size*2, size*2))
        
        # Health bar
        bar_width = self.CELL_SIZE * 1.5
        bar_height = 5
        hp_ratio = self.base_health / self.INITIAL_BASE_HEALTH
        bg_rect = pygame.Rect(center[0] - bar_width/2, center[1] - size - 15, bar_width, bar_height)
        fg_rect = pygame.Rect(center[0] - bar_width/2, center[1] - size - 15, bar_width * hp_ratio, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, fg_rect)


    def _draw_tower(self, tower):
        spec = self.TOWER_SPECS[tower['type']]
        center = self._grid_to_pixel(tower['pos'][0], tower['pos'][1])
        radius = self.CELL_SIZE // 3
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, spec['color'])
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, spec['color'])

    def _draw_enemy(self, enemy):
        pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
        radius = self.CELL_SIZE // 4
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
        
        # Health bar
        bar_width = self.CELL_SIZE * 0.8
        bar_height = 3
        hp_ratio = enemy['health'] / enemy['max_health']
        bg_rect = pygame.Rect(pos[0] - bar_width/2, pos[1] - radius - 8, bar_width, bar_height)
        fg_rect = pygame.Rect(pos[0] - bar_width/2, pos[1] - radius - 8, bar_width * hp_ratio, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, fg_rect)

    def _draw_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        # Check validity for color
        can_place = self.grid[x, y] == 0 and self.gold >= self.TOWER_SPECS[self.selected_tower_type]['cost']
        color = (0, 255, 0) if can_place else (255, 0, 0)
        
        pygame.draw.rect(self.screen, color, rect, 2)
        
        # Draw range indicator
        spec = self.TOWER_SPECS[self.selected_tower_type]
        center = self._grid_to_pixel(x, y)
        radius = int(spec['range'] * self.CELL_SIZE)
        
        # Draw a transparent circle for the range
        surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.aacircle(surface, radius, radius, radius - 1, (*color, 100))
        self.screen.blit(surface, (center[0] - radius, center[1] - radius))


    # --- Generation Helpers ---
    def _generate_path(self):
        path = []
        grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)

        start_y = self.np_random.integers(1, self.GRID_HEIGHT - 1)
        curr = [0, start_y]
        path.append(list(curr))
        grid[curr[0], curr[1]] = 1

        while curr[0] < self.GRID_WIDTH - 2:
            possible_moves = []
            # Prioritize right
            if curr[0] + 1 < self.GRID_WIDTH and grid[curr[0] + 1, curr[1]] == 0:
                possible_moves.extend([[1, 0]] * 5) # Weighted choice
            # Up
            if curr[1] - 1 >= 0 and grid[curr[0], curr[1] - 1] == 0:
                possible_moves.append([0, -1])
            # Down
            if curr[1] + 1 < self.GRID_HEIGHT and grid[curr[0], curr[1] + 1] == 0:
                possible_moves.append([0, 1])

            if not possible_moves: # Stuck, backtrack
                if len(path) > 1:
                    path.pop()
                    curr = path[-1]
                    continue
                else: # Should not happen with this logic, but as a fallback
                    break

            move = random.choice(possible_moves)
            curr[0] += move[0]
            curr[1] += move[1]
            path.append(list(curr))
            grid[curr[0], curr[1]] = 1
        
        # Connect to the end
        base_pos = [self.GRID_WIDTH - 1, curr[1]]
        path.append(base_pos)
        grid[base_pos[0], base_pos[1]] = 1
        
        for x, y in path:
            self.grid[x, y] = 1 # Mark path on main grid

        return path, base_pos

    def _generate_wave(self, wave_num):
        enemies = []
        num_enemies = 5 + wave_num * 2
        base_health = 20 + (wave_num - 1) * 10
        base_speed = 1.5 + (wave_num - 1) * 0.1
        base_gold = 2 + int((wave_num-1)/2)

        start_pos = self._grid_to_pixel(self.path[0][0], self.path[0][1])

        for _ in range(num_enemies):
            enemies.append({
                "pos": list(start_pos),
                "health": base_health,
                "max_health": base_health,
                "speed": base_speed + self.np_random.uniform(-0.2, 0.2),
                "gold_value": base_gold,
                "path_index": 0,
                "path_progress": 0.0,
            })
        return enemies

    def _create_particles(self, pos, count, color, speed, life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = [math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5),
                   math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5) - speed * 0.5]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": life + self.np_random.integers(-5, 5),
                "max_life": life,
                "color": color,
                "size": self.np_random.uniform(1, 3)
            })

    def _grid_to_pixel(self, x, y):
        return [
            self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
        ]

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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a separate pygame screen for display
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    done = False
    total_reward = 0
    
    # Game loop
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move_action = 0 # No-op
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering ---
        # The observation is a numpy array, convert it back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

    print(f"Game Over. Final Score: {total_reward:.2f}")
    pygame.quit()