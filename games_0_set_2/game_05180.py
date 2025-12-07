
# Generated: 2025-08-28T04:14:04.003027
# Source Brief: brief_05180.md
# Brief Index: 5180

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to place selected block. Shift to cycle block type (Wall/Turret)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Build a fortress of walls and turrets to defend your core against waves of geometric enemies in this isometric strategy game."
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
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 20
        self.MAX_STEPS = 2000
        self.MAX_WAVES = 20

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 48, 56)
        self.COLOR_CORE = (255, 200, 0)
        self.COLOR_CORE_DARK = (180, 140, 0)
        self.COLOR_WALL = (0, 180, 140)
        self.COLOR_WALL_DARK = (0, 120, 90)
        self.COLOR_TURRET = (0, 180, 220)
        self.COLOR_TURRET_DARK = (0, 120, 160)
        self.COLOR_ENEMY = (255, 70, 80)
        self.COLOR_ENEMY_DARK = (180, 50, 60)
        self.COLOR_PROJECTILE = (150, 200, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR_VALID = (255, 255, 255, 100)
        self.COLOR_CURSOR_INVALID = (255, 0, 0, 100)
        
        # Fonts
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # Isometric Projection
        self.TILE_WIDTH = 32
        self.TILE_HEIGHT = 16
        self.TILE_DEPTH = 20
        self.TILE_WIDTH_HALF = self.TILE_WIDTH // 2
        self.TILE_HEIGHT_HALF = self.TILE_HEIGHT // 2
        self.ORIGIN_X = self.SCREEN_WIDTH // 2
        self.ORIGIN_Y = 100

        # Block Types
        self.BLOCK_TYPES = [
            {'name': 'Wall', 'color': self.COLOR_WALL, 'dark_color': self.COLOR_WALL_DARK, 'max_health': 10, 'cost': 1},
            {'name': 'Turret', 'color': self.COLOR_TURRET, 'dark_color': self.COLOR_TURRET_DARK, 'max_health': 5, 'cost': 2, 'range': 5, 'fire_rate': 45, 'damage': 2}
        ]

        # Initialize state variables
        self.reset()
        
        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.grid = np.full((self.GRID_WIDTH, self.GRID_HEIGHT), -1, dtype=int)
        self.blocks = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.base_core = {
            'pos': (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2),
            'health': 100,
            'max_health': 100,
            'hit_timer': 0,
        }
        gx, gy = self.base_core['pos']
        self.grid[gx, gy] = -2  # Special value for base

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2 - 3]
        self.selected_block_type = 0
        self.block_budget = 10

        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_cooldown = 150 # 5s at 30fps
        self.enemies_to_spawn_this_wave = 0
        self.enemy_spawn_timer = 0

        self.previous_action = self.action_space.sample()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = self.game_over

        if not self.game_over:
            # Unpack factorized action and update logic
            self._handle_input(action)

            reward += self._update_wave_manager()
            self._update_enemies()
            self._update_turrets()
            reward += self._update_projectiles()
            reward += self._process_destroyed_entities()
            self._update_particles()
            
            self.steps += 1

            # Check termination conditions
            if self.base_core['health'] <= 0:
                self.game_over = True
                terminated = True
                reward -= 100
                # Sound: game_over_sound()
            elif self.wave_number > self.MAX_WAVES:
                self.game_over = True
                self.win = True
                terminated = True
                reward += 100
                # Sound: victory_sound()
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
                terminated = True
        
        self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Register actions on press (transition from 0 to 1)
        if space_held and not (self.previous_action[1] == 1):
            self._place_block()

        if shift_held and not (self.previous_action[2] == 1):
            self.selected_block_type = (self.selected_block_type + 1) % len(self.BLOCK_TYPES)
            # Sound: ui_cycle_sound()
        
        self.previous_action = action

    def _place_block(self):
        cx, cy = self.cursor_pos
        block_def = self.BLOCK_TYPES[self.selected_block_type]
        cost = block_def['cost']

        if self.grid[cx, cy] == -1 and self.block_budget >= cost:
            self.block_budget -= cost
            
            new_block = {
                'pos': (cx, cy),
                'type_idx': self.selected_block_type,
                'health': block_def['max_health'],
                'max_health': block_def['max_health'],
                'hit_timer': 0,
            }
            if block_def['name'] == 'Turret':
                new_block['fire_cooldown'] = self.np_random.integers(0, block_def['fire_rate'])

            self.blocks.append(new_block)
            self.grid[cx, cy] = len(self.blocks) - 1
            # Sound: place_block_sound()

    def _update_wave_manager(self):
        reward = 0
        if not self.wave_in_progress:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0:
                self.wave_in_progress = True
                self.wave_number += 1
                if self.wave_number > self.MAX_WAVES: return 0

                # Wave clear rewards and resource gain
                reward += len(self.blocks) * 0.1
                self.block_budget += 5 + self.wave_number

                self.enemies_to_spawn_this_wave = 2 + self.wave_number
                self.enemy_spawn_timer = 0
        else: # Wave is in progress
            self.enemy_spawn_timer -= 1
            if self.enemies_to_spawn_this_wave > 0 and self.enemy_spawn_timer <= 0:
                self._spawn_enemy()
                self.enemies_to_spawn_this_wave -= 1
                self.enemy_spawn_timer = 30 # 1s spawn interval
            
            if self.enemies_to_spawn_this_wave == 0 and not self.enemies:
                self.wave_in_progress = False
                self.wave_cooldown = 450 # 15s between waves
        return reward

    def _spawn_enemy(self):
        side = self.np_random.integers(4)
        if side == 0: pos = [self.np_random.uniform(0, self.GRID_WIDTH-1), -1.0]
        elif side == 1: pos = [self.np_random.uniform(0, self.GRID_WIDTH-1), self.GRID_HEIGHT]
        elif side == 2: pos = [-1.0, self.np_random.uniform(0, self.GRID_HEIGHT-1)]
        else: pos = [self.GRID_WIDTH, self.np_random.uniform(0, self.GRID_HEIGHT-1)]

        health = 5 + (self.wave_number - 1) // 2
        speed = 0.02 + (self.wave_number - 1) * 0.005
        
        self.enemies.append({
            'pos': np.array(pos, dtype=float), 'health': health, 'max_health': health,
            'speed': speed, 'attack_cooldown': 0, 'hit_timer': 0,
        })

    def _update_enemies(self):
        base_pos = np.array(self.base_core['pos'], dtype=float)
        for enemy in self.enemies:
            if enemy['attack_cooldown'] > 0: enemy['attack_cooldown'] -= 1
            if enemy['hit_timer'] > 0: enemy['hit_timer'] -= 1

            direction = base_pos - enemy['pos']
            dist = np.linalg.norm(direction)
            if dist == 0: continue
            
            next_pos = enemy['pos'] + (direction / dist) * enemy['speed']
            target_grid_pos = (int(round(next_pos[0])), int(round(next_pos[1])))
            
            is_blocked = False
            if 0 <= target_grid_pos[0] < self.GRID_WIDTH and 0 <= target_grid_pos[1] < self.GRID_HEIGHT:
                grid_val = self.grid[target_grid_pos]
                if grid_val >= 0: # Block
                    is_blocked = True
                    if enemy['attack_cooldown'] == 0:
                        block = self.blocks[grid_val]
                        block['health'] -= 1
                        block['hit_timer'] = 5
                        enemy['attack_cooldown'] = 60 # 2s
                        self._create_particles(self._iso_to_screen(*block['pos']), 3, self.COLOR_CORE, 0.5)
                        # Sound: hit_block_sound()
                elif grid_val == -2: # Base
                    is_blocked = True
                    if enemy['attack_cooldown'] == 0:
                        self.base_core['health'] -= 5
                        self.base_core['hit_timer'] = 5
                        enemy['attack_cooldown'] = 60
                        self._create_particles(self._iso_to_screen(*self.base_core['pos']), 5, self.COLOR_ENEMY, 1)
                        # Sound: hit_base_sound()
            
            if not is_blocked:
                enemy['pos'] = next_pos

    def _update_turrets(self):
        for block in self.blocks:
            block_def = self.BLOCK_TYPES[block['type_idx']]
            if block_def['name'] == 'Turret':
                if block['fire_cooldown'] > 0:
                    block['fire_cooldown'] -= 1
                elif self.enemies:
                    target = self._find_target_for_turret(block)
                    if target:
                        self._fire_projectile(block, target)
                        block['fire_cooldown'] = block_def['fire_rate']
                        # Sound: turret_fire_sound()

    def _find_target_for_turret(self, turret):
        turret_pos = np.array(turret['pos'])
        turret_def = self.BLOCK_TYPES[turret['type_idx']]
        best_target = None
        min_dist_sq = turret_def['range'] ** 2
        
        for enemy in self.enemies:
            dist_sq = np.sum((np.array(enemy['pos']) - turret_pos)**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_target = enemy
        return best_target

    def _fire_projectile(self, turret, target):
        start_pos = self._iso_to_screen(*turret['pos'])
        target_pos = self._iso_to_screen(*target['pos'])
        direction = target_pos - start_pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            self.projectiles.append({
                'pos': start_pos, 'vel': (direction / dist) * 8, 'life': 50
            })

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                projectiles_to_remove.append(i)
                continue

            for enemy in self.enemies:
                if enemy['health'] <= 0: continue
                enemy_screen_pos = self._iso_to_screen(*enemy['pos'])
                if np.linalg.norm(p['pos'] - enemy_screen_pos) < 10:
                    block_def = self.BLOCK_TYPES[1] # Turret
                    enemy['health'] -= block_def['damage']
                    enemy['hit_timer'] = 5
                    self._create_particles(p['pos'], 5, self.COLOR_PROJECTILE, 1)
                    if i not in projectiles_to_remove: projectiles_to_remove.append(i)
                    
                    if enemy['health'] <= 0:
                        reward += 1 # Reward for destroying an enemy
                        # Sound: enemy_die_sound()
                    break
        
        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
        return reward

    def _process_destroyed_entities(self):
        reward = 0
        
        # Destroyed enemies
        enemies_alive = []
        for enemy in self.enemies:
            if enemy['health'] > 0:
                enemies_alive.append(enemy)
            else:
                self._create_particles(self._iso_to_screen(*enemy['pos']), 30, self.COLOR_ENEMY_DARK, 2)
        self.enemies = enemies_alive
        
        # Destroyed blocks
        blocks_to_remove_indices = [i for i, b in enumerate(self.blocks) if b['health'] <= 0]
        if not blocks_to_remove_indices:
            return reward

        reward -= len(blocks_to_remove_indices) * 0.5 # Penalty for losing a block
        
        # Create particles and clear grid
        for i in blocks_to_remove_indices:
            block = self.blocks[i]
            self.grid[block['pos']] = -1
            block_def = self.BLOCK_TYPES[block['type_idx']]
            self._create_particles(self._iso_to_screen(*block['pos']), 20, block_def['dark_color'], 1.5)
            # Sound: block_destroy_sound()

        # Rebuild blocks list and remap grid indices
        new_blocks = []
        old_to_new_map = {}
        new_idx = 0
        for i, block in enumerate(self.blocks):
            if i not in blocks_to_remove_indices:
                old_to_new_map[i] = new_idx
                new_blocks.append(block)
                new_idx += 1
        self.blocks = new_blocks
        
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] >= 0:
                    self.grid[x, y] = old_to_new_map[self.grid[x, y]]

        return reward

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'][1] += 0.1 # Gravity
        self.particles = [p for p in self.particles if p['life'] > 0]

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
            "wave": self.wave_number,
            "blocks": len(self.blocks),
            "enemies": len(self.enemies),
            "core_health": self.base_core['health'],
            "block_budget": self.block_budget,
        }

    def _render_game(self):
        self._draw_grid()

        render_list = []
        for i, block in enumerate(self.blocks):
            render_list.append({'type': 'block', 'obj': block, 'pos': block['pos'], 'idx': i})
        for enemy in self.enemies:
            render_list.append({'type': 'enemy', 'obj': enemy, 'pos': enemy['pos']})
        render_list.append({'type': 'core', 'obj': self.base_core, 'pos': self.base_core['pos']})
        
        # Sort by a combination of y and x to get correct isometric draw order
        render_list.sort(key=lambda item: (item['pos'][1] + item['pos'][0], item['pos'][1] - item['pos'][0]))
        
        for item in render_list:
            if item['type'] == 'block': self._draw_iso_cube(item['obj'])
            elif item['type'] == 'enemy': self._draw_iso_cube(item['obj'], is_enemy=True)
            elif item['type'] == 'core': self._draw_iso_cube(item['obj'], is_core=True)

        self._render_cursor()
        self._render_projectiles()
        self._render_particles()
    
    def _draw_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

    def _iso_to_screen(self, gx, gy):
        return np.array([self.ORIGIN_X + (gx - gy) * self.TILE_WIDTH_HALF,
                         self.ORIGIN_Y + (gx + gy) * self.TILE_HEIGHT_HALF])

    def _draw_iso_cube(self, entity, is_enemy=False, is_core=False):
        pos = entity['pos']
        screen_pos = self._iso_to_screen(pos[0], pos[1])
        x, y = int(screen_pos[0]), int(screen_pos[1])
        
        health_ratio = max(0, entity['health'] / entity['max_health'])
        depth = self.TILE_DEPTH * health_ratio if not is_enemy else self.TILE_DEPTH / 2

        if is_enemy:
            color, dark_color = self.COLOR_ENEMY, self.COLOR_ENEMY_DARK
        elif is_core:
            color, dark_color = self.COLOR_CORE, self.COLOR_CORE_DARK
        else:
            block_def = self.BLOCK_TYPES[entity['type_idx']]
            color, dark_color = block_def['color'], block_def['dark_color']

        if entity['hit_timer'] > 0:
            color = (255, 255, 255)
            dark_color = (200, 200, 200)

        p = [
            (x, y - int(depth)), (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF - int(depth)),
            (x, y + self.TILE_HEIGHT - int(depth)), (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF - int(depth)),
            (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF), (x, y + self.TILE_HEIGHT),
            (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF)
        ]
        
        # Top face
        pygame.gfxdraw.filled_polygon(self.screen, [p[0], p[1], p[2], p[3]], color)
        # Left face
        pygame.gfxdraw.filled_polygon(self.screen, [p[3], p[2], p[5], p[6]], dark_color)
        # Right face
        pygame.gfxdraw.filled_polygon(self.screen, [p[2], p[1], p[4], p[5]], tuple(int(c*0.7) for c in color))

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        screen_pos = self._iso_to_screen(cx, cy)
        x, y = int(screen_pos[0]), int(screen_pos[1])
        points = [
            (x, y), (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),
            (x, y + self.TILE_HEIGHT), (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF)
        ]
        
        block_def = self.BLOCK_TYPES[self.selected_block_type]
        is_valid = self.grid[cx, cy] == -1 and self.block_budget >= block_def['cost']
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_projectiles(self):
        for p in self.projectiles:
            start = p['pos']
            end = p['pos'] - p['vel'] * 0.5
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start, end, 3)

    def _render_particles(self):
        for p in self.particles:
            size = int(p['life'] * p['size'] / 10)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], p['pos'], size)

    def _render_ui(self):
        # Top UI
        wave_text = f"Wave: {self.wave_number}/{self.MAX_WAVES}" if self.wave_number <= self.MAX_WAVES else "Victory!"
        self._draw_text(wave_text, (10, 10), self.font_m)
        
        core_hp_text = f"Core HP: {max(0, self.base_core['health'])}"
        self._draw_text(core_hp_text, (self.SCREEN_WIDTH // 2, 10), self.font_m, center=True)
        
        budget_text = f"Budget: {self.block_budget}"
        self._draw_text(budget_text, (self.SCREEN_WIDTH - 10, 10), self.font_m, right=True)

        # Bottom UI
        block_def = self.BLOCK_TYPES[self.selected_block_type]
        selected_text = f"Selected: {block_def['name']} (Cost: {block_def['cost']})"
        self._draw_text(selected_text, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30), self.font_m, center=True)
        
        # Inter-wave timer
        if not self.wave_in_progress and not self.game_over:
            timer_text = f"Next wave in: {math.ceil(self.wave_cooldown / 30)}s"
            self._draw_text(timer_text, (self.SCREEN_WIDTH // 2, 40), self.font_s, center=True)
        
        # Game Over / Win screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            message = "VICTORY" if self.win else "GAME OVER"
            self._draw_text(message, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20), self.font_l, center=True)

    def _draw_text(self, text, pos, font, color=None, center=False, right=False):
        if color is None: color = self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center: text_rect.center = pos
        elif right: text_rect.topright = pos
        else: text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, pos, count, color, size_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'color': color, 'size': size_mult})
    
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the game directly for testing
    import os
    # If you are not on a headless server, you can comment out the next line
    # to see the game window.
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    
    # To view the game, we need a window. This will fail on a headless server
    # if the dummy driver is not set.
    try:
        pygame.display.set_caption("Isometric Fortress")
        real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        interactive = True
    except pygame.error:
        interactive = False
        print("Pygame display could not be initialized. Running headlessly.")

    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        if interactive:
            action = [0, 0, 0] # Default no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False

            if not terminated:
                keys = pygame.key.get_pressed()
                # Movement
                if keys[pygame.K_UP]: action[0] = 1
                elif keys[pygame.K_DOWN]: action[0] = 2
                elif keys[pygame.K_LEFT]: action[0] = 3
                elif keys[pygame.K_RIGHT]: action[0] = 4
                else: action[0] = 0
                
                # Buttons
                if keys[pygame.K_SPACE]: action[1] = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

                obs, reward, terminated, truncated, info = env.step(action)
            
            # Blit the environment's screen to the real screen
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            real_screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            env.clock.tick(30) # Control the frame rate
        else: # Non-interactive loop for headless testing
            if not terminated:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                obs, info = env.reset()
                terminated = False
            if info['steps'] > 5000: # Run for a limited number of steps
                running = False

    pygame.quit()