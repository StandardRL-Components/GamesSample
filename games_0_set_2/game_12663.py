import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:14:02.843957
# Source Brief: brief_02663.md
# Brief Index: 2663
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
        "Defend your base crystal from waves of enemies by matching gems to power up your turrets. "
        "A hybrid of match-3 puzzle and tower defense."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to match crystals and "
        "generate energy. Press shift to select and upgrade your turrets."
    )
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 7
    CELL_SIZE = 32
    GRID_START_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_START_Y = 60

    # Colors
    COLOR_BG = (15, 10, 30)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_UI_BG = (30, 20, 60, 180)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_CURSOR_INVALID = (255, 50, 50)
    
    CRYSTAL_COLORS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (80, 80, 255),   # Blue
        (200, 50, 255),  # Purple
    ]
    NUM_CRYSTAL_TYPES = len(CRYSTAL_COLORS)
    MIN_MATCH_SIZE = 3

    # Game Parameters
    MAX_STEPS = 5000
    MAX_WAVES = 50
    INITIAL_BASE_HEALTH = 10
    TURRET_POSITIONS = [(80, 200), (SCREEN_WIDTH - 80, 200)]
    TURRET_UPGRADE_COST = [25, 50, 100, 200, 400, 800] # Cost for levels 2, 3, ...
    BASE_CRYSTAL_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 40)

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
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.energy = 0
        self.base_crystal_health = 0
        self.wave_number = 0
        self.wave_timer = 0
        self.enemies_to_spawn_this_wave = 0
        self.enemies_spawned_this_wave = 0
        self.active_enemies = []
        self.projectiles = []
        self.particles = []
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.cursor_pos = [0, 0]
        self.turrets = []
        self.selected_turret_for_upgrade = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.screen_shake = 0
        
        # self.reset() is called by the environment wrapper
        
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        self.energy = 0
        self.base_crystal_health = self.INITIAL_BASE_HEALTH
        self.wave_number = 1
        self._setup_wave()

        self.active_enemies = []
        self.projectiles = []
        self.particles = []
        
        self._populate_grid()
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.turrets = [
            {'pos': self.TURRET_POSITIONS[0], 'level': 1, 'fire_cooldown': 0, 'range': 120, 'damage': 1, 'fire_rate': 60},
            {'pos': self.TURRET_POSITIONS[1], 'level': 1, 'fire_cooldown': 0, 'range': 120, 'damage': 1, 'fire_rate': 60}
        ]
        self.selected_turret_for_upgrade = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.screen_shake = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # 1. Handle player input
        self._handle_movement(movement)
        match_reward = self._handle_matching(space_held)
        upgrade_reward = self._handle_upgrades(shift_held)
        reward += match_reward + upgrade_reward

        # 2. Update game logic
        self._update_wave_manager()
        self._update_turrets()
        self._update_projectiles()
        
        enemy_reward, wave_reward = self._update_enemies()
        reward += enemy_reward + wave_reward
        
        self._update_particles()
        self._apply_gravity_and_refill()

        # 3. Finalize step
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not truncated:
            if self.win_condition_met:
                reward += 100.0
            else: # Lost
                reward -= 100.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Game Logic Helpers ---

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = self.cursor_pos[0] % self.GRID_WIDTH
        self.cursor_pos[1] = self.cursor_pos[1] % self.GRID_HEIGHT

    def _handle_matching(self, space_held):
        reward = 0.0
        if space_held and not self.prev_space_held:
            match_group = self._find_match_group(self.cursor_pos[0], self.cursor_pos[1])
            if len(match_group) >= self.MIN_MATCH_SIZE:
                # SFX: Crystal Match
                num_matched = len(match_group)
                energy_gain = num_matched * num_matched # Bonus for larger matches
                self.energy += energy_gain
                
                reward += num_matched * 0.1
                reward += energy_gain * 0.01

                for x, y in match_group:
                    color_idx = self.grid[x, y] - 1
                    self.grid[x, y] = 0
                    self._create_particles(self._grid_to_pixel(x, y), self.CRYSTAL_COLORS[color_idx], 15, 2.0)
        return reward

    def _handle_upgrades(self, shift_held):
        if shift_held and not self.prev_shift_held:
            self.selected_turret_for_upgrade = (self.selected_turret_for_upgrade + 1) % len(self.turrets)
            # SFX: UI Select
            turret = self.turrets[self.selected_turret_for_upgrade]
            level_idx = turret['level'] - 1
            if level_idx < len(self.TURRET_UPGRADE_COST):
                cost = self.TURRET_UPGRADE_COST[level_idx]
                if self.energy >= cost:
                    # SFX: Upgrade Success
                    self.energy -= cost
                    turret['level'] += 1
                    turret['damage'] += 0.5
                    turret['range'] += 10
                    turret['fire_rate'] = max(20, turret['fire_rate'] - 5)
                    self._create_particles(turret['pos'], (255, 255, 100), 20, 3.0)
        return 0
    
    def _update_wave_manager(self):
        if self.enemies_spawned_this_wave >= self.enemies_to_spawn_this_wave and len(self.active_enemies) == 0:
            return # Wave is cleared, wait for _update_enemies to advance it

        self.wave_timer -= 1
        if self.wave_timer <= 0 and self.enemies_spawned_this_wave < self.enemies_to_spawn_this_wave:
            self._spawn_enemy()
            self.enemies_spawned_this_wave += 1
            spawn_rate_scaling = max(0.1, 3.0 - (self.wave_number // 10) * 0.5)
            self.wave_timer = int(spawn_rate_scaling * 30) # 30 FPS assumption

    def _spawn_enemy(self):
        # SFX: Shadow Spawn
        side = self.np_random.integers(4)
        if side == 0: # Top
            pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), -20]
        elif side == 1: # Right
            pos = [self.SCREEN_WIDTH + 20, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
        elif side == 2: # Bottom
            pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20]
        else: # Left
            pos = [-20, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
        
        health = 1 + (self.wave_number - 1) // 5
        self.active_enemies.append({
            'pos': np.array(pos, dtype=float),
            'health': health,
            'max_health': health,
            'speed': self.np_random.uniform(0.6, 1.0)
        })

    def _update_turrets(self):
        for turret in self.turrets:
            turret['fire_cooldown'] = max(0, turret['fire_cooldown'] - 1)
            if turret['fire_cooldown'] > 0:
                continue

            target = None
            min_dist = turret['range'] ** 2
            for enemy in self.active_enemies:
                dist_sq = sum((ep - tp)**2 for ep, tp in zip(enemy['pos'], turret['pos']))
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy
            
            if target:
                # SFX: Turret Fire
                self.projectiles.append({
                    'start_pos': np.array(turret['pos'], dtype=float),
                    'pos': np.array(turret['pos'], dtype=float),
                    'target': target,
                    'damage': turret['damage'],
                    'speed': 8.0
                })
                turret['fire_cooldown'] = turret['fire_rate']
                self._create_particles(turret['pos'], (255, 255, 200), 5, 1.0, count=3)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj['target'] not in self.active_enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj['target']['pos']
            direction = target_pos - proj['pos']
            dist = np.linalg.norm(direction)

            if dist < proj['speed']:
                # SFX: Projectile Hit
                proj['target']['health'] -= proj['damage']
                self._create_particles(target_pos, (255, 255, 255), 10, 1.5)
                self.projectiles.remove(proj)
            else:
                proj['pos'] += (direction / dist) * proj['speed']

    def _update_enemies(self):
        reward = 0.0
        wave_cleared_reward = 0.0
        for enemy in self.active_enemies[:]:
            if enemy['health'] <= 0:
                # SFX: Enemy Death
                self._create_particles(enemy['pos'], (50, 30, 80), 30, 4.0, 'shadow')
                self.active_enemies.remove(enemy)
                reward += 1.0
                continue

            direction = np.array(self.BASE_CRYSTAL_POS) - enemy['pos']
            dist = np.linalg.norm(direction)
            if dist < 15:
                # SFX: Base Hit
                self.base_crystal_health -= 1
                self.screen_shake = 10
                self.active_enemies.remove(enemy)
                self._create_particles(self.BASE_CRYSTAL_POS, (255, 80, 80), 25, 3.0)
            else:
                enemy['pos'] += (direction / dist) * enemy['speed']
        
        if self.enemies_spawned_this_wave >= self.enemies_to_spawn_this_wave and len(self.active_enemies) == 0:
            if not self._check_termination(): # Don't start a new wave if game is over
                # SFX: Wave Cleared
                self.wave_number += 1
                wave_cleared_reward = 5.0
                self._setup_wave()
        return reward, wave_cleared_reward

    def _update_particles(self):
        self.screen_shake = max(0, self.screen_shake - 1)
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['size'] *= 0.97
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_y = -1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == 0 and empty_y == -1:
                    empty_y = y
                elif self.grid[x, y] != 0 and empty_y != -1:
                    self.grid[x, empty_y] = self.grid[x, y]
                    self.grid[x, y] = 0
                    empty_y -= 1
        
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] == 0:
                    self.grid[x, y] = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1)
    
    def _find_match_group(self, start_x, start_y):
        if self.grid[start_x, start_y] == 0:
            return []
        
        target_color = self.grid[start_x, start_y]
        q = [(start_x, start_y)]
        visited = set(q)
        match_group = []

        while q:
            x, y = q.pop(0)
            match_group.append((x, y))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[nx, ny] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return match_group

    def _setup_wave(self):
        self.wave_timer = 120 # 4 seconds initial delay
        self.enemies_to_spawn_this_wave = 3 + self.wave_number * 2
        self.enemies_spawned_this_wave = 0
    
    def _populate_grid(self):
        self.grid = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        # Ensure no accidental starting matches
        for _ in range(3): # Run a few times to clear most matches
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT):
                    group = self._find_match_group(x, y)
                    if len(group) >= self.MIN_MATCH_SIZE:
                        for gx, gy in group:
                            self.grid[gx, gy] = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1)

    # --- Gym Interface Helpers ---

    def _get_observation(self):
        render_offset = (0, 0)
        if self.screen_shake > 0:
            render_offset = (self.np_random.integers(-self.screen_shake, self.screen_shake + 1),
                             self.np_random.integers(-self.screen_shake, self.screen_shake + 1))
        
        self.screen.fill(self.COLOR_BG)
        self._render_game(render_offset)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "wave": self.wave_number,
            "base_health": self.base_crystal_health,
        }

    def _check_termination(self):
        if self.base_crystal_health <= 0:
            self.game_over = True
            return True
        if self.wave_number > self.MAX_WAVES:
            self.win_condition_met = True
            self.game_over = True
            return True
        return False

    # --- Rendering Helpers ---

    def _render_game(self, offset):
        self._draw_base_crystal(offset)
        for turret in self.turrets:
            self._draw_turret(turret, offset)
        self._draw_grid(offset)
        self._draw_cursor(offset)
        
        for p in self.projectiles:
            self._draw_projectile(p, offset)
        for e in self.active_enemies:
            self._draw_enemy(e, offset)
        for p in self.particles:
            self._draw_particle(p, offset)
        
        if self.game_over:
            self._draw_game_over()

    def _render_ui(self):
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))

        texts = [
            f"WAVE: {self.wave_number}/{self.MAX_WAVES}",
            f"ENERGY: {self.energy}",
            f"BASE: {self.base_crystal_health}/{self.INITIAL_BASE_HEALTH}",
            f"SCORE: {int(self.score)}"
        ]
        for i, text in enumerate(texts):
            text_surf = self.font_small.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(text_surf, (10 + i * 150, 12))

    def _draw_grid(self, offset):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] != 0:
                    color_idx = self.grid[x, y] - 1
                    px, py = self._grid_to_pixel(x, y)
                    self._draw_crystal((px + offset[0], py + offset[1]), self.CRYSTAL_COLORS[color_idx])

    def _draw_crystal(self, pos, color):
        x, y = int(pos[0]), int(pos[1])
        radius = self.CELL_SIZE // 2 - 4
        glow_color = (*color, 60)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius + 3, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _draw_cursor(self, offset):
        x, y = self.cursor_pos
        px, py = self._grid_to_pixel(x, y)
        rect = pygame.Rect(px - self.CELL_SIZE//2 + offset[0], py - self.CELL_SIZE//2 + offset[1], 
                           self.CELL_SIZE, self.CELL_SIZE)
        
        match_group = self._find_match_group(x, y)
        color = self.COLOR_CURSOR if len(match_group) >= self.MIN_MATCH_SIZE else self.COLOR_CURSOR_INVALID
        
        pygame.draw.rect(self.screen, color, rect, 2, border_radius=4)
        for gx, gy in match_group:
            gpx, gpy = self._grid_to_pixel(gx, gy)
            highlight_rect = pygame.Rect(gpx - self.CELL_SIZE//2 + offset[0], gpy - self.CELL_SIZE//2 + offset[1], 
                                         self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, (*color, 80), highlight_rect, 0, border_radius=4)

    def _draw_base_crystal(self, offset):
        x, y = self.BASE_CRYSTAL_POS[0] + offset[0], self.BASE_CRYSTAL_POS[1] + offset[1]
        health_ratio = self.base_crystal_health / self.INITIAL_BASE_HEALTH
        
        pulse = (math.sin(self.steps * 0.05) + 1) / 2 * 5
        radius = 20 + pulse
        color = (180, 220, 255)
        glow_color = (*color, int(80 * health_ratio))

        pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius + 10), glow_color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, x, y, int(radius), color)

    def _draw_turret(self, turret, offset):
        x, y = int(turret['pos'][0] + offset[0]), int(turret['pos'][1] + offset[1])
        level = turret['level']
        is_selected = self.turrets.index(turret) == self.selected_turret_for_upgrade
        
        base_color = (200, 200, 220)
        top_color = (255, 255, 100)
        
        pygame.draw.circle(self.screen, base_color, (x, y), 10 + level)
        pygame.draw.circle(self.screen, top_color, (x, y), 5 + level)

        if is_selected:
            pygame.gfxdraw.aacircle(self.screen, x, y, 18 + level, (255, 255, 255, 150))
            pygame.gfxdraw.aacircle(self.screen, x, y, 19 + level, (255, 255, 255, 150))

    def _draw_enemy(self, enemy, offset):
        x, y = int(enemy['pos'][0] + offset[0]), int(enemy['pos'][1] + offset[1])
        size = 10
        points = []
        for i in range(5):
            angle = i * (2 * math.pi / 5) + self.steps * 0.1
            px = x + math.cos(angle) * size
            py = y + math.sin(angle) * size
            points.append((int(px), int(py)))
        
        shadow_color = (50, 30, 80)
        glow_color = (*shadow_color, 100)
        pygame.gfxdraw.filled_polygon(self.screen, points, glow_color)
        pygame.gfxdraw.filled_polygon(self.screen, points, shadow_color)
        
        # Health bar
        if enemy['health'] < enemy['max_health']:
            hp_ratio = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, (255, 0, 0), (x - 10, y - 15, 20, 3))
            pygame.draw.rect(self.screen, (0, 255, 0), (x - 10, y - 15, 20 * hp_ratio, 3))

    def _draw_projectile(self, projectile, offset):
        x, y = int(projectile['pos'][0] + offset[0]), int(projectile['pos'][1] + offset[1])
        color = (255, 255, 180)
        pygame.gfxdraw.filled_circle(self.screen, x, y, 4, (*color, 80))
        pygame.gfxdraw.filled_circle(self.screen, x, y, 2, color)
    
    def _draw_particle(self, p, offset):
        x, y = int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1])
        size = int(p['size'])
        if size > 0:
            alpha = int(max(0, min(255, p['lifetime'] * 10)))
            color = (*p['color'], alpha)
            if p['type'] == 'circle':
                pygame.gfxdraw.filled_circle(self.screen, x, y, size, color)
            elif p['type'] == 'shadow':
                points = []
                for i in range(3):
                    angle = i * (2 * math.pi / 3) + p['lifetime'] * 0.5
                    px = x + math.cos(angle) * size
                    py = y + math.sin(angle) * size
                    points.append((int(px), int(py)))
                if len(points) == 3:
                     pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        
        text = "VICTORY" if self.win_condition_met else "GAME OVER"
        text_surf = self.font_large.render(text, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        
        s.blit(text_surf, text_rect)
        self.screen.blit(s, (0, 0))

    def _grid_to_pixel(self, x, y):
        px = self.GRID_START_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_START_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
        return px, py

    def _create_particles(self, pos, color, num_particles, max_speed, p_type='circle', count=1):
        for _ in range(num_particles * count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'color': color,
                'lifetime': self.np_random.integers(20, 40),
                'size': self.np_random.uniform(2, 5),
                'type': p_type
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Caverns Defense")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # [movement, space, shift]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        keys = pygame.key.get_pressed()
        
        # Action mapping
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            pygame.time.wait(3000)
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()