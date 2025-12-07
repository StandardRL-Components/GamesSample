
# Generated: 2025-08-28T00:19:50.630728
# Source Brief: brief_03758.md
# Brief Index: 3758

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game objects
class Enemy:
    def __init__(self, path_nodes, health, speed, value, color):
        self.path_nodes = path_nodes
        self.health = self.max_health = health
        self.speed = speed
        self.value = value
        self.color = color
        self.path_index = 0
        self.progress = 0.0
        self.pos = self.path_nodes[0]
        self.alive = True

    def update(self, dt):
        if not self.alive:
            return

        if self.path_index >= len(self.path_nodes) - 1:
            return # Reached the end

        start_node = self.path_nodes[self.path_index]
        end_node = self.path_nodes[self.path_index + 1]
        
        segment_vec = (end_node[0] - start_node[0], end_node[1] - start_node[1])
        segment_len = math.sqrt(segment_vec[0]**2 + segment_vec[1]**2)
        if segment_len == 0:
            self.progress = 1.0
        else:
            self.progress += self.speed * dt / segment_len

        if self.progress >= 1.0:
            self.path_index += 1
            if self.path_index >= len(self.path_nodes) - 1:
                self.progress = 1.0 # Clamp at end
            else:
                self.progress -= 1.0
        
        start_node = self.path_nodes[self.path_index]
        end_node = self.path_nodes[min(self.path_index + 1, len(self.path_nodes) - 1)]
        
        self.pos = (
            start_node[0] + (end_node[0] - start_node[0]) * self.progress,
            start_node[1] + (end_node[1] - start_node[1]) * self.progress
        )

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.alive = False
            return True
        return False

class Tower:
    def __init__(self, grid_pos, stats):
        self.grid_pos = grid_pos
        self.range = stats['range']
        self.damage = stats['damage']
        self.fire_rate = stats['fire_rate']
        self.color = stats['color']
        self.projectile_speed = stats['projectile_speed']
        self.cooldown = 0.0

    def update(self, dt, enemies):
        self.cooldown = max(0.0, self.cooldown - dt)
        if self.cooldown > 0:
            return None

        target = self.find_target(enemies)
        if target:
            self.cooldown = 1.0 / self.fire_rate
            # SFX: Laser shoot
            return Projectile(self.grid_pos, target, self.damage, self.projectile_speed)
        return None

    def find_target(self, enemies):
        for enemy in enemies:
            if not enemy.alive:
                continue
            dist_sq = (self.grid_pos[0] - enemy.pos[0])**2 + (self.grid_pos[1] - enemy.pos[1])**2
            if dist_sq <= self.range**2:
                return enemy
        return None

class Projectile:
    def __init__(self, start_grid_pos, target, damage, speed):
        self.pos = start_grid_pos
        self.target = target
        self.damage = damage
        self.speed = speed

    def update(self, dt):
        if not self.target.alive:
            return True # Mark for deletion if target is dead

        target_pos = self.target.pos
        direction = (target_pos[0] - self.pos[0], target_pos[1] - self.pos[1])
        dist = math.sqrt(direction[0]**2 + direction[1]**2)
        
        if dist < 0.5: # Hit
            return True

        move_dist = self.speed * dt
        if dist < move_dist: # Prevent overshooting
            self.pos = target_pos
        else:
            self.pos = (
                self.pos[0] + direction[0] / dist * move_dist,
                self.pos[1] + direction[1] / dist * move_dist
            )
        return False

class Particle:
    def __init__(self, pos, vel, life, color, size):
        self.pos = list(pos)
        self.vel = list(vel)
        self.life = life
        self.max_life = life
        self.color = color
        self.size = size

    def update(self, dt):
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        self.life -= dt
        return self.life <= 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move the cursor. Press space to place a tower. Press shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in this isometric tower defense game."
    )

    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 22, 14
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 20, 10
    ORIGIN_X, ORIGIN_Y = SCREEN_WIDTH // 2, 80
    MAX_STEPS = 30 * 60 * 2 # 2 minutes at 30fps
    
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 50, 60)
    COLOR_PATH = (60, 70, 80)
    COLOR_BASE = (0, 150, 0)
    COLOR_BASE_DAMAGED = (150, 150, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GOLD = (255, 215, 0)
    COLOR_WAVE = (180, 180, 255)
    
    TOWER_SPECS = [
        {'name': 'Gatling', 'cost': 50, 'range': 3, 'damage': 5, 'fire_rate': 4, 'projectile_speed': 10, 'color': (0, 180, 255)},
        {'name': 'Cannon', 'cost': 120, 'range': 4, 'damage': 30, 'fire_rate': 1, 'projectile_speed': 7, 'color': (255, 120, 0)},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", 20, bold=True)

        self._define_path()
        self.reset()
        self.validate_implementation()
    
    def _define_path(self):
        self.path_nodes_grid = [
            (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (5, 2),
            (6, 2), (7, 2), (8, 2), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7),
            (10, 7), (11, 7), (12, 7), (13, 7), (14, 7), (15, 7), (15, 8), (15, 9),
            (15, 10), (14, 10), (13, 10), (12, 10), (11, 10), (11, 11), (11, 12),
            (12, 12), (13, 12), (14, 12), (15, 12), (16, 12), (17, 12), (18, 12),
            (19, 12), (20, 12), (21, 12)
        ]
        self.path_nodes_world = [self._iso_to_screen(p[0], p[1]) for p in self.path_nodes_grid]
        self.path_set = set(self.path_nodes_grid)
        self.base_pos_grid = self.path_nodes_grid[-1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = 100
        self.gold = 100
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.current_wave_index = -1
        self.wave_spawn_index = 0
        self.wave_timer = 0
        self.inter_wave_timer = 2.0 # Time before first wave
        self._init_waves()
        
        return self._get_observation(), self._get_info()

    def _init_waves(self):
        self.waves = []
        base_health = 10
        base_speed = 1.0
        num_enemies = 3
        for i in range(10):
            wave_enemies = []
            for _ in range(num_enemies + i * 2):
                # type, health, speed, value, color
                wave_enemies.append(('normal', base_health, base_speed, 5, (200, 50, 50)))
            self.waves.append(wave_enemies)
            base_health *= 1.10
            base_speed *= 1.05

    def step(self, action):
        dt = self.clock.tick(30) / 1000.0
        self.steps += 1
        reward = 0.01 # Small reward for surviving

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        
        tower_spec = self.TOWER_SPECS[self.selected_tower_type]
        can_place = self._is_valid_placement(self.cursor_pos) and self.gold >= tower_spec['cost']
        if space_held and not self.prev_space_held and can_place:
            self.gold -= tower_spec['cost']
            self.towers.append(Tower(tuple(self.cursor_pos), tower_spec))
            # SFX: Tower placed
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game State ---
        self._update_waves(dt)
        
        for t in self.towers:
            new_proj = t.update(dt, self.enemies)
            if new_proj:
                self.projectiles.append(new_proj)
        
        for p in self.projectiles[:]:
            if p.update(dt):
                if p.target.alive:
                    is_fatal = p.target.take_damage(p.damage)
                    # SFX: Enemy hit
                    self._create_particles(p.target.pos, 10, (255, 255, 100))
                    if is_fatal:
                        self.gold += p.target.value
                        reward += 1.0
                        # SFX: Enemy destroyed
                        self._create_particles(p.target.pos, 30, (255, 100, 100))
                self.projectiles.remove(p)
        
        for e in self.enemies[:]:
            e.update(dt)
            if e.path_index >= len(self.path_nodes_grid) - 1:
                self.base_health = max(0, self.base_health - 10)
                reward -= 1.0
                e.alive = False
                # SFX: Base takes damage
        
        for p in self.particles[:]:
            if p.update(dt):
                self.particles.remove(p)
        
        self.enemies = [e for e in self.enemies if e.alive]

        # --- Termination Check ---
        if self.base_health <= 0:
            self.game_over = True
            reward = -100.0
        elif self.win:
            self.game_over = True
            reward = 100.0
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_waves(self, dt):
        if self.win: return

        is_wave_active = self.current_wave_index < len(self.waves)
        
        if is_wave_active and not self.enemies and self.wave_spawn_index >= len(self.waves[self.current_wave_index]):
            # Wave complete
            if self.current_wave_index == len(self.waves) - 1:
                self.win = True
                return
            self.inter_wave_timer = 5.0 # Time between waves
            self.current_wave_index = -2 # Special state for inter-wave
        
        if self.inter_wave_timer > 0:
            self.inter_wave_timer -= dt
            if self.inter_wave_timer <= 0:
                self.current_wave_index = self.current_wave_index + 1 if self.current_wave_index != -2 else 0
                self.wave_spawn_index = 0
                self.wave_timer = 0
        
        if is_wave_active and self.wave_spawn_index < len(self.waves[self.current_wave_index]):
            self.wave_timer -= dt
            if self.wave_timer <= 0:
                self.wave_timer = 0.5 # Time between enemies
                enemy_data = self.waves[self.current_wave_index][self.wave_spawn_index]
                self.enemies.append(Enemy(self.path_nodes_world, enemy_data[1], enemy_data[2], enemy_data[3], enemy_data[4]))
                self.wave_spawn_index += 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Grid
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                x, y = self._iso_to_screen(c, r)
                pygame.gfxdraw.pixel(self.screen, int(x), int(y), self.COLOR_GRID)

        # Path
        if len(self.path_nodes_world) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_PATH, False, self.path_nodes_world, 10)
        
        # Base
        base_x, base_y = self._iso_to_screen(*self.base_pos_grid)
        base_color = self.COLOR_BASE if self.base_health > 30 else self.COLOR_BASE_DAMAGED
        pygame.draw.circle(self.screen, base_color, (int(base_x), int(base_y)), 15)
        pygame.draw.circle(self.screen, (255,255,255), (int(base_x), int(base_y)), 15, 1)

        # Towers
        for t in self.towers:
            tx, ty = self._iso_to_screen(*t.grid_pos)
            pygame.draw.rect(self.screen, t.color, (tx - 8, ty - 12, 16, 16))
            pygame.draw.rect(self.screen, (255,255,255), (tx - 8, ty - 12, 16, 16), 1)

        # Cursor
        cursor_spec = self.TOWER_SPECS[self.selected_tower_type]
        can_place = self._is_valid_placement(self.cursor_pos) and self.gold >= cursor_spec['cost']
        cursor_color = (0, 255, 0) if can_place else (255, 0, 0)
        cx, cy = self._iso_to_screen(*self.cursor_pos)
        
        # Range indicator
        range_px = cursor_spec['range'] * self.TILE_WIDTH_HALF * 1.414
        s = pygame.Surface((range_px*2, range_px*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*cursor_color, 50), (range_px, range_px), range_px)
        self.screen.blit(s, (cx - range_px, cy - range_px))
        
        # Cursor shape
        pygame.draw.rect(self.screen, cursor_color, (cx - 8, cy - 12, 16, 16), 2)

        # Projectiles
        for p in self.projectiles:
            px, py = self._iso_to_screen(*p.pos)
            pygame.draw.circle(self.screen, (255, 255, 0), (int(px), int(py)), 3)

        # Enemies
        for e in self.enemies:
            ex, ey = self._iso_to_screen(*e.pos)
            size = 8
            pygame.draw.rect(self.screen, e.color, (ex - size/2, ey - size, size, size))
            # Health bar
            health_pct = e.health / e.max_health
            pygame.draw.rect(self.screen, (50, 50, 50), (ex - size, ey - size - 8, size*2, 4))
            pygame.draw.rect(self.screen, (0, 255, 0), (ex - size, ey - size - 8, size*2 * health_pct, 4))

        # Particles
        for p in self.particles:
            px, py = self._iso_to_screen(*p.pos)
            alpha = int(255 * (p.life / p.max_life))
            color = (*p.color, alpha)
            s = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p.size, p.size), p.size)
            self.screen.blit(s, (px - p.size, py - p.size))
            
    def _render_ui(self):
        # Base Health Bar
        base_x, base_y = self._iso_to_screen(*self.base_pos_grid)
        health_pct = self.base_health / 100
        bar_width = 50
        pygame.draw.rect(self.screen, (50, 50, 50), (base_x - bar_width/2, base_y - 30, bar_width, 8))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (base_x - bar_width/2, base_y - 30, bar_width * health_pct, 8))
        
        # Top-left UI Panel
        panel_rect = pygame.Rect(10, 10, 200, 75)
        pygame.draw.rect(self.screen, (0,0,0,150), panel_rect, border_radius=5)
        
        gold_text = self.font_m.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (20, 15))
        
        wave_str = f"Wave: {self.current_wave_index+1}/{len(self.waves)}" if self.current_wave_index >= 0 else "Starting..."
        if self.win: wave_str = "You Win!"
        elif self.game_over: wave_str = "Game Over"
        wave_text = self.font_s.render(wave_str, True, self.COLOR_WAVE)
        self.screen.blit(wave_text, (20, 40))

        tower_spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_text = self.font_s.render(f"Placing: {tower_spec['name']} (${tower_spec['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (20, 60))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.current_wave_index + 1 if self.current_wave_index >= 0 else 0,
        }

    def _iso_to_screen(self, x, y):
        sx = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        sy = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return sx, sy

    def _is_valid_placement(self, grid_pos):
        if not (0 <= grid_pos[0] < self.GRID_WIDTH and 0 <= grid_pos[1] < self.GRID_HEIGHT):
            return False
        if tuple(grid_pos) in self.path_set:
            return False
        for tower in self.towers:
            if tower.grid_pos == tuple(grid_pos):
                return False
        return True

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            life = random.uniform(0.2, 0.5)
            size = random.randint(1, 3)
            self.particles.append(Particle(pos, vel, life, color, size))

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
    # To play the game manually
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'windows' or 'x11' or 'dummy'
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Isometric Tower Defense")
    
    running = True
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    print(env.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Movement
        up = keys[pygame.K_UP]
        down = keys[pygame.K_DOWN]
        left = keys[pygame.K_LEFT]
        right = keys[pygame.K_RIGHT]
        
        movement = 0
        if up: movement = 1
        elif down: movement = 2
        elif left: movement = 3
        elif right: movement = 4

        # Buttons
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

    pygame.quit()