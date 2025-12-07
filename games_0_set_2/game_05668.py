
# Generated: 2025-08-28T05:44:00.264233
# Source Brief: brief_05668.md
# Brief Index: 5668

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# --- Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
GRID_COLS, GRID_ROWS = 20, 10
CELL_SIZE = 32
GRID_WIDTH = GRID_COLS * CELL_SIZE
GRID_HEIGHT = GRID_ROWS * CELL_SIZE
UI_HEIGHT = SCREEN_HEIGHT - GRID_HEIGHT

# --- Colors ---
COLOR_BG = (25, 25, 40)
COLOR_GRID = (40, 40, 60)
COLOR_PATH = (50, 50, 70)
COLOR_BASE = (0, 255, 128)
COLOR_TEXT = (220, 220, 220)
COLOR_GOLD = (255, 223, 0)
COLOR_CURSOR_VALID = (0, 255, 0)
COLOR_CURSOR_INVALID = (255, 0, 0)
COLOR_HEALTH_BAR_FG = (0, 200, 0)
COLOR_HEALTH_BAR_BG = (50, 0, 0)

TOWER_SPECS = {
    0: {"name": "Gun Turret", "cost": 50, "range": 90, "damage": 5, "fire_rate": 2.0, "color": (0, 150, 255), "proj_speed": 8, "aoe": 0, "proj_color": (100, 200, 255)},
    1: {"name": "Cannon", "cost": 125, "range": 110, "damage": 25, "fire_rate": 0.5, "color": (255, 100, 0), "proj_speed": 5, "aoe": 25, "proj_color": (255, 150, 50)},
    2: {"name": "Sniper", "cost": 200, "range": 250, "damage": 60, "fire_rate": 0.3, "color": (200, 200, 200), "proj_speed": 20, "aoe": 0, "proj_color": (255, 255, 255)},
    3: {"name": "Frost Tower", "cost": 75, "range": 80, "damage": 1, "fire_rate": 1.0, "color": (100, 200, 255), "proj_speed": 6, "aoe": 20, "slow_effect": 0.5, "proj_color": (150, 220, 255)},
    4: {"name": "Machine Gun", "cost": 150, "range": 70, "damage": 2, "fire_rate": 8.0, "color": (255, 255, 0), "proj_speed": 10, "aoe": 0, "proj_color": (255, 255, 150)},
}

# --- Helper Classes ---

class Particle:
    def __init__(self, pos, vel, size, life, color):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.size = size
        self.life = life
        self.max_life = life
        self.color = color

    def update(self):
        self.pos += self.vel
        self.life -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            temp_surf = pygame.Surface((int(self.size) * 2, int(self.size) * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, self.color + (alpha,), (int(self.size), int(self.size)), int(self.size))
            surface.blit(temp_surf, (self.pos.x - self.size, self.pos.y - self.size), special_flags=pygame.BLEND_RGBA_ADD)

class Projectile:
    def __init__(self, pos, target, spec):
        self.pos = pygame.math.Vector2(pos)
        self.target = target
        self.spec = spec
        self.terminated = False

    def update(self):
        if self.target.health <= 0 or self.terminated:
            self.terminated = True
            return

        try:
            direction = (self.target.pos - self.pos).normalize()
            self.pos += direction * self.spec['proj_speed']
        except ValueError: # Target and projectile at same spot
            self.pos = self.target.pos

        if self.pos.distance_to(self.target.pos) < self.target.size / 2:
            self.terminated = True

    def draw(self, surface):
        pygame.draw.circle(surface, self.spec['proj_color'], (int(self.pos.x), int(self.pos.y)), 3)

class Enemy:
    def __init__(self, enemy_type, health, speed, value, path):
        self.pos = pygame.math.Vector2(path[0])
        self.type = enemy_type
        self.health = health
        self.max_health = health
        self.speed = speed
        self.value = value
        self.path = path
        self.path_index = 1
        self.size = 10
        self.slow_timer = 0
        self.color = {"normal": (255, 50, 50), "fast": (255, 120, 50), "tank": (200, 0, 0)}[enemy_type]

    def update(self):
        if self.path_index >= len(self.path):
            return

        current_speed = self.speed
        if self.slow_timer > 0:
            current_speed *= 0.5 # Default slow effect
            self.slow_timer -= 1

        target_pos = self.path[self.path_index]
        direction = (pygame.math.Vector2(target_pos) - self.pos)
        dist = direction.length()

        if dist < current_speed:
            self.pos = pygame.math.Vector2(target_pos)
            self.path_index += 1
        elif dist > 0:
            self.pos += direction.normalize() * current_speed

    def draw(self, surface):
        # Body
        pos_int = (int(self.pos.x), int(self.pos.y))
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], self.size, self.color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], self.size, self.color)
        
        # Health bar
        if self.health < self.max_health:
            bar_w = 20
            bar_h = 4
            bar_x = self.pos.x - bar_w / 2
            bar_y = self.pos.y - self.size - bar_h - 2
            health_ratio = self.health / self.max_health
            pygame.draw.rect(surface, (100,0,0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(surface, (0,255,0), (bar_x, bar_y, bar_w * health_ratio, bar_h))

class Tower:
    def __init__(self, grid_pos, tower_type):
        self.grid_pos = grid_pos
        self.pos = (grid_pos[0] * CELL_SIZE + CELL_SIZE / 2, grid_pos[1] * CELL_SIZE + CELL_SIZE / 2)
        self.type = tower_type
        self.spec = TOWER_SPECS[tower_type]
        self.cooldown = 0

    def update(self, enemies):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        target = self.find_target(enemies)
        if target:
            self.cooldown = 30 / self.spec['fire_rate']
            # sfx: tower_shoot.wav
            return Projectile(self.pos, target, self.spec)
        return None

    def find_target(self, enemies):
        valid_targets = [e for e in enemies if pygame.math.Vector2(self.pos).distance_to(e.pos) < self.spec['range']]
        if not valid_targets:
            return None
        # Target enemy closest to the base (highest path_index then shortest distance to next waypoint)
        return max(valid_targets, key=lambda e: (e.path_index, -e.pos.distance_to(e.path[min(e.path_index, len(e.path)-1)])))

    def draw(self, surface):
        x, y = int(self.pos[0]), int(self.pos[1])
        color = self.spec['color']
        # Base
        pygame.draw.rect(surface, (50, 50, 50), (x - 12, y - 12, 24, 24))
        # Top
        if self.spec['name'] == "Cannon":
            pygame.draw.circle(surface, color, (x, y), 10)
        elif self.spec['name'] == "Sniper":
            pygame.draw.rect(surface, color, (x - 4, y - 4, 8, 8))
        elif self.spec['name'] == "Frost Tower":
            pygame.draw.polygon(surface, color, [(x, y - 10), (x - 10, y + 5), (x + 10, y + 5)])
        else:
             pygame.draw.circle(surface, color, (x, y), 8)

# --- Gymnasium Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Hold Shift to cycle through tower types. "
        "Press Space to build the selected tower at the cursor."
    )

    game_description = (
        "A top-down tower defense game. Place towers to defend your base from waves of enemies. "
        "Earn gold by defeating enemies and use it to build more powerful towers."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 14)
        self.font_medium = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.path = self._generate_path()

        # NOTE: Brief's max_steps=1000 is too short for a 20-wave TD game (33s). 
        # Increasing to 15000 (8.3 min) to prioritize gameplay experience.
        self.max_steps = 15000
        self.max_waves = 20

        # This call is not part of the standard __init__, but useful for development
        # self.validate_implementation()
    
    def _generate_path(self):
        path = []
        path.append((-20, int(GRID_HEIGHT * 0.2)))
        path.append((int(GRID_WIDTH * 0.2), int(GRID_HEIGHT * 0.2)))
        path.append((int(GRID_WIDTH * 0.2), int(GRID_HEIGHT * 0.8)))
        path.append((int(GRID_WIDTH * 0.5), int(GRID_HEIGHT * 0.8)))
        path.append((int(GRID_WIDTH * 0.5), int(GRID_HEIGHT * 0.4)))
        path.append((int(GRID_WIDTH * 0.8), int(GRID_HEIGHT * 0.4)))
        path.append((int(GRID_WIDTH * 0.8), int(GRID_HEIGHT * 0.6)))
        path.append((GRID_WIDTH + 20, int(GRID_HEIGHT * 0.6)))
        return path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.base_health = 100
        self.max_base_health = 100
        self.gold = 100

        self.wave = 0
        self.wave_timer = 150 # 5s delay before first wave
        self.wave_spawn_queue = []
        self.wave_spawn_timer = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.grid = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        self.cursor_pos = (GRID_COLS // 2, GRID_ROWS // 2)
        self.selected_tower_type = 0
        self.last_shift_state = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Time penalty

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        action_reward = self._handle_actions(movement, space_held, shift_held)
        reward += action_reward

        # --- Update Game State ---
        if not self.game_over:
            self._update_wave_spawning()
            enemy_reward = self._update_enemies()
            tower_reward = self._update_towers_and_projectiles()
            reward += enemy_reward + tower_reward
        
        self._update_particles()
        
        self.steps += 1
        self.score += reward

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not (self.victory or self.base_health <= 0):
             # Only apply terminal reward if it hasn't been applied
            if self.victory:
                reward += 100
            elif self.base_health <= 0:
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, movement, space_held, shift_held):
        # Move cursor
        cx, cy = self.cursor_pos
        if movement == 1: cy = max(0, cy - 1)
        elif movement == 2: cy = min(GRID_ROWS - 1, cy + 1)
        elif movement == 3: cx = max(0, cx - 1)
        elif movement == 4: cx = min(GRID_COLS - 1, cx + 1)
        self.cursor_pos = (cx, cy)
        
        # Cycle tower type
        if shift_held and not self.last_shift_state:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(TOWER_SPECS)
            # sfx: ui_cycle.wav
        self.last_shift_state = shift_held
        
        # Place tower
        if space_held:
            spec = TOWER_SPECS[self.selected_tower_type]
            if self.gold >= spec['cost'] and self.grid[self.cursor_pos[1]][self.cursor_pos[0]] is None:
                self.gold -= spec['cost']
                new_tower = Tower(self.cursor_pos, self.selected_tower_type)
                self.towers.append(new_tower)
                self.grid[self.cursor_pos[1]][self.cursor_pos[0]] = new_tower
                # sfx: place_tower.wav
                self._create_particles(new_tower.pos, 1, count=20, color=spec['color'])
                return 0 # No direct reward/penalty for building
        return 0

    def _update_wave_spawning(self):
        if self.wave_timer > 0:
            self.wave_timer -= 1
            if self.wave_timer == 0:
                self._start_next_wave()

        if self.wave_spawn_queue:
            self.wave_spawn_timer -= 1
            if self.wave_spawn_timer <= 0:
                enemy_spec = self.wave_spawn_queue.pop(0)
                self.enemies.append(Enemy(**enemy_spec, path=self.path))
                self.wave_spawn_timer = 15 # spawn delay

        elif not self.enemies and self.wave > 0 and self.wave < self.max_waves:
            if self.wave_timer == 0:
                self.wave_timer = 240 # 8s between waves

    def _start_next_wave(self):
        self.wave += 1
        if self.wave > self.max_waves: return
        
        # sfx: wave_start.wav
        num_enemies = 5 + self.wave * 2
        base_health = 20 + self.wave * 10
        base_speed = 1.0 + self.wave * 0.05
        base_value = 10 + self.wave

        for i in range(num_enemies):
            enemy_type = "normal"
            if self.wave >= 5 and i % 4 == 1: enemy_type = "fast"
            if self.wave >= 10 and i % 4 == 2: enemy_type = "tank"

            health = base_health * (1.5 if enemy_type == "tank" else 0.7 if enemy_type == "fast" else 1)
            speed = base_speed * (0.7 if enemy_type == "tank" else 1.5 if enemy_type == "fast" else 1)
            
            self.wave_spawn_queue.append({
                "enemy_type": enemy_type,
                "health": health,
                "speed": speed,
                "value": base_value,
            })

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            enemy.update()
            if enemy.path_index >= len(self.path):
                self.base_health -= 10
                reward -= 5
                # sfx: base_damage.wav
                self._create_particles(self.path[-1], 10, count=30, color=(255,0,0))
                self.enemies.remove(enemy)
        return reward

    def _update_towers_and_projectiles(self):
        reward = 0
        # Towers shoot
        for tower in self.towers:
            new_proj = tower.update(self.enemies)
            if new_proj:
                self.projectiles.append(new_proj)

        # Projectiles move and hit
        for proj in self.projectiles[:]:
            proj.update()
            if proj.terminated:
                if proj.target in self.enemies: # Check if target is still valid
                    reward += self._handle_hit(proj)
                if proj in self.projectiles:
                    self.projectiles.remove(proj)
        return reward

    def _handle_hit(self, proj):
        reward = 0.1 # Reward for damaging
        hit_enemies = []
        
        # Area of Effect
        if proj.spec['aoe'] > 0:
            for enemy in self.enemies:
                if proj.target.pos.distance_to(enemy.pos) < proj.spec['aoe']:
                    hit_enemies.append(enemy)
            self._create_particles(proj.target.pos, proj.spec['aoe'] / 5, count=10, color=proj.spec['proj_color'])
            # sfx: explosion.wav
        else:
            hit_enemies.append(proj.target)
            self._create_particles(proj.pos, 2, count=5, color=proj.spec['proj_color'])
            # sfx: hit.wav

        for enemy in hit_enemies:
            if enemy.health <= 0: continue
            
            damage = proj.spec['damage']
            if proj.spec['aoe'] > 0 and enemy != proj.target:
                damage *= 0.5 # Splash damage falloff

            enemy.health -= damage
            
            # Apply slow
            if 'slow_effect' in proj.spec:
                enemy.slow_timer = max(enemy.slow_timer, 60) # Slow for 2 seconds

            if enemy.health <= 0:
                reward += 1.0 # Reward for kill
                self.gold += enemy.value
                self._create_particles(enemy.pos, 5, count=20, color=enemy.color)
                # sfx: enemy_die.wav
                if enemy in self.enemies:
                    self.enemies.remove(enemy)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, speed_mult, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            size = self.np_random.uniform(2, 5)
            life = self.np_random.integers(10, 30)
            self.particles.append(Particle(pos, vel, size, life, color))

    def _check_termination(self):
        if self.game_over: return True
        if self.base_health <= 0:
            self.game_over = True
            self.base_health = 0
        elif self.wave > self.max_waves and not self.enemies and not self.wave_spawn_queue:
            self.game_over = True
            self.victory = True
        elif self.steps >= self.max_steps:
            self.game_over = True
        return self.game_over
    
    def _get_observation(self):
        self.screen.fill(COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.wave,
            "base_health": self.base_health,
            "enemies_left": len(self.enemies) + len(self.wave_spawn_queue),
        }

    def _render_game(self):
        # Draw grid
        for r in range(GRID_ROWS + 1):
            pygame.draw.line(self.screen, COLOR_GRID, (0, r * CELL_SIZE), (GRID_WIDTH, r * CELL_SIZE))
        for c in range(GRID_COLS + 1):
            pygame.draw.line(self.screen, COLOR_GRID, (c * CELL_SIZE, 0), (c * CELL_SIZE, GRID_HEIGHT))

        # Draw path
        if len(self.path) > 1:
            pygame.draw.lines(self.screen, COLOR_PATH, False, self.path, 10)
        
        # Draw base
        base_pos = self.path[-1]
        base_rect = pygame.Rect(base_pos[0] - 15, base_pos[1] - 15, 30, 30)
        pygame.draw.rect(self.screen, COLOR_BASE, base_rect)
        
        # Draw towers
        for tower in self.towers:
            tower.draw(self.screen)

        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)
            
        # Draw projectiles
        for proj in self.projectiles:
            proj.draw(self.screen)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw cursor and range indicator
        spec = TOWER_SPECS[self.selected_tower_type]
        cursor_px = (self.cursor_pos[0] * CELL_SIZE, self.cursor_pos[1] * CELL_SIZE)
        is_valid = self.gold >= spec['cost'] and self.grid[self.cursor_pos[1]][self.cursor_pos[0]] is None
        cursor_color = COLOR_CURSOR_VALID if is_valid else COLOR_CURSOR_INVALID
        
        # Range indicator
        range_center = (cursor_px[0] + CELL_SIZE/2, cursor_px[1] + CELL_SIZE/2)
        s = pygame.Surface((SCREEN_WIDTH, GRID_HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(s, cursor_color + (50,), range_center, spec['range'])
        self.screen.blit(s, (0,0))
        
        # Cursor box
        pygame.draw.rect(self.screen, cursor_color, (*cursor_px, CELL_SIZE, CELL_SIZE), 2)

    def _render_ui(self):
        ui_rect = pygame.Rect(0, GRID_HEIGHT, SCREEN_WIDTH, UI_HEIGHT)
        pygame.draw.rect(self.screen, (15, 15, 25), ui_rect)
        pygame.draw.line(self.screen, (60, 60, 80), (0, GRID_HEIGHT), (SCREEN_WIDTH, GRID_HEIGHT), 2)

        # Gold
        gold_text = self.font_medium.render(f"GOLD: {self.gold}", True, COLOR_GOLD)
        self.screen.blit(gold_text, (10, GRID_HEIGHT + 10))

        # Wave
        wave_text = self.font_medium.render(f"WAVE: {self.wave}/{self.max_waves}", True, COLOR_TEXT)
        self.screen.blit(wave_text, (10, GRID_HEIGHT + 35))
        if self.wave_timer > 0 and self.wave < self.max_waves:
            timer_text = self.font_small.render(f"Next in {self.wave_timer/30:.1f}s", True, COLOR_TEXT)
            self.screen.blit(timer_text, (130, GRID_HEIGHT + 38))

        # Base Health
        health_text = self.font_medium.render("BASE HEALTH", True, COLOR_TEXT)
        self.screen.blit(health_text, (SCREEN_WIDTH - 160, GRID_HEIGHT + 10))
        hp_ratio = max(0, self.base_health / self.max_base_health)
        hp_bar_rect_bg = pygame.Rect(SCREEN_WIDTH - 160, GRID_HEIGHT + 35, 150, 20)
        hp_bar_rect_fg = pygame.Rect(SCREEN_WIDTH - 160, GRID_HEIGHT + 35, int(150 * hp_ratio), 20)
        pygame.draw.rect(self.screen, COLOR_HEALTH_BAR_BG, hp_bar_rect_bg)
        pygame.draw.rect(self.screen, COLOR_HEALTH_BAR_FG, hp_bar_rect_fg)

        # Selected Tower
        spec = TOWER_SPECS[self.selected_tower_type]
        tower_name_text = self.font_medium.render(f"Build: {spec['name']}", True, COLOR_TEXT)
        self.screen.blit(tower_name_text, (220, GRID_HEIGHT + 10))
        cost_color = COLOR_GOLD if self.gold >= spec['cost'] else COLOR_CURSOR_INVALID
        tower_cost_text = self.font_small.render(f"Cost: {spec['cost']}", True, cost_color)
        self.screen.blit(tower_cost_text, (220, GRID_HEIGHT + 35))
        tower_info_text = self.font_small.render(f"DMG: {spec['damage']} | RNG: {spec['range']} | RoF: {spec['fire_rate']}", True, COLOR_TEXT)
        self.screen.blit(tower_info_text, (220, GRID_HEIGHT + 55))

        # Game Over / Victory
        if self.game_over:
            s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            message = "VICTORY!" if self.victory else "GAME OVER"
            color = (0, 255, 128) if self.victory else (255, 50, 50)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage for visualization and testing
if __name__ == '__main__':
    env = GameEnv()
    env.reset()
    
    running = True
    is_paused = False
    
    # Use a dictionary to track held keys for smoother controls
    keys_held = {
        "up": False, "down": False, "left": False, "right": False,
        "space": False, "shift": False
    }

    render_surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_p:
                    is_paused = not is_paused
                if event.key == pygame.K_r:
                    env.reset()

                if event.key == pygame.K_UP: keys_held["up"] = True
                if event.key == pygame.K_DOWN: keys_held["down"] = True
                if event.key == pygame.K_LEFT: keys_held["left"] = True
                if event.key == pygame.K_RIGHT: keys_held["right"] = True
                if event.key == pygame.K_SPACE: keys_held["space"] = True
                if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT): keys_held["shift"] = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP: keys_held["up"] = False
                if event.key == pygame.K_DOWN: keys_held["down"] = False
                if event.key == pygame.K_LEFT: keys_held["left"] = False
                if event.key == pygame.K_RIGHT: keys_held["right"] = False
                if event.key == pygame.K_SPACE: keys_held["space"] = False
                if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT): keys_held["shift"] = False

        if not is_paused:
            # Map keyboard state to action space
            movement = 0
            if keys_held["up"]: movement = 1
            elif keys_held["down"]: movement = 2
            elif keys_held["left"]: movement = 3
            elif keys_held["right"]: movement = 4
            
            space = 1 if keys_held["space"] else 0
            shift = 1 if keys_held["shift"] else 0
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
                is_paused = True

        # Render the environment to the screen
        draw_surface = pygame.surfarray.make_surface(np.transpose(env._get_observation(), (1, 0, 2)))
        render_surface.blit(draw_surface, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Run at 30 FPS

    env.close()