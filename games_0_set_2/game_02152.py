
# Generated: 2025-08-27T19:28:01.375357
# Source Brief: brief_02152.md
# Brief Index: 2152

        
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


# Helper classes for game entities
class Enemy:
    def __init__(self, x, y, health, speed, wave):
        self.x = x
        self.y = y
        self.pixel_x = x * 32 + 16
        self.pixel_y = y * 32 + 16
        self.max_health = health * (1.05 ** (wave - 1))
        self.health = self.max_health
        self.speed = speed * (1.05 ** (wave - 1))
        self.move_cooldown = 0

class Projectile:
    def __init__(self, x, y, target_enemy, speed, damage):
        self.x = x
        self.y = y
        self.target = target_enemy
        self.speed = speed
        self.damage = damage
        self.trail = []

class Particle:
    def __init__(self, x, y, color, lifetime, size, velocity):
        self.x = x
        self.y = y
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size
        self.velocity = velocity

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to place selected block. Shift to cycle block type."
    )

    game_description = (
        "Defend your fortress from enemy waves by strategically placing defensive blocks on the grid."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 12
    CELL_SIZE = 32
    UI_HEIGHT = SCREEN_HEIGHT - (GRID_HEIGHT * CELL_SIZE) # 16 pixels

    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_GRID = (45, 45, 60)
    COLOR_TEXT = (240, 240, 240)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_FORTRESS = (0, 180, 120)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_HEALTH_BAR = (0, 255, 0)
    COLOR_HEALTH_BAR_BG = (120, 0, 0)
    COLOR_PROJECTILE = (0, 200, 255)

    # Block Types
    BLOCK_EMPTY = 0
    BLOCK_BASIC = 1
    BLOCK_REINFORCED = 2
    BLOCK_RANGED = 3
    BLOCK_AOE = 4 # Area of Effect
    BLOCK_GENERATOR = 5
    BLOCK_FORTRESS = 6

    BLOCK_SPECS = {
        BLOCK_BASIC: {"name": "WALL", "cost": 1, "health": 100, "color": (100, 100, 120)},
        BLOCK_REINFORCED: {"name": "HEAVY", "cost": 2, "health": 300, "color": (160, 160, 180)},
        BLOCK_RANGED: {"name": "TURRET", "cost": 3, "health": 50, "range": 4, "damage": 10, "cooldown": 2, "color": (50, 150, 200)},
        BLOCK_AOE: {"name": "BOMB", "cost": 4, "health": 20, "range": 1, "damage": 25, "cooldown": 5, "color": (200, 150, 50)},
        BLOCK_GENERATOR: {"name": "GEN", "cost": 5, "health": 40, "color": (200, 50, 200)},
        BLOCK_FORTRESS: {"name": "FORT", "cost": 0, "health": 100, "color": COLOR_FORTRESS}
    }

    MAX_WAVES = 20
    MAX_STEPS = 1000

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
        self.font_small = pygame.font.SysFont("sans-serif", 14)
        self.font_large = pygame.font.SysFont("sans-serif", 24)
        
        self.grid = None
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.fortress_health = 0
        self.resources = 0
        self.current_wave = 0
        self.wave_enemy_count = 0
        self.enemies_spawned_this_wave = 0
        self.score = 0
        self.steps = 0
        self.cursor_pos = [0, 0]
        self.selected_block_type = self.BLOCK_BASIC
        self.game_over = False
        self.rng = None

        self.reset()
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        # Grid stores [block_type, health, cooldown_timer]
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT, 3), dtype=np.int32)
        
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.fortress_health = 100
        self.resources = 10
        self.current_wave = 1
        self.wave_enemy_count = 3
        self.enemies_spawned_this_wave = 0
        self.spawn_cooldown = 0
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_block_type = self.BLOCK_BASIC
        
        # Place fortress
        self.fortress_coords = []
        for y in range(self.GRID_HEIGHT // 2 - 1, self.GRID_HEIGHT // 2 + 2):
            self.grid[0, y] = [self.BLOCK_FORTRESS, self.BLOCK_SPECS[self.BLOCK_FORTRESS]["health"], 0]
            self.fortress_coords.append((0, y))
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        step_reward = 0

        # 1. Player Actions
        self._handle_player_input(movement, space_action, shift_action)

        # 2. Spawn Enemies
        self._spawn_enemies()

        # 3. Block Actions
        step_reward += self._update_blocks()

        # 4. Update Projectiles
        step_reward += self._update_projectiles()

        # 5. Update Enemies
        fortress_damage = self._update_enemies()
        self.fortress_health = max(0, self.fortress_health - fortress_damage)

        # 6. Update Particles
        self._update_particles()
        
        # 7. Wave Management
        if self.enemies_spawned_this_wave >= self.wave_enemy_count and not self.enemies:
            step_reward += 1.0 # Wave survived
            self.current_wave += 1
            if self.current_wave > self.MAX_WAVES:
                self.game_over = True
                step_reward += 100 # Victory
            else:
                self.wave_enemy_count = 3 + (self.current_wave - 1)
                self.enemies_spawned_this_wave = 0
                self.spawn_cooldown = 60 # Breather between waves

        # 8. Update State
        self.score += step_reward
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over: # Only apply terminal reward once
            if self.fortress_health <= 0:
                step_reward -= 100 # Defeat
            self.game_over = True
        
        return self._get_observation(), step_reward, terminated, False, self._get_info()

    def _handle_player_input(self, movement, place_action, cycle_action):
        # Cycle block type
        if cycle_action:
            self.selected_block_type += 1
            if self.selected_block_type > self.BLOCK_GENERATOR:
                self.selected_block_type = self.BLOCK_BASIC
        
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Place block
        if place_action:
            x, y = self.cursor_pos
            spec = self.BLOCK_SPECS[self.selected_block_type]
            if self.grid[x, y, 0] == self.BLOCK_EMPTY and self.resources >= spec["cost"]:
                self.resources -= spec["cost"]
                self.grid[x, y] = [self.selected_block_type, spec["health"], 0]
                # sfx: place_block
                self._create_particles(x * self.CELL_SIZE + 16, y * self.CELL_SIZE + 16, spec["color"], 10, 15)

    def _spawn_enemies(self):
        if self.spawn_cooldown > 0:
            self.spawn_cooldown -= 1
            return
            
        if self.enemies_spawned_this_wave < self.wave_enemy_count:
            spawn_x = self.GRID_WIDTH - 1
            spawn_y = self.rng.integers(0, self.GRID_HEIGHT)
            if self.grid[spawn_x, spawn_y, 0] == self.BLOCK_EMPTY:
                self.enemies.append(Enemy(spawn_x, spawn_y, health=20, speed=0.1, wave=self.current_wave))
                self.enemies_spawned_this_wave += 1
                self.spawn_cooldown = 30 # Time between spawns

    def _update_blocks(self):
        reward = 0
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                block_type, health, cooldown = self.grid[x, y]
                if block_type == self.BLOCK_EMPTY:
                    continue
                
                if cooldown > 0:
                    self.grid[x,y,2] -= 1
                    cooldown -= 1

                if block_type == self.BLOCK_GENERATOR:
                    if cooldown == 0:
                        self.resources += 1
                        reward += 0.5
                        self.grid[x,y,2] = 60 # 2 seconds at 30fps
                        # sfx: resource_gain
                elif block_type == self.BLOCK_RANGED and cooldown == 0:
                    spec = self.BLOCK_SPECS[block_type]
                    target = self._find_closest_enemy(x, y, spec["range"])
                    if target:
                        px, py = x * self.CELL_SIZE + 16, y * self.CELL_SIZE + 16
                        self.projectiles.append(Projectile(px, py, target, 8, spec["damage"]))
                        self.grid[x,y,2] = spec["cooldown"] * 30 # Cooldown in steps
                        # sfx: turret_fire
                elif block_type == self.BLOCK_AOE and cooldown == 0:
                    spec = self.BLOCK_SPECS[block_type]
                    if self._find_closest_enemy(x, y, spec["range"]):
                        reward += self._aoe_explosion(x, y, spec)
                        self.grid[x, y, 0] = self.BLOCK_EMPTY # Bomb is consumed
                        # sfx: bomb_explode
        return reward

    def _aoe_explosion(self, x, y, spec):
        reward = 0
        px, py = x * self.CELL_SIZE + 16, y * self.CELL_SIZE + 16
        self._create_particles(px, py, spec["color"], 20, 20, is_explosion=True)
        for enemy in self.enemies:
            dist = math.hypot(enemy.pixel_x - px, enemy.pixel_y - py)
            if dist <= (spec["range"] + 0.5) * self.CELL_SIZE:
                enemy.health -= spec["damage"]
                reward += 0.1
        return reward
    
    def _find_closest_enemy(self, x, y, max_range):
        closest_enemy = None
        min_dist = float('inf')
        for enemy in self.enemies:
            dist = math.hypot(enemy.x - x, enemy.y - y)
            if dist <= max_range and dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        return closest_enemy

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj.target not in self.enemies:
                self.projectiles.remove(proj)
                continue

            dx = proj.target.pixel_x - proj.x
            dy = proj.target.pixel_y - proj.y
            dist = math.hypot(dx, dy)
            
            proj.trail.append((proj.x, proj.y))
            if len(proj.trail) > 5: proj.trail.pop(0)

            if dist < proj.speed:
                proj.target.health -= proj.damage
                reward += 0.1
                self._create_particles(proj.target.pixel_x, proj.target.pixel_y, self.COLOR_PROJECTILE, 5, 10)
                # sfx: enemy_hit
                self.projectiles.remove(proj)
            else:
                proj.x += (dx / dist) * proj.speed
                proj.y += (dy / dist) * proj.speed
        
        for enemy in self.enemies[:]:
            if enemy.health <= 0:
                self._create_particles(enemy.pixel_x, enemy.pixel_y, self.COLOR_ENEMY, 15, 20, is_explosion=True)
                # sfx: enemy_death
                self.enemies.remove(enemy)
        return reward

    def _update_enemies(self):
        fortress_damage = 0
        for enemy in self.enemies:
            enemy.move_cooldown -= 1
            if enemy.move_cooldown > 0:
                continue
            
            fx, fy = self.fortress_coords[len(self.fortress_coords)//2]
            dx, dy = fx - enemy.x, fy - enemy.y
            
            if abs(dx) > abs(dy):
                moves = [(np.sign(dx), 0), (0, np.sign(dy))]
            else:
                moves = [(0, np.sign(dy)), (np.sign(dx), 0)]

            moved = False
            for move_x, move_y in moves:
                if move_x == 0 and move_y == 0: continue
                next_x, next_y = int(enemy.x + move_x), int(enemy.y + move_y)
                if 0 <= next_x < self.GRID_WIDTH and 0 <= next_y < self.GRID_HEIGHT:
                    target_block_type = self.grid[next_x, next_y, 0]
                    if target_block_type == self.BLOCK_EMPTY:
                        enemy.x, enemy.y = next_x, next_y
                        moved = True
                        break
            
            if not moved:
                for move_x, move_y in moves:
                    if move_x == 0 and move_y == 0: continue
                    attack_x, attack_y = int(enemy.x + move_x), int(enemy.y + move_y)
                    if 0 <= attack_x < self.GRID_WIDTH and 0 <= attack_y < self.GRID_HEIGHT:
                        target_block_type = self.grid[attack_x, attack_y, 0]
                        if target_block_type != self.BLOCK_EMPTY:
                            damage = 5
                            if target_block_type == self.BLOCK_FORTRESS:
                                fortress_damage += damage
                            else:
                                self.grid[attack_x, attack_y, 1] -= damage
                                if self.grid[attack_x, attack_y, 1] <= 0:
                                    self.grid[attack_x, attack_y, 0] = self.BLOCK_EMPTY
                            self._create_particles(attack_x*32+16, attack_y*32+16, (255,255,255), 3, 5)
                            # sfx: block_hit
                            break
            
            target_pixel_x, target_pixel_y = enemy.x * 32 + 16, enemy.y * 32 + 16
            enemy.pixel_x += np.clip(target_pixel_x - enemy.pixel_x, -enemy.speed * 32, enemy.speed * 32)
            enemy.pixel_y += np.clip(target_pixel_y - enemy.pixel_y, -enemy.speed * 32, enemy.speed * 32)
            enemy.move_cooldown = int(1 / enemy.speed)

        return fortress_damage

    def _update_particles(self):
        for p in self.particles[:]:
            p.lifetime -= 1
            p.x += p.velocity[0]
            p.y += p.velocity[1]
            if p.lifetime <= 0:
                self.particles.remove(p)

    def _create_particles(self, x, y, color, count, lifetime, size=3, is_explosion=False):
        for _ in range(count):
            if is_explosion:
                angle = self.rng.uniform(0, 2 * math.pi)
                speed = self.rng.uniform(1, 4)
                velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
            else:
                velocity = (self.rng.uniform(-1, 1), self.rng.uniform(-1, 1))
            self.particles.append(Particle(x, y, color, self.rng.integers(lifetime//2, lifetime), self.rng.integers(1, size+1), velocity))
    
    def _check_termination(self):
        return self.fortress_health <= 0 or self.current_wave > self.MAX_WAVES or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_blocks()
        for proj in self.projectiles: self._render_projectile(proj)
        for enemy in self.enemies: self._render_enemy(enemy)
        for p in self.particles: self._render_particle(p)
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "fortress_health": self.fortress_health,
            "resources": self.resources,
            "enemies": len(self.enemies),
        }

    def _render_grid_and_blocks(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE + self.UI_HEIGHT, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                block_type, health, _ = self.grid[x, y]
                if block_type != self.BLOCK_EMPTY:
                    spec = self.BLOCK_SPECS[block_type]
                    pygame.draw.rect(self.screen, spec["color"], rect.inflate(-4, -4))
                    if health < spec["health"]:
                        health_ratio = health / spec["health"]
                        bar_rect = pygame.Rect(rect.left + 2, rect.bottom - 6, rect.width - 4, 4)
                        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bar_rect)
                        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_rect.left, bar_rect.top, max(0, bar_rect.width * health_ratio), bar_rect.height))

    def _render_enemy(self, enemy):
        pos_x, pos_y = int(enemy.pixel_x), int(enemy.pixel_y + self.UI_HEIGHT)
        points = [(pos_x, pos_y - 10), (pos_x - 8, pos_y + 6), (pos_x + 8, pos_y + 6)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        health_ratio = enemy.health / enemy.max_health
        bar_rect = pygame.Rect(pos_x - 12, pos_y - 20, 24, 4)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_rect.left, bar_rect.top, max(0, bar_rect.width * health_ratio), bar_rect.height))

    def _render_projectile(self, proj):
        px, py = int(proj.x), int(proj.y + self.UI_HEIGHT)
        pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (px, py), 4)
        for i, (tx, ty) in enumerate(proj.trail):
            alpha = int(255 * (i / len(proj.trail)))
            s = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_PROJECTILE, alpha), (2,2), 2)
            self.screen.blit(s, (int(tx)-2, int(ty)-2 + self.UI_HEIGHT))

    def _render_particle(self, p):
        alpha = int(255 * (p.lifetime / p.max_lifetime))
        s = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*p.color, alpha), (p.size, p.size), p.size)
        self.screen.blit(s, (int(p.x) - p.size, int(p.y) - p.size + self.UI_HEIGHT))

    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE + self.UI_HEIGHT, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)

    def _render_ui(self):
        pygame.draw.rect(self.screen, (15, 15, 25), (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        
        health_text = self.font_large.render(f"FORTRESS: {int(self.fortress_health)}/100", True, self.COLOR_FORTRESS)
        self.screen.blit(health_text, (10, 5))
        
        wave_text = self.font_large.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (240, 5))

        res_text = self.font_large.render(f"RESOURCES: {self.resources}", True, (255, 200, 0))
        self.screen.blit(res_text, (430, 5))
        
        spec = self.BLOCK_SPECS[self.selected_block_type]
        block_text = self.font_small.render(f"Selected: {spec['name']} (Cost: {spec['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(block_text, (10, self.UI_HEIGHT - 20))
        
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, self.UI_HEIGHT - 20))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            outcome_text = "VICTORY!" if self.fortress_health > 0 else "DEFEAT"
            color = self.COLOR_FORTRESS if self.fortress_health > 0 else self.COLOR_ENEMY
            text_surf = self.font_large.render(outcome_text, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Fortress Defense")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0
        
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1
                    action_taken = True

        if movement != 0 or space_held != 0:
            action_taken = True
        
        if action_taken:
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
        else: # If no key is pressed, pass a no-op to advance the game state
            obs, reward, terminated, truncated, info = env.step([0,0,0])

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.1f}, Survived to Wave: {info['wave']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30)

    env.close()