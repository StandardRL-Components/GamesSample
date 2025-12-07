
# Generated: 2025-08-27T12:58:49.033998
# Source Brief: brief_00212.md
# Brief Index: 212

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    def __init__(self, x, y, health, speed, value, path):
        self.x, self.y = x, y
        self.health = health
        self.max_health = health
        self.speed = speed
        self.value = value  # Gold dropped on death
        self.path = path
        self.waypoint_index = 0
        self.radius = 8
        self.target_x, self.target_y = self.path[0]
        self.is_alive = True

    def move(self):
        if self.waypoint_index >= len(self.path):
            self.is_alive = False # Reached the end
            return True # Reached base

        self.target_x, self.target_y = self.path[self.waypoint_index]
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.x, self.y = self.target_x, self.target_y
            self.waypoint_index += 1
        else:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
        return False

    def draw(self, surface):
        # Body
        pos = (int(self.x), int(self.y))
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], self.radius, (220, 50, 50))
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], self.radius, (255, 100, 100))
        # Health bar
        if self.health < self.max_health:
            bar_width = 20
            bar_height = 4
            health_pct = self.health / self.max_health
            fill_width = int(bar_width * health_pct)
            bg_rect = pygame.Rect(self.x - bar_width // 2, self.y - self.radius - 8, bar_width, bar_height)
            fill_rect = pygame.Rect(self.x - bar_width // 2, self.y - self.radius - 8, fill_width, bar_height)
            pygame.draw.rect(surface, (50, 50, 50), bg_rect)
            pygame.draw.rect(surface, (50, 200, 50), fill_rect)

class Tower:
    def __init__(self, grid_x, grid_y, spec):
        self.grid_x, self.grid_y = grid_x, grid_y
        self.x = grid_x * 40 + 20
        self.y = grid_y * 40 + 20
        self.spec = spec
        self.range = spec['range']
        self.damage = spec['damage']
        self.fire_rate = spec['fire_rate']
        self.type = spec['type']
        self.cooldown = 0
        self.target = None

    def find_target(self, enemies):
        if self.target and self.target.is_alive and math.hypot(self.x - self.target.x, self.y - self.target.y) <= self.range:
            return # Keep current target if still valid

        self.target = None
        closest_enemy = None
        min_dist = float('inf')
        for enemy in enemies:
            dist = math.hypot(self.x - enemy.x, self.y - enemy.y)
            if dist <= self.range and dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        self.target = closest_enemy

    def update(self, enemies):
        if self.cooldown > 0:
            self.cooldown -= 1
        self.find_target(enemies)
        if self.target and self.cooldown == 0:
            self.cooldown = self.fire_rate
            return self._fire() # Returns a projectile or None
        return None

    def _fire(self):
        # sfx: tower_fire.wav
        return Projectile(self.x, self.y, self.target, self.spec)

    def draw(self, surface):
        pos = (int(self.x), int(self.y))
        pygame.draw.rect(surface, self.spec['color'], (self.grid_x * 40 + 5, self.grid_y * 40 + 5, 30, 30))
        pygame.draw.rect(surface, tuple(min(255, c+40) for c in self.spec['color']), (self.grid_x * 40 + 8, self.grid_y * 40 + 8, 24, 24))
        if self.target:
            pygame.draw.aaline(surface, (200, 200, 200, 100), pos, (int(self.target.x), int(self.target.y)))

class Projectile:
    def __init__(self, x, y, target, tower_spec):
        self.x, self.y = x, y
        self.target = target
        self.speed = tower_spec['proj_speed']
        self.damage = tower_spec['damage']
        self.is_aoe = tower_spec['aoe_radius'] > 0
        self.aoe_radius = tower_spec['aoe_radius']
        self.color = tower_spec['color']

    def move(self):
        if not self.target.is_alive:
            return True # Target is dead, projectile fizzles

        dx = self.target.x - self.x
        dy = self.target.y - self.y
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.x, self.y = self.target.x, self.target.y
            return True # Reached target
        else:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
        return False

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), 4)

class Particle:
    def __init__(self, x, y, color, lifespan):
        self.x, self.y = x, y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        return self.lifespan <= 0

    def draw(self, surface):
        alpha = int(255 * (self.lifespan / self.max_lifespan))
        size = int(3 * (self.lifespan / self.max_lifespan))
        if size > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), size)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Use arrows to move the cursor. Press Shift to cycle tower types. Press Space to place a tower. Do nothing (no-op action) to start the next wave."
    game_description = "Defend your base from waves of invading enemies by strategically placing procedurally generated towers in a top-down tower defense game."
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 40
    GRID_W, GRID_H = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE
    TOTAL_WAVES = 10
    MAX_STEPS = 5000

    COLOR_BG = (20, 25, 30)
    COLOR_PATH = (40, 45, 50)
    COLOR_GRID = (30, 35, 40)
    COLOR_BASE = (50, 200, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    
    TOWER_SPECS = [
        {'type': 'Cannon', 'cost': 50, 'range': 100, 'damage': 10, 'fire_rate': 20, 'proj_speed': 8, 'aoe_radius': 0, 'color': (60, 120, 220)},
        {'type': 'Missile', 'cost': 120, 'range': 180, 'damage': 50, 'fire_rate': 80, 'proj_speed': 6, 'aoe_radius': 0, 'color': (220, 200, 60)},
        {'type': 'Splash', 'cost': 180, 'range': 120, 'damage': 25, 'fire_rate': 60, 'proj_speed': 5, 'aoe_radius': 40, 'color': (180, 60, 220)},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.render_mode = render_mode
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = 100
        self.gold = 150 # Start with enough for 3 cannons
        self.wave_number = 0
        self.game_phase = 'PLANNING' # 'PLANNING' or 'WAVE'

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.wave_spawn_timer = 0
        self.enemies_to_spawn_in_wave = 0
        self.enemies_alive = 0
        
        self.path = self._generate_path()
        self.buildable_tiles = self._get_buildable_tiles()

        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def _generate_path(self):
        path_grid = [(0, 2), (5, 2), (5, 7), (10, 7), (10, 2), (15, 2)]
        return [(x * self.GRID_SIZE + self.GRID_SIZE // 2, y * self.GRID_SIZE + self.GRID_SIZE // 2) for x, y in path_grid]

    def _get_buildable_tiles(self):
        buildable = set((x, y) for x in range(self.GRID_W) for y in range(self.GRID_H))
        for i in range(len(self.path) - 1):
            p1 = (self.path[i][0] // self.GRID_SIZE, self.path[i][1] // self.GRID_SIZE)
            p2 = (self.path[i+1][0] // self.GRID_SIZE, self.path[i+1][1] // self.GRID_SIZE)
            for x in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                if (x, p1[1]) in buildable: buildable.remove((x, p1[1]))
            for y in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                if (p2[0], y) in buildable: buildable.remove((p2[0], y))
        return buildable

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.game_phase == 'PLANNING':
            is_noop = (movement == 0 and not space_held and not shift_held)
            if is_noop and self.wave_number < self.TOTAL_WAVES:
                self._start_next_wave()
            else:
                step_reward += self._handle_planning_controls(movement, space_held, shift_held)
        
        elif self.game_phase == 'WAVE':
            step_reward += self._update_wave_phase()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        terminated = self.base_health <= 0 or (self.wave_number == self.TOTAL_WAVES and self._is_wave_over()) or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.base_health > 0 and self.wave_number == self.TOTAL_WAVES:
                step_reward += 100 # Victory
            else:
                step_reward -= 100 # Defeat

        self.score += step_reward
        return self._get_observation(), step_reward, terminated, False, self._get_info()

    def _handle_planning_controls(self, movement, space_held, shift_held):
        reward = 0
        # Cycle tower on shift press (rising edge)
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_Cycle.wav
        
        # Place tower on space press (rising edge)
        if space_held and not self.last_space_held:
            reward += self._place_tower()

        # Move cursor
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_H - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)
        return reward

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        cost = spec['cost']
        pos_tuple = tuple(self.cursor_pos)

        is_occupied = any(t.grid_x == pos_tuple[0] and t.grid_y == pos_tuple[1] for t in self.towers)

        if self.gold >= cost and pos_tuple in self.buildable_tiles and not is_occupied:
            self.gold -= cost
            self.towers.append(Tower(pos_tuple[0], pos_tuple[1], spec))
            # sfx: Tower_Place.wav
            return -cost * 0.01 # Reward penalty for spending gold
        else:
            # sfx: Error.wav
            return 0

    def _start_next_wave(self):
        self.game_phase = 'WAVE'
        self.wave_number += 1
        self.enemies_to_spawn_in_wave = 5 + self.wave_number * 2
        self.enemies_alive = self.enemies_to_spawn_in_wave
        self.wave_spawn_timer = 0
        # sfx: Wave_Start.wav

    def _update_wave_phase(self):
        reward = 0
        
        # 1. Spawn enemies
        self.wave_spawn_timer -= 1
        if self.enemies_to_spawn_in_wave > 0 and self.wave_spawn_timer <= 0:
            self.wave_spawn_timer = max(10, 30 - self.wave_number) # Faster spawns in later waves
            self.enemies_to_spawn_in_wave -= 1
            health = 50 * (1 + 0.05 * self.wave_number)
            speed = 1.5 * (1 + 0.02 * self.wave_number)
            value = 5 + self.wave_number
            start_pos = self.path[0]
            self.enemies.append(Enemy(start_pos[0], start_pos[1], health, speed, value, self.path))

        # 2. Update towers and create projectiles
        for tower in self.towers:
            projectile = tower.update(self.enemies)
            if projectile:
                self.projectiles.append(projectile)

        # 3. Update projectiles and handle hits
        new_projectiles = []
        for p in self.projectiles:
            hit = p.move()
            if hit:
                reward += self._handle_projectile_hit(p)
            else:
                new_projectiles.append(p)
        self.projectiles = new_projectiles
        
        # 4. Update enemies
        new_enemies = []
        for enemy in self.enemies:
            if enemy.move(): # Reached base
                self.base_health = max(0, self.base_health - 10)
                reward -= 10
                enemy.is_alive = False
                self.enemies_alive -= 1
                # sfx: Base_Damage.wav
            if enemy.is_alive:
                new_enemies.append(enemy)
        self.enemies = new_enemies
        
        # 5. Update particles
        self.particles = [p for p in self.particles if not p.update()]

        # 6. Check for wave end
        if self._is_wave_over():
            self.game_phase = 'PLANNING'
            # sfx: Wave_End.wav
        
        return reward

    def _handle_projectile_hit(self, p):
        reward = 0
        if not p.target.is_alive: return 0

        if not p.is_aoe:
            reward += self._damage_enemy(p.target, p.damage)
            self._create_particles(p.x, p.y, p.color, 5)
        else: # AOE damage
            # sfx: Explosion.wav
            self._create_particles(p.x, p.y, p.color, 20)
            for enemy in self.enemies:
                if math.hypot(p.x - enemy.x, p.y - enemy.y) <= p.aoe_radius:
                    reward += self._damage_enemy(enemy, p.damage)
        return reward

    def _damage_enemy(self, enemy, damage):
        reward = 0.1 # Reward for hitting
        enemy.health -= damage
        if enemy.health <= 0:
            enemy.is_alive = False
            self.gold += enemy.value
            reward += 1 # Reward for killing
            self.enemies_alive -= 1
            # sfx: Enemy_Die.wav
        return reward
        
    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            self.particles.append(Particle(x, y, color, random.randint(10, 20)))

    def _is_wave_over(self):
        return self.enemies_to_spawn_in_wave == 0 and self.enemies_alive == 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw path
        if len(self.path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, self.GRID_SIZE)
        
        # Draw base (end of path)
        base_pos = self.path[-1]
        base_rect = pygame.Rect(base_pos[0] - self.GRID_SIZE//2, base_pos[1] - self.GRID_SIZE//2, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        # Draw towers
        for tower in self.towers:
            tower.draw(self.screen)
        
        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)
            
        # Draw projectiles
        for p in self.projectiles:
            p.draw(self.screen)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw placement cursor
        if self.game_phase == 'PLANNING':
            self._render_cursor()

    def _render_cursor(self):
        cursor_x, cursor_y = self.cursor_pos
        screen_x, screen_y = cursor_x * self.GRID_SIZE, cursor_y * self.GRID_SIZE
        
        spec = self.TOWER_SPECS[self.selected_tower_type]
        can_afford = self.gold >= spec['cost']
        is_buildable = tuple(self.cursor_pos) in self.buildable_tiles
        is_occupied = any(t.grid_x == cursor_x and t.grid_y == cursor_y for t in self.towers)
        is_valid = can_afford and is_buildable and not is_occupied

        color = (0, 255, 0, 100) if is_valid else (255, 0, 0, 100)

        # Draw range indicator
        range_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(range_surface, screen_x + self.GRID_SIZE//2, screen_y + self.GRID_SIZE//2, spec['range'], color)
        self.screen.blit(range_surface, (0, 0))

        # Draw cursor box
        pygame.draw.rect(self.screen, tuple(c/2 for c in color[:3]), (screen_x, screen_y, self.GRID_SIZE, self.GRID_SIZE), 2)


    def _render_ui(self):
        # Top Bar
        bar_rect = pygame.Rect(0, 0, self.WIDTH, 30)
        pygame.draw.rect(self.screen, (10, 10, 15), bar_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 30), (self.WIDTH, 30))

        # Health
        health_text = self.font_small.render(f"❤️ Base Health: {self.base_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 7))

        # Gold
        gold_text = self.font_small.render(f"💰 Gold: {self.gold}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gold_text, (200, 7))

        # Wave
        wave_str = f"Wave: {self.wave_number}/{self.TOTAL_WAVES}"
        if self.game_phase == 'PLANNING' and self.wave_number < self.TOTAL_WAVES:
            wave_str = f"Next Wave: {self.wave_number + 1}"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (350, 7))

        # Tower Selection UI
        if self.game_phase == 'PLANNING':
            spec = self.TOWER_SPECS[self.selected_tower_type]
            tower_info_text = self.font_small.render(
                f"Selected: {spec['type']} | Cost: {spec['cost']} | Dmg: {spec['damage']} | Range: {spec['range']}",
                True, self.COLOR_UI_TEXT
            )
            self.screen.blit(tower_info_text, (10, self.HEIGHT - 22))

        # Game Over/Victory Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            if self.base_health > 0 and self.wave_number == self.TOTAL_WAVES:
                msg = "VICTORY!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.wave_number,
            "game_phase": self.game_phase,
        }

    def close(self):
        pygame.quit()

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
    # This block allows you to play the game directly
    # It will not be executed when the environment is imported by an RL agent
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    # Map Pygame keys to MultiDiscrete actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Track rising edge for single-press actions
    last_keys = pygame.key.get_pressed()
    
    while running:
        # In auto_advance=False mode, we only step on input
        # For human play, we need to poll for events and then step
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            
            movement_action = 0
            space_action = 0
            shift_action = 0
            
            # Use rising edge for discrete actions
            if keys[pygame.K_UP] and not last_keys[pygame.K_UP]: movement_action = 1
            elif keys[pygame.K_DOWN] and not last_keys[pygame.K_DOWN]: movement_action = 2
            elif keys[pygame.K_LEFT] and not last_keys[pygame.K_LEFT]: movement_action = 3
            elif keys[pygame.K_RIGHT] and not last_keys[pygame.K_RIGHT]: movement_action = 4
            
            if keys[pygame.K_SPACE]: space_action = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1
            
            # Special case for human: if Enter is pressed during planning, start wave
            if (keys[pygame.K_RETURN] and not last_keys[pygame.K_RETURN]) and info['game_phase'] == 'PLANNING':
                 action = [0, 0, 0] # This translates to a no-op action for the env
            else:
                 action = [movement_action, space_action, shift_action]
            
            # If during a wave, we just want to advance time
            if info['game_phase'] == 'WAVE':
                action = [0, 0, 0] # Send no-op to advance time
            
            obs, reward, terminated, truncated, info = env.step(action)
            last_keys = keys
        
        # Convert observation back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # During planning, wait for input. During wave, run at 30fps
        if info['game_phase'] == 'WAVE' and not terminated:
            clock.tick(30)
        else:
            clock.tick(15) # Slower tick for menuing

    env.close()