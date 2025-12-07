import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:59:06.743669
# Source Brief: brief_00138.md
# Brief Index: 138
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

# --- Helper Classes ---

class Microbe:
    """Represents a single player or enemy unit."""
    def __init__(self, microbe_type, q, r, microbe_info, wave_bonus):
        self.type = microbe_type
        self.q, self.r = q, r
        self.info = microbe_info
        
        base_hp = self.info['hp']
        base_dmg = self.info['damage']
        
        # Apply difficulty scaling for enemies
        self.max_hp = int(base_hp * wave_bonus) if self.info['faction'] == 'enemy' else base_hp
        self.hp = self.max_hp
        self.damage = int(base_dmg * wave_bonus) if self.info['faction'] == 'enemy' else base_dmg
        
        self.attack_range = self.info['range']
        self.attack_cooldown = 0
        self.attack_speed = self.info['attack_speed']
        self.move_cooldown = 0
        self.move_speed = self.info['move_speed']
        self.target = None
        self.bob_offset = random.uniform(0, math.pi * 2)

class Particle:
    """Represents a visual effect particle."""
    def __init__(self, x, y, p_type, color):
        self.x, self.y = x, y
        self.type = p_type
        self.color = color
        self.life = 1.0
        if self.type == 'attack':
            self.duration = 0.2
            self.vx = random.uniform(-150, 150)
            self.vy = random.uniform(-150, 150)
        elif self.type == 'death':
            self.duration = 0.5
            self.angle = random.uniform(0, 2 * math.pi)
            self.speed = random.uniform(50, 150)
        elif self.type == 'resource':
            self.duration = 0.6
            self.vy = -100
        
    def update(self, dt):
        self.life -= dt / self.duration
        if self.type == 'attack':
            self.x += self.vx * dt
            self.y += self.vy * dt
        elif self.type == 'death':
            self.x += math.cos(self.angle) * self.speed * dt
            self.y += math.sin(self.angle) * self.speed * dt
        elif self.type == 'resource':
            self.y += self.vy * dt
        return self.life > 0

# --- Main Environment Class ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your base by strategically placing microbes to fight off waves of attacking enemies in this hexagonal grid-based tower defense game."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press Shift to cycle through deployable microbes and Space to place one on the selected hex."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    
    # Visuals
    COLOR_BG = (15, 18, 33)
    COLOR_GRID = (40, 45, 70)
    COLOR_BASE = (30, 70, 90)
    COLOR_SPAWN = (90, 30, 50)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_ENEMY = (255, 60, 60)
    COLOR_RESOURCE = (0, 200, 255)
    COLOR_TEXT = (230, 230, 230)
    COLOR_CURSOR = (255, 255, 0)
    
    HEX_RADIUS = 18
    HEX_WIDTH = HEX_RADIUS * 2
    HEX_HEIGHT = math.sqrt(3) * HEX_RADIUS
    
    # Gameplay
    MAX_STEPS = 1000
    TOTAL_WAVES = 10
    RESOURCE_START = 25
    RESOURCE_RATE = 2.5  # per second
    RESOURCE_CAP = 100
    RESOURCE_LOW_THRESHOLD = 10

    # Microbe Definitions
    MICROBE_TYPES = {
        'guardian': {'name': 'Guardian', 'faction': 'player', 'cost': 10, 'hp': 100, 'damage': 12, 'range': 1.5, 'attack_speed': 1.0, 'move_speed': 0},
        'spitter': {'name': 'Spitter', 'faction': 'player', 'cost': 15, 'hp': 60, 'damage': 8, 'range': 4, 'attack_speed': 0.7, 'move_speed': 0},
        'runner': {'name': 'Runner', 'faction': 'enemy', 'hp': 40, 'damage': 5, 'range': 1.5, 'attack_speed': 1.2, 'move_speed': 0.5},
        'tank': {'name': 'Tank', 'faction': 'enemy', 'hp': 150, 'damage': 10, 'range': 1.5, 'attack_speed': 1.5, 'move_speed': 0.25},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.dt = 1 / self.FPS
        
        self.font_s = pygame.font.SysFont("Consolas", 14)
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Grid setup
        self.grid_width = 15
        self.grid_height = 9
        self.grid_offset_x = 60
        self.grid_offset_y = 50
        self.grid = {} # (q, r) -> Microbe or None
        for q in range(self.grid_width):
            for r in range(-(q//2), self.grid_height - (q//2)):
                if q + r >= 0 and q + r < self.grid_height:
                     self.grid[(q, r)] = None

        self.base_hexes = {qr for qr in self.grid if qr[0] == 0}
        self.spawn_hexes = {qr for qr in self.grid if qr[0] == self.grid_width - 1}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.resources = self.RESOURCE_START
        self.wave_num = 0
        self.wave_spawning = False
        self.enemies_in_wave = 0
        
        self.player_units = []
        self.enemy_units = []
        self.particles = []
        
        # Clear grid
        for qr in self.grid:
            self.grid[qr] = None

        self.cursor_q, self.cursor_r = 1, 3
        self.selected_microbe_idx = 0
        self.player_microbe_keys = [k for k, v in self.MICROBE_TYPES.items() if v['faction'] == 'player']
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._generate_all_waves()
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Handle Player Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        self._handle_player_actions(movement, space_pressed, shift_pressed)

        # --- 2. Update Game State ---
        # Resource generation
        new_resources = self.RESOURCE_RATE * self.dt
        if self.resources < self.RESOURCE_CAP:
            self.resources = min(self.RESOURCE_CAP, self.resources + new_resources)
            # Spawn resource particle effect
            if random.random() < 0.1:
                self.particles.append(Particle(self.SCREEN_WIDTH - 80, 50, 'resource', self.COLOR_RESOURCE))

        if self.resources < self.RESOURCE_LOW_THRESHOLD:
            reward -= 0.1

        # Wave spawning
        if self.wave_spawning and self.wave_spawn_queue:
            if self.steps >= self.wave_spawn_queue[0][1]:
                enemy_type, _ = self.wave_spawn_queue.popleft()
                self._spawn_enemy(enemy_type)
        
        if self.wave_spawning and not self.wave_spawn_queue:
            self.wave_spawning = False # All units for this wave have been spawned

        # --- 3. Update Units ---
        reward += self._update_units(self.player_units, self.enemy_units) # Player attacks
        reward += self._update_units(self.enemy_units, self.player_units) # Enemy attacks/moves
        
        # --- 4. Update Particles ---
        self.particles = [p for p in self.particles if p.update(self.dt)]

        # --- 5. Clean up and check wave completion ---
        if not self.wave_spawning and not self.enemy_units:
            if self.wave_num < self.TOTAL_WAVES:
                reward += 5.0 # Wave clear bonus
                self.score += 50
                self._start_next_wave()
            else: # All waves defeated
                self.game_over = True
                reward += 100.0 # Victory bonus
                self.score += 1000

        # --- 6. Check Termination Conditions ---
        if self._check_base_breach():
            self.game_over = True
            reward -= 100.0 # Defeat penalty
            self.score -= 500
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        terminated = self.game_over and not truncated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Gym Interface Helpers ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_num, "resources": self.resources}

    # --- Game Logic Sub-routines ---
    
    def _handle_player_actions(self, movement, space_pressed, shift_pressed):
        # Cursor movement
        q, r = self.cursor_q, self.cursor_r
        if movement == 1: r -= 1 # Up
        elif movement == 2: r += 1 # Down
        elif movement == 3: q -= 1; r += (1 if q % 2 == 0 else 0) # Left
        elif movement == 4: q += 1; r -= (1 if q % 2 != 0 else 0) # Right
        
        if (q, r) in self.grid:
            self.cursor_q, self.cursor_r = q, r
            
        # Cycle selected microbe
        if shift_pressed:
            self.selected_microbe_idx = (self.selected_microbe_idx + 1) % len(self.player_microbe_keys)
        
        # Deploy microbe
        if space_pressed:
            key = self.player_microbe_keys[self.selected_microbe_idx]
            cost = self.MICROBE_TYPES[key]['cost']
            if self.resources >= cost and self.grid.get((self.cursor_q, self.cursor_r)) is None:
                self.resources -= cost
                self.score += cost
                # sfx: deploy_unit.wav
                microbe_info = self.MICROBE_TYPES[key]
                new_microbe = Microbe(key, self.cursor_q, self.cursor_r, microbe_info, 1.0)
                self.player_units.append(new_microbe)
                self.grid[(self.cursor_q, self.cursor_r)] = new_microbe

    def _update_units(self, attackers, defenders):
        reward = 0
        
        for unit in attackers:
            # Cooldowns
            unit.attack_cooldown = max(0, unit.attack_cooldown - self.dt)
            unit.move_cooldown = max(0, unit.move_cooldown - self.dt)

            # Find target if needed
            if unit.target is None or unit.target.hp <= 0 or self._hex_distance(unit.q, unit.r, unit.target.q, unit.target.r) > unit.attack_range:
                unit.target = self._find_closest_target(unit, defenders)

            # Attack if possible
            if unit.target and unit.attack_cooldown == 0:
                unit.attack_cooldown = unit.attack_speed
                unit.target.hp -= unit.damage
                reward += 0.1 # Damage reward
                self.score += 1
                # sfx: attack_hit.wav
                # Particle effect
                px1, py1 = self._axial_to_pixel(unit.q, unit.r)
                px2, py2 = self._axial_to_pixel(unit.target.q, unit.target.r)
                for _ in range(3):
                    self.particles.append(Particle(px2, py2, 'attack', self.COLOR_ENEMY if unit.info['faction'] == 'player' else self.COLOR_PLAYER))

                if unit.target.hp <= 0:
                    reward += 0.5 # Kill reward
                    self.score += 25
                    # sfx: unit_death_explosion.wav
                    px, py = self._axial_to_pixel(unit.target.q, unit.target.r)
                    for _ in range(15):
                        self.particles.append(Particle(px, py, 'death', self.COLOR_ENEMY if unit.target.info['faction'] == 'enemy' else self.COLOR_PLAYER))
                    self.grid[(unit.target.q, unit.target.r)] = None
                    defenders.remove(unit.target)
                    unit.target = None
                    if unit.info['faction'] == 'player':
                        self.enemies_in_wave -= 1
            
            # Move if enemy and no target in range
            if unit.info['faction'] == 'enemy' and unit.move_cooldown == 0:
                if unit.target is None:
                    self._move_enemy(unit)
                    unit.move_cooldown = unit.move_speed
        return reward

    def _move_enemy(self, enemy):
        neighbors = self._get_neighbors(enemy.q, enemy.r)
        random.shuffle(neighbors)
        
        best_neighbor = None
        min_dist = float('inf')

        # Find neighbor closest to any base hex
        for nq, nr in neighbors:
            if self.grid.get((nq, nr)) is None: # Check if tile is empty
                # Find distance to the closest base hex from this neighbor
                dist_to_base = min(self._hex_distance(nq, nr, bq, br) for bq, br in self.base_hexes)
                if dist_to_base < min_dist:
                    min_dist = dist_to_base
                    best_neighbor = (nq, nr)

        if best_neighbor:
            self.grid[(enemy.q, enemy.r)] = None
            enemy.q, enemy.r = best_neighbor
            self.grid[(enemy.q, enemy.r)] = enemy

    def _check_base_breach(self):
        for enemy in self.enemy_units:
            if (enemy.q, enemy.r) in self.base_hexes:
                # sfx: base_alarm.wav
                return True
        return False
    
    def _generate_all_waves(self):
        self.all_waves = []
        for i in range(1, self.TOTAL_WAVES + 1):
            wave = []
            num_runners = i
            num_tanks = i // 2
            total_enemies = num_runners + num_tanks
            
            spawn_times = sorted([self.steps + 100 + k * (200 / max(1, total_enemies)) for k in range(total_enemies)])
            
            enemy_pool = ['runner'] * num_runners + ['tank'] * num_tanks
            random.shuffle(enemy_pool)
            
            for j in range(total_enemies):
                wave.append((enemy_pool[j], spawn_times[j]))
            self.all_waves.append(wave)

    def _start_next_wave(self):
        if self.wave_num < self.TOTAL_WAVES:
            self.wave_num += 1
            self.wave_spawn_queue = deque(self.all_waves[self.wave_num - 1])
            self.enemies_in_wave = len(self.wave_spawn_queue)
            self.wave_spawning = True

    def _spawn_enemy(self, enemy_type):
        spawn_hex = random.choice([qr for qr in self.spawn_hexes if self.grid[qr] is None])
        if spawn_hex:
            # sfx: enemy_spawn.wav
            wave_bonus = 1.0 + (0.05 * math.floor(self.wave_num / 5))
            microbe_info = self.MICROBE_TYPES[enemy_type]
            new_enemy = Microbe(enemy_type, spawn_hex[0], spawn_hex[1], microbe_info, wave_bonus)
            self.enemy_units.append(new_enemy)
            self.grid[spawn_hex] = new_enemy

    # --- Rendering ---

    def _render_game(self):
        # Draw grid and special zones
        for q, r in self.grid:
            color = self.COLOR_GRID
            if (q, r) in self.base_hexes: color = self.COLOR_BASE
            if (q, r) in self.spawn_hexes: color = self.COLOR_SPAWN
            self._draw_hexagon(self.screen, color, q, r, 1)

        # Draw units
        all_units = self.player_units + self.enemy_units
        for unit in sorted(all_units, key=lambda u: u.r): # Draw from top to bottom
            self._draw_microbe(unit)

        # Draw particles
        for p in self.particles:
            self._draw_particle(p)
            
        # Draw cursor
        self._draw_hexagon(self.screen, self.COLOR_CURSOR, self.cursor_q, self.cursor_r, 3)

    def _render_ui(self):
        # Wave Info
        wave_text = self.font_m.render(f"WAVE {self.wave_num}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (20, 15))
        enemies_text = self.font_s.render(f"ENEMIES: {self.enemies_in_wave}", True, self.COLOR_TEXT)
        self.screen.blit(enemies_text, (20, 35))
        
        # Resources
        res_text = self.font_l.render(f"{int(self.resources)}", True, self.COLOR_RESOURCE)
        res_icon = self.font_l.render("$", True, self.COLOR_RESOURCE)
        self.screen.blit(res_icon, (self.SCREEN_WIDTH - 120, 15))
        self.screen.blit(res_text, (self.SCREEN_WIDTH - 95, 15))
        
        # Selected Unit
        key = self.player_microbe_keys[self.selected_microbe_idx]
        info = self.MICROBE_TYPES[key]
        sel_text = self.font_m.render(f"Deploy: {info['name']} (${info['cost']})", True, self.COLOR_TEXT)
        text_rect = sel_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 25))
        self.screen.blit(sel_text, text_rect)
        
        # Game Over Text
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            status = "VICTORY" if self.wave_num >= self.TOTAL_WAVES else "DEFEAT"
            color = self.COLOR_PLAYER if status == "VICTORY" else self.COLOR_ENEMY
            end_text = self.font_l.render(status, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def _draw_microbe(self, unit):
        x, y = self._axial_to_pixel(unit.q, unit.r)
        color = self.COLOR_PLAYER if unit.info['faction'] == 'player' else self.COLOR_ENEMY
        
        # Bobbing animation
        bob = math.sin(self.steps * 0.1 + unit.bob_offset) * 2
        y += bob

        # Body
        radius = int(self.HEX_RADIUS * 0.7)
        pygame.gfxdraw.aacircle(self.screen, int(x), int(y), radius, color)
        pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), radius, color)
        
        # Glow effect
        glow_color = (*color, 60)
        s = pygame.Surface((radius*4, radius*4), pygame.SRCALPHA)
        pygame.draw.circle(s, glow_color, (radius*2, radius*2), int(radius*1.2 + abs(bob)))
        self.screen.blit(s, (int(x - radius*2), int(y - radius*2)))

        # Health bar
        bar_width = self.HEX_RADIUS * 1.2
        bar_height = 4
        hp_percent = max(0, unit.hp / unit.max_hp)
        
        bg_rect = pygame.Rect(x - bar_width/2, y - radius - 10, bar_width, bar_height)
        hp_rect = pygame.Rect(x - bar_width/2, y - radius - 10, bar_width * hp_percent, bar_height)
        
        pygame.draw.rect(self.screen, (50,50,50), bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, hp_rect)

    def _draw_particle(self, p):
        if p.life <= 0: return
        alpha = int(255 * p.life)
        color = (*p.color, alpha)
        
        if p.type == 'attack':
            s = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (2,2), 2)
            self.screen.blit(s, (int(p.x)-2, int(p.y)-2))
        elif p.type == 'death':
            radius = int((1.0 - p.life) * 15)
            pygame.gfxdraw.aacircle(self.screen, int(p.x), int(p.y), radius, (*p.color, max(0, alpha-150)))
        elif p.type == 'resource':
            text = self.font_s.render("+1", True, (*p.color, alpha))
            self.screen.blit(text, (int(p.x), int(p.y)))

    # --- Hex Grid Utilities ---
    
    def _axial_to_pixel(self, q, r):
        x = self.grid_offset_x + self.HEX_RADIUS * (3/2 * q)
        y = self.grid_offset_y + self.HEX_HEIGHT / 2 * q + self.HEX_HEIGHT * r
        return x, y

    def _get_neighbors(self, q, r):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        neighbors = []
        for dq, dr in directions:
            nq, nr = q + dq, r + dr
            if (nq, nr) in self.grid:
                neighbors.append((nq, nr))
        return neighbors

    def _hex_distance(self, q1, r1, q2, r2):
        return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2
        
    def _find_closest_target(self, unit, targets):
        closest_target = None
        min_dist = float('inf')
        for target in targets:
            dist = self._hex_distance(unit.q, unit.r, target.q, target.r)
            if dist <= unit.attack_range and dist < min_dist:
                min_dist = dist
                closest_target = target
        return closest_target

    def _draw_hexagon(self, surface, color, q, r, width=0):
        x, y = self._axial_to_pixel(q, r)
        points = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            points.append((x + self.HEX_RADIUS * math.cos(angle_rad),
                           y + self.HEX_RADIUS * math.sin(angle_rad)))
        if width == 0:
            pygame.gfxdraw.filled_polygon(surface, points, color)
        else:
            pygame.gfxdraw.aapolygon(surface, points, color)
            if width > 1: # Draw thicker lines by drawing multiple polygons
                for i in range(1, width):
                    points_inner = []
                    for p_x, p_y in points:
                        points_inner.append((x + (p_x-x)*(1-i/self.HEX_RADIUS), y + (p_y-y)*(1-i/self.HEX_RADIUS)))
                    pygame.gfxdraw.aapolygon(surface, points_inner, color)

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The original code had a separate display for human play,
    # which we need to set up since dummy is the default.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Microbe TD")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    truncated = False
    
    # Game loop for human play
    while running:
        # Action defaults
        movement = 0 # none
        space_held = 0 # released
        shift_held = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated and not truncated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
        if terminated or truncated:
            # Wait a bit before resetting on termination
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False
            truncated = False

    env.close()