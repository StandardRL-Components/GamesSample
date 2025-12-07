
# Generated: 2025-08-27T22:17:26.836077
# Source Brief: brief_03072.md
# Brief Index: 3072

        
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


# Helper classes for game objects
class Tower:
    def __init__(self, grid_pos, tower_type, iso_pos):
        self.grid_pos = grid_pos
        self.type = tower_type
        self.iso_pos = iso_pos
        self.cooldown = 0
        self.target = None

class Enemy:
    def __init__(self, path, health, speed, gold_value, color, size, np_random):
        self.path = path
        self.path_index = 0
        self.pos = np.array(self.path[0], dtype=float)
        self.health = health
        self.max_health = health
        self.speed = speed
        self.gold_value = gold_value
        self.color = color
        self.size = size
        self.np_random = np_random
        self.target_pos = np.array(self.path[1], dtype=float)
        self.progress = 0.0

    def move(self):
        if self.path_index >= len(self.path) - 1:
            return True  # Reached the end

        start_node = np.array(self.path[self.path_index], dtype=float)
        end_node = np.array(self.path[self.path_index + 1], dtype=float)
        
        distance_to_travel = self.speed
        
        while distance_to_travel > 0:
            remaining_dist_on_segment = np.linalg.norm(end_node - self.pos)
            
            if remaining_dist_on_segment == 0:
                self.path_index += 1
                if self.path_index >= len(self.path) - 1:
                    self.pos = end_node
                    return True
                start_node = np.array(self.path[self.path_index], dtype=float)
                end_node = np.array(self.path[self.path_index + 1], dtype=float)
                continue

            move_dist = min(distance_to_travel, remaining_dist_on_segment)
            direction = (end_node - self.pos) / remaining_dist_on_segment
            self.pos += direction * move_dist
            distance_to_travel -= move_dist

        return False

class Projectile:
    def __init__(self, start_pos, target_enemy, damage, color):
        self.pos = np.array(start_pos, dtype=float)
        self.target = target_enemy
        self.damage = damage
        self.speed = 15.0
        self.color = color

    def move(self):
        if self.target is None or self.target.health <= 0:
            return True # Target is gone
        
        direction = self.target.pos - self.pos
        dist = np.linalg.norm(direction)
        if dist < self.speed:
            self.pos = self.target.pos
            return True # Hit target
        
        self.pos += (direction / dist) * self.speed
        return False

class Particle:
    def __init__(self, pos, color, lifetime, size, velocity):
        self.pos = np.array(pos, dtype=float)
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size
        self.velocity = np.array(velocity, dtype=float)

    def update(self):
        self.pos += self.velocity
        self.lifetime -= 1
        return self.lifetime <= 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Arrows to move cursor, Shift to cycle towers, Space to place selected tower."
    game_description = "Minimalist isometric tower defense. Place towers to defend your base against waves of enemies."
    
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CONSTANTS ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 14, 11
        self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO = 40, 20
        self.ORIGIN_X = self.SCREEN_WIDTH // 2
        self.ORIGIN_Y = 80
        
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_PATH = (50, 65, 90)
        self.COLOR_BASE = (50, 180, 100)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_GOLD = (255, 200, 0)
        self.COLOR_HEALTH = (220, 50, 50)
        self.COLOR_CURSOR_VALID = (200, 255, 200, 100)
        self.COLOR_CURSOR_INVALID = (255, 100, 100, 100)

        self.MAX_STEPS = 5000
        self.TOTAL_WAVES = 10
        self.PREP_PHASE_DURATION = 150 # 5 seconds at 30fps

        # --- SPACES ---
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 20)
        self.font_m = pygame.font.Font(None, 28)
        self.font_l = pygame.font.Font(None, 48)

        # --- GAME DEFINITIONS ---
        self._define_path_and_grid()
        self._define_tower_types()

        self.reset()
        self.validate_implementation()
    
    def _define_path_and_grid(self):
        raw_path = [(0, 5), (3, 5), (3, 2), (10, 2), (10, 8), (13, 8)]
        self.path_nodes = []
        for i in range(len(raw_path) - 1):
            p1 = raw_path[i]
            p2 = raw_path[i+1]
            dx = np.sign(p2[0] - p1[0]) if p1[0] != p2[0] else 0
            dy = np.sign(p2[1] - p1[1]) if p1[1] != p2[1] else 0
            curr = p1
            while curr != p2:
                self.path_nodes.append(curr)
                curr = (curr[0] + dx, curr[1] + dy)
        self.path_nodes.append(raw_path[-1])
        
        self.iso_path = [self._iso_transform(p[0], p[1]) for p in self.path_nodes]
        self.path_tiles = set(self.path_nodes)
        self.base_pos = self.path_nodes[-1]

        self.grid_iso_points = {}
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                self.grid_iso_points[(x, y)] = self._iso_transform(x, y)

    def _define_tower_types(self):
        self.tower_types = [
            {"name": "Gatling", "cost": 50, "damage": 5, "range": 2.5, "rate": 10, "color": (100, 150, 255), "proj_color": (180, 220, 255)},
            {"name": "Cannon", "cost": 120, "damage": 25, "range": 3.5, "rate": 40, "color": (255, 150, 50), "proj_color": (255, 200, 150)},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = 100
        self.gold = 150
        self.wave_number = 0
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type_idx = 0
        
        self.game_phase = "PREP"
        self.prep_timer = self.PREP_PHASE_DURATION
        
        self.enemies_to_spawn = []
        self.wave_spawn_timer = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.step_reward = -0.001 # Small penalty for time passing

        self._handle_input(action)
        self._update_game_state()
        
        terminated = self._check_termination()
        
        self.score += self.step_reward
        
        return self._get_observation(), self.step_reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Actions on Press ---
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        
        if shift_press:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.tower_types)
        
        if space_press:
            self._place_tower()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_game_state(self):
        if self.game_phase == "PREP":
            self.prep_timer -= 1
            if self.prep_timer <= 0:
                self.game_phase = "WAVE_ACTIVE"
                self.wave_number += 1
                self._spawn_wave()
                self.wave_spawn_timer = 0
        
        elif self.game_phase == "WAVE_ACTIVE":
            self.wave_spawn_timer += 1
            if self.enemies_to_spawn and self.wave_spawn_timer % 30 == 0:
                self.enemies.append(self.enemies_to_spawn.pop(0))
            
            if not self.enemies_to_spawn and not self.enemies:
                self.game_phase = "PREP"
                self.prep_timer = self.PREP_PHASE_DURATION
                if self.wave_number < self.TOTAL_WAVES:
                    self.step_reward += 10
                    self.gold += 100 + self.wave_number * 10
                # Win condition is checked in _check_termination

        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
    def _spawn_wave(self):
        num_enemies = 3 + self.wave_number * 2
        base_health = 20
        base_speed = 0.03
        
        for _ in range(num_enemies):
            health = int(base_health * (1.15 ** self.wave_number))
            speed = base_speed * (1.05 ** self.wave_number)
            color_val = min(255, 100 + self.wave_number * 15)
            enemy = Enemy(self.iso_path, health, speed, 10, (color_val, 50, 80), 6, self.np_random)
            self.enemies_to_spawn.append(enemy)

    def _update_towers(self):
        for tower in self.towers:
            tower.cooldown = max(0, tower.cooldown - 1)
            if tower.cooldown > 0:
                continue
            
            # Find target
            tower.target = None
            min_dist = float('inf')
            for enemy in self.enemies:
                dist = np.linalg.norm(np.array(tower.iso_pos) - enemy.pos)
                if dist < tower.type['range'] * self.TILE_HEIGHT_ISO * 2: # Range in pixels
                    if dist < min_dist:
                        min_dist = dist
                        tower.target = enemy
            
            if tower.target:
                # Fire projectile
                # SFX: Pew! or Cannon Fire!
                proj = Projectile(tower.iso_pos, tower.target, tower.type['damage'], tower.type['proj_color'])
                self.projectiles.append(proj)
                tower.cooldown = tower.type['rate']
    
    def _update_projectiles(self):
        projectiles_to_remove = []
        for proj in self.projectiles:
            if proj.move():
                projectiles_to_remove.append(proj)
                if proj.target and proj.target.health > 0:
                    proj.target.health -= proj.damage
                    # SFX: Hit!
                    self._create_particles(proj.pos, (255,255,255), 3, 5)
                    if proj.target.health <= 0:
                        self._on_enemy_killed(proj.target)
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]

    def _update_enemies(self):
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy.health <= 0:
                enemies_to_remove.append(enemy)
                continue
            if enemy.move():
                enemies_to_remove.append(enemy)
                self.base_health -= 10
                # SFX: Base damage alarm!
                self._create_particles(enemy.pos, self.COLOR_HEALTH, 10, 15)
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]

    def _on_enemy_killed(self, enemy):
        self.gold += enemy.gold_value
        self.step_reward += 0.2
        # SFX: Enemy explosion!
        self._create_particles(enemy.pos, enemy.color, 15, 10)

    def _update_particles(self):
        self.particles = [p for p in self.particles if not p.update()]

    def _place_tower(self):
        pos_tuple = tuple(self.cursor_pos)
        selected_type = self.tower_types[self.selected_tower_type_idx]
        
        # Check validity
        is_on_path = pos_tuple in self.path_tiles
        is_occupied = any(t.grid_pos == pos_tuple for t in self.towers)
        has_enough_gold = self.gold >= selected_type['cost']

        if not is_on_path and not is_occupied and has_enough_gold:
            self.gold -= selected_type['cost']
            iso_pos = self.grid_iso_points[pos_tuple]
            new_tower = Tower(pos_tuple, selected_type, iso_pos)
            self.towers.append(new_tower)
            # SFX: Place tower
            self.step_reward += 0.1 # Small reward for a good action

    def _check_termination(self):
        if self.game_over:
            return True
        
        if self.base_health <= 0:
            self.game_over = True
            self.win = False
            self.step_reward -= 100
        elif self.wave_number >= self.TOTAL_WAVES and not self.enemies and not self.enemies_to_spawn:
            self.game_over = True
            self.win = True
            self.step_reward += 100
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
        
        return self.game_over

    def _iso_transform(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * self.TILE_WIDTH_ISO / 2
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * self.TILE_HEIGHT_ISO / 2
        return int(screen_x), int(screen_y)

    def _create_particles(self, pos, color, count, lifetime):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.uniform(2, 5)
            p = Particle(pos, color, lifetime, size, velocity)
            self.particles.append(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "gold": self.gold, "health": self.base_health, "wave": self.wave_number}

    def _render_game(self):
        # Render grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                points = [
                    self._iso_transform(x, y), self._iso_transform(x + 1, y),
                    self._iso_transform(x + 1, y + 1), self._iso_transform(x, y + 1)
                ]
                if (x,y) in self.path_tiles:
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PATH)
                else:
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Render base
        points = [ self._iso_transform(self.base_pos[0], self.base_pos[1]), self._iso_transform(self.base_pos[0] + 1, self.base_pos[1]),
                   self._iso_transform(self.base_pos[0] + 1, self.base_pos[1] + 1), self._iso_transform(self.base_pos[0], self.base_pos[1] + 1) ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BASE)
        
        # Create sorted list of dynamic objects for correct Z-ordering
        render_list = self.towers + self.enemies
        render_list.sort(key=lambda obj: obj.iso_pos[1] if isinstance(obj, Tower) else obj.pos[1])

        for obj in render_list:
            if isinstance(obj, Tower):
                self._render_tower(obj)
            elif isinstance(obj, Enemy):
                self._render_enemy(obj)

        self._render_projectiles()
        self._render_particles()
        self._render_cursor()

    def _render_tower(self, tower):
        center_x, center_y = tower.iso_pos
        size = self.TILE_WIDTH_ISO * 0.3
        points = [ (center_x, center_y - size), (center_x + size, center_y),
                   (center_x, center_y + size), (center_x - size, center_y) ]
        pygame.gfxdraw.filled_polygon(self.screen, points, tower.type['color'])
        pygame.gfxdraw.aapolygon(self.screen, points, tuple(c*0.7 for c in tower.type['color']))
        
        # Firing flash
        if tower.cooldown > tower.type['rate'] - 3:
             pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 5, (255,255,255,150))

    def _render_enemy(self, enemy):
        x, y = int(enemy.pos[0]), int(enemy.pos[1])
        pygame.gfxdraw.filled_circle(self.screen, x, y, int(enemy.size), enemy.color)
        pygame.gfxdraw.aacircle(self.screen, x, y, int(enemy.size), tuple(c*0.7 for c in enemy.color))
        
        # Health bar
        if enemy.health < enemy.max_health:
            bar_width = 20
            health_pct = enemy.health / enemy.max_health
            pygame.draw.rect(self.screen, (50,50,50), (x - bar_width/2, y - 15, bar_width, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH, (x - bar_width/2, y - 15, bar_width * health_pct, 5))

    def _render_projectiles(self):
        for proj in self.projectiles:
            x, y = int(proj.pos[0]), int(proj.pos[1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 3, proj.color)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p.lifetime / p.max_lifetime))
            color = (*p.color, alpha)
            size = int(p.size * (p.lifetime / p.max_lifetime))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), size, color)
    
    def _render_cursor(self):
        x, y = self.cursor_pos
        points = [ self._iso_transform(x, y), self._iso_transform(x + 1, y),
                   self._iso_transform(x + 1, y + 1), self._iso_transform(x, y + 1) ]
        
        pos_tuple = tuple(self.cursor_pos)
        selected_type = self.tower_types[self.selected_tower_type_idx]
        is_valid = not (pos_tuple in self.path_tiles) and not any(t.grid_pos == pos_tuple for t in self.towers) and self.gold >= selected_type['cost']
        
        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        pygame.gfxdraw.filled_polygon(self.screen, points, cursor_color)

        # Draw range indicator
        center_x, center_y = self.grid_iso_points[pos_tuple]
        radius = int(selected_type['range'] * self.TILE_HEIGHT_ISO * 2)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, cursor_color)

    def _render_ui(self):
        # Top Bar
        bar_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, (0,0,0,150), bar_rect)

        # Health
        health_text = self.font_m.render(f"Base Health: {self.base_health}", True, self.COLOR_HEALTH)
        self.screen.blit(health_text, (10, 10))

        # Gold
        gold_text = self.font_m.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (200, 10))

        # Wave
        wave_str = f"Wave: {self.wave_number}/{self.TOTAL_WAVES}"
        if self.game_phase == "PREP" and self.wave_number < self.TOTAL_WAVES:
            wave_str += f" (Starts in {self.prep_timer//30 + 1}s)"
        wave_text = self.font_m.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (350, 10))

        # Selected Tower Info
        tower = self.tower_types[self.selected_tower_type_idx]
        tower_text = self.font_s.render(f"Selected: {tower['name']} (Cost: {tower['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (10, self.SCREEN_HEIGHT - 25))

        # Game Over / Win
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_BASE if self.win else self.COLOR_HEALTH
            end_text = self.font_l.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    running = True
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # movement, space, shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            
    pygame.quit()