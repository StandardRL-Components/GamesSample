
# Generated: 2025-08-27T19:18:49.435996
# Source Brief: brief_02116.md
# Brief Index: 2116

        
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
    def __init__(self, grid_pos, tower_type, stats):
        self.grid_pos = grid_pos
        self.pixel_pos = (
            grid_pos[0] * 40 + 20,
            grid_pos[1] * 40 + 20,
        )
        self.type = tower_type
        self.stats = stats
        self.cooldown = 0
        self.target = None

class Enemy:
    def __init__(self, health, speed, value, path_points):
        self.path_points = path_points
        self.path_index = 0
        self.pos = np.array(self.path_points[0], dtype=float)
        self.max_health = health
        self.health = health
        self.speed_multiplier = 1.0
        self.base_speed = speed
        self.value = value
        self.slow_timer = 0
        self.size = 8 + int(math.log2(max(1, health / 10)))

    def move(self):
        if self.path_index >= len(self.path_points) - 1:
            return True  # Reached the end

        if self.slow_timer > 0:
            self.slow_timer -= 1
            self.speed_multiplier = 0.5
        else:
            self.speed_multiplier = 1.0

        target_point = np.array(self.path_points[self.path_index + 1], dtype=float)
        direction = target_point - self.pos
        distance = np.linalg.norm(direction)
        
        if distance < self.base_speed * self.speed_multiplier:
            self.pos = target_point
            self.path_index += 1
        else:
            direction /= distance
            self.pos += direction * self.base_speed * self.speed_multiplier
        
        return False

class Projectile:
    def __init__(self, start_pos, target, stats):
        self.pos = np.array(start_pos, dtype=float)
        self.target = target
        self.stats = stats
        self.is_aoe = stats.get("aoe_radius", 0) > 0

    def move(self):
        if self.target.health <= 0: # Target already dead
            return True, False 

        direction = self.target.pos - self.pos
        distance = np.linalg.norm(direction)
        
        if distance < self.stats["proj_speed"]:
            return True, True # Hit target
        
        direction /= distance
        self.pos += direction * self.stats["proj_speed"]
        return False, False # In transit

class Particle:
    def __init__(self, pos, vel, color, size, life):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life

    def update(self):
        self.pos += self.vel
        self.life -= 1
        self.size *= 0.98
        return self.life <= 0 or self.size < 0.5


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to place selected tower. Shift to cycle tower types."
    )

    game_description = (
        "A minimalist tower defense game. Place towers on the grid to defend your base from waves of enemies."
    )

    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    CELL_SIZE = 40
    MAX_STEPS = 30 * 120 # 2 minutes at 30fps
    TOTAL_WAVES = 10
    
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 45, 50)
    COLOR_PATH = (60, 80, 110)
    COLOR_BASE = (100, 120, 150)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_ENEMY = (217, 87, 99)
    COLOR_ENEMY_HEALTH = (100, 200, 100)
    
    # Tower stats
    TOWER_TYPES = {
        0: {"name": "Basic", "cost": 50, "range": 80, "damage": 10, "fire_rate": 30, "proj_speed": 8, "color": (0, 255, 127)},
        1: {"name": "Rapid", "cost": 75, "range": 70, "damage": 5, "fire_rate": 10, "proj_speed": 10, "color": (255, 255, 0)},
        2: {"name": "Sniper", "cost": 100, "range": 180, "damage": 35, "fire_rate": 75, "proj_speed": 20, "color": (0, 255, 255)},
        3: {"name": "Splash", "cost": 125, "range": 100, "damage": 15, "fire_rate": 50, "proj_speed": 6, "color": (255, 165, 0), "aoe_radius": 30},
        4: {"name": "Slow", "cost": 60, "range": 100, "damage": 2, "fire_rate": 40, "proj_speed": 7, "color": (255, 0, 255), "slow_duration": 60},
    }

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
        self.font_small = pygame.font.SysFont("sans", 18)
        self.font_large = pygame.font.SysFont("sans", 48)
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.gold = 0
        self.base_health = 0
        self.current_wave_num = 0
        self.wave_spawn_list = []
        self.wave_timer = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.grid = None
        self.path_grid_coords = set()
        self.path_points = self._generate_path()
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()
        self.validate_implementation()
    
    def _generate_path(self):
        path = []
        path_grid = []
        
        start_y = random.randint(2, self.GRID_HEIGHT - 3)
        path.append((-20, start_y * self.CELL_SIZE + 20))
        path_grid.append((0, start_y))

        x, y = 0, start_y
        while x < self.GRID_WIDTH:
            path.append((x * self.CELL_SIZE + 20, y * self.CELL_SIZE + 20))
            path_grid.append((x,y))

            if x > self.GRID_WIDTH - 3:
                x += 1
                continue

            move_y = random.choice([-1, 1]) if x % 3 == 1 else 0
            next_y = np.clip(y + move_y, 1, self.GRID_HEIGHT - 2)
            
            if move_y != 0:
                path.append((x * self.CELL_SIZE + 20, next_y * self.CELL_SIZE + 20))
                path_grid.append((x, next_y))
                y = next_y
            
            x += 1
        
        path.append((self.SCREEN_WIDTH + 20, y * self.CELL_SIZE + 20))
        path_grid.append((self.GRID_WIDTH-1, y))
        self.path_grid_coords = set(path_grid)
        return path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.gold = 100
        self.base_health = 20
        self.current_wave_num = 0
        self.wave_spawn_list = []
        self.wave_timer = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.path_points = self._generate_path()

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave_num += 1
        if self.current_wave_num > self.TOTAL_WAVES:
            self.win = True
            self.game_over = True
            return

        self.wave_spawn_list.clear()
        num_enemies = 3 + self.current_wave_num * 2
        health = 20 * (1.1 ** (self.current_wave_num - 1))
        speed = 1.0 * (1.05 ** (self.current_wave_num - 1))
        value = 5 + self.current_wave_num

        for i in range(num_enemies):
            spawn_time = i * (60 // self.current_wave_num + 5)
            self.wave_spawn_list.append({
                "time": spawn_time,
                "health": health * (1 + random.uniform(-0.1, 0.1)),
                "speed": speed * (1 + random.uniform(-0.1, 0.1)),
                "value": value
            })
        self.wave_timer = 0
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(30)
        reward = 0.01  # Small reward for surviving

        # --- 1. Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        if movement == 2: self.cursor_pos[1] += 1  # Down
        if movement == 3: self.cursor_pos[0] -= 1  # Left
        if movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        if space_held and not self.last_space_held:
            self._place_tower()
        if shift_held and not self.last_shift_held:
            self._cycle_tower_type()
            
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- 2. Update Game Logic ---
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_waves()
        self._update_particles()
        
        self.steps += 1
        self.score += reward

        # --- 3. Check Termination ---
        terminated = False
        if self.base_health <= 0:
            reward -= 100
            self.game_over = True
            terminated = True
        elif self.win:
            reward += 100
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            reward -= 50 # Penalty for timeout
            self.game_over = True
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_tower(self):
        x, y = self.cursor_pos
        stats = self.TOWER_TYPES[self.selected_tower_type]
        
        if self.gold >= stats["cost"] and self.grid[x, y] == 0 and (x, y) not in self.path_grid_coords:
            self.gold -= stats["cost"]
            self.grid[x, y] = self.selected_tower_type + 1
            self.towers.append(Tower((x, y), self.selected_tower_type, stats))
            # sfx: place_tower.wav
            for _ in range(20):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                pos = (x * self.CELL_SIZE + 20, y * self.CELL_SIZE + 20)
                life = random.randint(10, 20)
                self.particles.append(Particle(pos, vel, stats["color"], 5, life))


    def _cycle_tower_type(self):
        self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
        # sfx: cycle.wav

    def _update_waves(self):
        self.wave_timer += 1
        
        if self.wave_spawn_list:
            if self.wave_timer >= self.wave_spawn_list[0]["time"]:
                enemy_data = self.wave_spawn_list.pop(0)
                self.enemies.append(Enemy(enemy_data["health"], enemy_data["speed"], enemy_data["value"], self.path_points))
        
        elif not self.enemies and not self.game_over:
            self._start_next_wave()

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            if tower.cooldown > 0:
                tower.cooldown -= 1
                continue
            
            # Find target
            possible_targets = []
            for enemy in self.enemies:
                dist = np.linalg.norm(np.array(tower.pixel_pos) - enemy.pos)
                if dist <= tower.stats["range"]:
                    possible_targets.append(enemy)

            if possible_targets:
                # Target the enemy furthest along the path
                target = max(possible_targets, key=lambda e: e.path_index + np.linalg.norm(e.pos - e.path_points[e.path_index]))
                tower.target = target
                
                # Fire projectile
                self.projectiles.append(Projectile(tower.pixel_pos, target, tower.stats))
                tower.cooldown = tower.stats["fire_rate"]
                # sfx: shoot.wav
        return reward

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        for proj in self.projectiles:
            removed, hit = proj.move()
            if removed:
                projectiles_to_remove.append(proj)
                if hit:
                    # sfx: hit.wav
                    # Handle AoE damage
                    if proj.is_aoe:
                        for enemy in self.enemies:
                            if np.linalg.norm(enemy.pos - proj.target.pos) <= proj.stats["aoe_radius"]:
                                killed = self._damage_enemy(enemy, proj.stats["damage"])
                                if killed: reward += 1.0
                        # AoE particle effect
                        for _ in range(30):
                            angle = random.uniform(0, 2 * math.pi)
                            speed = random.uniform(0.5, 2.5)
                            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                            life = random.randint(15, 25)
                            self.particles.append(Particle(proj.target.pos, vel, proj.stats["color"], 7, life))
                    else: # Single target damage
                        killed = self._damage_enemy(proj.target, proj.stats["damage"])
                        if killed: reward += 1.0
                    
                    # Handle slow
                    if "slow_duration" in proj.stats:
                        proj.target.slow_timer = proj.stats["slow_duration"]

        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        return reward

    def _damage_enemy(self, enemy, damage):
        enemy.health -= damage
        if enemy.health <= 0 and enemy in self.enemies:
            # sfx: enemy_death.wav
            self.enemies.remove(enemy)
            self.gold += enemy.value
            # Death particle effect
            for _ in range(15):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 4)
                vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                life = random.randint(10, 20)
                self.particles.append(Particle(enemy.pos, vel, self.COLOR_ENEMY, enemy.size, life))
            return True
        return False

    def _update_enemies(self):
        reward = 0
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy.move():
                enemies_to_remove.append(enemy)
                self.base_health -= 1
                reward -= 0.5
                # sfx: base_damage.wav
                # Base damage particle effect
                for _ in range(25):
                    pos = (self.SCREEN_WIDTH - 10, enemy.pos[1])
                    vel = (random.uniform(-5, -1), random.uniform(-2, 2))
                    life = random.randint(20, 40)
                    self.particles.append(Particle(pos, vel, self.COLOR_ENEMY, 8, life))
        
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if not p.update()]

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
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.current_wave_num,
        }

    def _render_game(self):
        # Grid
        for x in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.CELL_SIZE, 0), (x * self.CELL_SIZE, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.CELL_SIZE), (self.SCREEN_WIDTH, y * self.CELL_SIZE))

        # Path and Base
        if len(self.path_points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_points, 38)
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_points, 36)
        base_y = self.path_points[-1][1]
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.SCREEN_WIDTH-10, base_y - 20, 10, 40))

        # Towers
        for tower in self.towers:
            x, y = tower.pixel_pos
            color = tower.stats["color"]
            if tower.type == 0: # Square
                pygame.draw.rect(self.screen, color, (x-10, y-10, 20, 20))
            elif tower.type == 1: # Triangle
                pygame.gfxdraw.aapolygon(self.screen, [(x, y-11), (x-10, y+8), (x+10, y+8)], color)
                pygame.gfxdraw.filled_polygon(self.screen, [(x, y-11), (x-10, y+8), (x+10, y+8)], color)
            elif tower.type == 2: # Circle
                pygame.gfxdraw.aacircle(self.screen, int(x), int(y), 10, color)
                pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 10, color)
            elif tower.type == 3: # Hexagon
                points = [(x + 12 * math.cos(math.pi/3 * i), y + 12 * math.sin(math.pi/3 * i)) for i in range(6)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            elif tower.type == 4: # Diamond
                points = [(x, y-12), (x+10, y), (x, y+12), (x-10, y)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Projectiles
        for proj in self.projectiles:
            pygame.draw.line(self.screen, proj.stats["color"], proj.pos, proj.pos + (proj.target.pos - proj.pos)*0.1, 3)

        # Enemies
        for enemy in self.enemies:
            x, y = int(enemy.pos[0]), int(enemy.pos[1])
            size = enemy.size
            pygame.gfxdraw.aacircle(self.screen, x, y, size, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, x, y, size, self.COLOR_ENEMY)
            # Health bar
            health_ratio = enemy.health / enemy.max_health
            if health_ratio < 1:
                pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH, (x - size, y - size - 5, int(size * 2 * health_ratio), 3))
            if enemy.slow_timer > 0:
                pygame.gfxdraw.aacircle(self.screen, x, y, size + 2, self.TOWER_TYPES[4]["color"])
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = (*p.color, alpha)
            temp_surf = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p.size, p.size), p.size)
            self.screen.blit(temp_surf, p.pos - (p.size, p.size))
            
        # Cursor
        cx, cy = self.cursor_pos
        px, py = cx * self.CELL_SIZE, cy * self.CELL_SIZE
        cursor_color = self.COLOR_CURSOR
        stats = self.TOWER_TYPES[self.selected_tower_type]
        is_valid_pos = self.grid[cx, cy] == 0 and (cx, cy) not in self.path_grid_coords and self.gold >= stats["cost"]
        
        # Range indicator
        range_color = (255, 255, 255, 30) if is_valid_pos else (255, 0, 0, 30)
        range_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(range_surf, px + 20, py + 20, stats["range"], range_color)
        self.screen.blit(range_surf, (0,0))
        
        # Cursor rect
        cursor_rect_color = (0, 255, 0) if is_valid_pos else (255, 0, 0)
        pygame.draw.rect(self.screen, cursor_rect_color, (px, py, self.CELL_SIZE, self.CELL_SIZE), 2)


    def _render_ui(self):
        # Top bar UI
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 30), pygame.SRCALPHA)
        ui_surf.fill((0,0,0,128))
        self.screen.blit(ui_surf, (0,0))

        gold_text = self.font_small.render(f"Gold: {self.gold}", True, (255, 215, 0))
        self.screen.blit(gold_text, (10, 5))

        wave_text = self.font_small.render(f"Wave: {self.current_wave_num}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 5))

        health_text = self.font_small.render(f"Base Health: {self.base_health}", True, (173, 216, 230))
        self.screen.blit(health_text, (self.SCREEN_WIDTH/2 - health_text.get_width()/2, 5))

        # Bottom bar UI (Tower Selection)
        stats = self.TOWER_TYPES[self.selected_tower_type]
        bar_width = 400
        bar_height = 50
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = self.SCREEN_HEIGHT - bar_height - 10
        
        bottom_surf = pygame.Surface((bar_width, bar_height), pygame.SRCALPHA)
        bottom_surf.fill((0,0,0,128))
        
        name_text = self.font_small.render(f"{stats['name']} Tower", True, stats["color"])
        cost_text = self.font_small.render(f"Cost: {stats['cost']}", True, self.COLOR_TEXT)
        info_text = self.font_small.render(f"DMG: {stats['damage']} | RNG: {stats['range']} | ROF: {60/stats['fire_rate']:.1f}/s", True, self.COLOR_TEXT)
        
        bottom_surf.blit(name_text, (10, 5))
        bottom_surf.blit(cost_text, (bar_width - cost_text.get_width() - 10, 5))
        bottom_surf.blit(info_text, (10, 25))

        self.screen.blit(bottom_surf, (bar_x, bar_y))

        # Game Over / Win Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.win:
                end_text = self.font_large.render("VICTORY", True, (0, 255, 127))
            else:
                end_text = self.font_large.render("GAME OVER", True, (217, 87, 99))
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            overlay.blit(end_text, text_rect)
            self.screen.blit(overlay, (0, 0))

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- To display the game in a window ---
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        # Get user input for manual play
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                terminated = False

        if terminated:
            print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for 'r' to reset
            pass

    pygame.quit()