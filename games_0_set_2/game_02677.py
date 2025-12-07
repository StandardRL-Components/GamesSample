
# Generated: 2025-08-28T05:35:06.227346
# Source Brief: brief_02677.md
# Brief Index: 2677

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import heapq
import os
import pygame


# --- Helper Classes ---

class Enemy:
    def __init__(self, pos, wave, grid_size):
        self.pos = pygame.Vector2(pos)
        self.max_health = 50 + (wave * 10)
        self.health = self.max_health
        self.speed = 1.0 + (wave * 0.05)
        self.damage = 10
        self.gold_value = 5 + wave
        self.path = []
        self.path_index = 0
        self.radius = grid_size / 2.5
        self.color = (220, 50, 50) # Bright Red

    def move(self):
        if self.path_index < len(self.path):
            target_pos = self.path[self.path_index]
            direction = target_pos - self.pos
            if direction.length() < self.speed:
                self.pos = target_pos
                self.path_index += 1
            else:
                self.pos += direction.normalize() * self.speed
        return self.path_index >= len(self.path)

    def draw(self, surface):
        # Body
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), int(self.radius))
        # Health bar
        if self.health < self.max_health:
            bar_width = self.radius * 2
            bar_height = 4
            health_pct = self.health / self.max_health
            fill_width = int(bar_width * health_pct)
            bg_rect = pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius - 8, bar_width, bar_height)
            fill_rect = pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius - 8, fill_width, bar_height)
            pygame.draw.rect(surface, (50, 50, 50), bg_rect)
            pygame.draw.rect(surface, (50, 200, 50), fill_rect)

class Tower:
    def __init__(self, grid_pos, tower_type, grid_size):
        self.grid_pos = grid_pos
        self.center_pos = pygame.Vector2(
            (grid_pos[0] + 0.5) * grid_size,
            (grid_pos[1] + 0.5) * grid_size
        )
        self.type = tower_type
        self.cooldown = 0
        self.size = grid_size * 0.4

        if self.type == 0: # Basic Gun
            self.range = grid_size * 3.5
            self.fire_rate = 30 # frames per shot
            self.damage = 12
            self.cost = 50
            self.color = (0, 150, 255) # Bright Blue
        elif self.type == 1: # Sniper
            self.range = grid_size * 7
            self.fire_rate = 90
            self.damage = 50
            self.cost = 120
            self.color = (150, 100, 255) # Purple

    def update(self, enemies, projectiles):
        if self.cooldown > 0:
            self.cooldown -= 1
            return

        target = self.find_target(enemies)
        if target:
            # sfx: tower_shoot.wav
            projectiles.append(Projectile(self.center_pos, target, self.damage))
            self.cooldown = self.fire_rate

    def find_target(self, enemies):
        in_range_enemies = [e for e in enemies if self.center_pos.distance_to(e.pos) <= self.range]
        if not in_range_enemies:
            return None
        # Target enemy closest to the base (last in its path)
        return min(in_range_enemies, key=lambda e: len(e.path) - e.path_index)

    def draw(self, surface):
        points = [
            (self.center_pos.x, self.center_pos.y - self.size),
            (self.center_pos.x - self.size, self.center_pos.y + self.size),
            (self.center_pos.x + self.size, self.center_pos.y + self.size),
        ]
        pygame.draw.polygon(surface, self.color, points)
        pygame.gfxdraw.aapolygon(surface, points, self.color)

class Projectile:
    def __init__(self, pos, target, damage):
        self.pos = pygame.Vector2(pos)
        self.target = target
        self.damage = damage
        self.speed = 10
        self.color = (255, 255, 0) # Yellow

    def update(self):
        if self.target.health <= 0:
            return True # Target is already dead

        direction = self.target.pos - self.pos
        if direction.length() < self.speed:
            self.target.health -= self.damage
            # sfx: enemy_hit.wav
            return True
        else:
            self.pos += direction.normalize() * self.speed
            return False

    def draw(self, surface):
        pygame.draw.line(surface, self.color, self.pos, self.pos + (self.target.pos - self.pos).normalize() * 5, 3)

class Particle:
    def __init__(self, pos, color):
        self.pos = pygame.Vector2(pos)
        self.color = color
        self.radius = random.uniform(4, 10)
        self.max_lifespan = 20
        self.lifespan = self.max_lifespan
        self.vel = pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        return self.lifespan <= 0

    def draw(self, surface):
        alpha = int(255 * (self.lifespan / self.max_lifespan))
        current_radius = int(self.radius * (1 - (self.lifespan / self.max_lifespan)))
        temp_surf = pygame.Surface((current_radius*2, current_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*self.color, alpha), (current_radius, current_radius), current_radius)
        surface.blit(temp_surf, (int(self.pos.x - current_radius), int(self.pos.y - current_radius)))

# --- A* Pathfinding ---
def a_star_search(grid, start, end):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = { (r, c): float('inf') for r in range(len(grid)) for c in range(len(grid[0])) }
    g_score[start] = 0
    f_score = { (r, c): float('inf') for r in range(len(grid)) for c in range(len(grid[0])) }
    f_score[start] = abs(start[0] - end[0]) + abs(start[1] - end[1])

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None # No path found

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor, Space to place tower, Shift to cycle tower type."
    )
    game_description = (
        "Defend your base from waves of enemies by strategically placing defensive towers in a top-down tactical arena."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.GRID_SIZE = self.WIDTH // self.GRID_W

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Colors and Fonts ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_BASE = (50, 200, 50) # Green
        self.COLOR_SPAWN = (100, 30, 30)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.FONT_UI = pygame.font.SysFont("sans-serif", 20, bold=True)
        self.FONT_MSG = pygame.font.SysFont("sans-serif", 40, bold=True)

        self.tower_types = [
            {"name": "Gun", "cost": 50},
            {"name": "Sniper", "cost": 120},
        ]

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.max_base_health = 100
        self.gold = 0
        self.current_wave = 0
        self.max_waves = 10
        self.wave_timer = 0
        self.wave_cooldown = 300 # frames (10 seconds at 30fps)
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        
        self.grid = []
        self.spawn_pos_grid = (0, self.GRID_H // 2)
        self.base_pos_grid = (self.GRID_W - 1, self.GRID_H // 2)

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_type = 0

        self.last_space_held = False
        self.last_shift_held = False

        self.reset()
        # self.validate_implementation() # Optional: call to verify setup

    def _grid_to_pixel(self, grid_pos):
        return pygame.Vector2(
            (grid_pos[0] + 0.5) * self.GRID_SIZE,
            (grid_pos[1] + 0.5) * self.GRID_SIZE
        )

    def _create_particle_burst(self, pos, color, count=15):
        for _ in range(count):
            self.particles.append(Particle(pos, color))

    def _recalculate_all_paths(self):
        for enemy in self.enemies:
            path_grid = a_star_search(self.grid, (int(enemy.pos.x // self.GRID_SIZE), int(enemy.pos.y // self.GRID_SIZE)), self.base_pos_grid)
            if path_grid:
                enemy.path = [self._grid_to_pixel(p) for p in path_grid]
                enemy.path.append(self._grid_to_pixel(self.base_pos_grid))
                enemy.path_index = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.max_base_health
        self.gold = 150
        self.current_wave = 0
        self.wave_timer = self.wave_cooldown
        self.enemies_to_spawn = 0
        
        self.grid = [[0] * self.GRID_H for _ in range(self.GRID_W)]
        self.grid[self.spawn_pos_grid[0]][self.spawn_pos_grid[1]] = 2 # Mark spawn
        self.grid[self.base_pos_grid[0]][self.base_pos_grid[1]] = 3 # Mark base

        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.max_waves:
            return
        # sfx: wave_start.wav
        self.enemies_to_spawn = 5 + self.current_wave
        self.spawn_timer = 0

    def step(self, action):
        reward = 0
        self.game_over = self.base_health <= 0 or (self.current_wave > self.max_waves and not self.enemies)
        if self.game_over:
            if self.base_health <= 0:
                reward -= 100
            else: # Win condition
                reward += 100
            return self._get_observation(), reward, True, False, self._get_info()

        # 1. Handle Input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_H - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)

        if space_held and not self.last_space_held:
            tower_cost = self.tower_types[self.selected_tower_type]["cost"]
            x, y = self.cursor_pos[0], self.cursor_pos[1]
            if self.gold >= tower_cost and self.grid[x][y] == 0:
                # Check if placement would block the path
                self.grid[x][y] = 1
                if a_star_search(self.grid, self.spawn_pos_grid, self.base_pos_grid):
                    self.gold -= tower_cost
                    self.towers.append(Tower((x, y), self.selected_tower_type, self.GRID_SIZE))
                    # sfx: place_tower.wav
                    self._recalculate_all_paths()
                else: # Path blocked, revert
                    self.grid[x][y] = 0
                    # sfx: error.wav
        
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_types)
            # sfx: cycle.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # 2. Update Game Logic
        # Wave management
        if not self.enemies and self.enemies_to_spawn == 0:
            if self.current_wave > 0 and self.current_wave <= self.max_waves:
                reward += 100 # Wave survived bonus
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_next_wave()
        
        # Enemy spawning
        if self.enemies_to_spawn > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                spawn_pixel_pos = self._grid_to_pixel(self.spawn_pos_grid)
                new_enemy = Enemy(spawn_pixel_pos, self.current_wave, self.GRID_SIZE)
                path_grid = a_star_search(self.grid, self.spawn_pos_grid, self.base_pos_grid)
                if path_grid:
                    new_enemy.path = [self._grid_to_pixel(p) for p in path_grid]
                    new_enemy.path.append(self._grid_to_pixel(self.base_pos_grid))
                self.enemies.append(new_enemy)
                self.enemies_to_spawn -= 1
                self.spawn_timer = 15 # frames between spawns

        # Update entities
        for enemy in self.enemies:
            if enemy.move():
                self.base_health -= enemy.damage
                reward -= enemy.damage * 0.01
                enemy.health = 0 # Mark for removal
                self._create_particle_burst(enemy.pos, self.COLOR_BASE, 20)
                # sfx: base_damage.wav
        
        for tower in self.towers:
            tower.update(self.enemies, self.projectiles)

        projectiles_to_remove = []
        for p in self.projectiles:
            if p.update():
                projectiles_to_remove.append(p)
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy.health <= 0:
                if enemy.path_index < len(enemy.path): # Didn't reach base
                    reward += 1.0 # Kill reward
                    self.gold += enemy.gold_value
                    reward += enemy.gold_value * 0.1
                    self._create_particle_burst(enemy.pos, enemy.color)
                    # sfx: enemy_death.wav
                enemies_to_remove.append(enemy)
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]

        particles_to_remove = []
        for p in self.particles:
            if p.update():
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]
        
        self.steps += 1
        terminated = self.base_health <= 0 or (self.current_wave > self.max_waves and not self.enemies) or self.steps >= 1000
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_W):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.GRID_SIZE, 0), (x * self.GRID_SIZE, self.HEIGHT))
        for y in range(self.GRID_H):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.GRID_SIZE), (self.WIDTH, y * self.GRID_SIZE))
        
        # Draw spawn and base
        spawn_rect = pygame.Rect(self.spawn_pos_grid[0] * self.GRID_SIZE, self.spawn_pos_grid[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        base_rect = pygame.Rect(self.base_pos_grid[0] * self.GRID_SIZE, self.base_pos_grid[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SPAWN, spawn_rect)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        # Draw entities
        for tower in self.towers:
            tower.draw(self.screen)
        for p in self.projectiles:
            p.draw(self.screen)
        for enemy in self.enemies:
            enemy.draw(self.screen)
        for p in self.particles:
            p.draw(self.screen)

        # Draw cursor and tower placement preview
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(cursor_x * self.GRID_SIZE, cursor_y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        
        tower_info = self.tower_types[self.selected_tower_type]
        can_afford = self.gold >= tower_info["cost"]
        is_empty = self.grid[cursor_x][cursor_y] == 0
        
        if is_empty:
            preview_color = (0, 255, 0, 100) if can_afford else (255, 0, 0, 100)
            
            # Draw tower range
            temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*preview_color[:3], 50), self._grid_to_pixel(self.cursor_pos), self.GRID_SIZE * (3.5 if self.selected_tower_type == 0 else 7.0), 0)
            self.screen.blit(temp_surf, (0,0))
            
            # Draw tower preview
            temp_surf = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
            size = self.GRID_SIZE * 0.4
            center = self.GRID_SIZE / 2
            points = [(center, center - size), (center - size, center + size), (center + size, center + size)]
            pygame.draw.polygon(temp_surf, preview_color, points)
            self.screen.blit(temp_surf, cursor_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)


    def _render_ui(self):
        # Top Bar
        bar_bg = pygame.Rect(0, 0, self.WIDTH, 30)
        pygame.draw.rect(self.screen, (0,0,0, 150), bar_bg)

        # Health
        health_text = self.FONT_UI.render(f"Base HP: {max(0, self.base_health)} / {self.max_base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 5))
        
        # Gold
        gold_text = self.FONT_UI.render(f"Gold: {self.gold}", True, (255, 223, 0))
        self.screen.blit(gold_text, (200, 5))

        # Wave
        wave_text = self.FONT_UI.render(f"Wave: {self.current_wave} / {self.max_waves}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (320, 5))
        
        # Selected Tower
        tower_info = self.tower_types[self.selected_tower_type]
        tower_text = self.FONT_UI.render(f"Build: {tower_info['name']} ({tower_info['cost']}G)", True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (450, 5))

        # Game Over / Win Message
        if self.game_over:
            msg = "YOU WIN!" if self.base_health > 0 else "GAME OVER"
            color = self.COLOR_BASE if self.base_health > 0 else (220, 50, 50)
            msg_surf = self.FONT_MSG.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0, 180), msg_rect.inflate(20, 20))
            self.screen.blit(msg_surf, msg_rect)

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
            "wave": self.current_wave,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headlessly

    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()

    # To run with display, comment out the os.environ line and run this:
    # os.environ["SDL_VIDEODRIVER"] = "x11" 
    # env = GameEnv(render_mode="rgb_array")
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # clock = pygame.time.Clock()
    # obs, info = env.reset()
    # done = False
    # while not done:
    #     action = env.action_space.sample() # Replace with your agent's action
    #     keys = pygame.key.get_pressed()
    #     mov = 0
    #     if keys[pygame.K_UP]: mov = 1
    #     if keys[pygame.K_DOWN]: mov = 2
    #     if keys[pygame.K_LEFT]: mov = 3
    #     if keys[pygame.K_RIGHT]: mov = 4
    #     space = 1 if keys[pygame.K_SPACE] else 0
    #     shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
    #     action = [mov, space, shift]

    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated

    #     # Display the frame
    #     frame = np.transpose(obs, (1, 0, 2))
    #     surf = pygame.surfarray.make_surface(frame)
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
        
    #     clock.tick(30)
    # env.close()