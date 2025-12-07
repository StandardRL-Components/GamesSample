import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper function for isometric projection
def to_iso(x, y, tile_w_half, tile_h_half, offset_x, offset_y):
    iso_x = (x - y) * tile_w_half + offset_x
    iso_y = (x + y) * tile_h_half + offset_y
    return int(iso_x), int(iso_y)

# Helper function to draw an isometric rectangle (cube top)
def draw_iso_poly(surface, color, x, y, tile_w_half, tile_h_half, offset_x, offset_y):
    points = [
        to_iso(x, y, tile_w_half, tile_h_half, offset_x, offset_y),
        to_iso(x + 1, y, tile_w_half, tile_h_half, offset_x, offset_y),
        to_iso(x + 1, y + 1, tile_w_half, tile_h_half, offset_x, offset_y),
        to_iso(x, y + 1, tile_w_half, tile_h_half, offset_x, offset_y),
    ]
    pygame.gfxdraw.aapolygon(surface, points, color)
    pygame.gfxdraw.filled_polygon(surface, points, color)

class Enemy:
    def __init__(self, wave, path, speed_multiplier, health_multiplier):
        self.path = path
        self.path_index = 0
        self.x, self.y = self.path[0]
        self.pixel_x, self.pixel_y = 0, 0 # Set during update
        self.health = int(50 * health_multiplier)
        self.max_health = self.health
        self.speed = 0.05 * speed_multiplier
        self.slow_timer = 0
        self.value = 10 + wave
        self.radius = 8

    def update(self, dt):
        if self.slow_timer > 0:
            self.slow_timer -= dt
            speed = self.speed / 2
        else:
            speed = self.speed

        if self.path_index < len(self.path) - 1:
            target_x, target_y = self.path[self.path_index + 1]
            dx, dy = target_x - self.x, target_y - self.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0:
                self.x += (dx / dist) * speed * dt
                self.y += (dy / dist) * speed * dt

            if math.sqrt((self.x - target_x)**2 + (self.y - target_y)**2) < 0.1:
                self.path_index += 1
                self.x, self.y = target_x, target_y
        
        return self.path_index >= len(self.path) - 1

class Tower:
    TOWER_SPECS = {
        0: {"name": "Gatling", "cost": 100, "damage": 5, "range": 3, "rate": 0.2, "color": (100, 255, 100), "projectile_speed": 10, "type": "normal"},
        1: {"name": "Cannon", "cost": 250, "damage": 40, "range": 4, "rate": 1.5, "color": (100, 100, 255), "projectile_speed": 6, "type": "splash"},
        2: {"name": "Frost", "cost": 150, "damage": 1, "range": 2.5, "rate": 1.0, "color": (150, 220, 255), "projectile_speed": 8, "type": "slow"},
    }

    def __init__(self, grid_x, grid_y, tower_type):
        self.grid_x, self.grid_y = grid_x, grid_y
        self.grid_pos = (grid_x, grid_y)
        self.spec = Tower.TOWER_SPECS[tower_type]
        self.cooldown = 0
        self.target = None

    def update(self, dt, enemies):
        self.cooldown = max(0, self.cooldown - dt)
        
        # Find new target if needed
        if self.target is None or self.target.health <= 0 or self.dist_to_enemy(self.target) > self.spec["range"]:
            self.target = self.find_target(enemies)
        
        # Fire if ready and has target
        if self.cooldown == 0 and self.target is not None:
            self.cooldown = self.spec["rate"]
            return Projectile(self, self.target)
        return None

    def dist_to_enemy(self, enemy):
        return math.sqrt((self.grid_x - enemy.x)**2 + (self.grid_y - enemy.y)**2)

    def find_target(self, enemies):
        in_range = [e for e in enemies if self.dist_to_enemy(e) <= self.spec["range"]]
        if not in_range:
            return None
        # Target enemy furthest along the path
        return max(in_range, key=lambda e: e.path_index + math.sqrt((e.x - e.path[e.path_index][0])**2 + (e.y - e.path[e.path_index][1])**2))

class Projectile:
    def __init__(self, tower, target):
        self.x, self.y = tower.grid_x + 0.5, tower.grid_y + 0.5
        self.target = target
        self.spec = tower.spec
        self.speed = self.spec["projectile_speed"]

    def update(self, dt):
        if self.target.health <= 0:
            return True # Reached destination (target is gone)

        dx, dy = self.target.x - self.x, self.target.y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < 0.2:
            return True # Reached destination

        self.x += (dx / dist) * self.speed * dt
        self.y += (dy / dist) * self.speed * dt
        return False

class Particle:
    def __init__(self, x, y, color, life, size, vel_range=(-1, 1)):
        self.x, self.y = x, y
        self.vx = random.uniform(vel_range[0], vel_range[1])
        self.vy = random.uniform(vel_range[0], vel_range[1])
        self.life = life
        self.max_life = life
        self.color = color
        self.size = size

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        return self.life <= 0


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. Press Shift to cycle tower types. Press Space to build a tower."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers along their path. Earn resources for each kill."
    )

    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_W, GRID_H = 32, 20
    TILE_W, TILE_H = 20, 10
    ISO_OFFSET_X = WIDTH // 2
    ISO_OFFSET_Y = 60
    MAX_STEPS = 5000 # ~2.7 minutes at 30fps
    MAX_WAVES = 10
    
    # --- Colors ---
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 55)
    COLOR_PATH = (60, 65, 75)
    COLOR_BASE = (80, 20, 20)
    COLOR_CURSOR_VALID = (255, 255, 0, 150)
    COLOR_CURSOR_INVALID = (255, 0, 0, 150)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH_GREEN = (0, 200, 0)
    COLOR_HEALTH_RED = (200, 0, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.path = self._generate_path()
        self.path_coords = set(self.path)
        
        self.tower_slots = []
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if (x, y) not in self.path_coords:
                    self.tower_slots.append((x, y))

        self.reset()

        # Call validation at the end of __init__
        # self.validate_implementation() # Commented out for final submission
    
    def _generate_path(self):
        # A fixed, winding path
        path = []
        for i in range(5): path.append((i, 4))
        for i in range(5, 12): path.append((4, i))
        for i in range(5, 15): path.append((i, 11))
        for i in range(11, 6, -1): path.append((14, i))
        for i in range(15, 25): path.append((i, 7))
        for i in range(8, 16): path.append((24, i))
        for i in range(25, 32): path.append((i, 15))
        return path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.base_health = 1000
        self.max_base_health = 1000
        self.resources = 300

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.current_wave = 0
        self.wave_timer = 5.0 # Time until first wave
        self.enemies_in_wave = 5
        self.enemy_speed_multiplier = 1.0
        self.enemy_health_multiplier = 1.0

        self.cursor_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        dt = self.clock.tick(30) / 1000.0 * 3 # Speed up simulation time
        reward = -0.001 # Small time penalty

        # --- 1. Handle Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1  # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1  # Right
        self.cursor_pos = (
            max(0, min(self.GRID_W - 1, self.cursor_pos[0] + dx)),
            max(0, min(self.GRID_H - 1, self.cursor_pos[1] + dy))
        )

        # Cycle tower type on shift press
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(Tower.TOWER_SPECS)
            # sfx: UI_cycle.wav
        
        # Place tower on space press
        if space_held and not self.prev_space_held:
            spec = Tower.TOWER_SPECS[self.selected_tower_type]
            can_place = (self.cursor_pos not in self.path_coords and
                         self.cursor_pos not in [t.grid_pos for t in self.towers] and
                         self.resources >= spec["cost"])
            if can_place:
                self.resources -= spec["cost"]
                self.towers.append(Tower(self.cursor_pos[0], self.cursor_pos[1], self.selected_tower_type))
                # sfx: tower_place.wav

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- 2. Update Game State ---
        if not self.game_over:
            # Wave Management
            if not self.enemies and self.current_wave < self.MAX_WAVES:
                self.wave_timer -= dt
                if self.wave_timer <= 0:
                    self.current_wave += 1
                    for _ in range(self.enemies_in_wave):
                        self.enemies.append(Enemy(self.current_wave, self.path, self.enemy_speed_multiplier, self.enemy_health_multiplier))
                    self.enemies_in_wave += 2
                    self.enemy_health_multiplier *= 1.1
                    self.enemy_speed_multiplier *= 1.05
                    self.wave_timer = 10.0 # Time between waves
            
            # Update Enemies
            enemies_reached_end = []
            for enemy in self.enemies:
                if enemy.update(dt):
                    enemies_reached_end.append(enemy)
            
            for enemy in enemies_reached_end:
                self.base_health -= enemy.max_health
                reward -= 10
                self.enemies.remove(enemy)
                # sfx: base_damage.wav

            # Update Towers and create Projectiles
            for tower in self.towers:
                projectile = tower.update(dt, self.enemies)
                if projectile:
                    self.projectiles.append(projectile)
                    # sfx: gatling_fire.wav or cannon_fire.wav
            
            # Update Projectiles and handle hits
            projectiles_to_remove = []
            for p in self.projectiles:
                if p.update(dt):
                    projectiles_to_remove.append(p)
                    # Handle hit logic
                    if p.target.health > 0:
                        reward += self._handle_hit(p)
            self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]

            # Remove dead enemies
            dead_enemies = [e for e in self.enemies if e.health <= 0]
            for enemy in dead_enemies:
                reward += 1
                self.score += enemy.value
                self.resources += enemy.value
                # sfx: enemy_explode.wav
                for _ in range(15):
                    self.particles.append(Particle(enemy.pixel_x, enemy.pixel_y, (255, 100, 0), 0.5, random.randint(2, 5)))
            self.enemies = [e for e in self.enemies if e.health > 0]

            # Update Particles
            self.particles = [p for p in self.particles if not p.update(dt)]

        # --- 3. Check Termination ---
        self.steps += 1
        terminated = False
        if self.base_health <= 0:
            self.base_health = 0
            terminated = True
            self.game_over = True
            reward -= 100
        elif self.current_wave >= self.MAX_WAVES and not self.enemies:
            terminated = True
            self.game_over = True
            self.victory = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Truncated in spirit, but terminated by API
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_hit(self, projectile):
        reward = 0.1
        if projectile.spec["type"] == "splash":
            # sfx: cannon_explosion.wav
            hit_pos = (projectile.target.pixel_x, projectile.target.pixel_y)
            for _ in range(30):
                self.particles.append(Particle(hit_pos[0], hit_pos[1], (255, 200, 50), 0.7, random.randint(2, 6), (-3, 3)))
            for enemy in self.enemies:
                dist = math.sqrt((enemy.pixel_x - hit_pos[0])**2 + (enemy.pixel_y - hit_pos[1])**2)
                if dist < 50: # Splash radius
                    damage = projectile.spec["damage"] * max(0, 1 - dist/50)
                    enemy.health -= damage
        elif projectile.spec["type"] == "slow":
            # sfx: frost_hit.wav
            projectile.target.health -= projectile.spec["damage"]
            projectile.target.slow_timer = 2.0 # Slow duration
            hit_pos = (projectile.target.pixel_x, projectile.target.pixel_y)
            for _ in range(10):
                self.particles.append(Particle(hit_pos[0], hit_pos[1], (200, 220, 255), 0.4, random.randint(1, 3)))
        else: # normal
            projectile.target.health -= projectile.spec["damage"]
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "resources": self.resources}
    
    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                draw_iso_poly(self.screen, self.COLOR_GRID, x, y, self.TILE_W, self.TILE_H, self.ISO_OFFSET_X, self.ISO_OFFSET_Y)
        
        # Draw path
        for x, y in self.path:
            draw_iso_poly(self.screen, self.COLOR_PATH, x, y, self.TILE_W, self.TILE_H, self.ISO_OFFSET_X, self.ISO_OFFSET_Y)
        
        # Draw base
        bx, by = self.path[-1]
        draw_iso_poly(self.screen, self.COLOR_BASE, bx, by, self.TILE_W, self.TILE_H, self.ISO_OFFSET_X, self.ISO_OFFSET_Y)

        # Draw towers and ranges
        for tower in self.towers:
            tx, ty = to_iso(tower.grid_x + 0.5, tower.grid_y + 0.5, self.TILE_W, self.TILE_H, self.ISO_OFFSET_X, self.ISO_OFFSET_Y)
            # Draw range circle
            range_px = tower.spec["range"] * self.TILE_W * 0.707 # Approximate isometric radius
            pygame.gfxdraw.aacircle(self.screen, tx, ty - self.TILE_H // 2, int(range_px), (*tower.spec["color"], 50))
            # Draw tower
            pygame.gfxdraw.filled_circle(self.screen, tx, ty - self.TILE_H, 6, tower.spec["color"])
            pygame.gfxdraw.aacircle(self.screen, tx, ty - self.TILE_H, 6, (255, 255, 255, 150))

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = to_iso(enemy.x, enemy.y, self.TILE_W, self.TILE_H, self.ISO_OFFSET_X, self.ISO_OFFSET_Y)
            enemy.pixel_x, enemy.pixel_y = ex, ey
            pygame.gfxdraw.filled_circle(self.screen, ex, ey - self.TILE_H // 2, enemy.radius, (200, 50, 50))
            pygame.gfxdraw.aacircle(self.screen, ex, ey - self.TILE_H // 2, enemy.radius, (255, 100, 100))
            # Health bar
            if enemy.health < enemy.max_health:
                health_pct = enemy.health / enemy.max_health
                bar_w = 20
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (ex - bar_w/2, ey - 25, bar_w, 4))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (ex - bar_w/2, ey - 25, bar_w * health_pct, 4))

        # Draw projectiles
        for p in self.projectiles:
            px, py = to_iso(p.x, p.y, self.TILE_W, self.TILE_H, self.ISO_OFFSET_X, self.ISO_OFFSET_Y)
            pygame.draw.line(self.screen, p.spec["color"], (px, py - 10), (px, py - 12), 3)

        # Draw cursor
        cx, cy = self.cursor_pos
        spec = Tower.TOWER_SPECS[self.selected_tower_type]
        is_valid = (cx, cy) not in self.path_coords and (cx, cy) not in [t.grid_pos for t in self.towers]
        color = self.COLOR_CURSOR_VALID if is_valid and self.resources >= spec["cost"] else self.COLOR_CURSOR_INVALID
        draw_iso_poly(self.screen, color, cx, cy, self.TILE_W, self.TILE_H, self.ISO_OFFSET_X, self.ISO_OFFSET_Y)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p.life / p.max_life))
            temp_surf = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p.color, alpha), (p.size, p.size), p.size)
            self.screen.blit(temp_surf, (p.x - p.size, p.y - p.size))
            
    def _render_ui(self):
        # Base Health
        pygame.draw.rect(self.screen, (50,50,50), (10, 10, 200, 20))
        health_pct = self.base_health / self.max_base_health if self.max_base_health > 0 else 0
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (10, 10, 200 * health_pct, 20))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, 10, 200, 20), 1)
        health_text = self.font_small.render(f"BASE HP: {int(self.base_health)}/{self.max_base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Resources
        res_text = self.font_medium.render(f"${self.resources}", True, (255, 223, 0))
        self.screen.blit(res_text, (10, 35))

        # Score and Wave
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        if self.current_wave > 0 and self.current_wave <= self.MAX_WAVES:
            wave_text = self.font_medium.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        else:
            wave_text = self.font_medium.render(f"WAVE: --/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 35))

        # Selected Tower Info
        spec = Tower.TOWER_SPECS[self.selected_tower_type]
        tower_info_text = self.font_medium.render(f"Selected: {spec['name']} (Cost: ${spec['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_info_text, (self.WIDTH // 2 - tower_info_text.get_width() // 2, self.HEIGHT - 30))

        # Game Over / Victory
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.victory:
                end_text = self.font_large.render("VICTORY", True, (0, 255, 0))
            else:
                end_text = self.font_large.render("GAME OVER", True, (255, 0, 0))
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    # Re-enable the display for direct play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Match the environment's internal clock for human play
        
    pygame.quit()