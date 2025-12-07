
# Generated: 2025-08-27T19:10:34.331284
# Source Brief: brief_02068.md
# Brief Index: 2068

        
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


# --- Helper Classes for Game Objects ---

class Tower:
    def __init__(self, grid_pos, tower_type):
        self.grid_pos = grid_pos
        self.type = tower_type  # 'basic' or 'advanced'
        if self.type == 'basic':
            self.range = 80
            self.damage = 10
            self.cooldown_max = 30  # frames
            self.color = (0, 200, 255)
        else: # 'advanced'
            self.range = 120
            self.damage = 25
            self.cooldown_max = 50
            self.color = (255, 100, 255)
        self.cooldown = 0
        self.target = None

    def update(self, enemies, projectiles):
        if self.cooldown > 0:
            self.cooldown -= 1
            return 0

        # Find a new target if needed
        if self.target is None or self.target.health <= 0 or self.target.distance_to_base < 0:
            self.find_target(enemies)

        if self.target:
            dist_to_target = math.hypot(self.target.pos[0] - self.pos[0], self.target.pos[1] - self.pos[1])
            if dist_to_target <= self.range:
                # Fire projectile
                # SFX: Laser_Shoot.wav
                projectiles.append(Projectile(self.pos, self.target, self.damage, self.color))
                self.cooldown = self.cooldown_max
            else:
                self.target = None # Target out of range
        return 0

    def find_target(self, enemies):
        closest_enemy = None
        min_dist = float('inf')
        for enemy in enemies:
            dist = math.hypot(enemy.pos[0] - self.pos[0], enemy.pos[1] - self.pos[1])
            if dist <= self.range and enemy.distance_to_base < min_dist:
                min_dist = enemy.distance_to_base
                closest_enemy = enemy
        self.target = closest_enemy

    def set_screen_pos(self, pos):
        self.pos = pos

class Enemy:
    def __init__(self, path, wave_num):
        self.path = path
        self.path_index = 0
        self.pos = list(self.path[0])
        self.speed = 0.8 + (wave_num * 0.02)
        self.max_health = 20 + (wave_num * 5)
        self.health = self.max_health
        self.distance_to_base = len(self.path) - 1

    def update(self):
        if self.path_index >= len(self.path) - 1:
            return True  # Reached base

        target_pos = self.path[self.path_index + 1]
        dx = target_pos[0] - self.pos[0]
        dy = target_pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.path_index += 1
            self.distance_to_base -= 1
            self.pos = list(self.path[self.path_index])
        else:
            self.pos[0] += (dx / dist) * self.speed
            self.pos[1] += (dy / dist) * self.speed
        return False

class Projectile:
    def __init__(self, start_pos, target, damage, color):
        self.pos = list(start_pos)
        self.target = target
        self.damage = damage
        self.speed = 10
        self.color = color

    def update(self):
        if self.target.health <= 0:
            return True, False, 0 # Expired

        dx = self.target.pos[0] - self.pos[0]
        dy = self.target.pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.target.health -= self.damage
            # SFX: Enemy_Hit.wav
            return True, True, self.damage # Hit target
        else:
            self.pos[0] += (dx / dist) * self.speed
            self.pos[1] += (dy / dist) * self.speed
            return False, False, 0 # In transit

class Particle:
    def __init__(self, pos, color, life, size_range, speed_range):
        self.pos = list(pos)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(*speed_range)
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.life = life
        self.max_life = life
        self.color = color
        self.size = random.uniform(*size_range)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1
        return self.life <= 0

# --- Main Game Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to build a Basic Tower "
        "and Shift to build an Advanced Tower on the selected tile."
    )
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers on a tiled isometric grid. "
        "Survive 10 waves to win."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 20, 15
        self.TILE_W, self.TILE_H = 32, 16
        self.OFFSET_X, self.OFFSET_Y = self.WIDTH // 2, 80
        self.MAX_STEPS = 5000 # Increased for longer games
        self.MAX_WAVES = 10
        self.WAVE_COOLDOWN = 150 # 5 seconds at 30fps

        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PATH = (50, 60, 85)
        self.COLOR_BASE = (150, 50, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_CURSOR = (0, 255, 150)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_UI_BG = (30, 40, 60, 180)

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans", 16)
        self.font_large = pygame.font.SysFont("sans", 24)
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = 0
        self.money = 0
        self.wave_num = 0
        self.wave_timer = 0
        self.enemies_in_wave = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.selected_tile = [0, 0]
        self.path_coords = []
        self.path_screen = []
        self.grid_occupied = set()
        self.last_action = np.zeros(self.action_space.shape)
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = 100
        self.money = 100
        self.wave_num = 0
        self.wave_timer = self.WAVE_COOLDOWN
        self.enemies_in_wave = 0

        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.grid_occupied.clear()

        self.selected_tile = [self.GRID_W // 2, self.GRID_H // 2]
        self.last_action = np.zeros(self.action_space.shape)

        self._create_path()
        self.base_pos_grid = self.path_coords[-1]
        self.base_pos_screen = self._iso_to_screen(*self.base_pos_grid)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Time penalty

        space_pressed = action[1] == 1 and self.last_action[1] == 0
        shift_pressed = action[2] == 1 and self.last_action[2] == 0
        self.last_action = action

        self._handle_input(action[0], space_pressed, shift_pressed)
        
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        reward += self._update_waves()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                reward += 100
                # SFX: Game_Win.wav
            else:
                reward -= 100
                # SFX: Game_Over.wav
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Update Logic ---

    def _handle_input(self, movement, place_basic, place_advanced):
        if movement == 1: self.selected_tile[1] -= 1 # Up
        elif movement == 2: self.selected_tile[1] += 1 # Down
        elif movement == 3: self.selected_tile[0] -= 1 # Left
        elif movement == 4: self.selected_tile[0] += 1 # Right
        
        self.selected_tile[0] %= self.GRID_W
        self.selected_tile[1] %= self.GRID_H
        
        tile_key = tuple(self.selected_tile)
        is_path = tile_key in self.path_coords
        is_occupied = tile_key in self.grid_occupied

        if not is_path and not is_occupied:
            if place_basic and self.money >= 25:
                self.money -= 25
                tower = Tower(tile_key, 'basic')
                self.towers.append(tower)
                self.grid_occupied.add(tile_key)
                self._create_particles(self._iso_to_screen(*tile_key), tower.color, 20, (1, 3), (0.5, 2))
                # SFX: Build_Tower.wav
            elif place_advanced and self.money >= 75:
                self.money -= 75
                tower = Tower(tile_key, 'advanced')
                self.towers.append(tower)
                self.grid_occupied.add(tile_key)
                self._create_particles(self._iso_to_screen(*tile_key), tower.color, 30, (2, 4), (1, 3))
                # SFX: Build_Tower_Advanced.wav
    
    def _update_towers(self):
        for tower in self.towers:
            screen_pos = self._iso_to_screen(*tower.grid_pos)
            tower.set_screen_pos(screen_pos)
            tower.update(self.enemies, self.projectiles)
        return 0

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        for proj in self.projectiles:
            expired, hit, damage = proj.update()
            if hit:
                reward += 0.1 # Reward for hitting
                self._create_particles(proj.pos, proj.color, 10, (1, 2), (0.5, 1.5))
            if expired:
                projectiles_to_remove.append(proj)
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        return reward

    def _update_enemies(self):
        reward = 0
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy.update(): # Reached base
                self.base_health -= 10
                enemies_to_remove.append(enemy)
                self._create_particles(self.base_pos_screen, self.COLOR_ENEMY, 50, (3, 6), (1, 4))
                # SFX: Base_Damage.wav
            elif enemy.health <= 0:
                reward += 1 # Reward for kill
                self.money += 5
                enemies_to_remove.append(enemy)
                self._create_particles(enemy.pos, self.COLOR_ENEMY, 30, (2, 4), (1, 3))
                # SFX: Enemy_Explode.wav

        if enemies_to_remove:
            self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if not p.update()]

    def _update_waves(self):
        reward = 0
        if not self.enemies and self.enemies_in_wave > 0: # Wave just cleared
            self.enemies_in_wave = 0
            if self.wave_num < self.MAX_WAVES:
                self.wave_timer = self.WAVE_COOLDOWN
                reward += 5 # Wave clear reward
                # SFX: Wave_Clear.wav

        if self.wave_timer > 0:
            self.wave_timer -= 1
            if self.wave_timer == 0 and self.wave_num < self.MAX_WAVES:
                self.wave_num += 1
                self._spawn_wave()
        return reward

    def _spawn_wave(self):
        num_enemies = 3 + self.wave_num * 2
        self.enemies_in_wave = num_enemies
        for _ in range(num_enemies):
            # Spawn enemies with a slight delay using their path index
            enemy = Enemy(self.path_screen, self.wave_num)
            enemy.path_index = -random.randint(10, 100) # Negative index to delay start
            self.enemies.append(enemy)

    def _check_termination(self):
        if self.game_over:
            return True
        
        if self.base_health <= 0:
            self.game_over = True
            self.win = False
            return True
            
        if self.wave_num >= self.MAX_WAVES and not self.enemies and self.enemies_in_wave == 0:
            self.game_over = True
            self.win = True
            return True
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            return True
            
        return False

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_path()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()

    def _render_grid(self):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], self.COLOR_GRID)

    def _render_path(self):
        for x, y in self.path_coords:
            p1 = self._iso_to_screen(x, y)
            p2 = self._iso_to_screen(x + 1, y)
            p3 = self._iso_to_screen(x + 1, y + 1)
            p4 = self._iso_to_screen(x, y + 1)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], self.COLOR_PATH)
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], self.COLOR_GRID)
    
    def _render_base(self):
        pos = self.base_pos_screen
        points = [
            (pos[0], pos[1] - 20), (pos[0] + 15, pos[1]),
            (pos[0], pos[1] + 20), (pos[0] - 15, pos[1])
        ]
        glow_color = (*self.COLOR_BASE, 50)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BASE)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_BASE)
        pygame.draw.circle(self.screen, glow_color, (int(pos[0]), int(pos[1])), 30)
        
        # Base health bar
        if self.base_health < 100:
            bar_w = 40
            bar_h = 5
            bar_x = pos[0] - bar_w / 2
            bar_y = pos[1] - 35
            health_pct = self.base_health / 100
            pygame.draw.rect(self.screen, (50, 0, 0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, bar_w * health_pct, bar_h))

    def _render_towers(self):
        for tower in self.towers:
            pos = tower.pos
            glow_color = (*tower.color, 30)
            pygame.draw.circle(self.screen, glow_color, (int(pos[0]), int(pos[1])), tower.range, 1)
            if tower.type == 'basic':
                pygame.draw.rect(self.screen, tower.color, (pos[0]-5, pos[1]-5, 10, 10))
            else:
                points = [(pos[0], pos[1]-8), (pos[0]+8, pos[1]+8), (pos[0]-8, pos[1]+8)]
                pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), tower.color)

    def _render_enemies(self):
        for enemy in self.enemies:
            if enemy.path_index < 0: continue
            pos = enemy.pos
            points = [(pos[0], pos[1]-8), (pos[0]+8, pos[1]), (pos[0], pos[1]+8), (pos[0]-8, pos[1])]
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_ENEMY)
            
            # Health bar
            bar_w = 20
            bar_h = 3
            bar_x = pos[0] - bar_w / 2
            bar_y = pos[1] - 15
            health_pct = enemy.health / enemy.max_health
            pygame.draw.rect(self.screen, (50, 0, 0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, bar_w * health_pct, bar_h))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj.pos[0]), int(proj.pos[1]), 3, proj.color)
            pygame.gfxdraw.aacircle(self.screen, int(proj.pos[0]), int(proj.pos[1]), 3, proj.color)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = (*p.color, alpha)
            temp_surf = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p.size, p.size), p.size)
            self.screen.blit(temp_surf, (p.pos[0] - p.size, p.pos[1] - p.size))
    
    def _render_cursor(self):
        x, y = self.selected_tile
        p1 = self._iso_to_screen(x, y)
        p2 = self._iso_to_screen(x + 1, y)
        p3 = self._iso_to_screen(x + 1, y + 1)
        p4 = self._iso_to_screen(x, y + 1)
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, [p1, p2, p3, p4], 2)
    
    def _render_ui(self):
        # Top-left info
        health_text = self.font_large.render(f"Base: {max(0, self.base_health)}%", True, self.COLOR_TEXT)
        money_text = self.font_large.render(f"$: {self.money}", True, self.COLOR_TEXT)
        wave_text = self.font_large.render(f"Wave: {self.wave_num}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        
        self.screen.blit(health_text, (10, 10))
        self.screen.blit(money_text, (10, 35))
        self.screen.blit(wave_text, (180, 10))
        self.screen.blit(score_text, (180, 35))

        # Bottom UI for tower costs
        s = pygame.Surface((self.WIDTH, 60), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (0, self.HEIGHT - 60))

        basic_cost_text = self.font_small.render("Basic Tower (Space) - Cost: 25", True, self.COLOR_TEXT)
        adv_cost_text = self.font_small.render("Advanced Tower (Shift) - Cost: 75", True, self.COLOR_TEXT)
        self.screen.blit(basic_cost_text, (20, self.HEIGHT - 45))
        self.screen.blit(adv_cost_text, (20, self.HEIGHT - 25))
        
        if self.wave_timer > 0:
            timer_val = math.ceil(self.wave_timer / 30)
            next_wave_text = self.font_large.render(f"Next wave in: {timer_val}", True, self.COLOR_TEXT)
            text_rect = next_wave_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 50))
            self.screen.blit(next_wave_text, text_rect)

        if self.game_over:
            status_text = "VICTORY!" if self.win else "GAME OVER"
            status_render = self.font_large.render(status_text, True, (255, 255, 100))
            text_rect = status_render.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(status_render, text_rect)

    # --- Helpers ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "money": self.money,
            "wave": self.wave_num,
        }

    def _iso_to_screen(self, x, y):
        screen_x = self.OFFSET_X + (x - y) * (self.TILE_W / 2)
        screen_y = self.OFFSET_Y + (x + y) * (self.TILE_H / 2)
        return int(screen_x), int(screen_y)

    def _create_path(self):
        self.path_coords.clear()
        # A winding path for enemies
        path_segments = [
            [(x, 7) for x in range(4)],
            [(3, y) for y in range(7, 3, -1)],
            [(x, 3) for x in range(3, 12)],
            [(11, y) for y in range(3, 11)],
            [(x, 10) for x in range(11, 17)],
            [(16, y) for y in range(10, 5, -1)],
            [(x, 5) for x in range(16, 19)],
        ]
        for segment in path_segments:
            self.path_coords.extend(segment)
        self.path_coords = list(dict.fromkeys(self.path_coords)) # Remove duplicates
        
        self.path_screen.clear()
        for gx, gy in self.path_coords:
            self.path_screen.append(self._iso_to_screen(gx, gy))

    def _create_particles(self, pos, color, count, size_range, speed_range):
        for _ in range(count):
            self.particles.append(Particle(pos, color, random.randint(15, 30), size_range, speed_range))

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        assert info['base_health'] == 100
        assert info['wave'] == 0
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
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
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()