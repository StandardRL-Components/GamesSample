
# Generated: 2025-08-28T04:16:38.878531
# Source Brief: brief_05196.md
# Brief Index: 5196

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the placement cursor. "
        "Press space to build a tower on a valid yellow tile. Survive 5 waves."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in this isometric tower defense game. "
        "Each tower costs resources, but you gain resources by defeating enemies."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    BASE_MAX_HEALTH = 100
    STARTING_RESOURCES = 25
    TOWER_COST = 10
    MAX_TOWERS = 10
    TOTAL_WAVES = 5
    ENEMIES_PER_WAVE = 15
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_PATH = (50, 60, 70)
    COLOR_PLACEMENT_ZONE = (70, 80, 50)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_INVALID = (255, 0, 0)
    COLOR_BASE = (0, 150, 200)
    COLOR_TOWER = (0, 200, 150)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_PROJECTILE = (100, 200, 255)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_HEALTH_BAR = (50, 220, 50)
    COLOR_HEALTH_BAR_BG = (120, 40, 40)
    COLOR_SCREEN_FLASH = (255, 0, 0, 100)

    # Grid & Isometric Projection
    GRID_SIZE_X, GRID_SIZE_Y = 16, 11
    TILE_W, TILE_H = 32, 16
    TILE_W_HALF, TILE_H_HALF = TILE_W // 2, TILE_H // 2
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 80


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
        try:
            self.font_s = pygame.font.Font(pygame.font.get_default_font(), 16)
            self.font_m = pygame.font.Font(pygame.font.get_default_font(), 20)
            self.font_l = pygame.font.Font(pygame.font.get_default_font(), 28)
        except:
            self.font_s = pygame.font.SysFont("monospace", 16)
            self.font_m = pygame.font.SysFont("monospace", 20)
            self.font_l = pygame.font.SysFont("monospace", 28)

        self._define_grid()
        self.reset()
        self.validate_implementation()

    def _define_grid(self):
        self.path_coords = []
        for x in range(1, 10): self.path_coords.append((x, 1))
        for y in range(1, 6): self.path_coords.append((9, y))
        for x in range(9, 3, -1): self.path_coords.append((x, 5))
        for y in range(5, 9): self.path_coords.append((4, y))
        for x in range(4, 14): self.path_coords.append((x, 8))
        self.path_coords.append((13, 9)) # Base

        self.placement_zones = []
        for x in range(self.GRID_SIZE_X):
            for y in range(self.GRID_SIZE_Y):
                if (x, y) not in self.path_coords:
                    self.placement_zones.append((x, y))

    def _iso_to_cart(self, ix, iy):
        return (ix - iy) * self.TILE_W_HALF + self.ORIGIN_X, (ix + iy) * self.TILE_H_HALF + self.ORIGIN_Y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = self.BASE_MAX_HEALTH
        self.resources = self.STARTING_RESOURCES
        self.current_wave = 0
        self.enemies_to_spawn = 0
        self.spawn_cooldown = 0
        self.wave_cleared_delay = 0

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.placement_cursor = [7, 3]
        self.base_hit_flash = 0

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.TOTAL_WAVES:
            self.win = True
            return
        self.enemies_to_spawn = self.ENEMIES_PER_WAVE
        self.spawn_cooldown = 60 # Initial delay before wave starts
        self.wave_cleared_delay = 0

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1
        self.base_hit_flash = max(0, self.base_hit_flash - 1)

        # --- Player Actions ---
        if movement != 0:
            px, py = self.placement_cursor
            if movement == 1: py -= 1 # Up
            elif movement == 2: py += 1 # Down
            elif movement == 3: px -= 1 # Left
            elif movement == 4: px += 1 # Right
            self.placement_cursor = [
                np.clip(px, 0, self.GRID_SIZE_X - 1),
                np.clip(py, 0, self.GRID_SIZE_Y - 1),
            ]

        if space_held:
            can_place = tuple(self.placement_cursor) in self.placement_zones
            spot_free = all(tower['grid_pos'] != self.placement_cursor for tower in self.towers)
            can_afford = self.resources >= self.TOWER_COST
            has_capacity = len(self.towers) < self.MAX_TOWERS

            if can_place and spot_free and can_afford and has_capacity:
                self.resources -= self.TOWER_COST
                self.towers.append({
                    'grid_pos': self.placement_cursor.copy(),
                    'attack_cooldown': random.randint(0, 30), # Stagger initial attacks
                    'muzzle_flash': 0,
                })
                # sfx: build_tower.wav
            else:
                # sfx: error.wav
                pass

        # --- Game Logic Update ---
        reward += self._update_towers()
        reward += self._update_projectiles()
        self._update_enemies()
        self._update_spawner()
        self._update_particles()

        # Wave management
        if self.enemies_to_spawn == 0 and not self.enemies:
            if not self.win:
                self.wave_cleared_delay += 1
                if self.wave_cleared_delay > 90: # 3 second pause between waves
                    self._start_next_wave()

        # --- Termination Check ---
        terminated = (self.base_health <= 0) or self.win or (self.steps >= self.MAX_STEPS)
        if terminated:
            if self.win:
                reward += 100
                self.score += 1000
            if self.base_health <= 0:
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_towers(self):
        for tower in self.towers:
            tower['muzzle_flash'] = max(0, tower['muzzle_flash'] - 1)
            tower['attack_cooldown'] -= 1
            if tower['attack_cooldown'] <= 0:
                target = self._find_closest_enemy(tower['grid_pos'], range_limit=4)
                if target:
                    tower['attack_cooldown'] = 45 # Attack speed
                    tower['muzzle_flash'] = 3
                    self.projectiles.append({
                        'pos': self._iso_to_cart(tower['grid_pos'][0], tower['grid_pos'][1]),
                        'target': target,
                        'speed': 5,
                    })
                    # sfx: tower_shoot.wav
        return 0

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            target_pos = self._iso_to_cart(p['target']['path_pos_idx'], p['target']['path_pos_idx']) # Simplified target pos
            target_pos = (target_pos[0], target_pos[1] - self.TILE_H_HALF)
            
            angle = math.atan2(target_pos[1] - p['pos'][1], target_pos[0] - p['pos'][0])
            p['pos'] = (p['pos'][0] + math.cos(angle) * p['speed'], p['pos'][1] + math.sin(angle) * p['speed'])

            if math.hypot(p['pos'][0] - target_pos[0], p['pos'][1] - target_pos[1]) < 8:
                self.projectiles.remove(p)
                p['target']['health'] -= 1
                reward += 0.1
                # sfx: enemy_hit.wav
                self._create_hit_particles(p['pos'])

                if p['target']['health'] <= 0 and p['target'] in self.enemies:
                    self.enemies.remove(p['target'])
                    reward += 1.0
                    self.resources += 2
                    self.score += 10 * self.current_wave
                    # sfx: enemy_destroy.wav
        return reward

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            enemy['move_cooldown'] -= 1
            if enemy['move_cooldown'] <= 0:
                enemy['move_cooldown'] = 3 # Enemy speed
                enemy['path_pos_idx'] += 1
                if enemy['path_pos_idx'] >= len(self.path_coords):
                    self.enemies.remove(enemy)
                    self.base_health -= 5 # Damage to base
                    self.base_health = max(0, self.base_health)
                    self.base_hit_flash = 15
                    # sfx: base_damage.wav
                else:
                    enemy['pos'] = self._iso_to_cart(*self.path_coords[enemy['path_pos_idx']])

    def _update_spawner(self):
        if self.enemies_to_spawn > 0:
            self.spawn_cooldown -= 1
            if self.spawn_cooldown <= 0:
                self.spawn_cooldown = 20 # Time between spawns
                self.enemies_to_spawn -= 1
                enemy_health = 4 + self.current_wave
                self.enemies.append({
                    'path_pos_idx': 0,
                    'pos': self._iso_to_cart(*self.path_coords[0]),
                    'health': enemy_health,
                    'max_health': enemy_health,
                    'move_cooldown': random.randint(0, 3) # Stagger movement
                })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _find_closest_enemy(self, tower_pos, range_limit):
        closest_enemy = None
        min_dist = float('inf')
        for enemy in self.enemies:
            enemy_pos = self.path_coords[enemy['path_pos_idx']]
            dist = math.hypot(tower_pos[0] - enemy_pos[0], tower_pos[1] - enemy_pos[1])
            if dist < range_limit and dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        return closest_enemy

    def _create_hit_particles(self, pos):
        for _ in range(5):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(5, 10),
                'color': self.COLOR_PROJECTILE,
                'radius': random.randint(1, 2)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_base()
        self._render_enemies()
        self._render_towers()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_effects()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(self.GRID_SIZE_X):
            for y in range(self.GRID_SIZE_Y):
                pos = self._iso_to_cart(x, y)
                points = [
                    (pos[0], pos[1] - self.TILE_H_HALF),
                    (pos[0] + self.TILE_W_HALF, pos[1]),
                    (pos[0], pos[1] + self.TILE_H_HALF),
                    (pos[0] - self.TILE_W_HALF, pos[1]),
                ]
                color = self.COLOR_BG
                if (x, y) in self.path_coords:
                    color = self.COLOR_PATH
                elif (x, y) in self.placement_zones:
                    color = self.COLOR_PLACEMENT_ZONE
                
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.aapolygon(self.screen, points, tuple(c*1.2 for c in color[:3]))


    def _render_base(self):
        base_pos = self._iso_to_cart(*self.path_coords[-1])
        pygame.gfxdraw.filled_circle(self.screen, int(base_pos[0]), int(base_pos[1] - self.TILE_H_HALF), 12, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, int(base_pos[0]), int(base_pos[1] - self.TILE_H_HALF), 12, tuple(c*1.2 for c in self.COLOR_BASE[:3]))

    def _render_cursor(self):
        cursor_pos = self._iso_to_cart(*self.placement_cursor)
        is_valid = tuple(self.placement_cursor) in self.placement_zones and \
                   all(tower['grid_pos'] != self.placement_cursor for tower in self.towers) and \
                   self.resources >= self.TOWER_COST and \
                   len(self.towers) < self.MAX_TOWERS

        color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID
        points = [
            (cursor_pos[0], cursor_pos[1] - self.TILE_H_HALF),
            (cursor_pos[0] + self.TILE_W_HALF, cursor_pos[1]),
            (cursor_pos[0], cursor_pos[1] + self.TILE_H_HALF),
            (cursor_pos[0] - self.TILE_W_HALF, cursor_pos[1]),
        ]
        pygame.draw.lines(self.screen, color, True, points, 2)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1] - self.TILE_H_HALF))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, tuple(c*0.8 for c in self.COLOR_ENEMY[:3]))
            
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_w = 12
            bar_h = 3
            bar_x = pos[0] - bar_w // 2
            bar_y = pos[1] - 12
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_w * health_ratio), bar_h))

    def _render_towers(self):
        for tower in self.towers:
            pos = self._iso_to_cart(*tower['grid_pos'])
            pos = (int(pos[0]), int(pos[1] - self.TILE_H_HALF))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_TOWER)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, tuple(c*1.2 for c in self.COLOR_TOWER[:3]))
            if tower['muzzle_flash'] > 0:
                flash_color = (255, 255, 150)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1] - 5, 3, flash_color)

    def _render_projectiles(self):
        for p in self.projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 10))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _render_effects(self):
        if self.base_hit_flash > 0:
            alpha = int(100 * (self.base_hit_flash / 15))
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 0, 0, alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # Base Health Bar
        health_ratio = self.base_health / self.BASE_MAX_HEALTH
        bar_w, bar_h = 200, 20
        bar_x, bar_y = self.SCREEN_WIDTH // 2 - bar_w // 2, 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_w * health_ratio), bar_h), border_radius=4)
        health_text = self.font_s.render(f"BASE: {self.base_health}/{self.BASE_MAX_HEALTH}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (bar_x + bar_w // 2 - health_text.get_width() // 2, bar_y + 2))

        # Score
        score_text = self.font_m.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Resources
        res_text = self.font_m.render(f"RESOURCES: {self.resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(res_text, (15, 35))
        
        # Towers
        tower_text = self.font_m.render(f"TOWERS: {len(self.towers)}/{self.MAX_TOWERS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(tower_text, (15, 60))

        # Wave
        wave_text = self.font_m.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 15, 10))

        if self.win:
            win_text = self.font_l.render("VICTORY!", True, self.COLOR_HEALTH_BAR)
            self.screen.blit(win_text, (self.SCREEN_WIDTH//2 - win_text.get_width()//2, self.SCREEN_HEIGHT//2 - win_text.get_height()//2))
        elif self.base_health <= 0:
            lose_text = self.font_l.render("GAME OVER", True, self.COLOR_ENEMY)
            self.screen.blit(lose_text, (self.SCREEN_WIDTH//2 - lose_text.get_width()//2, self.SCREEN_HEIGHT//2 - lose_text.get_height()//2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "resources": self.resources,
            "towers": len(self.towers),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert self.base_health == self.BASE_MAX_HEALTH
        assert self.resources == self.STARTING_RESOURCES
        assert self.current_wave == 1

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        # Test tower placement logic
        self.reset()
        initial_resources = self.resources
        initial_towers = len(self.towers)
        self.resources = self.TOWER_COST - 1
        self.step([0, 1, 0]) # Try to build with insufficient resources
        assert len(self.towers) == initial_towers, "Should not build tower without resources"
        self.resources = self.TOWER_COST
        self.step([0, 1, 0]) # Build tower
        assert len(self.towers) == initial_towers + 1, "Should build tower with sufficient resources"
        assert self.resources == 0, "Should deduct resources after building"

        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a window.
    # The environment is designed for headless operation, but we can display the frames.
    
    try:
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Tower Defense")
        
        obs, info = env.reset()
        terminated = False
        clock = pygame.time.Clock()

        print(env.user_guide)

        while not terminated:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)

            # Display the observation
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if info['score'] > 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Wave: {info['wave']}, Health: {info['base_health']}")
            
            clock.tick(30) # Limit to 30 FPS for playability

    finally:
        env.close()
        pygame.quit()