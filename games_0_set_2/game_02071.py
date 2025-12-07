
# Generated: 2025-08-28T03:35:42.458280
# Source Brief: brief_02071.md
# Brief Index: 2071

        
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
    def __init__(self, path, health, speed, value, wave):
        self.path = path
        self.path_index = 0
        self.x, self.y = self.path[0]
        self.health = health
        self.max_health = health
        self.speed = speed
        self.value = value
        self.wave = wave
        self.progress = 0.0

    def move(self):
        if self.path_index >= len(self.path) - 1:
            return True  # Reached the end

        start_node = self.path[self.path_index]
        end_node = self.path[self.path_index + 1]
        
        dist_to_travel = self.speed / 30.0 # Speed is per second, 30 FPS
        self.progress += dist_to_travel

        if self.progress >= 1.0:
            self.progress = 0.0
            self.path_index += 1
            if self.path_index >= len(self.path) - 1:
                self.x, self.y = end_node
                return True
            start_node = self.path[self.path_index]
            end_node = self.path[self.path_index + 1]

        self.x = start_node[0] + (end_node[0] - start_node[0]) * self.progress
        self.y = start_node[1] + (end_node[1] - start_node[1]) * self.progress
        return False

class Tower:
    def __init__(self, grid_pos, tower_type_info):
        self.grid_pos = grid_pos
        self.type_info = tower_type_info
        self.cooldown = 0

    def can_fire(self):
        return self.cooldown <= 0

    def fire(self):
        self.cooldown = self.type_info["fire_rate"]
        # sfx: tower_fire.wav

    def update(self):
        if self.cooldown > 0:
            self.cooldown -= 1

class Projectile:
    def __init__(self, start_pos, target_enemy, damage):
        self.x, self.y = start_pos
        self.target = target_enemy
        self.damage = damage
        self.speed = 15

    def move(self):
        if self.target is None:
            return True # Target is gone

        dx = self.target.x - self.x
        dy = self.target.y - self.y
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            return True  # Reached target
        
        self.x += (dx / dist) * self.speed
        self.y += (dy / dist) * self.speed
        return False

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-1.5, 1.5)
        self.lifespan = random.randint(10, 20)
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        return self.lifespan <= 0


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move selector, space to place a tower. Hold shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in a minimalist isometric world."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 12
        self.ISO_TILE_W, self.ISO_TILE_H = 40, 20
        self.GRID_OFFSET_X = self.WIDTH // 2
        self.GRID_OFFSET_Y = 100

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 14)
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 50, 62)
        self.COLOR_PATH = (65, 72, 89)
        self.COLOR_BASE = (34, 177, 76)
        self.COLOR_ENEMY = (237, 28, 36)
        self.COLOR_TOWER_1 = (63, 72, 204)
        self.COLOR_TOWER_2 = (0, 162, 232)
        self.COLOR_PROJECTILE = (255, 242, 0)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_GREEN = (0, 255, 0)
        self.COLOR_HEALTH_RED = (255, 0, 0)

        # Game rules and data
        self.MAX_STEPS = 30 * 60 * 2 # 2 minutes at 30fps
        self.MAX_WAVES = 10
        self.WAVE_COOLDOWN = 30 * 5 # 5 seconds
        self.ENEMY_PATH = [
            (-1, 5), (2, 5), (2, 2), (6, 2), (6, 8), 
            (10, 8), (10, 4), (13, 4), (13, 6), (self.GRID_W, 6)
        ]
        self.BASE_GRID_POS = (self.GRID_W, 6)
        self.TOWER_TYPES = [
            {"name": "Gun", "cost": 25, "range": 2.5, "damage": 10, "fire_rate": 30, "color": self.COLOR_TOWER_1},
            {"name": "Sniper", "cost": 75, "range": 5.0, "damage": 40, "fire_rate": 90, "color": self.COLOR_TOWER_2},
        ]

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.wave_timer = 0
        self.enemies_to_spawn = []
        self.spawn_timer = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        
        # Action cooldowns
        self.action_cooldowns = {'move': 0, 'place': 0, 'cycle': 0}

        # Initialize state
        self.reset()
        self.validate_implementation()
    
    def _to_iso(self, x, y):
        iso_x = self.GRID_OFFSET_X + (x - y) * (self.ISO_TILE_W / 2)
        iso_y = self.GRID_OFFSET_Y + (x + y) * (self.ISO_TILE_H / 2)
        return int(iso_x), int(iso_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = 100
        self.resources = 100
        self.current_wave = 0
        self.wave_timer = self.WAVE_COOLDOWN // 2
        self.enemies_to_spawn = []
        self.spawn_timer = 0
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()

        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_type = 0
        
        for key in self.action_cooldowns:
            self.action_cooldowns[key] = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty for time passing
        
        self._handle_actions(action)
        
        if not self.game_over:
            reward += self._update_game_state()

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.win and self.base_health <= 0:
            reward -= 100
        elif terminated and self.win:
            reward += 100

        self.score += reward
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action
        
        # Update cooldowns
        for key in self.action_cooldowns:
            if self.action_cooldowns[key] > 0:
                self.action_cooldowns[key] -= 1

        # Movement
        if movement != 0 and self.action_cooldowns['move'] == 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1
            elif movement == 2: dy = 1
            elif movement == 3: dx = -1
            elif movement == 4: dx = 1
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_W - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_H - 1)
            self.action_cooldowns['move'] = 5

        # Cycle tower type
        if shift_held and self.action_cooldowns['cycle'] == 0:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
            self.action_cooldowns['cycle'] = 10
            # sfx: menu_cycle.wav

        # Place tower
        if space_held and self.action_cooldowns['place'] == 0:
            tower_info = self.TOWER_TYPES[self.selected_tower_type]
            if self.resources >= tower_info["cost"]:
                is_on_path = any(self.cursor_pos[0] == p[0] and self.cursor_pos[1] == p[1] for p in self.ENEMY_PATH)
                is_occupied = any(t.grid_pos == self.cursor_pos for t in self.towers)
                if not is_on_path and not is_occupied:
                    self.towers.append(Tower(self.cursor_pos.copy(), tower_info))
                    self.resources -= tower_info["cost"]
                    self.action_cooldowns['place'] = 15
                    # sfx: place_tower.wav

    def _update_game_state(self):
        step_reward = 0

        # Wave management
        if not self.enemies and not self.enemies_to_spawn and self.current_wave <= self.MAX_WAVES:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._spawn_wave()
        
        # Spawn enemies from the current wave queue
        if self.enemies_to_spawn:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self.enemies.append(self.enemies_to_spawn.pop(0))
                self.spawn_timer = 15 # Stagger spawns

        # Update towers
        for tower in self.towers:
            tower.update()
            if tower.can_fire():
                target = None
                min_dist = tower.type_info["range"] ** 2
                for enemy in self.enemies:
                    dist_sq = (enemy.x - tower.grid_pos[0])**2 + (enemy.y - tower.grid_pos[1])**2
                    if dist_sq < min_dist:
                        min_dist = dist_sq
                        target = enemy
                if target:
                    tower.fire()
                    start_pos = self._to_iso(tower.grid_pos[0], tower.grid_pos[1])
                    self.projectiles.append(Projectile(start_pos, target, tower.type_info["damage"]))

        # Update projectiles
        for proj in self.projectiles[:]:
            enemy_pos = self._to_iso(proj.target.x, proj.target.y)
            dx, dy = enemy_pos[0] - proj.x, enemy_pos[1] - proj.y
            dist = math.hypot(dx, dy)
            if dist < 8: # Hit detection
                proj.target.health -= proj.damage
                step_reward += 0.1 # Reward for hitting
                # sfx: hit_enemy.wav
                for _ in range(5):
                    self.particles.append(Particle(proj.x, proj.y, self.COLOR_PROJECTILE))
                self.projectiles.remove(proj)
            else:
                proj.x += (dx / dist) * proj.speed
                proj.y += (dy / dist) * proj.speed

        # Update enemies
        for enemy in self.enemies[:]:
            if enemy.health <= 0:
                step_reward += 1.0 # Reward for kill
                self.resources += enemy.value
                for _ in range(10):
                    iso_pos = self._to_iso(enemy.x, enemy.y)
                    self.particles.append(Particle(iso_pos[0], iso_pos[1], self.COLOR_ENEMY))
                self.enemies.remove(enemy)
                # sfx: enemy_die.wav
                continue

            if enemy.move():
                self.base_health -= 10
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
                if self.base_health <= 0:
                    self.base_health = 0
                    self.game_over = True
        
        # Update particles
        for p in self.particles[:]:
            if p.update():
                self.particles.remove(p)

        # Check win condition
        if self.current_wave > self.MAX_WAVES and not self.enemies:
            self.win = True
            self.game_over = True

        return step_reward

    def _spawn_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            return

        num_enemies = 3 + self.current_wave * 2
        health = 10 * (1.1 ** (self.current_wave - 1))
        speed = 0.5 * (1.1 ** (self.current_wave - 1))
        value = 5 + self.current_wave
        
        self.enemies_to_spawn = [Enemy(self.ENEMY_PATH, health, speed, value, self.current_wave) for _ in range(num_enemies)]
        self.wave_timer = self.WAVE_COOLDOWN

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and path
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                p1 = self._to_iso(x, y)
                p2 = self._to_iso(x + 1, y)
                p3 = self._to_iso(x + 1, y + 1)
                p4 = self._to_iso(x, y + 1)
                pygame.draw.polygon(self.screen, self.COLOR_GRID, [p1, p2, p3, p4], 1)

        for i in range(len(self.ENEMY_PATH) - 1):
            p1_grid, p2_grid = self.ENEMY_PATH[i], self.ENEMY_PATH[i+1]
            for t in np.linspace(0, 1, 20):
                curr_grid = (p1_grid[0] * (1-t) + p2_grid[0] * t, p1_grid[1] * (1-t) + p2_grid[1] * t)
                x, y = int(curr_grid[0]), int(curr_grid[1])
                if 0 <= x < self.GRID_W and 0 <= y < self.GRID_H:
                    p1 = self._to_iso(x, y)
                    p2 = self._to_iso(x + 1, y)
                    p3 = self._to_iso(x + 1, y + 1)
                    p4 = self._to_iso(x, y + 1)
                    pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], self.COLOR_PATH)

        # Draw base
        base_pos = self._to_iso(self.BASE_GRID_POS[0]-1, self.BASE_GRID_POS[1])
        p1 = base_pos
        p2 = (base_pos[0] + self.ISO_TILE_W / 2, base_pos[1] + self.ISO_TILE_H / 2)
        p3 = (base_pos[0], base_pos[1] + self.ISO_TILE_H)
        p4 = (base_pos[0] - self.ISO_TILE_W / 2, base_pos[1] + self.ISO_TILE_H / 2)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], self.COLOR_BASE)
        pygame.draw.polygon(self.screen, tuple(c*0.8 for c in self.COLOR_BASE), [p1, p2, p3, p4], 2)

        # Draw towers
        for tower in self.towers:
            center_iso = self._to_iso(tower.grid_pos[0], tower.grid_pos[1])
            p1 = (center_iso[0], center_iso[1] - 8)
            p2 = (center_iso[0] - 8, center_iso[1] + 5)
            p3 = (center_iso[0] + 8, center_iso[1] + 5)
            color = tower.type_info["color"]
            if not tower.can_fire():
                color = tuple(c*0.5 for c in color)
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), color)
            pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), color)
        
        # Draw cursor
        cx, cy = self.cursor_pos
        p1 = self._to_iso(cx, cy)
        p2 = self._to_iso(cx + 1, cy)
        p3 = self._to_iso(cx + 1, cy + 1)
        p4 = self._to_iso(cx, cy + 1)
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, [p1, p2, p3, p4], 2)

        # Draw enemies
        for enemy in sorted(self.enemies, key=lambda e: e.y):
            iso_pos = self._to_iso(enemy.x, enemy.y)
            pygame.gfxdraw.filled_circle(self.screen, iso_pos[0], iso_pos[1], 6, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, iso_pos[0], iso_pos[1], 6, self.COLOR_ENEMY)
            # Health bar
            health_pct = enemy.health / enemy.max_health
            bar_w = 12
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (iso_pos[0] - bar_w/2, iso_pos[1] - 12, bar_w, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (iso_pos[0] - bar_w/2, iso_pos[1] - 12, bar_w * health_pct, 3))

        # Draw projectiles and particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), max(0, p.lifespan // 5), p.color)
        for proj in self.projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, (proj.x - 2, proj.y - 2, 4, 4))
    
    def _render_ui(self):
        # Info panel
        ui_texts = [
            f"HP: {self.base_health}/100",
            f"WAVE: {self.current_wave}/{self.MAX_WAVES}",
            f"$$: {self.resources}",
            f"SCORE: {int(self.score)}"
        ]
        for i, text in enumerate(ui_texts):
            surf = self.font_m.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(surf, (10, 10 + i * 20))
            
        # Tower selection panel
        tower_info = self.TOWER_TYPES[self.selected_tower_type]
        select_texts = [
            "TOWER:",
            f"{tower_info['name']}",
            f"Cost: {tower_info['cost']}",
            f"Dmg: {tower_info['damage']}",
            f"Rng: {tower_info['range']}"
        ]
        for i, text in enumerate(select_texts):
            surf = self.font_s.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(surf, (self.WIDTH - 100, 10 + i * 16))
        
        # Wave timer
        if self.wave_timer > 0 and not self.enemies and not self.enemies_to_spawn:
            text = f"NEXT WAVE IN {self.wave_timer / 30:.1f}s"
            surf = self.font_m.render(text, True, self.COLOR_UI_TEXT)
            text_rect = surf.get_rect(center=(self.WIDTH/2, 20))
            self.screen.blit(surf, text_rect)

        # Game over / Win screen
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_BASE if self.win else self.COLOR_ENEMY
            
            surf = self.font_l.render(message, True, color)
            text_rect = surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(surf, text_rect)
            
            score_text = f"Final Score: {int(self.score)}"
            score_surf = self.font_m.render(score_text, True, self.COLOR_UI_TEXT)
            score_rect = score_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "resources": self.resources,
            "enemies": len(self.enemies),
        }

    def close(self):
        pygame.quit()

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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)

        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0

        # Space
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        # Shift
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds before resetting
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()