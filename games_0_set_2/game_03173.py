
# Generated: 2025-08-28T07:13:11.574879
# Source Brief: brief_03173.md
# Brief Index: 3173

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame


# Helper class for 2D vector math
class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    @property
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)

    @property
    def normalized(self):
        mag = self.magnitude
        if mag == 0:
            return Vec2(0, 0)
        return Vec2(self.x / mag, self.y / mag)
        
    @property
    def as_tuple(self):
        return (int(self.x), int(self.y))

# --- Game Entity Classes ---

class Enemy:
    def __init__(self, path, speed, health):
        self.path = path
        self.path_index = 0
        self.pos = Vec2(path[0][0], path[0][1])
        self.speed = speed
        self.max_health = health
        self.health = health
        self.base_radius = 8

    def move(self):
        if self.path_index >= len(self.path) - 1:
            return True # Reached end

        target_pos = Vec2(self.path[self.path_index + 1][0], self.path[self.path_index + 1][1])
        direction = (target_pos - self.pos).normalized
        self.pos += direction * self.speed

        if (target_pos - self.pos).magnitude < self.speed:
            self.pos = target_pos
            self.path_index += 1
        
        return False

class Tower:
    def __init__(self, pos, tower_type, stats):
        self.pos = pos
        self.type = tower_type
        self.range = stats['range']
        self.damage = stats['damage']
        self.cooldown = stats['cooldown']
        self.color = stats['color']
        self.cooldown_timer = 0

    def update(self, enemies, projectiles):
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return

        target = self._find_target(enemies)
        if target:
            # Sound: Pew!
            projectiles.append(Projectile(self.pos, target, self.damage))
            self.cooldown_timer = self.cooldown

    def _find_target(self, enemies):
        in_range_enemies = [e for e in enemies if (e.pos - self.pos).magnitude <= self.range]
        if not in_range_enemies:
            return None
        # Target the enemy furthest along the path
        return max(in_range_enemies, key=lambda e: e.path_index)

class Projectile:
    def __init__(self, start_pos, target, damage):
        self.pos = Vec2(start_pos.x, start_pos.y)
        self.target = target
        self.damage = damage
        self.speed = 15

    def move(self):
        if self.target.health <= 0: # Target already dead
            return True, False # Remove projectile, no hit
            
        direction = (self.target.pos - self.pos).normalized
        self.pos += direction * self.speed
        
        if (self.target.pos - self.pos).magnitude < self.speed:
            return True, True # Remove projectile, hit occurred
        return False, False

class Particle:
    def __init__(self, pos, color, size, life, velocity, gravity=0.0):
        self.pos = Vec2(pos.x, pos.y)
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life
        self.velocity = velocity
        self.gravity = gravity

    def update(self):
        self.pos += self.velocity
        self.velocity.y += self.gravity
        self.life -= 1
        return self.life <= 0

# --- Main Gymnasium Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move cursor. Space for Gatling Tower. "
        "Shift+↑ for Cannon Tower. Shift+↓ for Sniper Tower."
    )

    game_description = (
        "A minimalist tower defense game. Strategically place towers to "
        "defend your base from waves of geometric enemies."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Game Constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 3000
        self.TOTAL_ENEMIES = 25
        self.ENEMY_SPAWN_INTERVAL = 45

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PATH = (50, 60, 80)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_ENEMY = (230, 25, 75)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_TEXT = (245, 245, 245)
        
        self.TOWER_STATS = {
            1: {'name': 'Gatling', 'range': 80, 'damage': 2, 'cooldown': 10, 'color': (0, 130, 200)},
            2: {'name': 'Cannon', 'range': 120, 'damage': 10, 'cooldown': 45, 'color': (245, 130, 48)},
            3: {'name': 'Sniper', 'range': 200, 'damage': 5, 'cooldown': 30, 'color': (255, 225, 25)},
        }

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_small = pygame.font.SysFont("sans-serif", 14)
        
        # State variables (initialized in reset)
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.path = []
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = Vec2(0, 0)
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        self.enemies_defeated = 0
        self.base_enemy_speed = 1.2
        self.current_enemy_speed = 1.2
        self.selected_tower_type = 0 # For UI display
        
        self.validate_implementation()
    
    def _generate_path(self):
        w, h = self.SCREEN_WIDTH, self.SCREEN_HEIGHT
        path_nodes = [
            Vec2(-20, h // 2), Vec2(w * 0.2, h // 2), Vec2(w * 0.2, h * 0.8),
            Vec2(w * 0.5, h * 0.8), Vec2(w * 0.5, h * 0.2), Vec2(w * 0.8, h * 0.2),
            Vec2(w * 0.8, h // 2), Vec2(w + 20, h // 2)
        ]
        detailed_path = []
        for i in range(len(path_nodes) - 1):
            p1, p2 = path_nodes[i], path_nodes[i+1]
            dist = (p2 - p1).magnitude
            steps = max(1, int(dist / 5))
            for j in range(steps + 1):
                t = j / steps
                px = p1.x + t * (p2.x - p1.x)
                py = p1.y + t * (p2.y - p1.y)
                detailed_path.append((px, py))
        return detailed_path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.path = self._generate_path()
        self.enemies, self.towers, self.projectiles, self.particles = [], [], [], []
        self.cursor_pos = Vec2(self.SCREEN_WIDTH // 3, self.SCREEN_HEIGHT // 3)
        self.enemies_to_spawn = self.TOTAL_ENEMIES
        self.spawn_timer = self.ENEMY_SPAWN_INTERVAL
        self.enemies_defeated = 0
        self.current_enemy_speed = self.base_enemy_speed
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance: self.clock.tick(30)
            
        reward = 0.0
        terminated = self.game_over
        
        if not terminated:
            # --- 1. Handle Player Input ---
            self._handle_action(action)

            # --- 2. Update Game Logic ---
            self._update_spawner()
            for tower in self.towers: tower.update(self.enemies, self.projectiles)
            
            projectile_reward = self._update_projectiles()
            reward += projectile_reward
            
            enemy_reward, terminated_by_enemy = self._update_enemies()
            reward += enemy_reward
            if terminated_by_enemy:
                reward = -100.0
                terminated = True
                self.game_over = True

        self._update_particles()

        # --- 3. Check Termination Conditions ---
        if not terminated:
            if self.enemies_defeated == self.TOTAL_ENEMIES and not self.enemies:
                reward = 100.0
                terminated = True
                self.game_over = True
            elif self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True
        
        self.steps += 1
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        tower_to_place = 0
        if space_held: tower_to_place = 1
        elif shift_held and movement == 1: tower_to_place = 2
        elif shift_held and movement == 2: tower_to_place = 3
        
        self.selected_tower_type = tower_to_place

        if tower_to_place > 0:
            if self._is_valid_placement(self.cursor_pos):
                stats = self.TOWER_STATS[tower_to_place]
                self.towers.append(Tower(Vec2(self.cursor_pos.x, self.cursor_pos.y), tower_to_place, stats))
                # Sound: Place tower
        else: # Move cursor
            cursor_speed = 10
            if movement == 1: self.cursor_pos.y -= cursor_speed
            elif movement == 2: self.cursor_pos.y += cursor_speed
            elif movement == 3: self.cursor_pos.x -= cursor_speed
            elif movement == 4: self.cursor_pos.x += cursor_speed
        
        self.cursor_pos.x = max(0, min(self.SCREEN_WIDTH, self.cursor_pos.x))
        self.cursor_pos.y = max(0, min(self.SCREEN_HEIGHT, self.cursor_pos.y))

    def _update_spawner(self):
        self.spawn_timer -= 1
        if self.enemies_to_spawn > 0 and self.spawn_timer <= 0:
            enemy_health = 10 + (self.TOTAL_ENEMIES - self.enemies_to_spawn) * 2.5
            self.enemies.append(Enemy(self.path, self.current_enemy_speed, enemy_health))
            self.enemies_to_spawn -= 1
            self.spawn_timer = self.ENEMY_SPAWN_INTERVAL

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            removed, hit = p.move()
            if hit:
                reward += 0.1
                p.target.health -= p.damage
                self._create_particles(p.pos, self.COLOR_PROJECTILE, 5, 0.5)
                # Sound: Hit!
            if removed:
                self.projectiles.remove(p)
        return reward

    def _update_enemies(self):
        reward = 0
        reached_base = False
        for e in self.enemies[:]:
            if e.health <= 0:
                reward += 1.0
                self.enemies_defeated += 1
                if self.enemies_defeated > 0 and self.enemies_defeated % 5 == 0:
                    self.current_enemy_speed += 0.15
                self._create_particles(e.pos, self.COLOR_ENEMY, 25, 2, True)
                self.enemies.remove(e)
                # Sound: Enemy explosion!
                continue
            if e.move():
                reached_base = True
        return reward, reached_base

    def _update_particles(self):
        for p in self.particles[:]:
            if p.update():
                self.particles.remove(p)

    def _is_valid_placement(self, pos):
        path_buffer = 25
        for i in range(len(self.path) - 1):
            p1 = Vec2(self.path[i][0], self.path[i][1])
            p2 = Vec2(self.path[i+1][0], self.path[i+1][1])
            rect = pygame.Rect(min(p1.x, p2.x) - path_buffer, min(p1.y, p2.y) - path_buffer,
                               abs(p1.x - p2.x) + 2 * path_buffer, abs(p1.y - p2.y) + 2 * path_buffer)
            if rect.collidepoint(pos.as_tuple): return False
        for tower in self.towers:
            if (tower.pos - pos).magnitude < 20: return False
        return True

    def _create_particles(self, pos, color, count, speed_mult, is_explosion=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = Vec2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(15, 30) if is_explosion else self.np_random.integers(5, 15)
            size = self.np_random.integers(2, 5) if is_explosion else self.np_random.integers(1, 3)
            self.particles.append(Particle(pos, color, size, life, vel))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        # Draw path
        if len(self.path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 40)
        # Draw base
        pygame.gfxdraw.box(self.screen, pygame.Rect(self.path[-1][0] - 15, self.path[-1][1] - 15, 30, 30), self.COLOR_BASE)
        
        # Draw towers
        for t in self.towers:
            pygame.gfxdraw.filled_circle(self.screen, t.pos.as_tuple[0], t.pos.as_tuple[1], 8, t.color)
            pygame.gfxdraw.aacircle(self.screen, t.pos.as_tuple[0], t.pos.as_tuple[1], 8, t.color)

        # Draw cursor and range indicator
        cursor_pos_tup = self.cursor_pos.as_tuple
        is_valid = self._is_valid_placement(self.cursor_pos)
        cursor_color = (0, 255, 0, 100) if is_valid else (255, 0, 0, 100)
        if self.selected_tower_type > 0:
            radius = self.TOWER_STATS[self.selected_tower_type]['range']
            pygame.gfxdraw.aacircle(self.screen, cursor_pos_tup[0], cursor_pos_tup[1], radius, cursor_color)
        pygame.gfxdraw.filled_circle(self.screen, cursor_pos_tup[0], cursor_pos_tup[1], 10, cursor_color)
        
        # Draw enemies
        for e in self.enemies:
            radius = int(e.base_radius * (e.health / e.max_health) + 2)
            pygame.gfxdraw.filled_circle(self.screen, e.pos.as_tuple[0], e.pos.as_tuple[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, e.pos.as_tuple[0], e.pos.as_tuple[1], radius, self.COLOR_ENEMY)

        # Draw projectiles and particles
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, p.pos.as_tuple[0], p.pos.as_tuple[1], 3, self.COLOR_PROJECTILE)
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = p.color + (alpha,) if len(p.color) == 3 else p.color
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), int(p.size), color)
        
        # Draw UI
        enemies_text = self.font_ui.render(f"ENEMIES: {self.enemies_defeated}/{self.TOTAL_ENEMIES}", True, self.COLOR_TEXT)
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(enemies_text, (10, 10))
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "enemies_defeated": self.enemies_defeated}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    import time

    # --- Headless Test ---
    print("--- Running Headless Test ---")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    start_time = time.time()
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated: break
    end_time = time.time()
    print(f"Headless simulation of 200 steps took {end_time - start_time:.2f} seconds.")
    print(f"Final Info: {info}")
    env.close()

    # --- Interactive Test ---
    if "SDL_VIDEODRIVER" in os.environ: del os.environ["SDL_VIDEODRIVER"]
    print("\n--- Starting Interactive Mode (close window to exit) ---")
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
    
    env.close()