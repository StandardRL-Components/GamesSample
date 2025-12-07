
# Generated: 2025-08-28T06:20:13.690862
# Source Brief: brief_05868.md
# Brief Index: 5868

        
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
class Tower:
    def __init__(self, grid_pos, tower_type, tile_size):
        self.grid_pos = grid_pos
        self.pixel_pos = (
            (grid_pos[0] + 0.5) * tile_size,
            (grid_pos[1] + 0.5) * tile_size,
        )
        self.type = tower_type
        self.stats = Tower.get_stats(tower_type)
        self.range = self.stats["range"] * tile_size
        self.damage = self.stats["damage"]
        self.cooldown_max = self.stats["cooldown"]
        self.cooldown = 0
        self.target = None
        self.placement_animation_timer = 30  # 1 second animation

    @staticmethod
    def get_stats(tower_type):
        if tower_type == 0:  # Gatling Tower
            return {"range": 3, "damage": 5, "cooldown": 10, "color": (0, 255, 0), "projectile_speed": 15, "projectile_color": (150, 255, 150)}
        elif tower_type == 1:  # Cannon Tower
            return {"range": 4, "damage": 25, "cooldown": 60, "color": (0, 150, 255), "projectile_speed": 10, "projectile_color": (150, 150, 255)}
        return {}

    def update(self, enemies, projectiles, tile_size):
        if self.placement_animation_timer > 0:
            self.placement_animation_timer -= 1
        
        self.cooldown = max(0, self.cooldown - 1)
        
        if self.target and (self.target.health <= 0 or self.dist_to(self.target) > self.range):
            self.target = None

        if not self.target:
            in_range_enemies = [e for e in enemies if self.dist_to(e) <= self.range]
            if in_range_enemies:
                self.target = min(in_range_enemies, key=lambda e: e.distance_traveled)

        if self.target and self.cooldown == 0:
            # sfx: tower_shoot
            projectiles.append(Projectile(self.pixel_pos, self.target, self.damage, self.stats["projectile_speed"], self.stats["projectile_color"]))
            self.cooldown = self.cooldown_max
            return True
        return False
    
    def dist_to(self, enemy):
        return math.hypot(self.pixel_pos[0] - enemy.pos[0], self.pixel_pos[1] - enemy.pos[1])

class Enemy:
    def __init__(self, path, base_health, base_speed, wave):
        self.path = path
        self.path_index = 0
        self.pos = list(path[0])
        self.speed = base_speed + (0.05 * math.floor((wave - 1) / 2))
        self.max_health = base_health * (1.05 ** (wave - 1))
        self.health = self.max_health
        self.distance_traveled = 0

    def update(self):
        if self.path_index >= len(self.path) - 1:
            return True  # Reached the end

        target_pos = self.path[self.path_index + 1]
        dx = target_pos[0] - self.pos[0]
        dy = target_pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.pos = list(target_pos)
            self.path_index += 1
            self.distance_traveled += dist
        else:
            self.pos[0] += (dx / dist) * self.speed
            self.pos[1] += (dy / dist) * self.speed
            self.distance_traveled += self.speed
        
        return False

class Projectile:
    def __init__(self, start_pos, target, damage, speed, color):
        self.pos = list(start_pos)
        self.target = target
        self.damage = damage
        self.speed = speed
        self.color = color

    def update(self):
        if self.target.health <= 0:
            return True # Target is dead, projectile fizzles

        dx = self.target.pos[0] - self.pos[0]
        dy = self.target.pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.target.health -= self.damage
            # sfx: enemy_hit
            return True # Hit target
        else:
            self.pos[0] += (dx / dist) * self.speed
            self.pos[1] += (dy / dist) * self.speed
        
        return False

class Particle:
    def __init__(self, pos, vel, color, size, lifespan):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.size = size
        self.lifespan = lifespan
        self.life = lifespan

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1
        return self.life <= 0

    def draw(self, surface):
        alpha = int(255 * (self.life / self.lifespan))
        temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.color + (alpha,), (self.size, self.size), self.size)
        surface.blit(temp_surf, (int(self.pos[0] - self.size), int(self.pos[1] - self.size)), special_flags=pygame.BLEND_RGBA_ADD)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to place tower. Shift to cycle tower type."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers on a grid."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 32, 20
        self.TILE_SIZE = self.WIDTH // self.GRID_COLS
        self.MAX_STEPS = 30 * 120 # 2 minutes at 30fps
        self.MAX_WAVES = 10
        self.MAX_TOWERS = 10
        self.BASE_ENEMY_HEALTH = 50
        self.BASE_ENEMY_SPEED = 1.0

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_PATH = (45, 55, 75)
        self.COLOR_BASE = (255, 200, 0)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BG = (80, 20, 20)
        self.COLOR_HEALTH_FG = (50, 255, 50)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Define enemy path
        self._define_path()
        
        # Initialize state variables
        self.cursor_pos = None
        self.grid = None
        self.towers = None
        self.enemies = None
        self.projectiles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.current_wave = None
        self.wave_timer = None
        self.enemies_to_spawn_in_wave = None
        self.enemies_spawned_in_wave = None
        self.selected_tower_type = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.towers_remaining = None
        self.initial_towers = self.MAX_TOWERS

        self.reset()
        
        # self.validate_implementation() # Uncomment to run self-checks

    def _define_path(self):
        path_grid = [
            (0, 10), (5, 10), (5, 5), (15, 5), (15, 15),
            (25, 15), (25, 2), (self.GRID_COLS - 1, 2)
        ]
        self.path_pixels = [( (p[0] + 0.5) * self.TILE_SIZE, (p[1] + 0.5) * self.TILE_SIZE ) for p in path_grid]
        self.path_tiles = set()
        for i in range(len(path_grid) - 1):
            p1 = path_grid[i]
            p2 = path_grid[i+1]
            x1, y1 = p1
            x2, y2 = p2
            for x in range(min(x1, x2), max(x1, x2) + 1):
                self.path_tiles.add((x, y1))
            for y in range(min(y1, y2), max(y1, y2) + 1):
                self.path_tiles.add((x2, y))
        self.base_pos = path_grid[-1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.grid = [[None for _ in range(self.GRID_ROWS)] for _ in range(self.GRID_COLS)]
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.current_wave = 0
        self.wave_timer = 150 # 5 seconds before first wave
        self.enemies_to_spawn_in_wave = 0
        self.enemies_spawned_in_wave = 0
        
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.towers_remaining = self.initial_towers
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = -0.01 # Time penalty
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle input
        self._handle_input(movement, space_held, shift_held)
        
        # Place towers
        if space_held and not self.prev_space_held:
            if self._is_valid_placement(self.cursor_pos):
                self._place_tower(self.cursor_pos)
        
        # Cycle tower types
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % 2 # 2 tower types

        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        # Update game state
        reward += self._update_waves()
        
        for tower in self.towers:
            if tower.update(self.enemies, self.projectiles, self.TILE_SIZE):
                self._create_particles(tower.pixel_pos, (255, 255, 0), 3)

        new_projectiles = []
        for p in self.projectiles:
            if p.update():
                reward += 0.1 # Hit reward
                self._create_particles(p.pos, p.color, 5)
            else:
                new_projectiles.append(p)
        self.projectiles = new_projectiles

        new_enemies = []
        for e in self.enemies:
            if e.health <= 0:
                reward += 1.0 # Defeat reward
                self.score += 10 * self.current_wave
                self._create_particles(e.pos, self.COLOR_ENEMY, 15, 2)
                # sfx: enemy_die
                continue
            if e.update():
                # Enemy reached base
                if self.towers:
                    self.towers.pop(0) # Destroy oldest tower
                    self.towers_remaining -= 1
                    reward -= 10
                    # sfx: tower_destroyed
                else: # No towers to destroy, but enemy reached base
                    reward -= 1 # Small penalty
                continue
            new_enemies.append(e)
        self.enemies = new_enemies
        
        self.particles = [p for p in self.particles if not p.update()]
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.current_wave > self.MAX_WAVES:
                reward += 100 # Win bonus
            elif self.towers_remaining <= 0 and self.initial_towers > 0:
                reward -= 100 # Loss penalty

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        if movement == 2: self.cursor_pos[1] += 1 # Down
        if movement == 3: self.cursor_pos[0] -= 1 # Left
        if movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

    def _update_waves(self):
        if self.current_wave > self.MAX_WAVES:
            return 0

        self.wave_timer -= 1
        if self.wave_timer <= 0 and self.current_wave == 0:
            self._start_next_wave()

        if self.current_wave > 0 and not self.enemies and self.enemies_spawned_in_wave == self.enemies_to_spawn_in_wave:
            # Wave complete
            if self.wave_timer <= 0:
                self._start_next_wave()
            return 0

        if self.current_wave > 0 and self.wave_timer <= 0 and self.enemies_spawned_in_wave < self.enemies_to_spawn_in_wave:
            self.enemies.append(Enemy(self.path_pixels, self.BASE_ENEMY_HEALTH, self.BASE_ENEMY_SPEED, self.current_wave))
            self.enemies_spawned_in_wave += 1
            self.wave_timer = 30 # Time between enemies
        
        return 0

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES: return
        self.enemies_to_spawn_in_wave = 5 + self.current_wave * 2
        self.enemies_spawned_in_wave = 0
        self.wave_timer = 150 # 5 seconds between waves
        self.score += 50 * (self.current_wave - 1)

    def _is_valid_placement(self, grid_pos):
        x, y = grid_pos
        if self.towers_remaining <= 0: return False
        if self.grid[x][y] is not None: return False
        if (x, y) in self.path_tiles: return False
        if (x,y) == self.base_pos: return False
        return True

    def _place_tower(self, grid_pos):
        # sfx: place_tower
        tower = Tower(grid_pos, self.selected_tower_type, self.TILE_SIZE)
        self.towers.append(tower)
        self.grid[grid_pos[0]][grid_pos[1]] = tower
        self.towers_remaining -= 1
        self._create_particles(tower.pixel_pos, (255, 255, 255), 20, 1)

    def _check_termination(self):
        if self.game_over: return True
        if self.steps >= self.MAX_STEPS: self.game_over = True
        if self.current_wave > self.MAX_WAVES: self.game_over = True
        if self.towers_remaining <= 0 and self.initial_towers > 0: self.game_over = True
        return self.game_over

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = random.randint(2, 5)
            lifespan = random.randint(10, 20)
            self.particles.append(Particle(pos, vel, color, size, lifespan))
    
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
            "wave": self.current_wave,
            "towers_remaining": self.towers_remaining,
        }

    def _render_game(self):
        # Draw grid and path
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if (x, y) in self.path_tiles:
                    pygame.draw.rect(self.screen, self.COLOR_PATH, rect)
        for x in range(0, self.WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # Draw base
        base_rect = (self.base_pos[0] * self.TILE_SIZE, self.base_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        
        # Draw towers
        for tower in self.towers:
            pos = (int(tower.pixel_pos[0]), int(tower.pixel_pos[1]))
            color = tower.stats["color"]
            if tower.placement_animation_timer > 0:
                rad = int(self.TILE_SIZE/2 * (1 - tower.placement_animation_timer/30))
            else:
                rad = int(self.TILE_SIZE/2 * 0.8)
            
            if tower.type == 0: # Gatling
                pygame.draw.circle(self.screen, color, pos, rad)
                pygame.draw.circle(self.screen, self.COLOR_BG, pos, rad-2)
            elif tower.type == 1: # Cannon
                side = rad * 2
                rect = pygame.Rect(pos[0]-rad, pos[1]-rad, side, side)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_BG, rect.inflate(-4, -4))

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy.pos[0]), int(enemy.pos[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, self.COLOR_ENEMY)
            # Health bar
            health_ratio = enemy.health / enemy.max_health
            bar_width = 12
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (pos[0] - bar_width/2, pos[1] - 12, bar_width, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (pos[0] - bar_width/2, pos[1] - 12, bar_width * health_ratio, 3))

        # Draw projectiles
        for p in self.projectiles:
            pos = (int(p.pos[0]), int(p.pos[1]))
            pygame.draw.circle(self.screen, p.color, pos, 3)

        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)
        
        # Draw cursor and placement info
        cursor_px = (self.cursor_pos[0] * self.TILE_SIZE, self.cursor_pos[1] * self.TILE_SIZE)
        valid_placement = self._is_valid_placement(self.cursor_pos)
        cursor_color = (0, 255, 0, 100) if valid_placement else (255, 0, 0, 100)
        
        s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        s.fill(cursor_color)
        self.screen.blit(s, cursor_px)
        
        # Draw selected tower preview in cursor
        stats = Tower.get_stats(self.selected_tower_type)
        preview_pos = (cursor_px[0] + self.TILE_SIZE//2, cursor_px[1] + self.TILE_SIZE//2)
        if stats:
            pygame.gfxdraw.aacircle(self.screen, preview_pos[0], preview_pos[1], int(stats["range"] * self.TILE_SIZE), (255,255,255,50))

    def _render_ui(self):
        # Top bar
        info_text = (
            f"Wave: {self.current_wave if self.current_wave <= self.MAX_WAVES else 'WIN'} | "
            f"Towers: {self.towers_remaining}/{self.initial_towers} | "
            f"Score: {self.score}"
        )
        text_surf = self.font_small.render(info_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 5))

        # Wave timer
        if self.wave_timer > 0 and (self.current_wave == 0 or (not self.enemies and self.enemies_spawned_in_wave == self.enemies_to_spawn_in_wave)):
            timer_text = f"Next wave in: {self.wave_timer / 30:.1f}s"
            text_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

        # Game Over / Win Text
        if self.game_over:
            msg = ""
            if self.current_wave > self.MAX_WAVES:
                msg = "YOU SURVIVED"
            elif self.towers_remaining <= 0 and self.initial_towers > 0:
                msg = "ALL TOWERS LOST"
            elif self.steps >= self.MAX_STEPS:
                msg = "TIME UP"
            
            if msg:
                text_surf = self.font_large.render(msg, True, self.COLOR_BASE)
                text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 30))
                self.screen.blit(text_surf, text_rect)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- To run with a random agent ---
    # obs, info = env.reset()
    # done = False
    # total_reward = 0
    # while not done:
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
    #     total_reward += reward
    #     # The environment is headless, but you could save frames:
    #     # from PIL import Image
    #     # img = Image.fromarray(obs)
    #     # img.save(f"frame_{info['steps']:04d}.png")
    # print(f"Episode finished. Total reward: {total_reward}, Info: {info}")

    # --- To run with manual control ---
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Key state tracking
    movement = 0
    space_held = False
    shift_held = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: space_held = True
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space_held = False
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        else: movement = 0

        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30)

    print(f"Game over! Final score: {info['score']}, Total reward: {total_reward:.2f}")
    pygame.quit()