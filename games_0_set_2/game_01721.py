
# Generated: 2025-08-28T02:29:41.771137
# Source Brief: brief_01721.md
# Brief Index: 1721

        
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
        "Controls: Press Space to cycle through tower placement spots and deploy a tower. "
        "Survive all waves to win."
    )

    game_description = (
        "An isometric tower defense game. Strategically place towers to defend your base "
        "against waves of enemies. Survive 10 waves to win."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 50, 62)
        self.COLOR_PATH = (60, 66, 82)
        self.COLOR_BASE = (255, 200, 0)
        self.COLOR_TOWER = (0, 255, 150)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_PROJECTILE = (0, 200, 255)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_GAMEOVER = (255, 0, 0)
        self.COLOR_WIN = (0, 255, 0)

        # EXACT spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Isometric grid setup
        self.TILE_WIDTH = 48
        self.TILE_HEIGHT = 24
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # Game constants
        self.MAX_WAVES = 10
        self.MAX_STEPS = 15000 # ~8 minutes at 30fps
        self.BASE_START_HEALTH = 100
        self.INTER_WAVE_DELAY = 3 * self.FPS # 3 seconds

        # Define tower placement spots and enemy path in grid coordinates
        self._define_world_layout()

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = 0
        self.current_wave = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.inter_wave_timer = 0
        self.enemies_to_spawn = []
        self.spawn_timer = 0
        self.placement_cursor_index = 0
        self.prev_space_held = False
        
        self.reset()
        
        self.validate_implementation()

    def _define_world_layout(self):
        """Pre-calculates screen positions for grid elements."""
        self.placement_spots_grid = [
            (3, 1), (5, 1), (1, 3), (3, 3), (5, 3), (7, 3),
            (1, 5), (3, 5), (5, 5), (7, 5), (3, 7), (5, 7)
        ]
        self.placement_spots_screen = [self._iso_to_screen(x, y) for x, y in self.placement_spots_grid]

        self.enemy_path_grid = [
            (-1, 4), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4),
            (6, 4), (6, 3), (6, 2), (5, 2), (4, 2), (3, 2), (2, 2),
            (2, 3), (2, 5), (2, 6), (2, 7), (3, 7), (4, 7), (5, 7),
            (6, 7), (7, 7), (8, 7), (9, 7)
        ]
        self.enemy_path_screen = [self._iso_to_screen(x, y) for x, y in self.enemy_path_grid]
        self.base_pos_grid = (9, 8)
        self.base_pos_screen = self._iso_to_screen(*self.base_pos_grid)

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * self.TILE_WIDTH / 2
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = self.BASE_START_HEALTH
        self.current_wave = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.enemies_to_spawn = []
        self.spawn_timer = 0
        self.placement_cursor_index = 0
        self.prev_space_held = False

        self.inter_wave_timer = self.INTER_WAVE_DELAY // 2
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle input for tower placement
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            reward += self._place_tower()
        self.prev_space_held = space_held
        
        # --- Game Logic Update ---
        self.steps += 1
        
        # Wave management
        if not self.enemies and not self.enemies_to_spawn:
            if self.current_wave >= self.MAX_WAVES:
                self.win = True
                self.game_over = True
            elif self.inter_wave_timer > 0:
                self.inter_wave_timer -= 1
            else:
                self._start_next_wave()
        
        # Enemy spawning
        if self.enemies_to_spawn:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self.enemies.append(self.enemies_to_spawn.pop(0))
                self.spawn_timer = self.FPS // 2 # 0.5 sec between spawns

        # Update towers
        for tower in self.towers:
            reward += tower.update(self.enemies, self.projectiles)
            
        # Update projectiles
        projectiles_to_remove = []
        for proj in self.projectiles:
            hit, destroyed = proj.update(self.particles)
            if hit:
                reward += 0.1 # Reward for hitting
                if destroyed:
                    reward += 1.0 # Reward for destroying
                    self.score += 10
            if proj.is_dead:
                projectiles_to_remove.append(proj)
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]

        # Update enemies
        enemies_to_remove = []
        for enemy in self.enemies:
            reached_base = enemy.update(self.enemy_path_screen)
            if reached_base:
                damage = enemy.damage
                self.base_health -= damage
                reward -= damage # Negative reward for base damage
                enemies_to_remove.append(enemy)
                self._create_particles(self.base_pos_screen, self.COLOR_ENEMY, 20)
            elif enemy.health <= 0:
                enemies_to_remove.append(enemy)
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        
        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # Check termination conditions
        terminated = False
        if self.base_health <= 0:
            self.base_health = 0
            reward -= 100
            self.game_over = True
            terminated = True
        elif self.win:
            reward += 100
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        self.inter_wave_timer = self.INTER_WAVE_DELAY
        
        num_enemies = 2 + self.current_wave
        speed_multiplier = 1.0 + (self.current_wave // 2) * 0.05
        base_speed = 0.8
        
        enemy_health = 80 + self.current_wave * 20
        
        self.enemies_to_spawn = []
        for _ in range(num_enemies):
            start_pos = self.enemy_path_screen[0]
            enemy = Enemy(start_pos, enemy_health, base_speed * speed_multiplier)
            self.enemies_to_spawn.append(enemy)
        self.spawn_timer = 0
    
    def _place_tower(self):
        occupied_spots = [t.grid_pos for t in self.towers]
        
        # Find next available spot
        initial_cursor = self.placement_cursor_index
        while self.placement_spots_grid[self.placement_cursor_index] in occupied_spots:
            self.placement_cursor_index = (self.placement_cursor_index + 1) % len(self.placement_spots_grid)
            if self.placement_cursor_index == initial_cursor: # All spots are full
                return 0 # No placement, no reward change

        # Place tower at the found spot
        grid_pos = self.placement_spots_grid[self.placement_cursor_index]
        screen_pos = self.placement_spots_screen[self.placement_cursor_index]
        new_tower = Tower(screen_pos, grid_pos)
        self.towers.append(new_tower)
        # sfx: tower_place.wav
        
        # Advance cursor for next time
        self.placement_cursor_index = (self.placement_cursor_index + 1) % len(self.placement_spots_grid)
        return 0

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(Particle(pos, color, self.np_random))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        if len(self.enemy_path_screen) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.enemy_path_screen, 10)
        
        # Draw placement spots
        for i, pos in enumerate(self.placement_spots_screen):
            is_occupied = self.placement_spots_grid[i] in [t.grid_pos for t in self.towers]
            color = self.COLOR_TOWER if is_occupied else self.COLOR_GRID
            pygame.draw.circle(self.screen, color, pos, 5, 1 if not is_occupied else 0)

        # Draw placement cursor
        if not self.game_over:
            cursor_pos = self.placement_spots_screen[self.placement_cursor_index]
            pulse = abs(math.sin(self.steps * 0.2))
            radius = int(8 + pulse * 4)
            pygame.draw.circle(self.screen, self.COLOR_CURSOR, cursor_pos, radius, 1)

        # Sort and draw all dynamic entities for correct isometric layering
        render_list = self.towers + self.enemies
        render_list.sort(key=lambda e: e.pos[1])
        for entity in render_list:
            entity.draw(self.screen)

        # Draw base
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.base_pos_screen[0]-10, self.base_pos_screen[1]-20, 20, 30))
        pygame.gfxdraw.box(self.screen, (self.base_pos_screen[0]-10, self.base_pos_screen[1]-20, 20, 30), (*self.COLOR_BASE, 100))
        
        # Draw projectiles and particles on top
        for proj in self.projectiles:
            proj.draw(self.screen)
        for particle in self.particles:
            particle.draw(self.screen)
            
    def _render_ui(self):
        # Base Health
        health_text = self.font_small.render(f"Base Health: {int(self.base_health)}%", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        health_bar_bg = pygame.Rect(10, 30, 150, 15)
        health_bar_fg = pygame.Rect(10, 30, int(150 * (self.base_health / self.BASE_START_HEALTH)), 15)
        pygame.draw.rect(self.screen, self.COLOR_GRID, health_bar_bg)
        pygame.draw.rect(self.screen, self.COLOR_BASE, health_bar_fg)
        
        # Wave Info
        if not self.enemies and not self.enemies_to_spawn and not self.win:
             wave_text_str = f"Wave {self.current_wave + 1} starting in {self.inter_wave_timer / self.FPS:.1f}s"
        else:
             wave_text_str = f"Wave: {self.current_wave}/{self.MAX_WAVES}"
        wave_text = self.font_small.render(wave_text_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 30))
        
        # Game Over / Win Message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_GAMEOVER
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "wave": self.current_wave,
            "enemies_left": len(self.enemies) + len(self.enemies_to_spawn),
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

class Tower:
    def __init__(self, pos, grid_pos):
        self.pos = pygame.Vector2(pos)
        self.grid_pos = grid_pos
        self.range = 100
        self.cooldown = 0
        self.fire_rate = 20 # frames
        self.color = (0, 255, 150)
        self.size = 8
        self.fire_glow = 0

    def update(self, enemies, projectiles):
        if self.fire_glow > 0:
            self.fire_glow -= 1
        
        if self.cooldown > 0:
            self.cooldown -= 1
            return 0
        
        target = self._find_target(enemies)
        if target:
            self._fire(target, projectiles)
            return 0.01 # Small reward for actively targeting
        return 0

    def _find_target(self, enemies):
        closest_enemy = None
        min_dist = self.range
        for enemy in enemies:
            dist = self.pos.distance_to(enemy.pos)
            if dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        return closest_enemy

    def _fire(self, target, projectiles):
        projectiles.append(Projectile(self.pos, target))
        self.cooldown = self.fire_rate
        self.fire_glow = 5
        # sfx: laser_shoot.wav

    def draw(self, screen):
        # Draw tower body
        points = [
            self.pos + (0, -self.size * 1.5),
            self.pos + (self.size, -self.size * 0.5),
            self.pos + (0, self.size * 0.5),
            self.pos + (-self.size, -self.size * 0.5),
        ]
        pygame.draw.polygon(screen, self.color, points)
        pygame.draw.polygon(screen, tuple(min(255, c+50) for c in self.color), points, 1)

        if self.fire_glow > 0:
            glow_color = (*self.color, self.fire_glow * 30)
            pygame.gfxdraw.filled_circle(screen, int(self.pos.x), int(self.pos.y), self.size, glow_color)

class Enemy:
    def __init__(self, start_pos, health, speed):
        self.pos = pygame.Vector2(start_pos)
        self.max_health = health
        self.health = health
        self.speed = speed
        self.path_index = 1
        self.color = (255, 50, 50)
        self.size = 7
        self.damage = 10

    def update(self, path):
        if self.path_index >= len(path):
            return True # Reached base
        
        target_pos = pygame.Vector2(path[self.path_index])
        direction = (target_pos - self.pos)
        
        if direction.length() < self.speed:
            self.pos = target_pos
            self.path_index += 1
        else:
            self.pos += direction.normalize() * self.speed
            
        return False

    def take_damage(self, amount):
        self.health -= amount
        return self.health <= 0

    def draw(self, screen):
        # Draw enemy body
        body_rect = pygame.Rect(self.pos.x - self.size, self.pos.y - self.size, self.size * 2, self.size * 2)
        pygame.draw.rect(screen, self.color, body_rect, border_radius=3)

        # Draw health bar
        if self.health < self.max_health:
            bar_width = 20
            bar_height = 4
            bar_x = self.pos.x - bar_width / 2
            bar_y = self.pos.y - self.size - 10
            health_ratio = self.health / self.max_health
            
            pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(screen, self.color, (bar_x, bar_y, bar_width * health_ratio, bar_height))

class Projectile:
    def __init__(self, start_pos, target_enemy):
        self.pos = pygame.Vector2(start_pos)
        self.target = target_enemy
        self.speed = 10
        self.damage = 45
        self.is_dead = False
        self.color = (0, 200, 255)

    def update(self, particles):
        if self.is_dead or self.target.health <= 0:
            self.is_dead = True
            return False, False
        
        direction = (self.target.pos - self.pos)
        dist = direction.length()
        
        if dist < self.speed:
            self.is_dead = True
            destroyed = self.target.take_damage(self.damage)
            # sfx: hit_enemy.wav
            for _ in range(5):
                particles.append(Particle(self.target.pos, self.color, np.random.default_rng()))
            return True, destroyed
        else:
            self.pos += direction.normalize() * self.speed
            return False, False

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), 3)
        pygame.gfxdraw.filled_circle(screen, int(self.pos.x), int(self.pos.y), 5, (*self.color, 100))

class Particle:
    def __init__(self, pos, color, rng):
        self.pos = pygame.Vector2(pos)
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 3)
        self.vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        self.lifespan = 15
        self.color = color
        self.size = rng.integers(2, 5)

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95
        self.lifespan -= 1
        return self.lifespan > 0

    def draw(self, screen):
        alpha = int(255 * (self.lifespan / 15))
        color_with_alpha = (*self.color, alpha)
        pygame.gfxdraw.filled_circle(screen, int(self.pos.x), int(self.pos.y), self.size, color_with_alpha)