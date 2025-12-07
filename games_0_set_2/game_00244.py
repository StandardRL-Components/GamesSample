
# Generated: 2025-08-27T13:03:05.362417
# Source Brief: brief_00244.md
# Brief Index: 244

        
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
class Alien:
    def __init__(self, x, y, health, speed):
        self.x = x
        self.y = y
        self.health = health
        self.max_health = health
        self.speed = speed
        self.hit_timer = 0

class Tower:
    def __init__(self, grid_x, grid_y, cell_size):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.x = (grid_x + 0.5) * cell_size
        self.y = (grid_y + 0.5) * cell_size
        self.range = 120
        self.fire_rate = 20  # steps per shot
        self.cooldown = 0
        self.build_progress = 0.0 # For animation

class Projectile:
    def __init__(self, start_x, start_y, target_alien, speed=15, damage=1):
        self.x = start_x
        self.y = start_y
        self.target = target_alien
        self.speed = speed
        self.damage = damage
        # Aim for the target's current position
        angle = math.atan2(target_alien.y - self.y, target_alien.x - self.x)
        self.vx = math.cos(angle) * self.speed
        self.vy = math.sin(angle) * self.speed
        self.lifespan = 40 # steps

class Particle:
    def __init__(self, x, y, color, size, lifespan):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.size = size
        self.lifespan = lifespan

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. Press Space to build a tower. Survive the waves!"
    )

    game_description = (
        "A top-down tower defense game. Defend your base from descending aliens by strategically placing defensive towers."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    CELL_SIZE = 40
    BASE_Y_START = SCREEN_HEIGHT - CELL_SIZE
    MAX_STEPS = 3000

    # --- Colors ---
    COLOR_BG = (15, 18, 28)
    COLOR_GRID = (30, 35, 50)
    COLOR_BASE = (30, 100, 70)
    COLOR_BASE_BORDER = (50, 160, 120)
    COLOR_ALIEN = (210, 50, 50)
    COLOR_ALIEN_HIT = (255, 255, 255)
    COLOR_TOWER = (60, 120, 220)
    COLOR_TOWER_BORDER = (100, 180, 255)
    COLOR_PROJECTILE = (255, 220, 100)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR = (80, 200, 120)
    COLOR_HEALTH_BAR_BG = (100, 40, 40)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.aliens = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.last_space_press = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.step_reward = 0

        # Player state
        self.base_health = 100
        self.max_base_health = 100
        self.resources = 80
        self.tower_cost = 25

        # Wave state
        self.wave_number = 0
        self.wave_cooldown = 0
        self.aliens_to_spawn = []

        # Entities
        self.aliens = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        # Controls
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2 - 1]
        self.last_space_press = False

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.step_reward = 0
        self.steps += 1

        if not self.game_over:
            self._handle_input(action)
            self._update_game_logic()

        terminated = self._check_termination()

        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 2) # Can't build on base row

        # Place tower
        if space_held and not self.last_space_press:
            self._place_tower()
        self.last_space_press = space_held

    def _place_tower(self):
        if self.resources >= self.tower_cost:
            is_occupied = any(t.grid_x == self.cursor_pos[0] and t.grid_y == self.cursor_pos[1] for t in self.towers)
            if not is_occupied:
                self.resources -= self.tower_cost
                self.towers.append(Tower(self.cursor_pos[0], self.cursor_pos[1], self.CELL_SIZE))
                # sfx: build_tower.wav

    def _update_game_logic(self):
        self._update_towers()
        self._update_projectiles()
        self._update_aliens()
        self._update_particles()
        self._update_waves()

    def _update_towers(self):
        for tower in self.towers:
            tower.build_progress = min(1.0, tower.build_progress + 0.1)
            if tower.cooldown > 0:
                tower.cooldown -= 1
                continue

            # Find a target
            target = None
            min_dist = tower.range ** 2
            for alien in self.aliens:
                dist_sq = (alien.x - tower.x)**2 + (alien.y - tower.y)**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = alien

            if target:
                self.projectiles.append(Projectile(tower.x, tower.y, target))
                tower.cooldown = tower.fire_rate
                # sfx: tower_shoot.wav

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p.lifespan -= 1
            if p.lifespan <= 0:
                self.projectiles.remove(p)
                continue

            p.x += p.vx
            p.y += p.vy
            
            # Simple collision check
            hitbox_size = 10
            if abs(p.x - p.target.x) < hitbox_size and abs(p.y - p.target.y) < hitbox_size:
                p.target.health -= p.damage
                p.target.hit_timer = 5  # Flash for 5 frames
                self.step_reward += 0.1
                self.projectiles.remove(p)
                self._create_particles(p.x, p.y, self.COLOR_PROJECTILE, 3, 1, 5)
                # sfx: alien_hit.wav

    def _update_aliens(self):
        for alien in self.aliens[:]:
            if alien.hit_timer > 0:
                alien.hit_timer -= 1

            alien.y += alien.speed

            # Alien reaches base
            if alien.y >= self.BASE_Y_START:
                self.base_health -= 10
                self.step_reward -= 5
                self.aliens.remove(alien)
                self._create_particles(alien.x, self.BASE_Y_START, self.COLOR_ALIEN, 15, 3, 15)
                # sfx: base_damage.wav
                continue

            # Alien destroyed
            if alien.health <= 0:
                self.score += 1
                self.step_reward += 1
                self.resources += 5
                self.aliens.remove(alien)
                self._create_particles(alien.x, alien.y, self.COLOR_ALIEN, 20, 3, 20)
                # sfx: alien_explode.wav

    def _update_particles(self):
        for p in self.particles[:]:
            p.lifespan -= 1
            if p.lifespan <= 0:
                self.particles.remove(p)
                continue
            p.x += p.vx
            p.y += p.vy
            p.size = max(0, p.size - 0.1)

    def _create_particles(self, x, y, color, count, size, lifespan):
        for _ in range(count):
            self.particles.append(Particle(x, y, color, size, lifespan))

    def _update_waves(self):
        if not self.aliens and not self.aliens_to_spawn:
            if self.wave_cooldown > 0:
                self.wave_cooldown -= 1
            else:
                if self.wave_number > 0:
                    self.step_reward += 100
                if self.wave_number >= 10:
                    self.game_over = True
                    self.victory = True
                else:
                    self._start_next_wave()
        
        if self.aliens_to_spawn and self.steps % 10 == 0: # Spawn one alien every 10 steps
            self.aliens.append(self.aliens_to_spawn.pop(0))

    def _start_next_wave(self):
        self.wave_number += 1
        self.wave_cooldown = 150 # 5 seconds at 30fps
        
        num_aliens = 5 + self.wave_number * 2
        alien_speed = 0.5 + self.wave_number * 0.05
        alien_health = 1 + self.wave_number
        
        for _ in range(num_aliens):
            x = random.uniform(self.CELL_SIZE, self.SCREEN_WIDTH - self.CELL_SIZE)
            y = random.uniform(-100, -20)
            self.aliens_to_spawn.append(Alien(x, y, alien_health, alien_speed))

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.base_health = 0
            self.game_over = True
            self.step_reward -= 100
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

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
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
        }
    
    def _render_game(self):
        self._render_grid()
        self._render_base()
        for tower in self.towers: self._render_tower(tower)
        for alien in self.aliens: self._render_alien(alien)
        for p in self.projectiles: self._render_projectile(p)
        for p in self.particles: self._render_particle(p)
        if not self.game_over:
            self._render_cursor()

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
    def _render_base(self):
        base_rect = pygame.Rect(0, self.BASE_Y_START, self.SCREEN_WIDTH, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.draw.rect(self.screen, self.COLOR_BASE_BORDER, base_rect, 2)
        
    def _render_tower(self, tower):
        radius = int(self.CELL_SIZE * 0.35 * tower.build_progress)
        border_radius = int(self.CELL_SIZE * 0.45 * tower.build_progress)
        pygame.gfxdraw.filled_circle(self.screen, int(tower.x), int(tower.y), border_radius, self.COLOR_TOWER_BORDER)
        pygame.gfxdraw.filled_circle(self.screen, int(tower.x), int(tower.y), radius, self.COLOR_TOWER)

    def _render_alien(self, alien):
        size = 10
        color = self.COLOR_ALIEN_HIT if alien.hit_timer > 0 else self.COLOR_ALIEN
        points = [
            (alien.x, alien.y - size),
            (alien.x - size / 1.5, alien.y + size / 2),
            (alien.x + size / 1.5, alien.y + size / 2)
        ]
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)
        
        # Health bar
        if alien.health < alien.max_health:
            bar_w = self.CELL_SIZE * 0.6
            bar_h = 4
            bar_x = alien.x - bar_w / 2
            bar_y = alien.y - size * 1.8
            health_ratio = alien.health / alien.max_health
            pygame.draw.rect(self.screen, (100, 40, 40), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (200, 80, 80), (bar_x, bar_y, bar_w * health_ratio, bar_h))


    def _render_projectile(self, p):
        end_x = int(p.x - self.vx * 0.5)
        end_y = int(p.y - self.vy * 0.5)
        pygame.draw.aaline(self.screen, self.COLOR_PROJECTILE, (int(p.x), int(p.y)), (end_x, end_y), 3)

    def _render_particle(self, p):
        if p.size > 0:
            pygame.draw.rect(self.screen, p.color, (p.x, p.y, p.size, p.size))

    def _render_cursor(self):
        x, y = self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        # Create a semi-transparent surface
        surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        alpha = 90 + math.sin(self.steps * 0.2) * 30
        
        can_build = self.resources >= self.tower_cost and not any(t.grid_x == self.cursor_pos[0] and t.grid_y == self.cursor_pos[1] for t in self.towers)
        color = (255, 255, 255, alpha) if can_build else (255, 0, 0, alpha)
        
        pygame.draw.rect(surface, color, surface.get_rect(), 0, border_radius=4)
        pygame.draw.rect(surface, (255, 255, 255, alpha+50), surface.get_rect(), 2, border_radius=4)
        self.screen.blit(surface, (x, y))

    def _render_ui(self):
        # Wave and Score
        wave_text = self.font_small.render(f"Wave: {self.wave_number}/10", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        score_text = self.font_small.render(f"Kills: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 30))

        # Resources
        res_text = self.font_small.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        res_rect = res_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(res_text, res_rect)
        
        # Base Health Bar
        bar_width = 200
        bar_height = 20
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = self.SCREEN_HEIGHT - bar_height - 5
        health_ratio = self.base_health / self.max_base_health
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, bar_width * health_ratio, bar_height), border_radius=4)
        health_text = self.font_small.render(f"{self.base_health}/{self.max_base_health}", True, self.COLOR_TEXT)
        health_text_rect = health_text.get_rect(center=(self.SCREEN_WIDTH / 2, bar_y + bar_height / 2))
        self.screen.blit(health_text, health_text_rect)

        # Game Over / Victory Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.victory else "GAME OVER"
            color = (100, 255, 150) if self.victory else (255, 100, 100)
            
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")