
# Generated: 2025-08-27T13:00:20.955670
# Source Brief: brief_00228.md
# Brief Index: 228

        
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
        "Controls: Use arrow keys to build cannons in the four quadrants. Press space to build a central laser tower."
    )

    game_description = (
        "A top-down tower defense game. Place towers strategically to defend your base against waves of geometric enemies."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 3000 # Increased to allow for 20 waves

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PATH = (40, 40, 60)
    COLOR_TOWER_SPOT = (50, 50, 70)
    COLOR_BASE = (0, 255, 150)
    COLOR_BASE_GLOW = (0, 255, 150, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 60)
    COLOR_TOWER_1 = (0, 150, 255) # Cannon
    COLOR_TOWER_2 = (255, 200, 0) # Laser
    COLOR_PROJECTILE_1 = (150, 200, 255)
    COLOR_PROJECTILE_2 = (255, 255, 150)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (50, 200, 50)
    COLOR_HEALTH_BAR_BG = (100, 50, 50)
    COLOR_SCREEN_FLASH = (255, 0, 0, 100)

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.game_path = [pygame.Vector2(-20, 100)] + [pygame.Vector2(p) for p in [(100, 100), (100, 300), (300, 300), (300, 100), (500, 100), (500, 300), (self.WIDTH + 20, 300)]]
        self.tower_spots = [
            pygame.Vector2(100, 200), # TL (for UP action)
            pygame.Vector2(300, 200), # TR (for RIGHT action)
            pygame.Vector2(500, 200), # BL (for LEFT action -> remapped)
            pygame.Vector2(100, 10),  # BR (for DOWN action -> remapped)
            pygame.Vector2(300, 10),  # Center (for SPACE action -> remapped)
        ]
        # Remap actions to be more intuitive
        # Up -> Top-Left
        # Right -> Top-Right
        # Left -> Bottom-Left
        # Down -> Bottom-Right
        # Space -> Center
        self.action_to_tower_spot_map = {
            1: 0, # Up -> spot 0
            4: 1, # Right -> spot 1
            3: 2, # Left -> spot 2
            2: 3, # Down -> spot 3
        }

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = 100
        self.wave_number = 1
        self.enemies_in_wave = 0
        self.enemies_spawned_in_wave = 0
        self.wave_timer = 3 * self.FPS # 3 second delay before first wave
        self.spawn_timer = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.screen_flash_timer = 0
        self._setup_wave()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        step_reward = 0

        if not self.game_over:
            self._handle_actions(action)
            
            # Update timers
            if self.wave_timer > 0:
                self.wave_timer -= 1
            if self.spawn_timer > 0:
                self.spawn_timer -= 1
            if self.screen_flash_timer > 0:
                self.screen_flash_timer -= 1

            # --- Game Logic ---
            self._update_spawning()
            
            # Update Towers
            for tower in self.towers:
                new_projectiles = tower.update(self.enemies)
                if new_projectiles:
                    self.projectiles.extend(new_projectiles)

            # Update and move Projectiles
            for p in self.projectiles[:]:
                p.update()
                if not p.is_alive:
                    self.projectiles.remove(p)

            # Update and move Enemies
            for enemy in self.enemies[:]:
                enemy.update(self.game_path)
                if enemy.reached_end:
                    self.base_health -= enemy.damage
                    self.enemies.remove(enemy)
                    self.screen_flash_timer = 5
                    # sfx: base_damage
                    self._create_particles(self.game_path[-1], 20, self.COLOR_ENEMY)

            # Handle Projectile Hits
            for p in self.projectiles[:]:
                if not p.target or p.target not in self.enemies:
                    p.is_alive = False # Target is already gone
                    continue

                if p.pos.distance_to(p.target.pos) < p.target.radius:
                    hit_enemy = p.target
                    hit_enemy.health -= p.damage
                    p.is_alive = False
                    self._create_particles(hit_enemy.pos, 5, self.COLOR_PROJECTILE_1)
                    # sfx: enemy_hit
                    if hit_enemy.health <= 0:
                        step_reward += 0.1
                        self.score += 10
                        self.enemies.remove(hit_enemy)
                        # sfx: enemy_destroyed
                        self._create_particles(hit_enemy.pos, 30, self.COLOR_ENEMY)
            
            self.projectiles = [p for p in self.projectiles if p.is_alive]

            # Update Particles
            for particle in self.particles[:]:
                particle.update()
                if not particle.is_alive:
                    self.particles.remove(particle)

            # Wave management
            if self.enemies_spawned_in_wave == self.enemies_in_wave and not self.enemies:
                if not self.game_over:
                    step_reward += 1.0
                    self.score += 100
                    self.wave_number += 1
                    if self.wave_number > 20:
                        self.win = True
                        self.game_over = True
                    else:
                        self.wave_timer = 5 * self.FPS # 5 seconds between waves
                        self._setup_wave()
                        # sfx: wave_complete

        # --- Termination and Reward ---
        terminated = self.base_health <= 0 or self.win or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
        
        reward = step_reward
        if self.game_over:
            if self.win:
                reward = 100
            elif self.base_health <= 0:
                reward = -100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        # Place Tower 1 (Cannon)
        if movement in self.action_to_tower_spot_map:
            spot_idx = self.action_to_tower_spot_map[movement]
            pos = self.tower_spots[spot_idx]
            if not any(t.pos == pos for t in self.towers):
                self.towers.append(Tower(pos, 1))
                # sfx: place_tower
        
        # Place Tower 2 (Laser)
        if space_held:
            pos = self.tower_spots[4] # Center spot
            if not any(t.pos == pos for t in self.towers):
                self.towers.append(Tower(pos, 2))
                # sfx: place_tower_alt

    def _setup_wave(self):
        self.enemies_in_wave = 5 + self.wave_number * 2
        self.enemies_spawned_in_wave = 0
        self.spawn_timer = 0
    
    def _update_spawning(self):
        if self.wave_timer == 0 and self.spawn_timer == 0 and self.enemies_spawned_in_wave < self.enemies_in_wave:
            speed = 1.0 + 0.05 * math.floor((self.wave_number - 1) / 2)
            health = 10 + 1 * math.floor((self.wave_number - 1) / 5)
            self.enemies.append(Enemy(self.game_path[0].copy(), health, speed))
            self.enemies_spawned_in_wave += 1
            self.spawn_timer = int(self.FPS * 0.5) # 0.5 sec between spawns

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append(Particle(pos, color))

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
            "base_health": self.base_health,
        }

    def _render_game(self):
        # Draw Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.game_path, 40)
        
        # Draw Tower Spots
        for spot in self.tower_spots:
            pygame.gfxdraw.aacircle(self.screen, int(spot.x), int(spot.y), 15, self.COLOR_TOWER_SPOT)

        # Draw Base
        base_pos = self.game_path[-1]
        pygame.gfxdraw.filled_circle(self.screen, int(base_pos.x), int(base_pos.y), 25, self.COLOR_BASE_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(base_pos.x), int(base_pos.y), 20, self.COLOR_BASE)
        
        # Draw Towers
        for tower in self.towers:
            tower.draw(self.screen)

        # Draw Enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)
        
        # Draw Projectiles
        for p in self.projectiles:
            p.draw(self.screen)

        # Draw Particles
        for particle in self.particles:
            particle.draw(self.screen)
        
        # Screen flash on base hit
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_SCREEN_FLASH)
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # Health Bar
        health_bar_width = 200
        health_ratio = max(0, self.base_health / 100)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, health_bar_width * health_ratio, 20))
        
        health_text = self.font_small.render(f"Base: {int(self.base_health)}/100", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Wave
        wave_text = self.font_small.render(f"Wave: {min(self.wave_number, 20)}/20", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 30))

        # Wave Timer
        if self.wave_timer > 0 and self.wave_number <= 20:
            timer_text_str = f"Wave {self.wave_number} in {math.ceil(self.wave_timer / self.FPS)}"
            timer_text = self.font_large.render(timer_text_str, True, self.COLOR_UI_TEXT)
            self.screen.blit(timer_text, (self.WIDTH/2 - timer_text.get_width()/2, self.HEIGHT/2 - timer_text.get_height()/2))

        # Game Over / Win Text
        if self.game_over:
            if self.win:
                end_text = self.font_large.render("VICTORY", True, self.COLOR_BASE)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))

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

class Enemy:
    def __init__(self, pos, health, speed):
        self.pos = pos
        self.start_health = health
        self.health = health
        self.speed = speed
        self.radius = 10
        self.path_idx = 1
        self.reached_end = False
        self.damage = 10 # Damage dealt to base

    def update(self, path):
        if self.reached_end:
            return

        target_pos = path[self.path_idx]
        direction = (target_pos - self.pos)
        
        if direction.length() < self.speed:
            self.pos = target_pos
            self.path_idx += 1
            if self.path_idx >= len(path):
                self.reached_end = True
        else:
            self.pos += direction.normalize() * self.speed

    def draw(self, screen):
        # Glow
        pygame.gfxdraw.filled_circle(screen, int(self.pos.x), int(self.pos.y), self.radius, GameEnv.COLOR_ENEMY_GLOW)
        # Body
        pygame.gfxdraw.filled_circle(screen, int(self.pos.x), int(self.pos.y), int(self.radius * 0.8), GameEnv.COLOR_ENEMY)
        # Health bar
        if self.health < self.start_health:
            bar_w = self.radius * 1.5
            bar_h = 4
            health_ratio = self.health / self.start_health
            pygame.draw.rect(screen, (50, 0, 0), (self.pos.x - bar_w/2, self.pos.y - self.radius - 8, bar_w, bar_h))
            pygame.draw.rect(screen, (200, 0, 0), (self.pos.x - bar_w/2, self.pos.y - self.radius - 8, bar_w * health_ratio, bar_h))

class Tower:
    def __init__(self, pos, tower_type):
        self.pos = pos
        self.type = tower_type
        self.cooldown = 0
        if self.type == 1: # Cannon
            self.range = 80
            self.fire_rate = 30 # frames per shot
            self.damage = 5
            self.color = GameEnv.COLOR_TOWER_1
            self.projectile_type = 1
        else: # Laser
            self.range = 150
            self.fire_rate = 60
            self.damage = 10
            self.color = GameEnv.COLOR_TOWER_2
            self.projectile_type = 2

    def update(self, enemies):
        if self.cooldown > 0:
            self.cooldown -= 1
            return []

        target = self._find_target(enemies)
        if target:
            self.cooldown = self.fire_rate
            # sfx: tower_shoot
            return [Projectile(self.pos.copy(), target, self.damage, self.projectile_type)]
        return []

    def _find_target(self, enemies):
        # Target enemy closest to the end of the path
        valid_targets = [e for e in enemies if self.pos.distance_to(e.pos) <= self.range]
        if not valid_targets:
            return None
        return max(valid_targets, key=lambda e: e.path_idx + e.pos.distance_to(e.path_idx))

    def draw(self, screen):
        size = 12
        rect = pygame.Rect(self.pos.x - size, self.pos.y - size, size * 2, size * 2)
        pygame.draw.rect(screen, self.color, rect, border_radius=4)
        pygame.draw.rect(screen, tuple(min(255, c + 50) for c in self.color), rect, width=2, border_radius=4)
        if self.cooldown > 0:
            # Draw cooldown indicator
            cooldown_ratio = self.cooldown / self.fire_rate
            pygame.draw.circle(screen, (255,255,255,50), self.pos, size * cooldown_ratio, 2)


class Projectile:
    def __init__(self, pos, target, damage, projectile_type):
        self.pos = pos
        self.target = target
        self.damage = damage
        self.type = projectile_type
        self.is_alive = True
        
        if self.type == 1: # Cannonball
            self.speed = 5
            self.color = GameEnv.COLOR_PROJECTILE_1
        else: # Laser beam
            self.speed = 10
            self.color = GameEnv.COLOR_PROJECTILE_2

    def update(self):
        if not self.target:
            self.is_alive = False
            return
        
        direction = (self.target.pos - self.pos)
        if direction.length() < self.speed:
            self.pos = self.target.pos
        else:
            self.pos += direction.normalize() * self.speed

    def draw(self, screen):
        if self.type == 1:
            pygame.draw.circle(screen, self.color, self.pos, 3)
        else:
            pygame.draw.line(screen, self.color, self.pos, self.pos - pygame.Vector2(self.speed, 0).rotate(self.pos.angle_to(self.target.pos)), 3)

class Particle:
    def __init__(self, pos, color):
        self.pos = pos.copy()
        self.vel = pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
        self.color = color
        self.lifetime = random.randint(10, 20)
        self.is_alive = True

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.is_alive = False

    def draw(self, screen):
        alpha = max(0, int(255 * (self.lifetime / 20)))
        temp_color = (*self.color, alpha)
        temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
        pygame.draw.rect(temp_surf, temp_color, (0, 0, 4, 4))
        screen.blit(temp_surf, self.pos)