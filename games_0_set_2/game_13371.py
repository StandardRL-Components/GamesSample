import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game entities
class Cell:
    CELL_STATS = {
        1: {"range": 100, "damage": 5, "fire_rate": 10, "cost": 10, "proj_speed": 8, "is_aoe": False},
        2: {"range": 150, "damage": 25, "fire_rate": 45, "cost": 25, "proj_speed": 10, "is_aoe": False},
        3: {"range": 80, "damage": 15, "fire_rate": 60, "cost": 50, "proj_speed": 6, "is_aoe": True, "aoe_radius": 40},
    }

    def __init__(self, pos, cell_type, aoe_callback):
        self.pos = pos
        self.type = cell_type
        self.level = 1
        self.cooldown = 0
        self.target = None
        self.pulse_timer = random.uniform(0, 2 * math.pi)
        self.aoe_callback = aoe_callback
        self.set_stats()

    def set_stats(self):
        stats = self.CELL_STATS[self.type]
        self.range = stats["range"] * (1 + 0.1 * (self.level - 1))
        self.damage = stats["damage"] * (1 + 0.2 * (self.level - 1))
        self.fire_rate = stats["fire_rate"]
        self.proj_speed = stats["proj_speed"]
        self.is_aoe = stats["is_aoe"]
        if self.is_aoe:
            self.aoe_radius = stats["aoe_radius"] * (1 + 0.1 * (self.level - 1))

    def upgrade(self):
        if self.level < 5:
            self.level += 1
            self.set_stats()
            return True
        return False

    def update(self, enemies):
        self.pulse_timer = (self.pulse_timer + 0.1) % (2 * math.pi)
        if self.cooldown > 0:
            self.cooldown -= 1
        
        # Find new target if needed
        if self.target is None or self.target.health <= 0 or math.dist(self.pos, self.target.pos) > self.range:
            self.target = self.find_target(enemies)
            
        # Fire if ready and has target
        if self.cooldown == 0 and self.target is not None:
            self.cooldown = self.fire_rate
            return Projectile(self.pos, self.target, self.damage, self.proj_speed, self.is_aoe, self.aoe_radius if self.is_aoe else 0, self.aoe_callback)
        return None

    def find_target(self, enemies):
        valid_targets = [e for e in enemies if math.dist(self.pos, e.pos) <= self.range]
        if not valid_targets:
            return None
        # Target enemy that is furthest along its path
        return max(valid_targets, key=lambda e: e.distance_traveled)

class Enemy:
    def __init__(self, enemy_type, path, wave_bonus):
        self.path = path
        self.pos = list(path[0])
        self.waypoint_index = 1
        self.distance_traveled = 0

        base_stats = {
            'grunt': {'health': 50, 'speed': 1.0, 'biomass': 2},
            'runner': {'health': 30, 'speed': 2.0, 'biomass': 3},
            'tank': {'health': 200, 'speed': 0.7, 'biomass': 5},
        }
        stats = base_stats[enemy_type]
        self.type = enemy_type
        self.max_health = stats['health'] * (1 + wave_bonus)
        self.health = self.max_health
        self.speed = stats['speed'] * (1 + wave_bonus)
        self.biomass_reward = stats['biomass']

    def update(self):
        if self.waypoint_index >= len(self.path):
            return True # Reached center

        target_pos = self.path[self.waypoint_index]
        dist = math.dist(self.pos, target_pos)
        
        if dist < self.speed:
            self.pos = list(target_pos)
            self.waypoint_index += 1
            self.distance_traveled += dist
        else:
            angle = math.atan2(target_pos[1] - self.pos[1], target_pos[0] - self.pos[0])
            self.pos[0] += math.cos(angle) * self.speed
            self.pos[1] += math.sin(angle) * self.speed
            self.distance_traveled += self.speed
        
        return False

class Projectile:
    def __init__(self, start_pos, target, damage, speed, is_aoe=False, aoe_radius=0, aoe_callback=None):
        self.pos = list(start_pos)
        self.target = target
        self.damage = damage
        self.speed = speed
        self.is_aoe = is_aoe
        self.aoe_radius = aoe_radius
        self.aoe_callback = aoe_callback

    def update(self, enemies):
        if self.target.health <= 0:
            return True, None # Hit (target already dead)
        
        target_pos = self.target.pos
        dist = math.dist(self.pos, target_pos)

        if dist < self.speed:
            # Hit
            self.target.health -= self.damage
            if self.is_aoe:
                self.aoe_callback(self.target.pos, self.aoe_radius, self.damage * 0.5, self.target)
            return True, self.target
        else:
            angle = math.atan2(target_pos[1] - self.pos[1], target_pos[0] - self.pos[0])
            self.pos[0] += math.cos(angle) * self.speed
            self.pos[1] += math.sin(angle) * self.speed
            return False, None

class Particle:
    def __init__(self, pos, vel, lifespan, color, size_range):
        self.pos = list(pos)
        self.vel = list(vel)
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color
        self.size = random.uniform(size_range[0], size_range[1])

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[0] *= 0.98 # friction
        self.vel[1] *= 0.98
        self.lifespan -= 1
        return self.lifespan <= 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A microscopic-themed tower defense game. Place and upgrade defensive cells to protect the central core from waves of enemies."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to cycle through and place cells. Press shift to upgrade a cell."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    PLAY_RADIUS = 180
    MAX_STEPS = 5000
    TOTAL_WAVES = 20
    
    # Colors
    COLOR_BG = (10, 0, 20)
    COLOR_GRID = (20, 10, 40)
    COLOR_PLAY_AREA = (100, 80, 200)
    COLOR_TEXT = (220, 220, 255)
    COLOR_HEALTH_BAR = (40, 200, 80)
    COLOR_HEALTH_BAR_BG = (200, 40, 80)
    COLOR_CURSOR = (255, 255, 255, 100)
    COLOR_CURSOR_INVALID = (255, 50, 50, 100)
    
    # Cell Type Colors
    CELL_COLORS = {
        1: {'base': (0, 255, 150), 'glow': (150, 255, 200)},
        2: {'base': (0, 150, 255), 'glow': (150, 200, 255)},
        3: {'base': (255, 150, 0), 'glow': (255, 200, 150)},
    }
    
    # Enemy Type Colors
    ENEMY_COLORS = {
        'grunt': {'base': (255, 50, 50), 'health': (255, 150, 150)},
        'runner': {'base': (255, 100, 200), 'health': (255, 180, 230)},
        'tank': {'base': (200, 0, 0), 'health': (255, 100, 100)},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        self.render_mode = render_mode
        self._initialize_paths_and_waves()
        
    def _initialize_state(self):
        # This is separated from reset to avoid re-calculating paths etc.
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 100
        self.biomass = 40
        self.current_wave = 0
        self.wave_cooldown = 200 # Time between waves
        self.enemies_in_wave = []
        self.enemies_spawned_in_wave = 0
        self.spawn_cooldown = 0
        self.cells = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = list(self.CENTER)
        self.cursor_speed = 8
        self.selected_cell_type = 1
        self.prev_space_held = False
        self.prev_shift_held = False
        self.step_reward = 0
        self.unlocked_cells = {1}
        self.game_won = False

    def _initialize_paths_and_waves(self):
        self.paths = []
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            start_x = self.CENTER[0] + (self.SCREEN_WIDTH/2) * math.cos(angle)
            start_y = self.CENTER[1] + (self.SCREEN_HEIGHT/2) * math.sin(angle)
            self.paths.append([(start_x, start_y), self.CENTER])

        self.wave_definitions = []
        for i in range(self.TOTAL_WAVES):
            wave = []
            num_grunts = 5 + i * 2
            wave.extend([('grunt', random.choice(self.paths)) for _ in range(num_grunts)])
            if i >= 3:
                num_runners = 2 + (i - 3)
                wave.extend([('runner', random.choice(self.paths)) for _ in range(num_runners)])
            if i >= 6:
                num_tanks = 1 + (i - 6) // 2
                wave.extend([('tank', random.choice(self.paths)) for _ in range(num_tanks)])
            random.shuffle(wave)
            self.wave_definitions.append(wave)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.step_reward = 0
        self.game_over = self._check_termination()
        if self.game_over:
            return self._get_observation(), self.step_reward, True, False, self._get_info()

        self._handle_input(action)
        self._update_game_state()
        
        self.steps += 1
        reward = self.step_reward
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if self.game_won:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos[1] -= self.cursor_speed
        if movement == 2: self.cursor_pos[1] += self.cursor_speed
        if movement == 3: self.cursor_pos[0] -= self.cursor_speed
        if movement == 4: self.cursor_pos[0] += self.cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # Place cell on space press
        if space_held and not self.prev_space_held:
            self._place_cell()

        # Upgrade cell on shift press
        if shift_held and not self.prev_shift_held:
            self._upgrade_cell()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _place_cell(self):
        # Cycle through available cell types
        available = sorted(list(self.unlocked_cells))
        current_idx = available.index(self.selected_cell_type)
        next_idx = (current_idx + 1) % len(available)
        self.selected_cell_type = available[next_idx]

        # Check placement validity
        if math.dist(self.cursor_pos, self.CENTER) > self.PLAY_RADIUS: return
        for cell in self.cells:
            if math.dist(self.cursor_pos, cell.pos) < 20: return

        # Check cost
        cost = Cell.CELL_STATS[self.selected_cell_type]["cost"]
        if self.biomass >= cost:
            self.biomass -= cost
            new_cell = Cell(tuple(self.cursor_pos), self.selected_cell_type, self._handle_aoe)
            self.cells.append(new_cell)
            self._create_particles(new_cell.pos, 15, (0, 255, 150), (1, 3))

    def _upgrade_cell(self):
        cell_to_upgrade = None
        for cell in self.cells:
            if math.dist(self.cursor_pos, cell.pos) < 20:
                cell_to_upgrade = cell
                break
        
        if cell_to_upgrade:
            cost = 15 * cell_to_upgrade.level
            if self.biomass >= cost and cell_to_upgrade.upgrade():
                self.biomass -= cost
                self._create_particles(cell_to_upgrade.pos, 20, (255, 255, 0), (2, 4))

    def _update_game_state(self):
        self._update_waves()
        self._update_enemies()
        self._update_cells()
        self._update_projectiles()
        self._update_particles()
        self.score += self.step_reward

    def _update_waves(self):
        if self.current_wave >= self.TOTAL_WAVES:
            if not self.enemies and not self.enemies_in_wave:
                self.game_won = True
            return

        if not self.enemies and not self.enemies_in_wave: # Wave cleared
            if self.wave_cooldown > 0:
                self.wave_cooldown -= 1
            else: # Start next wave
                self.current_wave += 1
                self.step_reward += 10 # Wave clear bonus
                self.wave_cooldown = 200
                self.enemies_in_wave = list(self.wave_definitions[self.current_wave - 1])
                self.enemies_spawned_in_wave = 0
                if self.current_wave in [5, 10, 15]:
                    self.unlocked_cells.add(self.current_wave // 5 + 1)
        
        if self.enemies_in_wave:
            if self.spawn_cooldown > 0:
                self.spawn_cooldown -= 1
            else:
                enemy_type, path = self.enemies_in_wave.pop(0)
                wave_bonus = 0.05 * (self.current_wave - 1)
                self.enemies.append(Enemy(enemy_type, path, wave_bonus))
                self.spawn_cooldown = max(10, 30 - self.current_wave)

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            if enemy.update():
                self.enemies.remove(enemy)
                self.player_health -= 10
                self.step_reward -= 0.1
                self._create_particles(self.CENTER, 30, (255, 0, 0), (3, 6))

    def _update_cells(self):
        for cell in self.cells:
            projectile = cell.update(self.enemies)
            if projectile:
                self.projectiles.append(projectile)

    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            hit, hit_enemy = proj.update(self.enemies)
            if hit:
                self.projectiles.remove(proj)
                if hit_enemy:
                    self.step_reward += 0.1 # Hit bonus
                    self._create_particles(proj.pos, 5, (150, 200, 255), (1, 2))
                    if hit_enemy.health <= 0:
                        self._on_enemy_killed(hit_enemy)

    def _update_particles(self):
        for p in reversed(self.particles):
            if p.update():
                self.particles.remove(p)

    def _on_enemy_killed(self, enemy):
        if enemy in self.enemies:
            self.enemies.remove(enemy)
            self.step_reward += 1.0 # Kill bonus
            self.biomass += enemy.biomass_reward
            self.score += 1
            self._create_particles(enemy.pos, 25, self.ENEMY_COLORS[enemy.type]['base'], (2, 5))

    def _handle_aoe(self, center, radius, damage, primary_target):
        hits = 0
        for enemy in self.enemies:
            if enemy is not primary_target and math.dist(center, enemy.pos) <= radius:
                enemy.health -= damage
                self.step_reward += 0.1 # AOE Hit bonus
                if enemy.health <= 0:
                    self._on_enemy_killed(enemy)
                hits += 1
        if hits > 1:
            self.step_reward += 5 # Combo bonus

    def _create_particles(self, pos, count, color, size_range):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = random.randint(20, 40)
            self.particles.append(Particle(pos, vel, lifespan, color, size_range))

    def _check_termination(self):
        return self.player_health <= 0 or self.steps >= self.MAX_STEPS or self.game_won

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Draw play area
        pygame.gfxdraw.aacircle(self.screen, self.CENTER[0], self.CENTER[1], self.PLAY_RADIUS, self.COLOR_PLAY_AREA)
        pygame.gfxdraw.aacircle(self.screen, self.CENTER[0], self.CENTER[1], self.PLAY_RADIUS+1, self.COLOR_PLAY_AREA)

        # Draw enemies
        for enemy in self.enemies:
            color_set = self.ENEMY_COLORS[enemy.type]
            pos_int = (int(enemy.pos[0]), int(enemy.pos[1]))
            size = int(8 + (enemy.health / enemy.max_health) * 6)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], size, color_set['base'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], size, color_set['base'])
            # Health bar
            health_ratio = enemy.health / enemy.max_health
            bar_width = 20
            bar_height = 4
            bar_x = pos_int[0] - bar_width // 2
            bar_y = pos_int[1] - size - 8
            pygame.draw.rect(self.screen, (100,0,0), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, color_set['health'], (bar_x, bar_y, int(bar_width * health_ratio), bar_height))

        # Draw cells
        for cell in self.cells:
            pos_int = (int(cell.pos[0]), int(cell.pos[1]))
            color_set = self.CELL_COLORS[cell.type]
            pulse = (math.sin(cell.pulse_timer) + 1) / 2
            glow_size = int(12 + pulse * 4 + cell.level)
            glow_color = (*color_set['glow'], int(50 + pulse * 50))
            
            s = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (glow_size, glow_size), glow_size)
            self.screen.blit(s, (pos_int[0] - glow_size, pos_int[1] - glow_size), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 10, color_set['base'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 10, color_set['base'])
            if cell.level > 1:
                level_text = self.font_small.render(str(cell.level), True, (0,0,0))
                self.screen.blit(level_text, (pos_int[0] - 4, pos_int[1] - 8))

        # Draw projectiles
        for proj in self.projectiles:
            pos_int = (int(proj.pos[0]), int(proj.pos[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 3, (150, 200, 255))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 3, (200, 220, 255))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p.lifespan / p.max_lifespan))
            color = (*p.color, alpha)
            s = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p.size, p.size), p.size)
            self.screen.blit(s, (int(p.pos[0] - p.size), int(p.pos[1] - p.size)), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw cursor
        cursor_color = self.COLOR_CURSOR
        if math.dist(self.cursor_pos, self.CENTER) > self.PLAY_RADIUS:
            cursor_color = self.COLOR_CURSOR_INVALID
        else:
            for cell in self.cells:
                if math.dist(self.cursor_pos, cell.pos) < 20:
                    cursor_color = self.COLOR_CURSOR_INVALID
                    break
        
        s = pygame.Surface((60, 60), pygame.SRCALPHA)
        pygame.gfxdraw.aacircle(s, 30, 30, 15, cursor_color)
        self.screen.blit(s, (int(self.cursor_pos[0]-30), int(self.cursor_pos[1]-30)))
        pygame.draw.line(self.screen, cursor_color, (int(self.cursor_pos[0])-5, int(self.cursor_pos[1])), (int(self.cursor_pos[0])+5, int(self.cursor_pos[1])))
        pygame.draw.line(self.screen, cursor_color, (int(self.cursor_pos[0]), int(self.cursor_pos[1])-5), (int(self.cursor_pos[0]), int(self.cursor_pos[1])+5))

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / 100)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(200 * health_ratio), 20))
        health_text = self.font_small.render(f"Base Integrity: {int(self.player_health)}%", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Biomass
        biomass_text = self.font_small.render(f"Biomass: {int(self.biomass)}", True, self.COLOR_TEXT)
        self.screen.blit(biomass_text, (10, 40))
        
        # Wave Info
        wave_str = f"Wave: {self.current_wave}/{self.TOTAL_WAVES}"
        if not self.enemies and not self.enemies_in_wave and self.current_wave < self.TOTAL_WAVES:
            wave_str += f" (Next in {self.wave_cooldown//30}s)"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Score
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 30))

        # Selected Cell
        selected_text = self.font_small.render("Selected Cell:", True, self.COLOR_TEXT)
        self.screen.blit(selected_text, (10, self.SCREEN_HEIGHT - 45))
        
        for i, cell_type in enumerate(sorted(list(self.unlocked_cells))):
            color_set = self.CELL_COLORS[cell_type]
            pos = (130 + i * 40, self.SCREEN_HEIGHT - 30)
            is_selected = cell_type == self.selected_cell_type
            
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, color_set['base'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, color_set['base'])
            if is_selected:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, (255, 255, 255))
        
        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY" if self.game_won else "BASE DESTROYED"
            color = (100, 255, 150) if self.game_won else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=self.CENTER)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "health": self.player_health,
            "biomass": self.biomass,
            "cells": len(self.cells),
            "enemies": len(self.enemies)
        }
        
    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # The main loop is for manual play and visual debugging.
    # It will not run in the evaluation environment.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS", etc.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Microscopic Tower Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Survived to Wave: {info['wave']}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    env.close()