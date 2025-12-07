
# Generated: 2025-08-28T06:10:04.788120
# Source Brief: brief_02839.md
# Brief Index: 2839

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper class for Zombies
class Zombie:
    def __init__(self, pos, health, speed, grid_cell_size, np_random):
        self.pos = np.array(pos, dtype=float)
        self.health = health
        self.max_health = health
        self.speed = speed
        self.size = grid_cell_size * 0.6
        self.color = (220, 20, 60) # Crimson
        self.animation_offset = np_random.uniform(0, 2 * math.pi)

    def update(self, base_pos):
        direction = base_pos - self.pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            direction /= dist
        self.pos += direction * self.speed

    def draw(self, surface):
        # Wiggle animation
        anim_x = math.sin(pygame.time.get_ticks() * 0.01 + self.animation_offset) * 2
        anim_y = math.cos(pygame.time.get_ticks() * 0.01 + self.animation_offset) * 2
        
        center_x = self.pos[0] + anim_x
        center_y = self.pos[1] + anim_y
        
        rect = pygame.Rect(center_x - self.size / 2, center_y - self.size / 2, self.size, self.size)
        pygame.draw.rect(surface, self.color, rect, border_radius=3)
        
        # Health bar
        if self.health < self.max_health:
            bar_width = self.size
            bar_height = 5
            health_pct = self.health / self.max_health
            
            bg_rect = pygame.Rect(center_x - bar_width / 2, rect.top - bar_height - 2, bar_width, bar_height)
            pygame.draw.rect(surface, (50, 50, 50), bg_rect)
            
            hp_rect = pygame.Rect(center_x - bar_width / 2, rect.top - bar_height - 2, bar_width * health_pct, bar_height)
            pygame.draw.rect(surface, (0, 255, 0), hp_rect)

# Helper class for Towers
class Tower:
    TOWER_TYPES = {
        0: {"name": "Arrow Tower", "cost": 50, "range": 100, "damage": 10, "fire_rate": 20, "color": (0, 200, 0), "projectile_speed": 8, "projectile_size": 4},
        1: {"name": "Cannon", "cost": 150, "range": 140, "damage": 30, "fire_rate": 70, "color": (100, 100, 200), "projectile_speed": 5, "projectile_size": 8, "splash_radius": 25},
    }
    
    def __init__(self, grid_pos, type_id, cell_size):
        self.grid_pos = grid_pos
        self.type_id = type_id
        self.stats = self.TOWER_TYPES[type_id]
        self.pos = np.array([(grid_pos[0] + 0.5) * cell_size, (grid_pos[1] + 0.5) * cell_size + 40]) # +40 for UI offset
        self.cooldown = 0
        self.target = None

    def update(self, zombies):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None # No new projectile

        # Find a target
        self.target = None
        min_dist = self.stats["range"]
        for zombie in zombies:
            dist = np.linalg.norm(self.pos - zombie.pos)
            if dist < min_dist:
                min_dist = dist
                self.target = zombie

        if self.target and self.cooldown <= 0:
            self.cooldown = self.stats["fire_rate"]
            # SFX: tower_shoot.wav
            return Projectile(self.pos.copy(), self.target, self.stats)
        return None

    def draw(self, surface, cell_size, ui_height):
        rect = pygame.Rect(self.grid_pos[0] * cell_size, ui_height + self.grid_pos[1] * cell_size, cell_size, cell_size)
        pygame.draw.rect(surface, self.stats["color"], rect.inflate(-8, -8), border_radius=5)
        pygame.draw.rect(surface, (255, 255, 255), rect.inflate(-8, -8), 2, border_radius=5)

# Helper class for Projectiles
class Projectile:
    def __init__(self, pos, target, tower_stats):
        self.pos = pos
        self.target = target
        self.stats = tower_stats
        self.speed = tower_stats["projectile_speed"]
        self.damage = tower_stats["damage"]
        self.size = tower_stats["projectile_size"]
        self.color = (135, 206, 250) # Sky Blue
        self.is_cannonball = "splash_radius" in tower_stats

        direction = self.target.pos - self.pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            self.velocity = (direction / dist) * self.speed
        else:
            self.velocity = np.array([0, -self.speed])

    def update(self):
        self.pos += self.velocity

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.pos.astype(int), self.size)
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), self.size, (255, 255, 255))

# Helper class for Particles
class Particle:
    def __init__(self, pos, color, life, size, velocity_spread, np_random):
        self.pos = pos.copy()
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(0.5, velocity_spread)
        self.velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed])

    def update(self):
        self.pos += self.velocity
        self.life -= 1

    def draw(self, surface):
        alpha = int(255 * (self.life / self.max_life))
        temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*self.color, alpha), (self.size, self.size), self.size)
        surface.blit(temp_surf, self.pos - self.size)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Arrow keys to move cursor. Space to place selected tower. Shift to cycle tower types."
    game_description = "Defend your base from waves of zombies in this real-time strategy survival game."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    UI_HEIGHT = 40
    GRID_WIDTH, GRID_HEIGHT = 16, 9
    CELL_SIZE = 40
    MAX_STEPS = 4000
    WIN_WAVE = 20

    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_BASE = (0, 100, 150)
    COLOR_CURSOR = (255, 255, 0, 100)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        self.base_grid_col = self.GRID_WIDTH - 1
        self.base_center_pos = np.array([(self.base_grid_col + 0.5) * self.CELL_SIZE, (self.SCREEN_HEIGHT - self.UI_HEIGHT) / 2 + self.UI_HEIGHT])
        
        # Initialize state variables
        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = 100
        self.max_base_health = 100
        self.resources = 150
        
        self.current_wave = 0
        self.wave_timer = 150
        self.zombies_to_spawn_this_wave = []
        self.spawn_cooldown = 0
        
        self.zombies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01
        
        self._handle_input(action)
        
        if not self.game_over:
            self._update_wave_system()
            self._update_towers()
            self._update_projectiles()
            reward += self._update_zombies_and_collisions()
            self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.base_health <= 0:
                reward -= 100
            elif self.current_wave > self.WIN_WAVE:
                reward += 100
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 2)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)
        
        if space_held and not self.prev_space_held:
            tower_stats = Tower.TOWER_TYPES[self.selected_tower_type]
            can_afford = self.resources >= tower_stats["cost"]
            is_occupied = any(t.grid_pos == self.cursor_pos for t in self.towers)
            
            if can_afford and not is_occupied:
                self.resources -= tower_stats["cost"]
                self.towers.append(Tower(self.cursor_pos.copy(), self.selected_tower_type, self.CELL_SIZE))
                # SFX: place_tower.wav
        
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(Tower.TOWER_TYPES)
            # SFX: cycle.wav

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_wave_system(self):
        if not self.zombies and not self.zombies_to_spawn_this_wave:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                if self.current_wave > self.WIN_WAVE:
                    return
                self._prepare_next_wave()
                self.wave_timer = 200

        if self.zombies_to_spawn_this_wave:
            self.spawn_cooldown -= 1
            if self.spawn_cooldown <= 0:
                zombie_data = self.zombies_to_spawn_this_wave.pop(0)
                self.zombies.append(Zombie(**zombie_data))
                self.spawn_cooldown = 15

    def _prepare_next_wave(self):
        num_zombies = 5 + (self.current_wave - 1) * 2
        speed = 0.6 + 0.05 * ((self.current_wave - 1) // 5)
        health = 50 + 20 * (self.current_wave - 1)
        
        for _ in range(num_zombies):
            spawn_y = self.np_random.uniform(self.UI_HEIGHT, self.SCREEN_HEIGHT)
            self.zombies_to_spawn_this_wave.append({
                "pos": [-20, spawn_y],
                "health": health,
                "speed": speed,
                "grid_cell_size": self.CELL_SIZE,
                "np_random": self.np_random
            })
            
    def _update_towers(self):
        for tower in self.towers:
            new_projectile = tower.update(self.zombies)
            if new_projectile:
                self.projectiles.append(new_projectile)
                self._create_particles(tower.pos, (255, 255, 100), 5, 10, 2, 2) # Muzzle flash

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p.update()
            if not (0 < p.pos[0] < self.SCREEN_WIDTH and 0 < p.pos[1] < self.SCREEN_HEIGHT):
                self.projectiles.remove(p)

    def _update_zombies_and_collisions(self):
        reward = 0
        base_rect = pygame.Rect(self.base_grid_col * self.CELL_SIZE, self.UI_HEIGHT, self.CELL_SIZE, self.SCREEN_HEIGHT - self.UI_HEIGHT)
        
        for z in self.zombies[:]:
            z.update(self.base_center_pos)
            zombie_rect = pygame.Rect(z.pos[0] - z.size/2, z.pos[1] - z.size/2, z.size, z.size)

            if zombie_rect.colliderect(base_rect):
                self.base_health = max(0, self.base_health - 1)
                reward -= 10
                self.zombies.remove(z)
                # SFX: base_hit.wav
                self._create_particles(z.pos, self.COLOR_BASE, 20, 15, 3, 3)
                continue
            
            for p in self.projectiles[:]:
                if np.linalg.norm(z.pos - p.pos) < (z.size + p.size) / 2:
                    reward += 0.1
                    z.health -= p.damage
                    # SFX: zombie_hit.wav
                    self._create_particles(p.pos, (255, 165, 0), 10, 10, 2, 2.5)

                    if p.is_cannonball:
                        for other_z in self.zombies:
                            if other_z is not z and np.linalg.norm(p.pos - other_z.pos) < p.stats["splash_radius"]:
                                other_z.health -= p.damage * 0.5
                                reward += 0.05
                        self._create_particles(p.pos, (255, 69, 0), 30, 20, 4, 4)
                        # SFX: explosion.wav
                    
                    if p in self.projectiles:
                        self.projectiles.remove(p)
                    
                    if z.health <= 0:
                        if z in self.zombies:
                            self.zombies.remove(z)
                        reward += 1
                        self.resources += 10
                        # SFX: zombie_die.wav
                        self._create_particles(z.pos, z.color, 25, 20, 2, 1.5)
                        break
        return reward
        
    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, life, size, velocity_spread):
        for _ in range(count):
            self.particles.append(Particle(pos, color, life, size, velocity_spread, self.np_random))

    def _check_termination(self):
        if self.base_health <= 0 or self.current_wave > self.WIN_WAVE or self.steps >= self.MAX_STEPS:
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
        return {"score": self.score, "steps": self.steps}
    
    def _render_game(self):
        for x in range(self.GRID_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.CELL_SIZE, self.UI_HEIGHT), (x * self.CELL_SIZE, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.UI_HEIGHT + y * self.CELL_SIZE), (self.SCREEN_WIDTH, self.UI_HEIGHT + y * self.CELL_SIZE))

        base_rect = pygame.Rect(self.base_grid_col * self.CELL_SIZE, self.UI_HEIGHT, self.CELL_SIZE, self.SCREEN_HEIGHT - self.UI_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.draw.rect(self.screen, (200, 200, 255), base_rect, 3)

        for tower in self.towers:
            tower.draw(self.screen, self.CELL_SIZE, self.UI_HEIGHT)
        for p in self.projectiles:
            p.draw(self.screen)
        for z in self.zombies:
            z.draw(self.screen)
        for p in self.particles:
            p.draw(self.screen)
            
        if not self.game_over:
            cursor_rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.UI_HEIGHT + self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            tower_stats = Tower.TOWER_TYPES[self.selected_tower_type]
            range_center = cursor_rect.center
            
            range_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(range_surface, range_center[0], range_center[1], tower_stats["range"], (255, 255, 255, 50))
            pygame.gfxdraw.filled_circle(range_surface, range_center[0], range_center[1], tower_stats["range"], (255, 255, 255, 30))
            self.screen.blit(range_surface, (0,0))
            
            cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            cursor_surface.fill(self.COLOR_CURSOR)
            self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_ui(self):
        pygame.draw.rect(self.screen, (10, 10, 15), (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        
        health_text = self.font_small.render(f"Base HP: {self.base_health}/{self.max_base_health}", True, (255, 255, 255))
        self.screen.blit(health_text, (10, 12))
        
        resource_text = self.font_small.render(f"Resources: {self.resources}", True, (255, 255, 255))
        self.screen.blit(resource_text, (180, 12))
        
        wave_text = self.font_small.render(f"Wave: {self.current_wave}", True, (255, 255, 255))
        self.screen.blit(wave_text, (320, 12))
        
        tower_stats = Tower.TOWER_TYPES[self.selected_tower_type]
        tower_text = self.font_small.render(f"Select: {tower_stats['name']} ({tower_stats['cost']})", True, (255, 255, 255))
        self.screen.blit(tower_text, (420, 12))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.current_wave > self.WIN_WAVE else "GAME OVER"
            text = self.font_large.render(msg, True, (255, 50, 50))
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    action = [0, 0, 0]

    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Zombie Defense")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_q:
                    running = False

        if not terminated:
            keys = pygame.key.get_pressed()
            action = [0, 0, 0]
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        env.clock.tick(30)

    env.close()