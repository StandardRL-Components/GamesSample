# Generated: 2025-08-28T02:36:33.099127
# Source Brief: brief_01753.md
# Brief Index: 1753

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame



# --- Helper Classes for Game Entities ---

class FloatingText:
    """Represents a piece of text that floats up and fades out."""
    def __init__(self, x, y, text, color, font, lifetime=30):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.font = font
        self.lifetime = lifetime
        self.alpha = 255

    def update(self):
        self.y -= 1
        self.lifetime -= 1
        if self.lifetime < 20:
            self.alpha = max(0, self.alpha - 15)

    def draw(self, surface, offset):
        if self.lifetime > 0:
            text_surf = self.font.render(self.text, True, self.color)
            text_surf.set_alpha(self.alpha)
            surface.blit(text_surf, (self.x - text_surf.get_width() // 2 + offset[0], self.y - text_surf.get_height() // 2 + offset[1]))

class Particle:
    """A simple particle for effects like explosions or muzzle flashes."""
    def __init__(self, x, y, color, size, lifetime, velocity):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.velocity = velocity

    def update(self):
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface, offset):
        if self.lifetime > 0 and self.size > 0:
            pygame.draw.circle(surface, self.color, (int(self.x + offset[0]), int(self.y + offset[1])), int(self.size))

class Projectile:
    """A projectile fired by a unit."""
    def __init__(self, x, y, target, damage):
        self.x = x
        self.y = y
        self.target = target
        self.damage = damage
        self.speed = 8
        self.lifetime = 100 # Failsafe

    def update(self):
        self.lifetime -= 1
        if self.target.health <= 0:
            return True # Target is dead, projectile should be removed

        dx = self.target.x - self.x
        dy = self.target.y - self.y
        dist = math.hypot(dx, dy)
        if dist < self.speed:
            self.target.health -= self.damage
            return True # Hit target

        self.x += (dx / dist) * self.speed
        self.y += (dy / dist) * self.speed
        return False

    def draw(self, surface, offset):
        pygame.draw.circle(surface, (200, 255, 255), (int(self.x + offset[0]), int(self.y + offset[1])), 3)
        pygame.gfxdraw.aacircle(surface, int(self.x + offset[0]), int(self.y + offset[1]), 3, (200, 255, 255))


class GameEntity:
    """Base class for all interactive game objects."""
    def __init__(self, x, y, size, health):
        self.x = x
        self.y = y
        self.size = size
        self.max_health = health
        self.health = health
        self.bob = 0
        self.bob_speed = 0.1

    def get_draw_pos(self, offset):
        iso_x = (self.x - self.y) * 16 + offset[0]
        iso_y = (self.x + self.y) * 8 + offset[1] - self.bob
        return int(iso_x), int(iso_y)

    def draw(self, surface, offset):
        """Draws a default representation for a GameEntity, used for resource nodes."""
        pos = self.get_draw_pos(offset)
        # Shadow
        pygame.gfxdraw.filled_ellipse(surface, pos[0], pos[1] + self.size // 4, self.size, self.size // 2, (0, 0, 0, 80))
        # Body (like a crystal)
        points = [
            (pos[0], pos[1] - self.size),
            (pos[0] + self.size // 2, pos[1] - self.size // 2),
            (pos[0], pos[1]),
            (pos[0] - self.size // 2, pos[1] - self.size // 2),
        ]
        pygame.draw.polygon(surface, (100, 220, 100), points)
        pygame.draw.aalines(surface, (180, 255, 180), True, points)

    def draw_health_bar(self, surface, pos):
        if self.health < self.max_health:
            bar_width = 30
            bar_height = 4
            x, y = pos[0] - bar_width // 2, pos[1] - self.size * 1.5 - 10
            
            health_ratio = self.health / self.max_health
            pygame.draw.rect(surface, (80, 0, 0), (x, y, bar_width, bar_height))
            pygame.draw.rect(surface, (0, 200, 0), (x, y, int(bar_width * health_ratio), bar_height))

class Unit(GameEntity):
    def __init__(self, x, y):
        super().__init__(x, y, 10, 100)
        self.attack_range = 8
        self.attack_cooldown = 0
        self.attack_speed = 20 # frames between attacks
        self.damage = 25

    def update(self, zombies, projectiles):
        self.bob = math.sin(pygame.time.get_ticks() * self.bob_speed + self.x) * 2
        self.attack_cooldown = max(0, self.attack_cooldown - 1)
        
        if self.attack_cooldown == 0:
            closest_zombie = None
            min_dist = float('inf')
            for zombie in zombies:
                dist = math.hypot(self.x - zombie.x, self.y - zombie.y)
                if dist < self.attack_range and dist < min_dist:
                    min_dist = dist
                    closest_zombie = zombie
            
            if closest_zombie:
                # Fire projectile
                projectiles.append(Projectile(self.x, self.y, closest_zombie, self.damage))
                self.attack_cooldown = self.attack_speed
                # Sound effect placeholder: # sfx_unit_fire.wav

    def draw(self, surface, offset):
        pos = self.get_draw_pos(offset)
        # Shadow
        pygame.gfxdraw.filled_ellipse(surface, pos[0], pos[1] + self.size // 2, self.size, self.size // 2, (0, 0, 0, 100))
        # Body
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1] - self.size // 2, self.size // 2, (50, 150, 255))
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1] - self.size // 2, self.size // 2, (150, 220, 255))
        self.draw_health_bar(surface, pos)

class Zombie(GameEntity):
    def __init__(self, x, y, speed, health):
        super().__init__(x, y, 9, health)
        self.speed = speed
        self.attack_cooldown = 0
        self.attack_speed = 30
        self.damage = 10
        self.bob_speed = 0.15

    def update(self, targets):
        self.bob = math.sin(pygame.time.get_ticks() * self.bob_speed + self.x) * 3
        self.attack_cooldown = max(0, self.attack_cooldown - 1)
        
        closest_target = None
        min_dist = float('inf')
        for target in targets:
            dist = math.hypot(self.x - target.x, self.y - target.y)
            if dist < min_dist:
                min_dist = dist
                closest_target = target

        if closest_target:
            if min_dist > 0.8: # Move towards target
                dx = closest_target.x - self.x
                dy = closest_target.y - self.y
                self.x += (dx / min_dist) * self.speed
                self.y += (dy / min_dist) * self.speed
            elif self.attack_cooldown == 0: # Attack target
                closest_target.health -= self.damage
                self.attack_cooldown = self.attack_speed
                # Sound effect placeholder: # sfx_zombie_attack.wav
    
    def draw(self, surface, offset):
        pos = self.get_draw_pos(offset)
        # Shadow
        pygame.gfxdraw.filled_ellipse(surface, pos[0], pos[1] + self.size // 2, self.size, self.size // 2, (0, 0, 0, 100))
        # Body
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1] - self.size // 2, self.size, (200, 50, 50))
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1] - self.size // 2, self.size, (255, 100, 100))
        self.draw_health_bar(surface, pos)


class Structure(GameEntity):
    def __init__(self, x, y):
        super().__init__(x, y, 15, 500)

    def draw(self, surface, offset):
        pos = self.get_draw_pos(offset)
        # Shadow
        pygame.gfxdraw.filled_ellipse(surface, pos[0], pos[1] + self.size // 2, self.size, self.size // 2, (0, 0, 0, 100))
        # Body
        rect = pygame.Rect(pos[0] - self.size, pos[1] - self.size, self.size * 2, self.size * 2)
        pygame.draw.rect(surface, (100, 100, 110), rect)
        pygame.draw.rect(surface, (150, 150, 160), rect, 2)
        self.draw_health_bar(surface, pos)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Arrows to move build cursor. Space to build a Unit. Shift to build a Wall."
    game_description = "Survive waves of zombies by building units and walls. Gather resources from nodes to fund your defense."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("monospace", 18, bold=True)
        self.floating_text_font = pygame.font.SysFont("monospace", 14, bold=True)
        
        # --- Game Constants ---
        self.MAX_STEPS = 6000
        self.GRID_WIDTH, self.GRID_HEIGHT = 22, 22
        self.ISO_OFFSET = (self.width // 2, 100)
        self.UNIT_COST = 25
        self.STRUCTURE_COST = 15
        self.MAX_UNITS = 20
        self.MAX_RESOURCES = 999

        # --- Colors ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        
        # --- State variables ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.resources = 0
        self.wave_level = 0
        self.zombie_spawn_timer = 0
        self.resource_timer = 0
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.base = None
        self.resource_nodes = []
        self.units = []
        self.zombies = []
        self.structures = []
        self.projectiles = []
        self.particles = []
        self.floating_texts = []

        # self.reset() # No need to call reset in init, it will be called by the runner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.resources = 50
        self.wave_level = 0
        self.zombie_spawn_timer = 0
        self.resource_timer = 0
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 4]
        
        self.base = Structure(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.base.health = 10000 # Effectively invincible
        
        self.resource_nodes = [
            GameEntity(5, 5, 12, 9999), 
            GameEntity(self.GRID_WIDTH - 5, 5, 12, 9999),
            GameEntity(5, self.GRID_HEIGHT - 5, 12, 9999),
            GameEntity(self.GRID_WIDTH - 5, self.GRID_HEIGHT - 5, 12, 9999)
        ]
        
        self.units = [Unit(self.base.x + i, self.base.y) for i in [-1, 1]]
        self.zombies = []
        self.structures = [self.base]
        self.projectiles = []
        self.particles = []
        self.floating_texts = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = self.steps >= self.MAX_STEPS or (len(self.units) == 0 and self.steps > 10)

        if self.game_over:
            if len(self.units) == 0:
                reward -= 100 # Loss penalty
            else:
                reward += 100 # Win bonus
            return self._get_observation(), reward, True, False, self._get_info()
        
        # --- Handle player input ---
        reward += self._handle_input(action)
        
        # --- Update game state ---
        update_reward = self._update_game_state()
        reward += update_reward

        self.steps += 1
        self.score += reward
        
        terminated = self.steps >= self.MAX_STEPS or (len(self.units) == 0 and self.steps > 10)
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)
        
        # Check if cursor position is occupied
        is_occupied = any(s.x == self.cursor_pos[0] and s.y == self.cursor_pos[1] for s in self.structures)

        # Build Unit (Space)
        if space_pressed and not is_occupied:
            if self.resources >= self.UNIT_COST and len(self.units) < self.MAX_UNITS:
                self.resources -= self.UNIT_COST
                self.units.append(Unit(self.cursor_pos[0], self.cursor_pos[1]))
                # Sound effect placeholder: # sfx_build_unit.wav
                self._create_particles(self.cursor_pos[0], self.cursor_pos[1], (50, 150, 255), 10)

        # Build Structure (Shift)
        if shift_pressed and not is_occupied:
            if self.resources >= self.STRUCTURE_COST:
                self.resources -= self.STRUCTURE_COST
                self.structures.append(Structure(self.cursor_pos[0], self.cursor_pos[1]))
                # Sound effect placeholder: # sfx_build_structure.wav
                self._create_particles(self.cursor_pos[0], self.cursor_pos[1], (150, 150, 160), 10)
        
        return 0 # No direct reward for actions

    def _update_game_state(self):
        reward = 0
        
        # --- Resource Generation ---
        self.resource_timer += 1
        if self.resource_timer >= 60:
            self.resource_timer = 0
            if self.resources < self.MAX_RESOURCES:
                self.resources += len(self.resource_nodes)
                self.resources = min(self.resources, self.MAX_RESOURCES)
                reward += 0.1 * len(self.resource_nodes)
                for node in self.resource_nodes:
                    self.floating_texts.append(FloatingText(node.x, node.y, "+1", (200, 255, 200), self.floating_text_font))

        # --- Difficulty Scaling ---
        self.wave_level = self.steps // 1000
        
        # --- Zombie Spawning ---
        zombie_spawn_interval = max(30, 150 - self.wave_level * 20)
        self.zombie_spawn_timer += 1
        if self.zombie_spawn_timer >= zombie_spawn_interval:
            self.zombie_spawn_timer = 0
            num_zombies = 1 + self.wave_level
            zombie_health = 80 + self.wave_level * 20
            zombie_speed = 0.02 + self.wave_level * 0.005
            for _ in range(num_zombies):
                edge = self.np_random.integers(4)
                if edge == 0: x, y = self.np_random.integers(self.GRID_WIDTH), -2
                elif edge == 1: x, y = self.GRID_WIDTH + 2, self.np_random.integers(self.GRID_HEIGHT)
                elif edge == 2: x, y = self.np_random.integers(self.GRID_WIDTH), self.GRID_HEIGHT + 2
                else: x, y = -2, self.np_random.integers(self.GRID_HEIGHT)
                self.zombies.append(Zombie(x, y, zombie_speed, zombie_health))

        # --- Update Entities ---
        all_targets = self.units + self.structures
        for zombie in self.zombies:
            zombie.update(all_targets)
        
        for unit in self.units:
            unit.update(self.zombies, self.projectiles)
            
        # --- Update Projectiles ---
        projectiles_to_remove = []
        for p in self.projectiles:
            if p.update() or p.lifetime <= 0:
                projectiles_to_remove.append(p)
                if p.lifetime > 0 and p.target: # Hit
                    self._create_particles(p.target.x, p.target.y, (255, 100, 100), 5)
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        
        # --- Handle Deaths ---
        dead_zombies = [z for z in self.zombies if z.health <= 0]
        for z in dead_zombies:
            reward += 1.0
            self.floating_texts.append(FloatingText(z.x, z.y, "+1", (255, 255, 100), self.floating_text_font))
            self._create_particles(z.x, z.y, (200, 50, 50), 20)
            # Sound effect placeholder: # sfx_zombie_death.wav
        self.zombies = [z for z in self.zombies if z.health > 0]
        
        dead_units = [u for u in self.units if u.health <= 0]
        for u in dead_units:
            reward -= 0.1
            self._create_particles(u.x, u.y, (50, 150, 255), 20)
        self.units = [u for u in self.units if u.health > 0]
        
        self.structures = [s for s in self.structures if s.health > 0 and s != self.base]
        self.structures.append(self.base) # Ensure base is not removed

        # --- Update Effects ---
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles: p.update()
        
        self.floating_texts = [t for t in self.floating_texts if t.lifetime > 0]
        for t in self.floating_texts: t.update()

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_world()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "resources": self.resources, "units": len(self.units), "zombies": len(self.zombies)}

    def _render_world(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                iso_x = (x - y) * 16 + self.ISO_OFFSET[0]
                iso_y = (x + y) * 8 + self.ISO_OFFSET[1]
                pygame.gfxdraw.line(self.screen, iso_x, iso_y, iso_x + 16, iso_y + 8, self.COLOR_GRID)
                pygame.gfxdraw.line(self.screen, iso_x, iso_y, iso_x - 16, iso_y + 8, self.COLOR_GRID)

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        iso_x = (cursor_x - cursor_y) * 16 + self.ISO_OFFSET[0]
        iso_y = (cursor_y + cursor_x) * 8 + self.ISO_OFFSET[1]
        
        is_occupied = any(s.x == cursor_x and s.y == cursor_y for s in self.structures)
        cursor_color = (255, 0, 0) if is_occupied else (255, 255, 0)

        points = [
            (iso_x, iso_y), (iso_x + 16, iso_y + 8),
            (iso_x, iso_y + 16), (iso_x - 16, iso_y + 8)
        ]
        pygame.draw.polygon(self.screen, (*cursor_color, 50), points)
        pygame.draw.aalines(self.screen, cursor_color, True, points, 2)

        # Sort all drawable entities by Y-position for correct layering
        drawable_entities = self.resource_nodes + self.structures + self.units + self.zombies
        drawable_entities.sort(key=lambda e: e.y + e.x * 0.1) # Sort by grid y primarily

        for entity in drawable_entities:
            entity.draw(self.screen, self.ISO_OFFSET)

        # Draw projectiles, particles, and floating text
        for p in self.projectiles: p.draw(self.screen, self.ISO_OFFSET)
        for p in self.particles: p.draw(self.screen, self.ISO_OFFSET)
        for t in self.floating_texts: t.draw(self.screen, self.ISO_OFFSET)

    def _render_ui(self):
        # Resources
        res_text = self.ui_font.render(f"RES: {self.resources}/{self.MAX_RESOURCES}", True, (200, 255, 200))
        self.screen.blit(res_text, (10, 10))
        
        # Units
        unit_text = self.ui_font.render(f"UNITS: {len(self.units)}/{self.MAX_UNITS}", True, (150, 220, 255))
        self.screen.blit(unit_text, (self.width - unit_text.get_width() - 10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 30
        time_text = self.ui_font.render(f"TIME: {int(time_left // 60):02}:{int(time_left % 60):02}", True, (255, 255, 255))
        self.screen.blit(time_text, (self.width // 2 - time_text.get_width() // 2, 10))

        # Wave
        wave_text = self.ui_font.render(f"WAVE: {self.wave_level + 1}", True, (255, 100, 100))
        self.screen.blit(wave_text, (self.width // 2 - wave_text.get_width() // 2, 30))

        if self.game_over:
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            end_font = pygame.font.SysFont("monospace", 40, bold=True)
            msg = "VICTORY" if len(self.units) > 0 else "DEFEAT"
            color = (100, 255, 100) if len(self.units) > 0 else (255, 100, 100)
            end_text = end_font.render(msg, True, color)
            self.screen.blit(end_text, (self.width//2 - end_text.get_width()//2, self.height//2 - end_text.get_height()//2))

    def _create_particles(self, grid_x, grid_y, color, count):
        iso_x = (grid_x - grid_y) * 16
        iso_y = (grid_x + grid_y) * 8
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            size = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append(Particle(iso_x, iso_y, color, size, lifetime, vel))

    def close(self):
        pygame.quit()