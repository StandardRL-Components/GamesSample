import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:48:25.017727
# Source Brief: brief_02604.md
# Brief Index: 2604
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes for Game Entities ---

class Particle:
    def __init__(self, pos, vel, life, start_color, end_color, start_size, end_size=0):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.life = self.max_life = life
        self.start_color = start_color
        self.end_color = end_color
        self.start_size = start_size
        self.end_size = end_size

    def update(self):
        self.pos += self.vel
        self.life -= 1
        self.vel *= 0.98  # Damping

    def draw(self, surface):
        if self.life > 0:
            life_ratio = self.life / self.max_life
            current_color = [
                self.start_color[i] * life_ratio + self.end_color[i] * (1 - life_ratio)
                for i in range(3)
            ]
            current_size = int(self.start_size * life_ratio + self.end_size * (1 - life_ratio))
            if current_size > 0:
                pygame.draw.circle(surface, current_color, self.pos, current_size)

class Player:
    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.size = 12
        self.color = (0, 255, 128)
        self.max_speed = 5.0
        self.acceleration = 0.6

    def update(self, movement_action, bounds):
        # Apply acceleration based on movement action
        if movement_action == 1: self.vel.y -= self.acceleration  # Up
        if movement_action == 2: self.vel.y += self.acceleration  # Down
        if movement_action == 3: self.vel.x -= self.acceleration  # Left
        if movement_action == 4: self.vel.x += self.acceleration  # Right

        # Limit speed
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)

        # Apply friction/damping
        self.vel *= 0.95
        if self.vel.length() < 0.1:
            self.vel.update(0, 0)

        # Update position and enforce boundaries
        self.pos += self.vel
        self.pos.x = np.clip(self.pos.x, self.size, bounds[0] - self.size)
        self.pos.y = np.clip(self.pos.y, self.size, bounds[1] - self.size)

    def draw(self, surface, particles):
        # Glow effect
        glow_radius = int(self.size * 2.5)
        glow_alpha = 60
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.color, glow_alpha), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (int(self.pos.x - glow_radius), int(self.pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Main body (triangle)
        p1 = self.pos + pygame.Vector2(0, -self.size)
        p2 = self.pos + pygame.Vector2(-self.size * 0.8, self.size * 0.8)
        p3 = self.pos + pygame.Vector2(self.size * 0.8, self.size * 0.8)
        points = [p1, p2, p3]
        pygame.gfxdraw.aapolygon(surface, [(int(p.x), int(p.y)) for p in points], self.color)
        pygame.gfxdraw.filled_polygon(surface, [(int(p.x), int(p.y)) for p in points], self.color)
        
        # Thrust particles
        if self.vel.length_squared() > 1:
            # sfx: player_thrust
            angle = self.vel.angle_to(pygame.Vector2(1, 0)) + 180
            for _ in range(2):
                p_angle = math.radians(angle + random.uniform(-20, 20))
                p_vel = pygame.Vector2(math.cos(p_angle), -math.sin(p_angle)) * random.uniform(1, 3)
                p_pos = self.pos - self.vel.normalize() * self.size
                particles.append(Particle(p_pos, p_vel, 15, (255, 200, 0), (255, 50, 0), 4, 1))

class Tile:
    FRACTAL_PATTERNS = {
        0: [(-1, 0), (0, 0), (1, 0), (0, 1)],  # T-shape
        1: [(0, 0), (1, 0), (0, 1), (1, 1)],  # Square
        2: [(-1, 0), (0, 0), (1, 0), (2, 0)],  # Line
        3: [(0, 0), (1, 0), (0, 1), (-1, 1)], # S-shape
    }
    COLORS = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 128, 0)]

    def __init__(self, pos, type_id, falling=True):
        self.pos = pygame.Vector2(pos)
        self.type_id = type_id
        self.color = self.COLORS[type_id]
        self.orientation = 0
        self.is_falling = falling
        self.fall_speed = 1.0
        self.block_size = 8

    def rotate(self):
        self.orientation = (self.orientation + 1) % 4
        # sfx: tile_rotate

    def get_block_positions(self):
        base_pattern = self.FRACTAL_PATTERNS[self.type_id]
        rotated_pattern = []
        for x, y in base_pattern:
            for _ in range(self.orientation):
                x, y = -y, x
            rotated_pattern.append(pygame.Vector2(x, y) * self.block_size)
        return [self.pos + p for p in rotated_pattern]

    def update(self, static_tiles):
        if self.is_falling:
            self.pos.y += self.fall_speed
            
            # Check for landing on bottom or other tiles
            for block_pos in self.get_block_positions():
                if block_pos.y >= 400 - self.block_size:
                    self.is_falling = False
                    break
                for static in static_tiles:
                    for static_block_pos in static.get_block_positions():
                        if pygame.Rect(block_pos.x, block_pos.y + self.fall_speed, self.block_size, self.block_size).colliderect(
                           pygame.Rect(static_block_pos.x, static_block_pos.y, self.block_size, self.block_size)):
                            self.is_falling = False
                            break
                    if not self.is_falling: break
            if not self.is_falling:
                # sfx: tile_land
                # Adjust position to sit perfectly on top
                self.pos.y = math.floor(self.pos.y / self.block_size) * self.block_size

    def draw(self, surface):
        for block_pos in self.get_block_positions():
            rect = pygame.Rect(int(block_pos.x), int(block_pos.y), self.block_size, self.block_size)
            pygame.draw.rect(surface, self.color, rect)
            pygame.draw.rect(surface, (255, 255, 255), rect, 1)

class Tower:
    def __init__(self, pos, type_id, particles):
        self.pos = pygame.Vector2(pos)
        self.type_id = type_id
        self.color = Tile.COLORS[type_id]
        self.range = 150 + type_id * 20
        self.fire_rate = 30 - type_id * 5
        self.cooldown = 0
        self.size = 15
        # sfx: tower_built
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1, 4)
            particles.append(Particle(self.pos, vel, 25, self.color, (50, 50, 50), 5))


    def update(self, enemies, projectiles):
        if self.cooldown > 0:
            self.cooldown -= 1
            return

        target = None
        min_dist = self.range
        for enemy in enemies:
            dist = self.pos.distance_to(enemy.pos)
            if dist < min_dist:
                min_dist = dist
                target = enemy
        
        if target:
            self.cooldown = self.fire_rate
            direction = (target.pos - self.pos).normalize()
            projectiles.append(Projectile(self.pos, direction * 8, self.color))
            # sfx: tower_shoot

    def draw(self, surface):
        # Glow
        glow_radius = int(self.size * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.color, 80), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (int(self.pos.x - glow_radius), int(self.pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Body
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.size, self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.size, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.size*0.6), (255,255,255))

class Projectile:
    def __init__(self, pos, vel, color):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.life = 120 # frames

    def update(self):
        self.pos += self.vel
        self.life -= 1

    def draw(self, surface):
        end_pos = self.pos - self.vel.normalize() * 10
        pygame.draw.line(surface, self.color, self.pos, end_pos, 3)

class Enemy:
    def __init__(self, pos, speed):
        self.pos = pygame.Vector2(pos)
        self.speed = speed
        self.size = 10
        self.color = (255, 50, 50)
        self.direction = 1 if random.random() > 0.5 else -1

    def update(self, bounds):
        self.pos.x += self.direction * self.speed
        if self.pos.x < self.size or self.pos.x > bounds[0] - self.size:
            self.direction *= -1
            self.pos.x = np.clip(self.pos.x, self.size, bounds[0] - self.size)

    def draw(self, surface):
        # Glow
        glow_radius = int(self.size * 2)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.color, 70), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (int(self.pos.x - glow_radius), int(self.pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Body
        rect = pygame.Rect(int(self.pos.x - self.size), int(self.pos.y - self.size), self.size * 2, self.size * 2)
        pygame.draw.rect(surface, self.color, rect)
        pygame.draw.rect(surface, (255, 255, 255), rect, 1)

class Obstacle:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.color = (100, 100, 120)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        pygame.draw.rect(surface, (150, 150, 180), self.rect, 2)


# --- Main Gymnasium Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a neon track, place portals to teleport, and match falling tiles to build defensive towers against enemies."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to place/use portals. Press shift to rotate falling tiles."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 20)
        
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PORTAL = (50, 150, 255)

        # Persistent state across levels
        self.level = 1
        self.max_episode_steps = 5000

        # self.reset() is called by the environment wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward = 0
        
        self.player = Player((self.width // 2, self.height - 50))
        
        self.enemies = []
        self.obstacles = []
        self.projectiles = []
        self.particles = []
        
        self.falling_tiles = []
        self.static_tiles = []
        self.towers = []
        self.tile_spawn_timer = 0

        self.portals = [None, None]
        self.portal_state = 0 # 0: none, 1: one placed, 2: pair active

        self._prev_space_held = False
        self._prev_shift_held = False
        
        self._generate_level()

        return self._get_observation(), self._get_info()
    
    def _win_level(self):
        self.level += 1
        # In a real game, you might show a "Level Complete" screen
        # Here, we just reset for the next level
        self.reset()
        # sfx: level_complete

    def _generate_level(self):
        num_enemies = 1 + self.level // 2
        enemy_speed = 1.0 + self.level * 0.05
        
        for _ in range(num_enemies):
            pos = (random.randint(50, self.width - 50), random.randint(50, self.height - 150))
            self.enemies.append(Enemy(pos, enemy_speed))

        # Procedural obstacles to form a "track"
        num_obstacles = self.level * 2
        for _ in range(num_obstacles):
            w = random.randint(50, 150)
            h = random.randint(20, 40)
            x = random.randint(0, self.width - w)
            y = random.randint(0, self.height - 100)
            self.obstacles.append(Obstacle((x, y, w, h)))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Actions ---
        space_pressed = space_held and not self._prev_space_held
        shift_pressed = shift_held and not self._prev_shift_held

        if shift_pressed and self.falling_tiles:
            self.falling_tiles[0].rotate()

        if space_pressed:
            self._handle_portal_placement()

        # --- Update Game Logic ---
        prev_y_pos = self.player.pos.y
        self.player.update(movement, (self.width, self.height))
        if self.player.pos.y < prev_y_pos: # Moved "forward" (up)
            self.reward += 0.1
        
        self._update_tiles()
        self._update_towers()
        self._update_projectiles_and_enemies()
        
        for p in self.particles: p.update()
        self.particles = [p for p in self.particles if p.life > 0]
        
        # --- Check Collisions & Termination ---
        terminated = self._check_collisions() or self.steps >= self.max_episode_steps
        truncated = self.steps >= self.max_episode_steps

        # Check for win condition (reaching top of screen)
        # In a scrolling world, this would be a world coordinate.
        # Here, we'll use a simplified version: stay at top for a bit.
        # For this brief, we'll use a step limit and enemy collision as main terminators.
        # A more complex implementation would scroll the world.
        # For simplicity, let's use a dummy finish line.
        if self.player.pos.y < 20:
             self.reward += 100
             terminated = True
             # sfx: win_game
             # In a multi-level scenario, we'd call self._win_level() here
             # but for a single episode, we terminate.

        self.game_over = terminated or truncated
        
        self._prev_space_held = space_held
        self._prev_shift_held = shift_held

        return (
            self._get_observation(),
            self.reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_portal_placement(self):
        if self.portal_state == 0:
            self.portals[0] = pygame.Vector2(self.player.pos)
            self.portal_state = 1
            # sfx: portal_place_1
            for _ in range(20):
                angle = random.uniform(0, 2*math.pi)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1,3)
                self.particles.append(Particle(self.portals[0], vel, 20, self.COLOR_PORTAL, (0,0,0), 4))

        elif self.portal_state == 1:
            if self.player.pos.distance_to(self.portals[0]) > 50:
                self.portals[1] = pygame.Vector2(self.player.pos)
                self.portal_state = 2
                self.reward += 5
                # sfx: portal_place_2
                for _ in range(20):
                    angle = random.uniform(0, 2*math.pi)
                    vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1,3)
                    self.particles.append(Particle(self.portals[1], vel, 20, self.COLOR_PORTAL, (0,0,0), 4))
        
        elif self.portal_state == 2:
            dist0 = self.player.pos.distance_to(self.portals[0])
            dist1 = self.player.pos.distance_to(self.portals[1])
            if dist0 < 25:
                self.player.pos = pygame.Vector2(self.portals[1])
                self._portal_teleport_effect(self.portals[0], self.portals[1])
            elif dist1 < 25:
                self.player.pos = pygame.Vector2(self.portals[0])
                self._portal_teleport_effect(self.portals[1], self.portals[0])

    def _portal_teleport_effect(self, start_pos, end_pos):
        # sfx: portal_teleport
        self.portals = [None, None]
        self.portal_state = 0
        self.player.vel *= 0.1 # Dampen velocity after teleport
        for _ in range(50):
            # Implosion at start
            angle = random.uniform(0, 2*math.pi)
            vel = (self.player.pos - start_pos).normalize() * random.uniform(2, 5)
            self.particles.append(Particle(start_pos, vel, 25, (255,255,255), self.COLOR_PORTAL, 6))
            # Explosion at end
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(2, 5)
            self.particles.append(Particle(end_pos, vel, 25, (255,255,255), self.COLOR_PORTAL, 6))

    def _update_tiles(self):
        self.tile_spawn_timer -= 1
        if self.tile_spawn_timer <= 0 and len(self.falling_tiles) < 3:
            self.tile_spawn_timer = random.randint(60, 120)
            pos = (random.randint(50, self.width - 50), -20)
            type_id = random.randint(0, len(Tile.FRACTAL_PATTERNS) - 1)
            self.falling_tiles.append(Tile(pos, type_id))

        for tile in self.falling_tiles:
            tile.update(self.static_tiles)
        
        newly_static = [t for t in self.falling_tiles if not t.is_falling]
        self.static_tiles.extend(newly_static)
        self.falling_tiles = [t for t in self.falling_tiles if t.is_falling]

        # Check for matches to create towers
        if newly_static:
            self._check_for_tile_matches()

    def _check_for_tile_matches(self):
        # This is a simplified adjacency check.
        # A more robust solution would use a grid.
        matched_indices = set()
        for i in range(len(self.static_tiles)):
            for j in range(i + 1, len(self.static_tiles)):
                if i in matched_indices or j in matched_indices:
                    continue
                
                tile1 = self.static_tiles[i]
                tile2 = self.static_tiles[j]
                
                if tile1.type_id == tile2.type_id:
                    for b1 in tile1.get_block_positions():
                        for b2 in tile2.get_block_positions():
                            if b1.distance_to(b2) < tile1.block_size * 1.5:
                                matched_indices.add(i)
                                matched_indices.add(j)
                                
                                # Create tower at midpoint
                                tower_pos = (tile1.pos + tile2.pos) / 2
                                self.towers.append(Tower(tower_pos, tile1.type_id, self.particles))
                                self.reward += 1
                                break
                        if i in matched_indices: break
        
        if matched_indices:
            self.static_tiles = [tile for i, tile in enumerate(self.static_tiles) if i not in matched_indices]

    def _update_towers(self):
        for tower in self.towers:
            tower.update(self.enemies, self.projectiles)

    def _update_projectiles_and_enemies(self):
        projectiles_to_keep = []
        enemies_hit_indices = set()

        for proj in self.projectiles:
            proj.update()
            hit = False
            if 0 < proj.pos.x < self.width and 0 < proj.pos.y < self.height and proj.life > 0:
                for i, enemy in enumerate(self.enemies):
                    if i in enemies_hit_indices: continue
                    if proj.pos.distance_to(enemy.pos) < enemy.size:
                        enemies_hit_indices.add(i)
                        hit = True
                        self.reward += 2
                        self.score += 10
                        # sfx: enemy_hit
                        for _ in range(15):
                            angle = random.uniform(0, 2*math.pi)
                            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(0.5, 3)
                            self.particles.append(Particle(enemy.pos, vel, 20, enemy.color, (255,255,0), 4))
                        break
                if not hit:
                    projectiles_to_keep.append(proj)
        
        self.projectiles = projectiles_to_keep
        if enemies_hit_indices:
            self.enemies = [enemy for i, enemy in enumerate(self.enemies) if i not in enemies_hit_indices]

        for enemy in self.enemies:
            enemy.update((self.width, self.height))

    def _check_collisions(self):
        # Player with enemies
        for enemy in self.enemies:
            if self.player.pos.distance_to(enemy.pos) < self.player.size + enemy.size:
                self.reward -= 100
                # sfx: player_death_enemy
                return True
        
        # Player with obstacles
        player_rect = pygame.Rect(self.player.pos.x - self.player.size, self.player.pos.y - self.player.size, self.player.size*2, self.player.size*2)
        for obs in self.obstacles:
            if player_rect.colliderect(obs.rect):
                self.reward -= 100
                # sfx: player_death_obstacle
                return True
        
        # Player with screen boundaries (soft collision)
        if not (self.player.size < self.player.pos.x < self.width - self.player.size) or \
           not (self.player.size < self.player.pos.y < self.height - self.player.size):
            self.reward -= 0.5

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
        }

    def _render_background(self):
        grid_color = (30, 25, 55)
        for i in range(0, self.width, 40):
            pygame.draw.line(self.screen, grid_color, (i, 0), (i, self.height))
        for i in range(0, self.height, 40):
            pygame.draw.line(self.screen, grid_color, (0, i), (self.width, i))

    def _render_game(self):
        for obs in self.obstacles: obs.draw(self.screen)
        
        # Portals
        if self.portals[0]: self._render_portal(self.portals[0], self.portal_state > 0)
        if self.portals[1]: self._render_portal(self.portals[1], self.portal_state > 1)
        if self.portal_state == 2:
            pygame.draw.aaline(self.screen, (*self.COLOR_PORTAL, 100), self.portals[0], self.portals[1])
        
        for tile in self.static_tiles: tile.draw(self.screen)
        for tower in self.towers: tower.draw(self.screen)
        for proj in self.projectiles: proj.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)
        for tile in self.falling_tiles: tile.draw(self.screen)
        
        self.player.draw(self.screen, self.particles)
        
        for p in self.particles: p.draw(self.screen)

    def _render_portal(self, pos, is_active):
        radius = 20
        alpha = 255 if is_active else 100
        color = (*self.COLOR_PORTAL, alpha)
        
        # Pulsing glow
        pulse = abs(math.sin(self.steps * 0.1))
        glow_radius = int(radius * (1.5 + pulse * 0.3))
        glow_alpha = int(90 + pulse * 40)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PORTAL, glow_alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(pos.x - glow_radius), int(pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Ring
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius - 2, color)


    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, (220, 220, 220))
        level_text = self.font.render(f"LEVEL: {self.level}", True, (220, 220, 220))
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.max_episode_steps}", True, (220, 220, 220))
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(level_text, (10, 35))
        self.screen.blit(steps_text, (self.width - steps_text.get_width() - 10, 10))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    # Re-enable video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Fractal Racer")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Human Controls to Action Mapping ---
        movement = 0 # None
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS
        
        if done:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    env.close()