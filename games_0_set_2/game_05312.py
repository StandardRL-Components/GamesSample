
# Generated: 2025-08-28T04:38:18.988531
# Source Brief: brief_05312.md
# Brief Index: 5312

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game entities
class Entity:
    def __init__(self, x, y, size, color):
        self.x, self.y = x, y
        self.size = size
        self.color = color
        self.vx, self.vy = 0, 0
        self.health = 100
        self.max_health = 100
        self.alive = True
        self.world_y = self.y * TILE_H_HALF + self.x * TILE_H_HALF

    def update_world_y(self):
        self.world_y = self.y * TILE_H_HALF + self.x * TILE_H_HALF

    def draw_health_bar(self, surface, offset_x, offset_y, iso_pos):
        if self.health < self.max_health:
            bar_width = 30
            bar_height = 5
            x, y = iso_pos
            bg_rect = pygame.Rect(x - bar_width // 2, y - self.size * 1.5 - bar_height, bar_width, bar_height)
            health_ratio = max(0, self.health / self.max_health)
            fg_rect = pygame.Rect(x - bar_width // 2, y - self.size * 1.5 - bar_height, bar_width * health_ratio, bar_height)
            pygame.draw.rect(surface, (50, 50, 50), bg_rect)
            pygame.draw.rect(surface, (255, 0, 0), fg_rect)


class Player(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, 12, (0, 150, 255))
        self.attack_power = 10
        self.ranged_power = 7
        self.attack_cooldown = 0
        self.ranged_cooldown = 0
        self.bob_angle = 0
        self.max_health = 100
        self.health = 100

    def update(self):
        self.attack_cooldown = max(0, self.attack_cooldown - 1)
        self.ranged_cooldown = max(0, self.ranged_cooldown - 1)
        self.bob_angle = (self.bob_angle + 0.2) % (2 * math.pi)

class Enemy(Entity):
    def __init__(self, x, y, level):
        super().__init__(x, y, 10, (255, 50, 50))
        self.max_health = 10 + 5 * (level - 1)
        self.health = self.max_health
        self.attack_power = 1 + 1 * (level - 1)
        self.attack_cooldown = 0
        self.ai_state = "patrol"
        self.patrol_target = (x, y)
        self.path = deque()
        self.bob_angle = random.uniform(0, 2 * math.pi)

    def update(self, player_pos, grid):
        self.attack_cooldown = max(0, self.attack_cooldown - 1)
        self.bob_angle = (self.bob_angle + 0.1) % (2 * math.pi)

class Boss(Enemy):
    def __init__(self, x, y, level):
        super().__init__(x, y, level)
        self.size = 20
        self.color = (150, 0, 200)
        self.max_health = 150
        self.health = self.max_health
        self.attack_power = 15

class Projectile:
    def __init__(self, x, y, target_x, target_y, speed, color, size):
        self.x, self.y = x, y
        self.color = color
        self.size = size
        angle = math.atan2(target_y - y, target_x - x)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = 60 # 2 seconds at 30fps

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        return self.lifetime > 0

class Particle:
    def __init__(self, x, y, color, lifetime):
        self.x, self.y = x, y
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1.5, -0.5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        return self.lifetime > 0

class FloatingText:
    def __init__(self, text, x, y, color, lifetime=45):
        self.text = text
        self.x, self.y = x, y
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.vy = -0.8

    def update(self):
        self.y += self.vy
        self.lifetime -= 1
        return self.lifetime > 0

class Chest:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.opened = False
        self.content = "gold" if random.random() > 0.3 else "health"
        self.world_y = self.y * TILE_H_HALF + self.x * TILE_H_HALF

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
TILE_W = 48
TILE_H = 24
TILE_W_HALF = TILE_W // 2
TILE_H_HALF = TILE_H // 2

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Space for melee attack. Shift for ranged attack."
    )
    game_description = (
        "Explore an isometric dungeon, fight monsters, and defeat the final boss."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_medium = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        self.COLOR_BG = (20, 25, 30)
        self.COLOR_FLOOR = (40, 50, 60)
        self.COLOR_WALL_TOP = (80, 90, 100)
        self.COLOR_WALL_SIDE = (60, 70, 80)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.max_levels = 5
        self.max_steps = 2500

        self.player = None
        self.enemies = []
        self.boss = None
        self.chests = []
        self.stairs = None
        self.projectiles = []
        self.particles = []
        self.floating_texts = []
        
        self.grid = []
        self.map_width = 50
        self.map_height = 50

        self.prev_space_held = False
        self.prev_shift_held = False
        self.prev_dist_to_goal = float('inf')

        self.reset()
        self.validate_implementation()
    
    def _iso_to_screen(self, x, y):
        screen_x = (x - y) * TILE_W_HALF
        screen_y = (x + y) * TILE_H_HALF
        return int(screen_x), int(screen_y)

    def _generate_dungeon(self):
        self.grid = np.zeros((self.map_width, self.map_height), dtype=int)
        rooms = []
        num_rooms = 10
        for _ in range(num_rooms * 3):
            w = self.np_random.integers(5, 10)
            h = self.np_random.integers(5, 10)
            x = self.np_random.integers(1, self.map_width - w - 1)
            y = self.np_random.integers(1, self.map_height - h - 1)
            new_room = pygame.Rect(x, y, w, h)
            
            if not any(new_room.colliderect(r) for r in rooms):
                rooms.append(new_room)
            if len(rooms) >= num_rooms:
                break
        
        if not rooms: # Failsafe
            rooms.append(pygame.Rect(10,10,10,10))

        for room in rooms:
            self.grid[room.left:room.right, room.top:room.bottom] = 1

        for i in range(len(rooms) - 1):
            x1, y1 = rooms[i].center
            x2, y2 = rooms[i+1].center
            if self.np_random.random() > 0.5:
                self.grid[min(x1, x2):max(x1, x2)+1, y1] = 1
                self.grid[x2, min(y1, y2):max(y1, y2)+1] = 1
            else:
                self.grid[x1, min(y1, y2):max(y1, y2)+1] = 1
                self.grid[min(x1, x2):max(x1, x2)+1, y2] = 1
        
        start_room = rooms[0]
        end_room = rooms[-1]
        
        player_start = start_room.center
        goal_pos = end_room.center

        return player_start, goal_pos, rooms

    def _spawn_entities(self, player_start, goal_pos, rooms):
        self.player = Player(player_start[0], player_start[1])
        self.enemies = []
        self.chests = []
        self.boss = None

        if self.level == self.max_levels:
            self.boss = Boss(goal_pos[0], goal_pos[1], self.level)
            self.stairs = None
        else:
            self.stairs = goal_pos
        
        for i, room in enumerate(rooms):
            if i == 0 or i == len(rooms) -1: continue # Skip start and end rooms
            
            # Spawn enemies
            num_enemies = self.np_random.integers(1, 2 + self.level)
            for _ in range(num_enemies):
                x = self.np_random.integers(room.left + 1, room.right - 1)
                y = self.np_random.integers(room.top + 1, room.bottom - 1)
                self.enemies.append(Enemy(x, y, self.level))
            
            # Spawn chests
            if self.np_random.random() < 0.4:
                x = self.np_random.integers(room.left + 1, room.right - 1)
                y = self.np_random.integers(room.top + 1, room.bottom - 1)
                self.chests.append(Chest(x, y))


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        
        player_start, goal_pos, rooms = self._generate_dungeon()
        self._spawn_entities(player_start, goal_pos, rooms)
        
        self.projectiles = []
        self.particles = []
        self.floating_texts = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        if self.boss:
            goal = self.boss
        else:
            goal = type('Stairs', (object,), {'x': self.stairs[0], 'y': self.stairs[1]})
        self.prev_dist_to_goal = math.hypot(self.player.x - goal.x, self.player.y - goal.y)
        
        return self._get_observation(), self._get_info()

    def _handle_input(self, action):
        movement, space_val, shift_val = action
        space_pressed = space_val == 1 and not self.prev_space_held
        shift_pressed = shift_val == 1 and not self.prev_shift_held
        self.prev_space_held = space_val == 1
        self.prev_shift_held = shift_val == 1

        move_speed = 0.15
        if movement == 1: self.player.vy = -move_speed; self.player.vx = -move_speed
        elif movement == 2: self.player.vy = move_speed; self.player.vx = move_speed
        elif movement == 3: self.player.vy = move_speed; self.player.vx = -move_speed
        elif movement == 4: self.player.vy = -move_speed; self.player.vx = move_speed
        else: self.player.vx, self.player.vy = 0, 0
        
        reward = 0
        # Melee Attack
        if space_pressed and self.player.attack_cooldown == 0:
            self.player.attack_cooldown = 20 # Cooldown frames
            # find enemies in range
            for enemy in self.enemies + ([self.boss] if self.boss else []):
                if enemy and enemy.alive:
                    dist = math.hypot(self.player.x - enemy.x, self.player.y - enemy.y)
                    if dist < 2.5:
                        damage = self.player.attack_power
                        enemy.health -= damage
                        px, py = self._iso_to_screen(enemy.x, enemy.y)
                        self.floating_texts.append(FloatingText(str(damage), px, py - 20, (255, 100, 100)))
                        for _ in range(5): self.particles.append(Particle(px, py, (255, 50, 50), 20))
                        # sfx: sword_hit
                        if enemy.health <= 0:
                            enemy.alive = False
                            reward += 10 if isinstance(enemy, Boss) else 1
                            self.score += 100 if isinstance(enemy, Boss) else 10
                            for _ in range(20): self.particles.append(Particle(px, py, enemy.color, 40))
                            # sfx: enemy_die
        
        # Ranged Attack
        if shift_pressed and self.player.ranged_cooldown == 0:
            self.player.ranged_cooldown = 35
            target = min(self.enemies + ([self.boss] if self.boss else []), 
                         key=lambda e: math.hypot(self.player.x - e.x, self.player.y - e.y) if e and e.alive else float('inf'), 
                         default=None)
            if target and target.alive:
                p_iso_x, p_iso_y = self._iso_to_screen(self.player.x, self.player.y)
                t_iso_x, t_iso_y = self._iso_to_screen(target.x, target.y)
                self.projectiles.append(Projectile(p_iso_x, p_iso_y, t_iso_x, t_iso_y, 5, (100, 200, 255), 4))
                # sfx: magic_shoot
        return reward

    def _update_game_state(self):
        reward = 0
        
        # Update player
        self.player.update()
        new_x, new_y = self.player.x + self.player.vx, self.player.y + self.player.vy
        if self.grid[int(new_x)][int(self.player.y)] == 1: self.player.x = new_x
        if self.grid[int(self.player.x)][int(new_y)] == 1: self.player.y = new_y
        self.player.update_world_y()

        # Update enemies
        for enemy in self.enemies + ([self.boss] if self.boss else []):
            if not enemy or not enemy.alive: continue
            enemy.update(self.player, self.grid)
            dist_to_player = math.hypot(self.player.x - enemy.x, self.player.y - enemy.y)
            
            if dist_to_player < 8: # Chase range
                dx, dy = self.player.x - enemy.x, self.player.y - enemy.y
                norm = math.hypot(dx, dy)
                if norm > 0:
                    move_x, move_y = dx/norm * 0.08, dy/norm * 0.08
                    new_ex, new_ey = enemy.x + move_x, enemy.y + move_y
                    if self.grid[int(new_ex)][int(enemy.y)] == 1: enemy.x = new_ex
                    if self.grid[int(enemy.x)][int(new_ey)] == 1: enemy.y = new_ey
                
                if dist_to_player < 1.8 and enemy.attack_cooldown == 0:
                    enemy.attack_cooldown = 60 # 2 sec cooldown
                    self.player.health -= enemy.attack_power
                    px, py = self._iso_to_screen(self.player.x, self.player.y)
                    self.floating_texts.append(FloatingText(str(enemy.attack_power), px, py - 20, (255, 0, 0)))
                    for _ in range(10): self.particles.append(Particle(px, py, self.player.color, 25))
                    # sfx: player_hurt
            enemy.update_world_y()

        # Update projectiles
        for p in self.projectiles[:]:
            if not p.update():
                self.projectiles.remove(p)
                continue
            for enemy in self.enemies + ([self.boss] if self.boss else []):
                if not enemy or not enemy.alive: continue
                e_iso_x, e_iso_y = self._iso_to_screen(enemy.x, enemy.y)
                if math.hypot(p.x - e_iso_x, p.y - e_iso_y) < enemy.size:
                    damage = self.player.ranged_power
                    enemy.health -= damage
                    self.floating_texts.append(FloatingText(str(damage), p.x, p.y, (255, 100, 100)))
                    for _ in range(5): self.particles.append(Particle(p.x, p.y, p.color, 20))
                    # sfx: ranged_hit
                    if enemy.health <= 0:
                        enemy.alive = False
                        reward += 10 if isinstance(enemy, Boss) else 1
                        self.score += 100 if isinstance(enemy, Boss) else 10
                        for _ in range(20): self.particles.append(Particle(e_iso_x, e_iso_y, enemy.color, 40))
                        # sfx: enemy_die
                    if p in self.projectiles: self.projectiles.remove(p)
                    break
        
        # Clean up dead enemies
        self.enemies = [e for e in self.enemies if e.alive]
        if self.boss and not self.boss.alive: self.boss = None

        # Update effects
        self.particles = [p for p in self.particles if p.update()]
        self.floating_texts = [t for t in self.floating_texts if t.update()]

        # Check chests
        for chest in self.chests:
            if not chest.opened and math.hypot(self.player.x - chest.x, self.player.y - chest.y) < 1.5:
                chest.opened = True
                px, py = self._iso_to_screen(chest.x, chest.y)
                if chest.content == "gold":
                    reward += 5
                    self.score += 50
                    self.floating_texts.append(FloatingText("+50 Gold", px, py, (255, 255, 0)))
                    # sfx: coin_pickup
                else: # health
                    reward += 2
                    heal_amount = 25
                    self.player.health = min(self.player.max_health, self.player.health + heal_amount)
                    self.floating_texts.append(FloatingText(f"+{heal_amount} HP", px, py, (0, 255, 0)))
                    # sfx: health_pickup
        
        # Check stairs
        if self.stairs and math.hypot(self.player.x - self.stairs[0], self.player.y - self.stairs[1]) < 1.5:
            self.level += 1
            self.score += self.level * 20
            player_start, goal_pos, rooms = self._generate_dungeon()
            self._spawn_entities(player_start, goal_pos, rooms)
            # sfx: level_up
        
        # Distance to goal reward
        if self.boss:
            goal = self.boss
        elif self.stairs:
            goal = type('Stairs', (object,), {'x': self.stairs[0], 'y': self.stairs[1]})
        else: # Game won
            goal = self.player
        
        dist_to_goal = math.hypot(self.player.x - goal.x, self.player.y - goal.y)
        if dist_to_goal < self.prev_dist_to_goal:
            reward += 0.1
        else:
            reward -= 0.1
        self.prev_dist_to_goal = dist_to_goal

        return reward

    def _check_termination(self):
        if self.player.health <= 0:
            self.game_over = True
            return True, -100 # Death penalty
        if self.level > self.max_levels: # Boss defeated
            self.game_over = True
            return True, 100 # Victory bonus
        if self.steps >= self.max_steps:
            self.game_over = True
            return True, -10 # Timeout penalty
        return False, 0
        
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = self._handle_input(action)
        reward += self._update_game_state()
        
        self.steps += 1
        terminated, term_reward = self._check_termination()
        reward += term_reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _render_game(self):
        # Camera offset
        cam_x, cam_y = self._iso_to_screen(self.player.x, self.player.y)
        offset_x = SCREEN_WIDTH // 2 - cam_x
        offset_y = SCREEN_HEIGHT // 2 - cam_y

        # Visible tile range
        player_pos_on_screen = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        
        # Render floor and walls
        view_dist_x = (SCREEN_WIDTH // TILE_W) + 5
        view_dist_y = (SCREEN_HEIGHT // TILE_H) + 5
        min_x = max(0, int(self.player.x - view_dist_x))
        max_x = min(self.map_width, int(self.player.x + view_dist_x))
        min_y = max(0, int(self.player.y - view_dist_y))
        max_y = min(self.map_height, int(self.player.y + view_dist_y))

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                if self.grid[x][y] == 1:
                    sx, sy = self._iso_to_screen(x, y)
                    sx += offset_x
                    sy += offset_y
                    
                    # Floor tile
                    pygame.gfxdraw.filled_polygon(self.screen, [
                        (sx, sy), (sx + TILE_W_HALF, sy + TILE_H_HALF),
                        (sx, sy + TILE_H), (sx - TILE_W_HALF, sy + TILE_H_HALF)
                    ], self.COLOR_FLOOR)
                    
                    # Walls
                    wall_height = TILE_H
                    if x > 0 and self.grid[x-1][y] == 0: # Left wall
                        p1 = (sx - TILE_W_HALF, sy + TILE_H_HALF)
                        p2 = (sx, sy + TILE_H)
                        pygame.gfxdraw.filled_polygon(self.screen, [p1, (p1[0], p1[1]-wall_height), (p2[0], p2[1]-wall_height), p2], self.COLOR_WALL_SIDE)
                    if y > 0 and self.grid[x][y-1] == 0: # Right wall (from iso perspective)
                        p1 = (sx, sy)
                        p2 = (sx + TILE_W_HALF, sy + TILE_H_HALF)
                        pygame.gfxdraw.filled_polygon(self.screen, [p1, (p1[0], p1[1]-wall_height), (p2[0], p2[1]-wall_height), p2], self.COLOR_WALL_TOP)

        # Draw stairs
        if self.stairs:
            sx, sy = self._iso_to_screen(self.stairs[0], self.stairs[1])
            rect = pygame.Rect(sx + offset_x - 10, sy + offset_y - 5, 20, 10)
            pygame.draw.rect(self.screen, (200, 200, 100), rect)

        # Collect and sort all drawable entities
        drawables = self.chests + self.enemies + [self.player] + ([self.boss] if self.boss else [])
        drawables = [d for d in drawables if d and getattr(d, 'alive', True)]
        drawables.sort(key=lambda e: e.world_y)

        for entity in drawables:
            iso_pos = self._iso_to_screen(entity.x, entity.y)
            screen_pos = (iso_pos[0] + offset_x, iso_pos[1] + offset_y)
            
            bob = math.sin(getattr(entity, 'bob_angle', 0)) * 2 if isinstance(entity, (Player, Enemy)) else 0
            
            if isinstance(entity, Player):
                # Glow effect
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], int(screen_pos[1] + bob), entity.size, (50, 180, 255, 80))
                # Player diamond
                points = [(screen_pos[0], screen_pos[1] - entity.size//2 + bob), (screen_pos[0] + entity.size//2, screen_pos[1] + bob),
                          (screen_pos[0], screen_pos[1] + entity.size//2 + bob), (screen_pos[0] - entity.size//2, screen_pos[1] + bob)]
                pygame.gfxdraw.filled_polygon(self.screen, points, entity.color)
            elif isinstance(entity, (Enemy, Boss)):
                # Shadow
                pygame.gfxdraw.filled_ellipse(self.screen, screen_pos[0], screen_pos[1] + entity.size // 2, entity.size, entity.size // 2, (0,0,0,100))
                # Body
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], int(screen_pos[1] + bob), entity.size, entity.color)
                entity.draw_health_bar(self.screen, offset_x, offset_y, screen_pos)
            elif isinstance(entity, Chest):
                color = (139, 69, 19) if not entity.opened else (80, 40, 10)
                rect = pygame.Rect(screen_pos[0] - 8, screen_pos[1] - 8, 16, 16)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (255, 215, 0), rect, 2)

        # Render projectiles, particles, text (on top)
        for p in self.projectiles:
            pos = (int(p.x + offset_x), int(p.y + offset_y))
            pygame.draw.circle(self.screen, p.color, pos, p.size)
        
        for p in self.particles:
            alpha = int(255 * (p.lifetime / p.max_lifetime))
            color = p.color + (alpha,)
            size = int(5 * (p.lifetime / p.max_lifetime))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p.x + offset_x), int(p.y + offset_y)), size)

        for t in self.floating_texts:
            alpha = int(255 * (t.lifetime / t.max_lifetime))
            text_surf = self.font_small.render(t.text, True, t.color)
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (t.x + offset_x - text_surf.get_width() // 2, t.y + offset_y))

    def _render_ui(self):
        # Player Health Bar
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, 204, 24))
        health_ratio = max(0, self.player.health / self.player.max_health)
        pygame.draw.rect(self.screen, (0, 200, 0), (12, 12, 200 * health_ratio, 20))
        health_text = self.font_medium.render(f"HP: {int(self.player.health)}/{self.player.max_health}", True, (255, 255, 255))
        self.screen.blit(health_text, (20, 14))

        # Score and Level
        score_text = self.font_large.render(f"Score: {self.score}", True, (255, 255, 100))
        self.screen.blit(score_text, (SCREEN_WIDTH - score_text.get_width() - 10, 10))
        level_text = self.font_large.render(f"Level: {self.level}", True, (200, 200, 255))
        self.screen.blit(level_text, (SCREEN_WIDTH - level_text.get_width() - 10, 40))

        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU DIED" if self.player.health <= 0 else "VICTORY!"
            end_text = self.font_large.render(msg, True, (255, 50, 50) if self.player.health <= 0 else (50, 255, 50))
            self.screen.blit(end_text, (SCREEN_WIDTH // 2 - end_text.get_width() // 2, SCREEN_HEIGHT // 2 - end_text.get_height() // 2))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level, "player_health": self.player.health}

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a separate display for human play
    render_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Dungeon Crawler")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)

        # Movement
        if keys[pygame.K_UP] and keys[pygame.K_LEFT]: action[0] = 3 # Iso up-left
        elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]: action[0] = 4 # Iso up-right
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]: action[0] = 1 # Iso down-left
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]: action[0] = 2 # Iso down-right
        elif keys[pygame.K_UP]: action[0] = 3
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 1
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Actions
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    pygame.quit()