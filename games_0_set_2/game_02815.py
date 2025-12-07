
# Generated: 2025-08-28T06:04:40.977690
# Source Brief: brief_02815.md
# Brief Index: 2815

        
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


# --- Constants ---
SCREEN_W, SCREEN_H = 640, 400
TILE_W, TILE_H = 48, 24
ISO_Z_SCALE = 16

# --- Colors ---
COLOR_BG = (25, 20, 35)
COLOR_FLOOR = (60, 50, 80)
COLOR_WALL_TOP = (110, 100, 130)
COLOR_WALL_SIDE = (85, 75, 105)

COLOR_PLAYER = (200, 255, 255)
COLOR_PLAYER_SHADOW = (0, 0, 0, 100)
COLOR_ENEMY = (255, 80, 80)
COLOR_BOSS = (255, 150, 80)
COLOR_POTION = (80, 255, 80)
COLOR_EXIT = (255, 220, 100)
COLOR_ATTACK = (255, 255, 255, 200)
COLOR_UI_TEXT = (220, 220, 240)
COLOR_DAMAGE_FLASH = (255, 0, 0, 180)

# --- Game Parameters ---
MAX_STEPS = 1000
FINAL_LEVEL = 5
PLAYER_MAX_HEALTH = 100
PLAYER_DAMAGE = 25
POTION_HEAL = 25
BASE_ENEMY_HEALTH = 20
BASE_ENEMY_DAMAGE = 5
BOSS_BASE_HEALTH = 150
BOSS_BASE_DAMAGE = 15
DUNGEON_SIZE = 20
DUNGEON_WALK_STEPS = 80
ENEMY_SIGHT_RANGE = 6

# --- Helper Functions and Classes ---

def to_iso(x, y):
    """Converts grid coordinates to isometric screen coordinates."""
    return (x - y) * TILE_W / 2, (x + y) * TILE_H / 2

class Particle:
    def __init__(self, x, y, z, color, np_random):
        self.x, self.y, self.z = x, y, z
        self.color = color
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.vz = np_random.uniform(2, 5)
        self.life = 20
        self.gravity = 0.3

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.z += self.vz
        self.vz -= self.gravity
        self.life -= 1
        return self.life > 0

class Entity:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.z, self.bob_angle = 0, random.uniform(0, 2 * math.pi)
        self.damage_flash = 0

    def update_anim(self):
        self.bob_angle += 0.1
        self.z = math.sin(self.bob_angle) * 2
        if self.damage_flash > 0:
            self.damage_flash -= 1

class Player(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.health, self.max_health = PLAYER_MAX_HEALTH, PLAYER_MAX_HEALTH
        self.damage = PLAYER_DAMAGE
        self.facing = 2  # 1:up, 2:down, 3:left, 4:right

class Enemy(Entity):
    def __init__(self, x, y, level):
        super().__init__(x, y)
        self.is_boss = False
        scale = 1 + 0.1 * (level - 1)
        self.max_health = int(BASE_ENEMY_HEALTH * scale)
        self.health = self.max_health
        self.damage = int(BASE_ENEMY_DAMAGE * scale)

class Boss(Enemy):
    def __init__(self, x, y, level):
        super().__init__(x, y, level)
        self.is_boss = True
        scale = 1 + 0.1 * (level - 1)
        self.max_health = int(BOSS_BASE_HEALTH * scale)
        self.health = self.max_health
        self.damage = int(BOSS_BASE_DAMAGE * scale)

class Potion(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.heal_amount = POTION_HEAL

class DungeonGenerator:
    def generate(self, size, steps, np_random):
        grid = {}
        x, y = size // 2, size // 2
        start_pos = (x, y)
        path = deque([(x, y)])
        grid[(x, y)] = 'floor'

        for _ in range(steps):
            dx, dy = np_random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            x, y = np.clip(x + dx, 1, size - 2), np.clip(y + dy, 1, size - 2)
            if (x, y) not in grid:
                grid[(x, y)] = 'floor'
                path.append((x, y))
        
        exit_pos = (x, y)
        
        walls = {}
        for (fx, fy) in grid:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if (dx, dy) == (0, 0): continue
                    nx, ny = fx + dx, fy + dy
                    if (nx, ny) not in grid:
                        walls[(nx, ny)] = 'wall'
        grid.update(walls)
        
        return grid, start_pos, exit_pos, list(path)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    user_guide = "Controls: ↑↓←→ to move. Press space to attack."
    game_description = "Explore a procedurally generated dungeon, battle monsters, and defeat the final boss."
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((SCREEN_W, SCREEN_H))
        self.font = pygame.font.Font(None, 24)
        self.dungeon_generator = DungeonGenerator()

        self.player = None
        self.enemies = []
        self.potions = []
        self.particles = []
        self.effects = []
        self.grid = {}
        self.exit_pos = (0, 0)
        
        self.steps = 0
        self.score = 0
        self.level = 1
        self.boss_defeated = False
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.level = 1
        self.boss_defeated = False
        self.particles = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.grid, start_pos, self.exit_pos, path = self.dungeon_generator.generate(
            DUNGEON_SIZE, DUNGEON_WALK_STEPS, self.np_random
        )
        
        self.player = Player(start_pos[0], start_pos[1])
        
        occupied_tiles = {start_pos, self.exit_pos}
        
        self.enemies = []
        if self.level == FINAL_LEVEL:
            self.enemies.append(Boss(self.exit_pos[0], self.exit_pos[1], self.level))
            self.exit_pos = None # No exit on boss level
        else:
            num_enemies = self.np_random.integers(2, 5) + self.level
            for _ in range(num_enemies):
                pos = self._get_random_floor_tile(path, occupied_tiles)
                if pos:
                    self.enemies.append(Enemy(pos[0], pos[1], self.level))
                    occupied_tiles.add(pos)

        self.potions = []
        num_potions = self.np_random.integers(1, 4)
        for _ in range(num_potions):
            pos = self._get_random_floor_tile(path, occupied_tiles)
            if pos:
                self.potions.append(Potion(pos[0], pos[1]))
                occupied_tiles.add(pos)
                
    def _get_random_floor_tile(self, path, occupied):
        attempts = 0
        while attempts < 50:
            pos = path[self.np_random.integers(0, len(path))]
            if pos not in occupied:
                return pos
            attempts += 1
        return None

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        self.effects = []

        # --- Player Action ---
        action_taken = False
        if space_held:
            # Attack
            action_taken = True
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(self.player.facing, (0, 0))
            target_pos = (self.player.x + dx, self.player.y + dy)
            self.effects.append({'type': 'attack', 'pos': target_pos, 'color': COLOR_ATTACK})
            # sfx: player_attack_swing.wav

            for enemy in self.enemies:
                if (enemy.x, enemy.y) == target_pos:
                    enemy.health -= self.player.damage
                    enemy.damage_flash = 5
                    # sfx: enemy_hit.wav
                    if enemy.health <= 0:
                        reward += 100 if enemy.is_boss else 10
                        self.score += 100 if enemy.is_boss else 10
                        if enemy.is_boss:
                            self.boss_defeated = True
                        self._create_particles(enemy.x, enemy.y, COLOR_ENEMY)
                        self.enemies.remove(enemy)
                        # sfx: enemy_die.wav
                    break
        elif movement > 0:
            # Move
            action_taken = True
            self.player.facing = movement
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            new_pos = (self.player.x + dx, self.player.y + dy)
            
            if self.grid.get(new_pos) == 'floor' and not any(e.x == new_pos[0] and e.y == new_pos[1] for e in self.enemies):
                self.player.x, self.player.y = new_pos
                # sfx: player_footstep.wav
            
            # Check for potion collection
            for potion in self.potions:
                if (potion.x, potion.y) == (self.player.x, self.player.y):
                    self.player.health = min(self.player.max_health, self.player.health + potion.heal_amount)
                    reward += 5
                    self.score += 5
                    self._create_particles(potion.x, potion.y, COLOR_POTION)
                    self.potions.remove(potion)
                    # sfx: potion_collect.wav
                    break
            
            # Check for level exit
            if self.exit_pos and (self.player.x, self.player.y) == self.exit_pos:
                reward += 20
                self.score += 20
                self.level += 1
                self._generate_level()
                # sfx: level_up.wav
        
        # --- Enemy Turn (only if player acted) ---
        if action_taken:
            for enemy in self.enemies:
                dist_to_player = math.hypot(self.player.x - enemy.x, self.player.y - enemy.y)
                
                if dist_to_player < 1.5: # Adjacent
                    self.player.health -= enemy.damage
                    self.player.damage_flash = 5
                    # sfx: player_hit.wav
                elif dist_to_player <= ENEMY_SIGHT_RANGE:
                    # Move towards player
                    dx, dy = np.sign(self.player.x - enemy.x), np.sign(self.player.y - enemy.y)
                    
                    # Try moving in the best direction
                    if dx != 0 and dy != 0: # Diagonal, pick one axis
                        if self.np_random.random() > 0.5:
                            move_pos = (enemy.x + dx, enemy.y)
                        else:
                            move_pos = (enemy.x, enemy.y + dy)
                    elif dx != 0:
                        move_pos = (enemy.x + dx, enemy.y)
                    elif dy != 0:
                        move_pos = (enemy.x, enemy.y + dy)
                    else: # On same tile (should not happen)
                        move_pos = (enemy.x, enemy.y)
                        
                    is_player_pos = (move_pos[0] == self.player.x and move_pos[1] == self.player.y)
                    is_occupied = any(e.x == move_pos[0] and e.y == move_pos[1] for e in self.enemies)
                    
                    if self.grid.get(move_pos) == 'floor' and not is_player_pos and not is_occupied:
                        enemy.x, enemy.y = move_pos

        # --- Update state ---
        self.steps += 1
        for p in self.particles[:]:
            if not p.update():
                self.particles.remove(p)
        for e in [self.player] + self.enemies + self.potions:
            e.update_anim()
        
        # --- Termination conditions ---
        if self.player.health <= 0:
            reward = -100
            terminated = True
        elif self.boss_defeated:
            reward = 100
            terminated = True
        elif self.steps >= MAX_STEPS:
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, x, y, color):
        iso_x, iso_y = to_iso(x, y)
        for _ in range(20):
            self.particles.append(Particle(iso_x, iso_y, -ISO_Z_SCALE, color, self.np_random))

    def _get_observation(self):
        self.screen.fill(COLOR_BG)
        
        cam_offset_x = SCREEN_W / 2 - (self.player.x - self.player.y) * TILE_W / 2
        cam_offset_y = SCREEN_H / 2 - (self.player.x + self.player.y) * TILE_H / 2

        # --- Render Game World ---
        draw_list = []
        for (gx, gy), type in self.grid.items():
            draw_list.append({'type': type, 'x': gx, 'y': gy, 'sort_key': gx + gy})
        
        for e in self.potions + self.enemies + [self.player]:
            draw_list.append({'type': 'entity', 'obj': e, 'sort_key': e.x + e.y + 0.5})

        draw_list.sort(key=lambda i: i['sort_key'])

        for item in draw_list:
            if item['type'] == 'floor':
                self._draw_iso_floor(item['x'], item['y'], COLOR_FLOOR, (cam_offset_x, cam_offset_y))
                if self.exit_pos and (item['x'], item['y']) == self.exit_pos:
                    self._draw_iso_floor(item['x'], item['y'], COLOR_EXIT, (cam_offset_x, cam_offset_y))
            elif item['type'] == 'wall':
                self._draw_iso_cube(item['x'], item['y'], (cam_offset_x, cam_offset_y))
            elif item['type'] == 'entity':
                self._draw_entity(item['obj'], (cam_offset_x, cam_offset_y))

        # --- Render Effects ---
        for effect in self.effects:
            if effect['type'] == 'attack':
                iso_x, iso_y = to_iso(effect['pos'][0], effect['pos'][1])
                self._draw_iso_floor(effect['pos'][0], effect['pos'][1], effect['color'], (cam_offset_x, cam_offset_y))

        for p in self.particles:
            pos = (int(p.x + cam_offset_x), int(p.y + cam_offset_y - p.z))
            pygame.draw.circle(self.screen, p.color, pos, int(max(0, p.life / 5)))
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_iso_floor(self, gx, gy, color, cam_offset):
        iso_x, iso_y = to_iso(gx, gy)
        points = [
            (iso_x - TILE_W / 2 + cam_offset[0], iso_y + cam_offset[1]),
            (iso_x + cam_offset[0], iso_y - TILE_H / 2 + cam_offset[1]),
            (iso_x + TILE_W / 2 + cam_offset[0], iso_y + cam_offset[1]),
            (iso_x + cam_offset[0], iso_y + TILE_H / 2 + cam_offset[1])
        ]
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)

    def _draw_iso_cube(self, gx, gy, cam_offset):
        iso_x, iso_y = to_iso(gx, gy)
        x, y = iso_x + cam_offset[0], iso_y + cam_offset[1]
        
        top_face = [
            (x - TILE_W / 2, y - ISO_Z_SCALE), (x, y - TILE_H / 2 - ISO_Z_SCALE),
            (x + TILE_W / 2, y - ISO_Z_SCALE), (x, y + TILE_H / 2 - ISO_Z_SCALE)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in top_face], COLOR_WALL_TOP)

        right_face = [
            (x, y + TILE_H / 2 - ISO_Z_SCALE), (x + TILE_W/2, y - ISO_Z_SCALE),
            (x + TILE_W/2, y), (x, y + TILE_H/2)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in right_face], COLOR_WALL_SIDE)

        left_face = [
            (x - TILE_W / 2, y - ISO_Z_SCALE), (x, y - TILE_H / 2 - ISO_Z_SCALE),
            (x, y - TILE_H/2), (x - TILE_W/2, y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in left_face], COLOR_WALL_SIDE)

    def _draw_entity(self, entity, cam_offset):
        iso_x, iso_y = to_iso(entity.x, entity.y)
        sx, sy = int(iso_x + cam_offset[0]), int(iso_y + cam_offset[1] - entity.z)
        
        # Shadow
        shadow_surface = pygame.Surface((TILE_W, TILE_H), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, COLOR_PLAYER_SHADOW, (0, 0, TILE_W-10, TILE_H-10))
        self.screen.blit(shadow_surface, (sx - (TILE_W/2), sy - (TILE_H/2)))

        # Determine color and size
        if isinstance(entity, Player):
            color, radius = COLOR_PLAYER, 8
        elif isinstance(entity, Boss):
            color, radius = COLOR_BOSS, 12
        elif isinstance(entity, Enemy):
            color, radius = COLOR_ENEMY, 7
        elif isinstance(entity, Potion):
            color, radius = COLOR_POTION, 5
        else: return
        
        # Main body
        pygame.draw.circle(self.screen, color, (sx, sy - ISO_Z_SCALE), radius)
        
        # Damage flash
        if entity.damage_flash > 0:
            flash_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(flash_surf, COLOR_DAMAGE_FLASH, (radius, radius), radius)
            self.screen.blit(flash_surf, (sx - radius, sy - ISO_Z_SCALE - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        health_text = f"Health: {self.player.health} / {self.player.max_health}"
        level_text = f"Level: {self.level}"
        score_text = f"Score: {self.score}"
        
        health_surf = self.font.render(health_text, True, COLOR_UI_TEXT)
        level_surf = self.font.render(level_text, True, COLOR_UI_TEXT)
        score_surf = self.font.render(score_text, True, COLOR_UI_TEXT)
        
        self.screen.blit(health_surf, (10, 10))
        self.screen.blit(level_surf, (10, 35))
        self.screen.blit(score_surf, (SCREEN_W - score_surf.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "player_health": self.player.health,
            "boss_defeated": self.boss_defeated,
        }

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Isometric Dungeon Crawler")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    
    terminated = False
    
    print(GameEnv.user_guide)

    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # Since it's turn based, we only step if an action is taken
        if any(action):
            obs, reward, terminated, _, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward}, Terminated: {terminated}")
        else: # Allow rendering even on no-op
             obs = env._get_observation()

        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Wait for next action
        pygame.time.wait(100) # Prevents fast repeated actions

    pygame.quit()
    print("Game Over!")