# Generated: 2025-08-27T12:57:10.314649
# Source Brief: brief_00210.md
# Brief Index: 210

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space to attack in the direction you are facing."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A grid-based dungeon crawler. Defeat enemies, collect gold, and find the exit to progress."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    TILE_SIZE = 40
    SCREEN_WIDTH, SCREEN_HEIGHT = GRID_WIDTH * TILE_SIZE, GRID_HEIGHT * TILE_SIZE

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_WALL_MAIN = (60, 60, 80)
    COLOR_WALL_ACCENT = (80, 80, 100)
    COLOR_FLOOR = (40, 40, 55)
    COLOR_EXIT = (255, 255, 255)
    COLOR_EXIT_PORTAL = (200, 100, 255)
    
    COLOR_PLAYER = (50, 200, 50)
    COLOR_PLAYER_ACCENT = (150, 255, 150)
    
    COLOR_GOLD = (255, 223, 0)
    
    ENEMY_COLORS = {
        'chaser': ((180, 50, 50), (220, 100, 100)),
        'patrol': ((50, 50, 180), (100, 100, 220)),
        'random': ((180, 100, 50), (220, 150, 100)),
    }
    
    # Game parameters
    PLAYER_MAX_HEALTH = 100
    PLAYER_ATTACK_DAMAGE = 25
    ENEMY_BASE_HEALTH = 20
    ENEMY_BASE_DAMAGE = 10
    MAX_STEPS = 1000
    FINAL_ROOM = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # State variables are initialized in reset()
        self.player_pos = (0, 0)
        self.player_health = 0
        self.player_gold = 0
        self.player_facing_dir = (0, 1)
        self.room_number = 1
        self.enemies = []
        self.gold_items = set()
        self.walls = set()
        self.exit_pos = (0, 0)
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_player_dist_to_exit = 0
        self.vfx = [] # Visual effects list
        
        # self.reset() is called by the environment runner
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_gold = 0
        self.room_number = 1
        self.game_over = False
        self.win = False
        self.player_facing_dir = (1, 0) # Start facing right
        
        self._generate_room()
        
        self.last_player_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
        
        return self._get_observation(), self._get_info()

    def _generate_room(self):
        self.walls = set()
        self.gold_items = set()
        self.enemies = []
        self.vfx = []

        # Create border walls
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, 0))
            self.walls.add((x, self.GRID_HEIGHT - 1))
        for y in range(self.GRID_HEIGHT):
            self.walls.add((0, y))
            self.walls.add((self.GRID_WIDTH - 1, y))

        # Player and Exit positions
        self.player_pos = (1, self.GRID_HEIGHT // 2)
        self.exit_pos = (self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2)
        
        # Make sure start/exit are not walled in
        if self.player_pos in self.walls: self.walls.remove(self.player_pos)
        if self.exit_pos in self.walls: self.walls.remove(self.exit_pos)

        # Generate some random internal walls
        num_walls = self.room_number * 2 + 2
        for _ in range(num_walls):
            pos = (self.np_random.integers(1, self.GRID_WIDTH - 1), self.np_random.integers(1, self.GRID_HEIGHT - 1))
            if pos != self.player_pos and pos != self.exit_pos:
                self.walls.add(pos)

        # Ensure connectivity with a simple corridor
        for x in range(1, self.GRID_WIDTH - 1):
            if (x, self.GRID_HEIGHT // 2) in self.walls:
                self.walls.remove((x, self.GRID_HEIGHT // 2))

        # Get all valid spawn points
        valid_spawns = []
        for x in range(1, self.GRID_WIDTH - 1):
            for y in range(1, self.GRID_HEIGHT - 1):
                pos = (x, y)
                if pos not in self.walls and pos != self.player_pos and pos != self.exit_pos:
                    valid_spawns.append(pos)
        
        # Use a consistent shuffling method for reproducibility if needed, but random.shuffle is fine for now
        self.np_random.shuffle(valid_spawns)

        # Spawn enemies
        num_enemies = min(len(valid_spawns), self.room_number + self.np_random.integers(0, 2))
        enemy_health = self.ENEMY_BASE_HEALTH + (self.room_number - 1) * 2
        for _ in range(num_enemies):
            if not valid_spawns: break
            pos = valid_spawns.pop()
            enemy_type = self.np_random.choice(list(self.ENEMY_COLORS.keys()))
            self.enemies.append(Enemy(pos, enemy_health, enemy_type))
            
        # Spawn gold
        num_gold = min(len(valid_spawns), 5)
        for _ in range(num_gold):
            if not valid_spawns: break
            pos = valid_spawns.pop()
            self.gold_items.add(pos)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # --- Player Turn ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # 1. Player Movement
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            self.player_facing_dir = (dx, dy)
            old_pos = self.player_pos
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            if new_pos not in self.walls:
                self.player_pos = new_pos
                self._add_vfx('dust', old_pos, 5)

        # 2. Player Interaction with Tile
        if self.player_pos in self.gold_items:
            self.gold_items.remove(self.player_pos)
            self.player_gold += 1
            reward += 0.5
            self._add_vfx('collect', self.player_pos, 8)
            # sfx: coin collect
        
        if self.player_pos == self.exit_pos:
            if self.room_number == self.FINAL_ROOM:
                self.win = True
            else:
                self.room_number += 1
                self._generate_room()
                reward += 10 # Reward for clearing a room
                self.last_player_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
        
        # 3. Player Attack
        if space_held:
            target_pos = (self.player_pos[0] + self.player_facing_dir[0], self.player_pos[1] + self.player_facing_dir[1])
            self._add_vfx('slash', target_pos, 4, data={'dir': self.player_facing_dir})
            # sfx: sword swing
            for enemy in self.enemies:
                if enemy.pos == target_pos:
                    enemy.take_damage(self.PLAYER_ATTACK_DAMAGE)
                    self._add_vfx('hit', enemy.pos, 5)
                    if enemy.health <= 0:
                        reward += 1.0

        # --- Enemy Turn ---
        for enemy in self.enemies:
            if enemy.health > 0:
                action = enemy.get_action(self.player_pos, self.walls, self.GRID_WIDTH, self.GRID_HEIGHT)
                if action['type'] == 'move':
                    enemy.pos = action['pos']
                elif action['type'] == 'attack':
                    self.player_health -= self.ENEMY_BASE_DAMAGE
                    self._add_vfx('hit', self.player_pos, 5, color=(255,0,0))
                    # sfx: player hurt
        
        # --- Cleanup ---
        self.enemies = [e for e in self.enemies if e.health > 0]
        
        # --- Reward Calculation ---
        new_dist = self._manhattan_distance(self.player_pos, self.exit_pos)
        dist_change = self.last_player_dist_to_exit - new_dist
        if dist_change > 0:
            reward += 0.1
        elif dist_change < 0:
            reward -= 0.1
        self.last_player_dist_to_exit = new_dist

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.player_health <= 0:
            self.player_health = 0
            self.game_over = True
            terminated = True
            reward = -100
        elif self.win:
            self.game_over = True
            terminated = True
            reward = 100
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Gymnasium standard: terminated can be true if truncated is true

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw floor
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                pygame.draw.rect(self.screen, self.COLOR_FLOOR, (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        # Draw exit
        ex, ey = self.exit_pos
        rect = pygame.Rect(ex * self.TILE_SIZE, ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT_PORTAL, rect)
        if self.room_number == self.FINAL_ROOM:
            pygame.gfxdraw.filled_circle(self.screen, int(rect.centerx), int(rect.centery), int(self.TILE_SIZE*0.3), self.COLOR_EXIT)

        # Draw walls
        for x, y in self.walls:
            rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL_MAIN, rect)
            pygame.draw.rect(self.screen, self.COLOR_WALL_ACCENT, rect.inflate(-4, -4))

        # Draw gold
        for x, y in self.gold_items:
            cx, cy = int((x + 0.5) * self.TILE_SIZE), int((y + 0.5) * self.TILE_SIZE)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, int(self.TILE_SIZE * 0.25), self.COLOR_GOLD)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, int(self.TILE_SIZE * 0.25), self.COLOR_GOLD)

        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen, self.TILE_SIZE, self.ENEMY_COLORS[enemy.type])

        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(px * self.TILE_SIZE, py * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-4,-4))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_ACCENT, player_rect.inflate(-12,-12))
        
        # Draw player facing indicator
        face_x = player_rect.centerx + self.player_facing_dir[0] * self.TILE_SIZE * 0.3
        face_y = player_rect.centery + self.player_facing_dir[1] * self.TILE_SIZE * 0.3
        pygame.draw.circle(self.screen, (255,255,255), (int(face_x), int(face_y)), 3)

        # Draw and update VFX
        new_vfx = []
        for vfx in self.vfx:
            vfx.update_and_draw(self.screen, self.TILE_SIZE)
            if not vfx.done:
                new_vfx.append(vfx)
        self.vfx = new_vfx

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        health_bar_width = 200
        health_bar_rect = pygame.Rect(10, 10, int(health_bar_width * health_ratio), 20)
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, (200, 50, 50), health_bar_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), (10, 10, health_bar_width, 20), 1)
        health_text = self.font_small.render(f"{self.player_health}/{self.PLAYER_MAX_HEALTH}", True, (255,255,255))
        self.screen.blit(health_text, (15, 12))

        # Gold Counter
        gold_text = self.font_small.render(f"Gold: {self.player_gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (self.SCREEN_WIDTH - gold_text.get_width() - 10, 10))

        # Room Number
        room_text = self.font_large.render(f"Room {self.room_number}/{self.FINAL_ROOM}", True, (200, 200, 200))
        self.screen.blit(room_text, (self.SCREEN_WIDTH // 2 - room_text.get_width() // 2, self.SCREEN_HEIGHT - room_text.get_height() - 5))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH//2 - end_text.get_width()//2, self.SCREEN_HEIGHT//2 - end_text.get_height()//2))

    def _get_info(self):
        return {
            "score": self.player_gold,
            "steps": self.steps,
            "room": self.room_number,
            "health": self.player_health
        }

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _add_vfx(self, vfx_type, pos, duration, color=(255,255,255), data=None):
        self.vfx.append(VFX(vfx_type, pos, duration, color, data))

class Enemy:
    def __init__(self, pos, health, enemy_type):
        self.pos = pos
        self.health = health
        self.max_health = health
        self.type = enemy_type
        self.damage_flash = 0
        self.patrol_dir = 1

    def take_damage(self, amount):
        self.health -= amount
        self.damage_flash = 5 # frames

    def draw(self, surface, tile_size, colors):
        main_color, accent_color = colors
        rect = pygame.Rect(self.pos[0] * tile_size, self.pos[1] * tile_size, tile_size, tile_size)
        pygame.draw.rect(surface, main_color, rect.inflate(-4,-4))
        pygame.draw.rect(surface, accent_color, rect.inflate(-12,-12))
        if self.damage_flash > 0:
            flash_surface = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
            flash_surface.fill((255, 100, 100, 150))
            surface.blit(flash_surface, rect.topleft)
            self.damage_flash -= 1

    def get_action(self, player_pos, walls, grid_w, grid_h):
        dist_to_player = abs(self.pos[0] - player_pos[0]) + abs(self.pos[1] - player_pos[1])

        if dist_to_player == 1:
            return {'type': 'attack'}
        
        if self.type == 'chaser' and dist_to_player < 6:
            return self._move_towards(player_pos, walls)
        elif self.type == 'patrol':
            return self._patrol(walls)
        else: # random
            return self._move_randomly(walls)

    def _get_valid_moves(self, walls):
        moves = []
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            new_pos = (self.pos[0] + dx, self.pos[1] + dy)
            if new_pos not in walls:
                moves.append(new_pos)
        return moves

    def _move_towards(self, target_pos, walls):
        valid_moves = self._get_valid_moves(walls)
        if not valid_moves: return {'type': 'none'}

        best_move = self.pos
        min_dist = abs(self.pos[0] - target_pos[0]) + abs(self.pos[1] - target_pos[1])

        for move in valid_moves:
            dist = abs(move[0] - target_pos[0]) + abs(move[1] - target_pos[1])
            if dist < min_dist:
                min_dist = dist
                best_move = move
        
        return {'type': 'move', 'pos': best_move}

    def _patrol(self, walls):
        next_pos = (self.pos[0] + self.patrol_dir, self.pos[1])
        if next_pos in walls:
            self.patrol_dir *= -1
            next_pos = (self.pos[0] + self.patrol_dir, self.pos[1])
        
        if next_pos not in walls:
            return {'type': 'move', 'pos': next_pos}
        return {'type': 'none'}

    def _move_randomly(self, walls):
        valid_moves = self._get_valid_moves(walls)
        if valid_moves:
            return {'type': 'move', 'pos': random.choice(valid_moves)}
        return {'type': 'none'}

class VFX:
    def __init__(self, vfx_type, pos, duration, color, data=None):
        self.type = vfx_type
        self.pos = pos # grid coordinates
        self.duration = duration
        self.max_duration = duration
        self.color = color
        self.data = data if data is not None else {}
        self.done = False

    def update_and_draw(self, surface, tile_size):
        if self.done: return
        self.duration -= 1
        if self.duration <= 0:
            self.done = True

        progress = 1.0 - (self.duration / self.max_duration)
        px, py = (self.pos[0] + 0.5) * tile_size, (self.pos[1] + 0.5) * tile_size

        if self.type == 'hit':
            radius = int(progress * tile_size * 0.5)
            alpha = int(255 * (1 - progress))
            pygame.gfxdraw.filled_circle(surface, int(px), int(py), radius, self.color + (alpha,))
        elif self.type == 'collect':
            for i in range(5):
                angle = 2 * math.pi * (i / 5 + progress * 2)
                dist = (1 - progress) * tile_size * 0.7
                x = px + math.cos(angle) * dist
                y = py + math.sin(angle) * dist
                pygame.draw.circle(surface, GameEnv.COLOR_GOLD, (int(x), int(y)), 2)
        elif self.type == 'slash':
            start_angle = -math.pi/2
            end_angle = math.pi/2
            angle = start_angle + (end_angle - start_angle) * progress
            
            dx, dy = self.data.get('dir', (1,0))
            if dy != 0: # Vertical
                start_pos = (px - tile_size//2, py)
                end_pos = (px + tile_size//2, py)
            else: # Horizontal
                start_pos = (px, py - tile_size//2)
                end_pos = (px, py + tile_size//2)

            pygame.draw.line(surface, (255,255,255), start_pos, end_pos, int(8 * (1 - progress)))
        elif self.type == 'dust':
            radius = int(progress * tile_size * 0.2)
            alpha = int(100 * (1 - progress))
            color = (150, 150, 150, alpha)
            pygame.gfxdraw.filled_circle(surface, int(px), int(py), radius, color)