import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:45:29.483508
# Source Brief: brief_01265.md
# Brief Index: 1265
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "As a stealthy ninja, match tiles to move, throw shurikens, and use disguises. "
        "Evade enemy patrols and reach the goal in each level."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to select movement tiles. Hold space to use shurikens and shift to activate your disguise. "
        "Match three tiles in a row or column to perform an action."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_AREA_HEIGHT = 320 # Top part for the game
    UI_AREA_HEIGHT = 80   # Bottom part for the tile board

    # Game Grid
    GRID_COLS = 20
    GRID_ROWS = 10
    TILE_SIZE = 32 # This applies to the game world grid, not the UI tile board

    # Colors
    COLOR_BG = (10, 4, 26) # #0a041a
    COLOR_PLAYER = (0, 255, 255) # Cyan
    COLOR_ENEMY = (255, 68, 68) # Red
    COLOR_GOAL = (68, 255, 68) # Green
    COLOR_WALL = (40, 20, 80)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (20, 10, 50)
    COLOR_TILE_BG = (30, 20, 60)
    COLOR_TILE_HIGHLIGHT = (255, 255, 0)

    # Tile Types & Colors
    TILE_TYPES = {
        'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'SHURIKEN': 5, 'DISGUISE': 6
    }
    TILE_COLORS = {
        1: (0, 150, 255), 2: (0, 150, 255), 3: (0, 150, 255), 4: (0, 150, 255), # Movement: Blue
        5: (255, 100, 0), # Shuriken: Orange
        6: (200, 0, 255)  # Disguise: Purple
    }

    # Rewards
    REWARD_MATCH_SUCCESS = 0.1
    REWARD_MATCH_FAIL = -0.1
    REWARD_ELIMINATE_ENEMY = 10.0
    REWARD_DETECTED = -10.0
    REWARD_WIN_LEVEL = 100.0

    # Game parameters
    MAX_STEPS = 2000
    ENEMY_BASE_SPEED = 0.5
    ENEMY_VISION_RADIUS = 4.5 # in grid units
    ENEMY_VISION_ANGLE = 45 # degrees
    SHURIKEN_SPEED = 12 # pixels per step
    DISGUISE_DURATION = 150 # steps

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
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 24)

        # Persistent state across resets
        self._level_number = 1
        
        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.enemies = []
        self.shurikens = []
        self.particles = []
        self.goal_pos = None
        self.tile_board = []
        self.shuriken_count = 0
        self.disguise_timer = 0
        self.last_action_feedback = None # For UI feedback

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Level setup
        self._generate_level()

        self.shurikens = []
        self.particles = []
        self.shuriken_count = 3 + self._level_number // 2
        self.disguise_timer = 0
        self.last_action_feedback = None
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- 1. Process Player Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        action_performed = False
        # Prioritize special actions over movement
        if space_held and self.shuriken_count > 0:
            action_performed, _ = self._process_tile_match(self.TILE_TYPES['SHURIKEN'])
            if action_performed:
                self.shuriken_count -= 1
                self._spawn_shuriken()
                reward += self.REWARD_MATCH_SUCCESS
                self.last_action_feedback = ("SHURIKEN", True)
            else:
                self.last_action_feedback = ("SHURIKEN", False)
        
        elif shift_held and not action_performed:
            action_performed, _ = self._process_tile_match(self.TILE_TYPES['DISGUISE'])
            if action_performed:
                self.disguise_timer = self.DISGUISE_DURATION
                reward += self.REWARD_MATCH_SUCCESS
                self.last_action_feedback = ("DISGUISE", True)
            else:
                self.last_action_feedback = ("DISGUISE", False)

        elif movement > 0 and not action_performed:
            action_map = {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT'}
            move_intent = self.TILE_TYPES[action_map[movement]]
            action_performed, _ = self._process_tile_match(move_intent)
            if action_performed:
                dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
                self.player['target_x'] = max(0, min(self.GRID_COLS - 1, self.player['x'] + dx))
                self.player['target_y'] = max(0, min(self.GRID_ROWS - 1, self.player['y'] + dy))
                self.player['last_dir'] = (dx, dy)
                reward += self.REWARD_MATCH_SUCCESS
                self.last_action_feedback = (action_map[movement], True)
            else:
                self.last_action_feedback = (action_map[movement], False)
        
        if not action_performed and self.last_action_feedback and not self.last_action_feedback[1]:
             reward += self.REWARD_MATCH_FAIL

        # --- 2. Update Game State ---
        self._update_player_movement()
        self._update_shurikens()
        reward += self._update_enemies()
        self._update_particles()
        if self.disguise_timer > 0:
            self.disguise_timer -= 1

        # --- 3. Calculate Rewards & Termination ---
        # Enemy detection reward is handled in _update_enemies
        
        terminated = self.game_over
        
        # Win condition
        player_grid_x = int(round(self.player['px'] / self.TILE_SIZE))
        player_grid_y = int(round(self.player['py'] / self.TILE_SIZE))
        if player_grid_x == self.goal_pos[0] and player_grid_y == self.goal_pos[1]:
            reward += self.REWARD_WIN_LEVEL
            terminated = True
            self._level_number += 1 # Progress to next level on next reset
            # 'WIN' feedback
            self.last_action_feedback = ("WIN", True)

        # Max steps condition
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True


        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_level(self):
        # Player
        self.player = {
            'x': 1, 'y': self.GRID_ROWS // 2,
            'px': 1 * self.TILE_SIZE + self.TILE_SIZE / 2,
            'py': 1 * self.TILE_SIZE + self.TILE_SIZE / 2,
            'target_x': 1, 'target_y': self.GRID_ROWS // 2,
            'last_dir': (1, 0) # Facing right
        }

        # Goal
        self.goal_pos = (self.GRID_COLS - 2, self.GRID_ROWS // 2)

        # Enemies
        self.enemies = []
        num_enemies = 1 + (self._level_number - 1) // 2
        for i in range(num_enemies):
            path_type = self.np_random.choice(['horizontal', 'vertical', 'box'])
            start_x = self.np_random.integers(5, self.GRID_COLS - 5)
            start_y = self.np_random.integers(2, self.GRID_ROWS - 2)
            
            path = []
            if path_type == 'horizontal':
                path = [(start_x, start_y), (start_x + 4, start_y)]
            elif path_type == 'vertical':
                path = [(start_x, start_y), (start_x, start_y + 3)]
            else: # box
                path = [(start_x, start_y), (start_x + 3, start_y), (start_x + 3, start_y + 3), (start_x, start_y + 3)]

            self.enemies.append({
                'x': path[0][0], 'y': path[0][1],
                'px': path[0][0] * self.TILE_SIZE + self.TILE_SIZE / 2,
                'py': path[0][1] * self.TILE_SIZE + self.TILE_SIZE / 2,
                'path': path,
                'path_index': 0,
                'state': 'neutral', # neutral, suspicious, alerted
                'dir': (1, 0)
            })

        # Tile Board
        self._generate_tile_board()

    def _generate_tile_board(self):
        self.tile_board = []
        # Ensure at least one of each action type is available for matching
        guaranteed_types = [
            self.TILE_TYPES['UP'], self.TILE_TYPES['DOWN'], self.TILE_TYPES['LEFT'], self.TILE_TYPES['RIGHT'],
            self.TILE_TYPES['SHURIKEN'], self.TILE_TYPES['DISGUISE']
        ]
        self.np_random.shuffle(guaranteed_types)

        for r in range(3):
            row = []
            for c in range(3):
                if guaranteed_types:
                    tile_type = guaranteed_types.pop()
                else:
                    tile_type = self.np_random.choice(list(self.TILE_TYPES.values()))
                row.append({'type': tile_type, 'y_offset': -self.UI_AREA_HEIGHT, 'matched': False})
            self.tile_board.append(row)
        
        # Shuffle board to make it random but fair
        flat_list = [tile for row in self.tile_board for tile in row]
        self.np_random.shuffle(flat_list)
        self.tile_board = [flat_list[i*3:(i+1)*3] for i in range(3)]


    def _process_tile_match(self, intent_type):
        matches = []
        # Check rows
        for r in range(3):
            if all(not self.tile_board[r][c]['matched'] and self.tile_board[r][c]['type'] == intent_type for c in range(3)):
                for c in range(3):
                    matches.append((r, c))
        # Check columns
        if not matches:
            for c in range(3):
                if all(not self.tile_board[r][c]['matched'] and self.tile_board[r][c]['type'] == intent_type for r in range(3)):
                    for r in range(3):
                        matches.append((r, c))
        
        if matches:
            unique_matches = list(set(matches))
            for r, c in unique_matches:
                if not self.tile_board[r][c]['matched']:
                    self.tile_board[r][c]['matched'] = True
                    self._spawn_particles(c * 60 + 190 + 30, self.SCREEN_HEIGHT - self.UI_AREA_HEIGHT / 2, self.TILE_COLORS[intent_type], 15)
            self._refill_board()
            return True, matches
        return False, []

    def _refill_board(self):
        for c in range(3):
            empty_count = 0
            for r in range(2, -1, -1):
                if self.tile_board[r][c]['matched']:
                    empty_count += 1
                elif empty_count > 0:
                    self.tile_board[r + empty_count][c] = self.tile_board[r][c]
                    self.tile_board[r + empty_count][c]['y_offset'] = -empty_count * 60
            
            for r in range(empty_count):
                self.tile_board[r][c] = {
                    'type': self.np_random.choice(list(self.TILE_TYPES.values())),
                    'y_offset': -self.UI_AREA_HEIGHT,
                    'matched': False
                }

    def _update_player_movement(self):
        target_px = self.player['target_x'] * self.TILE_SIZE + self.TILE_SIZE / 2
        target_py = self.player['target_y'] * self.TILE_SIZE + self.TILE_SIZE / 2
        
        # Interpolate position for smooth movement
        self.player['px'] += (target_px - self.player['px']) * 0.5
        self.player['py'] += (target_py - self.player['py']) * 0.5

        if abs(target_px - self.player['px']) < 1 and abs(target_py - self.player['py']) < 1:
            self.player['x'] = self.player['target_x']
            self.player['y'] = self.player['target_y']
            self.player['px'] = target_px
            self.player['py'] = target_py

    def _spawn_shuriken(self):
        # sound: whoosh
        self.shurikens.append({
            'x': self.player['px'],
            'y': self.player['py'],
            'dir': self.player['last_dir'],
            'life': 100
        })

    def _update_shurikens(self):
        reward = 0
        for shuriken in self.shurikens[:]:
            shuriken['x'] += shuriken['dir'][0] * self.SHURIKEN_SPEED
            shuriken['y'] += shuriken['dir'][1] * self.SHURIKEN_SPEED
            shuriken['life'] -= 1

            if shuriken['life'] <= 0 or not (0 < shuriken['x'] < self.SCREEN_WIDTH and 0 < shuriken['y'] < self.GAME_AREA_HEIGHT):
                self.shurikens.remove(shuriken)
                continue
            
            # Collision with enemies
            shuriken_rect = pygame.Rect(shuriken['x'] - 4, shuriken['y'] - 4, 8, 8)
            for enemy in self.enemies[:]:
                enemy_rect = pygame.Rect(enemy['px'] - self.TILE_SIZE/2, enemy['py'] - self.TILE_SIZE/2, self.TILE_SIZE, self.TILE_SIZE)
                if shuriken_rect.colliderect(enemy_rect):
                    # sound: enemy_hit
                    self._spawn_particles(enemy['px'], enemy['py'], self.COLOR_ENEMY, 30)
                    self.enemies.remove(enemy)
                    self.shurikens.remove(shuriken)
                    reward += self.REWARD_ELIMINATE_ENEMY
                    break
        return reward

    def _update_enemies(self):
        detection_reward = 0
        speed_multiplier = 1.0 + (self.steps // 500) * 0.05

        for enemy in self.enemies:
            # Movement
            path = enemy['path']
            target_node = path[enemy['path_index']]
            target_px = target_node[0] * self.TILE_SIZE + self.TILE_SIZE / 2
            target_py = target_node[1] * self.TILE_SIZE + self.TILE_SIZE / 2

            dist = math.hypot(target_px - enemy['px'], target_py - enemy['py'])
            if dist < 2:
                enemy['path_index'] = (enemy['path_index'] + 1) % len(path)
            else:
                angle = math.atan2(target_py - enemy['py'], target_px - enemy['px'])
                dx = math.cos(angle)
                dy = math.sin(angle)
                enemy['px'] += dx * self.ENEMY_BASE_SPEED * speed_multiplier
                enemy['py'] += dy * self.ENEMY_BASE_SPEED * speed_multiplier
                if abs(dx) > abs(dy):
                    enemy['dir'] = (np.sign(dx), 0)
                else:
                    enemy['dir'] = (0, np.sign(dy))

            enemy['x'] = int(enemy['px'] // self.TILE_SIZE)
            enemy['y'] = int(enemy['py'] // self.TILE_SIZE)
            
            # Detection logic
            dist_to_player = math.hypot(self.player['x'] - enemy['x'], self.player['y'] - enemy['y'])
            
            can_see_player = False
            if dist_to_player < self.ENEMY_VISION_RADIUS:
                # Check if player is within the vision cone
                enemy_angle = math.atan2(enemy['dir'][1], enemy['dir'][0])
                player_angle = math.atan2(self.player['y'] - enemy['y'], self.player['x'] - enemy['x'])
                angle_diff = (player_angle - enemy_angle + math.pi) % (2 * math.pi) - math.pi
                if abs(angle_diff) < math.radians(self.ENEMY_VISION_ANGLE):
                    can_see_player = True # Simplified, no raycasting for walls
            
            if can_see_player and self.disguise_timer <= 0:
                enemy['state'] = 'alerted'
                self.game_over = True
                detection_reward = self.REWARD_DETECTED
                # sound: detected_alarm
            elif dist_to_player < self.ENEMY_VISION_RADIUS / 2 and self.disguise_timer <= 0:
                enemy['state'] = 'suspicious'
            else:
                enemy['state'] = 'neutral'
        
        return detection_reward
    
    def _spawn_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vx'] *= 0.95
            p['vy'] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_city()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self._level_number,
            "shurikens": self.shuriken_count,
            "disguise_timer": self.disguise_timer,
        }

    def _render_background_city(self):
        for i in range(30):
            x = (i * 37 + self.steps * 0.1) % (self.SCREEN_WIDTH + 100) - 50
            h = 50 + (i * 13) % 100
            w = 20 + (i * 7) % 15
            color = (
                self.COLOR_BG[0] + 10 + (i*3)%10,
                self.COLOR_BG[1] + 5 + (i*5)%10,
                self.COLOR_BG[2] + 20 + (i*7)%20
            )
            pygame.draw.rect(self.screen, color, (x, self.GAME_AREA_HEIGHT - h, w, h))
            if i % 3 == 0: # Neon windows
                win_y = self.GAME_AREA_HEIGHT - self.np_random.uniform(10, h-10)
                win_color = self.np_random.choice([(255,0,150), (0,255,255)])
                pygame.draw.rect(self.screen, win_color, (x + 3, win_y, 2, 2))


    def _render_game(self):
        # Goal
        gx, gy = self.goal_pos
        goal_rect = pygame.Rect(gx * self.TILE_SIZE, gy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, goal_rect, 2)
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 40))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p['x'] - p['size'], p['y'] - p['size']), special_flags=pygame.BLEND_RGBA_ADD)

        # Enemies
        for enemy in self.enemies:
            ex, ey = int(enemy['px']), int(enemy['py'])
            # Vision cone
            if self.disguise_timer <= 0:
                state_color = {'neutral': (255, 255, 0), 'suspicious': (255, 165, 0), 'alerted': (255, 0, 0)}[enemy['state']]
                angle_rad = math.atan2(enemy['dir'][1], enemy['dir'][0])
                radius_px = self.ENEMY_VISION_RADIUS * self.TILE_SIZE
                angle_deg = math.radians(self.ENEMY_VISION_ANGLE)
                p1 = (ex, ey)
                p2 = (ex + radius_px * math.cos(angle_rad - angle_deg), ey + radius_px * math.sin(angle_rad - angle_deg))
                p3 = (ex + radius_px * math.cos(angle_rad + angle_deg), ey + radius_px * math.sin(angle_rad + angle_deg))
                pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], (*state_color, 50))
                pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], (*state_color, 50))

            # Body
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, (ex, ey), 10)
            pygame.gfxdraw.aacircle(self.screen, ex, ey, 10, self.COLOR_ENEMY)

        # Shurikens
        for s in self.shurikens:
            sx, sy = int(s['x']), int(s['y'])
            angle = (self.steps * 20) % 360
            p1 = (sx + 8 * math.cos(math.radians(angle)), sy + 8 * math.sin(math.radians(angle)))
            p2 = (sx + 8 * math.cos(math.radians(angle+180)), sy + 8 * math.sin(math.radians(angle+180)))
            p3 = (sx + 8 * math.cos(math.radians(angle+90)), sy + 8 * math.sin(math.radians(angle+90)))
            p4 = (sx + 8 * math.cos(math.radians(angle-90)), sy + 8 * math.sin(math.radians(angle-90)))
            pygame.draw.aaline(self.screen, self.COLOR_TEXT, p1, p2, 1)
            pygame.draw.aaline(self.screen, self.COLOR_TEXT, p3, p4, 1)

        # Player
        px, py = int(self.player['px']), int(self.player['py'])
        bob = math.sin(self.steps * 0.2) * 2
        
        # Disguise effect
        if self.disguise_timer > 0:
            alpha = 100 + math.sin(self.steps * 0.3) * 50
            pygame.gfxdraw.filled_circle(self.screen, px, int(py - bob), 14, (*self.TILE_COLORS[6], int(alpha/2)))
            pygame.gfxdraw.aacircle(self.screen, px, int(py - bob), 14, (*self.TILE_COLORS[6], int(alpha)))

        # Glow
        pygame.gfxdraw.filled_circle(self.screen, px, int(py - bob), 12, (*self.COLOR_PLAYER, 50))
        pygame.gfxdraw.aacircle(self.screen, px, int(py - bob), 12, (*self.COLOR_PLAYER, 100))
        # Body
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (px, int(py - bob)), 8)
        
    def _render_ui(self):
        # BG
        ui_rect = pygame.Rect(0, self.GAME_AREA_HEIGHT, self.SCREEN_WIDTH, self.UI_AREA_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (0, self.GAME_AREA_HEIGHT), (self.SCREEN_WIDTH, self.GAME_AREA_HEIGHT), 1)

        # Info Text
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        level_text = self.font_small.render(f"Level: {self._level_number}", True, self.COLOR_TEXT)
        steps_text = self.font_small.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, self.GAME_AREA_HEIGHT + 10))
        self.screen.blit(level_text, (10, self.GAME_AREA_HEIGHT + 30))
        self.screen.blit(steps_text, (10, self.GAME_AREA_HEIGHT + 50))
        
        # Resources
        shuriken_text = self.font_medium.render(f"x{self.shuriken_count}", True, self.COLOR_TEXT)
        self._draw_tile_icon(self.screen, self.TILE_TYPES['SHURIKEN'], 500, self.GAME_AREA_HEIGHT + 25, 20)
        self.screen.blit(shuriken_text, (525, self.GAME_AREA_HEIGHT + 15))

        if self.disguise_timer > 0:
            self._draw_tile_icon(self.screen, self.TILE_TYPES['DISGUISE'], 500, self.GAME_AREA_HEIGHT + 55, 20)
            disguise_bar_w = (self.disguise_timer / self.DISGUISE_DURATION) * 60
            pygame.draw.rect(self.screen, self.TILE_COLORS[6], (525, self.GAME_AREA_HEIGHT + 55, disguise_bar_w, 10))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (525, self.GAME_AREA_HEIGHT + 55, 60, 10), 1)


        # Tile Board
        board_x_start = 190
        board_y_start = self.GAME_AREA_HEIGHT + 10
        for r in range(3):
            for c in range(3):
                tile = self.tile_board[r][c]
                tile['y_offset'] *= 0.7 # Animate fall-in
                if abs(tile['y_offset']) < 1: tile['y_offset'] = 0

                tile_rect = pygame.Rect(board_x_start + c * 60, board_y_start + r * 20 + tile['y_offset'], 50, 18)
                
                pygame.draw.rect(self.screen, self.COLOR_TILE_BG, tile_rect, border_radius=3)
                self._draw_tile_icon(self.screen, tile['type'], tile_rect.centerx, tile_rect.centery, 14)

        # Action Feedback
        if self.last_action_feedback:
            action, success = self.last_action_feedback
            color = (0, 255, 0) if success else (255, 0, 0)
            text = self.font_small.render(f"{action}", True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH/2, self.GAME_AREA_HEIGHT - 20))
            self.screen.blit(text, text_rect)


    def _draw_tile_icon(self, surface, tile_type, cx, cy, size):
        color = self.TILE_COLORS[tile_type]
        s = size / 2
        if tile_type == self.TILE_TYPES['UP']:
            pygame.draw.polygon(surface, color, [(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)])
        elif tile_type == self.TILE_TYPES['DOWN']:
            pygame.draw.polygon(surface, color, [(cx, cy + s), (cx - s, cy - s), (cx + s, cy - s)])
        elif tile_type == self.TILE_TYPES['LEFT']:
            pygame.draw.polygon(surface, color, [(cx - s, cy), (cx + s, cy - s), (cx + s, cy + s)])
        elif tile_type == self.TILE_TYPES['RIGHT']:
            pygame.draw.polygon(surface, color, [(cx + s, cy), (cx - s, cy - s), (cx - s, cy + s)])
        elif tile_type == self.TILE_TYPES['SHURIKEN']:
            pygame.draw.line(surface, color, (cx-s, cy-s), (cx+s, cy+s), 2)
            pygame.draw.line(surface, color, (cx-s, cy+s), (cx+s, cy-s), 2)
        elif tile_type == self.TILE_TYPES['DISGUISE']:
            pygame.draw.circle(surface, color, (cx, cy), s, 2)
            pygame.draw.line(surface, color, (cx-s, cy), (cx+s, cy), 2)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Ninja Tile Match")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q: Quit")

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"--- Env Reset --- | Level: {info['level']}")
                # Handle held keys
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1

            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    if movement in [1,2,3,4]: # Only reset if it was the one being held
                        movement = 0
                elif event.key == pygame.K_SPACE:
                    space_held = 0
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 0
        
        action = [movement, space_held, shift_held]
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward

        if term or trunc:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Reset for next game
            obs, info = env.reset()
            total_reward = 0

        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS
        
    env.close()