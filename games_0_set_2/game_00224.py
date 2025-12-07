
# Generated: 2025-08-27T13:00:20.432652
# Source Brief: brief_00224.md
# Brief Index: 224

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space to attack. Shift + Arrow to dash-attack."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore a procedural dungeon, battle monsters, and find the exit to defeat the final boss."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.GRID_SIZE = 32
        self.TILE_WIDTH = 48
        self.TILE_HEIGHT = 24
        
        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_FLOOR = (40, 30, 60)
        self.COLOR_WALL = (70, 60, 90)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (180, 240, 255)
        self.COLOR_STAIRS = (180, 50, 255)
        self.COLOR_CHEST = (255, 190, 0)
        self.COLOR_CHEST_OPEN = (100, 80, 20)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_DAMAGE = (255, 50, 50)
        self.COLOR_HEAL = (50, 255, 50)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_max_health = None
        self.player_gold = None
        self.current_level = None
        self.dungeon_map = None
        self.enemies = None
        self.chests = None
        self.stairs_pos = None
        self.particles = None
        self.floating_texts = None
        self.game_over = None
        self.last_action_info = ""
        
        self.reset()

        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_health = 100
        self.player_max_health = 100
        self.player_gold = 0
        self.current_level = 1
        
        self.particles = []
        self.floating_texts = []
        self.last_action_info = "A new adventure begins!"

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # 1. Create a map full of walls
        self.dungeon_map = [[1 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        # 2. Carve out rooms with a random walker
        walker_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        num_floors = 0
        max_floors = int(self.GRID_SIZE * self.GRID_SIZE * 0.4) # 40% floor
        
        while num_floors < max_floors:
            if self.dungeon_map[walker_pos[1]][walker_pos[0]] == 1:
                self.dungeon_map[walker_pos[1]][walker_pos[0]] = 0
                num_floors += 1
            
            move = self.np_random.integers(0, 4)
            if move == 0 and walker_pos[0] > 1: walker_pos[0] -= 1
            elif move == 1 and walker_pos[0] < self.GRID_SIZE - 2: walker_pos[0] += 1
            elif move == 2 and walker_pos[1] > 1: walker_pos[1] -= 1
            elif move == 3 and walker_pos[1] < self.GRID_SIZE - 2: walker_pos[1] += 1

        # 3. Find valid spawn points
        floor_tiles = []
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.dungeon_map[y][x] == 0:
                    floor_tiles.append((x, y))
        
        self.np_random.shuffle(floor_tiles)
        
        # 4. Place player, stairs, chests, and enemies
        self.player_pos = list(floor_tiles.pop())

        if self.current_level < 10:
            self.stairs_pos = floor_tiles.pop()
        else: # Level 10 is the boss level
            self.stairs_pos = None

        self.chests = []
        num_chests = self.np_random.integers(2, 5)
        for _ in range(num_chests):
            if not floor_tiles: break
            self.chests.append({'pos': floor_tiles.pop(), 'opened': False})

        self.enemies = []
        num_enemies = self.np_random.integers(3, 7) + self.current_level
        
        if self.current_level == 10: # Boss level
            boss_pos = floor_tiles.pop()
            base_hp = 200 + 20 * self.current_level
            base_dmg = 15 + 3 * self.current_level
            self.enemies.append(self._create_enemy('Boss', boss_pos, base_hp, base_dmg))
        else:
            for _ in range(num_enemies):
                if not floor_tiles: break
                spawn_pos = floor_tiles.pop()
                dist_to_player = math.hypot(spawn_pos[0] - self.player_pos[0], spawn_pos[1] - self.player_pos[1])
                if dist_to_player < 5: continue # Don't spawn too close

                enemy_type = 'Goblin'
                if self.current_level >= 3 and self.np_random.random() > 0.6:
                    enemy_type = 'Skeleton Archer'
                if self.current_level >= 6 and self.np_random.random() > 0.7:
                    enemy_type = 'Ogre'
                
                base_hp = {'Goblin': 20, 'Skeleton Archer': 15, 'Ogre': 40}[enemy_type]
                base_dmg = {'Goblin': 5, 'Skeleton Archer': 8, 'Ogre': 10}[enemy_type]
                
                hp = int(base_hp * (1 + (self.current_level - 1) * 0.05))
                dmg = int(base_dmg * (1 + (self.current_level - 1) * 0.02))
                
                self.enemies.append(self._create_enemy(enemy_type, spawn_pos, hp, dmg))

    def _create_enemy(self, type, pos, hp, dmg):
        return {
            'type': type, 'pos': list(pos), 'health': hp, 'max_health': hp, 'damage': dmg,
            'attack_range': 1.5 if type != 'Skeleton Archer' else 5.5,
            'detection_range': 8,
            'boss_state': 'melee' if type == 'Boss' else None,
            'boss_state_timer': 3 if type == 'Boss' else 0,
            'color': {'Goblin': (50, 180, 50), 'Skeleton Archer': (200, 200, 200), 'Ogre': (180, 140, 80), 'Boss': (200, 20, 100)}[type]
        }

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.last_action_info = ""

        # Unpack factorized action
        movement_action = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        action_taken = False

        # 1. Player Action (prioritized: special > attack > move)
        if shift_held and movement_action > 0:
            action_taken = True
            reward += self._handle_player_dash_attack(movement_action)
        elif space_held:
            action_taken = True
            reward += self._handle_player_melee_attack()
        elif movement_action > 0:
            action_taken = True
            reward += self._handle_player_movement(movement_action)
        else: # No-op
            action_taken = True
            self.last_action_info = "Player waits."

        # 2. Enemy Actions (only if player acted)
        if action_taken:
            for enemy in self.enemies[:]:
                if enemy['health'] > 0:
                    reward += self._handle_enemy_turn(enemy)

        # 3. Update game state
        self._update_effects()
        self.enemies = [e for e in self.enemies if e['health'] > 0]
        self.score += reward

        # 4. Check for termination
        terminated = False
        if self.player_health <= 0:
            self.last_action_info = "You have been defeated!"
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = True
        
        boss = next((e for e in self.enemies if e['type'] == 'Boss'), None)
        if self.current_level == 10 and not boss:
            self.last_action_info = "VICTORY! The final boss is slain!"
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.last_action_info = "The journey is long... too long."

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_movement(self, move_action):
        reward = 0
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[move_action]
        target_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

        # Check for wall collision
        if self.dungeon_map[target_pos[1]][target_pos[0]] == 1:
            self.last_action_info = "You bump into a wall."
            return 0
        
        # Check for enemy collision
        if any(e['pos'] == target_pos for e in self.enemies):
            self.last_action_info = "An enemy blocks your path."
            return 0

        # Move player
        self.player_pos = target_pos
        self.last_action_info = "You move."

        # Check for interactions
        if self.stairs_pos and self.player_pos == self.stairs_pos:
            self.current_level += 1
            self._generate_level()
            reward += 10
            self.score += 10
            self.last_action_info = f"You descend to level {self.current_level}."
            return reward

        for chest in self.chests:
            if not chest['opened'] and chest['pos'] == self.player_pos:
                chest['opened'] = True
                gold_found = self.np_random.integers(10, 51)
                self.player_gold += gold_found
                self._create_floating_text(f"+{gold_found} Gold", self.player_pos, self.COLOR_CHEST)
                
                if self.np_random.random() < 0.2: # 20% chance of health potion
                    health_restored = self.np_random.integers(15, 31)
                    self.player_health = min(self.player_max_health, self.player_health + health_restored)
                    self._create_floating_text(f"+{health_restored} HP", [self.player_pos[0], self.player_pos[1]-0.5], self.COLOR_HEAL)

                reward += 5
                self.score += 5
                self.last_action_info = f"You found {gold_found} gold!"
                break
        
        return reward

    def _handle_player_melee_attack(self):
        reward = 0
        self.last_action_info = "You swing your weapon... and miss."
        attacked = False
        for enemy in self.enemies:
            if math.hypot(enemy['pos'][0] - self.player_pos[0], enemy['pos'][1] - self.player_pos[1]) < 1.5:
                attacked = True
                damage = self.np_random.integers(10, 16)
                enemy['health'] -= damage
                reward += 0.1 * damage
                self._create_particle_burst(enemy['pos'], self.COLOR_DAMAGE, 10)
                self._create_floating_text(str(damage), enemy['pos'], self.COLOR_DAMAGE)
                self.last_action_info = f"You hit a {enemy['type']} for {damage} damage!"
                if enemy['health'] <= 0:
                    reward += 1
                    self.score += 1
                    self.last_action_info = f"You defeated the {enemy['type']}!"
        return reward

    def _handle_player_dash_attack(self, move_action):
        self.player_health -= 10
        reward = -2 # -0.2 reward per damage point taken
        self._create_floating_text("-10 HP", self.player_pos, self.COLOR_DAMAGE)
        if self.player_health <= 0: return reward

        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[move_action]
        target_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

        # Check for wall collision
        if self.dungeon_map[target_pos[1]][target_pos[0]] == 1:
            self.last_action_info = "You dash into a wall! Ouch."
            return reward

        self.player_pos = target_pos
        enemy_hit = next((e for e in self.enemies if e['pos'] == target_pos), None)
        
        if enemy_hit:
            damage = self.np_random.integers(20, 31) # Double damage
            enemy_hit['health'] -= damage
            reward += 0.1 * damage
            self._create_particle_burst(enemy_hit['pos'], self.COLOR_PLAYER, 20)
            self._create_floating_text(str(damage), enemy_hit['pos'], self.COLOR_DAMAGE)
            self.last_action_info = f"You dash-attack the {enemy_hit['type']} for {damage} damage!"
            if enemy_hit['health'] <= 0:
                reward += 1
                self.score += 1
                self.last_action_info = f"You obliterated the {enemy_hit['type']}!"
        else:
            self.last_action_info = "You dash into empty space."

        return reward

    def _handle_enemy_turn(self, enemy):
        reward = 0
        dist_to_player = math.hypot(enemy['pos'][0] - self.player_pos[0], enemy['pos'][1] - self.player_pos[1])

        # Boss AI
        if enemy['type'] == 'Boss':
            enemy['boss_state_timer'] -= 1
            if enemy['boss_state_timer'] <= 0:
                if enemy['boss_state'] == 'melee':
                    enemy['boss_state'] = 'ranged'
                    enemy['boss_state_timer'] = 2
                elif enemy['boss_state'] == 'ranged':
                    enemy['boss_state'] = 'defend'
                    enemy['boss_state_timer'] = 2
                elif enemy['boss_state'] == 'defend':
                    enemy['boss_state'] = 'melee'
                    enemy['boss_state_timer'] = 3
            
            # Boss takes reduced damage in defend state
            if enemy['boss_state'] == 'defend':
                enemy['damage_mod'] = 0.5
            else:
                enemy['damage_mod'] = 1.0

            # Boss attack logic
            attack_range = 1.5 if enemy['boss_state'] == 'melee' else 10.0
            if dist_to_player < attack_range and enemy['boss_state'] != 'defend':
                 # Attack
                damage = self.np_random.integers(enemy['damage']-2, enemy['damage']+3)
                self.player_health -= damage
                reward -= 0.2 * damage
                self._create_particle_burst(self.player_pos, self.COLOR_DAMAGE, 15)
                self._create_floating_text(str(damage), self.player_pos, self.COLOR_DAMAGE)
                return reward

        # Standard AI
        elif dist_to_player < enemy['attack_range']:
            # Attack
            damage = self.np_random.integers(enemy['damage']-1, enemy['damage']+2)
            self.player_health -= damage
            reward -= 0.2 * damage
            self._create_particle_burst(self.player_pos, self.COLOR_DAMAGE, 10)
            self._create_floating_text(str(damage), self.player_pos, self.COLOR_DAMAGE)
            return reward

        if dist_to_player < enemy['detection_range']:
            # Move towards player
            dx = self.player_pos[0] - enemy['pos'][0]
            dy = self.player_pos[1] - enemy['pos'][1]
            
            if abs(dx) > abs(dy):
                move = [np.sign(dx), 0]
            else:
                move = [0, np.sign(dy)]
            
            target_pos = [enemy['pos'][0] + move[0], enemy['pos'][1] + move[1]]
            
            is_wall = self.dungeon_map[target_pos[1]][target_pos[0]] == 1
            is_player = target_pos == self.player_pos
            is_enemy = any(e['pos'] == target_pos for e in self.enemies)

            if not is_wall and not is_player and not is_enemy:
                enemy['pos'] = target_pos
        
        return reward

    def _iso_to_screen(self, x, y):
        screen_x = (x - y) * self.TILE_WIDTH / 2 + self.WIDTH / 2
        screen_y = (x + y) * self.TILE_HEIGHT / 2 + self.HEIGHT / 2 - self.GRID_SIZE * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_poly(self, surface, x, y, color, height=0):
        px, py = self._iso_to_screen(x, y)
        py -= height
        points = [
            (px, py - self.TILE_HEIGHT / 2),
            (px + self.TILE_WIDTH / 2, py),
            (px, py + self.TILE_HEIGHT / 2),
            (px - self.TILE_WIDTH / 2, py)
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_iso_cube(self, surface, x, y, color, size=0.8):
        px, py = self._iso_to_screen(x, y)
        w, h = self.TILE_WIDTH * size, self.TILE_HEIGHT * size
        
        top_points = [
            (px, py - h), (px + w/2, py - h/2), (px, py), (px - w/2, py - h/2)
        ]
        left_points = [
            (px - w/2, py - h/2), (px, py), (px, py + h), (px - w/2, py + h/2)
        ]
        right_points = [
            (px + w/2, py - h/2), (px, py), (px, py + h), (px + w/2, py + h/2)
        ]

        darker_color = tuple(max(0, c - 40) for c in color)
        darkest_color = tuple(max(0, c - 60) for c in color)
        
        pygame.draw.polygon(surface, color, top_points)
        pygame.draw.polygon(surface, darker_color, left_points)
        pygame.draw.polygon(surface, darkest_color, right_points)
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Center camera on player
        cam_offset_x = self.WIDTH/2 - (self.player_pos[0] - self.player_pos[1]) * self.TILE_WIDTH / 2
        cam_offset_y = self.HEIGHT/2 - (self.player_pos[0] + self.player_pos[1]) * self.TILE_HEIGHT / 2
        
        # Create a separate surface for the world to clip it
        world_surface = self.screen.copy()
        world_surface.fill(self.COLOR_BG)

        # 1. Render world (sorted for isometric perspective)
        render_queue = []
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.dungeon_map[y][x] == 0: # Floor
                    render_queue.append(('tile', (x, y), self.COLOR_FLOOR))
                else: # Wall
                    render_queue.append(('wall', (x, y), self.COLOR_WALL))
        
        if self.stairs_pos:
            render_queue.append(('stairs', self.stairs_pos, self.COLOR_STAIRS))
        for chest in self.chests:
            color = self.COLOR_CHEST_OPEN if chest['opened'] else self.COLOR_CHEST
            render_queue.append(('chest', chest['pos'], color))
        
        all_entities = self.enemies + [{'type': 'Player', 'pos': self.player_pos, 'color': self.COLOR_PLAYER}]
        for entity in all_entities:
            render_queue.append((entity['type'], entity['pos'], entity['color'], entity))

        # Sort by y-coord, then x-coord for correct isometric drawing
        render_queue.sort(key=lambda item: (item[1][1], item[1][0]))

        for item_type, pos, color, *extra in render_queue:
            px, py = self._iso_to_screen(pos[0], pos[1])
            px_cam = int(px - self.WIDTH/2 + cam_offset_x)
            py_cam = int(py - self.HEIGHT/2 + cam_offset_y)

            if item_type == 'tile':
                self._draw_iso_poly(world_surface, pos[0], pos[1], color)
            elif item_type == 'wall':
                self._draw_iso_cube(world_surface, pos[0], pos[1], color)
            elif item_type == 'stairs':
                self._draw_iso_poly(world_surface, pos[0], pos[1], color)
            elif item_type == 'chest':
                self._draw_iso_cube(world_surface, pos[0], pos[1], color, size=0.6)
            elif item_type == 'Player':
                # Glow effect
                pygame.gfxdraw.filled_circle(world_surface, px, py, 15, (*self.COLOR_PLAYER_GLOW, 50))
                pygame.gfxdraw.filled_circle(world_surface, px, py, 12, (*self.COLOR_PLAYER_GLOW, 80))
                self._draw_iso_cube(world_surface, pos[0], pos[1], color, size=0.7)
            else: # Enemy
                entity = extra[0]
                self._draw_iso_cube(world_surface, pos[0], pos[1], color, size=0.7)
                # Health bar
                if entity['health'] < entity['max_health']:
                    hp_ratio = max(0, entity['health'] / entity['max_health'])
                    bar_w = 20
                    bar_h = 3
                    bar_x = px - bar_w / 2
                    bar_y = py - 30
                    pygame.draw.rect(world_surface, (50,0,0), (bar_x, bar_y, bar_w, bar_h))
                    pygame.draw.rect(world_surface, self.COLOR_DAMAGE, (bar_x, bar_y, bar_w * hp_ratio, bar_h))

        # Blit the world surface to the main screen
        self.screen.blit(world_surface, (0,0))
        
        # 2. Render Effects (particles, floating text) - in screen space
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))
        
        for ft in self.floating_texts:
            text_surf = self.font_small.render(ft['text'], True, ft['color'])
            text_rect = text_surf.get_rect(center=(int(ft['pos'][0]), int(ft['pos'][1])))
            self.screen.blit(text_surf, text_rect)

        # 3. Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Player Health Bar
        hp_ratio = max(0, self.player_health / self.player_max_health)
        bar_w, bar_h = 200, 20
        pygame.draw.rect(self.screen, (50,0,0), (10, 10, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_DAMAGE, (10, 10, bar_w * hp_ratio, bar_h))
        hp_text = self.font_small.render(f"HP: {self.player_health}/{self.player_max_health}", True, self.COLOR_TEXT)
        self.screen.blit(hp_text, (15, 12))

        # Info text
        level_text = self.font_medium.render(f"Level: {self.current_level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 40))
        gold_text = self.font_medium.render(f"Gold: {self.player_gold}", True, self.COLOR_CHEST)
        self.screen.blit(gold_text, (10, 65))

        # Last action info
        action_text = self.font_small.render(self.last_action_info, True, self.COLOR_TEXT)
        action_rect = action_text.get_rect(centerx=self.WIDTH/2, y=10)
        self.screen.blit(action_text, action_rect)

        # Controls guide
        guide_text = self.font_small.render(self.user_guide, True, self.COLOR_TEXT)
        guide_rect = guide_text.get_rect(centerx=self.WIDTH/2, bottom=self.HEIGHT - 10)
        self.screen.blit(guide_text, guide_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.player_gold,
            "level": self.current_level,
            "health": self.player_health,
        }

    def _create_particle_burst(self, grid_pos, color, count):
        screen_pos = self._iso_to_screen(grid_pos[0], grid_pos[1])
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            self.particles.append({
                'pos': list(screen_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.random() * 3 + 2,
                'lifetime': self.np_random.integers(10, 20),
                'color': color
            })
    
    def _create_floating_text(self, text, grid_pos, color):
        screen_pos = self._iso_to_screen(grid_pos[0], grid_pos[1])
        self.floating_texts.append({
            'text': text,
            'pos': list(screen_pos),
            'lifetime': 30,
            'color': color
        })

    def _update_effects(self):
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            p['radius'] *= 0.95
            if p['lifetime'] <= 0:
                self.particles.remove(p)
        
        # Update floating texts
        for ft in self.floating_texts[:]:
            ft['pos'][1] -= 1
            ft['lifetime'] -= 1
            alpha = max(0, min(255, int(255 * (ft['lifetime'] / 30))))
            ft['color'] = (ft['color'][0], ft['color'][1], ft['color'][2], alpha)
            if ft['lifetime'] <= 0:
                self.floating_texts.remove(ft)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")