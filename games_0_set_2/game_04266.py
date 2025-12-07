
# Generated: 2025-08-28T01:53:25.907663
# Source Brief: brief_04266.md
# Brief Index: 4266

        
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
        "Controls: Arrow keys to move. Hold space to attack an adjacent enemy. Hold shift to defend (halves next damage)."
    )

    game_description = (
        "Explore a procedurally generated dungeon, fight monsters, and defeat the boss on level 5 to win."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    TILE_SIZE = 32
    MAX_STEPS = 2000
    NUM_LEVELS = 5

    # --- Colors ---
    COLOR_BG = (10, 10, 15)
    COLOR_WALL = (50, 50, 60)
    COLOR_FLOOR = (30, 30, 40)
    COLOR_PLAYER = (50, 200, 50)
    COLOR_ENEMY = (200, 50, 50)
    COLOR_BOSS = (255, 100, 100)
    COLOR_GOLD = (255, 215, 0)
    COLOR_STAIRS = (100, 100, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR = (50, 255, 50)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    COLOR_PARTICLE_HIT = (255, 150, 150)
    COLOR_PARTICLE_GOLD = (255, 230, 150)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.render_mode = render_mode
        self.np_random = None

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.info_text = ""

        self.current_level = 1
        self.player_pos = pygame.Vector2(0, 0)
        self.player_max_health = 100
        self.player_health = self.player_max_health
        self.player_attack = 10
        self.player_gold = 0
        self.is_defending = False

        self.entities = []
        self.particles = []
        self.dungeon_map = []
        self.stairs_pos = None

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.is_defending = False
        self.info_text = ""

        # 1. Handle Player Action
        reward += self._handle_player_action(action)

        # 2. Handle Enemy Turns
        if not self.game_over:
            enemy_damage = self._handle_enemy_turns()
            if enemy_damage > 0:
                self.player_health -= enemy_damage
                self.info_text = f"Took {enemy_damage} damage!"
                # Sound: player_hurt.wav
                self._spawn_particles(self.player_pos, self.COLOR_PARTICLE_HIT, 10)

        # 3. Update Game State
        self._update_particles()
        
        # 4. Check Termination Conditions
        terminated = False
        if self.player_health <= 0:
            self.game_over = True
            terminated = True
            reward = -100
            self.info_text = "You have been defeated!"
        elif self.victory:
            self.game_over = True
            terminated = True
            reward = 100
            self.info_text = "You defeated the boss! Victory!"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward = -50  # Penalize for running out of time
            self.info_text = "Out of time!"

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_player_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        action_taken = False

        # Action priority: Attack > Defend > Move
        if space_held:
            # Attack
            target = self._find_adjacent_enemy()
            if target:
                action_taken = True
                damage = self.player_attack
                target['health'] -= damage
                self.info_text = f"Dealt {damage} damage!"
                # Sound: player_attack.wav
                self._spawn_particles(target['pos'], self.COLOR_PARTICLE_HIT, 5, speed_mult=0.5)

                if target['health'] <= 0:
                    reward += 1.0  # Defeated enemy reward
                    self.info_text = f"Enemy defeated! +1 reward."
                    if target.get('is_boss', False):
                        self.victory = True
                    self.entities.remove(target)
                    # Sound: enemy_die.wav
            else:
                self.info_text = "Attack whiffed!"
        
        elif shift_held:
            # Defend
            action_taken = True
            self.is_defending = True
            self.info_text = "Defending!"

        elif movement != 0:
            # Move
            dist_before = self._distance_to_stairs(self.player_pos)
            moved = self._move_player(movement)
            if moved:
                action_taken = True
                dist_after = self._distance_to_stairs(self.player_pos)
                if dist_after < dist_before:
                    reward += 0.1
                else:
                    reward += -0.1
                
                # Check for collisions with entities
                collided_entity = self._get_entity_at(self.player_pos)
                if collided_entity:
                    if collided_entity['type'] == 'gold':
                        self.player_gold += collided_entity['value']
                        self.entities.remove(collided_entity)
                        reward += 10.0
                        self.info_text = f"Picked up {collided_entity['value']} gold! +10 reward."
                        # Sound: gold_pickup.wav
                        self._spawn_particles(self.player_pos, self.COLOR_PARTICLE_GOLD, 15)
                    elif collided_entity['type'] == 'stairs':
                        self.current_level += 1
                        if self.current_level > self.NUM_LEVELS:
                             self.victory = True # Should not happen, boss is on last level
                        else:
                            self._generate_level()
                            self.info_text = f"Descended to level {self.current_level}."
                            # Sound: level_up.wav
        
        if not action_taken:
            self.info_text = "Waiting..."

        return reward

    def _handle_enemy_turns(self):
        total_damage = 0
        for enemy in [e for e in self.entities if e['type'] == 'enemy']:
            dist_to_player = self._distance(self.player_pos, enemy['pos'])
            
            # Simple AI: If in range, attack. Else, move towards player.
            if dist_to_player < 1.5: # Adjacent
                # Sound: enemy_attack.wav
                damage = int(enemy['attack'])
                total_damage += damage
            elif dist_to_player < 8: # Aggro range
                # Move towards player
                dx, dy = self.player_pos.x - enemy['pos'].x, self.player_pos.y - enemy['pos'].y
                if abs(dx) > abs(dy):
                    new_pos = enemy['pos'] + pygame.Vector2(np.sign(dx), 0)
                else:
                    new_pos = enemy['pos'] + pygame.Vector2(0, np.sign(dy))
                
                if self._is_walkable(new_pos):
                    enemy['pos'] = new_pos
        
        if self.is_defending:
            total_damage = math.ceil(total_damage / 2)
        
        return total_damage

    def _move_player(self, movement):
        direction = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement)
        if direction:
            new_pos = self.player_pos + pygame.Vector2(direction)
            if self._is_walkable(new_pos):
                self.player_pos = new_pos
                return True
        return False

    def _is_walkable(self, pos):
        if not (0 <= pos.x < len(self.dungeon_map[0]) and 0 <= pos.y < len(self.dungeon_map)):
            return False
        if self.dungeon_map[int(pos.y)][int(pos.x)] == 1: # Wall
            return False
        if any(e['type'] == 'enemy' and e['pos'] == pos for e in self.entities):
            return False
        return True

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
            "level": self.current_level,
            "health": self.player_health,
            "gold": self.player_gold,
        }

    def _render_game(self):
        center_x = self.SCREEN_WIDTH // 2
        center_y = self.SCREEN_HEIGHT // 2
        
        # Calculate render offset
        offset_x = center_x - self.player_pos.x * self.TILE_SIZE
        offset_y = center_y - self.player_pos.y * self.TILE_SIZE

        # Render dungeon
        for y, row in enumerate(self.dungeon_map):
            for x, tile in enumerate(row):
                rect = pygame.Rect(offset_x + x * self.TILE_SIZE, offset_y + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if self.screen.get_rect().colliderect(rect):
                    color = self.COLOR_WALL if tile == 1 else self.COLOR_FLOOR
                    pygame.draw.rect(self.screen, color, rect)
        
        # Render entities
        for entity in self.entities:
            pos = entity['pos']
            rect = pygame.Rect(offset_x + pos.x * self.TILE_SIZE, offset_y + pos.y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            if not self.screen.get_rect().colliderect(rect):
                continue

            if entity['type'] == 'enemy':
                color = self.COLOR_BOSS if entity.get('is_boss') else self.COLOR_ENEMY
                pygame.draw.rect(self.screen, color, rect.inflate(-4, -4))
                self._render_health_bar(entity, rect.top - 8)
            elif entity['type'] == 'gold':
                points = [
                    (rect.centerx, rect.top + 4), (rect.centerx + 4, rect.centery - 2),
                    (rect.right - 4, rect.centery), (rect.centerx + 4, rect.centery + 2),
                    (rect.centerx, rect.bottom - 4), (rect.centerx - 4, rect.centery + 2),
                    (rect.left + 4, rect.centery), (rect.centerx - 4, rect.centery - 2),
                ]
                pygame.draw.polygon(self.screen, self.COLOR_GOLD, points)
            elif entity['type'] == 'stairs':
                pygame.draw.rect(self.screen, self.COLOR_STAIRS, rect.inflate(-8, -8))
                pygame.draw.rect(self.screen, (0,0,0), rect.inflate(-16, -16))

        # Render player
        player_rect = pygame.Rect(center_x, center_y, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-4, -4))
        self._render_health_bar({'health': self.player_health, 'max_health': self.player_max_health}, player_rect.top - 8)
        
        # Render particles
        for p in self.particles:
            render_pos = p['pos'] * self.TILE_SIZE + pygame.Vector2(offset_x, offset_y)
            pygame.draw.circle(self.screen, p['color'], (int(render_pos.x), int(render_pos.y)), int(p['size']))

    def _render_health_bar(self, entity, y_pos):
        health_pct = max(0, entity['health'] / entity['max_health'])
        center_x = self.SCREEN_WIDTH // 2
        
        if entity.get('type') == 'enemy':
            offset_x = center_x - self.player_pos.x * self.TILE_SIZE
            x_pos = offset_x + entity['pos'].x * self.TILE_SIZE
        else: # Player
             x_pos = center_x
        
        bar_width = self.TILE_SIZE
        bg_rect = pygame.Rect(x_pos, y_pos, bar_width, 5)
        fill_rect = pygame.Rect(x_pos, y_pos, int(bar_width * health_pct), 5)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, fill_rect)

    def _render_ui(self):
        # Top Bar
        bar_height = 30
        pygame.draw.rect(self.screen, (0, 0, 0, 150), (0, 0, self.SCREEN_WIDTH, bar_height))
        
        health_text = f"HP: {self.player_health}/{self.player_max_health}"
        gold_text = f"Gold: {self.player_gold}"
        level_text = f"Level: {self.current_level}/{self.NUM_LEVELS}"
        
        self._draw_text(health_text, (10, 5), self.font_small)
        self._draw_text(gold_text, (200, 5), self.font_small)
        self._draw_text(level_text, (400, 5), self.font_small)

        # Info text
        if self.info_text:
            self._draw_text(self.info_text, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30), self.font_large, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_UI_TEXT, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _generate_level(self):
        map_w, map_h = 50, 50
        self.dungeon_map = [[1 for _ in range(map_w)] for _ in range(map_h)]
        rooms = []
        
        for _ in range(15):
            w = self.np_random.integers(5, 10)
            h = self.np_random.integers(5, 10)
            x = self.np_random.integers(1, map_w - w - 1)
            y = self.np_random.integers(1, map_h - h - 1)
            new_room = pygame.Rect(x, y, w, h)
            
            if not any(new_room.colliderect(other) for other in rooms):
                for i in range(new_room.left, new_room.right):
                    for j in range(new_room.top, new_room.bottom):
                        self.dungeon_map[j][i] = 0
                rooms.append(new_room)
        
        for i in range(len(rooms) - 1):
            self._connect_rooms(rooms[i], rooms[i+1])
            
        self.player_pos = pygame.Vector2(rooms[0].center)
        self.stairs_pos = pygame.Vector2(rooms[-1].center)
        self.entities = [{'type': 'stairs', 'pos': self.stairs_pos}]
        self._populate_dungeon(rooms)
        
    def _connect_rooms(self, room1, room2):
        x1, y1 = room1.center
        x2, y2 = room2.center
        
        for x in range(min(x1, x2), max(x1, x2) + 1):
            self.dungeon_map[y1][x] = 0
        for y in range(min(y1, y2), max(y1, y2) + 1):
            self.dungeon_map[y][x2] = 0

    def _populate_dungeon(self, rooms):
        for room in rooms[1:]: # Don't populate start room
            if self.np_random.random() < 0.7: # 70% chance to have enemies
                num_enemies = self.np_random.integers(1, 4)
                for _ in range(num_enemies):
                    pos = self._get_random_pos_in_room(room)
                    if pos:
                        base_health = 10
                        base_attack = 2
                        enemy_health = base_health + self.current_level * 5
                        enemy_attack = base_attack + self.current_level * 2
                        self.entities.append({
                            'type': 'enemy', 'pos': pos, 'health': enemy_health,
                            'max_health': enemy_health, 'attack': enemy_attack
                        })

            if self.np_random.random() < 0.5: # 50% chance for gold
                pos = self._get_random_pos_in_room(room)
                if pos:
                    self.entities.append({'type': 'gold', 'pos': pos, 'value': 10})
        
        # Add boss on the final level
        if self.current_level == self.NUM_LEVELS:
            boss_room = rooms[-1]
            boss_health = 150
            boss_attack = 20
            self.entities.append({
                'type': 'enemy', 'pos': self.stairs_pos, 'health': boss_health,
                'max_health': boss_health, 'attack': boss_attack, 'is_boss': True
            })
            # Remove stairs from boss room
            self.entities = [e for e in self.entities if e['type'] != 'stairs']
            self.stairs_pos = None

    def _get_random_pos_in_room(self, room):
        for _ in range(10): # Try 10 times to find an empty spot
            pos = pygame.Vector2(
                self.np_random.integers(room.left, room.right),
                self.np_random.integers(room.top, room.bottom)
            )
            if self._is_walkable(pos) and pos != self.player_pos:
                return pos
        return None

    def _get_entity_at(self, pos):
        for entity in self.entities:
            if entity['pos'] == pos:
                return entity
        return None

    def _find_adjacent_enemy(self):
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            check_pos = self.player_pos + pygame.Vector2(dx, dy)
            entity = self._get_entity_at(check_pos)
            if entity and entity['type'] == 'enemy':
                return entity
        return None
        
    def _distance(self, pos1, pos2):
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def _distance_to_stairs(self, pos):
        if not self.stairs_pos:
            return float('inf')
        return self._distance(pos, self.stairs_pos)

    def _spawn_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            self.particles.append({
                'pos': pos + pygame.Vector2(0.5, 0.5), # Center of tile
                'vel': pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize() * self.np_random.uniform(1, 3) * speed_mult,
                'size': self.np_random.uniform(2, 5),
                'life': self.np_random.uniform(10, 20),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel'] / self.TILE_SIZE
            p['life'] -= 1
            p['size'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Dungeon Crawler")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        # --- Action mapping for human play ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    terminated = True
                else:
                    # For turn-based, we step on key press
                    obs, reward, term, trunc, info = env.step(action)
                    total_reward += reward
                    terminated = term
                    print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}")

        # --- Rendering ---
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit FPS

    print("Game Over!")
    env.close()