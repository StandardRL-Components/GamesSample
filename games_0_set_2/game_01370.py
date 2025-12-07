
# Generated: 2025-08-27T16:55:06.191245
# Source Brief: brief_01370.md
# Brief Index: 1370

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. In combat, hold Space to attack or Shift to defend. "
        "Your turn is consumed by any action."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based roguelike. Explore a dark dungeon, fight Lovecraftian horrors, and manage your sanity to find the exit and escape."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 25
    GRID_HEIGHT = 15
    TILE_SIZE = 24
    MAX_STEPS = 1000
    NUM_MONSTERS = 5

    # --- Colors ---
    COLOR_BG = (10, 5, 15)
    COLOR_WALL = (40, 30, 50)
    COLOR_FLOOR = (60, 50, 70)
    COLOR_PLAYER = (255, 255, 100)
    COLOR_EXIT = (100, 255, 255)
    COLOR_MONSTER = (150, 40, 180)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH = (200, 30, 30)
    COLOR_HEALTH_BG = (80, 15, 15)
    COLOR_SANITY = (180, 50, 220)
    COLOR_SANITY_BG = (60, 20, 80)
    COLOR_UI_BG = (20, 15, 30)

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
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)
        
        # This will be initialized in reset()
        self.dungeon_layout = None
        self.player_pos = None
        self.exit_pos = None
        self.monsters = None
        self.player_health = 0
        self.player_sanity = 0
        self.max_player_health = 10
        self.max_player_sanity = 5
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_message = ""
        self.current_monster_in_combat = None
        self.player_is_defending = False
        self.monster_stats_multiplier = 1.0
        self.effects = deque()
        self.last_dist_to_exit = 0
        self.last_sanity = 0

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.dungeon_layout = self._generate_dungeon(self.GRID_WIDTH, self.GRID_HEIGHT)
        self.player_pos = self._find_start_pos()
        self.exit_pos = self._find_furthest_pos(self.player_pos)
        self.dungeon_layout[self.exit_pos[1]][self.exit_pos[0]] = 2 # Mark exit tile

        self.player_health = self.max_player_health
        self.player_sanity = self.max_player_sanity
        
        self.monsters = self._place_monsters(self.NUM_MONSTERS)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_message = ""
        self.current_monster_in_combat = None
        self.player_is_defending = False
        self.monster_stats_multiplier = 1.0
        self.effects = deque()

        self.last_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
        self.last_sanity = self.player_sanity

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        player_action = "WAIT"
        if space_held: player_action = "ATTACK"
        elif shift_held: player_action = "DEFEND"
        elif movement != 0: player_action = "MOVE"

        # --- Player Turn ---
        if self.current_monster_in_combat: # In Combat
            if player_action == "ATTACK":
                reward += self._player_attack()
            elif player_action == "DEFEND":
                self.player_is_defending = True
                self._monster_turn()
            else: # Moving or waiting in combat forfeits turn
                self._monster_turn()
        else: # Exploration
            if player_action == "MOVE":
                self._player_move(movement)

        # --- Reward Calculation ---
        dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
        if dist_to_exit < self.last_dist_to_exit:
            reward += 0.1
        self.last_dist_to_exit = dist_to_exit
        
        sanity_loss = self.last_sanity - self.player_sanity
        if sanity_loss > 0:
            reward -= 0.1 * sanity_loss
        self.last_sanity = self.player_sanity

        self.score += reward
        terminated = self._check_termination()
        
        if terminated and not self.game_message:
            if self.player_health <= 0:
                reward -= 100
                self.game_message = "YOU DIED"
            elif self.player_sanity <= 0:
                reward -= 100
                self.game_message = "LOST TO MADNESS"
            elif self.steps >= self.MAX_STEPS:
                self.game_message = "TIME'S UP"
            elif tuple(self.player_pos) == self.exit_pos:
                reward += 100
                self.game_message = "YOU ESCAPED!"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _player_attack(self):
        # sound: player_attack.wav
        monster = self.current_monster_in_combat
        damage = self.np_random.integers(1, 4)
        monster['health'] = max(0, monster['health'] - damage)
        self._add_effect('hit', monster['pos'], f"-{damage}", self.COLOR_HEALTH)
        
        reward = 1.0
        
        if monster['health'] <= 0:
            # sound: monster_die.wav
            reward += 5.0
            self.monsters.remove(monster)
            self.current_monster_in_combat = None
            self.monster_stats_multiplier *= 1.10
        else:
            self._monster_turn()
        return reward

    def _player_move(self, movement_action):
        dx, dy = 0, 0
        if movement_action == 1: dy = -1 # Up
        elif movement_action == 2: dy = 1 # Down
        elif movement_action == 3: dx = -1 # Left
        elif movement_action == 4: dx = 1 # Right

        new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

        if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
            if self.dungeon_layout[new_pos[1]][new_pos[0]] != 1: # Not a wall
                self.player_pos = new_pos
                # Check for monster encounter
                for monster in self.monsters:
                    if tuple(monster['pos']) == tuple(self.player_pos):
                        self.current_monster_in_combat = monster
                        # sound: encounter.wav
                        break

    def _monster_turn(self):
        if not self.current_monster_in_combat: return

        monster = self.current_monster_in_combat
        damage = monster['attack']
        sanity_drain = monster['sanity_drain']

        if self.player_is_defending:
            damage = math.ceil(damage * 0.5)
            sanity_drain = math.ceil(sanity_drain * 0.5)
            self.player_is_defending = False
            # sound: player_defend.wav
        
        # sound: player_damage.wav
        self.player_health = max(0, self.player_health - damage)
        self.player_sanity = max(0, self.player_sanity - sanity_drain)
        
        if damage > 0:
            self._add_effect('hit', self.player_pos, f"-{damage}", self.COLOR_HEALTH)
        if sanity_drain > 0:
            self._add_effect('hit', [self.player_pos[0]+0.5, self.player_pos[1]-0.5], f"-{sanity_drain}", self.COLOR_SANITY)
            self._add_effect('sanity_drain_screen', None, None, None)

    def _check_termination(self):
        if self.game_over: return True
        
        if self.player_health <= 0 or self.player_sanity <= 0 or \
           tuple(self.player_pos) == self.exit_pos or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self._update_effects()
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "player_sanity": self.player_sanity,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
            "monsters_remaining": len(self.monsters),
        }

    # --- Rendering Methods ---
    def _render_game(self):
        cam_x = self.SCREEN_WIDTH / 2 - (self.player_pos[0] + 0.5) * self.TILE_SIZE
        cam_y = self.SCREEN_HEIGHT / 2 - (self.player_pos[1] + 0.5) * self.TILE_SIZE

        # Render dungeon
        for y, row in enumerate(self.dungeon_layout):
            for x, tile in enumerate(row):
                rect = pygame.Rect(cam_x + x * self.TILE_SIZE, cam_y + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if tile == 1: # Wall
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                elif tile == 0: # Floor
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
                elif tile == 2: # Exit
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
                    pygame.gfxdraw.filled_circle(self.screen, int(rect.centerx), int(rect.centery), int(self.TILE_SIZE * 0.4), self.COLOR_EXIT)
        
        # Render monsters
        for monster in self.monsters:
            m_pos_x = cam_x + monster['pos'][0] * self.TILE_SIZE + self.TILE_SIZE // 2
            m_pos_y = cam_y + monster['pos'][1] * self.TILE_SIZE + self.TILE_SIZE // 2
            size = int(self.TILE_SIZE * 0.35)
            points = [
                (m_pos_x, m_pos_y - size), (m_pos_x + size, m_pos_y + size),
                (m_pos_x - size, m_pos_y + size)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_MONSTER, points)
            if self.current_monster_in_combat == monster:
                 self._render_monster_health(monster, cam_x, cam_y)

        # Render player
        p_pos_x = self.SCREEN_WIDTH // 2
        p_pos_y = self.SCREEN_HEIGHT // 2
        pygame.gfxdraw.filled_circle(self.screen, p_pos_x, p_pos_y, int(self.TILE_SIZE * 0.3), self.COLOR_PLAYER)

        # Render torchlight effect
        self._render_torchlight(p_pos_x, p_pos_y)

        # Render effects
        for effect in list(self.effects):
            if effect['type'] == 'hit':
                e_pos_x = cam_x + effect['pos'][0] * self.TILE_SIZE + self.TILE_SIZE // 2
                e_pos_y = cam_y + effect['pos'][1] * self.TILE_SIZE + self.TILE_SIZE // 2
                alpha = effect['duration'] * 2.55
                text_surf = self.font_medium.render(effect['text'], True, effect['color'])
                text_surf.set_alpha(alpha)
                self.screen.blit(text_surf, (e_pos_x, e_pos_y - (100 - effect['duration'])))
            elif effect['type'] == 'sanity_drain_screen':
                overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                alpha = effect['duration'] * 2.55
                overlay.fill((self.COLOR_SANITY[0], self.COLOR_SANITY[1], self.COLOR_SANITY[2], alpha * 0.5))
                self.screen.blit(overlay, (0, 0))

    def _render_torchlight(self, p_x, p_y):
        light_radius = 200 + math.sin(pygame.time.get_ticks() * 0.01) * 5
        dark_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), flags=pygame.SRCALPHA)
        dark_surface.fill((0, 0, 0, 255))
        pygame.draw.circle(dark_surface, (0, 0, 0, 0), (p_x, p_y), light_radius)
        self.screen.blit(dark_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, 40))
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.SCREEN_WIDTH - 100, 0, 100, 100))

        # Health Bar
        self._render_bar(10, 10, 150, 20, self.player_health / self.max_player_health, self.COLOR_HEALTH, self.COLOR_HEALTH_BG, "HP")
        
        # Sanity Bar
        self._render_bar(170, 10, 150, 20, self.player_sanity / self.max_player_sanity, self.COLOR_SANITY, self.COLOR_SANITY_BG, "SAN")

        # Minimap
        self._render_minimap(self.SCREEN_WIDTH - 95, 5)

        # Combat Info
        if self.current_monster_in_combat:
            text = self.font_medium.render("COMBAT!", True, self.COLOR_HEALTH)
            self.screen.blit(text, (self.SCREEN_WIDTH / 2 - text.get_width() / 2, 10))

    def _render_monster_health(self, monster, cam_x, cam_y):
        bar_width = self.TILE_SIZE
        bar_height = 5
        x = cam_x + monster['pos'][0] * self.TILE_SIZE
        y = cam_y + monster['pos'][1] * self.TILE_SIZE - 10
        ratio = monster['health'] / monster['max_health']
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (x, y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (x, y, bar_width * ratio, bar_height))


    def _render_bar(self, x, y, w, h, percent, color, bg_color, label):
        percent = max(0, min(1, percent))
        pygame.draw.rect(self.screen, bg_color, (x, y, w, h))
        pygame.draw.rect(self.screen, color, (x, y, w * percent, h))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (x, y, w, h), 1)
        label_surf = self.font_small.render(label, True, self.COLOR_TEXT)
        self.screen.blit(label_surf, (x + 5, y + 2))

    def _render_minimap(self, x_offset, y_offset):
        map_size = 90
        tile_w = map_size / self.GRID_WIDTH
        tile_h = map_size / self.GRID_HEIGHT
        for y, row in enumerate(self.dungeon_layout):
            for x, tile in enumerate(row):
                color = self.COLOR_WALL
                if tile == 0: color = self.COLOR_FLOOR
                elif tile == 2: color = self.COLOR_EXIT
                pygame.draw.rect(self.screen, color, (x_offset + x * tile_w, y_offset + y * tile_h, tile_w, tile_h))

        # Player on minimap
        p_x = x_offset + self.player_pos[0] * tile_w
        p_y = y_offset + self.player_pos[1] * tile_h
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (p_x, p_y, tile_w, tile_h))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text_surf = self.font_large.render(self.game_message, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    # --- Game Logic Helpers ---
    def _generate_dungeon(self, width, height):
        grid = np.ones((height, width), dtype=int)
        start_x, start_y = (self.np_random.integers(1, width//2) * 2 - 1, self.np_random.integers(1, height//2) * 2 - 1)
        
        stack = [(start_x, start_y)]
        grid[start_y, start_x] = 0

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < width - 1 and 0 < ny < height - 1 and grid[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                grid[ny, nx] = 0
                grid[cy + (ny-cy)//2, cx + (nx-cx)//2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return grid.tolist()

    def _find_start_pos(self):
        for y, row in enumerate(self.dungeon_layout):
            for x, tile in enumerate(row):
                if tile == 0: return [x, y]
        return [1, 1]

    def _find_furthest_pos(self, start_pos):
        q = deque([(start_pos, 0)])
        visited = {tuple(start_pos)}
        furthest_pos, max_dist = start_pos, 0
        
        while q:
            pos, dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                furthest_pos = pos

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (pos[0] + dx, pos[1] + dy)
                if 0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT and \
                   self.dungeon_layout[next_pos[1]][next_pos[0]] == 0 and next_pos not in visited:
                    visited.add(next_pos)
                    q.append((next_pos, dist + 1))
        return list(furthest_pos)
    
    def _place_monsters(self, num_monsters):
        monsters = []
        floor_tiles = []
        for y, row in enumerate(self.dungeon_layout):
            for x, tile in enumerate(row):
                if tile == 0 and [x, y] != self.player_pos and [x, y] != self.exit_pos:
                    floor_tiles.append([x, y])
        
        self.np_random.shuffle(floor_tiles)
        
        for i in range(min(num_monsters, len(floor_tiles))):
            base_health = 5
            base_attack = 1
            base_sanity_drain = 1
            
            monster = {
                'pos': floor_tiles[i],
                'max_health': int(base_health * (1 + i * 0.1)),
                'health': int(base_health * (1 + i * 0.1)),
                'attack': int(base_attack * (1 + i * 0.1)),
                'sanity_drain': int(base_sanity_drain * (1 + i * 0.1)),
            }
            monsters.append(monster)
        return monsters

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _add_effect(self, type, pos, text, color):
        self.effects.append({'type': type, 'pos': pos, 'text': text, 'color': color, 'duration': 100})
    
    def _update_effects(self):
        for _ in range(len(self.effects)):
            effect = self.effects.popleft()
            effect['duration'] -= 10
            if effect['duration'] > 0:
                self.effects.append(effect)
    
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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # Manual play loop
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    print(env.user_guide)

    while not done:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # Only step if an action is taken in a turn-based game
        if movement or space or shift or True: # For this manual test, we always want to see the result
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Wait for the next action
        pygame.time.wait(100) # Wait 100ms to make turns manually playable

    pygame.quit()
    print(f"Game Over. Final Score: {info['score']:.2f}, Steps: {info['steps']}")