
# Generated: 2025-08-28T04:45:04.157871
# Source Brief: brief_05352.md
# Brief Index: 5352

        
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
        "Controls: Arrow keys to move. Space to attack. Shift to use a potion."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated dungeon, battling enemies and collecting gold to reach the exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 20, 12
        self.TILE_SIZE = min(self.WIDTH // self.GRID_W, self.HEIGHT // self.GRID_H)
        self.CENTER_X = (self.WIDTH - self.GRID_W * self.TILE_SIZE) // 2
        self.CENTER_Y = (self.HEIGHT - self.GRID_H * self.TILE_SIZE) // 2

        self.MAX_STEPS = 2000
        self.MAX_PLAYER_HEALTH = 10
        self.POTION_HEAL_AMOUNT = 2
        self.ENEMY_RESPAWN_TURNS = 5

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (60, 60, 80)
        self.COLOR_FLOOR = (40, 40, 60)
        self.COLOR_PLAYER = (50, 200, 50)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_POTION = (50, 150, 255)
        self.COLOR_EXIT = (150, 50, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HIT_FLASH = (255, 100, 100, 128)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # RNG
        self.np_random = None
        
        # Initialize state variables
        self.grid = None
        self.player_pos = None
        self.player_health = None
        self.player_potions = None
        self.player_last_move_dir = None
        self.enemies = None
        self.dead_enemies = None
        self.items = None
        self.exit_pos = None
        self.level = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.particles = None
        self.screen_flash_timer = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None or seed is not None:
             self.np_random = np.random.default_rng(seed=seed)
        
        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.screen_flash_timer = 0
        self.dead_enemies = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        while True:
            self._create_base_grid()
            self._place_features()
            if self._is_level_valid():
                break

        self.player_health = self.MAX_PLAYER_HEALTH
        self.player_potions = 1
        self.player_last_move_dir = (0, 1)

        self._spawn_entities()

    def _create_base_grid(self):
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        num_pillars = self.np_random.integers(3, 7)
        for _ in range(num_pillars):
            px = self.np_random.integers(2, self.GRID_W - 2)
            py = self.np_random.integers(2, self.GRID_H - 2)
            self.grid[px, py] = 1

    def _get_open_tiles(self):
        return list(zip(*np.where(self.grid == 0)))

    def _place_features(self):
        open_tiles = self._get_open_tiles()
        self.np_random.shuffle(open_tiles)
        
        self.exit_pos = open_tiles.pop()
        self.player_pos = open_tiles.pop()

    def _is_level_valid(self):
        if not self.player_pos or not self.exit_pos:
            return False
        
        q = deque([self.player_pos])
        visited = {self.player_pos}
        
        while q:
            x, y = q.popleft()
            if (x, y) == self.exit_pos:
                return True
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and self.grid[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False
        
    def _spawn_entities(self):
        self.enemies = []
        self.items = {}

        open_tiles = self._get_open_tiles()
        if self.player_pos in open_tiles: open_tiles.remove(self.player_pos)
        if self.exit_pos in open_tiles: open_tiles.remove(self.exit_pos)
        self.np_random.shuffle(open_tiles)

        num_enemies = 1 + (self.level - 1) // 2
        enemy_health = 2 + (self.level - 1)
        for _ in range(min(num_enemies, len(open_tiles))):
            pos = open_tiles.pop()
            self.enemies.append({'pos': pos, 'health': enemy_health, 'max_health': enemy_health})

        num_gold = self.np_random.integers(2, 5)
        for _ in range(min(num_gold, len(open_tiles))):
            pos = open_tiles.pop()
            self.items[pos] = 'gold'
            
        num_potions = self.np_random.integers(1, 3)
        for _ in range(min(num_potions, len(open_tiles))):
            pos = open_tiles.pop()
            self.items[pos] = 'potion'

        player_neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = self.player_pos[0] + dx, self.player_pos[1] + dy
                if (nx, ny) in open_tiles:
                    player_neighbors.append((nx, ny))
        
        if player_neighbors:
            potion_pos = player_neighbors[self.np_random.integers(len(player_neighbors))]
            self.items[potion_pos] = 'potion'
            
    def _next_level(self):
        self.level += 1
        self.score += 25
        self._generate_level()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        player_acted = False

        if shift_held and self.player_potions > 0 and self.player_health < self.MAX_PLAYER_HEALTH:
            self.player_potions -= 1
            healed_amount = min(self.POTION_HEAL_AMOUNT, self.MAX_PLAYER_HEALTH - self.player_health)
            self.player_health += healed_amount
            reward += 5
            player_acted = True
            self._spawn_particles(self.player_pos, self.COLOR_POTION, 20)
            # sfx: potion drink

        elif space_held:
            attack_pos = (self.player_pos[0] + self.player_last_move_dir[0], 
                          self.player_pos[1] + self.player_last_move_dir[1])
            self._spawn_attack_particles(attack_pos, self.COLOR_TEXT)
            # sfx: sword swing
            
            enemy_to_attack = next((e for e in self.enemies if e['pos'] == attack_pos), None)
            
            if enemy_to_attack:
                enemy_to_attack['health'] -= 1
                self._spawn_particles(enemy_to_attack['pos'], self.COLOR_ENEMY, 15)
                # sfx: enemy hit
                if enemy_to_attack['health'] <= 0:
                    self.enemies.remove(enemy_to_attack)
                    self.dead_enemies.append({'pos': enemy_to_attack['pos'], 'turn_died': self.steps})
                    reward += 1
                    self.score += 10
                    # sfx: enemy death
            player_acted = True

        elif movement > 0:
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = move_map[movement]
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

            is_wall = self.grid[new_pos[0], new_pos[1]] == 1
            is_enemy = any(e['pos'] == new_pos for e in self.enemies)

            if not is_wall and not is_enemy:
                dist_before = math.hypot(self.player_pos[0] - self.exit_pos[0], self.player_pos[1] - self.exit_pos[1])
                dist_after = math.hypot(new_pos[0] - self.exit_pos[0], new_pos[1] - self.exit_pos[1])
                reward += 0.1 if dist_after < dist_before else -0.1

                self.player_pos = new_pos
                self.player_last_move_dir = (dx, dy)

                if self.player_pos in self.items:
                    item_type = self.items.pop(self.player_pos)
                    color, points, item_reward = (self.COLOR_GOLD, 5, 2) if item_type == 'gold' else (self.COLOR_POTION, 0, 5)
                    self.score += points
                    reward += item_reward
                    if item_type == 'potion': self.player_potions += 1
                    self._spawn_particles(self.player_pos, color, 10)
                    # sfx: item pickup
                
                if self.player_pos == self.exit_pos:
                    reward += 100
                    self.score += 50
                    self._next_level()
                    # sfx: level complete
            player_acted = True
            
        elif movement == 0:
            player_acted = True

        if player_acted:
            if self.dead_enemies and self.steps - self.dead_enemies[0]['turn_died'] >= self.ENEMY_RESPAWN_TURNS:
                respawn_pos = self.dead_enemies[0]['pos']
                is_free = self.grid[respawn_pos] == 0 and self.player_pos != respawn_pos and not any(e['pos'] == respawn_pos for e in self.enemies)
                if is_free:
                    self.dead_enemies.pop(0)
                    enemy_health = 2 + (self.level - 1)
                    self.enemies.append({'pos': respawn_pos, 'health': enemy_health, 'max_health': enemy_health})
                    self._spawn_particles(respawn_pos, self.COLOR_ENEMY, 30)
                    # sfx: enemy respawn

            for enemy in self.enemies:
                ex, ey = enemy['pos']
                px, py = self.player_pos
                
                if abs(ex - px) + abs(ey - py) == 1:
                    self.player_health -= 1
                    self.screen_flash_timer = 5
                    self._spawn_particles(self.player_pos, self.COLOR_PLAYER, 10)
                    # sfx: player hit
                else:
                    best_move, min_dist = (0, 0), float('inf')
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = ex + dx, ey + dy
                        is_valid = self.grid[nx, ny] == 0 and (nx, ny) != self.player_pos and not any(e['pos'] == (nx, ny) for e in self.enemies if e is not enemy)
                        if is_valid:
                            dist = math.hypot(nx - px, ny - py)
                            if dist < min_dist:
                                min_dist, best_move = dist, (dx, dy)
                    enemy['pos'] = (ex + best_move[0], ey + best_move[1])

        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] = max(0, p['size'] * 0.95)
        if self.screen_flash_timer > 0: self.screen_flash_timer -= 1

        terminated = False
        if self.player_health <= 0:
            reward = -100
            terminated = self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                rect = (self.CENTER_X + x * self.TILE_SIZE, self.CENTER_Y + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.grid[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

        for pos, item_type in self.items.items():
            color = self.COLOR_GOLD if item_type == 'gold' else self.COLOR_POTION
            cx = self.CENTER_X + int((pos[0] + 0.5) * self.TILE_SIZE)
            cy = self.CENTER_Y + int((pos[1] + 0.5) * self.TILE_SIZE)
            pygame.draw.circle(self.screen, color, (cx, cy), self.TILE_SIZE // 4)

        ex_rect = pygame.Rect(self.CENTER_X + self.exit_pos[0] * self.TILE_SIZE, self.CENTER_Y + self.exit_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, ex_rect, border_radius=4)
        pygame.draw.rect(self.screen, tuple(min(255, c+40) for c in self.COLOR_EXIT), ex_rect.inflate(-8, -8), border_radius=4)

        for enemy in self.enemies:
            e_rect = pygame.Rect(self.CENTER_X + enemy['pos'][0] * self.TILE_SIZE, self.CENTER_Y + enemy['pos'][1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, e_rect)
            hp_ratio = enemy['health'] / enemy['max_health']
            hp_bar_width = int(self.TILE_SIZE * 0.8)
            hp_bar_rect_bg = (e_rect.left + (self.TILE_SIZE - hp_bar_width)//2, e_rect.top - 7, hp_bar_width, 5)
            hp_bar_rect_fg = (e_rect.left + (self.TILE_SIZE - hp_bar_width)//2, e_rect.top - 7, int(hp_bar_width * hp_ratio), 5)
            pygame.draw.rect(self.screen, (50,0,0), hp_bar_rect_bg)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, hp_bar_rect_fg)

        p_rect = pygame.Rect(self.CENTER_X + self.player_pos[0] * self.TILE_SIZE, self.CENTER_Y + self.player_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, p_rect)
        pygame.gfxdraw.filled_circle(self.screen, p_rect.centerx, p_rect.centery, self.TILE_SIZE//2 + 2, (*self.COLOR_PLAYER, 30))
        
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['size']))
            
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_HIT_FLASH)
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        health_text = self.font_small.render(f"HP: {self.player_health}/{self.MAX_PLAYER_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        potion_text = self.font_small.render(f"Potions: {self.player_potions}", True, self.COLOR_POTION)
        self.screen.blit(potion_text, (10, 30))
        
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_GOLD)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        level_text = self.font_small.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(bottomright=(self.WIDTH - 10, self.HEIGHT - 10))
        self.screen.blit(level_text, level_rect)

        if self.game_over and self.player_health <= 0:
            game_over_text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
            text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "player_health": self.player_health,
            "player_potions": self.player_potions
        }

    def _spawn_particles(self, pos_grid, color, count):
        px = self.CENTER_X + (pos_grid[0] + 0.5) * self.TILE_SIZE
        py = self.CENTER_Y + (pos_grid[1] + 0.5) * self.TILE_SIZE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'size': self.np_random.uniform(2, 6),
                'life': self.np_random.integers(10, 20)
            })

    def _spawn_attack_particles(self, pos_grid, color):
        px_start = self.CENTER_X + (self.player_pos[0] + 0.5) * self.TILE_SIZE
        py_start = self.CENTER_Y + (self.player_pos[1] + 0.5) * self.TILE_SIZE
        px_end = self.CENTER_X + (pos_grid[0] + 0.5) * self.TILE_SIZE
        py_end = self.CENTER_Y + (pos_grid[1] + 0.5) * self.TILE_SIZE
        
        for i in range(10):
            t = i / 9.0
            px = px_start + (px_end - px_start) * t
            py = py_start + (py_end - py_start) * t
            self.particles.append({
                'pos': [px, py],
                'vel': [(self.np_random.uniform(-0.5, 0.5)), (self.np_random.uniform(-0.5, 0.5))],
                'color': color,
                'size': self.np_random.uniform(1, 3),
                'life': self.np_random.integers(5, 10)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")