# Generated: 2025-08-27T16:55:18.635331
# Source Brief: brief_01372.md
# Brief Index: 1372

        
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
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Press space to attack in the direction you last moved."
    )

    game_description = (
        "Navigate a procedurally generated grid-based dungeon, battling simple enemies and collecting gold to reach the exit."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 20
    TILE_SIZE = 20
    
    MAX_EPISODE_STEPS = 1500
    MAX_STAGES = 3
    
    PLAYER_MAX_HEALTH = 10
    
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_WALL = (70, 70, 80)
    COLOR_FLOOR = (40, 40, 50)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_EXIT = (50, 255, 50)
    COLOR_GOLD = (255, 223, 0)
    COLOR_ENEMY_BLUE = (50, 150, 255)
    COLOR_ENEMY_YELLOW = (255, 200, 50)
    COLOR_ENEMY_PURPLE = (200, 50, 255)
    COLOR_HEALTH_BAR = (50, 200, 50)
    COLOR_HEALTH_BAR_BG = (100, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_WHITE = (255, 255, 255)

    # Tile types
    TILE_FLOOR = 0
    TILE_WALL = 1

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
        self.font_large = pygame.font.SysFont("monospace", 24)

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.enemies = []
        self.gold_pieces = []
        self.particles = []
        self._flash_color = None

        self.reset()
        
        # This check is disabled by default but can be run manually.
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_pos = (0, 0)
        self.player_last_move_dir = (0, 1) # Default: down
        self._flash_color = None

        self._generate_stage()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Cost of living
        self.steps += 1
        self._flash_color = None
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        old_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)

        # --- Player Action Phase ---
        player_acted = False
        if movement > 0: # 1-4 are moves
            player_acted = True
            move_dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = move_dirs[movement]
            self.player_last_move_dir = (dx, dy)
            
            target_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            if self.grid[target_pos[0]][target_pos[1]] != self.TILE_WALL:
                self.player_pos = target_pos
                
                # Check for gold
                if self.player_pos in self.gold_pieces:
                    self.gold_pieces.remove(self.player_pos)
                    self.score += 1
                    reward += 1.0
                    # Gold collection particles
                    self._spawn_particles(self.player_pos, self.COLOR_GOLD, 10)
                
                # Check for exit
                if self.player_pos == self.exit_pos:
                    if self.stage < self.MAX_STAGES:
                        self.stage += 1
                        reward += 10.0
                        self._generate_stage()
                        # Reset distance calculation for the new stage
                        old_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
                    else:
                        self.game_over = True # Win condition
                        reward += 100.0

        elif space_held: # Attack action
            player_acted = True
            target_pos = (self.player_pos[0] + self.player_last_move_dir[0], 
                          self.player_pos[1] + self.player_last_move_dir[1])
            
            enemy_to_attack = self._get_enemy_at(target_pos)
            if enemy_to_attack:
                player_attack_power = 1
                enemy_to_attack['health'] -= player_attack_power
                self._flash_color = self.COLOR_WHITE # Attack flash
                # Enemy hit particles
                self._spawn_particles(target_pos, self.COLOR_PLAYER, 5)

                if enemy_to_attack['health'] <= 0:
                    self.score += 5
                    reward += 5.0
                    # Enemy death particles
                    self._spawn_particles(target_pos, enemy_to_attack['color'], 20)
                    self.enemies.remove(enemy_to_attack)
            # else: no-op if no enemy to attack

        if player_acted:
            # --- Enemy Action Phase ---
            for enemy in self.enemies[:]: # Iterate over a copy
                if not self._is_enemy_alive(enemy): continue

                # Check for adjacent player
                attacked = False
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if (enemy['pos'][0] + dx, enemy['pos'][1] + dy) == self.player_pos:
                        self.player_health -= enemy['attack']
                        self._flash_color = self.COLOR_PLAYER # Damage flash
                        attacked = True
                        break
                
                if not attacked:
                    # Enemy movement AI
                    if enemy['type'] == 'yellow': # Horizontal patrol
                        tx, ty = enemy['pos'][0] + enemy['dir'], enemy['pos'][1]
                        if self._is_pos_free((tx, ty)):
                            enemy['pos'] = (tx, ty)
                        else:
                            enemy['dir'] *= -1
                    elif enemy['type'] == 'purple': # Random move
                        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                        self.rng.shuffle(moves)
                        for dx, dy in moves:
                            tx, ty = enemy['pos'][0] + dx, enemy['pos'][1] + dy
                            if self._is_pos_free((tx, ty)):
                                enemy['pos'] = (tx, ty)
                                break
        
        # --- Update rewards and termination ---
        new_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
        if new_dist_to_exit < old_dist_to_exit:
            reward += 0.1
        elif new_dist_to_exit > old_dist_to_exit:
            reward -= 0.1

        self._update_particles()
        
        if self.player_health <= 0:
            self.game_over = True
            reward = -100.0
        
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_stage(self):
        # 1. Generate a valid map layout
        while True:
            self.grid.fill(self.TILE_FLOOR)
            # Create borders
            self.grid[0, :] = self.TILE_WALL
            self.grid[-1, :] = self.TILE_WALL
            self.grid[:, 0] = self.TILE_WALL
            self.grid[:, -1] = self.TILE_WALL

            # Place random walls
            num_walls = self.rng.integers(40, 80)
            for _ in range(num_walls):
                x, y = self.rng.integers(1, self.GRID_WIDTH-1), self.rng.integers(1, self.GRID_HEIGHT-1)
                self.grid[x, y] = self.TILE_WALL
            
            # 2. Get all possible spawn points
            floor_tiles = self._get_all_floor_tiles()
            if len(floor_tiles) < 20: continue # Not enough space, retry

            self.rng.shuffle(floor_tiles)

            # 3. Place player and exit
            self.player_pos = floor_tiles.pop()
            self.exit_pos = floor_tiles.pop()

            # 4. Check for path, if not, regenerate
            if self._find_path(self.player_pos, self.exit_pos):
                break
        
        # 5. Place enemies and gold
        self.enemies.clear()
        self.gold_pieces.clear()
        floor_tiles = self._get_all_floor_tiles()
        self.rng.shuffle(floor_tiles)
        
        # Define enemy types and numbers
        enemy_counts = {'blue': 3, 'yellow': 2, 'purple': 2}
        stage_multiplier = self.stage - 1
        
        for _ in range(enemy_counts['blue']):
            if not floor_tiles: break
            self.enemies.append({
                'pos': floor_tiles.pop(), 'type': 'blue', 'color': self.COLOR_ENEMY_BLUE,
                'health': 1 + stage_multiplier, 'max_health': 1 + stage_multiplier,
                'attack': 1 + stage_multiplier
            })
        for _ in range(enemy_counts['yellow']):
            if not floor_tiles: break
            self.enemies.append({
                'pos': floor_tiles.pop(), 'type': 'yellow', 'color': self.COLOR_ENEMY_YELLOW,
                'health': 2 + stage_multiplier, 'max_health': 2 + stage_multiplier,
                'attack': 1 + stage_multiplier, 'dir': 1 if self.rng.random() > 0.5 else -1
            })
        for _ in range(enemy_counts['purple']):
            if not floor_tiles: break
            self.enemies.append({
                'pos': floor_tiles.pop(), 'type': 'purple', 'color': self.COLOR_ENEMY_PURPLE,
                'health': 1 + stage_multiplier, 'max_health': 1 + stage_multiplier,
                'attack': 2 + stage_multiplier
            })

        # Place gold
        num_gold = self.rng.integers(5, 11)
        for _ in range(num_gold):
            if not floor_tiles: break
            self.gold_pieces.append(floor_tiles.pop())
        
        self.particles.clear()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self._flash_color:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*self._flash_color, 100))
            self.screen.blit(flash_surface, (0, 0))
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "player_health": self.player_health
        }

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.grid[x, y] == self.TILE_WALL else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw exit
        ex, ey = self.exit_pos
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex * self.TILE_SIZE, ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        # Draw gold
        for gx, gy in self.gold_pieces:
            pygame.gfxdraw.filled_circle(self.screen, 
                                         int(gx * self.TILE_SIZE + self.TILE_SIZE / 2),
                                         int(gy * self.TILE_SIZE + self.TILE_SIZE / 2),
                                         int(self.TILE_SIZE * 0.3), self.COLOR_GOLD)

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            rect = (ex * self.TILE_SIZE, ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, enemy['color'], rect)

        # Draw player
        px, py = self.player_pos
        player_rect = (px * self.TILE_SIZE, py * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, player_rect, 1) # Outline

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

    def _render_ui(self):
        # Health bar
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_small.render(f"HP: {self.player_health}/{self.PLAYER_MAX_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Stage text
        stage_text = self.font_large.render(f"Stage: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH - stage_text.get_width() - 10, 10))

        # Score text
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH / 2 - score_text.get_width() / 2, 10))

    # --- Helper Functions ---
    
    def _get_all_floor_tiles(self):
        return [(x, y) for x in range(1, self.GRID_WIDTH - 1) for y in range(1, self.GRID_HEIGHT - 1) if self.grid[x, y] == self.TILE_FLOOR]

    def _find_path(self, start, end):
        q = deque([start])
        visited = {start}
        while q:
            x, y = q.popleft()
            if (x, y) == end:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and self.grid[nx, ny] == self.TILE_FLOOR:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def _get_enemy_at(self, pos):
        for enemy in self.enemies:
            if enemy['pos'] == pos:
                return enemy
        return None

    def _is_pos_free(self, pos):
        if self.grid[pos[0]][pos[1]] == self.TILE_WALL: return False
        if self.player_pos == pos: return False
        if self._get_enemy_at(pos): return False
        return True

    def _is_enemy_alive(self, enemy):
        return enemy in self.enemies

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _spawn_particles(self, grid_pos, color, count):
        px, py = (grid_pos[0] + 0.5) * self.TILE_SIZE, (grid_pos[1] + 0.5) * self.TILE_SIZE
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2 + 1
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.rng.random() * 2 + 2,
                'color': color,
                'life': self.rng.integers(10, 20)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
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

    def close(self):
        pygame.quit()