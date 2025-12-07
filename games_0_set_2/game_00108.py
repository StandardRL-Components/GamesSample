
# Generated: 2025-08-27T12:37:37.757070
# Source Brief: brief_00108.md
# Brief Index: 108

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move, Space to attack adjacent enemies, Shift to defend (halves incoming damage)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore a procedurally generated dungeon, battling enemies and collecting gold to defeat the final boss."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Game world
    GRID_WIDTH = 40
    GRID_HEIGHT = 25
    TILE_SIZE = 16
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    MAX_LEVEL = 5

    # Colors
    COLOR_BG = (10, 8, 15)
    COLOR_FLOOR = (50, 40, 30)
    COLOR_WALL = (80, 70, 60)
    COLOR_PLAYER = (50, 200, 255)
    COLOR_PLAYER_GLOW = (50, 200, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_BOSS = (255, 100, 150)
    COLOR_GOLD = (255, 223, 0)
    COLOR_STAIRS = (150, 100, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BG = (30, 30, 40, 180)
    COLOR_HEALTH_BAR = (50, 255, 50)
    COLOR_HEALTH_BAR_BG = (200, 50, 50)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = (0, 0)
        self.player_health = 100
        self.player_max_health = 100
        self.is_defending = False
        self.level = 1
        self.grid = []
        self.enemies = []
        self.gold_pieces = []
        self.stairs_pos = (0, 0)
        self.boss = None
        self.particles = []
        self.rng = None
        self.last_distance_to_stairs = 0

        # Initialize state variables
        self.reset()

        # Run validation check
        # self.validate_implementation() # Commented out for final submission

    def _generate_level(self):
        self.grid = [[1] * self.GRID_HEIGHT for _ in range(self.GRID_WIDTH)] # 1 = wall
        self.enemies = []
        self.gold_pieces = []
        self.boss = None

        # Carve out rooms and corridors using random walks
        num_walks = 150
        walk_length = 50
        for _ in range(num_walks):
            cx, cy = self.rng.integers(1, self.GRID_WIDTH-1), self.rng.integers(1, self.GRID_HEIGHT-1)
            for _ in range(walk_length):
                if 0 < cx < self.GRID_WIDTH - 1 and 0 < cy < self.GRID_HEIGHT - 1:
                    self.grid[cx][cy] = 0 # 0 = floor
                dx, dy = self.rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
                cx, cy = cx + dx, cy + dy
        
        # Get all valid floor tiles
        floor_tiles = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT) if self.grid[x][y] == 0]
        if not floor_tiles: # Failsafe if generation is empty
            self.reset()
            return
            
        self.rng.shuffle(floor_tiles)

        # Place player
        self.player_pos = floor_tiles.pop()

        # Place stairs
        self.stairs_pos = floor_tiles.pop()

        # Place gold
        num_gold = 20
        for _ in range(min(num_gold, len(floor_tiles))):
            self.gold_pieces.append(floor_tiles.pop())

        # Place enemies or boss
        if self.level == self.MAX_LEVEL:
            boss_pos = floor_tiles.pop()
            self.boss = {
                'pos': boss_pos, 'health': 50, 'max_health': 50,
                'damage': 20 + (self.level - 1)
            }
        else:
            num_enemies = 3 + self.level
            for _ in range(min(num_enemies, len(floor_tiles))):
                enemy_pos = floor_tiles.pop()
                self.enemies.append({
                    'pos': enemy_pos, 'health': 20, 'max_health': 20,
                    'damage': 10 + (self.level - 1),
                    'patrol_dir': 1, 'patrol_steps': 0
                })
        
        self.last_distance_to_stairs = self._manhattan_distance(self.player_pos, self.stairs_pos)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.player_health = 100
        self.is_defending = False
        self.particles = []

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.is_defending = False
        
        # --- Player Action Phase ---
        action_taken = True
        if space_pressed: # Attack
            # sfx: player_attack.wav
            attacked = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                target_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
                
                # Check for enemy
                for enemy in self.enemies:
                    if enemy['pos'] == target_pos:
                        enemy['health'] -= 15
                        self._create_hit_particle(target_pos)
                        attacked = True
                        if enemy['health'] <= 0:
                            reward += 5
                            self.score += 10
                            self.enemies.remove(enemy)
                            # sfx: enemy_die.wav
                        break
                
                # Check for boss
                if self.boss and self.boss['pos'] == target_pos:
                    self.boss['health'] -= 15
                    self._create_hit_particle(target_pos)
                    attacked = True
                    if self.boss['health'] <= 0:
                        reward += 100
                        self.score += 100
                        self.boss = None
                        # sfx: boss_die.wav
                    break
        elif shift_pressed: # Defend
            self.is_defending = True
            # sfx: defend.wav
            self._create_shield_particle(self.player_pos)
        elif movement != 0: # Move
            px, py = self.player_pos
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
            nx, ny = px + dx, py + dy

            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx][ny] == 0:
                self.player_pos = (nx, ny)
                # sfx: player_step.wav
                
                # Reward for moving towards stairs
                new_dist = self._manhattan_distance(self.player_pos, self.stairs_pos)
                if new_dist < self.last_distance_to_stairs:
                    reward += 0.1
                else:
                    reward -= 0.1
                self.last_distance_to_stairs = new_dist
        else: # No-op
            action_taken = False

        # --- Game State Update Phase ---
        if action_taken:
            self.steps += 1
            
            # Gold collection
            if self.player_pos in self.gold_pieces:
                self.gold_pieces.remove(self.player_pos)
                reward += 1
                self.score += 5
                # sfx: collect_gold.wav
                self._create_gold_particle(self.player_pos)

            # Level transition
            if self.player_pos == self.stairs_pos and self.level < self.MAX_LEVEL:
                self.level += 1
                self.score += 50
                reward += 10
                self._generate_level()
                # sfx: level_up.wav

            # Enemy turn
            self._update_enemies()
            
            # Spawn new enemy
            if self.level < self.MAX_LEVEL and self.steps > 0 and self.steps % 25 == 0:
                self._spawn_enemy()
        
        self._update_particles()

        # --- Termination Check ---
        terminated = (
            self.player_health <= 0 or
            (self.level == self.MAX_LEVEL and not self.boss) or
            self.steps >= self.MAX_STEPS
        )
        if self.player_health <= 0:
            reward -= 20 # Penalty for dying
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_enemies(self):
        entities_to_attack = self.enemies + ([self.boss] if self.boss else [])
        for entity in entities_to_attack:
            ex, ey = entity['pos']
            px, py = self.player_pos
            
            # Attack if adjacent
            if abs(ex - px) + abs(ey - py) == 1:
                damage = entity['damage']
                if self.is_defending:
                    damage = math.ceil(damage / 2)
                self.player_health -= damage
                self.player_health = max(0, self.player_health)
                self._create_hit_particle(self.player_pos, self.COLOR_ENEMY)
                # sfx: player_hurt.wav
            # Move (only for non-boss enemies)
            elif 'patrol_dir' in entity:
                # Simple patrol: 2 steps up, 2 steps down
                if entity['patrol_steps'] < 2:
                    ney = ey - entity['patrol_dir']
                else:
                    ney = ey + entity['patrol_dir']
                
                if 0 <= ney < self.GRID_HEIGHT and self.grid[ex][ney] == 0 and (ex, ney) != self.player_pos:
                    entity['pos'] = (ex, ney)
                
                entity['patrol_steps'] = (entity['patrol_steps'] + 1) % 4

    def _spawn_enemy(self):
        floor_tiles = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT) if self.grid[x][y] == 0]
        occupied_tiles = [e['pos'] for e in self.enemies] + [self.player_pos]
        valid_spawns = [t for t in floor_tiles if t not in occupied_tiles and self._manhattan_distance(t, self.player_pos) > 5]
        
        if valid_spawns:
            pos = self.rng.choice(valid_spawns)
            self.enemies.append({
                'pos': tuple(pos), 'health': 20, 'max_health': 20,
                'damage': 10 + (self.level - 1),
                'patrol_dir': 1, 'patrol_steps': 0
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level, "health": self.player_health}

    def _render_game(self):
        # Camera offset to center player
        cam_x = self.player_pos[0] * self.TILE_SIZE - self.SCREEN_WIDTH / 2 + self.TILE_SIZE / 2
        cam_y = self.player_pos[1] * self.TILE_SIZE - self.SCREEN_HEIGHT / 2 + self.TILE_SIZE / 2

        # Render grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                screen_x, screen_y = int(x * self.TILE_SIZE - cam_x), int(y * self.TILE_SIZE - cam_y)
                if -self.TILE_SIZE < screen_x < self.SCREEN_WIDTH and -self.TILE_SIZE < screen_y < self.SCREEN_HEIGHT:
                    color = self.COLOR_WALL if self.grid[x][y] == 1 else self.COLOR_FLOOR
                    pygame.draw.rect(self.screen, color, (screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE))

        # Render gold
        for gx, gy in self.gold_pieces:
            screen_x, screen_y = int(gx * self.TILE_SIZE - cam_x), int(gy * self.TILE_SIZE - cam_y)
            pygame.draw.rect(self.screen, self.COLOR_GOLD, (screen_x + 4, screen_y + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8))

        # Render stairs
        sx, sy = self.stairs_pos
        screen_x, screen_y = int(sx * self.TILE_SIZE - cam_x), int(sy * self.TILE_SIZE - cam_y)
        pygame.draw.rect(self.screen, self.COLOR_STAIRS, (screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE))
        pygame.draw.rect(self.screen, (0,0,0), (screen_x+2, screen_y+2, self.TILE_SIZE-4, self.TILE_SIZE-4), 2)


        # Render enemies
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            screen_x, screen_y = int(ex * self.TILE_SIZE - cam_x), int(ey * self.TILE_SIZE - cam_y)
            bob = int(math.sin(self.steps * 0.3 + ex) * 2)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (screen_x + 2, screen_y + 2 + bob, self.TILE_SIZE - 4, self.TILE_SIZE - 4))
            self._draw_health_bar(screen_x, screen_y - 5, self.TILE_SIZE, 4, enemy['health'], enemy['max_health'])
            
        # Render boss
        if self.boss:
            bx, by = self.boss['pos']
            screen_x, screen_y = int(bx * self.TILE_SIZE - cam_x), int(by * self.TILE_SIZE - cam_y)
            bob = int(math.sin(self.steps * 0.2) * 3)
            pygame.draw.rect(self.screen, self.COLOR_BOSS, (screen_x-2, screen_y-2+bob, self.TILE_SIZE+4, self.TILE_SIZE+4))
            self._draw_health_bar(screen_x-2, screen_y - 8, self.TILE_SIZE+4, 6, self.boss['health'], self.boss['max_health'])


        # Render player
        px, py = self.player_pos
        screen_x, screen_y = int(px * self.TILE_SIZE - cam_x), int(py * self.TILE_SIZE - cam_y)
        bob = int(math.sin(self.steps * 0.5) * 3)
        
        # Glow effect
        glow_surf = pygame.Surface((self.TILE_SIZE * 2, self.TILE_SIZE * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE, self.COLOR_PLAYER_GLOW)
        self.screen.blit(glow_surf, (screen_x - self.TILE_SIZE/2, screen_y - self.TILE_SIZE/2 + bob))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (screen_x, screen_y + bob, self.TILE_SIZE, self.TILE_SIZE))

        # Render particles
        for p in self.particles:
            p_x, p_y = int(p['pos'][0] * self.TILE_SIZE - cam_x + self.TILE_SIZE/2 + p['offset'][0]), int(p['pos'][1] * self.TILE_SIZE - cam_y + self.TILE_SIZE/2 + p['offset'][1])
            pygame.draw.circle(self.screen, p['color'], (p_x, p_y), int(p['size']))

    def _render_ui(self):
        # Health Bar
        health_bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (8, 8, health_bar_width + 4, 24))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, health_bar_width, 20))
        health_ratio = self.player_health / self.player_max_health
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(health_bar_width * health_ratio), 20))
        health_text = self.font_small.render(f"HP: {self.player_health}/{self.player_max_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Gold Display
        gold_text = self.font_large.render(f"Gold: {self.score}", True, self.COLOR_GOLD)
        text_rect = gold_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, text_rect.inflate(10, 10))
        self.screen.blit(gold_text, text_rect)

        # Level Display
        level_text = self.font_large.render(f"Dungeon Level: {self.level}", True, self.COLOR_UI_TEXT)
        text_rect = level_text.get_rect(midbottom=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 10))
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, text_rect.inflate(10, 10))
        self.screen.blit(level_text, text_rect)

    def _draw_health_bar(self, x, y, w, h, current, maximum):
        if maximum > 0:
            ratio = max(0, current / maximum)
            pygame.draw.rect(self.screen, (80, 0, 0), (x, y, w, h))
            pygame.draw.rect(self.screen, (0, 150, 0), (x, y, int(w * ratio), h))

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # --- Particle Effects ---
    def _create_hit_particle(self, pos, color=(255, 255, 255)):
        for _ in range(10):
            self.particles.append({
                'pos': pos,
                'offset': (self.rng.uniform(-8, 8), self.rng.uniform(-8, 8)),
                'color': color,
                'size': self.rng.uniform(2, 5),
                'lifetime': 10
            })

    def _create_gold_particle(self, pos):
        for _ in range(15):
            self.particles.append({
                'pos': pos,
                'offset': (self.rng.uniform(-10, 10), self.rng.uniform(-10, 10)),
                'color': self.COLOR_GOLD,
                'size': self.rng.uniform(1, 4),
                'lifetime': 15
            })

    def _create_shield_particle(self, pos):
        for i in range(12):
            angle = (i / 12) * 2 * math.pi
            self.particles.append({
                'pos': pos,
                'offset': (math.cos(angle) * 10, math.sin(angle) * 10),
                'color': (100, 150, 255),
                'size': 4,
                'lifetime': 8
            })

    def _update_particles(self):
        for p in self.particles:
            p['lifetime'] -= 1
            p['size'] *= 0.9
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    # Note: Requires a non-dummy SDL_VIDEODRIVER
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Dungeon Crawler")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # In a human-playable mode, we only step when an action is taken.
        # A no-op is a valid action that advances the turn.
        # We need a key to trigger a no-op, let's use 'w' for wait.
        if any(keys):
            if keys[pygame.K_w]: # Explicit wait action
                action = [0, 0, 0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")
            
            if terminated:
                print("Game Over!")
                obs, info = env.reset()
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Limit frame rate for human playability

    env.close()