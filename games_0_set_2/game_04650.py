
# Generated: 2025-08-28T03:02:44.303483
# Source Brief: brief_04650.md
# Brief Index: 4650

        
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

    user_guide = (
        "Controls: Arrow keys to move, Space to attack in the last moved direction. Collect gold and reach the blue exit."
    )

    game_description = (
        "Grid-based dungeon crawler. Navigate a maze, fight enemies, and collect gold to reach the exit. Your health is limited, and enemies will respawn."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 31, 19 # Odd numbers for maze generation
        self.TILE_SIZE = min(self.SCREEN_WIDTH // (self.GRID_WIDTH + 1), self.SCREEN_HEIGHT // (self.GRID_HEIGHT + 1))
        self.WORLD_WIDTH = self.GRID_WIDTH * self.TILE_SIZE
        self.WORLD_HEIGHT = self.GRID_HEIGHT * self.TILE_SIZE
        self.X_OFFSET = (self.SCREEN_WIDTH - self.WORLD_WIDTH) // 2
        self.Y_OFFSET = (self.SCREEN_HEIGHT - self.WORLD_HEIGHT) // 2
        
        self.MAX_STEPS = 5000
        self.MAX_LEVELS = 5
        
        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (60, 60, 80)
        self.COLOR_FLOOR = (40, 40, 55)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_OUTLINE = (20, 150, 20)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_OUTLINE = (150, 20, 20)
        self.COLOR_GOLD = (255, 223, 0)
        self.COLOR_EXIT = (50, 150, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_HEALTH_BAR = (0, 200, 0)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_ATTACK = (200, 200, 255)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 48)
        
        # State variables (initialized in reset)
        self.grid = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 1
        
        self.player_pos = None
        self.player_health = 100
        self.player_max_health = 100
        self.player_last_move_dir = (0, 1) # Default: down
        self.player_hit_flash = 0
        
        self.enemies = []
        self.gold_piles = set()
        self.exit_pos = None
        
        self.visual_effects = []
        
        self.reset()
        self.validate_implementation()

    def _generate_level(self):
        self.grid = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.uint8) # 1 = wall
        
        # Randomized DFS for maze generation
        start_x, start_y = (self.np_random.integers(0, self.GRID_WIDTH // 2) * 2 + 1,
                            self.np_random.integers(0, self.GRID_HEIGHT // 2) * 2 + 1)
        stack = [(start_x, start_y)]
        self.grid[start_y, start_x] = 0 # 0 = floor
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                self.grid[ny, nx] = 0
                self.grid[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

        floor_tiles = list(zip(*np.where(self.grid == 0)))
        self.np_random.shuffle(floor_tiles)

        self.player_pos = floor_tiles.pop(0)
        
        # Place exit far from player
        max_dist = -1
        for tile in reversed(floor_tiles):
            dist = math.dist(self.player_pos, tile)
            if dist > max_dist:
                max_dist = dist
                self.exit_pos = tile
        floor_tiles.remove(self.exit_pos)

        # Place gold
        self.gold_piles = set()
        num_gold = self.np_random.integers(5, 11)
        for _ in range(num_gold):
            if not floor_tiles: break
            gold_pos = floor_tiles.pop(0)
            self.gold_piles.add(gold_pos)

        # Place enemies
        self.enemies = []
        num_enemies = self.current_level
        for _ in range(num_enemies):
            if not floor_tiles: break
            enemy_pos = floor_tiles.pop(0)
            self.enemies.append({
                'pos': enemy_pos,
                'health': 20,
                'max_health': 20,
                'death_timer': 0,
                'hit_flash': 0
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 1
        
        self.player_health = self.player_max_health
        self.player_last_move_dir = (0, 1)
        self.player_hit_flash = 0
        
        self.visual_effects = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        self.steps += 1
        
        # Decay visual effects
        self.visual_effects = [fx for fx in self.visual_effects if fx['timer'] > 1]
        for fx in self.visual_effects: fx['timer'] -= 1
        self.player_hit_flash = max(0, self.player_hit_flash - 1)
        for enemy in self.enemies: enemy['hit_flash'] = max(0, enemy['hit_flash'] - 1)

        # Enemy respawn logic
        for enemy in self.enemies:
            if enemy['health'] <= 0:
                enemy['death_timer'] -= 1
                if enemy['death_timer'] <= 0:
                    floor_tiles = list(zip(*np.where(self.grid == 0)))
                    occupied_tiles = {self.player_pos, self.exit_pos} | self.gold_piles
                    for e in self.enemies: occupied_tiles.add(e['pos'])
                    
                    valid_spawns = [t for t in floor_tiles if t not in occupied_tiles]
                    if valid_spawns:
                        enemy['pos'] = self.np_random.choice(valid_spawns, axis=0)
                        enemy['health'] = enemy['max_health']

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Action handling: Attack takes precedence over movement
        if space_held:
            # Attack action
            target_pos = (self.player_pos[0] + self.player_last_move_dir[0], 
                          self.player_pos[1] + self.player_last_move_dir[1])
            
            self._add_visual_effect('slash', target_pos, 1, {'dir': self.player_last_move_dir})
            # SFX: Player_Attack_Swing.wav

            for enemy in self.enemies:
                if enemy['pos'] == target_pos and enemy['health'] > 0:
                    damage = 10
                    enemy['health'] -= damage
                    enemy['hit_flash'] = 2 # Flash for 2 turns
                    self._add_visual_effect('text', enemy['pos'], 2, {'text': f"-{damage}", 'color': self.COLOR_WHITE})
                    # SFX: Enemy_Hit.wav

                    if enemy['health'] <= 0:
                        reward += 2
                        enemy['death_timer'] = 10 # Respawn in 10 turns
                        self._add_visual_effect('explosion', enemy['pos'], 3)
                        # SFX: Enemy_Defeat.wav
                        break
        
        elif movement > 0:
            # Movement action
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = move_map.get(movement, (0, 0))
            
            if dx != 0 or dy != 0:
                self.player_last_move_dir = (dx, dy)
                next_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
                
                # Check for wall collision
                if self.grid[next_pos[1], next_pos[0]] == 0:
                    dist_before = math.dist(self.player_pos, self.exit_pos)
                    self.player_pos = next_pos
                    dist_after = math.dist(self.player_pos, self.exit_pos)
                    
                    # Reward for moving closer to exit
                    if dist_after < dist_before: reward += 1.0
                    else: reward -= 0.2
                    
                    # Check for interactions on new tile
                    # Exit
                    if self.player_pos == self.exit_pos:
                        self.current_level += 1
                        if self.current_level > self.MAX_LEVELS:
                            reward += 500
                            terminated = True
                            self.game_over = True
                        else:
                            reward += 100
                            self._generate_level()
                            # SFX: Level_Up.wav
                    # Gold
                    elif self.player_pos in self.gold_piles:
                        self.gold_piles.remove(self.player_pos)
                        reward += 10
                        self.score += 10
                        self._add_visual_effect('text', self.player_pos, 2, {'text': "+10", 'color': self.COLOR_GOLD})
                        # SFX: Gold_Collect.wav
                    # Enemy
                    else:
                        for enemy in self.enemies:
                            if enemy['pos'] == self.player_pos and enemy['health'] > 0:
                                self.player_health -= 10
                                self.player_hit_flash = 2
                                # SFX: Player_Hurt.wav
                                break

        # Check termination conditions
        if self.player_health <= 0:
            self.player_health = 0
            reward -= 100
            terminated = True
            self.game_over = True
            # SFX: Game_Over.wav
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.current_level, "health": self.player_health}

    def _grid_to_screen(self, x, y):
        return (self.X_OFFSET + x * self.TILE_SIZE, self.Y_OFFSET + y * self.TILE_SIZE)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_x, screen_y = self._grid_to_screen(x, y)
                rect = pygame.Rect(screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.grid[y, x] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

        # Draw exit
        ex, ey = self._grid_to_screen(*self.exit_pos)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex, ey, self.TILE_SIZE, self.TILE_SIZE))
        pygame.gfxdraw.filled_circle(self.screen, ex + self.TILE_SIZE//2, ey + self.TILE_SIZE//2, self.TILE_SIZE//3, (20,100,200))

        # Draw gold
        for gx, gy in self.gold_piles:
            sx, sy = self._grid_to_screen(gx, gy)
            pygame.gfxdraw.filled_circle(self.screen, sx + self.TILE_SIZE//2, sy + self.TILE_SIZE//2, self.TILE_SIZE//3, self.COLOR_GOLD)
            pygame.gfxdraw.aacircle(self.screen, sx + self.TILE_SIZE//2, sy + self.TILE_SIZE//2, self.TILE_SIZE//3, (255,255,100))

        # Draw enemies
        for enemy in self.enemies:
            if enemy['health'] > 0:
                ex, ey = self._grid_to_screen(*enemy['pos'])
                color = self.COLOR_WHITE if enemy['hit_flash'] > 0 else self.COLOR_ENEMY
                outline_color = self.COLOR_WHITE if enemy['hit_flash'] > 0 else self.COLOR_ENEMY_OUTLINE
                pygame.draw.rect(self.screen, outline_color, (ex + 2, ey + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4))
                pygame.draw.rect(self.screen, color, (ex + 4, ey + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8))
        
        # Draw player
        px, py = self._grid_to_screen(*self.player_pos)
        player_color = self.COLOR_ENEMY if self.player_hit_flash > 0 else self.COLOR_PLAYER
        outline_color = self.COLOR_ENEMY if self.player_hit_flash > 0 else self.COLOR_PLAYER_OUTLINE
        pygame.draw.rect(self.screen, outline_color, (px + 1, py + 1, self.TILE_SIZE - 2, self.TILE_SIZE - 2))
        pygame.draw.rect(self.screen, player_color, (px + 3, py + 3, self.TILE_SIZE - 6, self.TILE_SIZE - 6))

        # Draw visual effects
        for fx in self.visual_effects:
            self._render_effect(fx)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.player_max_health
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, max(0, bar_width * health_ratio), bar_height))
        health_text = self.font_small.render(f"HP: {self.player_health}/{self.player_max_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_GOLD)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        # Level
        level_text = self.font_medium.render(f"Level: {self.current_level}/{self.MAX_LEVELS}", True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(midbottom=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 10))
        self.screen.blit(level_text, level_rect)

    def _add_visual_effect(self, type, pos, timer, data=None):
        self.visual_effects.append({'type': type, 'pos': pos, 'timer': timer, 'data': data or {}})

    def _render_effect(self, fx):
        sx, sy = self._grid_to_screen(*fx['pos'])
        center_x, center_y = sx + self.TILE_SIZE // 2, sy + self.TILE_SIZE // 2
        
        if fx['type'] == 'text':
            text_surf = self.font_small.render(fx['data']['text'], True, fx['data']['color'])
            alpha = int(255 * (fx['timer'] / 2.0))
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(center_x, center_y - (2 - fx['timer']) * 10))
            self.screen.blit(text_surf, text_rect)

        elif fx['type'] == 'slash':
            dx, dy = fx['data']['dir']
            p1 = (center_x - dx * self.TILE_SIZE // 4 - dy * self.TILE_SIZE // 4, 
                  center_y - dy * self.TILE_SIZE // 4 - dx * self.TILE_SIZE // 4)
            p2 = (center_x + dx * self.TILE_SIZE // 2, center_y + dy * self.TILE_SIZE // 2)
            p3 = (center_x - dx * self.TILE_SIZE // 4 + dy * self.TILE_SIZE // 4, 
                  center_y - dy * self.TILE_SIZE // 4 + dx * self.TILE_SIZE // 4)
            pygame.draw.aaline(self.screen, self.COLOR_ATTACK, p1, p3, 2)
            
        elif fx['type'] == 'explosion':
            radius = int(self.TILE_SIZE // 2 * (1 - (fx['timer'] / 3.0)))
            color = (255, 150 + (3-fx['timer'])*30, (3-fx['timer'])*30)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)


    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # To display the game, we need a Pygame screen
    pygame.display.set_caption("Dungeon Crawler")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    # Game loop
    running = True
    while running:
        # Get user input for manual play
        action = [0, 0, 0] # Default: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # This is a simplified mapping for manual play.
                # An agent would provide an action array directly.
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                
                # If any key is pressed, step the environment
                if action != [0, 0, 0]:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
                    
                    if terminated or truncated:
                        print("Game Over! Resetting.")
                        obs, info = env.reset()

        # Render the observation to the display screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we don't need a clock tick in the main loop,
        # the game only advances on key presses.
        
    pygame.quit()