import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space to attack adjacent enemies. Reach the treasure with 10 gold to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down roguelike. Navigate a dungeon, fight enemies, and collect gold to claim the treasure."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_SIZE = 8
    MAX_STEPS = 200
    START_HEALTH = 3
    WIN_GOLD = 10
    NUM_ENEMIES_MIN = 3
    NUM_ENEMIES_MAX = 5
    NUM_GOLD_MIN = 12
    NUM_GOLD_MAX = 18

    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 48
    GRID_WIDTH = GRID_SIZE * TILE_SIZE
    GRID_HEIGHT = GRID_SIZE * TILE_SIZE
    OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    COLOR_BG = (20, 20, 30)
    COLOR_WALL = (70, 70, 80)
    COLOR_FLOOR = (100, 80, 60)
    COLOR_PLAYER = (255, 70, 70)
    COLOR_PLAYER_GLOW = (255, 150, 150)
    COLOR_ENEMY_1 = (70, 70, 255)
    COLOR_ENEMY_2 = (120, 120, 255)
    COLOR_GOLD = (255, 215, 0)
    COLOR_TREASURE = (255, 215, 0)
    COLOR_TREASURE_LOCK = (139, 69, 19)
    COLOR_HEALTH_BG = (150, 0, 0)
    COLOR_HEALTH_FG = (0, 200, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_ATTACK_FX = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
            self.font_small = pygame.font.SysFont("monospace", 16)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 22)

        self.game_over = False
        self.steps = 0
        self.score = 0
        self.player_pos = (0, 0)
        self.player_health = 0
        self.player_gold = 0
        self.treasure_pos = (0, 0)
        self.grid = []
        self.enemies = []
        self.gold_coins = []
        self.attack_fx = []
        # self.np_random is initialized in super().reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.START_HEALTH
        self.player_gold = 0
        self.attack_fx = []

        self._generate_dungeon()
        
        self.last_dist_to_treasure = self._manhattan_distance(self.player_pos, self.treasure_pos)

        return self._get_observation(), self._get_info()
    
    def _generate_dungeon(self):
        # 1. Initialize grid with walls
        self.grid = [['wall' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        # 2. Create a path using random walk
        start_pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
        self.player_pos = start_pos
        
        q = deque([start_pos])
        visited = {start_pos}
        
        while q:
            x, y = q.popleft()
            self.grid[y][x] = 'floor'
            
            neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
            self.np_random.shuffle(neighbors)
            
            for nx, ny in neighbors:
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in visited:
                    # Count visited neighbors to avoid creating large open areas
                    visited_neighbors = 0
                    for nnx, nny in [(nx, ny - 1), (nx, ny + 1), (nx - 1, ny), (nx + 1, y)]:
                        if (nnx, nny) in visited:
                            visited_neighbors += 1
                    
                    if visited_neighbors <= 2:
                        visited.add((nx, ny))
                        q.append((nx, ny))

        floor_tiles = list(visited)
        
        # 3. Place treasure far from player
        min_dist = self.GRID_SIZE // 2
        possible_treasure_locs = [p for p in floor_tiles if self._manhattan_distance(self.player_pos, p) >= min_dist]
        if not possible_treasure_locs:
            possible_treasure_locs = [p for p in floor_tiles if p != self.player_pos]
        if not possible_treasure_locs: # Player is the only floor tile
            possible_treasure_locs = floor_tiles

        idx = self.np_random.integers(len(possible_treasure_locs))
        self.treasure_pos = possible_treasure_locs[idx]

        # 4. Place gold and enemies
        available_tiles = [p for p in floor_tiles if p != self.player_pos and p != self.treasure_pos]
        self.np_random.shuffle(available_tiles)
        
        num_gold = self.np_random.integers(self.NUM_GOLD_MIN, self.NUM_GOLD_MAX + 1)
        self.gold_coins = []
        for _ in range(num_gold):
            if available_tiles:
                self.gold_coins.append(available_tiles.pop())
        
        num_enemies = self.np_random.integers(self.NUM_ENEMIES_MIN, self.NUM_ENEMIES_MAX + 1)
        self.enemies = []
        for _ in range(num_enemies):
            if available_tiles:
                pos = available_tiles.pop()
                self.enemies.append({'pos': pos, 'health': 1})

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0, True, False, info

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        self.attack_fx = [] # Clear attack effects each turn

        # --- Player Action Phase ---
        action_taken = False
        if space_held:
            # Player attacks
            action_taken = True
            px, py = self.player_pos
            attack_coords = [(px, py - 1), (px, py + 1), (px - 1, py), (px + 1, py)]
            
            enemies_hit = []
            for enemy in self.enemies:
                if enemy['pos'] in attack_coords:
                    enemies_hit.append(enemy)
                    self.attack_fx.append(enemy['pos'])
            
            if enemies_hit:
                # sound_effect: player_attack_hit.wav
                for enemy in enemies_hit:
                    self.enemies.remove(enemy)
                    reward += 10 # Defeated an enemy
                    self.score += 100
            else:
                # sound_effect: player_attack_miss.wav
                pass
        
        elif movement != 0:
            # Player moves
            action_taken = True
            px, py = self.player_pos
            nx, ny = px, py

            if movement == 1: ny -= 1 # Up
            elif movement == 2: ny += 1 # Down
            elif movement == 3: nx -= 1 # Left
            elif movement == 4: nx += 1 # Right

            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid[ny][nx] == 'floor':
                is_enemy_at_target = any(enemy['pos'] == (nx, ny) for enemy in self.enemies)
                if not is_enemy_at_target:
                    self.player_pos = (nx, ny)
                    # sound_effect: player_move.wav
            
        # --- Post-Player Action Logic ---
        if action_taken:
            # Gold collection
            if self.player_pos in self.gold_coins:
                self.gold_coins.remove(self.player_pos)
                self.player_gold += 1
                reward += 1
                self.score += 10
                # sound_effect: collect_gold.wav

            # Distance to treasure reward
            new_dist = self._manhattan_distance(self.player_pos, self.treasure_pos)
            if new_dist < self.last_dist_to_treasure:
                reward += 0.1
            elif new_dist > self.last_dist_to_treasure:
                reward -= 0.1
            self.last_dist_to_treasure = new_dist

            # --- Enemy Action Phase ---
            for enemy in self.enemies:
                ex, ey = enemy['pos']
                px, py = self.player_pos
                
                # Check if adjacent to player to attack
                if abs(ex - px) + abs(ey - py) == 1:
                    self.player_health -= 1
                    reward -= 1
                    self.score -= 20
                    # sound_effect: player_hurt.wav
                else: # Move randomly
                    neighbors = [(ex, ey - 1), (ex, ey + 1), (ex - 1, ey), (ex + 1, ey)]
                    valid_moves = []
                    for nx, ny in neighbors:
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid[ny][nx] == 'floor':
                            is_occupied = (nx, ny) == self.player_pos or any(e['pos'] == (nx, ny) for e in self.enemies if e is not enemy)
                            if not is_occupied:
                                valid_moves.append((nx, ny))
                    if valid_moves:
                        idx = self.np_random.integers(len(valid_moves))
                        enemy['pos'] = valid_moves[idx]

        # --- Termination Check ---
        self.steps += 1

        if self.player_health <= 0:
            terminated = True
            reward -= 100
            # sound_effect: game_over.wav
        elif self.player_pos == self.treasure_pos and self.player_gold >= self.WIN_GOLD:
            terminated = True
            reward += 100
            self.score += 1000
            # sound_effect: victory.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        
        # Ensure reward is float
        reward = float(reward)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(self.OFFSET_X + x * self.TILE_SIZE, self.OFFSET_Y + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if self.grid[y][x] == 'wall':
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)

        # Draw treasure
        tx, ty = self.treasure_pos
        treasure_rect = pygame.Rect(self.OFFSET_X + tx * self.TILE_SIZE, self.OFFSET_Y + ty * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_TREASURE, treasure_rect.inflate(-8, -8))
        lock_rect = pygame.Rect(0,0, self.TILE_SIZE // 3, self.TILE_SIZE // 3)
        lock_rect.center = treasure_rect.center
        pygame.draw.rect(self.screen, self.COLOR_TREASURE_LOCK, lock_rect)

        # Draw gold
        for gx, gy in self.gold_coins:
            cx = int(self.OFFSET_X + gx * self.TILE_SIZE + self.TILE_SIZE / 2)
            cy = int(self.OFFSET_Y + gy * self.TILE_SIZE + self.TILE_SIZE / 2)
            radius = int(self.TILE_SIZE / 4)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, self.COLOR_GOLD)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_GOLD)

        # Draw enemies
        enemy_color = self.COLOR_ENEMY_1 if self.steps % 10 < 5 else self.COLOR_ENEMY_2
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            rect = pygame.Rect(self.OFFSET_X + ex * self.TILE_SIZE, self.OFFSET_Y + ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, enemy_color, rect.inflate(-12, -12))
            
        # Draw attack FX
        for ax, ay in self.attack_fx:
            rect = pygame.Rect(self.OFFSET_X + ax * self.TILE_SIZE, self.OFFSET_Y + ay * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_ATTACK_FX, rect.inflate(-20, -20), 3)

        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(self.OFFSET_X + px * self.TILE_SIZE, self.OFFSET_Y + py * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        
        # Glow effect
        glow_rect = player_rect.inflate(8, 8)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PLAYER_GLOW, 100), glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-8, -8), border_radius=4)


    def _render_ui(self):
        # UI Background
        ui_bg_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.OFFSET_Y)
        pygame.draw.rect(self.screen, (0,0,0, 150), ui_bg_rect)
        
        # Gold display
        gold_text = self.font_large.render(f"GOLD: {self.player_gold}/{self.WIN_GOLD}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (20, 8))

        # Health display
        health_bar_width = 150
        health_bar_height = 20
        health_bar_x = self.SCREEN_WIDTH - health_bar_width - 20
        health_bar_y = 12

        health_pct = max(0, self.player_health / self.START_HEALTH)
        bg_rect = pygame.Rect(health_bar_x, health_bar_y, health_bar_width, health_bar_height)
        fg_rect = pygame.Rect(health_bar_x, health_bar_y, int(health_bar_width * health_pct), health_bar_height)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, fg_rect, border_radius=5)
        
        health_text = self.font_small.render(f"HP: {self.player_health}/{self.START_HEALTH}", True, self.COLOR_TEXT)
        text_rect = health_text.get_rect(center=bg_rect.center)
        self.screen.blit(health_text, text_rect)

        # Step/Turn display
        step_text = self.font_small.render(f"TURN: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(step_text, (self.SCREEN_WIDTH // 2 - step_text.get_width() // 2, 14))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "player_gold": self.player_gold,
            "player_pos": self.player_pos,
            "treasure_pos": self.treasure_pos,
            "dist_to_treasure": self._manhattan_distance(self.player_pos, self.treasure_pos)
        }

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # This is not called in the final version, but is useful for development
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    # It will not work in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv()
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Dungeon Crawler")
        clock = pygame.time.Clock()
        
        running = True
        game_over_screen = False
        
        while running:
            action = [0, 0, 0] # [movement, space, shift]
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and not game_over_screen:
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
                    
                    # If any key was pressed that corresponds to an action, step the environment
                    if any(a != 0 for a in action) or action[0] != 0:
                        obs, reward, terminated, _, info = env.step(np.array(action))
                        print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
                        if terminated:
                            game_over_screen = True
                            print("GAME OVER. Press 'R' to restart.")

                if event.type == pygame.KEYDOWN:
                     if event.key == pygame.K_r: # Reset game
                        obs, info = env.reset()
                        game_over_screen = False

            # Draw the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(30)
            
        env.close()
    except pygame.error as e:
        print(f"Could not run in graphical mode: {e}")
        print("Running validation check in headless mode...")
        env = GameEnv()
        env.validate_implementation()
        env.close()