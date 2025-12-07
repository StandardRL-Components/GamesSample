
# Generated: 2025-08-27T20:24:49.709071
# Source Brief: brief_02450.md
# Brief Index: 2450

        
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
        "Controls: Arrow keys to move. Space to attack in your last moved direction. Wait a turn by not moving."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based roguelike. Explore the dungeon, collect gold, defeat enemies, and find the exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    TILE_SIZE = 20
    SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE
    SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE

    COLOR_BG = (20, 20, 30)
    COLOR_FLOOR = (50, 50, 70)
    COLOR_WALL = (100, 100, 120)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_GOLD = (255, 220, 0)
    COLOR_EXIT = (255, 255, 100)
    COLOR_HEALTH_FG = (0, 200, 0)
    COLOR_HEALTH_BG = (100, 0, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_ATTACK_FLASH = (255, 255, 255)

    MAX_STEPS = 1000
    INITIAL_PLAYER_HEALTH = 5
    INITIAL_ENEMY_COUNT = 1
    MAX_ENEMIES = 10
    ENEMY_SPAWN_INTERVAL = 500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.big_font = pygame.font.SysFont("monospace", 40, bold=True)

        # RNG
        self.np_random = None
        
        # Initialize state variables
        self.grid = None
        self.player_pos = None
        self.player_health = 0
        self.player_gold = 0
        self.player_facing_dir = (0, 1) # Default down
        self.enemies = []
        self.gold_piles = []
        self.exit_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.attack_effects = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.INITIAL_PLAYER_HEALTH
        self.player_gold = 0
        self.player_facing_dir = (0, 1) # Down
        self.attack_effects = []

        self._generate_dungeon()
        self._place_entities()
        
        return self._get_observation(), self._get_info()

    def _generate_dungeon(self):
        self.grid = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.uint8) # 1 = wall
        stack = deque()
        
        # Start DFS from a random point
        start_x, start_y = (self.np_random.integers(1, self.GRID_WIDTH//2) * 2, 
                            self.np_random.integers(1, self.GRID_HEIGHT//2) * 2)
        self.grid[start_y, start_x] = 0 # 0 = floor
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.GRID_WIDTH-1 and 0 < ny < self.GRID_HEIGHT-1 and self.grid[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                self.grid[ny, nx] = 0
                self.grid[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
    
    def _place_entities(self):
        floor_tiles = np.argwhere(self.grid == 0).tolist()
        self.np_random.shuffle(floor_tiles)

        # Player and Exit
        self.player_pos = floor_tiles.pop()
        
        # Find a distant exit
        min_dist = (self.GRID_WIDTH + self.GRID_HEIGHT) // 3
        exit_candidate = None
        for tile in reversed(floor_tiles):
            dist = abs(tile[1] - self.player_pos[1]) + abs(tile[0] - self.player_pos[0])
            if dist > min_dist:
                exit_candidate = tile
                floor_tiles.remove(tile)
                break
        self.exit_pos = exit_candidate if exit_candidate else floor_tiles.pop()
        
        # Gold Piles (5% of floor tiles)
        num_gold = max(1, int(len(floor_tiles) * 0.05))
        self.gold_piles = [floor_tiles.pop() for _ in range(num_gold) if floor_tiles]

        # Enemies
        self.enemies = []
        num_enemies = self.INITIAL_ENEMY_COUNT
        for _ in range(num_enemies):
            if not floor_tiles: break
            pos = floor_tiles.pop()
            self.enemies.append({
                "pos": pos,
                "health": 1,
                "patrol_dir": self.np_random.choice([(0,1), (0,-1), (1,0), (-1,0)], axis=0),
                "patrol_steps": 0,
                "patrol_len": self.np_random.integers(2, 5)
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = 0
        self.attack_effects.clear() # Clear last turn's effects

        # --- Player Turn ---
        prev_dist_to_exit = abs(self.player_pos[1] - self.exit_pos[1]) + abs(self.player_pos[0] - self.exit_pos[0])

        if space_held: # Player attacks
            # sfx: player_attack_whoosh.wav
            attack_pos = (self.player_pos[1] + self.player_facing_dir[1], self.player_pos[0] + self.player_facing_dir[0])
            self.attack_effects.append(attack_pos)
            
            enemy_hit = None
            for enemy in self.enemies:
                if tuple(enemy["pos"]) == attack_pos:
                    enemy_hit = enemy
                    break
            
            if enemy_hit:
                # sfx: enemy_hit.wav
                enemy_hit["health"] -= 1
                if enemy_hit["health"] <= 0:
                    self.enemies.remove(enemy_hit)
                    reward += 1.0
        
        else: # Player moves or waits
            move_dir = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement)
            if move_dir:
                self.player_facing_dir = move_dir
                new_pos = [self.player_pos[0] + move_dir[1], self.player_pos[1] + move_dir[0]]
                
                if self.grid[new_pos[0], new_pos[1]] == 0: # Is a floor tile
                    self.player_pos = new_pos
                    # sfx: player_step.wav
        
        # Check for gold collection
        if self.player_pos in self.gold_piles:
            self.gold_piles.remove(self.player_pos)
            self.player_gold += 1
            reward += 1.0
            # sfx: gold_pickup.wav

        # Distance to exit reward
        new_dist_to_exit = abs(self.player_pos[1] - self.exit_pos[1]) + abs(self.player_pos[0] - self.exit_pos[0])
        if new_dist_to_exit < prev_dist_to_exit:
            reward += 0.1
        elif new_dist_to_exit > prev_dist_to_exit:
            reward -= 0.1

        # --- Enemy Turn ---
        for enemy in self.enemies:
            if abs(enemy["pos"][1] - self.player_pos[1]) + abs(enemy["pos"][0] - self.player_pos[0]) == 1:
                # Attack player if adjacent
                self.player_health -= 1
                # sfx: player_hurt.wav
            else:
                # Patrol
                new_enemy_pos = [enemy["pos"][0] + enemy["patrol_dir"][1], enemy["pos"][1] + enemy["patrol_dir"][0]]
                if self.grid[new_enemy_pos[0], new_enemy_pos[1]] == 0:
                    enemy["pos"] = new_enemy_pos
                    enemy["patrol_steps"] += 1
                else: # Hit a wall, reverse
                    enemy["patrol_dir"] = (-enemy["patrol_dir"][0], -enemy["patrol_dir"][1])
                    enemy["patrol_steps"] = 0

                if enemy["patrol_steps"] >= enemy["patrol_len"]:
                    enemy["patrol_dir"] = (-enemy["patrol_dir"][0], -enemy["patrol_dir"][1])
                    enemy["patrol_steps"] = 0
        
        # --- Update State ---
        self.steps += 1
        
        # Spawn more enemies over time
        if self.steps > 0 and self.steps % self.ENEMY_SPAWN_INTERVAL == 0 and len(self.enemies) < self.MAX_ENEMIES:
            floor_tiles = np.argwhere(self.grid == 0).tolist()
            valid_spawns = [t for t in floor_tiles if abs(t[1] - self.player_pos[1]) + abs(t[0] - self.player_pos[0]) > 5]
            if valid_spawns:
                pos = self.np_random.choice(valid_spawns, axis=0).tolist()
                self.enemies.append({
                    "pos": pos, "health": 1, 
                    "patrol_dir": self.np_random.choice([(0,1), (0,-1), (1,0), (-1,0)], axis=0),
                    "patrol_steps": 0, "patrol_len": self.np_random.integers(2, 5)
                })

        # --- Check Termination ---
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 100.0
            terminated = True
            # sfx: victory_fanfare.wav
        elif self.player_health <= 0:
            reward -= 100.0
            terminated = True
            # sfx: player_death.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        ts = self.TILE_SIZE
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color = self.COLOR_WALL if self.grid[y, x] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, (x * ts, y * ts, ts, ts))
        
        # Draw exit
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (self.exit_pos[1] * ts, self.exit_pos[0] * ts, ts, ts))
        pygame.gfxdraw.rectangle(self.screen, (self.exit_pos[1] * ts, self.exit_pos[0] * ts, ts, ts), (255,255,255,100))

        # Draw gold
        for y, x in self.gold_piles:
            pygame.draw.rect(self.screen, self.COLOR_GOLD, (x * ts + ts//4, y * ts + ts//4, ts//2, ts//2))
        
        # Draw enemies
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (enemy["pos"][1] * ts, enemy["pos"][0] * ts, ts, ts))

        # Draw player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (self.player_pos[1] * ts, self.player_pos[0] * ts, ts, ts))
        
        # Draw attack effects
        for y, x in self.attack_effects:
             pygame.gfxdraw.box(self.screen, (x * ts, y * ts, ts, ts), (*self.COLOR_ATTACK_FLASH, 150))

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.INITIAL_PLAYER_HEALTH)
        health_bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, int(health_bar_width * health_ratio), 20))
        health_text = self.font.render(f"HP: {self.player_health}/{self.INITIAL_PLAYER_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Gold Counter
        gold_text = self.font.render(f"Gold: {self.player_gold}", True, self.COLOR_GOLD)
        text_rect = gold_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(gold_text, text_rect)
        
        # Step Counter
        step_text = self.font.render(f"Turn: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        step_rect = step_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 35))
        self.screen.blit(step_text, step_rect)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status = "VICTORY!" if self.player_health > 0 and self.player_pos == self.exit_pos else "GAME OVER"
            status_text = self.big_font.render(status, True, self.COLOR_EXIT if status == "VICTORY!" else self.COLOR_ENEMY)
            status_rect = status_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(status_text, status_rect)

            score_text = self.font.render(f"Final Score: {self.score:.1f}", True, self.COLOR_TEXT)
            score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30))
            self.screen.blit(score_text, score_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.player_gold,
            "health": self.player_health,
        }

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
        
        # Custom Assertions from brief
        assert self.player_health <= self.INITIAL_PLAYER_HEALTH
        assert 0 <= self.player_pos[1] < self.GRID_WIDTH and 0 <= self.player_pos[0] < self.GRID_HEIGHT
        for enemy in self.enemies:
            assert enemy["health"] <= 1
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Roguelike Dungeon")
    
    running = True
    game_over = False
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not game_over:
                # Movement
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                # Attack
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                # Reset
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    game_over = False
                    continue # Skip step for this frame

                # If any key was pressed that constitutes an action, step the environment
                if action != [0, 0, 0]:
                    obs, reward, terminated, truncated, info = env.step(np.array(action))
                    game_over = terminated

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()