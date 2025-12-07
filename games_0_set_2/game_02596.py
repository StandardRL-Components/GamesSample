
# Generated: 2025-08-27T20:50:51.612901
# Source Brief: brief_02596.md
# Brief Index: 2596

        
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
        "Controls: Use arrow keys to move one space at a time. "
        "Collect all 3 keys to unlock the exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated dungeon, collect keys to unlock the exit, "
        "and evade lurking enemies before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 32, 20
    CELL_SIZE = 20
    SCREEN_WIDTH, SCREEN_HEIGHT = GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE

    MAX_STEPS = 1000
    NUM_KEYS = 3
    NUM_ENEMIES = 2
    ENEMY_PATH_LENGTH = 8

    # --- Colors ---
    COLOR_BG = (10, 15, 20)
    COLOR_WALL = (30, 35, 40)
    COLOR_FLOOR = (50, 55, 60)
    COLOR_GRID = (40, 45, 50)
    
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255, 50)

    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_FLICKER = (200, 20, 20)
    COLOR_ENEMY_GLOW = (255, 0, 0, 70)

    COLOR_KEY = (255, 220, 0)
    COLOR_KEY_GLOW = (255, 220, 0, 80)

    COLOR_EXIT_LOCKED = (10, 100, 10)
    COLOR_EXIT_UNLOCKED = (10, 255, 10)
    COLOR_EXIT_GLOW_LOCKED = (10, 100, 10, 60)
    COLOR_EXIT_GLOW_UNLOCKED = (10, 255, 10, 90)

    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_KEY = (255, 220, 0)

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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables are initialized in reset()
        self.grid = None
        self.player_pos = None
        self.enemy_positions = None
        self.enemy_paths = None
        self.enemy_path_indices = None
        self.key_positions = None
        self.exit_pos = None
        self.keys_collected = None
        self.steps = None
        self.score = None
        self.game_over_message = None

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.keys_collected = 0
        self.time_left = self.MAX_STEPS
        self.game_over_message = None

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = 0
        terminated = False

        # --- Update game logic ---
        self.steps += 1
        self.time_left -= 1
        
        # 1. Calculate player movement and rewards
        prev_player_pos = self.player_pos
        self._move_player(movement)
        
        # Calculate distance-based rewards
        reward += self._calculate_proximity_rewards(prev_player_pos, self.player_pos)

        # 2. Move enemies
        self._move_enemies()

        # 3. Check for key collection
        if self.player_pos in self.key_positions:
            self.key_positions.remove(self.player_pos)
            self.keys_collected += 1
            self.score += 50
            reward += 5.0
            # sfx: key pickup sound

        # 4. Check for termination conditions
        # Collision with enemy
        if self.player_pos in self.enemy_positions:
            terminated = True
            reward = -10.0
            self.score -= 100
            self.game_over_message = "CAUGHT!"
            # sfx: player death sound

        # Reached exit
        elif self.player_pos == self.exit_pos and self.keys_collected == self.NUM_KEYS:
            terminated = True
            reward = 100.0
            self.score += 1000
            self.game_over_message = "ESCAPED!"
            # sfx: level complete fanfare

        # Timer runs out
        elif self.time_left <= 0:
            terminated = True
            reward = -50.0
            self.score -= 500
            self.game_over_message = "TIME'S UP!"
            # sfx: failure sound

        # Max steps reached (fallback)
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        # Add small survival reward
        if not terminated:
            reward += 0.01
            self.score += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        """Procedurally generates a valid, solvable level."""
        while True:
            self.grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)  # 1 = wall
            
            # Simple random walk to carve out floor
            px, py = self.np_random.integers(1, self.GRID_WIDTH-1), self.np_random.integers(1, self.GRID_HEIGHT-1)
            self.grid[px, py] = 0
            num_floors = 1
            for _ in range(self.GRID_WIDTH * self.GRID_HEIGHT * 2):
                nx, ny = px, py
                d = self.np_random.integers(0, 4)
                if d == 0: nx += 1
                elif d == 1: nx -= 1
                elif d == 2: ny += 1
                else: ny -= 1

                if 0 < nx < self.GRID_WIDTH - 1 and 0 < ny < self.GRID_HEIGHT - 1:
                    if self.grid[nx, ny] == 1:
                        self.grid[nx, ny] = 0
                        num_floors += 1
                    px, py = nx, ny

            # Find all floor tiles
            floor_tiles = list(zip(*np.where(self.grid == 0)))
            if not floor_tiles: continue

            # Check for reachability
            start_node = random.choice(floor_tiles)
            q = deque([start_node])
            reachable = {start_node}
            while q:
                x, y = q.popleft()
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if self.grid[nx, ny] == 0 and (nx, ny) not in reachable:
                        reachable.add((nx, ny))
                        q.append((nx, ny))
            
            # Ensure enough space and all items can be placed
            if len(reachable) > len(floor_tiles) * 0.7 and len(reachable) >= self.NUM_KEYS + self.NUM_ENEMIES + 2:
                valid_spawns = list(reachable)
                self.np_random.shuffle(valid_spawns)
                
                self.player_pos = valid_spawns.pop()
                self.exit_pos = valid_spawns.pop()
                self.key_positions = [valid_spawns.pop() for _ in range(self.NUM_KEYS)]
                
                enemy_starts = [valid_spawns.pop() for _ in range(self.NUM_ENEMIES)]
                self.enemy_positions = list(enemy_starts)
                self.enemy_paths = [self._find_enemy_path(start) for start in enemy_starts]
                self.enemy_path_indices = [0] * self.NUM_ENEMIES
                break

    def _find_enemy_path(self, start_pos):
        """Finds a short, looping path for an enemy."""
        path = [start_pos]
        for _ in range(self.ENEMY_PATH_LENGTH - 1):
            current_pos = path[-1]
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current_pos[0] + dx, current_pos[1] + dy
                if self.grid[nx, ny] == 0 and (nx, ny) not in path:
                    neighbors.append((nx, ny))
            if not neighbors: # If stuck, just stay put
                path.append(current_pos)
            else:
                path.append(random.choice(neighbors))
        return path + list(reversed(path[:-1])) # Create a loop back and forth

    def _move_player(self, movement):
        px, py = self.player_pos
        if movement == 1:  # Up
            py -= 1
        elif movement == 2:  # Down
            py += 1
        elif movement == 3:  # Left
            px -= 1
        elif movement == 4:  # Right
            px += 1
        
        if self.grid[px, py] == 0:  # 0 is floor
            self.player_pos = (px, py)
        else:
            pass # sfx: bump into wall

    def _move_enemies(self):
        for i in range(self.NUM_ENEMIES):
            path = self.enemy_paths[i]
            if not path: continue
            self.enemy_path_indices[i] = (self.enemy_path_indices[i] + 1) % len(path)
            self.enemy_positions[i] = path[self.enemy_path_indices[i]]

    def _calculate_proximity_rewards(self, old_pos, new_pos):
        if not self.key_positions:
            return 0

        def dist(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        nearest_key = min(self.key_positions, key=lambda k: dist(k, old_pos))
        
        old_dist = dist(old_pos, nearest_key)
        new_dist = dist(new_pos, nearest_key)
        
        if new_dist < old_dist:
            return 0.1  # Moved closer to a key
        elif new_dist > old_dist:
            return -0.2 # Moved away from a key
        return 0

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
            "time_left": self.time_left,
            "keys_collected": self.keys_collected,
            "player_pos": self.player_pos,
        }

    def _grid_to_pixels(self, pos):
        x, y = pos
        return x * self.CELL_SIZE, y * self.CELL_SIZE
    
    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                color = self.COLOR_WALL if self.grid[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        center_offset = self.CELL_SIZE // 2

        # Draw exit
        ex, ey = self._grid_to_pixels(self.exit_pos)
        is_unlocked = self.keys_collected == self.NUM_KEYS
        exit_color = self.COLOR_EXIT_UNLOCKED if is_unlocked else self.COLOR_EXIT_LOCKED
        glow_color = self.COLOR_EXIT_GLOW_UNLOCKED if is_unlocked else self.COLOR_EXIT_GLOW_LOCKED
        pygame.gfxdraw.filled_circle(self.screen, ex + center_offset, ey + center_offset, int(self.CELL_SIZE * 0.6), glow_color)
        pygame.draw.rect(self.screen, exit_color, (ex + 4, ey + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8))
        
        # Draw keys
        pulse = abs(math.sin(self.steps * 0.2))
        key_size = int((self.CELL_SIZE - 8) * (0.8 + pulse * 0.2))
        glow_radius = int(self.CELL_SIZE * 0.5 * (1.0 + pulse * 0.2))
        for kx, ky in self.key_positions:
            px, py = self._grid_to_pixels((kx, ky))
            pygame.gfxdraw.filled_circle(self.screen, px + center_offset, py + center_offset, glow_radius, self.COLOR_KEY_GLOW)
            pygame.draw.rect(self.screen, self.COLOR_KEY, (px + center_offset - key_size//2, py + center_offset - key_size//2, key_size, key_size), border_radius=2)
            
        # Draw enemies
        flicker = self.steps % 6 < 4
        enemy_color = self.COLOR_ENEMY if flicker else self.COLOR_ENEMY_FLICKER
        for ex, ey in self.enemy_positions:
            px, py = self._grid_to_pixels((ex, ey))
            pygame.gfxdraw.filled_circle(self.screen, px + center_offset, py + center_offset, int(self.CELL_SIZE * 0.7), self.COLOR_ENEMY_GLOW)
            pygame.draw.rect(self.screen, enemy_color, (px + 5, py + 5, self.CELL_SIZE - 10, self.CELL_SIZE - 10))
            
        # Draw player
        px, py = self._grid_to_pixels(self.player_pos)
        pygame.gfxdraw.filled_circle(self.screen, px + center_offset, py + center_offset, int(self.CELL_SIZE * 0.6), self.COLOR_PLAYER_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px + 4, py + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8))

    def _render_ui(self):
        # Render key count
        key_text = self.font_small.render(f"KEYS: {self.keys_collected}/{self.NUM_KEYS}", True, self.COLOR_TEXT_KEY)
        self.screen.blit(key_text, (10, 5))

        # Render timer
        time_text = self.font_small.render(f"TIME: {self.time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 5))

        # Render game over message
        if self.game_over_message:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space after reset
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Dungeon Crawler")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        movement = 0 # No-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        # The environment expects an action on every frame because auto_advance is False
        action = [movement, 0, 0] # Space and Shift are not used
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause to see the result

        clock.tick(10) # Control the speed of the game for human playability
        
    env.close()