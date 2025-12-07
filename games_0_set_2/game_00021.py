import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character (green square) one tile at a time. "
        "The goal is to reach the exit (blue square)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based roguelike. Navigate a dungeon, collect gold (yellow circles), and fight enemies (red squares) "
        "by moving into them. Reach level 5 to win. Your health is shown in the top-left."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 25
    GRID_HEIGHT = 17
    TILE_SIZE = 20
    GAME_AREA_WIDTH = GRID_WIDTH * TILE_SIZE
    GAME_AREA_HEIGHT = GRID_HEIGHT * TILE_SIZE
    GAME_AREA_X_OFFSET = (SCREEN_WIDTH - GAME_AREA_WIDTH) // 2
    GAME_AREA_Y_OFFSET = (SCREEN_HEIGHT - GAME_AREA_HEIGHT) // 2 + 10

    MAX_STEPS = 1000
    STARTING_HEALTH = 10
    WIN_LEVEL = 5

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (60, 60, 80)
    COLOR_FLOOR = (30, 30, 45)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_GOLD = (255, 223, 0)
    COLOR_EXIT = (50, 150, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR_BG = (120, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 200, 0)
    
    # Grid cell types
    CELL_FLOOR = 0
    CELL_WALL = 1

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.grid = None
        self.player_pos = None
        self.exit_pos = None
        self.enemies = None
        self.gold_items = None
        self.player_health = 0
        self.score = 0
        self.level = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.last_message = ""

        # This will be initialized in reset()
        self.np_random = None

        # self.validate_implementation() # This is called by the test runner, not needed here

    def _generate_level(self):
        """Generates a new level using randomized DFS for a perfect maze."""
        self.grid = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.uint8) * self.CELL_WALL
        
        # Use integer coordinates for grid operations
        # The upper bound is exclusive, and we need a wall border, so we use DIM - 1
        # to generate coordinates within [1, DIM - 2].
        start_x = self.np_random.integers(1, self.GRID_WIDTH - 1)
        start_y = self.np_random.integers(1, self.GRID_HEIGHT - 1)
        
        # Ensure starting position is on an odd-numbered tile for the maze algorithm
        start_x |= 1
        start_y |= 1
        
        stack = deque([(start_x, start_y)])
        self.grid[start_y, start_x] = self.CELL_FLOOR
        
        floor_tiles = []

        while stack:
            cx, cy = stack[-1]
            floor_tiles.append((cx, cy))
            
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.GRID_WIDTH - 1 and 0 < ny < self.GRID_HEIGHT - 1 and self.grid[ny, nx] == self.CELL_WALL:
                    neighbors.append((nx, ny))

            if neighbors:
                # np.random.Generator.choice doesn't work on a list of tuples as one might expect.
                # A robust way is to select an index.
                chosen_neighbor = neighbors[self.np_random.integers(len(neighbors))]
                nx, ny = chosen_neighbor
                # Carve path
                self.grid[ny, nx] = self.CELL_FLOOR
                self.grid[(cy + ny) // 2, (cx + nx) // 2] = self.CELL_FLOOR
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Place player, exit, enemies, and gold
        self.np_random.shuffle(floor_tiles)
        self.player_pos = np.array(floor_tiles.pop())

        # Find the furthest tile for the exit to ensure a challenge
        distances = [math.hypot(x - self.player_pos[0], y - self.player_pos[1]) for x, y in floor_tiles]
        self.exit_pos = np.array(floor_tiles.pop(np.argmax(distances)))

        # Place enemies
        num_enemies = self.level
        self.enemies = []
        for _ in range(num_enemies):
            if floor_tiles:
                self.enemies.append(np.array(floor_tiles.pop()))

        # Place gold
        num_gold = 5
        self.gold_items = []
        for _ in range(num_gold):
            if floor_tiles:
                self.gold_items.append(np.array(floor_tiles.pop()))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.level = 1
        self.score = 0
        self.player_health = self.STARTING_HEALTH
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.last_message = f"Level {self.level}"
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # Calculate distance to exit before moving
        dist_before = math.hypot(self.player_pos[0] - self.exit_pos[0], self.player_pos[1] - self.exit_pos[1])

        target_pos = self.player_pos.copy()
        if movement == 1:  # Up
            target_pos[1] -= 1
        elif movement == 2:  # Down
            target_pos[1] += 1
        elif movement == 3:  # Left
            target_pos[0] -= 1
        elif movement == 4:  # Right
            target_pos[0] += 1
        
        # Only process move if an actual move action was taken
        if movement != 0:
            tx, ty = target_pos
            if 0 <= tx < self.GRID_WIDTH and 0 <= ty < self.GRID_HEIGHT and self.grid[ty, tx] == self.CELL_FLOOR:
                self.player_pos = target_pos
                self.last_message = ""

                # Reward for moving closer/further from exit
                dist_after = math.hypot(self.player_pos[0] - self.exit_pos[0], self.player_pos[1] - self.exit_pos[1])
                if dist_after < dist_before:
                    reward += 0.01
                else:
                    reward -= 0.01

            else: # Bumped into a wall
                self.last_message = "Ouch! A wall."
                reward -= 0.05

        # Check for interactions at the new position
        # Enemy interaction
        enemy_idx = self._get_entity_index_at(self.player_pos, self.enemies)
        if enemy_idx is not None:
            self.player_health -= 1
            reward -= 1
            self.enemies.pop(enemy_idx)
            self.last_message = "Fought an enemy! -1 HP"
            # sfx: player_hit

        # Gold interaction
        gold_idx = self._get_entity_index_at(self.player_pos, self.gold_items)
        if gold_idx is not None:
            self.score += 1
            reward += 1
            self.gold_items.pop(gold_idx)
            self.last_message = "Found gold! +1"
            # sfx: collect_gold
        
        # Exit interaction
        if np.array_equal(self.player_pos, self.exit_pos):
            self.level += 1
            if self.level > self.WIN_LEVEL:
                self.game_over = True
                self.game_won = True
                reward += 100
                self.last_message = "You escaped the dungeon! YOU WIN!"
                # sfx: win_game
            else:
                reward += 10 # Reward for completing a level
                self.last_message = f"Descended to Level {self.level}!"
                self._generate_level()
                # sfx: next_level

        # Update steps and check for termination
        self.steps += 1
        terminated = False
        truncated = False

        if self.player_health <= 0:
            self.game_over = True
            terminated = True
            reward -= 100
            self.last_message = "You have fallen. GAME OVER."
            # sfx: player_death
        
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            truncated = True # Use truncated for time limit
            self.last_message = "Out of time. GAME OVER."

        if self.game_over:
            terminated = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_entity_index_at(self, pos, entity_list):
        for i, entity_pos in enumerate(entity_list):
            if np.array_equal(pos, entity_pos):
                return i
        return None

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
            "health": self.player_health,
            "level": self.level,
            "player_pos": self.player_pos.tolist(),
            "exit_pos": self.exit_pos.tolist(),
        }

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.GAME_AREA_X_OFFSET + x * self.TILE_SIZE,
                    self.GAME_AREA_Y_OFFSET + y * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                color = self.COLOR_WALL if self.grid[y, x] == self.CELL_WALL else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(
            self.GAME_AREA_X_OFFSET + ex * self.TILE_SIZE,
            self.GAME_AREA_Y_OFFSET + ey * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        
        # Draw gold
        for gx, gy in self.gold_items:
            pygame.gfxdraw.filled_circle(
                self.screen,
                self.GAME_AREA_X_OFFSET + gx * self.TILE_SIZE + self.TILE_SIZE // 2,
                self.GAME_AREA_Y_OFFSET + gy * self.TILE_SIZE + self.TILE_SIZE // 2,
                self.TILE_SIZE // 3,
                self.COLOR_GOLD
            )
        
        # Draw enemies
        for ex, ey in self.enemies:
            enemy_rect = pygame.Rect(
                self.GAME_AREA_X_OFFSET + ex * self.TILE_SIZE + 2,
                self.GAME_AREA_Y_OFFSET + ey * self.TILE_SIZE + 2,
                self.TILE_SIZE - 4, self.TILE_SIZE - 4
            )
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, enemy_rect)

        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(
            self.GAME_AREA_X_OFFSET + px * self.TILE_SIZE + 2,
            self.GAME_AREA_Y_OFFSET + py * self.TILE_SIZE + 2,
            self.TILE_SIZE - 4, self.TILE_SIZE - 4
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)
        # Add a small highlight to the player
        highlight_rect = player_rect.inflate(-6, -6)
        highlight_color = tuple(min(255, c + 50) for c in self.COLOR_PLAYER)
        pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_radius=2)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.STARTING_HEALTH)
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (20, 15, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (20, 15, bar_width * health_ratio, bar_height))
        health_text = self.font_small.render(f"HP: {self.player_health}/{self.STARTING_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (25, 17))

        # Score (Gold)
        score_text = self.font_large.render(f"Gold: {self.score}", True, self.COLOR_GOLD)
        score_rect = score_text.get_rect(centerx=self.SCREEN_WIDTH // 2, y=12)
        self.screen.blit(score_text, score_rect)
        
        # Level
        level_text_str = f"Level: {self.level}" if self.level <= self.WIN_LEVEL else "Victory!"
        level_text = self.font_large.render(level_text_str, True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(right=self.SCREEN_WIDTH - 20, y=12)
        self.screen.blit(level_text, level_rect)

        # Last message
        if self.last_message:
            message_surf = self.font_small.render(self.last_message, True, self.COLOR_TEXT)
            message_rect = message_surf.get_rect(centerx=self.SCREEN_WIDTH // 2, bottom=self.SCREEN_HEIGHT - 10)
            self.screen.blit(message_surf, message_rect)

    def close(self):
        pygame.font.quit()
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly.
    # It will not work with the "dummy" video driver, so you might need to
    # comment out the os.environ line at the top of the file to run this.
    
    # To run in interactive mode, comment out this line at the top of the file:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if os.getenv("SDL_VIDEODRIVER") == "dummy":
        print("Cannot run interactive test with SDL_VIDEODRIVER=dummy.")
        print("Comment out the os.environ line at the top of the file to run this.")
    else:
        env = GameEnv()
        obs, info = env.reset()
        done = False
        
        # Create a window to display the game
        pygame.display.set_caption("Dungeon Crawler")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        clock = pygame.time.Clock()

        print(env.game_description)
        print(env.user_guide)

        while not done:
            action = np.array([0, 0, 0])  # Default to no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    elif event.key == pygame.K_r: # Reset
                        obs, info = env.reset()
                    
                    # Since auto_advance is False, we only step on key presses
                    if action[0] != 0 or event.key == pygame.K_r:
                        if event.key != pygame.K_r:
                            obs, reward, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            print(f"Game Over! Final Score: {info['score']}. Resetting in 3 seconds...")
                            # Display final state
                            frame = np.transpose(obs, (1, 0, 2))
                            surf = pygame.surfarray.make_surface(frame)
                            screen.blit(surf, (0, 0))
                            pygame.display.flip()
                            pygame.time.wait(3000)
                            obs, info = env.reset()


            # Render the current state
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(30) # Limit FPS for human play

        env.close()