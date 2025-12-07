
# Generated: 2025-08-28T06:50:09.455991
# Source Brief: brief_03061.md
# Brief Index: 3061

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze."
    )

    game_description = (
        "Navigate an isometric maze, collecting all the gems before you run out of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and rendering setup
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game constants
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_WALL = (60, 70, 80)
        self.COLOR_PATH = (40, 45, 55)
        self.COLOR_PLAYER = (57, 255, 20)
        self.COLOR_GEM = (255, 223, 0)
        self.COLOR_TEXT = (230, 230, 230)

        self.MAX_MOVES = 20
        self.NUM_GEMS = 10
        self.MAX_LEVEL = 10 # Cap maze size increase

        # Game state variables
        self.level = 0
        self.maze_base_size = 13 # Must be odd
        self.maze = np.array([])
        self.player_pos = (0, 0)
        self.gem_locations = []
        self.gems_collected = 0
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        
        # Isometric projection parameters
        self.tile_width = 32
        self.tile_height = self.tile_width / 2
        self.origin_x = self.WIDTH // 2
        self.origin_y = 80
        
        self.validate_implementation()

    def _generate_maze(self, width, height):
        # Ensure odd dimensions for maze generation algorithm
        width = width if width % 2 != 0 else width + 1
        height = height if height % 2 != 0 else height + 1
        
        maze = np.ones((height, width), dtype=np.uint8) # 1 for wall
        
        def is_valid(r, c):
            return 0 <= r < height and 0 <= c < width

        def carve(r, c):
            maze[r, c] = 0 # 0 for path
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            self.np_random.shuffle(directions)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                wall_r, wall_c = r + dr // 2, c + dc // 2
                if is_valid(nr, nc) and maze[nr, nc] == 1:
                    maze[wall_r, wall_c] = 0
                    carve(nr, nc)
        
        # Start carving from a random odd position
        start_r = self.np_random.integers(0, (height // 2)) * 2 + 1
        start_c = self.np_random.integers(0, (width // 2)) * 2 + 1
        carve(start_r, start_c)
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # On loss, reset level
        if self.game_over and not self.game_won:
            self.level = 0
            
        maze_dim = self.maze_base_size + min(self.level * 2, self.MAX_LEVEL * 2)
        self.maze = self._generate_maze(maze_dim, maze_dim)
        
        path_cells = np.argwhere(self.maze == 0).tolist()
        self.np_random.shuffle(path_cells)
        
        # Place player
        player_r, player_c = path_cells.pop()
        self.player_pos = (player_r, player_c)
        
        # Place gems
        self.gem_locations = []
        for _ in range(min(self.NUM_GEMS, len(path_cells))):
            gem_r, gem_c = path_cells.pop()
            self.gem_locations.append((gem_r, gem_c))
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.gems_collected = 0
        self.game_over = False
        self.game_won = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # Calculate distance to nearest gem before moving
        dist_before = self._get_dist_to_nearest_gem()

        # Handle movement
        if movement > 0:
            dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
            new_r, new_c = self.player_pos[0] + dr, self.player_pos[1] + dc
            
            # Check boundaries and walls
            if 0 <= new_r < self.maze.shape[0] and 0 <= new_c < self.maze.shape[1] and self.maze[new_r, new_c] == 0:
                self.player_pos = (new_r, new_c)
                # // Play move sound
            else:
                pass # // Play bump sound

        self.moves_left -= 1

        # Calculate distance-based reward
        dist_after = self._get_dist_to_nearest_gem()
        if dist_after < dist_before:
            reward += 1.0
        elif dist_after > dist_before:
            reward -= 0.1

        # Check for gem collection
        if self.player_pos in self.gem_locations:
            self.gem_locations.remove(self.player_pos)
            self.gems_collected += 1
            self.score += 10
            reward += 10
            # // Play gem collection sound

        # Check for termination conditions
        terminated = False
        if self.gems_collected >= self.NUM_GEMS:
            self.game_over = True
            self.game_won = True
            self.level += 1
            self.score += 50
            reward += 50
            terminated = True
            # // Play win sound
        elif self.moves_left <= 0:
            self.game_over = True
            self.game_won = False
            self.score -= 50
            reward -= 50
            terminated = True
            # // Play lose sound

        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_dist_to_nearest_gem(self):
        if not self.gem_locations:
            return 0
        player_r, player_c = self.player_pos
        min_dist = float('inf')
        for gem_r, gem_c in self.gem_locations:
            dist = abs(player_r - gem_r) + abs(player_c - gem_c) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist
        
    def _iso_to_screen(self, r, c):
        screen_x = self.origin_x + (c - r) * self.tile_width / 2
        screen_y = self.origin_y + (c + r) * self.tile_height / 2
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Draw maze tiles
        for r in range(self.maze.shape[0]):
            for c in range(self.maze.shape[1]):
                screen_x, screen_y = self._iso_to_screen(r, c)
                points = [
                    (screen_x, screen_y - self.tile_height / 2),
                    (screen_x + self.tile_width / 2, screen_y),
                    (screen_x, screen_y + self.tile_height / 2),
                    (screen_x - self.tile_width / 2, screen_y),
                ]
                color = self.COLOR_WALL if self.maze[r, c] == 1 else self.COLOR_PATH
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.aapolygon(self.screen, points, color)

        # Draw gems
        for r, c in self.gem_locations:
            screen_x, screen_y = self._iso_to_screen(r, c)
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            radius = int(self.tile_width / 4 + pulse * 2)
            pygame.gfxdraw.filled_circle(self.screen, screen_x, screen_y, radius, self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, screen_x, screen_y, radius, self.COLOR_GEM)
            
            # Sparkle effect
            for i in range(4):
                angle = self.steps * 0.1 + i * math.pi / 2
                sparkle_len = radius + 3 + pulse * 2
                start_pos = (
                    screen_x + math.cos(angle) * (radius - 2),
                    screen_y + math.sin(angle) * (radius - 2)
                )
                end_pos = (
                    screen_x + math.cos(angle) * sparkle_len,
                    screen_y + math.sin(angle) * sparkle_len
                )
                pygame.draw.aaline(self.screen, self.COLOR_GEM, start_pos, end_pos)

        # Draw player
        player_r, player_c = self.player_pos
        screen_x, screen_y = self._iso_to_screen(player_r, player_c)
        points = [
            (screen_x, screen_y - self.tile_height / 2),
            (screen_x + self.tile_width / 2, screen_y),
            (screen_x, screen_y + self.tile_height / 2),
            (screen_x - self.tile_width / 2, screen_y),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Gem count
        gem_text = self.font_small.render(f"Gems: {self.gems_collected} / {self.NUM_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (10, 10))

        # Move counter
        move_text = self.font_small.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(move_text, (self.WIDTH - move_text.get_width() - 10, 10))

        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 30))
        self.screen.blit(score_text, score_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_PLAYER if self.game_won else (255, 50, 50)
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

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
            "gems_collected": self.gems_collected,
            "moves_left": self.moves_left,
            "level": self.level,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for interactive testing
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Gem Collector")
    clock = pygame.time.Clock()
    
    done = False
    print(env.user_guide)
    
    while not done:
        action = [0, 0, 0] # Default to no-op
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1 # Up-Left
                elif event.key == pygame.K_DOWN:
                    action[0] = 2 # Down-Right
                elif event.key == pygame.K_LEFT:
                    action[0] = 3 # Down-Left
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4 # Up-Right
                
                # We only process one keydown event per frame for this turn-based game
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
                    if terminated:
                        print("Episode finished. Resetting in 3 seconds...")
                        pygame.time.wait(3000)
                        obs, info = env.reset()

        # Rendering
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS
        
    env.close()