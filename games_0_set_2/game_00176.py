
# Generated: 2025-08-27T12:49:36.132942
# Source Brief: brief_00176.md
# Brief Index: 176

        
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
        "Controls: Arrow keys to move your blue square. Collect all green gems to win, but avoid the red traps!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A grid-based puzzle game. Navigate the maze to collect all the gems while avoiding hidden traps."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.CELL_WIDTH = self.SCREEN_WIDTH // self.GRID_WIDTH
        self.CELL_HEIGHT = self.SCREEN_HEIGHT // self.GRID_HEIGHT
        self.NUM_GEMS = 10
        self.NUM_TRAPS = 15
        self.MIN_REACHABLE_GEMS = 3
        self.MAX_STEPS = 200 # Reduced from 1000 for a tighter puzzle feel

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_GEM = (0, 255, 150)
        self.COLOR_TRAP = (255, 50, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_WIN = (255, 215, 0)
        self.COLOR_LOSE = self.COLOR_TRAP

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.player_pos = (0, 0)
        self.gems = []
        self.traps = []
        self.rng = None

        # Initialize state
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Use a default generator if no seed is provided
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False

        # Generate a valid level
        level_valid = False
        while not level_valid:
            self._generate_level_data()
            level_valid = self._validate_level()

        return self._get_observation(), self._get_info()

    def _generate_level_data(self):
        """Places player, gems, and traps on the grid."""
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.rng.shuffle(all_coords)

        self.player_pos = all_coords.pop()
        
        # Ensure we don't try to pop more items than available
        num_gems_to_place = min(self.NUM_GEMS, len(all_coords))
        self.gems = [all_coords.pop() for _ in range(num_gems_to_place)]

        num_traps_to_place = min(self.NUM_TRAPS, len(all_coords))
        self.traps = [all_coords.pop() for _ in range(num_traps_to_place)]

    def _bfs_pathfinder(self, start_pos, targets=None, get_path_dist=False):
        """
        A flexible BFS implementation.
        - Finds if any target is reachable.
        - Can count all reachable targets.
        - Can return distance to the first found target.
        """
        queue = deque([(start_pos, 0)])
        visited = {start_pos}
        reachable_targets = set()
        
        while queue:
            (x, y), dist = queue.popleft()

            if targets and (x, y) in targets:
                if get_path_dist:
                    return dist
                reachable_targets.add((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited and (nx, ny) not in self.traps:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
        
        if get_path_dist:
            return float('inf') # No path found
        return reachable_targets

    def _validate_level(self):
        """Ensures at least MIN_REACHABLE_GEMS are accessible from the start."""
        if not self.gems: return True # No gems to reach
        reachable_gems = self._bfs_pathfinder(self.player_pos, targets=set(self.gems))
        return len(reachable_gems) >= self.MIN_REACHABLE_GEMS

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        reward = 0
        terminated = False
        
        old_pos = self.player_pos
        
        # Calculate distance to nearest gem before moving
        dist_before = self._get_dist_to_nearest_gem(old_pos)

        # Update player position based on movement action
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
        
        moved = False
        if movement != 0 and 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
            self.player_pos = new_pos
            moved = True

        # Check for collisions and events
        if self.player_pos in self.traps:
            # sfx: player_fall
            reward = -50
            terminated = True
            self.game_over = True
        elif self.player_pos in self.gems:
            # sfx: gem_collect
            self.gems.remove(self.player_pos)
            self.score += 1
            reward = 10
            if self.score >= self.NUM_GEMS:
                # sfx: level_win
                reward += 50
                terminated = True
                self.game_over = True
                self.win_condition = True
        elif moved:
            # Continuous reward for moving towards/away from nearest gem
            dist_after = self._get_dist_to_nearest_gem(self.player_pos)
            if dist_after < dist_before:
                reward += 1
            elif dist_after > dist_before:
                reward -= 0.1

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_dist_to_nearest_gem(self, pos):
        if not self.gems:
            return 0
        
        min_dist = float('inf')
        for gem_pos in self.gems:
            dist = self._bfs_pathfinder(pos, targets={gem_pos}, get_path_dist=True)
            if dist < min_dist:
                min_dist = dist
        return min_dist

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
            "player_pos": self.player_pos,
            "gems_left": len(self.gems)
        }
        
    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(x * self.CELL_WIDTH, y * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw traps
        for x, y in self.traps:
            center_x = int(x * self.CELL_WIDTH + self.CELL_WIDTH / 2)
            center_y = int(y * self.CELL_HEIGHT + self.CELL_HEIGHT / 2)
            size = int(min(self.CELL_WIDTH, self.CELL_HEIGHT) * 0.3)
            pygame.draw.line(self.screen, self.COLOR_TRAP, (center_x - size, center_y - size), (center_x + size, center_y + size), 3)
            pygame.draw.line(self.screen, self.COLOR_TRAP, (center_x - size, center_y + size), (center_x + size, center_y - size), 3)

        # Draw gems
        for x, y in self.gems:
            center_x = x * self.CELL_WIDTH + self.CELL_WIDTH // 2
            center_y = y * self.CELL_HEIGHT + self.CELL_HEIGHT // 2
            
            # Shimmer effect
            shimmer = (math.sin(self.steps * 0.2 + x + y) + 1) / 2 # range 0-1
            size_mod = 0.8 + shimmer * 0.2 # range 0.8-1.0
            
            w = int(self.CELL_WIDTH * 0.3 * size_mod)
            h = int(self.CELL_HEIGHT * 0.4 * size_mod)
            
            points = [
                (center_x, center_y - h), (center_x + w, center_y),
                (center_x, center_y + h), (center_x - w, center_y)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GEM)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GEM)

        # Draw player
        player_rect = pygame.Rect(
            self.player_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH * 0.15,
            self.player_pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT * 0.15,
            self.CELL_WIDTH * 0.7,
            self.CELL_HEIGHT * 0.7
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        
    def _render_ui(self):
        # Draw score
        score_text = self.font_main.render(f"GEMS: {self.score}/{self.NUM_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Draw steps
        steps_text = self.font_main.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        steps_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(steps_text, steps_rect)

        # Draw game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_condition:
                msg_text = "YOU WIN!"
                msg_color = self.COLOR_WIN
            else:
                msg_text = "GAME OVER"
                msg_color = self.COLOR_LOSE
            
            game_over_surf = self.font_game_over.render(msg_text, True, msg_color)
            game_over_rect = game_over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_surf, game_over_rect)
            
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
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Collector")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if action[0] != 0: # Only step if a move key is pressed
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            if terminated:
                print("--- Episode Finished ---")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(10) # Limit frame rate for manual play
        
    env.close()