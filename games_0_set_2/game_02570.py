
# Generated: 2025-08-27T20:46:17.452043
# Source Brief: brief_02570.md
# Brief Index: 2570

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric grid-based puzzle game where the player collects gems while avoiding traps.
    The game is presented as a Gymnasium environment.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character on the grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Navigate the isometric grid to collect all 10 gems. "
        "Plan your moves carefully to avoid the red traps!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_W = 15
    GRID_H = 15
    TILE_W_ISO = 32
    TILE_H_ISO = 16
    NUM_GEMS_TO_WIN = 10
    NUM_TRAPS = 25
    MAX_STEPS = 1000

    # Colors (Vibrant and High Contrast)
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (50, 55, 65)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_GEM = (0, 255, 120)
    COLOR_GEM_GLOW = (0, 255, 120, 60)
    COLOR_TRAP = (255, 50, 50)
    COLOR_TRAP_GLOW = (255, 50, 50, 70)
    COLOR_UI_TEXT = (220, 220, 220)


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
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 50, bold=True)
        
        # Centering the grid
        self.grid_origin_x = self.SCREEN_WIDTH // 2
        self.grid_origin_y = (self.SCREEN_HEIGHT - self.GRID_H * self.TILE_H_ISO) // 2 + 20

        # Initialize state variables to be populated in reset()
        self.player_pos = None
        self.gems = None
        self.traps = None
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.win = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.win = False

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """
        Procedurally generates a new level, ensuring all gems and traps are reachable.
        """
        self.player_pos = (self.GRID_W // 2, self.GRID_H // 2)

        # Use BFS to find all reachable cells from the player's start position
        queue = collections.deque([self.player_pos])
        reachable_cells = {self.player_pos}
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and (nx, ny) not in reachable_cells:
                    reachable_cells.add((nx, ny))
                    queue.append((nx, ny))

        # We can't place items on the player's starting cell
        possible_placements = list(reachable_cells - {self.player_pos})
        
        num_placements = min(len(possible_placements), self.NUM_GEMS_TO_WIN + self.NUM_TRAPS)
        
        # Use numpy's random generator for reproducibility with seeding
        indices = self.np_random.choice(len(possible_placements), num_placements, replace=False)
        placements = [possible_placements[i] for i in indices]

        self.gems = placements[:self.NUM_GEMS_TO_WIN]
        self.traps = placements[self.NUM_GEMS_TO_WIN:]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        old_pos = self.player_pos
        dist_before = self._find_nearest_gem_dist(old_pos)

        # --- Update Player Position ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if dx != 0 or dy != 0:
            new_x = max(0, min(self.GRID_W - 1, self.player_pos[0] + dx))
            new_y = max(0, min(self.GRID_H - 1, self.player_pos[1] + dy))
            self.player_pos = (new_x, new_y)

        # --- Check Collisions and Calculate Rewards ---
        if self.player_pos in self.traps:
            # sound: player_hit_trap.wav
            reward = -50
            self.score -= 50
            self.game_over = True
            self.win = False
        elif self.player_pos in self.gems:
            # sound: collect_gem.wav
            self.gems.remove(self.player_pos)
            self.gems_collected += 1
            self.score += 10
            reward = 10
            if self.gems_collected >= self.NUM_GEMS_TO_WIN:
                # sound: win_level.wav
                self.score += 100
                reward += 100
                self.game_over = True
                self.win = True
        elif movement != 0: # No collision, calculate distance-based reward
            dist_after = self._find_nearest_gem_dist(self.player_pos)
            if dist_after < dist_before:
                reward = 1  # Moved closer
                self.score += 1
            else:
                reward = -0.1 # Moved further or same distance

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
             self.game_over = True # Ran out of time
             self.win = False

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _find_nearest_gem_dist(self, pos):
        """Calculates Manhattan distance to the nearest gem."""
        if not self.gems:
            return 0
        px, py = pos
        min_dist = float('inf')
        for gx, gy in self.gems:
            dist = abs(px - gx) + abs(py - gy)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _grid_to_iso(self, x, y):
        """Converts grid coordinates to isometric screen coordinates."""
        screen_x = self.grid_origin_x + (x - y) * (self.TILE_W_ISO / 2)
        screen_y = self.grid_origin_y + (x + y) * (self.TILE_H_ISO / 2)
        return int(screen_x), int(screen_y)
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the main game elements (grid, items, player)."""
        # Draw grid lines
        for i in range(self.GRID_W + 1):
            start = self._grid_to_iso(i, 0)
            end = self._grid_to_iso(i, self.GRID_H)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for i in range(self.GRID_H + 1):
            start = self._grid_to_iso(0, i)
            end = self._grid_to_iso(self.GRID_W, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # Draw traps
        for x, y in self.traps:
            sx, sy = self._grid_to_iso(x, y)
            # Pulsating glow effect
            pulse_radius = 8 + 3 * math.sin(self.steps * 0.2 + x + y)
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, int(pulse_radius), self.COLOR_TRAP_GLOW)
            # Main trap shape
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, 6, self.COLOR_TRAP)
            pygame.gfxdraw.aacircle(self.screen, sx, sy, 6, self.COLOR_TRAP)
            
        # Draw gems
        for x, y in self.gems:
            sx, sy = self._grid_to_iso(x, y)
            # Sparkling effect
            sparkle_size = 10 + 2 * math.sin(self.steps * 0.3 + x)
            points = [
                (sx, sy - sparkle_size),
                (sx + sparkle_size * 0.6, sy),
                (sx, sy + sparkle_size),
                (sx - sparkle_size * 0.6, sy)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GEM)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GEM)

        # Draw player
        px, py = self.player_pos
        sx, sy = self._grid_to_iso(px, py)
        player_size = 8
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, player_size + 5, self.COLOR_PLAYER_GLOW)
        # Player shape (diamond)
        points = [
            (sx, sy - player_size),
            (sx + player_size, sy),
            (sx, sy + player_size),
            (sx - player_size, sy)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        """Renders the UI overlay (score, gems, game over text)."""
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))
        
        gems_text = self.font_ui.render(f"GEMS: {self.gems_collected} / {self.NUM_GEMS_TO_WIN}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gems_text, (15, 35))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                end_text = self.font_game_over.render("YOU WIN!", True, self.COLOR_GEM)
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.COLOR_TRAP)
                
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "player_pos": self.player_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Isometric Gem Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.user_guide)
    print(env.game_description)

    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if terminated:
                    # If game is over, any key press resets the environment
                    obs, info = env.reset()
                    terminated = False
                    continue

                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                # Since auto_advance is False, we only step on key press
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # Draw the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for human play

    env.close()