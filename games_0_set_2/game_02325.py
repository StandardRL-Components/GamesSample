
# Generated: 2025-08-28T04:27:13.104489
# Source Brief: brief_02325.md
# Brief Index: 2325

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your avatar on the grid. "
        "Space and Shift have no function in this game."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate an isometric grid, collecting all the green gems while avoiding red traps. "
        "The game is turn-based. Each move counts. Complete 3 levels to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 10, 10
        self.MAX_STEPS = 1000
        self.MAX_LEVELS = 3
        self.NUM_GEMS = 5

        # Visual constants
        self.TILE_W, self.TILE_H = 40, 20
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = self.HEIGHT // 2 - (self.GRID_H * self.TILE_H // 2) + 20

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_GRID_BORDER = (60, 70, 80)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_OUTLINE = (200, 220, 255)
        self.COLOR_GEM = (50, 255, 150)
        self.COLOR_GEM_OUTLINE = (200, 255, 220)
        self.COLOR_TRAP = (255, 80, 80)
        self.COLOR_TRAP_OUTLINE = (150, 40, 40)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_SHADOW = (10, 10, 10)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 24)
        except pygame.error:
            self.font_large = pygame.font.SysFont("sans", 36)
            self.font_small = pygame.font.SysFont("sans", 24)

        # Initialize state variables
        self.player_pos = [0, 0]
        self.gems = []
        self.traps = []
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self._setup_level()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Calculate reward based on distance to nearest gem ---
        old_pos = list(self.player_pos)
        dist_before = self._get_distance_to_nearest_gem(old_pos)
        
        # --- Update game logic ---
        self.steps += 1
        if movement == 1:  # Up
            self.player_pos[1] -= 1
        elif movement == 2:  # Down
            self.player_pos[1] += 1
        elif movement == 3:  # Left
            self.player_pos[0] -= 1
        elif movement == 4:  # Right
            self.player_pos[0] += 1
        
        # Clamp player position to grid
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_W - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_H - 1)

        # --- Calculate rewards and check for events ---
        reward = 0
        
        # Distance-based reward
        dist_after = self._get_distance_to_nearest_gem(self.player_pos)
        if dist_after < dist_before:
            reward += 1.0  # Closer to a gem
        elif dist_after > dist_before:
            reward -= 0.1  # Further from a gem
            
        # Event-based rewards
        if self.player_pos in self.gems:
            # sfx: gem_collect
            self.gems.remove(self.player_pos)
            self.score += 10
            reward += 10.0
            
            if not self.gems: # Level complete
                # sfx: level_complete
                self.score += 50
                reward += 50.0
                self.level += 1
                if self.level > self.MAX_LEVELS:
                    self.game_over = True
                else:
                    self._setup_level()

        elif self.player_pos in self.traps:
            # sfx: trap_hit
            self.score -= 100
            reward -= 100.0
            self.game_over = True
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "player_pos": self.player_pos,
            "gems_left": len(self.gems)
        }

    def _setup_level(self):
        num_traps = 4 + self.level
        all_coords = set((x, y) for x in range(self.GRID_W) for y in range(self.GRID_H))

        # Ensure player starts away from the center to avoid trivial solutions
        edge_coords = list({c for c in all_coords if c[0] == 0 or c[0] == self.GRID_W - 1 or c[1] == 0 or c[1] == self.GRID_H - 1})
        player_start_tuple = edge_coords[self.np_random.integers(len(edge_coords))]
        self.player_pos = list(player_start_tuple)
        all_coords.remove(player_start_tuple)

        # Place gems
        gem_coords_tuples = self.np_random.choice(list(all_coords), size=self.NUM_GEMS, replace=False)
        self.gems = [list(c) for c in gem_coords_tuples]
        for c in gem_coords_tuples:
            all_coords.remove(tuple(c))

        # Place traps
        trap_coords_tuples = self.np_random.choice(list(all_coords), size=num_traps, replace=False)
        self.traps = [list(c) for c in trap_coords_tuples]

    def _isometric_to_screen(self, gx, gy):
        sx = self.ORIGIN_X + (gx - gy) * (self.TILE_W / 2)
        sy = self.ORIGIN_Y + (gx + gy) * (self.TILE_H / 2)
        return int(sx), int(sy)

    def _draw_iso_tile(self, surface, color, gx, gy, height=0, outline_color=None):
        sx, sy = self._isometric_to_screen(gx, gy)
        sy -= height
        points = [
            (sx, sy - self.TILE_H / 2),
            (sx + self.TILE_W / 2, sy),
            (sx, sy + self.TILE_H / 2),
            (sx - self.TILE_W / 2, sy),
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        if outline_color:
            pygame.gfxdraw.aapolygon(surface, points, outline_color)

    def _draw_text(self, surface, text, font, pos, color, shadow_color):
        x, y = pos
        shadow_img = font.render(text, True, shadow_color)
        surface.blit(shadow_img, (x + 2, y + 2))
        text_img = font.render(text, True, color)
        surface.blit(text_img, (x, y))

    def _render_game(self):
        # Draw grid tiles
        for gy in range(self.GRID_H):
            for gx in range(self.GRID_W):
                self._draw_iso_tile(self.screen, self.COLOR_GRID, gx, gy, 0, self.COLOR_GRID_BORDER)

        # Draw traps (underneath player/gems)
        for tx, ty in self.traps:
            self._draw_iso_tile(self.screen, self.COLOR_TRAP, tx, ty, 0, self.COLOR_TRAP_OUTLINE)

        # Draw gems
        gem_pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 3  # 0 to 3 pixels
        for gx, gy in self.gems:
            sx, sy = self._isometric_to_screen(gx, gy)
            radius = int(self.TILE_W / 4 + gem_pulse)
            pygame.gfxdraw.filled_circle(self.screen, sx, sy - 8, radius, self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, sx, sy - 8, radius, self.COLOR_GEM_OUTLINE)

        # Draw player
        px, py = self.player_pos
        sx, sy = self._isometric_to_screen(px, py)
        player_height = 12
        player_rect = pygame.Rect(sx - 10, sy - player_height - 10, 20, 20)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, width=2, border_radius=4)
        
    def _render_ui(self):
        self._draw_text(self.screen, f"Score: {self.score}", self.font_large, (10, 10), self.COLOR_TEXT, self.COLOR_SHADOW)
        self._draw_text(self.screen, f"Level: {self.level}/{self.MAX_LEVELS}", self.font_large, (self.WIDTH - 150, 10), self.COLOR_TEXT, self.COLOR_SHADOW)
        self._draw_text(self.screen, f"Steps: {self.steps}/{self.MAX_STEPS}", self.font_small, (10, 45), self.COLOR_TEXT, self.COLOR_SHADOW)

    def _get_distance_to_nearest_gem(self, pos):
        if not self.gems:
            return 0
        
        distances = [abs(pos[0] - gx) + abs(pos[1] - gy) for gx, gy in self.gems]
        return min(distances)

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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Isometric Gem Collector")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default to no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                elif event.key == pygame.K_q: # Quit on 'q' key
                    running = False

        if action[0] != 0: # Only step if a move key was pressed
             obs, reward, terminated, truncated, info = env.step(action)
             print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
             if terminated:
                 print("Game Over! Press 'r' to restart.")

        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for manual play

    env.close()