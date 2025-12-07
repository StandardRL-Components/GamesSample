
# Generated: 2025-08-27T22:03:39.130258
# Source Brief: brief_02998.md
# Brief Index: 2998

        
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
        "Controls: Use arrow keys (↑↓←→) to move your character on the grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate an isometric grid, collecting all the gems while avoiding the traps to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_WIDTH = 12
        self.GRID_HEIGHT = 12
        self.NUM_GEMS = 15
        self.NUM_TRAPS = 20
        self.MAX_STEPS = 250 # Reduced for tighter episodes

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Visual constants
        self.TILE_WIDTH_HALF = 32
        self.TILE_HEIGHT_HALF = 16
        self.ORIGIN_X = 640 // 2
        self.ORIGIN_Y = 60

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 60, 80)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_SHADOW = (10, 15, 20, 100)
        self.COLOR_TRAP = (180, 20, 20)
        self.COLOR_TRAP_X = (0, 0, 0)
        self.COLOR_TEXT = (220, 220, 230)
        self.GEM_COLOR_PALETTE = [
            ((255, 223, 0), (255, 191, 0)),    # Gold
            ((0, 255, 127), (0, 220, 100)),   # Spring Green
            ((255, 0, 255), (220, 0, 220)),   # Magenta
            ((255, 165, 0), (230, 140, 0)),   # Orange
            ((138, 43, 226), (110, 30, 200)), # BlueViolet
            ((220, 20, 60), (200, 10, 50)),   # Crimson
        ]

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)

        # State variables (initialized in reset)
        self.player_pos = (0, 0)
        self.gems = set()
        self.gem_colors = {}
        self.traps = set()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.np_random = None
        self.last_player_action = 0

        self.reset()
        self.validate_implementation()

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _get_manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _find_nearest(self, pos, item_list):
        if not item_list:
            return float('inf')
        return min(self._get_manhattan_distance(pos, item_pos) for item_pos in item_list)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_player_action = 0

        # --- Level Generation with Solvability Guarantee ---
        while True:
            all_coords = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
            
            # Use choice for sampling sets without replacement
            trap_coords_list = self.np_random.choice(list(all_coords), size=self.NUM_TRAPS, replace=False)
            self.traps = {tuple(c) for c in trap_coords_list}
            
            free_cells = all_coords - self.traps
            if not free_cells: continue

            # Find the largest connected component of free cells using BFS
            max_component = set()
            visited_overall = set()
            for start_cell in free_cells:
                if start_cell in visited_overall: continue
                
                component = set()
                q = [start_cell]
                visited_component = {start_cell}
                
                while q:
                    curr = q.pop(0)
                    component.add(curr)
                    visited_overall.add(curr)
                    
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        neighbor = (curr[0] + dx, curr[1] + dy)
                        if neighbor in free_cells and neighbor not in visited_component:
                            visited_component.add(neighbor)
                            q.append(neighbor)
                
                if len(component) > len(max_component):
                    max_component = component
            
            if len(max_component) >= self.NUM_GEMS + 1:
                placement_options = list(max_component)
                chosen_indices = self.np_random.choice(len(placement_options), size=self.NUM_GEMS + 1, replace=False)
                
                self.player_pos = placement_options[chosen_indices[0]]
                self.gems = {placement_options[i] for i in chosen_indices[1:]}

                self.gem_colors = {}
                gem_color_palette = self.np_random.permutation(self.GEM_COLOR_PALETTE)
                for i, gem_pos in enumerate(self.gems):
                    self.gem_colors[gem_pos] = gem_color_palette[i % len(gem_color_palette)]
                
                break # Valid level generated

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        self.last_player_action = movement

        reward = 0
        old_pos = self.player_pos
        old_dist_gem = self._find_nearest(old_pos, self.gems)
        old_dist_trap = self._find_nearest(old_pos, self.traps)

        # Update player position
        if movement == 1: # Up
            self.player_pos = (self.player_pos[0], self.player_pos[1] - 1)
        elif movement == 2: # Down
            self.player_pos = (self.player_pos[0], self.player_pos[1] + 1)
        elif movement == 3: # Left
            self.player_pos = (self.player_pos[0] - 1, self.player_pos[1])
        elif movement == 4: # Right
            self.player_pos = (self.player_pos[0] + 1, self.player_pos[1])

        # Clamp position to grid boundaries
        self.player_pos = (
            max(0, min(self.GRID_WIDTH - 1, self.player_pos[0])),
            max(0, min(self.GRID_HEIGHT - 1, self.player_pos[1]))
        )

        # Distance-based rewards
        if movement != 0:
            new_dist_gem = self._find_nearest(self.player_pos, self.gems)
            new_dist_trap = self._find_nearest(self.player_pos, self.traps)
            if new_dist_gem < old_dist_gem:
                reward += 1 # Moved closer to a gem
            if new_dist_trap < old_dist_trap:
                reward -= 1 # Moved closer to a trap

        # Check for events at the new position
        if self.player_pos in self.gems:
            # sfx: gem collect
            self.gems.remove(self.player_pos)
            self.score += 10
            reward += 10
        
        if self.player_pos in self.traps:
            # sfx: player falls in trap
            self.game_over = True
            reward -= 100

        self.steps += 1
        terminated = False

        # Check termination conditions
        if self.game_over:
            terminated = True
        elif not self.gems: # All gems collected
            # sfx: level complete
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _render_game(self):
        # Render grid tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                sx, sy = self._iso_to_screen(x, y)
                points = [
                    (sx, sy),
                    (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
                    (sx, sy + self.TILE_HEIGHT_HALF * 2),
                    (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Render traps
        for tx, ty in self.traps:
            sx, sy = self._iso_to_screen(tx, ty)
            points = [
                    (sx, sy),
                    (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
                    (sx, sy + self.TILE_HEIGHT_HALF * 2),
                    (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF)
                ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_TRAP)
            pygame.gfxdraw.line(self.screen, sx - 12, sy + 16, sx + 12, sy + 48, self.COLOR_TRAP_X)
            pygame.gfxdraw.line(self.screen, sx + 12, sy + 16, sx - 12, sy + 48, self.COLOR_TRAP_X)


        # Render gems
        for gx, gy in self.gems:
            sx, sy = self._iso_to_screen(gx, gy)
            color_bright, color_dark = self.gem_colors.get((gx, gy), self.GEM_COLOR_PALETTE[0])
            
            # Diamond shape
            points_top = [(sx, sy + 10), (sx + 12, sy + 22), (sx, sy + 34)]
            points_bottom = [(sx, sy + 10), (sx - 12, sy + 22), (sx, sy + 34)]
            pygame.gfxdraw.filled_polygon(self.screen, points_top, color_bright)
            pygame.gfxdraw.filled_polygon(self.screen, points_bottom, color_dark)
            pygame.gfxdraw.aapolygon(self.screen, points_top, color_bright)
            pygame.gfxdraw.aapolygon(self.screen, points_bottom, color_dark)
            # Glint
            pygame.gfxdraw.pixel(self.screen, sx - 2, sy + 15, (255, 255, 255))


        # Render player
        px, py = self.player_pos
        sx, sy = self._iso_to_screen(px, py)
        
        # Bobbing animation
        bob_offset = math.sin(self.steps * 0.5 + (px + py)) * 2
        sy += bob_offset

        # Shadow
        shadow_rect = (sx - 16, sy + 40, 32, 12)
        shadow_surface = pygame.Surface((32, 12), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, self.COLOR_PLAYER_SHADOW, (0, 0, 32, 12))
        self.screen.blit(shadow_surface, (sx - 16, sy + 40))

        # Player Body (cylinder-like)
        pygame.draw.ellipse(self.screen, (20,40,80), (sx-14, sy+28, 28, 14))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (sx-14, sy+10, 28, 25))
        pygame.draw.ellipse(self.screen, self.COLOR_PLAYER, (sx-14, sy+5, 28, 14))
        pygame.draw.ellipse(self.screen, (100, 230, 255), (sx-12, sy+6, 24, 12))

    def _render_ui(self):
        # Score display
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps display
        steps_text = self.font_medium.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.screen.get_width() - steps_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2))
            self.screen.blit(end_text, text_rect)

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
            "gems_remaining": len(self.gems),
            "player_pos": self.player_pos,
        }

    def close(self):
        pygame.font.quit()
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Isometric Gem Collector")
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    
    print(env.user_guide)
    print(env.game_description)

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    terminated = False
                    obs, info = env.reset()
                
                # Only register one-time key presses for turn-based movement
                if not terminated:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
        
        # Only step the environment if an action was taken or game is over
        if action[0] != 0 or terminated:
            if not terminated:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")

        # Update the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()