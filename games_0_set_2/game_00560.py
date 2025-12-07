
# Generated: 2025-08-27T14:01:04.825399
# Source Brief: brief_00560.md
# Brief Index: 560

        
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
        "Controls: Use arrow keys to move the gem on the grid. Collect all 15 gems to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic isometric puzzle. Move the gem to its target, avoiding obstacles, to collect all 15 gems before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # --- Game Constants ---
        self.GRID_SIZE = (12, 12)
        self.TILE_WIDTH = 48
        self.TILE_HEIGHT = 24
        self.TILE_WIDTH_HALF = self.TILE_WIDTH // 2
        self.TILE_HEIGHT_HALF = self.TILE_HEIGHT // 2
        self.ORIGIN_X = self.screen.get_width() // 2
        self.ORIGIN_Y = 60
        self.TOTAL_GEMS_TO_COLLECT = 15
        self.MOVES_PER_GEM = 30
        
        # --- Colors ---
        self.COLOR_BG = (45, 52, 54)
        self.COLOR_GRID = (99, 110, 114)
        self.COLOR_OBSTACLE_TOP = (129, 140, 144)
        self.COLOR_OBSTACLE_SIDE = (80, 90, 94)
        self.COLOR_TARGET = (178, 190, 195)
        self.COLOR_TEXT = (223, 230, 233)
        self.GEM_COLORS = [
            (231, 76, 60), (230, 126, 34), (241, 196, 15), (46, 204, 113),
            (26, 188, 156), (52, 152, 219), (155, 89, 182), (52, 73, 94),
            (236, 240, 241), (149, 165, 166), (211, 84, 0), (192, 57, 43),
            (39, 174, 96), (41, 128, 185), (142, 68, 173)
        ]

        # --- Fonts ---
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 50)
        
        # --- State Variables ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gems_collected = 0
        self.moves_left = 0
        self.gem_pos = (0, 0)
        self.target_pos = (0, 0)
        self.current_gem_color = (0, 0, 0)
        self.obstacles = []
        self.message = ""
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gems_collected = 0
        self.message = ""
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()
    
    def _setup_level(self):
        self.moves_left = self.MOVES_PER_GEM
        self.current_gem_color = self.GEM_COLORS[self.gems_collected % len(self.GEM_COLORS)]

        # Difficulty scaling
        obstacle_density_factor = self.gems_collected // 3
        obstacle_density = min(0.4, 0.1 * obstacle_density_factor)
        num_obstacles = int(self.GRID_SIZE[0] * self.GRID_SIZE[1] * obstacle_density)

        while True:
            all_cells = [(x, y) for x in range(self.GRID_SIZE[0]) for y in range(self.GRID_SIZE[1])]
            
            # Ensure we don't try to pick more unique items than available cells
            num_to_pick = 2 + num_obstacles
            if num_to_pick > len(all_cells):
                 num_obstacles = len(all_cells) - 2
                 num_to_pick = len(all_cells)
            
            chosen_indices = self.np_random.choice(len(all_cells), size=num_to_pick, replace=False)
            chosen_cells = [all_cells[i] for i in chosen_indices]

            self.gem_pos = chosen_cells[0]
            self.target_pos = chosen_cells[1]
            self.obstacles = chosen_cells[2:]

            if self._has_path(self.gem_pos, self.target_pos, self.obstacles):
                break

    def _has_path(self, start, end, obstacles):
        q = deque([start])
        visited = {start}
        obstacle_set = set(obstacles)
        
        while q:
            x, y = q.popleft()
            if (x, y) == end:
                return True
            
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.GRID_SIZE[0] and 0 <= ny < self.GRID_SIZE[1] and
                        (nx, ny) not in obstacle_set and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1  # Cost for taking a step
        self.steps += 1
        self.moves_left -= 1
        
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if dx != 0 or dy != 0:
            next_pos = (self.gem_pos[0] + dx, self.gem_pos[1] + dy)
            
            # Check boundaries and obstacles
            if (0 <= next_pos[0] < self.GRID_SIZE[0] and
                0 <= next_pos[1] < self.GRID_SIZE[1] and
                next_pos not in self.obstacles):
                self.gem_pos = next_pos
                # sound placeholder: # sfx_move.play()
        
        terminated = False
        
        # Check for gem collection
        if self.gem_pos == self.target_pos:
            self.gems_collected += 1
            gem_reward = 10.0
            reward += gem_reward
            self.score += gem_reward
            # sound placeholder: # sfx_collect_gem.play()

            if self.gems_collected >= self.TOTAL_GEMS_TO_COLLECT:
                # Win condition
                win_reward = 100.0
                reward += win_reward
                self.score += win_reward
                self.game_over = True
                terminated = True
                self.message = "YOU WIN!"
                # sound placeholder: # sfx_win_game.play()
            else:
                self._setup_level()
        
        # Check for out of moves
        if self.moves_left <= 0 and not self.game_over:
            loss_penalty = -50.0
            reward += loss_penalty
            self.score += loss_penalty
            self.game_over = True
            terminated = True
            self.message = "OUT OF MOVES"
            # sound placeholder: # sfx_lose_game.play()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, surface, color, x, y, height_offset=0, outline_color=None, outline_width=2):
        sx, sy = self._iso_to_screen(x, y)
        sy -= height_offset
        points = [
            (sx, sy),
            (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
            (sx, sy + self.TILE_HEIGHT),
            (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF)
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)
        if outline_color:
            pygame.draw.aalines(surface, outline_color, True, points, True)


    def _draw_iso_block(self, surface, top_color, side_color, x, y, height):
        sx, sy = self._iso_to_screen(x, y)
        
        top_points = [
            (sx, sy - height),
            (sx + self.TILE_WIDTH_HALF, sy - height + self.TILE_HEIGHT_HALF),
            (sx, sy - height + self.TILE_HEIGHT),
            (sx - self.TILE_WIDTH_HALF, sy - height + self.TILE_HEIGHT_HALF)
        ]
        
        side_1_points = [
            (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
            (sx, sy + self.TILE_HEIGHT),
            (sx, sy - height + self.TILE_HEIGHT),
            (sx - self.TILE_WIDTH_HALF, sy - height + self.TILE_HEIGHT_HALF)
        ]
        
        side_2_points = [
            (sx, sy + self.TILE_HEIGHT),
            (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
            (sx + self.TILE_WIDTH_HALF, sy - height + self.TILE_HEIGHT_HALF),
            (sx, sy - height + self.TILE_HEIGHT)
        ]

        pygame.gfxdraw.filled_polygon(surface, side_1_points, side_color)
        pygame.gfxdraw.filled_polygon(surface, side_2_points, side_color)
        pygame.gfxdraw.aapolygon(surface, side_1_points, side_color)
        pygame.gfxdraw.aapolygon(surface, side_2_points, side_color)
        
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)


    def _render_game(self):
        # Render grid
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0]):
                self._draw_iso_tile(self.screen, self.COLOR_GRID, x, y)
        
        # Render target
        self._draw_iso_tile(self.screen, self.COLOR_TARGET, self.target_pos[0], self.target_pos[1], outline_color=self.current_gem_color)
        
        # Render obstacles
        for ox, oy in self.obstacles:
            self._draw_iso_block(self.screen, self.COLOR_OBSTACLE_TOP, self.COLOR_OBSTACLE_SIDE, ox, oy, self.TILE_HEIGHT)

        # Render gem
        gem_height = int(self.TILE_HEIGHT * 0.2 + abs(math.sin(self.steps * 0.2)) * 5)
        self._draw_iso_block(self.screen, self.current_gem_color, tuple(c*0.7 for c in self.current_gem_color), self.gem_pos[0], self.gem_pos[1], gem_height)

    def _render_ui(self):
        # Gems collected
        gem_text = f"Gems: {self.gems_collected} / {self.TOTAL_GEMS_TO_COLLECT}"
        gem_surf = self.font_ui.render(gem_text, True, self.COLOR_TEXT)
        self.screen.blit(gem_surf, (15, 15))

        # Moves left
        moves_text = f"Moves Left: {self.moves_left}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (self.screen.get_width() - moves_surf.get_width() - 15, 15))

        # Score
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 40))

        # Game over message
        if self.game_over:
            msg_surf = self.font_msg.render(self.message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2))
            
            # Draw a semi-transparent background for the message
            bg_rect = msg_rect.inflate(40, 40)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))
            self.screen.blit(bg_surf, bg_rect.topleft)
            
            self.screen.blit(msg_surf, msg_rect)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # for testing and demonstration purposes.
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
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
                
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                    
                if event.key == pygame.K_r: # Press 'r' to reset
                    obs, info = env.reset()
                    done = False
                
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not done and action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Pygame rendering ---
        # The environment's observation is already a rendered frame.
        # We just need to display it.
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # Create a display if one doesn't exist
        try:
            display_surface = pygame.display.get_surface()
            if display_surface is None:
                raise Exception
        except Exception:
            display_surface = pygame.display.set_mode(render_surface.get_size())
            pygame.display.set_caption("Gem Collector")

        display_surface.blit(render_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS

    env.close()