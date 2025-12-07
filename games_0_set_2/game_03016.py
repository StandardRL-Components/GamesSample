
# Generated: 2025-08-27T22:07:13.388350
# Source Brief: brief_03016.md
# Brief Index: 3016

        
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
    """
    An isometric puzzle game where the player drops colored crystals into a cavern.
    The goal is to create large connected groups of the same color to score points.
    A move is consumed each time a crystal is dropped. The game ends when a
    winning cluster is formed or the player runs out of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the falling crystal. Press space to drop it. "
        "Create connected groups of same-colored crystals to score points."
    )

    game_description = (
        "An isometric crystal-matching puzzle game. Strategically drop crystals "
        "to form large color groups and achieve a high score before you run out of moves."
    )

    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 28, 44)
    COLOR_GRID = (50, 55, 75)
    COLOR_WHITE = (240, 240, 240)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    CRYSTAL_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]

    # Grid and Rendering
    GRID_WIDTH = 8
    GRID_HEIGHT = 10
    TILE_WIDTH_ISO = 64
    TILE_HEIGHT_ISO = 32
    CRYSTAL_HEIGHT = 48
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Mechanics
    TOTAL_MOVES = 20 # Increased from 10 for more gameplay
    WIN_CLUSTER_SIZE = 10
    REWARD_CLUSTER_BONUS_THRESHOLD = 5

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        self.grid_origin_x = self.SCREEN_WIDTH // 2
        self.grid_origin_y = 80

        # State variables are initialized in reset()
        self.grid = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.win = None
        self.falling_crystal = None
        self.steps = 0
        self.last_drop_info = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), -1, dtype=int)
        self.score = 0
        self.moves_left = self.TOTAL_MOVES
        self.game_over = False
        self.win = False
        self.steps = 0
        self.last_drop_info = None
        
        self._spawn_crystal()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing and just return the final state
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()
            
        self.steps += 1
        self.last_drop_info = None # Clear landing effects
        reward = 0
        terminated = False

        movement = action[0]
        space_pressed = action[1] == 1

        if space_pressed:
            # Drop the crystal
            drop_x = self.falling_crystal['x']
            drop_y = self._find_drop_row(drop_x)

            if drop_y is not None:
                # Sound effect placeholder: # sfx_crystal_land()
                self.grid[drop_y, drop_x] = self.falling_crystal['color_index']
                self.moves_left -= 1
                
                # Store drop info for visual effects
                self.last_drop_info = {'x': drop_x, 'y': drop_y, 'frame': self.steps}

                # Check for connections and calculate score/reward
                cluster = self._find_connected_cluster(drop_x, drop_y)
                cluster_size = len(cluster)
                
                self.score += cluster_size * cluster_size # Exponential score for bigger clusters

                if cluster_size == 1:
                    reward = -0.2
                else:
                    reward = cluster_size
                
                if cluster_size >= self.REWARD_CLUSTER_BONUS_THRESHOLD:
                    reward += 10
                
                if cluster_size >= self.WIN_CLUSTER_SIZE:
                    self.win = True
                    reward += 100
                    
                if self.win or self.moves_left == 0:
                    terminated = True
                    self.game_over = True
                    if not self.win:
                        reward -= 10 # Penalty for losing

                if not self.game_over:
                    self._spawn_crystal()
            else:
                # Column is full, invalid move
                reward = -1.0

        else: # Not dropping, just moving
            if movement == 3:  # Left
                self.falling_crystal['x'] = max(0, self.falling_crystal['x'] - 1)
            elif movement == 4:  # Right
                self.falling_crystal['x'] = min(self.GRID_WIDTH - 1, self.falling_crystal['x'] + 1)
            # No reward or penalty for just moving horizontally
            reward = 0.0
            
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

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "steps": self.steps,
            "win": self.win,
        }

    def close(self):
        pygame.quit()

    # --- Helper Methods ---

    def _spawn_crystal(self):
        self.falling_crystal = {
            'x': self.GRID_WIDTH // 2,
            'color_index': self.np_random.integers(0, len(self.CRYSTAL_COLORS)),
        }

    def _find_drop_row(self, x):
        for y in range(self.GRID_HEIGHT - 1, -1, -1):
            if self.grid[y, x] == -1:
                return y
        return None # Column is full

    def _find_connected_cluster(self, start_x, start_y):
        if self.grid[start_y, start_x] == -1:
            return set()

        target_color = self.grid[start_y, start_x]
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        
        while q:
            x, y = q.popleft()
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 4-way connection
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[ny, nx] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return visited

    def _iso_to_screen(self, x, y):
        screen_x = self.grid_origin_x + (x - y) * (self.TILE_WIDTH_ISO / 2)
        screen_y = self.grid_origin_y + (x + y) * (self.TILE_HEIGHT_ISO / 2)
        return int(screen_x), int(screen_y)

    def _render_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    # --- Rendering Methods ---

    def _render_game(self):
        # Render grid cells
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                sx, sy = self._iso_to_screen(x, y)
                points = [
                    (sx, sy),
                    (sx + self.TILE_WIDTH_ISO / 2, sy + self.TILE_HEIGHT_ISO / 2),
                    (sx, sy + self.TILE_HEIGHT_ISO),
                    (sx - self.TILE_WIDTH_ISO / 2, sy + self.TILE_HEIGHT_ISO / 2)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Render placed crystals
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != -1:
                    sx, sy = self._iso_to_screen(x, y)
                    color = self.CRYSTAL_COLORS[self.grid[y, x]]
                    self._draw_crystal(self.screen, sx, sy, color)
        
        # Render landing effect
        if self.last_drop_info:
            lx, ly = self.last_drop_info['x'], self.last_drop_info['y']
            sx, sy = self._iso_to_screen(lx, ly)
            radius = 20
            alpha = 100
            # Sound effect placeholder: # sfx_landing_sparkle()
            circle_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, (*self.COLOR_WHITE, alpha), (radius, radius), radius)
            self.screen.blit(circle_surf, (sx - radius, sy + self.TILE_HEIGHT_ISO/2 - radius))


        if not self.game_over:
            # Render falling crystal and its preview
            fc_x = self.falling_crystal['x']
            fc_color_idx = self.falling_crystal['color_index']
            fc_color = self.CRYSTAL_COLORS[fc_color_idx]
            
            # Bobbing animation for falling crystal
            bob_offset = math.sin(self.steps * 0.2) * 5
            
            # Position above the grid
            fall_sx, fall_sy = self._iso_to_screen(fc_x, -2)
            self._draw_crystal(self.screen, fall_sx, fall_sy + bob_offset, fc_color)
            
            # Render drop preview/shadow
            drop_y = self._find_drop_row(fc_x)
            if drop_y is not None:
                psx, psy = self._iso_to_screen(fc_x, drop_y)
                points = [
                    (psx, psy),
                    (psx + self.TILE_WIDTH_ISO / 2, psy + self.TILE_HEIGHT_ISO / 2),
                    (psx, psy + self.TILE_HEIGHT_ISO),
                    (psx - self.TILE_WIDTH_ISO / 2, psy + self.TILE_HEIGHT_ISO / 2)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, (*self.COLOR_WHITE, 100))
                pygame.gfxdraw.filled_polygon(self.screen, points, (*self.COLOR_WHITE, 50))


    def _draw_crystal(self, surface, sx, sy, color):
        top_color = tuple(min(255, c + 40) for c in color)
        right_color = color
        left_color = tuple(max(0, c - 40) for c in color)

        # Base of crystal is at sy
        crystal_base_y = sy + self.TILE_HEIGHT_ISO / 2
        
        # Top face
        top_points = [
            (sx, crystal_base_y - self.CRYSTAL_HEIGHT),
            (sx + self.TILE_WIDTH_ISO / 2, crystal_base_y - self.CRYSTAL_HEIGHT + self.TILE_HEIGHT_ISO / 2),
            (sx, crystal_base_y - self.CRYSTAL_HEIGHT + self.TILE_HEIGHT_ISO),
            (sx - self.TILE_WIDTH_ISO / 2, crystal_base_y - self.CRYSTAL_HEIGHT + self.TILE_HEIGHT_ISO / 2)
        ]
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)

        # Left face
        left_points = [
            (sx - self.TILE_WIDTH_ISO / 2, crystal_base_y - self.CRYSTAL_HEIGHT + self.TILE_HEIGHT_ISO / 2),
            (sx, crystal_base_y - self.CRYSTAL_HEIGHT + self.TILE_HEIGHT_ISO),
            (sx, crystal_base_y + self.TILE_HEIGHT_ISO),
            (sx - self.TILE_WIDTH_ISO / 2, crystal_base_y + self.TILE_HEIGHT_ISO / 2)
        ]
        pygame.gfxdraw.filled_polygon(surface, left_points, left_color)
        pygame.gfxdraw.aapolygon(surface, left_points, left_color)

        # Right face
        right_points = [
            (sx + self.TILE_WIDTH_ISO / 2, crystal_base_y - self.CRYSTAL_HEIGHT + self.TILE_HEIGHT_ISO / 2),
            (sx, crystal_base_y - self.CRYSTAL_HEIGHT + self.TILE_HEIGHT_ISO),
            (sx, crystal_base_y + self.TILE_HEIGHT_ISO),
            (sx + self.TILE_WIDTH_ISO / 2, crystal_base_y + self.TILE_HEIGHT_ISO / 2)
        ]
        pygame.gfxdraw.filled_polygon(surface, right_points, right_color)
        pygame.gfxdraw.aapolygon(surface, right_points, right_color)


    def _render_ui(self):
        # Score display
        self._render_text(f"Score: {self.score}", self.font_main, self.COLOR_TEXT, (10, 10))
        
        # Moves left display
        moves_text = f"Moves: {self.moves_left}"
        text_width = self.font_main.size(moves_text)[0]
        self._render_text(moves_text, self.font_main, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_width - 10, 10))

        if self.game_over:
            # Overlay
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(overlay, (0, 0))

            # Win/Loss message
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.CRYSTAL_COLORS[3] if self.win else self.CRYSTAL_COLORS[0]
            # Sound effect placeholder: # sfx_win() or sfx_lose()
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            shadow_surf = self.font_large.render(message, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (text_rect.x + 3, text_rect.y + 3))
            self.screen.blit(text_surf, text_rect)


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

# Example usage for human play
if __name__ == '__main__':
    # Set this to True to run the game in a window
    HUMAN_PLAY = True

    if HUMAN_PLAY:
        # Override render_mode for human play
        env = GameEnv(render_mode="human")
        env.metadata["render_modes"] = ["human", "rgb_array"]
        pygame.display.set_caption(env.game_description)
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    else:
        env = GameEnv()

    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        if HUMAN_PLAY:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # Continuous key presses
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    action[0] = 3
                elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    action[0] = 4
                
                if keys[pygame.K_SPACE]:
                    action[1] = 1

            # Only step if an action is taken in human mode
            if not np.array_equal(action, np.array([0, 0, 0])):
                 obs, reward, terminated, truncated, info = env.step(action)
        else: # AI play
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        if HUMAN_PLAY:
            # Get the rendered surface from the environment and draw it to the display
            rendered_frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(rendered_frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30) # Limit FPS

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Win: {info['win']}")
            if HUMAN_PLAY:
                pygame.time.wait(3000) # Pause before reset
            obs, info = env.reset()

    env.close()