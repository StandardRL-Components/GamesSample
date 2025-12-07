
# Generated: 2025-08-28T01:41:25.156839
# Source Brief: brief_04192.md
# Brief Index: 4192

        
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
        "Controls: Use arrow keys to shift the row/column your character is on. "
        "Your goal is to create a path to the red goal tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based isometric puzzle. Shift tiles to connect a path from your "
        "character to the goal before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 5, 4
        self.MAX_MOVES_PER_LEVEL = 10
        self.NUM_LEVELS = 3

        # Exact spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_TILE = (70, 80, 90)
        self.COLOR_TILE_SIDE = (50, 60, 70)
        self.COLOR_GRID_LINE = (40, 50, 60)
        self.COLOR_PATH = (60, 180, 255)
        self.COLOR_PATH_SIDE = (40, 120, 190)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_GOAL = (255, 60, 60)
        self.COLOR_GOAL_SIDE = (190, 40, 40)
        self.COLOR_START = (60, 255, 60)
        self.COLOR_START_SIDE = (40, 190, 40)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # Isometric projection constants
        self.TILE_WIDTH_HALF = 32
        self.TILE_HEIGHT_HALF = 16
        self.TILE_DEPTH = 10
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 120

        # Precompute tile and level data
        self._define_tiles()
        self._define_levels()

        # Initialize state variables
        self.grid = None
        self.player_pos = None
        self.goal_pos = None
        self.start_pos = None
        self.path = []
        self.moves_left = 0
        self.current_level = 0
        self.total_score = 0
        self.steps = 0
        self.game_over = False
        self.best_path_len = {}

        self.reset()
        
        # Run validation check
        try:
            self.validate_implementation()
        except AssertionError as e:
            print(f"Validation failed: {e}")


    def _define_tiles(self):
        """Defines tile connections. 'n', 's', 'e', 'w'."""
        self.connections = {
            1: {'n', 's'},  # Vertical
            2: {'e', 'w'},  # Horizontal
            3: {'n', 'e'},  # Corner NE
            4: {'s', 'e'},  # Corner SE
            5: {'s', 'w'},  # Corner SW
            6: {'n', 'w'},  # Corner NW
        }
        self.opposites = {'n': 's', 's': 'n', 'e': 'w', 'w': 'e'}

    def _define_levels(self):
        """Defines the grid layout, start, and goal for each level."""
        self.level_layouts = [
            # Level 1: Simple shift
            np.array([
                [2, 3, 5, 2],
                [1, 6, 4, 1],
                [2, 3, 5, 2],
                [1, 6, 4, 1],
                [2, 3, 5, 2],
            ]),
            # Level 2: Two shifts required
            np.array([
                [6, 4, 6, 4],
                [3, 5, 3, 5],
                [1, 2, 1, 2],
                [2, 1, 2, 1],
                [6, 4, 6, 4],
            ]),
            # Level 3: Misleading paths
            np.array([
                [4, 1, 3, 5],
                [2, 5, 6, 2],
                [3, 2, 4, 1],
                [1, 6, 3, 5],
                [5, 2, 6, 4],
            ]),
        ]
        self.level_starts = [(0, 0), (2, 1), (0, 3)]
        self.level_goals = [(4, 3), (2, 2), (4, 0)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_score = 0
        self.steps = 0
        self.game_over = False
        self.current_level = 0
        self.best_path_len = {i: 999 for i in range(self.NUM_LEVELS)}
        self._load_level(self.current_level)
        return self._get_observation(), self._get_info()

    def _load_level(self, level_idx):
        """Initializes state for a specific level."""
        self.grid = self.level_layouts[level_idx].copy()
        self.start_pos = self.level_starts[level_idx]
        self.player_pos = self.start_pos
        self.goal_pos = self.level_goals[level_idx]
        self.moves_left = self.MAX_MOVES_PER_LEVEL
        self._find_path()
        # Sound effect placeholder:
        # play_sound('level_start')

    def step(self, action):
        reward = 0
        terminated = self.game_over

        if not terminated:
            movement = action[0]
            self.steps += 1

            if movement != 0:
                self.moves_left -= 1
                reward -= 0.1  # Cost for making a move
                
                player_r, player_c = self.player_pos
                
                # Apply shift based on action
                if movement == 1: # Up
                    self.grid[:, player_c] = np.roll(self.grid[:, player_c], -1)
                elif movement == 2: # Down
                    self.grid[:, player_c] = np.roll(self.grid[:, player_c], 1)
                elif movement == 3: # Left
                    self.grid[player_r, :] = np.roll(self.grid[player_r, :], -1)
                elif movement == 4: # Right
                    self.grid[player_r, :] = np.roll(self.grid[player_r, :], 1)

                # Sound effect placeholder:
                # play_sound('tile_shift')

                # After shifting, re-evaluate the path and player position
                self._find_path()

                # Reward for finding a shorter path
                path_len_to_goal = self._get_path_len_to_goal()
                if path_len_to_goal != -1 and path_len_to_goal < self.best_path_len[self.current_level]:
                    reward += 1.0
                    self.best_path_len[self.current_level] = path_len_to_goal

            # Check for win condition
            if self.player_pos == self.goal_pos:
                reward += 100
                self.total_score += 1 # A "win" point
                self.current_level += 1
                # Sound effect placeholder:
                # play_sound('level_complete')

                if self.current_level >= self.NUM_LEVELS:
                    terminated = True
                    self.game_over = True
                    # Sound effect placeholder:
                    # play_sound('game_win')
                else:
                    self._load_level(self.current_level)

            # Check for loss condition
            elif self.moves_left <= 0:
                reward -= 10
                terminated = True
                self.game_over = True
                # Sound effect placeholder:
                # play_sound('game_over')
        
        # Max steps termination
        if self.steps >= 1000:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _find_path(self):
        """Follows the path from the current player pos until it ends."""
        self.path = [self.start_pos]
        curr_pos = self.start_pos
        
        # The pathfinding needs to start from the fixed start tile, not the player
        # as the player moves along the path.
        
        visited = {self.start_pos}
        
        for _ in range(self.GRID_ROWS * self.GRID_COLS): # Safety break
            r, c = curr_pos
            tile_id = self.grid[r, c]
            
            if tile_id not in self.connections:
                break # Path ends
                
            possible_moves = self.connections[tile_id]
            found_next = False
            for move_dir in possible_moves:
                dr, dc = {'n': (-1, 0), 's': (1, 0), 'w': (0, -1), 'e': (0, 1)}[move_dir]
                next_pos = (r + dr, c + dc)
                
                if next_pos in visited:
                    continue

                # Check if next tile connects back
                nr, nc = next_pos
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                    next_tile_id = self.grid[nr, nc]
                    if next_tile_id in self.connections and self.opposites[move_dir] in self.connections[next_tile_id]:
                        self.path.append(next_pos)
                        visited.add(next_pos)
                        curr_pos = next_pos
                        found_next = True
                        break # Found the one exit
            
            if not found_next:
                break # Dead end

        self.player_pos = self.path[-1]

    def _get_path_len_to_goal(self):
        """Returns the number of steps along the current path to the goal, or -1 if not on path."""
        try:
            return self.path.index(self.goal_pos)
        except ValueError:
            return -1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.total_score,
            "steps": self.steps,
            "level": self.current_level,
            "moves_left": self.moves_left,
        }
        
    def _world_to_iso(self, r, c):
        """Converts grid coordinates to isometric screen coordinates."""
        iso_x = self.ORIGIN_X + (c - r) * self.TILE_WIDTH_HALF
        iso_y = self.ORIGIN_Y + (c + r) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _draw_iso_poly(self, points, color, outline_color=None):
        """Draws an antialiased, filled polygon."""
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        if outline_color:
            pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

    def _render_game(self):
        """Renders the main game grid, tiles, and character."""
        # Draw grid base
        for r in range(self.GRID_ROWS + 1):
            start_pos = self._world_to_iso(r, 0)
            end_pos = self._world_to_iso(r, self.GRID_COLS)
            pygame.draw.aaline(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos)
        for c in range(self.GRID_COLS + 1):
            start_pos = self._world_to_iso(0, c)
            end_pos = self._world_to_iso(self.GRID_ROWS, c)
            pygame.draw.aaline(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos)

        # Draw tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                iso_x, iso_y = self._world_to_iso(r, c)
                
                top_points = [
                    (iso_x, iso_y - self.TILE_HEIGHT_HALF),
                    (iso_x + self.TILE_WIDTH_HALF, iso_y),
                    (iso_x, iso_y + self.TILE_HEIGHT_HALF),
                    (iso_x - self.TILE_WIDTH_HALF, iso_y),
                ]
                
                side_l_points = [
                    (iso_x - self.TILE_WIDTH_HALF, iso_y),
                    (iso_x, iso_y + self.TILE_HEIGHT_HALF),
                    (iso_x, iso_y + self.TILE_HEIGHT_HALF + self.TILE_DEPTH),
                    (iso_x - self.TILE_WIDTH_HALF, iso_y + self.TILE_DEPTH),
                ]

                side_r_points = [
                    (iso_x + self.TILE_WIDTH_HALF, iso_y),
                    (iso_x, iso_y + self.TILE_HEIGHT_HALF),
                    (iso_x, iso_y + self.TILE_HEIGHT_HALF + self.TILE_DEPTH),
                    (iso_x + self.TILE_WIDTH_HALF, iso_y + self.TILE_DEPTH),
                ]

                is_path = (r, c) in self.path
                is_goal = (r, c) == self.goal_pos
                is_start = (r, c) == self.start_pos

                top_color = self.COLOR_TILE
                side_color = self.COLOR_TILE_SIDE
                if is_path:
                    top_color = self.COLOR_PATH
                    side_color = self.COLOR_PATH_SIDE
                if is_start:
                    top_color = self.COLOR_START
                    side_color = self.COLOR_START_SIDE
                if is_goal:
                    top_color = self.COLOR_GOAL
                    side_color = self.COLOR_GOAL_SIDE
                
                self._draw_iso_poly(side_l_points, side_color)
                self._draw_iso_poly(side_r_points, side_color)
                self._draw_iso_poly(top_points, top_color, self.COLOR_GRID_LINE)
                
                # Draw tile connection graphics
                tile_id = self.grid[r, c]
                if tile_id in self.connections:
                    for direction in self.connections[tile_id]:
                        if direction == 'n':
                            pygame.draw.line(self.screen, side_color, (iso_x, iso_y - self.TILE_HEIGHT_HALF), (iso_x, iso_y), 2)
                        elif direction == 's':
                            pygame.draw.line(self.screen, side_color, (iso_x, iso_y), (iso_x, iso_y + self.TILE_HEIGHT_HALF), 2)
                        elif direction == 'w':
                            pygame.draw.line(self.screen, side_color, (iso_x - self.TILE_WIDTH_HALF, iso_y), (iso_x, iso_y), 2)
                        elif direction == 'e':
                            pygame.draw.line(self.screen, side_color, (iso_x, iso_y), (iso_x + self.TILE_WIDTH_HALF, iso_y), 2)


        # Draw player
        player_r, player_c = self.player_pos
        p_iso_x, p_iso_y = self._world_to_iso(player_r, player_c)
        player_rect = pygame.Rect(0, 0, 16, 16)
        player_rect.center = (p_iso_x, p_iso_y - self.TILE_HEIGHT_HALF)
        pygame.draw.ellipse(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.ellipse(self.screen, (255, 255, 255), player_rect, 2)


    def _render_ui(self):
        """Renders UI text like score and moves left."""
        ui_surface = pygame.Surface((self.WIDTH, 60), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surface, (0, self.HEIGHT - 60))

        level_text = self.font_large.render(f"Level: {self.current_level + 1}/{self.NUM_LEVELS}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (20, self.HEIGHT - 45))

        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(right=self.WIDTH - 20, top=self.HEIGHT - 45)
        self.screen.blit(moves_text, moves_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.player_pos == self.goal_pos and self.current_level >= self.NUM_LEVELS:
                end_text = "You Win!"
            else:
                end_text = "Game Over"
            
            end_surf = self.font_large.render(end_text, True, self.COLOR_TEXT)
            end_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_surf, end_rect)


    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To display the game, we need a Pygame window
    pygame.display.set_caption("Isometric Tile Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
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
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    action[0] = 0 # No-op for the reset frame
                
                # Only step if a key was pressed (turn-based)
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
                    if terminated:
                        print("--- GAME OVER ---")
                        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we only step on input.
        # The loop will spin without a clock tick, so let's add one.
        env.clock.tick(30)

    pygame.quit()