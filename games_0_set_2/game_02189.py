
# Generated: 2025-08-28T04:00:30.916552
# Source Brief: brief_02189.md
# Brief Index: 2189

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric puzzle game where the player must collect all crystals within a limited number of moves.

    The game is presented in a fixed isometric view. The player navigates a cavern grid,
    and each move consumes one of the limited moves available. The goal is to find an
    optimal path to collect all 25 crystals before running out of 20 moves.

    **State:**
    - Player's grid position (x, y).
    - A list of active crystal grid positions.
    - A list of collected crystal grid positions.
    - The number of moves remaining.

    **Actions:**
    - The `MultiDiscrete([5, 2, 2])` action space is used.
    - `action[0]` controls movement (0: No-op, 1: Up, 2: Down, 3: Left, 4: Right).
    - `action[1]` (Space) and `action[2]` (Shift) have no effect in this game.

    **Rewards:**
    - +1 for each crystal collected.
    - +100 bonus for collecting all 25 crystals (winning the game).
    - 0 for all other actions.

    **Termination:**
    - The episode ends when the player collects all 25 crystals (win).
    - The episode ends when the player runs out of moves (20 moves used) (loss).
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character one tile at a time. "
        "Collect all the yellow crystals before you run out of moves."
    )

    game_description = (
        "A turn-based puzzle game. Navigate an isometric cavern to collect all 25 crystals "
        "within a strict 20-move limit. Plan your path carefully to succeed!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 12
        self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF = 28, 14
        self.TILE_Z_HEIGHT = 20

        self.MAX_MOVES = 20
        self.NUM_CRYSTALS = 25

        # --- Colors ---
        self.COLOR_BG = (26, 28, 44)
        self.COLOR_FLOOR = (75, 77, 89)
        self.COLOR_WALL_TOP = (51, 53, 65)
        self.COLOR_WALL_SIDE = (45, 47, 58)
        self.COLOR_PLAYER_TOP = (110, 162, 247)
        self.COLOR_PLAYER_SIDE = (66, 135, 245)
        self.COLOR_CRYSTAL_TOP = (247, 232, 115)
        self.COLOR_CRYSTAL_SIDE = (245, 221, 66)
        self.COLOR_COLLECTED_CRYSTAL = (120, 120, 120, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # --- Game Layout ---
        self._define_layout()

        # --- State Variables (initialized in reset) ---
        self.player_pos = None
        self.crystals = None
        self.collected_crystals = None
        self.moves_left = None
        self.score = None
        self.game_over = None
        self.win = None

        self.reset()
        
        # This check is for development and ensures the implementation matches the spec
        # self.validate_implementation()

    def _define_layout(self):
        """Defines the static layout of the cavern."""
        layout_str = [
            "################",
            "#..............#",
            "#.############.#",
            "#.#..........#.#",
            "#.#.########.#.#",
            "#.#.#......#.#.#",
            "#...#......#...#",
            "#.#.#......#.#.#",
            "#.#.########.#.#",
            "#.#..........#.#",
            "#.############.#",
            "################",
        ]
        self.cavern_layout = [[(1 if char == '#' else 0) for char in row] for row in layout_str]
        self.floor_tiles = []
        for r, row in enumerate(self.cavern_layout):
            for c, tile in enumerate(row):
                if tile == 0:
                    self.floor_tiles.append((c, r))
        self.player_start_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT - 3]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = list(self.player_start_pos)
        
        # Place crystals randomly on valid floor tiles
        crystal_indices = self.np_random.choice(
            len(self.floor_tiles), self.NUM_CRYSTALS, replace=False
        )
        self.crystals = {tuple(self.floor_tiles[i]) for i in crystal_indices if tuple(self.floor_tiles[i]) != tuple(self.player_pos)}
        # Ensure exactly NUM_CRYSTALS, handling case where player start pos was chosen
        while len(self.crystals) < self.NUM_CRYSTALS:
            new_idx = self.np_random.choice(len(self.floor_tiles))
            new_pos = tuple(self.floor_tiles[new_idx])
            if new_pos != tuple(self.player_pos) and new_pos not in self.crystals:
                self.crystals.add(new_pos)

        self.collected_crystals = set()
        
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.win = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # Only consume a move if a move action is taken
        if movement != 0:
            self.moves_left -= 1
            
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1  # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1  # Right
            
            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            
            # Wall collision check
            if (0 <= new_pos[0] < self.GRID_WIDTH and
                0 <= new_pos[1] < self.GRID_HEIGHT and
                self.cavern_layout[new_pos[1]][new_pos[0]] == 0):
                self.player_pos = new_pos
                # Sound effect placeholder: play_move_sound()

        # Crystal collection check
        player_pos_tuple = tuple(self.player_pos)
        if player_pos_tuple in self.crystals:
            self.crystals.remove(player_pos_tuple)
            self.collected_crystals.add(player_pos_tuple)
            self.score += 1
            reward += 1
            # Sound effect placeholder: play_crystal_collect_sound()

        # Termination check
        terminated = False
        if len(self.collected_crystals) == self.NUM_CRYSTALS:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100  # Win bonus
            # Sound effect placeholder: play_win_jingle()
        elif self.moves_left <= 0:
            self.game_over = True
            terminated = True
            # Sound effect placeholder: play_lose_sound()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _iso_to_screen(self, x, y):
        """Converts isometric grid coordinates to screen pixel coordinates."""
        screen_x = (self.SCREEN_WIDTH / 2) + (x - y) * self.TILE_WIDTH_HALF
        screen_y = 60 + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, x, y, top_color, side_color):
        """Draws a 3D isometric cube at a grid position."""
        sx, sy = self._iso_to_screen(x, y)
        
        # Points for the cube
        p_top_center = (sx, sy - self.TILE_Z_HEIGHT)
        p_top_left = (sx - self.TILE_WIDTH_HALF, sy - self.TILE_Z_HEIGHT + self.TILE_HEIGHT_HALF)
        p_top_right = (sx + self.TILE_WIDTH_HALF, sy - self.TILE_Z_HEIGHT + self.TILE_HEIGHT_HALF)
        p_top_front = (sx, sy - self.TILE_Z_HEIGHT + self.TILE_HEIGHT_HALF * 2)
        
        p_bottom_left = (p_top_left[0], p_top_left[1] + self.TILE_Z_HEIGHT)
        p_bottom_right = (p_top_right[0], p_top_right[1] + self.TILE_Z_HEIGHT)
        p_bottom_front = (p_top_front[0], p_top_front[1] + self.TILE_Z_HEIGHT)

        # Draw faces
        # Top face
        top_points = [p_top_center, p_top_right, p_top_front, p_top_left]
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)

        # Left face
        left_points = [p_top_left, p_top_front, p_bottom_front, p_bottom_left]
        pygame.gfxdraw.filled_polygon(surface, left_points, side_color)
        pygame.gfxdraw.aapolygon(surface, left_points, side_color)

        # Right face
        right_points = [p_top_right, p_top_front, p_bottom_front, p_bottom_right]
        pygame.gfxdraw.filled_polygon(surface, right_points, side_color)
        pygame.gfxdraw.aapolygon(surface, right_points, side_color)

    def _draw_iso_diamond(self, surface, x, y, top_color, side_color):
        """Draws a floating isometric diamond (crystal)."""
        sx, sy = self._iso_to_screen(x, y)
        sy -= 10 # Float above ground

        points = [
            (sx, sy - self.TILE_HEIGHT_HALF), # Top
            (sx + self.TILE_WIDTH_HALF / 2, sy), # Right
            (sx, sy + self.TILE_HEIGHT_HALF), # Bottom
            (sx - self.TILE_WIDTH_HALF / 2, sy), # Left
        ]
        
        # Draw two halves for a 3D effect
        pygame.gfxdraw.filled_polygon(surface, [points[0], points[1], points[2]], side_color)
        pygame.gfxdraw.filled_polygon(surface, [points[0], points[3], points[2]], top_color)
        pygame.gfxdraw.aapolygon(surface, points, side_color)

    def _render_game(self):
        """Renders all game elements onto the screen."""
        # Render tiles, crystals, and player in correct Z-order
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                sx, sy = self._iso_to_screen(x, y)
                floor_points = [
                    (sx, sy),
                    (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
                    (sx, sy + self.TILE_HEIGHT_HALF * 2),
                    (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
                ]

                if self.cavern_layout[y][x] == 1: # Wall
                    self._draw_iso_cube(self.screen, x, y, self.COLOR_WALL_TOP, self.COLOR_WALL_SIDE)
                else: # Floor
                    pygame.gfxdraw.filled_polygon(self.screen, floor_points, self.COLOR_FLOOR)
                    # pygame.gfxdraw.aapolygon(self.screen, floor_points, self.COLOR_WALL_SIDE)

                # Draw collected crystals (faded)
                if (x, y) in self.collected_crystals:
                    s = pygame.Surface((self.TILE_WIDTH_HALF*2, self.TILE_HEIGHT_HALF*2), pygame.SRCALPHA)
                    sx_c, sy_c = self._iso_to_screen(x, y)
                    diamond_points = [
                        (self.TILE_WIDTH_HALF, 0),
                        (self.TILE_WIDTH_HALF*1.5, self.TILE_HEIGHT_HALF),
                        (self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF*2),
                        (self.TILE_WIDTH_HALF*0.5, self.TILE_HEIGHT_HALF),
                    ]
                    pygame.gfxdraw.filled_polygon(s, diamond_points, self.COLOR_COLLECTED_CRYSTAL)
                    self.screen.blit(s, (sx_c - self.TILE_WIDTH_HALF, sy_c - self.TILE_HEIGHT_HALF - 10))

                # Draw active crystals
                if (x, y) in self.crystals:
                    self._draw_iso_diamond(self.screen, x, y, self.COLOR_CRYSTAL_TOP, self.COLOR_CRYSTAL_SIDE)
                
                # Draw player
                if self.player_pos == [x, y]:
                    self._draw_iso_cube(self.screen, x, y, self.COLOR_PLAYER_TOP, self.COLOR_PLAYER_SIDE)

    def _render_ui(self):
        """Renders the UI text overlay."""
        # Score and Moves
        score_text = f"Crystals: {self.score} / {self.NUM_CRYSTALS}"
        moves_text = f"Moves Left: {self.moves_left}"
        
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(moves_surf, (10, 40))

        # Game Over Message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "OUT OF MOVES"
                color = self.COLOR_LOSE
            
            game_over_surf = self.font_game_over.render(msg, True, color)
            pos = game_over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Add a semi-transparent background for readability
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(game_over_surf, pos)

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
            "crystals_collected": len(self.collected_crystals),
            "steps": self.MAX_MOVES - self.moves_left,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
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
        assert len(self.crystals) == self.NUM_CRYSTALS
        assert self.moves_left == self.MAX_MOVES
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        if test_action[0] != 0:
            assert info['moves_left'] == self.MAX_MOVES - 1
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Isometric Crystal Collector")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default to no-op
        
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
                    env.reset()
                    action[0] = 0 # No move on reset
                elif event.key == pygame.K_ESCAPE:
                    running = False

                # Only step if a key was pressed
                if action[0] != 0 or event.key == pygame.K_r:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
                    if terminated:
                        print("Game Over. Press 'R' to reset.")

        # Always render the current state
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

    env.close()