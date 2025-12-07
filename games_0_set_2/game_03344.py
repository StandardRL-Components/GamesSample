
# Generated: 2025-08-27T23:05:57.442419
# Source Brief: brief_03344.md
# Brief Index: 3344

        
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
        "Controls: Arrow keys to move your character (white). Push crystals to clear a path to the golden exit tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Navigate a cavern by pushing crystals to reach the exit within a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 12
    NUM_CRYSTALS = 15
    MAX_GEN_ATTEMPTS = 100

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_FLOOR = (60, 70, 80)
    COLOR_FLOOR_LINE = (80, 90, 100)
    COLOR_WALL_TOP = (40, 45, 50)
    COLOR_WALL_SIDE = (30, 35, 40)
    COLOR_PLAYER = [(255, 255, 255), (220, 220, 255), (200, 200, 235)] # Top, Right, Left
    CRYSTAL_COLORS = [
        [(50, 220, 220), (40, 180, 180), (30, 160, 160)],  # Cyan
        [(230, 50, 120), (190, 40, 100), (170, 30, 80)],   # Magenta
        [(120, 230, 50), (100, 190, 40), (80, 170, 30)],   # Lime
    ]
    COLOR_EXIT_TOP = (255, 215, 0)
    COLOR_EXIT_SIDE = (200, 160, 0)
    COLOR_TEXT = (240, 240, 240)
    
    # Isometric projection values
    TILE_WIDTH_HALF = 24
    TILE_HEIGHT_HALF = 12
    CUBE_HEIGHT = 20
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font = pygame.font.Font(None, 28)
        
        # Etc...        
        self.np_random = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.moves_remaining = 0
        self.initial_moves = 0
        
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.walls = set()
        self.crystals = {} # pos -> color_index
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self._generate_level()
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """
        Generates a new level layout, ensuring it is solvable.
        1. Places walls, player, and exit.
        2. Finds a guaranteed path from player to exit on an empty grid.
        3. Sets the move limit based on this path length.
        4. Populates the grid with crystals, avoiding the guaranteed path to ensure solvability.
        """
        for _ in range(self.MAX_GEN_ATTEMPTS):
            self.walls = set()
            self.crystals = {}

            # 1. Place perimeter walls
            for i in range(-1, self.GRID_WIDTH + 1):
                self.walls.add((i, -1))
                self.walls.add((i, self.GRID_HEIGHT))
            for i in range(self.GRID_HEIGHT):
                self.walls.add((-1, i))
                self.walls.add((self.GRID_WIDTH, i))

            # 2. Place exit and player far apart
            exit_side = self.np_random.integers(4)
            if exit_side == 0: # Top
                self.exit_pos = (self.np_random.integers(1, self.GRID_WIDTH - 1), 0)
                self.player_pos = (self.np_random.integers(1, self.GRID_WIDTH - 1), self.GRID_HEIGHT - 1)
            elif exit_side == 1: # Bottom
                self.exit_pos = (self.np_random.integers(1, self.GRID_WIDTH - 1), self.GRID_HEIGHT - 1)
                self.player_pos = (self.np_random.integers(1, self.GRID_WIDTH - 1), 0)
            elif exit_side == 2: # Left
                self.exit_pos = (0, self.np_random.integers(1, self.GRID_HEIGHT - 1))
                self.player_pos = (self.GRID_WIDTH - 1, self.np_random.integers(1, self.GRID_HEIGHT - 1))
            else: # Right
                self.exit_pos = (self.GRID_WIDTH - 1, self.np_random.integers(1, self.GRID_HEIGHT - 1))
                self.player_pos = (0, self.np_random.integers(1, self.GRID_HEIGHT - 1))
            
            # 3. Find a guaranteed path using BFS
            path_to_exit = self._find_path(self.player_pos, self.exit_pos)
            if not path_to_exit:
                continue

            # 4. Set move limit based on path length with a buffer
            self.initial_moves = int(len(path_to_exit) * 1.5) + 5
            self.moves_remaining = self.initial_moves

            # 5. Place crystals on tiles not on the direct path
            possible_crystal_locs = []
            for r in range(self.GRID_HEIGHT):
                for c in range(self.GRID_WIDTH):
                    pos = (c, r)
                    if pos != self.player_pos and pos != self.exit_pos and pos not in path_to_exit:
                        possible_crystal_locs.append(pos)
            
            self.np_random.shuffle(possible_crystal_locs)

            for i in range(min(self.NUM_CRYSTALS, len(possible_crystal_locs))):
                pos = possible_crystal_locs[i]
                color_idx = self.np_random.integers(len(self.CRYSTAL_COLORS))
                self.crystals[pos] = color_idx
            
            return
        
        raise RuntimeError("Failed to generate a solvable level after multiple attempts.")

    def _find_path(self, start, end):
        q = deque([(start, [start])])
        visited = {start}
        
        while q:
            current_pos, path = q.popleft()
            
            if current_pos == end:
                return path

            cx, cy = current_pos
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                next_pos = (cx + dx, cy + dy)
                if 0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT and next_pos not in visited:
                    visited.add(next_pos)
                    new_path = list(path)
                    new_path.append(next_pos)
                    q.append((next_pos, new_path))
        return None
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
            
        if movement != 0: # A move was attempted
            self.steps += 1
            self.moves_remaining -= 1

            old_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)

            # Determine target position
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            target_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            # Check what is at the target position
            if target_pos in self.walls:
                # Blocked by wall, do nothing. Sound: *thud*
                pass
            elif target_pos == self.exit_pos:
                # Reached exit. Sound: *level_complete_jingle*
                self.player_pos = target_pos
                self.game_over = True
            elif target_pos in self.crystals:
                # Attempt to push crystal
                crystal_pos = target_pos
                behind_crystal_pos = (crystal_pos[0] + dx, crystal_pos[1] + dy)
                
                if behind_crystal_pos in self.walls or behind_crystal_pos in self.crystals:
                    # Crystal is blocked. Sound: *clink*
                    pass
                else:
                    # Push successful. Sound: *crystal_slide*
                    old_crystal_dist = self._manhattan_distance(crystal_pos, self.exit_pos)
                    
                    crystal_color_idx = self.crystals.pop(crystal_pos)
                    self.crystals[behind_crystal_pos] = crystal_color_idx
                    self.player_pos = crystal_pos
                    
                    new_crystal_dist = self._manhattan_distance(behind_crystal_pos, self.exit_pos)
                    
                    if new_crystal_dist < old_crystal_dist:
                        reward += 5.0
                    else:
                        reward -= 1.0
            else:
                # Empty space, move player
                self.player_pos = target_pos

            # Calculate movement-based distance reward
            new_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
            if new_dist_to_exit < old_dist_to_exit:
                reward += 0.1
            elif new_dist_to_exit > old_dist_to_exit:
                reward -= 0.1
        
        # Calculate terminal rewards and check termination
        terminated = False
        if self.player_pos == self.exit_pos:
            self.game_over = True
            terminated = True
            reward += 100.0
        elif self.moves_remaining <= 0:
            self.game_over = True
            terminated = True
            reward -= 50.0

        self.score += reward
        
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
    
    def _render_game(self):
        # Render floor tiles
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                self._draw_iso_tile((c, r), self.COLOR_FLOOR, self.COLOR_FLOOR_LINE)
        
        # Render exit tile
        self._draw_iso_tile(self.exit_pos, self.COLOR_EXIT_TOP, self.COLOR_EXIT_SIDE)
        
        # Render all objects, sorted by grid position for correct occlusion
        sorted_objects = []
        for pos in self.walls:
            sorted_objects.append((pos, 'wall'))
        for pos in self.crystals:
            sorted_objects.append((pos, 'crystal'))
        sorted_objects.append((self.player_pos, 'player'))

        sorted_objects.sort(key=lambda item: (item[0][1], item[0][0]))

        for pos, obj_type in sorted_objects:
            if obj_type == 'wall':
                self._draw_iso_cube(pos, self.CUBE_HEIGHT, self.COLOR_WALL_TOP, self.COLOR_WALL_SIDE)
            elif obj_type == 'crystal':
                color_idx = self.crystals[pos]
                colors = self.CRYSTAL_COLORS[color_idx]
                self._draw_iso_cube(pos, self.CUBE_HEIGHT, colors[0], colors[1], colors[2])
            elif obj_type == 'player':
                 self._draw_iso_cube(pos, self.CUBE_HEIGHT, self.COLOR_PLAYER[0], self.COLOR_PLAYER[1], self.COLOR_PLAYER[2])

    def _render_ui(self):
        moves_text = f"Moves Left: {self.moves_remaining}"
        text_surface = self.font.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        score_text = f"Score: {self.score:.1f}"
        score_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surface, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }
        
    def _world_to_iso(self, x, y):
        iso_x = (self.WIDTH / 2) + (x - y) * self.TILE_WIDTH_HALF
        iso_y = 60 + (x + y) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _draw_iso_tile(self, pos, color_fill, color_line):
        x, y = pos
        iso_x, iso_y = self._world_to_iso(x, y)
        points = [
            (iso_x, iso_y - self.TILE_HEIGHT_HALF),
            (iso_x + self.TILE_WIDTH_HALF, iso_y),
            (iso_x, iso_y + self.TILE_HEIGHT_HALF),
            (iso_x - self.TILE_WIDTH_HALF, iso_y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color_fill)
        pygame.gfxdraw.aapolygon(self.screen, points, color_line)

    def _draw_iso_cube(self, pos, height, top_color, right_color, left_color=None):
        if left_color is None:
            left_color = right_color
        x, y = pos
        iso_x, iso_y = self._world_to_iso(x, y)
        
        # Top face
        top_points = [
            (iso_x, iso_y - self.TILE_HEIGHT_HALF),
            (iso_x + self.TILE_WIDTH_HALF, iso_y),
            (iso_x, iso_y + self.TILE_HEIGHT_HALF),
            (iso_x - self.TILE_WIDTH_HALF, iso_y)
        ]
        
        # Right face
        right_points = [
            (iso_x + self.TILE_WIDTH_HALF, iso_y),
            (iso_x, iso_y + self.TILE_HEIGHT_HALF),
            (iso_x, iso_y + self.TILE_HEIGHT_HALF + height),
            (iso_x + self.TILE_WIDTH_HALF, iso_y + height)
        ]

        # Left face
        left_points = [
            (iso_x - self.TILE_WIDTH_HALF, iso_y),
            (iso_x, iso_y + self.TILE_HEIGHT_HALF),
            (iso_x, iso_y + self.TILE_HEIGHT_HALF + height),
            (iso_x - self.TILE_WIDTH_HALF, iso_y + height)
        ]

        pygame.gfxdraw.filled_polygon(self.screen, right_points, right_color)
        pygame.gfxdraw.aapolygon(self.screen, right_points, right_color)
        pygame.gfxdraw.filled_polygon(self.screen, left_points, left_color)
        pygame.gfxdraw.aapolygon(self.screen, left_points, left_color)
        pygame.gfxdraw.filled_polygon(self.screen, top_points, top_color)
        pygame.gfxdraw.aapolygon(self.screen, top_points, top_color)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Cavern Crystal Push")
    clock = pygame.time.Clock()
    running = True
    key_pressed_in_frame = False

    print("\n" + "="*30)
    print("Cavern Crystal Push - Manual Play")
    print(env.user_guide)
    print("Press 'R' to reset. Press 'ESC' to quit.")
    print("="*30 + "\n")

    while running:
        move_action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("--- Level Reset ---")
                elif event.key == pygame.K_UP:
                    move_action = 1
                elif event.key == pygame.K_DOWN:
                    move_action = 2
                elif event.key == pygame.K_LEFT:
                    move_action = 3
                elif event.key == pygame.K_RIGHT:
                    move_action = 4
        
        if move_action != 0:
            action = env.action_space.sample() # Get a template action
            action[0] = move_action
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_remaining']}")

            if terminated:
                print("\n--- GAME OVER ---")
                if info['player_pos'] == info['exit_pos']:
                    print("Result: You reached the exit!")
                else:
                    print("Result: You ran out of moves.")
                print(f"Final Score: {info['score']:.2f}")
                print("Press 'R' to play again or ESC to quit.")

        # Draw the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(15) # Limit frame rate to avoid multiple inputs from one key press

    env.close()