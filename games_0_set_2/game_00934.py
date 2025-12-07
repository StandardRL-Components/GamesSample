
# Generated: 2025-08-27T15:15:02.196028
# Source Brief: brief_00934.md
# Brief Index: 934

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move the robot one tile at a time."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide the robot through a maze to its charging station before the battery runs out. Each move depletes the battery."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_EPISODE_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_WALL = (70, 80, 90)
        self.COLOR_PATH = (40, 45, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_ACCENT = (200, 240, 255)
        self.COLOR_GOAL = (0, 200, 80)
        self.COLOR_GOAL_ACCENT = (180, 255, 200)
        self.COLOR_BATTERY = (255, 200, 0)
        self.COLOR_BATTERY_BG = (50, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_MOVE_FEEDBACK = (255, 255, 255, 100) # RGBA for transparency

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
        self.font_level = pygame.font.Font(None, 40)

        # --- Game State ---
        self.level = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.maze = None
        self.player_pos = None
        self.goal_pos = None
        self.moves_remaining = 0
        self.max_moves = 0
        self.maze_width = 0
        self.maze_height = 0
        self.move_feedback_timer = 0
        self.last_move_pos = None

        self.reset()
        
        # This is a critical self-check.
        self.validate_implementation()


    def _init_level_properties(self):
        """Initializes properties for the current level."""
        self.level += 1
        # Maze dimensions must be odd
        base_size = 7
        self.maze_width = min(base_size + (self.level - 1) * 2, 39)
        self.maze_height = min(base_size + (self.level - 1) * 2, 23)
        
        self.max_moves = 15 + ((self.level - 1) // 3) * 5
        self.moves_remaining = self.max_moves


    def _generate_maze(self):
        """
        Generates a maze using Randomized Depth-First Search.
        Ensures a path exists and places the player and goal.
        """
        # Grid: 1 for wall, 0 for path
        grid = np.ones((self.maze_height, self.maze_width), dtype=np.uint8)
        
        # DFS traversal to carve paths
        stack = deque()
        start_pos = (1, 1)
        grid[start_pos] = 0
        stack.append(start_pos)
        
        path_nodes = {start_pos}

        while stack:
            cy, cx = stack[-1]
            neighbors = []
            for dy, dx in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                ny, nx = cy + dy, cx + dx
                if 0 < ny < self.maze_height - 1 and 0 < nx < self.maze_width - 1 and grid[ny, nx] == 1:
                    neighbors.append((ny, nx))
            
            if neighbors:
                ny, nx = self.np_random.choice(neighbors, axis=0)
                wall_y, wall_x = cy + (ny - cy) // 2, cx + (nx - cx) // 2
                grid[ny, nx] = 0
                grid[wall_y, wall_x] = 0
                path_nodes.add((ny,nx))
                stack.append((ny, nx))
            else:
                stack.pop()

        self.maze = grid
        
        # Place player
        self.player_pos = list(start_pos)

        # Place goal at the furthest point from the player using BFS
        queue = deque([(start_pos, 0)])
        visited = {start_pos}
        farthest_node = start_pos
        max_dist = 0

        while queue:
            (y, x), dist = queue.popleft()

            if dist > max_dist:
                max_dist = dist
                farthest_node = (y, x)

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.maze_height and 0 <= nx < self.maze_width and self.maze[ny, nx] == 0 and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    queue.append(((ny, nx), dist + 1))
        
        self.goal_pos = list(farthest_node)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.game_over: # Increment level only after completing a level
            pass # Level is incremented in step() on win
        else: # Full reset or first start
            self.level = 0
        
        self._init_level_properties()
        self._generate_maze()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.move_feedback_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self.steps += 1
        reward = -0.1  # Cost for taking a turn
        terminated = False

        # --- Handle Movement ---
        py, px = self.player_pos
        self.last_move_pos = (py, px) # For visual feedback
        
        dy, dx = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        # A no-op (movement == 0) or any other action still consumes a turn
        self.moves_remaining -= 1
        
        if movement != 0:
            ny, nx = py + dy, px + dx
            # Check for valid move (within bounds and not a wall)
            if 0 <= ny < self.maze_height and 0 <= nx < self.maze_width and self.maze[ny, nx] == 0:
                self.player_pos = [ny, nx]
                # _move_sound_.play()
                self.move_feedback_timer = 5 # Start visual feedback animation
            else:
                # _bump_wall_sound_.play()
                pass # Invalid move, position doesn't change

        # --- Check Termination Conditions ---
        if self.player_pos == self.goal_pos:
            reward += 10.0
            terminated = True
            self.game_over = True
            # _win_sound_.play()

        elif self.moves_remaining <= 0:
            terminated = True
            self.game_over = True
            # _lose_sound_.play()
            
        if self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the maze, player, and goal."""
        # Calculate tile size and maze offset to center it
        tile_h = self.SCREEN_HEIGHT * 0.8 / self.maze_height
        tile_w = tile_h # Square tiles
        
        maze_pixel_w = self.maze_width * tile_w
        maze_pixel_h = self.maze_height * tile_h
        
        offset_x = (self.SCREEN_WIDTH - maze_pixel_w) / 2
        offset_y = (self.SCREEN_HEIGHT - maze_pixel_h) / 2
        
        # Draw move feedback
        if self.move_feedback_timer > 0 and self.last_move_pos is not None:
            py, px = self.last_move_pos
            rect = pygame.Rect(
                offset_x + px * tile_w,
                offset_y + py * tile_h,
                tile_w,
                tile_h
            )
            alpha = self.move_feedback_timer * 30
            shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (255, 255, 255, alpha), shape_surf.get_rect())
            self.screen.blit(shape_surf, rect)
            self.move_feedback_timer -= 1

        # Draw maze tiles
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                rect = pygame.Rect(
                    offset_x + x * tile_w,
                    offset_y + y * tile_h,
                    tile_w,
                    tile_h
                )
                color = self.COLOR_WALL if self.maze[y, x] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)

        # Draw Goal
        gy, gx = self.goal_pos
        goal_rect = pygame.Rect(
            offset_x + gx * tile_w,
            offset_y + gy * tile_h,
            tile_w,
            tile_h
        )
        pygame.gfxdraw.box(self.screen, goal_rect, self.COLOR_GOAL)
        # Goal "power" symbol
        symbol_cx = int(goal_rect.centerx)
        symbol_cy = int(goal_rect.centery)
        radius = int(min(tile_w, tile_h) * 0.25)
        pygame.gfxdraw.aacircle(self.screen, symbol_cx, symbol_cy, radius, self.COLOR_GOAL_ACCENT)
        pygame.gfxdraw.filled_circle(self.screen, symbol_cx, symbol_cy, radius, self.COLOR_GOAL_ACCENT)


        # Draw Player
        py, px = self.player_pos
        player_cx = int(offset_x + px * tile_w + tile_w / 2)
        player_cy = int(offset_y + py * tile_h + tile_h / 2)
        player_radius = int(min(tile_w, tile_h) * 0.35)
        
        pygame.gfxdraw.aacircle(self.screen, player_cx, player_cy, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_cx, player_cy, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_cx, player_cy, int(player_radius*0.6), self.COLOR_PLAYER_ACCENT)
        
        # Draw Battery Bar above player
        bar_width = tile_w * 0.8
        bar_height = tile_h * 0.1
        bar_x = player_cx - bar_width / 2
        bar_y = (offset_y + py * tile_h) - bar_height * 2

        # Background of the bar
        pygame.draw.rect(self.screen, self.COLOR_BATTERY_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=2)
        # Foreground (current battery)
        battery_ratio = max(0, self.moves_remaining / self.max_moves)
        current_bar_width = bar_width * battery_ratio
        if current_bar_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_BATTERY, (bar_x, bar_y, current_bar_width, bar_height), border_radius=2)


    def _render_ui(self):
        """Renders UI text like level and moves remaining."""
        level_text = self.font_level.render(f"LEVEL {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (20, 20))

        moves_text = self.font_ui.render(f"BATTERY: {max(0, self.moves_remaining)}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 25))
        self.screen.blit(moves_text, moves_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_remaining": self.moves_remaining,
        }

    def close(self):
        pygame.font.quit()
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

# --- Example of how to run the environment ---
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11", "dummy" or "windows" depending on your system

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame window for human play
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Robot")
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print("      MAZE ROBOT - MANUAL PLAY")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("Press Q or close the window to quit.")
    print("="*30 + "\n")

    running = True
    while running:
        action = np.array([0, 0, 0])  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                
                # Since auto_advance is False, we only step on key press
                move = 0
                if event.key == pygame.K_UP:
                    move = 1
                elif event.key == pygame.K_DOWN:
                    move = 2
                elif event.key == pygame.K_LEFT:
                    move = 3
                elif event.key == pygame.K_RIGHT:
                    move = 4
                
                # Only process a step if a move key was pressed
                if move != 0:
                    action[0] = move
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

                    if terminated:
                        print("\n--- LEVEL END ---")
                        print(f"Final Score: {info['score']:.2f}")
                        print("Resetting environment...\n")
                        obs, info = env.reset()

        # Render the observation to the human-visible screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    env.close()
    print("Environment closed.")