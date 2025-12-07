
# Generated: 2025-08-27T23:04:26.958202
# Source Brief: brief_03342.md
# Brief Index: 3342

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ↑↓←→ to move the selected robot. Space/Shift to cycle through robots."
    )

    # Short, user-facing description of the game
    game_description = (
        "Guide your squad of robots through isometric mazes to the exit before you run out of moves."
    )

    # Frames only advance when an action is received
    auto_advance = False
    
    # --- Colors and Visuals ---
    COLOR_BG = (25, 30, 35)
    COLOR_WALL_TOP = (60, 70, 80)
    COLOR_WALL_SIDE = (50, 60, 70)
    COLOR_FLOOR = (40, 45, 50)
    COLOR_EXIT = (60, 220, 120)
    COLOR_EXIT_GLOW = (150, 255, 200)
    COLOR_ROBOTS = [
        (255, 80, 80), (80, 150, 255), (255, 220, 80),
        (180, 80, 255), (255, 150, 80), (80, 220, 220)
    ]
    COLOR_SELECT_GLOW = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_WARN = (255, 100, 100)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Game state variables
        self.level = 1
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.maze_dim = 0
        self.max_moves = 0
        self.moves_left = 0
        self.robots = []
        self.exit_pos = (0, 0)
        self.maze = np.array([[]])
        self.selected_robot_idx = 0
        self.animation_timer = 0
        
        self.reset()
        self.validate_implementation()
    
    def _calculate_level_params(self):
        """Calculates game parameters based on the current level."""
        self.maze_dim = min(20, 8 + 2 * self.level)
        num_robots = min(len(self.COLOR_ROBOTS), 2 + self.level)
        self.max_moves = 20 + num_robots * 5 + self.level * 5
        return num_robots

    def _generate_maze(self, width, height):
        """Generates a perfect maze using iterative randomized DFS."""
        maze = np.ones((height, width), dtype=np.uint8) # 1 = wall, 0 = path
        stack = deque()
        
        # Start carving from a random position
        start_x, start_y = self.np_random.integers(0, width//2)*2, self.np_random.integers(0, height//2)*2
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]
                # Carve path
                maze[ny, nx] = 0
                maze[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and options.get("reset_level", False):
            self.level = 1

        num_robots = self._calculate_level_params()
        self.maze = self._generate_maze(self.maze_dim, self.maze_dim)
        
        # Get all possible floor locations for spawning
        floor_tiles = np.argwhere(self.maze == 0).tolist()
        self.np_random.shuffle(floor_tiles)

        spawn_locations = floor_tiles[:num_robots + 1]
        self.exit_pos = tuple(spawn_locations.pop())
        
        self.robots = []
        for i in range(num_robots):
            pos = tuple(spawn_locations[i])
            self.robots.append({
                'pos': pos,
                'color': self.COLOR_ROBOTS[i % len(self.COLOR_ROBOTS)],
                'active': True,
                'id': i
            })
        
        self.selected_robot_idx = 0
        self.moves_left = self.max_moves
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.animation_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0
        
        active_robots = [r for r in self.robots if r['active']]
        if not active_robots:
            self.game_over = True
            return self._get_observation(), 100, True, False, self._get_info()

        # --- 1. Handle Robot Selection ---
        if space_pressed or shift_pressed:
            # sfx: robot_select
            current_selection_id = active_robots[self.selected_robot_idx]['id']
            all_active_ids = [r['id'] for r in active_robots]
            
            try:
                current_idx_in_active = all_active_ids.index(current_selection_id)
                if space_pressed:
                    new_idx_in_active = (current_idx_in_active + 1) % len(all_active_ids)
                else: # shift_pressed
                    new_idx_in_active = (current_idx_in_active - 1 + len(all_active_ids)) % len(all_active_ids)
                
                new_robot_id = all_active_ids[new_idx_in_active]
                self.selected_robot_idx = [i for i, r in enumerate(self.robots) if r['id'] == new_robot_id][0]

            except (ValueError, IndexError): # If selection is somehow invalid, reset to first active
                 if active_robots:
                    self.selected_robot_idx = self.robots.index(active_robots[0])


        # --- 2. Handle Movement ---
        if movement > 0:
            self.moves_left -= 1
            reward -= 0.1 # Penalty for taking a move
            
            robot = self.robots[self.selected_robot_idx]
            if not robot['active']: # Can't move inactive robots
                movement = 0

            if movement > 0:
                # Iso directions: 1=up(NE), 2=down(SW), 3=left(NW), 4=right(SE)
                moves = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}
                dx, dy = moves[movement]
                
                old_pos = robot['pos']
                new_pos = (old_pos[0] + dx, old_pos[1] + dy)
                
                # Check boundaries
                if not (0 <= new_pos[0] < self.maze_dim and 0 <= new_pos[1] < self.maze_dim):
                    # sfx: bump_wall
                    pass # Hit edge of map
                # Check walls
                elif self.maze[new_pos[1], new_pos[0]] == 1:
                    # sfx: bump_wall
                    pass # Hit a wall
                # Check other robots
                elif any(r['pos'] == new_pos for r in self.robots if r['active']):
                    # sfx: bump_robot
                    pass # Blocked by another robot
                else:
                    # sfx: robot_move
                    robot['pos'] = new_pos

            # Check if robot reached exit
            if robot['active'] and robot['pos'] == self.exit_pos:
                # sfx: robot_exit
                robot['active'] = False
                reward += 1.0
                
                # If this was the last active robot, select the next available one
                if not any(r['active'] for r in self.robots):
                    pass # Game will end on next check
                elif self.selected_robot_idx == self.robots.index(robot):
                    # Find first active robot and select it
                    for i, r in enumerate(self.robots):
                        if r['active']:
                            self.selected_robot_idx = i
                            break

        # --- 3. Check Termination Conditions ---
        terminated = False
        all_exited = not any(r['active'] for r in self.robots)
        out_of_moves = self.moves_left <= 0

        if all_exited:
            # sfx: level_complete
            reward += 100
            terminated = True
            self.game_over = True
            self.level += 1
        elif out_of_moves:
            # sfx: game_over
            reward -= 10
            terminated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _iso_to_screen(self, x, y, tile_w, tile_h):
        """Converts isometric grid coordinates to screen coordinates."""
        screen_x = self.origin_x + (x - y) * tile_w / 2
        screen_y = self.origin_y + (x + y) * tile_h / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, x, y, tile_w, tile_h, height, top_color, side_color):
        """Draws a single isometric cube."""
        px, py = self._iso_to_screen(x, y, tile_w, tile_h)
        
        points_top = [
            (px, py - height),
            (px + tile_w / 2, py - tile_h / 2 - height),
            (px, py - tile_h - height),
            (px - tile_w / 2, py - tile_h / 2 - height),
        ]
        points_left = [
            (px - tile_w / 2, py - tile_h / 2 - height),
            (px, py - tile_h - height),
            (px, py - tile_h),
            (px - tile_w / 2, py - tile_h / 2),
        ]
        points_right = [
            (px + tile_w / 2, py - tile_h / 2 - height),
            (px, py - tile_h - height),
            (px, py - tile_h),
            (px + tile_w / 2, py - tile_h / 2),
        ]

        pygame.gfxdraw.filled_polygon(surface, [(int(p[0]), int(p[1])) for p in points_left], side_color)
        pygame.gfxdraw.aapolygon(surface, [(int(p[0]), int(p[1])) for p in points_left], side_color)
        pygame.gfxdraw.filled_polygon(surface, [(int(p[0]), int(p[1])) for p in points_right], side_color)
        pygame.gfxdraw.aapolygon(surface, [(int(p[0]), int(p[1])) for p in points_right], side_color)
        pygame.gfxdraw.filled_polygon(surface, [(int(p[0]), int(p[1])) for p in points_top], top_color)
        pygame.gfxdraw.aapolygon(surface, [(int(p[0]), int(p[1])) for p in points_top], top_color)

    def _render_game(self):
        """Renders the main game elements (maze, robots, exit)."""
        self.animation_timer = (self.animation_timer + 1) % 60
        
        # Define tile dimensions based on maze size to fit screen
        tile_w = min(60, 600 / self.maze_dim)
        tile_h = tile_w * 0.5
        wall_height = tile_h * 1.5
        floor_height = 0
        
        self.origin_x = 320
        self.origin_y = 50 + (self.maze_dim * tile_h / 2)

        # Draw floor and exit
        for y in range(self.maze_dim):
            for x in range(self.maze_dim):
                if self.maze[y, x] == 0:
                    color = self.COLOR_EXIT if (x, y) == self.exit_pos else self.COLOR_FLOOR
                    self._draw_iso_cube(self.screen, x, y, tile_w, tile_h, floor_height, color, self.COLOR_BG)
        
        # Draw exit glow
        if not self.game_over:
            glow_alpha = 90 + int(math.sin(self.animation_timer * 0.1) * 30)
            glow_surf = pygame.Surface((tile_w, tile_h), pygame.SRCALPHA)
            pygame.gfxdraw.filled_polygon(glow_surf, [(tile_w/2, 0), (tile_w, tile_h/2), (tile_w/2, tile_h), (0, tile_h/2)], self.COLOR_EXIT_GLOW + (glow_alpha,))
            ex, ey = self._iso_to_screen(self.exit_pos[0], self.exit_pos[1], tile_w, tile_h)
            self.screen.blit(glow_surf, (int(ex - tile_w/2), int(ey - tile_h)))
        
        # Draw walls
        for y in range(self.maze_dim):
            for x in range(self.maze_dim):
                if self.maze[y, x] == 1:
                    self._draw_iso_cube(self.screen, x, y, tile_w, tile_h, wall_height, self.COLOR_WALL_TOP, self.COLOR_WALL_SIDE)

        # Draw robots
        for i, robot in enumerate(self.robots):
            if not robot['active']:
                continue
            
            rx, ry = self._iso_to_screen(robot['pos'][0], robot['pos'][1], tile_w, tile_h)
            robot_radius = int(tile_w * 0.2)
            
            # Selection highlight
            if i == self.selected_robot_idx:
                pulse = 1 + abs(math.sin(self.animation_timer * 0.2))
                glow_radius = int(robot_radius * (1.2 + pulse * 0.4))
                glow_alpha = 150 - int(pulse * 50)
                pygame.gfxdraw.filled_circle(self.screen, rx, int(ry-tile_h/2), glow_radius, self.COLOR_SELECT_GLOW + (glow_alpha,))

            # Robot body
            pygame.gfxdraw.filled_circle(self.screen, rx, int(ry-tile_h/2), robot_radius, robot['color'])
            pygame.gfxdraw.aacircle(self.screen, rx, int(ry-tile_h/2), robot_radius, robot['color'])

    def _render_ui(self):
        """Renders the UI overlay (score, moves, etc.)."""
        # Level
        level_text = self.font_small.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        # Score
        score_text = self.font_large.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(centerx=320, top=10)
        self.screen.blit(score_text, score_rect)

        # Moves Left
        moves_color = self.COLOR_TEXT if self.moves_left > 10 else self.COLOR_TEXT_WARN
        moves_text = self.font_large.render(f"MOVES: {self.moves_left}", True, moves_color)
        moves_rect = moves_text.get_rect(right=630, top=10)
        self.screen.blit(moves_text, moves_rect)

        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            result_text = "LEVEL COMPLETE" if any(r['active'] == False for r in self.robots) and self.moves_left >=0 else "OUT OF MOVES"
            
            end_font = pygame.font.Font(None, 60)
            text_surf = end_font.render(result_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(320, 200))
            overlay.blit(text_surf, text_rect)
            self.screen.blit(overlay, (0, 0))


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        active_robots = sum(1 for r in self.robots if r['active'])
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_left": self.moves_left,
            "robots_left": active_robots
        }

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

# Example usage to run and display the game
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Must be set for pygame to run headlessly
    
    env = GameEnv()
    
    # To run with a display:
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.display.init()
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Isometric Robot Maze")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)
    
    while not done:
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # [movement, space, shift]
        
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Process events
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                action_taken = True
                if event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                
        # In a turn-based game, we only step when a key is pressed
        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Drawing
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    print("Game Over!")
    env.close()