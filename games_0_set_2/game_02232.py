import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


# Set headless mode for Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selected robot. Pressing no direction switches to the next robot."
    )

    game_description = (
        "Guide three robots (red, blue, yellow) through increasingly complex isometric mazes to their "
        "corresponding green escape zones. You have a limited number of moves to rescue all three."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
        # EXACT spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup (headless)
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_ui = pygame.font.SysFont("dejavusansmono", 18)
        self.font_msg = pygame.font.SysFont("dejavusans", 30)

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_WALL = (60, 68, 87)
        self.COLOR_WALL_TOP = (90, 98, 117)
        self.COLOR_FLOOR = (40, 45, 58)
        self.COLOR_GRID = (50, 55, 68)
        self.COLOR_ESCAPE = (58, 158, 112)
        self.COLOR_ESCAPE_LIT = (88, 208, 142)
        self.COLOR_HINT = (255, 255, 255, 50)
        self.COLOR_SELECT_GLOW = (255, 255, 100)
        self.ROBOT_COLORS = [(220, 70, 70), (70, 130, 220), (220, 200, 70)]
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # Isometric projection settings
        self.tile_width = 32
        self.tile_height = 16
        self.tile_depth = 16

        # Game state variables initialized in reset()
        self.rng = None
        self.steps = 0
        self.score = 0
        self.level = 0
        self.moves_left = 0
        self.maze_dim = (0, 0)
        self.maze = None
        self.robots = []
        self.escape_zones = []
        self.selected_robot_idx = 0
        self.game_over = False
        self.game_over_message = ""
        self.terminal_reward_applied = False
        
        # The original code called reset() here, which is good practice.
        # However, it caused the error during initialization. We defer it
        # to the first explicit call by the user.
        # self.reset()
        # self.validate_implementation() # This would also fail before a reset.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        if options and "level" in options:
            self.level = options.get("level", 1)
        else:
            self.level = 1

        self.game_over = False
        self.game_over_message = ""
        self.terminal_reward_applied = False
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.1  # Cost for taking any action
        self.steps += 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]

        if movement == 0: # Switch robot
            self.selected_robot_idx = (self.selected_robot_idx + 1) % len(self.robots)
        else: # Move robot
            self.moves_left -= 1
            move_reward = self._move_robot(self.selected_robot_idx, movement)
            reward += move_reward

        terminated = self._check_termination()
        if terminated and not self.terminal_reward_applied:
            self.terminal_reward_applied = True
            if all(r['rescued'] for r in self.robots): # Win
                reward += 50
                self.game_over_message = f"ALL ROBOTS RESCUED! Score: {self.score:.1f}"
                self.level += 1 # For next reset
            elif self.moves_left <= 0: # Lose (out of moves)
                reward -= 50
                self.game_over_message = "OUT OF MOVES!"
            else: # Lose (trapped)
                reward -= 5 # This was already applied in _move_robot
                self.game_over_message = "ROBOT TRAPPED!"

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        while True:
            rows = 8 + int(self.level * 1.5)
            cols = 12 + int(self.level * 2.0)
            self.maze_dim = (rows, cols)

            # 1 = path, 0 = wall
            self.maze = self._generate_maze(rows, cols)
            
            # Place robots and escape zones
            self.robots = []
            self.escape_zones = []
            
            available_pos = [(r, c) for r in range(rows) for c in range(cols) if self.maze[r, c] == 1]
            self.rng.shuffle(available_pos)
            
            if len(available_pos) < 6: continue # Not enough space, regenerate

            for i in range(3):
                robot_pos = available_pos.pop()
                self.robots.append({'pos': robot_pos, 'rescued': False, 'id': i})
                
                escape_pos = available_pos.pop()
                self.escape_zones.append(escape_pos)
            
            if self._is_solvable():
                break

        self.moves_left = 100
        self.selected_robot_idx = 0

    def _generate_maze(self, rows, cols):
        maze = np.zeros((rows, cols), dtype=int)
        
        def is_valid(r, c):
            return 0 <= r < rows and 0 <= c < cols

        def carve(r, c):
            maze[r, c] = 1
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            self.rng.shuffle(directions)
            
            for dr, dc in directions:
                nr, nc = r + dr * 2, c + dc * 2
                if is_valid(nr, nc) and maze[nr, nc] == 0:
                    maze[r + dr, c + dc] = 1
                    carve(nr, nc)

        start_r, start_c = self.rng.integers(0, rows // 2) * 2, self.rng.integers(0, cols // 2) * 2
        carve(start_r, start_c)
        
        # Add some loops by removing a few walls
        num_loops = (rows * cols) // 25
        for _ in range(num_loops):
            r, c = self.rng.integers(1, rows - 1), self.rng.integers(1, cols - 1)
            if maze[r,c] == 0:
                maze[r,c] = 1
                
        return maze

    def _is_solvable(self):
        for robot in self.robots:
            path_found = False
            for zone in self.escape_zones:
                if self._find_path(robot['pos'], zone):
                    path_found = True
                    break
            if not path_found:
                return False
        return True

    def _find_path(self, start, end):
        q = deque([start])
        visited = {start}
        while q:
            r, c = q.popleft()
            if (r, c) == end:
                return True
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if self._is_valid_pos(nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return False

    def _move_robot(self, robot_idx, movement):
        robot = self.robots[robot_idx]
        if robot['rescued']:
            return 0

        r, c = robot['pos']
        dr, dc = 0, 0
        if movement == 1: dr = -1  # Up
        elif movement == 2: dr = 1   # Down
        elif movement == 3: dc = -1  # Left
        elif movement == 4: dc = 1   # Right

        nr, nc = r + dr, c + dc
        
        # Check for collisions
        if not self._is_valid_pos(nr, nc):
            return 0
        if any(other_robot['pos'] == (nr, nc) for other_robot in self.robots):
            return 0

        # Move is valid
        robot['pos'] = (nr, nc)
        reward = 0

        # Check for rescue
        if robot['pos'] in self.escape_zones:
            if not robot['rescued']:
                robot['rescued'] = True
                reward += 5
        
        # Check if trapped
        if self._is_trapped(robot_idx) and not robot['rescued']:
            self.game_over = True
            reward -= 5
        
        return reward

    def _is_trapped(self, robot_idx):
        r, c = self.robots[robot_idx]['pos']
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if self._is_valid_pos(nr, nc) and not any(rb['pos'] == (nr, nc) for rb in self.robots):
                return False
        return True

    def _is_valid_pos(self, r, c):
        rows, cols = self.maze_dim
        return 0 <= r < rows and 0 <= c < cols and self.maze[r, c] == 1

    def _check_termination(self):
        if self.game_over:
            return True
        if all(r['rescued'] for r in self.robots):
            self.game_over = True
            return True
        if self.moves_left <= 0:
            self.game_over = True
            return True
        if self.steps >= 1000:
            return True
        return False

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
            "level": self.level,
            "moves_left": self.moves_left,
            "robots_rescued": sum(1 for r in self.robots if r['rescued']),
        }

    def _grid_to_iso(self, r, c):
        screen_x = (self.SCREEN_WIDTH / 2) + (c - r) * (self.tile_width / 2)
        screen_y = (self.SCREEN_HEIGHT / 4) + (c + r) * (self.tile_height / 2)
        return int(screen_x), int(screen_y)

    def _render_game(self):
        rows, cols = self.maze_dim
        
        # Draw floors, grid lines, and escape zones
        for r in range(rows):
            for c in range(cols):
                if self.maze[r, c] == 1:
                    sx, sy = self._grid_to_iso(r, c)
                    points = [
                        (sx, sy - self.tile_height // 2),
                        (sx + self.tile_width // 2, sy),
                        (sx, sy + self.tile_height // 2),
                        (sx - self.tile_width // 2, sy)
                    ]
                    color = self.COLOR_ESCAPE if (r, c) in self.escape_zones else self.COLOR_FLOOR
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw movement hints for selected robot
        if not self.game_over:
            selected_robot = self.robots[self.selected_robot_idx]
            if not selected_robot['rescued']:
                r, c = selected_robot['pos']
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if self._is_valid_pos(nr, nc) and not any(rb['pos'] == (nr, nc) for rb in self.robots):
                        sx, sy = self._grid_to_iso(nr, nc)
                        points = [
                            (sx, sy - self.tile_height // 2),
                            (sx + self.tile_width // 2, sy),
                            (sx, sy + self.tile_height // 2),
                            (sx - self.tile_width // 2, sy)
                        ]
                        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HINT)

        # Draw walls and robots
        for r in range(rows):
            for c in range(cols):
                # Draw walls
                if self.maze[r, c] == 0:
                    self._draw_iso_cube(r, c, self.tile_depth, self.COLOR_WALL, self.COLOR_WALL_TOP)

                # Draw robots
                for i, robot in enumerate(self.robots):
                    if robot['pos'] == (r, c):
                        self._draw_robot(robot, i == self.selected_robot_idx)

    def _draw_iso_cube(self, r, c, height, side_color, top_color):
        sx, sy = self._grid_to_iso(r, c)
        hw, hh = self.tile_width / 2, self.tile_height / 2
        
        top_points = [
            (sx, sy - hh - height), (sx + hw, sy - height),
            (sx, sy + hh - height), (sx - hw, sy - height)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, top_points, top_color)
        pygame.gfxdraw.aapolygon(self.screen, top_points, side_color)

        # Left side
        left_points = [
            (sx - hw, sy - height), (sx, sy + hh - height),
            (sx, sy + hh), (sx - hw, sy)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, left_points, side_color)
        
        # Right side
        right_points = [
            (sx + hw, sy - height), (sx, sy + hh - height),
            (sx, sy + hh), (sx + hw, sy)
        ]
        # FIX: Create a new, darker color tuple by multiplying the components.
        # Multiplying a pygame.Color object by a float is not supported.
        darker_side_color = tuple(int(c * 0.8) for c in side_color)
        pygame.gfxdraw.filled_polygon(self.screen, right_points, darker_side_color)

    def _draw_robot(self, robot, is_selected):
        r, c = robot['pos']
        sx, sy = self._grid_to_iso(r, c)
        
        # Shadow
        shadow_radius = int(self.tile_width * 0.35)
        shadow_surf = pygame.Surface((shadow_radius * 2, shadow_radius * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 70), (0, 0, shadow_radius * 2, shadow_radius * 1.5))
        self.screen.blit(shadow_surf, (sx - shadow_radius, sy - shadow_radius // 2))

        # Selection glow
        if is_selected and not self.game_over:
            glow_radius = int(self.tile_width * 0.45)
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, glow_radius, self.COLOR_SELECT_GLOW + (50,))
            pygame.gfxdraw.aacircle(self.screen, sx, sy, glow_radius, self.COLOR_SELECT_GLOW)

        # Robot body
        robot_color = self.ROBOT_COLORS[robot['id']]
        if robot['rescued']:
            robot_color = self.COLOR_ESCAPE_LIT
        
        radius = int(self.tile_width * 0.25)
        body_y = sy - radius // 2
        pygame.gfxdraw.filled_circle(self.screen, sx, body_y, radius, robot_color)
        pygame.gfxdraw.aacircle(self.screen, sx, body_y, radius, (0,0,0,50))

        # Eye
        eye_color = (255, 255, 255) if not robot['rescued'] else (25, 28, 36)
        pygame.draw.circle(self.screen, eye_color, (sx, body_y), radius // 3)

    def _render_ui(self):
        # Top UI panel
        panel_surf = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        panel_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(panel_surf, (0, 0))
        
        # UI Text
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        moves_text = self.font_ui.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        level_text = self.font_ui.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(moves_text, (self.SCREEN_WIDTH // 2 - moves_text.get_width() // 2, 10))
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            msg_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            msg_surf.fill((0, 0, 0, 150))
            
            msg_text = self.font_msg.render(self.game_over_message, True, self.COLOR_TEXT)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            msg_surf.blit(msg_text, msg_rect)
            self.screen.blit(msg_surf, (0, 0))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Validating implementation...")
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

# Example of how to run the environment
if __name__ == '__main__':
    # For human play, we need to set up a display
    # Un-set the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Robot Maze")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default action: switch robot
        
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
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if action[0] != 0 and not env.game_over:
             obs, reward, terminated, truncated, info = env.step(action)
        elif action[0] == 0: # Allow switching even if game is over
             obs, reward, terminated, truncated, info = env.step(action)
        
        if env.game_over:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Render one last time to show the game over message
            obs = env._get_observation()
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            # Wait for a moment before auto-resetting or allow manual reset
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Render the observation from the environment to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()