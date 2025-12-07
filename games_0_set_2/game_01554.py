
# Generated: 2025-08-27T17:29:43.512873
# Source Brief: brief_01554.md
# Brief Index: 1554

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the robot through the maze."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Guide the robot to the green exit through a "
        "randomly generated maze before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    MAX_MOVES = 25

    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_WALL = (180, 180, 200)
    COLOR_ROBOT = (255, 50, 50)
    COLOR_ROBOT_GLOW = (255, 100, 100, 50)
    COLOR_EXIT = (50, 255, 50)
    COLOR_EXIT_GLOW = (100, 255, 100, 50)
    COLOR_TEXT = (220, 220, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.moves_left = 0
        self.robot_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.walls = set() # Stores tuples of ((x1, y1), (x2, y2)) representing walls

        # Calculate grid rendering properties once
        self.cell_size = min(
            (self.SCREEN_WIDTH - 80) // self.GRID_WIDTH,
            (self.SCREEN_HEIGHT - 80) // self.GRID_HEIGHT
        )
        self.grid_area_width = self.cell_size * self.GRID_WIDTH
        self.grid_area_height = self.cell_size * self.GRID_HEIGHT
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_area_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_area_height) // 2

        self.reset()
        
        # self.validate_implementation() # Optional: Call for self-check

    def _generate_maze(self):
        """Generates a perfect maze using randomized DFS, guarantees a path."""
        self.walls = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if x < self.GRID_WIDTH - 1:
                    self.walls.add(tuple(sorted(((x, y), (x + 1, y)))))
                if y < self.GRID_HEIGHT - 1:
                    self.walls.add(tuple(sorted(((x, y), (x, y + 1)))))

        visited = set()
        stack = [(self.robot_pos)]
        visited.add(self.robot_pos)

        while stack:
            current_cell = stack[-1]
            cx, cy = current_cell

            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    neighbors.append((nx, ny))

            if neighbors:
                next_cell = self.np_random.choice(len(neighbors))
                next_cell = neighbors[next_cell]
                
                wall_to_remove = tuple(sorted((current_cell, next_cell)))
                if wall_to_remove in self.walls:
                    self.walls.remove(wall_to_remove)
                
                visited.add(next_cell)
                stack.append(next_cell)
            else:
                stack.pop()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.moves_left = self.MAX_MOVES

        # Set start and end points
        self.robot_pos = (0, 0)
        self.exit_pos = (self.GRID_WIDTH - 1, self.GRID_HEIGHT - 1)

        self._generate_maze()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = -1.0  # Cost for taking a step/turn
        
        if movement != 0: # Any action other than no-op costs a move
            self.moves_left -= 1
            self.steps += 1

            # Calculate potential new position
            current_x, current_y = self.robot_pos
            next_pos = self.robot_pos
            if movement == 1:  # Up
                next_pos = (current_x, current_y - 1)
            elif movement == 2:  # Down
                next_pos = (current_x, current_y + 1)
            elif movement == 3:  # Left
                next_pos = (current_x - 1, current_y)
            elif movement == 4:  # Right
                next_pos = (current_x + 1, current_y)

            # Check for wall collision
            wall = tuple(sorted((self.robot_pos, next_pos)))
            if wall not in self.walls:
                # Check for boundary collision
                nx, ny = next_pos
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    self.robot_pos = next_pos
                    # sfx: player_move.wav

        self.score += reward

        # Check for termination conditions
        terminated = False
        if self.robot_pos == self.exit_pos:
            reward += 100.0
            self.score += 100.0
            terminated = True
            # sfx: win_level.wav
        elif self.moves_left <= 0:
            terminated = True
            # sfx: lose_level.wav
        
        if terminated:
            self.game_over = True

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
            "steps": self.steps,
            "moves_left": self.moves_left,
            "robot_pos": self.robot_pos,
            "exit_pos": self.exit_pos
        }

    def _render_game(self):
        # Draw grid background cells
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(
            self.grid_offset_x + ex * self.cell_size,
            self.grid_offset_y + ey * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect.inflate(-4, -4))
        pygame.gfxdraw.aacircle(
            self.screen,
            exit_rect.centerx,
            exit_rect.centery,
            self.cell_size // 2,
            self.COLOR_EXIT_GLOW
        )


        # Draw robot
        rx, ry = self.robot_pos
        robot_rect = pygame.Rect(
            self.grid_offset_x + rx * self.cell_size,
            self.grid_offset_y + ry * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect.inflate(-4, -4))
        pygame.gfxdraw.aacircle(
            self.screen,
            robot_rect.centerx,
            robot_rect.centery,
            self.cell_size // 2,
            self.COLOR_ROBOT_GLOW
        )
        
        # Draw walls
        for wall in self.walls:
            p1, p2 = wall
            x1, y1 = p1
            x2, y2 = p2
            
            start_pos = (0, 0)
            end_pos = (0, 0)

            if y1 == y2:  # Vertical wall
                start_pos = (
                    self.grid_offset_x + (x1 + 1) * self.cell_size,
                    self.grid_offset_y + y1 * self.cell_size
                )
                end_pos = (
                    self.grid_offset_x + (x1 + 1) * self.cell_size,
                    self.grid_offset_y + (y1 + 1) * self.cell_size
                )
            else:  # Horizontal wall
                start_pos = (
                    self.grid_offset_x + x1 * self.cell_size,
                    self.grid_offset_y + (y1 + 1) * self.cell_size
                )
                end_pos = (
                    self.grid_offset_x + (x1 + 1) * self.cell_size,
                    self.grid_offset_y + (y1 + 1) * self.cell_size
                )
            pygame.draw.line(self.screen, self.COLOR_WALL, start_pos, end_pos, 3)

    def _render_ui(self):
        moves_text = f"Moves Left: {self.moves_left}"
        text_surface = self.font.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (20, 20))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            
            end_text = "SUCCESS!" if self.robot_pos == self.exit_pos else "OUT OF MOVES"
            end_font = pygame.font.SysFont("monospace", 60, bold=True)
            end_surface = end_font.render(end_text, True, self.COLOR_EXIT if self.robot_pos == self.exit_pos else self.COLOR_ROBOT)
            end_rect = end_surface.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))

            self.screen.blit(overlay, (0, 0))
            self.screen.blit(end_surface, end_rect)


    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "dummy" to run headlessly
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    action = [0, 0, 0] # No-op

    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("Press R to reset. Press Q to quit.")
    print("="*30 + "\n")

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                    action = [0, 0, 0]
                
                # Map keys to actions
                if not done:
                    if event.key == pygame.K_UP:
                        action = [1, 0, 0]
                    elif event.key == pygame.K_DOWN:
                        action = [2, 0, 0]
                    elif event.key == pygame.K_LEFT:
                        action = [3, 0, 0]
                    elif event.key == pygame.K_RIGHT:
                        action = [4, 0, 0]
                    else:
                        action = [0, 0, 0] # No-op for other keys

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_left']}, Terminated: {done}")
                    action = [0, 0, 0] # Reset action after one step

        # Rendering
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()