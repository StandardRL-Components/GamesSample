
# Generated: 2025-08-27T16:46:59.056185
# Source Brief: brief_01328.md
# Brief Index: 1328

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to move the robot one tile at a time. "
        "Avoid the red lasers!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down puzzle game. Guide your robot through a maze of rotating lasers to reach the green exit tile. "
        "Plan your moves carefully as the lasers turn every 5 steps."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.TILE_SIZE = 32
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = (self.SCREEN_HEIGHT - 40) // self.TILE_SIZE # Reserve 40px for UI
        self.UI_HEIGHT = self.SCREEN_HEIGHT - (self.GRID_HEIGHT * self.TILE_SIZE)
        
        self.MAX_STEPS = 1000
        self.LASER_ROTATION_INTERVAL = 5

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 50)
        self.COLOR_WALL = (50, 60, 70)
        self.COLOR_ROBOT = (0, 150, 255)
        self.COLOR_ROBOT_GLOW = (100, 200, 255)
        self.COLOR_EXIT = (0, 200, 100)
        self.COLOR_EXIT_GLOW = (100, 255, 150)
        self.COLOR_LASER = (255, 50, 50)
        self.COLOR_LASER_GLOW = (255, 100, 100, 100)
        self.COLOR_LASER_SOURCE = (255, 150, 150)
        self.COLOR_TEXT = (220, 220, 220)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.robot_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.walls = set()
        self.lasers = [] # List of dicts: {"pos": (x,y), "angle": degrees}
        self.np_random = None
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Comment out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        if options and "level" in options:
            self.level = options["level"]
        else:
            self.level = 1
            
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        self.robot_pos = (1, self.GRID_HEIGHT // 2)
        self.exit_pos = (self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2)
        
        self._generate_walls()
        self._generate_lasers()

    def _generate_walls(self):
        self.walls = set()
        path_found = False
        while not path_found:
            self.walls.clear()
            # Add border walls
            for x in range(self.GRID_WIDTH):
                self.walls.add((x, -1))
                self.walls.add((x, self.GRID_HEIGHT))
            for y in range(self.GRID_HEIGHT):
                self.walls.add((-1, y))
                self.walls.add((self.GRID_WIDTH, y))

            # Add random internal walls
            num_walls = int(self.GRID_WIDTH * self.GRID_HEIGHT * 0.2)
            for _ in range(num_walls):
                wall_pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
                if wall_pos != self.robot_pos and wall_pos != self.exit_pos:
                    self.walls.add(wall_pos)
            
            # Check for a valid path using BFS
            queue = deque([self.robot_pos])
            visited = {self.robot_pos}
            found = False
            while queue:
                x, y = queue.popleft()
                if (x, y) == self.exit_pos:
                    found = True
                    break
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in self.walls and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            path_found = found

    def _generate_lasers(self):
        self.lasers = []
        num_lasers = 8 + 2 * self.level
        
        available_pos = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x,y) not in self.walls and (x,y) != self.robot_pos and (x,y) != self.exit_pos:
                    available_pos.add((x,y))
        
        for _ in range(num_lasers):
            if not available_pos: break
            pos = self.np_random.choice(list(available_pos))
            available_pos.remove(tuple(pos))
            angle = self.np_random.integers(0, 360)
            self.lasers.append({"pos": tuple(pos), "angle": angle})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = -0.01 # Small penalty for each step
        
        # --- Update Robot Position ---
        px, py = self.robot_pos
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        
        if (px, py) not in self.walls and 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
            self.robot_pos = (px, py)

        # --- Update Lasers ---
        if self.steps % self.LASER_ROTATION_INTERVAL == 0:
            # sfx: laser_charge.wav
            rotation_amount = 45 + (self.level - 1)
            for laser in self.lasers:
                laser["angle"] = (laser["angle"] + rotation_amount) % 360
        
        # --- Check for Termination ---
        terminated = False
        if self._check_laser_collision():
            # sfx: player_hit.wav
            reward = -100.0
            terminated = True
            self.game_over = True
        
        if self.robot_pos == self.exit_pos:
            # sfx: level_complete.wav
            reward = 100.0
            self.score += 1
            terminated = True
            self.game_over = True # Or could lead to next level
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            # No extra penalty, step penalty has accumulated

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _check_laser_collision(self):
        robot_rect = pygame.Rect(
            self.robot_pos[0] * self.TILE_SIZE,
            self.robot_pos[1] * self.TILE_SIZE + self.UI_HEIGHT,
            self.TILE_SIZE,
            self.TILE_SIZE
        )
        for laser in self.lasers:
            start_pos_px = (
                laser["pos"][0] * self.TILE_SIZE + self.TILE_SIZE // 2,
                laser["pos"][1] * self.TILE_SIZE + self.TILE_SIZE // 2 + self.UI_HEIGHT
            )
            angle_rad = math.radians(laser["angle"])
            # Raycast to find end point
            end_pos_px = (
                start_pos_px[0] + math.cos(angle_rad) * self.SCREEN_WIDTH * 2,
                start_pos_px[1] + math.sin(angle_rad) * self.SCREEN_WIDTH * 2
            )
            
            if robot_rect.clipline(start_pos_px, end_pos_px):
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        grid_surface = self.screen.subsurface(pygame.Rect(0, self.UI_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.UI_HEIGHT))
        
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.TILE_SIZE):
            pygame.draw.line(grid_surface, self.COLOR_GRID, (x, 0), (x, grid_surface.get_height()))
        for y in range(0, grid_surface.get_height(), self.TILE_SIZE):
            pygame.draw.line(grid_surface, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw walls
        for x, y in self.walls:
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(grid_surface, self.COLOR_WALL, rect)

        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(ex * self.TILE_SIZE, ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(grid_surface, self.COLOR_EXIT, exit_rect, border_radius=4)
        pygame.draw.rect(grid_surface, self.COLOR_EXIT_GLOW, exit_rect.inflate(4, 4), 1, border_radius=6)

        # Draw lasers
        for laser in self.lasers:
            self._render_laser(grid_surface, laser)

        # Draw robot
        rx, ry = self.robot_pos
        robot_rect = pygame.Rect(rx * self.TILE_SIZE, ry * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(grid_surface, self.COLOR_ROBOT, robot_rect.inflate(-6, -6), border_radius=4)
        pygame.draw.rect(grid_surface, self.COLOR_ROBOT_GLOW, robot_rect.inflate(-2, -2), 1, border_radius=6)
        
    def _render_laser(self, surface, laser):
        start_pos = (
            laser["pos"][0] * self.TILE_SIZE + self.TILE_SIZE // 2,
            laser["pos"][1] * self.TILE_SIZE + self.TILE_SIZE // 2
        )
        angle_rad = math.radians(laser["angle"])
        
        # Raycast to find the wall intersection point for drawing
        end_pos = list(start_pos)
        for _ in range(self.SCREEN_WIDTH): # Max distance
            end_pos[0] += math.cos(angle_rad)
            end_pos[1] += math.sin(angle_rad)
            grid_x, grid_y = int(end_pos[0] // self.TILE_SIZE), int(end_pos[1] // self.TILE_SIZE)
            if (grid_x, grid_y) in self.walls:
                break
        
        # Draw glow
        pygame.draw.line(surface, self.COLOR_LASER_GLOW, start_pos, end_pos, 5)
        # Draw core beam
        pygame.draw.aaline(surface, self.COLOR_LASER, start_pos, end_pos, 1)

        # Draw source
        pulse_size = int(3 + 2 * math.sin(self.steps * 0.3))
        pygame.gfxdraw.filled_circle(surface, int(start_pos[0]), int(start_pos[1]), pulse_size + 2, self.COLOR_LASER_GLOW)
        pygame.gfxdraw.filled_circle(surface, int(start_pos[0]), int(start_pos[1]), pulse_size, self.COLOR_LASER_SOURCE)
        pygame.gfxdraw.aacircle(surface, int(start_pos[0]), int(start_pos[1]), pulse_size, self.COLOR_LASER)

        # Flash effect before rotation
        if self.steps % self.LASER_ROTATION_INTERVAL == self.LASER_ROTATION_INTERVAL - 1:
            pygame.gfxdraw.filled_circle(surface, int(start_pos[0]), int(start_pos[1]), 8, (255, 255, 255, 50))


    def _render_ui(self):
        ui_surface = self.screen.subsurface(pygame.Rect(0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        ui_surface.fill(self.COLOR_BG)
        pygame.draw.line(ui_surface, self.COLOR_GRID, (0, self.UI_HEIGHT - 1), (self.SCREEN_WIDTH, self.UI_HEIGHT - 1))

        # Render Moves
        moves_text = self.font_main.render(f"MOVES: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        ui_surface.blit(moves_text, (10, (self.UI_HEIGHT - moves_text.get_height()) // 2))

        # Render Score
        score_text = self.font_main.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(centerx=self.SCREEN_WIDTH / 2, centery=self.UI_HEIGHT / 2)
        ui_surface.blit(score_text, score_rect)

        # Render Level
        level_text = self.font_main.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(right=self.SCREEN_WIDTH - 10, centery=self.UI_HEIGHT / 2)
        ui_surface.blit(level_text, level_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "robot_pos": self.robot_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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


if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Laser Grid Maze")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    
    # --- Action mapping for human play ---
    # action = [movement, space, shift]
    action = [0, 0, 0] 
    
    print(env.game_description)
    print(env.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Reset on 'r' key
                    print("Resetting environment.")
                    obs, info = env.reset()
                    action = [0, 0, 0]
                    continue

                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

                # Reset action after processing
                action = [0, 0, 0]

                if terminated:
                    print("--- GAME OVER ---")
                    print(f"Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")

        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human play

    env.close()