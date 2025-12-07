import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


# Set the SDL_VIDEODRIVER to "dummy" to ensure Pygame runs headless
os.environ['SDL_VIDEODRIVER'] = 'dummy'

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the robot one cell at a time. Reach the green exit square."
    )

    game_description = (
        "Navigate a robot through a procedurally generated maze to reach the exit. "
        "Avoid blue obstacles. Find the shortest path to maximize your score!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 32
    GRID_ROWS = 20
    CELL_SIZE = 20
    MAX_STEPS = 500
    
    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_ROBOT_MAIN = (255, 50, 50)
    COLOR_ROBOT_GLOW = (255, 50, 50, 50)
    COLOR_OBSTACLE = (50, 150, 255)
    COLOR_EXIT = (50, 255, 150)
    COLOR_TRAIL = (255, 100, 100, 150)
    COLOR_TEXT = (220, 220, 220)
    
    # --- Difficulty ---
    INITIAL_OBSTACLES = 10
    MAX_OBSTACLES = 80 
    OBSTACLE_INCREMENT = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        self.num_obstacles = self.INITIAL_OBSTACLES
        
        # State variables are initialized in reset()
        self.robot_pos = None
        self.last_robot_pos = None
        self.exit_pos = None
        self.obstacles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.terminal_particles = []

        self.validate_implementation()
        
    def _is_path_available(self, grid, start, end):
        """Checks for a path using Breadth-First Search."""
        q = deque([start])
        visited = {start}
        while q:
            x, y = q.popleft()
            if (x, y) == end:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS and (nx, ny) not in visited and grid[ny][nx] == 0:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def _generate_level(self):
        """Generates a level with a guaranteed path from start to exit."""
        self.robot_pos = (1, 1)
        self.exit_pos = (self.GRID_COLS - 2, self.GRID_ROWS - 2)
        
        while True:
            grid = np.zeros((self.GRID_ROWS, self.GRID_COLS))
            
            # Get all possible obstacle locations
            possible_locs = list(set((c, r) for c in range(self.GRID_COLS) for r in range(self.GRID_ROWS)))
            possible_locs.remove(self.robot_pos)
            possible_locs.remove(self.exit_pos)
            
            # Place obstacles
            num_to_place = min(self.num_obstacles, len(possible_locs))
            chosen_indices = self.np_random.choice(len(possible_locs), size=num_to_place, replace=False)
            self.obstacles = {possible_locs[i] for i in chosen_indices}

            for ox, oy in self.obstacles:
                grid[oy][ox] = 1

            if self._is_path_available(grid, self.robot_pos, self.exit_pos):
                break # Valid level found

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._generate_level()
        
        self.last_robot_pos = self.robot_pos
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.terminal_particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        # space_held = action[1] == 1 # unused
        # shift_held = action[2] == 1 # unused
        
        self.last_robot_pos = self.robot_pos
        
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        reward = -0.1 # Cost of living
        terminated = False
        truncated = False
        
        if movement != 0:
            new_pos = (self.robot_pos[0] + dx, self.robot_pos[1] + dy)
            
            # Boundary check
            if 0 <= new_pos[0] < self.GRID_COLS and 0 <= new_pos[1] < self.GRID_ROWS:
                self.robot_pos = new_pos
        
        # Check for events
        if self.robot_pos == self.exit_pos:
            reward = 100.0
            terminated = True
            self.game_over = True
            self.num_obstacles = min(self.MAX_OBSTACLES, self.num_obstacles + self.OBSTACLE_INCREMENT)
            self._create_terminal_effect(self.exit_pos, self.COLOR_EXIT)

        elif self.robot_pos in self.obstacles:
            reward = -10.0
            terminated = True
            self.game_over = True
            self._create_terminal_effect(self.robot_pos, self.COLOR_OBSTACLE)
            
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True # Use terminated for time limit in this context
            self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
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
            "obstacles": self.num_obstacles
        }
    
    def _create_terminal_effect(self, pos, color):
        """Generates particles for a win/loss event."""
        cx = (pos[0] + 0.5) * self.CELL_SIZE
        cy = (pos[1] + 0.5) * self.CELL_SIZE
        self.terminal_particles = []
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            px = cx + math.cos(angle) * speed * self.np_random.uniform(1, 4)
            py = cy + math.sin(angle) * speed * self.np_random.uniform(1, 4)
            radius = self.np_random.uniform(1, 4)
            self.terminal_particles.append({'pos': (px, py), 'radius': radius, 'color': color})

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_COLS + 1):
            x = i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for i in range(self.GRID_ROWS + 1):
            y = i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw exit (only if initialized)
        if self.exit_pos:
            ex, ey = self.exit_pos
            exit_rect = pygame.Rect(ex * self.CELL_SIZE, ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
            pygame.gfxdraw.rectangle(self.screen, exit_rect, (255,255,255,80))

        # Draw obstacles (only if initialized)
        if self.obstacles:
            for ox, oy in self.obstacles:
                obs_rect = pygame.Rect(ox * self.CELL_SIZE, oy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect)
                pygame.draw.rect(self.screen, (255,255,255,40), obs_rect.inflate(2,2), 1)

        # Draw movement trail (only if positions are valid)
        if self.last_robot_pos and self.robot_pos and self.last_robot_pos != self.robot_pos:
            start_center = ((self.last_robot_pos[0] + 0.5) * self.CELL_SIZE, (self.last_robot_pos[1] + 0.5) * self.CELL_SIZE)
            end_center = ((self.robot_pos[0] + 0.5) * self.CELL_SIZE, (self.robot_pos[1] + 0.5) * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_TRAIL, start_center, end_center, 4)

        # Draw robot (only if initialized)
        if self.robot_pos:
            rx, ry = self.robot_pos
            robot_center_x = int((rx + 0.5) * self.CELL_SIZE)
            robot_center_y = int((ry + 0.5) * self.CELL_SIZE)
            
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, robot_center_x, robot_center_y, int(self.CELL_SIZE * 0.7), self.COLOR_ROBOT_GLOW)
            
            # Body
            robot_rect = pygame.Rect(rx * self.CELL_SIZE, ry * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE).inflate(-4, -4)
            pygame.draw.rect(self.screen, self.COLOR_ROBOT_MAIN, robot_rect, border_radius=3)
            pygame.draw.rect(self.screen, (255, 150, 150), robot_rect.inflate(-4,-4), border_radius=2)
        
        # Draw terminal particles
        for p in self.terminal_particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        steps_text = self.font_main.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        obs_text = self.font_small.render(f"Obstacles: {self.num_obstacles}", True, self.COLOR_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))
        self.screen.blit(obs_text, (10, 35))
        
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will use a visible display instead of the dummy one
    os.environ['SDL_VIDEODRIVER'] = 'x11'
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Robot Maze")
    
    running = True
    
    # Game loop for human play
    while running:
        action_movement = 0 # No-op default
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action_movement = 1
                elif event.key == pygame.K_DOWN:
                    action_movement = 2
                elif event.key == pygame.K_LEFT:
                    action_movement = 3
                elif event.key == pygame.K_RIGHT:
                    action_movement = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    continue
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    continue
                
                # Construct the MultiDiscrete action
                action = [action_movement, 0, 0] # space/shift are not used
                
                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated:
                    print(f"Episode finished! Final Score: {info['score']:.1f} in {info['steps']} steps.")
                    # Render the final frame with effects
                    frame = np.transpose(obs, (1, 0, 2))
                    surf = pygame.surfarray.make_surface(frame)
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    
                    pygame.time.wait(2000) # Wait 2 seconds before resetting
                    obs, info = env.reset()
                    terminated = False
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()