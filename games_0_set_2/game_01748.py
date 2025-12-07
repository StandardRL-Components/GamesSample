
# Generated: 2025-08-27T18:09:21.784238
# Source Brief: brief_01748.md
# Brief Index: 1748

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import collections
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the robot. Reach the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a robot through a procedurally generated maze to reach the exit. "
        "Each successful run increases the maze's complexity."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()

        # Visuals
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_WALL = (60, 60, 70)
        self.COLOR_PATH = (40, 45, 50)
        self.COLOR_ROBOT = (50, 150, 255)
        self.COLOR_ROBOT_GLOW = (150, 200, 255)
        self.COLOR_EXIT = (50, 200, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_OVERLAY = (0, 0, 0, 180)
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state
        self.level_wins = 0
        self.max_steps = 1000

        # Initialize state variables
        self.np_random = None # Set in reset
        self.robot_pos = None
        self.exit_pos = None
        self.obstacles = None
        self.grid_w = None
        self.grid_h = None
        self.path_taken = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_message = ""
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.path_taken = []
        self.win_message = ""
        
        self._generate_maze()
        self.path_taken.append(self.robot_pos)
        
        return self._get_observation(), self._get_info()

    def _generate_maze(self):
        # Difficulty scaling
        self.grid_w = min(20, 10 + 2 * (self.level_wins // 5))
        self.grid_h = self.grid_w
        max_possible_obstacles = int((self.grid_w * self.grid_h) * 0.6)
        num_obstacles = min(max_possible_obstacles, 10 + (self.level_wins // 2))

        # Generate start and end points
        while True:
            self.robot_pos = (self.np_random.integers(0, self.grid_w), self.np_random.integers(0, self.grid_h))
            self.exit_pos = (self.np_random.integers(0, self.grid_w), self.np_random.integers(0, self.grid_h))
            if self.robot_pos != self.exit_pos:
                break
        
        # Find a guaranteed path using Breadth-First Search (BFS)
        q = collections.deque([(self.robot_pos, [self.robot_pos])])
        visited = {self.robot_pos}
        solution_path = None
        
        while q:
            (x, y), path = q.popleft()
            if (x, y) == self.exit_pos:
                solution_path = path
                break
            
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = list(path)
                    new_path.append((nx, ny))
                    q.append(((nx, ny), new_path))
        
        if solution_path is None: # Should be virtually impossible on an empty grid
            self.reset() # Failsafe, try again
            return

        # Place obstacles in cells not on the solution path
        all_cells = set((x, y) for x in range(self.grid_w) for y in range(self.grid_h))
        path_cells = set(solution_path)
        obstacle_candidates = list(all_cells - path_cells)
        
        if len(obstacle_candidates) >= num_obstacles:
            indices = self.np_random.choice(len(obstacle_candidates), num_obstacles, replace=False)
            self.obstacles = {obstacle_candidates[i] for i in indices}
        else:
            self.obstacles = set(obstacle_candidates)
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.steps += 1
        reward = -0.1  # Cost of taking a step
        terminated = False

        if movement != 0: # Process movement
            # sfx: robot_move
            px, py = self.robot_pos
            if movement == 1: py -= 1  # Up
            elif movement == 2: py += 1 # Down
            elif movement == 3: px -= 1 # Left
            elif movement == 4: px += 1 # Right
            
            next_pos = (px, py)
            
            # Check for collisions
            if not (0 <= px < self.grid_w and 0 <= py < self.grid_h) or next_pos in self.obstacles:
                # sfx: collision_wall
                reward = -10.0
                terminated = True
                self.win_message = "COLLISION!"
            else:
                self.robot_pos = next_pos
                if self.robot_pos not in self.path_taken:
                    self.path_taken.append(self.robot_pos)

                # Check for win condition
                if self.robot_pos == self.exit_pos:
                    # sfx: victory_fanfare
                    reward = 100.0
                    terminated = True
                    self.level_wins += 1
                    self.win_message = "VICTORY!"

        # Check for max steps termination
        if not terminated and self.steps >= self.max_steps:
            # sfx: failure_sound
            reward = -10.0 # Penalty for running out of time
            terminated = True
            self.win_message = "OUT OF STEPS"

        self.score += reward
        self.game_over = terminated
        
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
        cell_w = self.width / self.grid_w
        cell_h = self.height / self.grid_h

        # Draw path taken
        for pos in self.path_taken:
            px, py = pos[0] * cell_w, pos[1] * cell_h
            pygame.draw.rect(self.screen, self.COLOR_PATH, (px, py, cell_w, cell_h))

        # Draw obstacles
        for pos in self.obstacles:
            px, py = pos[0] * cell_w, pos[1] * cell_h
            pygame.draw.rect(self.screen, self.COLOR_WALL, (px, py, cell_w, cell_h))

        # Draw exit
        ex, ey = self.exit_pos[0] * cell_w, self.exit_pos[1] * cell_h
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex, ey, cell_w, cell_h))
        
        # Draw robot with glow
        rx, ry = self.robot_pos[0] * cell_w, self.robot_pos[1] * cell_h
        glow_size_w = cell_w * 1.4
        glow_size_h = cell_h * 1.4
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_GLOW, (rx - (glow_size_w - cell_w)/2, ry - (glow_size_h - cell_h)/2, glow_size_w, glow_size_h), border_radius=int(cell_w/4))
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, (rx, ry, cell_w, cell_h), border_radius=int(cell_w/5))

    def _render_ui(self):
        # Info text
        steps_text = self.font_small.render(f"Steps: {self.steps}/{self.max_steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 10))
        
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(score_text, score_rect)

        level_text = self.font_small.render(f"Level: {self.level_wins + 1}", True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(midtop=(self.width // 2, 10))
        self.screen.blit(level_text, level_rect)
        
        # Game over screen
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level_wins,
            "robot_pos": self.robot_pos,
            "exit_pos": self.exit_pos,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Maze Robot")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # Continuous key presses for movement
        keys = pygame.key.get_pressed()
        moved = False
        if keys[pygame.K_UP]:
            action[0] = 1
            moved = True
        elif keys[pygame.K_DOWN]:
            action[0] = 2
            moved = True
        elif keys[pygame.K_LEFT]:
            action[0] = 3
            moved = True
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            moved = True
        
        # Only step if a move was made (since it's turn-based)
        if moved and not env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Game Over! Score: {info['score']:.1f}, Steps: {info['steps']}")
                
        # Render the environment observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate
        
    env.close()