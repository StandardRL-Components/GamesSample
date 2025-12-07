
# Generated: 2025-08-27T18:50:56.221474
# Source Brief: brief_01976.md
# Brief Index: 1976

        
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
        "Controls: ↑↓←→ to navigate the maze."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze to reach the exit within a limited number of steps."
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
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (80, 80, 90)
        self.COLOR_PATH = (40, 40, 50)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_OUTLINE = (255, 255, 255)
        self.COLOR_EXIT = (0, 255, 100)
        self.COLOR_TEXT = (255, 255, 255)
        
        # Game state (difficulty persists across resets)
        self.maze_width = 11
        self.maze_height = 11
        self.max_maze_dim = 31

        # Initialize state variables that are reset each episode
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.path_traversed = None
        self.steps_remaining = None
        self.score = None
        self.game_over = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps_remaining = 50
        self.score = 0
        self.game_over = False
        
        self.maze = self._generate_maze(self.maze_width, self.maze_height)
        self.player_pos = np.array([1, 1])
        self.exit_pos = np.array([self.maze_width - 2, self.maze_height - 2])
        self.path_traversed = [tuple(self.player_pos)]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        moved = False
        
        if movement != 0: # A step is taken
            self.steps_remaining -= 1
            reward -= 0.1 # Small penalty for each step
            
            target_pos = self.player_pos.copy()
            if movement == 1:  # Up
                target_pos[1] -= 1
            elif movement == 2: # Down
                target_pos[1] += 1
            elif movement == 3: # Left
                target_pos[0] -= 1
            elif movement == 4: # Right
                target_pos[0] += 1

            # Check for wall collision
            if self.maze[target_pos[1], target_pos[0]] == 0:
                self.player_pos = target_pos
                moved = True
                if tuple(self.player_pos) not in self.path_traversed:
                    self.path_traversed.append(tuple(self.player_pos))
            # else: # Hit a wall, no position change
                # sound_effect: 'wall_thud'

        terminated = False
        if np.array_equal(self.player_pos, self.exit_pos):
            # Victory condition
            reward += 10 # Discover exit
            reward += 50 # Reach exit
            terminated = True
            self.game_over = True
            # Increase difficulty for the next round
            self.maze_width = min(self.max_maze_dim, self.maze_width + 2)
            self.maze_height = min(self.max_maze_dim, self.maze_height + 2)
            # sound_effect: 'level_complete'
        
        elif self.steps_remaining <= 0:
            # Failure condition
            reward = -100 # Penalty for running out of steps
            terminated = True
            self.game_over = True
            # sound_effect: 'game_over'

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
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps_remaining": self.steps_remaining,
            "player_pos": tuple(self.player_pos),
            "maze_size": (self.maze_width, self.maze_height)
        }
        
    def _render_game(self):
        screen_w, screen_h = self.screen.get_size()
        
        # Calculate cell size and centering offset
        cell_w = screen_w / self.maze_width
        cell_h = screen_h / self.maze_height
        cell_size = min(cell_w, cell_h)
        
        offset_x = (screen_w - self.maze_width * cell_size) / 2
        offset_y = (screen_h - self.maze_height * cell_size) / 2

        # Draw traversed path
        for x, y in self.path_traversed:
            rect = pygame.Rect(
                offset_x + x * cell_size,
                offset_y + y * cell_size,
                cell_size,
                cell_size
            )
            pygame.draw.rect(self.screen, self.COLOR_PATH, rect)

        # Draw maze walls
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                if self.maze[y, x] == 1:
                    rect = pygame.Rect(
                        offset_x + x * cell_size,
                        offset_y + y * cell_size,
                        cell_size,
                        cell_size
                    )
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw exit
        exit_rect = pygame.Rect(
            offset_x + self.exit_pos[0] * cell_size,
            offset_y + self.exit_pos[1] * cell_size,
            cell_size,
            cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        
        # Draw player
        player_center_x = int(offset_x + (self.player_pos[0] + 0.5) * cell_size)
        player_center_y = int(offset_y + (self.player_pos[1] + 0.5) * cell_size)
        player_radius = int(cell_size * 0.35)
        
        # Draw with anti-aliasing for smooth look
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
        # Add outline for visibility
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, player_radius + 1, self.COLOR_PLAYER_OUTLINE)

    def _render_ui(self):
        # Render steps remaining
        steps_text = f"Steps: {self.steps_remaining}"
        text_surface = self.font_ui.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Render score
        score_text = f"Score: {self.score:.1f}"
        text_surface = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(topright=(self.screen.get_width() - 10, 10))
        self.screen.blit(text_surface, text_rect)
        
        # Render game over messages
        if self.game_over:
            if np.array_equal(self.player_pos, self.exit_pos):
                msg = "YOU WIN!"
                color = self.COLOR_EXIT
            else:
                msg = "GAME OVER"
                color = (255, 50, 50)
            
            msg_surface = self.font_msg.render(msg, True, color)
            msg_rect = msg_surface.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2))
            self.screen.blit(msg_surface, msg_rect)

    def _generate_maze(self, width, height):
        # Ensure dimensions are odd
        width = width if width % 2 != 0 else width + 1
        height = height if height % 2 != 0 else height + 1
        
        # Initialize maze with walls
        maze = np.ones((height, width), dtype=np.int8)
        
        # Start carving from (1, 1)
        stack = [(1, 1)]
        maze[1, 1] = 0
        
        while stack:
            x, y = stack[-1]
            
            # Find unvisited neighbors (2 cells away)
            neighbors = []
            if y > 1 and maze[y - 2, x] == 1: neighbors.append((x, y - 2))
            if y < height - 2 and maze[y + 2, x] == 1: neighbors.append((x, y + 2))
            if x > 1 and maze[y, x - 2] == 1: neighbors.append((x - 2, y))
            if x < width - 2 and maze[y, x + 2] == 1: neighbors.append((x + 2, y))
            
            if neighbors:
                # Use numpy's seeded random generator
                self.np_random.shuffle(neighbors)
                nx, ny = neighbors[0]
                
                # Carve path to neighbor
                maze[ny, nx] = 0
                maze[(y + ny) // 2, (x + nx) // 2] = 0
                
                stack.append((nx, ny))
            else:
                # Backtrack
                stack.pop()
        
        return maze

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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run headless
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Maze Runner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)
    print(env.game_description)

    while running:
        action = [0, 0, 0] # Default action: no-op, buttons released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # In turn-based mode, we only step if an action is taken.
        # This loop waits for a key press (other than modifiers).
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            
            if terminated or truncated:
                print("Episode finished. Press 'R' to reset.")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for human play

    pygame.quit()