import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to navigate the maze."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze to find the green exit before time runs out."
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
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font = pygame.font.Font(None, 32)
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_WALL = (60, 80, 100)
        self.COLOR_PATH = (210, 220, 230)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_TEXT = (255, 255, 255)
        
        # Game state variables
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.maze_width = 0
        self.maze_height = 0
        self.level = 1
        self.time_remaining = 0
        self.initial_time = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_steps = 1000
        
        # Initialize state variables
        self.reset()
        
        # Run validation
        # self.validate_implementation() # Commented out for submission, can be re-enabled for testing
    
    def _generate_maze(self, width, height):
        maze = [[{'visited': False, 'walls': [True, True, True, True]} for _ in range(width)] for _ in range(height)]
        
        stack = []
        start_cell = (0, 0)
        maze[start_cell[1]][start_cell[0]]['visited'] = True
        stack.append(start_cell)

        while stack:
            x, y = stack[-1]
            
            neighbors = []
            # Check top neighbor
            if y > 0 and not maze[y - 1][x]['visited']:
                neighbors.append((x, y - 1))
            # Check right neighbor
            if x < width - 1 and not maze[y][x + 1]['visited']:
                neighbors.append((x + 1, y))
            # Check bottom neighbor
            if y < height - 1 and not maze[y + 1][x]['visited']:
                neighbors.append((x, y + 1))
            # Check left neighbor
            if x > 0 and not maze[y][x - 1]['visited']:
                neighbors.append((x - 1, y))

            if neighbors:
                # FIX: self.np_random.choice on a list of tuples returns a numpy array of shape (2,)
                # which can be unpacked into nx, ny. The original code tried to unpack an integer.
                nx, ny = self.np_random.choice(neighbors)
                
                # Remove walls
                if nx == x + 1: # Right
                    maze[y][x]['walls'][1] = False
                    maze[ny][nx]['walls'][3] = False
                elif nx == x - 1: # Left
                    maze[y][x]['walls'][3] = False
                    maze[ny][nx]['walls'][1] = False
                elif ny == y + 1: # Down
                    maze[y][x]['walls'][2] = False
                    maze[ny][nx]['walls'][0] = False
                elif ny == y - 1: # Up
                    maze[y][x]['walls'][0] = False
                    maze[ny][nx]['walls'][2] = False
                
                maze[ny][nx]['visited'] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Difficulty scaling: Increase maze size with level, capped to fit screen
        self.maze_width = min(38, 8 + 2 * self.level)
        self.maze_height = min(24, 8 + 2 * self.level)
        
        self.initial_time = 60 + (self.level - 1) * 15
        self.time_remaining = self.initial_time
        
        self.maze = self._generate_maze(self.maze_width, self.maze_height)
        self.player_pos = (0, 0)
        self.exit_pos = (self.maze_width - 1, self.maze_height - 1)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.steps += 1
        reward = -0.1  # Step penalty
        
        old_pos = self.player_pos
        px, py = self.player_pos
        
        moved = False
        if movement == 1 and py > 0 and not self.maze[py][px]['walls'][0]: # Up
            self.player_pos = (px, py - 1)
            moved = True
        elif movement == 2 and py < self.maze_height - 1 and not self.maze[py][px]['walls'][2]: # Down
            self.player_pos = (px, py + 1)
            moved = True
        elif movement == 3 and px > 0 and not self.maze[py][px]['walls'][3]: # Left
            self.player_pos = (px - 1, py)
            moved = True
        elif movement == 4 and px < self.maze_width - 1 and not self.maze[py][px]['walls'][1]: # Right
            self.player_pos = (px + 1, py)
            moved = True

        if moved:
            # Dead end penalty
            new_px, new_py = self.player_pos
            if self.player_pos != self.exit_pos:
                num_walls = sum(self.maze[new_py][new_px]['walls'])
                if num_walls == 3:
                    reward -= 1.0
                    # Sound effect placeholder: # sfx_dead_end_thud

        # Decrement time only on a move or attempted move (not no-op)
        if movement != 0:
            self.time_remaining -= 1

        # Check for win condition
        if self.player_pos == self.exit_pos:
            win_bonus = 100.0 + (max(0, self.time_remaining) / self.initial_time * 50.0)
            reward += win_bonus
            self.game_over = True
            self.level += 1
            # Sound effect placeholder: # sfx_level_complete

        self.score += reward
        
        # Check for termination conditions
        terminated = self.game_over or self.time_remaining <= 0 or self.steps >= self.max_steps
        if terminated and not self.game_over:
            self.game_over = True # Ensure game over state is consistent
            # Sound effect placeholder: # sfx_time_up
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _render_game(self):
        # Calculate maze rendering dimensions to fit and center
        render_area_height = self.screen_height - 40 # Reserve space for UI
        
        cell_dim_w = self.screen_width / self.maze_width
        cell_dim_h = render_area_height / self.maze_height
        self.cell_size = int(min(cell_dim_w, cell_dim_h))
        
        maze_render_width = self.cell_size * self.maze_width
        maze_render_height = self.cell_size * self.maze_height
        
        self.offset_x = (self.screen_width - maze_render_width) // 2
        self.offset_y = (self.screen_height - maze_render_height) // 2

        # Draw paths
        pygame.draw.rect(self.screen, self.COLOR_PATH, 
                         (self.offset_x, self.offset_y, maze_render_width, maze_render_height))

        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(
            self.offset_x + ex * self.cell_size,
            self.offset_y + ey * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Draw walls
        wall_thickness = max(1, self.cell_size // 10)
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                cell = self.maze[y][x]
                px = self.offset_x + x * self.cell_size
                py = self.offset_y + y * self.cell_size
                
                if cell['walls'][0]: # Top
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.cell_size, py), wall_thickness)
                if cell['walls'][1]: # Right
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.cell_size, py), (px + self.cell_size, py + self.cell_size), wall_thickness)
                if cell['walls'][2]: # Bottom
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.cell_size), (px + self.cell_size, py + self.cell_size), wall_thickness)
                if cell['walls'][3]: # Left
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + self.cell_size), wall_thickness)

        # Draw player
        px, py = self.player_pos
        player_center_x = self.offset_x + int((px + 0.5) * self.cell_size)
        player_center_y = self.offset_y + int((py + 0.5) * self.cell_size)
        player_radius = int(self.cell_size * 0.35)
        
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Time display
        time_text = f"Time: {max(0, self.time_remaining)}"
        time_surf = self.font.render(time_text, True, self.COLOR_TEXT)
        time_rect = time_surf.get_rect(topright=(self.screen_width - 10, 5))
        self.screen.blit(time_surf, time_rect)
        
        # Score display
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topleft=(10, 5))
        self.screen.blit(score_surf, score_rect)

        # Level display
        level_text = f"Level: {self.level}"
        level_surf = self.font.render(level_text, True, self.COLOR_TEXT)
        level_rect = level_surf.get_rect(midtop=(self.screen_width // 2, 5))
        self.screen.blit(level_surf, level_rect)

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
            "time_remaining": self.time_remaining,
            "player_pos": self.player_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("✓ Running implementation validation...")
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
    # This block allows you to play the game manually for testing
    # To run, you may need to unset the dummy video driver
    # comment out: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # and install pygame: pip install pygame
    
    # For automated testing, the dummy driver is fine.
    # To test manually, you need a display.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Running in headless mode. Manual play is disabled.")
        print("To play manually, comment out the line `os.environ.setdefault(\"SDL_VIDEODRIVER\", \"dummy\")`")
        # Quick test of the environment logic
        env = GameEnv()
        env.validate_implementation()
        env.reset()
        env.step(env.action_space.sample())
        env.close()
    else:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        # Create a window to display the game
        pygame.display.set_caption(env.game_description)
        screen = pygame.display.set_mode((640, 400))
        
        terminated = False
        
        # Game loop
        while not terminated:
            movement = 0 # No-op by default
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        movement = 1
                    elif event.key == pygame.K_DOWN:
                        movement = 2
                    elif event.key == pygame.K_LEFT:
                        movement = 3
                    elif event.key == pygame.K_RIGHT:
                        movement = 4
                    elif event.key == pygame.K_r: # Reset on 'r' key
                        obs, info = env.reset()
                        print(f"Game Reset. Level: {info['level']}")
                        continue
                    elif event.key == pygame.K_q: # Quit on 'q' key
                        terminated = True
                        
            if movement != 0:
                action = [movement, 0, 0] # space/shift not used
                obs, reward, term, trunc, info = env.step(action)
                terminated = term
                
                if reward != -0.1:
                    print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {term}")

                if terminated:
                    print(f"Episode Finished! Final Score: {info['score']:.2f}")

            # Render the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # If game is over, wait for a moment before resetting
            if terminated and info.get('player_pos') == env.exit_pos:
                pygame.time.wait(2000) # Pause on win
                obs, info = env.reset()
                terminated = False
            elif terminated:
                pygame.time.wait(2000) # Pause on loss
                env.level = 1 # Reset level on loss
                obs, info = env.reset()
                terminated = False

        env.close()