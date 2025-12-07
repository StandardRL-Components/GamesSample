
# Generated: 2025-08-27T15:37:56.519618
# Source Brief: brief_01032.md
# Brief Index: 1032

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import random
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to navigate the maze. Reach the red exit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze to find the exit before the timer expires. The maze gets larger on each success."
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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Visuals
        self.COLOR_BG = (100, 100, 100) # Light Gray floor
        self.COLOR_WALL = (40, 40, 40)   # Dark Gray walls
        self.COLOR_PLAYER = (50, 200, 50) # Bright Green
        self.COLOR_EXIT = (200, 50, 50)   # Bright Red
        self.COLOR_TEXT = (255, 255, 255) # White
        
        # Game State
        self.maze_width = 0
        self.maze_height = 0
        self.difficulty_level = 0
        self.max_difficulty = 7  # 10 + 7*2 = 24. Capped at 25x25.
        self.maze = []
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.time_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def _generate_maze(self, width, height):
        # Using iterative depth-first search (recursive backtracking)
        maze = [[{'N': True, 'S': True, 'W': True, 'E': True} for _ in range(width)] for _ in range(height)]
        visited = [[False for _ in range(width)] for _ in range(height)]
        
        # Start at a random cell
        start_r, start_c = (0, 0) # For consistency, let's start at top-left
        stack = [(start_r, start_c)]
        visited[start_r][start_c] = True

        while stack:
            cr, cc = stack[-1]
            
            # Find unvisited neighbors
            neighbors = []
            if cr > 0 and not visited[cr - 1][cc]: neighbors.append(('N', cr - 1, cc))
            if cr < height - 1 and not visited[cr + 1][cc]: neighbors.append(('S', cr + 1, cc))
            if cc > 0 and not visited[cr][cc - 1]: neighbors.append(('W', cr, cc - 1))
            if cc < width - 1 and not visited[cr][cc + 1]: neighbors.append(('E', cr, cc + 1))

            if neighbors:
                # Choose a random neighbor
                direction, nr, nc = neighbors[self.np_random.integers(len(neighbors))]
                
                # Knock down walls between current cell and neighbor
                if direction == 'N':
                    maze[cr][cc]['N'] = False
                    maze[nr][nc]['S'] = False
                elif direction == 'S':
                    maze[cr][cc]['S'] = False
                    maze[nr][nc]['N'] = False
                elif direction == 'W':
                    maze[cr][cc]['W'] = False
                    maze[nr][nc]['E'] = False
                elif direction == 'E':
                    maze[cr][cc]['E'] = False
                    maze[nr][nc]['W'] = False
                
                visited[nr][nc] = True
                stack.append((nr, nc))
            else:
                # Backtrack
                stack.pop()
        
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Set maze dimensions based on difficulty
        self.maze_width = min(25, 10 + self.difficulty_level * 2)
        self.maze_height = min(25, 10 + self.difficulty_level * 2)
        
        # Generate a new maze
        self.maze = self._generate_maze(self.maze_width, self.maze_height)
        
        # Set start and end points (guaranteed to be different and solvable)
        self.player_pos = (0, 0)
        self.exit_pos = (self.maze_height - 1, self.maze_width - 1)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.time_remaining = 60
        self.game_over = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _move_player(self, movement):
        y, x = self.player_pos
        
        # 1=up, 2=down, 3=left, 4=right
        if movement == 1 and y > 0 and not self.maze[y][x]['N']:
            self.player_pos = (y - 1, x)
        elif movement == 2 and y < self.maze_height - 1 and not self.maze[y][x]['S']:
            self.player_pos = (y + 1, x)
        elif movement == 3 and x > 0 and not self.maze[y][x]['W']:
            self.player_pos = (y, x - 1)
        elif movement == 4 and x < self.maze_width - 1 and not self.maze[y][x]['E']:
            self.player_pos = (y, x + 1)
        # No-op (movement == 0) or illegal move does nothing.

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self._move_player(movement)
        
        # Update game logic
        self.steps += 1
        self.time_remaining -= 1
        
        reward = -0.1
        terminated = False
        
        # Check for win condition
        if self.player_pos == self.exit_pos:
            reward = 10.0
            terminated = True
            # Increase difficulty for the next round
            self.difficulty_level = min(self.max_difficulty, self.difficulty_level + 1)
        
        # Check for loss condition (time out)
        if self.time_remaining <= 0:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _render_game(self):
        # Calculate cell size and maze offset to center it
        padding = 40
        maze_render_width = self.screen.get_width() - 2 * padding
        maze_render_height = self.screen.get_height() - 2 * padding
        
        cell_w = maze_render_width / self.maze_width
        cell_h = maze_render_height / self.maze_height
        cell_size = min(cell_w, cell_h)

        total_maze_w = cell_size * self.maze_width
        total_maze_h = cell_size * self.maze_height
        offset_x = (self.screen.get_width() - total_maze_w) / 2
        offset_y = (self.screen.get_height() - total_maze_h) / 2

        # Render Exit
        exit_r, exit_c = self.exit_pos
        exit_rect = pygame.Rect(
            int(offset_x + exit_c * cell_size), 
            int(offset_y + exit_r * cell_size), 
            math.ceil(cell_size), 
            math.ceil(cell_size)
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        
        # Render Player
        player_r, player_c = self.player_pos
        player_rect = pygame.Rect(
            int(offset_x + player_c * cell_size), 
            int(offset_y + player_r * cell_size), 
            math.ceil(cell_size), 
            math.ceil(cell_size)
        )
        # Inset the player rect slightly to not overlap walls
        player_rect.inflate_ip(-cell_size * 0.3, -cell_size * 0.3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=int(max(1, cell_size * 0.15)))

        # Render maze walls
        wall_thickness = 2
        for r in range(self.maze_height):
            for c in range(self.maze_width):
                x1 = int(offset_x + c * cell_size)
                y1 = int(offset_y + r * cell_size)
                x2 = int(x1 + cell_size)
                y2 = int(y1 + cell_size)
                
                if self.maze[r][c]['N']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (x1, y1), (x2, y1), wall_thickness)
                if self.maze[r][c]['S']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (x1, y2), (x2, y2), wall_thickness)
                if self.maze[r][c]['W']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (x1, y1), (x1, y2), wall_thickness)
                if self.maze[r][c]['E']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (x2, y1), (x2, y2), wall_thickness)

    def _render_ui(self):
        time_text = f"Time: {self.time_remaining}"
        score_text = f"Score: {self.score:.1f}"

        time_surface = self.font.render(time_text, True, self.COLOR_TEXT)
        score_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        
        # Shadow effect for readability
        time_shadow = self.font.render(time_text, True, (0,0,0))
        score_shadow = self.font.render(score_text, True, (0,0,0))
        self.screen.blit(time_shadow, (self.screen.get_width() - time_surface.get_width() - 9, 11))
        self.screen.blit(score_shadow, (11, 11))

        self.screen.blit(time_surface, (self.screen.get_width() - time_surface.get_width() - 10, 10))
        self.screen.blit(score_surface, (10, 10))
        
        if self.game_over:
            message = "SUCCESS!" if self.player_pos == self.exit_pos else "TIME UP!"
            msg_surface = self.font.render(message, True, self.COLOR_TEXT)
            msg_shadow = self.font.render(message, True, (0,0,0))
            
            msg_x = (self.screen.get_width() - msg_surface.get_width()) / 2
            msg_y = (self.screen.get_height() - msg_surface.get_height()) / 2
            
            self.screen.blit(msg_shadow, (int(msg_x + 2), int(msg_y + 2)))
            self.screen.blit(msg_surface, (int(msg_x), int(msg_y)))

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
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "difficulty": self.difficulty_level,
        }

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
        assert self.player_pos != self.exit_pos
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert self.time_remaining == 59
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Use a separate screen for rendering if playing directly
    render_screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Maze Runner")
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Since auto_advance is False, we only step when a key is pressed
            # or on a timer to allow for visual updates even with no input.
            # Let's use a simple key press trigger for turn-based play.
            
            # We need to detect a key press, not just held down
            # For simplicity in this test runner, we will just step on any key
            # A better human player would use pygame.KEYDOWN events.
            
            # Let's step once per frame if a key is held, for simple playability
            if action[0] != 0:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
        else:
            # If terminated, wait for a key to reset
            keys = pygame.key.get_pressed()
            if any(keys):
                print("Resetting game...")
                obs, info = env.reset()
                terminated = False

        # Render the observation to the display screen
        # Need to transpose back for pygame's display format
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate for human play
        env.clock.tick(10) # Slower for turn-based feel
        
    pygame.quit()