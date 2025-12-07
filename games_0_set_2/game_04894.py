
# Generated: 2025-08-28T03:20:58.181364
# Source Brief: brief_04894.md
# Brief Index: 4894

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to navigate the maze. Find the green exit before you run out of steps."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Navigate a procedurally generated maze to find the exit within a limited number of steps. Each move counts!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Class-level persistent state ---
    # This will persist across resets, allowing for difficulty progression
    total_wins = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 100

        # Maze generation parameters
        self.MAZE_W = 31  # Must be odd
        self.MAZE_H = 19  # Must be odd
        self.CELL_SIZE = 20
        self.RENDER_X_OFFSET = (self.SCREEN_WIDTH - self.MAZE_W * self.CELL_SIZE) // 2
        self.RENDER_Y_OFFSET = (self.SCREEN_HEIGHT - self.MAZE_H * self.CELL_SIZE) // 2

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (45, 50, 60)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_ACCENT = (150, 200, 255)
        self.COLOR_EXIT = (50, 255, 150)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 24)
        except IOError:
            # Fallback if default font is not found
            self.font_large = pygame.font.SysFont("sans", 36)
            self.font_small = pygame.font.SysFont("sans", 24)

        # --- Game State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.maze = []
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.np_random = None

        # Initialize state for the first time
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # If the last game was a win, increment the total wins for difficulty scaling
        if self.win:
            GameEnv.total_wins += 1

        # Initialize all game state for a new episode
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False

        # Generate a new maze
        # Difficulty increases every 5 wins by adding more loops/complexity
        complexity = GameEnv.total_wins // 5
        self.maze, self.player_pos, self.exit_pos = self._generate_maze(self.MAZE_W, self.MAZE_H, complexity)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # If game is over, do nothing
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        # Every action, valid or not, costs a step
        self.steps += 1
        reward = -0.1  # Small penalty for each step taken

        # --- Player Movement ---
        px, py = self.player_pos
        if movement == 1:  # Up
            py -= 1
        elif movement == 2:  # Down
            py += 1
        elif movement == 3:  # Left
            px -= 1
        elif movement == 4:  # Right
            px += 1

        # Check for valid move (within bounds and not a wall)
        if 0 <= px < self.MAZE_W and 0 <= py < self.MAZE_H and self.maze[py][px] == 0:
            self.player_pos = (px, py)
            # Placeholder for move sound: sfx: player_move.wav

        # --- Check Game Conditions ---
        # 1. Win condition
        if self.player_pos == self.exit_pos:
            self.win = True
            self.game_over = True
            reward = 100.0
            # Placeholder for win sound: sfx: win_jingle.wav

        # 2. Lose condition
        if not self.win and self.steps >= self.MAX_STEPS:
            self.game_over = True
            reward = -10.0
            # Placeholder for lose sound: sfx: lose_buzzer.wav
        
        self.score += reward
        terminated = self.game_over

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _generate_maze(self, width, height, complexity):
        # Maze grid: 1 for wall, 0 for path
        maze = np.ones((height, width), dtype=np.uint8)
        
        # Use randomized depth-first search
        stack = []
        start_x, start_y = (1, 1) # Start top-left
        
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            
            # Check potential neighbors (2 cells away)
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                idx = self.np_random.choice(len(neighbors))
                nx, ny = neighbors[idx]
                
                # Carve path
                maze[ny, nx] = 0
                maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Add complexity by removing some dead-ends to create loops
        # The number of loops is based on the complexity parameter
        for _ in range(complexity):
            # Find a random wall that is not on the border
            wall_y, wall_x = self.np_random.integers(1, height - 1), self.np_random.integers(1, width - 1)
            # Ensure it's a wall with paths on opposite sides to create a loop
            if maze[wall_y, wall_x] == 1:
                # Horizontal wall between two vertical paths
                if wall_x % 2 == 1 and maze[wall_y, wall_x-1] == 0 and maze[wall_y, wall_x+1] == 0:
                     maze[wall_y, wall_x] = 0
                # Vertical wall between two horizontal paths
                elif wall_y % 2 == 1 and maze[wall_y-1, wall_x] == 0 and maze[wall_y+1, wall_x] == 0:
                     maze[wall_y, wall_x] = 0

        player_start = (1, 1)
        exit_pos = (width - 2, height - 2)
        
        return maze, player_start, exit_pos

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

    def _render_game(self):
        # Draw the maze
        for y in range(self.MAZE_H):
            for x in range(self.MAZE_W):
                if self.maze[y][x] == 1:
                    rect = pygame.Rect(
                        self.RENDER_X_OFFSET + x * self.CELL_SIZE,
                        self.RENDER_Y_OFFSET + y * self.CELL_SIZE,
                        self.CELL_SIZE,
                        self.CELL_SIZE
                    )
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw the exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(
            self.RENDER_X_OFFSET + ex * self.CELL_SIZE,
            self.RENDER_Y_OFFSET + ey * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Draw the player
        px, py = self.player_pos
        player_rect_outer = pygame.Rect(
            self.RENDER_X_OFFSET + px * self.CELL_SIZE,
            self.RENDER_Y_OFFSET + py * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect_outer)
        
        # Add an inner highlight for better visibility
        player_rect_inner = player_rect_outer.inflate(-self.CELL_SIZE * 0.4, -self.CELL_SIZE * 0.4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_ACCENT, player_rect_inner, border_radius=2)

    def _render_ui(self):
        # Helper to draw text with a shadow
        def draw_text(text, font, color, x, y, center=False):
            shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surface = font.render(text, True, color)
            
            shadow_rect = shadow_surface.get_rect()
            text_rect = text_surface.get_rect()

            if center:
                shadow_rect.center = (x + 2, y + 2)
                text_rect.center = (x, y)
            else:
                shadow_rect.topleft = (x + 2, y + 2)
                text_rect.topleft = (x, y)

            self.screen.blit(shadow_surface, shadow_rect)
            self.screen.blit(text_surface, text_rect)

        # Display remaining steps
        steps_left = max(0, self.MAX_STEPS - self.steps)
        draw_text(f"Steps: {steps_left}", self.font_small, self.COLOR_TEXT, 20, 10)

        # Display score
        draw_text(f"Score: {self.score:.1f}", self.font_small, self.COLOR_TEXT, self.SCREEN_WIDTH - 150, 10)
        
        # Display win/loss message
        if self.game_over:
            message = "YOU WIN!" if self.win else "OUT OF STEPS"
            color = self.COLOR_EXIT if self.win else (255, 100, 100)
            draw_text(message, self.font_large, color, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "total_wins": GameEnv.total_wins,
            "win": self.win
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                should_step = True
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    should_step = False
                elif event.key == pygame.K_q: # Quit
                    running = False
                    should_step = False
                else:
                    should_step = False

                # Since auto_advance is False, we step on key press
                if should_step:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
                    if terminated:
                        print("\n--- GAME OVER ---")
                        print(f"Final Score: {info['score']:.1f}, Total Wins: {info['total_wins']}")
                        print("Press 'R' to play again or 'Q' to quit.")

        # Draw the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate

    env.close()