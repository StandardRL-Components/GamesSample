import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set SDL to dummy driver for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    """
    A top-down puzzle game where the player navigates a procedurally generated maze
    to find the exit within a limited number of steps.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ↑, ↓, ←, → to move your avatar one cell at a time."
    )

    # User-facing game description
    game_description = (
        "Navigate procedurally generated mazes to find the green exit. Each move "
        "costs one step. Reach the exit in under 50 steps to win."
    )

    # Frames advance only when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 50
        self.MAX_EPISODE_STEPS = 1000 # Hard limit for safety

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_WALLS = (50, 50, 60)
        self.COLOR_PATH = (100, 100, 110)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (30, 90, 180)
        self.COLOR_EXIT = (50, 255, 150)
        self.COLOR_TEXT = (255, 255, 255)

        # Action and Observation Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Difficulty scaling state
        self.initial_maze_dim = 7
        self.max_maze_dim = 31
        self.maze_width = self.initial_maze_dim
        self.maze_height = self.initial_maze_dim
        self.successful_episodes = 0

        # Initialize state variables (will be properly set in reset)
        self.maze = []
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.path_taken = []
        self.steps = 0
        self.score = 0
        self.game_over = False

        # self.reset() # This is called by the wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Difficulty scaling: increase maze size every 3 wins
        if self.successful_episodes > 0 and self.successful_episodes % 3 == 0:
            new_dim = min(self.maze_width + 2, self.max_maze_dim)
            if new_dim > self.maze_width:
                self.maze_width = new_dim
                self.maze_height = new_dim
        
        # Generate maze and set positions
        self.maze = self._generate_maze(self.maze_width, self.maze_height)
        self.player_pos = (1, 1)
        self.exit_pos = (self.maze_width - 2, self.maze_height - 2)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.path_taken = [self.player_pos]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0=none, 1=up, 2=down, 3=left, 4=right
        
        moved = self._move_player(movement)
        
        # A step is counted even for no-ops or invalid moves
        self.steps += 1
        
        # Calculate reward
        reward = -1  # Cost for taking a step
        terminated = False

        # Check for win condition
        if self.player_pos == self.exit_pos:
            reward += 100  # Large bonus for winning
            self.score += reward
            self.game_over = True
            terminated = True
            self.successful_episodes += 1
        
        # Check for lose condition (step limit)
        if not terminated and self.steps >= self.MAX_STEPS:
            self.score += reward # Add final step cost
            self.game_over = True
            terminated = True
        
        # Check for hard step limit
        if not terminated and self.steps >= self.MAX_EPISODE_STEPS:
            self.score += reward
            self.game_over = True
            terminated = True
            
        if not terminated:
            self.score += reward

        truncated = False # This environment does not truncate based on time limit

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _move_player(self, movement):
        px, py = self.player_pos
        nx, ny = px, py

        if movement == 1:  # Up
            ny -= 1
        elif movement == 2:  # Down
            ny += 1
        elif movement == 3:  # Left
            nx -= 1
        elif movement == 4:  # Right
            nx += 1
        
        # Check for wall collision
        if self.maze[ny][nx] == 0:  # 0 is a path
            self.player_pos = (nx, ny)
            if self.player_pos not in self.path_taken:
                self.path_taken.append(self.player_pos)
            return True
        return False

    def _generate_maze(self, width, height):
        # Maze must have odd dimensions
        width = width if width % 2 != 0 else width + 1
        height = height if height % 2 != 0 else height + 1
        
        # Initialize maze with walls (1)
        maze = np.ones((height, width), dtype=np.uint8)
        
        # Use randomized DFS to carve paths
        stack = []
        start_x, start_y = (1, 1)
        maze[start_y, start_x] = 0 # 0 is a path
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            
            # Find unvisited neighbors (2 cells away)
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # Choose a random neighbor
                neighbor_index = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[neighbor_index]

                # Carve path to neighbor
                maze[ny, nx] = 0
                maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                
                stack.append((nx, ny))
            else:
                stack.pop()
        
        return maze.tolist()

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
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
            "maze_dim": (self.maze_width, self.maze_height)
        }

    def _render_game(self):
        # Calculate cell dimensions and centering offset
        cell_w = self.WIDTH / self.maze_width
        cell_h = self.HEIGHT / self.maze_height
        offset_x = (self.WIDTH - self.maze_width * cell_w) / 2
        offset_y = (self.HEIGHT - self.maze_height * cell_h) / 2

        # Draw traversed path
        for x, y in self.path_taken:
            rect = pygame.Rect(
                offset_x + x * cell_w,
                offset_y + y * cell_h,
                math.ceil(cell_w),
                math.ceil(cell_h)
            )
            pygame.draw.rect(self.screen, self.COLOR_PATH, rect)

        # Draw maze walls
        for y, row in enumerate(self.maze):
            for x, cell in enumerate(row):
                if cell == 1:  # Wall
                    rect = pygame.Rect(
                        offset_x + x * cell_w,
                        offset_y + y * cell_h,
                        math.ceil(cell_w),
                        math.ceil(cell_h)
                    )
                    pygame.draw.rect(self.screen, self.COLOR_WALLS, rect)
        
        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(
            offset_x + ex * cell_w,
            offset_y + ey * cell_h,
            cell_w,
            cell_h
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Draw player with glow effect
        px, py = self.player_pos
        player_center_x = int(offset_x + (px + 0.5) * cell_w)
        player_center_y = int(offset_y + (py + 0.5) * cell_h)
        
        # Glow
        glow_radius = int(min(cell_w, cell_h) * 0.6)
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Player square
        player_size = min(cell_w, cell_h) * 0.7
        player_rect = pygame.Rect(
            offset_x + px * cell_w + (cell_w - player_size) / 2,
            offset_y + py * cell_h + (cell_h - player_size) / 2,
            player_size,
            player_size
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=int(player_size*0.2))


    def _render_ui(self):
        steps_left = max(0, self.MAX_STEPS - self.steps)
        text_surface = self.font.render(f"Steps Left: {steps_left}", True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        score_surface = self.font.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_surface.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_surface, score_rect)

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # For interactive play, switch the SDL driver
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy' or 'windows' as appropriate

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Maze Runner")
    
    running = True
    terminated = False
    
    print(env.user_guide)
    print("Press 'r' to reset.")

    while running:
        action = np.array([0, 0, 0]) # Default no-op action
        
        # Poll for events and take action on key press
        event_happened = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    terminated = False
                    obs, info = env.reset()
                    event_happened = True
                if terminated:
                    continue
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                # Since it's turn based, we step on each key press
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward}, Score: {info['score']}, Terminated: {terminated}")
                event_happened = True
        
        # Render the observation to the display
        # The observation is (H, W, C), but pygame wants (W, H, C)
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
    env.close()