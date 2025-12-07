import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the robot through the maze. "
        "Reach the green survivor before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Guide your robot through a procedurally generated maze to rescue a survivor. "
        "Each move costs a point, so find the most efficient path. Red tiles are risky but might be necessary."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAX_MOVES = 50
        self.MAX_EPISODE_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_WALL = (60, 70, 80)
        self.COLOR_PATH = (35, 40, 50)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_ROBOT = (50, 150, 255)
        self.COLOR_ROBOT_GLOW = (50, 150, 255, 50)
        self.COLOR_SURVIVOR = (50, 255, 150)
        self.COLOR_SURVIVOR_GLOW = (50, 255, 150, 50)
        self.COLOR_RISKY = (255, 80, 80)
        self.COLOR_RISKY_GLOW = (255, 80, 80, 50)
        self.COLOR_TEXT = (220, 220, 220)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 32, bold=True)

        # Initialize state variables
        self.maze_w = 0
        self.maze_h = 0
        self.maze = []
        self.robot_pos = (0, 0)
        self.survivor_pos = (0, 0)
        self.risky_tiles = []
        self.steps = 0
        self.level = 0
        self.score = 0
        self.moves_remaining = 0
        self.game_over = False
        self.np_random = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        if self.level == 0:
            self.level = 1
            self.score = 0
            
        self.steps = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False

        self._set_maze_dimensions()
        self._generate_maze()

        return self._get_observation(), self._get_info()

    def _set_maze_dimensions(self):
        level_group = (self.level - 1) // 3
        size = min(29, 5 + level_group * 2)
        self.maze_w = size
        self.maze_h = size

    def _generate_maze(self):
        # 0: path, 1: wall
        self.maze = np.ones((self.maze_h, self.maze_w), dtype=np.uint8)
        
        # Randomized DFS
        stack = []
        start_x, start_y = (1, 1)
        self.maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.maze_w - 1 and 0 < ny < self.maze_h - 1 and self.maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # Choose a random neighbor by index
                chosen_index = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[chosen_index]
                
                # Carve path
                self.maze[ny, nx] = 0
                self.maze[(y + ny) // 2, (x + nx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Place entities
        self.robot_pos = (1, 1)
        self.survivor_pos = (self.maze_w - 2, self.maze_h - 2)

        # Place risky tiles
        path_tiles = np.argwhere(self.maze == 0).tolist()
        path_tiles = [tuple(t[::-1]) for t in path_tiles] # argwhere returns (row, col) which is (y, x)
        
        # Exclude start and end points
        if self.robot_pos in path_tiles: path_tiles.remove(self.robot_pos)
        if self.survivor_pos in path_tiles: path_tiles.remove(self.survivor_pos)
        
        num_risky = int(math.sqrt(self.maze_w * self.maze_h) / 4)
        self.risky_tiles = []
        if len(path_tiles) > num_risky:
            indices = self.np_random.choice(len(path_tiles), size=num_risky, replace=False)
            self.risky_tiles = [path_tiles[i] for i in indices]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        moved = False

        if movement > 0:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            nx, ny = self.robot_pos[0] + dx, self.robot_pos[1] + dy

            # Check for valid move (within bounds and not a wall)
            if 0 <= nx < self.maze_w and 0 <= ny < self.maze_h and self.maze[ny, nx] == 0:
                self.robot_pos = (nx, ny)
                moved = True
                # sfx: move_blip
        
        # Apply costs and rewards only if a move was attempted (action > 0)
        if movement > 0:
            self.steps += 1
            if moved:
                self.moves_remaining -= 1
                reward -= 1  # Cost for moving

                # Check for risky tile
                if self.robot_pos in self.risky_tiles:
                    reward += -5 + 2 # -5 penalty, +2 bonus
                    # sfx: risky_tile_land
                
                # Check for survivor
                if self.robot_pos == self.survivor_pos:
                    reward += 100
                    self.score += 1
                    self.level += 1
                    self.game_over = True
                    # sfx: win_level
            # No penalty for bumping into a wall, but it uses a step

        # Termination conditions
        terminated = self.game_over or self.moves_remaining <= 0 or self.steps >= self.MAX_EPISODE_STEPS
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if not self.game_over and self.moves_remaining <= 0:
            # sfx: lose_game
            self.game_over = True # Set game over for rendering message
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Calculate grid rendering properties
        grid_area_w = self.SCREEN_WIDTH - 150 # Reserve space for UI
        grid_area_h = self.SCREEN_HEIGHT - 20
        cell_size = min(grid_area_w // self.maze_w, grid_area_h // self.maze_h)
        
        total_grid_w = self.maze_w * cell_size
        total_grid_h = self.maze_h * cell_size
        offset_x = (grid_area_w - total_grid_w) // 2 + 10
        offset_y = (self.SCREEN_HEIGHT - total_grid_h) // 2

        # Draw maze
        for y in range(self.maze_h):
            for x in range(self.maze_w):
                rect = pygame.Rect(offset_x + x * cell_size, offset_y + y * cell_size, cell_size, cell_size)
                color = self.COLOR_WALL if self.maze[y, x] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)
                if (x, y) in self.risky_tiles:
                    # Pulsing effect for risky tiles
                    pulse = (math.sin(self.steps * 0.5) + 1) / 2
                    risky_color = tuple(int(c * pulse + self.COLOR_RISKY[i] * (1-pulse)) for i, c in enumerate(self.COLOR_RISKY_GLOW[:3]))
                    pygame.draw.rect(self.screen, risky_color, rect.inflate(-cell_size*0.2, -cell_size*0.2))


        # Draw grid lines
        for i in range(self.maze_w + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (offset_x + i * cell_size, offset_y), (offset_x + i * cell_size, offset_y + total_grid_h))
        for i in range(self.maze_h + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (offset_x, offset_y + i * cell_size), (offset_x + total_grid_w, offset_y + i * cell_size))

        # Draw survivor
        sx, sy = self.survivor_pos
        survivor_center = (int(offset_x + (sx + 0.5) * cell_size), int(offset_y + (sy + 0.5) * cell_size))
        survivor_radius = int(cell_size * 0.35)
        pygame.gfxdraw.filled_circle(self.screen, survivor_center[0], survivor_center[1], survivor_radius, self.COLOR_SURVIVOR)
        pygame.gfxdraw.aacircle(self.screen, survivor_center[0], survivor_center[1], survivor_radius, self.COLOR_SURVIVOR)

        # Draw robot
        rx, ry = self.robot_pos
        robot_center = (int(offset_x + (rx + 0.5) * cell_size), int(offset_y + (ry + 0.5) * cell_size))
        robot_radius = int(cell_size * 0.4)
        pygame.gfxdraw.filled_circle(self.screen, robot_center[0], robot_center[1], robot_radius, self.COLOR_ROBOT)
        pygame.gfxdraw.aacircle(self.screen, robot_center[0], robot_center[1], robot_radius, self.COLOR_ROBOT)
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, robot_center[0], robot_center[1], int(robot_radius * 1.5), self.COLOR_ROBOT_GLOW)


    def _render_ui(self):
        ui_x = self.SCREEN_WIDTH - 130
        
        # Moves remaining
        moves_text = self.font_ui.render("Moves", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (ui_x, 30))
        moves_val = self.font_msg.render(f"{self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_val, (ui_x, 50))
        
        # Level
        level_text = self.font_ui.render("Level", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (ui_x, 120))
        level_val = self.font_msg.render(f"{self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_val, (ui_x, 140))
        
        # Score
        score_text = self.font_ui.render("Rescued", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_x, 210))
        score_val = self.font_msg.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_val, (ui_x, 230))

        # Game Over Message
        if self.game_over:
            if self.robot_pos == self.survivor_pos:
                msg = "LEVEL COMPLETE"
                color = self.COLOR_SURVIVOR
            else:
                msg = "OUT OF MOVES"
                color = self.COLOR_RISKY
            
            end_text = self.font_msg.render(msg, True, color)
            text_rect = end_text.get_rect(center=( (self.SCREEN_WIDTH - 150) / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 180))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "level": self.level,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
        }
        
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # You can also use it for initial testing and debugging
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    # This will not work in a headless environment
    try:
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Maze Rescue")
        human_play = True
    except pygame.error:
        print("Pygame display not available. Running in headless mode. Manual play disabled.")
        human_play = False

    if not human_play:
        # If you cannot play, just run a few random steps to test the environment
        print("Testing with 10 random actions...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
            if terminated or truncated:
                print("Episode finished. Resetting.")
                obs, info = env.reset()
        print("Test complete.")

    else:
        # Human play loop
        running = True
        terminated = False
        
        print(env.user_guide)
        print(env.game_description)

        while running:
            action = np.array([0, 0, 0]) # Default to no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # If the game is over, any key press resets the environment
                    if terminated:
                        terminated = False
                        obs, info = env.reset()
                        continue

                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    
                    # Only step if a movement key was pressed
                    if action[0] != 0:
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
                        if terminated or truncated:
                            # The game will wait for a key press to reset
                            pass

            # Draw the observation to the display screen
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
        
    env.close()