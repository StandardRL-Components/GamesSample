import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import collections
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An arcade-style, top-down maze puzzle game.

    The player must navigate a procedurally generated maze to find the exit
    before running out of moves. The game is turn-based, with each move
    attempt consuming one step from a limited pool. The view is centered on
    the player, revealing only a small portion of the maze at a time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze."
    )

    # Short, user-facing description of the game
    game_description = (
        "Navigate a procedurally generated maze to reach the green exit tile. "
        "You have a limited number of moves for each maze. Plan your path carefully!"
    )

    # Frames only advance on action
    auto_advance = False

    # --- Constants ---
    # Maze and rendering
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAZE_WIDTH = 31  # Must be odd
    MAZE_HEIGHT = 21 # Must be odd
    VIEW_WIDTH_TILES = 15
    VIEW_HEIGHT_TILES = 11
    TILE_WIDTH = SCREEN_WIDTH / VIEW_WIDTH_TILES
    TILE_HEIGHT = SCREEN_HEIGHT / VIEW_HEIGHT_TILES

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (40, 40, 55)
    COLOR_PATH_UNVISITED = (70, 70, 90)
    COLOR_PATH_VISITED = (120, 120, 140)
    COLOR_PLAYER = (255, 220, 0)
    COLOR_PLAYER_GLOW = (255, 220, 0, 50)
    COLOR_EXIT = (0, 255, 120)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)

    # Game rules
    INITIAL_MOVES = 100
    MAX_EPISODE_STEPS = 1000

    # Maze cell types
    PATH = 0
    WALL = 1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.visited_tiles = None
        self.steps = 0
        self.score = 0
        self.remaining_moves = 0
        self.game_over_message = ""
        self.start_pos = None

        # This will be initialized in reset()
        self.np_random = None

        # The validation function needs a valid state, which is created by reset().
        # We call reset() inside the validation function to ensure the environment
        # is correctly initialized before tests are run.
        self.validate_implementation()

    def _generate_maze(self):
        """Generates a perfect maze using recursive backtracking."""
        grid = np.full((self.MAZE_HEIGHT, self.MAZE_WIDTH), self.WALL, dtype=np.uint8)
        stack = []

        # Start carving from (1, 1)
        start_x, start_y = (1, 1)
        grid[start_y, start_x] = self.PATH
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []

            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.MAZE_WIDTH - 1 and 0 < ny < self.MAZE_HEIGHT - 1 and grid[ny, nx] == self.WALL:
                    neighbors.append((nx, ny))

            if neighbors:
                # np.random.Generator.choice returns an array if size is specified
                idx = self.np_random.choice(len(neighbors))
                nx, ny = neighbors[idx]
                
                # Carve path
                grid[ny, nx] = self.PATH
                grid[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = self.PATH
                stack.append((nx, ny))
            else:
                stack.pop()

        self.maze = grid
        self.start_pos = (1, 1)

        # Find the furthest point from the start to place the exit
        q = collections.deque([(self.start_pos, 0)])
        distances = {self.start_pos: 0}
        max_dist = 0
        furthest_cell = self.start_pos

        while q:
            (vx, vy), dist = q.popleft()

            if dist > max_dist:
                max_dist = dist
                furthest_cell = (vx, vy)

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = vx + dx, vy + dy
                if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[ny, nx] == self.PATH and (nx, ny) not in distances:
                    distances[(nx, ny)] = dist + 1
                    q.append(((nx, ny), dist + 1))
        
        self.exit_pos = furthest_cell

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._generate_maze()
        self.player_pos = self.start_pos
        self.visited_tiles = {self.player_pos}

        self.steps = 0
        self.score = 0
        self.remaining_moves = self.INITIAL_MOVES
        self.game_over_message = ""

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = 0
        terminated = False

        if self.game_over_message:
            # If game is over, do nothing until reset
            return self._get_observation(), 0, True, False, self._get_info()

        # Update step count
        self.steps += 1
        
        # --- Handle player movement ---
        moved = False
        if movement > 0: # Any move attempt costs a move
            self.remaining_moves -= 1
            reward = -0.1 # Small penalty for taking a step

            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            next_x, next_y = self.player_pos[0] + dx, self.player_pos[1] + dy

            # Check for valid move (within bounds and not a wall)
            if 0 <= next_x < self.MAZE_WIDTH and 0 <= next_y < self.MAZE_HEIGHT and self.maze[next_y, next_x] == self.PATH:
                self.player_pos = (next_x, next_y)
                self.visited_tiles.add(self.player_pos)
                moved = True
                # sfx: player_move.wav

        # --- Check for termination conditions ---
        if self.player_pos == self.exit_pos:
            reward += 10.0  # Big reward for winning
            self.score += 10 + self.remaining_moves # Bonus score for finishing
            terminated = True
            self.game_over_message = "MAZE COMPLETE!"
            # sfx: win_level.wav
        
        elif self.remaining_moves <= 0:
            terminated = True
            self.game_over_message = "OUT OF MOVES"
            # sfx: lose_level.wav

        elif self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over_message = "TIME LIMIT REACHED"
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.game_over_message:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Calculate camera offset to center player
        cam_origin_x = self.player_pos[0] - self.VIEW_WIDTH_TILES // 2
        cam_origin_y = self.player_pos[1] - self.VIEW_HEIGHT_TILES // 2

        # Draw the visible portion of the maze
        for y in range(self.VIEW_HEIGHT_TILES):
            for x in range(self.VIEW_WIDTH_TILES):
                maze_x, maze_y = cam_origin_x + x, cam_origin_y + y
                screen_x, screen_y = x * self.TILE_WIDTH, y * self.TILE_HEIGHT
                
                rect = pygame.Rect(screen_x, screen_y, math.ceil(self.TILE_WIDTH), math.ceil(self.TILE_HEIGHT))

                if 0 <= maze_x < self.MAZE_WIDTH and 0 <= maze_y < self.MAZE_HEIGHT:
                    cell_type = self.maze[maze_y, maze_x]
                    pos = (maze_x, maze_y)
                    
                    if cell_type == self.WALL:
                        pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                    else: # It's a path
                        if pos in self.visited_tiles:
                            color = self.COLOR_PATH_VISITED
                        else:
                            color = self.COLOR_PATH_UNVISITED
                        pygame.draw.rect(self.screen, color, rect)
                        
                        if pos == self.exit_pos:
                            pygame.draw.rect(self.screen, self.COLOR_EXIT, rect.inflate(-self.TILE_WIDTH*0.2, -self.TILE_HEIGHT*0.2))
                else:
                    # Draw out-of-bounds as walls
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw the player in the center of the screen
        player_center_x = int((self.VIEW_WIDTH_TILES // 2 + 0.5) * self.TILE_WIDTH)
        player_center_y = int((self.VIEW_HEIGHT_TILES // 2 + 0.5) * self.TILE_HEIGHT)
        player_radius = int(min(self.TILE_WIDTH, self.TILE_HEIGHT) * 0.35)

        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, int(player_radius * 1.5), self.COLOR_PLAYER_GLOW)
        # Player body
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Render Moves Remaining
        moves_text = f"Moves: {self.remaining_moves}"
        self._draw_text(moves_text, (15, 10), self.font_ui)

        # Render Score
        score_text = f"Score: {int(self.score)}"
        text_width = self.font_ui.size(score_text)[0]
        self._draw_text(score_text, (self.SCREEN_WIDTH - text_width - 15, 10), self.font_ui)
        
    def _render_game_over(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Game over text
        text_width, text_height = self.font_game_over.size(self.game_over_message)
        pos_x = (self.SCREEN_WIDTH - text_width) / 2
        pos_y = (self.SCREEN_HEIGHT - text_height) / 2
        self._draw_text(self.game_over_message, (pos_x, pos_y), self.font_game_over)

    def _draw_text(self, text, pos, font):
        # Shadow
        shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
        # Main text
        text_surface = font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_moves": self.remaining_moves,
            "player_pos": self.player_pos,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        This function must call reset() to initialize the state before testing.
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to initialize the environment state
        obs, info = self.reset()
        
        # Test observation space using the observation from reset
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Reset again to ensure the environment is in a clean state after validation
        self.reset()
        
        # The print statement is commented out as it's not part of the standard
        # environment output, but it's useful for debugging.
        # print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To play, you might need to unset the headless environment variable
    # and use a regular pygame.display.set_mode
    # Example:
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Set up Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Runner")
    clock = pygame.time.Clock()
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    total_reward = 0
    
    running = True
    last_action_time = 0
    action_delay = 100 # milliseconds

    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- MAZE RESET ---")
                if event.key == pygame.K_q:
                    running = False

        # Turn-based logic for human play
        if not env.game_over_message and (current_time - last_action_time > action_delay):
            keys = pygame.key.get_pressed()
            move_action = 0
            if keys[pygame.K_UP]: move_action = 1
            elif keys[pygame.K_DOWN]: move_action = 2
            elif keys[pygame.K_LEFT]: move_action = 3
            elif keys[pygame.K_RIGHT]: move_action = 4
            
            if move_action > 0:
                action = np.array([move_action, 0, 0])
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
                last_action_time = current_time

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate
        
    env.close()