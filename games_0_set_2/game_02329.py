import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use Arrow Keys (↑, ↓, ←, →) to move one cell at a time. "
        "Avoid the enemies and reach the green exit."
    )

    game_description = (
        "A top-down puzzle game. Navigate a procedurally generated maze to find the exit "
        "while avoiding patrolling enemies. Plan your moves carefully to solve the maze in the fewest steps."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAZE_WIDTH, self.MAZE_HEIGHT = 20, 12
        self.CELL_SIZE = 32
        self.MAZE_OFFSET_X = (self.WIDTH - self.MAZE_WIDTH * self.CELL_SIZE) // 2
        self.MAZE_OFFSET_Y = (self.HEIGHT - self.MAZE_HEIGHT * self.CELL_SIZE) // 2

        self.MAX_STEPS = 1000
        self.INITIAL_ENEMY_MOVE_PERIOD = 3.0
        self.ENEMY_COUNT = 5

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_WALL = (40, 60, 100)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        self.player_pos = [0, 0]
        self.exit_pos = [0, 0]
        self.maze = []
        self.enemies = []
        self.enemy_move_counter = 0

        # --- Difficulty Progression ---
        self.enemy_move_period = self.INITIAL_ENEMY_MOVE_PERIOD
        
        # Initial call to reset is needed to set up the maze and other game state variables
        # A seed is not passed here as it will be passed in the public reset call
        # self.reset() is called by the environment wrapper, but for initialization, we need a state.

    def _generate_maze(self):
        # Using Randomized DFS to generate a perfect maze
        w, h = self.MAZE_WIDTH, self.MAZE_HEIGHT
        maze = [[{'n': True, 's': True, 'e': True, 'w': True, 'visited': False} for _ in range(w)] for _ in range(h)]
        
        stack = []
        x, y = self.np_random.integers(0, w), self.np_random.integers(0, h)
        maze[y][x]['visited'] = True
        stack.append((x, y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            if cy > 0 and not maze[cy - 1][cx]['visited']: neighbors.append(('n', cx, cy - 1))
            if cy < h - 1 and not maze[cy + 1][cx]['visited']: neighbors.append(('s', cx, cy + 1))
            if cx < w - 1 and not maze[cy][cx + 1]['visited']: neighbors.append(('e', cx + 1, cy))
            if cx > 0 and not maze[cy][cx - 1]['visited']: neighbors.append(('w', cx - 1, cy))

            if neighbors:
                # FIX: Use np_random.integers to get an index, then select from the list.
                # np.random.choice converts mixed-type lists to strings, causing the TypeError.
                idx = self.np_random.integers(len(neighbors))
                direction, nx, ny = neighbors[idx]

                if direction == 'n':
                    maze[cy][cx]['n'] = False
                    maze[ny][nx]['s'] = False
                elif direction == 's':
                    maze[cy][cx]['s'] = False
                    maze[ny][nx]['n'] = False
                elif direction == 'e':
                    maze[cy][cx]['e'] = False
                    maze[ny][nx]['w'] = False
                elif direction == 'w':
                    maze[cy][cx]['w'] = False
                    maze[ny][nx]['e'] = False
                
                maze[ny][nx]['visited'] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _find_valid_enemy_paths(self):
        paths = []
        # Find horizontal paths
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH - 2):
                if not self.maze[y][x]['e'] and not self.maze[y][x+1]['e']:
                    paths.append([(x, y), (x + 1, y), (x + 2, y)])
        # Find vertical paths
        for x in range(self.MAZE_WIDTH):
            for y in range(self.MAZE_HEIGHT - 2):
                if not self.maze[y][x]['s'] and not self.maze[y+1][x]['s']:
                    paths.append([(x, y), (x, y + 1), (x, y + 2)])
        return paths if paths else [[(self.MAZE_WIDTH // 2, self.MAZE_HEIGHT // 2)]]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.maze = self._generate_maze()
        self.player_pos = [0, 0]
        self.exit_pos = [self.MAZE_WIDTH - 1, self.MAZE_HEIGHT - 1]

        self.enemies = []
        valid_paths = self._find_valid_enemy_paths()
        if valid_paths:
            self.np_random.shuffle(valid_paths)
            for i in range(min(self.ENEMY_COUNT, len(valid_paths))):
                path = valid_paths[i]
                # Ensure enemy doesn't spawn on player or exit
                if tuple(path[0]) != tuple(self.player_pos) and tuple(path[0]) != tuple(self.exit_pos):
                    self.enemies.append({
                        'path': path,
                        'index': 0,
                        'direction': 1
                    })

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        self.enemy_move_counter = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1  # Cost of living
        terminated = False

        # --- Player Movement ---
        px, py = self.player_pos
        current_cell = self.maze[py][px]
        if movement == 1 and not current_cell['n']:  # Up
            self.player_pos[1] -= 1
        elif movement == 2 and not current_cell['s']:  # Down
            self.player_pos[1] += 1
        elif movement == 3 and not current_cell['w']:  # Left
            self.player_pos[0] -= 1
        elif movement == 4 and not current_cell['e']:  # Right
            self.player_pos[0] += 1
        
        # --- Enemy Movement ---
        self.enemy_move_counter += 1
        if self.enemy_move_counter >= self.enemy_move_period:
            self.enemy_move_counter = 0
            for enemy in self.enemies:
                path_len = len(enemy['path'])
                if path_len > 1:
                    if not (0 <= enemy['index'] + enemy['direction'] < path_len):
                        enemy['direction'] *= -1
                    enemy['index'] += enemy['direction']

        # --- Check Game State ---
        # 1. Collision with enemy
        for enemy in self.enemies:
            enemy_pos = enemy['path'][enemy['index']]
            if self.player_pos[0] == enemy_pos[0] and self.player_pos[1] == enemy_pos[1]:
                reward = -10.0
                terminated = True
                self.game_over = True
                self.win_message = "CAUGHT!"
                break
        
        if not terminated:
            # 2. Reached exit
            if self.player_pos[0] == self.exit_pos[0] and self.player_pos[1] == self.exit_pos[1]:
                reward = 10.0
                terminated = True
                self.game_over = True
                self.win_message = "ESCAPED!"
                self.enemy_move_period = max(1.0, self.enemy_move_period - 0.1) # Difficulty increase

            # 3. Timeout
            elif self.steps >= self.MAX_STEPS -1:
                terminated = True
                self.game_over = True
                self.win_message = "TIME OUT!"

        self.steps += 1
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Render Maze Walls
        wall_thickness = 2
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                cell = self.maze[y][x]
                px, py = self.MAZE_OFFSET_X + x * self.CELL_SIZE, self.MAZE_OFFSET_Y + y * self.CELL_SIZE
                if cell['n']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.CELL_SIZE, py), wall_thickness)
                if cell['s']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.CELL_SIZE), (px + self.CELL_SIZE, py + self.CELL_SIZE), wall_thickness)
                if cell['w']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + self.CELL_SIZE), wall_thickness)
                if cell['e']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.CELL_SIZE, py), (px + self.CELL_SIZE, py + self.CELL_SIZE), wall_thickness)

        # Render Exit
        exit_px, exit_py = self._grid_to_pixel(self.exit_pos)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (exit_px, exit_py, self.CELL_SIZE, self.CELL_SIZE))

        # Render Enemies
        for enemy in self.enemies:
            ex, ey = self._grid_to_pixel(enemy['path'][enemy['index']])
            center_x, center_y = ex + self.CELL_SIZE // 2, ey + self.CELL_SIZE // 2
            size = self.CELL_SIZE * 0.35
            
            # Pulsing effect for animation
            pulse = (math.sin(self.steps * 0.2) * 0.1) + 1.0
            
            points = [
                (center_x, center_y - size * pulse),
                (center_x - size * pulse, center_y + size * 0.7 * pulse),
                (center_x + size * pulse, center_y + size * 0.7 * pulse)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        # Render Player
        px, py = self._grid_to_pixel(self.player_pos)
        center_x, center_y = int(px + self.CELL_SIZE // 2), int(py + self.CELL_SIZE // 2)
        radius = int(self.CELL_SIZE * 0.3)
        
        # Breathing effect
        pulse = (math.sin(self.steps * 0.25 + math.pi) * 2) + radius
        int_pulse = int(pulse)
        
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int_pulse, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int_pulse, self.COLOR_PLAYER)

    def _render_ui(self):
        # Render Timer
        time_left = self.MAX_STEPS - self.steps
        time_text = f"TIME: {time_left}"
        text_surface = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Render Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surface = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surface.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_surface, score_rect)

        # Render Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_surface = self.font_game_over.render(self.win_message, True, self.COLOR_TEXT)
            end_text_rect = end_text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text_surface, end_text_rect)

    def _grid_to_pixel(self, grid_pos):
        x = self.MAZE_OFFSET_X + grid_pos[0] * self.CELL_SIZE
        y = self.MAZE_OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        return x, y

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
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    # The validation requires a state to be set, which reset() does.
    # In a typical Gymnasium workflow, reset() would be called by the user/wrapper before anything else.
    # For standalone validation, we call it here.
    env.reset(seed=42)
    env.validate_implementation()
    
    # --- Manual Play ---
    # This part is for human testing and requires a display.
    # It will not run in a headless environment.
    try:
        # Check if running in a headless environment
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
             print("\nManual play is not available in a headless environment.")
        else:
            print("\n--- Manual Control ---")
            print(env.user_guide)
            
            screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            pygame.display.set_caption("Maze Runner")
            
            obs, info = env.reset()
            done = False
            
            # Game loop for human play
            while not done:
                action = [0, 0, 0]  # Default no-op
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            action[0] = 1
                        elif event.key == pygame.K_DOWN:
                            action[0] = 2
                        elif event.key == pygame.K_LEFT:
                            action[0] = 3
                        elif event.key == pygame.K_RIGHT:
                            action[0] = 4
                        elif event.key == pygame.K_r: # Reset on 'r'
                            obs, info = env.reset()
                            continue
                
                # Since auto_advance is False, we only step when a key is pressed
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated:
                        print(f"Game Over! Score: {info['score']:.1f}, Steps: {info['steps']}")
                        # Short pause before auto-reset
                        frame = np.transpose(obs, (1, 0, 2))
                        pygame.surfarray.blit_array(screen, frame)
                        pygame.display.flip()
                        pygame.time.wait(2000)
                        obs, info = env.reset()

                # Render the current state
                frame = np.transpose(obs, (1, 0, 2))
                pygame.surfarray.blit_array(screen, frame)
                pygame.display.flip()
                
                env.clock.tick(30) # Limit FPS for human play
                
            env.close()
    except Exception as e:
        print(f"An error occurred during manual play: {e}")
        env.close()