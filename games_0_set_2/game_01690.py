
# Generated: 2025-08-27T17:57:27.023311
# Source Brief: brief_01690.md
# Brief Index: 1690

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to rotate the selected pipe."
    )

    game_description = (
        "A time-based puzzle game. Rotate pipe segments to connect the blue water source to the green exit before the timer runs out."
    )

    auto_advance = False

    # --- Constants ---
    # Grid and Display
    GRID_WIDTH = 12
    GRID_HEIGHT = 8
    CELL_SIZE = 50
    SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE + 40  # 640
    SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE    # 400

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (40, 50, 60)
    COLOR_PIPE = (150, 160, 170)
    COLOR_WATER = (50, 150, 255)
    COLOR_SOURCE = (30, 100, 220)
    COLOR_EXIT = (30, 220, 100)
    COLOR_CURSOR = (255, 220, 0)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TIMER_WARN = (255, 80, 80)

    # Pipe Types
    PIPE_STRAIGHT = 0
    PIPE_CORNER = 1
    PIPE_T_SHAPE = 2
    PIPE_CROSS = 3

    # Directions (match action space)
    DIR_NONE = 0
    DIR_UP = 1
    DIR_DOWN = 2
    DIR_LEFT = 3
    DIR_RIGHT = 4
    
    # Internal direction vectors for logic
    VEC = {
        DIR_UP: np.array([0, -1]),
        DIR_DOWN: np.array([0, 1]),
        DIR_LEFT: np.array([-1, 0]),
        DIR_RIGHT: np.array([1, 0]),
    }
    
    OPPOSITE_DIR = {
        DIR_UP: DIR_DOWN, DIR_DOWN: DIR_UP,
        DIR_LEFT: DIR_RIGHT, DIR_RIGHT: DIR_LEFT
    }

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
        self.font_main = pygame.font.SysFont("sans-serif", 24)
        self.font_timer = pygame.font.SysFont("monospace", 32, bold=True)
        
        self._init_pipe_exits()
        
        # State variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.source_pos = None
        self.exit_pos = None
        self.source_dir = None
        self.exit_dir = None
        self.watered_pipes = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.max_steps = 1000
        self.last_space_held = False

        self.validate_implementation()
    
    def _init_pipe_exits(self):
        """Pre-calculates the exit directions for each pipe type and orientation."""
        self.PIPE_EXITS = {
            self.PIPE_STRAIGHT: [
                {self.DIR_UP, self.DIR_DOWN}, {self.DIR_LEFT, self.DIR_RIGHT}
            ],
            self.PIPE_CORNER: [
                {self.DIR_UP, self.DIR_RIGHT}, {self.DIR_RIGHT, self.DIR_DOWN},
                {self.DIR_DOWN, self.DIR_LEFT}, {self.DIR_LEFT, self.DIR_UP}
            ],
            self.PIPE_T_SHAPE: [
                {self.DIR_LEFT, self.DIR_UP, self.DIR_RIGHT}, {self.DIR_UP, self.DIR_RIGHT, self.DIR_DOWN},
                {self.DIR_RIGHT, self.DIR_DOWN, self.DIR_LEFT}, {self.DIR_DOWN, self.DIR_LEFT, self.DIR_UP}
            ],
            self.PIPE_CROSS: [
                {self.DIR_UP, self.DIR_DOWN, self.DIR_LEFT, self.DIR_RIGHT}
            ]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.max_steps
        self.last_space_held = False
        
        self._generate_puzzle()
        self.cursor_pos = np.array(self.source_pos)
        
        self.watered_pipes = set()
        self._update_water_flow()
        # Reset score after initial flow check to not reward the initial state
        self.score = 0
        
        return self._get_observation(), self._get_info()
    
    def _generate_puzzle(self):
        self.source_pos = (0, self.np_random.integers(self.GRID_HEIGHT))
        self.source_dir = self.DIR_RIGHT
        self.exit_pos = (self.GRID_WIDTH - 1, self.np_random.integers(self.GRID_HEIGHT))
        self.exit_dir = self.DIR_LEFT
        
        # 1. Create a grid with placeholders
        # grid stores (pipe_type, orientation)
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT, 2), dtype=int)
        
        # 2. Generate a guaranteed solvable path using randomized DFS
        path = self._find_solution_path()
        
        # 3. Place pipes along the path and fill the rest
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) in path:
                    # Determine pipe type for the solution path
                    prev_node = path[(x, y)]
                    
                    # Find the next node in the path to determine exit direction
                    next_node = None
                    for key, val in path.items():
                        if val == (x, y):
                            next_node = key
                            break
                    
                    dirs = set()
                    if prev_node:
                        dx, dy = x - prev_node[0], y - prev_node[1]
                        dirs.add(self._vector_to_dir((dx, dy)))
                    if next_node:
                        dx, dy = next_node[0] - x, next_node[1] - y
                        dirs.add(self._vector_to_dir((dx, dy)))
                    
                    if len(dirs) == 2 and self.OPPOSITE_DIR[list(dirs)[0]] == list(dirs)[1]:
                        pipe_type = self.PIPE_STRAIGHT
                    else:
                        pipe_type = self.PIPE_CORNER
                    
                    self.grid[x, y] = [pipe_type, 0] # Initial orientation
                else:
                    # Fill non-path cells with random pipes
                    pipe_type = self.np_random.choice([self.PIPE_STRAIGHT, self.PIPE_CORNER, self.PIPE_T_SHAPE])
                    self.grid[x, y] = [pipe_type, 0]
                    
        # 4. Randomly rotate all pipes to create the puzzle
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                pipe_type = self.grid[x, y, 0]
                if pipe_type == self.PIPE_STRAIGHT:
                    self.grid[x, y, 1] = self.np_random.integers(2)
                else:
                    self.grid[x, y, 1] = self.np_random.integers(4)

    def _find_solution_path(self):
        start_node = (self.source_pos[0] + 1, self.source_pos[1])
        end_node = (self.exit_pos[0] - 1, self.exit_pos[1])

        q = collections.deque([(start_node, {start_node: self.source_pos})])
        visited = {self.source_pos, start_node}

        while q:
            current, path = q.popleft()
            if current == end_node:
                path[self.exit_pos] = current
                return path

            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
            
            self.np_random.shuffle(neighbors)
            
            for neighbor in neighbors:
                visited.add(neighbor)
                new_path = path.copy()
                new_path[neighbor] = current
                q.append((neighbor, new_path))
        
        # Fallback if pathfinding fails (should not happen in this setup)
        return {end_node: start_node, start_node: self.source_pos, self.exit_pos: end_node}

    def _vector_to_dir(self, vec):
        if vec == (0, -1): return self.DIR_UP
        if vec == (0, 1): return self.DIR_DOWN
        if vec == (-1, 0): return self.DIR_LEFT
        if vec == (1, 0): return self.DIR_RIGHT
        return self.DIR_NONE

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01  # Time penalty
        self.timer -= 1
        self.steps += 1
        
        # 1. Handle cursor movement
        if movement != self.DIR_NONE:
            delta = self.VEC.get(movement, np.array([0, 0]))
            new_pos = self.cursor_pos + delta
            # Wrap around edges
            new_pos[0] %= self.GRID_WIDTH
            new_pos[1] %= self.GRID_HEIGHT
            self.cursor_pos = new_pos
            
        # 2. Handle pipe rotation
        rotated = False
        if space_held and not self.last_space_held:
            # // sfx: pipe_rotate.wav
            pipe_x, pipe_y = self.cursor_pos
            pipe_type, orientation = self.grid[pipe_x, pipe_y]
            
            max_orientations = 2 if pipe_type == self.PIPE_STRAIGHT else 4
            if pipe_type != self.PIPE_CROSS:
                self.grid[pipe_x, pipe_y, 1] = (orientation + 1) % max_orientations
                rotated = True
        
        self.last_space_held = space_held
        
        # 3. Update water flow and calculate rewards if state changed
        if rotated:
            old_watered_count = len(self.watered_pipes)
            self._update_water_flow()
            new_watered_count = len(self.watered_pipes)
            
            # Reward for increasing path length
            reward += 5.0 * max(0, new_watered_count - old_watered_count)
            # Reward for total connected pipes
            reward += 0.1 * new_watered_count
            
        # 4. Check for termination
        terminated = False
        if self.exit_pos in self.watered_pipes:
            # // sfx: win_jingle.wav
            terminated = True
            reward += 100
        elif self.timer <= 0:
            # // sfx: loss_buzzer.wav
            terminated = True
            reward += -10
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_water_flow(self):
        new_watered = set()
        q = collections.deque([(self.source_pos, self.source_dir)])
        visited = {self.source_pos}

        while q:
            pos, entry_dir = q.popleft()
            
            if not (0 <= pos[0] < self.GRID_WIDTH and 0 <= pos[1] < self.GRID_HEIGHT):
                continue

            if pos == self.exit_pos:
                if entry_dir == self.exit_dir:
                    new_watered.add(self.exit_pos)
                continue

            # It's a pipe
            pipe_type, orientation = self.grid[pos[0], pos[1]]
            
            exits = self._get_pipe_exits(pipe_type, orientation)
            
            if entry_dir in exits:
                new_watered.add(pos)
                for exit_dir in exits:
                    if exit_dir != entry_dir:
                        next_pos_arr = np.array(pos) + self.VEC[exit_dir]
                        next_pos = tuple(next_pos_arr)
                        if next_pos not in visited:
                            visited.add(next_pos)
                            q.append((next_pos, self.OPPOSITE_DIR[exit_dir]))
                            
        self.watered_pipes = new_watered

    def _get_pipe_exits(self, pipe_type, orientation):
        if pipe_type == self.PIPE_CROSS:
            return self.PIPE_EXITS[pipe_type][0]
        return self.PIPE_EXITS[pipe_type][orientation]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_pipes()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_and_pipes(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x * self.CELL_SIZE + 20, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                
                # Draw grid cell background
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                pipe_type, orientation = self.grid[x, y]
                is_watered = (x, y) in self.watered_pipes
                self._draw_pipe(self.screen, rect, pipe_type, orientation, is_watered)

        # Draw Source and Exit
        src_rect = pygame.Rect(self.source_pos[0] * self.CELL_SIZE + 20, self.source_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SOURCE, src_rect)
        
        exit_rect = pygame.Rect(self.exit_pos[0] * self.CELL_SIZE + 20, self.exit_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        if self.exit_pos in self.watered_pipes:
             pygame.draw.rect(self.screen, self.COLOR_WATER, exit_rect.inflate(-10,-10))


    def _draw_pipe(self, surface, rect, pipe_type, orientation, is_watered):
        center = rect.center
        radius = int(self.CELL_SIZE * 0.2)
        
        color = self.COLOR_WATER if is_watered else self.COLOR_PIPE
        
        # Use thicker lines for better visibility
        pipe_width = 8

        # Draw arms
        exits = self._get_pipe_exits(pipe_type, orientation)
        for exit_dir in exits:
            if exit_dir == self.DIR_UP:
                pygame.draw.line(surface, color, center, (center[0], rect.top), pipe_width)
            elif exit_dir == self.DIR_DOWN:
                pygame.draw.line(surface, color, center, (center[0], rect.bottom), pipe_width)
            elif exit_dir == self.DIR_LEFT:
                pygame.draw.line(surface, color, center, (rect.left, center[1]), pipe_width)
            elif exit_dir == self.DIR_RIGHT:
                pygame.draw.line(surface, color, center, (rect.right, center[1]), pipe_width)

        # Draw center circle to smooth connections
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)


    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(x * self.CELL_SIZE + 20, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        # Draw a thick, glowing border
        border_size = 3
        for i in range(5):
            alpha = 150 - i * 30
            color = (*self.COLOR_CURSOR, alpha)
            s = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect(), border_size + i)
            self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (30, 10))
        
        # Timer
        timer_color = self.COLOR_TIMER_WARN if self.timer < self.max_steps * 0.1 else self.COLOR_TEXT
        timer_text = self.font_timer.render(f"{self.timer:04d}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "cursor_pos": tuple(self.cursor_pos),
            "solution_possible": self.exit_pos in self.watered_pipes
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pipe Puzzle Environment")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = GameEnv.DIR_NONE
        space_held = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = GameEnv.DIR_UP
        elif keys[pygame.K_DOWN]:
            movement = GameEnv.DIR_DOWN
        elif keys[pygame.K_LEFT]:
            movement = GameEnv.DIR_LEFT
        elif keys[pygame.K_RIGHT]:
            movement = GameEnv.DIR_RIGHT
            
        if keys[pygame.K_SPACE]:
            space_held = True

        # Construct the action based on MultiDiscrete space
        action = [movement, 1 if space_held else 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            
        clock.tick(30) # Control the speed of the game loop
        
    env.close()