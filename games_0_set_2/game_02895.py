import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Space to interact with green gems. "
        "Shift to reveal the optimal path."
    )

    game_description = (
        "Navigate a cursed maze, solving puzzles to reach the golden exit. "
        "Each action costs a move. Run out of moves and you lose."
    )

    auto_advance = False

    # --- Colors ---
    COLOR_BG = (10, 5, 20)
    COLOR_WALL = (40, 30, 50)
    COLOR_FLOOR = (20, 15, 30)
    COLOR_PLAYER = (100, 150, 255)
    COLOR_EXIT = (255, 200, 0)
    COLOR_TRAP = (255, 50, 50)
    COLOR_PUZZLE = (50, 255, 100)
    COLOR_HINT = (200, 200, 255)
    COLOR_TEXT = (220, 220, 220)

    # --- Game Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    CELL_SIZE = 32
    MAZE_WIDTH, MAZE_HEIGHT = 41, 31  # Odd numbers for maze generation
    MAX_STEPS = 1000
    INITIAL_MOVES = 150
    TRAP_PENALTY = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        except pygame.error:
            self.font_large = pygame.font.SysFont("monospace", 24)
            self.font_small = pygame.font.SysFont("monospace", 16)

        # Persistent state across resets
        self.successful_escapes = 0
        self.difficulty_level = 0

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.maze = np.ones((self.MAZE_WIDTH, self.MAZE_HEIGHT))
        self.player_pos = [1, 1]
        self.exit_pos = [1, 1]
        self.remaining_moves = 0
        self.traps = set()
        self.triggered_traps = set()
        self.puzzles = {}
        self.solution_path = []
        self.particles = []
        self.hint_active_steps = 0

        # `reset()` is called here to set up the initial state.
        # It's important to call super().reset() first inside our `reset()`
        # to initialize the np_random generator.
        # self.reset()

    def _generate_maze(self):
        maze = np.ones((self.MAZE_WIDTH, self.MAZE_HEIGHT), dtype=np.uint8)
        stack = [(1, 1)]
        maze[1, 1] = 0

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.MAZE_WIDTH - 1 and 0 < ny < self.MAZE_HEIGHT - 1 and maze[nx, ny] == 1:
                    neighbors.append((nx, ny))

            if neighbors:
                # FIX: Choose a random neighbor tuple from the list and unpack it.
                # The original code incorrectly tried to unpack an integer index.
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]
                mx, my = (cx + nx) // 2, (cy + ny) // 2
                maze[nx, ny] = 0
                maze[mx, my] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

        self.maze = maze
        self.player_pos = [1, 1]
        self.exit_pos = [self.MAZE_WIDTH - 2, self.MAZE_HEIGHT - 2]
        self.maze[self.exit_pos[0], self.exit_pos[1]] = 0  # Ensure exit is reachable

    def _find_path(self, start, end):
        q = deque([(start, [start])])
        visited = {tuple(start)}
        while q:
            (vx, vy), path = q.popleft()
            if vx == end[0] and vy == end[1]:
                return path
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = vx + dx, vy + dy
                if (0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and
                        self.maze[nx, ny] == 0 and tuple((nx, ny)) not in visited):
                    visited.add(tuple((nx, ny)))
                    new_path = list(path)
                    new_path.append((nx, ny))
                    q.append(((nx, ny), new_path))
        return []  # No path found

    def _place_game_elements(self):
        path_cells = np.argwhere(self.maze == 0)
        self.np_random.shuffle(path_cells)

        # Avoid placing on start/end
        path_cells = [tuple(c) for c in path_cells if
                      tuple(c) != tuple(self.player_pos) and tuple(c) != tuple(self.exit_pos)]

        # Place Traps
        trap_count = min(len(path_cells) // 20, int(5 + self.difficulty_level * 0.5))
        self.traps = set(tuple(c) for c in path_cells[:trap_count])

        # Place Puzzles
        self.puzzles = {}
        puzzle_count = min(len(path_cells[trap_count:]) // 20, int(1 + self.difficulty_level * 0.2))

        # Find a wall to remove for each puzzle
        for i in range(puzzle_count):
            if len(path_cells) > trap_count + i:
                puzzle_pos = path_cells[trap_count + i]
                # Find an adjacent wall that, when removed, connects two paths
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    wall_to_remove = (puzzle_pos[0] + dx, puzzle_pos[1] + dy)
                    opposite_cell = (puzzle_pos[0] + 2 * dx, puzzle_pos[1] + 2 * dy)
                    if (0 < wall_to_remove[0] < self.MAZE_WIDTH - 1 and
                            0 < wall_to_remove[1] < self.MAZE_HEIGHT - 1 and
                            self.maze[wall_to_remove[0], wall_to_remove[1]] == 1 and
                            0 < opposite_cell[0] < self.MAZE_WIDTH - 1 and
                            0 < opposite_cell[1] < self.MAZE_HEIGHT - 1 and
                            self.maze[opposite_cell[0], opposite_cell[1]] == 0):
                        self.puzzles[tuple(puzzle_pos)] = {
                            "solved": False,
                            "wall_to_remove": wall_to_remove,
                            "anim_state": self.np_random.random() * math.pi * 2
                        }
                        break

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.remaining_moves = self.INITIAL_MOVES

        self._generate_maze()
        self._place_game_elements()
        self.triggered_traps.clear()

        # Ensure a path exists after modifications
        self.solution_path = self._find_path(self.player_pos, self.exit_pos)
        while not self.solution_path:
            self._generate_maze()
            self._place_game_elements()
            self.solution_path = self._find_path(self.player_pos, self.exit_pos)

        self.particles = []
        self.hint_active_steps = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.remaining_moves -= 1

        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        old_pos = list(self.player_pos)
        old_dist = len(self._find_path(self.player_pos, self.exit_pos) or [0] * 1000)

        # --- Action Handling ---
        # Prioritize movement
        if movement > 0:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            nx, ny = self.player_pos[0] + dx, self.player_pos[1] + dy
            if self.maze[nx, ny] == 0:
                self.player_pos = [nx, ny]
        # Then interaction
        elif space_held:
            player_pos_tuple = tuple(self.player_pos)
            if player_pos_tuple in self.puzzles and not self.puzzles[player_pos_tuple]["solved"]:
                puzzle = self.puzzles[player_pos_tuple]
                puzzle["solved"] = True
                wall_x, wall_y = puzzle["wall_to_remove"]
                self.maze[wall_x, wall_y] = 0
                reward += 5.0
                # Recalculate solution path
                self.solution_path = self._find_path(self.player_pos, self.exit_pos)
                self._add_particles(self.player_pos, self.COLOR_PUZZLE, 30)
                # Sound: puzzle solve
        # Then hint
        elif shift_held:
            self.hint_active_steps = 10
            # Cost for using hint
            reward -= 0.5

        # --- State Updates ---
        # Distance-based reward if moved
        if tuple(old_pos) != tuple(self.player_pos):
            new_dist = len(self._find_path(self.player_pos, self.exit_pos) or [0] * 1000)
            if new_dist < old_dist:
                reward += 0.1
            else:
                reward -= 0.2

        # Check for traps
        player_pos_tuple = tuple(self.player_pos)
        if player_pos_tuple in self.traps and player_pos_tuple not in self.triggered_traps:
            self.remaining_moves = max(0, self.remaining_moves - self.TRAP_PENALTY)
            self.triggered_traps.add(player_pos_tuple)
            reward -= 2.0
            self._add_particles(self.player_pos, self.COLOR_TRAP, 50)
            # Sound: trap trigger

        self.score += reward

        # --- Termination Check ---
        terminated = False
        if tuple(self.player_pos) == tuple(self.exit_pos):
            reward += 100.0
            self.score += 100.0
            terminated = True
            self.game_over = True
            self.successful_escapes += 1
            if self.successful_escapes > 0 and self.successful_escapes % 5 == 0:
                self.difficulty_level += 1
            self._add_particles(self.player_pos, self.COLOR_EXIT, 100)
            # Sound: win
        elif self.remaining_moves <= 0:
            reward -= 50.0
            self.score -= 50.0
            terminated = True
            self.game_over = True
            # Sound: lose
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _add_particles(self, pos, color, count):
        px, py = (pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2,
                  pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({
                "pos": [px, py],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": color
            })

    def _update_and_draw_particles(self, camera_offset):
        cam_x, cam_y = camera_offset
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95
            p["vel"][1] *= 0.95
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
                color = (*p["color"], alpha)
                pos = (int(p["pos"][0] - cam_x), int(p["pos"][1] - cam_y))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)

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
            "remaining_moves": self.remaining_moves,
            "player_pos": self.player_pos,
            "difficulty": self.difficulty_level,
        }

    def _render_game(self):
        # --- Camera Calculation ---
        cam_x = self.player_pos[0] * self.CELL_SIZE - self.SCREEN_WIDTH / 2
        cam_y = self.player_pos[1] * self.CELL_SIZE - self.SCREEN_HEIGHT / 2
        cam_x = max(0, min(cam_x, self.MAZE_WIDTH * self.CELL_SIZE - self.SCREEN_WIDTH))
        cam_y = max(0, min(cam_y, self.MAZE_HEIGHT * self.CELL_SIZE - self.SCREEN_HEIGHT))

        # --- Render Maze ---
        start_gx = int(cam_x // self.CELL_SIZE)
        start_gy = int(cam_y // self.CELL_SIZE)
        end_gx = start_gx + (self.SCREEN_WIDTH // self.CELL_SIZE) + 2
        end_gy = start_gy + (self.SCREEN_HEIGHT // self.CELL_SIZE) + 2

        for gy in range(start_gy, end_gy):
            for gx in range(start_gx, end_gx):
                if not (0 <= gx < self.MAZE_WIDTH and 0 <= gy < self.MAZE_HEIGHT):
                    continue

                sx, sy = int(gx * self.CELL_SIZE - cam_x), int(gy * self.CELL_SIZE - cam_y)
                rect = pygame.Rect(sx, sy, self.CELL_SIZE, self.CELL_SIZE)

                if self.maze[gx, gy] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)

                # Render Traps
                if (gx, gy) in self.traps and (gx, gy) not in self.triggered_traps:
                    pulse = (math.sin(self.steps * 0.2) + 1) / 2
                    size = int(self.CELL_SIZE * 0.3 + pulse * self.CELL_SIZE * 0.1)
                    color = (
                    self.COLOR_TRAP[0], int(self.COLOR_TRAP[1] * pulse + 50), int(self.COLOR_TRAP[2] * pulse + 50))
                    pygame.gfxdraw.filled_circle(self.screen, sx + self.CELL_SIZE // 2, sy + self.CELL_SIZE // 2, size,
                                                 color)

                # Render Puzzles
                if (gx, gy) in self.puzzles:
                    puzzle = self.puzzles[(gx, gy)]
                    if not puzzle["solved"]:
                        angle = (self.steps * 0.05 + puzzle["anim_state"]) % (2 * math.pi)
                        points = []
                        for i in range(4):
                            a = angle + i * math.pi / 2
                            px = sx + self.CELL_SIZE // 2 + math.cos(a) * self.CELL_SIZE * 0.35
                            py = sy + self.CELL_SIZE // 2 + math.sin(a) * self.CELL_SIZE * 0.35
                            points.append((int(px), int(py)))
                        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PUZZLE)
                        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PUZZLE)
                    else:  # Solved puzzle leaves a faint mark
                        pygame.gfxdraw.filled_circle(self.screen, sx + self.CELL_SIZE // 2, sy + self.CELL_SIZE // 2,
                                                     self.CELL_SIZE // 8, (50, 80, 60))

        # --- Render Exit ---
        ex, ey = self.exit_pos
        sx, sy = int(ex * self.CELL_SIZE - cam_x), int(ey * self.CELL_SIZE - cam_y)
        for i in range(15, 0, -1):
            pulse = (math.sin(self.steps * 0.1 + i / 5) + 1) / 2
            alpha = int(100 * (1 - i / 15) * pulse + 20)
            color = (*self.COLOR_EXIT, alpha)
            radius = int(self.CELL_SIZE * 0.5 * (i / 15))
            pygame.gfxdraw.filled_circle(self.screen, sx + self.CELL_SIZE // 2, sy + self.CELL_SIZE // 2, radius, color)

        # --- Render Player ---
        px, py = self.player_pos
        sx, sy = int(px * self.CELL_SIZE - cam_x), int(py * self.CELL_SIZE - cam_y)
        for i in range(10, 0, -1):
            alpha = int(150 * (1 - i / 10))
            color = (*self.COLOR_PLAYER, alpha)
            radius = int(self.CELL_SIZE * 0.3 + i)
            pygame.gfxdraw.filled_circle(self.screen, sx + self.CELL_SIZE // 2, sy + self.CELL_SIZE // 2, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, sx + self.CELL_SIZE // 2, sy + self.CELL_SIZE // 2,
                                     int(self.CELL_SIZE * 0.3), self.COLOR_PLAYER)

        # --- Render Hint ---
        if self.hint_active_steps > 0:
            self.hint_active_steps -= 1
            if len(self.solution_path) > 1:
                p1 = self.solution_path[0]
                p2 = self.solution_path[1]
                start_pos = (int(p1[0] * self.CELL_SIZE - cam_x + self.CELL_SIZE / 2),
                             int(p1[1] * self.CELL_SIZE - cam_y + self.CELL_SIZE / 2))
                end_pos = (int(p2[0] * self.CELL_SIZE - cam_x + self.CELL_SIZE / 2),
                           int(p2[1] * self.CELL_SIZE - cam_y + self.CELL_SIZE / 2))
                alpha = int(200 * (self.hint_active_steps / 10))
                pygame.draw.line(self.screen, (*self.COLOR_HINT, alpha), start_pos, end_pos, 3)

        # --- Render Particles ---
        self._update_and_draw_particles((cam_x, cam_y))

    def _render_ui(self):
        moves_text = self.font_large.render(f"Moves: {self.remaining_moves}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()


if __name__ == '__main__':
    # This block is for human play and debugging.
    # It is not part of the Gymnasium environment API.
    # To use, run `python your_file_name.py`
    
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption(GameEnv.game_description)
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    print(GameEnv.user_guide)

    while running:
        movement = 0  # No-op
        space_held = 0
        shift_held = 0
        
        # Poll for events
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            obs, info = env.reset()
            print("--- Game Reset ---")
            continue

        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        # Step the environment if an action is taken
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"Action: {action}, Reward: {reward:.2f}, Moves Left: {info['remaining_moves']}, Terminated: {terminated or truncated}")

            if terminated or truncated:
                print(f"--- Episode Finished --- Score: {info['score']:.2f}")
                print("Press 'R' to reset.")
                # Wait for reset key
                wait_for_reset = True
                while wait_for_reset:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            wait_for_reset = False
                            running = False
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                            obs, info = env.reset()
                            print("--- Game Reset ---")
                            wait_for_reset = False
                continue

        # --- Drawing ---
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()

        clock.tick(30)  # Limit FPS for human play

    env.close()