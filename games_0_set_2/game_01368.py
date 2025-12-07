import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold Shift to sprint (move two squares). "
        "Avoid the red rats and reach the green exit before time runs out."
    )

    game_description = (
        "Navigate a procedurally generated sewer maze, dodging ravenous rats, "
        "to reach the exit within a tight time limit. A top-down survival horror game."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAZE_WIDTH, self.MAZE_HEIGHT = 32, 20
        self.CELL_SIZE = self.WIDTH // self.MAZE_WIDTH
        self.MAX_TIME = 600  # 60 seconds at 10 steps/sec
        self.MAX_RATS_TOUCHED = 5
        self.MAX_RATS = 20
        self.MIN_RAT_SPAWN_DIST = 8

        # --- Colors ---
        self.COLOR_BG = (15, 15, 20)
        self.COLOR_WALL = (40, 40, 50)
        self.COLOR_WALL_OUTLINE = (60, 60, 70)
        self.COLOR_PATH = (25, 25, 30)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (255, 255, 255, 50)
        self.COLOR_RAT = (220, 50, 50)
        self.COLOR_EXIT = (50, 220, 50)
        self.COLOR_EXIT_GLOW = (50, 220, 50, 70)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_DANGER = (220, 50, 50)
        self.COLOR_SPLASH = (220, 50, 50)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # --- State Variables ---
        self.maze = None
        self.start_pos = None
        self.exit_pos = None
        self.player_pos = None
        self.rat_positions = None
        self.time_remaining = None
        self.rats_touched = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.particles = None
        self.np_random = None

    def _generate_maze(self):
        # 0: path, 1: wall
        maze = np.ones((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.uint8)
        
        # Randomized DFS
        stack = deque()
        start_x = self.np_random.integers(0, self.MAZE_WIDTH)
        start_y = self.np_random.integers(0, self.MAZE_HEIGHT)
        stack.append((start_x, start_y))
        maze[start_y, start_x] = 0
        
        visited_cells = {(start_x, start_y)}

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx * 2, cy + dy * 2
                if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and (nx, ny) not in visited_cells:
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                # Need to convert to a list of tuples for np_random.choice
                neighbor_list = [tuple(n) for n in neighbors]
                chosen_neighbor = neighbor_list[self.np_random.integers(len(neighbor_list))]
                nx, ny, dx, dy = chosen_neighbor
                
                maze[cy + dy, cx + dx] = 0
                maze[ny, nx] = 0
                visited_cells.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        
        self.maze = maze
        
        # Set start and exit
        possible_starts = np.argwhere(self.maze == 0)
        start_idx = self.np_random.integers(0, len(possible_starts))
        self.start_pos = tuple(possible_starts[start_idx][::-1]) # (x, y)
        
        farthest_pos = None
        max_dist = -1
        for pos in possible_starts:
            dist = self._manhattan_distance(self.start_pos, tuple(pos[::-1]))
            if dist > max_dist:
                max_dist = dist
                farthest_pos = tuple(pos[::-1])
        self.exit_pos = farthest_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None or self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self._generate_maze()
        self.player_pos = self.start_pos
        
        # Spawn rats
        self.rat_positions = []
        possible_spawns = np.argwhere(self.maze == 0)
        valid_spawns = []
        for pos in possible_spawns:
            pos_xy = tuple(pos[::-1])
            if pos_xy != self.start_pos and pos_xy != self.exit_pos and \
               self._manhattan_distance(pos_xy, self.start_pos) > self.MIN_RAT_SPAWN_DIST:
                valid_spawns.append(pos_xy)

        num_rats = self.np_random.integers(self.MAX_RATS // 2, self.MAX_RATS + 1)
        if len(valid_spawns) > num_rats:
            spawn_indices = self.np_random.choice(len(valid_spawns), num_rats, replace=False)
            for i in spawn_indices:
                self.rat_positions.append(valid_spawns[i])

        self.time_remaining = self.MAX_TIME
        self.rats_touched = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Reward Calculation Setup ---
        dist_before = self._manhattan_distance(self.player_pos, self.exit_pos)
        reward = -0.1  # Time penalty

        # --- Player Movement ---
        sprint_mult = 2 if shift_held else 1
        dx, dy = {
            1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)
        }.get(movement, (0, 0))

        current_pos = self.player_pos
        for _ in range(sprint_mult):
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if self._is_valid_path(next_pos):
                current_pos = next_pos
            else:
                break # Hit a wall
        self.player_pos = current_pos

        # --- Rat Movement ---
        new_rat_positions = []
        for rat_pos in self.rat_positions:
            rat_x, rat_y = rat_pos
            player_x, player_y = self.player_pos
            
            rdx = np.sign(player_x - rat_x)
            rdy = np.sign(player_y - rat_y)
            
            move_options = []
            if rdx != 0: move_options.append((rat_x + rdx, rat_y))
            if rdy != 0: move_options.append((rat_x, rat_y + rdy))
            
            if move_options:
                self.np_random.shuffle(move_options)
                moved = False
                for next_pos in move_options:
                    if self._is_valid_path(next_pos):
                        new_rat_positions.append(next_pos)
                        moved = True
                        break
                if not moved:
                    new_rat_positions.append(rat_pos)
            else:
                new_rat_positions.append(rat_pos)
        self.rat_positions = new_rat_positions

        # --- Collision Detection ---
        rats_after_collision = []
        collided = False
        for rat_pos in self.rat_positions:
            if rat_pos == self.player_pos:
                self.rats_touched += 1
                collided = True
                self._create_splash_particles(self.player_pos)
            else:
                rats_after_collision.append(rat_pos)
        self.rat_positions = rats_after_collision
        
        if collided:
            reward -= 10

        # --- Update State & Finalize Reward ---
        self.steps += 1
        self.time_remaining -= 1
        dist_after = self._manhattan_distance(self.player_pos, self.exit_pos)

        if dist_after < dist_before:
            reward += 1.0
        elif dist_after > dist_before:
            reward -= 1.0

        # --- Termination Check ---
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 100
            terminated = True
            self.win = True
        elif self.rats_touched >= self.MAX_RATS_TOUCHED:
            terminated = True
        elif self.time_remaining <= 0:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        self._update_particles()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_valid_path(self, pos):
        x, y = pos
        return 0 <= x < self.MAZE_WIDTH and 0 <= y < self.MAZE_HEIGHT and self.maze[y, x] == 0

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _grid_to_pixel(self, grid_pos):
        x, y = grid_pos
        return (x * self.CELL_SIZE, y * self.CELL_SIZE)

    def _create_splash_particles(self, grid_pos):
        px, py = self._grid_to_pixel(grid_pos)
        center_x, center_y = px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'max_life': life})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Drag
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw maze
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                rect = (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if self.maze[y, x] == 1:
                    pygame.gfxdraw.box(self.screen, rect, self.COLOR_WALL)
                    pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_WALL_OUTLINE)
                else:
                    pygame.gfxdraw.box(self.screen, rect, self.COLOR_PATH)

        # Draw exit
        exit_px, exit_py = self._grid_to_pixel(self.exit_pos)
        glow_rect = pygame.Rect(exit_px-2, exit_py-2, self.CELL_SIZE+4, self.CELL_SIZE+4)
        pygame.gfxdraw.box(self.screen, glow_rect, self.COLOR_EXIT_GLOW)
        exit_rect = pygame.Rect(exit_px, exit_py, self.CELL_SIZE, self.CELL_SIZE)
        pygame.gfxdraw.box(self.screen, exit_rect, self.COLOR_EXIT)

        # Draw rats
        for rat_pos in self.rat_positions:
            rat_px, rat_py = self. _grid_to_pixel(rat_pos)
            rat_rect = pygame.Rect(rat_px + 2, rat_py + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
            pygame.gfxdraw.box(self.screen, rat_rect, self.COLOR_RAT)
        
        # Draw player
        player_px, player_py = self._grid_to_pixel(self.player_pos)
        glow_rect = pygame.Rect(player_px - 4, player_py - 4, self.CELL_SIZE + 8, self.CELL_SIZE + 8)
        pygame.gfxdraw.box(self.screen, glow_rect, self.COLOR_PLAYER_GLOW)
        player_rect = pygame.Rect(player_px, player_py, self.CELL_SIZE, self.CELL_SIZE)
        pygame.gfxdraw.box(self.screen, player_rect, self.COLOR_PLAYER)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = self.COLOR_SPLASH + (alpha,)
            size = int(self.CELL_SIZE / 4 * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

    def _render_ui(self):
        # Rats touched display
        rat_text = self.font_ui.render(f"TOUCHES: {self.rats_touched}/{self.MAX_RATS_TOUCHED}", True, self.COLOR_TEXT_DANGER)
        self.screen.blit(rat_text, (10, 5))

        # Time display
        time_str = f"TIME: {self.time_remaining / 10.0:.1f}"
        time_text = self.font_ui.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 5))

        # Game Over display
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU ESCAPED!" if self.win else "GAME OVER"
            color = self.COLOR_EXIT if self.win else self.COLOR_RAT
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "rats_touched": self.rats_touched,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
            "win": self.win
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually
    # You need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    pygame.init() # Re-init with display
    
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Sewer Maze")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    print(env.user_guide)
    
    running = True
    while running:
        # --- Action mapping for human play ---
        movement = 0 # no-op
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        # For turn-based, we wait for an event to step
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action_taken = True
                if event.key == pygame.K_r:
                    print("Resetting game.")
                    obs, info = env.reset()
                    terminated = False

        if action_taken and not terminated:
            action = [movement, 0, shift_held] # space is unused
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Press 'R' to reset.")

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    env.close()