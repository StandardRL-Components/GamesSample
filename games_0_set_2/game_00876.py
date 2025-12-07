
# Generated: 2025-08-27T15:03:32.105245
# Source Brief: brief_00876.md
# Brief Index: 876

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import heapq
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Follow the pulsating arrow cue to find the path."
    )

    game_description = (
        "A rhythm-puzzle game. Navigate a maze by following a directional cue that reveals the "
        "optimal path. Reach the green exit before the 60-second timer runs out. Your score is "
        "based on following the cues and finishing quickly."
    )

    auto_advance = True

    # --- Colors and Fonts ---
    COLOR_BG = (15, 20, 30)
    COLOR_WALL = (40, 60, 80)
    COLOR_PATH = (25, 35, 50)
    COLOR_PLAYER = (255, 220, 0)
    COLOR_EXIT = (0, 255, 120)
    COLOR_TRAIL = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_FLASH = (255, 50, 50)
    COLOR_CUE = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game state persistence
        self.maze_dims = (5, 5) # Start with 5x5, increases on win

        # Initialize state variables
        self.maze = []
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.path_trail = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.time_limit = 60 * 30  # 60 seconds at 30 FPS
        self.cue_timer = 0
        self.cue_period = 15 # 0.5s * 30fps
        self.current_cue = 0
        self.flash_timer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.time_limit
        
        self._generate_maze(*self.maze_dims)
        
        self.path_trail = []
        self.flash_timer = 0
        self.cue_timer = self.cue_period
        self._update_cue()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = 0

        # --- Update Timers ---
        self.timer -= 1
        self.cue_timer -= 1
        self.flash_timer = max(0, self.flash_timer - 1)
        
        if self.cue_timer <= 0:
            self._update_cue()
            self.cue_timer = self.cue_period

        # --- Handle Player Action ---
        if movement != 0: # An action was taken
            target_pos = list(self.player_pos)
            if movement == 1: target_pos[1] -= 1 # Up
            elif movement == 2: target_pos[1] += 1 # Down
            elif movement == 3: target_pos[0] -= 1 # Left
            elif movement == 4: target_pos[0] += 1 # Right
            
            # Check if move is valid (not into a wall)
            if self.maze[target_pos[1]][target_pos[0]] == 0:
                if movement == self.current_cue:
                    reward = 1.0 # Correct move
                    # sound: positive_beep.wav
                    self.player_pos = tuple(target_pos)
                    self.path_trail.append([self.player_pos, 30]) # pos, lifetime
                    self._update_cue() # Get next cue immediately
                    self.cue_timer = self.cue_period
                else:
                    reward = -1.0 # Incorrect move
                    # sound: error_buzz.wav
                    self.flash_timer = 5
            else:
                reward = -1.0 # Hit a wall
                # sound: wall_thud.wav
                self.flash_timer = 5
        
        self.score += reward
        
        # --- Update Path Trail ---
        self.path_trail = [[pos, life - 1] for pos, life in self.path_trail if life > 1]

        # --- Check Termination Conditions ---
        terminated = False
        if self.player_pos == self.exit_pos:
            # sound: level_complete.wav
            win_bonus = 50 * (self.timer / self.time_limit)
            reward += win_bonus
            self.score += win_bonus
            terminated = True
            
            # Increase difficulty for next round
            new_w = min(20, self.maze_dims[0] + 1)
            new_h = min(20, self.maze_dims[1] + 1)
            self.maze_dims = (new_w, new_h)

        if self.timer <= 0:
            # sound: game_over.wav
            terminated = True
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "time_left": self.timer / 30.0,
            "maze_size": self.maze_dims
        }

    def _render_game(self):
        # Calculate cell size and offsets to center the maze
        maze_pixel_w = self.maze_dims[0] * 32
        maze_pixel_h = self.maze_dims[1] * 32
        cell_w = (self.screen.get_width() - 40) / self.maze_dims[0]
        cell_h = (self.screen.get_height() - 80) / self.maze_dims[1]
        cell_size = int(min(cell_w, cell_h))
        offset_x = (self.screen.get_width() - self.maze_dims[0] * cell_size) // 2
        offset_y = (self.screen.get_height() - self.maze_dims[1] * cell_size) // 2 + 30

        # Render maze
        for y, row in enumerate(self.maze):
            for x, cell in enumerate(row):
                rect = pygame.Rect(offset_x + x * cell_size, offset_y + y * cell_size, cell_size, cell_size)
                color = self.COLOR_WALL if cell == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)

        # Render path trail
        for pos, life in self.path_trail:
            center_x = int(offset_x + (pos[0] + 0.5) * cell_size)
            center_y = int(offset_y + (pos[1] + 0.5) * cell_size)
            radius = int(cell_size * 0.2 * (life / 30.0))
            alpha = int(150 * (life / 30.0))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, (*self.COLOR_TRAIL, alpha))

        # Render exit
        exit_center_x = int(offset_x + (self.exit_pos[0] + 0.5) * cell_size)
        exit_center_y = int(offset_y + (self.exit_pos[1] + 0.5) * cell_size)
        glow_size = int(cell_size * 0.8)
        for i in range(glow_size, 0, -2):
            alpha = int(80 * (1 - i / glow_size))
            pygame.gfxdraw.filled_circle(self.screen, exit_center_x, exit_center_y, i, (*self.COLOR_EXIT, alpha))
        pygame.gfxdraw.filled_circle(self.screen, exit_center_x, exit_center_y, int(cell_size * 0.35), self.COLOR_EXIT)
        
        # Render player
        player_center_x = int(offset_x + (self.player_pos[0] + 0.5) * cell_size)
        player_center_y = int(offset_y + (self.player_pos[1] + 0.5) * cell_size)
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, int(cell_size * 0.4), self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, int(cell_size * 0.4), self.COLOR_PLAYER)

        # Render cue indicator
        pulse = abs(math.sin(self.cue_timer * math.pi / self.cue_period))
        arrow_size = int(cell_size * 0.2 * (1 + pulse * 0.5))
        self._draw_arrow(self.screen, self.COLOR_CUE, (player_center_x, player_center_y), self.current_cue, arrow_size)

        # Render incorrect move flash
        if self.flash_timer > 0:
            flash_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            alpha = int(100 * (self.flash_timer / 5.0))
            flash_surface.fill((*self.COLOR_FLASH, alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {max(0, self.timer / 30.0):.1f}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (20, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.screen.get_width() - 20, 10))
        self.screen.blit(score_surf, score_rect)
        
        if self.game_over:
            if self.player_pos == self.exit_pos:
                msg = "LEVEL COMPLETE"
            else:
                msg = "TIME UP"
            msg_surf = self.font_large.render(msg, True, self.COLOR_PLAYER)
            msg_rect = msg_surf.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _generate_maze(self, width, height):
        # Ensure dimensions are odd for the algorithm
        w, h = (width // 2 * 2 + 1, height // 2 * 2 + 1)
        self.maze = [[1] * w for _ in range(h)]
        
        # Randomized DFS
        start_x, start_y = (1, 1)
        stack = [(start_x, start_y)]
        self.maze[start_y][start_x] = 0
        visited_cells = {(start_x, start_y)}
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < w - 1 and 0 < ny < h - 1 and (nx, ny) not in visited_cells:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                # Carve path
                self.maze[ny][nx] = 0
                self.maze[cy + (ny - cy) // 2][cx + (nx - cx) // 2] = 0
                visited_cells.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        self.player_pos = (start_x, start_y)
        
        # Find the furthest point for the exit
        distances = self._bfs_distances(self.player_pos)
        max_dist = -1
        furthest_pos = self.player_pos
        for pos, dist in distances.items():
            if dist > max_dist:
                max_dist = dist
                furthest_pos = pos
        self.exit_pos = furthest_pos
        
        # Update maze_dims to reflect actual generated size
        self.maze_dims = (w, h)

    def _bfs_distances(self, start):
        q = [(start, 0)]
        visited = {start}
        distances = {start: 0}
        while q:
            (cx, cy), dist = q.pop(0)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= ny < len(self.maze) and 0 <= nx < len(self.maze[0]) and \
                   self.maze[ny][nx] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    distances[(nx, ny)] = dist + 1
                    q.append(((nx, ny), dist + 1))
        return distances

    def _a_star_path(self, start, end):
        open_set = [(0, start)]
        came_from = {}
        g_score = { (x,y): float('inf') for y, r in enumerate(self.maze) for x, c in enumerate(r) }
        g_score[start] = 0
        f_score = { (x,y): float('inf') for y, r in enumerate(self.maze) for x, c in enumerate(r) }
        f_score[start] = abs(start[0] - end[0]) + abs(start[1] - end[1])

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.maze_dims[0] and 0 <= neighbor[1] < self.maze_dims[1]):
                    continue
                if self.maze[neighbor[1]][neighbor[0]] == 1:
                    continue
                
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def _update_cue(self):
        path = self._a_star_path(self.player_pos, self.exit_pos)
        if path and len(path) > 0:
            next_pos = path[0]
            dx = next_pos[0] - self.player_pos[0]
            dy = next_pos[1] - self.player_pos[1]
            if dy == -1: self.current_cue = 1 # Up
            elif dy == 1: self.current_cue = 2 # Down
            elif dx == -1: self.current_cue = 3 # Left
            elif dx == 1: self.current_cue = 4 # Right
        else:
            self.current_cue = 0 # No path found, no cue

    def _draw_arrow(self, surface, color, pos, direction, size):
        x, y = pos
        if direction == 1: # Up
            points = [(x, y - size), (x - size, y), (x + size, y)]
        elif direction == 2: # Down
            points = [(x, y + size), (x - size, y), (x + size, y)]
        elif direction == 3: # Left
            points = [(x - size, y), (x, y - size), (x, y + size)]
        elif direction == 4: # Right
            points = [(x + size, y), (x, y - size), (x, y + size)]
        else:
            return
        pygame.draw.polygon(surface, color, points)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Rhythm Maze")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Final Score: {info['score']:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()