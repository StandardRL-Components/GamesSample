import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Navigate a procedurally generated, color-coded momentum maze,
    conserving speed to reach the exit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a procedurally generated maze, conserving speed to reach the exit. "
        "Avoid traps and manage your momentum to progress."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move. Avoid traps and manage your momentum to "
        "stay alive and find the exit."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 50)
        self.np_random = None

        # --- Game Constants ---
        self.MAX_STEPS = 2000
        self.MAX_MOMENTUM = 100.0
        self.INITIAL_MOMENTUM = 25.0
        self.MOMENTUM_GAIN = 2.0
        self.MOMENTUM_DECAY = 1.0
        self.MOMENTUM_THRESHOLD = self.MAX_MOMENTUM * 0.2
        self.TRAP_RED_PENALTY = 0.5 # Multiplier
        self.TRAP_BLUE_PENALTY = 0.0 # Sets to this value

        # --- Visuals ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_WALL = (50, 60, 80)
        self.COLOR_PATH = (70, 80, 100)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_TRAP_RED = (255, 50, 50)
        self.COLOR_TRAP_BLUE = (50, 150, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_MOMENTUM_BAR_BG = (40, 40, 60)
        self.COLOR_MOMENTUM_BAR_FG = (40, 200, 100)

        # --- State Variables ---
        self.maze_level = 0
        self.maze_base_size = (10, 8)  # width, height in cells
        self.maze_size = self.maze_base_size
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.traps = {}
        self.momentum = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.cell_width = 0
        self.cell_height = 0
        self.grid_offset_x = 0
        self.grid_offset_y = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.momentum = self.INITIAL_MOMENTUM

        if self.maze_level > 0 and self.maze_level % 5 == 0:
            w, h = self.maze_base_size
            scale_factor = 1 + (0.05 * (self.maze_level // 5))
            new_w = max(self.maze_base_size[0], int(w * scale_factor))
            new_h = max(self.maze_base_size[1], int(h * scale_factor))
            self.maze_size = (new_w, new_h)

        self._generate_maze()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held and shift_held are ignored per brief
        
        self.steps += 1
        reward = 0
        
        # --- Update Player Position ---
        prev_pos = self.player_pos
        px, py = self.player_pos
        moved = False
        if movement == 1 and py > 0 and self.maze[py - 1][px] == 1: # Up
            self.player_pos = (px, py - 1)
            moved = True
        elif movement == 2 and py < self.maze_size[1] - 1 and self.maze[py + 1][px] == 1: # Down
            self.player_pos = (px, py + 1)
            moved = True
        elif movement == 3 and px > 0 and self.maze[py][px - 1] == 1: # Left
            self.player_pos = (px - 1, py)
            moved = True
        elif movement == 4 and px < self.maze_size[0] - 1 and self.maze[py][px + 1] == 1: # Right
            self.player_pos = (px + 1, py)
            moved = True

        # --- Update Momentum and Rewards ---
        if moved:
            self.momentum = min(self.MAX_MOMENTUM, self.momentum + self.MOMENTUM_GAIN)
        elif movement != 0: # Attempted to move into a wall
            reward -= 0.5
            self.momentum -= self.MOMENTUM_DECAY * 2
        else: # No-op
            reward -= 0.1
            self.momentum -= self.MOMENTUM_DECAY

        if self.momentum > self.MOMENTUM_THRESHOLD:
            reward += 0.1

        # --- Check Traps ---
        if self.player_pos in self.traps:
            trap_type = self.traps.pop(self.player_pos)
            if trap_type == 'red':
                reward -= 5
                self.momentum *= self.TRAP_RED_PENALTY
                self._create_particles(self.player_pos, self.COLOR_TRAP_RED, 20)
            elif trap_type == 'blue':
                reward -= 10
                self.momentum = self.TRAP_BLUE_PENALTY
                self._create_particles(self.player_pos, self.COLOR_TRAP_BLUE, 30)

        self.momentum = max(0, self.momentum)
        
        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.player_pos == self.exit_pos:
            reward += 100
            terminated = True
            self.game_over = True
            self.maze_level += 1
        elif self.momentum <= 0:
            reward -= 20 # Small penalty for running out of steam
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_particles()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.maze_level}

    def _grid_to_pixel(self, pos):
        gx, gy = pos
        x = self.grid_offset_x + gx * self.cell_width + self.cell_width / 2
        y = self.grid_offset_y + gy * self.cell_height + self.cell_height / 2
        return int(x), int(y)

    def _render_game(self):
        # Render maze
        for r in range(self.maze_size[1]):
            for c in range(self.maze_size[0]):
                color = self.COLOR_PATH if self.maze[r][c] == 1 else self.COLOR_WALL
                rect = pygame.Rect(
                    self.grid_offset_x + c * self.cell_width,
                    self.grid_offset_y + r * self.cell_height,
                    self.cell_width, self.cell_height
                )
                pygame.draw.rect(self.screen, color, rect)

        # Render traps
        trap_radius = int(min(self.cell_width, self.cell_height) * 0.25)
        for pos, trap_type in self.traps.items():
            px, py = self._grid_to_pixel(pos)
            color = self.COLOR_TRAP_RED if trap_type == 'red' else self.COLOR_TRAP_BLUE
            pygame.gfxdraw.filled_circle(self.screen, px, py, trap_radius, color)
            pygame.gfxdraw.aacircle(self.screen, px, py, trap_radius, color)

        # Render exit
        exit_px, exit_py = self._grid_to_pixel(self.exit_pos)
        exit_radius = int(min(self.cell_width, self.cell_height) * 0.35)
        glow = int(exit_radius * (1.2 + 0.2 * math.sin(self.steps * 0.1)))
        pygame.gfxdraw.filled_circle(self.screen, exit_px, exit_py, glow, (*self.COLOR_EXIT, 50))
        pygame.gfxdraw.filled_circle(self.screen, exit_px, exit_py, exit_radius, self.COLOR_EXIT)
        pygame.gfxdraw.aacircle(self.screen, exit_px, exit_py, exit_radius, self.COLOR_EXIT)

        # Render player
        player_px, player_py = self._grid_to_pixel(self.player_pos)
        player_radius = int(min(self.cell_width, self.cell_height) * 0.3)
        path_pulse_radius = int(player_radius * (1.5 + 0.3 * math.sin(self.steps * 0.2)))
        pygame.gfxdraw.filled_circle(self.screen, player_px, player_py, path_pulse_radius, (*self.COLOR_PLAYER, 30))
        glow_radius = int(player_radius * (1.3 + 0.2 * math.sin(self.steps * 0.25)))
        pygame.gfxdraw.filled_circle(self.screen, player_px, player_py, glow_radius, (*self.COLOR_PLAYER, 80))
        pygame.gfxdraw.filled_circle(self.screen, player_px, player_py, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_px, player_py, player_radius, self.COLOR_PLAYER)
        
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _render_ui(self):
        # Momentum Bar
        bar_width = 200
        bar_height = 20
        bar_x = (self.screen_width - bar_width) / 2
        bar_y = 15
        pygame.draw.rect(self.screen, self.COLOR_MOMENTUM_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        fill_width = bar_width * (self.momentum / self.MAX_MOMENTUM)
        pygame.draw.rect(self.screen, self.COLOR_MOMENTUM_BAR_FG, (bar_x, bar_y, fill_width, bar_height), border_radius=4)

        # Score Text
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 20, 15))

        # Level Text
        level_text = self.font_ui.render(f"Level: {self.maze_level + 1}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (20, 15))

    def _generate_maze(self):
        w, h = self.maze_size
        self.maze = np.zeros((h, w), dtype=int)
        
        stack = [(0, 0)]
        visited = set([(0, 0)])
        self.maze[0][0] = 1
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx * 2, cy + dy * 2
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                    neighbors.append((dx, dy, nx, ny))
            
            if neighbors:
                idx = self.np_random.integers(len(neighbors))
                dx, dy, nx, ny = neighbors[idx]
                self.maze[cy + dy][cx + dx] = 1
                self.maze[ny][nx] = 1
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        path_nodes = [(c, r) for r in range(h) for c in range(w) if self.maze[r][c] == 1]
        
        start_node = path_nodes[0]
        farthest_node, _ = self._bfs(start_node)
        self.exit_pos, path = self._bfs(farthest_node)
        self.player_pos, _ = self._bfs(self.exit_pos)

        self.traps = {}
        num_traps = int(len(path) * 0.1)
        path_candidates = [p for p in path if p != self.player_pos and p != self.exit_pos]
        
        if len(path_candidates) > 0:
            trap_indices = self.np_random.choice(len(path_candidates), size=min(num_traps, len(path_candidates)), replace=False)
            for i in trap_indices:
                pos = path_candidates[i]
                trap_type = self.np_random.choice(['red', 'blue'])
                self.traps[pos] = trap_type

        self.cell_width = self.screen_width / w
        self.cell_height = (self.screen_height - 60) / h
        self.grid_offset_x = 0
        self.grid_offset_y = 50

    def _bfs(self, start_node):
        q = [(start_node, [start_node])]
        visited = {start_node}
        farthest_node = start_node
        longest_path = [start_node]

        while q:
            (vx, vy), path = q.pop(0)
            
            if len(path) > len(longest_path):
                longest_path = path
                farthest_node = (vx, vy)

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = vx + dx, vy + dy
                if 0 <= nx < self.maze_size[0] and 0 <= ny < self.maze_size[1] and self.maze[ny][nx] == 1 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append(((nx, ny), path + [(nx, ny)]))
        return farthest_node, longest_path

    def _create_particles(self, grid_pos, color, count):
        pixel_pos = self._grid_to_pixel(grid_pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pixel_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.uniform(2, 6),
                'color': color,
                'life': self.np_random.uniform(20, 40),
                'max_life': 40
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] *= 0.97
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Use a real display for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Momentum Maze")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    truncated = False
    
    while running:
        if terminated or truncated:
            print(f"Game Over! Score: {env.score:.2f}, Steps: {env.steps}, Level: {env.maze_level}")
            obs, info = env.reset()
            terminated = False
            truncated = False

        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    terminated = False
                    truncated = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(15) # Run at 15 FPS for turn-based feel
        
    env.close()