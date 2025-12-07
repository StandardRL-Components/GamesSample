
# Generated: 2025-08-28T04:12:02.398887
# Source Brief: brief_02235.md
# Brief Index: 2235

        
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

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to guide the glowing line through the maze. Avoid touching the walls."
    )

    game_description = (
        "Guide a glowing line through a procedurally generated, shifting labyrinth to reach the exit within the time limit."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAZE_COLS, self.MAZE_ROWS = 31, 19  # Must be odd
        self.CELL_W = self.WIDTH // self.MAZE_COLS
        self.CELL_H = self.HEIGHT // self.MAZE_ROWS
        self.MAX_STEPS = 180 * self.FPS # 180 seconds

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_WALL = (26, 35, 126)
        self.COLOR_EXIT = (0, 230, 118)
        self.COLOR_PLAYER = (255, 214, 0)
        self.COLOR_PARTICLE = (255, 23, 68)
        self.COLOR_TEXT = (255, 255, 255)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # State variables (initialized in reset)
        self.maze = None
        self.player_grid_pos = None
        self.player_pixel_pos = None
        self.player_trail = None
        self.exit_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.shift_timer = 0
        self.shift_freq = 0
        self.shift_animations = []
        self.last_dist_to_exit = 0
        self.player_move_cooldown = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.shift_animations = []
        self.player_move_cooldown = 0
        
        self.maze = self._generate_maze(self.MAZE_COLS, self.MAZE_ROWS)
        self.start_pos = [1, 1]
        self.exit_pos = [self.MAZE_COLS - 2, self.MAZE_ROWS - 2]
        
        self.player_grid_pos = list(self.start_pos)
        self.player_pixel_pos = [
            self.start_pos[0] * self.CELL_W + self.CELL_W / 2,
            self.start_pos[1] * self.CELL_H + self.CELL_H / 2
        ]
        self.player_trail = [list(self.player_pixel_pos)] * 15
        
        self.shift_freq = 0.2  # Hz
        self.shift_timer = int(self.FPS / self.shift_freq)
        
        self.last_dist_to_exit = self._get_dist_to_exit()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        self._update_maze_shifts()
        
        if self.maze[self.player_grid_pos[1], self.player_grid_pos[0]] == 1:
            # sound: player_death_explosion.wav
            reward = -10
            self._create_explosion(self.player_trail[-1])
            self.game_over = True
        else:
            self._handle_input(action)
        
        self._update_player_movement_and_trail()
        self._update_particles()
        
        terminated = self.game_over
        if not terminated:
            if self._check_exit_reached():
                # sound: level_complete.wav
                reward += 100
                terminated = True
            elif self.steps >= self.MAX_STEPS:
                # sound: time_up.wav
                terminated = True
        
        if not terminated:
            new_dist = self._get_dist_to_exit()
            if new_dist < self.last_dist_to_exit:
                reward += 0.1
            elif new_dist > self.last_dist_to_exit:
                reward -= 0.2
            self.last_dist_to_exit = new_dist
        
        self.score += reward
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        if self.player_move_cooldown > 0:
            return

        movement = action[0]
        dx, dy = 0, 0
        if movement == 1: dy = -1
        elif movement == 2: dy = 1
        elif movement == 3: dx = -1
        elif movement == 4: dx = 1

        if dx != 0 or dy != 0:
            next_grid_pos = [self.player_grid_pos[0] + dx, self.player_grid_pos[1] + dy]
            if 0 <= next_grid_pos[0] < self.MAZE_COLS and 0 <= next_grid_pos[1] < self.MAZE_ROWS:
                if self.maze[next_grid_pos[1], next_grid_pos[0]] == 0:
                    self.player_grid_pos = next_grid_pos
                    self.player_move_cooldown = 4 # frames per cell

    def _update_player_movement_and_trail(self):
        if self.player_move_cooldown > 0:
            self.player_move_cooldown -= 1

        target_px = [
            self.player_grid_pos[0] * self.CELL_W + self.CELL_W / 2,
            self.player_grid_pos[1] * self.CELL_H + self.CELL_H / 2
        ]
        self.player_pixel_pos[0] += (target_px[0] - self.player_pixel_pos[0]) * 0.5
        self.player_pixel_pos[1] += (target_px[1] - self.player_pixel_pos[1]) * 0.5

        adjusted_pixel_pos = list(self.player_pixel_pos)
        row_width_pixels = (self.MAZE_COLS - 2) * self.CELL_W

        for anim in self.shift_animations:
            if anim['row'] == self.player_grid_pos[1] and anim['progress'] < 1.0:
                offset = anim['dir'] * anim['progress'] * self.CELL_W
                
                base_x = (self.player_pixel_pos[0] - self.CELL_W)
                anim_x = (base_x + offset) % row_width_pixels
                adjusted_pixel_pos[0] = anim_x + self.CELL_W
                break

        self.player_trail.pop(0)
        self.player_trail.append(adjusted_pixel_pos)

    def _update_maze_shifts(self):
        self.shift_freq = min(2.0, 0.2 + 0.05 * (self.steps // 100))
        
        finished_anims = []
        for anim in self.shift_animations:
            anim['progress'] += 1 / 15  # 15-frame animation
            if anim['progress'] >= 1.0:
                finished_anims.append(anim)
        
        for anim in finished_anims:
            row_idx = anim["row"]
            row_data = self.maze[row_idx, 1:-1]
            shifted_row = np.roll(row_data, anim["dir"])
            self.maze[row_idx, 1:-1] = shifted_row
            self.shift_animations.remove(anim)
        
        self.shift_timer -= 1
        if self.shift_timer <= 0 and not self.shift_animations:
            # sound: wall_shift.wav
            self.shift_timer = int(self.FPS / self.shift_freq)
            shiftable_rows = [r for r in range(1, self.MAZE_ROWS - 1) if r % 2 != 0]
            if shiftable_rows:
                row_to_shift = random.choice(shiftable_rows)
                direction = random.choice([-1, 1])
                self.shift_animations.append({"row": row_to_shift, "dir": direction, "progress": 0})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_maze()
        self._render_exit()
        self._render_particles()
        if not self.game_over:
            self._render_player()
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_maze(self):
        row_width_pixels = (self.MAZE_COLS - 2) * self.CELL_W

        for r in range(self.MAZE_ROWS):
            is_shifting = any(anim['row'] == r for anim in self.shift_animations)
            
            if not is_shifting:
                for c in range(self.MAZE_COLS):
                    if self.maze[r, c] == 1:
                        pygame.draw.rect(self.screen, self.COLOR_WALL, (c * self.CELL_W, r * self.CELL_H, self.CELL_W + 1, self.CELL_H + 1))
            else:
                # Draw static side walls
                pygame.draw.rect(self.screen, self.COLOR_WALL, (0, r * self.CELL_H, self.CELL_W, self.CELL_H))
                pygame.draw.rect(self.screen, self.COLOR_WALL, ((self.MAZE_COLS - 1) * self.CELL_W, r * self.CELL_H, self.CELL_W, self.CELL_H))

                anim = next(a for a in self.shift_animations if a['row'] == r)
                offset = anim['dir'] * anim['progress'] * self.CELL_W
                
                # Erase the area to be redrawn
                pygame.draw.rect(self.screen, self.COLOR_BG, (self.CELL_W, r * self.CELL_H, row_width_pixels, self.CELL_H))

                for c in range(1, self.MAZE_COLS - 1):
                    if self.maze[r, c] == 1:
                        base_x = (c - 1) * self.CELL_W
                        anim_x = (base_x + offset) % row_width_pixels
                        final_x = anim_x + self.CELL_W
                        pygame.draw.rect(self.screen, self.COLOR_WALL, (final_x, r * self.CELL_H, self.CELL_W + 1, self.CELL_H + 1))

    def _render_player(self):
        # Trail
        for i, pos in enumerate(self.player_trail):
            lerp = i / len(self.player_trail)
            radius = int((self.CELL_W / 2.5) * lerp)
            if radius < 1: continue
            
            color = (
                int(self.COLOR_PLAYER[0] * lerp + self.COLOR_BG[0] * (1 - lerp)),
                int(self.COLOR_PLAYER[1] * lerp + self.COLOR_BG[1] * (1 - lerp)),
                int(self.COLOR_PLAYER[2] * lerp + self.COLOR_BG[2] * (1 - lerp)),
            )
            pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), radius)

        # Head
        px, py = self.player_trail[-1]
        
        # Glow
        for i in range(4, 0, -1):
            alpha = 100 - i * 20
            radius = int(self.CELL_W / 3 + i * 1.5)
            glow_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER, alpha), (radius, radius), radius)
            self.screen.blit(glow_surf, (int(px) - radius, int(py) - radius))
        
        # Core
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (int(px), int(py)), int(self.CELL_W / 3))

    def _render_exit(self):
        ex, ey = self.exit_pos
        rect = pygame.Rect(ex * self.CELL_W, ey * self.CELL_H, self.CELL_W, self.CELL_H)
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in self.COLOR_EXIT)
        pygame.draw.rect(self.screen, color, rect)

    def _create_explosion(self, pos):
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 6)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.randint(20, 40)
            self.particles.append([list(pos), [vx, vy], lifetime])

    def _update_particles(self):
        for p in self.particles[:]:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[1][1] += 0.1  # Gravity
            p[2] -= 1
            if p[2] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            pos, _, lifetime = p
            radius = int(max(0, lifetime / 8))
            if radius == 0: continue
            alpha = int(255 * (lifetime / 40))
            color = (*self.COLOR_PARTICLE, alpha)
            
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (int(pos[0]) - radius, int(pos[1]) - radius))

    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {time_left:.1f}"
        score_text = f"SCORE: {int(self.score)}"
        
        time_surf = self.font.render(time_text, True, self.COLOR_TEXT)
        score_surf = self.font.render(score_text, True, self.COLOR_TEXT)
        
        self.screen.blit(time_surf, (10, 10))
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_dist_to_exit(self):
        return math.hypot(self.player_grid_pos[0] - self.exit_pos[0], self.player_grid_pos[1] - self.exit_pos[1])

    def _check_exit_reached(self):
        return self.player_grid_pos == self.exit_pos

    def _generate_maze(self, width, height):
        maze = np.ones((height, width), dtype=np.uint8)
        stack = [(1, 1)]
        maze[1, 1] = 0
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < width and 0 < ny < height and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            if neighbors:
                nx, ny = random.choice(neighbors)
                maze[ny, nx] = 0
                maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()