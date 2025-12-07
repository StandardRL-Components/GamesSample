
# Generated: 2025-08-27T16:58:32.422027
# Source Brief: brief_01382.md
# Brief Index: 1382

        
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
        "Controls: Use arrow keys to move the selector. Press Space to clear a cluster of 3 or more same-colored blocks."
    )

    game_description = (
        "Match adjacent colored blocks to clear them from the board. Clear 5 rows before the 60-second timer runs out to win!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        self.GRID_ROWS, self.GRID_COLS = 8, 10
        self.BLOCK_SIZE = 40
        self.MIN_CLUSTER_SIZE = 3
        
        self.BOARD_WIDTH = self.GRID_COLS * self.BLOCK_SIZE
        self.BOARD_HEIGHT = self.GRID_ROWS * self.BLOCK_SIZE
        self.BOARD_OFFSET_X = (self.WIDTH - self.BOARD_WIDTH) // 2
        self.BOARD_OFFSET_Y = (self.HEIGHT - self.BOARD_HEIGHT) // 2
        
        self.WIN_ROWS = 5
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS + 10 # A bit over the time limit

        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_WARN = (255, 100, 100)
        self.COLOR_PROGRESS_BG = (40, 50, 70)
        self.COLOR_PROGRESS_FG = (100, 220, 150)
        
        self.BLOCK_COLORS = [
            (0, 0, 0), # 0: Empty
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.NUM_COLORS = len(self.BLOCK_COLORS) - 1

        # --- Rewards ---
        self.REWARD_PER_BLOCK = 0.05
        self.REWARD_PER_ROW = 5.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -100.0

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.timer = None
        self.score = None
        self.rows_cleared = None
        self.game_over = None
        self.steps = None
        self.prev_space_held = None
        self.particles = None
        self.np_random = None

        self.reset()
        
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.rows_cleared = 0
        self.timer = self.TIME_LIMIT_SECONDS
        self.game_over = False
        self.prev_space_held = False
        self.particles = []
        
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self._create_initial_grid()
        
        return self._get_observation(), self._get_info()

    def _create_initial_grid(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_ROWS, self.GRID_COLS))
        # Ensure no valid clusters exist at the start
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                while len(self._find_cluster(r, c)) >= self.MIN_CLUSTER_SIZE:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def step(self, action):
        self.clock.tick(self.FPS)
        self.steps += 1
        reward = 0.0
        
        if not self.game_over:
            self.timer -= 1.0 / self.FPS
            reward = self._handle_logic(action)

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.rows_cleared >= self.WIN_ROWS:
                reward += self.REWARD_WIN
            else: # Timeout
                reward += self.REWARD_LOSE

        self._update_particles()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_logic(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        step_reward = 0

        # --- Handle Input ---
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)
        
        space_press = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        if space_press:
            # --- Find and Clear Cluster ---
            cluster = self._find_cluster(self.cursor_pos[0], self.cursor_pos[1])
            if len(cluster) >= self.MIN_CLUSTER_SIZE:
                # sfx: block_clear.wav
                blocks_cleared = len(cluster)
                step_reward += blocks_cleared * self.REWARD_PER_BLOCK
                self.score += blocks_cleared

                color_idx = self.grid[self.cursor_pos[0], self.cursor_pos[1]]
                for r, c in cluster:
                    self.grid[r, c] = 0 # Mark as empty
                    self._create_particles(r, c, color_idx)
                
                # --- Apply Gravity and Refill ---
                self._apply_gravity()
                rows_cleared_this_step = self._check_and_clear_rows()
                if rows_cleared_this_step > 0:
                    # sfx: row_clear_fanfare.wav
                    step_reward += rows_cleared_this_step * self.REWARD_PER_ROW
                    self.score += rows_cleared_this_step * 100
                    self.rows_cleared += rows_cleared_this_step
                self._refill_board()
        
        return step_reward

    def _find_cluster(self, r_start, c_start):
        if self.grid[r_start, c_start] == 0:
            return set()
        
        target_color = self.grid[r_start, c_start]
        q = [(r_start, c_start)]
        visited = set(q)
        
        while q:
            r, c = q.pop(0)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                    if (nr, nc) not in visited and self.grid[nr, nc] == target_color:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return visited

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1
    
    def _check_and_clear_rows(self):
        cleared_rows_indices = [r for r in range(self.GRID_ROWS) if np.all(self.grid[r] == 0)]
        if not cleared_rows_indices:
            return 0
        
        num_cleared = len(cleared_rows_indices)
        
        new_grid = np.zeros_like(self.grid)
        new_grid_row = self.GRID_ROWS - 1
        for r in range(self.GRID_ROWS - 1, -1, -1):
            if r not in cleared_rows_indices:
                new_grid[new_grid_row] = self.grid[r]
                new_grid_row -= 1
        
        self.grid = new_grid
        return num_cleared

    def _refill_board(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _check_termination(self):
        return self.rows_cleared >= self.WIN_ROWS or self.timer <= 0 or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        board_rect = pygame.Rect(self.BOARD_OFFSET_X, self.BOARD_OFFSET_Y, self.BOARD_WIDTH, self.BOARD_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, board_rect)

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_idx = self.grid[r, c]
                if color_idx > 0:
                    block_rect = pygame.Rect(
                        self.BOARD_OFFSET_X + c * self.BLOCK_SIZE,
                        self.BOARD_OFFSET_Y + r * self.BLOCK_SIZE,
                        self.BLOCK_SIZE, self.BLOCK_SIZE
                    )
                    main_color = self.BLOCK_COLORS[color_idx]
                    border_color = tuple(max(0, x - 50) for x in main_color)
                    pygame.gfxdraw.box(self.screen, block_rect.inflate(-4, -4), main_color)
                    pygame.draw.rect(self.screen, border_color, block_rect, 2)

        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        cursor_alpha = 100 + int(pulse * 100)
        cursor_color = (*self.COLOR_TEXT, cursor_alpha)
        
        cursor_rect = pygame.Rect(
            self.BOARD_OFFSET_X + self.cursor_pos[1] * self.BLOCK_SIZE,
            self.BOARD_OFFSET_Y + self.cursor_pos[0] * self.BLOCK_SIZE,
            self.BLOCK_SIZE, self.BLOCK_SIZE
        )
        
        surf = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(surf, cursor_color, surf.get_rect(), 5, border_radius=5)
        self.screen.blit(surf, cursor_rect.topleft)

        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        timer_val = max(0, self.timer)
        minutes = int(timer_val) // 60
        seconds = int(timer_val) % 60
        timer_color = self.COLOR_TEXT if timer_val > 10 else self.COLOR_TEXT_WARN
        timer_text = self.font_main.render(f"{minutes:02}:{seconds:02}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        progress_bar_y = self.HEIGHT - 30
        progress_bar_width = self.WIDTH - 20
        
        bg_rect = pygame.Rect(10, progress_bar_y, progress_bar_width, 20)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BG, bg_rect, border_radius=5)
        
        progress = min(1.0, self.rows_cleared / self.WIN_ROWS)
        fg_width = progress * (progress_bar_width - 4)
        fg_rect = pygame.Rect(12, progress_bar_y + 2, fg_width, 16)
        if fg_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_PROGRESS_FG, fg_rect, border_radius=4)
        
        progress_text = self.font_small.render(f"Rows Cleared: {self.rows_cleared} / {self.WIN_ROWS}", True, self.COLOR_TEXT)
        progress_text_rect = progress_text.get_rect(center=bg_rect.center)
        self.screen.blit(progress_text, progress_text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.rows_cleared >= self.WIN_ROWS else "TIME'S UP!"
            msg_text = self.font_main.render(message, True, self.COLOR_PROGRESS_FG if self.rows_cleared >= self.WIN_ROWS else self.COLOR_TEXT_WARN)
            msg_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "rows_cleared": self.rows_cleared,
            "cursor_pos": list(self.cursor_pos),
        }

    def _create_particles(self, r, c, color_idx):
        center_x = self.BOARD_OFFSET_X + c * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        center_y = self.BOARD_OFFSET_Y + r * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        color = self.BLOCK_COLORS[color_idx]

        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': vel,
                'radius': random.uniform(3, 7),
                'life': self.FPS * 0.5, # 0.5 seconds
                'color': color
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['radius'] *= 0.95
            p['life'] -= 1
            if p['life'] > 0 and p['radius'] > 0.5:
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Block Breaker Gym Env")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    
    while not terminated:
        movement = 0 # none
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Rows: {info['rows_cleared']}")

        if terminated:
            print("Game Over!")
            print(f"Final Score: {info['score']}, Rows Cleared: {info['rows_cleared']}")
            
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()