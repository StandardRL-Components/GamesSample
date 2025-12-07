
# Generated: 2025-08-28T03:08:08.316989
# Source Brief: brief_04830.md
# Brief Index: 4830

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to clear a cluster of blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the grid of colorful blocks by clicking on clusters of the same color before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.BLOCK_SIZE = self.HEIGHT // self.GRID_HEIGHT
        
        self.MAX_STEPS = 1800 # 60 seconds * 30 fps
        self.INITIAL_TIME = 60.0
        self.MOVE_COOLDOWN_FRAMES = 4 # Cooldown for cursor movement

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.BLOCK_COLORS = {
            1: (255, 80, 80),   # Red
            2: (80, 255, 80),   # Green
            3: (80, 150, 255),  # Blue
            4: (255, 255, 80),  # Yellow
            5: (200, 80, 255),  # Purple
        }
        self.BLOCK_HIGHLIGHT_COLORS = {k: tuple(min(255, c + 50) for c in v) for k, v in self.BLOCK_COLORS.items()}
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GAMEOVER = (255, 50, 50)
        self.COLOR_WIN = (50, 255, 50)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_gameover = pygame.font.Font(None, 80)
        
        # Etc...
        self.grid = None
        self.cursor_pos = None
        self.score = 0
        self.steps = 0
        self.timer = 0.0
        self.game_over = False
        self.win = False
        self.space_was_held = False
        self.move_cooldown = 0
        self.particles = []
        self.rng = None
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = self.INITIAL_TIME
        self.space_was_held = False
        self.move_cooldown = 0
        self.particles = []
        
        self._generate_grid()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        dt = self.clock.tick(30) / 1000.0
        self.steps += 1
        self.timer = max(0, self.timer - dt)
        
        reward = -0.01 # Small penalty for time passing

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # Update game logic
        self.move_cooldown = max(0, self.move_cooldown - 1)
        if self.move_cooldown == 0 and movement != 0:
            if movement == 1: self.cursor_pos[1] -= 1 # Up
            elif movement == 2: self.cursor_pos[1] += 1 # Down
            elif movement == 3: self.cursor_pos[0] -= 1 # Left
            elif movement == 4: self.cursor_pos[0] += 1 # Right
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)
            self.move_cooldown = self.MOVE_COOLDOWN_FRAMES

        click_reward = 0
        if space_held and not self.space_was_held:
            # sfx: click
            click_reward = self._perform_click()
        self.space_was_held = space_held
        
        reward += click_reward

        self._update_particles(dt)
        
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 50
                # sfx: win_jingle
            else:
                reward -= 50
                # sfx: lose_buzzer
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _check_termination(self):
        if self.game_over:
            return True
        if np.sum(self.grid > 0) == 0:
            self.win = True
            self.game_over = True
            return True
        if self.timer <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _generate_grid(self):
        self.grid = self.rng.integers(1, len(self.BLOCK_COLORS) + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.int8)

    def _perform_click(self):
        cx, cy = self.cursor_pos
        if self.grid[cx, cy] == 0:
            return 0

        cluster = self._find_cluster(cx, cy)
        
        if len(cluster) < 2: # Do not clear single blocks
            # sfx: invalid_click
            return 0

        reward = 0
        num_cleared = len(cluster)
        reward += num_cleared # +1 per block
        if num_cleared >= 5:
            reward += 5 # Bonus for large clusters
        
        # sfx: block_break
        for x, y in cluster:
            color = self.BLOCK_COLORS[self.grid[x, y]]
            self._create_particles(x, y, color)
            self.grid[x, y] = 0
        
        self.score += num_cleared
        self._apply_gravity()
        return reward

    def _find_cluster(self, start_x, start_y):
        target_color = self.grid[start_x, start_y]
        if target_color == 0:
            return set()

        q = [(start_x, start_y)]
        visited = set(q)
        head = 0
        while head < len(q):
            x, y = q[head]
            head += 1
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[nx, ny] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return visited

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            col = self.grid[x, :]
            non_empty = col[col > 0]
            num_empty = self.GRID_HEIGHT - len(non_empty)
            new_col = np.concatenate((np.zeros(num_empty, dtype=np.int8), non_empty))
            self.grid[x, :] = new_col

    def _create_particles(self, grid_x, grid_y, color):
        px = (grid_x + 0.5) * self.BLOCK_SIZE
        py = (grid_y + 0.5) * self.BLOCK_SIZE
        for _ in range(10):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 80 + 20
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.rng.random() * 0.5 + 0.3
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self, dt):
        gravity = 200.0
        for p in self.particles:
            p['pos'][0] += p['vel'][0] * dt
            p['pos'][1] += p['vel'][1] * dt
            p['vel'][1] += gravity * dt
            p['life'] -= dt
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(1, self.GRID_WIDTH):
            px = x * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.HEIGHT))
        for y in range(1, self.GRID_HEIGHT):
            py = y * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.WIDTH, py))

        # Draw blocks
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color_idx = self.grid[x, y]
                if color_idx > 0:
                    px, py = x * self.BLOCK_SIZE, y * self.BLOCK_SIZE
                    color = self.BLOCK_COLORS[color_idx]
                    highlight_color = self.BLOCK_HIGHLIGHT_COLORS[color_idx]
                    
                    rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(self.screen, color, rect)
                    
                    inner_rect = pygame.Rect(px + 4, py + 4, self.BLOCK_SIZE - 8, self.BLOCK_SIZE - 8)
                    pygame.draw.rect(self.screen, highlight_color, inner_rect, border_radius=4)

        # Draw particles
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            size = int(alpha * 6)
            color = p['color']
            pygame.draw.rect(self.screen, color, (int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), size, size))

        # Draw cursor
        if not self.game_over:
            cx, cy = self.cursor_pos
            cpx, cpy = cx * self.BLOCK_SIZE, cy * self.BLOCK_SIZE
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            line_width = int(2 + pulse * 2)
            cursor_rect = pygame.Rect(cpx, cpy, self.BLOCK_SIZE, self.BLOCK_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=line_width, border_radius=4)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        mins, secs = divmod(self.timer, 60)
        timer_color = self.COLOR_TEXT if self.timer > 10 else self.COLOR_GAMEOVER
        timer_text = self.font_ui.render(f"Time: {int(mins):02}:{int(secs):02}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.win:
                msg_text = self.font_gameover.render("YOU WIN!", True, self.COLOR_WIN)
            else:
                msg_text = self.font_gameover.render("GAME OVER", True, self.COLOR_GAMEOVER)
            
            msg_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "blocks_remaining": int(np.sum(self.grid > 0))
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")