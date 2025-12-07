
# Generated: 2025-08-28T03:40:31.057241
# Source Brief: brief_05002.md
# Brief Index: 5002

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to clear a selected group of blocks."
    )

    game_description = (
        "Clear the grid of colored blocks by clicking adjacent groups of the same color before the time runs out."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_ROWS = 8
    GRID_COLS = 8
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    GAME_DURATION_SECONDS = 60

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (30, 40, 50)
    COLOR_GRID_LINES = (50, 60, 70)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    BLOCK_COLORS = [
        (0, 0, 0),  # 0: Empty
        (231, 76, 60),   # 1: Red
        (52, 152, 219),  # 2: Blue
        (46, 204, 113),  # 3: Green
        (241, 196, 15),   # 4: Yellow
        (155, 89, 182),  # 5: Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 60, bold=True)
        self.font_floating = pygame.font.SysFont("Consolas", 18, bold=True)

        # --- Grid & Block Sizing ---
        self.grid_area_width = 320
        self.grid_area_height = 320
        self.grid_top_left_x = (self.SCREEN_WIDTH - self.grid_area_width) // 2
        self.grid_top_left_y = (self.SCREEN_HEIGHT - self.grid_area_height) // 2
        self.block_size = self.grid_area_width // self.GRID_COLS

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_status = None
        self.previous_space_held = None
        self.particles = None
        self.floating_texts = None
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Optional: Call to self-check during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = None
        self.previous_space_held = False
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self._generate_board()

        self.particles = []
        self.floating_texts = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = 0
        
        if not self.game_over:
            self.steps += 1
            
            # Unpack actions
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

            # --- Handle Input ---
            self._handle_movement(movement)
            
            # Detect space press (rising edge)
            if space_held and not self.previous_space_held:
                reward += self._handle_click()
                # sfx: click_block.wav
            
            self.previous_space_held = space_held

            # --- Update Game Systems ---
            self._update_particles()
            self._update_floating_texts()

        # --- Check Termination ---
        terminated = False
        is_board_clear = self._is_board_clear()
        no_moves_left = not self._has_valid_moves() and not is_board_clear
        
        if is_board_clear:
            if not self.game_over: # First frame of win
                reward += 100
                self.win_status = "YOU WIN!"
                # sfx: win_fanfare.wav
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS or no_moves_left:
            if not self.game_over: # First frame of loss
                reward -= 100
                self.win_status = "GAME OVER" if no_moves_left else "TIME'S UP!"
                # sfx: lose_buzzer.wav
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] -= 1
        elif movement == 2: # Down
            self.cursor_pos[1] += 1
        elif movement == 3: # Left
            self.cursor_pos[0] -= 1
        elif movement == 4: # Right
            self.cursor_pos[0] += 1
        
        # Wrap cursor around edges
        self.cursor_pos[0] %= self.GRID_COLS
        self.cursor_pos[1] %= self.GRID_ROWS

    def _handle_click(self):
        col, row = self.cursor_pos
        color_index = self.grid[row][col]
        
        if color_index == 0: # Clicked an empty space
            # sfx: click_empty.wav
            return 0

        cluster = self._find_cluster(col, row)
        
        if len(cluster) < 2: # Not a valid cluster
            # sfx: click_invalid.wav
            return 0
        
        # Valid click, process it
        # sfx: clear_cluster.wav
        num_cleared = len(cluster)
        reward = num_cleared  # +1 per block cleared
        self.score += num_cleared

        for r, c in cluster:
            self.grid[r][c] = 0

        self._apply_gravity()
        
        # Visual feedback
        self._spawn_particles(col, row, self.BLOCK_COLORS[color_index], num_cleared)
        self._spawn_floating_text(f"+{num_cleared}", col, row)
        
        if self._is_board_clear():
            reward += 10 # Bonus for clearing board

        return reward

    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(1, len(self.BLOCK_COLORS), size=(self.GRID_ROWS, self.GRID_COLS), dtype=int).tolist()
            if self._has_valid_moves():
                break

    def _has_valid_moves(self):
        visited = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] != 0 and (r, c) not in visited:
                    cluster = self._find_cluster(c, r)
                    if len(cluster) >= 2:
                        return True
                    visited.update(cluster)
        return False

    def _find_cluster(self, start_col, start_row):
        target_color = self.grid[start_row][start_col]
        if target_color == 0:
            return []

        q = deque([(start_row, start_col)])
        cluster = set([(start_row, start_col)])
        
        while q:
            r, c = q.popleft()
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                    if (nr, nc) not in cluster and self.grid[nr][nc] == target_color:
                        cluster.add((nr, nc))
                        q.append((nr, nc))
        return list(cluster)

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] != 0:
                    self.grid[empty_row][c], self.grid[r][c] = self.grid[r][c], self.grid[empty_row][c]
                    empty_row -= 1

    def _is_board_clear(self):
        return all(self.grid[r][c] == 0 for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_top_left_x, self.grid_top_left_y, self.grid_area_width, self.grid_area_height)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)
        
        # Draw blocks and grid lines
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_index = self.grid[r][c]
                block_rect = pygame.Rect(
                    self.grid_top_left_x + c * self.block_size,
                    self.grid_top_left_y + r * self.block_size,
                    self.block_size,
                    self.block_size
                )
                if color_index != 0:
                    color = self.BLOCK_COLORS[color_index]
                    # Draw block with a subtle 3D effect
                    pygame.draw.rect(self.screen, color, block_rect)
                    pygame.draw.rect(self.screen, tuple(min(255, x + 30) for x in color), block_rect.inflate(-6, -6), 0)
                
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, block_rect, 1)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))
        
        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_top_left_x + cursor_x * self.block_size,
            self.grid_top_left_y + cursor_y * self.block_size,
            self.block_size,
            self.block_size
        )
        # Pulsing effect for cursor
        alpha = 128 + 64 * math.sin(self.steps * 0.2)
        cursor_surface = pygame.Surface((self.block_size, self.block_size), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), (0, 0, self.block_size, self.block_size), 4)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_ui(self):
        # --- Score Display ---
        self._draw_text(f"Score: {self.score}", self.font_main, (20, 20))
        
        # --- Timer Display ---
        time_left = max(0, self.GAME_DURATION_SECONDS - (self.steps / self.MAX_STEPS * self.GAME_DURATION_SECONDS))
        timer_text = f"Time: {time_left:.1f}"
        text_width = self.font_main.size(timer_text)[0]
        self._draw_text(timer_text, self.font_main, (self.SCREEN_WIDTH - text_width - 20, 20))

        # --- Floating Text ---
        for ft in self.floating_texts:
            alpha = ft['life'] / 30.0 * 255
            text_surf = self.font_floating.render(ft['text'], True, (*self.COLOR_TEXT, alpha))
            text_surf.set_colorkey((0,0,0))
            self.screen.blit(text_surf, (int(ft['x']), int(ft['y'])))

        # --- Game Over/Win Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            self._draw_text(self.win_status, self.font_large, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), center=True)

    def _draw_text(self, text, font, pos, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surface = font.render(text, True, color)
        shadow_surface = font.render(text, True, shadow_color)
        x, y = pos
        if center:
            text_rect = text_surface.get_rect(center=(x, y))
        else:
            text_rect = text_surface.get_rect(topleft=(x, y))

        self.screen.blit(shadow_surface, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _spawn_particles(self, grid_col, grid_row, color, count):
        center_x = self.grid_top_left_x + (grid_col + 0.5) * self.block_size
        center_y = self.grid_top_left_y + (grid_row + 0.5) * self.block_size

        for _ in range(count * 3):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            self.particles.append({
                'x': center_x,
                'y': center_y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(20, 40),
                'size': self.np_random.uniform(2, 5),
                'color': color,
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
            p['size'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]

    def _spawn_floating_text(self, text, grid_col, grid_row):
        start_x = self.grid_top_left_x + (grid_col + 0.5) * self.block_size - self.font_floating.size(text)[0] / 2
        start_y = self.grid_top_left_y + (grid_row + 0.5) * self.block_size
        self.floating_texts.append({
            'text': text,
            'x': start_x,
            'y': start_y,
            'vy': -1.5,
            'life': 30
        })

    def _update_floating_texts(self):
        for ft in self.floating_texts:
            ft['y'] += ft['vy']
            ft['life'] -= 1
        self.floating_texts = [ft for ft in self.floating_texts if ft['life'] > 0]

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

# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- For interactive play ---
    import pygame
    
    pygame.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Clearer")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    
    running = True
    while running:
        movement = 0 # No-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1

        action = [movement, space, 0] # shift is unused
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the observation from the environment to the display screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Match the environment's internal clock
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause before resetting
            obs, info = env.reset()
            terminated = False

    env.close()
    pygame.quit()