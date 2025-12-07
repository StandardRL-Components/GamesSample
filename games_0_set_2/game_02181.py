
# Generated: 2025-08-28T03:57:49.340540
# Source Brief: brief_02181.md
# Brief Index: 2181

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import Counter
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        self.color = color
        self.np_random = np_random
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = self.np_random.integers(10, 20)  # Frames
        self.radius = self.np_random.integers(3, 6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        self.radius -= 0.2

    def draw(self, surface):
        if self.lifetime > 0 and self.radius > 0:
            pygame.gfxdraw.filled_circle(
                surface, int(self.x), int(self.y), int(self.radius), self.color
            )
            pygame.gfxdraw.aacircle(
                surface, int(self.x), int(self.y), int(self.radius), self.color
            )

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Use arrow keys to move the cursor. Press Space to select and merge a group of colored squares. The game ends when the timer runs out or the board is a single color."
    game_description = "A strategic puzzle game where you merge adjacent same-colored squares. Clear the board by creating a single color block before the 60-second timer expires. Grey blocks are obstacles."
    
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 10
    CELL_SIZE = 32
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2 + 20
    MAX_STEPS = 1800  # 60 seconds at 30fps
    MAX_TIMER = 60.0

    # --- Colors ---
    COLOR_BG = (25, 25, 35)
    COLOR_GRID_LINES = (50, 50, 60)
    PLAYABLE_COLORS = [
        (255, 87, 87),    # Red
        (87, 255, 87),    # Green
        (87, 87, 255),    # Blue
        (255, 255, 87),   # Yellow
        (255, 87, 255),   # Magenta
        (87, 255, 255),   # Cyan
    ]
    COLOR_GREY = (128, 128, 128) # Obstacle
    COLOR_WHITE = (240, 240, 240)
    
    # Color indices
    COLOR_INDEX_GREY = len(PLAYABLE_COLORS)
    COLOR_INDEX_EMPTY = -1

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
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        self.grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.win = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []

        self.reset()
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.MAX_TIMER
        self.game_over = False
        self.win = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        # Initialize grid
        self.grid = self.np_random.integers(
            0, len(self.PLAYABLE_COLORS), size=(self.GRID_HEIGHT, self.GRID_WIDTH)
        )
        
        # Add obstacles
        num_obstacles = int(self.GRID_WIDTH * self.GRID_HEIGHT * 0.1)
        obstacle_indices = self.np_random.choice(
            self.GRID_WIDTH * self.GRID_HEIGHT, num_obstacles, replace=False
        )
        for idx in obstacle_indices:
            row, col = divmod(idx, self.GRID_WIDTH)
            self.grid[row][col] = self.COLOR_INDEX_GREY
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            # If the game is over, do nothing but return the final state
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        reward = -0.01  # Small penalty for each step (time passing)
        self.steps += 1
        self.timer -= 1 / 30.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Handle "click" action (on press, not hold)
        if space_held and not self.prev_space_held:
            # sfx: click
            reward += self._handle_click()
        
        # Handle "restart" action (on press, not hold)
        if shift_held and not self.prev_shift_held:
            # This is an agent-forced termination, penalize heavily
            self.game_over = True
            reward -= 100
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        self.score += reward
        
        # Update particles
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += terminal_reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_click(self):
        cx, cy = self.cursor_pos
        target_color_idx = self.grid[cy][cx]

        if target_color_idx == self.COLOR_INDEX_GREY:
            # sfx: buzz_error
            return -1.0 # Penalty for clicking an obstacle

        # Flood fill to find connected squares
        q = [(cx, cy)]
        visited = set(q)
        connected = []
        while q:
            x, y = q.pop(0)
            connected.append((x, y))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and \
                   (nx, ny) not in visited and self.grid[ny][nx] == target_color_idx:
                    visited.add((nx, ny))
                    q.append((nx, ny))

        if len(connected) <= 1:
            # sfx: buzz_error
            return -1.0 # Penalty for clicking an isolated square

        # sfx: merge_pop
        # Calculate reward for successful merge
        reward = len(connected)  # +1 for each square
        if len(connected) >= 4:
            reward += 5  # Bonus for large merge

        # Remove merged squares and spawn particles
        merge_color = self.PLAYABLE_COLORS[target_color_idx]
        for x, y in connected:
            self.grid[y][x] = self.COLOR_INDEX_EMPTY
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE / 2
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE / 2
            for _ in range(3):
                self.particles.append(Particle(px, py, merge_color, self.np_random))

        # Apply gravity
        for col in range(self.GRID_WIDTH):
            write_row = self.GRID_HEIGHT - 1
            for read_row in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[read_row][col] != self.COLOR_INDEX_EMPTY:
                    if read_row != write_row:
                        self.grid[write_row][col] = self.grid[read_row][col]
                        self.grid[read_row][col] = self.COLOR_INDEX_EMPTY
                    write_row -= 1
        
        # Refill empty top cells with new random colors
        for row in range(self.GRID_HEIGHT):
            for col in range(self.GRID_WIDTH):
                if self.grid[row][col] == self.COLOR_INDEX_EMPTY:
                    self.grid[row][col] = self.np_random.integers(0, len(self.PLAYABLE_COLORS))

        return reward

    def _check_termination(self):
        if self.game_over: # Already terminated by shift or other means
            return True, 0

        # Check for win condition
        first_color = -1
        is_win = True
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.grid[r][c]
                if color_idx != self.COLOR_INDEX_GREY:
                    if first_color == -1:
                        first_color = color_idx
                    elif color_idx != first_color:
                        is_win = False
                        break
            if not is_win:
                break
        
        if is_win:
            # sfx: win_jingle
            self.game_over = True
            self.win = True
            return True, 100.0

        # Check for loss conditions
        if self.timer <= 0 or self.steps >= self.MAX_STEPS:
            # sfx: lose_sound
            self.game_over = True
            return True, -100.0
            
        return False, 0

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
            "timer": self.timer,
        }

    def _render_game(self):
        # Draw grid and cells
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                
                color_idx = self.grid[y][x]
                if color_idx == self.COLOR_INDEX_GREY:
                    color = self.COLOR_GREY
                else:
                    color = self.PLAYABLE_COLORS[color_idx]
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect, 1)

        # Draw cursor
        cursor_x = self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE
        cursor_y = self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.CELL_SIZE, self.CELL_SIZE)
        
        # Pulsing glow effect for cursor
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        glow_size = int(pulse * 3)
        glow_alpha = int(pulse * 100 + 50)
        
        glow_color = (*self.COLOR_WHITE[:3], glow_alpha)
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=4, width=glow_size+2)
        self.screen.blit(s, cursor_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, cursor_rect, 2, border_radius=4)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Timer display
        timer_text = f"TIME: {max(0, int(self.timer)):02}"
        timer_surf = self.font_medium.render(timer_text, True, self.COLOR_WHITE)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 20, 10))

        # Score display
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_WHITE)
        self.screen.blit(score_surf, (20, 10))

        # Color counts display
        counts = Counter(c for row in self.grid for c in row if c != self.COLOR_INDEX_GREY)
        start_x = self.GRID_OFFSET_X
        y_pos = self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE + 10
        
        for i, color in enumerate(self.PLAYABLE_COLORS):
            count = counts.get(i, 0)
            rect = pygame.Rect(start_x, y_pos, 16, 16)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            
            count_text = f"{count}"
            count_surf = self.font_small.render(count_text, True, self.COLOR_WHITE)
            self.screen.blit(count_surf, (start_x + 20, y_pos))
            start_x += 55

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.PLAYABLE_COLORS[1] if self.win else self.PLAYABLE_COLORS[0]
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
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
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Color Merge")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Action Mapping for Human Play ---
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()