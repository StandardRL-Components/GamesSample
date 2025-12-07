
# Generated: 2025-08-28T04:39:55.565444
# Source Brief: brief_02375.md
# Brief Index: 2375

        
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
        "Controls: Arrow keys to move the cursor. Press space to select a block group. "
        "Clear the board before you run out of moves!"
    )

    game_description = (
        "Clear the grid by strategically matching adjacent colored blocks in this fast-paced puzzle game. "
        "Selecting a block group removes it, and blocks above collapse down. Plan ahead to create large chain reactions!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.W, self.H = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 12, 8
        self.BLOCK_SIZE = 40
        self.GRID_WIDTH = self.GRID_COLS * self.BLOCK_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.BLOCK_SIZE
        self.GRID_X_OFFSET = (self.W - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.H - self.GRID_HEIGHT) // 2 + 30

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()

        # --- Colors & Style ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 50, 62)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_CURSOR = (255, 255, 255)
        self.BLOCK_COLORS = [
            None,  # 0 is empty
            (231, 76, 60),   # Red
            (46, 204, 113),  # Green
            (52, 152, 219),  # Blue
            (241, 196, 15),   # Yellow
            (155, 89, 182),  # Purple
        ]
        self.BLOCK_HIGHLIGHT_COLORS = [
            None,
            (252, 134, 120),
            (118, 225, 162),
            (123, 189, 235),
            (247, 215, 96),
            (194, 149, 212),
        ]
        
        # --- Fonts ---
        try:
            self.FONT_UI = pygame.font.SysFont("Consolas", 24, bold=True)
            self.FONT_MSG = pygame.font.SysFont("Arial", 48, bold=True)
        except pygame.error:
            self.FONT_UI = pygame.font.Font(None, 28)
            self.FONT_MSG = pygame.font.Font(None, 52)


        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.victory = False
        self.last_space_held = False
        self.particles = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.moves_left = 40
        self.steps = 0
        self.game_over = False
        self.victory = False
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.last_space_held = False
        self.particles = []
        
        self._generate_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        self.steps += 1
        
        self._handle_movement(movement)
        
        space_pressed = space_held and not self.last_space_held
        if space_pressed:
            reward = self._handle_selection()
        
        self.last_space_held = space_held
        
        self._update_particles()
        
        board_cleared = np.all(self.grid == 0)
        out_of_moves = self.moves_left <= 0
        
        if board_cleared or out_of_moves:
            terminated = True
            self.game_over = True
            if board_cleared:
                self.victory = True
                reward += 100  # Victory bonus
            else:
                reward += -50  # Loss penalty
        
        if self.steps >= 1000 and not terminated:
            terminated = True
            self.game_over = True
            reward += -50 # Out of time penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_COLS
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_ROWS

    def _handle_selection(self):
        x, y = self.cursor_pos
        color_to_match = self.grid[x, y]

        if color_to_match == 0:
            self.moves_left -= 1
            return -0.2

        connected_blocks = self._find_connected_blocks(x, y)

        if len(connected_blocks) < 2:
            self.moves_left -= 1
            return -0.2
        
        # Match found
        self.moves_left -= 1
        num_cleared = len(connected_blocks)
        reward = float(num_cleared)
        if num_cleared > 5:
            reward += 5.0
        
        self.score += num_cleared * 10 * (num_cleared - 1) # Combo scoring

        for bx, by in connected_blocks:
            block_color_index = self.grid[bx, by]
            self.grid[bx, by] = 0
            # SFX: Block break
            self._create_particles(bx, by, self.BLOCK_COLORS[block_color_index])
        
        self._collapse_grid()
        return reward

    def _find_connected_blocks(self, start_x, start_y):
        color_to_match = self.grid[start_x, start_y]
        if color_to_match == 0:
            return []

        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        connected = []

        while q:
            x, y = q.popleft()
            connected.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    if (nx, ny) not in visited and self.grid[nx, ny] == color_to_match:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return connected

    def _collapse_grid(self):
        for x in range(self.GRID_COLS):
            write_idx = self.GRID_ROWS - 1
            for y in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[x, y] != 0:
                    if y != write_idx:
                        self.grid[x, write_idx] = self.grid[x, y]
                        self.grid[x, y] = 0
                    write_idx -= 1

    def _generate_grid(self):
        self.grid = self.np_random.integers(1, len(self.BLOCK_COLORS), size=(self.GRID_COLS, self.GRID_ROWS))

    def _create_particles(self, grid_x, grid_y, color):
        px = self.GRID_X_OFFSET + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        py = self.GRID_Y_OFFSET + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = random.uniform(2, 5)
            lifespan = random.uniform(0.5, 1.0)
            self.particles.append({'pos': [px, py], 'vel': vel, 'size': size, 'lifespan': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifespan'] -= 1/30.0 # Assuming 30fps for visual effect
            if p['lifespan'] > 0:
                active_particles.append(p)
        self.particles = active_particles
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT), 1)

        # Draw blocks
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_index = self.grid[c, r]
                if color_index != 0:
                    x = self.GRID_X_OFFSET + c * self.BLOCK_SIZE
                    y = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE
                    
                    main_color = self.BLOCK_COLORS[color_index]
                    highlight_color = self.BLOCK_HIGHLIGHT_COLORS[color_index]
                    
                    block_rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    highlight_rect = pygame.Rect(x + 4, y + 4, self.BLOCK_SIZE - 8, self.BLOCK_SIZE - 8)

                    pygame.draw.rect(self.screen, main_color, block_rect, border_radius=5)
                    pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_radius=3)

        # Draw particles
        for p in self.particles:
            life_ratio = max(0, p['lifespan'] / p['max_life'])
            alpha = int(255 * life_ratio)
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['size'] * life_ratio)
            if size > 0:
                # Using a temporary surface for alpha blending
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (pos[0]-size, pos[1]-size))

        # Draw cursor
        cursor_x = self.GRID_X_OFFSET + self.cursor_pos[0] * self.BLOCK_SIZE
        cursor_y = self.GRID_Y_OFFSET + self.cursor_pos[1] * self.BLOCK_SIZE
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

    def _render_ui(self):
        # Moves Left
        moves_text = self.FONT_UI.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 15))

        # Score
        score_text = self.FONT_UI.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.W - 20, 15))
        self.screen.blit(score_text, score_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg_text_str = "VICTORY!" if self.victory else "GAME OVER"
            msg_color = (100, 255, 100) if self.victory else (255, 100, 100)
            
            msg_surf = self.FONT_MSG.render(msg_text_str, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.W // 2, self.H // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Block Clear")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        movement = 0 # No-op
        space_held = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = True

        action = [movement, 1 if space_held else 0, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Limit frame rate for human play

    env.close()