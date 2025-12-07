
# Generated: 2025-08-27T23:50:24.223573
# Source Brief: brief_03590.md
# Brief Index: 3590

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a block. "
        "Select an adjacent, matching block to clear it and all connected blocks of the same color. "
        "Press Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the grid by matching adjacent colored blocks. Plan your moves to create large combos "
        "and clear the board before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 5, 5
        self.NUM_COLORS = 4
        self.MAX_MOVES = 30
        self.MAX_STEPS = 500

        # Visuals
        self.BLOCK_SIZE = 60
        self.GRID_LINE_WIDTH = 2
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2 + 20

        self.COLOR_BG = (44, 62, 80)
        self.COLOR_GRID_BG = (52, 73, 94)
        self.COLOR_CURSOR = (241, 196, 15)
        self.COLOR_TEXT = (236, 240, 241)
        self.BLOCK_COLORS = [
            (231, 76, 60),   # Red
            (46, 204, 113),  # Green
            (52, 152, 219),  # Blue
            (155, 89, 182),  # Purple
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        
        # State variables (will be reset)
        self.grid = None
        self.cursor_pos = None
        self.selected_block = None
        self.moves_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.np_random = None
        self.visual_cursor_pos = [0, 0]
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.visual_cursor_pos = [
            self.GRID_OFFSET_X + self.cursor_pos[0] * self.BLOCK_SIZE,
            self.GRID_OFFSET_Y + self.cursor_pos[1] * self.BLOCK_SIZE,
        ]
        self.selected_block = None
        self.particles = []

        self._generate_grid()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        # Generates a grid and ensures at least one move is possible
        while True:
            self.grid = self.np_random.integers(
                1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH)
            ).tolist()
            if self._has_possible_moves():
                break

    def _has_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color = self.grid[y][x]
                if x + 1 < self.GRID_WIDTH and self.grid[y][x + 1] == color:
                    return True
                if y + 1 < self.GRID_HEIGHT and self.grid[y + 1][x] == color:
                    return True
        return False
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # 1. Handle Deselection (Shift)
        if shift_press and self.selected_block is not None:
            self.selected_block = None
            # sound: "deselect_sound"

        # 2. Handle Selection / Match Attempt (Space)
        elif space_press:
            cursor_x, cursor_y = self.cursor_pos
            
            if self.selected_block is None:
                if self.grid[cursor_y][cursor_x] is not None:
                    self.selected_block = (cursor_x, cursor_y)
                    # sound: "select_sound"
            else:
                sel_x, sel_y = self.selected_block
                
                is_adjacent = abs(sel_x - cursor_x) + abs(sel_y - cursor_y) == 1
                color1 = self.grid[sel_y][sel_x]
                color2 = self.grid[cursor_y][cursor_x]
                is_match = (color1 is not None) and (color1 == color2)
                
                self.moves_remaining -= 1
                step_score = 0

                if is_adjacent and is_match:
                    # sound: "match_success_sound"
                    connected_blocks = self._find_connected_blocks((cursor_x, cursor_y), color2)
                    num_cleared = len(connected_blocks)
                    
                    step_score += num_cleared
                    reward += num_cleared
                    if num_cleared > 2:
                        step_score += 5
                        reward += 5
                    
                    for x, y in connected_blocks:
                        self._create_particles(x, y, self.grid[y][x])
                        self.grid[y][x] = None
                        
                    self._apply_gravity()
                else:
                    # sound: "match_fail_sound"
                    reward -= 0.1
                
                self.score += step_score
                self.selected_block = None
        
        # 3. Handle Movement
        dx, dy = 0, 0
        if movement == 1: dy = -1
        elif movement == 2: dy = 1
        elif movement == 3: dx = -1
        elif movement == 4: dx = 1
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_HEIGHT - 1)
        
        # Update game logic
        self.steps += 1
        terminated = False
        
        if self._is_board_clear():
            reward += 50
            self.score += 50
            terminated = True
            self.game_over = True
        elif self.moves_remaining <= 0:
            reward -= 50
            self.score -= 50
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _find_connected_blocks(self, start_pos, color_id):
        if color_id is None: return []
        q = deque([start_pos])
        visited = {start_pos}
        connected = []
        while q:
            x, y = q.popleft()
            connected.append((x, y))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[ny][nx] == color_id:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return connected

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] is not None:
                    if y != empty_row:
                        self.grid[empty_row][x] = self.grid[y][x]
                        self.grid[y][x] = None
                    empty_row -= 1

    def _is_board_clear(self):
        for row in self.grid:
            if any(cell is not None for cell in row):
                return False
        return True
    
    def _get_observation(self):
        self._update_particles()
        
        # Interpolate cursor for smooth movement
        target_x = self.GRID_OFFSET_X + self.cursor_pos[0] * self.BLOCK_SIZE
        target_y = self.GRID_OFFSET_Y + self.cursor_pos[1] * self.BLOCK_SIZE
        self.visual_cursor_pos[0] += (target_x - self.visual_cursor_pos[0]) * 0.6
        self.visual_cursor_pos[1] += (target_y - self.visual_cursor_pos[1]) * 0.6

        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_blocks()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_remaining": self.moves_remaining}

    def _render_grid_and_blocks(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.BLOCK_SIZE,
                    self.GRID_OFFSET_Y + y * self.BLOCK_SIZE,
                    self.BLOCK_SIZE, self.BLOCK_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID_BG, rect)
                
                color_id = self.grid[y][x]
                if color_id is not None:
                    color = self.BLOCK_COLORS[color_id - 1]
                    block_rect = rect.inflate(-self.GRID_LINE_WIDTH * 2, -self.GRID_LINE_WIDTH * 2)
                    pygame.draw.rect(self.screen, color, block_rect, border_radius=5)
                    
                    highlight = tuple(min(255, c + 30) for c in color)
                    shadow = tuple(max(0, c - 30) for c in color)
                    
                    pygame.draw.line(self.screen, highlight, block_rect.topleft, block_rect.topright, 2)
                    pygame.draw.line(self.screen, highlight, block_rect.topleft, block_rect.bottomleft, 2)
                    pygame.draw.line(self.screen, shadow, block_rect.bottomright, block_rect.topright, 2)
                    pygame.draw.line(self.screen, shadow, block_rect.bottomright, block_rect.bottomleft, 2)

                    if self.selected_block == (x, y):
                        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
                        alpha = 50 + pulse * 100
                        selection_surf = pygame.Surface(block_rect.size, pygame.SRCALPHA)
                        pygame.draw.rect(selection_surf, (255, 255, 255, alpha), selection_surf.get_rect(), border_radius=5)
                        self.screen.blit(selection_surf, block_rect.topleft)

    def _render_cursor(self):
        cursor_rect = pygame.Rect(
            int(self.visual_cursor_pos[0]), int(self.visual_cursor_pos[1]),
            self.BLOCK_SIZE, self.BLOCK_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=4, border_radius=8)

    def _render_ui(self):
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        moves_text = self.font_medium.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 15))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "You Win!" if self._is_board_clear() else "Game Over"
            color = (46, 204, 113) if self._is_board_clear() else (231, 76, 60)
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)))

    def _create_particles(self, grid_x, grid_y, color_id):
        if color_id is None: return
        color = self.BLOCK_COLORS[color_id - 1]
        px = self.GRID_OFFSET_X + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        py = self.GRID_OFFSET_Y + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = random.randint(20, 40)
            size = random.randint(3, 7)
            self.particles.append([px, py, vx, vy, life, color, size])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1

    def _render_particles(self):
        for px, py, vx, vy, life, color, size in self.particles:
            alpha = int(255 * (life / 40.0))
            alpha = max(0, min(255, alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(size * (life / 40.0)), (*color, alpha))
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    import os
    import time

    # --- Headless Test ---
    print("--- Running Headless Test ---")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = GameEnv()
    obs, info = env.reset()
    start_time = time.time()
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if (i + 1) % 20 == 0:
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print(f"Episode terminated at step {i+1}.")
            env.reset()
    end_time = time.time()
    print(f"Headless test for 100 steps finished in {end_time - start_time:.2f} seconds.")
    env.close()

    # --- Interactive Test ---
    print("\n--- Starting Interactive Mode (close window to exit) ---")
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Puzzle Gym Env")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0]
        
        event_happened = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                event_happened = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                elif event.key == pygame.K_q:
                     running = False
                
                # Map keys to actions for a single step
                if not terminated:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    
                    if event.key == pygame.K_SPACE: action[1] = 1
                    if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
        
        if event_happened and not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Convert observation to pygame surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()
    pygame.quit()