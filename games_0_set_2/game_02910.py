
# Generated: 2025-08-27T21:48:19.385496
# Source Brief: brief_02910.md
# Brief Index: 2910

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to clear a group of matching blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the board by matching groups of same-colored blocks. Plan your moves to create large combos and maximize your score before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 14
    GRID_HEIGHT = 10
    BLOCK_SIZE = 32
    GRID_LINE_WIDTH = 2
    NUM_COLORS = 5
    MIN_MATCH_SIZE = 2
    
    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_TEXT = (220, 230, 240)
    BLOCK_COLORS = [
        (0, 0, 0),  # 0: Empty
        (231, 76, 60),   # 1: Red
        (46, 204, 113),  # 2: Green
        (52, 152, 219),  # 3: Blue
        (241, 196, 15),  # 4: Yellow
        (155, 89, 182),  # 5: Purple
    ]
    BLOCK_SHADOW_COLORS = [
        (0, 0, 0),
        (192, 57, 43),
        (39, 174, 96),
        (41, 128, 185),
        (243, 156, 18),
        (142, 68, 173),
    ]

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.grid_render_offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.grid_render_offset_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2
        
        self.grid = None
        self.cursor_pos = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = []

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = 50
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.particles = []

        self._generate_board()
        self._ensure_valid_moves()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        space_pressed = space_pressed == 1
        reward = 0
        
        self.particles.clear() # Particles only last one frame

        # 1. Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        # 2. Handle activation
        if space_pressed:
            self.moves_left -= 1
            # sfx: click_sound
            
            x, y = self.cursor_pos
            if self.grid[y, x] > 0:
                matches = self._find_matches(x, y)
                
                if len(matches) >= self.MIN_MATCH_SIZE:
                    # sfx: match_clear_sound
                    num_cleared = len(matches)
                    reward += num_cleared  # +1 for each block
                    if num_cleared > 5:
                        reward += 10 # Bonus for large clears
                    
                    self.score += reward

                    for pos_x, pos_y in matches:
                        self._create_particles(pos_x, pos_y, self.grid[pos_y, pos_x])
                        self.grid[pos_y, pos_x] = 0 # Clear block
                    
                    self._apply_gravity_and_refill()
                    
                    if not self._check_for_any_valid_moves():
                        self._ensure_valid_moves()
                        # sfx: board_shuffle_sound
        
        self.steps += 1
        
        # 3. Check for termination conditions
        terminated = False
        if self._is_board_clear():
            reward += 100 # Win bonus
            self.score += 100
            terminated = True
            self.game_over = True
        elif self.moves_left <= 0:
            terminated = True
            self.game_over = True

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
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
        }

    def _generate_board(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))

    def _ensure_valid_moves(self):
        while not self._check_for_any_valid_moves():
            self._generate_board()

    def _check_for_any_valid_moves(self):
        visited = np.zeros_like(self.grid, dtype=bool)
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if not visited[r, c] and self.grid[r, c] > 0:
                    matches = self._find_matches(c, r, visited_mask=visited)
                    if len(matches) >= self.MIN_MATCH_SIZE:
                        return True
        return False

    def _is_board_clear(self):
        return np.all(self.grid == 0)

    def _find_matches(self, start_x, start_y, visited_mask=None):
        if self.grid[start_y, start_x] == 0:
            return []

        target_color = self.grid[start_y, start_x]
        q = deque([(start_x, start_y)])
        matches = set([(start_x, start_y)])
        
        if visited_mask is not None:
            visited_mask[start_y, start_x] = True

        while q:
            x, y = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in matches and self.grid[ny, nx] == target_color:
                        matches.add((nx, ny))
                        q.append((nx, ny))
                        if visited_mask is not None:
                            visited_mask[ny, nx] = True
        return list(matches)

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1
            
            # Refill from top
            for r in range(empty_row, -1, -1):
                self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)
    
    def _create_particles(self, block_x, block_y, color_index):
        center_x = self.grid_render_offset_x + block_x * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        center_y = self.grid_render_offset_y + block_y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        color = self.BLOCK_COLORS[color_index]
        for _ in range(10): # 10 particles per block
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.uniform(2, 5)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'size': size, 'color': color})

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_render_offset_x, self.grid_render_offset_y,
                                self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_index = self.grid[r, c]
                if color_index > 0:
                    block_rect = pygame.Rect(
                        self.grid_render_offset_x + c * self.BLOCK_SIZE,
                        self.grid_render_offset_y + r * self.BLOCK_SIZE,
                        self.BLOCK_SIZE, self.BLOCK_SIZE
                    )
                    # Draw shadow/base
                    pygame.draw.rect(self.screen, self.BLOCK_SHADOW_COLORS[color_index], block_rect, border_radius=5)
                    # Draw main color
                    inner_rect = block_rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, self.BLOCK_COLORS[color_index], inner_rect, border_radius=4)

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_render_offset_x + cursor_x * self.BLOCK_SIZE,
            self.grid_render_offset_y + cursor_y * self.BLOCK_SIZE,
            self.BLOCK_SIZE, self.BLOCK_SIZE
        )
        pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, width=3, border_radius=6)

        # Draw particles
        for p in self.particles:
            # In a turn-based game, we just displace them for one frame
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            pygame.draw.circle(self.screen, p['color'], p['pos'], p['size'])

    def _render_ui(self):
        # Render Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 10))
        
        # Render Moves Left
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            result_text_str = "YOU WIN!" if self._is_board_clear() else "GAME OVER"
            result_text = self.font_large.render(result_text_str, True, (255, 255, 255))
            text_rect = result_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(result_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    env = GameEnv(render_mode="rgb_array")
    
    # Run validation
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    # We need a separate pygame window to display the rendering
    pygame.display.set_caption("Block Breaker Gym Environment")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_pressed = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_pressed = 1
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        # Only step if an action is taken in this turn-based game
        if movement != 0 or space_pressed != 0:
            action = [movement, space_pressed, 0] # Shift is not used
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if reward > 0:
                print(f"Reward: {reward}")
            if done:
                print(f"Game Over! Final Score: {info['score']}")
                
        # --- Rendering ---
        # Convert the observation (H, W, C) to a pygame surface (W, H)
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play

    env.close()