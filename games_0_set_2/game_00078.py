
# Generated: 2025-08-27T12:32:31.031156
# Source Brief: brief_00078.md
# Brief Index: 78

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to select a group of blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match colored blocks in a grid to reach a target score before running out of moves. Select a group of 3 or more adjacent, same-colored blocks to clear them and score points. Larger groups and chain reactions score more!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 10
        self.BLOCK_SIZE = 32
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) + 10
        self.NUM_COLORS = 5
        self.TARGET_SCORE = 1000
        self.MAX_MOVES = 50
        
        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_UI_TEXT = (220, 220, 230)
        self.COLOR_CURSOR = (255, 255, 255)
        self.BLOCK_COLORS = [
            (0, 0, 0),  # 0: Empty
            (255, 80, 80),   # 1: Red
            (80, 255, 80),   # 2: Green
            (80, 150, 255),  # 3: Blue
            (255, 255, 80),  # 4: Yellow
            (200, 80, 255),  # 5: Purple
        ]
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # State variables (will be properly initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.last_cleared_blocks = set()

        # Initialize state variables
        self.reset()
        
        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.steps = 0
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.last_space_held = False
        self.last_cleared_blocks = set()

        self._initialize_board()
        
        return self._get_observation(), self._get_info()

    def _initialize_board(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        self._ensure_valid_moves()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Clear last frame's visual effects
        self.last_cleared_blocks = set()
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # 1. Handle cursor movement
        if movement == 1: # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
        elif movement == 2: # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_HEIGHT
        elif movement == 3: # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
        elif movement == 4: # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_WIDTH

        # 2. Handle block selection on key press (rising edge)
        reward = 0
        space_pressed = space_held and not self.last_space_held
        if space_pressed:
            self.moves_left -= 1
            reward = self._process_match_at_cursor()
        self.last_space_held = space_held

        # 3. Check for termination
        terminated = self.score >= self.TARGET_SCORE or self.moves_left <= 0
        if terminated:
            self.game_over = True
            if self.score >= self.TARGET_SCORE:
                reward += 100 # Win bonus
            else:
                reward -= 50 # Loss penalty

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _process_match_at_cursor(self):
        y, x = self.cursor_pos
        if self.grid[y, x] == 0:
            # sfx: click_empty
            return 0

        # Find initial group
        initial_matches = self._find_matches(y, x)
        if len(initial_matches) < 3:
            # sfx: click_fail
            return 0
        
        # sfx: match_success
        total_reward = 0
        combo_multiplier = 1.0
        
        all_cleared_this_turn = set()
        current_matches = initial_matches

        # Cascade loop
        while len(current_matches) >= 3:
            all_cleared_this_turn.update(current_matches)
            
            # Calculate score and reward for this wave
            num_cleared = len(current_matches)
            self.score += int((num_cleared ** 1.5) * 10 * combo_multiplier)
            total_reward += num_cleared
            if num_cleared > 5:
                total_reward += 10
            
            # Clear blocks from grid
            for by, bx in current_matches:
                self.grid[by, bx] = 0
            
            # sfx: blocks_fall
            self._handle_gravity_and_refill()

            # Find new matches caused by the cascade
            new_match_sets = self._find_all_matches_on_board()
            current_matches = set().union(*new_match_sets) if new_match_sets else set()
            combo_multiplier += 0.5

        self.last_cleared_blocks = all_cleared_this_turn
        self._ensure_valid_moves()
        return total_reward

    def _find_matches(self, y_start, x_start):
        if self.grid[y_start, x_start] == 0:
            return set()
            
        target_color = self.grid[y_start, x_start]
        q = deque([(y_start, x_start)])
        matched = set([(y_start, x_start)])

        while q:
            y, x = q.popleft()
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.GRID_HEIGHT and 0 <= nx < self.GRID_WIDTH and
                        (ny, nx) not in matched and self.grid[ny, nx] == target_color):
                    matched.add((ny, nx))
                    q.append((ny, nx))
        return matched
    
    def _find_all_matches_on_board(self):
        all_match_sets = []
        visited = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r, c) not in visited:
                    matches = self._find_matches(r, c)
                    if len(matches) >= 3:
                        all_match_sets.append(matches)
                    visited.update(matches)
        return all_match_sets

    def _handle_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            write_y = self.GRID_HEIGHT - 1
            for read_y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[read_y, x] != 0:
                    if read_y != write_y:
                        self.grid[write_y, x] = self.grid[read_y, x]
                        self.grid[read_y, x] = 0
                    write_y -= 1
            for y in range(write_y, -1, -1):
                self.grid[y, x] = self.np_random.integers(1, self.NUM_COLORS + 1)
                
    def _check_for_valid_moves(self):
        visited = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r, c) in visited or self.grid[r,c] == 0:
                    continue
                matches = self._find_matches(r, c)
                if len(matches) >= 3:
                    return True
                visited.update(matches)
        return False

    def _ensure_valid_moves(self):
        while not self._check_for_valid_moves():
            # sfx: board_shuffle
            flat_grid = self.grid.flatten()
            self.np_random.shuffle(flat_grid)
            self.grid = flat_grid.reshape((self.GRID_HEIGHT, self.GRID_WIDTH))
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        
        # Draw flashing effect for recently cleared blocks
        for r, c in self.last_cleared_blocks:
            x = self.GRID_OFFSET_X + c * self.BLOCK_SIZE
            y = self.GRID_OFFSET_Y + r * self.BLOCK_SIZE
            flash_rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)
            pygame.draw.rect(self.screen, (255, 255, 255), flash_rect, border_radius=8)

        # Draw blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_index = self.grid[r, c]
                if color_index > 0:
                    x = self.GRID_OFFSET_X + c * self.BLOCK_SIZE
                    y = self.GRID_OFFSET_Y + r * self.BLOCK_SIZE
                    color = self.BLOCK_COLORS[color_index]
                    
                    # Draw block with a slight 3D effect
                    block_rect = pygame.Rect(x + 2, y + 2, self.BLOCK_SIZE - 4, self.BLOCK_SIZE - 4)
                    shadow_color = tuple(max(0, val - 40) for val in color)
                    highlight_color = tuple(min(255, val + 40) for val in color)
                    
                    pygame.draw.rect(self.screen, shadow_color, block_rect.move(2, 2), border_radius=6)
                    pygame.draw.rect(self.screen, color, block_rect, border_radius=6)
                    pygame.gfxdraw.arc(self.screen, block_rect.x + 6, block_rect.y + 6, 4, 120, 240, highlight_color)


        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_x = self.GRID_OFFSET_X + cursor_c * self.BLOCK_SIZE
        cursor_y = self.GRID_OFFSET_Y + cursor_r * self.BLOCK_SIZE
        
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        alpha = 100 + int(pulse * 155)
        
        cursor_surface = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, self.COLOR_CURSOR + (alpha,), (0, 0, self.BLOCK_SIZE, self.BLOCK_SIZE), 4, border_radius=8)
        self.screen.blit(cursor_surface, (cursor_x, cursor_y))

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Moves display
        moves_text = self.font_large.render(f"MOVES: {self.moves_left}", True, self.COLOR_UI_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WON!" if self.score >= self.TARGET_SCORE else "GAME OVER"
            end_text = self.font_large.render(message, True, (255, 255, 100))
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Block Matcher")
    
    running = True
    clock = pygame.time.Clock()
    
    # Game loop
    terminated = False
    while running:
        # Pygame event handling
        action = np.array([0, 0, 0]) # no-op, released, released
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Moves: {info['moves_left']}, Reward: {reward:.2f}")

            if terminated:
                print(f"Game Over! Final Score: {info['score']}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Control the speed of human play
        
    env.close()