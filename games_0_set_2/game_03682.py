
# Generated: 2025-08-28T00:05:52.649009
# Source Brief: brief_03682.md
# Brief Index: 3682

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem. "
        "Select an adjacent gem to swap. Match 3 or more to score."
    )

    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of three or more. "
        "Create chain reactions for bonus points and reach the target score before you run out of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.NUM_GEM_TYPES = 6
        self.TARGET_SCORE = 5000
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_SELECT = (255, 255, 255, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.GEM_COLORS = [
            (255, 50, 50),   # Red
            (50, 255, 50),   # Green
            (50, 150, 255),  # Blue
            (255, 255, 50),  # Yellow
            (255, 50, 255),  # Magenta
            (50, 255, 255),  # Cyan
        ]

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # --- Grid and Gem Sizing ---
        self.GEM_SIZE = 40
        self.GRID_LINE_WIDTH = 2
        total_grid_w = self.GRID_COLS * self.GEM_SIZE
        total_grid_h = self.GRID_ROWS * self.GEM_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - total_grid_w) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - total_grid_h) // 2 + 20

        # --- Game State Initialization ---
        self.grid = None
        self.cursor_pos = None
        self.selected_gem = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.steps = None
        self.particles = []
        self.prev_space_held = False
        self.rng = None

        self.reset()
        
        # self.validate_implementation() # Run self-check

    def _create_valid_board(self):
        while True:
            board = self.rng.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))
            if not self._find_matches_on_board(board) and self._has_valid_moves(board):
                return board

    def _find_matches_on_board(self, board):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Check horizontal
                if c < self.GRID_COLS - 2 and board[r, c] == board[r, c+1] == board[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Check vertical
                if r < self.GRID_ROWS - 2 and board[r, c] == board[r+1, c] == board[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _has_valid_moves(self, board):
        temp_board = np.copy(board)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Check swap right
                if c < self.GRID_COLS - 1:
                    temp_board[r, c], temp_board[r, c+1] = temp_board[r, c+1], temp_board[r, c]
                    if self._find_matches_on_board(temp_board):
                        return True
                    temp_board[r, c], temp_board[r, c+1] = temp_board[r, c+1], temp_board[r, c] # Swap back
                # Check swap down
                if r < self.GRID_ROWS - 1:
                    temp_board[r, c], temp_board[r+1, c] = temp_board[r+1, c], temp_board[r, c]
                    if self._find_matches_on_board(temp_board):
                        return True
                    temp_board[r, c], temp_board[r+1, c] = temp_board[r+1, c], temp_board[r, c] # Swap back
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.grid = self._create_valid_board()
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_gem = None
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.steps = 0
        self.particles = []
        self.prev_space_held = True # Prevent action on first frame

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        reward = 0
        terminated = self.game_over

        if not terminated:
            # --- Handle Input ---
            if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 2: self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
            elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 4: self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)
            
            if space_pressed:
                r, c = self.cursor_pos
                if self.selected_gem is None:
                    # Select first gem
                    self.selected_gem = (r, c)
                else:
                    # Attempt to swap with second gem
                    r1, c1 = self.selected_gem
                    r2, c2 = r, c
                    
                    # Deselect if same gem is clicked
                    if (r1, c1) == (r2, c2):
                        self.selected_gem = None
                    # Check for adjacency
                    elif abs(r1 - r2) + abs(c1 - c2) == 1:
                        self.moves_left -= 1
                        reward += self._attempt_swap(r1, c1, r2, c2)
                        self.selected_gem = None
                    else:
                        # Select new gem if not adjacent
                        self.selected_gem = (r2, c2)

        # --- Update game state ---
        self._update_particles()
        self.steps += 1

        if not self.game_over:
            if self.moves_left <= 0:
                self.game_over = True
                reward -= 10 # Lose penalty
            elif self.score >= self.TARGET_SCORE:
                self.game_over = True
                reward += 100 # Win reward
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _attempt_swap(self, r1, c1, r2, c2):
        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        matches = self._find_matches_on_board(self.grid)
        if not matches:
            # Invalid swap, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            return -0.1 # Penalty for invalid move

        # Valid swap, process matches and cascades
        total_reward = 0
        is_cascade = False
        while matches:
            # Add rewards
            if is_cascade:
                total_reward += 5 # Cascade bonus
            total_reward += len(matches) # Reward per gem

            # Update score
            self.score += len(matches) * 10 * (1.5 if is_cascade else 1)

            # Create particle effects for matched gems
            for r_match, c_match in matches:
                gem_type = self.grid[r_match, c_match]
                if gem_type != -1: # Avoid re-processing
                    self._create_explosion(r_match, c_match, self.GEM_COLORS[gem_type])
            
            # Remove matched gems (set to -1)
            for r_match, c_match in matches:
                self.grid[r_match, c_match] = -1

            # Gravity: make gems fall
            self._apply_gravity()

            # Fill top rows with new gems
            self._refill_gems()

            # Check for new matches
            matches = self._find_matches_on_board(self.grid)
            is_cascade = True
        
        # Ensure the board is still playable
        if not self._has_valid_moves(self.grid):
            self.grid = self._create_valid_board() # Reshuffle if stuck

        return total_reward

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1

    def _refill_gems(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.rng.integers(0, self.NUM_GEM_TYPES)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_gems()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_COLS * self.GEM_SIZE, y), self.GRID_LINE_WIDTH)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_ROWS * self.GEM_SIZE), self.GRID_LINE_WIDTH)

    def _render_gems(self):
        gem_rect_inner = pygame.Rect(0, 0, self.GEM_SIZE - 6, self.GEM_SIZE - 6)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    color = self.GEM_COLORS[gem_type]
                    center_x = self.GRID_OFFSET_X + c * self.GEM_SIZE + self.GEM_SIZE // 2
                    center_y = self.GRID_OFFSET_Y + r * self.GEM_SIZE + self.GEM_SIZE // 2
                    
                    gem_rect_inner.center = (center_x, center_y)
                    pygame.draw.rect(self.screen, color, gem_rect_inner, border_radius=8)
                    
                    # Highlight effect
                    highlight_color = tuple(min(255, val + 60) for val in color)
                    pygame.gfxdraw.arc(self.screen, center_x, center_y, self.GEM_SIZE // 3, 120, 210, highlight_color)
                    pygame.gfxdraw.arc(self.screen, center_x, center_y, self.GEM_SIZE // 3 -1, 120, 210, highlight_color)


    def _render_cursor(self):
        r, c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + c * self.GEM_SIZE,
            self.GRID_OFFSET_Y + r * self.GEM_SIZE,
            self.GEM_SIZE, self.GEM_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)

        if self.selected_gem:
            r_sel, c_sel = self.selected_gem
            select_rect = pygame.Rect(
                self.GRID_OFFSET_X + c_sel * self.GEM_SIZE,
                self.GRID_OFFSET_Y + r_sel * self.GEM_SIZE,
                self.GEM_SIZE, self.GEM_SIZE
            )
            s = pygame.Surface((self.GEM_SIZE, self.GEM_SIZE), pygame.SRCALPHA)
            s.fill(self.COLOR_SELECT)
            self.screen.blit(s, select_rect.topleft)

    def _create_explosion(self, r, c, color):
        # Sound placeholder: # pygame.mixer.Sound('match.wav').play()
        center_x = self.GRID_OFFSET_X + c * self.GEM_SIZE + self.GEM_SIZE // 2
        center_y = self.GRID_OFFSET_Y + r * self.GEM_SIZE + self.GEM_SIZE // 2
        for _ in range(20):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.rng.uniform(2, 5)
            life = self.rng.integers(15, 30)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': vel,
                'radius': radius,
                'color': color,
                'life': life
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['radius'] -= 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]
    
    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, p['radius']))

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            result_text_str = "You Win!" if self.score >= self.TARGET_SCORE else "Game Over"
            result_text = self.font_large.render(result_text_str, True, self.COLOR_CURSOR)
            text_rect = result_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(result_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
            "selected_gem": self.selected_gem,
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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        # --- Human Input to Action Mapping ---
        movement = 0 # none
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        
        # Note: Since auto_advance is False, we only step when there's an input or a key is held.
        # For a better human experience, we run the loop at 10 FPS to catch key presses.
        action = np.array([movement, space, 0])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Control human play speed

    env.close()
    pygame.quit()