
# Generated: 2025-08-27T19:17:27.740051
# Source Brief: brief_02102.md
# Brief Index: 2102

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem. "
        "Move to an adjacent gem and press Space again to swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match gems in a grid to clear them and achieve the highest score within a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Game constants
        self.GRID_DIM = 10
        self.NUM_GEM_TYPES = 6
        self.CELL_SIZE = 36
        self.GRID_WIDTH = self.GRID_DIM * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_DIM * self.CELL_SIZE
        self.GRID_X = (640 - self.GRID_WIDTH) // 2
        self.GRID_Y = (400 - self.GRID_HEIGHT) // 2
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000

        # Visuals
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_LINES = (40, 50, 60)
        self.COLOR_TEXT = (220, 220, 230)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 160, 80),  # Orange
        ]
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 48, bold=True)

        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.score = None
        self.moves_remaining = None
        self.game_over = None
        self.steps = None
        self.animation_effects = []
        
        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.steps = 0
        self.cursor_pos = (self.GRID_DIM // 2, self.GRID_DIM // 2)
        self.selected_gem_pos = None
        self.animation_effects.clear()
        
        self._generate_valid_grid()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.animation_effects.clear()
        reward = 0
        terminated = False

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        if shift_pressed:
            self.selected_gem_pos = None
        
        if movement != 0:
            self._move_cursor(movement)

        if space_pressed:
            reward += self._handle_selection()
        
        if self.moves_remaining <= 0:
            reward += -50  # Penalty for running out of moves
            terminated = True
        
        if self._is_grid_clear():
            reward += 100 # Bonus for clearing the grid
            terminated = True
            
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        if not self._has_possible_moves(self.grid):
             self._shuffle_board()

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, movement):
        r, c = self.cursor_pos
        if movement == 1: r -= 1  # Up
        elif movement == 2: r += 1  # Down
        elif movement == 3: c -= 1  # Left
        elif movement == 4: c += 1  # Right
        
        if 0 <= r < self.GRID_DIM and 0 <= c < self.GRID_DIM:
            self.cursor_pos = (r, c)

    def _handle_selection(self):
        if self.selected_gem_pos is None:
            self.selected_gem_pos = self.cursor_pos
            return 0
        
        # If selecting the same gem again, deselect
        if self.selected_gem_pos == self.cursor_pos:
            self.selected_gem_pos = None
            return 0
        
        # Check for adjacency for a swap
        (r1, c1), (r2, c2) = self.selected_gem_pos, self.cursor_pos
        if abs(r1 - r2) + abs(c1 - c2) == 1:
            return self._perform_swap_and_cascade(self.selected_gem_pos, self.cursor_pos)
        else: # Not adjacent, so select the new gem instead
            self.selected_gem_pos = self.cursor_pos
            return 0

    def _perform_swap_and_cascade(self, pos1, pos2):
        self.moves_remaining -= 1
        self._swap_gems(pos1, pos2)
        
        matches = self._find_all_matches(self.grid)
        if not matches:
            # Invalid swap, swap back
            self._swap_gems(pos1, pos2) # No move cost, but a small penalty
            self.selected_gem_pos = None
            return -0.1

        # Successful swap, handle cascades
        total_reward = 0
        while matches:
            reward_for_match = self._calculate_match_reward(matches)
            total_reward += reward_for_match
            
            # Add particle effects for matched gems
            for r, c in matches:
                # sound: gem_destroy.wav
                self.animation_effects.append(('particle_burst', (r, c), self.GEM_COLORS[self.grid[r, c] - 1]))

            self._clear_gems(matches)
            self._apply_gravity()
            self._fill_new_gems()
            matches = self._find_all_matches(self.grid)
        
        self.selected_gem_pos = None
        return total_reward

    def _generate_valid_grid(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_DIM, self.GRID_DIM))
            if not self._find_all_matches(self.grid) and self._has_possible_moves(self.grid):
                break

    def _swap_gems(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

    def _find_all_matches(self, grid):
        matched_gems = set()
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM - 2):
                if grid[r, c] == grid[r, c + 1] == grid[r, c + 2] and grid[r, c] != 0:
                    matched_gems.update([(r, c), (r, c + 1), (r, c + 2)])
        for c in range(self.GRID_DIM):
            for r in range(self.GRID_DIM - 2):
                if grid[r, c] == grid[r + 1, c] == grid[r + 2, c] and grid[r, c] != 0:
                    matched_gems.update([(r, c), (r + 1, c), (r + 2, c)])
        return matched_gems
    
    def _calculate_match_reward(self, matches):
        num_gems = len(matches)
        reward = num_gems # +1 per gem
        if num_gems == 4: reward += 5
        if num_gems >= 5: reward += 10
        self.score += reward
        return reward

    def _clear_gems(self, matches):
        for r, c in matches:
            self.grid[r, c] = 0

    def _apply_gravity(self):
        for c in range(self.GRID_DIM):
            empty_row = self.GRID_DIM - 1
            for r in range(self.GRID_DIM - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1
    
    def _fill_new_gems(self):
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                if self.grid[r, c] == 0:
                    # sound: gem_fall.wav
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

    def _has_possible_moves(self, grid):
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                # Check swap right
                if c < self.GRID_DIM - 1:
                    temp_grid = grid.copy()
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    if self._find_all_matches(temp_grid): return True
                # Check swap down
                if r < self.GRID_DIM - 1:
                    temp_grid = grid.copy()
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    if self._find_all_matches(temp_grid): return True
        return False
    
    def _shuffle_board(self):
        # sound: shuffle.wav
        flat_gems = self.grid.flatten()
        self.np_random.shuffle(flat_gems)
        self.grid = flat_gems.reshape((self.GRID_DIM, self.GRID_DIM))
        if not self._has_possible_moves(self.grid) or self._find_all_matches(self.grid):
             self._generate_valid_grid() # Failsafe reshuffle

    def _is_grid_clear(self):
        return np.all(self.grid == 0)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid_lines()
        self._render_gems()
        self._render_cursor_and_selection()
        self._render_effects()

    def _render_grid_lines(self):
        for i in range(self.GRID_DIM + 1):
            # Vertical
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES,
                             (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y),
                             (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT), 1)
            # Horizontal
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES,
                             (self.GRID_X, self.GRID_Y + i * self.CELL_SIZE),
                             (self.GRID_X + self.GRID_WIDTH, self.GRID_Y + i * self.CELL_SIZE), 1)

    def _render_gems(self):
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                gem_type = self.grid[r, c]
                if gem_type == 0:
                    continue
                
                color = self.GEM_COLORS[gem_type - 1]
                outline_color = tuple(max(0, x - 50) for x in color)
                
                center_x = self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                radius = self.CELL_SIZE // 2 - 4

                # Draw gem shapes using gfxdraw for anti-aliasing
                if gem_type == 1: # Circle
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, outline_color)
                elif gem_type == 2: # Square
                    rect = (center_x - radius, center_y - radius, radius*2, radius*2)
                    pygame.draw.rect(self.screen, color, rect, border_radius=3)
                    pygame.draw.rect(self.screen, outline_color, rect, width=1, border_radius=3)
                elif gem_type == 3: # Triangle
                    points = [
                        (center_x, center_y - radius),
                        (center_x - radius, center_y + radius * 0.7),
                        (center_x + radius, center_y + radius * 0.7)
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)
                    pygame.gfxdraw.aapolygon(self.screen, points, outline_color)
                elif gem_type == 4: # Diamond
                    points = [
                        (center_x, center_y - radius), (center_x + radius, center_y),
                        (center_x, center_y + radius), (center_x - radius, center_y)
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)
                    pygame.gfxdraw.aapolygon(self.screen, points, outline_color)
                elif gem_type == 5: # Hexagon
                    points = [(center_x + radius * math.cos(math.pi/3 * i),
                               center_y + radius * math.sin(math.pi/3 * i)) for i in range(6)]
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)
                    pygame.gfxdraw.aapolygon(self.screen, points, outline_color)
                elif gem_type == 6: # Star
                    points = []
                    for i in range(10):
                        r_val = radius if i % 2 == 0 else radius * 0.5
                        angle = i * math.pi / 5
                        points.append((center_x + r_val * math.sin(angle), center_y - r_val * math.cos(angle)))
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)
                    pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

    def _render_cursor_and_selection(self):
        # Render cursor
        r, c = self.cursor_pos
        rect = (self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        alpha = int(128 + 127 * math.sin(self.steps * 0.3))
        cursor_color = (255, 255, 0, alpha)
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, cursor_color, s.get_rect(), 4, border_radius=4)
        self.screen.blit(s, rect)

        # Render selection
        if self.selected_gem_pos:
            r, c = self.selected_gem_pos
            rect = (self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 4, border_radius=4)

    def _render_effects(self):
        for effect in self.animation_effects:
            effect_type, pos, color = effect
            if effect_type == 'particle_burst':
                r, c = pos
                center_x = self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                for _ in range(15): # 15 particles
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(5, self.CELL_SIZE * 0.6)
                    p_x = int(center_x + distance * math.cos(angle))
                    p_y = int(center_y + distance * math.sin(angle))
                    p_radius = random.randint(2, 4)
                    pygame.gfxdraw.filled_circle(self.screen, p_x, p_y, p_radius, color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        # Moves
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (640 - moves_text.get_width() - 20, 10))
        # Game Over
        if self.game_over:
            s = pygame.Surface((640, 400), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            over_text = self.font_game_over.render("GAME OVER", True, (255, 50, 50))
            text_rect = over_text.get_rect(center=(640 / 2, 400 / 2))
            self.screen.blit(over_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "cursor_pos": self.cursor_pos,
            "selected_gem_pos": self.selected_gem_pos,
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
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "dummy" for headless, "x11" or "windows" for visible
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Gem Matcher")
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_pressed = keys[pygame.K_SPACE]
        shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_pressed else 0, 1 if shift_pressed else 0]
        
        # --- Event handling ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Any keydown triggers a step in this turn-based game
                action_taken = True

        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")
            
            if terminated:
                print("Game Over! Final Score:", info['score'])
                # Optional: auto-reset after a delay
                pygame.time.wait(2000)
                obs, info = env.reset()

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to blit it to the display screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate
        
    env.close()