
# Generated: 2025-08-28T05:42:21.524549
# Source Brief: brief_05670.md
# Brief Index: 5670

        
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
        "Move the cursor to an adjacent gem and press Space again to swap. "
        "Press Shift to deselect."
    )

    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to form lines of 3 or more. "
        "Create combos and chain reactions to maximize your score. "
        "Reach 5000 points in 20 moves to win!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_GEM_TYPES = 5
        self.GEM_SIZE = 40
        self.BOARD_WIDTH = self.GRID_SIZE * self.GEM_SIZE
        self.BOARD_OFFSET = ((self.WIDTH - self.BOARD_WIDTH) // 2, (self.HEIGHT - self.BOARD_WIDTH) // 2)
        self.WIN_SCORE = 5000
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 60, 80)
        self.COLOR_CURSOR = (255, 255, 255, 150)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_COMBO = (255, 220, 0)
        self.GEM_COLORS = [
            (220, 50, 50),   # Red
            (50, 220, 50),   # Green
            (50, 100, 220),  # Blue
            (220, 220, 50),  # Yellow
            (180, 50, 220),  # Purple
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
        self.font_main = pygame.font.Font(None, 36)
        self.font_combo = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.score = None
        self.moves_left = None
        self.steps = None
        self.game_over = None
        self.effects = []
        self.combo_display = None
        
        # --- RNG ---
        self.np_random = None

        self.reset()
        
        # Run validation check
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # If no seed, create a new generator, but subsequent resets won't re-seed
            # unless a new seed is passed. This is Gymnasium's standard behavior.
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_gem_pos = None
        self.effects = []
        self.combo_display = None
        
        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # Unpack factorized action
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

        # --- Handle Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] %= self.GRID_SIZE
        self.cursor_pos[1] %= self.GRID_SIZE

        # --- Handle Deselection ---
        if shift_press and self.selected_gem_pos:
            self.selected_gem_pos = None
            # sound: deselect_sound.wav

        # --- Handle Selection and Swapping ---
        if space_press:
            if not self.selected_gem_pos:
                self.selected_gem_pos = list(self.cursor_pos)
                # sound: select_gem.wav
            else:
                if self.selected_gem_pos == list(self.cursor_pos):
                     self.selected_gem_pos = None # Deselect if same gem is pressed again
                elif self._is_adjacent(self.selected_gem_pos, self.cursor_pos):
                    # This is a valid swap attempt, which constitutes a move.
                    self.moves_left -= 1
                    # sound: swap_attempt.wav
                    reward += self._execute_swap(self.selected_gem_pos, self.cursor_pos)
                    self.selected_gem_pos = None
                else:
                    # Selected a non-adjacent gem, so just move selection
                    self.selected_gem_pos = list(self.cursor_pos)
                    # sound: select_gem.wav
        
        terminated = self.score >= self.WIN_SCORE or self.moves_left <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
                # sound: game_win.wav
            else:
                # sound: game_over.wav
                pass

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _execute_swap(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        
        # Perform swap
        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]

        matches1 = self._find_matches_at(p1)
        matches2 = self._find_matches_at(p2)
        all_matches = matches1.union(matches2)

        if not all_matches:
            # No match, swap back
            self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
            return -0.1 # Penalty for invalid move

        # --- Process Matches and Cascades ---
        total_reward = 0
        combo_multiplier = 0
        
        current_matches = all_matches
        while current_matches:
            combo_multiplier += 1
            
            # Add reward for current matches
            num_gems_matched = len(current_matches)
            total_reward += num_gems_matched
            
            # Add visual effects for matched gems
            for (y, x) in current_matches:
                self._add_effect('match', (x, y), duration=15)
            # sound: match_clear.wav
            
            # Remove matched gems
            for r, c in current_matches:
                self.grid[r, c] = -1
            
            # Apply gravity and refill
            self._apply_gravity_and_refill()
            
            # Check for new matches (combos)
            current_matches = self._find_all_matches()

        if combo_multiplier > 1:
            total_reward += 10 # Combo bonus
            self.combo_display = {'text': f"COMBO x{combo_multiplier}!", 'timer': 30}
            # sound: combo_bonus.wav

        # Ensure board is not stuck
        if not self._find_possible_moves():
            self._reshuffle()
            self._add_effect('reshuffle', (self.GRID_SIZE//2 - 0.5, self.GRID_SIZE//2 - 0.5), duration=45)

        return total_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    # --- Rendering Methods ---

    def _render_game(self):
        self._draw_grid_lines()
        self._draw_gems()
        self._draw_cursor()
        self._draw_selection()
        self._draw_effects()

    def _draw_grid_lines(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.BOARD_OFFSET[0] + i * self.GEM_SIZE, self.BOARD_OFFSET[1])
            end_pos = (self.BOARD_OFFSET[0] + i * self.GEM_SIZE, self.BOARD_OFFSET[1] + self.BOARD_WIDTH)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.BOARD_OFFSET[0], self.BOARD_OFFSET[1] + i * self.GEM_SIZE)
            end_pos = (self.BOARD_OFFSET[0] + self.BOARD_WIDTH, self.BOARD_OFFSET[1] + i * self.GEM_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _draw_gems(self):
        radius = self.GEM_SIZE // 2 - 4
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    center_x = self.BOARD_OFFSET[0] + c * self.GEM_SIZE + self.GEM_SIZE // 2
                    center_y = self.BOARD_OFFSET[1] + r * self.GEM_SIZE + self.GEM_SIZE // 2
                    color = self.GEM_COLORS[gem_type]
                    
                    # Draw anti-aliased filled circle
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    
                    # Add a subtle highlight for 3D effect
                    highlight_color = tuple(min(255, val + 60) for val in color)
                    pygame.gfxdraw.arc(self.screen, center_x, center_y, radius - 2, 120, 240, highlight_color)


    def _draw_cursor(self):
        x = self.BOARD_OFFSET[0] + self.cursor_pos[0] * self.GEM_SIZE
        y = self.BOARD_OFFSET[1] + self.cursor_pos[1] * self.GEM_SIZE
        rect = pygame.Rect(x, y, self.GEM_SIZE, self.GEM_SIZE)
        
        # Use a surface for transparency
        s = pygame.Surface((self.GEM_SIZE, self.GEM_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), border_radius=6)
        self.screen.blit(s, (x, y))

    def _draw_selection(self):
        if self.selected_gem_pos:
            x, y = self.selected_gem_pos
            center_x = self.BOARD_OFFSET[0] + x * self.GEM_SIZE + self.GEM_SIZE // 2
            center_y = self.BOARD_OFFSET[1] + y * self.GEM_SIZE + self.GEM_SIZE // 2
            
            # Pulsating effect
            pulse = (math.sin(self.steps * 0.3) + 1) / 2
            radius = int(self.GEM_SIZE // 2 * (1.0 + pulse * 0.15))
            alpha = int(100 + pulse * 100)
            
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, (255, 255, 255, alpha))

    def _draw_effects(self):
        new_effects = []
        for effect in self.effects:
            effect['timer'] -= 1
            if effect['timer'] > 0:
                if effect['type'] == 'match':
                    x, y = effect['pos']
                    center_x = self.BOARD_OFFSET[0] + x * self.GEM_SIZE + self.GEM_SIZE // 2
                    center_y = self.BOARD_OFFSET[1] + y * self.GEM_SIZE + self.GEM_SIZE // 2
                    progress = 1 - (effect['timer'] / effect['duration'])
                    radius = int(progress * self.GEM_SIZE * 0.8)
                    alpha = int(255 * (1 - progress))
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, (255, 255, 255, alpha))
                elif effect['type'] == 'reshuffle':
                    text_surf = self.font_combo.render("RESHUFFLE!", True, self.COLOR_COMBO)
                    text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
                    self.screen.blit(text_surf, text_rect)
                new_effects.append(effect)
        self.effects = new_effects

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 20))

        # Moves
        moves_surf = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        moves_rect = moves_surf.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(moves_surf, moves_rect)
        
        # Combo Text
        if self.combo_display and self.combo_display['timer'] > 0:
            progress = self.combo_display['timer'] / 30.0
            size_scale = 1.0 + (1.0 - progress) * 0.5
            alpha = int(255 * min(1.0, progress * 3))
            
            font = pygame.font.Font(None, int(50 * size_scale))
            text_surf = font.render(self.combo_display['text'], True, self.COLOR_COMBO)
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
            self.screen.blit(text_surf, text_rect)
            self.combo_display['timer'] -= 1

    # --- Game Logic Helpers ---
    
    def _generate_board(self):
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
        while self._find_all_matches() or not self._find_possible_moves():
            # Clear existing matches
            matches = self._find_all_matches()
            while matches:
                for r, c in matches:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)
                matches = self._find_all_matches()
            
            # If still no moves, reshuffle completely
            if not self._find_possible_moves():
                self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
    
    def _find_matches_at(self, pos):
        r, c = pos
        if not (0 <= r < self.GRID_SIZE and 0 <= c < self.GRID_SIZE):
            return set()

        color = self.grid[r, c]
        matches = set()

        # Horizontal
        h_matches = {(r, c)}
        for i in range(c - 1, -1, -1):
            if self.grid[r, i] == color: h_matches.add((r, i))
            else: break
        for i in range(c + 1, self.GRID_SIZE):
            if self.grid[r, i] == color: h_matches.add((r, i))
            else: break
        if len(h_matches) >= 3: matches.update(h_matches)

        # Vertical
        v_matches = {(r, c)}
        for i in range(r - 1, -1, -1):
            if self.grid[i, c] == color: v_matches.add((i, c))
            else: break
        for i in range(r + 1, self.GRID_SIZE):
            if self.grid[i, c] == color: v_matches.add((i, c))
            else: break
        if len(v_matches) >= 3: matches.update(v_matches)
        
        return matches

    def _find_all_matches(self):
        all_matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color = self.grid[r, c]
                if color == -1: continue

                # Horizontal check
                if c < self.GRID_SIZE - 2 and self.grid[r, c+1] == color and self.grid[r, c+2] == color:
                    all_matches.add((r, c)); all_matches.add((r, c+1)); all_matches.add((r, c+2))
                
                # Vertical check
                if r < self.GRID_SIZE - 2 and self.grid[r+1, c] == color and self.grid[r+2, c] == color:
                    all_matches.add((r, c)); all_matches.add((r+1, c)); all_matches.add((r+2, c))
        return all_matches

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if empty_row != r:
                        self.grid[empty_row, c] = self.grid[r, c]
                    empty_row -= 1
            
            for r in range(empty_row, -1, -1):
                self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Try swapping right
                if c < self.GRID_SIZE - 1:
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                    if self._find_matches_at((r,c)) or self._find_matches_at((r,c+1)):
                        moves.append(((r,c), (r,c+1)))
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c] # Swap back
                # Try swapping down
                if r < self.GRID_SIZE - 1:
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                    if self._find_matches_at((r,c)) or self._find_matches_at((r+1,c)):
                        moves.append(((r,c), (r+1,c)))
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c] # Swap back
        return moves

    def _reshuffle(self):
        # Flatten, shuffle, and reshape, then ensure validity
        flat_grid = list(self.grid.flatten())
        self.np_random.shuffle(flat_grid)
        self.grid = np.array(flat_grid).reshape((self.GRID_SIZE, self.GRID_SIZE))
        
        # Keep reshuffling until a valid board is made
        while self._find_all_matches() or not self._find_possible_moves():
            flat_grid = list(self.grid.flatten())
            self.np_random.shuffle(flat_grid)
            self.grid = np.array(flat_grid).reshape((self.GRID_SIZE, self.GRID_SIZE))
        
    def _is_adjacent(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

    def _add_effect(self, type, pos, duration):
        self.effects.append({'type': type, 'pos': pos, 'timer': duration, 'duration': duration})

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
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up a window for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Match-3 Gym Environment")
    
    # Game loop for human control
    running = True
    while running:
        # Action defaults to NO-OP
        action = [0, 0, 0] # movement, space, shift

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Press 'r' to reset
                    obs, info = env.reset()
                    print("--- Environment Reset ---")
        
        # Only step if an action was taken (or if auto_advance is True)
        if any(action) or GameEnv.auto_advance:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")

            if terminated:
                print("--- GAME OVER ---")
                if info['score'] >= env.WIN_SCORE:
                    print("YOU WIN!")
                else:
                    print("Better luck next time!")
                
                # Wait for a moment, then reset
                pygame.time.wait(2000)
                obs, info = env.reset()
                print("--- New Game Started ---")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we don't need to cap the framerate.
        # The loop waits for user input.
    
    env.close()