import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a tile. "
        "Select an adjacent tile to swap. Shift to reshuffle (costs 5 moves)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent tiles to create matches of 3 or more. "
        "Clear the entire board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 8, 6
    TILE_SIZE = 48
    GRID_LINE_WIDTH = 2
    
    # Centering the grid
    GRID_WIDTH = GRID_COLS * (TILE_SIZE + GRID_LINE_WIDTH) - GRID_LINE_WIDTH
    GRID_HEIGHT = GRID_ROWS * (TILE_SIZE + GRID_LINE_WIDTH) - GRID_LINE_WIDTH
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID_BG = (30, 40, 60)
    COLOR_GRID_LINE = (50, 60, 80)
    TILE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECT = (255, 255, 0)
    COLOR_TEXT = (220, 220, 230)
    
    MAX_MOVES = 25
    MAX_STEPS = 500
    RESHUFFLE_COST = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.grid = None
        self.cursor_pos = None
        self.selected_tile = None
        self.steps = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.particles = []
        
        self.previous_space_held = False
        self.previous_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        
        self.cursor_pos = [0, 0]
        self.selected_tile = None
        self.particles = []
        
        self.previous_space_held = False
        self.previous_shift_held = False
        
        self._generate_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

        # --- Handle Input ---
        # Movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
        
        # Shift (Reshuffle) - rising edge
        if shift_held and not self.previous_shift_held:
            if self.moves_left >= self.RESHUFFLE_COST:
                self.moves_left -= self.RESHUFFLE_COST
                self._reshuffle_board()
                reward -= 1 # Small penalty for reshuffling
                # SFX: board_shuffle
        
        # Space (Select/Swap) - rising edge
        if space_held and not self.previous_space_held:
            # SFX: select_tile
            cursor_pos_tuple = tuple(self.cursor_pos)
            if self.selected_tile is None:
                self.selected_tile = cursor_pos_tuple
            else:
                if cursor_pos_tuple == self.selected_tile:
                    self.selected_tile = None # Deselect if same tile
                elif self._is_adjacent(self.selected_tile, cursor_pos_tuple):
                    reward += self._attempt_swap(self.selected_tile, cursor_pos_tuple)
                    self.selected_tile = None
                else:
                    self.selected_tile = cursor_pos_tuple # Select new tile
        
        self.previous_space_held = space_held
        self.previous_shift_held = shift_held
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and self.score > 0 and np.sum(self.grid) == 0:
            reward += 100 # Big bonus for clearing the board
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _attempt_swap(self, pos1, pos2):
        self.moves_left -= 1
        
        # Perform swap
        self.grid[pos1], self.grid[pos2] = self.grid[pos2], self.grid[pos1]
        
        # Check for matches
        all_matches = self._find_all_matches()
        
        if not all_matches:
            # No match, swap back
            self.grid[pos1], self.grid[pos2] = self.grid[pos2], self.grid[pos1]
            # SFX: invalid_swap
            return -0.1

        # Cascade matches
        # SFX: match_found
        cascade_reward = 0
        while all_matches:
            # Calculate reward for this wave of matches
            for match in all_matches:
                cascade_reward += (len(match) - 2) # +1 for 3, +2 for 4, etc.
                for r, c in match:
                    self._create_particles(r, c, self.grid[r, c])

            # Clear matched tiles
            matched_indices = set()
            for match in all_matches:
                matched_indices.update(match)
            
            for r, c in matched_indices:
                self.grid[r, c] = 0 # 0 represents empty

            # Apply gravity and refill
            self._apply_gravity_and_refill()
            
            # Check for new matches
            all_matches = self._find_all_matches()
        
        self.score += cascade_reward
        
        # Anti-softlock check
        if not self._find_possible_moves():
            self._reshuffle_board()
            # SFX: board_shuffle_auto
            
        return cascade_reward

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
            "cursor_pos": list(self.cursor_pos),
            "selected_tile": self.selected_tile,
        }
        
    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, 
                         (self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT), 
                         border_radius=5)

        # Draw tiles and grid lines
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_val = self.grid[r, c]
                
                # Grid lines (drawn as gaps between tiles)
                if c > 0:
                    pygame.draw.line(self.screen, self.COLOR_GRID_LINE,
                                     (self.GRID_X + c * (self.TILE_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH, self.GRID_Y + r * (self.TILE_SIZE + self.GRID_LINE_WIDTH)),
                                     (self.GRID_X + c * (self.TILE_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH, self.GRID_Y + (r + 1) * (self.TILE_SIZE + self.GRID_LINE_WIDTH)),
                                     self.GRID_LINE_WIDTH)
                if r > 0:
                    pygame.draw.line(self.screen, self.COLOR_GRID_LINE,
                                     (self.GRID_X + c * (self.TILE_SIZE + self.GRID_LINE_WIDTH), self.GRID_Y + r * (self.TILE_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH),
                                     (self.GRID_X + (c + 1) * (self.TILE_SIZE + self.GRID_LINE_WIDTH), self.GRID_Y + r * (self.TILE_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH),
                                     self.GRID_LINE_WIDTH)
                
                if tile_val > 0:
                    color = self.TILE_COLORS[tile_val - 1]
                    rect = self._get_tile_rect(r, c)
                    pygame.draw.rect(self.screen, color, rect, border_radius=8)
                    
                    # 3D effect
                    highlight = tuple(min(255, x + 40) for x in color)
                    shadow = tuple(max(0, x - 40) for x in color)
                    pygame.draw.line(self.screen, highlight, rect.topleft, rect.topright, 2)
                    pygame.draw.line(self.screen, highlight, rect.topleft, rect.bottomleft, 2)
                    pygame.draw.line(self.screen, shadow, rect.bottomright, rect.topright, 2)
                    pygame.draw.line(self.screen, shadow, rect.bottomright, rect.bottomleft, 2)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color_with_alpha = p['color'] + (alpha,)
            pos = [int(p['pos'][0]), int(p['pos'][1])]
            radius = int(p['size'] * (p['life'] / p['max_life']))
            if radius > 0:
                try:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color_with_alpha)
                except TypeError: # Sometimes color can be invalid on the very last frame
                    pass


        # Draw selection highlight
        if self.selected_tile is not None:
            r, c = self.selected_tile
            rect = self._get_tile_rect(r, c)
            pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, 4, border_radius=10)
        
        # Draw cursor
        r, c = self.cursor_pos
        rect = self._get_tile_rect(r, c)
        time_factor = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
        line_width = 2 + int(time_factor * 3)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect.inflate(4,4), line_width, border_radius=12)
        
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Moves left
        moves_text = self.font_large.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 20))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            message = "BOARD CLEARED!" if np.sum(self.grid) == 0 else "GAME OVER"
            end_text = self.font_large.render(message, True, self.COLOR_SELECT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(end_text, text_rect)
    
    # --- Helper Functions ---
    
    def _get_tile_rect(self, r, c):
        x = self.GRID_X + c * (self.TILE_SIZE + self.GRID_LINE_WIDTH)
        y = self.GRID_Y + r * (self.TILE_SIZE + self.GRID_LINE_WIDTH)
        return pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
        
    def _is_adjacent(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2) == 1

    def _generate_board(self):
        num_colors = len(self.TILE_COLORS)
        self.grid = self.np_random.integers(1, num_colors + 1, size=(self.GRID_ROWS, self.GRID_COLS), dtype=np.int8)
        
        while self._find_all_matches() or not self._find_possible_moves():
            self.grid = self.np_random.integers(1, num_colors + 1, size=(self.GRID_ROWS, self.GRID_COLS), dtype=np.int8)
    
    def _reshuffle_board(self):
        flat_grid = self.grid.flatten()
        self.np_random.shuffle(flat_grid)
        self.grid = flat_grid.reshape((self.GRID_ROWS, self.GRID_COLS))

        while self._find_all_matches() or not self._find_possible_moves():
            self.np_random.shuffle(flat_grid)
            self.grid = flat_grid.reshape((self.GRID_ROWS, self.GRID_COLS))

    def _find_all_matches(self):
        matches = set()
        
        # Horizontal
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[r, c] > 0 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    match = {(r, c), (r, c+1), (r, c+2)}
                    i = c + 3
                    while i < self.GRID_COLS and self.grid[r, i] == self.grid[r, c]:
                        match.add((r, i))
                        i += 1
                    matches.add(frozenset(match))

        # Vertical
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if self.grid[r, c] > 0 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    match = {(r, c), (r+1, c), (r+2, c)}
                    i = r + 3
                    while i < self.GRID_ROWS and self.grid[i, c] == self.grid[r, c]:
                        match.add((i, c))
                        i += 1
                    matches.add(frozenset(match))
        
        return list(matches)

    def _find_possible_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Test swap right
                if c < self.GRID_COLS - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_all_matches():
                        self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                # Test swap down
                if r < self.GRID_ROWS - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_all_matches():
                        self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return False

    def _apply_gravity_and_refill(self):
        num_colors = len(self.TILE_COLORS)
        for c in range(self.GRID_COLS):
            empty_count = 0
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    self.grid[r + empty_count, c] = self.grid[r, c]
                    self.grid[r, c] = 0
            
            for r in range(empty_count):
                self.grid[r, c] = self.np_random.integers(1, num_colors + 1)
                
    def _create_particles(self, r, c, tile_val):
        rect = self._get_tile_rect(r, c)
        center_x, center_y = rect.center
        base_color = self.TILE_COLORS[tile_val - 1]
        
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': base_color,
                'size': self.np_random.uniform(2, 6)
            })

    def _check_termination(self):
        if self.game_over: return True
        
        board_cleared = np.sum(self.grid) == 0
        
        if board_cleared or self.moves_left <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # Set up display for human play
    pygame.display.set_caption("Match-3 Game")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    # Manual play loop
    running = True
    while running:
        action = [0, 0, 0] # no-op, released, released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Pause before reset
            obs, info = env.reset()

        env.clock.tick(30) # Limit frame rate for human play

    env.close()