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
        "Controls: Arrow keys to move the cursor. Space to select a gem, then move the cursor and press Space again to swap. Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent gems to create matches of 3 or more. Collect 50 gems in 50 moves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 8
    NUM_GEM_TYPES = 6
    TILE_WIDTH_HALF = 28
    TILE_HEIGHT_HALF = 14
    GEM_Y_OFFSET = -8  # To make gems look like they sit on top of tiles

    WIN_SCORE = 50
    MAX_MOVES = 50

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    GRID_COLOR = (40, 60, 80)
    CURSOR_COLOR = (255, 255, 255, 150)
    SELECTION_COLOR = (255, 255, 0, 200)
    TEXT_COLOR = (240, 240, 240)
    SHADOW_COLOR = (10, 15, 20)

    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 150, 80),  # Orange
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('sans-serif', 36, bold=True)
        self.font_small = pygame.font.SysFont('sans-serif', 24)

        # Game state variables are initialized in reset()
        self.grid = None
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.prev_space_held = 0
        self.prev_shift_held = 0
        self.last_match_effects = []
        self.rng = None

        # self.reset() is called by the wrapper or user, not in __init__
        # to be compliant with Gymnasium API.
        # We initialize a default RNG here for cases where reset() is not called with a seed first.
        self.rng = np.random.default_rng()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_gem_pos = None
        self.prev_space_held = 0
        self.prev_shift_held = 0
        self.last_match_effects = []
        self._create_initial_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action
        reward = 0
        terminated = False

        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Up
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1) # Down
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Left
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1) # Right

        # 2. Handle deselection
        if shift_pressed and self.selected_gem_pos:
            self.selected_gem_pos = None

        # 3. Handle selection/swap
        elif space_pressed:
            if not self.selected_gem_pos:
                self.selected_gem_pos = list(self.cursor_pos)
            else:
                # Attempt a swap
                r1, c1 = self.selected_gem_pos
                r2, c2 = self.cursor_pos
                is_adjacent = abs(r1 - r2) + abs(c1 - c2) == 1

                if is_adjacent:
                    self.moves_left -= 1
                    self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c1], self.grid[r1][c1]

                    total_gems_cleared_this_turn = 0
                    cascade_bonus = 0
                    
                    # Cascade loop
                    while True:
                        matches = self._find_all_matches()
                        if not matches:
                            break
                        
                        num_cleared_this_cascade = len(matches)
                        total_gems_cleared_this_turn += num_cleared_this_cascade
                        
                        reward += num_cleared_this_cascade
                        if num_cleared_this_cascade >= 4:
                            reward += 5 # Bonus for 4+ match
                        if cascade_bonus > 0:
                            reward += cascade_bonus # Cascade bonus
                        cascade_bonus += 2

                        self._add_match_effects(matches)
                        self._remove_gems(matches)
                        self._drop_and_fill_gems()
                    
                    if total_gems_cleared_this_turn > 0:
                        self.score += total_gems_cleared_this_turn
                    else:
                        # No match, swap back
                        self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c1], self.grid[r1][c1]
                        reward = -0.2
                    
                    self.selected_gem_pos = None # Deselect after swap attempt
                else:
                    # Not adjacent, just move selection to new cursor pos
                    self.selected_gem_pos = list(self.cursor_pos)

        # 4. Check for post-cascade soft-lock
        if not self._check_for_possible_moves():
            self._reshuffle_board()

        # 5. Check for termination
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.moves_left <= 0:
            reward += -10
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "moves_left": self.moves_left}

    # --- Helper Methods ---

    def _grid_to_screen(self, r, c):
        origin_x = self.WIDTH / 2
        origin_y = self.HEIGHT / 2 - (self.GRID_SIZE * self.TILE_HEIGHT_HALF / 2) + 20
        screen_x = origin_x + (c - r) * self.TILE_WIDTH_HALF
        screen_y = origin_y + (c + r) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Draw grid tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                x, y = self._grid_to_screen(r, c)
                points = [
                    (x, y - self.TILE_HEIGHT_HALF),
                    (x + self.TILE_WIDTH_HALF, y),
                    (x, y + self.TILE_HEIGHT_HALF),
                    (x - self.TILE_WIDTH_HALF, y)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.GRID_COLOR)
                pygame.gfxdraw.aapolygon(self.screen, points, self.GRID_COLOR)

        # Draw gems
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.grid[r][c]
                if gem_type != -1:
                    x, y = self._grid_to_screen(r, c)
                    self._render_gem(x, y + self.GEM_Y_OFFSET, self.GEM_COLORS[gem_type])
        
        # Draw selection and cursor
        if self.selected_gem_pos:
            r, c = self.selected_gem_pos
            x, y = self._grid_to_screen(r, c)
            self._render_highlight(x, y, self.SELECTION_COLOR, 4)
        
        r, c = self.cursor_pos
        x, y = self._grid_to_screen(r, c)
        self._render_highlight(x, y, self.CURSOR_COLOR, 2)

        # Draw match effects and clear them
        for effect in self.last_match_effects:
            self._render_match_effect(effect['pos'])
        self.last_match_effects = []

    def _render_gem(self, x, y, color):
        w, h = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        
        # Darker base for 3D effect
        base_color = tuple(max(0, val - 60) for val in color)
        
        # Main gem shape (rhombus)
        points = [(x, y - h), (x + w, y), (x, y + h), (x - w, y)]
        pygame.gfxdraw.filled_polygon(self.screen, points, base_color)

        # Bright inner highlight
        highlight_points = [(x, y - h*0.6), (x + w*0.6, y), (x, y + h*0.6), (x - w*0.6, y)]
        pygame.gfxdraw.filled_polygon(self.screen, highlight_points, color)
        pygame.gfxdraw.aapolygon(self.screen, highlight_points, color)

    def _render_highlight(self, x, y, color, thickness):
        w, h = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        points = [
            (x, y - h), (x + w, y), (x, y + h), (x - w, y)
        ]
        # Create a surface for transparency, required for alpha colors
        highlight_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.draw.lines(highlight_surface, color, True, points, width=thickness)
        self.screen.blit(highlight_surface, (0, 0))

    def _render_match_effect(self, pos):
        x, y = pos
        y += self.GEM_Y_OFFSET
        # Simple starburst effect
        for i in range(8):
            angle = i * (math.pi / 4)
            end_x = x + math.cos(angle) * 25
            end_y = y + math.sin(angle) * 25
            pygame.draw.line(self.screen, (255, 255, 200), (x, y), (int(end_x), int(end_y)), 2)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.TEXT_COLOR)
        self.screen.blit(score_text, (20, 10))

        # Moves
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.TEXT_COLOR)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(msg, True, (255, 255, 100))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_initial_grid(self):
        while True:
            self.grid = self.rng.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
            # Ensure no matches on start
            while self._find_all_matches():
                matches = self._find_all_matches()
                self._remove_gems(matches)
                self._drop_and_fill_gems()
            
            # Ensure at least one move is possible
            if self._check_for_possible_moves():
                break

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem = self.grid[r][c]
                if gem == -1: continue
                
                # Horizontal check
                if c < self.GRID_SIZE - 2 and self.grid[r][c+1] == gem and self.grid[r][c+2] == gem:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                
                # Vertical check
                if r < self.GRID_SIZE - 2 and self.grid[r+1][c] == gem and self.grid[r+2][c] == gem:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _check_for_possible_moves(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Try swapping with right neighbor
                if c < self.GRID_SIZE - 1:
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                    if self._find_all_matches():
                        self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c] # Swap back
                        return True
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c] # Swap back
                
                # Try swapping with down neighbor
                if r < self.GRID_SIZE - 1:
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
                    if self._find_all_matches():
                        self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c] # Swap back
                        return True
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c] # Swap back
        return False

    def _reshuffle_board(self):
        flat_gems = [gem for row in self.grid for gem in row if gem != -1]
        self.rng.shuffle(flat_gems)
        
        while True:
            shuffled_gems = list(flat_gems)
            new_grid = [[-1] * self.GRID_SIZE for _ in range(self.GRID_SIZE)]
            idx = 0
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    if idx < len(shuffled_gems):
                        new_grid[r][c] = shuffled_gems[idx]
                        idx += 1
            
            self.grid = new_grid
            if not self._find_all_matches() and self._check_for_possible_moves():
                break

    def _add_match_effects(self, matches):
        for r, c in matches:
            screen_pos = self._grid_to_screen(r, c)
            self.last_match_effects.append({'pos': screen_pos})

    def _remove_gems(self, matches):
        for r, c in matches:
            self.grid[r][c] = -1 # Mark as empty

    def _drop_and_fill_gems(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r][c] != -1:
                    if r != empty_row:
                        self.grid[empty_row][c] = self.grid[r][c]
                        self.grid[r][c] = -1
                    empty_row -= 1
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r][c] == -1:
                    self.grid[r][c] = self.rng.integers(0, self.NUM_GEM_TYPES)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # no-op, released, released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    print("Game Reset")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
        # Only step if an action is taken to avoid rapid-fire actions
        if action != [0,0,0] or env.auto_advance:
            obs, reward, terminated, truncated, info = env.step(action)
        
            if terminated:
                print(f"Game Over! Final Score: {info['score']}")
                # Render one last time to show final state
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                pygame.time.wait(2000) # Wait 2 seconds
                obs, info = env.reset() # Reset for a new game

        # Display the frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    env.close()