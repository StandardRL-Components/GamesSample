
# Generated: 2025-08-27T21:29:28.388437
# Source Brief: brief_02800.md
# Brief Index: 2800

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to tilt the cavern. "
        "Align 5 identical crystals in a row or column to score."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Tilt a 5x5 grid to slide crystals. "
        "Create horizontal or vertical lines of 5 identical crystals to score points. "
        "Achieve 5 matched lines before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 5
    MAX_MOVES = 20
    WIN_CONDITION_MATCHES = 5

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (50, 60, 80)
    CRYSTAL_PALETTE = {
        1: {'base': (255, 50, 50), 'light': (255, 120, 120), 'dark': (180, 20, 20)},   # Red
        2: {'base': (50, 255, 50), 'light': (120, 255, 120), 'dark': (20, 180, 20)},   # Green
        3: {'base': (50, 150, 255), 'light': (120, 200, 255), 'dark': (20, 100, 180)}, # Blue
        4: {'base': (255, 255, 50), 'light': (255, 255, 120), 'dark': (180, 180, 20)}, # Yellow
        5: {'base': (200, 50, 255), 'light': (230, 120, 255), 'dark': (140, 20, 180)}, # Purple
    }
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)
    COLOR_WIN = (100, 255, 150)
    COLOR_LOSE = (255, 100, 100)
    
    # Isometric rendering parameters
    TILE_WIDTH_ISO = 64
    TILE_HEIGHT_ISO = 32
    CRYSTAL_HEIGHT = 40
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Isometric grid origin
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_SIZE * self.TILE_HEIGHT_ISO) // 2 + 30

        # State variables (will be initialized in reset)
        self.grid = None
        self.moves_left = 0
        self.score = 0
        self.matched_lines = set()
        self.game_over = False
        self.win = False
        self.steps = 0
        
        # Initialize state variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self.np_random.integers(1, len(self.CRYSTAL_PALETTE) + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.matched_lines = set()
        self.game_over = False
        self.win = False
        self.steps = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.steps += 1
        self.moves_left -= 1
        
        moved_crystals_count = self._apply_gravity(movement)
        
        # Movement reward
        if moved_crystals_count > 0:
            reward += moved_crystals_count * 0.1 # Scaled down from +1 to keep rewards reasonable
        elif movement != 0:
            reward -= 0.1

        # Check for new matches and calculate match reward
        match_reward = self._check_matches()
        reward += match_reward

        terminated = self.moves_left <= 0 or len(self.matched_lines) >= self.WIN_CONDITION_MATCHES
        
        if terminated:
            self.game_over = True
            if len(self.matched_lines) >= self.WIN_CONDITION_MATCHES:
                self.win = True
                reward += 100 # Win bonus
            else:
                self.win = False
                reward -= 10 # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _apply_gravity(self, direction):
        if direction == 0: # No-op
            return 0

        grid_before = self.grid.copy()
        
        if direction == 1: # Up
            for c in range(self.GRID_SIZE):
                col = [grid_before[r, c] for r in range(self.GRID_SIZE) if grid_before[r, c] != 0]
                for r in range(self.GRID_SIZE):
                    self.grid[r, c] = col[r] if r < len(col) else 0
        elif direction == 2: # Down
            for c in range(self.GRID_SIZE):
                col = [grid_before[r, c] for r in range(self.GRID_SIZE) if grid_before[r, c] != 0]
                for r in range(self.GRID_SIZE):
                    self.grid[self.GRID_SIZE - 1 - r, c] = col[len(col) - 1 - r] if r < len(col) else 0
        elif direction == 3: # Left
            for r in range(self.GRID_SIZE):
                row = [grid_before[r, c] for c in range(self.GRID_SIZE) if grid_before[r, c] != 0]
                for c in range(self.GRID_SIZE):
                    self.grid[r, c] = row[c] if c < len(row) else 0
        elif direction == 4: # Right
            for r in range(self.GRID_SIZE):
                row = [grid_before[r, c] for c in range(self.GRID_SIZE) if grid_before[r, c] != 0]
                for c in range(self.GRID_SIZE):
                    self.grid[r, self.GRID_SIZE - 1 - c] = row[len(row) - 1 - c] if c < len(row) else 0
        
        # Count how many non-zero cells are now in a different position
        moved_count = np.count_nonzero((grid_before != 0) & (grid_before != self.grid))
        return moved_count

    def _check_matches(self):
        match_reward = 0
        # Check rows
        for r in range(self.GRID_SIZE):
            if len(set(self.grid[r, :])) == 1 and self.grid[r, 0] != 0:
                if ('row', r) not in self.matched_lines:
                    self.matched_lines.add(('row', r))
                    self.score += 10
                    match_reward += 10
                    # Sound effect placeholder: pygame.mixer.Sound('match.wav').play()

        # Check columns
        for c in range(self.GRID_SIZE):
            if len(set(self.grid[:, c])) == 1 and self.grid[0, c] != 0:
                if ('col', c) not in self.matched_lines:
                    self.matched_lines.add(('col', c))
                    self.score += 10
                    match_reward += 10
                    # Sound effect placeholder: pygame.mixer.Sound('match.wav').play()
        
        return match_reward
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_grid()
        self._render_crystals()

    def _iso_transform(self, grid_x, grid_y):
        screen_x = self.origin_x + (grid_x - grid_y) * (self.TILE_WIDTH_ISO / 2)
        screen_y = self.origin_y + (grid_x + grid_y) * (self.TILE_HEIGHT_ISO / 2)
        return screen_x, screen_y

    def _render_grid(self):
        for r in range(self.GRID_SIZE + 1):
            start_pos = self._iso_transform(r, 0)
            end_pos = self._iso_transform(r, self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)
        for c in range(self.GRID_SIZE + 1):
            start_pos = self._iso_transform(0, c)
            end_pos = self._iso_transform(self.GRID_SIZE, c)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)

    def _render_crystals(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                crystal_color_id = self.grid[r, c]
                if crystal_color_id != 0:
                    iso_x, iso_y = self._iso_transform(c, r)
                    # Adjust for tile center
                    iso_x += self.TILE_WIDTH_ISO / 2
                    iso_y += self.TILE_HEIGHT_ISO / 2
                    self._draw_iso_cube(self.screen, iso_x, iso_y, self.CRYSTAL_PALETTE[crystal_color_id])
    
    def _draw_iso_cube(self, surface, iso_x, iso_y, palette):
        h = self.CRYSTAL_HEIGHT
        w_half = self.TILE_WIDTH_ISO / 2
        h_half = self.TILE_HEIGHT_ISO / 2

        # Points are relative to the top-center of the cube's footprint
        top_points = [
            (iso_x, iso_y - h),
            (iso_x + w_half, iso_y - h + h_half),
            (iso_x, iso_y - h + h_half * 2),
            (iso_x - w_half, iso_y - h + h_half),
        ]
        left_face_points = [
            (iso_x - w_half, iso_y + h_half),
            (iso_x, iso_y + h_half * 2),
            (iso_x, iso_y - h + h_half * 2),
            (iso_x - w_half, iso_y - h + h_half),
        ]
        right_face_points = [
            (iso_x + w_half, iso_y + h_half),
            (iso_x, iso_y + h_half * 2),
            (iso_x, iso_y - h + h_half * 2),
            (iso_x + w_half, iso_y - h + h_half),
        ]

        # Use integer coordinates for drawing
        def to_int_coords(points):
            return [(int(p[0]), int(p[1])) for p in points]

        pygame.gfxdraw.filled_polygon(surface, to_int_coords(left_face_points), palette['dark'])
        pygame.gfxdraw.filled_polygon(surface, to_int_coords(right_face_points), palette['base'])
        pygame.gfxdraw.filled_polygon(surface, to_int_coords(top_points), palette['light'])
        
        # Anti-aliased outlines for a clean look
        pygame.gfxdraw.aapolygon(surface, to_int_coords(left_face_points), palette['dark'])
        pygame.gfxdraw.aapolygon(surface, to_int_coords(right_face_points), palette['base'])
        pygame.gfxdraw.aapolygon(surface, to_int_coords(top_points), palette['light'])
        
    def _render_ui(self):
        # Helper to draw text with a shadow
        def draw_text(text, font, color, x, y):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow, (x + 2, y + 2))
            main_text = font.render(text, True, color)
            self.screen.blit(main_text, (x, y))

        # Score display
        score_text = f"SCORE: {self.score}"
        draw_text(score_text, self.font_ui, self.COLOR_TEXT, 10, 10)
        
        # Moves left display
        moves_text = f"MOVES: {self.moves_left}"
        text_width = self.font_ui.size(moves_text)[0]
        draw_text(moves_text, self.font_ui, self.COLOR_TEXT, self.SCREEN_WIDTH - text_width - 10, 10)
        
    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.win:
            text = "YOU WIN!"
            color = self.COLOR_WIN
        else:
            text = "OUT OF MOVES"
            color = self.COLOR_LOSE
            
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        shadow_surf = self.font_game_over.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (text_rect.x + 3, text_rect.y + 3))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "matches": len(self.matched_lines),
        }

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
    # This block allows you to run the game directly to test it
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Cavern")
    clock = pygame.time.Clock()
    
    running = True
    game_over = False
    
    print(env.user_guide)

    while running:
        action = np.array([0, 0, 0]) # Default action is no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if not game_over and event.type == pygame.KEYDOWN:
                # Map keys to actions
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                # Only step if a valid key was pressed
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    game_over = terminated
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
        
        # Get the current observation from the environment
        frame = env._get_observation()
        # Pygame uses (width, height), numpy uses (height, width)
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()