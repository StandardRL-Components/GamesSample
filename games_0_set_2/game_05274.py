
# Generated: 2025-08-28T04:31:10.052675
# Source Brief: brief_05274.md
# Brief Index: 5274

        
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
        "Controls: Use arrow keys to move the selector. Press Space to reveal a tile. "
        "Match all pairs before you run out of moves."
    )

    game_description = (
        "A minimalist memory puzzle game. Select tiles to reveal hidden symbols. "
        "Find all the matching pairs to clear the board and win."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 4, 6
    NUM_PAIRS = (GRID_ROWS * GRID_COLS) // 2
    MAX_MOVES = 20
    MAX_STEPS = MAX_MOVES * 2 + NUM_PAIRS * 2 # Generous upper bound

    # Colors
    COLOR_BG = (30, 35, 40)
    COLOR_TILE_HIDDEN = (60, 70, 80)
    COLOR_TILE_REVEALED = (120, 130, 140)
    COLOR_CURSOR = (255, 220, 0)
    COLOR_MATCH = (70, 220, 120)
    COLOR_MISMATCH = (220, 70, 70)
    COLOR_TEXT = (230, 230, 240)
    
    SYMBOL_COLORS = [
        (255, 107, 107), (255, 184, 107), (230, 255, 107), (107, 255, 133),
        (107, 255, 255), (107, 133, 255), (184, 107, 255), (255, 107, 230),
        (255, 99, 71),   (64, 224, 208),  (218, 112, 214), (100, 149, 237)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Grid and tile dimensions
        self.grid_area_height = self.SCREEN_HEIGHT - 60
        self.padding = 10
        self.tile_size = min(
            (self.SCREEN_WIDTH - self.padding * (self.GRID_COLS + 1)) // self.GRID_COLS,
            (self.grid_area_height - self.padding * (self.GRID_ROWS + 1)) // self.GRID_ROWS
        )
        self.grid_width = self.GRID_COLS * (self.tile_size + self.padding) - self.padding
        self.grid_height = self.GRID_ROWS * (self.tile_size + self.padding) - self.padding
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height - 30) // 2 + 30

        # Initialize state variables
        self.grid = None
        self.revealed_grid = None
        self.matched_pairs = None
        self.cursor_pos = None
        self.selected_tiles = None
        self.moves_left = None
        self.last_match_info = None
        self.last_space_held = False
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [0, 0] # [row, col]
        self.selected_tiles = []
        self.matched_pairs = set()
        self.last_match_info = None # Stores info for visual feedback
        self.last_space_held = False

        self._generate_grid()
        
        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        symbols = list(range(self.NUM_PAIRS)) * 2
        self.np_random.shuffle(symbols)
        self.grid = np.array(symbols).reshape((self.GRID_ROWS, self.GRID_COLS))
        self.revealed_grid = np.zeros_like(self.grid, dtype=bool)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.last_match_info = None

        # --- 1. Process results from the previous action ---
        if len(self.selected_tiles) == 2:
            (r1, c1), (r2, c2) = self.selected_tiles
            symbol1 = self.grid[r1, c1]
            symbol2 = self.grid[r2, c2]

            if symbol1 == symbol2:
                # MATCH
                reward += 5.0
                self.score += 5.0
                self.matched_pairs.add(symbol1)
                self.last_match_info = {'status': 'correct', 'tiles': self.selected_tiles}
                # Sound: # a_success_chime.wav
            else:
                # MISMATCH
                reward -= 0.1
                self.score -= 0.1
                self.revealed_grid[r1, c1] = False
                self.revealed_grid[r2, c2] = False
                self.last_match_info = {'status': 'incorrect', 'tiles': self.selected_tiles}
                # Sound: # a_failure_buzz.wav
            
            self.selected_tiles = []

        # --- 2. Process the new action ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        self.last_space_held = space_held

        # Handle Movement
        if movement == 1: self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_ROWS
        elif movement == 2: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_ROWS
        elif movement == 3: self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_COLS
        elif movement == 4: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_COLS

        # Handle Tile Selection (Space Press)
        if space_press and len(self.selected_tiles) < 2:
            r, c = self.cursor_pos
            symbol_id = self.grid[r, c]
            is_already_revealed = self.revealed_grid[r, c]
            is_matched = symbol_id in self.matched_pairs
            
            if not is_already_revealed and not is_matched:
                self.revealed_grid[r, c] = True
                self.selected_tiles.append(tuple(self.cursor_pos))
                # Sound: # a_tile_flip.wav
                
                if len(self.selected_tiles) == 2:
                    self.moves_left -= 1

        # --- 3. Check for termination ---
        all_matched = len(self.matched_pairs) == self.NUM_PAIRS
        out_of_moves = self.moves_left <= 0
        
        terminated = all_matched or out_of_moves or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            if all_matched:
                reward += 50.0
                self.score += 50.0
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "pairs_matched": len(self.matched_pairs),
        }

    def _render_game(self):
        # Draw all tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                symbol_id = self.grid[r, c]
                is_revealed = self.revealed_grid[r, c]
                is_matched = symbol_id in self.matched_pairs
                
                tile_rect = self._get_tile_rect(r, c)
                
                if is_matched:
                    continue # Don't draw matched tiles

                # Determine tile color
                tile_color = self.COLOR_TILE_REVEALED if is_revealed else self.COLOR_TILE_HIDDEN
                pygame.draw.rect(self.screen, tile_color, tile_rect, border_radius=5)
                
                if is_revealed:
                    self._draw_symbol(symbol_id, tile_rect.center)
        
        # Draw match/mismatch feedback borders
        if self.last_match_info:
            color = self.COLOR_MATCH if self.last_match_info['status'] == 'correct' else self.COLOR_MISMATCH
            for r, c in self.last_match_info['tiles']:
                rect = self._get_tile_rect(r, c)
                pygame.draw.rect(self.screen, color, rect, width=4, border_radius=5)

        # Draw cursor
        if not self.game_over:
            cursor_rect = self._get_tile_rect(self.cursor_pos[0], self.cursor_pos[1])
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=3, border_radius=5)

    def _render_ui(self):
        # Draw UI background bar
        ui_bar_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 50)
        pygame.draw.rect(self.screen, (40, 45, 50), ui_bar_rect)
        pygame.draw.line(self.screen, (20,25,30), (0, 50), (self.SCREEN_WIDTH, 50), 2)

        # Moves Left
        moves_text = self.font_medium.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 12))

        # Score
        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(centerx=self.SCREEN_WIDTH / 2, y=12)
        self.screen.blit(score_text, score_rect)
        
        # Pairs Matched
        pairs_text = self.font_medium.render(f"Pairs: {len(self.matched_pairs)}/{self.NUM_PAIRS}", True, self.COLOR_TEXT)
        pairs_rect = pairs_text.get_rect(right=self.SCREEN_WIDTH - 20, y=12)
        self.screen.blit(pairs_text, pairs_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win = len(self.matched_pairs) == self.NUM_PAIRS
            message = "YOU WIN!" if win else "GAME OVER"
            color = self.COLOR_MATCH if win else self.COLOR_MISMATCH
            
            text = self.font_large.render(message, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _get_tile_rect(self, r, c):
        x = self.grid_offset_x + c * (self.tile_size + self.padding)
        y = self.grid_offset_y + r * (self.tile_size + self.padding)
        return pygame.Rect(x, y, self.tile_size, self.tile_size)

    def _draw_symbol(self, symbol_id, center):
        size = self.tile_size * 0.6
        half = size / 2
        cx, cy = center
        color = self.SYMBOL_COLORS[symbol_id % len(self.SYMBOL_COLORS)]
        
        points = []
        if symbol_id == 0: # Circle
            pygame.gfxdraw.filled_circle(self.screen, int(cx), int(cy), int(half), color)
            pygame.gfxdraw.aacircle(self.screen, int(cx), int(cy), int(half), color)
        elif symbol_id == 1: # Square
            pygame.draw.rect(self.screen, color, (cx - half, cy - half, size, size), border_radius=3)
        elif symbol_id == 2: # Triangle
            points = [(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        elif symbol_id == 3: # Diamond
            points = [(cx, cy - half), (cx - half, cy), (cx, cy + half), (cx + half, cy)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        elif symbol_id == 4: # 'X'
            pygame.draw.line(self.screen, color, (cx - half, cy - half), (cx + half, cy + half), 7)
            pygame.draw.line(self.screen, color, (cx - half, cy + half), (cx + half, cy - half), 7)
        elif symbol_id == 5: # Plus
            pygame.draw.line(self.screen, color, (cx, cy - half), (cx, cy + half), 7)
            pygame.draw.line(self.screen, color, (cx - half, cy), (cx + half, cy), 7)
        elif symbol_id == 6: # Star
            points = []
            for i in range(5):
                angle_rad = math.radians(i * 72 - 90)
                outer_x = cx + half * math.cos(angle_rad)
                outer_y = cy + half * math.sin(angle_rad)
                points.append((outer_x, outer_y))
                
                angle_rad_inner = math.radians((i * 72 - 90) + 36)
                inner_x = cx + (half/2.5) * math.cos(angle_rad_inner)
                inner_y = cy + (half/2.5) * math.sin(angle_rad_inner)
                points.insert(-1, (inner_x, inner_y))
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        elif symbol_id == 7: # Hexagon
            points = [(cx + half * math.cos(math.radians(60*i)), cy + half * math.sin(math.radians(60*i))) for i in range(6)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        elif symbol_id == 8: # Pentagon
            points = [(cx + half * math.cos(math.radians(72*i - 90)), cy + half * math.sin(math.radians(72*i - 90))) for i in range(5)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        elif symbol_id == 9: # Circle with dot
            pygame.gfxdraw.filled_circle(self.screen, int(cx), int(cy), int(half), color)
            pygame.gfxdraw.aacircle(self.screen, int(cx), int(cy), int(half), color)
            pygame.gfxdraw.filled_circle(self.screen, int(cx), int(cy), int(half * 0.3), self.COLOR_TILE_REVEALED)
        elif symbol_id == 10: # Square with dot
            pygame.draw.rect(self.screen, color, (cx - half, cy - half, size, size), border_radius=3)
            pygame.draw.circle(self.screen, self.COLOR_TILE_REVEALED, (cx, cy), half * 0.3)
        elif symbol_id == 11: # Inverted Triangle
            points = [(cx, cy + half), (cx - half, cy - half), (cx + half, cy - half)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        ''' Call this at the end of __init__ to verify implementation. '''
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
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Memory Match")
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # No-op
    
    print(GameEnv.user_guide)

    while not terminated:
        # --- Human Controls ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        
        # Movement (prioritize one direction)
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, term, trunc, info = env.step(action)
        terminated = term

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we need to decide when to "tick".
        # For human play, we can tick at a steady rate.
        clock.tick(15) # Controls how fast you can press buttons
        
    env.close()
    print(f"Game Over. Final Score: {info['score']:.1f}")