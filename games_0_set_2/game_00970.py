
# Generated: 2025-08-27T15:21:34.156352
# Source Brief: brief_00970.md
# Brief Index: 970

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to reveal a tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced memory game. Race against the clock to find all 8 pairs of matching symbols on the grid."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
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
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_large = pygame.font.Font(None, 72)

        # Game constants
        self.FPS = 30
        self.GRID_SIZE = 4
        self.NUM_PAIRS = 8
        self.INITIAL_TIME = 60.0  # seconds
        self.MAX_STEPS = int(self.INITIAL_TIME * self.FPS)
        self.MISMATCH_DELAY = int(0.75 * self.FPS)  # frames to show mismatch

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_TILE_HIDDEN = (90, 105, 120)
        self.COLOR_TILE_REVEALED = (0, 150, 255)
        self.COLOR_TILE_MATCHED = (40, 180, 130)
        self.COLOR_TILE_MISMATCH = (220, 50, 50)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_UI_BG = (40, 50, 60, 180) # RGBA for transparency
        self.SYMBOL_COLORS = [
            (255, 87, 34), (255, 193, 7), (139, 195, 74), (0, 188, 212),
            (33, 150, 243), (103, 58, 183), (233, 30, 99), (158, 158, 158)
        ]

        # Grid layout
        self.grid_area_size = 320
        self.tile_size = self.grid_area_size // self.GRID_SIZE
        self.tile_margin = 8
        self.grid_offset_x = (self.screen_width - self.grid_area_size) // 2
        self.grid_offset_y = (self.screen_height - self.grid_area_size) + 20
        
        # State variables are initialized in reset()
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = self.INITIAL_TIME
        
        # Grid and tile state
        symbols = list(range(self.NUM_PAIRS)) * 2
        self.np_random.shuffle(symbols)
        self.grid_symbols = np.array(symbols).reshape((self.GRID_SIZE, self.GRID_SIZE))
        self.grid_states = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int) # 0:hidden, 1:revealed, 2:matched
        
        # Cursor and interaction state
        self.cursor_pos = [0, 0] # row, col
        self.last_space_held = False
        self.revealed_tiles = [] # Stores (r, c) of up to 2 revealed tiles
        self.mismatch_info = {'timer': 0, 'tiles': []}
        
        # Effects
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            
            # 1. Handle player input
            reward += self._handle_input(movement, space_held)
            
            # 2. Update game logic
            self._update_game_state()
            
            # 3. Update timer and step count
            self.timer = max(0, self.timer - 1.0 / self.FPS)
            self.steps += 1
        
        # 4. Check for termination conditions
        matched_pairs = np.sum(self.grid_states == 2) // 2
        if not self.game_over and matched_pairs == self.NUM_PAIRS:
            self.win = True
            self.game_over = True
            reward += 100 # Goal-oriented reward
            self.score += 1000

        terminated = self.game_over or self.timer <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Time ran out
            self.game_over = True
            # No penalty for timeout, just absence of win reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        reward = 0
        
        # Movement (action[0])
        if movement != 0:
            prev_pos = list(self.cursor_pos)
            if movement == 1: self.cursor_pos[0] -= 1 # Up
            elif movement == 2: self.cursor_pos[0] += 1 # Down
            elif movement == 3: self.cursor_pos[1] -= 1 # Left
            elif movement == 4: self.cursor_pos[1] += 1 # Right
            
            # Wrap around grid
            self.cursor_pos[0] %= self.GRID_SIZE
            self.cursor_pos[1] %= self.GRID_SIZE

            # Small penalty for moving onto an already solved tile
            if self.grid_states[self.cursor_pos[0], self.cursor_pos[1]] in [1, 2]:
                reward -= 0.01

        # Reveal tile (action[1]) - on rising edge of space press
        is_space_press = space_held and not self.last_space_held
        if is_space_press:
            reward += self._reveal_tile_at_cursor()
        self.last_space_held = space_held
        
        return reward

    def _reveal_tile_at_cursor(self):
        r, c = self.cursor_pos
        
        # Can't reveal if a mismatch is being shown, or tile is already matched/revealed
        if self.mismatch_info['timer'] > 0 or self.grid_states[r, c] != 0:
            return 0
        
        # Can't reveal more than 2 tiles at a time
        if len(self.revealed_tiles) >= 2:
            return 0

        # Reveal the tile
        # SFX: Tile flip
        self.grid_states[r, c] = 1 # 1: revealed
        if (r, c) not in self.revealed_tiles:
            self.revealed_tiles.append((r, c))
        
        # If two tiles are now revealed, check for match
        if len(self.revealed_tiles) == 2:
            r1, c1 = self.revealed_tiles[0]
            r2, c2 = self.revealed_tiles[1]
            
            symbol1 = self.grid_symbols[r1, c1]
            symbol2 = self.grid_symbols[r2, c2]
            
            if symbol1 == symbol2:
                # Match found
                # SFX: Match success
                self.grid_states[r1, c1] = 2 # 2: matched
                self.grid_states[r2, c2] = 2
                self.revealed_tiles.clear()
                self.score += 100
                self._create_particles(r1, c1)
                self._create_particles(r2, c2)
                return 10.0 # Match reward
            else:
                # Mismatch
                # SFX: Mismatch fail
                self.mismatch_info['timer'] = self.MISMATCH_DELAY
                self.mismatch_info['tiles'] = list(self.revealed_tiles)
                self.revealed_tiles.clear()
                self.score -= 10
                return -1.0 # Mismatch penalty
        
        return 0.1 # Reward for revealing one tile

    def _update_game_state(self):
        # Mismatch timer
        if self.mismatch_info['timer'] > 0:
            self.mismatch_info['timer'] -= 1
            if self.mismatch_info['timer'] == 0:
                for r, c in self.mismatch_info['tiles']:
                    if self.grid_states[r, c] == 1: # Ensure it hasn't been matched by a bug
                        self.grid_states[r, c] = 0 # Hide tile
                self.mismatch_info['tiles'].clear()

        # Particle physics
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles first
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['x']), int(p['y']), int(p['size']),
                (*p['color'], alpha)
            )

        # Draw grid tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_rect = pygame.Rect(
                    self.grid_offset_x + c * self.tile_size,
                    self.grid_offset_y + r * self.tile_size,
                    self.tile_size, self.tile_size
                ).inflate(-self.tile_margin, -self.tile_margin)

                state = self.grid_states[r, c]
                color = self.COLOR_TILE_HIDDEN
                
                is_mismatch_tile = (r, c) in self.mismatch_info['tiles'] and self.mismatch_info['timer'] > 0
                
                if is_mismatch_tile:
                    # Flash red for mismatch
                    flash_progress = self.mismatch_info['timer'] / self.MISMATCH_DELAY
                    if (self.mismatch_info['timer'] // 4) % 2 == 0:
                        color = self.COLOR_TILE_MISMATCH
                    else:
                        color = self.COLOR_TILE_REVEALED
                elif state == 1:
                    color = self.COLOR_TILE_REVEALED
                elif state == 2:
                    color = self.COLOR_TILE_MATCHED

                pygame.draw.rect(self.screen, color, tile_rect, border_radius=8)

                # Draw symbol if not hidden
                if state in [1, 2] or is_mismatch_tile:
                    symbol_id = self.grid_symbols[r, c]
                    self._render_symbol(self.screen, symbol_id, tile_rect)
        
        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cursor_c * self.tile_size,
            self.grid_offset_y + cursor_r * self.tile_size,
            self.tile_size, self.tile_size
        ).inflate(-self.tile_margin / 2, -self.tile_margin / 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=10)

    def _render_symbol(self, surface, symbol_id, rect):
        center_x, center_y = rect.center
        size = rect.width * 0.35
        color = self.SYMBOL_COLORS[symbol_id % len(self.SYMBOL_COLORS)]
        
        stype = symbol_id % 8 # Use modulo to be safe
        
        if stype == 0: # Circle
            pygame.gfxdraw.aacircle(surface, center_x, center_y, int(size), color)
            pygame.gfxdraw.filled_circle(surface, center_x, center_y, int(size), color)
        elif stype == 1: # Square
            s_rect = pygame.Rect(0, 0, size*2, size*2)
            s_rect.center = rect.center
            pygame.draw.rect(surface, color, s_rect)
        elif stype == 2: # Triangle
            points = [
                (center_x, center_y - size),
                (center_x - size, center_y + size * 0.7),
                (center_x + size, center_y + size * 0.7)
            ]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        elif stype == 3: # Diamond
            points = [
                (center_x, center_y - size), (center_x + size, center_y),
                (center_x, center_y + size), (center_x - size, center_y)
            ]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        elif stype == 4: # X
            pygame.draw.line(surface, color, (center_x - size, center_y - size), (center_x + size, center_y + size), 6)
            pygame.draw.line(surface, color, (center_x - size, center_y + size), (center_x + size, center_y - size), 6)
        elif stype == 5: # Star
            num_points = 5
            outer_radius = size
            inner_radius = size * 0.5
            points = []
            for i in range(num_points * 2):
                angle = math.pi / num_points * i - math.pi / 2
                radius = outer_radius if i % 2 == 0 else inner_radius
                points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        elif stype == 6: # Pentagon
            num_points = 5
            radius = size
            points = []
            for i in range(num_points):
                angle = 2 * math.pi / num_points * i - math.pi / 2
                points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        elif stype == 7: # Plus
            pygame.draw.line(surface, color, (center_x, center_y - size), (center_x, center_y + size), 6)
            pygame.draw.line(surface, color, (center_x - size, center_y), (center_x + size, center_y), 6)

    def _render_ui(self):
        # UI Background
        ui_panel = pygame.Surface((self.screen_width, 60), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 60), (self.screen_width, 60))

        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Timer
        time_str = f"{int(self.timer // 60):02}:{int(self.timer % 60):02}"
        time_color = self.COLOR_TEXT if self.timer > 10 else self.COLOR_TILE_MISMATCH
        time_text = self.font_medium.render(time_str, True, time_color)
        time_rect = time_text.get_rect(centerx=self.screen_width / 2, centery=30)
        self.screen.blit(time_text, time_rect)

        # Matched Pairs
        matched_pairs = np.sum(self.grid_states == 2) // 2
        pairs_text = self.font_medium.render(f"{matched_pairs}/{self.NUM_PAIRS}", True, self.COLOR_TEXT)
        pairs_rect = pairs_text.get_rect(right=self.screen_width - 20, centery=30)
        self.screen.blit(pairs_text, pairs_rect)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "TIME'S UP!"
            msg_text = self.font_large.render(message, True, self.COLOR_CURSOR if self.win else self.COLOR_TEXT)
            msg_rect = msg_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "matched_pairs": np.sum(self.grid_states == 2) // 2
        }

    def _create_particles(self, r, c):
        # SFX: Particle burst
        center_x = self.grid_offset_x + c * self.tile_size + self.tile_size / 2
        center_y = self.grid_offset_y + r * self.tile_size + self.tile_size / 2
        symbol_id = self.grid_symbols[r, c]
        color = self.SYMBOL_COLORS[symbol_id % len(self.SYMBOL_COLORS)]
        
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': center_x,
                'y': center_y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(self.FPS // 2, self.FPS),
                'max_life': self.FPS,
                'size': self.np_random.uniform(2, 5),
                'color': color
            })
    
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