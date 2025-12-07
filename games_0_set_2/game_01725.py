
# Generated: 2025-08-28T02:31:01.183852
# Source Brief: brief_01725.md
# Brief Index: 1725

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # User-facing control string for the puzzle game
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a tile. "
        "Space again on a second tile to connect. Shift to deselect."
    )

    # User-facing description of the game
    game_description = (
        "A strategic puzzle game. Connect all matching colored tiles by creating wormholes. "
        "Plan your path carefully, as you only have a limited number of moves!"
    )

    # The game is turn-based; state is static until an action is received.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 8, 6
        self.MAX_MOVES = 20
        self.FONT_SIZE_UI = 24
        self.FONT_SIZE_MSG = 48

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_SELECTION = (255, 255, 255)
        self.COLOR_CONNECTED = (80, 90, 110)
        self.COLOR_TEXT = (220, 220, 240)
        self.TILE_COLORS = [
            (255, 87, 34),   # Deep Orange
            (33, 150, 243),  # Blue
            (76, 175, 80),   # Green
            (253, 216, 53),  # Yellow
            (156, 39, 176),  # Purple
            (0, 188, 212),   # Cyan
            (233, 30, 99),   # Pink
            (103, 58, 183),  # Deep Purple
        ]
        self.NUM_PAIRS_PER_COLOR = (self.GRID_COLS * self.GRID_ROWS) // (len(self.TILE_COLORS) * 2)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", self.FONT_SIZE_UI, bold=True)
        self.font_msg = pygame.font.SysFont("Arial", self.FONT_SIZE_MSG, bold=True)

        # --- Grid & Tile Sizing ---
        self.GRID_AREA_HEIGHT = self.HEIGHT - 50 # Reserve space for UI
        self.TILE_SIZE = min(self.WIDTH // self.GRID_COLS, self.GRID_AREA_HEIGHT // self.GRID_ROWS) - 4
        self.GRID_WIDTH_PX = self.GRID_COLS * (self.TILE_SIZE + 4)
        self.GRID_HEIGHT_PX = self.GRID_ROWS * (self.TILE_SIZE + 4)
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH_PX) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT_PX) // 2 + 20

        # --- Game State Initialization ---
        self.grid = None
        self.cursor_pos = None
        self.selected_tile_pos = None
        self.moves_remaining = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.total_pairs = None
        self.pairs_connected = None
        self.last_connection_effect = None
        self.np_random = None

        # Initialize state variables
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.cursor_pos = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
        self.selected_tile_pos = None
        self.moves_remaining = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_connection_effect = None
        self._generate_grid()

        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        self.total_pairs = (self.GRID_COLS * self.GRID_ROWS) // 2
        self.pairs_connected = 0
        
        colors = []
        for i in range(len(self.TILE_COLORS)):
            colors.extend([i] * self.NUM_PAIRS_PER_COLOR * 2)

        self.np_random.shuffle(colors)
        self.grid = np.array(colors).reshape((self.GRID_ROWS, self.GRID_COLS))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.last_connection_effect = None # Clear effect from previous step

        reward = self._handle_actions(action)
        self._check_termination()
        
        if self.game_over:
            if self.win:
                reward += 100  # Goal-oriented win bonus
                # sfx: game_win_fanfare
            else:
                reward += -50 # Goal-oriented loss penalty
                # sfx: game_over_sound

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # 1. Handle Deselection (highest priority)
        if shift_pressed and self.selected_tile_pos is not None:
            self.selected_tile_pos = None
            # sfx: deselect_sound

        # 2. Handle Selection/Connection
        elif space_pressed:
            cursor_y, cursor_x = self.cursor_pos
            
            # If no tile is selected, try to select one
            if self.selected_tile_pos is None:
                if self.grid[cursor_y, cursor_x] != -1: # Not already connected
                    self.selected_tile_pos = self.cursor_pos
                    # sfx: select_tile_1
            
            # If a tile is already selected, attempt a connection
            else:
                sel_y, sel_x = self.selected_tile_pos
                
                # Connecting to self is an invalid move
                if self.selected_tile_pos == self.cursor_pos:
                    reward -= 0.2
                    self.moves_remaining -= 1
                    self.selected_tile_pos = None # Deselect on invalid move
                    # sfx: error_buzz
                    return reward

                # This action consumes a move
                self.moves_remaining -= 1

                color1 = self.grid[sel_y, sel_x]
                color2 = self.grid[cursor_y, cursor_x]

                # Successful connection
                if color1 != -1 and color1 == color2:
                    reward += 5.0 # Event-based reward for a pair
                    self.score += 10
                    self.grid[sel_y, sel_x] = -1
                    self.grid[cursor_y, cursor_x] = -1
                    self.pairs_connected += 1
                    self.last_connection_effect = (self.selected_tile_pos, self.cursor_pos, self.TILE_COLORS[color1])
                    # sfx: successful_connection
                # Failed connection
                else:
                    reward -= 0.2
                    self.score -= 1
                    # sfx: failed_connection

                self.selected_tile_pos = None # Always deselect after an attempt

        # 3. Handle Movement
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1 # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1 # Right
        
        if dx != 0 or dy != 0:
            new_y = (self.cursor_pos[0] + dy) % self.GRID_ROWS
            new_x = (self.cursor_pos[1] + dx) % self.GRID_COLS
            self.cursor_pos = (new_y, new_x)
            # sfx: cursor_move

        return reward

    def _check_termination(self):
        if self.pairs_connected == self.total_pairs:
            self.game_over = True
            self.win = True
        elif self.moves_remaining <= 0:
            self.game_over = True
            self.win = False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_tiles()
        self._render_cursor_and_selection()
        if self.last_connection_effect:
            self._render_connection_effect()
        self._render_ui()
        if self.game_over:
            self._render_game_over_message()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for row in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + row * (self.TILE_SIZE + 4)
            start_pos = (self.GRID_OFFSET_X - 2, y - 2)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH_PX - 6, y - 2)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        for col in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + col * (self.TILE_SIZE + 4)
            start_pos = (x - 2, self.GRID_OFFSET_Y - 2)
            end_pos = (x - 2, self.GRID_OFFSET_Y + self.GRID_HEIGHT_PX - 6)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_tiles(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_idx = self.grid[r, c]
                color = self.COLOR_CONNECTED if color_idx == -1 else self.TILE_COLORS[color_idx]
                
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * (self.TILE_SIZE + 4),
                    self.GRID_OFFSET_Y + r * (self.TILE_SIZE + 4),
                    self.TILE_SIZE,
                    self.TILE_SIZE
                )
                pygame.draw.rect(self.screen, color, rect, border_radius=4)

                # Draw a small indicator on unconnected tiles
                if color_idx != -1:
                    pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, 3, (255, 255, 255, 50))
                    pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, 3, (255, 255, 255, 50))

    def _render_cursor_and_selection(self):
        # Render cursor
        cur_r, cur_c = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_OFFSET_X + cur_c * (self.TILE_SIZE + 4) - 3,
            self.GRID_OFFSET_Y + cur_r * (self.TILE_SIZE + 4) - 3,
            self.TILE_SIZE + 6,
            self.TILE_SIZE + 6
        )
        # Glow effect for cursor
        for i in range(4):
            alpha = 150 - i * 30
            pygame.draw.rect(self.screen, (*self.COLOR_CURSOR, alpha), rect.inflate(i*2, i*2), 1, border_radius=6)
        
        # Render selection
        if self.selected_tile_pos is not None:
            sel_r, sel_c = self.selected_tile_pos
            rect = pygame.Rect(
                self.GRID_OFFSET_X + sel_c * (self.TILE_SIZE + 4) - 2,
                self.GRID_OFFSET_Y + sel_r * (self.TILE_SIZE + 4) - 2,
                self.TILE_SIZE + 4,
                self.TILE_SIZE + 4
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTION, rect, 3, border_radius=6)

    def _render_connection_effect(self):
        pos1_rc, pos2_rc, color = self.last_connection_effect
        
        y1, x1 = pos1_rc
        center1 = (
            self.GRID_OFFSET_X + x1 * (self.TILE_SIZE + 4) + self.TILE_SIZE // 2,
            self.GRID_OFFSET_Y + y1 * (self.TILE_SIZE + 4) + self.TILE_SIZE // 2
        )
        
        y2, x2 = pos2_rc
        center2 = (
            self.GRID_OFFSET_X + x2 * (self.TILE_SIZE + 4) + self.TILE_SIZE // 2,
            self.GRID_OFFSET_Y + y2 * (self.TILE_SIZE + 4) + self.TILE_SIZE // 2
        )

        # Draw wormhole line
        pygame.draw.aaline(self.screen, color, center1, center2, 3)
        
        # Draw glowing endpoints
        for i in range(5, 0, -1):
            alpha = 255 - i * 40
            pygame.gfxdraw.aacircle(self.screen, center1[0], center1[1], self.TILE_SIZE // 4 + i, (*color, alpha))
            pygame.gfxdraw.aacircle(self.screen, center2[0], center2[1], self.TILE_SIZE // 4 + i, (*color, alpha))

    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_ui.render(f"MOVES: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 15, 10))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))
        
        # Pairs connected
        pairs_text = self.font_ui.render(f"PAIRS: {self.pairs_connected}/{self.total_pairs}", True, self.COLOR_TEXT)
        self.screen.blit(pairs_text, (self.WIDTH // 2 - pairs_text.get_width() // 2, 10))

    def _render_game_over_message(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        msg_text = "LEVEL CLEAR" if self.win else "OUT OF MOVES"
        color = (100, 255, 100) if self.win else (255, 100, 100)
        
        text_surf = self.font_msg.render(msg_text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "pairs_connected": self.pairs_connected,
            "total_pairs": self.total_pairs
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset(seed=random.randint(0, 1000))
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Wormhole Connect")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset(seed=random.randint(0, 1000))
                    terminated = False
                    continue
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            # Only step if an action was taken (for manual play)
            if any(a != 0 for a in action):
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()