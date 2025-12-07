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
        "Use arrow keys to move the cursor. Press Space to pick up a pixel, "
        "and arrow keys to move it. Press Shift to drop a pixel."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target image by moving pixels on your grid. You have a limited number of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_DIM = 10
    CELL_SIZE = 30
    GRID_WIDTH = GRID_DIM * CELL_SIZE
    GRID_HEIGHT = GRID_DIM * CELL_SIZE
    MAX_MOVES = 20
    MIN_INCORRECT_START = 20

    # --- Colors ---
    COLOR_BG = (30, 30, 40)
    COLOR_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECT = (255, 255, 0)
    COLOR_FLASH = (255, 255, 255)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    
    PALETTE = [
        (255, 87, 51),   # Vermilion
        (255, 195, 0),   # Saffron
        (51, 187, 255),  # Capri
        (102, 255, 102), # Light Green
        (255, 102, 178), # Pink
        (178, 102, 255), # Lavender
    ]

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.target_grid = None
        self.player_grid = None
        self.cursor_pos = None
        self.selected_pixel_pos = None
        self.moves_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.move_flash_timer = 0
        self.last_move_from = None
        self.last_move_to = None
        self.np_random = np.random.default_rng()

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self._generate_grids()
        
        self.cursor_pos = np.array([self.GRID_DIM // 2, self.GRID_DIM // 2])
        self.selected_pixel_pos = None
        
        self.moves_remaining = self.MAX_MOVES
        self.score = 0.0
        self.steps = 0
        self.game_over = False
        self.win = False
        
        self.move_flash_timer = 0
        self.last_move_from = None
        self.last_move_to = None
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_grids(self):
        # Generate a simple, symmetric target grid
        self.target_grid = np.zeros((self.GRID_DIM, self.GRID_DIM, 3), dtype=np.uint8)
        center = self.GRID_DIM // 2
        color1_idx, color2_idx = self.np_random.choice(len(self.PALETTE), 2, replace=False)
        color1, color2 = self.PALETTE[color1_idx], self.PALETTE[color2_idx]

        for i in range(self.GRID_DIM):
            self.target_grid[i, center] = color1
            self.target_grid[center, i] = color1
        for i in range(center - 1, center + 2):
            for j in range(center - 1, center + 2):
                if (i, j) != (center, center):
                    self.target_grid[i, j] = color2

        # Create player grid by shuffling until at least MIN_INCORRECT_START pixels are wrong
        while True:
            flat_pixels = self.target_grid.reshape(-1, 3).copy()
            self.np_random.shuffle(flat_pixels, axis=0)
            self.player_grid = flat_pixels.reshape(self.GRID_DIM, self.GRID_DIM, 3)
            
            incorrect_pixels = np.sum(np.any(self.player_grid != self.target_grid, axis=2))
            if incorrect_pixels >= self.MIN_INCORRECT_START:
                break
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = 0.0
        moved_pixel = False

        # --- Action Logic ---
        # 1. Handle deselect (Shift)
        if shift_held and self.selected_pixel_pos is not None:
            self.selected_pixel_pos = None
            # sfx_deselect

        # 2. Handle select (Space)
        elif space_held and self.selected_pixel_pos is None:
            self.selected_pixel_pos = self.cursor_pos.copy()
            # sfx_select

        # 3. Handle movement (Arrows)
        elif movement != 0:
            direction = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}.get(movement)
            direction = np.array(direction)
            
            if self.selected_pixel_pos is not None:
                # Move the selected pixel
                start_pos = self.selected_pixel_pos.copy()
                end_pos = (start_pos + direction) % self.GRID_DIM
                
                # Swap pixels
                start_color = self.player_grid[start_pos[1], start_pos[0]].copy()
                end_color = self.player_grid[end_pos[1], end_pos[0]].copy()
                self.player_grid[start_pos[1], start_pos[0]] = end_color
                self.player_grid[end_pos[1], end_pos[0]] = start_color
                
                self.selected_pixel_pos = end_pos
                self.cursor_pos = end_pos.copy()

                self.moves_remaining -= 1
                moved_pixel = True
                
                self.move_flash_timer = 5
                self.last_move_from = start_pos
                self.last_move_to = end_pos
                # sfx_swap
            else:
                # Move the cursor
                self.cursor_pos = (self.cursor_pos + direction) % self.GRID_DIM
                # sfx_cursor_move

        # --- State and Reward Update ---
        if moved_pixel:
            correct_pixels = np.sum(np.all(self.player_grid == self.target_grid, axis=2))
            move_reward = float(correct_pixels) - 0.2
            reward += move_reward
            self.score += move_reward

        self.steps += 1
        
        # --- Termination Check ---
        self.win = np.array_equal(self.player_grid, self.target_grid)
        is_loss = self.moves_remaining <= 0
        terminated = self.win or is_loss

        if terminated:
            self.game_over = True
            if self.win:
                win_bonus = 100.0
                reward += win_bonus
                self.score += win_bonus
                # sfx_win
            else:
                # sfx_lose
                pass
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
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
        target_grid_pos = (30, 80)
        player_grid_pos = (self.SCREEN_WIDTH - self.GRID_WIDTH - 30, 80)

        self._render_grid(self.target_grid, target_grid_pos, desaturate_incorrect=False)
        self._render_grid(self.player_grid, player_grid_pos, desaturate_incorrect=True)
        self._render_cursor(player_grid_pos)

        if self.selected_pixel_pos is not None:
            self._render_selection(player_grid_pos)

        if self.move_flash_timer > 0:
            self._render_move_flash(player_grid_pos)
            self.move_flash_timer -= 1

    def _render_grid(self, grid_data, top_left, desaturate_incorrect):
        for y in range(self.GRID_DIM):
            for x in range(self.GRID_DIM):
                px, py = top_left[0] + x * self.CELL_SIZE, top_left[1] + y * self.CELL_SIZE
                rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
                
                # Convert numpy array to list of python ints for pygame
                color = grid_data[y, x].tolist()
                
                if desaturate_incorrect:
                    is_correct = np.array_equal(grid_data[y, x], self.target_grid[y, x])
                    if not is_correct:
                        color = self._desaturate(color, 0.3)
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0, 50), rect, 1)

    def _render_cursor(self, player_grid_pos):
        cx, cy = self.cursor_pos
        px = player_grid_pos[0] + cx * self.CELL_SIZE
        py = player_grid_pos[1] + cy * self.CELL_SIZE
        
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        cursor_color = (*self.COLOR_CURSOR, alpha)
        
        cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, cursor_color, cursor_surface.get_rect(), 4)
        self.screen.blit(cursor_surface, (px, py))

    def _render_selection(self, player_grid_pos):
        sx, sy = self.selected_pixel_pos
        px = player_grid_pos[0] + sx * self.CELL_SIZE
        py = player_grid_pos[1] + sy * self.CELL_SIZE
        rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, 4)

    def _render_move_flash(self, player_grid_pos):
        alpha = (self.move_flash_timer / 5) * 255
        flash_color = (*self.COLOR_FLASH, alpha)
        flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        flash_surface.fill(flash_color)
        
        for pos in [self.last_move_from, self.last_move_to]:
            if pos is not None:
                px = player_grid_pos[0] + pos[0] * self.CELL_SIZE
                py = player_grid_pos[1] + pos[1] * self.CELL_SIZE
                self.screen.blit(flash_surface, (px, py))

    def _render_ui(self):
        target_title = self.font_medium.render("TARGET", True, self.COLOR_TEXT)
        player_title = self.font_medium.render("YOUR GRID", True, self.COLOR_TEXT)
        self.screen.blit(target_title, (30 + self.GRID_WIDTH // 2 - target_title.get_width() // 2, 40))
        self.screen.blit(player_title, (self.SCREEN_WIDTH - 30 - self.GRID_WIDTH // 2 - player_title.get_width() // 2, 40))

        moves_text = self.font_medium.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH // 2 - moves_text.get_width() // 2, 10))
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 35))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text_str = "PERFECT MATCH!" if self.win else "OUT OF MOVES"
            end_text_color = self.COLOR_WIN if self.win else self.COLOR_LOSE
            end_text = self.font_large.render(end_text_str, True, end_text_color)
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        correct_pixels = np.sum(np.all(self.player_grid == self.target_grid, axis=2))
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "correct_pixels": int(correct_pixels),
            "is_win": self.win,
        }

    def _desaturate(self, color, factor):
        h, s, l, a = pygame.Color(*color).hsla
        s = max(0, s * factor)
        desaturated_color = pygame.Color(0, 0, 0)
        desaturated_color.hsla = (int(h), int(s), int(l), int(a))
        return (desaturated_color.r, desaturated_color.g, desaturated_color.b)

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
        
        # print("✓ Implementation validated successfully")