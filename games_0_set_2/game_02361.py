
# Generated: 2025-08-28T04:36:17.414685
# Source Brief: brief_02361.md
# Brief Index: 2361

        
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
        "Controls: ↑↓←→ to move the cursor. Space to select the first tile, Shift to select the second and swap."
    )

    game_description = (
        "Swap adjacent colored tiles in a grid to create matches of 3 or more, aiming for a high score before running out of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.NUM_COLORS = 6
        self.TILE_SIZE = 40
        self.GAP_SIZE = 4
        self.GRID_START_X = (self.SCREEN_WIDTH - (self.GRID_COLS * (self.TILE_SIZE + self.GAP_SIZE) - self.GAP_SIZE)) // 2
        self.GRID_START_Y = (self.SCREEN_HEIGHT - (self.GRID_ROWS * (self.TILE_SIZE + self.GAP_SIZE) - self.GAP_SIZE)) // 2

        self.MAX_MOVES = 20
        self.WIN_SCORE = 1000
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_BG = (40, 50, 60)
        self.COLOR_UI_TEXT = (220, 220, 230)
        self.TILE_COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 150, 255),   # Blue
            (255, 255, 80),   # Yellow
            (200, 80, 255),   # Purple
            (255, 150, 50),   # Orange
        ]
        self.SELECTOR_COLOR = (255, 255, 255)
        self.SELECTION_COLOR = (255, 255, 0)

        # Animation constants
        self.SWAP_FRAMES = 6
        self.FALL_FRAMES = 8
        self.MATCH_FRAMES = 10

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables are initialized in reset()
        self.grid = None
        self.selector_pos = None
        self.first_selection = None
        self.steps = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.win = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.game_state = None
        self.animation_timer = None
        self.swapping_tiles = None
        self.matched_tiles = None
        self.falling_tiles = None
        self.particles = None
        self.reward_buffer = None
        self.cascade_count = None

        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False

        self.selector_pos = np.array([self.GRID_ROWS // 2, self.GRID_COLS // 2])
        self.first_selection = None
        self.prev_space_held = False
        self.prev_shift_held = False

        self.game_state = 'IDLE' # 'IDLE', 'SWAP', 'MATCH', 'FALL', 'INVALID_SWAP'
        self.animation_timer = 0
        self.swapping_tiles = {}
        self.matched_tiles = set()
        self.falling_tiles = {}
        self.particles = []
        self.reward_buffer = 0
        self.cascade_count = 0

        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Update animations if any are running
        if self.game_state != 'IDLE':
            self._update_animations()
        # If animations just finished, process the consequences
        else:
            reward += self.reward_buffer
            self.reward_buffer = 0

            # Check for win/loss conditions after a move resolves
            if self.score >= self.WIN_SCORE:
                self.game_over = True
                self.win = True
                reward += 100
            elif self.moves_left <= 0:
                self.game_over = True
                reward -= 10
            elif not self._find_possible_moves():
                self.game_over = True
                reward -= 10 # No moves left is a loss
            
            if self.game_over:
                terminated = True
            else:
                 # Only process new input if the board is idle
                self._handle_input(action)
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        # --- Movement ---
        if movement == 1: self.selector_pos[0] -= 1  # Up
        elif movement == 2: self.selector_pos[0] += 1  # Down
        elif movement == 3: self.selector_pos[1] -= 1  # Left
        elif movement == 4: self.selector_pos[1] += 1  # Right
        self.selector_pos[0] = np.clip(self.selector_pos[0], 0, self.GRID_ROWS - 1)
        self.selector_pos[1] = np.clip(self.selector_pos[1], 0, self.GRID_COLS - 1)

        # --- Space (Select first tile) ---
        if space_press:
            pos = tuple(self.selector_pos)
            if self.first_selection == pos:
                self.first_selection = None # Deselect
            else:
                self.first_selection = pos
                # sfx: select_tile

        # --- Shift (Select second tile & Swap) ---
        if shift_press and self.first_selection is not None:
            pos1 = self.first_selection
            pos2 = tuple(self.selector_pos)
            
            # Check for adjacency
            if pos1 != pos2 and abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1:
                self.moves_left -= 1
                self.swapping_tiles = {pos1: pos2, pos2: pos1}
                
                # Check if swap is valid (creates a match)
                if self._check_swap_validity(pos1, pos2):
                    self.game_state = 'SWAP'
                    self.animation_timer = self.SWAP_FRAMES
                    self.cascade_count = 0
                    # sfx: swap_start
                else:
                    self.game_state = 'INVALID_SWAP'
                    self.animation_timer = self.SWAP_FRAMES
                    self.reward_buffer -= 0.1
                    # sfx: invalid_swap

                self.first_selection = None # Reset selection after swap attempt
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_animations(self):
        self.animation_timer -= 1
        if self.animation_timer > 0:
            return

        if self.game_state == 'SWAP':
            pos1, pos2 = list(self.swapping_tiles.keys())
            self.grid[pos1], self.grid[pos2] = self.grid[pos2], self.grid[pos1]
            self.swapping_tiles = {}
            self._process_matches()

        elif self.game_state == 'INVALID_SWAP':
            self.swapping_tiles = {}
            self.game_state = 'IDLE'

        elif self.game_state == 'MATCH':
            self._handle_cleared_tiles()
            # sfx: tiles_fall
        
        elif self.game_state == 'FALL':
            self._apply_fall()
            self._process_matches()

    def _process_matches(self):
        matches = self._find_all_matches()
        if matches:
            if self.cascade_count > 0:
                self.reward_buffer += 5 # Cascade bonus
            self.cascade_count += 1

            self.matched_tiles = matches
            self.game_state = 'MATCH'
            self.animation_timer = self.MATCH_FRAMES
            
            # Add reward and spawn particles
            for r, c in matches:
                self.reward_buffer += 1
                self._spawn_particles(r, c, self.grid[r, c])
            # sfx: match_success
        else:
            self.game_state = 'IDLE'

    def _handle_cleared_tiles(self):
        if not self.matched_tiles:
            self.game_state = 'IDLE'
            return
        
        cols_to_update = sorted(list(set(c for r, c in self.matched_tiles)))
        
        for c in cols_to_update:
            col_tiles = [self.grid[r, c] for r in range(self.GRID_ROWS)]
            cleared_in_col = sorted([r for r, c_ in self.matched_tiles if c_ == c], reverse=True)
            
            for r in cleared_in_col:
                col_tiles.pop(r)

            new_tiles_count = self.GRID_ROWS - len(col_tiles)
            new_tiles = self.np_random.integers(1, self.NUM_COLORS + 1, size=new_tiles_count).tolist()
            
            final_col = new_tiles + col_tiles

            # Set up falling animation data
            current_row = self.GRID_ROWS - 1
            for r_idx in range(self.GRID_ROWS - 1, -1, -1):
                old_r = r_idx - new_tiles_count
                while (old_r, c) in self.matched_tiles:
                    old_r -= 1

                if old_r >= 0:
                    self.falling_tiles[(r_idx, c)] = (old_r, c)
                else: # New tile
                    self.falling_tiles[(r_idx, c)] = (r_idx - self.GRID_ROWS, c) # Start off-screen
                
                self.grid[r_idx, c] = final_col[r_idx]

        self.matched_tiles = set()
        self.game_state = 'FALL'
        self.animation_timer = self.FALL_FRAMES

    def _apply_fall(self):
        self.falling_tiles = {}
        # The grid is already correct, this just finishes the animation state
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    # --- Generation and Logic Helpers ---

    def _generate_board(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_ROWS, self.GRID_COLS))
        while self._find_all_matches() or not self._find_possible_moves():
            matches = self._find_all_matches()
            while matches:
                for r, c in matches:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)
                matches = self._find_all_matches()

    def _find_all_matches(self, grid=None):
        if grid is None:
            grid = self.grid
        
        matches = set()
        rows, cols = grid.shape
        # Check rows
        for r in range(rows):
            for c in range(cols - 2):
                if grid[r, c] == grid[r, c + 1] == grid[r, c + 2] != 0:
                    matches.update([(r, c), (r, c + 1), (r, c + 2)])
        # Check columns
        for c in range(cols):
            for r in range(rows - 2):
                if grid[r, c] == grid[r + 1, c] == grid[r + 2, c] != 0:
                    matches.update([(r, c), (r + 1, c), (r + 2, c)])
        return matches

    def _check_swap_validity(self, pos1, pos2):
        temp_grid = self.grid.copy()
        temp_grid[pos1], temp_grid[pos2] = temp_grid[pos2], temp_grid[pos1]
        return bool(self._find_all_matches(grid=temp_grid))

    def _find_possible_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Check swap right
                if c < self.GRID_COLS - 1:
                    if self._check_swap_validity((r, c), (r, c + 1)):
                        return True
                # Check swap down
                if r < self.GRID_ROWS - 1:
                    if self._check_swap_validity((r, c), (r + 1, c)):
                        return True
        return False

    def _spawn_particles(self, r, c, color_idx):
        center_x, center_y = self._get_pixel_pos(r, c, center=True)
        color = self.TILE_COLORS[color_idx - 1]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(10, 20)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'lifetime': lifetime, 'color': color})

    # --- Rendering Helpers ---

    def _get_pixel_pos(self, r, c, center=False):
        x = self.GRID_START_X + c * (self.TILE_SIZE + self.GAP_SIZE)
        y = self.GRID_START_Y + r * (self.TILE_SIZE + self.GAP_SIZE)
        if center:
            x += self.TILE_SIZE // 2
            y += self.TILE_SIZE // 2
        return x, y
    
    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(
            self.GRID_START_X - self.GAP_SIZE * 2,
            self.GRID_START_Y - self.GAP_SIZE * 2,
            self.GRID_COLS * (self.TILE_SIZE + self.GAP_SIZE) - self.GAP_SIZE + self.GAP_SIZE * 4,
            self.GRID_ROWS * (self.TILE_SIZE + self.GAP_SIZE) - self.GAP_SIZE + self.GAP_SIZE * 4
        )
        pygame.gfxdraw.box(self.screen, grid_rect, self.COLOR_GRID_BG)
        
        # Draw tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == 0 or (r, c) in self.matched_tiles:
                    continue
                
                x, y = self._get_pixel_pos(r, c)
                
                # Handle animations
                anim_progress = 0
                if self.game_state in ['SWAP', 'INVALID_SWAP'] and (r,c) in self.swapping_tiles:
                    anim_progress = (self.SWAP_FRAMES - self.animation_timer) / self.SWAP_FRAMES
                    target_r, target_c = self.swapping_tiles[(r, c)]
                    target_x, target_y = self._get_pixel_pos(target_r, target_c)
                    x = int(x + (target_x - x) * anim_progress)
                    y = int(y + (target_y - y) * anim_progress)
                    if self.game_state == 'INVALID_SWAP': # Shake effect
                        angle = anim_progress * math.pi * 2
                        x += int(math.sin(angle) * 4)

                elif self.game_state == 'FALL' and (r,c) in self.falling_tiles:
                    anim_progress = (self.FALL_FRAMES - self.animation_timer) / self.FALL_FRAMES
                    start_r, start_c = self.falling_tiles[(r,c)]
                    start_x, start_y = self._get_pixel_pos(start_r, start_c)
                    y = int(start_y + (y - start_y) * anim_progress)

                self._draw_tile(self.screen, x, y, self.grid[r, c])

        # Draw matching animation
        if self.game_state == 'MATCH':
            progress = self.animation_timer / self.MATCH_FRAMES
            for r, c in self.matched_tiles:
                x, y = self._get_pixel_pos(r, c)
                self._draw_tile(self.screen, x, y, self.grid[r, c], scale=progress, alpha=int(progress * 255))
        
        # Draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
            else:
                radius = int((p['lifetime'] / 20) * 5)
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])

        # Draw selectors
        if not self.game_over:
            # First selection highlight
            if self.first_selection:
                r, c = self.first_selection
                x, y = self._get_pixel_pos(r, c)
                rect = pygame.Rect(x - 2, y - 2, self.TILE_SIZE + 4, self.TILE_SIZE + 4)
                pygame.draw.rect(self.screen, self.SELECTION_COLOR, rect, 3, border_radius=8)
            
            # Cursor
            r, c = self.selector_pos
            x, y = self._get_pixel_pos(r, c)
            rect = pygame.Rect(x-3, y-3, self.TILE_SIZE + 6, self.TILE_SIZE + 6)
            pygame.draw.rect(self.screen, self.SELECTOR_COLOR, rect, 2, border_radius=8)

    def _draw_tile(self, surface, x, y, color_idx, scale=1.0, alpha=255):
        if color_idx == 0: return
        color = self.TILE_COLORS[color_idx - 1]
        
        size = int(self.TILE_SIZE * scale)
        offset = (self.TILE_SIZE - size) // 2
        rect = pygame.Rect(x + offset, y + offset, size, size)
        
        if alpha < 255:
            temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.gfxdraw.box(temp_surf, temp_surf.get_rect(), (*color, alpha))
            pygame.gfxdraw.aacircle(temp_surf, size // 2, size // 2, size // 2 - 2, (*color, alpha))
            surface.blit(temp_surf, rect.topleft)
        else:
            pygame.gfxdraw.box(surface, rect, color)
            # A simple shape for visual flair
            pygame.gfxdraw.filled_circle(surface, rect.centerx, rect.centery, int(size * 0.3), (255, 255, 255, 60))
            pygame.gfxdraw.aacircle(surface, rect.centerx, rect.centery, int(size * 0.3), (255, 255, 255, 120))


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Moves
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            msg_text = self.font_msg.render(msg, True, self.SELECTION_COLOR if self.win else self.TILE_COLORS[0])
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_text, msg_rect)

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
        
        print("✓ Implementation validated successfully")