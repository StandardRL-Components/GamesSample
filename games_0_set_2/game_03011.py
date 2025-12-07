import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A fast-paced, grid-based puzzle game where the player manipulates falling
    blocks to clear lines and achieve a target score before the stack reaches the top.
    This environment prioritizes visual quality and satisfying gameplay feel.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑↓ to rotate. Hold space for soft drop, tap shift for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear lines by fitting falling blocks together. Score points and combos, but don't let the stack reach the top!"
    )

    # Frames auto-advance at 30fps for smooth, real-time gameplay.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    BLOCK_SIZE = 18
    
    # Play area positioning
    PLAY_AREA_X = (SCREEN_WIDTH - GRID_WIDTH * BLOCK_SIZE) // 2
    PLAY_AREA_Y = (SCREEN_HEIGHT - GRID_HEIGHT * BLOCK_SIZE) // 2
    
    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 50)
    COLOR_WHITE = (240, 240, 240)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_GAMEOVER = (200, 50, 50)
    
    # Tetromino shapes and colors
    PIECE_SHAPES = [
        [[1, 1, 1, 1]],  # I
        [[1, 1, 0], [0, 1, 1]],  # Z
        [[0, 1, 1], [1, 1, 0]],  # S
        [[1, 1, 1], [0, 1, 0]],  # T
        [[1, 1, 1], [1, 0, 0]],  # L
        [[1, 1, 1], [0, 0, 1]],  # J
        [[1, 1], [1, 1]],  # O
    ]
    PIECE_COLORS = [
        (0, 240, 240),  # I (Cyan)
        (240, 0, 0),    # Z (Red)
        (0, 240, 0),    # S (Green)
        (160, 0, 240),  # T (Purple)
        (240, 160, 0),  # L (Orange)
        (0, 0, 240),    # J (Blue)
        (240, 240, 0),  # O (Yellow)
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
        self.font_main = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 18)
        self.font_title = pygame.font.SysFont("consolas", 48, bold=True)

        # Initialize state variables
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.combo = 0
        self.fall_timer = 0
        self.fall_speed = 30 # Frames per grid cell drop
        self.last_shift_held = False
        self.line_clear_animation = [] # Stores (row_index, timer)
        self.last_reward = 0
        
        # Initialize state
        # self.reset() is called by the wrapper, but we can call it here for standalone use
        # In this specific case, the failing test calls __init__ then reset, so we'll match that.
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.combo = 0
        self.fall_speed = 30 # 1 drop per second at 30fps
        self.fall_timer = 0
        self.last_shift_held = False
        self.line_clear_animation = []
        self.last_reward = 0
        
        self._spawn_piece() # Spawns next_piece
        self._spawn_piece() # Spawns current_piece, moves old next_piece
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed
        self.steps += 1

        if not self.game_over:
            self._handle_action(action)
            reward += self._update_game_state()
            
        terminated = self._check_termination()
        if terminated and not self.game_over: # Win condition
            reward += 100
            self.game_over = True # To show win message
        elif terminated and self.game_over: # Lose condition
            reward -= 100

        self.last_reward = reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement and Rotation ---
        if movement == 1: self._rotate_piece(clockwise=True)  # Up -> Rotate CW
        if movement == 2: self._rotate_piece(clockwise=False) # Down -> Rotate CCW
        if movement == 3: self._move_piece(-1, 0) # Left
        if movement == 4: self._move_piece(1, 0)  # Right
        
        # --- Soft Drop ---
        if space_held:
            self.fall_timer += self.fall_speed * 0.7 # Speed up fall significantly
        
        # --- Hard Drop (on key press, not hold) ---
        if shift_held and not self.last_shift_held:
            # Sound: Hard drop thud
            while self._move_piece(0, 1):
                self.score += 2 # Small bonus for hard dropping
            self.fall_timer = self.fall_speed # Force lock
        self.last_shift_held = shift_held

    def _update_game_state(self):
        reward = 0
        self.fall_timer += 1
        
        if self.fall_timer >= self.fall_speed:
            self.fall_timer = 0
            if not self._move_piece(0, 1): # Try to move down
                # If it can't move down, lock it
                self._lock_piece()
                
                cleared = self._clear_lines()
                if cleared > 0:
                    # Sound: Line clear
                    reward += cleared * (cleared * 0.5) # 1->0.5, 2->2, 3->4.5, 4->8
                    reward += self.combo # Add combo bonus
                    self.score += (100 * cleared * self.combo) + (50 * cleared)
                    self.lines_cleared += cleared
                    self.combo += 1
                    
                    # Increase difficulty every 2 lines
                    if self.lines_cleared % 2 == 0:
                        self.fall_speed = max(5, self.fall_speed * 0.95)
                else:
                    reward -= 0.2 # Penalty for placement without a clear
                    self.combo = 0 # Reset combo
                    
                self._spawn_piece()
                if not self._is_valid_position(self.current_piece['shape'], self.current_piece['pos']):
                    self.game_over = True
                    # Sound: Game over
        return reward

    def _spawn_piece(self):
        if self.next_piece is None:
            self.next_piece = self._new_random_piece()
            
        self.current_piece = self.next_piece
        self.next_piece = self._new_random_piece()
        self.current_piece['pos'] = [self.GRID_WIDTH // 2 - 1, 0]

    def _new_random_piece(self):
        idx = self.np_random.integers(0, len(self.PIECE_SHAPES))
        return {'shape': self.PIECE_SHAPES[idx], 'color_idx': idx, 'pos': [0,0]}

    def _is_valid_position(self, shape, pos):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_c, grid_r = pos[0] + c, pos[1] + r
                    if not (0 <= grid_c < self.GRID_WIDTH and 0 <= grid_r < self.GRID_HEIGHT):
                        return False
                    if self.grid[grid_r, grid_c] != 0:
                        return False
        return True

    def _move_piece(self, dx, dy):
        new_pos = [self.current_piece['pos'][0] + dx, self.current_piece['pos'][1] + dy]
        if self._is_valid_position(self.current_piece['shape'], new_pos):
            self.current_piece['pos'] = new_pos
            return True
        return False

    def _rotate_piece(self, clockwise=True):
        shape = self.current_piece['shape']
        if clockwise:
            new_shape = [list(row) for row in zip(*shape[::-1])]
        else:
            new_shape = [list(row) for row in zip(*shape)][::-1]
        
        # --- Wall Kick Logic ---
        original_pos = self.current_piece['pos']
        for dc in [0, -1, 1, -2, 2]: # Standard wall kick offsets
            new_pos = [original_pos[0] + dc, original_pos[1]]
            if self._is_valid_position(new_shape, new_pos):
                self.current_piece['shape'] = new_shape
                self.current_piece['pos'] = new_pos
                # Sound: Rotate click
                return
    
    def _lock_piece(self):
        # Sound: Block lock
        shape = self.current_piece['shape']
        pos = self.current_piece['pos']
        color_idx = self.current_piece['color_idx'] + 1 # Use 1-based index for grid
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self.grid[pos[1] + r, pos[0] + c] = color_idx
    
    def _clear_lines(self):
        lines_to_clear = [r for r, row in enumerate(self.grid) if np.all(row)]
        if not lines_to_clear:
            return 0
            
        for r in lines_to_clear:
            self.line_clear_animation.append([r, 10]) # 10 frames of animation

        # Create new grid and copy non-cleared lines down
        new_grid = np.zeros_like(self.grid)
        new_row = self.GRID_HEIGHT - 1
        for r in range(self.GRID_HEIGHT - 1, -1, -1):
            if r not in lines_to_clear:
                new_grid[new_row] = self.grid[r]
                new_row -= 1
        self.grid = new_grid
        
        return len(lines_to_clear)

    def _check_termination(self):
        return self.game_over or self.lines_cleared >= 10 or self.steps >= 1000

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Grid ---
        for r in range(self.GRID_HEIGHT + 1):
            y = self.PLAY_AREA_Y + r * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAY_AREA_X, y), (self.PLAY_AREA_X + self.GRID_WIDTH * self.BLOCK_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.PLAY_AREA_X + c * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.PLAY_AREA_Y), (x, self.PLAY_AREA_Y + self.GRID_HEIGHT * self.BLOCK_SIZE))
        
        # --- Draw Placed Blocks ---
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    self._draw_block(c, r, self.grid[r, c] - 1)

        if not self.game_over:
            # --- Draw Ghost Piece ---
            ghost_pos = list(self.current_piece['pos'])
            temp_piece = {'shape': self.current_piece['shape'], 'pos': ghost_pos, 'color_idx': self.current_piece['color_idx']}
            while self._is_valid_position(temp_piece['shape'], [temp_piece['pos'][0], temp_piece['pos'][1] + 1]):
                temp_piece['pos'][1] += 1
            self._draw_piece(temp_piece, ghost=True)

            # --- Draw Current Piece ---
            self._draw_piece(self.current_piece)

        # --- Line Clear Animation ---
        new_anim_list = []
        for r, timer in self.line_clear_animation:
            rect = pygame.Rect(self.PLAY_AREA_X, self.PLAY_AREA_Y + r * self.BLOCK_SIZE, self.GRID_WIDTH * self.BLOCK_SIZE, self.BLOCK_SIZE)
            alpha = int(255 * (timer / 10))
            flash_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, rect.topleft)
            if timer > 0:
                new_anim_list.append([r, timer - 1])
        self.line_clear_animation = new_anim_list

    def _draw_piece(self, piece, ghost=False):
        shape = piece['shape']
        pos = piece['pos']
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_block(pos[0] + c, pos[1] + r, piece['color_idx'], ghost)

    def _draw_block(self, c, r, color_idx, ghost=False):
        x = self.PLAY_AREA_X + c * self.BLOCK_SIZE
        y = self.PLAY_AREA_Y + r * self.BLOCK_SIZE
        color = self.PIECE_COLORS[color_idx]
        
        rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)

        if ghost:
            # Draw an outline for the ghost piece
            pygame.gfxdraw.rectangle(self.screen, rect, (*color, 100))
        else:
            # Draw a filled, slightly shaded block for a 3D feel
            darker_color = tuple(max(0, val - 50) for val in color)
            lighter_color = tuple(min(255, val + 50) for val in color)
            
            pygame.gfxdraw.box(self.screen, rect, color)
            
            # Top and left highlight
            pygame.draw.line(self.screen, lighter_color, (x, y), (x + self.BLOCK_SIZE - 1, y))
            pygame.draw.line(self.screen, lighter_color, (x, y), (x, y + self.BLOCK_SIZE - 1))
            
            # Bottom and right shadow
            pygame.draw.line(self.screen, darker_color, (x + self.BLOCK_SIZE - 1, y), (x + self.BLOCK_SIZE - 1, y + self.BLOCK_SIZE - 1))
            pygame.draw.line(self.screen, darker_color, (x, y + self.BLOCK_SIZE - 1), (x + self.BLOCK_SIZE - 1, y + self.BLOCK_SIZE - 1))


    def _render_ui(self):
        # --- Helper for shadowed text ---
        def draw_text(text, font, color, x, y, shadow_color=self.COLOR_TEXT_SHADOW, center=False):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            text_rect = text_surf.get_rect()
            if center:
                text_rect.center = (x, y)
            else:
                text_rect.topleft = (x, y)
            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
            self.screen.blit(text_surf, text_rect)

        # --- Score and Lines Display ---
        right_panel_x = self.PLAY_AREA_X + self.GRID_WIDTH * self.BLOCK_SIZE + 30
        draw_text("SCORE", self.font_small, self.COLOR_TEXT, right_panel_x, 50)
        draw_text(f"{self.score:06d}", self.font_main, self.COLOR_WHITE, right_panel_x, 75)
        
        draw_text("LINES", self.font_small, self.COLOR_TEXT, right_panel_x, 125)
        draw_text(f"{self.lines_cleared}", self.font_main, self.COLOR_WHITE, right_panel_x, 150)

        if self.combo > 1:
            draw_text(f"COMBO x{self.combo}", self.font_main, self.PIECE_COLORS[self.combo % len(self.PIECE_COLORS)], right_panel_x, 200)

        # --- Next Piece Preview ---
        draw_text("NEXT", self.font_small, self.COLOR_TEXT, right_panel_x, 250)
        next_piece_surf = pygame.Surface((4 * self.BLOCK_SIZE, 4 * self.BLOCK_SIZE), pygame.SRCALPHA)
        if self.next_piece:
            shape = self.next_piece['shape']
            w, h = len(shape[0]), len(shape)
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        x = ((4 - w) / 2 + c) * self.BLOCK_SIZE
                        y = ((4 - h) / 2 + r) * self.BLOCK_SIZE
                        color = self.PIECE_COLORS[self.next_piece['color_idx']]
                        pygame.gfxdraw.box(next_piece_surf, (x, y, self.BLOCK_SIZE, self.BLOCK_SIZE), color)
        self.screen.blit(next_piece_surf, (right_panel_x, 275))
        
        # --- Game Over / Win Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.lines_cleared >= 10:
                draw_text("YOU WIN!", self.font_title, self.PIECE_COLORS[6], self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, center=True)
            else:
                draw_text("GAME OVER", self.font_title, self.COLOR_GAMEOVER, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "combo": self.combo,
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

# Example of how to run the environment
if __name__ == '__main__':
    # To run with a display, you might need to unset the SDL_VIDEODRIVER
    # e.g., run from the terminal: SDL_VIDEODRIVER=x11 python your_script_name.py
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    done = False
    
    try:
        # Set up a window to display the game
        pygame.display.set_caption("Gymnasium Block Puzzle")
        display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    except pygame.error as e:
        print(f"Could not create display: {e}")
        print("This is expected if you are running in a headless environment.")
        print("To play manually, run with a display driver, e.g.:")
        print("SDL_VIDEODRIVER=x11 python your_script.py")
        display_screen = None

    # Game loop for manual control
    while not done:
        action = [0, 0, 0] # [movement, space, shift]
        
        if display_screen:
            # Action mapping for human player
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
            # Pygame event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
        else: # No display, just run a random agent
            action = env.action_space.sample()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if display_screen:
            # Render the observation to the display window
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()

        # Control the frame rate
        env.clock.tick(30)
        
    env.close()
    print("Game Over!")
    print(f"Final Score: {info['score']}, Lines: {info['lines_cleared']}")