# Generated: 2025-08-27T19:51:57.889915
# Source Brief: brief_02276.md
# Brief Index: 2276

        
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


# Ensure Pygame runs headless
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to rotate clockwise, ↓ to rotate counter-clockwise. "
        "Hold Space to soft drop, press Shift to hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Rotate and drop blocks to clear lines and score points. "
        "Survive for 90 seconds to win, but don't let the stack reach the top!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT = 10, 20
    BLOCK_SIZE = 18
    PLAYFIELD_PX_WIDTH = PLAYFIELD_WIDTH * BLOCK_SIZE
    PLAYFIELD_PX_HEIGHT = PLAYFIELD_HEIGHT * BLOCK_SIZE
    
    TOP_MARGIN = (SCREEN_HEIGHT - PLAYFIELD_PX_HEIGHT) // 2
    LEFT_MARGIN = (SCREEN_WIDTH - PLAYFIELD_PX_WIDTH) // 2 - 80

    NEXT_BOX_X = LEFT_MARGIN + PLAYFIELD_PX_WIDTH + 40
    NEXT_BOX_Y = TOP_MARGIN
    NEXT_BOX_W = BLOCK_SIZE * 5
    NEXT_BOX_H = BLOCK_SIZE * 5

    FPS = 30
    GAME_DURATION_SECONDS = 90
    TOTAL_GAME_FRAMES = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_BOUNDARY = (200, 200, 220)
    COLOR_TEXT = (220, 220, 230)
    COLOR_GAMEOVER = (255, 50, 50)
    COLOR_WIN = (50, 255, 50)

    PIECE_COLORS = [
        (0, 240, 240),  # I - Cyan
        (240, 240, 0),  # O - Yellow
        (160, 0, 240),  # T - Purple
        (0, 0, 240),    # J - Blue
        (240, 160, 0),  # L - Orange
        (0, 240, 0),    # S - Green
        (240, 0, 0),    # Z - Red
    ]

    TETROMINOS = {
        'I': [
            [(0, 1), (1, 1), (2, 1), (3, 1)],
            [(2, 0), (2, 1), (2, 2), (2, 3)],
            [(0, 2), (1, 2), (2, 2), (3, 2)],
            [(1, 0), (1, 1), (1, 2), (1, 3)]
        ],
        'O': [
            [(1, 0), (2, 0), (1, 1), (2, 1)]
        ],
        'T': [
            [(0, 1), (1, 1), (2, 1), (1, 0)],
            [(1, 0), (1, 1), (1, 2), (2, 1)],
            [(0, 1), (1, 1), (2, 1), (1, 2)],
            [(1, 0), (1, 1), (1, 2), (0, 1)]
        ],
        'J': [
            [(0, 0), (0, 1), (1, 1), (2, 1)],
            [(1, 0), (2, 0), (1, 1), (1, 2)],
            [(0, 1), (1, 1), (2, 1), (2, 2)],
            [(1, 0), (1, 1), (1, 2), (0, 2)]
        ],
        'L': [
            [(2, 0), (0, 1), (1, 1), (2, 1)],
            [(1, 0), (1, 1), (1, 2), (2, 2)],
            [(0, 1), (1, 1), (2, 1), (0, 2)],
            [(0, 0), (1, 0), (1, 1), (1, 2)]
        ],
        'S': [
            [(1, 0), (2, 0), (0, 1), (1, 1)],
            [(1, 0), (1, 1), (2, 1), (2, 2)],
            [(1, 1), (2, 1), (0, 2), (1, 2)],
            [(0, 0), (0, 1), (1, 1), (1, 2)]
        ],
        'Z': [
            [(0, 0), (1, 0), (1, 1), (2, 1)],
            [(2, 0), (1, 1), (2, 1), (1, 2)],
            [(0, 1), (1, 1), (1, 2), (2, 2)],
            [(1, 0), (0, 1), (1, 1), (0, 2)]
        ]
    }
    
    SHAPES = list(TETROMINOS.keys())

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
        
        self.render_mode = render_mode

        # Etc...        
        
        # Initialize state variables
        # FIX: Initialize grid here to prevent TypeError during validation
        self.grid = np.zeros((self.PLAYFIELD_HEIGHT, self.PLAYFIELD_WIDTH), dtype=int)
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.steps = 0
        self.game_timer = 0
        self.game_over = False
        self.reward_this_step = 0
        self.fall_progress = 0.0
        self.fall_speed = 0.0
        self.lines_to_clear = []
        self.clear_animation_timer = 0
        self.lock_timer = 0
        self.lock_delay_frames = self.FPS // 2  # 0.5 second lock delay
        self.das_delay = self.FPS // 6 # Delay before auto-repeat
        self.das_repeat = self.FPS // 20 # Auto-repeat speed
        self.das_timer = 0
        self.last_horiz_move_action = 0
        self.last_rot_action = 0
        self.soft_drop_active = False
        self.bag = []

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.grid = np.zeros((self.PLAYFIELD_HEIGHT, self.PLAYFIELD_WIDTH), dtype=int)
        
        self.bag = self.SHAPES[:]
        random.shuffle(self.bag)
        
        self.next_piece = None
        self._spawn_piece()
        self._spawn_piece() # First one becomes current, second becomes next

        self.steps = 0
        self.score = 0
        self.game_timer = self.TOTAL_GAME_FRAMES
        self.game_over = False
        self.reward_this_step = 0
        self.fall_progress = 0.0
        self.fall_speed = (self.BLOCK_SIZE / self.FPS) * 1.0 # 1 block per second
        
        self.lines_to_clear = []
        self.clear_animation_timer = 0
        self.lock_timer = 0
        self.das_timer = 0
        self.last_horiz_move_action = 0
        self.last_rot_action = 0

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _spawn_piece(self):
        self.current_piece = self.next_piece
        
        if not self.bag:
            self.bag = self.SHAPES[:]
            random.shuffle(self.bag)
        
        shape_type = self.bag.pop(0)
        
        self.next_piece = {
            'shape': shape_type,
            'rotation': 0,
            'x': self.PLAYFIELD_WIDTH // 2 - 2,
            'y': 0,
            'color_idx': self.SHAPES.index(shape_type) + 1
        }
        
        if self.current_piece and self._check_collision(self.current_piece):
            self.game_over = True

    def _get_piece_coords(self, piece):
        shape_coords = self.TETROMINOS[piece['shape']][piece['rotation'] % len(self.TETROMINOS[piece['shape']])]
        return [(int(piece['x'] + x), int(piece['y'] + y)) for x, y in shape_coords]

    def _check_collision(self, piece):
        coords = self._get_piece_coords(piece)
        for x, y in coords:
            if not (0 <= x < self.PLAYFIELD_WIDTH and 0 <= y < self.PLAYFIELD_HEIGHT):
                return True  # Wall collision
            if y >= 0 and self.grid[y, x] > 0:
                return True  # Grid collision
        return False

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Update game logic
        self.reward_this_step = -0.01 # Time penalty
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Handle line clear animation
        if self.clear_animation_timer > 0:
            self.clear_animation_timer -= 1
            if self.clear_animation_timer == 0:
                self._remove_cleared_lines()
        else:
            self._handle_input(action)
            self._update_piece_fall()

        self.game_timer -= 1
        self.steps += 1
        
        # Update difficulty
        if self.game_timer == self.TOTAL_GAME_FRAMES * 2 // 3 or self.game_timer == self.TOTAL_GAME_FRAMES // 3:
            self.fall_speed += (self.BLOCK_SIZE / self.FPS) * 0.25 # Increase speed by 0.25 blocks/sec

        terminated = self._check_termination()
        if terminated and not self.game_over and self.game_timer <= 0: # Survived
            self.reward_this_step += 100
            
        reward = self._calculate_reward()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Hard drop (Shift) - takes precedence
        if shift_held:
            # sfx: hard_drop_sound
            if self.current_piece:
                temp_piece = self.current_piece.copy()
                while not self._check_collision(temp_piece):
                    self.current_piece['y'] = temp_piece['y']
                    temp_piece['y'] += 1
                self._lock_and_spawn()
            return # End turn after hard drop

        # Horizontal Movement (Left/Right) with DAS
        horiz_move = 0
        if movement == 3: horiz_move = -1
        if movement == 4: horiz_move = 1

        if horiz_move != 0:
            if self.last_horiz_move_action != movement: # New press
                self._move(horiz_move)
                self.das_timer = self.das_delay
            else: # Held
                self.das_timer -= 1
                if self.das_timer <= 0:
                    self._move(horiz_move)
                    self.das_timer = self.das_repeat # Use faster repeat delay
        self.last_horiz_move_action = movement

        # Rotation (Up/Down)
        rot_dir = 0
        if movement == 1: rot_dir = 1  # Clockwise
        if movement == 2: rot_dir = -1 # Counter-clockwise

        if rot_dir != 0 and self.last_rot_action != movement:
            self._rotate(rot_dir)
        self.last_rot_action = movement
        
        # Soft drop (Space)
        self.soft_drop_active = space_held

    def _move(self, dx):
        if not self.current_piece: return
        test_piece = self.current_piece.copy()
        test_piece['x'] += dx
        if not self._check_collision(test_piece):
            # sfx: move_sound
            self.current_piece['x'] = test_piece['x']
            self.lock_timer = 0 # Reset lock timer on successful move

    def _rotate(self, dr):
        if not self.current_piece: return
        test_piece = self.current_piece.copy()
        test_piece['rotation'] += dr
        
        # Simple wall kick: try to shift left/right if rotation fails
        if self._check_collision(test_piece):
            for kick in [-1, 1, -2, 2]: # Try kicking
                kicked_piece = test_piece.copy()
                kicked_piece['x'] += kick
                if not self._check_collision(kicked_piece):
                    # sfx: rotate_sound
                    self.current_piece = kicked_piece
                    self.lock_timer = 0 # Reset lock timer
                    return
        else:
            # sfx: rotate_sound
            self.current_piece['rotation'] = test_piece['rotation']
            self.lock_timer = 0 # Reset lock timer

    def _update_piece_fall(self):
        if not self.current_piece: return
        is_on_ground = self._is_on_ground()

        if is_on_ground:
            if self.lock_timer == 0:
                self.lock_timer = self.lock_delay_frames
            else:
                self.lock_timer -= 1
                if self.lock_timer <= 0:
                    self._lock_and_spawn()
                    return
        else:
            self.lock_timer = 0
            
            soft_drop_multiplier = 10 if self.soft_drop_active else 1
            self.fall_progress += self.fall_speed * soft_drop_multiplier

            if self.fall_progress >= 1.0:
                dy = math.floor(self.fall_progress)
                self.fall_progress -= dy
                
                for _ in range(dy):
                    self.current_piece['y'] += 1
                    if self._check_collision(self.current_piece):
                        self.current_piece['y'] -= 1
                        # sfx: land_sound
                        if self.lock_timer == 0:
                           self.lock_timer = self.lock_delay_frames
                        break

    def _is_on_ground(self):
        if not self.current_piece: return False
        test_piece = self.current_piece.copy()
        test_piece['y'] += 1
        return self._check_collision(test_piece)

    def _lock_and_spawn(self):
        self._lock_piece()
        lines_cleared = self._check_and_start_clear_animation()
        if lines_cleared == 0:
            self._spawn_piece()
        # sfx: lock_piece_sound

    def _lock_piece(self):
        if not self.current_piece: return
        coords = self._get_piece_coords(self.current_piece)
        hole_penalty = 0
        for x, y in coords:
            if 0 <= x < self.PLAYFIELD_WIDTH and 0 <= y < self.PLAYFIELD_HEIGHT:
                self.grid[y, x] = self.current_piece['color_idx']
                # Check for created holes/overhangs
                if y + 1 < self.PLAYFIELD_HEIGHT and self.grid[y + 1, x] == 0:
                    hole_penalty += 1
        
        self.reward_this_step -= hole_penalty
        self.current_piece = None

    def _check_and_start_clear_animation(self):
        self.lines_to_clear = []
        for r in range(self.PLAYFIELD_HEIGHT):
            if np.all(self.grid[r, :] > 0):
                self.lines_to_clear.append(r)
        
        if self.lines_to_clear:
            # sfx: line_clear_sound
            self.clear_animation_timer = self.FPS // 5 # 0.2 second animation
            num_lines = len(self.lines_to_clear)
            # Reward: 10 per line, with a bonus for more lines (tetris)
            self.reward_this_step += (10 * num_lines) * num_lines 
            self.score += (100 * num_lines) * num_lines
        return len(self.lines_to_clear)

    def _remove_cleared_lines(self):
        if not self.lines_to_clear: return

        self.lines_to_clear.sort(reverse=True)
        for r in self.lines_to_clear:
            self.grid[1:r+1, :] = self.grid[0:r, :]
            self.grid[0, :] = 0
        
        self.lines_to_clear = []
        self._spawn_piece()

    def _calculate_reward(self):
        return self.reward_this_step

    def _check_termination(self):
        if self.game_over:
            return True
        if self.game_timer <= 0:
            return True
        return False

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
        # Draw playfield boundary and grid
        boundary_rect = pygame.Rect(self.LEFT_MARGIN - 2, self.TOP_MARGIN - 2, self.PLAYFIELD_PX_WIDTH + 4, self.PLAYFIELD_PX_HEIGHT + 4)
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, boundary_rect, 2, border_radius=3)
        for x in range(self.PLAYFIELD_WIDTH):
            for y in range(self.PLAYFIELD_HEIGHT):
                rect = pygame.Rect(self.LEFT_MARGIN + x * self.BLOCK_SIZE, self.TOP_MARGIN + y * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw landed blocks
        for y in range(self.PLAYFIELD_HEIGHT):
            for x in range(self.PLAYFIELD_WIDTH):
                if self.grid[y, x] > 0:
                    self._draw_block(x, y, self.grid[y, x])
        
        # Draw current piece and ghost piece
        if self.current_piece and not self.game_over:
            # Ghost piece
            ghost_piece = self.current_piece.copy()
            while not self._check_collision(ghost_piece):
                ghost_piece['y'] += 1
            ghost_piece['y'] -= 1
            ghost_coords = self._get_piece_coords(ghost_piece)
            for x, y in ghost_coords:
                self._draw_block(x, y, self.current_piece['color_idx'], is_ghost=True)

            # Current piece
            coords = self._get_piece_coords(self.current_piece)
            for x, y in coords:
                self._draw_block(x, y, self.current_piece['color_idx'])

        # Draw line clear animation
        if self.clear_animation_timer > 0:
            for r in self.lines_to_clear:
                rect = pygame.Rect(self.LEFT_MARGIN, self.TOP_MARGIN + r * self.BLOCK_SIZE, self.PLAYFIELD_PX_WIDTH, self.BLOCK_SIZE)
                flash_color = (255, 255, 255) if (self.clear_animation_timer // 2) % 2 == 0 else self.COLOR_BG
                pygame.draw.rect(self.screen, flash_color, rect)

    def _draw_block(self, grid_x, grid_y, color_idx, is_ghost=False):
        px, py = self.LEFT_MARGIN + grid_x * self.BLOCK_SIZE, self.TOP_MARGIN + grid_y * self.BLOCK_SIZE
        color = self.PIECE_COLORS[color_idx - 1]
        
        if is_ghost:
            rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
            pygame.draw.rect(self.screen, color, rect, 2, border_radius=2)
            return

        outer_rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
        inner_rect = pygame.Rect(px + 2, py + 2, self.BLOCK_SIZE - 4, self.BLOCK_SIZE - 4)
        
        light_color = tuple(min(255, c + 60) for c in color)
        dark_color = tuple(max(0, c - 60) for c in color)
        
        pygame.draw.rect(self.screen, dark_color, outer_rect, 0, border_radius=3)
        pygame.draw.rect(self.screen, color, inner_rect, 0, border_radius=2)
        
        # Highlight
        pygame.draw.line(self.screen, light_color, (px + 2, py + 2), (px + self.BLOCK_SIZE - 3, py + 2), 1)
        pygame.draw.line(self.screen, light_color, (px + 2, py + 2), (px + 2, py + self.BLOCK_SIZE - 3), 1)

    def _render_ui(self):
        # Score
        score_surf = self.font_medium.render(f"SCORE", True, self.COLOR_TEXT)
        score_val_surf = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (30, 30))
        self.screen.blit(score_val_surf, (30, 60))

        # Timer
        time_left = max(0, self.game_timer / self.FPS)
        time_surf = self.font_medium.render(f"TIME", True, self.COLOR_TEXT)
        time_val_surf = self.font_large.render(f"{time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - 120, 30))
        self.screen.blit(time_val_surf, (self.SCREEN_WIDTH - 120, 60))

        # Next Piece
        next_text = self.font_medium.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.NEXT_BOX_X, self.NEXT_BOX_Y - 30))
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.NEXT_BOX_X, self.NEXT_BOX_Y, self.NEXT_BOX_W, self.NEXT_BOX_H), 2, border_radius=3)
        if self.next_piece:
            shape_coords = self.TETROMINOS[self.next_piece['shape']][0]
            for x, y in shape_coords:
                px = self.NEXT_BOX_X + self.BLOCK_SIZE // 2 + x * self.BLOCK_SIZE
                py = self.NEXT_BOX_Y + self.BLOCK_SIZE // 2 + y * self.BLOCK_SIZE
                
                # Adjust for 'I' and 'O' piece centering
                if self.next_piece['shape'] == 'I': py += self.BLOCK_SIZE // 2
                if self.next_piece['shape'] == 'O': px -= self.BLOCK_SIZE // 2
                
                outer_rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
                inner_rect = pygame.Rect(px + 2, py + 2, self.BLOCK_SIZE - 4, self.BLOCK_SIZE - 4)
                color = self.PIECE_COLORS[self.next_piece['color_idx'] - 1]
                pygame.draw.rect(self.screen, color, outer_rect, 0, border_radius=3)
                pygame.draw.rect(self.screen, tuple(min(255, c + 60) for c in color), inner_rect, 0, border_radius=2)

        # Game Over / Win message
        if self.game_over:
            self._render_message("GAME OVER", self.COLOR_GAMEOVER)
        elif self.game_timer <= 0:
            self._render_message("YOU WIN!", self.COLOR_WIN)

    def _render_message(self, text, color):
        msg_surf = self.font_large.render(text, True, color)
        msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        bg_rect = msg_rect.inflate(20, 20)
        s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        s.fill((20, 20, 30, 200))
        self.screen.blit(s, bg_rect)
        pygame.draw.rect(self.screen, color, bg_rect, 2, border_radius=5)
        
        self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.game_timer / self.FPS),
        }
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Validating implementation...")
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
    # It will not be executed when the environment is used by Gymnasium
    
    # Unset the dummy video driver if you want to see the game window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Puzzle Fall")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("--- Playing Game ---")
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    # To handle single key presses for rotation
    key_up_pressed = False
    key_down_pressed = False

    while running:
        # --- Human Input to Action ---
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        
        # Continuous actions
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                # Discrete actions (rotation)
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
            
        action = [movement, space_held, shift_held]

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to Screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for reset key
            finished = True
            while finished:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        finished = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        print("--- Game Reset ---")
                        finished = False

        clock.tick(GameEnv.FPS)
        
    pygame.quit()