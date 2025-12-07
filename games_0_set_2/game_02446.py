
# Generated: 2025-08-27T20:23:47.450420
# Source Brief: brief_02446.md
# Brief Index: 2446

        
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
        "Controls: ↑/↓ to rotate, ←/→ to move. Press space to hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade puzzler. Place falling blocks to clear lines and score big."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAY_WIDTH, PLAY_HEIGHT = 200, 400  # 10 blocks wide, 20 blocks high
    BLOCK_SIZE = 20
    GRID_TOP_LEFT_X = (SCREEN_WIDTH - PLAY_WIDTH) // 2 - 100
    GRID_TOP_LEFT_Y = 0

    # --- Colors ---
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_BG = (35, 35, 45)
    COLOR_UI_HEADER = (180, 180, 200)
    COLOR_UI_VALUE = (255, 255, 255)
    COLOR_CLEAR_ANIM = (240, 240, 240)

    # --- Tetromino Shapes ---
    SHAPES = {
        'S': [[[0, 1, 1], [1, 1, 0], [0, 0, 0]], [[0, 1, 0], [0, 1, 1], [0, 0, 1]]],
        'Z': [[[1, 1, 0], [0, 1, 1], [0, 0, 0]], [[0, 0, 1], [0, 1, 1], [0, 1, 0]]],
        'I': [[[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]],
        'O': [[[1, 1], [1, 1]]],
        'J': [[[0, 1, 0], [0, 1, 0], [1, 1, 0]], [[1, 0, 0], [1, 1, 1], [0, 0, 0]], [[0, 1, 1], [0, 1, 0], [0, 1, 0]], [[0, 0, 0], [1, 1, 1], [0, 0, 1]]],
        'L': [[[0, 1, 0], [0, 1, 0], [0, 1, 1]], [[0, 0, 0], [1, 1, 1], [1, 0, 0]], [[1, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 1], [0, 0, 0]]],
        'T': [[[0, 1, 0], [1, 1, 1], [0, 0, 0]], [[0, 1, 0], [0, 1, 1], [0, 1, 0]], [[0, 0, 0], [1, 1, 1], [0, 1, 0]], [[0, 1, 0], [1, 1, 0], [0, 1, 0]]]
    }
    SHAPE_COLORS = {
        'S': (0, 255, 127),   # SpringGreen
        'Z': (255, 69, 0),    # OrangeRed
        'I': (0, 191, 255),   # DeepSkyBlue
        'O': (255, 215, 0),   # Gold
        'J': (0, 0, 255),     # Blue
        'L': (255, 165, 0),   # Orange
        'T': (128, 0, 128)    # Purple
    }

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
        
        self.font_header = pygame.font.Font(None, 28)
        self.font_value = pygame.font.Font(None, 36)

        self.game_state_attrs = [
            "steps", "score", "game_over", "board", "lines_cleared", "level",
            "current_block", "next_block", "fall_time", "fall_speed",
            "clear_animation_timer", "lines_being_cleared"
        ]
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.board = np.zeros((20, 10), dtype=int)
        self.shape_keys = list(self.SHAPES.keys())
        
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.game_over = False
        
        self.fall_speed = 1.0  # seconds per grid cell
        self.fall_time = 0

        self.clear_animation_timer = 0
        self.lines_being_cleared = []

        self.next_block = self._get_new_block()
        self.current_block = self._get_new_block()
        self.current_block['y'] = self._get_spawn_y()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for every step to encourage speed
        self.steps += 1
        
        if self.clear_animation_timer > 0:
            self.clear_animation_timer -= 1
            if self.clear_animation_timer == 0:
                self._finish_line_clear()
            # Game is paused during animation, only return observation
            return self._get_observation(), 0, self.game_over, False, self._get_info()

        movement = action[0]
        hard_drop = action[1] == 1
        
        # --- Action Handling ---
        if not self.game_over:
            if hard_drop:
                # Sound: Hard drop thud
                drop_dist = self._hard_drop()
                reward += drop_dist * 0.02 # Small reward for dropping
                self._place_block()
            else:
                # 1. Handle Horizontal Movement
                if movement == 3:  # Left
                    self.current_block['x'] -= 1
                    if not self._is_valid_position():
                        self.current_block['x'] += 1
                elif movement == 4:  # Right
                    self.current_block['x'] += 1
                    if not self._is_valid_position():
                        self.current_block['x'] -= 1

                # 2. Handle Rotation
                if movement == 1:  # Up -> Rotate Clockwise
                    self._rotate_block(1)
                elif movement == 2:  # Down -> Rotate Counter-Clockwise
                    self._rotate_block(-1)

                # 3. Handle Gravity
                self.fall_time += 1 / 30.0 # Assuming 30fps call rate
                if self.fall_time >= self.fall_speed:
                    self.fall_time = 0
                    self.current_block['y'] += 1
                    if not self._is_valid_position():
                        self.current_block['y'] -= 1
                        self._place_block() # Landed
        
        # --- Termination ---
        terminated = self.game_over or self.steps >= 10000
        if self.game_over:
            reward -= 10
        if self.lines_cleared >= 100 and not self.game_over:
            reward += 100
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Game Logic Helpers ---

    def _get_new_block(self):
        shape_key = self.np_random.choice(self.shape_keys)
        shape = self.SHAPES[shape_key]
        color_idx = self.shape_keys.index(shape_key) + 1
        return {
            'x': (10 - len(shape[0][0])) // 2,
            'y': 0,
            'shape_key': shape_key,
            'rotation': 0,
            'color_idx': color_idx
        }
    
    def _get_spawn_y(self):
        shape = self.SHAPES[self.current_block['shape_key']][self.current_block['rotation']]
        for r, row in enumerate(shape):
            if any(row):
                return -r
        return 0

    def _is_valid_position(self, block=None, pos=None):
        if block is None:
            block = self.current_block
        if pos is None:
            pos = (block['x'], block['y'])

        shape_to_check = self.SHAPES[block['shape_key']][block['rotation']]
        for r, row in enumerate(shape_to_check):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = pos[0] + c, pos[1] + r
                    if not (0 <= grid_x < 10 and 0 <= grid_y < 20 and self.board[grid_y][grid_x] == 0):
                        return False
        return True

    def _rotate_block(self, direction):
        # Sound: Rotation swish
        initial_rotation = self.current_block['rotation']
        num_rotations = len(self.SHAPES[self.current_block['shape_key']])
        self.current_block['rotation'] = (initial_rotation + direction) % num_rotations
        
        if not self._is_valid_position():
            # Wall kick logic
            for offset in [1, -1, 2, -2]:
                self.current_block['x'] += offset
                if self._is_valid_position():
                    return
                self.current_block['x'] -= offset
            # If all kicks fail, revert rotation
            self.current_block['rotation'] = initial_rotation

    def _hard_drop(self):
        original_y = self.current_block['y']
        while self._is_valid_position():
            self.current_block['y'] += 1
        self.current_block['y'] -= 1
        return self.current_block['y'] - original_y

    def _place_block(self):
        # Sound: Block placement click
        shape = self.SHAPES[self.current_block['shape_key']][self.current_block['rotation']]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = self.current_block['x'] + c
                    grid_y = self.current_block['y'] + r
                    if 0 <= grid_y < 20:
                        self.board[grid_y][grid_x] = self.current_block['color_idx']

        self.score += 1 # Base score for placing a block
        cleared = self._check_and_clear_lines()
        
        if not cleared:
            self._spawn_next_block()

    def _check_and_clear_lines(self):
        lines_to_clear = [r for r, row in enumerate(self.board) if all(row)]
        if lines_to_clear:
            # Sound: Line clear chime
            self.lines_being_cleared = lines_to_clear
            self.clear_animation_timer = 5 # Animate for 5 frames
            
            # --- Reward Calculation ---
            num_cleared = len(lines_to_clear)
            reward_map = {1: 10, 2: 30, 3: 60, 4: 100}
            self.score += reward_map.get(num_cleared, 0)
            
            self.lines_cleared += num_cleared
            
            # Update level and speed
            new_level = 1 + self.lines_cleared // 10
            if new_level > self.level:
                self.level = new_level
                self.fall_speed = max(0.1, 1.0 - (self.level - 1) * 0.05)
                # Sound: Level up fanfare
            return True
        return False
    
    def _finish_line_clear(self):
        for r in sorted(self.lines_being_cleared, reverse=True):
            self.board = np.delete(self.board, r, axis=0)
            new_row = np.zeros((1, 10), dtype=int)
            self.board = np.insert(self.board, 0, new_row, axis=0)
        self.lines_being_cleared = []
        self._spawn_next_block()

    def _spawn_next_block(self):
        self.current_block = self.next_block
        self.current_block['y'] = self._get_spawn_y()
        self.next_block = self._get_new_block()
        
        if not self._is_valid_position():
            self.game_over = True
            # Sound: Game over sound

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(21):
            y = self.GRID_TOP_LEFT_Y + r * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_TOP_LEFT_X, y), (self.GRID_TOP_LEFT_X + self.PLAY_WIDTH, y))
        for c in range(11):
            x = self.GRID_TOP_LEFT_X + c * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_TOP_LEFT_Y), (x, self.GRID_TOP_LEFT_Y + self.PLAY_HEIGHT))

        # Draw placed blocks
        for r, row in enumerate(self.board):
            for c, color_idx in enumerate(row):
                if color_idx > 0:
                    color = self.SHAPE_COLORS[self.shape_keys[color_idx-1]]
                    self._draw_block(c, r, color)
        
        # Draw line clear animation
        if self.clear_animation_timer > 0:
            for r in self.lines_being_cleared:
                for c in range(10):
                    self._draw_block(c, r, self.COLOR_CLEAR_ANIM, is_flash=True)

        if not self.game_over:
            # Draw ghost piece
            ghost_y = self.current_block['y']
            while self._is_valid_position(pos=(self.current_block['x'], ghost_y + 1)):
                ghost_y += 1
            self._draw_piece(self.current_block, (self.current_block['x'], ghost_y), ghost=True)

            # Draw current piece
            self._draw_piece(self.current_block, (self.current_block['x'], self.current_block['y']))

    def _draw_block(self, grid_x, grid_y, color, is_flash=False):
        x, y = self.GRID_TOP_LEFT_X + grid_x * self.BLOCK_SIZE, self.GRID_TOP_LEFT_Y + grid_y * self.BLOCK_SIZE
        rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        if is_flash:
            pygame.draw.rect(self.screen, color, rect)
            return

        # Main color fill
        pygame.draw.rect(self.screen, color, rect)
        
        # 3D effect
        light_color = tuple(min(255, c + 40) for c in color)
        dark_color = tuple(max(0, c - 40) for c in color)
        
        pygame.draw.line(self.screen, light_color, (x, y), (x + self.BLOCK_SIZE - 1, y), 2)
        pygame.draw.line(self.screen, light_color, (x, y), (x, y + self.BLOCK_SIZE - 1), 2)
        pygame.draw.line(self.screen, dark_color, (x + 1, y + self.BLOCK_SIZE - 1), (x + self.BLOCK_SIZE - 1, y + self.BLOCK_SIZE - 1), 2)
        pygame.draw.line(self.screen, dark_color, (x + self.BLOCK_SIZE - 1, y + 1), (x + self.BLOCK_SIZE - 1, y + self.BLOCK_SIZE - 1), 2)

    def _draw_piece(self, piece, pos, ghost=False):
        shape = self.SHAPES[piece['shape_key']][piece['rotation']]
        color = self.SHAPE_COLORS[piece['shape_key']]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = pos[0] + c, pos[1] + r
                    if grid_y >= 0:
                        if ghost:
                            x, y = self.GRID_TOP_LEFT_X + grid_x * self.BLOCK_SIZE, self.GRID_TOP_LEFT_Y + grid_y * self.BLOCK_SIZE
                            rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)
                            pygame.draw.rect(self.screen, color, rect, 2)
                        else:
                            self._draw_block(grid_x, grid_y, color)

    def _render_ui(self):
        ui_x = self.GRID_TOP_LEFT_X + self.PLAY_WIDTH + 30
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (ui_x - 15, 15, 180, 370), border_radius=10)

        # Score
        self._draw_ui_text("SCORE", (ui_x, 30))
        self._draw_ui_text(f"{self.score}", (ui_x, 55), value=True)
        
        # Lines
        self._draw_ui_text("LINES", (ui_x, 110))
        self._draw_ui_text(f"{self.lines_cleared}", (ui_x, 135), value=True)

        # Level
        self._draw_ui_text("LEVEL", (ui_x, 190))
        self._draw_ui_text(f"{self.level}", (ui_x, 215), value=True)

        # Next Piece
        self._draw_ui_text("NEXT", (ui_x, 270))
        next_shape = self.SHAPES[self.next_block['shape_key']][0]
        next_color = self.SHAPE_COLORS[self.next_block['shape_key']]
        shape_w = len(next_shape[0]) * self.BLOCK_SIZE
        shape_h = len(next_shape) * self.BLOCK_SIZE
        start_x = ui_x + (150 - shape_w) / 2
        start_y = 305 + (70 - shape_h) / 2
        
        for r, row in enumerate(next_shape):
            for c, cell in enumerate(row):
                if cell:
                    x = start_x + c * self.BLOCK_SIZE
                    y = start_y + r * self.BLOCK_SIZE
                    rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(self.screen, next_color, rect)

        if self.game_over:
            overlay = pygame.Surface((self.PLAY_WIDTH, self.PLAY_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (self.GRID_TOP_LEFT_X, self.GRID_TOP_LEFT_Y))
            
            game_over_text = self.font_value.render("GAME OVER", True, (255, 50, 50))
            text_rect = game_over_text.get_rect(center=(self.GRID_TOP_LEFT_X + self.PLAY_WIDTH / 2, self.GRID_TOP_LEFT_Y + self.PLAY_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def _draw_ui_text(self, text, pos, value=False):
        font = self.font_value if value else self.font_header
        color = self.COLOR_UI_VALUE if value else self.COLOR_UI_HEADER
        surface = font.render(text, True, color)
        self.screen.blit(surface, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "level": self.level
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        
        # Test game-specific assertions
        self.reset()
        self.lines_cleared = 9
        self.board[-1, :] = 1 # Fill bottom row
        self.step(self.action_space.sample()) # Step to process clear
        self.step(self.action_space.sample()) # Step to process clear
        self.step(self.action_space.sample()) # Step to process clear
        self.step(self.action_space.sample()) # Step to process clear
        self.step(self.action_space.sample()) # Step to process clear
        self.step(self.action_space.sample()) # Step to process clear
        
        assert self.lines_cleared == 10
        assert self.level == 2
        assert math.isclose(self.fall_speed, 0.95)
        
        self.reset()
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Manual Controls ---
    # Arrows for movement/rotation
    # Space for hard drop
    # Q to quit

    action = np.array([0, 0, 0]) # no-op, no space, no shift
    
    # Create a window to display the game
    pygame.display.set_caption("Puzzle Game")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while not terminated:
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True
        
        # --- Map keyboard to action space ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Unused
        
        action = np.array([movement, space_held, shift_held])

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != -0.01:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")

        # --- Render the observation ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # The game logic advances per step, not per frame.
        # To make it playable, we add a small delay.
        clock.tick(15) # Limit to 15 actions per second for human playability

    env.close()
    print("Game Over!")
    print(f"Final Score: {info['score']}, Lines Cleared: {info['lines_cleared']}")