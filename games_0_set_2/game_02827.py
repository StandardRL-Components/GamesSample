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
        "Controls: ←→ to move, ↑ to rotate, ↓ for soft drop. Space for hard drop, Shift to hold piece."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade puzzle game. Position falling blocks to clear lines and score points before the stack reaches the top. Game speed increases with your score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAYFIELD_W, PLAYFIELD_H = 10, 20
    BLOCK_SIZE = 18
    PLAYFIELD_X_OFFSET = (SCREEN_WIDTH - PLAYFIELD_W * BLOCK_SIZE) // 2 - 120
    PLAYFIELD_Y_OFFSET = (SCREEN_HEIGHT - PLAYFIELD_H * BLOCK_SIZE) // 2

    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_TEXT = (220, 220, 240)
    COLOR_PANEL_BG = (30, 30, 45)

    TETROMINO_SHAPES = {
        'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],
        'J': [[[1, 0, 0], [1, 1, 1]], [[1, 1], [1, 0], [1, 0]], [[1, 1, 1], [0, 0, 1]], [[0, 1], [0, 1], [1, 1]]],
        'L': [[[0, 0, 1], [1, 1, 1]], [[1, 0], [1, 0], [1, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]]],
        'O': [[[1, 1], [1, 1]]],
        'S': [[[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]]],
        'T': [[[0, 1, 0], [1, 1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1, 1], [0, 1, 0]], [[0, 1], [1, 1], [0, 1]]],
        'Z': [[[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]]]
    }

    TETROMINO_COLORS = {
        'I': (0, 240, 240), 'J': (0, 0, 240), 'L': (240, 160, 0),
        'O': (240, 240, 0), 'S': (0, 240, 0), 'T': (160, 0, 240),
        'Z': (240, 0, 0)
    }

    SCORE_TARGET = 5000
    MAX_STEPS = 10000

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_title = pygame.font.SysFont("Consolas", 48, bold=True)

        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.held_piece = None
        self.can_hold = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.fall_timer = None
        self.fall_delay = None
        self.last_action = None
        self.line_clear_effects = None
        self.tetris_effect_timer = None

        # self.reset() is called here, which requires self.np_random to be initialized.
        # Gymnasium's base __init__ doesn't set it up, so we call reset with a seed first.
        self.reset(seed=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = [[0 for _ in range(self.PLAYFIELD_W)] for _ in range(self.PLAYFIELD_H)]
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_delay = 0.8  # Initial fall speed in seconds per row
        self.fall_timer = 0
        self.last_action = self.action_space.sample() * 0  # Initialize with no action
        self.line_clear_effects = []
        self.tetris_effect_timer = 0

        self.current_piece = None
        self.next_piece = None
        self.held_piece = None
        self.can_hold = True

        self._create_new_piece()  # First one becomes current
        self._create_new_piece()  # Second one becomes next

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_action, shift_action = action[0], action[1], action[2]

        up_pressed = movement == 1 and self.last_action[0] != 1
        soft_drop = movement == 2
        left_action = movement == 3
        right_action = movement == 4
        space_pressed = space_action == 1 and self.last_action[1] != 1
        shift_pressed = shift_action == 1 and self.last_action[2] != 1

        self.last_action = action

        # --- Game Logic ---
        if up_pressed:
            self._rotate_piece()
        if left_action:
            self._move_horizontal(-1)
        if right_action:
            self._move_horizontal(1)

        if shift_pressed:
            if self._hold_piece():
                pass

        if space_pressed:
            reward += 0.1 * self._hard_drop()  # Reward for distance dropped
            self._lock_piece()
        else:
            # Normal fall and soft drop
            time_delta = 1 / 30.0  # Since auto_advance=True, we assume 30fps
            self.fall_timer += time_delta
            if soft_drop:
                self.fall_timer += time_delta * 5  # Accelerate fall
                reward += 0.01  # Small incentive for soft dropping

            if self.fall_timer >= self.fall_delay:
                self.fall_timer = 0
                if self._move_down():
                    reward += 0.1
                else:  # Piece has landed
                    self._lock_piece()

        # Update line clear effects
        self.line_clear_effects = [e for e in self.line_clear_effects if e['timer'] > 0]
        for effect in self.line_clear_effects:
            effect['timer'] -= 1
        if self.tetris_effect_timer > 0:
            self.tetris_effect_timer -= 1

        # --- Calculate Rewards from Locking ---
        if 'lines_cleared' in self.current_piece:
            lines = self.current_piece['lines_cleared']
            if lines > 0:
                line_scores = {1: 100, 2: 300, 3: 500, 4: 800}
                line_rewards = {1: 1, 2: 3, 3: 5, 4: 8}
                self.score += line_scores.get(lines, 0)
                reward += line_rewards.get(lines, 0)
                self._update_fall_speed()
                if lines == 4:
                    self.tetris_effect_timer = 15  # frames
                else:
                    pass
            del self.current_piece['lines_cleared']

        terminated = self._check_termination()
        if terminated:
            if self.score >= self.SCORE_TARGET:
                reward += 100  # Victory bonus
            else:
                reward -= 100  # Game over penalty

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Game Logic Helpers ---
    def _create_new_piece(self):
        if not hasattr(self, 'np_random') or self.np_random is None:
            self.reset()  # Fallback for initialization

        piece_type = self.np_random.choice(list(self.TETROMINO_SHAPES.keys()))
        shape = self.TETROMINO_SHAPES[piece_type][0]
        new_piece = {
            'type': piece_type,
            'shape': shape,
            'rotation': 0,
            'color': self.TETROMINO_COLORS[piece_type],
            'x': self.PLAYFIELD_W // 2 - len(shape[0]) // 2,
            'y': 0
        }

        if self.current_piece is None:
            self.current_piece = new_piece
        else:
            # On the second call during reset, next_piece is None.
            # We just want to populate it without advancing current_piece.
            if self.next_piece is None:
                self.next_piece = new_piece
            else:
                # Normal game loop: advance the queue.
                self.current_piece = self.next_piece
                self.next_piece = new_piece

        if not self._is_valid_position(self.current_piece):
            self.game_over = True

    def _is_valid_position(self, piece, offset_x=0, offset_y=0):
        if piece is None:
            return False
        px, py = piece['x'] + offset_x, piece['y'] + offset_y
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    if not (0 <= px + c < self.PLAYFIELD_W and 0 <= py + r < self.PLAYFIELD_H):
                        return False
                    if self.grid[py + r][px + c] != 0:
                        return False
        return True

    def _rotate_piece(self):
        piece = self.current_piece
        rotations = self.TETROMINO_SHAPES[piece['type']]
        next_rotation_idx = (piece['rotation'] + 1) % len(rotations)

        test_piece = piece.copy()
        test_piece['rotation'] = next_rotation_idx
        test_piece['shape'] = rotations[next_rotation_idx]

        # Wall kick logic
        for kick_x in [0, -1, 1, -2, 2]:
            if self._is_valid_position(test_piece, offset_x=kick_x):
                piece['rotation'] = test_piece['rotation']
                piece['shape'] = test_piece['shape']
                piece['x'] += kick_x
                return True
        return False

    def _move_horizontal(self, dx):
        if self._is_valid_position(self.current_piece, offset_x=dx):
            self.current_piece['x'] += dx
            return True
        return False

    def _move_down(self):
        if self._is_valid_position(self.current_piece, offset_y=1):
            self.current_piece['y'] += 1
            return True
        return False

    def _hard_drop(self):
        dist = 0
        while self._is_valid_position(self.current_piece, offset_y=1):
            self.current_piece['y'] += 1
            dist += 1
        return dist

    def _lock_piece(self):
        piece = self.current_piece
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    self.grid[piece['y'] + r][piece['x'] + c] = piece['color']

        lines_cleared = self._clear_lines()
        self.current_piece['lines_cleared'] = lines_cleared

        self._create_new_piece()
        self.can_hold = True

    def _clear_lines(self):
        lines_to_clear = []
        for r, row in enumerate(self.grid):
            if all(cell != 0 for cell in row):
                lines_to_clear.append(r)

        if lines_to_clear:
            for r in sorted(lines_to_clear, reverse=True):
                self.grid.pop(r)
                self.grid.insert(0, [0 for _ in range(self.PLAYFIELD_W)])
                self.line_clear_effects.append({'y': r, 'timer': 10})  # 10 frames of effect

        return len(lines_to_clear)

    def _hold_piece(self):
        if not self.can_hold:
            return False

        if self.held_piece is None:
            self.held_piece = self.current_piece
            self._create_new_piece()
        else:
            self.current_piece, self.held_piece = self.held_piece, self.current_piece
            self.current_piece['x'] = self.PLAYFIELD_W // 2 - len(self.current_piece['shape'][0]) // 2
            self.current_piece['y'] = 0

            if not self._is_valid_position(self.current_piece):
                # Swap back if new piece is invalid
                self.current_piece, self.held_piece = self.held_piece, self.current_piece
                return False

        self.can_hold = False
        return True

    def _update_fall_speed(self):
        level = self.score // 500
        self.fall_delay = max(0.2, 0.8 - level * 0.02)

    def _check_termination(self):
        if self.game_over: return True
        if self.score >= self.SCORE_TARGET: return True
        if self.steps >= self.MAX_STEPS: return True
        return False

    # --- Rendering Helpers ---
    def _render_game(self):
        self._render_grid()
        self._render_blocks()
        if not self.game_over and self.current_piece:
            self._render_ghost_piece()
            self._render_active_piece()
        self._render_line_clear_effects()
        self._render_tetris_effect()

    def _render_grid(self):
        pygame.draw.rect(self.screen, self.COLOR_BG, (
        self.PLAYFIELD_X_OFFSET, self.PLAYFIELD_Y_OFFSET, self.PLAYFIELD_W * self.BLOCK_SIZE,
        self.PLAYFIELD_H * self.BLOCK_SIZE))
        for x in range(self.PLAYFIELD_W + 1):
            px = self.PLAYFIELD_X_OFFSET + x * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.PLAYFIELD_Y_OFFSET),
                             (px, self.PLAYFIELD_Y_OFFSET + self.PLAYFIELD_H * self.BLOCK_SIZE))
        for y in range(self.PLAYFIELD_H + 1):
            py = self.PLAYFIELD_Y_OFFSET + y * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAYFIELD_X_OFFSET, py),
                             (self.PLAYFIELD_X_OFFSET + self.PLAYFIELD_W * self.BLOCK_SIZE, py))

    def _render_blocks(self):
        for r, row in enumerate(self.grid):
            for c, color in enumerate(row):
                if color != 0:
                    self._draw_block(c, r, color)

    def _render_active_piece(self):
        piece = self.current_piece
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_block(piece['x'] + c, piece['y'] + r, piece['color'], is_active=True)

    def _render_ghost_piece(self):
        ghost_piece = self.current_piece.copy()
        dist = 0
        while self._is_valid_position(ghost_piece, offset_y=1):
            ghost_piece['y'] += 1
            dist += 1

        if dist > 0:
            piece = ghost_piece
            color = piece['color']
            ghost_color = (color[0] * 0.3, color[1] * 0.3, color[2] * 0.3)
            for r, row in enumerate(piece['shape']):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block(piece['x'] + c, piece['y'] + r, ghost_color, is_ghost=True)

    def _render_ui(self):
        # --- Score Panel ---
        panel_x = self.PLAYFIELD_X_OFFSET + self.PLAYFIELD_W * self.BLOCK_SIZE + 20
        panel_y = self.PLAYFIELD_Y_OFFSET
        panel_w = 160
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BG, (panel_x, panel_y, panel_w, 80), border_radius=5)

        score_text = self.font_small.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (panel_x + 15, panel_y + 15))
        score_val = self.font_main.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_val, (panel_x + 15, panel_y + 35))

        # --- Next Piece Panel ---
        next_panel_y = panel_y + 90
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BG, (panel_x, next_panel_y, panel_w, 120), border_radius=5)
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (panel_x + 15, next_panel_y + 15))
        if self.next_piece:
            self._render_preview_piece(self.next_piece, panel_x, next_panel_y)

        # --- Hold Piece Panel ---
        hold_panel_y = next_panel_y + 130
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BG, (panel_x, hold_panel_y, panel_w, 120), border_radius=5)
        hold_text = self.font_small.render("HOLD", True, self.COLOR_TEXT)
        self.screen.blit(hold_text, (panel_x + 15, hold_panel_y + 15))
        if self.held_piece:
            self._render_preview_piece(self.held_piece, panel_x, hold_panel_y)

    def _render_preview_piece(self, piece, panel_x, panel_y):
        shape = self.TETROMINO_SHAPES[piece['type']][0]  # Always show base rotation
        shape_w = len(shape[0])
        shape_h = len(shape)

        preview_bs = 20  # Block size for preview
        offset_x = panel_x + (160 - shape_w * preview_bs) / 2
        offset_y = panel_y + 30 + (90 - shape_h * preview_bs) / 2

        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    rect = pygame.Rect(offset_x + c * preview_bs, offset_y + r * preview_bs, preview_bs, preview_bs)
                    pygame.draw.rect(self.screen, piece['color'], rect)
                    pygame.draw.rect(self.screen, tuple(min(255, x + 50) for x in piece['color']), rect.inflate(-4, -4))

    def _draw_block(self, c, r, color, is_active=False, is_ghost=False):
        px = self.PLAYFIELD_X_OFFSET + c * self.BLOCK_SIZE
        py = self.PLAYFIELD_Y_OFFSET + r * self.BLOCK_SIZE
        rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)

        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2, border_radius=2)
            return

        main_color = color
        light_color = tuple(min(255, x + 60) for x in main_color)
        dark_color = tuple(max(0, x - 60) for x in main_color)

        pygame.draw.rect(self.screen, dark_color, rect, border_radius=3)
        inner_rect = rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, main_color, inner_rect, border_radius=2)

        if is_active:
            # Add a glow/highlight for active piece
            highlight_rect = rect.inflate(-8, -8)
            pygame.draw.rect(self.screen, light_color, highlight_rect, border_radius=1)

    def _render_line_clear_effects(self):
        for effect in self.line_clear_effects:
            alpha = int(255 * (effect['timer'] / 10))
            if alpha > 0:
                y = self.PLAYFIELD_Y_OFFSET + effect['y'] * self.BLOCK_SIZE
                width = self.PLAYFIELD_W * self.BLOCK_SIZE
                height = self.BLOCK_SIZE

                flash_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                flash_surface.fill((255, 255, 255, alpha))
                self.screen.blit(flash_surface, (self.PLAYFIELD_X_OFFSET, y))

    def _render_tetris_effect(self):
        if self.tetris_effect_timer > 0:
            alpha = int(255 * math.sin(math.pi * (self.tetris_effect_timer / 15)))
            text = self.font_title.render("TETRIS", True, (255, 255, 0))
            text.set_alpha(alpha)
            text_rect = text.get_rect(
                center=(self.PLAYFIELD_X_OFFSET + self.PLAYFIELD_W * self.BLOCK_SIZE / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        text_str = "VICTORY!" if self.score >= self.SCORE_TARGET else "GAME OVER"
        text_color = (255, 215, 0) if self.score >= self.SCORE_TARGET else (200, 50, 50)

        text = self.font_title.render(text_str, True, text_color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
        self.screen.blit(text, text_rect)

        score_text = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30))
        self.screen.blit(score_text, score_rect)


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass

    env = GameEnv()
    obs, info = env.reset()
    done = False

    # Set human_mode to True to control with keyboard
    human_mode = True

    if not human_mode:
        # Test with random actions
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
                obs, info = env.reset()
        env.close()
    else:
        # Manual control loop
        pygame.display.set_caption("Gym Tetris")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        clock = pygame.time.Clock()

        action = np.array([0, 0, 0])  # no-op, released, released

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()

            # Movement
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            else:
                action[0] = 0

            # Space and Shift
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            obs, reward, terminated, truncated, info = env.step(action)

            if reward != 0:
                pass

            # Render to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
                pygame.time.wait(2000)  # Pause before reset
                obs, info = env.reset()

            clock.tick(30)  # Match the assumed FPS

        pygame.quit()