
# Generated: 2025-08-27T19:06:03.629771
# Source Brief: brief_02044.md
# Brief Index: 2044

        
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
        "Controls: ←→ to move, ↑ to rotate, ↓ for soft drop. Space for hard drop, Shift to hold a piece."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based puzzle game. Manipulate falling tetrominoes to clear lines and achieve a target score before the stack reaches the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    CELL_SIZE = 18
    PLAYFIELD_WIDTH = GRID_WIDTH * CELL_SIZE
    PLAYFIELD_HEIGHT = GRID_HEIGHT * CELL_SIZE
    PLAYFIELD_X = (SCREEN_WIDTH - PLAYFIELD_WIDTH) // 2
    PLAYFIELD_Y = (SCREEN_HEIGHT - PLAYFIELD_HEIGHT) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_GHOST = (255, 255, 255, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_WARN = (255, 0, 0)
    
    TETROMINOES = {
        'I': {'shapes': [[[1, 0], [1, 1], [1, 2], [1, 3]], [[0, 2], [1, 2], [2, 2], [3, 2]]], 'color': (66, 215, 245)},
        'O': {'shapes': [[[0, 1], [0, 2], [1, 1], [1, 2]]], 'color': (245, 227, 66)},
        'T': {'shapes': [[[1, 0], [1, 1], [1, 2], [0, 1]], [[0, 1], [1, 1], [2, 1], [1, 0]], [[1, 0], [1, 1], [1, 2], [2, 1]], [[0, 1], [1, 1], [2, 1], [1, 2]]], 'color': (176, 66, 245)},
        'J': {'shapes': [[[0, 0], [1, 0], [1, 1], [1, 2]], [[0, 1], [0, 2], [1, 1], [2, 1]], [[1, 0], [1, 1], [1, 2], [2, 2]], [[0, 1], [1, 1], [2, 0], [2, 1]]], 'color': (66, 81, 245)},
        'L': {'shapes': [[[0, 2], [1, 0], [1, 1], [1, 2]], [[0, 1], [1, 1], [2, 1], [2, 2]], [[1, 0], [1, 1], [1, 2], [2, 0]], [[0, 0], [0, 1], [1, 1], [2, 1]]], 'color': (245, 161, 66)},
        'S': {'shapes': [[[1, 1], [1, 2], [0, 0], [0, 1]], [[0, 1], [1, 1], [1, 2], [2, 2]]], 'color': (66, 245, 84)},
        'Z': {'shapes': [[[0, 1], [0, 2], [1, 0], [1, 1]], [[0, 2], [1, 1], [1, 2], [2, 1]]], 'color': (245, 66, 66)},
    }
    TETROMINO_KEYS = list(TETROMINOES.keys())
    
    # Game settings
    MAX_STEPS = 5000
    WIN_SCORE = 1000
    INITIAL_FALL_SPEED = 15 # frames (0.5s at 30fps)
    LINE_CLEAR_ANIM_FRAMES = 6 # 0.2s at 30fps

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 18, bold=True)
        
        # Etc...
        self.grid = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.active_piece = None
        self.next_piece_key = None
        self.held_piece_key = None
        self.can_swap_hold = True
        self.fall_timer = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.last_action = (0, 0, 0)
        self.line_clear_animation = []
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.piece_bag = []
        self._refill_piece_bag()
        
        self.next_piece_key = self.piece_bag.pop()
        self._spawn_new_piece()

        self.held_piece_key = None
        self.can_swap_hold = True
        
        self.fall_timer = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.last_action = (0, 0, 0)
        self.line_clear_animation = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = -0.01
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Update game logic
        self.steps += 1
        
        # If in line clear animation, pause gameplay
        if self.line_clear_animation:
            self._update_line_clear_animation()
        else:
            # --- Handle Input ---
            # Rising edge detection for instant actions
            space_pressed = space_held and not self.last_action[1]
            shift_pressed = shift_held and not self.last_action[2]

            piece_placed_by_action = False
            if shift_pressed and self.can_swap_hold:
                self._hold_piece() # sfx: hold_piece.wav
            elif space_pressed:
                placement_reward = self._hard_drop() # sfx: hard_drop.wav
                reward += placement_reward
                piece_placed_by_action = True
            else:
                # Handle continuous actions
                if movement == 1: self._rotate_piece() # sfx: rotate.wav
                if movement == 3: self._move_piece(-1, 0) # sfx: move.wav
                if movement == 4: self._move_piece(1, 0) # sfx: move.wav

            # --- Handle Gravity & Placement ---
            if not piece_placed_by_action:
                soft_drop = (movement == 2)
                self.fall_timer += 2 if soft_drop else 1 # Soft drop effectively doubles fall speed

                if self.fall_timer >= self.fall_speed:
                    self.fall_timer = 0
                    if not self._move_piece(0, 1):
                        # Could not move down, so place piece
                        placement_reward = self._place_piece()
                        reward += placement_reward
                        # sfx: place_piece.wav

        self.last_action = (movement, space_held, shift_held)
        
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    # --- Game Logic Helpers ---

    def _refill_piece_bag(self):
        self.piece_bag = self.TETROMINO_KEYS[:]
        self.np_random.shuffle(self.piece_bag)

    def _spawn_new_piece(self):
        if not self.piece_bag:
            self._refill_piece_bag()
        
        self.active_piece = {
            'key': self.next_piece_key,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - 2,
            'y': 0
        }
        self.can_swap_hold = True
        
        self.next_piece_key = self.piece_bag.pop()
        
        if self._check_collision(self.active_piece):
            self.game_over = True

    def _get_piece_coords(self, piece):
        shape_coords = self.TETROMINOES[piece['key']]['shapes'][piece['rotation']]
        return [(piece['y'] + r, piece['x'] + c) for r, c in shape_coords]

    def _check_collision(self, piece):
        coords = self._get_piece_coords(piece)
        for r, c in coords:
            if not (0 <= c < self.GRID_WIDTH and 0 <= r < self.GRID_HEIGHT):
                return True
            if self.grid[r, c] != 0:
                return True
        return False

    def _move_piece(self, dx, dy):
        if not self.active_piece: return False
        self.active_piece['x'] += dx
        self.active_piece['y'] += dy
        if self._check_collision(self.active_piece):
            self.active_piece['x'] -= dx
            self.active_piece['y'] -= dy
            return False
        return True

    def _rotate_piece(self):
        if not self.active_piece: return
        original_rotation = self.active_piece['rotation']
        num_rotations = len(self.TETROMINOES[self.active_piece['key']]['shapes'])
        self.active_piece['rotation'] = (original_rotation + 1) % num_rotations
        
        if self._check_collision(self.active_piece):
            for dx in [1, -1, 2, -2]: # Wall kick tests
                self.active_piece['x'] += dx
                if not self._check_collision(self.active_piece): return
                self.active_piece['x'] -= dx
            self.active_piece['rotation'] = original_rotation

    def _hard_drop(self):
        if not self.active_piece: return 0
        while self._move_piece(0, 1):
            pass
        return self._place_piece()

    def _hold_piece(self):
        if self.held_piece_key is None:
            self.held_piece_key = self.active_piece['key']
            self._spawn_new_piece()
        else:
            self.active_piece['key'], self.held_piece_key = self.held_piece_key, self.active_piece['key']
            self.active_piece['rotation'] = 0
            self.active_piece['x'] = self.GRID_WIDTH // 2 - 2
            self.active_piece['y'] = 0
            if self._check_collision(self.active_piece):
                self.game_over = True
        self.can_swap_hold = False
    
    def _place_piece(self):
        if not self.active_piece: return 0
        coords = self._get_piece_coords(self.active_piece)
        color_index = self.TETROMINO_KEYS.index(self.active_piece['key']) + 1
        for r, c in coords:
            if 0 <= r < self.GRID_HEIGHT and 0 <= c < self.GRID_WIDTH:
                self.grid[r, c] = color_index
        
        self.active_piece = None
        lines_cleared = self._check_and_start_line_clear()
        
        if not self.line_clear_animation:
            self._spawn_new_piece()

        score_map = {1: 100, 2: 300, 3: 500, 4: 800}
        self.score += score_map.get(lines_cleared, 0)
        
        level = self.score // 200
        self.fall_speed = max(3, self.INITIAL_FALL_SPEED - level * 0.3)

        reward_map = {1: 1, 2: 3, 3: 5, 4: 10}
        if lines_cleared > 0:
            # sfx: line_clear.wav
            return reward_map.get(lines_cleared, 0)
        else:
            return -0.2

    def _check_and_start_line_clear(self):
        lines_to_clear = [r for r in range(self.GRID_HEIGHT) if np.all(self.grid[r, :] != 0)]
        if lines_to_clear:
            self.line_clear_animation = [(row, self.LINE_CLEAR_ANIM_FRAMES) for row in lines_to_clear]
        return len(lines_to_clear)

    def _update_line_clear_animation(self):
        if not self.line_clear_animation: return
        self.line_clear_animation = [(r, t - 1) for r, t in self.line_clear_animation]
        if self.line_clear_animation[0][1] <= 0:
            self._finish_line_clear()

    def _finish_line_clear(self):
        rows_cleared = sorted([row for row, timer in self.line_clear_animation], reverse=True)
        for r in rows_cleared:
            self.grid[1:r+1, :] = self.grid[0:r, :]
            self.grid[0, :] = 0
        self.line_clear_animation = []
        self._spawn_new_piece()
        
    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            # sfx: win.wav
            return True, 100
        if self.game_over:
            # sfx: lose.wav
            return True, -50
        if self.steps >= self.MAX_STEPS:
            return True, 0
        return False, 0

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.PLAYFIELD_X - 2, self.PLAYFIELD_Y - 2, self.PLAYFIELD_WIDTH + 4, self.PLAYFIELD_HEIGHT + 4), 2)

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    color_key = self.TETROMINO_KEYS[self.grid[r, c] - 1]
                    color = self.TETROMINOES[color_key]['color']
                    self._draw_block(c, r, color)
            if np.any(self.grid[r, :]) and r < 4:
                 warn_surface = pygame.Surface((self.PLAYFIELD_WIDTH, self.CELL_SIZE), pygame.SRCALPHA)
                 warn_surface.fill((255, 0, 0, 20 + (4-r)*10))
                 self.screen.blit(warn_surface, (self.PLAYFIELD_X, self.PLAYFIELD_Y + r * self.CELL_SIZE))

        if not self.game_over:
            if self.active_piece:
                self._render_ghost_piece()
                self._render_active_piece()
        
        if self.line_clear_animation:
            self._render_line_clear_animation()

    def _render_active_piece(self):
        color = self.TETROMINOES[self.active_piece['key']]['color']
        coords = self._get_piece_coords(self.active_piece)
        for r, c in coords:
            self._draw_block(c, r, color)
            
    def _render_ghost_piece(self):
        ghost_piece = self.active_piece.copy()
        while not self._check_collision(ghost_piece):
            ghost_piece['y'] += 1
        ghost_piece['y'] -= 1
        coords = self._get_piece_coords(ghost_piece)
        for r, c in coords:
            if r >= 0:
                rect = pygame.Rect(self.PLAYFIELD_X + c * self.CELL_SIZE, self.PLAYFIELD_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GHOST, rect, 2)

    def _draw_block(self, c, r, color):
        rect = pygame.Rect(self.PLAYFIELD_X + c * self.CELL_SIZE, self.PLAYFIELD_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        darker_color = tuple(max(0, val - 50) for val in color)
        pygame.draw.rect(self.screen, darker_color, rect)
        inner_rect = rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, color, inner_rect)
        
    def _render_line_clear_animation(self):
        for row, timer in self.line_clear_animation:
            alpha = 255 - abs(self.LINE_CLEAR_ANIM_FRAMES/2 - timer) * (510 / self.LINE_CLEAR_ANIM_FRAMES)
            surface = pygame.Surface((self.PLAYFIELD_WIDTH, self.CELL_SIZE), pygame.SRCALPHA)
            surface.fill((255, 255, 255, alpha))
            self.screen.blit(surface, (self.PLAYFIELD_X, self.PLAYFIELD_Y + row * self.CELL_SIZE))

    def _render_ui(self):
        score_text = self.font_main.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 20))
        score_title = self.font_title.render("SCORE", True, self.COLOR_GRID)
        self.screen.blit(score_title, (self.SCREEN_WIDTH - score_text.get_width() - 20, 45))

        self._render_piece_preview(self.next_piece_key, self.PLAYFIELD_X + self.PLAYFIELD_WIDTH + 20, 80, "NEXT")
        self._render_piece_preview(self.held_piece_key, 40, 80, "HOLD")
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            status_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            text_surf = self.font_main.render(status_text, True, self.COLOR_TEXT)
            self.screen.blit(text_surf, (self.SCREEN_WIDTH/2 - text_surf.get_width()/2, self.SCREEN_HEIGHT/2 - text_surf.get_height()/2))

    def _render_piece_preview(self, piece_key, x, y, title):
        title_surf = self.font_title.render(title, True, self.COLOR_GRID)
        self.screen.blit(title_surf, (x, y))
        box_size = self.CELL_SIZE * 4.5
        pygame.draw.rect(self.screen, self.COLOR_GRID, (x - 5, y + 25, box_size, box_size), 2)
        if piece_key:
            color = self.TETROMINOES[piece_key]['color']
            shape_coords = self.TETROMINOES[piece_key]['shapes'][0]
            min_r, max_r = min(c[0] for c in shape_coords), max(c[0] for c in shape_coords)
            min_c, max_c = min(c[1] for c in shape_coords), max(c[1] for c in shape_coords)
            offset_x = x + (box_size - (max_c - min_c + 1) * self.CELL_SIZE) / 2 - min_c * self.CELL_SIZE
            offset_y = y + 25 + (box_size - (max_r - min_r + 1) * self.CELL_SIZE) / 2 - min_r * self.CELL_SIZE
            for r, c in shape_coords:
                rect = pygame.Rect(offset_x + c * self.CELL_SIZE, offset_y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                darker_color = tuple(max(0, val - 50) for val in color)
                pygame.draw.rect(self.screen, darker_color, rect)
                inner_rect = rect.inflate(-4, -4)
                pygame.draw.rect(self.screen, color, inner_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
        
    def close(self):
        pygame.font.quit()
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
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gymnasium Tetris")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000)
            
        clock.tick(30)
        
    env.close()