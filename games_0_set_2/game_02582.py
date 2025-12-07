
# Generated: 2025-08-27T20:48:27.458758
# Source Brief: brief_02582.md
# Brief Index: 2582

        
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
        "Controls: ←→ to move, ↑ to rotate clockwise, SHIFT to rotate counter-clockwise. ↓ for soft drop, SPACE for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, procedurally generated falling block puzzle. Clear lines to score points, but be careful not to stack blocks to the top!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT = 10, 20
    CELL_SIZE = 18
    PLAYFIELD_PIXEL_WIDTH = PLAYFIELD_WIDTH * CELL_SIZE
    PLAYFIELD_PIXEL_HEIGHT = PLAYFIELD_HEIGHT * CELL_SIZE
    SIDEBAR_WIDTH = 200

    PLAYFIELD_X_OFFSET = (SCREEN_WIDTH - PLAYFIELD_PIXEL_WIDTH - SIDEBAR_WIDTH) // 2
    PLAYFIELD_Y_OFFSET = (SCREEN_HEIGHT - PLAYFIELD_PIXEL_HEIGHT) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_GHOST = (255, 255, 255)
    TETROMINO_COLORS = [
        (0, 0, 0),  # 0 is empty
        (40, 180, 255),   # I (Cyan)
        (0, 80, 255),     # J (Blue)
        (255, 140, 0),    # L (Orange)
        (255, 220, 0),    # O (Yellow)
        (80, 220, 40),    # S (Green)
        (180, 40, 220),   # T (Purple)
        (255, 40, 80),    # Z (Red)
    ]

    # Tetromino shapes and their rotations
    TETROMINOES = {
        'I': [[(0, -1), (0, 0), (0, 1), (0, 2)], [(-1, 0), (0, 0), (1, 0), (2, 0)]],
        'O': [[(0, 0), (1, 0), (0, 1), (1, 1)]],
        'J': [[(-1, -1), (-1, 0), (0, 0), (1, 0)], [(0, -1), (1, -1), (0, 0), (0, 1)], [(-1, 0), (0, 0), (1, 0), (1, 1)], [(0, -1), (0, 0), (0, 1), (-1, 1)]],
        'L': [[(1, -1), (-1, 0), (0, 0), (1, 0)], [(0, -1), (0, 0), (0, 1), (1, 1)], [(-1, 0), (0, 0), (1, 0), (-1, 1)], [(0, -1), (0, 0), (-1, -1), (0, 1)]],
        'S': [[(0, -1), (0, 0), (1, 0), (1, 1)], [(-1, 0), (0, 0), (0, -1), (1, -1)]],
        'T': [[(0, -1), (-1, 0), (0, 0), (1, 0)], [(0, -1), (0, 0), (1, 0), (0, 1)], [(-1, 0), (0, 0), (1, 0), (0, 1)], [(0, -1), (-1, 0), (0, 0), (0, 1)]],
        'Z': [[(0, -1), (0, 0), (-1, 0), (-1, 1)], [(-1, -1), (0, -1), (0, 0), (1, 0)]]
    }
    
    FALL_SPEED_NORMAL = 15 # Frames per grid cell
    SOFT_DROP_MULTIPLIER = 10
    MAX_SCORE = 1000
    MAX_STEPS = 5000
    
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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        self.playfield = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.current_piece = None
        self.next_piece = None
        self.fall_timer = 0
        self.line_clear_animation = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.playfield = np.zeros((self.PLAYFIELD_HEIGHT, self.PLAYFIELD_WIDTH), dtype=int)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_timer = 0
        self.line_clear_animation = []
        
        self._spawn_new_piece()
        self._spawn_new_piece() # Call twice to populate current and next
        
        return self._get_observation(), self._get_info()
    
    def _spawn_new_piece(self):
        self.current_piece = self.next_piece
        
        shape_name = self.np_random.choice(list(self.TETROMINOES.keys()))
        color_idx = list(self.TETROMINOES.keys()).index(shape_name) + 1
        
        self.next_piece = {
            'shape_name': shape_name,
            'shape': self.TETROMINOES[shape_name],
            'rotation': 0,
            'x': self.PLAYFIELD_WIDTH // 2 - 1,
            'y': 0,
            'color_idx': color_idx
        }

        if self.current_piece and not self._is_valid_position(self.current_piece):
            self.game_over = True

    def _get_piece_coords(self, piece):
        coords = []
        shape_template = piece['shape'][piece['rotation']]
        for dx, dy in shape_template:
            coords.append((piece['x'] + dx, piece['y'] + dy))
        return coords

    def _is_valid_position(self, piece, offset_x=0, offset_y=0, rotation=None):
        if piece is None: return False
        
        test_piece = piece.copy()
        test_piece['x'] += offset_x
        test_piece['y'] += offset_y
        if rotation is not None:
            test_piece['rotation'] = rotation

        coords = self._get_piece_coords(test_piece)
        for x, y in coords:
            if not (0 <= x < self.PLAYFIELD_WIDTH and y < self.PLAYFIELD_HEIGHT):
                return False
            if y >= 0 and self.playfield[y, x] > 0:
                return False
        return True

    def _lock_piece(self):
        reward = 0
        max_height_before = self._get_max_height()
        
        coords = self._get_piece_coords(self.current_piece)
        for x, y in coords:
            if y >= 0:
                self.playfield[y, x] = self.current_piece['color_idx']
        
        # sound placeholder: # sfx_lock.play()
        lines_cleared, line_reward = self._clear_lines()
        reward += line_reward
        
        if lines_cleared == 0:
            max_height_after = self._get_max_height()
            if max_height_after > max_height_before:
                reward -= 0.2 # Penalty for increasing height without clearing lines
        
        self.score += lines_cleared * 10 # Base score for lines
        
        self._spawn_new_piece()
        return reward

    def _get_max_height(self):
        for i in range(self.PLAYFIELD_HEIGHT):
            if np.any(self.playfield[i, :]):
                return self.PLAYFIELD_HEIGHT - i
        return 0

    def _clear_lines(self):
        lines_to_clear = []
        for i in range(self.PLAYFIELD_HEIGHT):
            if np.all(self.playfield[i, :] > 0):
                lines_to_clear.append(i)
        
        if lines_to_clear:
            # sound placeholder: # sfx_clear.play()
            for i in lines_to_clear:
                self.line_clear_animation.append({'y': i, 'timer': 5})
            
            # Defer actual clearing to allow animation
            # This is a bit of a trick: we'll clear them after the animation timer runs out
            # For the RL agent, the effect is immediate via reward and state change
            cleared_rows = self.playfield[lines_to_clear, :]
            remaining_rows = np.delete(self.playfield, lines_to_clear, axis=0)
            new_top_rows = np.zeros((len(lines_to_clear), self.PLAYFIELD_WIDTH), dtype=int)
            self.playfield = np.vstack((new_top_rows, remaining_rows))
            
            num_cleared = len(lines_to_clear)
            reward_map = {1: 1, 2: 2, 3: 4, 4: 8}
            self.score += reward_map.get(num_cleared, 0) * 10 # Tetris bonus
            return num_cleared, reward_map.get(num_cleared, 0)
        
        return 0, 0

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed
        terminated = False
        
        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

            # --- Handle Input ---
            # Hard drop (Space) takes precedence
            if space_held:
                # sound placeholder: # sfx_hard_drop.play()
                while self._is_valid_position(self.current_piece, offset_y=1):
                    self.current_piece['y'] += 1
                reward += self._lock_piece()
            else:
                # Rotations
                if movement == 1: # Clockwise (Up)
                    next_rot = (self.current_piece['rotation'] + 1) % len(self.current_piece['shape'])
                    if self._is_valid_position(self.current_piece, rotation=next_rot):
                        self.current_piece['rotation'] = next_rot
                        # sound placeholder: # sfx_rotate.play()
                if shift_held: # Counter-clockwise
                    next_rot = (self.current_piece['rotation'] - 1 + len(self.current_piece['shape'])) % len(self.current_piece['shape'])
                    if self._is_valid_position(self.current_piece, rotation=next_rot):
                        self.current_piece['rotation'] = next_rot
                        # sound placeholder: # sfx_rotate.play()

                # Horizontal Movement
                if movement == 3: # Left
                    if self._is_valid_position(self.current_piece, offset_x=-1):
                        self.current_piece['x'] -= 1
                        # sound placeholder: # sfx_move.play()
                elif movement == 4: # Right
                    if self._is_valid_position(self.current_piece, offset_x=1):
                        self.current_piece['x'] += 1
                        # sound placeholder: # sfx_move.play()
                
                # --- Apply Gravity ---
                self.fall_timer += self.SOFT_DROP_MULTIPLIER if movement == 2 else 1
                if self.fall_timer >= self.FALL_SPEED_NORMAL:
                    self.fall_timer = 0
                    if self._is_valid_position(self.current_piece, offset_y=1):
                        self.current_piece['y'] += 1
                    else:
                        reward += self._lock_piece()

        # --- Update animations ---
        self.line_clear_animation = [a for a in self.line_clear_animation if a['timer'] > 0]
        for anim in self.line_clear_animation:
            anim['timer'] -= 1

        # --- Check Termination Conditions ---
        self.steps += 1
        if self.game_over:
            reward -= 10
            terminated = True
        elif self.score >= self.MAX_SCORE:
            reward += 100
            terminated = True
            self.game_over = True # To show "You Win!"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _draw_block(self, surface, x, y, color_idx, is_ghost=False):
        color = self.TETROMINO_COLORS[color_idx]
        px, py = x * self.CELL_SIZE, y * self.CELL_SIZE
        rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)

        if is_ghost:
            pygame.draw.rect(surface, self.COLOR_GHOST, rect, 2, border_radius=3)
        else:
            # Beveled look
            light_color = tuple(min(255, c + 40) for c in color)
            dark_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(surface, dark_color, rect, border_radius=3)
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(surface, color, inner_rect, border_radius=2)
            pygame.gfxdraw.aacircle(surface, rect.left + 3, rect.top + 3, 2, light_color)


    def _render_game(self):
        game_surface = pygame.Surface((self.PLAYFIELD_PIXEL_WIDTH, self.PLAYFIELD_PIXEL_HEIGHT))
        game_surface.fill(self.COLOR_BG)
        
        # Draw grid
        for x in range(self.PLAYFIELD_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(game_surface, self.COLOR_GRID, (px, 0), (px, self.PLAYFIELD_PIXEL_HEIGHT))
        for y in range(self.PLAYFIELD_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(game_surface, self.COLOR_GRID, (0, py), (self.PLAYFIELD_PIXEL_WIDTH, py))

        # Draw locked blocks
        for y in range(self.PLAYFIELD_HEIGHT):
            for x in range(self.PLAYFIELD_WIDTH):
                if self.playfield[y, x] > 0:
                    self._draw_block(game_surface, x, y, int(self.playfield[y, x]))

        # Draw ghost piece
        if not self.game_over and self.current_piece:
            ghost_piece = self.current_piece.copy()
            while self._is_valid_position(ghost_piece, offset_y=1):
                ghost_piece['y'] += 1
            for x, y in self._get_piece_coords(ghost_piece):
                if y >= 0:
                    self._draw_block(game_surface, x, y, ghost_piece['color_idx'], is_ghost=True)

        # Draw current piece
        if not self.game_over and self.current_piece:
            for x, y in self._get_piece_coords(self.current_piece):
                if y >= 0:
                    self._draw_block(game_surface, x, y, self.current_piece['color_idx'])

        # Draw line clear animation
        for anim in self.line_clear_animation:
            flash_color = (255, 255, 255, 150 + anim['timer'] * 20)
            flash_rect = pygame.Rect(0, anim['y'] * self.CELL_SIZE, self.PLAYFIELD_PIXEL_WIDTH, self.CELL_SIZE)
            flash_surf = pygame.Surface(flash_rect.size, pygame.SRCALPHA)
            flash_surf.fill(flash_color)
            game_surface.blit(flash_surf, flash_rect.topleft)

        self.screen.blit(game_surface, (self.PLAYFIELD_X_OFFSET, self.PLAYFIELD_Y_OFFSET))

    def _render_ui(self):
        ui_x = self.PLAYFIELD_X_OFFSET + self.PLAYFIELD_PIXEL_WIDTH + 40
        
        # Score display
        score_text = self.font_main.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_x, self.PLAYFIELD_Y_OFFSET))
        score_val = self.font_main.render(f"{self.score}", True, self.TETROMINO_COLORS[4])
        self.screen.blit(score_val, (ui_x, self.PLAYFIELD_Y_OFFSET + 30))

        # Next piece display
        next_text = self.font_main.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (ui_x, self.PLAYFIELD_Y_OFFSET + 100))
        
        next_box_rect = pygame.Rect(ui_x, self.PLAYFIELD_Y_OFFSET + 130, 4 * self.CELL_SIZE, 4 * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, next_box_rect, 2, border_radius=5)
        
        if self.next_piece:
            next_surf = pygame.Surface((4 * self.CELL_SIZE, 4 * self.CELL_SIZE), pygame.SRCALPHA)
            temp_piece = self.next_piece.copy()
            temp_piece['x'], temp_piece['y'] = 1, 1
            temp_piece['rotation'] = 0
            for dx, dy in temp_piece['shape'][0]:
                self._draw_block(next_surf, 1.5 + dx, 1.5 + dy, temp_piece['color_idx'])
            self.screen.blit(next_surf, next_box_rect.topleft)

        # Game Over / Win display
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.score >= self.MAX_SCORE else "GAME OVER"
            end_text = self.font_main.render(message, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for display
    pygame.display.set_caption("Falling Block Puzzle")
    screen_display = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample() # Start with a random action
    action.fill(0) # Default to no-op

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Map keyboard to MultiDiscrete action ---
        keys = pygame.key.get_pressed()
        
        # Movement (mutually exclusive)
        mov = 0 # no-op
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([mov, space, shift])

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # 30 FPS

    pygame.quit()
    print(f"Game Over. Final Score: {info['score']}, Steps: {info['steps']}")