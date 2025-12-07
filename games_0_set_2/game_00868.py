
# Generated: 2025-08-27T15:03:03.921389
# Source Brief: brief_00868.md
# Brief Index: 868

        
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
        "Controls: ↑/↓ to rotate, ←/→ to move. Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a relentless onslaught of falling blocks by clearing lines. Survive for 60 seconds to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- Game Constants ---
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.GRID_X = (self.screen_width - self.GRID_WIDTH * self.CELL_SIZE) // 2 - 100
        self.GRID_Y = (self.screen_height - self.GRID_HEIGHT * self.CELL_SIZE) // 2

        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_DANGER = (80, 20, 20)
        self.COLOR_GHOST = (255, 255, 255)

        self._define_tetrominoes()

        # Control feel (Delayed Auto Shift / Auto Repeat Rate)
        self.DAS_THRESHOLD = 8  # frames
        self.ARR_RATE = 2  # frames
        
        # --- State Variables (initialized in reset) ---
        self.grid = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.time_remaining = None
        
        self.current_piece = None
        self.next_piece = None
        
        self.fall_progress = None
        self.fall_speed_seconds = None
        
        self.last_action = None
        self.move_action_held_frames = None
        self.line_clear_animation = None
        self.lock_delay_timer = None

        self.reset()
        self.validate_implementation()

    def _define_tetrominoes(self):
        # Shapes are defined by coordinates relative to a pivot point
        self.TETROMINOES = {
            'I': {'shape': [(0, -1), (0, 0), (0, 1), (0, 2)], 'color': (66, 215, 245)},
            'J': {'shape': [(-1, -1), (0, -1), (0, 0), (0, 1)], 'color': (66, 105, 245)},
            'L': {'shape': [(1, -1), (0, -1), (0, 0), (0, 1)], 'color': (245, 166, 66)},
            'O': {'shape': [(0, 0), (1, 0), (0, 1), (1, 1)], 'color': (245, 225, 66)},
            'S': {'shape': [(0, 0), (1, 0), (-1, 1), (0, 1)], 'color': (141, 245, 66)},
            'T': {'shape': [(-1, 0), (0, 0), (1, 0), (0, 1)], 'color': (188, 66, 245)},
            'Z': {'shape': [(-1, 0), (0, 0), (0, 1), (1, 1)], 'color': (245, 66, 75)}
        }
        self.TETROMINO_KEYS = list(self.TETROMINOES.keys())
        # Assign an integer index to each color for grid storage
        self.COLOR_MAP = {val['color']: i + 1 for i, val in enumerate(self.TETROMINOES.values())}
        self.INDEX_TO_COLOR = {i + 1: val['color'] for i, val in enumerate(self.TETROMINOES.values())}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        
        self.fall_progress = 0
        self.fall_speed_seconds = 0.5
        
        self.piece_bag = []
        
        self._spawn_new_piece() # Populates next_piece
        self._spawn_new_piece() # Populates current_piece from next_piece

        self.last_action = self.action_space.sample()
        self.move_action_held_frames = 0
        self.line_clear_animation = []
        self.lock_delay_timer = None
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        reward = -0.001 # Small penalty per step to encourage speed

        # Handle input
        reward += self._handle_input(action)
        
        # Update game state
        state_reward, terminated = self._update_game_state()
        reward += state_reward

        self.game_over = terminated or self.time_remaining <= 0
        
        # Calculate terminal rewards
        if self.game_over:
            if self.time_remaining <= 0:
                reward += 100 # Win reward
            else:
                reward += -10 # Loss penalty

        self.last_action = action

        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Hard Drop ---
        is_space_press = space_held and not (self.last_action[1] == 1)
        if is_space_press:
            # Sound: Hard drop thud
            while self._move_piece(0, 1):
                self.score += 0.1 # Small reward for dropping
            reward += self._lock_piece()
            self.game_over = self.game_over or self._check_game_over_on_spawn()
            return reward

        # --- Movement and Rotation (with DAS/ARR) ---
        is_move_press = movement != 0 and movement != self.last_action[0]
        is_move_hold = movement != 0 and movement == self.last_action[0]

        if is_move_press:
            self.move_action_held_frames = 0
            if self._process_movement(movement):
                reward -= 0.01 # Small penalty for horizontal movement
        elif is_move_hold:
            self.move_action_held_frames += 1
            if self.move_action_held_frames > self.DAS_THRESHOLD and \
               (self.move_action_held_frames - self.DAS_THRESHOLD) % self.ARR_RATE == 0:
                if self._process_movement(movement):
                    reward -= 0.01
        
        return reward

    def _process_movement(self, movement_action):
        moved = False
        if movement_action == 1: # Rotate Left
            moved = self._rotate_piece(-1)
        elif movement_action == 2: # Rotate Right
            moved = self._rotate_piece(1)
        elif movement_action == 3: # Move Left
            moved = self._move_piece(-1, 0)
        elif movement_action == 4: # Move Right
            moved = self._move_piece(1, 0)
        
        if moved:
            # Reset lock delay if piece moves
            if self.lock_delay_timer is not None:
                self.lock_delay_timer = int(0.5 * self.FPS)
        
        return moved

    def _update_game_state(self):
        reward = 0
        terminated = False
        
        # Update line clear animations
        if self.line_clear_animation:
            self.line_clear_animation[0] = (self.line_clear_animation[0][0], self.line_clear_animation[0][1] - 1)
            if self.line_clear_animation[0][1] <= 0:
                rows_to_clear = self.line_clear_animation.pop(0)[0]
                self._remove_cleared_lines(rows_to_clear)
                self._spawn_new_piece()
                terminated = self._check_game_over_on_spawn()

        # Don't process gravity if animating or no piece
        if self.line_clear_animation or self.current_piece is None:
            return reward, terminated

        # Check for landing and update lock delay
        if not self._check_collision(self.current_piece['shape'], (self.current_piece['x'], self.current_piece['y'] + 1)):
            self.lock_delay_timer = None # Not on the ground
        else: # On the ground or another piece
            if self.lock_delay_timer is None:
                self.lock_delay_timer = int(0.5 * self.FPS) # Start lock delay
            else:
                self.lock_delay_timer -= 1

        # Apply gravity or lock piece
        self.fall_progress += 1
        fall_threshold = self.fall_speed_seconds * self.FPS
        
        if self.lock_delay_timer is not None and self.lock_delay_timer <= 0:
            reward += self._lock_piece()
            terminated = self._check_game_over_on_spawn()
        elif self.fall_progress >= fall_threshold:
            self.fall_progress = 0
            if not self._move_piece(0, 1):
                # Cannot move down, but lock delay is active
                pass

        # Difficulty scaling
        if self.steps > 0 and self.steps % 60 == 0:
            self.fall_speed_seconds = max(0.05, self.fall_speed_seconds - 0.01)

        return reward, terminated

    def _spawn_new_piece(self):
        if not self.piece_bag:
            self.piece_bag = self.TETROMINO_KEYS[:]
            self.np_random.shuffle(self.piece_bag)

        self.current_piece = self.next_piece
        
        next_key = self.piece_bag.pop()
        self.next_piece = {
            'key': next_key,
            'shape': self.TETROMINOES[next_key]['shape'],
            'color': self.TETROMINOES[next_key]['color'],
            'x': self.GRID_WIDTH // 2,
            'y': 1 if next_key != 'I' else 0,
            'rotation': 0
        }

    def _check_game_over_on_spawn(self):
        if self.current_piece and self._check_collision(self.current_piece['shape'], (self.current_piece['x'], self.current_piece['y'])):
            self.current_piece = None # Don't draw the overlapping piece
            return True
        return False

    def _lock_piece(self):
        # Sound: Piece lock click
        if self.current_piece is None: return 0
        
        for dx, dy in self.current_piece['shape']:
            x, y = self.current_piece['x'] + dx, self.current_piece['y'] + dy
            if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                self.grid[y, x] = self.COLOR_MAP[self.current_piece['color']]
        
        self.current_piece = None
        self.lock_delay_timer = None
        
        reward = self._check_and_clear_lines()
        
        if not self.line_clear_animation: # If no lines cleared, spawn next piece immediately
            self._spawn_new_piece()

        return reward

    def _check_and_clear_lines(self):
        full_lines = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                full_lines.append(r)

        if full_lines:
            # Sound: Line clear chime
            self.line_clear_animation.append((full_lines, int(0.2 * self.FPS)))
            
            # Calculate reward
            num_lines = len(full_lines)
            if num_lines == 4:
                return 5.0  # Tetris bonus
            else:
                return 1.0 * num_lines
        return 0

    def _remove_cleared_lines(self, rows_to_clear):
        rows_to_clear.sort(reverse=True)
        for r in rows_to_clear:
            self.grid[1:r+1, :] = self.grid[0:r, :]
            self.grid[0, :] = 0
        self.score += len(rows_to_clear) * 100

    def _move_piece(self, dx, dy):
        if self.current_piece is None: return False
        
        new_x = self.current_piece['x'] + dx
        new_y = self.current_piece['y'] + dy
        if not self._check_collision(self.current_piece['shape'], (new_x, new_y)):
            self.current_piece['x'] = new_x
            self.current_piece['y'] = new_y
            return True
        return False

    def _rotate_piece(self, direction):
        if self.current_piece is None or self.current_piece['key'] == 'O': return False

        original_shape = self.current_piece['shape']
        
        rotated_shape = []
        for x, y in original_shape:
            if direction == 1: rotated_shape.append((y, -x))
            else: rotated_shape.append((-y, x))
        
        kick_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (-2, 0), (2, 0)]
        
        for kx, ky in kick_offsets:
            new_x = self.current_piece['x'] + kx
            new_y = self.current_piece['y'] + ky
            if not self._check_collision(rotated_shape, (new_x, new_y)):
                self.current_piece['shape'] = rotated_shape
                self.current_piece['x'] = new_x
                self.current_piece['y'] = new_y
                self.current_piece['rotation'] = (self.current_piece['rotation'] + direction) % 4
                return True
        return False

    def _check_collision(self, shape, pos):
        px, py = pos
        for dx, dy in shape:
            x, y = px + dx, py + dy
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return True # Wall collision
            if y >= 0 and self.grid[y, x] != 0:
                return True # Grid collision
        return False

    def _get_ghost_piece_y(self):
        if self.current_piece is None: return self.current_piece
        y = self.current_piece['y']
        while not self._check_collision(self.current_piece['shape'], (self.current_piece['x'], y + 1)):
            y += 1
        return y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))
        grid_surface.fill(self.COLOR_GRID)
        danger_rect = pygame.Rect(0, 0, self.GRID_WIDTH * self.CELL_SIZE, 4 * self.CELL_SIZE)
        pygame.draw.rect(grid_surface, self.COLOR_DANGER, danger_rect)
        
        for i in range(self.GRID_WIDTH + 1):
            pygame.draw.line(grid_surface, self.COLOR_BG, (i * self.CELL_SIZE, 0), (i * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(grid_surface, self.COLOR_BG, (0, i * self.CELL_SIZE), (self.GRID_WIDTH * self.CELL_SIZE, i * self.CELL_SIZE))
        
        self.screen.blit(grid_surface, (self.GRID_X, self.GRID_Y))

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    self._draw_block(self.screen, self.INDEX_TO_COLOR[self.grid[r, c]], c, r)

        if self.line_clear_animation:
            rows, timer = self.line_clear_animation[0]
            alpha = 255 * (timer / (0.2 * self.FPS))
            flash_color = (255, 255, 255, alpha)
            for r in rows:
                flash_rect = pygame.Rect(self.GRID_X, self.GRID_Y + r * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
                flash_surf = pygame.Surface(flash_rect.size, pygame.SRCALPHA)
                flash_surf.fill(flash_color)
                self.screen.blit(flash_surf, flash_rect.topleft)

        if self.current_piece:
            ghost_y = self._get_ghost_piece_y()
            for dx, dy in self.current_piece['shape']:
                self._draw_block(self.screen, self.current_piece['color'], self.current_piece['x'] + dx, ghost_y + dy, is_ghost=True)
                self._draw_block(self.screen, self.current_piece['color'], self.current_piece['x'] + dx, self.current_piece['y'] + dy)

    def _draw_block(self, surface, color, grid_x, grid_y, is_ghost=False):
        screen_x = self.GRID_X + grid_x * self.CELL_SIZE
        screen_y = self.GRID_Y + grid_y * self.CELL_SIZE
        
        rect = pygame.Rect(screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.draw.rect(surface, color, rect, width=2, border_radius=3)
        else:
            pygame.draw.rect(surface, color, rect, border_radius=4)
            highlight_color = tuple(min(255, c + 40) for c in color)
            pygame.draw.rect(surface, highlight_color, rect.inflate(-self.CELL_SIZE*0.5, -self.CELL_SIZE*0.5), border_radius=3)
            pygame.draw.rect(surface, self.COLOR_BG, rect, width=1, border_radius=4)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE", True, self.COLOR_TEXT)
        score_val = self.font_main.render(f"{self.score:06}", True, (255, 255, 255))
        self.screen.blit(score_text, (30, 30))
        self.screen.blit(score_val, (30, 60))

        time_text = self.font_main.render(f"TIME", True, self.COLOR_TEXT)
        secs = max(0, self.time_remaining // self.FPS)
        time_val = self.font_main.render(f"{secs:02}", True, (255, 255, 255))
        self.screen.blit(time_text, (self.screen_width - 120, 30))
        self.screen.blit(time_val, (self.screen_width - 120, 60))

        next_text = self.font_main.render(f"NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.screen_width - 180, 150))
        if self.next_piece:
            center_x = self.screen_width - 140
            center_y = 220
            for dx, dy in self.next_piece['shape']:
                x = center_x + dx * self.CELL_SIZE
                y = center_y + dy * self.CELL_SIZE
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.next_piece['color'], rect, border_radius=4)
                pygame.draw.rect(self.screen, self.COLOR_BG, rect, width=1, border_radius=4)

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.time_remaining <= 0 else "GAME OVER"
            end_text = self.font_main.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": max(0, self.time_remaining // self.FPS),
        }

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Tetris Survival")
    
    running = True
    total_reward = 0
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        # --- Human Input Processing ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = np.array([movement, space, shift])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(env.FPS)
        
    env.close()