
# Generated: 2025-08-28T02:52:04.301383
# Source Brief: brief_04594.md
# Brief Index: 4594

        
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
        "Controls: ←→ to move, ↑↓ to rotate. Hold Shift for soft drop, press Space for hard drop."
    )

    game_description = (
        "A fast-paced, grid-based falling block puzzle game where strategic rotations and risky placements are rewarded."
    )

    auto_advance = True

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAY_AREA_WIDTH = 250
    CELL_SIZE = PLAY_AREA_WIDTH // GRID_WIDTH
    PLAY_AREA_HEIGHT = CELL_SIZE * GRID_HEIGHT
    SIDE_PANEL_WIDTH = 150

    PLAY_AREA_X = (SCREEN_WIDTH - PLAY_AREA_WIDTH - SIDE_PANEL_WIDTH - 20) // 2
    PLAY_AREA_Y = (SCREEN_HEIGHT - PLAY_AREA_HEIGHT) // 2

    SIDE_PANEL_X = PLAY_AREA_X + PLAY_AREA_WIDTH + 20

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID_BG = (40, 40, 55)
    COLOR_GRID_LINES = (60, 60, 80)
    COLOR_TEXT = (220, 220, 240)
    COLOR_GHOST = (255, 255, 255, 50)
    COLOR_WHITE = (255, 255, 255)

    TETROMINOES = {
        'I': {'shape': [[1, 1, 1, 1]], 'color': (50, 220, 220)},
        'O': {'shape': [[1, 1], [1, 1]], 'color': (240, 220, 80)},
        'T': {'shape': [[0, 1, 0], [1, 1, 1]], 'color': (180, 80, 240)},
        'L': {'shape': [[0, 0, 1], [1, 1, 1]], 'color': (240, 160, 50)},
        'J': {'shape': [[1, 0, 0], [1, 1, 1]], 'color': (80, 100, 240)},
        'S': {'shape': [[0, 1, 1], [1, 1, 0]], 'color': (80, 240, 90)},
        'Z': {'shape': [[1, 1, 0], [0, 1, 1]], 'color': (240, 80, 80)}
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        try:
            self.font_large = pygame.font.Font(None, 36)
            self.font_medium = pygame.font.Font(None, 28)
            self.font_small = pygame.font.Font(None, 22)
        except pygame.error:
            self.font_large = pygame.font.SysFont("sans-serif", 36)
            self.font_medium = pygame.font.SysFont("sans-serif", 28)
            self.font_small = pygame.font.SysFont("sans-serif", 22)

        self.game_state_vars = [
            'grid', 'score', 'lines_cleared', 'steps', 'game_over',
            'current_piece', 'next_piece', 'fall_timer', 'fall_speed',
            'lines_to_clear', 'line_clear_animation_timer', 'last_action',
            'particles'
        ]
        for var in self.game_state_vars:
            setattr(self, var, None)

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = [[(0, 0, 0) for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False

        self.piece_sequence = list(self.TETROMINOES.keys())
        random.shuffle(self.piece_sequence)
        self.piece_idx = 0

        self._spawn_new_piece()
        self._spawn_new_piece()

        self.fall_timer = 0
        self.fall_speed = 15  # 30fps / 2 -> 0.5s per drop
        self.lines_to_clear = []
        self.line_clear_animation_timer = 0
        self.last_action = np.array([0, 0, 0])
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Time penalty
        self.steps += 1

        if self.game_over:
            terminated = self._check_termination()
            return self._get_observation(), -100, terminated, False, self._get_info()

        if self.line_clear_animation_timer > 0:
            self.line_clear_animation_timer -= 1
            if self.line_clear_animation_timer == 0:
                self._execute_line_clear()
        else:
            reward += self._handle_input(action)
            reward += self._update_physics(action)

        self.last_action = action
        terminated = self._check_termination()
        
        if terminated and not self.game_over: # Win condition
            reward += 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        if self.current_piece is None:
            return 0

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        last_movement, last_space_held, _ = self.last_action[0], self.last_action[1] == 1, self.last_action[2] == 1
        
        # --- Handle one-shot actions (Rotation, Hard Drop) ---
        # These only trigger on the rising edge (0 -> 1)
        moved = False
        if movement != 0 and movement != last_movement:
            if movement == 1: # Rotate CW
                self._rotate_piece(1)
            elif movement == 2: # Rotate CCW
                self._rotate_piece(-1)
        
        # --- Handle continuous actions (Left/Right) ---
        if movement == 3: # Move Left
            self.current_piece['x'] -= 1
            if self._check_collision(self.current_piece['shape'], (self.current_piece['x'], self.current_piece['y'])):
                self.current_piece['x'] += 1
        elif movement == 4: # Move Right
            self.current_piece['x'] += 1
            if self._check_collision(self.current_piece['shape'], (self.current_piece['x'], self.current_piece['y'])):
                self.current_piece['x'] -= 1

        if space_held and not last_space_held:
            return self._hard_drop()
        
        return 0

    def _update_physics(self, action):
        if self.current_piece is None:
            return 0

        _, _, shift_held = action
        reward = 0

        fall_increment = 3 if shift_held else 1
        if shift_held:
            reward += 0.01 # Reward for soft dropping
        self.fall_timer += fall_increment

        if self.fall_timer >= self.fall_speed:
            self.fall_timer = 0
            self.current_piece['y'] += 1
            if self._check_collision(self.current_piece['shape'], (self.current_piece['x'], self.current_piece['y'])):
                self.current_piece['y'] -= 1
                reward += self._lock_piece()
        
        return reward

    def _lock_piece(self):
        if self.current_piece is None: return 0
        reward = 0
        
        # Calculate stack height penalty
        prev_height = self._get_stack_height()

        shape = self.current_piece['shape']
        px, py = self.current_piece['x'], self.current_piece['y']
        color = self.current_piece['color']
        
        min_y = self.GRID_HEIGHT
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = px + x, py + y
                    if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
                        self.grid[grid_y][grid_x] = color
                        min_y = min(min_y, grid_y)
        
        # Risky play reward
        if min_y <= 1:
            reward += 2.0

        # Stack height penalty
        new_height = self._get_stack_height()
        if new_height > prev_height:
            reward -= 0.2

        # Check for line clears
        lines_cleared_this_turn = 0
        full_rows = []
        for r_idx, row in enumerate(self.grid):
            if all(cell != (0,0,0) for cell in row):
                full_rows.append(r_idx)
                lines_cleared_this_turn += 1
        
        if lines_cleared_this_turn > 0:
            reward += lines_cleared_this_turn * 1.0
            if lines_cleared_this_turn > 1:
                reward += 1.5 # Multi-line bonus
            self.lines_to_clear = full_rows
            self.line_clear_animation_timer = 8 # frames
            # sound: line_clear.wav
            for r_idx in full_rows:
                for c_idx in range(self.GRID_WIDTH):
                    self._create_particles(c_idx, r_idx, self.grid[r_idx][c_idx])

        self.score += int(reward * 10) # Update score based on reward
        self._spawn_new_piece()

        # Check for game over
        if self._check_collision(self.current_piece['shape'], (self.current_piece['x'], self.current_piece['y'])):
            self.game_over = True
            self.current_piece = None
            # sound: game_over.wav
            reward -= 100
        else:
            # sound: piece_lock.wav
            pass
        
        return reward

    def _hard_drop(self):
        if self.current_piece is None: return 0
        
        ghost_y = self._get_ghost_piece_y()
        # Create trail particles
        for y in range(self.current_piece['y'], ghost_y):
            if y % 2 == 0:
                for x_offset, _ in enumerate(self.current_piece['shape'][0]):
                    px = self.current_piece['x'] + x_offset
                    self.particles.append([
                        self.PLAY_AREA_X + px * self.CELL_SIZE + self.CELL_SIZE/2,
                        self.PLAY_AREA_Y + y * self.CELL_SIZE + self.CELL_SIZE/2,
                        (random.uniform(-0.5, 0.5), random.uniform(-1, 0)),
                        self.current_piece['color'],
                        random.randint(10, 20)
                    ])

        self.current_piece['y'] = ghost_y
        return self._lock_piece()

    def _execute_line_clear(self):
        self.lines_cleared += len(self.lines_to_clear)
        self.score += len(self.lines_to_clear) * 100 # Bonus score for clears
        
        for r_idx in sorted(self.lines_to_clear, reverse=True):
            self.grid.pop(r_idx)
            self.grid.insert(0, [(0, 0, 0) for _ in range(self.GRID_WIDTH)])
        self.lines_to_clear = []

    def _spawn_new_piece(self):
        self.current_piece = self.next_piece
        
        if self.piece_idx >= len(self.piece_sequence):
            self.piece_idx = 0
            random.shuffle(self.piece_sequence)

        next_shape_key = self.piece_sequence[self.piece_idx]
        self.piece_idx += 1
        
        shape_info = self.TETROMINOES[next_shape_key]
        shape = shape_info['shape']
        
        self.next_piece = {
            'shape': shape,
            'color': shape_info['color'],
            'x': self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0
        }

    def _rotate_piece(self, direction):
        if self.current_piece is None: return
        
        shape = self.current_piece['shape']
        # Transpose and reverse rows for rotation
        if direction == 1: # Clockwise
            new_shape = [list(row) for row in zip(*shape[::-1])]
        else: # Counter-clockwise
            new_shape = [list(row) for row in zip(*shape)][::-1]

        # Wall kick logic
        original_x = self.current_piece['x']
        for offset in [0, 1, -1, 2, -2]:
            self.current_piece['x'] = original_x + offset
            if not self._check_collision(new_shape, (self.current_piece['x'], self.current_piece['y'])):
                self.current_piece['shape'] = new_shape
                # sound: rotate.wav
                return
        self.current_piece['x'] = original_x # Revert if no valid position found
        
    def _check_collision(self, shape, offset):
        off_x, off_y = offset
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = off_x + x, off_y + y
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return True
                    if self.grid[grid_y][grid_x] != (0, 0, 0):
                        return True
        return False

    def _get_stack_height(self):
        for y, row in enumerate(self.grid):
            if any(cell != (0,0,0) for cell in row):
                return self.GRID_HEIGHT - y
        return 0

    def _get_ghost_piece_y(self):
        if self.current_piece is None: return 0
        y = self.current_piece['y']
        while not self._check_collision(self.current_piece['shape'], (self.current_piece['x'], y + 1)):
            y += 1
        return y
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_play_area()
        self._render_ui()
        self._update_and_draw_particles()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_play_area(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, 
                         (self.PLAY_AREA_X, self.PLAY_AREA_Y, self.PLAY_AREA_WIDTH, self.PLAY_AREA_HEIGHT))

        # Draw placed blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] != (0, 0, 0):
                    self._draw_block(x, y, self.grid[y][x])
        
        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_y = self._get_ghost_piece_y()
            self._draw_piece(self.current_piece, (self.current_piece['x'], ghost_y), self.COLOR_GHOST)

        # Draw current piece
        if self.current_piece and not self.game_over:
            self._draw_piece(self.current_piece, (self.current_piece['x'], self.current_piece['y']), self.current_piece['color'])

        # Draw line clear animation
        if self.line_clear_animation_timer > 0:
            alpha = 255 * (self.line_clear_animation_timer / 8)
            flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            for y in self.lines_to_clear:
                for x in range(self.GRID_WIDTH):
                    self.screen.blit(flash_surface, (self.PLAY_AREA_X + x * self.CELL_SIZE, self.PLAY_AREA_Y + y * self.CELL_SIZE))

        # Draw grid lines
        for x in range(1, self.GRID_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES,
                             (self.PLAY_AREA_X + x * self.CELL_SIZE, self.PLAY_AREA_Y),
                             (self.PLAY_AREA_X + x * self.CELL_SIZE, self.PLAY_AREA_Y + self.PLAY_AREA_HEIGHT))
        for y in range(1, self.GRID_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES,
                             (self.PLAY_AREA_X, self.PLAY_AREA_Y + y * self.CELL_SIZE),
                             (self.PLAY_AREA_X + self.PLAY_AREA_WIDTH, self.PLAY_AREA_Y + y * self.CELL_SIZE))
        
        # Draw border
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, 
                         (self.PLAY_AREA_X, self.PLAY_AREA_Y, self.PLAY_AREA_WIDTH, self.PLAY_AREA_HEIGHT), 2)

    def _render_ui(self):
        # --- Score Box ---
        score_text = self.font_medium.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SIDE_PANEL_X, self.PLAY_AREA_Y))
        score_val = self.font_large.render(f"{self.score:06d}", True, self.COLOR_WHITE)
        self.screen.blit(score_val, (self.SIDE_PANEL_X, self.PLAY_AREA_Y + 30))

        # --- Lines Box ---
        lines_text = self.font_medium.render("LINES", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (self.SIDE_PANEL_X, self.PLAY_AREA_Y + 90))
        lines_val = self.font_large.render(f"{self.lines_cleared} / 15", True, self.COLOR_WHITE)
        self.screen.blit(lines_val, (self.SIDE_PANEL_X, self.PLAY_AREA_Y + 120))

        # --- Next Piece Box ---
        next_text = self.font_medium.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.SIDE_PANEL_X, self.PLAY_AREA_Y + 180))
        
        next_box_rect = (self.SIDE_PANEL_X, self.PLAY_AREA_Y + 210, 4 * self.CELL_SIZE, 4 * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, next_box_rect)

        if self.next_piece:
            shape = self.next_piece['shape']
            w, h = len(shape[0]), len(shape)
            off_x = (4 - w) / 2
            off_y = (4 - h) / 2
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        self._draw_block(x + off_x, y + off_y, self.next_piece['color'], 
                                         base_pos=(next_box_rect[0], next_box_rect[1]))

        # --- Game Over Text ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "YOU WIN!" if self.lines_cleared >= 15 else "GAME OVER"
            text_surf = self.font_large.render(status_text, True, self.COLOR_WHITE)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _draw_piece(self, piece, offset, color):
        px, py = offset
        for y, row in enumerate(piece['shape']):
            for x, cell in enumerate(row):
                if cell:
                    self._draw_block(px + x, py + y, color)
    
    def _draw_block(self, x, y, color, base_pos=None):
        if base_pos is None:
            base_pos = (self.PLAY_AREA_X, self.PLAY_AREA_Y)
            
        is_ghost = len(color) == 4
        
        rect = pygame.Rect(
            base_pos[0] + x * self.CELL_SIZE,
            base_pos[1] + y * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 0)
        else:
            inner_rect = rect.inflate(-4, -4)
            highlight_color = tuple(min(255, c + 50) for c in color)
            shadow_color = tuple(max(0, c - 50) for c in color)
            
            pygame.draw.rect(self.screen, highlight_color, rect, 0, border_radius=4)
            pygame.draw.rect(self.screen, shadow_color, rect.move(0, 2), 0, border_radius=4)
            pygame.draw.rect(self.screen, color, inner_rect, 0, border_radius=3)
            
    def _create_particles(self, grid_x, grid_y, color):
        px = self.PLAY_AREA_X + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.PLAY_AREA_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(5):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append([px, py, vel, color, random.randint(15, 30)])

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p[0] += p[2][0]
            p[1] += p[2][1]
            p[4] -= 1
            if p[4] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p[4] / 30))
                color = p[3] + (alpha,)
                size = int(p[4] / 5)
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), size, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }

    def _check_termination(self):
        return self.game_over or self.lines_cleared >= 15 or self.steps >= 1000

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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Map keyboard keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Pygame setup for human play
    pygame.display.set_caption("Gymnasium Block Puzzle")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    while running:
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # This input handling is continuous, not edge-triggered like the agent's
        # It's a simplification for human testing
        for key, move_val in key_map.items():
            if keys[key]:
                movement = move_val
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = np.array([movement, space, shift])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Lines: {info['lines_cleared']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            env.reset()
            
        env.clock.tick(30) # Run at 30 FPS

    pygame.quit()