
# Generated: 2025-08-27T18:35:21.965552
# Source Brief: brief_01878.md
# Brief Index: 1878

        
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
        "Controls: ←→ to move, ↑ to rotate clockwise, ↓ to soft drop. "
        "Hold Shift to rotate counter-clockwise. Press Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A falling block puzzle. Clear 10 lines to win. Don't let the stack reach the top!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.BLOCK_SIZE = 18
        self.MAX_STEPS = 2500

        # Positioning
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2 - 100
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2

        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PLACED = (100, 100, 120)
        self.PIECE_COLORS = [
            (0, 0, 0),          # 0: Empty
            (0, 240, 240),      # 1: I (Cyan)
            (0, 0, 240),        # 2: J (Blue)
            (240, 160, 0),      # 3: L (Orange)
            (240, 240, 0),      # 4: O (Yellow)
            (0, 240, 0),        # 5: S (Green)
            (160, 0, 240),      # 6: T (Purple)
            (240, 0, 0),        # 7: Z (Red)
        ]

        # Tetromino shapes [shape][rotation]
        self.PIECE_SHAPES = {
            1: [[(0, 1), (1, 1), (2, 1), (3, 1)], [(1, 0), (1, 1), (1, 2), (1, 3)]],  # I
            2: [[(0, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (2, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (2, 1), (2, 2)], [(1, 0), (1, 1), (0, 2), (1, 2)]],  # J
            3: [[(2, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (1, 2), (2, 2)], [(0, 1), (1, 1), (2, 1), (0, 2)], [(0, 0), (1, 0), (1, 1), (1, 2)]],  # L
            4: [[(1, 0), (2, 0), (1, 1), (2, 1)]],  # O
            5: [[(1, 0), (2, 0), (0, 1), (1, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)]],  # S
            6: [[(1, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (0, 1), (1, 1), (1, 2)], [(0, 1), (1, 1), (2, 1), (1, 2)], [(1, 0), (1, 1), (2, 1), (1, 2)]],  # T
            7: [[(0, 0), (1, 0), (1, 1), (2, 1)], [(2, 0), (1, 1), (2, 1), (1, 2)]],  # Z
        }
        
        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # --- State Variables ---
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.piece_bag = []
        self.fall_time = None
        self.fall_counter = None
        self.score = None
        self.lines_cleared = None
        self.steps = None
        self.game_over = None
        self.particles = []
        self.last_action = np.array([0, 0, 0])
        self.np_random = None

        # Initialize state
        self.reset()
        
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        
        self.fall_time = 1.0
        self.fall_counter = 0.0
        
        self.particles = []
        self.piece_bag = []
        self._fill_piece_bag()
        
        self.current_piece = None
        self.next_piece = self._create_new_piece()
        self._spawn_new_piece()
        
        self.last_action = np.array([0, 0, 0])
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty per step to encourage speed

        # --- Action Handling ---
        movement, space_btn, shift_btn = action[0], action[1] == 1, action[2] == 1
        
        # Debounced actions (trigger on press)
        rot_cw = movement == 1 and self.last_action[0] != 1
        hard_drop = space_btn and not self.last_action[1]
        rot_ccw = shift_btn and not self.last_action[2]
        
        # Continuous actions
        move_dx = 0
        if movement == 3: move_dx = -1
        elif movement == 4: move_dx = 1
        soft_drop = movement == 2

        self.last_action = action

        if rot_cw: self._rotate_piece(1)
        if rot_ccw: self._rotate_piece(-1)
        if move_dx != 0: self._move_piece(move_dx, 0)
        
        piece_locked = False
        if hard_drop:
            # sfx: hard_drop_sound
            lines_dropped = self._hard_drop()
            reward += lines_dropped * 0.01 # Small reward for distance
            piece_locked = True
        else:
            # --- Game Logic (Gravity) ---
            self.fall_counter += 1 / 30.0  # Fixed timestep for 30fps
            drop_interval = self.fall_time / 5.0 if soft_drop else self.fall_time
            if self.fall_counter >= drop_interval:
                self.fall_counter = 0
                if not self._move_piece(0, 1):
                    # sfx: piece_land_sound
                    piece_locked = True

        # --- Lock Piece and Check Lines ---
        if piece_locked:
            placement_reward = self._calculate_placement_reward()
            reward += placement_reward
            self._lock_piece()
            
            lines = self._clear_lines()
            if lines > 0:
                # sfx: line_clear_sound
                self.lines_cleared += lines
                line_rewards = {1: 1, 2: 3, 3: 5, 4: 8}
                score_gain = line_rewards[lines] * 10
                self.score += score_gain
                reward += line_rewards[lines]

                # Increase difficulty
                self.fall_time = max(0.1, 1.0 - (self.lines_cleared // 2) * 0.05)

            self._spawn_new_piece()

        # --- Termination Check ---
        terminated = self.game_over or self.lines_cleared >= 10 or self.steps >= self.MAX_STEPS
        if terminated:
            if self.lines_cleared >= 10:
                reward += 100  # Win bonus
            elif self.game_over:
                reward -= 100  # Lose penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Helper Functions: Game Logic ---

    def _fill_piece_bag(self):
        bag = list(self.PIECE_SHAPES.keys())
        self.np_random.shuffle(bag)
        self.piece_bag.extend(bag)

    def _create_new_piece(self):
        if len(self.piece_bag) < 2:
            self._fill_piece_bag()
        shape_idx = self.piece_bag.pop(0)
        return {
            "shape": shape_idx,
            "rotation": 0,
            "x": self.GRID_WIDTH // 2 - 2,
            "y": 0,
            "color_idx": shape_idx,
        }

    def _spawn_new_piece(self):
        self.current_piece = self.next_piece
        self.next_piece = self._create_new_piece()
        self.current_piece['x'] = self.GRID_WIDTH // 2 - 2
        self.current_piece['y'] = 0 if self.current_piece['shape'] != 1 else -1 # I piece spawns higher

        if self._check_collision(self.current_piece):
            self.game_over = True

    def _move_piece(self, dx, dy):
        test_piece = self.current_piece.copy()
        test_piece['x'] += dx
        test_piece['y'] += dy
        if not self._check_collision(test_piece):
            self.current_piece = test_piece
            return True
        return False

    def _rotate_piece(self, direction):
        if not self.current_piece: return
        p = self.current_piece
        num_rotations = len(self.PIECE_SHAPES[p['shape']])
        if num_rotations <= 1: return

        test_piece = p.copy()
        test_piece['rotation'] = (p['rotation'] + direction) % num_rotations

        # Wall kick checks
        for kick_dx in [0, -1, 1, -2, 2]:
            test_piece['x'] = p['x'] + kick_dx
            if not self._check_collision(test_piece):
                self.current_piece = test_piece
                # sfx: rotate_sound
                return
    
    def _hard_drop(self):
        if not self.current_piece: return 0
        start_y = self.current_piece['y']
        while self._move_piece(0, 1):
            pass
        return self.current_piece['y'] - start_y

    def _check_collision(self, piece):
        shape_coords = self.PIECE_SHAPES[piece['shape']][piece['rotation']]
        for x_off, y_off in shape_coords:
            x, y = piece['x'] + x_off, piece['y'] + y_off
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return True
            if y >= 0 and self.grid[y, x] != 0:
                return True
        return False

    def _lock_piece(self):
        p = self.current_piece
        shape_coords = self.PIECE_SHAPES[p['shape']][p['rotation']]
        for x_off, y_off in shape_coords:
            x, y = p['x'] + x_off, p['y'] + y_off
            if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                self.grid[y, x] = p['color_idx']

    def _clear_lines(self):
        lines_to_clear = [y for y in range(self.GRID_HEIGHT) if np.all(self.grid[y, :] != 0)]
        if not lines_to_clear:
            return 0
        
        for y in lines_to_clear:
            self.grid[1:y+1, :] = self.grid[0:y, :]
            self.grid[0, :] = 0
            # Create particle effects
            for i in range(30):
                self._create_particle(self.GRID_X + self.np_random.random() * self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_Y + y * self.BLOCK_SIZE)

        return len(lines_to_clear)

    def _calculate_placement_reward(self):
        is_risky = False
        p = self.current_piece
        shape_coords = self.PIECE_SHAPES[p['shape']][p['rotation']]
        
        temp_grid = self.grid.copy()
        # Temporarily place piece to check for gaps
        for x_off, y_off in shape_coords:
            x, y = p['x'] + x_off, p['y'] + y_off
            if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                temp_grid[y, x] = p['color_idx']

        for x_off, y_off in shape_coords:
            x, y = p['x'] + x_off, p['y'] + y_off
            if y + 1 < self.GRID_HEIGHT and temp_grid[y + 1, x] == 0:
                is_risky = True
                break
        
        return 2.0 if is_risky else -0.2

    # --- Helper Functions: Rendering ---
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._draw_placed_blocks()
        if not self.game_over and self.current_piece:
            self._draw_ghost_piece()
            self._draw_piece(self.current_piece)
        self._update_and_draw_particles()
        self._draw_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_grid(self):
        # Grid border
        border_rect = pygame.Rect(self.GRID_X - 2, self.GRID_Y - 2, self.GRID_WIDTH * self.BLOCK_SIZE + 4, self.GRID_HEIGHT * self.BLOCK_SIZE + 4)
        pygame.draw.rect(self.screen, self.COLOR_GRID, border_rect, 0, 5)
        
        # Grid background
        bg_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BG, bg_rect)
        
        # Grid lines
        for x in range(1, self.GRID_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X + x * self.BLOCK_SIZE, self.GRID_Y), (self.GRID_X + x * self.BLOCK_SIZE, self.GRID_Y + self.GRID_HEIGHT * self.BLOCK_SIZE))
        for y in range(1, self.GRID_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y + y * self.BLOCK_SIZE), (self.GRID_X + self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_Y + y * self.BLOCK_SIZE))

    def _draw_placed_blocks(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != 0:
                    color = self.PIECE_COLORS[self.grid[y, x]]
                    self._draw_block(x, y, color, self.GRID_X, self.GRID_Y)

    def _draw_piece(self, piece, grid_x_offset=None, grid_y_offset=None, alpha=255):
        if not piece: return
        grid_x_offset = grid_x_offset if grid_x_offset is not None else self.GRID_X
        grid_y_offset = grid_y_offset if grid_y_offset is not None else self.GRID_Y
        
        shape_coords = self.PIECE_SHAPES[piece['shape']][piece['rotation']]
        color = self.PIECE_COLORS[piece['color_idx']]
        
        for x_off, y_off in shape_coords:
            x, y = piece['x'] + x_off, piece['y'] + y_off
            if y >= 0: # Don't draw blocks above the ceiling
                self._draw_block(x, y, color, grid_x_offset, grid_y_offset, alpha)

    def _draw_ghost_piece(self):
        ghost_piece = self.current_piece.copy()
        ghost_piece['y'] = self._get_ghost_y()
        self._draw_piece(ghost_piece, alpha=50)

    def _get_ghost_y(self):
        test_piece = self.current_piece.copy()
        while not self._check_collision(test_piece):
            test_piece['y'] += 1
        return test_piece['y'] - 1

    def _draw_block(self, x, y, color, grid_x, grid_y, alpha=255):
        rect = pygame.Rect(grid_x + x * self.BLOCK_SIZE, grid_y + y * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        if alpha < 255: # For ghost piece
            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(surf, (*color, alpha), (0, 0, *rect.size), 0, 3)
            pygame.draw.rect(surf, (*[min(255, c+50) for c in color], alpha), (0, 0, *rect.size), 2, 3)
            self.screen.blit(surf, rect.topleft)
        else: # For solid pieces
            pygame.draw.rect(self.screen, color, rect, 0, 3)
            # Add a subtle highlight for 3D effect
            highlight_color = tuple(min(255, c + 40) for c in color)
            pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.topright, 2)
            pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.bottomleft, 2)
            shadow_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.line(self.screen, shadow_color, rect.bottomright, rect.topright, 2)
            pygame.draw.line(self.screen, shadow_color, rect.bottomright, rect.bottomleft, 2)

    def _draw_ui(self):
        ui_x = self.GRID_X + self.GRID_WIDTH * self.BLOCK_SIZE + 40
        
        # --- Next Piece ---
        next_text = self.font_main.render("NEXT", True, (255, 255, 255))
        self.screen.blit(next_text, (ui_x, self.GRID_Y))
        
        next_piece_copy = self.next_piece.copy()
        next_piece_copy['x'], next_piece_copy['y'] = 0, 0
        self._draw_piece(next_piece_copy, grid_x_offset=ui_x, grid_y_offset=self.GRID_Y + 40)
        
        # --- Score ---
        score_text = self.font_main.render("SCORE", True, (255, 255, 255))
        self.screen.blit(score_text, (ui_x, self.GRID_Y + 150))
        score_val = self.font_small.render(f"{self.score:06d}", True, (200, 200, 255))
        self.screen.blit(score_val, (ui_x, self.GRID_Y + 180))

        # --- Lines ---
        lines_text = self.font_main.render("LINES", True, (255, 255, 255))
        self.screen.blit(lines_text, (ui_x, self.GRID_Y + 220))
        lines_val = self.font_small.render(f"{self.lines_cleared} / 10", True, (200, 200, 255))
        self.screen.blit(lines_val, (ui_x, self.GRID_Y + 250))

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            msg = "GAME OVER" if self.lines_cleared < 10 else "YOU WIN!"
            end_text = self.font_main.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particle(self, x, y):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        color = self.np_random.choice(self.PIECE_COLORS[1:], 1)[0]
        self.particles.append({
            "pos": [x, y],
            "vel": vel,
            "alpha": 255,
            "size": self.np_random.uniform(2, 6),
            "color": color
        })

    def _update_and_draw_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['alpha'] -= 5
            p['size'] *= 0.98
            
            if p['alpha'] > 0 and p['size'] > 1:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), (*p['color'], int(p['alpha']))
                )
        self.particles = [p for p in self.particles if p['alpha'] > 0 and p['size'] > 1]
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Falling Block Puzzle")
    clock = pygame.time.Clock()
    total_reward = 0

    while not done:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()