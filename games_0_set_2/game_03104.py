
# Generated: 2025-08-27T22:23:20.637341
# Source Brief: brief_03104.md
# Brief Index: 3104

        
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
        "Controls: ←→ to move, ↓ to soft drop, ↑ to rotate. "
        "Press space to hard drop. Press shift to hold/swap piece."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically place falling blocks in a grid to clear rows and achieve the "
        "target score before the stack reaches the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_X_OFFSET, GRID_Y_OFFSET = 220, 0
    CELL_SIZE = 20

    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GAMEOVER = (255, 50, 50)
    COLOR_WIN = (50, 255, 50)

    PIECE_COLORS = [
        (0, 255, 255),    # I (Cyan)
        (255, 255, 0),    # O (Yellow)
        (128, 0, 128),    # T (Purple)
        (0, 0, 255),      # J (Blue)
        (255, 165, 0),    # L (Orange)
        (0, 255, 0),      # S (Green)
        (255, 0, 0),      # Z (Red)
    ]
    
    SHAPES = [
        [[1, 1, 1, 1]],         # I
        [[1, 1], [1, 1]],       # O
        [[0, 1, 0], [1, 1, 1]], # T
        [[1, 0, 0], [1, 1, 1]], # J
        [[0, 0, 1], [1, 1, 1]], # L
        [[0, 1, 1], [1, 1, 0]], # S
        [[1, 1, 0], [0, 1, 1]]  # Z
    ]

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
        self.font_small = pygame.font.Font(None, 24)
        
        self.grid_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))

        self.reset()
        
        self.validate_implementation()

    def _create_rotated_shapes(self):
        rotated_shapes = []
        for shape in self.SHAPES:
            rotations = [np.array(shape, dtype=int)]
            current_shape = rotations[0]
            for _ in range(3):
                current_shape = np.rot90(current_shape)
                if not any(np.array_equal(current_shape, r) for r in rotations):
                    rotations.append(current_shape)
            rotated_shapes.append(rotations)
        return rotated_shapes

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.rows_cleared = 0
        self.game_over = False
        self.win = False
        
        self.rotated_shapes = self._create_rotated_shapes()
        self.bag = list(range(len(self.SHAPES)))
        random.shuffle(self.bag)

        self.current_piece = None
        self.next_piece_shape_idx = self._get_next_from_bag()
        self._spawn_piece()

        self.held_piece_shape_idx = None
        self.can_swap = True
        
        self.fall_timer = 0
        self.base_fall_delay = 30 
        
        self.move_timer = 0
        self.move_delay = 5
        self.move_initial_delay = 15

        self.lock_timer = 0
        self.lock_delay = 15

        self.particles = []
        self.last_action = np.array([0, 0, 0])
        
        return self._get_observation(), self._get_info()

    def _get_next_from_bag(self):
        if not self.bag:
            self.bag = list(range(len(self.SHAPES)))
            random.shuffle(self.bag)
        return self.bag.pop()

    def _spawn_piece(self):
        self.current_piece = {
            'shape_idx': self.next_piece_shape_idx,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - 1,
            'y': 0,
            'color_idx': self.next_piece_shape_idx + 1
        }
        self.next_piece_shape_idx = self._get_next_from_bag()
        self.can_swap = True

        if not self._is_valid_position(self.current_piece):
            self.game_over = True

    def _is_valid_position(self, piece, offset_x=0, offset_y=0):
        shape = self.rotated_shapes[piece['shape_idx']][piece['rotation']]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    x = piece['x'] + c + offset_x
                    y = piece['y'] + r + offset_y
                    if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT and self.grid[y, x] == 0):
                        return False
        return True

    def step(self, action):
        reward = -0.01
        self.steps += 1

        if not self.game_over and not self.win:
            placement_reward = self._handle_input(action)
            game_update_reward = self._update_game_state()
            reward += placement_reward + game_update_reward
        
        if self.rows_cleared >= 10:
            self.win = True
        
        terminated = self.game_over or self.win or self.steps >= 1000
        if terminated:
            if self.win:
                reward += 100
            elif self.game_over:
                reward -= 100
        
        self.last_action = action
        
        return (self._get_observation(), reward, terminated, False, self._get_info())

    def _handle_input(self, action):
        movement, space, shift = action[0], action[1], action[2]
        is_press = lambda i, val: action[i] == val and self.last_action[i] != val
        
        if is_press(0, 1): self._rotate_piece()
        
        move_dir = 0
        if movement == 3: move_dir = -1
        if movement == 4: move_dir = 1
        
        is_move_press = movement in [3, 4] and self.last_action[0] != movement
        is_move_hold = movement in [3, 4] and self.last_action[0] == movement
        
        if is_move_press:
            self._move_piece(move_dir)
            self.move_timer = 0
        elif is_move_hold:
            self.move_timer += 1
            if self.move_timer > self.move_initial_delay and (self.move_timer - self.move_initial_delay) % self.move_delay == 0:
                self._move_piece(move_dir)
        else:
            self.move_timer = 0

        if movement == 2: self.fall_timer += 5

        if is_press(1, 1): return self._hard_drop() # Sound: Hard drop
        
        if is_press(2, 1): self._hold_piece() # Sound: Hold piece
        
        return 0.0

    def _update_game_state(self):
        reward = 0.0
        self.fall_timer += 1
        
        fall_speed_multiplier = 1.0 + (self.rows_cleared // 2) * 0.05
        current_fall_delay = self.base_fall_delay / fall_speed_multiplier

        is_on_ground = not self._is_valid_position(self.current_piece, offset_y=1)

        if is_on_ground: self.lock_timer += 1
        else: self.lock_timer = 0
        
        if self.fall_timer >= current_fall_delay:
            self.fall_timer = 0
            if not is_on_ground:
                self.current_piece['y'] += 1 # Sound: piece moves down
            else:
                self.lock_timer += 5 

        if is_on_ground and self.lock_timer >= self.lock_delay:
            reward += self._lock_piece()

        self._update_particles()
        return reward

    def _move_piece(self, dx):
        if self._is_valid_position(self.current_piece, offset_x=dx):
            self.current_piece['x'] += dx
            self.lock_timer = 0 # Sound: piece moves sideways

    def _rotate_piece(self):
        original_rotation = self.current_piece['rotation']
        num_rotations = len(self.rotated_shapes[self.current_piece['shape_idx']])
        self.current_piece['rotation'] = (self.current_piece['rotation'] + 1) % num_rotations
        
        for kick_x in [0, -1, 1, -2, 2]: # Wall kick
            if self._is_valid_position(self.current_piece, offset_x=kick_x):
                self.current_piece['x'] += kick_x
                self.lock_timer = 0 # Sound: piece rotates
                return
        
        self.current_piece['rotation'] = original_rotation

    def _hard_drop(self):
        while self._is_valid_position(self.current_piece, offset_y=1):
            self.current_piece['y'] += 1
        return self._lock_piece()

    def _hold_piece(self):
        if not self.can_swap: return
        
        if self.held_piece_shape_idx is None:
            self.held_piece_shape_idx = self.current_piece['shape_idx']
            self._spawn_piece()
        else:
            held_idx_temp, self.held_piece_shape_idx = self.held_piece_shape_idx, self.current_piece['shape_idx']
            self.next_piece_shape_idx = self._get_next_from_bag()
            self._spawn_piece()
            self.next_piece_shape_idx = self.current_piece['shape_idx'] # to avoid skipping a piece
            self.current_piece['shape_idx'] = held_idx_temp
            self.current_piece['color_idx'] = held_idx_temp + 1

        self.can_swap = False

    def _lock_piece(self):
        shape = self.rotated_shapes[self.current_piece['shape_idx']][self.current_piece['rotation']]
        px, py = self.current_piece['x'], self.current_piece['y']
        
        num_supporting_blocks = 0
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self.grid[py + r, px + c] = self.current_piece['color_idx']
                    if py + r + 1 >= self.GRID_HEIGHT or self.grid[py + r + 1, px + c] != 0:
                        num_supporting_blocks += 1

        # Sound: piece locks
        reward = 0
        if num_supporting_blocks == 1: reward += 2
        elif num_supporting_blocks >= 2: reward -= 0.2

        reward += self._clear_rows()
        self._spawn_piece()
        self.lock_timer = 0
        return reward

    def _clear_rows(self):
        rows_to_clear = [r for r, row in enumerate(self.grid) if np.all(row)]
        if not rows_to_clear: return 0
        
        # Sound: line clear
        for r in rows_to_clear:
            self._create_clear_particles(r)
        
        self.grid = np.delete(self.grid, rows_to_clear, axis=0)
        new_rows = np.zeros((len(rows_to_clear), self.GRID_WIDTH), dtype=int)
        self.grid = np.vstack((new_rows, self.grid))
        
        num_cleared = len(rows_to_clear)
        self.rows_cleared += num_cleared
        
        reward = num_cleared
        self.score += (10 * num_cleared) * num_cleared # Bonus for multi-line clears
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self.grid_surface.fill(self.COLOR_BG)
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                pygame.draw.rect(self.grid_surface, self.COLOR_GRID, (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE), 1)
                if self.grid[r, c] != 0:
                    self._draw_block(self.grid_surface, c, r, self.grid[r, c] - 1)
        self.screen.blit(self.grid_surface, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET))

        if not self.game_over and not self.win:
            self._render_ghost_piece(self.screen)
            self._render_piece(self.screen, self.current_piece)
        
        self._render_particles(self.screen)

    def _draw_block(self, surface, x, y, color_idx, alpha=255, offset=(0,0)):
        color = self.PIECE_COLORS[color_idx]
        x_pos, y_pos = int(offset[0] + x * self.CELL_SIZE), int(offset[1] + y * self.CELL_SIZE)
        
        outer_rect = (x_pos, y_pos, self.CELL_SIZE, self.CELL_SIZE)
        inner_rect = (x_pos + 3, y_pos + 3, self.CELL_SIZE - 6, self.CELL_SIZE - 6)

        if alpha == 255:
            pygame.gfxdraw.box(surface, outer_rect, color)
            brighter_color = tuple(min(255, c + 40) for c in color)
            pygame.gfxdraw.box(surface, inner_rect, brighter_color)
        else:
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.set_alpha(alpha)
            pygame.gfxdraw.box(s, (0, 0, self.CELL_SIZE, self.CELL_SIZE), color)
            surface.blit(s, (x_pos, y_pos))

    def _render_piece(self, surface, piece, alpha=255, offset=(GRID_X_OFFSET, GRID_Y_OFFSET)):
        shape = self.rotated_shapes[piece['shape_idx']][piece['rotation']]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_block(surface, piece['x'] + c, piece['y'] + r, piece['shape_idx'], alpha, offset)

    def _render_ghost_piece(self, surface):
        ghost_piece = self.current_piece.copy()
        while self._is_valid_position(ghost_piece, offset_y=1):
            ghost_piece['y'] += 1
        self._render_piece(surface, ghost_piece, alpha=60)

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        rows_text = self.font_main.render(f"Rows: {self.rows_cleared} / 10", True, self.COLOR_TEXT)
        self.screen.blit(rows_text, (20, 60))

        next_text = self.font_small.render("Next:", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (480, 20))
        next_piece_preview = {'shape_idx': self.next_piece_shape_idx, 'rotation': 0}
        self._render_piece_in_box(self.screen, next_piece_preview, (480, 50))
        
        held_text = self.font_small.render("Hold:", True, self.COLOR_TEXT)
        self.screen.blit(held_text, (20, 150))
        if self.held_piece_shape_idx is not None:
            held_piece_preview = {'shape_idx': self.held_piece_shape_idx, 'rotation': 0}
            self._render_piece_in_box(self.screen, held_piece_preview, (20, 180))

        if self.game_over or self.win:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "YOU WIN!" if self.win else "GAME OVER"
            end_color = self.COLOR_WIN if self.win else self.COLOR_GAMEOVER
            end_render = self.font_main.render(end_text, True, end_color)
            text_rect = end_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_render, text_rect)

    def _render_piece_in_box(self, surface, piece, pos):
        box_size = 4 * self.CELL_SIZE
        pygame.draw.rect(surface, self.COLOR_GRID, (*pos, box_size, box_size), 2)
        
        shape = self.rotated_shapes[piece['shape_idx']][piece['rotation']]
        shape_h, shape_w = shape.shape
        
        offset_x = pos[0] + (box_size - shape_w * self.CELL_SIZE) / 2
        offset_y = pos[1] + (box_size - shape_h * self.CELL_SIZE) / 2
        
        self._render_piece(surface, {**piece, 'x': 0, 'y': 0}, offset=(offset_x, offset_y))

    def _create_clear_particles(self, row_idx):
        y = self.GRID_Y_OFFSET + (row_idx + 0.5) * self.CELL_SIZE
        for _ in range(40):
            x = self.GRID_X_OFFSET + random.uniform(0, self.GRID_WIDTH * self.CELL_SIZE)
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            color = random.choice(self.PIECE_COLORS)
            lifespan = random.randint(20, 40)
            self.particles.append({'pos': [x, y], 'vel': vel, 'life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self, surface):
        for p in self.particles:
            size = max(0, int(p['life'] / 8))
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.rect(surface, p['color'], (*pos, size, size))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "rows_cleared": self.rows_cleared}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = np.array([movement, space, shift])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Episode End! Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30)
        
    env.close()