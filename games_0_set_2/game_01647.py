
# Generated: 2025-08-28T02:15:54.923538
# Source Brief: brief_01647.md
# Brief Index: 1647

        
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
    """
    A Tetris-like puzzle game environment for Gymnasium.

    The player must place falling tetrominoes to complete horizontal lines.
    Clearing lines awards points and increases the game speed. The goal is to
    clear a set number of lines without letting the blocks stack to the top.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑↓ to rotate. Hold shift to hold a piece and press space to drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically place falling blocks in a grid to complete lines and achieve a target score before the board fills up."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.BLOCK_SIZE = 18
        self.WIN_CONDITION = 20
        self.MAX_STEPS = 3000

        # Centering the grid
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.PIECE_COLORS = [
            (0, 0, 0),          # 0: Empty
            (3, 252, 248),      # 1: I piece (Cyan)
            (252, 248, 3),      # 2: O piece (Yellow)
            (173, 3, 252),      # 3: T piece (Purple)
            (3, 32, 252),       # 4: J piece (Blue)
            (252, 128, 3),      # 5: L piece (Orange)
            (57, 255, 20),      # 6: S piece (Green)
            (252, 3, 3),        # 7: Z piece (Red)
        ]

        # Tetromino shapes (rotations handled by code)
        self.PIECE_SHAPES = {
            1: [(0, -1), (0, 0), (0, 1), (0, 2)],    # I
            2: [(0, 0), (0, 1), (1, 0), (1, 1)],    # O
            3: [(-1, 0), (0, 0), (1, 0), (0, -1)],   # T
            4: [(-1, -1), (0, -1), (0, 0), (0, 1)],  # J
            5: [(-1, 1), (0, -1), (0, 0), (0, 1)],   # L
            6: [(-1, 0), (0, 0), (0, -1), (1, -1)],  # S
            7: [(-1, -1), (0, -1), (0, 0), (1, 0)],  # Z
        }

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Etc...
        self.grid = []
        self.current_piece = None
        self.next_piece_type = 0
        self.held_piece_type = 0
        self.can_hold = True
        self.fall_time = 0
        self.fall_speed = 0
        self.move_cooldown = 0
        self.rotate_cooldown = 0
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.clear_animation_timer = 0
        self.lines_to_clear_anim = []
        self.last_action_reward = 0
        self.piece_bag = []

        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.grid = [[0 for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = 1000  # ms per grid cell
        self.fall_time = 0
        self.move_cooldown = 0
        self.rotate_cooldown = 0
        self.held_piece_type = 0
        self.can_hold = True
        self.clear_animation_timer = 0
        self.lines_to_clear_anim = []
        self.last_action_reward = 0

        self.piece_bag = list(range(1, 8))
        random.shuffle(self.piece_bag)

        self.next_piece_type = self._get_piece_from_bag()
        self._new_piece()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False
        self.steps += 1
        
        # Advance frame clock for time-based logic
        dt = self.clock.tick(30)

        # Handle clear animation delay without processing other logic
        if self.clear_animation_timer > 0:
            self.clear_animation_timer -= dt
            if self.clear_animation_timer <= 0:
                self._perform_line_clear()
        elif not self.game_over:
            self._handle_input(action, dt)
            self._update_fall(dt)

        # Check termination conditions
        if self.game_over:
            reward -= 100
            terminated = True
        elif self.lines_cleared >= self.WIN_CONDITION:
            reward += 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        reward += self.last_action_reward
        self.last_action_reward = 0 # Consume the reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

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

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }

    def _get_piece_from_bag(self):
        if not self.piece_bag:
            self.piece_bag = list(range(1, 8))
            random.shuffle(self.piece_bag)
        return self.piece_bag.pop()

    def _new_piece(self):
        self.current_piece = {
            "type": self.next_piece_type,
            "shape": self.PIECE_SHAPES[self.next_piece_type],
            "row": -1, # Start slightly above the grid
            "col": self.GRID_WIDTH // 2 - 1
        }
        self.next_piece_type = self._get_piece_from_bag()
        self.can_hold = True
        self.fall_time = 0

        if self._check_collision(self.current_piece["shape"], (self.current_piece["row"], self.current_piece["col"])):
            self.game_over = True

    def _check_collision(self, shape, pos):
        row, col = pos
        for r_off, c_off in shape:
            r, c = int(row + r_off), int(col + c_off)
            if not (0 <= c < self.GRID_WIDTH and r < self.GRID_HEIGHT):
                return True # Out of bounds (left, right, bottom)
            if r >= 0 and self.grid[r][c] != 0:
                return True # Collides with existing block
        return False

    def _handle_input(self, action, dt):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        cooldown_time = 120 # ms
        self.move_cooldown = max(0, self.move_cooldown - dt)
        self.rotate_cooldown = max(0, self.rotate_cooldown - dt)

        if self.move_cooldown == 0:
            if movement == 3: self._move(-1); self.move_cooldown = cooldown_time
            elif movement == 4: self._move(1); self.move_cooldown = cooldown_time
        
        if self.rotate_cooldown == 0:
            if movement == 1: self._rotate(True); self.rotate_cooldown = cooldown_time
            elif movement == 2: self._rotate(False); self.rotate_cooldown = cooldown_time
        
        if space_held: self._hard_drop() # sfx: hard_drop
        if shift_held and self.can_hold: self._hold_piece() # sfx: hold_piece

    def _move(self, direction):
        new_col = self.current_piece["col"] + direction
        if not self._check_collision(self.current_piece["shape"], (self.current_piece["row"], new_col)):
            self.current_piece["col"] = new_col # sfx: move_sound

    def _rotate(self, clockwise):
        if self.current_piece["type"] == 2: return # O-piece doesn't rotate

        new_shape = []
        for r_off, c_off in self.current_piece["shape"]:
            new_shape.append((c_off, -r_off) if clockwise else (-c_off, r_off))

        for kick_offset in [(0, 0), (0, -1), (0, 1), (0, -2), (0, 2), (1, 0), (-1, 0)]:
            if not self._check_collision(new_shape, (self.current_piece["row"] + kick_offset[0], self.current_piece["col"] + kick_offset[1])):
                self.current_piece["shape"] = new_shape
                self.current_piece["row"] += kick_offset[0]
                self.current_piece["col"] += kick_offset[1]
                # sfx: rotate_sound
                return

    def _update_fall(self, dt):
        self.fall_time += dt
        if self.fall_time >= self.fall_speed:
            self.fall_time = 0
            new_row = self.current_piece["row"] + 1
            if self._check_collision(self.current_piece["shape"], (new_row, self.current_piece["col"])):
                self._lock_piece()
            else:
                self.current_piece["row"] = new_row

    def _hard_drop(self):
        ghost_pos = self._get_ghost_position()
        self.current_piece["row"] = ghost_pos["row"]
        self._lock_piece()

    def _lock_piece(self):
        # sfx: lock_sound
        shape, pos = self.current_piece["shape"], (self.current_piece["row"], self.current_piece["col"])
        
        for r_off, c_off in shape:
            r, c = int(pos[0] + r_off), int(pos[1] + c_off)
            if 0 <= r < self.GRID_HEIGHT and 0 <= c < self.GRID_WIDTH:
                self.grid[r][c] = self.current_piece["type"]
        
        self.last_action_reward -= self._calculate_holes_under_piece(shape, pos) * 0.1
        self._check_and_start_clear_animation()

        if self.clear_animation_timer <= 0: self._new_piece()

    def _calculate_holes_under_piece(self, shape, pos):
        holes = 0
        min_rows = {}
        for r_off, c_off in shape:
            r, c = int(pos[0] + r_off), int(pos[1] + c_off)
            if c not in min_rows or r > min_rows[c]: min_rows[c] = r
        
        for c, r_start in min_rows.items():
            if 0 <= c < self.GRID_WIDTH:
                for r in range(r_start + 1, self.GRID_HEIGHT):
                    if self.grid[r][c] == 0: holes += 1
                    else: break
        return holes

    def _check_and_start_clear_animation(self):
        full_lines = [r for r in range(self.GRID_HEIGHT) if all(self.grid[r])]
        if full_lines:
            self.lines_to_clear_anim = full_lines
            self.clear_animation_timer = 200 # sfx: line_clear_start
        
    def _perform_line_clear(self):
        num_cleared = len(self.lines_to_clear_anim)
        line_rewards = {1: 1, 2: 3, 3: 5, 4: 10}
        reward = line_rewards.get(num_cleared, 0)
        self.last_action_reward += reward
        self.score += reward * 10 * num_cleared
        self.lines_cleared += num_cleared
        # sfx: combo_clear_sound

        for r in sorted(self.lines_to_clear_anim, reverse=True): self.grid.pop(r)
        for _ in range(num_cleared): self.grid.insert(0, [0] * self.GRID_WIDTH)

        self.fall_speed = max(100, 1000 - self.lines_cleared * 25)
        self.lines_to_clear_anim = []
        self._new_piece()

    def _hold_piece(self):
        self.can_hold = False
        if self.held_piece_type == 0:
            self.held_piece_type = self.current_piece["type"]
            self._new_piece()
        else:
            held, self.held_piece_type = self.held_piece_type, self.current_piece["type"]
            self.current_piece = {"type": held, "shape": self.PIECE_SHAPES[held], "row": -1, "col": self.GRID_WIDTH // 2 - 1}
            if self._check_collision(self.current_piece["shape"], (self.current_piece["row"], self.current_piece["col"])):
                self.game_over = True
    
    def _get_ghost_position(self):
        ghost = self.current_piece.copy()
        while not self._check_collision(ghost["shape"], (ghost["row"] + 1, ghost["col"])):
            ghost["row"] += 1
        return ghost

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE))

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] != 0: self._draw_block(c, r, self.grid[r][c])
        
        if not self.game_over and self.current_piece:
            ghost = self._get_ghost_position()
            for r_off, c_off in ghost["shape"]: self._draw_block(ghost["col"] + c_off, ghost["row"] + r_off, ghost["type"], True)
            for r_off, c_off in self.current_piece["shape"]: self._draw_block(self.current_piece["col"] + c_off, self.current_piece["row"] + r_off, self.current_piece["type"])

        for r in range(self.GRID_HEIGHT + 1): pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_X, self.GRID_Y + r * self.BLOCK_SIZE), (self.GRID_X + self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_Y + r * self.BLOCK_SIZE))
        for c in range(self.GRID_WIDTH + 1): pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_X + c * self.BLOCK_SIZE, self.GRID_Y), (self.GRID_X + c * self.BLOCK_SIZE, self.GRID_Y + self.GRID_HEIGHT * self.BLOCK_SIZE))

        if self.clear_animation_timer > 0:
            for r in self.lines_to_clear_anim: pygame.draw.rect(self.screen, (255, 255, 255), (self.GRID_X, self.GRID_Y + r * self.BLOCK_SIZE, self.GRID_WIDTH * self.BLOCK_SIZE, self.BLOCK_SIZE))

    def _draw_block(self, c, r, piece_type, is_ghost=False):
        x, y = self.GRID_X + c * self.BLOCK_SIZE, self.GRID_Y + r * self.BLOCK_SIZE
        if y < self.GRID_Y: return # Don't draw above the visible grid
        
        color = self.PIECE_COLORS[piece_type]
        if is_ghost:
            s = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            s.fill((color[0], color[1], color[2], 60))
            self.screen.blit(s, (x, y))
            pygame.draw.rect(self.screen, (255, 255, 255), (x, y, self.BLOCK_SIZE, self.BLOCK_SIZE), 1)
        else:
            outer = tuple(max(0, val - 60) for val in color)
            pygame.draw.rect(self.screen, outer, (x, y, self.BLOCK_SIZE, self.BLOCK_SIZE))
            pygame.draw.rect(self.screen, color, (x + 2, y + 2, self.BLOCK_SIZE - 4, self.BLOCK_SIZE - 4))

    def _render_ui(self):
        self.screen.blit(self.font_main.render(f"SCORE: {self.score}", 1, self.COLOR_TEXT), (self.GRID_X + self.GRID_WIDTH * self.BLOCK_SIZE + 20, self.GRID_Y + 200))
        self.screen.blit(self.font_main.render(f"LINES: {self.lines_cleared}/{self.WIN_CONDITION}", 1, self.COLOR_TEXT), (self.GRID_X + self.GRID_WIDTH * self.BLOCK_SIZE + 20, self.GRID_Y + 230))
        self._render_preview_box(self.GRID_X + self.GRID_WIDTH * self.BLOCK_SIZE + 20, self.GRID_Y, "NEXT", self.next_piece_type)
        self._render_preview_box(self.GRID_X - 120, self.GRID_Y, "HOLD", self.held_piece_type)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            end_text = "YOU WIN!" if self.lines_cleared >= self.WIN_CONDITION else "GAME OVER"
            text_surf = self.font_main.render(end_text, 1, (255, 50, 50))
            self.screen.blit(text_surf, text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def _render_preview_box(self, x, y, title, piece_type):
        box_w, box_h = 100, 100
        pygame.draw.rect(self.screen, self.COLOR_GRID, (x, y, box_w, box_h), border_radius=5)
        title_surf = self.font_small.render(title, 1, self.COLOR_TEXT)
        self.screen.blit(title_surf, (x + (box_w - title_surf.get_width()) // 2, y + 5))
        
        if piece_type != 0:
            shape = self.PIECE_SHAPES[piece_type]
            color = self.PIECE_COLORS[piece_type]
            min_r, max_r = min(r for r,c in shape), max(r for r,c in shape)
            min_c, max_c = min(c for r,c in shape), max(c for r,c in shape)
            shape_h, shape_w = (max_r-min_r+1)*self.BLOCK_SIZE, (max_c-min_c+1)*self.BLOCK_SIZE
            start_x, start_y = x + (box_w-shape_w)//2, y + (box_h-shape_h)//2 + 10

            for r_off, c_off in shape:
                dx, dy = start_x+(c_off-min_c)*self.BLOCK_SIZE, start_y+(r_off-min_r)*self.BLOCK_SIZE
                outer = tuple(max(0, val - 60) for val in color)
                pygame.draw.rect(self.screen, outer, (dx, dy, self.BLOCK_SIZE, self.BLOCK_SIZE))
                pygame.draw.rect(self.screen, color, (dx + 2, dy + 2, self.BLOCK_SIZE - 4, self.BLOCK_SIZE - 4))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
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
        assert not trunc and isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gymnasium Block Puzzle")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not terminated:
        movement, space_held, shift_held = 0, 0, 0
        
        # Continuous key presses for movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        elif keys[pygame.K_SPACE]: space_held = 1
        
        # Event-based key presses for rotation/hold
        for event in pygame.event.get():
            if event.type == pygame.QUIT: terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()