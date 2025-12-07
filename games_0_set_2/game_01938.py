
# Generated: 2025-08-27T18:44:44.047357
# Source Brief: brief_01938.md
# Brief Index: 1938

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to rotate, ←→ to move, ↓ for soft drop. Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down falling block puzzle game. Clear 10 lines to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAYFIELD_WIDTH = 10
    PLAYFIELD_HEIGHT = 20
    CELL_SIZE = 18
    GRID_COLOR = (40, 40, 60)
    BG_COLOR = (20, 20, 30)
    TEXT_COLOR = (220, 220, 240)
    FLASH_COLOR = (255, 255, 255)

    # Tetromino shapes and their colors
    TETROMINOES = {
        'I': ([[1, 1, 1, 1]], (50, 180, 220)),
        'O': ([[1, 1], [1, 1]], (220, 220, 50)),
        'T': ([[0, 1, 0], [1, 1, 1]], (180, 50, 220)),
        'J': ([[1, 0, 0], [1, 1, 1]], (50, 50, 220)),
        'L': ([[0, 0, 1], [1, 1, 1]], (220, 120, 50)),
        'S': ([[0, 1, 1], [1, 1, 0]], (50, 220, 50)),
        'Z': ([[1, 1, 0], [0, 1, 1]], (220, 50, 50))
    }
    TETROMINO_KEYS = list(TETROMINOES.keys())
    PLACED_BLOCK_COLOR = (80, 80, 100)
    GHOST_ALPHA = 80

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
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        self.playfield_x = (self.SCREEN_WIDTH - self.PLAYFIELD_WIDTH * self.CELL_SIZE) // 2
        self.playfield_y = (self.SCREEN_HEIGHT - self.PLAYFIELD_HEIGHT * self.CELL_SIZE) // 2

        self.particles = []
        self.reset()
        
        # This check is disabled in the final deliverable but is useful for development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.grid = np.zeros((self.PLAYFIELD_HEIGHT, self.PLAYFIELD_WIDTH), dtype=int)
        self.bag = list(self.TETROMINO_KEYS)
        self.np_random.shuffle(self.bag)

        self.current_piece = self._new_piece()
        self.next_piece = self._new_piece()

        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.winner = False

        self.fall_timer = 0.0
        self.fall_speed = 1.0  # seconds per grid cell
        self.base_fall_speed = 1.0
        self.fall_speed_reduction_per_line = 0.05

        self.flash_timer = 0
        self.flash_rows = []
        
        self.particles = []

        self.prev_action = np.array([0, 0, 0])
        self.move_timer = 0.0
        self.move_delay = 0.15 # seconds

        return self._get_observation(), self._get_info()

    def _new_piece(self):
        if not self.bag:
            self.bag = list(self.TETROMINO_KEYS)
            self.np_random.shuffle(self.bag)
        shape_key = self.bag.pop()
        shape, color = self.TETROMINOES[shape_key]
        return {
            "shape": np.array(shape),
            "color": color,
            "x": self.PLAYFIELD_WIDTH // 2 - len(shape[0]) // 2,
            "y": 0
        }

    def _check_collision(self, piece, offset_x=0, offset_y=0):
        shape = piece["shape"]
        pos_x, pos_y = piece["x"] + offset_x, piece["y"] + offset_y
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = pos_x + c, pos_y + r
                    if not (0 <= grid_x < self.PLAYFIELD_WIDTH and 0 <= grid_y < self.PLAYFIELD_HEIGHT):
                        return True  # Wall collision
                    if self.grid[grid_y, grid_x] != 0:
                        return True  # Placed block collision
        return False

    def _place_piece(self):
        shape, color = self.current_piece["shape"], self.current_piece["color"]
        pos_x, pos_y = self.current_piece["x"], self.current_piece["y"]
        color_index = self.TETROMINO_KEYS.index(self._get_key_from_color(color)) + 1
        
        # sound placeholder: # sfx_place_block()
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self.grid[pos_y + r, pos_x + c] = color_index
                    self._create_particles((pos_x + c, pos_y + r), color)

    def _get_key_from_color(self, color):
        for key, val in self.TETROMINOES.items():
            if val[1] == color:
                return key
        return None

    def _clear_lines(self):
        lines_to_clear = [r for r, row in enumerate(self.grid) if np.all(row != 0)]
        if not lines_to_clear:
            return 0, 0
        
        # sound placeholder: # sfx_clear_line()
        self.flash_rows = lines_to_clear
        self.flash_timer = 0.2 # seconds

        # Calculate reward for clearing lines
        reward = len(lines_to_clear) * 10

        # Create particles for cleared lines
        for r in lines_to_clear:
            for c in range(self.PLAYFIELD_WIDTH):
                color = self.TETROMINOES[self.TETROMINO_KEYS[int(self.grid[r, c]) - 1]][1]
                self._create_particles((c, r), color, count=3)

        # Clear lines and shift down
        self.grid = np.delete(self.grid, lines_to_clear, axis=0)
        new_rows = np.zeros((len(lines_to_clear), self.PLAYFIELD_WIDTH), dtype=int)
        self.grid = np.vstack((new_rows, self.grid))

        self.lines_cleared += len(lines_to_clear)
        self.fall_speed = max(0.1, self.base_fall_speed - self.lines_cleared * self.fall_speed_reduction_per_line)
        return len(lines_to_clear), reward
        
    def _calculate_risk_reward(self):
        shape = self.current_piece["shape"]
        pos_x, pos_y = self.current_piece["x"], self.current_piece["y"]
        
        holes_created = 0
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_y = pos_y + r + 1
                    grid_x = pos_x + c
                    if grid_y < self.PLAYFIELD_HEIGHT and self.grid[grid_y, grid_x] == 0:
                        holes_created += 1
        
        if holes_created > 0:
            return 5 # Risky placement reward
        else:
            return -2 # Safe placement reward

    def step(self, action):
        reward = -0.1  # Small penalty per step to encourage speed
        self.steps += 1
        dt = self.clock.tick(30) / 1000.0  # Delta time in seconds

        # --- Handle one-shot actions ---
        movement, space_btn, shift_btn = action
        prev_movement, prev_space, _ = self.prev_action

        rotate_action = (movement == 1 and prev_movement != 1)
        hard_drop_action = (space_btn == 1 and prev_space != 1)

        # --- Handle held actions ---
        soft_drop_held = (movement == 2)
        move_left_held = (movement == 3)
        move_right_held = (movement == 4)
        
        self.move_timer += dt
        
        if self.move_timer > self.move_delay:
            self.move_timer = 0
            if move_left_held and not self._check_collision(self.current_piece, offset_x=-1):
                self.current_piece["x"] -= 1
            if move_right_held and not self._check_collision(self.current_piece, offset_x=1):
                self.current_piece["x"] += 1

        if rotate_action:
            # sound placeholder: # sfx_rotate()
            rotated_shape = np.rot90(self.current_piece["shape"], k=-1)
            original_pos = self.current_piece["x"]
            test_piece = {**self.current_piece, "shape": rotated_shape}
            
            # Basic wall kick
            if not self._check_collision(test_piece):
                self.current_piece["shape"] = rotated_shape
            else:
                # Try kicking left/right
                if not self._check_collision(test_piece, offset_x=-1):
                    self.current_piece["shape"] = rotated_shape
                    self.current_piece["x"] -= 1
                elif not self._check_collision(test_piece, offset_x=1):
                    self.current_piece["shape"] = rotated_shape
                    self.current_piece["x"] += 1
                elif len(rotated_shape) > 2 and not self._check_collision(test_piece, offset_x=-2): # For 'I' piece
                    self.current_piece["shape"] = rotated_shape
                    self.current_piece["x"] -= 2
                elif len(rotated_shape) > 2 and not self._check_collision(test_piece, offset_x=2): # For 'I' piece
                    self.current_piece["shape"] = rotated_shape
                    self.current_piece["x"] += 2


        if hard_drop_action:
            # sound placeholder: # sfx_hard_drop()
            while not self._check_collision(self.current_piece, offset_y=1):
                self.current_piece["y"] += 1
                reward += 0.1 # Small reward for dropping
            self.fall_timer = self.fall_speed # Force placement on next logic tick
        
        # --- Game Logic Update ---
        self.fall_timer += dt * (4.0 if soft_drop_held else 1.0)
        if soft_drop_held:
            reward += 0.1 # Small reward for soft dropping

        if self.fall_timer >= self.fall_speed:
            self.fall_timer = 0
            if not self._check_collision(self.current_piece, offset_y=1):
                self.current_piece["y"] += 1
            else:
                # Piece has landed
                reward += self._calculate_risk_reward()
                self._place_piece()
                reward += 1 # Reward for placing a piece
                
                lines_cleared, line_reward = self._clear_lines()
                reward += line_reward
                self.score += (lines_cleared ** 2) * 100 # Score based on lines

                self.current_piece = self.next_piece
                self.next_piece = self._new_piece()

                if self._check_collision(self.current_piece):
                    self.game_over = True
        
        # Update animations
        if self.flash_timer > 0:
            self.flash_timer -= dt
            if self.flash_timer <= 0:
                self.flash_rows = []

        self._update_particles(dt)

        self.prev_action = action
        terminated = self._check_termination()

        if self.winner:
            reward += 100
        elif self.game_over:
            reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.game_over:
            return True
        if self.lines_cleared >= 10:
            self.winner = True
            return True
        if self.steps >= 1000:
            self.game_over = True # Treat max steps as a loss
            return True
        return False
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines": self.lines_cleared,
            "game_over": self.game_over,
            "winner": self.winner
        }

    def _get_observation(self):
        self.screen.fill(self.BG_COLOR)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_playfield_bg()
        self._draw_placed_blocks()
        if not self.game_over:
            self._draw_ghost_piece()
            self._draw_piece(self.current_piece)
        self._draw_particles()
        self._draw_flash_animation()
        self._draw_playfield_grid()
        
    def _draw_playfield_bg(self):
        pygame.draw.rect(
            self.screen,
            (0,0,0),
            (self.playfield_x, self.playfield_y, self.PLAYFIELD_WIDTH * self.CELL_SIZE, self.PLAYFIELD_HEIGHT * self.CELL_SIZE)
        )

    def _draw_playfield_grid(self):
        for x in range(self.PLAYFIELD_WIDTH + 1):
            px = self.playfield_x + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.GRID_COLOR, (px, self.playfield_y), (px, self.playfield_y + self.PLAYFIELD_HEIGHT * self.CELL_SIZE))
        for y in range(self.PLAYFIELD_HEIGHT + 1):
            py = self.playfield_y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.GRID_COLOR, (self.playfield_x, py), (self.playfield_x + self.PLAYFIELD_WIDTH * self.CELL_SIZE, py))

    def _draw_cell(self, grid_x, grid_y, color, alpha=255):
        rect = pygame.Rect(
            self.playfield_x + grid_x * self.CELL_SIZE,
            self.playfield_y + grid_y * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        if alpha < 255:
            surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(surface, (*color, alpha), (1, 1, self.CELL_SIZE - 2, self.CELL_SIZE - 2))
            self.screen.blit(surface, rect.topleft)
        else:
            pygame.draw.rect(self.screen, color, rect.inflate(-2, -2))

    def _draw_placed_blocks(self):
        for r in range(self.PLAYFIELD_HEIGHT):
            for c in range(self.PLAYFIELD_WIDTH):
                if self.grid[r, c] != 0:
                    self._draw_cell(c, r, self.PLACED_BLOCK_COLOR)

    def _draw_piece(self, piece, offset_x=0, offset_y=0, alpha=255):
        shape, color = piece["shape"], piece["color"]
        pos_x, pos_y = piece["x"] + offset_x, piece["y"] + offset_y
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_cell(pos_x + c, pos_y + r, color, alpha)

    def _draw_ghost_piece(self):
        ghost = self.current_piece.copy()
        while not self._check_collision(ghost, offset_y=1):
            ghost["y"] += 1
        self._draw_piece(ghost, alpha=self.GHOST_ALPHA)

    def _draw_flash_animation(self):
        if self.flash_timer > 0:
            for r in self.flash_rows:
                rect = pygame.Rect(
                    self.playfield_x,
                    self.playfield_y + r * self.CELL_SIZE,
                    self.PLAYFIELD_WIDTH * self.CELL_SIZE,
                    self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.FLASH_COLOR, rect)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE", True, self.TEXT_COLOR)
        score_val = self.font_large.render(f"{self.score:06}", True, self.TEXT_COLOR)
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(score_val, (20, 45))

        # Lines
        lines_text = self.font_large.render(f"LINES", True, self.TEXT_COLOR)
        lines_val = self.font_large.render(f"{self.lines_cleared}/10", True, self.TEXT_COLOR)
        self.screen.blit(lines_text, (self.SCREEN_WIDTH - 120, 20))
        self.screen.blit(lines_val, (self.SCREEN_WIDTH - 120, 45))

        # Next Piece
        next_text = self.font_large.render(f"NEXT", True, self.TEXT_COLOR)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 120, 280))
        
        next_piece_surf = pygame.Surface((4 * self.CELL_SIZE, 4 * self.CELL_SIZE), pygame.SRCALPHA)
        shape = self.next_piece["shape"]
        color = self.next_piece["color"]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(next_piece_surf, color, (c*self.CELL_SIZE, r*self.CELL_SIZE, self.CELL_SIZE-2, self.CELL_SIZE-2))
        self.screen.blit(next_piece_surf, (self.SCREEN_WIDTH - 120, 310))

        # Game Over / Winner
        if self.game_over or self.winner:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.winner else "GAME OVER"
            msg_surf = self.font_large.render(message, True, self.FLASH_COLOR)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _create_particles(self, grid_pos, color, count=5):
        px = self.playfield_x + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.playfield_y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(count):
            self.particles.append({
                "pos": [px, py],
                "vel": [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -1)],
                "life": self.np_random.uniform(0.3, 0.7),
                "color": color
            })

    def _update_particles(self, dt):
        for p in self.particles:
            p["pos"][0] += p["vel"][0] * 60 * dt
            p["pos"][1] += p["vel"][1] * 60 * dt
            p["vel"][1] += 0.1 * 60 * dt # Gravity
            p["life"] -= dt
        self.particles = [p for p in self.particles if p["life"] > 0]
        
    def _draw_particles(self):
        for p in self.particles:
            size = max(1, int(p["life"] * 5))
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), size)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    # Set the video driver to dummy to run headless
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    pygame.display.set_caption("Falling Block Puzzle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Map Pygame keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP:    (1, 0, 0),
        pygame.K_DOWN:  (2, 0, 0),
        pygame.K_LEFT:  (3, 0, 0),
        pygame.K_RIGHT: (4, 0, 0),
        pygame.K_SPACE: (0, 1, 0),
        pygame.K_LSHIFT: (0, 2, 0), # Not used, but for completeness
        pygame.K_RSHIFT: (0, 2, 0),
    }

    running = True
    while running:
        # --- Human Input ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = np.array([movement, space, shift])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()