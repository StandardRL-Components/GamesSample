
# Generated: 2025-08-28T05:48:03.817092
# Source Brief: brief_02729.md
# Brief Index: 2729

        
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
        "Controls: ←→ to move, ↑ to rotate clockwise, Shift to rotate counter-clockwise. "
        "↓ for soft drop, Space for hard drop."
    )

    game_description = (
        "A fast-paced falling block puzzle. Strategically place pieces to clear lines, "
        "score points, and prevent the stack from reaching the top. The game speeds up as you clear more lines."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_WARN = (255, 50, 50)
        
        # Tetromino shapes and colors (I, O, T, L, J, S, Z)
        self.SHAPES = [
            [[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]],  # I
            [[1,1], [1,1]],                               # O
            [[0,1,0], [1,1,1], [0,0,0]],                   # T
            [[0,0,1], [1,1,1], [0,0,0]],                   # L
            [[1,0,0], [1,1,1], [0,0,0]],                   # J
            [[0,1,1], [1,1,0], [0,0,0]],                   # S
            [[1,1,0], [0,1,1], [0,0,0]],                   # Z
        ]
        self.COLORS = [
            (50, 200, 200),  # I - Cyan
            (220, 220, 50),  # O - Yellow
            (180, 50, 180),  # T - Purple
            (220, 120, 50),  # L - Orange
            (50, 100, 220),  # J - Blue
            (50, 200, 50),   # S - Green
            (220, 50, 50),   # Z - Red
        ]

        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # --- Game State ---
        self.grid = None
        self.current_piece = None
        self.next_piece_shape_idx = None
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.steps = 0
        self.fall_timer = 0
        self.fall_speed_frames = 0
        self.das_counter = 0
        self.das_direction = 0
        self.particles = []
        self.line_clear_animation = []
        
        self.reset()
        
        # --- Self-Validation ---
        # self.validate_implementation()

    def _spawn_new_piece(self):
        self.current_piece = {
            "shape_idx": self.next_piece_shape_idx,
            "rotation": 0,
            "x": self.GRID_WIDTH // 2 - 1,
            "y": 0,
        }
        self.next_piece_shape_idx = self.np_random.integers(0, len(self.SHAPES))
        
        # Adjust spawn position for I and O pieces
        if self.SHAPES[self.current_piece["shape_idx"]] == self.SHAPES[0]: # I
            self.current_piece['y'] = -1

        if self._check_collision(self.current_piece['x'], self.current_piece['y'], self._get_rotated_piece()):
            self.game_over = True
            return -100 # Game over penalty

        return 0

    def _get_rotated_piece(self, piece=None):
        if piece is None:
            piece = self.current_piece
        
        shape = self.SHAPES[piece["shape_idx"]]
        rot = piece["rotation"]
        
        if rot == 0: return shape
        
        rotated = shape
        for _ in range(rot % 4):
            rotated = list(zip(*rotated[::-1]))
        return rotated

    def _check_collision(self, x, y, shape):
        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    grid_x = x + c_idx
                    grid_y = y + r_idx
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT) or self.grid[grid_y][grid_x] != 0:
                        return True
        return False

    def _place_piece(self):
        shape = self._get_rotated_piece()
        px, py = self.current_piece['x'], self.current_piece['y']
        color_idx = self.current_piece['shape_idx'] + 1
        
        # Calculate hole penalty before placing
        holes_before = self._count_holes()
        
        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    self.grid[py + r_idx][px + c_idx] = color_idx
        
        holes_after = self._count_holes()
        hole_penalty = max(0, holes_after - holes_before) * -0.5
        
        # SFX: Block placed
        return self._clear_lines() + hole_penalty

    def _count_holes(self):
        holes = 0
        for c in range(self.GRID_WIDTH):
            block_found = False
            for r in range(self.GRID_HEIGHT):
                if self.grid[r][c] != 0:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes

    def _clear_lines(self):
        lines_to_clear = []
        for r_idx, row in enumerate(self.grid):
            if all(cell != 0 for cell in row):
                lines_to_clear.append(r_idx)

        if not lines_to_clear:
            return 0

        for r_idx in lines_to_clear:
            self.line_clear_animation.append({"y": r_idx, "timer": 10})
            # SFX: Line clear
            for x in range(self.GRID_WIDTH):
                for _ in range(5): # Spawn 5 particles per cell
                    self.particles.append({
                        "pos": [self.GRID_X + x * self.CELL_SIZE + self.CELL_SIZE / 2, self.GRID_Y + r_idx * self.CELL_SIZE + self.CELL_SIZE / 2],
                        "vel": [(self.np_random.random() - 0.5) * 4, (self.np_random.random() - 0.5) * 4],
                        "life": self.np_random.integers(15, 30),
                        "color": (255, 255, 255)
                    })

        # Remove lines after a delay for animation
        for r_idx in sorted(lines_to_clear, reverse=True):
            del self.grid[r_idx]
            self.grid.insert(0, [0] * self.GRID_WIDTH)

        num_cleared = len(lines_to_clear)
        self.lines_cleared += num_cleared
        
        # Score based on Tetris guidelines (simplified)
        score_map = {1: 100, 2: 300, 3: 500, 4: 800}
        self.score += score_map.get(num_cleared, 0) * (self.lines_cleared // 20 + 1)

        # Update fall speed every 20 lines
        self.fall_speed_frames = max(5, 15 - (self.lines_cleared // 20))
        
        # Reward: +1 per line, +2 bonus for multi-line
        reward = num_cleared + (2 if num_cleared > 1 else 0)
        return reward

    def _handle_input(self, action):
        if self.game_over:
            return

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- One-shot actions ---
        if space_held: # Hard drop
            # SFX: Hard drop
            while not self._check_collision(self.current_piece['x'], self.current_piece['y'] + 1, self._get_rotated_piece()):
                self.current_piece['y'] += 1
                self.score += 2 # Small bonus for hard dropping
            self.fall_timer = self.fall_speed_frames # Force placement on next step
            return

        if shift_held: # Rotate CCW
            self.current_piece['rotation'] = (self.current_piece['rotation'] - 1 + 4) % 4
            if self._check_collision(self.current_piece['x'], self.current_piece['y'], self._get_rotated_piece()):
                # Wall kick
                if not self._check_collision(self.current_piece['x'] + 1, self.current_piece['y'], self._get_rotated_piece()):
                    self.current_piece['x'] += 1
                elif not self._check_collision(self.current_piece['x'] - 1, self.current_piece['y'], self._get_rotated_piece()):
                    self.current_piece['x'] -= 1
                else:
                    self.current_piece['rotation'] = (self.current_piece['rotation'] + 1) % 4 # Revert
            else:
                # SFX: Rotate
                pass

        # --- Movement/Rotation from `actions[0]` ---
        dx = 0
        if movement == 1: # Rotate CW
            self.current_piece['rotation'] = (self.current_piece['rotation'] + 1) % 4
            if self._check_collision(self.current_piece['x'], self.current_piece['y'], self._get_rotated_piece()):
                if not self._check_collision(self.current_piece['x'] + 1, self.current_piece['y'], self._get_rotated_piece()):
                    self.current_piece['x'] += 1
                elif not self._check_collision(self.current_piece['x'] - 1, self.current_piece['y'], self._get_rotated_piece()):
                    self.current_piece['x'] -= 1
                else:
                    self.current_piece['rotation'] = (self.current_piece['rotation'] - 1 + 4) % 4
            else:
                # SFX: Rotate
                pass
        
        if movement == 2: # Soft drop
            self.fall_timer += 2 # Speed up fall
            self.score += 1 # Small bonus for soft dropping

        if movement == 3: dx = -1 # Left
        if movement == 4: dx = 1  # Right
            
        # Delayed Auto Shift (DAS)
        if dx != 0:
            if self.das_direction != dx:
                self.das_counter = 0
                self.das_direction = dx
                if not self._check_collision(self.current_piece['x'] + dx, self.current_piece['y'], self._get_rotated_piece()):
                    self.current_piece['x'] += dx
                    # SFX: Move
            else:
                self.das_counter += 1
                if self.das_counter > 10 and (self.das_counter - 10) % 2 == 0:
                    if not self._check_collision(self.current_piece['x'] + dx, self.current_piece['y'], self._get_rotated_piece()):
                        self.current_piece['x'] += dx
                        # SFX: Move
        else:
            self.das_direction = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = [[0] * self.GRID_WIDTH for _ in range(self.GRID_HEIGHT)]
        self.next_piece_shape_idx = self.np_random.integers(0, len(self.SHAPES))
        self._spawn_new_piece()
        
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.steps = 0
        self.fall_timer = 0
        self.fall_speed_frames = 15 # ~0.5s at 30fps
        self.das_counter = 0
        self.das_direction = 0
        self.particles = []
        self.line_clear_animation = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = -0.01  # Small penalty per step to encourage speed

        if not self.game_over:
            self._handle_input(action)
            
            self.fall_timer += 1
            if self.fall_timer >= self.fall_speed_frames:
                self.fall_timer = 0
                
                if not self._check_collision(self.current_piece['x'], self.current_piece['y'] + 1, self._get_rotated_piece()):
                    self.current_piece['y'] += 1
                else:
                    placement_reward = self._place_piece()
                    reward += placement_reward
                    spawn_reward = self._spawn_new_piece()
                    if self.game_over:
                        reward += spawn_reward

        terminated = self.game_over or self.lines_cleared >= 100 or self.steps >= 10000
        if terminated and not self.game_over and self.lines_cleared >= 100:
            reward += 100 # Win bonus
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))

        # Draw warning zone
        if not self.game_over:
            s = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, 4 * self.CELL_SIZE))
            s.set_alpha(30 + math.sin(self.steps * 0.2) * 20)
            s.fill(self.COLOR_WARN)
            self.screen.blit(s, (self.GRID_X, self.GRID_Y))

        # Draw placed blocks
        for r_idx, row in enumerate(self.grid):
            for c_idx, cell in enumerate(row):
                if cell != 0:
                    self._draw_cell(c_idx, r_idx, self.COLORS[cell - 1])

        # Draw ghost piece
        if not self.game_over and self.current_piece:
            ghost_y = self.current_piece['y']
            shape = self._get_rotated_piece()
            while not self._check_collision(self.current_piece['x'], ghost_y + 1, shape):
                ghost_y += 1
            
            color = self.COLORS[self.current_piece['shape_idx']]
            ghost_color = (color[0] * 0.3, color[1] * 0.3, color[2] * 0.3)
            for r_idx, row in enumerate(shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        self._draw_cell(self.current_piece['x'] + c_idx, ghost_y + r_idx, ghost_color, is_ghost=True)

        # Draw current piece
        if not self.game_over and self.current_piece:
            shape = self._get_rotated_piece()
            color = self.COLORS[self.current_piece['shape_idx']]
            for r_idx, row in enumerate(shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        self._draw_cell(self.current_piece['x'] + c_idx, self.current_piece['y'] + r_idx, color)

        # Draw grid lines on top
        for i in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y), (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_X, self.GRID_Y + i * self.CELL_SIZE), (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y + i * self.CELL_SIZE))

    def _draw_cell(self, grid_x, grid_y, color, is_ghost=False):
        if grid_y < 0: return # Don't draw above the grid
        px, py = self.GRID_X + grid_x * self.CELL_SIZE, self.GRID_Y + grid_y * self.CELL_SIZE
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, (px + 1, py + 1, self.CELL_SIZE - 2, self.CELL_SIZE - 2), 1)
        else:
            # Main cell
            pygame.gfxdraw.box(self.screen, (px + 1, py + 1, self.CELL_SIZE - 2, self.CELL_SIZE - 2), color)
            # 3D-effect highlights/shadows
            highlight = tuple(min(255, c + 40) for c in color)
            shadow = tuple(max(0, c - 40) for c in color)
            pygame.draw.line(self.screen, highlight, (px, py), (px + self.CELL_SIZE - 2, py))
            pygame.draw.line(self.screen, highlight, (px, py), (px, py + self.CELL_SIZE - 2))
            pygame.draw.line(self.screen, shadow, (px + self.CELL_SIZE - 1, py), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1))
            pygame.draw.line(self.screen, shadow, (px, py + self.CELL_SIZE - 1), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1))

    def _render_effects(self):
        # Line clear animation
        for anim in self.line_clear_animation[:]:
            anim['timer'] -= 1
            if anim['timer'] <= 0:
                self.line_clear_animation.remove(anim)
            else:
                alpha = 255 * (anim['timer'] / 10)
                s = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE))
                s.set_alpha(alpha)
                s.fill((255, 255, 255))
                self.screen.blit(s, (self.GRID_X, self.GRID_Y + anim['y'] * self.CELL_SIZE))

        # Particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = 255 * (p['life'] / 30)
                size = self.CELL_SIZE / 4 * (p['life'] / 30)
                pygame.gfxdraw.box(self.screen, (int(p['pos'][0]), int(p['pos'][1]), int(size), int(size)), (*p['color'], int(alpha)))

    def _render_ui(self):
        # --- Score Display ---
        score_text = self.font_large.render(f"{self.score:08}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y))
        
        # --- Lines Display ---
        lines_label = self.font_small.render("LINES", True, self.COLOR_TEXT)
        self.screen.blit(lines_label, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y + 50))
        lines_text = self.font_large.render(f"{self.lines_cleared}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y + 70))

        # --- Next Piece Display ---
        next_box_x = self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20
        next_box_y = self.GRID_Y + 120
        pygame.draw.rect(self.screen, self.COLOR_GRID, (next_box_x, next_box_y, 4 * self.CELL_SIZE + 20, 4 * self.CELL_SIZE + 20), 0, 5)
        next_label = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_label, (next_box_x + 10, next_box_y + 10))

        if self.next_piece_shape_idx is not None:
            shape = self.SHAPES[self.next_piece_shape_idx]
            color = self.COLORS[self.next_piece_shape_idx]
            w, h = len(shape[0]), len(shape)
            off_x = next_box_x + (4 * self.CELL_SIZE + 20 - w * self.CELL_SIZE) // 2
            off_y = next_box_y + (4 * self.CELL_SIZE + 20 - h * self.CELL_SIZE) // 2 + 10
            for r_idx, row in enumerate(shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        px = off_x + c_idx * self.CELL_SIZE
                        py = off_y + r_idx * self.CELL_SIZE
                        pygame.gfxdraw.box(self.screen, (px+1, py+1, self.CELL_SIZE-2, self.CELL_SIZE-2), color)

        # --- Game Over ---
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT))
            s.set_alpha(180)
            s.fill((0, 0, 0))
            self.screen.blit(s, (0, 0))
            
            status_text = "GAME OVER" if self.lines_cleared < 100 else "YOU WIN!"
            text = self.font_large.render(status_text, True, self.COLOR_WARN if self.lines_cleared < 100 else (100, 255, 100))
            text_rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gymnasium Tetris")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # --- Human Controls ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    action = [0, 0, 0]
    
    while not done:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Key Presses ---
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        else:
            action[0] = 0
        
        # Space and Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()