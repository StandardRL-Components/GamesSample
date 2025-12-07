
# Generated: 2025-08-28T05:15:10.830740
# Source Brief: brief_05514.md
# Brief Index: 5514

        
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
        "Controls: ←/→ to move, ↑ to rotate clockwise, ↓ to soft drop. "
        "Space to rotate counter-clockwise, Shift to hard drop."
    )

    game_description = (
        "A fast-paced, grid-based falling block puzzle game. "
        "Clear lines by filling them with blocks. Win by clearing 10 lines."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2

    COLOR_BG = (26, 26, 46)
    COLOR_GRID = (40, 40, 60)
    COLOR_UI_TEXT = (230, 230, 255)
    COLOR_UI_ACCENT = (233, 69, 96)
    COLOR_WHITE = (255, 255, 255)

    BLOCK_SHAPES = {
        'I': ([[1, 1, 1, 1]], (255, 100, 100)),
        'O': ([[1, 1], [1, 1]], (255, 212, 35)),
        'T': ([[0, 1, 0], [1, 1, 1]], (170, 0, 255)),
        'L': ([[0, 0, 1], [1, 1, 1]], (255, 150, 0)),
        'J': ([[1, 0, 0], [1, 1, 1]], (0, 100, 255)),
        'S': ([[0, 1, 1], [1, 1, 0]], (100, 255, 100)),
        'Z': ([[1, 1, 0], [0, 1, 1]], (255, 0, 100)),
    }
    
    WIN_CONDITION_LINES = 10
    MAX_STEPS = 3000 # Increased to allow for more complex play

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
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_title = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.shape_keys = list(self.BLOCK_SHAPES.keys())

        # Initialize state variables
        self.grid = None
        self.current_block = None
        self.next_block = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        
        self.fall_timer = 0
        self.fall_speed = 1.0 # seconds per drop
        
        self.line_clear_animation = None
        
        self.action_cooldowns = {'move': 0, 'rotate': 0}
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        self.current_block = self._new_block()
        self.next_block = self._new_block()
        
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        
        self.fall_timer = 0
        self.fall_speed = 1.0
        
        self.line_clear_animation = None
        
        self.action_cooldowns = {'move': 0, 'rotate': 0}

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01  # Small penalty per step to encourage speed
        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        if self.line_clear_animation:
            self.line_clear_animation['timer'] -= 1
            if self.line_clear_animation['timer'] <= 0:
                self._finish_line_clear()
            return self._get_observation(), reward, False, False, self._get_info()
        
        # --- Handle Input ---
        self._update_cooldowns()
        
        if shift_held: # Hard Drop
            reward += self._hard_drop()
        else:
            # Movement
            if movement == 3 and self.action_cooldowns['move'] == 0: # Left
                self._move(-1)
                self.action_cooldowns['move'] = 5
            elif movement == 4 and self.action_cooldowns['move'] == 0: # Right
                self._move(1)
                self.action_cooldowns['move'] = 5
            
            # Rotation
            if movement == 1 and self.action_cooldowns['rotate'] == 0: # Up -> Rotate CW
                self._rotate(1)
                self.action_cooldowns['rotate'] = 8
            elif space_held and self.action_cooldowns['rotate'] == 0: # Space -> Rotate CCW
                self._rotate(-1)
                self.action_cooldowns['rotate'] = 8

            # Soft Drop
            soft_drop_speed = 0.05 if movement == 2 else self.fall_speed
            
            # --- Game Logic Update ---
            self.fall_timer += self.clock.get_time() / 1000.0
            if self.fall_timer >= soft_drop_speed:
                self.fall_timer = 0
                self.current_block['y'] += 1
                if self._check_collision(self.current_block):
                    self.current_block['y'] -= 1
                    reward += self._lock_block()

        # --- Termination Check ---
        if self.lines_cleared >= self.WIN_CONDITION_LINES:
            self.game_over = True
            self.game_won = True
            reward += 100
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            reward -= 10 # Penalty for running out of time
            
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Helper Methods ---

    def _new_block(self):
        shape_key = self.np_random.choice(self.shape_keys)
        shape_info = self.BLOCK_SHAPES[shape_key]
        return {
            'key': shape_key,
            'shape': shape_info[0],
            'color_idx': self.shape_keys.index(shape_key) + 1,
            'color': shape_info[1],
            'x': self.GRID_WIDTH // 2 - len(shape_info[0][0]) // 2,
            'y': 0,
        }

    def _check_collision(self, block):
        for y, row in enumerate(block['shape']):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = block['x'] + x, block['y'] + y
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT and self.grid[grid_x, grid_y] == 0):
                        return True
        return False

    def _move(self, dx):
        self.current_block['x'] += dx
        if self._check_collision(self.current_block):
            self.current_block['x'] -= dx

    def _rotate(self, direction):
        original_shape = self.current_block['shape']
        if direction == 1: # Clockwise
            new_shape = [list(row) for row in zip(*original_shape[::-1])]
        else: # Counter-Clockwise
            new_shape = [list(row) for row in zip(*original_shape)][::-1]
        
        self.current_block['shape'] = new_shape
        if self._check_collision(self.current_block):
            # Wall kick
            for dx in [-1, 1, -2, 2]:
                self.current_block['x'] += dx
                if not self._check_collision(self.current_block):
                    return
                self.current_block['x'] -= dx
            self.current_block['shape'] = original_shape # Revert if all kicks fail

    def _hard_drop(self):
        while not self._check_collision(self.current_block):
            self.current_block['y'] += 1
        self.current_block['y'] -= 1
        return self._lock_block()

    def _lock_block(self):
        # Sound: Block lock
        reward = 0
        is_touching = False
        
        for y, row in enumerate(self.current_block['shape']):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = self.current_block['x'] + x, self.current_block['y'] + y
                    if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
                        self.grid[grid_x, grid_y] = self.current_block['color_idx']
                        
                        # Check adjacency for reward
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = grid_x + dx, grid_y + dy
                            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx, ny] != 0:
                                is_touching = True
                                break
                        if is_touching: break
            if is_touching: break
        
        if is_touching:
            reward += 0.1
        elif self.current_block['y'] + len(self.current_block['shape']) >= self.GRID_HEIGHT:
            # Placed on the floor, not touching anything
            reward -= 0.2
        
        reward += self._check_and_clear_lines()
        
        self.current_block = self.next_block
        self.next_block = self._new_block()
        
        if self._check_collision(self.current_block):
            self.game_over = True
            reward -= 100 # Loss penalty
            
        return reward

    def _check_and_clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if all(self.grid[:, y]):
                lines_to_clear.append(y)
        
        if lines_to_clear:
            # Sound: Line clear
            self.line_clear_animation = {'lines': lines_to_clear, 'timer': 5}
            
            num_cleared = len(lines_to_clear)
            self.lines_cleared += num_cleared
            self.fall_speed = max(0.1, 1.0 - self.lines_cleared * 0.05)
            
            # Reward for line clears
            if num_cleared == 1: return 1
            if num_cleared == 2: return 3
            if num_cleared == 3: return 5
            if num_cleared >= 4: return 10
        
        return 0

    def _finish_line_clear(self):
        lines_to_clear = self.line_clear_animation['lines']
        for y in sorted(lines_to_clear, reverse=True):
            self.grid[:, y:] = np.roll(self.grid[:, y:], -1, axis=1)
            self.grid[:, 0] = 0
        self.line_clear_animation = None
        
    def _update_cooldowns(self):
        for key in self.action_cooldowns:
            if self.action_cooldowns[key] > 0:
                self.action_cooldowns[key] -= 1

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GRID_X + x * self.CELL_SIZE, self.GRID_Y)
            end = (self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_BG, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_X, self.GRID_Y + y * self.CELL_SIZE)
            end = (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_BG, start, end)
            
        # Draw placed blocks
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] != 0:
                    color_idx = int(self.grid[x, y]) - 1
                    color = self.BLOCK_SHAPES[self.shape_keys[color_idx]][1]
                    self._draw_cell(x, y, color)
        
        # Draw line clear animation
        if self.line_clear_animation:
            flash_color = self.COLOR_WHITE if (self.line_clear_animation['timer'] % 2) == 1 else self.COLOR_BG
            for y in self.line_clear_animation['lines']:
                for x in range(self.GRID_WIDTH):
                    self._draw_cell(x, y, flash_color)
            return # Pause rendering of moving blocks during animation
            
        # Draw ghost piece
        ghost_block = self.current_block.copy()
        while not self._check_collision(ghost_block):
            ghost_block['y'] += 1
        ghost_block['y'] -= 1
        self._draw_block(ghost_block, ghost=True)
        
        # Draw current block
        self._draw_block(self.current_block)

    def _draw_block(self, block, ghost=False):
        for y, row in enumerate(block['shape']):
            for x, cell in enumerate(row):
                if cell:
                    self._draw_cell(block['x'] + x, block['y'] + y, block['color'], ghost)

    def _draw_cell(self, grid_x, grid_y, color, ghost=False):
        screen_x = self.GRID_X + grid_x * self.CELL_SIZE
        screen_y = self.GRID_Y + grid_y * self.CELL_SIZE
        rect = pygame.Rect(screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE)

        if ghost:
            pygame.draw.rect(self.screen, color, rect, 2)
        else:
            pygame.gfxdraw.box(self.screen, rect, color)
            
            # 3D effect
            darker_color = tuple(max(0, c - 40) for c in color)
            lighter_color = tuple(min(255, c + 40) for c in color)
            pygame.draw.line(self.screen, darker_color, rect.bottomleft, rect.bottomright, 2)
            pygame.draw.line(self.screen, darker_color, rect.topright, rect.bottomright, 2)
            pygame.draw.line(self.screen, lighter_color, rect.topleft, rect.topright, 2)
            pygame.draw.line(self.screen, lighter_color, rect.topleft, rect.bottomleft, 2)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_main.render(f"{self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, 40))
        self.screen.blit(score_val, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, 65))
        
        # Lines
        lines_text = self.font_main.render(f"LINES", True, self.COLOR_UI_TEXT)
        lines_val = self.font_main.render(f"{self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_WHITE)
        self.screen.blit(lines_text, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, 115))
        self.screen.blit(lines_val, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, 140))
        
        # Next Block
        next_text = self.font_main.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, 200))
        
        next_box_rect = pygame.Rect(self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 15, 230, 5 * self.CELL_SIZE, 5 * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, next_box_rect)
        
        if self.next_block:
            shape = self.next_block['shape']
            w, h = len(shape[0]), len(shape)
            start_x = next_box_rect.centerx - (w * self.CELL_SIZE) / 2
            start_y = next_box_rect.centery - (h * self.CELL_SIZE) / 2
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        rect = pygame.Rect(start_x + x * self.CELL_SIZE, start_y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                        pygame.gfxdraw.box(self.screen, rect, self.next_block['color'])

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else self.COLOR_UI_ACCENT
            
            title_surf = self.font_title.render(message, True, color)
            title_rect = title_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20))
            self.screen.blit(title_surf, title_rect)
            
            score_surf = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_WHITE)
            score_rect = score_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 30))
            self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }

    def close(self):
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override auto_advance for human play
    env.auto_advance = True 
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Falling Block Puzzle")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    running = True
    while running:
        # Reset action at the start of each frame for held keys
        action.fill(0)
        action[1] = 0 # Space released
        action[2] = 0 # Shift released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Map keys to the MultiDiscrete action space
        if keys[pygame.K_UP]:
            action[0] = 1 # Rotate CW
        if keys[pygame.K_DOWN]:
            action[0] = 2 # Soft drop
        if keys[pygame.K_LEFT]:
            action[0] = 3 # Move left
        if keys[pygame.K_RIGHT]:
            action[0] = 4 # Move right
        
        if keys[pygame.K_SPACE]:
            action[1] = 1 # Rotate CCW
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1 # Hard drop
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Lines: {info['lines_cleared']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
        
        clock.tick(30) # Run at 30 FPS

    env.close()