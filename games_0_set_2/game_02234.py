
# Generated: 2025-08-27T19:43:19.988634
# Source Brief: brief_02234.md
# Brief Index: 2234

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a square. "
        "Select a second, adjacent, same-colored square to connect and clear them. "
        "Press Shift to deselect."
    )

    game_description = (
        "A minimalist puzzle game. Connect all same-colored blocks into single groups "
        "before you run out of moves. Clearing blocks causes new ones to fall from the top."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.CELL_SIZE = 40
        self.MAX_MOVES = 50
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_EMPTY = self.COLOR_BG
        self.PALETTE = [
            (255, 70, 70),   # Red
            (70, 200, 255),  # Blue
            (100, 255, 100), # Green
            (255, 220, 90),  # Yellow
            (200, 100, 255), # Purple
        ]
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 230)
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # State variables (initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.selected_squares = None
        self.moves_remaining = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        
        self.reset()

        # Run validation check
        # self.validate_implementation() # Comment out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.grid = self._generate_grid()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_squares = []
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self._update_particles()
        
        # 1. Handle cursor movement
        self._move_cursor(movement)

        # 2. Handle deselection (Shift)
        if shift_pressed and self.selected_squares:
            self.selected_squares.clear()
            # sfx: deselect_sound

        # 3. Handle selection/connection (Space)
        if space_pressed:
            reward += self._handle_selection()
        
        self.steps += 1
        
        # 4. Check for termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if not self.game_over and self.moves_remaining <= 0:
            if self._check_win_condition():
                reward += 100
                self.score += 100
                # sfx: win_sound
            else:
                reward -= 100
                self.score -= 100
                # sfx: lose_sound
            self.game_over = True
            terminated = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

    def _handle_selection(self):
        cx, cy = self.cursor_pos
        
        if (cx, cy) in self.selected_squares:
            return 0 # Cannot select the same square twice

        if not self.selected_squares:
            self.selected_squares.append((cx, cy))
            # sfx: select_1_sound
            return 0
        
        if len(self.selected_squares) == 1:
            self.selected_squares.append((cx, cy))
            x1, y1 = self.selected_squares[0]
            x2, y2 = self.selected_squares[1]
            
            is_adjacent = abs(x1 - x2) + abs(y1 - y2) == 1
            is_same_color = self.grid[y1][x1] == self.grid[y2][x2]

            if is_adjacent and is_same_color:
                # sfx: connect_success_sound
                color_index = self.grid[y1][x1]
                
                self.grid[y1][x1] = -1 # Mark for removal
                self.grid[y2][x2] = -1
                
                self._create_particles(x1, y1, color_index)
                self._create_particles(x2, y2, color_index)
                
                self._apply_gravity_and_refill()
                
                self.moves_remaining -= 1
                self.selected_squares.clear()
                
                # Check for rewards and win condition
                reward = 1.0
                
                # Check if this completed a color group
                if self._is_color_group_complete(color_index):
                    reward += 10.0
                
                # Check for overall win
                if self._check_win_condition():
                    reward += 100.0
                    self.game_over = True
                    # sfx: win_sound
                
                return reward
            else:
                # sfx: connect_fail_sound
                self.selected_squares.clear()
                return 0 # Invalid connection
        return 0

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_slots = []
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] == -1:
                    empty_slots.append(y)
                elif empty_slots:
                    new_y = empty_slots.pop(0)
                    self.grid[new_y][x] = self.grid[y][x]
                    self.grid[y][x] = -1
                    empty_slots.append(y)
            
            for y in empty_slots:
                self.grid[y][x] = self.np_random.integers(0, len(self.PALETTE))


    def _is_color_group_complete(self, color_index):
        all_squares = []
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] == color_index:
                    all_squares.append((x, y))
        
        if not all_squares:
            return True # Color is cleared
        
        return self._bfs_check(all_squares[0], color_index) == len(all_squares)

    def _check_win_condition(self):
        colors_on_grid = set()
        for row in self.grid:
            for cell in row:
                if cell != -1:
                    colors_on_grid.add(cell)

        if not colors_on_grid:
            return True

        for color_index in colors_on_grid:
            if not self._is_color_group_complete(color_index):
                return False
        return True

    def _bfs_check(self, start_node, color_index):
        q = deque([start_node])
        visited = {start_node}
        count = 0
        while q:
            x, y = q.popleft()
            count += 1
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    neighbor = (nx, ny)
                    if neighbor not in visited and self.grid[ny][nx] == color_index:
                        visited.add(neighbor)
                        q.append(neighbor)
        return count

    def _generate_grid(self):
        grid = [[self.np_random.integers(0, len(self.PALETTE)) for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        # Ensure at least one valid move
        while not self._has_valid_move(grid):
             grid = [[self.np_random.integers(0, len(self.PALETTE)) for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        return grid

    def _has_valid_move(self, grid):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color = grid[y][x]
                # Check right
                if x + 1 < self.GRID_WIDTH and grid[y][x+1] == color:
                    return True
                # Check down
                if y + 1 < self.GRID_HEIGHT and grid[y+1][x] == color:
                    return True
        return False

    def _create_particles(self, grid_x, grid_y, color_index):
        center_x = grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        center_y = grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        color = self.PALETTE[color_index]
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 40)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'color': color, 'radius': radius})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT), 1)
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py), 1)
        
        # Draw squares
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_index = self.grid[y][x]
                if color_index != -1:
                    color = self.PALETTE[color_index]
                    rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    
                    # Bevel effect
                    highlight = tuple(min(255, c + 40) for c in color)
                    shadow = tuple(max(0, c - 40) for c in color)
                    pygame.draw.rect(self.screen, shadow, rect)
                    pygame.draw.rect(self.screen, highlight, rect.inflate(-4, -4))
                    pygame.draw.rect(self.screen, color, rect.inflate(-8, -8))

        # Draw selected squares highlight
        for i, (sx, sy) in enumerate(self.selected_squares):
            rect = pygame.Rect(sx * self.CELL_SIZE, sy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            highlight_color = (255, 255, 255) if i == 0 else (200, 200, 255)
            pygame.draw.rect(self.screen, highlight_color, rect, 4, border_radius=4)
        
        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(cx * self.CELL_SIZE, cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=4)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 40.0))))
            color_with_alpha = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color_with_alpha)


    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (10, 10))

        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            won = self._check_win_condition()
            end_text_str = "YOU WIN!" if won else "GAME OVER"
            end_color = (100, 255, 100) if won else (255, 70, 70)
            
            end_text = self.font_game_over.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "is_win": self.game_over and self._check_win_condition()
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a real screen for manual play
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Connect")
    
    terminated = False
    clock = pygame.time.Clock()
    
    while not terminated:
        movement = 0 # No-op
        space_pressed = 0
        shift_pressed = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_pressed = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_pressed = 1
            
        action = [movement, space_pressed, shift_pressed]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")

        # Blit the observation from the env's buffer to the real screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(10) # Control the speed of manual play

    print("Game Over!")
    print(f"Final Info: {info}")
    pygame.time.wait(2000)
    env.close()