
# Generated: 2025-08-27T18:26:29.258708
# Source Brief: brief_01830.md
# Brief Index: 1830

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A puzzle game where the player clears groups of same-colored blocks from a grid.

    The goal is to clear the entire board within a limited number of moves, maximizing the score.
    Score is awarded based on the number of blocks cleared in a single move.
    The game is turn-based and controlled via a cursor.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to clear the highlighted group of blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the board by selecting groups of 2 or more same-colored blocks. "
        "You have a limited number of moves. Plan your moves to create larger groups for a higher score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.BLOCK_SIZE = 32
        self.GRID_LINE_WIDTH = 2
        
        self.GRID_AREA_WIDTH = self.GRID_WIDTH * self.BLOCK_SIZE
        self.GRID_AREA_HEIGHT = self.GRID_HEIGHT * self.BLOCK_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_AREA_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_AREA_HEIGHT) - 10

        self.MAX_MOVES = 25
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = pygame.Color("#101018")
        self.COLOR_GRID = pygame.Color("#202030")
        self.COLOR_TEXT = pygame.Color("#E0E0FF")
        self.COLOR_SCORE = pygame.Color("#F0C040")
        self.COLOR_MOVES = pygame.Color("#40C0F0")
        self.COLOR_CURSOR = pygame.Color("#FFFFFF")
        
        self.BLOCK_COLORS = [
            pygame.Color("#000000"),  # 0: Empty
            pygame.Color("#E63946"),  # 1: Red
            pygame.Color("#F1FAEE"),  # 2: White
            pygame.Color("#A8DADC"),  # 3: Light Blue
            pygame.Color("#457B9D"),  # 4: Dark Blue
            pygame.Color("#1D3557"),  # 5: Navy
        ]
        self.NUM_COLORS = len(self.BLOCK_COLORS) - 1
        
        self.HIGHLIGHT_BRIGHTNESS = 40

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # --- Game State ---
        self.np_random = None
        self.grid = None
        self.cursor_pos = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.highlighted_group = set()
        
        # Initialize state variables
        self.reset()
        
        # Self-validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False
        self.particles = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        # Keep generating grids until we get one with a valid move
        while True:
            self._generate_grid()
            if self._has_valid_moves():
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # 1. Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
        elif movement == 2:  # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_HEIGHT
        elif movement == 3:  # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
        elif movement == 4:  # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_WIDTH
        
        # 2. Handle block clearing action if space is pressed
        if space_pressed:
            self.steps += 1
            self.moves_left -= 1

            group_to_clear = self._find_connected_group(self.cursor_pos[0], self.cursor_pos[1])
            num_cleared = len(group_to_clear)

            if num_cleared > 1:
                # sfx_clear_blocks
                color_index = self.grid[self.cursor_pos[0], self.cursor_pos[1]]
                
                for x, y in group_to_clear:
                    self.grid[x, y] = 0
                    self._create_particles(x, y, self.BLOCK_COLORS[color_index])

                self._apply_gravity_and_collapse()
                
                # Reward calculation
                reward += num_cleared  # +1 per block
                if num_cleared >= 10:
                    reward += 5
                elif num_cleared == 2: # Penalty only for smallest valid group
                    reward -= 1
            else:
                # sfx_invalid_move
                reward = -0.2
            
            self.score += reward
        
        # 3. Update animations
        self._update_particles()
        
        # 4. Check for termination
        terminated = False
        board_cleared = np.all(self.grid == 0)
        no_more_moves = not self._has_valid_moves() if not board_cleared else False

        if self.moves_left <= 0 or board_cleared or no_more_moves or self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            if board_cleared:
                # sfx_win_game
                self.win = True
                final_bonus = 100
                reward += final_bonus
                self.score += final_bonus

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_highlighted_group()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}
        
    def _generate_grid(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))

    def _find_connected_group(self, start_x, start_y):
        if self.grid[start_x, start_y] == 0:
            return set()

        target_color = self.grid[start_x, start_y]
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        
        while q:
            x, y = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    if self.grid[nx, ny] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return visited

    def _apply_gravity_and_collapse(self):
        new_grid = np.zeros_like(self.grid)
        new_col_idx = 0
        for x in range(self.GRID_WIDTH):
            col_has_blocks = False
            new_row_idx = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] != 0:
                    col_has_blocks = True
                    new_grid[new_col_idx, new_row_idx] = self.grid[x, y]
                    new_row_idx -= 1
            if col_has_blocks:
                new_col_idx += 1
        self.grid = new_grid

    def _has_valid_moves(self):
        visited = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] != 0 and (x, y) not in visited:
                    group = self._find_connected_group(x, y)
                    if len(group) > 1:
                        return True
                    visited.update(group)
        return False
        
    def _update_highlighted_group(self):
        if self.game_over:
            self.highlighted_group = set()
            return
            
        cx, cy = self.cursor_pos
        if self.grid[cx, cy] != 0:
            group = self._find_connected_group(cx, cy)
            if len(group) > 1:
                self.highlighted_group = group
            else:
                self.highlighted_group = set()
        else:
            self.highlighted_group = set()

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_AREA_WIDTH, self.GRID_AREA_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw blocks
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color_index = self.grid[x, y]
                if color_index == 0:
                    continue

                base_color = self.BLOCK_COLORS[color_index]
                is_highlighted = (x, y) in self.highlighted_group
                
                r, g, b = base_color.r, base_color.g, base_color.b
                if is_highlighted:
                    pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2 * 20
                    r = min(255, r + self.HIGHLIGHT_BRIGHTNESS + pulse)
                    g = min(255, g + self.HIGHLIGHT_BRIGHTNESS + pulse)
                    b = min(255, b + self.HIGHLIGHT_BRIGHTNESS + pulse)
                
                block_rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.BLOCK_SIZE,
                    self.GRID_OFFSET_Y + y * self.BLOCK_SIZE,
                    self.BLOCK_SIZE, self.BLOCK_SIZE
                )
                inner_rect = block_rect.inflate(-self.GRID_LINE_WIDTH, -self.GRID_LINE_WIDTH)
                pygame.gfxdraw.box(self.screen, inner_rect, (r, g, b))

        # Draw particles
        for p in self.particles:
            p_color = p['color']
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), (*p_color[:3], alpha))

        # Draw cursor
        if not self.game_over:
            cx, cy = self.cursor_pos
            cursor_rect = pygame.Rect(
                self.GRID_OFFSET_X + cx * self.BLOCK_SIZE,
                self.GRID_OFFSET_Y + cy * self.BLOCK_SIZE,
                self.BLOCK_SIZE, self.BLOCK_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=3)
    
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (15, 15))

        # Moves
        moves_text = self.font_ui.render(f"MOVES: {self.moves_left}", True, self.COLOR_MOVES)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 15, 15))
        self.screen.blit(moves_text, moves_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_SCORE if self.win else self.COLOR_TEXT
            
            end_text = self.font_game_over.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(end_text, end_rect)

            final_score_text = self.font_ui.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 30))
            self.screen.blit(final_score_text, final_score_rect)

    def _create_particles(self, grid_x, grid_y, color):
        px = self.GRID_OFFSET_X + (grid_x + 0.5) * self.BLOCK_SIZE
        py = self.GRID_OFFSET_Y + (grid_y + 0.5) * self.BLOCK_SIZE
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [px, py], 'vel': vel, 'color': color, 
                'life': life, 'max_life': life, 'radius': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

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
    # This block allows you to play the game with a keyboard for testing.
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Clear Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    last_space_press = False # To detect a single press event

    while running:
        # Action defaults
        movement = 0 # No-op
        space_pressed = 0
        shift_pressed = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get key presses
        keys = pygame.key.get_pressed()
        
        # For turn-based, only take one action per frame
        # This simple logic prioritizes movement over clearing
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # Handle single press for space bar
        current_space_press = keys[pygame.K_SPACE]
        if current_space_press and not last_space_press:
            space_pressed = 1
        last_space_press = current_space_press
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_pressed = 1
        
        action = [movement, space_pressed, shift_pressed]
        
        # Only step the environment if an action was taken
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                print(f"Game Over! Final Score: {info['score']}")
                # Render the final frame
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                # Wait a bit before resetting
                pygame.time.wait(3000)
                obs, info = env.reset()
        else:
            # If no action, just re-render the current state
            obs = env._get_observation()

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(15) # Limit to 15 FPS for human play

    pygame.quit()