import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Block:
    """Helper class to manage individual block state and animations."""
    def __init__(self, color_idx, col, row, cell_size):
        self.color_idx = color_idx
        self.col = col
        self.row = row
        self.cell_size = cell_size
        self.visual_y = (row - 1) * cell_size
        self.target_y = row * cell_size
        self.state = 'falling'  # 'falling', 'idle', 'clearing'
        self.anim_progress = 0.0

    def update(self):
        """Update block animations. Returns False if the block should be removed."""
        # Smoothly move to target y-position
        if self.state == 'falling':
            lerp_speed = 0.2
            self.visual_y += (self.target_y - self.visual_y) * lerp_speed
            if abs(self.target_y - self.visual_y) < 1:
                self.visual_y = self.target_y
                self.state = 'idle'

        # Process clearing animation
        if self.state == 'clearing':
            self.anim_progress += 0.1
            if self.anim_progress >= 1.0:
                return False  # Signal to remove this block
        return True

    def draw(self, surface, grid_offset, colors, is_in_path, is_cursor):
        """Draw the block on the given surface."""
        x = grid_offset[0] + self.col * self.cell_size
        y = grid_offset[1] + self.visual_y
        
        base_color = colors[self.color_idx]
        shadow_color = tuple(max(0, c - 40) for c in base_color)
        
        size = self.cell_size
        if self.state == 'clearing':
            size *= (1.0 - self.anim_progress)
        
        rect = pygame.Rect(
            x + (self.cell_size - size) / 2,
            y + (self.cell_size - size) / 2,
            size, size
        )
        
        # Draw shadow and block
        shadow_rect = rect.move(0, 4)
        pygame.draw.rect(surface, shadow_color, shadow_rect, border_radius=6)
        pygame.draw.rect(surface, base_color, rect, border_radius=6)
        
        # Highlight if in path or under cursor
        if is_in_path or (is_cursor and not is_in_path):
            highlight_color = (255, 255, 255, 150) if is_in_path else (255, 255, 255, 80)
            highlight_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(highlight_surface, highlight_color, highlight_surface.get_rect(), border_radius=8)
            surface.blit(highlight_surface, rect.topleft)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Hold Space and move to connect blocks. Release Space to clear. Shift to cancel."
    )
    game_description = (
        "Connect three or more same-colored blocks to clear them. New blocks spawn over time. Clear the board to win!"
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame and Display ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- Game Grid and Layout ---
        self.GRID_COLS, self.GRID_ROWS = 10, 10
        self.CELL_SIZE = 36
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_OFFSET = (
            (self.WIDTH - self.GRID_WIDTH) // 2,
            (self.HEIGHT - self.GRID_HEIGHT) // 2 + 10
        )
        
        # --- Colors and Style ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_UI_TEXT = (220, 220, 230)
        self.BLOCK_COLORS = [
            (231, 76, 60),   # Red
            (52, 152, 219),  # Blue
            (46, 204, 113),  # Green
            (241, 196, 15),  # Yellow
            (155, 89, 182),  # Purple
        ]
        
        # --- Game Constants ---
        self.MAX_STEPS = 2000
        self.FPS = 30
        self.BASE_SPAWN_INTERVAL = self.FPS * 2  # Spawn every 2 seconds initially

        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.is_dragging = None
        self.drag_path = None
        self.drag_color_idx = None
        self.last_space_held = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.spawn_timer = None
        self.spawn_rate_modifier = None
        self.particles = None
        self.steps_since_clear = None

        # Call reset to initialize the state.
        # A seed is not passed here, but `reset` will still use the environment's RNG.
        self.reset()
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = [[None for _ in range(self.GRID_ROWS)] for _ in range(self.GRID_COLS)]
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.is_dragging = False
        self.drag_path = []
        self.drag_color_idx = -1
        self.last_space_held = False
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.spawn_timer = 0
        self.spawn_rate_modifier = 1.0
        self.particles = []
        self.steps_since_clear = 0

        # Initial board population
        for _ in range(25):
            col, row = self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(5, self.GRID_ROWS)
            if self.grid[col][row] is None:
                color_idx = self.np_random.integers(0, len(self.BLOCK_COLORS))
                self.grid[col][row] = Block(color_idx, col, row, self.CELL_SIZE)
        
        self._ensure_valid_move()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.steps += 1
        self.steps_since_clear += 1

        # Handle player input
        reward += self._handle_input(movement, space_held, shift_held)
        
        # Update game state (time-based events)
        self._update_game_state()
        
        # Update animations
        self._update_blocks_and_particles()

        # Anti-softlock check
        if self.steps_since_clear > 150:
            if not self._is_any_move_possible():
                self._remove_random_block()
                self.steps_since_clear = 0 # Give player time
            else:
                self.steps_since_clear = 0 # Reset timer if a move is possible

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self._is_grid_empty():
                reward += 100 # Win bonus
            elif self._is_grid_full():
                reward -= 100 # Loss penalty
        
        self.last_space_held = space_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # --- Shift: Cancel Drag ---
        if shift_held and self.is_dragging:
            self.is_dragging = False
            self.drag_path = []
            self.drag_color_idx = -1

        # --- Space: Start/End Drag ---
        space_pressed = space_held and not self.last_space_held
        space_released = not space_held and self.last_space_held
        
        if space_pressed:
            self._start_drag()
        elif space_released:
            reward += self._end_drag()

        # --- Movement: Move Cursor or Extend Drag ---
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = [self.cursor_pos[0] + dx, self.cursor_pos[1] + dy]
            
            if 0 <= new_pos[0] < self.GRID_COLS and 0 <= new_pos[1] < self.GRID_ROWS:
                if self.is_dragging:
                    self._update_drag(new_pos)
                self.cursor_pos = new_pos
        
        return reward

    def _start_drag(self):
        block = self.grid[self.cursor_pos[0]][self.cursor_pos[1]]
        if block and not self.is_dragging:
            self.is_dragging = True
            self.drag_path = [self.cursor_pos]
            self.drag_color_idx = block.color_idx

    def _update_drag(self, new_pos):
        if not self.drag_path: return
        
        last_pos = self.drag_path[-1]
        is_adjacent = abs(new_pos[0] - last_pos[0]) + abs(new_pos[1] - last_pos[1]) == 1
        
        if new_pos in self.drag_path: # Allow backtracking to remove last segment
            if len(self.drag_path) > 1 and new_pos == self.drag_path[-2]:
                self.drag_path.pop()
        elif is_adjacent:
            block = self.grid[new_pos[0]][new_pos[1]]
            if block and block.color_idx == self.drag_color_idx:
                self.drag_path.append(new_pos)

    def _end_drag(self):
        reward = 0
        if self.is_dragging and len(self.drag_path) >= 3:
            num_cleared = len(self.drag_path)
            reward += num_cleared # +1 per block
            if num_cleared > 3:
                reward += 5 # Bonus for longer chains

            for pos in self.drag_path:
                block = self.grid[pos[0]][pos[1]]
                if block:
                    block.state = 'clearing'
                    self._create_particles(pos, block.color_idx, 15)
            self.steps_since_clear = 0
        
        self.is_dragging = False
        self.drag_path = []
        self.drag_color_idx = -1
        return reward

    def _apply_gravity(self):
        for col in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for row in range(self.GRID_ROWS - 1, -1, -1):
                block = self.grid[col][row]
                if block:
                    if row != empty_row:
                        # Move block in logical grid
                        self.grid[col][empty_row] = block
                        self.grid[col][row] = None
                        # Update block's internal state for animation
                        block.row = empty_row
                        block.target_y = empty_row * self.CELL_SIZE
                        block.state = 'falling'
                    empty_row -= 1

    def _update_game_state(self):
        # Increase difficulty over time
        if self.steps > 0 and self.steps % 200 == 0:
            self.spawn_rate_modifier = max(0.2, self.spawn_rate_modifier * 0.95)

        # Spawn new blocks
        self.spawn_timer += 1
        if self.spawn_timer >= self.BASE_SPAWN_INTERVAL * self.spawn_rate_modifier:
            self.spawn_timer = 0
            self._spawn_blocks()
            self._apply_gravity()

    def _spawn_blocks(self):
        empty_top_cells = [c for c in range(self.GRID_COLS) if self.grid[c][0] is None]
        if not empty_top_cells:
            return

        num_to_spawn = self.np_random.integers(1, min(4, len(empty_top_cells) + 1))
        spawn_cols = self.np_random.choice(empty_top_cells, size=num_to_spawn, replace=False)

        for col in spawn_cols:
            color_idx = self.np_random.integers(0, len(self.BLOCK_COLORS))
            # Heuristic to make games more solvable
            if self.np_random.random() < 0.3:
                neighbors = []
                if col > 0 and self.grid[col - 1][0]: neighbors.append(self.grid[col-1][0].color_idx)
                if self.grid[col][1]: neighbors.append(self.grid[col][1].color_idx)
                if neighbors:
                    color_idx = self.np_random.choice(neighbors)

            self.grid[col][0] = Block(color_idx, col, 0, self.CELL_SIZE)
    
    def _update_blocks_and_particles(self):
        # Update blocks and remove cleared ones
        new_grid = [[None for _ in range(self.GRID_ROWS)] for _ in range(self.GRID_COLS)]
        blocks_moved = False
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                block = self.grid[c][r]
                if block:
                    if block.update():
                        new_grid[c][r] = block
                    else: # Block finished clearing animation
                        blocks_moved = True

        self.grid = new_grid
        if blocks_moved:
            self._apply_gravity()

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity on particles
            p['life'] -= 1

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if self._is_grid_full():
            return True
        if self._is_grid_empty():
            return True
        return False

    def _is_grid_full(self):
        return any(self.grid[c][0] is not None for c in range(self.GRID_COLS))

    def _is_grid_empty(self):
        return all(self.grid[c][r] is None for c in range(self.GRID_COLS) for r in range(self.GRID_ROWS))

    def _is_any_move_possible(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                block = self.grid[c][r]
                if block:
                    # Check for chains of 3 starting from this block
                    q = [(c, r)]
                    visited = set([(c, r)])
                    chain = []
                    while q:
                        cx, cy = q.pop(0)
                        chain.append((cx, cy))
                        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS and (nx, ny) not in visited:
                                neighbor = self.grid[nx][ny]
                                if neighbor and neighbor.color_idx == block.color_idx:
                                    visited.add((nx,ny))
                                    q.append((nx,ny))
                    if len(chain) >= 3:
                        return True
        return False
    
    def _ensure_valid_move(self):
        """
        Ensures the initial board has at least one valid move.
        If not, it reshuffles the board until a valid move is present.
        """
        attempts = 0
        while not self._is_any_move_possible():
            attempts += 1
            if attempts > 100: # Failsafe to prevent infinite loop
                self.grid = [[None for _ in range(self.GRID_ROWS)] for _ in range(self.GRID_COLS)]
                return

            # Clear and re-populate the grid, similar to reset()
            self.grid = [[None for _ in range(self.GRID_ROWS)] for _ in range(self.GRID_COLS)]
            for _ in range(25):
                col, row = self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(5, self.GRID_ROWS)
                if self.grid[col][row] is None:
                    color_idx = self.np_random.integers(0, len(self.BLOCK_COLORS))
                    self.grid[col][row] = Block(color_idx, col, row, self.CELL_SIZE)

    def _remove_random_block(self):
        non_empty_indices = [(c, r) for c in range(self.GRID_COLS) for r in range(self.GRID_ROWS) if self.grid[c][r]]
        if non_empty_indices:
            idx = self.np_random.integers(len(non_empty_indices))
            c, r = non_empty_indices[idx]
            block = self.grid[c][r]
            if block:
                self._create_particles((c,r), block.color_idx, 20)
                self.grid[c][r] = None
                self._apply_gravity()

    def _create_particles(self, pos, color_idx, count):
        px, py = self.GRID_OFFSET[0] + pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2, \
                 self.GRID_OFFSET[1] + pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        color = self.BLOCK_COLORS[color_idx]
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(20, 40),
                'color': color
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET[1] + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET[0], y), (self.GRID_OFFSET[0] + self.GRID_WIDTH, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET[0] + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET[1]), (x, self.GRID_OFFSET[1] + self.GRID_HEIGHT), 1)

        # Draw blocks
        path_set = set(map(tuple, self.drag_path))
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                block = self.grid[c][r]
                if block:
                    is_in_path = (c, r) in path_set
                    is_cursor = (c, r) == tuple(self.cursor_pos)
                    block.draw(self.screen, self.GRID_OFFSET, self.BLOCK_COLORS, is_in_path, is_cursor)
        
        # Draw drag line
        if len(self.drag_path) > 1:
            color = self.BLOCK_COLORS[self.drag_color_idx]
            points = [(self.GRID_OFFSET[0] + (c + 0.5) * self.CELL_SIZE, self.GRID_OFFSET[1] + (r + 0.5) * self.CELL_SIZE) for c, r in self.drag_path]
            pygame.draw.lines(self.screen, color, False, points, width=8)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_OFFSET[0] + cx * self.CELL_SIZE, self.GRID_OFFSET[1] + cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, 2, border_radius=4)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), 3, (*p['color'], alpha))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Blocks remaining
        block_count = sum(1 for r in self.grid for block in r if block)
        blocks_text = self.font_main.render(f"Blocks: {block_count}", True, self.COLOR_UI_TEXT)
        self.screen.blit(blocks_text, (self.WIDTH - blocks_text.get_width() - 20, 20))

        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self._is_grid_empty() else "GAME OVER"
            end_text = self.font_main.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks": sum(1 for r in self.grid for block in r if block),
        }

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Validating implementation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # This is a simplified mapping for human play.
    # The agent uses the MultiDiscrete action space.
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Setup Pygame window for human play
    # We need to re-set the video driver to something other than "dummy"
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Clear")
    clock = pygame.time.Clock()

    while not done:
        # --- Event Handling ---
        movement_action = 0
        space_action = 0
        shift_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]: # Reset game
             obs, info = env.reset()

        # Check for pressed keys once per frame
        moved = False
        for key, move in key_to_action.items():
            if keys[key] and not moved:
                movement_action = move
                moved = True
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    env.close()