
# Generated: 2025-08-28T03:45:04.460353
# Source Brief: brief_02117.md
# Brief Index: 2117

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a block, "
        "then move to an adjacent, same-colored block and press Space again to connect and clear."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect adjacent same-colored blocks to clear them. Clearing an entire horizontal "
        "row scores a Line Clear. Clear 20 lines in 50 moves to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 12
    GRID_ROWS = 8
    BLOCK_SIZE = 40
    GRID_WIDTH = GRID_COLS * BLOCK_SIZE
    GRID_HEIGHT = GRID_ROWS * BLOCK_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2
    
    WIN_LINES = 20
    MAX_MOVES = 50
    MAX_STEPS = 500

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_TEXT = (230, 230, 240)
    COLOR_CURSOR = (0, 255, 255)
    COLOR_SELECT = (255, 255, 255)
    BLOCK_COLORS = [
        (255, 65, 54),    # Red
        (46, 204, 64),    # Green
        (0, 116, 217),    # Blue
        (255, 220, 0),    # Yellow
        (177, 13, 201),    # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_info = pygame.font.SysFont("Arial", 20)
        self.font_gameover = pygame.font.SysFont("Arial", 48, bold=True)

        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.cursor_pos = [0, 0]
        self.selected_pos = None
        self.moves_remaining = 0
        self.lines_cleared = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.prev_space_held = False
        self.particles = []
        self.invalid_move_flash = 0
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._initialize_grid()
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.lines_cleared = 0
        self.game_over = False
        self.win = False
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_pos = None
        self.prev_space_held = False
        self.particles = []
        self.invalid_move_flash = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self._handle_input(movement, space_held)
        
        connection_result = self._process_connection(space_held)
        if connection_result:
            reward += connection_result["reward"]
            if connection_result["valid"]:
                # Sound: block_clear.wav
                pass

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.lines_cleared >= self.WIN_LINES:
                self.win = True
                reward += 100
                # Sound: win_game.wav
            else:
                reward -= 100
                # Sound: lose_game.wav
        
        if not terminated:
            reward -= 0.01 # Small penalty for each step to encourage efficiency

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
    
    def _process_connection(self, space_held):
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        if not space_pressed:
            return None

        result = {"reward": 0, "valid": False}

        if not self.selected_pos:
            self.selected_pos = list(self.cursor_pos)
            # Sound: select.wav
        else:
            pos1 = self.selected_pos
            pos2 = self.cursor_pos
            
            is_adjacent = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1
            color1 = self.grid[pos1[1], pos1[0]]
            color2 = self.grid[pos2[1], pos2[0]]
            is_same_color = color1 == color2 and color1 > 0
            
            if is_adjacent and is_same_color:
                result["valid"] = True
                self.moves_remaining -= 1
                
                blocks_to_clear = self._find_connected_blocks(pos1)
                result["reward"] += len(blocks_to_clear)
                
                temp_grid = self.grid.copy()
                for x, y in blocks_to_clear:
                    temp_grid[y, x] = 0
                
                lines_just_cleared = self._check_line_clears(temp_grid, blocks_to_clear)
                self.lines_cleared += lines_just_cleared
                result["reward"] += lines_just_cleared * 5
                
                self._spawn_particles(blocks_to_clear, self.BLOCK_COLORS[color1 - 1])
                self._apply_gravity_and_refill(blocks_to_clear)
            else:
                result["reward"] -= 0.1
                self.invalid_move_flash = 10
                # Sound: invalid_move.wav
                
            self.selected_pos = None
        
        return result

    def _initialize_grid(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                self.grid[r, c] = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)

    def _find_connected_blocks(self, start_pos):
        q = [tuple(start_pos)]
        visited = {tuple(start_pos)}
        color_to_match = self.grid[start_pos[1], start_pos[0]]
        
        head = 0
        while head < len(q):
            x, y = q[head]
            head += 1
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    if (nx, ny) not in visited and self.grid[ny, nx] == color_to_match:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return list(visited)

    def _check_line_clears(self, grid_state, cleared_blocks):
        cleared_rows_indices = {y for _, y in cleared_blocks}
        lines = 0
        for r in cleared_rows_indices:
            if np.all(grid_state[r, :] == 0):
                lines += 1
        return lines

    def _apply_gravity_and_refill(self, cleared_blocks):
        cleared_cols = sorted(list({x for x, y in cleared_blocks}))
        
        for c in cleared_cols:
            write_idx = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if (c, r) not in cleared_blocks:
                    self.grid[write_idx, c] = self.grid[r, c]
                    write_idx -= 1
            
            for r in range(write_idx, -1, -1):
                self.grid[r, c] = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)

    def _check_termination(self):
        return (
            self.lines_cleared >= self.WIN_LINES or
            self.moves_remaining <= 0 or
            self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        if self.invalid_move_flash > 0:
            self.screen.fill((80, 20, 20), special_flags=pygame.BLEND_RGB_ADD)
            self.invalid_move_flash -= 1

        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + r * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X + c * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT), 1)

        # Draw blocks
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_idx = self.grid[r, c]
                if color_idx > 0:
                    color = self.BLOCK_COLORS[color_idx - 1]
                    rect = pygame.Rect(
                        self.GRID_X + c * self.BLOCK_SIZE,
                        self.GRID_Y + r * self.BLOCK_SIZE,
                        self.BLOCK_SIZE, self.BLOCK_SIZE
                    )
                    # Draw a slightly smaller inner rect for a border effect
                    inner_rect = rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, color, inner_rect, border_radius=5)

        # Draw selected block highlight
        if self.selected_pos:
            x, y = self.selected_pos
            rect = pygame.Rect(
                self.GRID_X + x * self.BLOCK_SIZE,
                self.GRID_Y + y * self.BLOCK_SIZE,
                self.BLOCK_SIZE, self.BLOCK_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECT, rect.inflate(-2,-2), 2, border_radius=7)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_X + cx * self.BLOCK_SIZE,
            self.GRID_Y + cy * self.BLOCK_SIZE,
            self.BLOCK_SIZE, self.BLOCK_SIZE
        )
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        line_width = int(2 + pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width, border_radius=8)

        self._update_and_draw_particles()

    def _render_ui(self):
        # Lines Cleared
        lines_text = f"LINES: {self.lines_cleared} / {self.WIN_LINES}"
        lines_surf = self.font_info.render(lines_text, True, self.COLOR_TEXT)
        self.screen.blit(lines_surf, (15, 10))

        # Moves Remaining
        moves_text = f"MOVES: {self.moves_remaining}"
        moves_surf = self.font_info.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (self.SCREEN_WIDTH - moves_surf.get_width() - 15, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            text_surf = self.font_gameover.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "moves_remaining": self.moves_remaining,
        }
        
    def _spawn_particles(self, positions, color):
        for x, y in positions:
            for _ in range(5):
                px = self.GRID_X + x * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
                py = self.GRID_Y + y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = self.np_random.integers(20, 40)
                self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # a little gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                radius = max(0, (p['life'] / 40) * 4)
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), int(radius), p['color']
                )

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("BlockLink")
    clock = pygame.time.Clock()
    
    movement = 0
    space = 0
    shift = 0

    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        # Get key states for continuous actions
        keys = pygame.key.get_pressed()
        
        # Reset actions each frame
        movement = 0
        space = 0
        
        if not done:
            # Map keys to actions
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1
            
            # Since auto_advance is False, we only step when a key is pressed
            if movement != 0 or space != 0 or (event.type == pygame.KEYUP and event.key == pygame.K_SPACE):
                action = [movement, space, shift]
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Lines: {info['lines_cleared']}, Moves: {info['moves_remaining']}")

        # Rendering
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(15) # Limit frame rate for manual play

    env.close()
    pygame.quit()