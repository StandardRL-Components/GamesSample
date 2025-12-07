
# Generated: 2025-08-28T03:17:59.829393
# Source Brief: brief_04880.md
# Brief Index: 4880

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a block. "
        "Select an adjacent, matching block to clear them. Press Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect adjacent blocks of the same color to clear them from the grid. "
        "Plan your moves to clear the entire board for a big bonus!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Visuals & Colors
        self.COLOR_BG = (20, 25, 35)
        self.COLOR_GRID = (40, 50, 65)
        self.COLOR_UI_TEXT = (220, 220, 230)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECT_GLOW = (255, 255, 100)
        self.BLOCK_COLORS = [
            (0, 0, 0),  # 0: Empty
            (220, 50, 50),   # 1: Red
            (50, 220, 50),   # 2: Green
            (50, 100, 220),  # 3: Blue
            (230, 230, 50),  # 4: Yellow
            (180, 50, 230),  # 5: Purple
        ]
        
        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_huge = pygame.font.Font(None, 80)

        # Game constants
        self.GRID_SIZE = 5
        self.NUM_COLORS = 5
        self.MAX_STEPS = 2000
        self.grid_area_width = self.screen_height - 40
        self.block_size = self.grid_area_width // self.GRID_SIZE
        self.grid_offset_x = (self.screen_width - self.grid_area_width) // 2
        self.grid_offset_y = (self.screen_height - self.grid_area_width) // 2
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.selected_block_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []
        self.connection_effect = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._generate_board()
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_block_pos = None
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []
        self.connection_effect = None
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1
        
        reward = 0
        self.connection_effect = None # Clear one-frame effect

        if not self.game_over:
            # Action priority: Shift > Space > Movement
            if shift_pressed:
                # Deselect
                if self.selected_block_pos is not None:
                    self.selected_block_pos = None
                    # Sound: Deselect_SFX

            elif space_pressed:
                # Select or attempt connection
                reward += self._handle_selection()

            elif movement > 0:
                # Move cursor
                if movement == 1: # Up
                    self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
                elif movement == 2: # Down
                    self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE
                elif movement == 3: # Left
                    self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
                elif movement == 4: # Right
                    self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE
                # Sound: Cursor_Move_SFX

        self.steps += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self._is_board_clear():
                reward += 100
                self.win_message = "BOARD CLEARED!"
                # Sound: Win_Jingle_SFX
            elif self.steps >= self.MAX_STEPS:
                self.win_message = "TIME UP"
            else: # No more moves
                reward += -10
                self.win_message = "NO MOVES LEFT"
                # Sound: Lose_Sting_SFX
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_selection(self):
        reward = 0
        cx, cy = self.cursor_pos
        
        if self.grid[cy, cx] == 0:
            # Clicked an empty space
            self.selected_block_pos = None
            # Sound: Error_SFX
            return reward

        if self.selected_block_pos is None:
            # First selection
            self.selected_block_pos = (cx, cy)
            if not self._has_valid_moves(cx, cy):
                reward -= 0.1
            # Sound: Select_SFX
        else:
            # Second selection (attempt to connect)
            sx, sy = self.selected_block_pos
            if self._is_adjacent((sx, sy), (cx, cy)) and self.grid[sy, sx] == self.grid[cy, cx]:
                # Successful connection
                color_index = self.grid[sy, sx]
                color = self.BLOCK_COLORS[color_index]
                
                # Check for color clear bonus before removing blocks
                color_counts = self._get_color_counts()
                if color_counts.get(color_index, 0) == 2:
                    reward += 5 # Bonus for clearing the last two of a color
                
                # Remove blocks and add effects
                self.grid[sy, sx] = 0
                self.grid[cy, cx] = 0
                self._spawn_particles(self._grid_to_pixel(sx, sy), color)
                self._spawn_particles(self._grid_to_pixel(cx, cy), color)
                self.connection_effect = (self._grid_to_pixel(sx, sy), self._grid_to_pixel(cx, cy), color)
                
                self.selected_block_pos = None
                reward += 2 # +1 for each block
                # Sound: Connect_Success_SFX
            else:
                # Failed connection (different color, not adjacent, or same block)
                if (sx, sy) == (cx, cy): # Clicked same block again
                    self.selected_block_pos = None # Deselect
                    # Sound: Deselect_SFX
                else: # Clicked a different, invalid block
                    self.selected_block_pos = (cx, cy) # Select the new block instead
                    # Sound: Select_SFX
        return reward

    def _check_termination(self):
        if self._is_board_clear():
            return True
        if not self._has_any_valid_moves():
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_draw_particles()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": list(self.cursor_pos),
            "selected_pos": self.selected_block_pos,
            "color_counts": self._get_color_counts(),
        }
        
    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            start_x = self.grid_offset_x + i * self.block_size
            start_y = self.grid_offset_y + i * self.block_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, self.grid_offset_y), (start_x, self.grid_offset_y + self.grid_area_width), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, start_y), (self.grid_offset_x + self.grid_area_width, start_y), 1)

        # Draw blocks
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = self.grid[r, c]
                if color_index > 0:
                    px, py = self._grid_to_pixel(c, r)
                    rect = pygame.Rect(px - self.block_size // 2, py - self.block_size // 2, self.block_size, self.block_size)
                    inner_rect = rect.inflate(-8, -8)
                    pygame.draw.rect(self.screen, self.BLOCK_COLORS[color_index], inner_rect, border_radius=8)

        # Draw selection glow
        if self.selected_block_pos is not None:
            sx, sy = self.selected_block_pos
            px, py = self._grid_to_pixel(sx, sy)
            for i in range(1, 6):
                glow_size = self.block_size - 8 + i * 2
                glow_alpha = 100 - i * 15
                s = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
                pygame.draw.rect(s, (*self.COLOR_SELECT_GLOW, glow_alpha), s.get_rect(), border_radius=10)
                self.screen.blit(s, (px - glow_size // 2, py - glow_size // 2))
        
        # Draw connection effect
        if self.connection_effect:
            p1, p2, color = self.connection_effect
            pygame.draw.line(self.screen, color, p1, p2, 6)

        # Draw cursor
        cx, cy = self.cursor_pos
        px, py = self._grid_to_pixel(cx, cy)
        cursor_rect = pygame.Rect(px - self.block_size // 2, py - self.block_size // 2, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=10)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Color counts
        color_counts = self._get_color_counts()
        for i in range(1, self.NUM_COLORS + 1):
            count = color_counts.get(i, 0)
            color = self.BLOCK_COLORS[i]
            y_pos = 60 + (i - 1) * 25
            pygame.draw.rect(self.screen, color, (20, y_pos, 20, 20), border_radius=4)
            count_text = self.font_small.render(f": {count}", True, self.COLOR_UI_TEXT)
            self.screen.blit(count_text, (45, y_pos))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg_surf = self.font_huge.render(self.win_message, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _grid_to_pixel(self, c, r):
        px = self.grid_offset_x + c * self.block_size + self.block_size // 2
        py = self.grid_offset_y + r * self.block_size + self.block_size // 2
        return px, py

    def _generate_board(self):
        while True:
            board = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
            if self._has_any_valid_moves(board):
                return board

    def _is_valid(self, r, c):
        return 0 <= r < self.GRID_SIZE and 0 <= c < self.GRID_SIZE

    def _is_adjacent(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) + abs(y1 - y2) == 1

    def _has_valid_moves(self, c, r, board=None):
        if board is None:
            board = self.grid
        
        color = board[r, c]
        if color == 0:
            return False
            
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if self._is_valid(nr, nc) and board[nr, nc] == color:
                return True
        return False

    def _has_any_valid_moves(self, board=None):
        if board is None:
            board = self.grid
            
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self._has_valid_moves(c, r, board):
                    return True
        return False

    def _is_board_clear(self):
        return np.all(self.grid == 0)

    def _get_color_counts(self):
        unique, counts = np.unique(self.grid[self.grid > 0], return_counts=True)
        return dict(zip(unique, counts))

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            self.particles.append(self.Particle(pos, color, self.np_random))

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p.update()
            if p.is_alive():
                p.draw(self.screen)
                active_particles.append(p)
        self.particles = active_particles

    class Particle:
        def __init__(self, pos, color, rng):
            self.x, self.y = pos
            angle = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(1, 4)
            self.vx = math.cos(angle) * speed
            self.vy = math.sin(angle) * speed
            self.color = color
            self.lifetime = rng.integers(15, 30)
            self.size = rng.integers(3, 7)

        def update(self):
            self.x += self.vx
            self.y += self.vy
            self.lifetime -= 1
            self.size = max(0, self.size - 0.2)

        def is_alive(self):
            return self.lifetime > 0

        def draw(self, surface):
            if self.size > 1:
                pygame.gfxdraw.filled_circle(
                    surface, int(self.x), int(self.y), int(self.size), self.color
                )
    
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a different screen for rendering if playing directly
    render_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Block Connector")
    
    terminated = False
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    
    while not terminated:
        # --- Human Controls ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        # This is a bit tricky for one-off key presses. We'll use the event queue.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 1
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()

        action = [movement, space, shift]
        
        # Only step if an action was taken
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # Render the observation to the display screen
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(render_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS
        
    print("Game Over!")
    pygame.time.wait(2000)
    env.close()