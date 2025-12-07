
# Generated: 2025-08-28T03:34:19.629007
# Source Brief: brief_02027.md
# Brief Index: 2027

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move the cursor. Press space to reveal a square. Hold shift to flag/unflag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a grid, revealing safe squares while avoiding hidden mines to clear the board."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = (10, 10)
        self.NUM_MINES = 10
        self.MAX_STEPS = 1000

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24)
            self.font_large = pygame.font.SysFont("Consolas", 64)
            self.font_ui = pygame.font.SysFont("Lucida Console", 18)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 28)
            self.font_large = pygame.font.SysFont(None, 72)
            self.font_ui = pygame.font.SysFont(None, 22)

        # Colors
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_GRID = (60, 60, 70)
        self.COLOR_UNREVEALED = (120, 120, 140)
        self.COLOR_REVEALED_SAFE = (80, 90, 100)
        self.COLOR_REVEALED_MINE = (200, 50, 50)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_FLAG = (255, 120, 0)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_NUMBERS = [
            (0, 0, 0), # 0 is not rendered
            (100, 150, 255), # 1
            (100, 200, 100), # 2
            (255, 100, 100), # 3
            (150, 100, 255), # 4
            (255, 150, 50),  # 5
            (100, 220, 220), # 6
            (220, 220, 100), # 7
            (200, 200, 200), # 8
        ]

        # Grid positioning
        self.cell_size = 32
        self.grid_width = self.GRID_SIZE[0] * self.cell_size
        self.grid_height = self.GRID_SIZE[1] * self.cell_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2
        
        # Initialize state variables
        self.mine_grid = None
        self.revealed_grid = None
        self.flagged_grid = None
        self.number_grid = None
        self.cursor_pos = None
        self.game_over = None
        self.win = None
        self.steps = None
        self.score = None
        self.last_space_press = False
        self.last_shift_press = False
        self.explosion_particles = []

        self.reset()

        # Run validation check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_space_press = False
        self.last_shift_press = False
        self.explosion_particles = []

        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        
        # Initialize grids
        self.mine_grid = np.zeros(self.GRID_SIZE, dtype=bool)
        self.revealed_grid = np.zeros(self.GRID_SIZE, dtype=bool)
        self.flagged_grid = np.zeros(self.GRID_SIZE, dtype=bool)
        self.number_grid = np.zeros(self.GRID_SIZE, dtype=int)
        
        # Place mines
        mine_indices = self.np_random.choice(self.GRID_SIZE[0] * self.GRID_SIZE[1], self.NUM_MINES, replace=False)
        self.mine_grid.flat[mine_indices] = True
        
        # Calculate adjacent mine counts
        for r in range(self.GRID_SIZE[1]):
            for c in range(self.GRID_SIZE[0]):
                if not self.mine_grid[r, c]:
                    count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.GRID_SIZE[1] and 0 <= nc < self.GRID_SIZE[0] and self.mine_grid[nr, nc]:
                                count += 1
                    self.number_grid[r, c] = count
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if not self.game_over:
            movement = action[0]
            space_pressed = action[1] == 1
            shift_pressed = action[2] == 1

            # Handle cursor movement
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1 # Down
            elif movement == 3: self.cursor_pos[0] -= 1 # Left
            elif movement == 4: self.cursor_pos[0] += 1 # Right
            
            # Wrap cursor around edges
            self.cursor_pos[0] %= self.GRID_SIZE[0]
            self.cursor_pos[1] %= self.GRID_SIZE[1]

            # Handle actions (rising edge detection)
            action_taken = False
            if space_pressed and not self.last_space_press:
                action_taken = True
                reward += self._reveal_square(self.cursor_pos)
            elif shift_pressed and not self.last_shift_press:
                action_taken = True
                reward += self._toggle_flag(self.cursor_pos)
            
            self.last_space_press = space_pressed
            self.last_shift_press = shift_pressed

            # Check for win condition
            if not self.game_over:
                num_safe_squares = self.GRID_SIZE[0] * self.GRID_SIZE[1] - self.NUM_MINES
                if np.sum(self.revealed_grid) == num_safe_squares:
                    self.game_over = True
                    self.win = True
                    reward += 100

        self.score += reward
        self.steps += 1
        
        # Check termination conditions
        if self.game_over or self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reveal_square(self, pos):
        r, c = pos[1], pos[0]
        
        if self.revealed_grid[r, c] or self.flagged_grid[r, c]:
            return -0.1 # Penalty for wasted action

        self.revealed_grid[r, c] = True

        if self.mine_grid[r, c]:
            self.game_over = True
            self.win = False
            self._create_explosion(pos)
            # Reveal all mines on loss
            for r_idx in range(self.GRID_SIZE[1]):
                for c_idx in range(self.GRID_SIZE[0]):
                    if self.mine_grid[r_idx, c_idx]:
                        self.revealed_grid[r_idx, c_idx] = True
            return -100
        
        # Reward for revealing a safe square
        reward = 1.0

        # Discourage revealing squares adjacent to zeros (less informative moves)
        is_adj_to_zero = False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_SIZE[1] and 0 <= nc < self.GRID_SIZE[0] and self.revealed_grid[nr, nc] and self.number_grid[nr, nc] == 0:
                    is_adj_to_zero = True
                    break
            if is_adj_to_zero: break
        if is_adj_to_zero:
            reward -= 0.2

        if self.number_grid[r, c] == 0:
            self._flood_fill(pos)
        
        return reward

    def _flood_fill(self, pos):
        q = deque([pos])
        visited = {tuple(pos)}
        
        while q:
            c, r = q.popleft()
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    
                    nr, nc = r + dr, c + dc
                    
                    if 0 <= nr < self.GRID_SIZE[1] and 0 <= nc < self.GRID_SIZE[0]:
                        if not self.revealed_grid[nr, nc] and not self.flagged_grid[nr, nc] and (nr, nc) not in visited:
                            self.revealed_grid[nr, nc] = True
                            if self.number_grid[nr, nc] == 0:
                                q.append((nc, nr))
                                visited.add((nc, nr))

    def _toggle_flag(self, pos):
        r, c = pos[1], pos[0]
        if not self.revealed_grid[r, c]:
            self.flagged_grid[r, c] = not self.flagged_grid[r, c]
        return 0 # No reward for flagging

    def _create_explosion(self, pos):
        # sound: play explosion
        c, r = pos[0], pos[1]
        center_x = self.grid_offset_x + c * self.cell_size + self.cell_size // 2
        center_y = self.grid_offset_y + r * self.cell_size + self.cell_size // 2
        
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            dx = math.cos(angle) * speed
            dy = math.sin(angle) * speed
            color = random.choice([(255, 100, 0), (255, 200, 0), (200, 200, 200)])
            lifetime = random.randint(20, 40)
            self.explosion_particles.append([center_x, center_y, dx, dy, lifetime, color])

    def _update_and_draw_particles(self):
        remaining_particles = []
        for p in self.explosion_particles:
            p[0] += p[2] # x += dx
            p[1] += p[3] # y += dy
            p[4] -= 1    # lifetime--
            if p[4] > 0:
                remaining_particles.append(p)
                size = max(1, int(p[4] / 8))
                pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), size)
        self.explosion_particles = remaining_particles
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid cells
        for r in range(self.GRID_SIZE[1]):
            for c in range(self.GRID_SIZE[0]):
                rect = pygame.Rect(
                    self.grid_offset_x + c * self.cell_size,
                    self.grid_offset_y + r * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                if self.revealed_grid[r, c]:
                    if self.mine_grid[r, c]:
                        pygame.draw.rect(self.screen, self.COLOR_REVEALED_MINE, rect)
                        # Draw mine symbol
                        center = rect.center
                        pygame.draw.circle(self.screen, (20, 20, 20), center, self.cell_size // 3)
                    else:
                        pygame.draw.rect(self.screen, self.COLOR_REVEALED_SAFE, rect)
                        num = self.number_grid[r, c]
                        if num > 0:
                            num_text = self.font_main.render(str(num), True, self.COLOR_NUMBERS[num])
                            text_rect = num_text.get_rect(center=rect.center)
                            self.screen.blit(num_text, text_rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)
                    if self.flagged_grid[r, c]:
                        # Draw flag
                        flag_center_x = rect.centerx
                        flag_top_y = rect.centery - 8
                        pygame.draw.line(self.screen, self.COLOR_FLAG, (flag_center_x, rect.top + 5), (flag_center_x, rect.bottom - 5), 2)
                        pygame.draw.polygon(self.screen, self.COLOR_FLAG, [(flag_center_x, flag_top_y), (flag_center_x + 10, flag_top_y + 5), (flag_center_x, flag_top_y + 10)])

                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.cell_size,
            self.grid_offset_y + self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        # Draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Mines remaining
        num_flags = np.sum(self.flagged_grid)
        mines_left = self.NUM_MINES - num_flags
        mines_text = self.font_ui.render(f"Mines: {int(mines_left)}", True, self.COLOR_TEXT)
        self.screen.blit(mines_text, (15, 15))

        # Score
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 15))
        self.screen.blit(score_text, score_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.win:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = self.COLOR_REVEALED_MINE
                
            text = self.font_large.render(msg, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "mines_left": self.NUM_MINES - np.sum(self.flagged_grid),
            "game_over": self.game_over,
            "win": self.win
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
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Minesweeper Gym Environment")
    
    terminated = False
    running = True
    clock = pygame.time.Clock()

    # Action state
    movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
    space_held = 0
    shift_held = 0

    print(GameEnv.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        # Continuous key state checking for smooth controls
        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.1f}, Win: {info['win']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(1000)

        # Since auto_advance is False, we need to control the step rate manually
        # This gives the player time to react between moves
        clock.tick(10) # Process player input 10 times per second

    env.close()