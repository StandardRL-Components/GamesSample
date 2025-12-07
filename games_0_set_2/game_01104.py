
# Generated: 2025-08-27T16:02:50.750939
# Source Brief: brief_01104.md
# Brief Index: 1104

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor, Space to reveal tile, Shift to flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic puzzle game. Reveal all safe tiles on the grid while avoiding hidden mines. Use flags to mark suspected mines."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_ROWS, self.GRID_COLS = 5, 5
        self.NUM_MINES = 10
        self.MAX_STEPS = 1000
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.CELL_SIZE = 60
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID_LINES = (50, 60, 70)
        self.COLOR_TILE_HIDDEN = (70, 80, 95)
        self.COLOR_TILE_REVEALED = (110, 120, 135)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_FLAG = (255, 180, 0)
        self.COLOR_MINE = (255, 80, 80)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TEXT_GAMEOVER = (255, 50, 50)
        self.COLOR_TEXT_WIN = (50, 255, 50)
        self.NUMBER_COLORS = [
            self.COLOR_TILE_REVEALED, # 0
            (80, 150, 255),  # 1
            (80, 200, 80),   # 2
            (255, 80, 80),   # 3
            (80, 80, 200),   # 4
            (150, 80, 80),   # 5
            (80, 200, 200),  # 6
            (40, 40, 40),    # 7
            (120, 120, 120)  # 8
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_tile = pygame.font.Font(None, 40)
        self.font_gameover = pygame.font.Font(None, 80)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.solution_grid = None
        self.visible_grid = None
        self.cursor_pos = None
        self.safe_tiles_to_reveal = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []

        # Initialize state
        self.reset()
        
        # Run validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.safe_tiles_to_reveal = (self.GRID_ROWS * self.GRID_COLS) - self.NUM_MINES
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []

        self._initialize_grids()

        return self._get_observation(), self._get_info()

    def _initialize_grids(self):
        # 0: hidden, 1: revealed, 2: flagged
        self.visible_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.int8)
        # -1: mine, 0-8: number of adjacent mines
        self.solution_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.int8)

        # Place mines
        mine_indices = self.np_random.choice(
            self.GRID_ROWS * self.GRID_COLS, self.NUM_MINES, replace=False
        )
        mine_coords = [(i // self.GRID_COLS, i % self.GRID_COLS) for i in mine_indices]
        for r, c in mine_coords:
            self.solution_grid[r, c] = -1

        # Calculate numbers
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.solution_grid[r, c] == -1:
                    continue
                mine_count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS
                            and self.solution_grid[nr, nc] == -1):
                            mine_count += 1
                self.solution_grid[r, c] = mine_count

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        movement = action[0]
        space_pressed = action[1] == 1 and not self.last_space_held
        shift_pressed = action[2] == 1 and not self.last_shift_held
        
        self.last_space_held = action[1] == 1
        self.last_shift_held = action[2] == 1

        # 1. Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_ROWS) % self.GRID_ROWS
        elif movement == 2:  # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_ROWS
        elif movement == 3:  # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_COLS) % self.GRID_COLS
        elif movement == 4:  # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_COLS

        # 2. Handle tile reveal (space)
        if space_pressed:
            r, c = self.cursor_pos
            if self.visible_grid[r, c] == 0: # Can only reveal hidden tiles
                reward += self._reveal_tile(r, c)

        # 3. Handle flagging (shift)
        if shift_pressed:
            r, c = self.cursor_pos
            if self.visible_grid[r, c] == 0:
                self.visible_grid[r, c] = 2 # Flag
            elif self.visible_grid[r, c] == 2:
                self.visible_grid[r, c] = 0 # Unflag

        # Update particles for explosion
        self._update_particles()
        
        self.score += reward
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reveal_tile(self, r, c):
        if self.solution_grid[r, c] == -1:
            # Game Over: Hit a mine
            self.game_over = True
            self.visible_grid[r, c] = 1
            self._create_explosion(r, c)
            # SFX: explosion
            return -100
        
        revealed_count = self._flood_fill(r, c)
        
        if revealed_count > 0:
            reward = 0
            # Base reward for revealing safe tiles
            reward += revealed_count
            # Penalty for revealing a numbered tile (risk)
            if self.solution_grid[r, c] > 0:
                reward -= 0.2
            
            self.safe_tiles_to_reveal -= revealed_count
            
            if self.safe_tiles_to_reveal <= 0:
                self.game_over = True
                self.win = True
                # SFX: win_jingle
                return reward + 100 # Win bonus
            return reward
        
        return 0 # Revealed an already revealed tile (shouldn't happen with current logic)

    def _flood_fill(self, r, c):
        if not (0 <= r < self.GRID_ROWS and 0 <= c < self.GRID_COLS):
            return 0
        if self.visible_grid[r, c] != 0: # Not hidden
            return 0

        q = deque([(r, c)])
        self.visible_grid[r, c] = 1
        count = 1
        
        while q:
            curr_r, curr_c = q.popleft()
            
            if self.solution_grid[curr_r, curr_c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = curr_r + dr, curr_c + dc
                        
                        if (0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and self.visible_grid[nr, nc] == 0):
                            self.visible_grid[nr, nc] = 1
                            count += 1
                            q.append((nr, nc))
        return count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": list(self.cursor_pos),
            "win": self.win
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y), 2)
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT), 2)

        # Draw tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = pygame.Rect(self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                
                tile_state = self.visible_grid[r,c]
                solution_val = self.solution_grid[r,c]

                # If game over, reveal all mines
                if self.game_over and not self.win and solution_val == -1:
                    tile_state = 1
                
                if tile_state == 0: # Hidden
                    pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, rect.inflate(-2, -2))
                elif tile_state == 1: # Revealed
                    pygame.draw.rect(self.screen, self.COLOR_TILE_REVEALED, rect.inflate(-2, -2))
                    if solution_val > 0:
                        num_text = self.font_tile.render(str(solution_val), True, self.NUMBER_COLORS[solution_val])
                        text_rect = num_text.get_rect(center=rect.center)
                        self.screen.blit(num_text, text_rect)
                    elif solution_val == -1:
                        # Draw Mine
                        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, self.CELL_SIZE // 4, self.COLOR_MINE)
                        pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, self.CELL_SIZE // 4, self.COLOR_MINE)
                elif tile_state == 2: # Flagged
                    pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, rect.inflate(-2, -2))
                    # Draw Flag
                    flag_points = [
                        (rect.centerx - 10, rect.centery - 15),
                        (rect.centerx + 10, rect.centery - 10),
                        (rect.centerx - 10, rect.centery - 5)
                    ]
                    pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)
                    pygame.draw.line(self.screen, self.COLOR_FLAG, (rect.centerx - 10, rect.centery - 15), (rect.centerx - 10, rect.centery + 15), 3)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_X + cursor_c * self.CELL_SIZE, self.GRID_Y + cursor_r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['radius']), p['color'])

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_main.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        steps_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(steps_text, steps_rect)

        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_TEXT_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_TEXT_GAMEOVER
            
            end_text = self.font_gameover.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            # Draw a semi-transparent background for the text
            bg_surf = pygame.Surface(end_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))
            self.screen.blit(bg_surf, end_rect.topleft)
            self.screen.blit(end_text, end_rect)

    def _create_explosion(self, r, c):
        center_x = self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': center_x,
                'y': center_y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'radius': random.uniform(3, 8),
                'life': random.randint(20, 40),
                'color': random.choice([self.COLOR_MINE, self.COLOR_FLAG, (255, 150, 0)])
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Minesweeper Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.1f}, Win: {info['win']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Since auto_advance is False, we control the "frame rate" here
        # A lower rate makes it easier for humans to play
        clock.tick(15) 

    env.close()