
# Generated: 2025-08-28T02:53:48.998982
# Source Brief: brief_04606.md
# Brief Index: 4606

        
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
        "Controls: Arrow keys to move cursor. Space to reveal a square. Shift to place/remove a flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic puzzle game. Navigate a grid, revealing safe squares while avoiding hidden mines. "
        "Numbers indicate adjacent mines. Clear all safe squares to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 20, 12
        self.CELL_SIZE = 30
        self.NUM_MINES = 30
        self.MAX_STEPS = 1000

        self.GRID_PX_W = self.GRID_W * self.CELL_SIZE
        self.GRID_PX_H = self.GRID_H * self.CELL_SIZE
        self.OFFSET_X = (self.WIDTH - self.GRID_PX_W) // 2
        self.OFFSET_Y = (self.HEIGHT - self.GRID_PX_H) // 2

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID_LINE = (40, 44, 52)
        self.COLOR_CELL_HIDDEN = (60, 66, 82)
        self.COLOR_CELL_HIDDEN_LITE = (75, 83, 102)
        self.COLOR_CELL_HIDDEN_DARK = (45, 50, 62)
        self.COLOR_CELL_REVEALED = (40, 44, 52)
        self.COLOR_CURSOR = (76, 175, 80, 150)
        self.COLOR_FLAG = (255, 193, 7)
        self.COLOR_MINE = (239, 83, 80)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_WIN = (76, 175, 80)
        self.COLOR_LOSE = (239, 83, 80)
        
        self.number_colors = {
            1: (66, 165, 245), 2: (102, 187, 106), 3: (239, 83, 80),
            4: (126, 87, 194), 5: (255, 167, 38), 6: (0, 172, 193),
            7: (212, 212, 212), 8: (170, 170, 170)
        }

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_cell = pygame.font.Font(None, 28)
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 60)
        
        # Initialize state variables
        self.mine_grid = None
        self.revealed_grid = None
        self.adjacency_grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.total_safe_squares = 0
        self.revealed_safe_squares = 0
        self.flags_placed = 0
        self.last_action_was_reveal = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.revealed_safe_squares = 0
        self.flags_placed = 0
        self.last_action_was_reveal = False
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        
        self.mine_grid = np.zeros((self.GRID_W, self.GRID_H), dtype=np.int8)
        self.revealed_grid = np.zeros((self.GRID_W, self.GRID_H), dtype=np.int8) # 0: hidden, 1: revealed, 2: flagged
        
        # Place mines
        flat_indices = self.np_random.choice(self.GRID_W * self.GRID_H, self.NUM_MINES, replace=False)
        mine_coords = np.unravel_index(flat_indices, (self.GRID_W, self.GRID_H))
        self.mine_grid[mine_coords] = 1
        
        self.total_safe_squares = (self.GRID_W * self.GRID_H) - self.NUM_MINES
        
        # Calculate adjacency
        self.adjacency_grid = np.zeros((self.GRID_W, self.GRID_H), dtype=np.int8)
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if self.mine_grid[x, y] == 1:
                    continue
                count = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and self.mine_grid[nx, ny] == 1:
                            count += 1
                self.adjacency_grid[x, y] = count

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        self.last_action_was_reveal = False

        if not self.game_over and not self.game_won:
            # 1. Process movement
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            
            # Wrap cursor around edges
            self.cursor_pos[0] %= self.GRID_W
            self.cursor_pos[1] %= self.GRID_H
            
            cx, cy = self.cursor_pos
            
            # 2. Process actions (prioritize reveal over flag)
            if space_held:
                self.last_action_was_reveal = True
                if self.revealed_grid[cx, cy] == 0: # Can only reveal hidden squares
                    # Sound effect placeholder: # sfx_reveal.play()
                    if self.mine_grid[cx, cy] == 1:
                        self.game_over = True
                        reward = -100.0
                        self.revealed_grid[cx, cy] = 1
                        # Sound effect placeholder: # sfx_explosion.play()
                    else:
                        adj_mines = self.adjacency_grid[cx, cy]
                        reward = 1.0
                        if adj_mines == 0:
                            reward -= 0.2
                            self._flood_fill_reveal(cx, cy)
                        else:
                            if self.revealed_grid[cx, cy] == 0:
                                self.revealed_grid[cx, cy] = 1
                                self.revealed_safe_squares += 1
            
            elif shift_held:
                if self.revealed_grid[cx, cy] != 1: # Can't flag revealed squares
                    if self.revealed_grid[cx, cy] == 2:
                        self.revealed_grid[cx, cy] = 0
                        self.flags_placed -= 1
                        # Sound effect placeholder: # sfx_unflag.play()
                    else:
                        self.revealed_grid[cx, cy] = 2
                        self.flags_placed += 1
                        # Sound effect placeholder: # sfx_flag.play()

            # 3. Check for win condition
            if self.revealed_safe_squares == self.total_safe_squares:
                self.game_won = True
                reward += 100.0
                # Sound effect placeholder: # sfx_win.play()

        self.steps += 1
        self.score += reward
        
        terminated = self.game_over or self.game_won or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _flood_fill_reveal(self, x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if not (0 <= cx < self.GRID_W and 0 <= cy < self.GRID_H):
                continue
            if self.revealed_grid[cx, cy] != 0:
                continue

            self.revealed_grid[cx, cy] = 1
            self.revealed_safe_squares += 1

            if self.adjacency_grid[cx, cy] == 0:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        stack.append((cx + dx, cy + dy))
    
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
            "cursor_pos": self.cursor_pos,
            "mines_remaining": self.NUM_MINES - self.flags_placed,
            "game_won": self.game_won
        }

    def _render_game(self):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                rect = pygame.Rect(
                    self.OFFSET_X + x * self.CELL_SIZE,
                    self.OFFSET_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                
                state = self.revealed_grid[x, y]
                
                if state == 1: # Revealed
                    pygame.draw.rect(self.screen, self.COLOR_CELL_REVEALED, rect)
                    if self.mine_grid[x, y] == 1:
                        # Draw mine explosion
                        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, self.CELL_SIZE // 3, self.COLOR_MINE)
                    else:
                        adj_mines = self.adjacency_grid[x, y]
                        if adj_mines > 0:
                            color = self.number_colors.get(adj_mines, self.COLOR_TEXT)
                            text_surf = self.font_cell.render(str(adj_mines), True, color)
                            text_rect = text_surf.get_rect(center=rect.center)
                            self.screen.blit(text_surf, text_rect)
                else: # Hidden or Flagged
                    # 3D button effect
                    pygame.draw.rect(self.screen, self.COLOR_CELL_HIDDEN, rect)
                    pygame.draw.line(self.screen, self.COLOR_CELL_HIDDEN_LITE, rect.topleft, rect.topright, 1)
                    pygame.draw.line(self.screen, self.COLOR_CELL_HIDDEN_LITE, rect.topleft, rect.bottomleft, 1)
                    pygame.draw.line(self.screen, self.COLOR_CELL_HIDDEN_DARK, rect.bottomleft, rect.bottomright, 1)
                    pygame.draw.line(self.screen, self.COLOR_CELL_HIDDEN_DARK, rect.topright, rect.bottomright, 1)

                    if state == 2: # Flagged
                        self._draw_flag(rect)

        # Draw grid lines
        for i in range(self.GRID_W + 1):
            start_pos = (self.OFFSET_X + i * self.CELL_SIZE, self.OFFSET_Y)
            end_pos = (self.OFFSET_X + i * self.CELL_SIZE, self.OFFSET_Y + self.GRID_PX_H)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 1)
        for i in range(self.GRID_H + 1):
            start_pos = (self.OFFSET_X, self.OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.OFFSET_X + self.GRID_PX_W, self.OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 1)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.OFFSET_X + cx * self.CELL_SIZE,
            self.OFFSET_Y + cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        
        # Animate cursor on reveal
        if self.last_action_was_reveal:
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((255,255,255, 100))
            self.screen.blit(s, cursor_rect.topleft)
        else:
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        # Reveal all mines if game over
        if self.game_over:
            for y in range(self.GRID_H):
                for x in range(self.GRID_W):
                    if self.mine_grid[x, y] == 1 and self.revealed_grid[x, y] != 1:
                        rect = pygame.Rect(
                            self.OFFSET_X + x * self.CELL_SIZE,
                            self.OFFSET_Y + y * self.CELL_SIZE,
                            self.CELL_SIZE, self.CELL_SIZE
                        )
                        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, self.CELL_SIZE // 4, self.COLOR_TEXT)

    def _draw_flag(self, rect):
        pole_x = rect.centerx - self.CELL_SIZE // 6
        pole_start = (pole_x, rect.top + self.CELL_SIZE // 4)
        pole_end = (pole_x, rect.bottom - self.CELL_SIZE // 4)
        pygame.draw.line(self.screen, self.COLOR_FLAG, pole_start, pole_end, 2)
        
        flag_points = [
            pole_start,
            (rect.right - self.CELL_SIZE // 4, rect.centery - self.CELL_SIZE // 8),
            (pole_x, rect.centery)
        ]
        pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_FLAG)
        pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FLAG)
        
    def _render_ui(self):
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 15))
        
        # Mines Remaining
        mines_text = f"MINES: {max(0, self.NUM_MINES - self.flags_placed)}"
        mines_surf = self.font_ui.render(mines_text, True, self.COLOR_TEXT)
        mines_rect = mines_surf.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(mines_surf, mines_rect)

        # Game Over / Win message
        if self.game_over or self.game_won:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_WIN if self.game_won else self.COLOR_LOSE
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            msg_surf = self.font_game_over.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

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

        # Test specific game logic
        assert self.total_safe_squares == (self.GRID_W * self.GRID_H) - self.NUM_MINES
        assert np.sum(self.mine_grid) == self.NUM_MINES
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Create a display window
    pygame.display.set_caption("Minesweeper Gym Env")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = [0, 0, 0] # no-op, no-space, no-shift
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: action[2] = 1
                
                # Since auto_advance is False, we step on any key press
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Reset action after processing
                action = [0, 0, 0]

                print(f"Step: {info['steps']}, Score: {info['score']:.1f}, Reward: {reward:.1f}, Terminated: {terminated}")
        
        # Update the display
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    print("\nGame session finished.")
    env.close()