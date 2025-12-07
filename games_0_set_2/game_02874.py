
# Generated: 2025-08-27T21:41:23.889371
# Source Brief: brief_02874.md
# Brief Index: 2874

        
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
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to reveal a square. Shift to flag/unflag a square."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic Minesweeper puzzle. Reveal all safe squares without hitting a mine. Numbers indicate adjacent mines."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    GRID_SIZE = (20, 12)  # Width, Height
    NUM_MINES = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (40, 42, 54)
    COLOR_GRID = (70, 72, 84)
    COLOR_UNREVEALED = (98, 114, 164)
    COLOR_REVEALED = (68, 71, 90)
    COLOR_FLAG = (241, 250, 140)
    COLOR_MINE = (255, 85, 85)
    COLOR_CURSOR = (80, 250, 123, 150) # RGBA
    COLOR_TEXT_UI = (248, 248, 242)
    COLOR_NUMBERS = [
        (0, 0, 0), # 0 is not drawn
        (98, 160, 255),  # 1
        (80, 250, 123),  # 2
        (255, 121, 198), # 3
        (189, 147, 249), # 4
        (255, 184, 108), # 5
        (139, 233, 253), # 6
        (255, 85, 85),   # 7
        (200, 200, 200)  # 8
    ]
    COLOR_WIN = (80, 250, 123)
    COLOR_LOSE = (255, 85, 85)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.width, self.height = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_cell = pygame.font.SysFont('Consolas', 18, bold=True)
        self.font_ui = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_game_over = pygame.font.SysFont('Consolas', 50, bold=True)
        
        # Calculate cell dimensions
        self.cell_w = self.width // self.GRID_SIZE[0]
        self.cell_h = (self.height - 40) // self.GRID_SIZE[1] # Reserve 40px for UI
        self.grid_offset_x = (self.width - self.GRID_SIZE[0] * self.cell_w) // 2
        self.grid_offset_y = 40 + ((self.height - 40) - self.GRID_SIZE[1] * self.cell_h) // 2
        
        # State variables are initialized in reset()
        self.grid = None
        self.revealed = None
        self.flagged = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.np_random = None
        
        if self.render_mode == "human":
            self.human_screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Minesweeper Gym Environment")
            
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.cursor_pos = np.array([self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2])
        
        self.grid = np.zeros(self.GRID_SIZE, dtype=int)
        self.revealed = np.zeros(self.GRID_SIZE, dtype=bool)
        self.flagged = np.zeros(self.GRID_SIZE, dtype=bool)
        
        self._place_mines()
        self._calculate_numbers()

        return self._get_observation(), self._get_info()

    def _place_mines(self):
        mine_indices = self.np_random.choice(
            self.GRID_SIZE[0] * self.GRID_SIZE[1], self.NUM_MINES, replace=False
        )
        mine_coords = np.unravel_index(mine_indices, self.GRID_SIZE)
        self.grid[mine_coords] = -1 # -1 represents a mine

    def _calculate_numbers(self):
        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1]):
                if self.grid[x, y] == -1:
                    continue
                count = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_SIZE[0] and 0 <= ny < self.GRID_SIZE[1]:
                            if self.grid[nx, ny] == -1:
                                count += 1
                self.grid[x, y] = count

    def step(self, action):
        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, reveal_action, flag_action = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # 1. Handle Movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE[1] - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE[0] - 1, self.cursor_pos[0] + 1)

        cx, cy = self.cursor_pos[0], self.cursor_pos[1]

        # 2. Handle Actions (Reveal > Flag)
        if reveal_action and not self.flagged[cx, cy] and not self.revealed[cx, cy]:
            # sound: reveal_sfx()
            self.revealed[cx, cy] = True
            if self.grid[cx, cy] == -1:
                # sound: explosion_sfx()
                self.game_over = True
                reward = -100.0
            else:
                if self.grid[cx, cy] == 0:
                    reward -= 0.2 # Penalty for revealing a non-informative '0'
                    reward += self._flood_fill(cx, cy)
                else:
                    reward = 1.0

        elif flag_action and not self.revealed[cx, cy]:
            # sound: flag_sfx()
            is_mine = self.grid[cx, cy] == -1
            if not self.flagged[cx, cy]: # Placing a flag
                self.flagged[cx, cy] = True
                reward = 5.0 if is_mine else -1.0
            else: # Removing a flag
                self.flagged[cx, cy] = False
                reward = -5.0 if is_mine else 1.0
        
        self.score += reward
        self.steps += 1
        
        # 3. Check for win/loss/termination
        if not self.game_over and self._check_win_condition():
            # sound: win_sfx()
            self.game_won = True
            self.score += 100.0 # Win bonus
        
        terminated = self.game_over or self.game_won or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _flood_fill(self, x, y):
        revealed_count = 1
        q = deque([(x, y)])
        visited = set([(x, y)])
        
        while q:
            cx, cy = q.popleft()
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    
                    if 0 <= nx < self.GRID_SIZE[0] and 0 <= ny < self.GRID_SIZE[1] and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        if not self.revealed[nx, ny] and not self.flagged[nx, ny]:
                            self.revealed[nx, ny] = True
                            revealed_count += 1
                            if self.grid[nx, ny] == 0:
                                q.append((nx, ny))
        return revealed_count

    def _check_win_condition(self):
        non_mine_cells = self.GRID_SIZE[0] * self.GRID_SIZE[1] - self.NUM_MINES
        return np.sum(self.revealed) == non_mine_cells

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.render_mode == "human":
            self.human_screen.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1]):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_w,
                    self.grid_offset_y + y * self.cell_h,
                    self.cell_w,
                    self.cell_h
                )
                
                # Draw cell background
                if self.revealed[x, y]:
                    pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                    if self.grid[x, y] == -1: # Exploded mine
                        pygame.draw.rect(self.screen, self.COLOR_MINE, rect)
                        self._draw_mine(rect.center, self.cell_w * 0.6)
                    elif self.grid[x, y] > 0:
                        self._draw_text(
                            str(self.grid[x, y]),
                            self.font_cell,
                            self.COLOR_NUMBERS[self.grid[x,y]],
                            rect.center
                        )
                elif self.flagged[x, y]:
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)
                    self._draw_flag(rect.center, self.cell_h * 0.6)
                else: # Unrevealed
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)

                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.cell_w,
            self.grid_offset_y + self.cursor_pos[1] * self.cell_h,
            self.cell_w,
            self.cell_h
        )
        cursor_surface = pygame.Surface((self.cell_w, self.cell_h), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, self.COLOR_CURSOR, cursor_surface.get_rect(), border_radius=3)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_ui(self):
        # UI Background
        ui_rect = pygame.Rect(0, 0, self.width, 40)
        pygame.draw.rect(self.screen, self.COLOR_GRID, ui_rect)
        
        # Score
        score_text = f"Score: {int(self.score)}"
        self._draw_text(score_text, self.font_ui, self.COLOR_TEXT_UI, (80, 20))

        # Steps
        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        self._draw_text(steps_text, self.font_ui, self.COLOR_TEXT_UI, (self.width / 2, 20))

        # Mines remaining
        flags_placed = np.sum(self.flagged)
        mines_text = f"Mines: {self.NUM_MINES - flags_placed}"
        self._draw_text(mines_text, self.font_ui, self.COLOR_TEXT_UI, (self.width - 80, 20))
        
        # Game Over / Win message
        if self.game_over or self.game_won:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_WIN if self.game_won else self.COLOR_LOSE
            self._draw_text(msg, self.font_game_over, color, (self.width/2, self.height/2))

    def _draw_text(self, text, font, color, center_pos):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=center_pos)
        self.screen.blit(text_surf, text_rect)

    def _draw_flag(self, center, size):
        pole_x = center[0]
        pole_top = center[1] - size / 2
        pole_bottom = center[1] + size / 2
        pygame.draw.line(self.screen, (0,0,0), (pole_x, pole_top), (pole_x, pole_bottom), 3)
        
        flag_points = [
            (pole_x, pole_top),
            (pole_x, pole_top + size * 0.4),
            (pole_x - size * 0.6, pole_top + size * 0.2)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_MINE, flag_points)
        
    def _draw_mine(self, center, size):
        pygame.draw.circle(self.screen, (20,20,20), center, size/2)
        for i in range(8):
            angle = i * math.pi / 4
            start_pos = (
                center[0] + (size/2) * math.cos(angle),
                center[1] + (size/2) * math.sin(angle)
            )
            end_pos = (
                center[0] + (size*0.8) * math.cos(angle),
                center[1] + (size*0.8) * math.sin(angle)
            )
            pygame.draw.line(self.screen, (20,20,20), start_pos, end_pos, 3)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos.tolist(),
            "game_won": self.game_won,
            "game_over": self.game_over
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game with a keyboard
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print("Minesweeper Gym Environment")
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")
    
    while not done:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        # Poll for events
        event_processed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                event_processed = True
            if event.type == pygame.KEYDOWN and not event_processed:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                
                # Only step if an action was taken
                if sum(action) > 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated
                    print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")
                
                event_processed = True

        # Since auto_advance is False, we only need to render when an action is taken.
        # However, to keep the window responsive, we render continuously.
        env._get_observation()
        
    print("--- Final State ---")
    print(f"Score: {info['score']:.2f}, Steps: {info['steps']}, Win: {info['game_won']}")
    
    # Wait a bit before closing
    pygame.time.wait(3000)
    env.close()