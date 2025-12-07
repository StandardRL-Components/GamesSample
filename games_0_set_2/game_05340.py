import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the selector. Space to shift the selected row right. Shift to shift the selected column down."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target pattern by shifting rows and columns of pixels. You have a limited number of moves for each stage."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # --- Visuals ---
        self.FONT_UI = pygame.font.SysFont("Consolas", 20)
        self.FONT_TITLE = pygame.font.SysFont("Consolas", 24, bold=True)
        self.FONT_MSG = pygame.font.SysFont("Consolas", 32, bold=True)

        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID_BG = (35, 38, 46)
        self.COLOR_GRID_LINES = (55, 58, 66)
        self.COLOR_UI_TEXT = (220, 225, 235)
        self.COLOR_UI_VALUE = (137, 221, 255)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_FLASH = (255, 255, 255, 100)
        self.PIXEL_COLORS = [
            self.COLOR_GRID_BG,       # 0: Empty
            (255, 80, 80),          # 1: Red
            (80, 255, 80),          # 2: Green
            (80, 120, 255),         # 3: Blue
            (255, 255, 80),         # 4: Yellow
            (255, 80, 255),         # 5: Magenta
        ]

        # --- Game Layout ---
        self.GRID_SIZE = 10
        self.CELL_SIZE = 36
        self.GRID_MARGIN = 20
        self.GRID_RECT = pygame.Rect(
            self.GRID_MARGIN, 
            self.GRID_MARGIN, 
            self.GRID_SIZE * self.CELL_SIZE, 
            self.GRID_SIZE * self.CELL_SIZE
        )
        
        self.TARGET_CELL_SIZE = 18
        self.TARGET_RECT = pygame.Rect(
            self.GRID_RECT.right + 40,
            self.GRID_MARGIN + 30,
            self.GRID_SIZE * self.TARGET_CELL_SIZE,
            self.GRID_SIZE * self.TARGET_CELL_SIZE
        )
        
        # --- Game State ---
        self.grid = None
        self.target_grid = None
        self.cursor_pos = None
        self.moves_left = None
        self.score = None
        self.stage = None
        self.game_over = None
        self.win = None
        self.steps = None
        self.max_steps = 600
        self.flash_effect = None # {'type': 'row'/'col', 'index': N}
        
        self.patterns = self._define_patterns()

        # Initialize state variables - seed is set here for the first time
        # self.reset() will be called by the parent environment later, so we just init
        super().reset(seed=0)
        self.stage = 1
        self._start_stage()
        
    def _define_patterns(self):
        patterns = {}
        # Stage 1: Simple cross
        p1 = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        p1[4, :] = 1
        p1[:, 4] = 1
        patterns[1] = p1

        # Stage 2: Arrow
        p2 = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        p2[4, 2:8] = 2
        p2[3, 3:7] = 2
        p2[2, 4:6] = 2
        p2[5:8, 4] = 3
        p2[5:8, 5] = 3
        patterns[2] = p2

        # Stage 3: Checkerboard with a twist
        p3 = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (r + c) % 2 == 0:
                    p3[r, c] = 4
        p3[2:8, 2:8] = 5
        p3[4:6, 4:6] = 0
        patterns[3] = p3
        
        return patterns

    def _start_stage(self):
        self.target_grid = self.patterns[self.stage]
        self.moves_left = 20
        self.game_over = False
        self.win = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        # Scramble the target grid to create the starting grid
        self.grid = self.target_grid.copy()
        num_scrambles = self.stage * 3 + 2 # More scrambles for harder stages
        for _ in range(num_scrambles):
            if self.np_random.random() > 0.5: # Shift row
                row = self.np_random.integers(0, self.GRID_SIZE)
                amount = self.np_random.integers(1, self.GRID_SIZE)
                self.grid[row, :] = np.roll(self.grid[row, :], amount)
            else: # Shift column
                col = self.np_random.integers(0, self.GRID_SIZE)
                amount = self.np_random.integers(1, self.GRID_SIZE)
                self.grid[:, col] = np.roll(self.grid[:, col], amount)
        
        # Ensure it's not solved by a lucky scramble
        if np.array_equal(self.grid, self.target_grid):
            self.grid[0, :] = np.roll(self.grid[0, :], 1)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.stage = 1
        self._start_stage()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        self.flash_effect = None
        reward = 0.0 # Use float for rewards
        
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        action_taken = False
        
        # --- Handle Grid Shifts (costs a move) ---
        if self.moves_left > 0:
            if space_held:
                # Shift row right
                row_idx = self.cursor_pos[1]
                self.grid[row_idx, :] = np.roll(self.grid[row_idx, :], 1)
                self.flash_effect = {'type': 'row', 'index': row_idx}
                action_taken = True
                # sfx: row_shift.wav
            
            if shift_held:
                # Shift column down
                col_idx = self.cursor_pos[0]
                self.grid[:, col_idx] = np.roll(self.grid[:, col_idx], 1)
                self.flash_effect = {'type': 'col', 'index': col_idx}
                action_taken = True
                # sfx: col_shift.wav

        if action_taken:
            self.moves_left -= 1
        
        # --- Handle Cursor Movement (no move cost) ---
        if movement != 0:
            if movement == 1: # Up
                self.cursor_pos[1] -= 1
            elif movement == 2: # Down
                self.cursor_pos[1] += 1
            elif movement == 3: # Left
                self.cursor_pos[0] -= 1
            elif movement == 4: # Right
                self.cursor_pos[0] += 1
            
            self.cursor_pos[0] %= self.GRID_SIZE
            self.cursor_pos[1] %= self.GRID_SIZE
            # sfx: cursor_move.wav

        self.steps += 1
        
        # --- Calculate Reward & Check Termination ---
        correct_pixels = np.sum(self.grid == self.target_grid)
        reward = float(correct_pixels) # Dense reward
        
        is_match = (correct_pixels == self.GRID_SIZE * self.GRID_SIZE)
        
        if is_match:
            # sfx: stage_clear.wav
            self.score += int(correct_pixels) # Add final pixel score
            reward += 5.0 # Stage complete bonus
            self.stage += 1
            
            if self.stage > len(self.patterns):
                # Game Won
                self.win = True
                self.game_over = True
                reward += 50.0 # Final win bonus
                # sfx: game_win.wav
            else:
                # Go to next stage
                self._start_stage()
        
        elif self.moves_left <= 0:
            self.game_over = True
            # sfx: game_over.wav
            
        if self.steps >= self.max_steps:
            self.game_over = True

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_grid(self, surface, rect, cell_size, grid_data, draw_grid_lines=True):
        # Draw grid background
        pygame.draw.rect(surface, self.COLOR_GRID_BG, rect)
        
        # Draw pixels
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = grid_data[r, c]
                pixel_color = self.PIXEL_COLORS[color_idx]
                pixel_rect = pygame.Rect(
                    rect.left + c * cell_size,
                    rect.top + r * cell_size,
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(surface, pixel_color, pixel_rect)
        
        # Draw grid lines
        if draw_grid_lines:
            for i in range(self.GRID_SIZE + 1):
                # Vertical
                pygame.draw.line(surface, self.COLOR_GRID_LINES, 
                                 (rect.left + i * cell_size, rect.top), 
                                 (rect.left + i * cell_size, rect.bottom))
                # Horizontal
                pygame.draw.line(surface, self.COLOR_GRID_LINES, 
                                 (rect.left, rect.top + i * cell_size), 
                                 (rect.right, rect.top + i * cell_size))

    def _render_game(self):
        # Draw main grid
        self._render_grid(self.screen, self.GRID_RECT, self.CELL_SIZE, self.grid)
        
        # Draw target pattern
        self._render_grid(self.screen, self.TARGET_RECT, self.TARGET_CELL_SIZE, self.target_grid, draw_grid_lines=False)
        
        # Draw flash effect
        if self.flash_effect:
            flash_rect = None
            if self.flash_effect['type'] == 'row':
                idx = self.flash_effect['index']
                flash_rect = pygame.Rect(
                    self.GRID_RECT.left,
                    self.GRID_RECT.top + idx * self.CELL_SIZE,
                    self.GRID_RECT.width,
                    self.CELL_SIZE
                )
            elif self.flash_effect['type'] == 'col':
                idx = self.flash_effect['index']
                flash_rect = pygame.Rect(
                    self.GRID_RECT.left + idx * self.CELL_SIZE,
                    self.GRID_RECT.top,
                    self.CELL_SIZE,
                    self.GRID_RECT.height
                )
            if flash_rect:
                s = pygame.Surface(flash_rect.size, pygame.SRCALPHA)
                s.fill(self.COLOR_FLASH)
                self.screen.blit(s, flash_rect.topleft)

        # Draw cursor
        if not self.game_over:
            cx, cy = self.cursor_pos
            cursor_rect = pygame.Rect(
                self.GRID_RECT.left + cx * self.CELL_SIZE,
                self.GRID_RECT.top + cy * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            # Glow effect
            glow_rect = cursor_rect.inflate(8, 8)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_CURSOR, 50), s.get_rect(), border_radius=5)
            self.screen.blit(s, glow_rect.topleft)
            # Outline
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=3)

    def _render_ui(self):
        ui_x = self.TARGET_RECT.left
        
        # Target Title
        title_surf = self.FONT_TITLE.render("TARGET", True, self.COLOR_UI_TEXT)
        self.screen.blit(title_surf, (ui_x, self.GRID_MARGIN))
        
        # Stats
        stats_y = self.TARGET_RECT.bottom + 30
        
        stage_surf = self.FONT_UI.render("STAGE", True, self.COLOR_UI_TEXT)
        stage_val_surf = self.FONT_TITLE.render(f"{self.stage} / {len(self.patterns)}", True, self.COLOR_UI_VALUE)
        self.screen.blit(stage_surf, (ui_x, stats_y))
        self.screen.blit(stage_val_surf, (ui_x, stats_y + 20))
        
        moves_surf = self.FONT_UI.render("MOVES", True, self.COLOR_UI_TEXT)
        moves_val_surf = self.FONT_TITLE.render(str(self.moves_left), True, self.COLOR_UI_VALUE)
        self.screen.blit(moves_surf, (ui_x, stats_y + 70))
        self.screen.blit(moves_val_surf, (ui_x, stats_y + 90))
        
        score_surf = self.FONT_UI.render("SCORE", True, self.COLOR_UI_TEXT)
        score_val_surf = self.FONT_TITLE.render(str(self.score), True, self.COLOR_UI_VALUE)
        self.screen.blit(score_surf, (ui_x, stats_y + 140))
        self.screen.blit(score_val_surf, (ui_x, stats_y + 160))
        
        # Game Over / Win Message
        if self.game_over:
            msg = "GAME OVER"
            color = self.PIXEL_COLORS[1] # Red
            if self.win:
                msg = "YOU WIN!"
                color = self.PIXEL_COLORS[2] # Green
            
            msg_surf = self.FONT_MSG.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=self.screen.get_rect().center)
            
            # Draw a semi-transparent background for the message
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(msg_surf, msg_rect)


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        correct_pixels = np.sum(self.grid == self.target_grid) if self.grid is not None else 0
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "moves_left": self.moves_left,
            "correct_pixels": int(correct_pixels),
        }

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # To run headlessly, SDL_VIDEODRIVER is set to "dummy" in __init__
    # To run with a window, comment out the line in __init__
    # import os
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame window for human play ---
    pygame.display.set_caption("Pixel Shift")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)
    print("Press 'R' to reset the game.")

    while running:
        # --- Human Controls ---
        movement = 0 # none
        space = 0
        shift = 0
        
        keys = pygame.key.get_pressed()
        
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # We only care about keydown events for discrete actions
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("\n--- Game Reset ---")
        
        # These actions are continuous (held down)
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        # Step the environment only if an action is taken
        # Since auto_advance is False, the state only changes on a step
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"Step: {info['steps']}, Action: {action}, Reward: {int(reward)}, Total Reward: {int(total_reward)}, Terminated: {terminated}", end='\r')

            if terminated:
                print(f"\n--- Episode Finished ---")
                print(f"Final Score: {info['score']}")
                # Wait a bit before allowing a reset
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0
                print("\n--- New Game Started ---")

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = env._get_observation()
        # Pygame uses (width, height), numpy uses (height, width)
        # We need to transpose back for display
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate for human play

    env.close()