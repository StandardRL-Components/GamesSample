
# Generated: 2025-08-27T13:55:07.272655
# Source Brief: brief_00526.md
# Brief Index: 526

        
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
        "Controls: Use arrow keys (↑↓←→) to move your selector. Match 3 or more "
        "blocks of the same color horizontally or vertically to clear them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Clear lines to score points and advance through "
        "stages. The grid fills up over time, so think fast! Clear multiple lines "
        "at once for a big bonus."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 10
    CELL_SIZE = 36
    GRID_PIXEL_WIDTH = GRID_WIDTH * CELL_SIZE
    GRID_PIXEL_HEIGHT = GRID_HEIGHT * CELL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_PIXEL_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_PIXEL_HEIGHT) // 2 + 20
    NUM_COLORS = 8

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID_LINES = (45, 50, 62)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    COLOR_UI_BG = (35, 40, 50, 180)
    COLOR_PLAYER_HIGHLIGHT = (255, 255, 255)
    
    # Palette based on 'tab10'
    COLORS = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
            self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 16)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 28)
            self.font_small = pygame.font.Font(None, 22)

        self.grid = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), -1, dtype=int)
        self.player_pos = [0, 0]
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.stage = 1
        self.lines_cleared_in_stage = 0
        self.stage_steps = 0
        self.max_steps_per_stage = 600 # 60 seconds at 10 steps/sec
        self.row_add_intervals = [10, 8, 6]
        self.last_action_info = ""
        self.np_random = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.stage = 1
        self.lines_cleared_in_stage = 0
        self.stage_steps = 0
        self.particles = []
        self.last_action_info = ""

        # Initialize grid with 20% fill
        self.grid = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), -1, dtype=int)
        num_cells_to_fill = int(self.GRID_WIDTH * self.GRID_HEIGHT * 0.2)
        flat_indices = self.np_random.choice(self.GRID_WIDTH * self.GRID_HEIGHT, num_cells_to_fill, replace=False)
        row_indices, col_indices = np.unravel_index(flat_indices, self.grid.shape)
        colors = self.np_random.integers(0, self.NUM_COLORS, size=num_cells_to_fill)
        self.grid[row_indices, col_indices] = colors

        self.player_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        self.last_action_info = ""
        
        self.steps += 1
        self.stage_steps += 1

        # 1. Handle player movement
        prev_pos = list(self.player_pos)
        moved = self._move_player(movement)
        if moved:
            reward = -0.02 # Penalty for moving without clearing
        
        # 2. Check for line clears
        cleared_cells, num_lines = self._check_and_process_clears()
        
        if cleared_cells:
            # sound: line_clear.wav
            if moved: reward += 0.02 # Nullify move penalty
            
            reward += len(cleared_cells) * 0.1 # Per-cell bonus
            
            if num_lines == 1:
                reward += 1
                self.last_action_info = "Clear! +1"
            elif num_lines == 2:
                reward += 3
                self.last_action_info = "Double! +3"
            elif num_lines >= 3:
                reward += 10
                self.last_action_info = "Multi! +10"
            
            self.lines_cleared_in_stage += num_lines
            self._collapse_grid(cleared_cells)
            self._create_particles(cleared_cells)
        
        # 3. Handle periodic row addition
        self._add_new_row()
        
        # 4. Update stage and check for win/loss
        terminated = self._update_game_state()
        
        # Apply terminal rewards
        if terminated:
            if self.win:
                reward += 100
                self.last_action_info = "YOU WIN!"
            else: # Loss
                reward -= 50
                self.last_action_info = "GAME OVER"

        self.score += reward
        self._update_particles()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _move_player(self, movement):
        y, x = self.player_pos
        if movement == 1 and y > 0: self.player_pos[0] -= 1
        elif movement == 2 and y < self.GRID_HEIGHT - 1: self.player_pos[0] += 1
        elif movement == 3 and x > 0: self.player_pos[1] -= 1
        elif movement == 4 and x < self.GRID_WIDTH - 1: self.player_pos[1] += 1
        else: return False
        return True

    def _check_and_process_clears(self):
        y, x = self.player_pos
        color_idx = self.grid[y, x]
        if color_idx == -1:
            return set(), 0

        cleared_cells = set()
        h_line, v_line = set(), set()
        
        # Horizontal check
        temp_line = set([(y,x)])
        # Left
        for i in range(x - 1, -1, -1):
            if self.grid[y, i] == color_idx: temp_line.add((y, i))
            else: break
        # Right
        for i in range(x + 1, self.GRID_WIDTH):
            if self.grid[y, i] == color_idx: temp_line.add((y, i))
            else: break
        if len(temp_line) >= 3:
            h_line = temp_line

        # Vertical check
        temp_line = set([(y,x)])
        # Up
        for i in range(y - 1, -1, -1):
            if self.grid[i, x] == color_idx: temp_line.add((i, x))
            else: break
        # Down
        for i in range(y + 1, self.GRID_HEIGHT):
            if self.grid[i, x] == color_idx: temp_line.add((i, x))
            else: break
        if len(temp_line) >= 3:
            v_line = temp_line

        cleared_cells = h_line.union(v_line)
        num_lines = (1 if h_line else 0) + (1 if v_line else 0)
        
        return cleared_cells, num_lines

    def _collapse_grid(self, cleared_cells):
        cols_affected = set(x for _, x in cleared_cells)
        for x in cols_affected:
            cleared_in_col = sorted([y for y, cx in cleared_cells if cx == x], reverse=True)
            
            # Create a list of surviving cells in the column
            survivors = [self.grid[y, x] for y in range(self.GRID_HEIGHT) if (y, x) not in cleared_cells]
            
            # Create new cells to fill the gap
            num_new_cells = self.GRID_HEIGHT - len(survivors)
            new_cells = self.np_random.integers(0, self.NUM_COLORS, size=num_new_cells).tolist()
            
            # Combine and update the grid column
            self.grid[:, x] = new_cells + survivors

    def _add_new_row(self):
        interval = self.row_add_intervals[min(self.stage - 1, len(self.row_add_intervals) - 1)]
        if self.stage_steps > 0 and self.stage_steps % interval == 0:
            # Check for game over condition (top row is not empty)
            if np.any(self.grid[0] != -1):
                self.game_over = True
                # sound: game_over_grid_full.wav
                return

            # Shift grid down
            self.grid[1:, :] = self.grid[:-1, :]
            
            # Add new row at the top
            self.grid[0, :] = self.np_random.integers(0, self.NUM_COLORS, size=self.GRID_WIDTH)
            
            # Shift player down
            self.player_pos[0] = min(self.GRID_HEIGHT - 1, self.player_pos[0] + 1)
            # sound: new_row.wav
    
    def _update_game_state(self):
        if self.lines_cleared_in_stage >= 3:
            # sound: stage_clear.wav
            self.stage += 1
            self.lines_cleared_in_stage = 0
            self.stage_steps = 0
            self.score += 50
            self.last_action_info = f"Stage {self.stage-1} Clear! +50"
            if self.stage > 3:
                self.win = True
                self.game_over = True

        if self.stage_steps >= self.max_steps_per_stage and not self.game_over:
            # sound: game_over_timeout.wav
            self.game_over = True
        
        return self.game_over

    def _create_particles(self, cleared_cells):
        for y, x in cleared_cells:
            color_idx = self.grid[y, x]
            if color_idx == -1: continue # Should not happen if logic is correct
            
            px = self.GRID_X + x * self.CELL_SIZE + self.CELL_SIZE / 2
            py = self.GRID_Y + y * self.CELL_SIZE + self.CELL_SIZE / 2
            base_color = self.COLORS[color_idx]
            
            for _ in range(10): # 10 particles per cleared cell
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                radius = self.np_random.uniform(2, 5)
                lifetime = self.np_random.uniform(15, 30)
                self.particles.append([px, py, vx, vy, radius, lifetime, base_color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[5] > 0]
        for p in self.particles:
            p[0] += p[1]  # x += vx
            p[1] += p[2]  # y += vy
            p[5] -= 1     # lifetime -= 1
            p[4] *= 0.95  # radius shrinks

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_PIXEL_HEIGHT))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X, y), (self.GRID_X + self.GRID_PIXEL_WIDTH, y))

        # Draw grid cells
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.grid[y, x]
                if color_idx != -1:
                    rect = pygame.Rect(self.GRID_X + x * self.CELL_SIZE + 2, self.GRID_Y + y * self.CELL_SIZE + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
                    pygame.draw.rect(self.screen, self.COLORS[color_idx], rect, border_radius=4)
        
        # Draw player highlight
        py, px = self.player_pos
        player_rect = pygame.Rect(self.GRID_X + px * self.CELL_SIZE, self.GRID_Y + py * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        # Pulsing highlight effect
        pulse = (math.sin(self.steps * 0.3) + 1) / 2
        highlight_color = tuple(min(255, c + int(pulse * 50)) for c in self.COLOR_PLAYER_HIGHLIGHT)
        pygame.draw.rect(self.screen, highlight_color, player_rect, width=3, border_radius=6)
        
        # Draw particles
        for x, y, _, _, radius, lifetime, color in self.particles:
            alpha = max(0, min(255, int(255 * (lifetime / 30))))
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color + (alpha,), (radius, radius), radius)
            self.screen.blit(s, (int(x - radius), int(y - radius)))

        # Draw near-full warning
        if not self.game_over and np.any(self.grid[0:2] != -1):
            pulse_alpha = 70 + 50 * math.sin(self.steps * 0.4)
            warning_surface = pygame.Surface((self.GRID_PIXEL_WIDTH, self.GRID_PIXEL_HEIGHT), pygame.SRCALPHA)
            warning_surface.fill((214, 39, 40, int(pulse_alpha)))
            self.screen.blit(warning_surface, (self.GRID_X, self.GRID_Y))

    def _render_ui(self):
        # Score
        self._draw_text(f"Score: {int(self.score)}", (self.SCREEN_WIDTH - 10, 10), self.font_medium, self.COLOR_TEXT, align="topright")
        # Stage
        self._draw_text(f"Stage: {self.stage}", (10, 10), self.font_medium, self.COLOR_TEXT, align="topleft")
        # Time
        time_left = max(0, self.max_steps_per_stage - self.stage_steps)
        time_color = (214, 39, 40) if time_left < 100 else self.COLOR_TEXT
        self._draw_text(f"Turns: {time_left}", (self.SCREEN_WIDTH // 2, 10), self.font_medium, time_color, align="midtop")
        # Lines to clear
        self._draw_text(f"Lines: {self.lines_cleared_in_stage} / 3", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 10), self.font_small, self.COLOR_TEXT, align="midbottom")
        
        # Last action info
        if self.last_action_info and not self.game_over:
            self._draw_text(self.last_action_info, (self.GRID_X + self.GRID_PIXEL_WIDTH / 2, self.GRID_Y - 15), self.font_small, (255, 220, 100), align="midbottom")

        # Game Over / Win Text
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            self._draw_text(msg, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20), self.font_large, color, align="center")
            self._draw_text(f"Final Score: {int(self.score)}", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 30), self.font_medium, self.COLOR_TEXT, align="center")

    def _draw_text(self, text, pos, font, color, align="topleft"):
        shadow_color = self.COLOR_TEXT_SHADOW
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()

        if align == "topleft": text_rect.topleft = pos
        elif align == "topright": text_rect.topright = pos
        elif align == "midtop": text_rect.midtop = pos
        elif align == "midbottom": text_rect.midbottom = pos
        elif align == "center": text_rect.center = pos
        
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "lines_cleared": self.lines_cleared_in_stage,
            "game_over": self.game_over,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Color Grid Puzzle")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                action.fill(0)

        # --- Human Controls ---
        # This part is for human play and is not part of the Gym environment logic
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        current_action = np.array([movement, space_held, shift_held])
        
        # Only step if an action is taken (turn-based)
        if np.any(current_action != 0) or True: # Step on any key press for responsiveness
            obs, reward, terminated, truncated, info = env.step(current_action)
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}")
            if terminated:
                print("Game Over!")

        # --- Rendering ---
        # The environment returns an RGB array, we need to display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(15) # Limit human play speed

    env.close()