import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Match-3 style puzzle game where the player swaps adjacent numbers on a grid
    to create matches of three or more. The goal is to reach a score of 500
    before the 60-second timer runs out. The environment is designed for
    visual quality and satisfying gameplay, with smooth animations and clear feedback.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A fast-paced match-3 puzzle game. Swap adjacent numbers to create matches of "
        "three or more and race against the clock to reach the target score."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press 'space' to select a number, "
        "then move to an adjacent number and press 'space' again to swap. Press 'shift' to deselect."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 4
    CELL_SIZE = 80
    GRID_ORIGIN_X = (SCREEN_WIDTH - GRID_SIZE * CELL_SIZE) // 2
    GRID_ORIGIN_Y = (SCREEN_HEIGHT - GRID_SIZE * CELL_SIZE) // 2 + 20
    MAX_SCORE = 500
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = 1000 # Safety termination

    # --- Animation States ---
    STATE_IDLE = 0
    STATE_SWAP_ANIM = 1
    STATE_MATCH_ANIM = 2
    STATE_FALL_ANIM = 3
    STATE_GAME_OVER = 4

    # --- Colors ---
    COLOR_BG = (15, 20, 30)
    COLOR_GRID_LINE = (50, 60, 70)
    COLOR_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTED = (0, 255, 255)
    COLOR_MATCH_FLASH = (255, 255, 255)
    COLOR_MAP = {
        1: (255, 80, 80),   # Red
        2: (80, 255, 80),   # Green
        3: (80, 150, 255),  # Blue
        4: (255, 255, 80),  # Yellow
        5: (255, 80, 255),  # Magenta
        6: (80, 255, 255),  # Cyan
        7: (255, 160, 80),  # Orange
        8: (160, 80, 255),  # Purple
        9: (200, 200, 200)  # White/Gray
    }

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
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # Game state variables
        self.grid = None
        self.cursor_pos = None
        self.selected_cell = None
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.match_count = 0
        self.match_multiplier = 1.0
        self.game_over = False
        self.win_condition_met = False
        
        # Animation state
        self.game_state = self.STATE_IDLE
        self.animation_progress = 0.0
        self.animation_duration = 0
        self.swap_info = None
        self.matched_cells = set()
        self.fall_map = {}
        self.step_reward = 0.0

        # Action handling
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reset()
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        while True:
            self.grid = self.np_random.integers(1, 10, size=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
            if not self._find_all_matches():
                break
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.timer = self.GAME_DURATION_SECONDS * 30 # Assuming 30 FPS
        self.match_count = 0
        self.match_multiplier = 1.0
        
        self.cursor_pos = [0, 0]
        self.selected_cell = None
        
        self.game_state = self.STATE_IDLE
        self.animation_progress = 0.0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.step_reward = 0.0
        self.steps += 1
        self.timer = max(0, self.timer - 1)

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        
        if self.game_state == self.STATE_IDLE:
            self._handle_input(movement, space_press, shift_press)
        else:
            self._update_animations()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            self.game_state = self.STATE_GAME_OVER
            if self.win_condition_met:
                self.step_reward += 100
            else: # Timer ran out or max steps
                self.step_reward += -10
        
        reward = self.step_reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_press, shift_press):
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        
        if shift_press and self.selected_cell is not None:
            self.selected_cell = None
            # sfx: cancel_selection

        if space_press:
            if self.selected_cell is None:
                self.selected_cell = list(self.cursor_pos)
                # sfx: select_gem
            else:
                r1, c1 = self.selected_cell
                r2, c2 = self.cursor_pos
                
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    self._initiate_swap((r1, c1), (r2, c2))
                else: # Not adjacent, treat as new selection
                    self.selected_cell = list(self.cursor_pos)
                    # sfx: reselect_gem

    def _initiate_swap(self, cell1, cell2):
        r1, c1 = cell1
        r2, c2 = cell2
        
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        matches = self._find_all_matches()
        
        if matches:
            self.step_reward = 0.1
            self.swap_info = (cell1, cell2, True) # True for valid swap
            self.matched_cells = matches
            self._start_animation(self.STATE_SWAP_ANIM, 8)
            # sfx: swap_success
        else:
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1] # Swap back
            self.step_reward = -0.1
            self.swap_info = (cell1, cell2, False) # False for invalid
            self._start_animation(self.STATE_SWAP_ANIM, 12)
            # sfx: swap_fail
        
        self.selected_cell = None
    
    def _update_animations(self):
        self.animation_progress += 1.0
        if self.animation_progress < self.animation_duration:
            return

        if self.game_state == self.STATE_SWAP_ANIM:
            _, _, success = self.swap_info
            if success:
                self._process_matches()
            else:
                self.game_state = self.STATE_IDLE

        elif self.game_state == self.STATE_MATCH_ANIM:
            self._apply_gravity_and_refill()
            self._start_animation(self.STATE_FALL_ANIM, 10)

        elif self.game_state == self.STATE_FALL_ANIM:
            self._finalize_fall()
            new_matches = self._find_all_matches()
            if new_matches:
                self.matched_cells = new_matches
                self._process_matches() # Cascade
            else:
                self.game_state = self.STATE_IDLE

    def _process_matches(self):
        # sfx: match_found
        for r, c in self.matched_cells:
            if self.grid[r, c] != 0: # Avoid double counting score in complex cascades
                self.score += self.grid[r, c] * self.match_multiplier
        
        # Reward based on match length
        # This is a simplification; a more complex system could analyze individual match groups
        match_size = len(self.matched_cells)
        if match_size >= 3:
            self.step_reward += (match_size - 2)
            
        self.match_count += 1
        if self.match_count % 5 == 0:
            self.match_multiplier += 0.1
        
        self._start_animation(self.STATE_MATCH_ANIM, 15)

    def _apply_gravity_and_refill(self):
        self.fall_map = {}
        for r_match, c_match in self.matched_cells:
            self.grid[r_match, c_match] = 0 # Mark for removal

        for c in range(self.GRID_SIZE):
            fall_dist = 0
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] == 0:
                    fall_dist += 1
                elif fall_dist > 0:
                    self.fall_map[(r, c)] = fall_dist
    
    def _finalize_fall(self):
        new_grid = self.np_random.integers(1, 10, size=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        
        for c in range(self.GRID_SIZE):
            write_idx = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    new_grid[write_idx, c] = self.grid[r, c]
                    write_idx -= 1
        self.grid = new_grid
        # sfx: gems_fall

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2] != 0:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c] != 0:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _start_animation(self, state, duration):
        self.game_state = state
        self.animation_duration = duration
        self.animation_progress = 0

    def _check_termination(self):
        self.win_condition_met = self.score >= self.MAX_SCORE
        return self.timer <= 0 or self.win_condition_met or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer / 30, "multiplier": self.match_multiplier}

    def _render_game(self):
        self._draw_grid_lines()
        
        # Draw static cells first
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                is_swapping = self.game_state == self.STATE_SWAP_ANIM and ((r,c) in self.swap_info[:2])
                is_falling = self.game_state == self.STATE_FALL_ANIM and (r,c) in self.fall_map
                if not is_swapping and not is_falling:
                    self._draw_cell_at(r, c)
        
        # Draw animated cells on top
        if self.game_state == self.STATE_SWAP_ANIM:
            self._draw_swap_animation()
        elif self.game_state == self.STATE_FALL_ANIM:
            self._draw_fall_animation()

        self._draw_cursor()
    
    def _draw_grid_lines(self):
        for i in range(self.GRID_SIZE + 1):
            # Horizontal
            start_pos = (self.GRID_ORIGIN_X, self.GRID_ORIGIN_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_ORIGIN_X + self.GRID_SIZE * self.CELL_SIZE, self.GRID_ORIGIN_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 1)
            # Vertical
            start_pos = (self.GRID_ORIGIN_X + i * self.CELL_SIZE, self.GRID_ORIGIN_Y)
            end_pos = (self.GRID_ORIGIN_X + i * self.CELL_SIZE, self.GRID_ORIGIN_Y + self.GRID_SIZE * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 1)

    def _draw_cell_at(self, r, c, offset_px=(0,0), scale=1.0, alpha=255):
        if self.grid[r, c] == 0: return

        is_matching = self.game_state == self.STATE_MATCH_ANIM and (r,c) in self.matched_cells
        
        # Calculate animation effects
        if is_matching:
            p = self.animation_progress / self.animation_duration
            # Flash white then shrink
            if p < 0.5:
                scale = 1.0 + 0.2 * math.sin(p * 2 * math.pi) # Pulse
                color_lerp = p * 2
            else:
                scale = 1.0 - ((p - 0.5) * 2)
                color_lerp = 1.0
        else:
            color_lerp = 0.0

        number = self.grid[r, c]
        color = self.COLOR_MAP[number]
        
        if color_lerp > 0:
            color = tuple(int(color[i] + (self.COLOR_MATCH_FLASH[i] - color[i]) * color_lerp) for i in range(3))

        size = int(self.CELL_SIZE * scale)
        if size <= 0: return
        
        x = self.GRID_ORIGIN_X + c * self.CELL_SIZE + (self.CELL_SIZE - size) // 2 + offset_px[0]
        y = self.GRID_ORIGIN_Y + r * self.CELL_SIZE + (self.CELL_SIZE - size) // 2 + offset_px[1]
        
        cell_rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, color, cell_rect, border_radius=8)
        
        text_surf = self.font_medium.render(str(number), True, self.COLOR_BG)
        text_rect = text_surf.get_rect(center=cell_rect.center)
        self.screen.blit(text_surf, text_rect)

    def _draw_swap_animation(self):
        p = self.animation_progress / self.animation_duration
        # Ease in-out cubic
        t = 4 * p * p * p if p < 0.5 else 1 - pow(-2 * p + 2, 3) / 2
        
        (r1, c1), (r2, c2), _ = self.swap_info
        
        dx = (c2 - c1) * self.CELL_SIZE * t
        dy = (r2 - r1) * self.CELL_SIZE * t
        
        self._draw_cell_at(r2, c2, offset_px=(int(dx), int(dy))) # The one that was at (r2,c2) and is moving to (r1,c1)
        self._draw_cell_at(r1, c1, offset_px=(int(-dx), int(-dy))) # The one that was at (r1,c1) and is moving to (r2,c2)

    def _draw_fall_animation(self):
        p = self.animation_progress / self.animation_duration
        t = 1 - (1 - p)**3 # Ease-out cubic

        # Draw falling pieces
        for (r,c), dist in self.fall_map.items():
            offset_y = -dist * self.CELL_SIZE * (1 - t)
            self._draw_cell_at(r, c, offset_px=(0, int(offset_y)))
        
        # Draw new pieces appearing from top
        cols_with_new = {c for r,c in self.matched_cells}
        for c in cols_with_new:
            empty_count = sum(1 for r in range(self.GRID_SIZE) if self.grid[r,c] == 0)
            for i in range(empty_count):
                r = i
                offset_y = -self.CELL_SIZE * (1-t)
                # Need to get the number from the *final* grid state
                final_grid_val = self.grid[r,c] # This is a bit of a look-ahead
                if final_grid_val != 0:
                    self._draw_new_cell(r, c, final_grid_val, offset_px=(0, int(offset_y)))

    def _draw_new_cell(self, r, c, number, offset_px):
        color = self.COLOR_MAP[number]
        size = self.CELL_SIZE
        x = self.GRID_ORIGIN_X + c * self.CELL_SIZE + offset_px[0]
        y = self.GRID_ORIGIN_Y + (r - self.GRID_SIZE) * self.CELL_SIZE + offset_px[1] # Start from above the grid
        y = self.GRID_ORIGIN_Y + r * self.CELL_SIZE + offset_px[1]

        cell_rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, color, cell_rect, border_radius=8)
        
        text_surf = self.font_medium.render(str(number), True, self.COLOR_BG)
        text_rect = text_surf.get_rect(center=cell_rect.center)
        self.screen.blit(text_surf, text_rect)

    def _draw_cursor(self):
        r, c = self.cursor_pos
        x = self.GRID_ORIGIN_X + c * self.CELL_SIZE
        y = self.GRID_ORIGIN_Y + r * self.CELL_SIZE
        
        # Pulsating effect for selected cell
        if self.selected_cell and self.selected_cell == [r,c]:
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            width = 2 + int(pulse * 2)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, (x, y, self.CELL_SIZE, self.CELL_SIZE), width, border_radius=8)
        else: # Normal cursor
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, (x, y, self.CELL_SIZE, self.CELL_SIZE), 3, border_radius=8)

        # Draw border for the first selected cell even if cursor moves away
        if self.selected_cell and self.selected_cell != [r,c]:
            r_sel, c_sel = self.selected_cell
            x_sel = self.GRID_ORIGIN_X + c_sel * self.CELL_SIZE
            y_sel = self.GRID_ORIGIN_Y + r_sel * self.CELL_SIZE
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, (x_sel, y_sel, self.CELL_SIZE, self.CELL_SIZE), 3, border_radius=8)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Multiplier
        mult_text = f"MULT: {self.match_multiplier:.1f}x"
        mult_surf = self.font_small.render(mult_text, True, self.COLOR_TEXT)
        self.screen.blit(mult_surf, (20, 40))

        # Timer bar
        timer_bar_width = 200
        timer_bar_height = 20
        timer_x = self.SCREEN_WIDTH - timer_bar_width - 20
        timer_y = 15
        
        ratio = self.timer / (self.GAME_DURATION_SECONDS * 30)
        current_width = int(timer_bar_width * ratio)
        
        if ratio < 0.25: color = (255, 80, 80)
        elif ratio < 0.6: color = (255, 255, 80)
        else: color = (80, 255, 80)
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, (timer_x, timer_y, timer_bar_width, timer_bar_height))
        if current_width > 0:
            pygame.draw.rect(self.screen, color, (timer_x, timer_y, current_width, timer_bar_height))

        # Game Over Screen
        if self.game_state == self.STATE_GAME_OVER:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(overlay, (0,0))
            
            msg = "YOU WIN!" if self.win_condition_met else "TIME'S UP!"
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Match-3 Gym Environment")
    clock = pygame.time.Clock()

    movement = 0
    space = 0
    shift = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # Keydown events for continuous actions
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
            
            # Keyup events to reset actions
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    movement = 0
                if event.key == pygame.K_SPACE:
                    space = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 0

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Timer: {info['timer']:.2f}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()