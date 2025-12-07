import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to select a gem, then move to an adjacent gem and press Space again to swap. Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match 3 or more gems to score points. Create combos for higher scores. Reach 1000 points before the time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    GEM_TYPES = 3
    CELL_SIZE = 36
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_SIZE * CELL_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_SIZE * CELL_SIZE) // 2 + 20

    WIN_SCORE = 1000
    TIME_LIMIT_SECONDS = 60
    FPS = 30

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECT = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
    ]
    
    # Animation timings (in frames)
    ANIM_SWAP_DURATION = 6
    ANIM_FALL_SPEED = 6 # pixels per frame
    ANIM_CLEAR_DURATION = 8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Set Pygame to headless mode
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        self.grid = None
        self.gem_visual_y = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.cursor_pos = [0, 0]
        self.selected_gem_pos = None
        self.board_state = "IDLE" # IDLE, SWAPPING, REVERTING, CLEARING, FALLING
        self.animation_timer = 0
        self.swap_info = {}
        self.clearing_info = []
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.combo_multiplier = 1

        # self.reset() and self.validate_implementation() are called here
        # to ensure the environment is ready and valid upon creation.
        # The timeout error occurs in reset, specifically _create_initial_grid.
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_gem_pos = None
        self.board_state = "IDLE"
        self.animation_timer = 0
        self.swap_info = {}
        self.clearing_info = []
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.combo_multiplier = 1
        
        self._create_initial_grid()
        self.gem_visual_y = np.array([[y * self.CELL_SIZE for x in range(self.GRID_SIZE)] for y in range(self.GRID_SIZE)], dtype=float)

        return self._get_observation(), self._get_info()

    def _create_initial_grid(self):
        # FIX: The original method could enter an infinite loop.
        # This new method constructs the grid cell by cell, ensuring no matches are created.
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                possible_gems = list(range(1, self.GEM_TYPES + 1))
                
                # Check for horizontal match with the previous two gems
                if c >= 2 and self.grid[r, c - 1] == self.grid[r, c - 2]:
                    if self.grid[r, c - 1] in possible_gems:
                        possible_gems.remove(self.grid[r, c - 1])
                
                # Check for vertical match with the two gems above
                if r >= 2 and self.grid[r - 1, c] == self.grid[r - 2, c]:
                    if self.grid[r - 1, c] in possible_gems:
                        possible_gems.remove(self.grid[r - 1, c])

                self.grid[r, c] = self.np_random.choice(possible_gems)

    def step(self, action):
        reward = 0
        self.steps += 1
        self.time_left = max(0, self.time_left - 1)

        self._handle_input(action)
        reward += self._update_board_state()
        self._update_particles()
        
        terminated = self.game_over
        if not terminated and (self.time_left <= 0 or self.score >= self.WIN_SCORE):
            terminated = True
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Time out penalty

        truncated = False # This environment does not truncate based on steps
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        if self.game_over or self.board_state != "IDLE":
            # Store current state for next frame's press detection
            self.prev_space_held = (action[1] == 1)
            self.prev_shift_held = (action[2] == 1)
            return

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        # --- Movement ---
        if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE
        elif movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE

        # --- Shift (Deselect) ---
        if shift_press:
            self.selected_gem_pos = None

        # --- Space (Select/Swap) ---
        if space_press:
            if self.selected_gem_pos is None:
                self.selected_gem_pos = list(self.cursor_pos)
            else:
                if self._is_adjacent(self.selected_gem_pos, self.cursor_pos):
                    # Start swap
                    self.board_state = "SWAPPING"
                    self.animation_timer = self.ANIM_SWAP_DURATION
                    self.swap_info = {'pos1': self.selected_gem_pos, 'pos2': list(self.cursor_pos)}
                    self.selected_gem_pos = None
                else:
                    self.selected_gem_pos = list(self.cursor_pos)
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _update_board_state(self):
        reward = 0
        if self.board_state in ["SWAPPING", "REVERTING"]:
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                p1, p2 = self.swap_info['pos1'], self.swap_info['pos2']
                g = self.grid
                g[p1[1], p1[0]], g[p2[1], p2[0]] = g[p2[1], p2[0]], g[p1[1], p1[0]]
                
                if self.board_state == "SWAPPING":
                    matches = self._find_matches()
                    if matches:
                        self.combo_multiplier = 1
                        reward += self._process_matches(matches)
                    else: # No match, revert swap
                        self.board_state = "REVERTING"
                        self.animation_timer = self.ANIM_SWAP_DURATION
                else: # REVERTING finished
                    self.board_state = "IDLE"

        elif self.board_state == "CLEARING":
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                for y, x in self.clearing_info:
                    self.grid[y, x] = 0
                self.clearing_info = []
                self.board_state = "FALLING"

        elif self.board_state == "FALLING":
            is_stable = self._apply_gravity_and_refill()
            if is_stable:
                matches = self._find_matches()
                if matches:
                    self.combo_multiplier += 1
                    reward += self._process_matches(matches)
                else:
                    self.board_state = "IDLE"
                    self.combo_multiplier = 1
        return reward

    def _process_matches(self, matches):
        reward = 0
        self.board_state = "CLEARING"
        self.animation_timer = self.ANIM_CLEAR_DURATION
        self.clearing_info = list(matches)
        
        # Calculate score and reward from raw match count
        match_count = len(matches)
        if match_count == 3: reward += 1
        elif match_count == 4: reward += 5
        elif match_count >= 5: reward += 10
        
        self.score += match_count * self.combo_multiplier * 10
        
        for y, x in matches:
            self._create_particles(x, y, self.GEM_COLORS[self.grid[y, x] - 1])
        return reward

    def _find_matches(self):
        to_match = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2] != 0:
                    to_match.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c] != 0:
                    to_match.update([(r, c), (r+1, c), (r+2, c)])
        return to_match

    def _apply_gravity_and_refill(self):
        is_stable = True
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                target_y = y * self.CELL_SIZE
                if self.gem_visual_y[y, x] < target_y:
                    self.gem_visual_y[y, x] += self.ANIM_FALL_SPEED
                    self.gem_visual_y[y, x] = min(self.gem_visual_y[y, x], target_y)
                    is_stable = False

        if not is_stable:
            return False

        for c in range(self.GRID_SIZE):
            empty_count = 0
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    self.grid[r + empty_count, c] = self.grid[r, c]
                    self.gem_visual_y[r + empty_count, c] = self.gem_visual_y[r, c]
                    self.grid[r, c] = 0
                    is_stable = False
        
        if not is_stable: return False

        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.GEM_TYPES + 1)
                    self.gem_visual_y[r, c] = (r - self.GRID_SIZE) * self.CELL_SIZE # Start off-screen
                    is_stable = False
        return is_stable

    def _create_particles(self, grid_x, grid_y, color):
        cx = self.GRID_OFFSET_X + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        cy = self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([cx, cy, vx, vy, lifetime, color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1 # lifetime

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_grid_bg()
        self._render_gems()
        self._render_cursor_and_selection()
        self._render_particles()

    def _render_grid_bg(self):
        for y in range(self.GRID_SIZE + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_SIZE + 1):
            start = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_SIZE * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

    def _render_gems(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                gem_type = self.grid[y, x]
                if gem_type == 0:
                    continue

                anim_progress = 0
                dx, dy = 0, 0
                
                # Swap animation
                if self.board_state in ["SWAPPING", "REVERTING"] and self.animation_timer > 0:
                    p1, p2 = self.swap_info['pos1'], self.swap_info['pos2']
                    is_p1 = (x, y) == (p1[0], p1[1])
                    is_p2 = (x, y) == (p2[0], p2[1])
                    if is_p1 or is_p2:
                        anim_progress = (self.ANIM_SWAP_DURATION - self.animation_timer) / self.ANIM_SWAP_DURATION
                        target_pos = p2 if is_p1 else p1
                        dx = (target_pos[0] - x) * self.CELL_SIZE * anim_progress
                        dy = (target_pos[1] - y) * self.CELL_SIZE * anim_progress

                # Clear animation
                scale = 1.0
                if self.board_state == "CLEARING" and (y, x) in self.clearing_info:
                    scale = self.animation_timer / self.ANIM_CLEAR_DURATION
                
                visual_y = self.gem_visual_y[y, x]
                center_x = self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE / 2 + dx
                center_y = self.GRID_OFFSET_Y + visual_y + self.CELL_SIZE / 2 + dy

                self._draw_gem(self.screen, gem_type, center_x, center_y, scale)

    def _draw_gem(self, surface, gem_type, cx, cy, scale):
        size = int(self.CELL_SIZE * 0.75 * scale)
        if size <= 0: return
        
        color = self.GEM_COLORS[gem_type - 1]
        rect = pygame.Rect(cx - size/2, cy - size/2, size, size)

        if gem_type == 1: # Circle (Red)
            pygame.draw.circle(surface, color, (int(cx), int(cy)), int(size/2))
            pygame.draw.circle(surface, (255,255,255), (int(cx-size*0.1), int(cy-size*0.1)), int(size/6), 2)
        elif gem_type == 2: # Square (Green)
            pygame.draw.rect(surface, color, rect, border_radius=int(size*0.1))
            pygame.draw.rect(surface, (255,255,255), rect.inflate(-size*0.4, -size*0.4), 2)
        elif gem_type == 3: # Triangle (Blue)
            points = [
                (cx, cy - size/2),
                (cx - size/2, cy + size/2),
                (cx + size/2, cy + size/2),
            ]
            pygame.draw.polygon(surface, color, points)
            pygame.gfxdraw.aapolygon(surface, [(int(p[0]), int(p[1])) for p in points], color)

    def _render_cursor_and_selection(self):
        # Cursor
        cx, cy = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_OFFSET_X + cx * self.CELL_SIZE,
            self.GRID_OFFSET_Y + cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=4)
        
        # Selection
        if self.selected_gem_pos is not None:
            sx, sy = self.selected_gem_pos
            rect = pygame.Rect(
                self.GRID_OFFSET_X + sx * self.CELL_SIZE,
                self.GRID_OFFSET_Y + sy * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, 3, border_radius=4)
    
    def _render_particles(self):
        for x, y, vx, vy, life, color in self.particles:
            alpha = int(255 * (life / 30))
            radius = int(life / 6)
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color + (alpha,), (radius, radius), radius)
            self.screen.blit(temp_surf, (x - radius, y - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        # Time bar
        time_ratio = self.time_left / (self.TIME_LIMIT_SECONDS * self.FPS)
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        
        fill_width = int(bar_width * time_ratio)
        bar_color = (0, 255, 0)
        if time_ratio < 0.5: bar_color = (255, 255, 0)
        if time_ratio < 0.2: bar_color = (255, 0, 0)

        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, fill_width, bar_height))

        # Combo
        if self.combo_multiplier > 1:
            combo_text = self.font_large.render(f"x{self.combo_multiplier} COMBO!", True, self.COLOR_SELECT)
            text_rect = combo_text.get_rect(center=(self.SCREEN_WIDTH/2, self.GRID_OFFSET_Y - 25))
            self.screen.blit(combo_text, text_rect)

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME UP!"
            end_text = self.font_large.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "combo": self.combo_multiplier,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly.
    # NOTE: It requires a graphical display. The environment class itself is headless.
    # To run this, you might need to comment out `os.environ["SDL_VIDEODRIVER"] = "dummy"`
    # in the __init__ method.
    try:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        # We need to unset the dummy driver to create a display
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Gem Matcher")
        clock = pygame.time.Clock()

        running = True
        total_reward = 0
        
        while running:
            movement = 0 # No-op
            space_held = 0
            shift_held = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Transpose observation for Pygame display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0

            clock.tick(GameEnv.FPS)

        env.close()
    except pygame.error as e:
        print("\nCould not run in display mode. This is expected in a headless environment.")
        print("The GameEnv class is working correctly in headless mode.")
        print(f"Pygame error: {e}")