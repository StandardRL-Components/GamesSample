
# Generated: 2025-08-28T03:29:01.327351
# Source Brief: brief_04932.md
# Brief Index: 4932

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to match a group of 3 or more fruits."
    )

    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )
    
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_COLS = 10
        self.GRID_ROWS = 12
        self.CELL_SIZE = 32
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_COLS * self.CELL_SIZE) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_ROWS * self.CELL_SIZE) + 10

        self.NUM_FRUIT_TYPES = 5
        self.FPS = 30
        self.GAME_DURATION = 60.0
        self.TARGET_SCORE = 5000
        self.MIN_MATCH_SIZE = 3
        self.COMBO_RESET_SECONDS = 2.0

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_UI_BG = (10, 20, 30, 180)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_RED = (255, 80, 80)
        
        self.FRUIT_COLORS = [
            (255, 50, 50),   # Red
            (50, 200, 50),   # Green
            (50, 150, 255),  # Blue
            (255, 150, 50),  # Orange
            (150, 50, 200),  # Purple
        ]
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_big = pygame.font.Font(None, 64)
            self.font_float = pygame.font.Font(None, 22)
        except IOError:
            self.font_ui = pygame.font.SysFont("sans-serif", 28)
            self.font_big = pygame.font.SysFont("sans-serif", 64)
            self.font_float = pygame.font.SysFont("sans-serif", 22)

        # State variables
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.timer = None
        self.combo_multiplier = None
        self.last_space_held = None
        self.game_over = None
        self.steps = None
        self.rng = None
        self.particles = []
        self.floating_texts = []
        self.last_match_step = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.timer = self.GAME_DURATION
        self.game_over = False
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.last_space_held = False
        self.combo_multiplier = 1
        self.last_match_step = - (self.FPS * self.COMBO_RESET_SECONDS) # Allow combo immediately
        self.particles.clear()
        self.floating_texts.clear()
        
        self._init_grid()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        self.timer -= 1.0 / self.FPS

        if self.steps - self.last_match_step > self.FPS * self.COMBO_RESET_SECONDS:
            self.combo_multiplier = 1

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        space_press = space_held and not self.last_space_held
        if space_press:
            match_info = self._process_match_at_cursor()
            reward += match_info["reward"]
            
            if match_info.get("fruits_cleared", 0) > 0:
                self._apply_gravity()
                self._refill_grid()
                
                is_cascade = True
                while True:
                    cascade_matches = self._find_all_grid_matches()
                    if not cascade_matches: break
                    
                    cascade_info = self._handle_matches(cascade_matches, is_cascade=is_cascade)
                    reward += cascade_info["reward"]
                    is_cascade = True
                    
                    self._apply_gravity()
                    self._refill_grid()
                    # // Sound: cascade_fall

        self.last_space_held = space_held
        
        terminated = False
        if self.timer <= 0:
            self.timer = 0
            terminated = True
            if self.score < self.TARGET_SCORE: reward -= 100
        if self.score >= self.TARGET_SCORE:
            terminated = True
            reward += 100
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _init_grid(self):
        self.grid = self.rng.integers(1, self.NUM_FRUIT_TYPES + 1, size=(self.GRID_ROWS, self.GRID_COLS))
        while True:
            matches = self._find_all_grid_matches()
            if not matches: break
            for group in matches:
                for r, c in group:
                    self.grid[r, c] = 0
            self._apply_gravity()
            self._refill_grid()

    def _process_match_at_cursor(self):
        col, row = self.cursor_pos
        if self.grid[row, col] == 0: return {"reward": -0.1, "fruits_cleared": 0}

        q = [(row, col)]
        visited = set(q)
        match_group = []
        fruit_type = self.grid[row, col]

        while q:
            r, c = q.pop(0)
            match_group.append((r, c))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and \
                   (nr, nc) not in visited and self.grid[nr, nc] == fruit_type:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        
        if len(match_group) >= self.MIN_MATCH_SIZE:
            # // Sound: match_success
            info = self._handle_matches([match_group], is_cascade=False)
            info["fruits_cleared"] = len(match_group)
            return info
        else:
            # // Sound: buzz_error
            self._create_floating_text("X", self.cursor_pos, self.COLOR_RED)
            return {"reward": -0.1, "fruits_cleared": 0}

    def _find_all_grid_matches(self):
        potentially_matched = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == 0: continue
                
                if c <= self.GRID_COLS - self.MIN_MATCH_SIZE:
                    if all(self.grid[r, c+i] == self.grid[r,c] for i in range(self.MIN_MATCH_SIZE)):
                        for i in range(self.MIN_MATCH_SIZE): potentially_matched.add((r, c + i))

                if r <= self.GRID_ROWS - self.MIN_MATCH_SIZE:
                    if all(self.grid[r+i, c] == self.grid[r,c] for i in range(self.MIN_MATCH_SIZE)):
                        for i in range(self.MIN_MATCH_SIZE): potentially_matched.add((r + i, c))
        
        all_groups = []
        visited = set()
        for r_start, c_start in potentially_matched:
            if (r_start, c_start) not in visited:
                fruit_type = self.grid[r_start, c_start]
                group, q = [], [(r_start, c_start)]
                visited.add((r_start, c_start))
                while q:
                    r, c = q.pop(0)
                    group.append((r, c))
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in potentially_matched and (nr, nc) not in visited and self.grid[nr, nc] == fruit_type:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                all_groups.append(group)
        return all_groups

    def _handle_matches(self, all_matches, is_cascade):
        total_reward = 0
        if not all_matches: return {"reward": 0}

        if self.steps - self.last_match_step > self.FPS * self.COMBO_RESET_SECONDS:
            self.combo_multiplier = 1
        
        if is_cascade:
            self.combo_multiplier += 1
            total_reward += 10 # Combo activation reward
            # // Sound: combo_tick
        elif self.combo_multiplier == 1:
            self.combo_multiplier = 2
            total_reward += 10 # Start combo
        else:
            self.combo_multiplier += 1
            total_reward += 10 # Continue combo

        for group in all_matches:
            num_fruits = len(group)
            base_score = num_fruits * 10
            bonus_score = (num_fruits - self.MIN_MATCH_SIZE) ** 2 * 10
            score_gain = (base_score + bonus_score) * self.combo_multiplier
            
            self.score += score_gain
            total_reward += num_fruits
            if num_fruits >= 6: total_reward += 5

            avg_x = sum(c for r, c in group) / num_fruits
            avg_y = sum(r for r, c in group) / num_fruits
            self._create_particles((avg_x, avg_y), self.FRUIT_COLORS[self.grid[group[0]] - 1], num_fruits)
            self._create_floating_text(f"+{score_gain}", (avg_x, avg_y), self.COLOR_WHITE)

            for r, c in group: self.grid[r, c] = 0

        self.last_match_step = self.steps
        return {"reward": total_reward}

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], 0
                    empty_row -= 1

    def _refill_grid(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.rng.integers(1, self.NUM_FRUIT_TYPES + 1)
                    
    def _create_particles(self, pos, color, count):
        grid_x, grid_y = pos
        screen_x = self.GRID_X_OFFSET + (grid_x + 0.5) * self.CELL_SIZE
        screen_y = self.GRID_Y_OFFSET + (grid_y + 0.5) * self.CELL_SIZE
        for _ in range(count * 3):
            angle = self.rng.random() * 2 * math.pi
            speed = 2 + self.rng.random() * 4
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = 20 + self.rng.integers(0, 20)
            self.particles.append([[screen_x, screen_y], vel, lifetime, color])

    def _create_floating_text(self, text, pos, color):
        grid_x, grid_y = pos
        screen_x = self.GRID_X_OFFSET + (grid_x + 0.5) * self.CELL_SIZE
        screen_y = self.GRID_Y_OFFSET + (grid_y + 0.5) * self.CELL_SIZE
        self.floating_texts.append({
            "pos": [screen_x, screen_y], "text": text, "color": color, "lifetime": self.FPS * 1.5
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_COLS * self.CELL_SIZE, self.GRID_ROWS * self.CELL_SIZE), border_radius=5)

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                fruit_type = self.grid[r, c]
                if fruit_type > 0:
                    color = self.FRUIT_COLORS[fruit_type - 1]
                    cx, cy = self.GRID_X_OFFSET + c * self.CELL_SIZE + self.CELL_SIZE // 2, self.GRID_Y_OFFSET + r * self.CELL_SIZE + self.CELL_SIZE // 2
                    radius = self.CELL_SIZE // 2 - 4
                    pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, tuple(max(0, x-50) for x in color))
                    pygame.gfxdraw.filled_circle(self.screen, cx-radius//3, cy-radius//3, radius//4, (255,255,255,100))
        
        cursor_pulse = (math.sin(self.steps * 0.2) + 1) / 2
        cursor_alpha = 100 + cursor_pulse * 100
        cursor_rect = pygame.Rect(self.GRID_X_OFFSET + self.cursor_pos[0] * self.CELL_SIZE, self.GRID_Y_OFFSET + self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, (255, 255, 255, cursor_alpha), s.get_rect(), border_radius=4)
        self.screen.blit(s, cursor_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, cursor_rect, 2, border_radius=4)

        for p in self.particles[:]:
            p[0][0] += p[1][0]; p[0][1] += p[1][1]; p[1][1] += 0.2; p[2] -= 1
            if p[2] <= 0: self.particles.remove(p)
            else:
                alpha = max(0, min(255, p[2] * 10)); color = p[3] + (alpha,)
                s = pygame.Surface((4,4), pygame.SRCALPHA); s.fill(color)
                self.screen.blit(s, (int(p[0][0]), int(p[0][1])))

        for ft in self.floating_texts[:]:
            ft["pos"][1] -= 0.5; ft["lifetime"] -= 1
            if ft["lifetime"] <= 0: self.floating_texts.remove(ft)
            else:
                alpha = max(0, min(255, ft["lifetime"] * 4))
                text_surf = self.font_float.render(ft["text"], True, ft["color"])
                text_surf.set_alpha(alpha)
                self.screen.blit(text_surf, text_surf.get_rect(center=(int(ft["pos"][0]), int(ft["pos"][1]))))

    def _render_ui(self):
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA); ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))

        score_surf = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_surf, (15, 12))

        time_color = self.COLOR_RED if self.timer < 10 and self.steps % self.FPS < self.FPS/2 else self.COLOR_WHITE
        time_surf = self.font_ui.render(f"Time: {max(0, math.ceil(self.timer))}", True, time_color)
        self.screen.blit(time_surf, time_surf.get_rect(centerx=self.SCREEN_WIDTH / 2, y=12))

        combo_color = self.COLOR_GOLD if self.combo_multiplier > 1 else self.COLOR_WHITE
        combo_surf = self.font_ui.render(f"Combo: x{self.combo_multiplier}", True, combo_color)
        self.screen.blit(combo_surf, combo_surf.get_rect(right=self.SCREEN_WIDTH - 15, y=12))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.score >= self.TARGET_SCORE else "TIME'S UP!"
            color = self.COLOR_GOLD if self.score >= self.TARGET_SCORE else self.COLOR_RED
            msg_surf = self.font_big.render(msg, True, color)
            self.screen.blit(msg_surf, msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

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

    def close(self):
        pygame.quit()