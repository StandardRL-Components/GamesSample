
# Generated: 2025-08-27T19:26:11.070960
# Source Brief: brief_02148.md
# Brief Index: 2148

        
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
        "Controls: Use arrow keys to move the cursor. "
        "Press Space to select a monster, then select an adjacent monster to swap them and make a match!"
    )

    game_description = (
        "A fast-paced puzzle game. Match 3 or more monsters of the same color to score points. "
        "Reach the target score before the timer runs out!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20
        self.NUM_MONSTER_TYPES = 6
        self.WIN_SCORE = 500
        self.INITIAL_TIMER = 90.0
        self.MAX_STEPS = int(self.INITIAL_TIMER * self.FPS) + 100 # Safety buffer

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_LINES = (50, 60, 70)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_SELECTED = (0, 255, 255)
        self.COLOR_TEXT = (220, 220, 230)
        self.MONSTER_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)

        # --- Game State Initialization ---
        self.grid = None
        self.cursor_pos = None
        self.selected_cell = None
        self.timer = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.last_space_held = False
        self.rng = None
        self.pending_reward = 0
        
        # Animation and Effects State
        self.animation_state = "IDLE"
        self.animation_progress = 0
        self.animation_data = {}
        self.particles = []
        self.combo_multiplier = 1
        self.combo_display_timer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.grid = self._create_initial_grid()
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_cell = None
        self.timer = self.INITIAL_TIMER
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.pending_reward = 0
        
        self.animation_state = "IDLE"
        self.animation_progress = 0
        self.animation_data = {}
        self.particles = []
        self.combo_multiplier = 1
        self.combo_display_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        
        if not self.game_over:
            self.timer -= 1.0 / self.FPS
            self._update_animations()

            if self.animation_state == "IDLE":
                self._handle_input(action)
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                self.pending_reward += 100
            elif self.timer <= 0:
                self.pending_reward -= 100
            self.game_over = True
            self.animation_state = "GAME_OVER"

        reward = self.pending_reward
        self.pending_reward = 0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        # --- Selection Logic (on key press) ---
        is_space_press = space_held and not self.last_space_held
        if is_space_press:
            cx, cy = self.cursor_pos
            if self.selected_cell is None:
                self.selected_cell = [cx, cy]
            else:
                sx, sy = self.selected_cell
                if (cx, cy) == (sx, sy): # Deselect if same cell
                    self.selected_cell = None
                elif abs(cx - sx) + abs(cy - sy) == 1: # Adjacent cell
                    self.pending_reward += 0.1 # Swap attempt reward
                    self._start_swap([sx, sy], [cx, cy])
                    self.selected_cell = None
                else: # Non-adjacent cell, make it the new selection
                    self.selected_cell = [cx, cy]
        
        self.last_space_held = space_held

    def _check_termination(self):
        return self.score >= self.WIN_SCORE or self.timer <= 0 or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_monsters()
        self._render_cursor_and_selection()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Game Logic ---
    def _create_initial_grid(self):
        while True:
            grid = self.rng.integers(0, self.NUM_MONSTER_TYPES, size=(self.GRID_COLS, self.GRID_ROWS), dtype=np.int8)
            if not self._find_matches(grid):
                return grid

    def _find_matches(self, grid):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if grid[c, r] == -1: continue
                # Horizontal
                if c < self.GRID_COLS - 2 and grid[c, r] == grid[c + 1, r] == grid[c + 2, r]:
                    for i in range(3): matches.add((c + i, r))
                # Vertical
                if r < self.GRID_ROWS - 2 and grid[c, r] == grid[c, r + 1] == grid[c, r + 2]:
                    for i in range(3): matches.add((c, r + i))
        return list(matches)

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[c, r] != -1:
                    if r != empty_row:
                        self.grid[c, empty_row] = self.grid[c, r]
                        self.grid[c, r] = -1
                    empty_row -= 1
        
    def _fill_top_rows(self):
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS):
                if self.grid[c, r] == -1:
                    self.grid[c, r] = self.rng.integers(0, self.NUM_MONSTER_TYPES)

    # --- Animation State Machine ---
    def _start_swap(self, cell1, cell2):
        self.animation_state = "SWAP"
        self.animation_progress = 0
        self.animation_data = {'cell1': cell1, 'cell2': cell2, 'is_swap_back': False}
        # sound: swap_start.wav

    def _update_animations(self):
        if self.animation_state == "SWAP":
            self._update_swap()
        elif self.animation_state == "MATCH":
            self._update_match()
        elif self.animation_state == "FALL":
            self._update_fall()

    def _update_swap(self):
        self.animation_progress += 1.0 / (self.FPS * 0.2) # 0.2 second swap
        if self.animation_progress >= 1.0:
            c1, c2 = self.animation_data['cell1'], self.animation_data['cell2']
            
            val1 = self.grid[c1[0], c1[1]]
            self.grid[c1[0], c1[1]] = self.grid[c2[0], c2[1]]
            self.grid[c2[0], c2[1]] = val1

            if self.animation_data['is_swap_back']:
                self.animation_state = "IDLE"
                return

            matches = self._find_matches(self.grid)
            if not matches:
                self.pending_reward -= 0.1 # Invalid swap penalty
                self.animation_data = {'cell1': c2, 'cell2': c1, 'is_swap_back': True}
                self.animation_progress = 0
                # sound: invalid_swap.wav
            else:
                self.animation_state = "MATCH"
                self.animation_progress = 0
                self.combo_multiplier = 1
                self._process_matches(matches)

    def _process_matches(self, matches):
        num_matched = len(matches)
        self.pending_reward += num_matched + max(0, num_matched - 3)
        self.score += (num_matched + max(0, num_matched - 3)) * 10 * self.combo_multiplier
        
        # sound: match_success.wav
        for x, y in matches:
            if self.grid[x, y] != -1:
                self._create_particles(x, y, self.grid[x, y])
                self.grid[x, y] = -1
        
        self.combo_display_timer = self.FPS * 1.5
        self.animation_data = {'matches': matches}
        
    def _update_match(self):
        self.animation_progress += 1.0 / (self.FPS * 0.3)
        if self.animation_progress >= 1.0:
            self.animation_state = "FALL"
            self.animation_progress = 0
            self._apply_gravity()
            self._fill_top_rows()

    def _update_fall(self):
        self.animation_progress += 1.0 / (self.FPS * 0.25)
        if self.animation_progress >= 1.0:
            matches = self._find_matches(self.grid)
            if matches:
                self.combo_multiplier += 1
                self.animation_state = "MATCH"
                self.animation_progress = 0
                self._process_matches(matches)
                # sound: combo.wav
            else:
                self.animation_state = "IDLE"
                self.combo_multiplier = 1

    # --- Rendering ---
    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT), 1)

    def _render_monsters(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                monster_type = self.grid[c, r]
                if monster_type == -1: continue

                center_x = self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                size_mod = 1.0

                if self.animation_state == "SWAP":
                    c1, c2 = self.animation_data['cell1'], self.animation_data['cell2']
                    prog = min(1.0, self.animation_progress)
                    if (c, r) == tuple(c1):
                        target_x = self.GRID_X + c2[0] * self.CELL_SIZE + self.CELL_SIZE // 2
                        target_y = self.GRID_Y + c2[1] * self.CELL_SIZE + self.CELL_SIZE // 2
                        center_x = int(pygame.math.lerp(center_x, target_x, prog))
                        center_y = int(pygame.math.lerp(center_y, target_y, prog))
                    elif (c, r) == tuple(c2):
                        target_x = self.GRID_X + c1[0] * self.CELL_SIZE + self.CELL_SIZE // 2
                        target_y = self.GRID_Y + c1[1] * self.CELL_SIZE + self.CELL_SIZE // 2
                        center_x = int(pygame.math.lerp(center_x, target_x, prog))
                        center_y = int(pygame.math.lerp(center_y, target_y, prog))
                
                if self.animation_state == "MATCH" and (c,r) in self.animation_data.get('matches', []):
                    size_mod = 1.0 - self.animation_progress
                    if int(self.animation_progress * 10) % 2 == 0:
                        size_mod *= 1.2 # Flash effect

                self._draw_monster(self.screen, (center_x, center_y), monster_type, int(self.CELL_SIZE * 0.8 * size_mod))

    def _draw_monster(self, surface, pos, m_type, size):
        if size <= 1: return
        color = self.MONSTER_COLORS[m_type]
        rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
        
        if m_type == 0: pygame.draw.circle(surface, color, pos, size // 2)
        elif m_type == 1: pygame.draw.rect(surface, color, rect, border_radius=size//5)
        elif m_type == 2:
            points = [(pos[0], pos[1] - size // 2), (pos[0] + size // 2, pos[1]), 
                      (pos[0], pos[1] + size // 2), (pos[0] - size // 2, pos[1])]
            pygame.draw.polygon(surface, color, points)
        elif m_type == 3:
            points = [(pos[0], pos[1] - size // 2), (pos[0] - size // 2, pos[1] + size // 2),
                      (pos[0] + size // 2, pos[1] + size // 2)]
            pygame.draw.polygon(surface, color, points)
        elif m_type == 4:
             points = [(pos[0] + math.cos(math.radians(a)) * size//2, pos[1] + math.sin(math.radians(a)) * size//2) for a in range(30, 361, 60)]
             pygame.draw.polygon(surface, color, points)
        else:
            pygame.draw.line(surface, color, (rect.left, rect.top), (rect.right, rect.bottom), max(1, size // 5))
            pygame.draw.line(surface, color, (rect.left, rect.bottom), (rect.right, rect.top), max(1, size // 5))

        eye_size = max(1, size // 8)
        eye_y = pos[1] - size // 6
        pygame.draw.circle(surface, (255, 255, 255), (pos[0] - size // 4, eye_y), eye_size)
        pygame.draw.circle(surface, (255, 255, 255), (pos[0] + size // 4, eye_y), eye_size)
        pygame.draw.circle(surface, (0, 0, 0), (pos[0] - size // 4, eye_y), max(1, eye_size // 2))
        pygame.draw.circle(surface, (0, 0, 0), (pos[0] + size // 4, eye_y), max(1, eye_size // 2))

    def _render_cursor_and_selection(self):
        if self.animation_state != "IDLE": return
        
        if self.selected_cell:
            sx, sy = self.selected_cell
            rect = pygame.Rect(self.GRID_X + sx * self.CELL_SIZE, self.GRID_Y + sy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, 3, border_radius=5)

        cx, cy = self.cursor_pos
        rect = pygame.Rect(self.GRID_X + cx * self.CELL_SIZE, self.GRID_Y + cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=5)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 10))

        timer_text = self.font_small.render(f"TIME: {max(0, int(self.timer))}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (20, 10))

        if self.combo_display_timer > 0 and self.combo_multiplier > 1:
            self.combo_display_timer -= 1
            alpha = min(255, int(255 * (self.combo_display_timer / (self.FPS * 0.5))))
            combo_surf = self.font_large.render(f"COMBO x{self.combo_multiplier}!", True, self.COLOR_TEXT)
            combo_surf.set_alpha(alpha)
            pos = (self.WIDTH // 2 - combo_surf.get_width() // 2, self.HEIGHT - 70)
            self.screen.blit(combo_surf, pos)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME'S UP!"
            text = self.font_large.render(msg, True, self.COLOR_TEXT)
            pos = (self.WIDTH // 2 - text.get_width() // 2, self.HEIGHT // 2 - text.get_height() // 2)
            self.screen.blit(text, pos)

    def _create_particles(self, grid_c, grid_r, monster_type):
        center_x = self.GRID_X + grid_c * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.GRID_Y + grid_r * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.MONSTER_COLORS[monster_type]
        for _ in range(15):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'color': color, 'life': self.rng.integers(20, 40)})

    def _render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                radius = int(p['life'] / 8)
                if radius > 0:
                    pygame.draw.circle(self.screen, p['color'], [int(p['pos'][0]), int(p['pos'][1])], radius)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Monster Match")
    
    running = True
    while running:
        action = [0, 0, 0]
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        env.clock.tick(env.FPS)
        
    pygame.quit()