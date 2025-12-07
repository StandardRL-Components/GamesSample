import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to select a number. Match two identical numbers to clear them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Find and match pairs of numbers on the grid to clear the board before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 5
    MAX_STEPS = 200

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (50, 60, 70)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTED = (0, 255, 255)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TIMER_BAR_BG = (40, 50, 60)
    COLOR_TIMER_GOOD = (0, 200, 100)
    COLOR_TIMER_BAD = (220, 50, 50)
    
    NUMBER_COLORS = [
        (100, 150, 255),  # 1: Blue
        (100, 255, 150),  # 2: Green
        (255, 100, 100),  # 3: Red
        (255, 255, 100),  # 4: Yellow
        (200, 100, 255),  # 5: Purple
        (255, 150, 100),  # 6: Orange
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
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24)

        # Game state variables will be initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.effects = []

        # Grid rendering properties
        self.grid_area_width = self.SCREEN_HEIGHT - 40
        self.cell_size = self.grid_area_width // self.GRID_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_area_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_area_width) // 2

        # The initial reset call is omitted here because the user of the class
        # is expected to call it.
        
    def _create_grid(self):
        # Create 6 sets of 4 identical numbers (12 pairs total)
        numbers = [i for i in range(1, 7) for _ in range(4)]
        self.np_random.shuffle(numbers)
        
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        # Fill the grid, leaving one cell (bottom-right) empty
        idx = 0
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if idx < len(numbers):
                    self.grid[r, c] = numbers[idx]
                    idx += 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._create_grid()
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.effects = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        
        reward = self._handle_movement(movement)
        
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            reward += self._handle_selection()
        self.prev_space_held = bool(space_held)

        self.steps += 1
        
        self._update_effects()

        terminated = self._check_termination()
        truncated = False # No truncation condition other than termination
        if terminated and not self.game_over:
            self.game_over = True
            if self._is_board_clear():
                reward += 100
                self.score += 100
            else:
                reward += -50
                self.score -= 50

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)
        return 0

    def _handle_selection(self):
        x, y = self.cursor_pos
        num_at_cursor = self.grid[y, x]

        if num_at_cursor == 0:
            self.selected_pos = None
            return -0.1

        if self.selected_pos is None:
            self.selected_pos = (x, y)
            return 1.0
        
        if self.selected_pos == (x, y):
            self.selected_pos = None
            return 0.0

        sel_x, sel_y = self.selected_pos
        num_at_selected = self.grid[sel_y, sel_x]

        if num_at_cursor == num_at_selected:
            self.grid[y, x] = 0
            self.grid[sel_y, sel_x] = 0
            self._create_match_effect((x, y))
            self._create_match_effect((sel_x, sel_y))
            self.selected_pos = None
            self.score += 10
            return 10.0
        else:
            self._create_mismatch_effect((x, y))
            self._create_mismatch_effect(self.selected_pos)
            self.selected_pos = None
            return -1.0

    def _check_termination(self):
        return self._is_board_clear() or self.steps >= self.MAX_STEPS

    def _is_board_clear(self):
        return np.sum(self.grid) == 0

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        self._render_grid_lines()
        self._render_highlights()
        self._render_numbers()
        self._render_effects()

    def _render_grid_lines(self):
        for i in range(self.GRID_SIZE + 1):
            start_pos_v = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y)
            end_pos_v = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y + self.grid_area_width)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos_v, end_pos_v, 2)
            
            start_pos_h = (self.grid_offset_x, self.grid_offset_y + i * self.cell_size)
            end_pos_h = (self.grid_offset_x + self.grid_area_width, self.grid_offset_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos_h, end_pos_h, 2)

    def _cell_to_pixel(self, grid_pos):
        x = self.grid_offset_x + grid_pos[0] * self.cell_size
        y = self.grid_offset_y + grid_pos[1] * self.cell_size
        return x, y

    def _render_highlights(self):
        if self.selected_pos is not None:
            px, py = self._cell_to_pixel(self.selected_pos)
            alpha = 128 + int(127 * math.sin(self.steps * 0.3))
            s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_SELECTED, alpha), s.get_rect(), border_radius=8)
            self.screen.blit(s, (px, py))

        cx, cy = self._cell_to_pixel(self.cursor_pos)
        rect = pygame.Rect(cx, cy, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 5, border_radius=8)

    def _render_numbers(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                num = self.grid[r, c]
                if num > 0:
                    color = self.NUMBER_COLORS[(num - 1) % len(self.NUMBER_COLORS)]
                    text_surf = self.font_large.render(str(num), True, color)
                    px, py = self._cell_to_pixel((c, r))
                    text_rect = text_surf.get_rect(center=(px + self.cell_size // 2, py + self.cell_size // 2))
                    self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        timer_ratio = max(0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        
        timer_color = (
            int(self.COLOR_TIMER_BAD[0] * (1 - timer_ratio) + self.COLOR_TIMER_GOOD[0] * timer_ratio),
            int(self.COLOR_TIMER_BAD[1] * (1 - timer_ratio) + self.COLOR_TIMER_GOOD[1] * timer_ratio),
            int(self.COLOR_TIMER_BAD[2] * (1 - timer_ratio) + self.COLOR_TIMER_GOOD[2] * timer_ratio),
        )

        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, timer_color, (bar_x, bar_y, int(bar_width * timer_ratio), bar_height), border_radius=5)

    def _create_match_effect(self, grid_pos):
        px, py = self._cell_to_pixel(grid_pos)
        center_x = px + self.cell_size // 2
        center_y = py + self.cell_size // 2
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.effects.append({
                'type': 'particle',
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': (255, 255, 255)
            })

    def _create_mismatch_effect(self, grid_pos):
        px, py = self._cell_to_pixel(grid_pos)
        self.effects.append({
            'type': 'flash',
            'pos': (px, py),
            'life': 10,
            'color': (255, 50, 50)
        })

    def _update_effects(self):
        for effect in self.effects[:]:
            effect['life'] -= 1
            if effect['life'] <= 0:
                self.effects.remove(effect)
            elif effect['type'] == 'particle':
                effect['pos'][0] += effect['vel'][0]
                effect['pos'][1] += effect['vel'][1]
    
    def _render_effects(self):
        for effect in self.effects:
            if effect['type'] == 'particle':
                pos = (int(effect['pos'][0]), int(effect['pos'][1]))
                radius = int(effect['life'] * 0.2)
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, effect['color'])
            elif effect['type'] == 'flash':
                alpha = int(200 * (effect['life'] / 10))
                s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                pygame.draw.rect(s, (*effect['color'], alpha), s.get_rect(), border_radius=8)
                self.screen.blit(s, effect['pos'])

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and visualization
    # It will not be run by the evaluation server.
    # Set the video driver to a real one
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Number Match Puzzle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        movement = 0
        space_held = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = True
            
        action = np.array([movement, 1 if space_held else 0, 0])
        
        obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    env.close()
    pygame.quit()
    print(f"Game Over! Final Score: {info['score']}")