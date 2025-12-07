
# Generated: 2025-08-27T18:02:33.816884
# Source Brief: brief_01710.md
# Brief Index: 1710

        
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
        "Controls: Use arrow keys to move the cursor. Press space to clear a selected group of 2 or more blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Clear adjacent blocks of the same color to score points. Plan your moves to clear the board before you run out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 12
        self.GRID_HEIGHT = 8
        self.BLOCK_SIZE = 40
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2
        self.MAX_MOVES = 25
        self.NUM_COLORS = 5

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_LINES = (40, 50, 60)
        self.BLOCK_COLORS = [
            (255, 87, 87),    # Red
            (87, 187, 255),   # Blue
            (87, 255, 127),   # Green
            (255, 187, 87),   # Orange
            (187, 87, 255),   # Purple
        ]
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_POPUP_GOOD = (255, 215, 0)
        self.COLOR_POPUP_BAD = (255, 50, 50)
        self.COLOR_POPUP_NEUTRAL = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.SysFont("dejavusansmono", 20)
            self.font_large = pygame.font.SysFont("dejavusansmono", 48)
        except pygame.error:
            self.font_main = pygame.font.SysFont("monospace", 22)
            self.font_large = pygame.font.SysFont("monospace", 50)
        
        # --- Game State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.score = 0
        self.moves_remaining = 0
        self.total_steps = 0
        self.game_over = False
        self.win = False
        self.prev_space_held = False
        self.particles = []
        self.text_popups = []
        
        self.reset()
        # self.validate_implementation() # Optional: call for testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._initialize_grid()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.total_steps = 0
        self.game_over = False
        self.win = False
        self.prev_space_held = False
        self.particles = []
        self.text_popups = []
        
        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
            if self._has_valid_moves():
                break

    def step(self, action):
        reward = 0
        self.total_steps += 1

        if self.game_over:
            # If the game is over, subsequent steps do nothing and return the final state
            terminated = True
            return self._get_observation(), 0, terminated, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement)
        
        space_press = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        if space_press:
            clear_reward, cleared_count = self._attempt_clear()
            reward += clear_reward
            if cleared_count == 0 and movement == 0:
                reward -= 0.1 # Penalty for invalid clear attempt with no movement
        elif movement == 0:
            reward -= 0.1 # Penalty for no-op

        terminated = self._check_termination()
        
        if terminated and not self.game_over: # First frame of termination
            self.game_over = True
            if self.win:
                reward += 100
                self._create_text_popup("YOU WIN!", (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2), self.font_large, self.COLOR_POPUP_GOOD, 120)
            else: # Ran out of moves or got stuck
                reward -= 100
                if self.moves_remaining <= 0:
                    msg = "GAME OVER"
                else:
                    msg = "NO MOVES!"
                self._create_text_popup(msg, (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2), self.font_large, self.COLOR_POPUP_BAD, 120)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement):
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

    def _attempt_clear(self):
        cx, cy = self.cursor_pos
        connected_blocks = self._find_connected_blocks(cx, cy)
        
        if len(connected_blocks) < 2:
            # sfx: invalid move sound
            return 0, 0

        self.moves_remaining -= 1
        
        reward = len(connected_blocks)
        self.score += len(connected_blocks)

        cleared_color = self.grid[cx, cy]
        is_color_clear = np.count_nonzero(self.grid == cleared_color) == len(connected_blocks)
        if is_color_clear:
            reward += 10
            self.score += 10
            px, py = self.grid_to_pixel(cx, cy)
            self._create_text_popup("Color Clear!", (px, py - 20), self.font_main, self.COLOR_POPUP_GOOD, 60)
        
        for x, y in connected_blocks:
            px, py = self.grid_to_pixel(x, y)
            color = self.BLOCK_COLORS[self.grid[x, y]-1]
            self._create_particles(px + self.BLOCK_SIZE/2, py + self.BLOCK_SIZE/2, color)
            self.grid[x, y] = 0
            # sfx: block clear sound

        self._apply_gravity()
        
        px, py = self.grid_to_pixel(cx, cy)
        self._create_text_popup(f"+{len(connected_blocks)}", (px, py), self.font_main, self.COLOR_POPUP_NEUTRAL, 45)
        
        return reward, len(connected_blocks)

    def _find_connected_blocks(self, start_x, start_y):
        if self.grid[start_x, start_y] == 0: return []
        
        target_color = self.grid[start_x, start_y]
        q = [(start_x, start_y)]
        visited = set(q)
        
        head = 0
        while head < len(q):
            x, y = q[head]
            head += 1
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[nx, ny] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return q

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            write_ptr = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] != 0:
                    if y != write_ptr:
                        self.grid[x, write_ptr] = self.grid[x, y]
                        self.grid[x, y] = 0
                    write_ptr -= 1

    def _has_valid_moves(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x,y] != 0 and len(self._find_connected_blocks(x, y)) > 1:
                    return True
        return False

    def _check_termination(self):
        if np.all(self.grid == 0):
            self.win = True
            return True
        if self.moves_remaining <= 0:
            return True
        if not self._has_valid_moves():
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_render_effects()
        self._render_grid_lines()
        self._render_blocks()
        if not self.game_over:
            self._render_selection_highlight()
            self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "moves_remaining": self.moves_remaining, "steps": self.total_steps}

    def _update_and_render_effects(self):
        # Particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
            else:
                size = max(1, int(p['lifespan'] / p['max_lifespan'] * 8))
                pygame.draw.circle(self.screen, p['color'], p['pos'], size)

        # Text Popups
        for t in self.text_popups[:]:
            t['pos'][1] -= 0.5
            t['lifespan'] -= 1
            if t['lifespan'] <= 0:
                self.text_popups.remove(t)
            else:
                alpha = max(0, min(255, int(255 * (t['lifespan'] / t['max_lifespan']))))
                text_surf = t['font'].render(t['text'], True, t['color'])
                text_surf.set_alpha(alpha)
                text_rect = text_surf.get_rect(center=t['pos'])
                self.screen.blit(text_surf, text_rect)

    def _render_grid_lines(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_OFFSET_X + x * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.BLOCK_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_OFFSET_Y + y * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.BLOCK_SIZE, py))

    def _render_blocks(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color_idx = self.grid[x, y]
                if color_idx > 0:
                    px, py = self.grid_to_pixel(x, y)
                    color = self.BLOCK_COLORS[color_idx-1]
                    rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    darker_color = tuple(max(0, c - 40) for c in color)
                    pygame.draw.rect(self.screen, darker_color, rect, border_radius=5)
                    inner_rect = pygame.Rect(px+2, py, self.BLOCK_SIZE-4, self.BLOCK_SIZE-4)
                    pygame.draw.rect(self.screen, color, inner_rect, border_radius=5)

    def _render_selection_highlight(self):
        cx, cy = self.cursor_pos
        if self.grid[cx, cy] > 0:
            connected = self._find_connected_blocks(cx, cy)
            if len(connected) > 1:
                pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
                alpha = 50 + pulse * 50
                highlight_surface = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(highlight_surface, (255, 255, 255, int(alpha)), highlight_surface.get_rect(), border_radius=5)
                for x, y in connected:
                    px, py = self.grid_to_pixel(x, y)
                    self.screen.blit(highlight_surface, (px, py))

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        px, py = self.grid_to_pixel(cx, cy)
        rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=5)

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)

    def _create_particles(self, x, y, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(20, 41)
            self.particles.append({
                'pos': [x, y], 'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color, 'lifespan': lifespan, 'max_lifespan': lifespan
            })
            
    def _create_text_popup(self, text, pos, font, color, lifespan):
        self.text_popups.append({
            'text': text, 'pos': list(pos), 'font': font, 'color': color,
            'lifespan': lifespan, 'max_lifespan': lifespan
        })

    def grid_to_pixel(self, grid_x, grid_y):
        return (self.GRID_OFFSET_X + grid_x * self.BLOCK_SIZE, 
                self.GRID_OFFSET_Y + grid_y * self.BLOCK_SIZE)

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
        pygame.font.quit()
        pygame.quit()