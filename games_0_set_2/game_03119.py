import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem. "
        "Move the cursor to an adjacent gem and press Space again to swap. "
        "Press Shift to deselect."
    )

    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of 3 or "
        "more of the same type. Score points by clearing gems and creating combos. "
        "Reach 1000 points to win, but the game ends if no more moves are possible."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 8
        self.NUM_GEM_TYPES = 6
        self.WIN_SCORE = 1000
        self.MAX_STEPS = 1000

        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.CELL_SIZE = 40
        self.GRID_AREA_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.GRID_AREA_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_AREA_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_AREA_HEIGHT) // 2

        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 160, 80),  # Orange
        ]

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
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 24)

        # --- Game State Variables ---
        self.board = None
        self.cursor_pos = None
        self.selected_gem = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.possible_moves = 0
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.rng = None

        # This call was causing the timeout due to an infinite loop.
        # The loop is now fixed in _create_initial_board.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_gem = None
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False

        self._create_initial_board()
        self.possible_moves = self._count_possible_moves(self.board)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        # Move cursor
        if movement == 1 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 2 and self.cursor_pos[0] < self.GRID_HEIGHT - 1: self.cursor_pos[0] += 1
        elif movement == 3 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 4 and self.cursor_pos[1] < self.GRID_WIDTH - 1: self.cursor_pos[1] += 1

        # Deselect
        if shift_pressed and self.selected_gem is not None:
            self.selected_gem = None

        # Select / Swap
        if space_pressed:
            if self.selected_gem is None:
                self.selected_gem = tuple(self.cursor_pos)
            else:
                r1, c1 = self.selected_gem
                r2, c2 = self.cursor_pos
                
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    reward += self._attempt_swap((r1, c1), (r2, c2))
                else: 
                    reward -= 0.1
                self.selected_gem = None
        
        self._update_particles()
        
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.possible_moves == 0:
            reward -= 10
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _attempt_swap(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        
        temp_board = np.copy(self.board)
        temp_board[r1, c1], temp_board[r2, c2] = temp_board[r2, c2], temp_board[r1, c1]

        matches = self._find_matches(temp_board)
        
        if not matches:
            return -0.1
        
        self.board = temp_board
        total_reward = 0
        combo = 0
        
        current_matches = matches
        while current_matches:
            combo += 1
            
            num_cleared = len(current_matches)
            total_reward += num_cleared
            if num_cleared == 4: total_reward += 5
            elif num_cleared >= 5: total_reward += 10
            
            self.score += num_cleared * combo

            for r, c in current_matches:
                self._add_particles(r, c, self.board[r,c])
                self.board[r, c] = -1
            
            self._apply_gravity()
            self._refill_board()
            
            current_matches = self._find_matches(self.board)

        self.possible_moves = self._count_possible_moves(self.board)
        return total_reward

    def _create_initial_board(self):
        while True:
            board = self.rng.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            
            while True:
                matches = self._find_matches(board)
                if not matches:
                    break
                
                for r, c in matches:
                    board[r, c] = -1
                
                self._apply_gravity(board)
                self._refill_board(board)
            
            if self._count_possible_moves(board) > 0:
                self.board = board
                break

    def _find_matches(self, board):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if board[r, c] != -1 and board[r, c] == board[r, c+1] == board[r, c+2]:
                    gem_type = board[r, c]
                    i = c
                    while i < self.GRID_WIDTH and board[r, i] == gem_type:
                        matches.add((r, i))
                        i += 1
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if board[r, c] != -1 and board[r, c] == board[r+1, c] == board[r+2, c]:
                    gem_type = board[r, c]
                    i = r
                    while i < self.GRID_HEIGHT and board[i, c] == gem_type:
                        matches.add((i, c))
                        i += 1
        return matches

    def _count_possible_moves(self, board):
        count = 0
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if c < self.GRID_WIDTH - 1:
                    temp_board = np.copy(board)
                    temp_board[r, c], temp_board[r, c+1] = temp_board[r, c+1], temp_board[r, c]
                    if self._find_matches(temp_board):
                        count += 1
                if r < self.GRID_HEIGHT - 1:
                    temp_board = np.copy(board)
                    temp_board[r, c], temp_board[r+1, c] = temp_board[r+1, c], temp_board[r, c]
                    if self._find_matches(temp_board):
                        count += 1
        return count

    def _apply_gravity(self, board=None):
        target_board = board if board is not None else self.board
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if target_board[r, c] != -1:
                    if r != empty_row:
                        target_board[empty_row, c] = target_board[r, c]
                        target_board[r, c] = -1
                    empty_row -= 1
        return target_board

    def _refill_board(self, board=None):
        target_board = board if board is not None else self.board
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if target_board[r, c] == -1:
                    target_board[r, c] = self.rng.integers(0, self.NUM_GEM_TYPES)
        return target_board

    def _add_particles(self, r, c, gem_type):
        px = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.GEM_COLORS[gem_type]
        for _ in range(15):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.rng.integers(15, 30)
            self.particles.append({'pos': [px, py], 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_AREA_HEIGHT))

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.board[r, c]
                if gem_type != -1:
                    self._draw_gem(r, c, gem_type)
        
        self._draw_cursor_and_selection()

        for p in self.particles:
            color = p['color']
            size = max(1, int(p['lifespan'] / 6))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

    def _draw_gem(self, r, c, gem_type):
        rect = pygame.Rect(
            self.GRID_OFFSET_X + c * self.CELL_SIZE,
            self.GRID_OFFSET_Y + r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        center_x, center_y = rect.center
        color = self.GEM_COLORS[gem_type]
        radius = int(self.CELL_SIZE * 0.38)
        
        darker_color = tuple(max(0, val - 50) for val in color)
        pygame.draw.circle(self.screen, darker_color, rect.center, radius + 2)

        if gem_type == 0:
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        elif gem_type == 1:
            size = int(self.CELL_SIZE * 0.7)
            pygame.draw.rect(self.screen, color, (center_x - size//2, center_y - size//2, size, size))
        elif gem_type == 2:
            size = int(self.CELL_SIZE * 0.4)
            points = [(center_x, center_y - size), (center_x - size, center_y + size), (center_x + size, center_y + size)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 3:
            size = int(self.CELL_SIZE * 0.45)
            points = [(center_x, center_y - size), (center_x + size, center_y), (center_x, center_y + size), (center_x - size, center_y)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 4:
            size = int(self.CELL_SIZE * 0.4)
            points = [(center_x + size, center_y), (center_x + size/2, center_y - size * 0.866), (center_x - size/2, center_y - size * 0.866), (center_x - size, center_y), (center_x - size/2, center_y + size * 0.866), (center_x + size/2, center_y + size * 0.866)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 5:
            size = int(self.CELL_SIZE * 0.45)
            points = []
            for i in range(10):
                angle = i * math.pi / 5
                r_val = size if i % 2 == 0 else size * 0.5
                points.append((center_x + r_val * math.sin(angle), center_y - r_val * math.cos(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_cursor_and_selection(self):
        r, c = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_OFFSET_X + c * self.CELL_SIZE, self.GRID_OFFSET_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        if self.selected_gem is not None:
            r_sel, c_sel = self.selected_gem
            sel_rect = pygame.Rect(self.GRID_OFFSET_X + c_sel * self.CELL_SIZE, self.GRID_OFFSET_Y + r_sel * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pulse = (math.sin(self.steps * 0.3) + 1) / 2
            width = 2 + int(pulse * 2)
            alpha = int(150 + pulse * 105)
            
            glow_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_CURSOR + (alpha,), glow_surf.get_rect(), width, border_radius=4)
            self.screen.blit(glow_surf, sel_rect.topleft)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        moves_text = self.font_main.render(f"MOVES: {self.possible_moves}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_main.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "possible_moves": self.possible_moves,
            "cursor_pos": list(self.cursor_pos),
            "selected_gem": self.selected_gem,
        }
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    terminated, truncated = False, False
    
    pygame.display.set_caption("Gem Swap")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        movement = 0
        space_held = False
        shift_held = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        should_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                should_step = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated, truncated = False, False
                    should_step = False
        
        if should_step and not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Term: {terminated}, Trunc: {truncated}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    env.close()