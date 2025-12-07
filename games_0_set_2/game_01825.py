
# Generated: 2025-08-28T02:50:09.650295
# Source Brief: brief_01825.md
# Brief Index: 1825

        
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
        "Controls: Use arrow keys to move the selector. Press space to select a gem group. Match 3 or more to score."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A colorful gem-matching puzzle game. Select groups of 3 or more adjacent gems to collect them. Reach 50 gems in 20 moves to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.GEM_SIZE = 40
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_COLS * self.GEM_SIZE) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_ROWS * self.GEM_SIZE) // 2
        self.NUM_GEM_TYPES = 6
        self.GEM_GOAL = 50
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_BG = (30, 45, 60)
        self.COLOR_GRID_LINE = (50, 65, 80)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 150, 50),  # Orange
        ]
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_PROGRESS_BAR = (80, 255, 80)
        self.COLOR_PROGRESS_BAR_BG = (60, 60, 60)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.font_title = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.win = None
        self.steps = None
        self.particles = []
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False
        self.steps = 0
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.particles = []

        self._initialize_grid()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        """Creates a new grid, ensuring no initial matches and at least one possible move."""
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))
        while True:
            self._remove_all_matches_on_board()
            if self._has_possible_moves():
                break
            self._reshuffle_board()

    def _remove_all_matches_on_board(self):
        """Finds and removes all matches on the board, refilling it until stable."""
        while True:
            matches = self._find_all_board_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = -1
            self._apply_gravity_and_refill()

    def _find_all_board_matches(self):
        """Scans the entire board for any horizontal or vertical matches of 3+."""
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                gem_type = self.grid[r, c]
                if gem_type == -1: continue
                if c < self.GRID_COLS - 2 and self.grid[r, c+1] == gem_type and self.grid[r, c+2] == gem_type:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                if r < self.GRID_ROWS - 2 and self.grid[r+1, c] == gem_type and self.grid[r+2, c] == gem_type:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _has_possible_moves(self):
        """Checks if there's at least one group of 3+ that can be selected."""
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if len(self._find_connected_gems(r, c)) >= 3:
                    return True
        return False

    def _reshuffle_board(self):
        """Flattens, shuffles, and rebuilds the board."""
        flat_gems = self.grid.flatten().tolist()
        self.np_random.shuffle(flat_gems)
        self.grid = np.array(flat_gems).reshape((self.GRID_ROWS, self.GRID_COLS))
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False

        movement, space_pressed, _ = action
        space_pressed = space_pressed == 1

        # Handle Cursor Movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        # Handle Gem Selection
        if space_pressed:
            self.moves_left -= 1
            cursor_c, cursor_r = self.cursor_pos
            connected_gems = self._find_connected_gems(cursor_r, cursor_c)
            
            if len(connected_gems) >= 3:
                num_matched = len(connected_gems)
                self.score += num_matched
                reward += num_matched
                if num_matched > 3: reward += 5
                
                gem_type = self.grid[cursor_r, cursor_c]
                for r, c in connected_gems:
                    self.grid[r, c] = -1
                    self._create_particles(c, r, self.GEM_COLORS[gem_type])
                
                # sfx: play_sound("match_success")
                self._apply_gravity_and_refill()
                self._remove_all_matches_on_board()
                
                if not self._has_possible_moves():
                    self._reshuffle_board()
                    # sfx: play_sound("reshuffle")
            else:
                reward = -0.1
                # sfx: play_sound("match_fail")
        
        self._update_particles()
        
        # Check Termination Conditions
        if self.score >= self.GEM_GOAL:
            reward += 100
            terminated = True
            self.game_over = True
            self.win = True
        elif self.moves_left <= 0 or self.steps >= self.MAX_STEPS:
            if not self.game_over: reward += -10
            terminated = True
            self.game_over = True
            self.win = False

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _find_connected_gems(self, start_r, start_c):
        """Finds all connected gems of the same type using BFS."""
        if not (0 <= start_r < self.GRID_ROWS and 0 <= start_c < self.GRID_COLS): return []
        target_type = self.grid[start_r, start_c]
        if target_type == -1: return []
            
        q = [(start_r, start_c)]
        visited, connected = set(q), []
        
        while q:
            r, c = q.pop(0)
            connected.append((r, c))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and
                        (nr, nc) not in visited and self.grid[nr, nc] == target_type):
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return connected

    def _apply_gravity_and_refill(self):
        """Shifts gems down and fills empty top spaces with new gems."""
        for c in range(self.GRID_COLS):
            write_idx = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != -1:
                    self.grid[write_idx, c], self.grid[r, c] = self.grid[r, c], self.grid[write_idx, c]
                    write_idx -= 1
            for r in range(write_idx + 1):
                self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _create_particles(self, c, r, color):
        """Create a burst of particles at a gem's location."""
        px = self.GRID_X_OFFSET + c * self.GEM_SIZE + self.GEM_SIZE // 2
        py = self.GRID_Y_OFFSET + r * self.GEM_SIZE + self.GEM_SIZE // 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append([px, py, vx, vy, life, color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2]; p[1] += p[3]; p[4] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_COLS * self.GEM_SIZE, self.GRID_ROWS * self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=10)

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    gem_color = self.GEM_COLORS[gem_type]
                    rect = pygame.Rect(self.GRID_X_OFFSET + c * self.GEM_SIZE + 3, self.GRID_Y_OFFSET + r * self.GEM_SIZE + 3, self.GEM_SIZE - 6, self.GEM_SIZE - 6)
                    highlight = tuple(min(255, val + 50) for val in gem_color)
                    shadow = tuple(max(0, val - 50) for val in gem_color)
                    pygame.draw.rect(self.screen, shadow, rect, border_radius=8)
                    pygame.draw.rect(self.screen, gem_color, rect.inflate(-4, -4), border_radius=6)
                    pygame.gfxdraw.arc(self.screen, int(rect.left + 8), int(rect.top + 8), 5, 120, 220, highlight)

        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_COLS * self.GEM_SIZE, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_ROWS * self.GEM_SIZE))

        for p in self.particles:
            alpha = max(0, min(255, int(p[4] * (255/30))))
            temp_surf = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p[5], alpha), (2,2), 2)
            self.screen.blit(temp_surf, (int(p[0]-2), int(p[1]-2)))

        if not self.game_over:
            cursor_c, cursor_r = self.cursor_pos
            cursor_rect = pygame.Rect(self.GRID_X_OFFSET + cursor_c * self.GEM_SIZE, self.GRID_Y_OFFSET + cursor_r * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

    def _render_ui(self):
        score_text = self.font_title.render(f"Gems: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        moves_text = self.font_title.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 20))
        
        bar_y, bar_width = self.HEIGHT - 40, self.WIDTH - 40
        progress = min(1.0, self.score / self.GEM_GOAL)
        bg_rect = pygame.Rect(20, bar_y, bar_width, 20)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, bg_rect, border_radius=5)
        
        if progress > 0:
            fill_rect = pygame.Rect(20, bar_y, bar_width * progress, 20)
            pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR, fill_rect, border_radius=5)
            
        progress_text = self.font_small.render(f"{self.score} / {self.GEM_GOAL}", True, self.COLOR_TEXT)
        self.screen.blit(progress_text, progress_text.get_rect(center=bg_rect.center))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            end_text_str = "YOU WIN!" if self.win else "GAME OVER"
            end_text_color = self.COLOR_PROGRESS_BAR if self.win else (255, 80, 80)
            end_text = self.font_large.render(end_text_str, True, end_text_color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left, "win": self.win}

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Puzzle")
    running, total_reward = True, 0

    while running:
        action_to_take = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action_to_take = [1, 0, 0]
                elif event.key == pygame.K_DOWN: action_to_take = [2, 0, 0]
                elif event.key == pygame.K_LEFT: action_to_take = [3, 0, 0]
                elif event.key == pygame.K_RIGHT: action_to_take = [4, 0, 0]
                elif event.key == pygame.K_SPACE: action_to_take = [0, 1, 0]
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                
        if action_to_take:
            obs, reward, terminated, truncated, info = env.step(action_to_take)
            total_reward += reward
            print(f"Action: {action_to_take}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
            if terminated: print(f"Game Over! Final Info: {info}")
        
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        env.clock.tick(30)
        
    env.close()