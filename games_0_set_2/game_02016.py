
# Generated: 2025-08-27T19:00:17.129313
# Source Brief: brief_02016.md
# Brief Index: 2016

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a group of matching blocks. "
        "Hold Shift to reshuffle the board (costs one move)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match groups of 2 or more same-colored blocks to score points. Reach 5000 points in 30 moves to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 10
    GRID_HEIGHT = 8
    BLOCK_SIZE = 40
    GRID_LINE_WIDTH = 2
    
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    GRID_AREA_WIDTH = GRID_WIDTH * (BLOCK_SIZE + GRID_LINE_WIDTH) - GRID_LINE_WIDTH
    GRID_AREA_HEIGHT = GRID_HEIGHT * (BLOCK_SIZE + GRID_LINE_WIDTH) - GRID_LINE_WIDTH
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) - 20

    COLOR_BG = (20, 25, 40)
    COLOR_GRID_BG = (40, 45, 60)
    COLOR_HIGHLIGHT = (255, 255, 100, 150)

    BLOCK_COLORS = {
        0: (220, 50, 50),   # Red
        1: (50, 220, 50),   # Green
        2: (50, 100, 220),  # Blue
        3: (220, 220, 50),  # Yellow
        4: (150, 50, 220),  # Purple
        5: (100, 100, 100)  # Gray (unmatchable - not used in generation)
    }
    NUM_COLORS = 5
    
    WIN_SCORE = 5000
    STARTING_MOVES = 30
    MAX_STEPS = 1000

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
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.last_action_feedback = ""
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._generate_grid()
        while not self._check_for_possible_moves():
            self.grid = self._generate_grid()

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.score = 0
        self.moves_left = self.STARTING_MOVES
        self.game_over = False
        self.steps = 0
        self.last_action_feedback = ""
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        action_taken = False
        if shift_pressed:
            # sound: "swoosh"
            self._reshuffle_grid()
            self.moves_left -= 1
            action_taken = True
            self.last_action_feedback = "Board Reshuffled!"
            # No direct reward/penalty for reshuffle, it's a strategic cost.
        elif space_pressed:
            action_taken = True
            reward = self._handle_selection()
            self.moves_left -= 1
        
        self._update_particles()
        
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward = 100.0
            self.last_action_feedback = "YOU WIN!"
            terminated = True
            self.game_over = True
        elif self.moves_left <= 0:
            reward = -100.0
            self.last_action_feedback = "OUT OF MOVES!"
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        if action_taken and not self.game_over and not self._check_for_possible_moves():
            self._reshuffle_grid()
            self.last_action_feedback = "No moves! Auto-shuffled."

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_grid(self):
        return [
            [self.np_random.integers(0, self.NUM_COLORS) for _ in range(self.GRID_WIDTH)]
            for _ in range(self.GRID_HEIGHT)
        ]

    def _handle_selection(self):
        cx, cy = self.cursor_pos
        group = self._find_match_group(cx, cy)
        num_cleared = len(group)
        
        if num_cleared <= 1:
            # sound: "fail_buzz"
            self.last_action_feedback = "Invalid Match"
            return -0.1
        
        # sound: "pop"
        base_score = num_cleared * 10
        combo_bonus = (num_cleared - 2) ** 2 * 5
        self.score += base_score + combo_bonus
        self.last_action_feedback = f"+{base_score + combo_bonus} ({num_cleared} blocks)"
        
        # Scaled reward for RL agent
        reward = num_cleared * 0.1
        if num_cleared >= 4: reward += 1.0
        if num_cleared >= 7: reward += 5.0
        
        for x, y in group:
            self._create_particles(x, y, self.grid[y][x])
            self.grid[y][x] = -1
        
        self._apply_gravity()
        self._fill_empty_spaces()
        
        return reward

    def _find_match_group(self, start_x, start_y):
        target_color = self.grid[start_y][start_x]
        q = [(start_x, start_y)]
        visited = set(q)
        group = []
        
        while q:
            x, y = q.pop(0)
            group.append((x, y))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[ny][nx] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] != -1:
                    if y != empty_row:
                        self.grid[empty_row][x] = self.grid[y][x]
                        self.grid[y][x] = -1
                    empty_row -= 1

    def _fill_empty_spaces(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] == -1:
                    self.grid[y][x] = self.np_random.integers(0, self.NUM_COLORS)

    def _reshuffle_grid(self):
        flat_list = [self.grid[y][x] for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH)]
        self.np_random.shuffle(flat_list)
        self.grid = [flat_list[i * self.GRID_WIDTH:(i + 1) * self.GRID_WIDTH] for i in range(self.GRID_HEIGHT)]
        if not self._check_for_possible_moves():
            self._reshuffle_grid()

    def _check_for_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color = self.grid[y][x]
                if x + 1 < self.GRID_WIDTH and self.grid[y][x+1] == color: return True
                if y + 1 < self.GRID_HEIGHT and self.grid[y+1][x] == color: return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_bg_rect = pygame.Rect(
            self.GRID_OFFSET_X - 5, self.GRID_OFFSET_Y - 5,
            self.GRID_AREA_WIDTH + 10, self.GRID_AREA_HEIGHT + 10
        )
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_bg_rect, border_radius=8)

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_id = self.grid[y][x]
                if color_id != -1:
                    self._draw_block(x, y, self.BLOCK_COLORS[color_id])
        
        self._draw_particles()

        cx, cy = self.cursor_pos
        hx = self.GRID_OFFSET_X + cx * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
        hy = self.GRID_OFFSET_Y + cy * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
        
        highlight_surface = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
        highlight_surface.fill(self.COLOR_HIGHLIGHT)
        self.screen.blit(highlight_surface, (hx, hy))
        pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT[:3], (hx, hy, self.BLOCK_SIZE, self.BLOCK_SIZE), 3)

    def _draw_block(self, grid_x, grid_y, color):
        px = self.GRID_OFFSET_X + grid_x * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
        py = self.GRID_OFFSET_Y + grid_y * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
        rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        highlight_color = tuple(min(255, c + 40) for c in color)
        shadow_color = tuple(max(0, c - 40) for c in color)
        
        pygame.draw.line(self.screen, highlight_color, (px + 3, py + 2), (px + self.BLOCK_SIZE - 4, py + 2), 2)
        pygame.draw.line(self.screen, highlight_color, (px + 2, py + 3), (px + 2, py + self.BLOCK_SIZE - 4), 2)
        pygame.draw.line(self.screen, shadow_color, (px + self.BLOCK_SIZE - 3, py + 3), (px + self.BLOCK_SIZE - 3, py + self.BLOCK_SIZE - 3), 2)
        pygame.draw.line(self.screen, shadow_color, (px + 3, py + self.BLOCK_SIZE - 3), (px + self.BLOCK_SIZE - 3, py + self.BLOCK_SIZE - 3), 2)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 15))
        
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, (255, 255, 255))
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(moves_text, moves_rect)
        
        if self.last_action_feedback:
            feedback_text = self.font_small.render(self.last_action_feedback, True, (200, 200, 220))
            feedback_rect = feedback_text.get_rect(center=(self.SCREEN_WIDTH / 2, 35))
            self.screen.blit(feedback_text, feedback_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(status, True, (255, 255, 100))
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(end_text, end_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, (255, 255, 255))
            final_score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)

    def _create_particles(self, grid_x, grid_y, color_id):
        px = self.GRID_OFFSET_X + grid_x * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH) + self.BLOCK_SIZE / 2
        py = self.GRID_OFFSET_Y + grid_y * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH) + self.BLOCK_SIZE / 2
        color = self.BLOCK_COLORS[color_id]
        
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py], 'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(20, 40), 'color': color, 'size': random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1
            p['life'] -= 1
            p['size'] *= 0.97
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0.5]

    def _draw_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['size']))

    def _get_info(self):
        return {"score": self.score, "moves_left": self.moves_left, "steps": self.steps}

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
        
        # print("✓ Implementation validated successfully")