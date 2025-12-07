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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a group of 3 or more adjacent fruits of the same color."
    )

    game_description = (
        "Match cascading fruits in a frantic race against time to reach a target score. Chain reactions from cascades award bonus points."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game Constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 10
        self.CELL_SIZE = 38
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_COLS * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_ROWS * self.CELL_SIZE) // 2 + 30

        self.FRUIT_TYPES = {1: "Apple", 2: "Lime", 3: "Blueberry", 4: "Lemon", 5: "Grape"}
        self.COLORS = {
            "BG": (20, 30, 40),
            "GRID_BG": (35, 45, 55),
            "GRID_LINE": (60, 70, 80),
            "CURSOR": (255, 255, 0),
            "TEXT": (220, 220, 230),
            "SCORE_POPUP": (255, 255, 100),
            1: (220, 50, 50),   # Red
            2: (50, 220, 50),   # Green
            3: (80, 80, 230),   # Blue
            4: (230, 230, 50),  # Yellow
            5: (150, 50, 200),  # Purple
        }

        self.FPS = 30
        self.GAME_DURATION_SECONDS = 120
        self.TARGET_SCORE = 1000
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        self.MOVE_COOLDOWN_FRAMES = 4

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 42, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 20, bold=True)

        # State variables initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.steps = None
        self.time_remaining = None
        self.game_over = None
        self.last_space_held = None
        self.cascade_multiplier = None
        self.reward_this_step = None
        self.game_state = None
        self.animations = None
        self.particles = None
        self.move_cooldown = None
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._fill_board()
        while not self._check_for_possible_moves():
            self._fill_board()

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.score = 0
        self.steps = 0
        self.time_remaining = float(self.GAME_DURATION_SECONDS)
        self.game_over = False
        self.last_space_held = False
        self.cascade_multiplier = 1.0
        self.game_state = "IDLE"
        self.animations = []
        self.particles = []
        self.move_cooldown = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        self.steps += 1
        if not self.game_over:
            self.time_remaining -= 1.0 / self.FPS

        self._handle_input(action)
        self._update_animations_and_particles()

        reward = self._calculate_reward()
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        if self.game_state != "IDLE" or self.game_over:
            return

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        elif movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
            new_x = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_COLS - 1)
            new_y = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_ROWS - 1)
            if self.cursor_pos != [new_x, new_y]:
                self.cursor_pos = [new_x, new_y]
                self.move_cooldown = self.MOVE_COOLDOWN_FRAMES

        if space_held and not self.last_space_held:
            self._handle_selection()
        self.last_space_held = space_held

    def _handle_selection(self):
        x, y = self.cursor_pos
        fruit_type = self.grid[y, x]
        if fruit_type == 0:
            self.reward_this_step -= 0.1
            return

        group = self._find_adjacent_group(x, y)
        if len(group) >= 3:
            self.cascade_multiplier = 1.0
            self._process_match(group)
            self._process_cascades()
        else:
            self.reward_this_step -= 0.1

    def _process_match(self, group):
        num_matched = len(group)
        score_gain = int(10 * num_matched * self.cascade_multiplier)
        self.score += score_gain
        self.reward_this_step += num_matched
        if self.cascade_multiplier > 1.0:
            self.reward_this_step += 5

        fruit_type = self.grid[group[0]]
        avg_r = int(sum(r for r, c in group) / len(group))
        avg_c = int(sum(c for r, c in group) / len(group))
        self._create_score_popup(avg_c, avg_r, f"+{score_gain}")

        for r, c in group:
            self.grid[r, c] = 0
            self._create_particles(c, r, self.COLORS[fruit_type])

    def _process_cascades(self):
        self.game_state = "ANIMATING"
        made_change = True
        while made_change:
            made_change = False
            # 1. Fruits fall down
            for c in range(self.GRID_COLS):
                empty_row = -1
                for r in range(self.GRID_ROWS - 1, -1, -1):
                    if self.grid[r, c] == 0 and empty_row == -1:
                        empty_row = r
                    elif self.grid[r, c] != 0 and empty_row != -1:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                        empty_row -= 1
                        made_change = True

            # 2. Refill from top
            for c in range(self.GRID_COLS):
                for r in range(self.GRID_ROWS):
                    if self.grid[r, c] == 0:
                        self.grid[r, c] = self.np_random.integers(1, len(self.FRUIT_TYPES) + 1)
                        made_change = True

            # 3. Check for new matches
            all_matches = self._find_all_matches()
            if all_matches:
                self.cascade_multiplier += 0.5
                for group in all_matches:
                    self._process_match(group)
                made_change = True
            
        self.game_state = "IDLE"
        if not self._check_for_possible_moves():
            self._reshuffle_board()

    def _find_adjacent_group(self, start_x, start_y):
        fruit_type = self.grid[start_y, start_x]
        if fruit_type == 0: return []
        q, visited, group = [(start_y, start_x)], set([(start_y, start_x)]), []
        while q:
            r, c = q.pop(0)
            group.append((r, c))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and (nr, nc) not in visited and self.grid[nr, nc] == fruit_type:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return group

    def _find_all_matches(self):
        all_matches, visited = [], set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if (r, c) not in visited and self.grid[r, c] != 0:
                    group = self._find_adjacent_group(c, r)
                    if len(group) >= 3:
                        all_matches.append(group)
                        visited.update(group)
        return all_matches

    def _fill_board(self):
        self.grid = self.np_random.integers(1, len(self.FRUIT_TYPES) + 1, size=(self.GRID_ROWS, self.GRID_COLS))

    def _check_for_possible_moves(self):
        return bool(self._find_all_matches())

    def _reshuffle_board(self):
        self._fill_board()
        while not self._check_for_possible_moves():
            self._fill_board()
        self.animations.append({"type": "reshuffle_text", "duration": 60, "progress": 0})

    def _update_animations_and_particles(self):
        self.animations = [a for a in self.animations if a.get('progress', 0) < a.get('duration', 60)]
        for anim in self.animations: anim['progress'] += 1
        
        next_particles = []
        for p in self.particles:
            p['x'] += p['vx']; p['y'] += p['vy']; p['vy'] += 0.1; p['life'] -= 1
            if p['life'] > 0: next_particles.append(p)
        self.particles = next_particles

    def _create_particles(self, grid_c, grid_r, color):
        cx = self.GRID_OFFSET_X + grid_c * self.CELL_SIZE + self.CELL_SIZE / 2
        cy = self.GRID_OFFSET_Y + grid_r * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(12):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({'x': cx, 'y': cy, 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed, 'life': random.randint(20, 40), 'color': color})

    def _create_score_popup(self, grid_c, grid_r, text):
        x = self.GRID_OFFSET_X + grid_c * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + grid_r * self.CELL_SIZE
        self.animations.append({"type": "score_popup", "text": text, "pos": [x, y], "duration": 45, "progress": 0})

    def _calculate_reward(self):
        reward = self.reward_this_step
        if self.score >= self.TARGET_SCORE and not self.game_over:
            reward += 100
        if self.time_remaining <= 0 and not self.game_over:
            reward -= 100
        return reward

    def _check_termination(self):
        return self.score >= self.TARGET_SCORE or self.time_remaining <= 0

    def _get_observation(self):
        self.screen.fill(self.COLORS["BG"])
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_remaining": self.time_remaining}

    def _render_game(self):
        self._draw_grid_bg()
        self._draw_fruits()
        if not self.game_over: self._draw_cursor()
        self._draw_particles()
        self._draw_animations()

    def _draw_grid_bg(self):
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_COLS * self.CELL_SIZE, self.GRID_ROWS * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLORS["GRID_BG"], grid_rect, border_radius=8)
        for r in range(1, self.GRID_ROWS):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLORS["GRID_LINE"], (grid_rect.left, y), (grid_rect.right, y))
        for c in range(1, self.GRID_COLS):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLORS["GRID_LINE"], (x, grid_rect.top), (x, grid_rect.bottom))
        pygame.draw.rect(self.screen, self.COLORS["GRID_LINE"], grid_rect, 2, border_radius=8)

    def _draw_fruits(self):
        radius = int(self.CELL_SIZE * 0.42)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                fruit_type = self.grid[r, c]
                if fruit_type != 0:
                    cx = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                    cy = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                    color = self.COLORS[fruit_type]
                    dark_color = tuple(max(0, val - 50) for val in color)
                    pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, dark_color)
                    pygame.gfxdraw.filled_circle(self.screen, cx - radius // 3, cy - radius // 3, radius // 4, (255, 255, 255, 120))

    def _draw_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLORS["CURSOR"], rect, 4, border_radius=6)

    def _draw_particles(self):
        for p in self.particles:
            size = max(1, int(p['life'] / 8))
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), size)

    def _draw_animations(self):
        for anim in self.animations:
            if anim['type'] == 'score_popup':
                ratio = anim['progress'] / anim['duration']
                y_offset = -int(30 * math.sin(ratio * math.pi))
                alpha = int(255 * (1 - ratio**2))
                text_surf = self.font_medium.render(anim['text'], True, self.COLORS["SCORE_POPUP"])
                text_surf.set_alpha(max(0, alpha))
                pos = (anim['pos'][0], anim['pos'][1] + y_offset)
                self.screen.blit(text_surf, pos)
            elif anim['type'] == 'reshuffle_text':
                ratio = anim['progress'] / anim['duration']
                alpha = int(255 * math.sin(ratio * math.pi))
                text_surf = self.font_large.render("Reshuffle!", True, self.COLORS["TEXT"])
                text_surf.set_alpha(max(0, alpha))
                text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
                self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        score_text = self.font_large.render(f"{self.score}", True, self.COLORS["TEXT"])
        score_label = self.font_small.render("SCORE", True, self.COLORS["TEXT"])
        self.screen.blit(score_label, (20, 5))
        self.screen.blit(score_text, (20, 25))

        time_str = f"{max(0, int(self.time_remaining))}"
        timer_text = self.font_large.render(time_str, True, self.COLORS["TEXT"])
        timer_label = self.font_small.render("TIME", True, self.COLORS["TEXT"])
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 25))
        label_rect = timer_label.get_rect(topright=(self.SCREEN_WIDTH - 20, 5))
        self.screen.blit(timer_label, label_rect)
        self.screen.blit(timer_text, timer_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 190))
            self.screen.blit(overlay, (0, 0))
            win = self.score >= self.TARGET_SCORE
            msg = "YOU WIN!" if win else "TIME'S UP!"
            color = (100, 255, 150) if win else (255, 100, 100)
            msg_text = self.font_large.render(msg, True, color)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(msg_text, msg_rect)
            final_text = self.font_medium.render(f"Final Score: {self.score}", True, self.COLORS["TEXT"])
            final_rect = final_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 30))
            self.screen.blit(final_text, final_rect)

    def close(self):
        pygame.quit()