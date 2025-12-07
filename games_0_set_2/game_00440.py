
# Generated: 2025-08-27T13:40:39.041471
# Source Brief: brief_00440.md
# Brief Index: 440

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a fruit cluster."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match cascading fruits in a grid-based puzzle to reach a target score before running out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 8
        self.CELL_SIZE = 40
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) - 10
        self.NUM_FRUIT_TYPES = 5
        self.MIN_CLUSTER_SIZE = 3
        self.TARGET_SCORE = 1000
        self.INITIAL_MOVES = 30

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_LINES = (40, 60, 80)
        self.FRUIT_COLORS = [
            (255, 80, 80),   # 1: Cherry Red
            (80, 200, 255),  # 2: Sky Blue
            (100, 255, 100), # 3: Lime Green
            (255, 220, 50),  # 4: Sunny Yellow
            (200, 80, 255),  # 5: Royal Purple
        ]
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_HIGHLIGHT = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_WARN = (255, 100, 100)
        self.COLOR_TEXT_WIN = (100, 255, 100)

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_popup = pygame.font.SysFont("Consolas", 18, bold=True)
        
        # Etc...        
        self.grid = []
        self.cursor_pos = [0, 0]
        self.score = 0
        self.steps = 0
        self.moves_remaining = 0
        self.game_over = False
        self.prev_space_held = False
        self.particles = []
        self.score_popups = []
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.score = 0
        self.steps = 0
        self.moves_remaining = self.INITIAL_MOVES
        self.game_over = False
        self.prev_space_held = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.particles.clear()
        self.score_popups.clear()

        self._create_and_stabilize_grid()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        space_press = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        reward = 0
        
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        # 2. Handle match attempt on space press
        if space_press:
            self.moves_remaining -= 1
            cluster = self._find_cluster_at(self.cursor_pos[0], self.cursor_pos[1])

            if len(cluster) >= self.MIN_CLUSTER_SIZE:
                # Sound: Match success
                total_reward, total_score = self._process_match_and_cascades(cluster)
                reward += total_reward
                self.score += total_score
                
                if not self.game_over and not self._is_move_possible():
                    # Sound: Reshuffle
                    self._create_and_stabilize_grid()
            else:
                # Sound: Invalid move
                reward -= 0.1

        self.steps += 1
        
        # 3. Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.score >= self.TARGET_SCORE:
                reward += 100
            else:
                reward -= 50
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= self.TARGET_SCORE:
            self.score = self.TARGET_SCORE # Cap score
            self.game_over = True
            return True
        if self.moves_remaining <= 0:
            self.game_over = True
            return True
        return False

    def _create_and_stabilize_grid(self):
        while True:
            self.grid = [
                [self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1) for _ in range(self.GRID_WIDTH)]
                for _ in range(self.GRID_HEIGHT)
            ]
            while self._find_and_clear_all_matches(is_initial_setup=True):
                self._cascade_fruits()
                self._refill_grid()
            if self._is_move_possible():
                break
    
    def _find_cluster_at(self, start_x, start_y):
        if not (0 <= start_x < self.GRID_WIDTH and 0 <= start_y < self.GRID_HEIGHT):
            return set()
        fruit_type = self.grid[start_y][start_x]
        if fruit_type == 0:
            return set()
        q = [(start_x, start_y)]
        visited, cluster = set(q), set(q)
        while q:
            x, y = q.pop(0)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and
                        (nx, ny) not in visited and self.grid[ny][nx] == fruit_type):
                    visited.add((nx, ny))
                    cluster.add((nx, ny))
                    q.append((nx, ny))
        return cluster

    def _process_match_and_cascades(self, initial_cluster):
        total_reward, total_score, combo = 0, 0, 1
        clusters_to_clear = [initial_cluster]
        while clusters_to_clear:
            for cluster in clusters_to_clear:
                score, rew = self._clear_cluster(cluster, combo)
                total_score += score
                total_reward += rew
            self._cascade_fruits()
            self._refill_grid()
            combo += 1
            clusters_to_clear = self._find_all_matches()
        return total_reward, total_score

    def _clear_cluster(self, cluster, combo_multiplier):
        if not cluster: return 0, 0
        num_cleared = len(cluster)
        fruit_type = self.grid[list(cluster)[0][1]][list(cluster)[0][0]]
        reward = num_cleared
        score = num_cleared * 10 * combo_multiplier
        if num_cleared > self.MIN_CLUSTER_SIZE:
            bonus = 5 * (num_cleared - self.MIN_CLUSTER_SIZE)
            reward += bonus
            score += bonus * 10 * combo_multiplier
        if combo_multiplier > 1:
            reward += 5 * combo_multiplier
        px = sum(c[0] for c in cluster) / num_cleared
        py = sum(c[1] for c in cluster) / num_cleared
        rx, ry = self.GRID_OFFSET_X + px * self.CELL_SIZE, self.GRID_OFFSET_Y + py * self.CELL_SIZE
        self.score_popups.append([f"+{score}", rx, ry, 30, self.FRUIT_COLORS[fruit_type - 1]])
        for x, y in cluster:
            self.grid[y][x] = 0
            for _ in range(5):
                self.particles.append([
                    self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE / 2,
                    self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE / 2,
                    (self.np_random.random() - 0.5) * 4, (self.np_random.random() - 0.5) * 4,
                    self.np_random.integers(20, 40), self.FRUIT_COLORS[fruit_type - 1]
                ])
        return score, reward

    def _find_all_matches(self):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.grid[y][x] != 0 and self.grid[y][x] == self.grid[y][x+1] == self.grid[y][x+2]:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.grid[y][x] != 0 and self.grid[y][x] == self.grid[y+1][x] == self.grid[y+2][x]:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        if not matches: return []
        all_clusters, visited = [], set()
        for x, y in matches:
            if (x, y) not in visited:
                cluster = self._find_cluster_at(x, y)
                all_clusters.append(cluster)
                visited.update(cluster)
        return all_clusters

    def _find_and_clear_all_matches(self, is_initial_setup=False):
        all_match_clusters = self._find_all_matches()
        if not all_match_clusters: return False
        for cluster in all_match_clusters:
            for x, y in cluster: self.grid[y][x] = 0
        return True

    def _cascade_fruits(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] != 0:
                    if y != empty_row:
                        self.grid[empty_row][x], self.grid[y][x] = self.grid[y][x], 0
                    empty_row -= 1

    def _refill_grid(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] == 0:
                    self.grid[y][x] = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1)
    
    def _is_move_possible(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if len(self._find_cluster_at(x, y)) >= self.MIN_CLUSTER_SIZE:
                    return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_remaining": self.moves_remaining}

    def _render_game(self):
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, y))

        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2]; p[1] += p[3]; p[4] -= 1
            alpha = max(0, min(255, int(255 * (p[4] / 30)))); color = p[5] + (alpha,)
            radius = int(self.CELL_SIZE / 4 * (p[4] / 30))
            if radius > 0: pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), radius, color)

        radius = self.CELL_SIZE // 2 - 4
        for y, row in enumerate(self.grid):
            for x, fruit_type in enumerate(row):
                if fruit_type > 0:
                    cx, cy = self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE // 2, self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
                    color = self.FRUIT_COLORS[fruit_type - 1]
                    pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, color)
                    highlight_color = tuple(min(255, c + 60) for c in color)
                    pygame.gfxdraw.filled_circle(self.screen, cx - radius // 3, cy - radius // 3, radius // 3, highlight_color)

        if not self.game_over:
            cx, cy = self.cursor_pos
            cursor_rect = pygame.Rect(self.GRID_OFFSET_X + cx * self.CELL_SIZE, self.GRID_OFFSET_Y + cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA); s.fill(self.COLOR_CURSOR); self.screen.blit(s, cursor_rect.topleft)
            cluster = self._find_cluster_at(cx, cy)
            if len(cluster) >= self.MIN_CLUSTER_SIZE:
                for x, y in cluster:
                    rect = pygame.Rect(self.GRID_OFFSET_X + x * self.CELL_SIZE + 2, self.GRID_OFFSET_Y + y * self.CELL_SIZE + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
                    pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, rect, 3, border_radius=8)

        self.score_popups = [p for p in self.score_popups if p[3] > 0]
        for p in self.score_popups:
            p[1] += (self.np_random.random() - 0.5) * 0.5; p[2] -= 0.5; p[3] -= 1
            alpha = max(0, min(255, int(255 * (p[3] / 30)))); text_surf = self.font_popup.render(p[0], True, p[4]); text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (int(p[1] - text_surf.get_width() / 2), int(p[2])))

    def _render_ui(self):
        pygame.draw.rect(self.screen, (10, 20, 30), (0, 0, self.WIDTH, 40))
        pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (0, 40), (self.WIDTH, 40))
        score_text = self.font_ui.render(f"Score: {self.score}/{self.TARGET_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        moves_color = self.COLOR_TEXT if self.moves_remaining > 5 else self.COLOR_TEXT_WARN
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, moves_color)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 5))
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180)); self.screen.blit(overlay, (0, 0))
            end_text = self.font_game_over.render("YOU WIN!" if self.score >= self.TARGET_SCORE else "GAME OVER", True, self.COLOR_TEXT_WIN if self.score >= self.TARGET_SCORE else self.COLOR_TEXT_WARN)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2)))

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(reward, (int, float))
        assert isinstance(term, bool) and trunc is False and isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Matcher")
    clock = pygame.time.Clock()
    
    while not terminated:
        mov, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [mov, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)
        
    print("Game Over!")
    pygame.quit()