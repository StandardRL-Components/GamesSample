
# Generated: 2025-08-28T02:25:40.927497
# Source Brief: brief_04440.md
# Brief Index: 4440

        
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
        "Controls: Arrow keys to move cursor. Space to select a gem. "
        "Move to an adjacent gem and press Space again to swap."
    )

    game_description = (
        "Swap adjacent gems in an isometric grid to match 3 or more. "
        "Collect 20 gems within 50 moves to win!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    NUM_GEM_TYPES = 6
    TILE_W_HALF = 30
    TILE_H_HALF = 15
    GEM_Z_OFFSET = 5

    GOAL_GEMS = 20
    MAX_MOVES = 50
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BAR_BG = (50, 60, 90)
    COLOR_UI_BAR_FG = (100, 180, 255)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 255, 80),   # Yellow
        (255, 80, 255),   # Magenta
        (80, 255, 255),   # Cyan
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)

        self.grid_origin_x = self.SCREEN_WIDTH // 2
        self.grid_origin_y = 120

        # Initialize state variables
        self.grid = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_left = 0
        self.gems_collected = 0
        self.cursor_pos = [0, 0]
        self.selected_gem_pos = None
        self.last_space_held = False
        self.particles = []
        
        self.reset()

        # self.validate_implementation() # Uncomment to run self-check

    def _grid_to_screen(self, r, c):
        x = self.grid_origin_x + (c - r) * self.TILE_W_HALF
        y = self.grid_origin_y + (c + r) * self.TILE_H_HALF
        return int(x), int(y)

    def _generate_board(self):
        while True:
            grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            while self._find_matches(grid):
                matches = self._find_matches(grid)
                for r, c in matches:
                    grid[r, c] = 0
                
                for c in range(self.GRID_WIDTH):
                    empty_count = np.sum(grid[:, c] == 0)
                    if empty_count > 0:
                        non_empty = grid[:, c][grid[:, c] != 0]
                        grid[:, c] = np.concatenate([
                            self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=empty_count),
                            non_empty
                        ])

            if self._find_all_possible_moves(grid):
                return grid

    def _find_all_possible_moves(self, grid):
        moves = []
        temp_grid = grid.copy()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Check swap right
                if c < self.GRID_WIDTH - 1:
                    temp_grid[r, c], temp_grid[r, c + 1] = temp_grid[r, c + 1], temp_grid[r, c]
                    if self._find_matches(temp_grid):
                        moves.append(((r, c), (r, c + 1)))
                    temp_grid[r, c], temp_grid[r, c + 1] = temp_grid[r, c + 1], temp_grid[r, c] # Swap back
                # Check swap down
                if r < self.GRID_HEIGHT - 1:
                    temp_grid[r, c], temp_grid[r + 1, c] = temp_grid[r + 1, c], temp_grid[r, c]
                    if self._find_matches(temp_grid):
                        moves.append(((r, c), (r + 1, c)))
                    temp_grid[r, c], temp_grid[r + 1, c] = temp_grid[r + 1, c], temp_grid[r, c] # Swap back
        return moves

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self._generate_board()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.moves_left = self.MAX_MOVES
        self.gems_collected = 0

        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_gem_pos = None

        self.last_space_held = False
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, _ = action
        reward = 0
        
        self.steps += 1
        
        # --- Handle Input ---
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_HEIGHT - 1, self.cursor_pos[0] + 1)
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_WIDTH - 1, self.cursor_pos[1] + 1)

        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        if space_pressed:
            # sfx: select_gem
            cursor_r, cursor_c = self.cursor_pos
            if self.selected_gem_pos is None:
                self.selected_gem_pos = [cursor_r, cursor_c]
            else:
                sel_r, sel_c = self.selected_gem_pos
                dist = abs(sel_r - cursor_r) + abs(sel_c - cursor_c)
                
                if dist == 1: # Adjacent, attempt swap
                    reward = self._attempt_swap((sel_r, sel_c), (cursor_r, cursor_c))
                    self.selected_gem_pos = None
                elif dist == 0: # Clicked same gem
                    self.selected_gem_pos = None
                else: # Clicked different, non-adjacent gem
                    self.selected_gem_pos = [cursor_r, cursor_c]
        
        self._update_particles()
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.win:
                reward += 100 # Goal-oriented reward for winning
            else:
                reward -= 10 # Penalty for losing
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _attempt_swap(self, pos1, pos2):
        self.moves_left -= 1
        r1, c1 = pos1
        r2, c2 = pos2

        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        # sfx: gem_swap

        matches = self._find_matches(self.grid)
        if not matches:
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            # sfx: invalid_swap
            return -0.1

        step_reward = 0
        chain = 0
        while True:
            matches = self._find_matches(self.grid)
            if not matches:
                break
            
            chain += 1
            num_matched = len(matches)
            
            # sfx: match_3 (or match_4, match_5)
            step_reward += num_matched
            self.gems_collected += num_matched
            self.score += num_matched * chain

            if num_matched >= 4:
                step_reward += 5
                self.score += 10
            
            for r, c in matches:
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    self._spawn_particles(r, c, gem_type)
            
            for r, c in matches:
                self.grid[r, c] = 0

            self._drop_and_fill()

        return step_reward

    def _find_matches(self, grid):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if grid[r, c] != 0 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if grid[r, c] != 0 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches
    
    def _drop_and_fill(self):
        # sfx: gems_fall
        for c in range(self.GRID_WIDTH):
            empty_indices = np.where(self.grid[:, c] == 0)[0]
            if len(empty_indices) > 0:
                non_empty_gems = self.grid[:, c][self.grid[:, c] != 0]
                new_gems = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=len(empty_indices))
                self.grid[:, c] = np.concatenate((new_gems, non_empty_gems))

    def _spawn_particles(self, r, c, gem_type):
        screen_x, screen_y = self._grid_to_screen(r, c)
        color = self.GEM_COLORS[gem_type - 1]
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': [screen_x, screen_y], 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1
            p['life'] -= 1

    def _check_termination(self):
        if self.gems_collected >= self.GOAL_GEMS:
            self.win = True
            return True
        if self.moves_left <= 0:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "moves_left": self.moves_left,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.GRID_HEIGHT + 1):
            start = self._grid_to_screen(r, 0)
            end = self._grid_to_screen(r, self.GRID_WIDTH)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for c in range(self.GRID_WIDTH + 1):
            start = self._grid_to_screen(0, c)
            end = self._grid_to_screen(self.GRID_HEIGHT, c)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    self._draw_gem(r, c, gem_type)
        
        self._draw_cursor_and_selection()

        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

    def _draw_gem(self, r, c, gem_type):
        x, y = self._grid_to_screen(r, c)
        y -= self.GEM_Z_OFFSET
        color = self.GEM_COLORS[gem_type - 1]
        
        top_color = tuple(min(255, val + 40) for val in color)
        side_color = tuple(max(0, val - 40) for val in color)

        points_top = [(x, y - self.TILE_H_HALF), (x + self.TILE_W_HALF, y), (x, y + self.TILE_H_HALF), (x - self.TILE_W_HALF, y)]
        points_left = [(x - self.TILE_W_HALF, y), (x, y + self.TILE_H_HALF), (x, y + self.TILE_H_HALF * 2), (x - self.TILE_W_HALF, y + self.TILE_H_HALF)]
        points_right = [(x + self.TILE_W_HALF, y), (x, y + self.TILE_H_HALF), (x, y + self.TILE_H_HALF * 2), (x + self.TILE_W_HALF, y + self.TILE_H_HALF)]
        
        pygame.gfxdraw.filled_polygon(self.screen, points_left, side_color)
        pygame.gfxdraw.filled_polygon(self.screen, points_right, side_color)
        pygame.gfxdraw.filled_polygon(self.screen, points_top, top_color)
        pygame.gfxdraw.aapolygon(self.screen, points_top, color)

    def _draw_cursor_and_selection(self):
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 155 + 100

        if self.selected_gem_pos:
            r, c = self.selected_gem_pos
            x, y = self._grid_to_screen(r, c)
            y += self.TILE_H_HALF 
            points = [(x, y - self.TILE_H_HALF), (x + self.TILE_W_HALF, y), (x, y + self.TILE_H_HALF), (x - self.TILE_W_HALF, y)]
            temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.draw.polygon(temp_surf, (255, 255, 255, int(pulse)), points, 3)
            self.screen.blit(temp_surf, (0,0))

        r, c = self.cursor_pos
        x, y = self._grid_to_screen(r, c)
        y += self.TILE_H_HALF
        points = [(x, y - self.TILE_H_HALF - 2), (x + self.TILE_W_HALF + 2, y), (x, y + self.TILE_H_HALF + 2), (x - self.TILE_W_HALF - 2, y)]
        pygame.draw.lines(self.screen, (255, 255, 0), True, points, 2)

    def _render_ui(self):
        bar_w, bar_h, bar_x, bar_y = 200, 20, (self.SCREEN_WIDTH - 200) / 2, 20
        moves_ratio = max(0, self.moves_left / self.MAX_MOVES)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FG, (bar_x, bar_y, bar_w * moves_ratio, bar_h))
        moves_text = self.font_small.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (bar_x + bar_w + 10, bar_y))
        
        gems_text = self.font_small.render(f"Gems: {self.gems_collected} / {self.GOAL_GEMS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gems_text, gems_text.get_rect(topleft=(20, 20)))
        
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20)))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text_str = "YOU WIN!" if self.win else "OUT OF MOVES"
        text_color = self.COLOR_WIN if self.win else self.COLOR_LOSE
        text = self.font_large.render(text_str, True, text_color)
        
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)

    def close(self):
        pygame.quit()

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    env.reset()
    
    running = True
    game_over_screen = False

    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                    game_over_screen = False
        
        if not game_over_screen:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            obs, reward, terminated, _, info = env.step(action)
            
            if terminated:
                game_over_screen = True
                print(f"Game Over! Final Info: {info}")

        display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Gem Swap")
        
        surf = pygame.surfarray.make_surface(np.transpose(env._get_observation(), (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    env.close()