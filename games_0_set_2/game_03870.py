
# Generated: 2025-08-28T00:42:07.908323
# Source Brief: brief_03870.md
# Brief Index: 3870

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem. "
        "With a gem selected, move the cursor to an adjacent tile and press Space again to swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic match-3 puzzle game. Swap adjacent gems to create lines of 3 or more of the same color. "
        "Your goal is to collect 10 gems of the special target color before you run out of 20 moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # Constants
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    GEM_SIZE = 40
    GRID_AREA_WIDTH = GRID_WIDTH * GEM_SIZE
    GRID_AREA_HEIGHT = GRID_HEIGHT * GEM_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) // 2
    MAX_MOVES = 20
    TARGET_GEM_GOAL = 10
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (50, 60, 80)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 120, 255),   # Blue
        (255, 255, 80),   # Yellow
        (200, 80, 255),   # Purple
        (255, 160, 80),   # Orange
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.grid = []
        self.cursor_pos = []
        self.selected_pos = None
        self.moves_left = 0
        self.score = 0
        self.target_color_idx = 0
        self.target_gems_collected = 0
        self.game_over = False
        self.steps = 0
        self.particles = []
        self.last_action_was_swap = False

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_pos = None
        self.target_color_idx = self.np_random.integers(0, len(self.GEM_COLORS))
        self.target_gems_collected = 0
        self.particles = []
        self.last_action_was_swap = False

        self._populate_grid()
        
        return self._get_observation(), self._get_info()

    def _populate_grid(self):
        self.grid = [[0] * self.GRID_WIDTH for _ in range(self.GRID_HEIGHT)]
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                invalid_colors = []
                if x >= 2 and self.grid[y][x-1] == self.grid[y][x-2]:
                    invalid_colors.append(self.grid[y][x-1])
                if y >= 2 and self.grid[y-1][x] == self.grid[y-2][x]:
                    invalid_colors.append(self.grid[y-1][x])
                
                possible_colors = [i for i in range(len(self.GEM_COLORS)) if i not in invalid_colors]
                self.grid[y][x] = self.np_random.choice(possible_colors)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.last_action_was_swap = False

        movement, space_press, _ = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)
            
        # 2. Handle selection and swapping
        if space_press:
            cx, cy = self.cursor_pos
            if self.selected_pos is None:
                self.selected_pos = (cx, cy)
                # sfx: select_gem.wav
            else:
                sx, sy = self.selected_pos
                dist = abs(cx - sx) + abs(cy - sy)
                
                if (cx, cy) == (sx, sy):
                    self.selected_pos = None
                    # sfx: deselect.wav
                elif dist == 1:
                    reward, match_found = self._perform_swap(sx, sy, cx, cy)
                    if match_found:
                        self.moves_left -= 1
                        reward -= 0.1 # Cost of a move
                        self.last_action_was_swap = True
                        # sfx: swap_success.wav
                    else: # sfx: swap_fail.wav
                        pass
                    self.selected_pos = None
                else:
                    self.selected_pos = None
                    # sfx: swap_fail.wav
        
        if self.last_action_was_swap:
            cascade_reward, _ = self._process_cascades()
            reward += cascade_reward

        self.score += reward

        # 4. Check for termination
        terminated = False
        if self.target_gems_collected >= self.TARGET_GEM_GOAL:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.moves_left <= 0:
            reward -= 10
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _perform_swap(self, x1, y1, x2, y2):
        self.grid[y1][x1], self.grid[y2][x2] = self.grid[y2][x2], self.grid[y1][x1]
        matches = self._find_matches()
        if not matches:
            self.grid[y1][x1], self.grid[y2][x2] = self.grid[y2][x2], self.grid[y1][x1]
            return 0, False
        
        reward = self._clear_matches(matches)
        return reward, True

    def _process_cascades(self):
        total_reward, total_cleared = 0, 0
        while True:
            self._drop_gems()
            self._refill_grid()
            matches = self._find_matches()
            if not matches: break
            
            reward = self._clear_matches(matches)
            total_reward += reward
            total_cleared += len(matches)
            # sfx: cascade_match.wav
        return total_reward, total_cleared

    def _clear_matches(self, matches):
        reward = 0
        for x, y in matches:
            color_idx = self.grid[y][x]
            if color_idx is not None:
                reward += 1
                if color_idx == self.target_color_idx: self.target_gems_collected += 1
                self._create_particles(x, y, self.GEM_COLORS[color_idx])
                self.grid[y][x] = None
                # sfx: gem_clear.wav
        return reward

    def _find_matches(self):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.grid[y][x] is not None and self.grid[y][x] == self.grid[y][x+1] == self.grid[y][x+2]:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.grid[y][x] is not None and self.grid[y][x] == self.grid[y+1][x] == self.grid[y+2][x]:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return matches

    def _drop_gems(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] is not None:
                    if y != empty_row:
                        self.grid[empty_row][x], self.grid[y][x] = self.grid[y][x], None
                    empty_row -= 1

    def _refill_grid(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] is None:
                    self.grid[y][x] = self.np_random.integers(0, len(self.GEM_COLORS))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        self._draw_grid_lines()
        self._draw_gems()
        self._draw_cursor()
        self._update_and_draw_particles()

    def _draw_grid_lines(self):
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_X_OFFSET + i * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_AREA_HEIGHT))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y_OFFSET + i * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_AREA_WIDTH, y))

    def _draw_gems(self):
        radius = self.GEM_SIZE // 2 - 4
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.grid[y][x]
                if color_idx is not None:
                    center_x = self.GRID_X_OFFSET + x * self.GEM_SIZE + self.GEM_SIZE // 2
                    center_y = self.GRID_Y_OFFSET + y * self.GEM_SIZE + self.GEM_SIZE // 2
                    color = self.GEM_COLORS[color_idx]
                    
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, tuple(min(255, c+30) for c in color))
                    
                    highlight_pos = (center_x - radius // 2, center_y - radius // 2)
                    pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], radius // 3, (255, 255, 255, 60))

                    if color_idx == self.target_color_idx:
                        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius + 2, self.COLOR_CURSOR)

    def _draw_cursor(self):
        cx, cy = self.cursor_pos
        rect = pygame.Rect(self.GRID_X_OFFSET + cx * self.GEM_SIZE, self.GRID_Y_OFFSET + cy * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=4)
        
        if self.selected_pos is not None:
            sx, sy = self.selected_pos
            rect = pygame.Rect(self.GRID_X_OFFSET + sx * self.GEM_SIZE, self.GRID_Y_OFFSET + sy * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
            pulse = (math.sin(self.steps * 0.3) + 1) / 2
            color = (200 + 55 * pulse, 200 + 55 * pulse, 255)
            pygame.draw.rect(self.screen, color, rect, 3, border_radius=6)

    def _create_particles(self, grid_x, grid_y, color):
        center_x = self.GRID_X_OFFSET + grid_x * self.GEM_SIZE + self.GEM_SIZE // 2
        center_y = self.GRID_Y_OFFSET + grid_y * self.GEM_SIZE + self.GEM_SIZE // 2
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({"pos": [center_x, center_y], "vel": vel, "color": color, "life": self.np_random.integers(15, 30)})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]; p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1
            p["life"] -= 1
            if p["life"] > 0:
                alpha = max(0, min(255, int(255 * (p["life"] / 20))))
                color_with_alpha = p["color"] + (alpha,)
                size = max(1, int(p["life"] / 6))
                
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color_with_alpha, (size, size), size)
                self.screen.blit(particle_surf, (int(p["pos"][0] - size), int(p["pos"][1] - size)))
                active_particles.append(p)
        self.particles = active_particles

    def _render_ui(self):
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        target_color_surf = pygame.Surface((24, 24), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(target_color_surf, 12, 12, 10, self.GEM_COLORS[self.target_color_idx])
        pygame.gfxdraw.aacircle(target_color_surf, 12, 12, 10, self.COLOR_CURSOR)
        self.screen.blit(target_color_surf, (self.SCREEN_WIDTH - 180, 18))
        
        target_text = self.font_main.render(f"Collect: {self.target_gems_collected} / {self.TARGET_GEM_GOAL}", True, self.COLOR_TEXT)
        self.screen.blit(target_text, (self.SCREEN_WIDTH - 150, 20))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.target_gems_collected >= self.TARGET_GEM_GOAL else "GAME OVER"
            end_color = (100, 255, 100) if self.target_gems_collected >= self.TARGET_GEM_GOAL else (255, 100, 100)
            end_text = self.font_main.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(end_text, text_rect)
            
            score_text = self.font_small.render(f"Final Score: {self.score:.1f}", True, self.COLOR_TEXT)
            score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(score_text, score_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Gem Swap")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    total_reward = 0.0
    
    running = True
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    done = False
                    continue
                elif event.key == pygame.K_q:
                    running = False
                    continue

        if not done and any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Moves: {info['moves_left']}")
            done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()