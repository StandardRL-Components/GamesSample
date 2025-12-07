import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Set the SDL video driver to "dummy" to run Pygame in headless mode.
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to select a gem. Arrows again to swap. Shift to cancel selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent gems to create matches of 3 or more. Reach the target score of 100 gems within 20 moves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # Constants
    WIDTH, HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    GEM_SIZE = 40
    GRID_OFFSET_X = (WIDTH - GRID_WIDTH * GEM_SIZE) // 2
    GRID_OFFSET_Y = (HEIGHT - GRID_HEIGHT * GEM_SIZE) // 2
    NUM_GEM_TYPES = 6
    TARGET_SCORE = 100
    MAX_MOVES = 20
    
    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_GRID = (30, 40, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SCORE = (255, 215, 0)
    COLOR_MOVES = (255, 100, 100)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED = (255, 255, 0)
    
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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # Etc...        
        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.win = None
        self.steps = None
        self.particles = []
        self.last_action_was_swap = False
        
        # Initialize state variables
        # self.reset() is implicitly called by the environment wrapper, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_gem_pos = None
        self.particles = []
        self.last_action_was_swap = False
        
        self._generate_board()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Update game logic
        self.steps += 1
        reward = 0
        self.last_action_was_swap = False

        if shift_held and self.selected_gem_pos:
            self.selected_gem_pos = None
            # sfx: cancel_selection
        
        elif space_held and not self.selected_gem_pos:
            self.selected_gem_pos = list(self.cursor_pos)
            # sfx: select_gem

        elif movement != 0:
            if self.selected_gem_pos:
                # Attempt a swap, which consumes a move
                self.moves_left -= 1
                reward, match_found = self._handle_swap(movement)
                self.selected_gem_pos = None
                self.last_action_was_swap = True
                if match_found:
                    # sfx: match_success
                    cascade_reward = self._process_cascades()
                    reward += cascade_reward
                else:
                    # sfx: swap_fail
                    pass
            else:
                # Move cursor (does not consume a move)
                dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
                self.cursor_pos[0] = (self.cursor_pos[0] + dx + self.GRID_WIDTH) % self.GRID_WIDTH
                self.cursor_pos[1] = (self.cursor_pos[1] + dy + self.GRID_HEIGHT) % self.GRID_HEIGHT
                # sfx: cursor_move
        
        terminated = self._check_termination()
        if terminated and self.win:
            reward += 100
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _check_termination(self):
        if self.score >= self.TARGET_SCORE:
            self.game_over = True
            self.win = True
            return True
        if self.moves_left <= 0:
            self.game_over = True
            return True
        if self.last_action_was_swap and not self._find_possible_moves():
             self.game_over = True
             return True
        return False

    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
            
            # Resolve any initial cascades until the board is stable.
            while True:
                matches = self._find_matches()
                if not np.any(matches):
                    break
                
                # Set matched gems to 0 (to be removed), without creating particles.
                self.grid[matches] = 0
                
                self._handle_gravity()
                self._fill_top_rows()

            # After the board is settled (no initial matches), check if it's playable.
            if self._find_possible_moves():
                break # If it has moves, we are done.

    def _find_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Check swap right
                if x < self.GRID_WIDTH - 1:
                    self.grid[x,y], self.grid[x+1,y] = self.grid[x+1,y], self.grid[x,y]
                    if np.any(self._find_matches()):
                        self.grid[x,y], self.grid[x+1,y] = self.grid[x+1,y], self.grid[x,y]
                        return True
                    self.grid[x,y], self.grid[x+1,y] = self.grid[x+1,y], self.grid[x,y]
                # Check swap down
                if y < self.GRID_HEIGHT - 1:
                    self.grid[x,y], self.grid[x,y+1] = self.grid[x,y+1], self.grid[x,y]
                    if np.any(self._find_matches()):
                        self.grid[x,y], self.grid[x,y+1] = self.grid[x,y+1], self.grid[x,y]
                        return True
                    self.grid[x,y], self.grid[x,y+1] = self.grid[x,y+1], self.grid[x,y]
        return False

    def _handle_swap(self, movement):
        x1, y1 = self.selected_gem_pos
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
        x2, y2 = x1 + dx, y1 + dy

        if not (0 <= x2 < self.GRID_WIDTH and 0 <= y2 < self.GRID_HEIGHT):
            return 0, False

        self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]
        
        matches = self._find_matches()
        if not np.any(matches):
            self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]
            return -0.1, False
        
        reward = self._calculate_reward(matches)
        self._remove_gems(matches)
        return reward, True

    def _process_cascades(self):
        total_cascade_reward = 0
        while True:
            self._handle_gravity()
            self._fill_top_rows()
            matches = self._find_matches()
            if not np.any(matches): break
            # sfx: cascade_match
            reward = self._calculate_reward(matches)
            self._remove_gems(matches)
            total_cascade_reward += reward
        return total_cascade_reward

    def _find_matches(self):
        matches = np.zeros_like(self.grid, dtype=bool)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.grid[x,y] != 0 and self.grid[x,y] == self.grid[x+1,y] == self.grid[x+2,y]:
                    matches[x:x+3, y] = True
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.grid[x,y] != 0 and self.grid[x,y] == self.grid[x,y+1] == self.grid[x,y+2]:
                    matches[x, y:y+3] = True
        return matches

    def _calculate_reward(self, matches):
        reward = 0
        num_matched = np.sum(matches)
        reward += num_matched
        self.score += num_matched

        # Simplified bonus for long matches
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if matches[x, y]:
                    h_len = 0
                    while x + h_len < self.GRID_WIDTH and matches[x + h_len, y]: h_len += 1
                    if h_len >= 5: reward += 10
                    elif h_len >= 4: reward += 5
                    
                    v_len = 0
                    while y + v_len < self.GRID_HEIGHT and matches[x, y + v_len]: v_len += 1
                    if v_len >= 5: reward += 10
                    elif v_len >= 4: reward += 5
                    
                    # Avoid double counting bonuses by zeroing out processed matches
                    if h_len > 0: matches[x:x+h_len, y] = False
                    if v_len > 0: matches[x, y:y+v_len] = False
        return reward

    def _remove_gems(self, matches):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if matches[x, y]:
                    gem_type = self.grid[x, y]
                    self._create_particles(x, y, gem_type)
                    self.grid[x, y] = 0

    def _handle_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] != 0:
                    if y != empty_row:
                        self.grid[x, empty_row] = self.grid[x, y]
                        self.grid[x, y] = 0
                    empty_row -= 1

    def _fill_top_rows(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] == 0:
                    self.grid[x, y] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
                    # sfx: gem_fall

    def _create_particles(self, grid_x, grid_y, gem_type):
        px = self.GRID_OFFSET_X + grid_x * self.GEM_SIZE + self.GEM_SIZE // 2
        py = self.GRID_OFFSET_Y + grid_y * self.GEM_SIZE + self.GEM_SIZE // 2
        color = self.GEM_COLORS[gem_type - 1]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append([px, py, vel, lifespan, color])

    def _update_and_draw_particles(self):
        self.particles = [p for p in self.particles if p[3] > 0]
        for p in self.particles:
            p[0] += p[2][0]; p[1] += p[2][1]; p[3] -= 1
            radius = int(max(0, p[3] / 5))
            if radius > 0:
                pygame.draw.circle(self.screen, p[4], (int(p[0]), int(p[1])), radius)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_gems()
        self._update_and_draw_particles()
        self._render_cursor()
        if self.game_over: self._render_game_over()

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GRID_OFFSET_X + x * self.GEM_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.GEM_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.GEM_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.GEM_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_WIDTH * self.GEM_SIZE, self.GRID_OFFSET_Y + y * self.GEM_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

    def _render_gems(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] > 0: self._draw_gem(x, y, self.grid[x, y])

    def _draw_gem(self, grid_x, grid_y, gem_type):
        px, py = self.GRID_OFFSET_X + grid_x * self.GEM_SIZE, self.GRID_OFFSET_Y + grid_y * self.GEM_SIZE
        center_x, center_y = px + self.GEM_SIZE // 2, py + self.GEM_SIZE // 2
        color = self.GEM_COLORS[gem_type - 1]
        dark_color = tuple(c * 0.6 for c in color)
        light_color = tuple(min(255, c * 1.4) for c in color)
        inset = 5
        
        if gem_type == 1: # Square
            rect = (px + inset, py + inset, self.GEM_SIZE - 2*inset, self.GEM_SIZE - 2*inset)
            pygame.gfxdraw.box(self.screen, rect, color)
            pygame.gfxdraw.rectangle(self.screen, rect, dark_color)
        elif gem_type == 2: # Circle
            radius = self.GEM_SIZE // 2 - inset
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, dark_color)
        elif gem_type == 3: # Diamond
            points = [(center_x, py + inset), (px + self.GEM_SIZE - inset, center_y), (center_x, py + self.GEM_SIZE - inset), (px + inset, center_y)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, dark_color)
        elif gem_type == 4: # Triangle
            points = [(center_x, py + inset), (px + self.GEM_SIZE - inset, py + self.GEM_SIZE - inset), (px + inset, py + self.GEM_SIZE - inset)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, dark_color)
        elif gem_type == 5: # Hexagon
            radius = self.GEM_SIZE // 2 - inset
            points = [(center_x + radius * math.cos(math.pi / 3 * i), center_y + radius * math.sin(math.pi / 3 * i)) for i in range(6)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, dark_color)
        elif gem_type == 6: # Star
            r1, r2 = self.GEM_SIZE // 2 - inset, (self.GEM_SIZE // 2 - inset) * 0.5
            points = [(center_x + (r1 if i % 2 == 0 else r2) * math.sin(i * math.pi / 5), center_y - (r1 if i % 2 == 0 else r2) * math.cos(i * math.pi / 5)) for i in range(10)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, dark_color)
        
        pygame.gfxdraw.filled_circle(self.screen, center_x - 6, center_y - 6, 3, light_color)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        px, py = self.GRID_OFFSET_X + cx * self.GEM_SIZE, self.GRID_OFFSET_Y + cy * self.GEM_SIZE
        color = self.COLOR_SELECTED if self.selected_gem_pos == [cx, cy] else self.COLOR_CURSOR
        width = 4 if self.selected_gem_pos == [cx, cy] else 2
        pygame.draw.rect(self.screen, color, (px, py, self.GEM_SIZE, self.GEM_SIZE), width)

    def _render_ui(self):
        score_text = self.font_large.render(f"Gems: {self.score}/{self.TARGET_SCORE}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 20))
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_MOVES)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 20))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        text = "YOU WIN!" if self.win else "GAME OVER"
        color = (100, 255, 100) if self.win else (255, 100, 100)
        title_surf = self.font_large.render(text, True, color)
        title_rect = title_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
        score_surf = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 20))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(title_surf, title_rect)
        self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
            "is_selected": self.selected_gem_pos is not None,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # The main loop is for interactive testing, not for the headless environment
    # Re-enable the video driver for display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    pygame.display.set_caption("Gem Swap")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    running = True
    clock = pygame.time.Clock()
    
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: action[2] = 1
                elif event.key == pygame.K_r: obs, info = env.reset()
                
        if any(action):
             obs, reward, terminated, truncated, info = env.step(action)
             print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}, Terminated: {terminated}")
             if terminated: print("Game Over! Press R to restart.")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)
        
    env.close()