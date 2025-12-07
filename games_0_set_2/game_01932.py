
# Generated: 2025-08-27T18:43:45.733544
# Source Brief: brief_01932.md
# Brief Index: 1932

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem. "
        "Move to an adjacent gem and press Space again to swap. Press Shift to deselect."
    )

    game_description = (
        "Swap adjacent gems to create matches of 3 or more in this isometric puzzle game. "
        "Reach the target score before you run out of moves!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    BOARD_WIDTH, BOARD_HEIGHT = 8, 8
    WIN_SCORE = 5000
    MAX_MOVES = 25
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (15, 25, 40)
    COLOR_GRID = (30, 45, 70)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 150, 50),  # Orange
    ]
    COLOR_WHITE = (255, 255, 255)
    COLOR_GOLD = (255, 215, 0)
    COLOR_SILVER = (192, 192, 192)

    # Isometric projection
    TILE_W = 32
    TILE_H = 16
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 100

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        self.board = None
        self.cursor_pos = None
        self.selected_pos = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.animations = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False
        self.cursor_pos = (self.BOARD_HEIGHT // 2, self.BOARD_WIDTH // 2)
        self.selected_pos = None
        self.particles = []
        self.animations = []

        self._init_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0
        
        # --- Handle Actions ---
        if not self.game_over:
            # 1. Handle cursor movement
            if movement == 1: # Up
                self.cursor_pos = ((self.cursor_pos[0] - 1 + self.BOARD_HEIGHT) % self.BOARD_HEIGHT, self.cursor_pos[1])
            elif movement == 2: # Down
                self.cursor_pos = ((self.cursor_pos[0] + 1) % self.BOARD_HEIGHT, self.cursor_pos[1])
            elif movement == 3: # Left
                self.cursor_pos = (self.cursor_pos[0], (self.cursor_pos[1] - 1 + self.BOARD_WIDTH) % self.BOARD_WIDTH)
            elif movement == 4: # Right
                self.cursor_pos = (self.cursor_pos[0], (self.cursor_pos[1] + 1) % self.BOARD_WIDTH)

            # 2. Handle deselection
            if shift_press:
                self.selected_pos = None
                # SFX: Deselect_sound

            # 3. Handle selection/swap
            elif space_press:
                if self.selected_pos is None:
                    self.selected_pos = self.cursor_pos
                    # SFX: Select_gem_sound
                else:
                    if self._are_adjacent(self.selected_pos, self.cursor_pos):
                        reward += self._attempt_swap(self.selected_pos, self.cursor_pos)
                        self.selected_pos = None # Deselect after any swap attempt
                    else:
                        self.selected_pos = self.cursor_pos # Not adjacent, so just re-select
                        # SFX: Select_gem_sound
        
        # --- Update Game State ---
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE and not self.game_over:
            self.game_over = True
            self.win = True
            reward += 100
        elif self.moves_left <= 0 and not self.game_over:
            self.game_over = True
            self.win = False
            reward -= 100
        
        if self.game_over or self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "win": self.win
        }

    def close(self):
        pygame.quit()

    # --- Helper Methods ---

    def _init_board(self):
        self.board = self.np_random.integers(0, len(self.GEM_COLORS), size=(self.BOARD_HEIGHT, self.BOARD_WIDTH))
        while self._find_and_remove_matches(add_score=False):
            self._refill_board()
        
        if not self._find_possible_moves():
            self._shuffle_board()

    def _grid_to_iso(self, r, c):
        x = self.ORIGIN_X + (c - r) * self.TILE_W
        y = self.ORIGIN_Y + (c + r) * self.TILE_H
        return int(x), int(y)

    def _are_adjacent(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2) == 1

    def _attempt_swap(self, pos1, pos2):
        self.moves_left -= 1
        r1, c1 = pos1
        r2, c2 = pos2
        
        # Perform swap
        self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]
        # SFX: Gem_swap_swoosh

        total_reward = 0
        cascade_multiplier = 1.0

        while True:
            matches = self._find_matches()
            if not matches:
                # If the very first check finds no matches, it was an invalid move
                if cascade_multiplier == 1.0:
                    self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1] # Swap back
                    # SFX: Invalid_move_buzz
                    return -0.1
                else: # End of cascade
                    break
            
            # --- Process Matches ---
            num_cleared = len(matches)
            # Base reward: +1 per gem
            match_reward = num_cleared
            # Bonus for larger matches
            if num_cleared == 4: match_reward += 5
            if num_cleared >= 5: match_reward += 10
            
            total_reward += match_reward * cascade_multiplier

            self.score += int(match_reward * cascade_multiplier * 10)
            
            for r, c in matches:
                self._spawn_particles(r, c, self.board[r, c])
                self.board[r, c] = -1 # Mark for removal
            
            # SFX: Match_success_chime
            
            self._refill_board()
            cascade_multiplier += 0.5 # Increase multiplier for next cascade

        if not self._find_possible_moves():
            self._shuffle_board()
            # SFX: Board_shuffle_whoosh

        return total_reward

    def _find_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH - 2):
                if self.board[r, c] != -1 and self.board[r, c] == self.board[r, c+1] == self.board[r, c+2]:
                    gem_type = self.board[r, c]
                    i = c
                    while i < self.BOARD_WIDTH and self.board[r, i] == gem_type:
                        matches.add((r, i))
                        i += 1
        # Vertical
        for c in range(self.BOARD_WIDTH):
            for r in range(self.BOARD_HEIGHT - 2):
                if self.board[r, c] != -1 and self.board[r, c] == self.board[r+1, c] == self.board[r+2, c]:
                    gem_type = self.board[r, c]
                    i = r
                    while i < self.BOARD_HEIGHT and self.board[i, c] == gem_type:
                        matches.add((i, c))
                        i += 1
        return matches

    def _find_and_remove_matches(self, add_score=True):
        matches = self._find_matches()
        if not matches:
            return False
        
        if add_score:
            self.score += len(matches) * 10
        
        for r, c in matches:
            self.board[r, c] = -1
        return True

    def _refill_board(self):
        for c in range(self.BOARD_WIDTH):
            empty_count = 0
            for r in range(self.BOARD_HEIGHT - 1, -1, -1):
                if self.board[r, c] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    self.board[r + empty_count, c] = self.board[r, c]
                    self.board[r, c] = -1
            
            for r in range(empty_count):
                self.board[r, c] = self.np_random.integers(0, len(self.GEM_COLORS))

    def _find_possible_moves(self):
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH):
                # Check swap right
                if c < self.BOARD_WIDTH - 1:
                    self.board[r,c], self.board[r,c+1] = self.board[r,c+1], self.board[r,c]
                    if self._find_matches():
                        self.board[r,c], self.board[r,c+1] = self.board[r,c+1], self.board[r,c]
                        return True
                    self.board[r,c], self.board[r,c+1] = self.board[r,c+1], self.board[r,c]
                # Check swap down
                if r < self.BOARD_HEIGHT - 1:
                    self.board[r,c], self.board[r+1,c] = self.board[r+1,c], self.board[r,c]
                    if self._find_matches():
                        self.board[r,c], self.board[r+1,c] = self.board[r+1,c], self.board[r,c]
                        return True
                    self.board[r,c], self.board[r+1,c] = self.board[r+1,c], self.board[r,c]
        return False

    def _shuffle_board(self):
        flat_board = self.board.flatten()
        self.np_random.shuffle(flat_board)
        self.board = flat_board.reshape((self.BOARD_HEIGHT, self.BOARD_WIDTH))
        while self._find_and_remove_matches(add_score=False):
            self._refill_board()
        if not self._find_possible_moves():
            self._shuffle_board() # Recurse if shuffle is unlucky

    def _spawn_particles(self, r, c, gem_type):
        x, y = self._grid_to_iso(r, c)
        color = self.GEM_COLORS[gem_type]
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([x, y, vx, vy, lifetime, color])

    def _update_particles(self):
        self.particles = [
            [p[0] + p[2], p[1] + p[3], p[2], p[3] * 0.95 + 0.1, p[4] - 1, p[5]]
            for p in self.particles if p[4] > 0
        ]

    # --- Rendering Methods ---
    def _render_game(self):
        # Draw grid lines
        for r in range(self.BOARD_HEIGHT + 1):
            p1 = self._grid_to_iso(r, -0.5)
            p2 = self._grid_to_iso(r, self.BOARD_WIDTH - 0.5)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for c in range(self.BOARD_WIDTH + 1):
            p1 = self._grid_to_iso(-0.5, c)
            p2 = self._grid_to_iso(self.BOARD_HEIGHT - 0.5, c)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

        # Draw gems
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH):
                gem_type = self.board[r, c]
                if gem_type != -1:
                    x, y = self._grid_to_iso(r, c)
                    self._draw_gem(self.screen, (x, y), gem_type)
        
        # Draw cursor
        if self.cursor_pos:
            r, c = self.cursor_pos
            self._draw_highlight(self.screen, r, c, self.COLOR_WHITE, 2, 0)
        
        # Draw selection
        if self.selected_pos:
            r, c = self.selected_pos
            pulse = abs(math.sin(self.steps * 0.2))
            color = tuple(min(255, int(c + (255-c)*pulse)) for c in self.COLOR_GOLD)
            self._draw_highlight(self.screen, r, c, color, 3, pulse)

        # Draw particles
        for p in self.particles:
            x, y, _, _, lifetime, color = p
            alpha = max(0, min(255, int(255 * (lifetime / 30.0))))
            size = max(1, int(3 * (lifetime / 30.0)))
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (size, size), size)
            self.screen.blit(s, (int(x) - size, int(y) - size))

    def _draw_gem(self, surface, pos, gem_type):
        x, y = pos
        color = self.GEM_COLORS[gem_type]
        dark_color = tuple(c * 0.6 for c in color)
        light_color = tuple(min(255, c * 1.4) for c in color)

        points = [
            (x, y - self.TILE_H * 1.2),
            (x + self.TILE_W * 0.8, y - self.TILE_H * 0.4),
            (x + self.TILE_W * 0.6, y + self.TILE_H * 0.6),
            (x, y + self.TILE_H * 1.1),
            (x - self.TILE_W * 0.6, y + self.TILE_H * 0.6),
            (x - self.TILE_W * 0.8, y - self.TILE_H * 0.4),
        ]
        
        pygame.gfxdraw.aapolygon(surface, points, dark_color)
        pygame.gfxdraw.filled_polygon(surface, points, dark_color)
        
        inner_points = [
            (x, y - self.TILE_H * 0.9),
            (x + self.TILE_W * 0.6, y - self.TILE_H * 0.3),
            (x + self.TILE_W * 0.45, y + self.TILE_H * 0.45),
            (x, y + self.TILE_H * 0.8),
            (x - self.TILE_W * 0.45, y + self.TILE_H * 0.45),
            (x - self.TILE_W * 0.6, y - self.TILE_H * 0.3),
        ]
        pygame.gfxdraw.aapolygon(surface, inner_points, color)
        pygame.gfxdraw.filled_polygon(surface, inner_points, color)
        
        # Glint
        pygame.gfxdraw.filled_circle(surface, int(x - self.TILE_W*0.2), int(y - self.TILE_H*0.2), 3, light_color)
        pygame.gfxdraw.aacircle(surface, int(x - self.TILE_W*0.2), int(y - self.TILE_H*0.2), 3, light_color)

    def _draw_highlight(self, surface, r, c, color, width, pulse=0):
        x, y = self._grid_to_iso(r, c)
        size_w = self.TILE_W + 4 + pulse * 4
        size_h = self.TILE_H + 4 + pulse * 4
        points = [
            (x, y - size_h),
            (x + size_w, y),
            (x, y + size_h),
            (x - size_w, y),
        ]
        pygame.draw.lines(surface, color, True, points, width)

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_GOLD)
        self.screen.blit(score_text, (20, 10))

        # Moves display
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_SILVER)
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_GOLD if self.win else self.COLOR_RED
            
            msg_surf = self.font_large.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(msg_surf, msg_rect)
            
            final_score_surf = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_WHITE)
            final_score_rect = final_score_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 40))
            self.screen.blit(final_score_surf, final_score_rect)

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

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    pygame.display.set_caption("Gem Swap")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        if not terminated:
            # Only step if an action was taken
            if any(a != 0 for a in action):
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()