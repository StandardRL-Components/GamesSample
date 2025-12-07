
# Generated: 2025-08-28T01:15:14.656865
# Source Brief: brief_04050.md
# Brief Index: 4050

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrows to move the cursor. Press space to swap the gem "
        "with the one in the direction you last moved."
    )

    game_description = (
        "Swap adjacent gems to create matches of 3 or more and reach a target "
        "score within a limited number of moves."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    
    WIN_SCORE = 5000
    MAX_MOVES = 20
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 100)
    
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 160, 80)   # Orange
    ]
    NUM_GEM_TYPES = len(GEM_COLORS)

    # Rewards
    REWARD_WIN = 100.0
    REWARD_LOSS = -10.0
    REWARD_INVALID_SWAP = -0.1
    REWARD_PER_GEM = 1.0
    REWARD_MATCH_3 = 10.0
    REWARD_MATCH_4 = 20.0
    REWARD_MATCH_5 = 50.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.board_offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * 40) // 2
        self.board_offset_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * 40) // 2 + 20
        self.gem_size = 40
        
        self.board = None
        self.score = 0
        self.moves_remaining = 0
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.last_move_dir = None # (dx, dy)
        self.particles = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.last_move_dir = None
        self.particles = []
        
        self._create_board()
        while not self._find_possible_moves():
            self._reshuffle_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

        # 1. Handle cursor movement
        if movement > 0:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            self.cursor_pos[0] = (self.cursor_pos[0] + dx + self.GRID_WIDTH) % self.GRID_WIDTH
            self.cursor_pos[1] = (self.cursor_pos[1] + dy + self.GRID_HEIGHT) % self.GRID_HEIGHT
            self.last_move_dir = (dx, dy)

        # 2. Handle special actions (Shift for reshuffle)
        if shift_press:
            self._reshuffle_board()
            # sfx: board_reshuffle

        # 3. Handle primary action (Space for swap)
        if space_press and self.last_move_dir:
            reward += self._attempt_swap()
            self.last_move_dir = None # Consume the direction

        # 4. Check for game end conditions
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += self.REWARD_WIN
            # sfx: game_win
        elif self.moves_remaining <= 0:
            terminated = True
            reward += self.REWARD_LOSS
            # sfx: game_over
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _attempt_swap(self):
        self.moves_remaining -= 1
        
        x1, y1 = self.cursor_pos
        dx, dy = self.last_move_dir
        x2, y2 = x1 + dx, y1 + dy
        
        if not (0 <= x2 < self.GRID_WIDTH and 0 <= y2 < self.GRID_HEIGHT):
            return self.REWARD_INVALID_SWAP # Swap out of bounds

        # Perform swap
        self.board[y1, x1], self.board[y2, x2] = self.board[y2, x2], self.board[y1, x1]
        # sfx: gem_swap

        match_reward, matches_found = self._process_board_state()
        
        if not matches_found:
            # Invalid swap, reverse it
            self.board[y1, x1], self.board[y2, x2] = self.board[y2, x2], self.board[y1, x1]
            # sfx: invalid_swap
            return self.REWARD_INVALID_SWAP
        
        # Auto-reshuffle if board is stuck
        if not self._find_possible_moves():
            self._reshuffle_board()
            # sfx: board_reshuffle_auto

        return match_reward

    def _process_board_state(self):
        total_reward = 0
        matches_found_in_total = False
        chain_level = 0

        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            
            matches_found_in_total = True
            chain_level += 1
            if chain_level > 1:
                # sfx: chain_reaction
                pass

            # Calculate reward and clear gems
            num_cleared = len(matches)
            total_reward += num_cleared * self.REWARD_PER_GEM
            
            # This part is just for scoring, not for reward, to avoid double counting
            match_groups = self._get_match_groups(matches)
            for group in match_groups:
                self.score += [0, 0, 0, 100, 200, 500, 750, 1000][min(len(group), 7)]
                reward_bonus = {3: self.REWARD_MATCH_3, 4: self.REWARD_MATCH_4}.get(len(group), self.REWARD_MATCH_5)
                total_reward += reward_bonus
            
            for r, c in matches:
                self._create_particles(c, r, self.board[r, c])
                self.board[r, c] = -1
            # sfx: match_clear

            # Apply gravity and fill
            self._apply_gravity_and_fill()
        
        return total_reward, matches_found_in_total

    def _create_board(self):
        self.board = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            for r, c in matches:
                self.board[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _find_all_matches(self):
        matches = set()
        # Horizontal matches
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.board[r, c] != -1 and self.board[r, c] == self.board[r, c+1] == self.board[r, c+2]:
                    gem_type = self.board[r, c]
                    i = c
                    while i < self.GRID_WIDTH and self.board[r, i] == gem_type:
                        matches.add((r, i))
                        i += 1
        # Vertical matches
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.board[r, c] != -1 and self.board[r, c] == self.board[r+1, c] == self.board[r+2, c]:
                    gem_type = self.board[r, c]
                    i = r
                    while i < self.GRID_HEIGHT and self.board[i, c] == gem_type:
                        matches.add((i, c))
                        i += 1
        return matches
    
    def _get_match_groups(self, match_coords):
        # Helper to group connected match coordinates for scoring
        coords = set(match_coords)
        groups = []
        while coords:
            group = set()
            q = deque([coords.pop()])
            group.add(q[0])
            while q:
                r, c = q.popleft()
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (r + dr, c + dc)
                    if neighbor in coords:
                        coords.remove(neighbor)
                        group.add(neighbor)
                        q.append(neighbor)
            groups.append(group)
        return groups

    def _apply_gravity_and_fill(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[r, c] != -1:
                    self.board[empty_row, c], self.board[r, c] = self.board[r, c], self.board[empty_row, c]
                    empty_row -= 1
            for r in range(empty_row, -1, -1):
                self.board[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Try swapping right
                if c < self.GRID_WIDTH - 1:
                    self.board[r, c], self.board[r, c+1] = self.board[r, c+1], self.board[r, c]
                    if self._find_all_matches(): moves.append(((r, c), (r, c+1)))
                    self.board[r, c], self.board[r, c+1] = self.board[r, c+1], self.board[r, c] # Swap back
                # Try swapping down
                if r < self.GRID_HEIGHT - 1:
                    self.board[r, c], self.board[r+1, c] = self.board[r+1, c], self.board[r, c]
                    if self._find_all_matches(): moves.append(((r, c), (r+1, c)))
                    self.board[r, c], self.board[r+1, c] = self.board[r+1, c], self.board[r, c] # Swap back
        return moves

    def _reshuffle_board(self):
        flat_board = self.board.flatten()
        self.np_random.shuffle(flat_board)
        self.board = flat_board.reshape((self.GRID_HEIGHT, self.GRID_WIDTH))
        
        # Ensure no matches on reshuffle and that moves are possible
        while self._find_all_matches() or not self._find_possible_moves():
            self._create_board()

    def _create_particles(self, c, r, gem_type):
        px = self.board_offset_x + c * self.gem_size + self.gem_size // 2
        py = self.board_offset_y + r * self.gem_size + self.gem_size // 2
        color = self.GEM_COLORS[gem_type]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([px, py, vx, vy, color, lifetime])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.board_offset_y + r * self.gem_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.board_offset_x, y), (self.board_offset_x + self.GRID_WIDTH * self.gem_size, y), 1)
        for c in range(self.GRID_WIDTH + 1):
            x = self.board_offset_x + c * self.gem_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.board_offset_y), (x, self.board_offset_y + self.GRID_HEIGHT * self.gem_size), 1)
            
        # Draw gems
        gem_radius = self.gem_size // 2 - 4
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.board[r, c]
                if gem_type != -1:
                    cx = self.board_offset_x + c * self.gem_size + self.gem_size // 2
                    cy = self.board_offset_y + r * self.gem_size + self.gem_size // 2
                    color = self.GEM_COLORS[gem_type]
                    
                    pygame.gfxdraw.aacircle(self.screen, cx, cy, gem_radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, cx, cy, gem_radius, color)
                    
                    # Highlight
                    highlight_color = (min(255, c+80) for c in color)
                    pygame.gfxdraw.aacircle(self.screen, cx - 3, cy - 3, gem_radius // 2, (*highlight_color, 80))

        # Draw particles
        self._render_particles()

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        rect = pygame.Rect(
            self.board_offset_x + cursor_x * self.gem_size,
            self.board_offset_y + cursor_y * self.gem_size,
            self.gem_size, self.gem_size
        )
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        alpha = 150 + pulse * 105
        cursor_color = (*self.COLOR_CURSOR, alpha)
        
        # Create a temporary surface for the glowing rectangle
        glow_surface = pygame.Surface((self.gem_size, self.gem_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, cursor_color, glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=8)


    def _render_particles(self):
        for p in self.particles[:]:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[5] -= 1    # lifetime -= 1
            if p[5] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p[5] / 30.0))))
                color = (*p[4], alpha)
                temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (2, 2), 2)
                self.screen.blit(temp_surf, (int(p[0]), int(p[1])))

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                end_text_str = "YOU WIN!"
                end_color = (100, 255, 100)
            else:
                end_text_str = "GAME OVER"
                end_color = (255, 100, 100)
            
            end_text = self.font_large.render(end_text_str, True, end_color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "cursor_pos": list(self.cursor_pos),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a display for human play
    pygame.display.set_caption("Gem Swap")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    clock = pygame.time.Clock()
    
    print(GameEnv.user_guide)

    while not terminated:
        movement = 0 # no-op
        space_press = 0
        shift_press = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_press = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_press = 1
                elif event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                elif event.key == pygame.K_q:
                    terminated = True

        action = [movement, space_press, shift_press]
        obs, reward, terminated, truncated, info = env.step(action)

        # Draw the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    env.close()