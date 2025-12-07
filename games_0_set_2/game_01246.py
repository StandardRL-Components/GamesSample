
# Generated: 2025-08-27T16:31:30.686478
# Source Brief: brief_01246.md
# Brief Index: 1246

        
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
        "Controls: Use arrows to move cursor. Space to select a gem. "
        "Move to an adjacent gem and press Space again to swap. Shift to cancel selection."
    )

    game_description = (
        "Swap adjacent gems to create matches of 3 or more and reach a target score within a limited number of moves."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    BOARD_WIDTH = 8
    BOARD_HEIGHT = 8
    GEM_SIZE = 40
    NUM_GEM_TYPES = 6 # 5 colors + 1 for empty
    TARGET_SCORE = 5000
    MAX_MOVES = 20
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (50, 60, 80)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 0, 100)
    COLOR_SELECT = (255, 255, 255, 150)
    GEM_COLORS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (50, 150, 255),  # Blue
        (255, 255, 50),  # Yellow
        (200, 50, 255),  # Purple
        (255, 150, 50),  # Orange
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
        self.font_large = pygame.font.Font(None, 72)
        
        self.grid_offset_x = (self.SCREEN_WIDTH - self.BOARD_WIDTH * self.GEM_SIZE) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.BOARD_HEIGHT * self.GEM_SIZE) // 2

        self.np_random = None
        self.board = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.match_effects = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             self.np_random = np.random.default_rng(seed=seed)
        else:
             self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False
        self.cursor_pos = [self.BOARD_WIDTH // 2, self.BOARD_HEIGHT // 2]
        self.selected_gem_pos = None
        self.particles = []
        self.match_effects = []
        
        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        self.match_effects.clear() # Clear effects from previous step
        reward = 0.0
        terminated = False
        
        # Handle player input and game logic for one "turn"
        turn_taken, swap_success = self._handle_input(action)

        if turn_taken:
            self.moves_left -= 1
            if not swap_success:
                reward = -0.1 # Penalty for invalid swap
            else:
                # Process matches and cascades
                cascade_reward, score_gain = self._process_cascades()
                reward += cascade_reward
                self.score += score_gain

        # Check for termination conditions
        if self.score >= self.TARGET_SCORE:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100.0
        elif self.moves_left <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            if not self.win: # Don't penalize if won on the last move
                reward += -50.0
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        turn_taken = False
        swap_success = False

        # --- Handle Deselection ---
        if shift_press and self.selected_gem_pos:
            self.selected_gem_pos = None
            # Sound: Cancel/Deselect

        # --- Handle Cursor Movement ---
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.BOARD_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.BOARD_WIDTH - 1, self.cursor_pos[0] + 1)

        # --- Handle Selection/Swap ---
        if space_press:
            if not self.selected_gem_pos:
                # First selection
                self.selected_gem_pos = tuple(self.cursor_pos)
                # Sound: Select
            else:
                # Second selection - attempt swap
                turn_taken = True
                target_pos = tuple(self.cursor_pos)
                
                # Check for adjacency
                dx = abs(self.selected_gem_pos[0] - target_pos[0])
                dy = abs(self.selected_gem_pos[1] - target_pos[1])

                if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                    # Perform swap
                    self._swap_gems(self.selected_gem_pos, target_pos)
                    # Sound: Swap
                    
                    # Check if swap results in a match
                    matches = self._find_all_matches()
                    if len(matches) > 0:
                        swap_success = True
                    else:
                        # No match, swap back
                        self._swap_gems(self.selected_gem_pos, target_pos) # Swap back
                        swap_success = False
                        # Sound: Invalid Swap
                else:
                    # Not adjacent, invalid move
                    swap_success = False

                # Reset selection after swap attempt
                self.selected_gem_pos = None
        
        return turn_taken, swap_success

    def _process_cascades(self):
        total_reward = 0
        total_score = 0
        
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            
            # Sound: Match
            
            # Calculate reward and score
            num_matched = len(matches)
            total_reward += num_matched * 1.0
            total_score += num_matched * 10
            
            if num_matched >= 5:
                total_reward += 10.0
                total_score += 100 # Bonus for big match

            # Create visual effects and remove gems
            for r, c in matches:
                self._spawn_particles(c, r, self.GEM_COLORS[self.board[r, c]])
                self.match_effects.append((c, r))
                self.board[r, c] = -1 # Mark as empty

            # Apply gravity and fill new gems
            self._apply_gravity()
            self._fill_top_rows()
            
        return total_reward, total_score

    def _generate_board(self):
        self.board = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.BOARD_HEIGHT, self.BOARD_WIDTH))
        while len(self._find_all_matches()) > 0:
             self.board = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.BOARD_HEIGHT, self.BOARD_WIDTH))

    def _swap_gems(self, pos1, pos2):
        r1, c1 = pos1[1], pos1[0]
        r2, c2 = pos2[1], pos2[0]
        self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]

    def _find_all_matches(self):
        matches = set()
        # Horizontal matches
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH - 2):
                if self.board[r, c] == self.board[r, c+1] == self.board[r, c+2] != -1:
                    matches.add((r, c))
                    matches.add((r, c+1))
                    matches.add((r, c+2))
        # Vertical matches
        for c in range(self.BOARD_WIDTH):
            for r in range(self.BOARD_HEIGHT - 2):
                if self.board[r, c] == self.board[r+1, c] == self.board[r+2, c] != -1:
                    matches.add((r, c))
                    matches.add((r+1, c))
                    matches.add((r+2, c))
        
        # Expand matches to find 4s and 5s
        expanded_matches = set(matches)
        for r_start, c_start in list(matches):
            if (r_start, c_start) not in expanded_matches: continue
            gem_type = self.board[r_start, c_start]
            # Expand right
            c = c_start + 1
            while c < self.BOARD_WIDTH and self.board[r_start, c] == gem_type:
                expanded_matches.add((r_start, c))
                c += 1
            # Expand left
            c = c_start - 1
            while c >= 0 and self.board[r_start, c] == gem_type:
                expanded_matches.add((r_start, c))
                c -= 1
            # Expand down
            r = r_start + 1
            while r < self.BOARD_HEIGHT and self.board[r, c_start] == gem_type:
                expanded_matches.add((r, c_start))
                r += 1
            # Expand up
            r = r_start - 1
            while r >= 0 and self.board[r, c_start] == gem_type:
                expanded_matches.add((r, c_start))
                r -= 1

        return expanded_matches
    
    def _apply_gravity(self):
        for c in range(self.BOARD_WIDTH):
            empty_row = self.BOARD_HEIGHT - 1
            for r in range(self.BOARD_HEIGHT - 1, -1, -1):
                if self.board[r, c] != -1:
                    if r != empty_row:
                        self.board[empty_row, c] = self.board[r, c]
                        self.board[r, c] = -1
                    empty_row -= 1
    
    def _fill_top_rows(self):
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH):
                if self.board[r, c] == -1:
                    self.board[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _spawn_particles(self, c, r, color):
        center_x = self.grid_offset_x + c * self.GEM_SIZE + self.GEM_SIZE // 2
        center_y = self.grid_offset_y + r * self.GEM_SIZE + self.GEM_SIZE // 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for r in range(self.BOARD_HEIGHT + 1):
            y = self.grid_offset_y + r * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, y), (self.grid_offset_x + self.BOARD_WIDTH * self.GEM_SIZE, y))
        for c in range(self.BOARD_WIDTH + 1):
            x = self.grid_offset_x + c * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_offset_y), (x, self.grid_offset_y + self.BOARD_HEIGHT * self.GEM_SIZE))
        
        # Draw match effects (under gems)
        for c, r in self.match_effects:
            center_x = self.grid_offset_x + c * self.GEM_SIZE + self.GEM_SIZE // 2
            center_y = self.grid_offset_y + r * self.GEM_SIZE + self.GEM_SIZE // 2
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.GEM_SIZE // 2, (255, 255, 255, 100))

        # Draw gems
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH):
                gem_type = self.board[r, c]
                if gem_type != -1:
                    self._render_gem(c, r, gem_type)

        # Draw selection highlight
        if self.selected_gem_pos:
            c, r = self.selected_gem_pos
            rect = pygame.Rect(self.grid_offset_x + c * self.GEM_SIZE, self.grid_offset_y + r * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, 4, border_radius=5)
        
        # Draw cursor
        c, r = self.cursor_pos
        cursor_rect = pygame.Rect(self.grid_offset_x + c * self.GEM_SIZE, self.grid_offset_y + r * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        alpha = 50 + pulse * 100
        color = (*self.COLOR_CURSOR[:3], alpha)
        s = pygame.Surface((self.GEM_SIZE, self.GEM_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, color, s.get_rect(), 3, border_radius=5)
        self.screen.blit(s, cursor_rect.topleft)
        
        # Update and draw particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['life'] -= 1
            radius = max(0, int(p['life'] * 0.15))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_gem(self, c, r, gem_type):
        x = self.grid_offset_x + c * self.GEM_SIZE
        y = self.grid_offset_y + r * self.GEM_SIZE
        center_x, center_y = x + self.GEM_SIZE // 2, y + self.GEM_SIZE // 2
        radius = self.GEM_SIZE // 2 - 4
        color = self.GEM_COLORS[gem_type]
        
        # Main gem body
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, tuple(val // 2 for val in color))
        
        # Highlight
        highlight_color = tuple(min(255, val + 80) for val in color)
        pygame.gfxdraw.filled_circle(self.screen, center_x - radius//3, center_y - radius//3, radius//3, highlight_color)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Moves
        moves_text = self.font_small.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
            "selected_gem": self.selected_gem_pos,
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

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override screen for display
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Swap")

    action = [0, 0, 0] # No-op, no space, no shift
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while not done:
        # --- Human Input to Action Mapping ---
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
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
                    space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 1
        
        # Only step if an action is taken
        if movement or space or shift:
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            if reward != 0:
                print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")

        # Render the current state
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human playability

    print("Game Over!")
    env.close()