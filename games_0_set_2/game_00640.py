
# Generated: 2025-08-27T14:18:22.216712
# Source Brief: brief_00640.md
# Brief Index: 640

        
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
        "Controls: Use arrow keys to move the cursor. Hold Space and press an arrow key to swap gems."
    )

    game_description = (
        "Swap adjacent gems to create matches of 3 or more and reach a target score in a limited number of moves."
    )

    auto_advance = False

    # --- Constants ---
    BOARD_WIDTH, BOARD_HEIGHT = 8, 8
    GEM_TYPES = 6
    GEM_SIZE = 40
    TARGET_SCORE = 1000
    MAX_MOVES = 20
    MAX_STEPS = MAX_MOVES + 50 # Allow for some non-move actions

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SCORE = (255, 220, 100)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_CURSOR_SELECTED = (100, 255, 100)

    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width = 640
        self.height = 400

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        self.board_offset_x = (self.width - self.BOARD_WIDTH * self.GEM_SIZE) // 2
        self.board_offset_y = (self.height - self.BOARD_HEIGHT * self.GEM_SIZE) // 2 + 20

        self.board = None
        self.cursor_pos = None
        self.selected_gem = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_action_was_swap = False
        self.particles = []
        
        self._np_random = None

        self.reset()
        
        # This is a non-standard but useful check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        else:
            self._np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [self.BOARD_HEIGHT // 2, self.BOARD_WIDTH // 2]
        self.selected_gem = None
        self.particles = []
        self.last_action_was_swap = False

        self._generate_initial_board()
        
        return self._get_observation(), self._get_info()

    def _generate_initial_board(self):
        self.board = self._np_random.integers(1, self.GEM_TYPES + 1, size=(self.BOARD_HEIGHT, self.BOARD_WIDTH))
        while self._find_matches():
            self._remove_matches(self._find_matches())
            self._apply_gravity()
            self._fill_top_rows()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.last_action_was_swap = False

        movement, space_held, _ = action
        space_press = space_held == 1

        if space_press:
            # Select/Deselect/Swap logic
            if self.selected_gem is None:
                # Select gem at cursor
                self.selected_gem = list(self.cursor_pos)
            elif self.selected_gem == self.cursor_pos:
                # Deselect if cursor is on selected gem
                self.selected_gem = None
            else:
                # Attempt swap if cursor is adjacent to selected gem
                if self._is_adjacent(self.selected_gem, self.cursor_pos):
                    reward += self._attempt_swap(self.selected_gem, self.cursor_pos)
                    self.selected_gem = None # Deselect after swap attempt
                    self.last_action_was_swap = True
                else:
                    # If not adjacent, move selection to cursor
                    self.selected_gem = list(self.cursor_pos)
        else:
            # Cursor movement
            if movement != 0:
                dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
                self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dy, 0, self.BOARD_HEIGHT - 1)
                self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dx, 0, self.BOARD_WIDTH - 1)

        # A non-swap action that consumes a move is an invalid move in this context
        if movement != 0 and not self.last_action_was_swap:
            pass # Moving cursor doesn't cost a move or give reward

        # Update particles
        self._update_particles()

        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _attempt_swap(self, pos1, pos2):
        self.moves_left -= 1
        
        r1, c1 = pos1
        r2, c2 = pos2
        
        # Perform swap
        self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]
        
        matches = self._find_matches()
        if not matches:
            # Invalid swap, swap back
            self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]
            return -0.1 # Penalty for invalid move
        
        # Valid swap, process matches
        total_reward = 0
        chain_multiplier = 1.0
        
        while matches:
            num_gems_cleared = len(matches)
            
            # Base reward for clearing gems
            reward_for_this_chain = num_gems_cleared * chain_multiplier
            
            # Bonus for large clears
            if num_gems_cleared >= 5:
                reward_for_this_chain += 10
            
            total_reward += reward_for_this_chain
            self.score += int(reward_for_this_chain * 10) # Scale reward for score

            self._create_match_particles(matches)
            # sfx: gem_match.wav
            self._remove_matches(matches)
            # sfx: gems_fall.wav
            self._apply_gravity()
            self._fill_top_rows()
            
            matches = self._find_matches()
            chain_multiplier += 0.5 # Increase multiplier for chain reactions
            
        return total_reward

    def _find_matches(self):
        matches = set()
        # Horizontal matches
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH - 2):
                if self.board[r, c] != 0 and self.board[r, c] == self.board[r, c+1] == self.board[r, c+2]:
                    gem_type = self.board[r, c]
                    match_len = 3
                    while c + match_len < self.BOARD_WIDTH and self.board[r, c+match_len] == gem_type:
                        match_len += 1
                    for i in range(match_len):
                        matches.add((r, c+i))
        # Vertical matches
        for c in range(self.BOARD_WIDTH):
            for r in range(self.BOARD_HEIGHT - 2):
                if self.board[r, c] != 0 and self.board[r, c] == self.board[r+1, c] == self.board[r+2, c]:
                    gem_type = self.board[r, c]
                    match_len = 3
                    while r + match_len < self.BOARD_HEIGHT and self.board[r+match_len, c] == gem_type:
                        match_len += 1
                    for i in range(match_len):
                        matches.add((r+i, c))
        return list(matches)

    def _remove_matches(self, matches):
        for r, c in matches:
            self.board[r, c] = 0

    def _apply_gravity(self):
        for c in range(self.BOARD_WIDTH):
            empty_row = self.BOARD_HEIGHT - 1
            for r in range(self.BOARD_HEIGHT - 1, -1, -1):
                if self.board[r, c] != 0:
                    if r != empty_row:
                        self.board[empty_row, c] = self.board[r, c]
                        self.board[r, c] = 0
                    empty_row -= 1

    def _fill_top_rows(self):
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH):
                if self.board[r, c] == 0:
                    self.board[r, c] = self._np_random.integers(1, self.GEM_TYPES + 1)

    def _check_termination(self):
        if self.score >= self.TARGET_SCORE or self.moves_left <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
    
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
            "cursor_pos": list(self.cursor_pos),
            "selected_gem": self.selected_gem,
        }

    def _render_game(self):
        # Draw grid
        for r in range(self.BOARD_HEIGHT + 1):
            y = self.board_offset_y + r * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.board_offset_x, y), (self.board_offset_x + self.BOARD_WIDTH * self.GEM_SIZE, y))
        for c in range(self.BOARD_WIDTH + 1):
            x = self.board_offset_x + c * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.board_offset_y), (x, self.board_offset_y + self.BOARD_HEIGHT * self.GEM_SIZE))

        # Draw gems
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH):
                gem_type = self.board[r, c]
                if gem_type > 0:
                    self._draw_gem(r, c, gem_type)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], (*p['pos'], p['size'], p['size']))

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_x = self.board_offset_x + cursor_c * self.GEM_SIZE
        cursor_y = self.board_offset_y + cursor_r * self.GEM_SIZE
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, self.GEM_SIZE, self.GEM_SIZE), 3)

        # Draw selected gem highlight
        if self.selected_gem:
            sel_r, sel_c = self.selected_gem
            sel_x = self.board_offset_x + sel_c * self.GEM_SIZE
            sel_y = self.board_offset_y + sel_r * self.GEM_SIZE
            pygame.draw.rect(self.screen, self.COLOR_CURSOR_SELECTED, (sel_x, sel_y, self.GEM_SIZE, self.GEM_SIZE), 3)

    def _draw_gem(self, r, c, gem_type):
        x = self.board_offset_x + c * self.GEM_SIZE + self.GEM_SIZE // 2
        y = self.board_offset_y + r * self.GEM_SIZE + self.GEM_SIZE // 2
        color = self.GEM_COLORS[gem_type - 1]
        
        radius = self.GEM_SIZE // 2 - 5
        
        # Draw a different shape for each gem type for accessibility
        if gem_type == 1: # Circle (Red)
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        elif gem_type == 2: # Square (Green)
            rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
            pygame.draw.rect(self.screen, color, rect)
        elif gem_type == 3: # Triangle (Blue)
            points = [(x, y - radius), (x - radius, y + radius // 2), (x + radius, y + radius // 2)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        elif gem_type == 4: # Diamond (Yellow)
            points = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        elif gem_type == 5: # Pentagon (Magenta)
            points = []
            for i in range(5):
                angle = math.pi / 2 + 2 * math.pi * i / 5
                points.append((int(x + radius * math.cos(angle)), int(y + radius * math.sin(angle))))
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        elif gem_type == 6: # Hexagon (Cyan)
            points = []
            for i in range(6):
                angle = 2 * math.pi * i / 6
                points.append((int(x + radius * math.cos(angle)), int(y + radius * math.sin(angle))))
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_ui(self):
        score_text = f"Score: {self.score}"
        moves_text = f"Moves: {self.moves_left}"

        score_surf = self.font_main.render(score_text, True, self.COLOR_SCORE)
        moves_surf = self.font_main.render(moves_text, True, self.COLOR_TEXT)
        
        self.screen.blit(score_surf, (20, 10))
        self.screen.blit(moves_surf, (self.width - moves_surf.get_width() - 20, 10))

        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "Goal Reached!" if self.score >= self.TARGET_SCORE else "Game Over"
            end_surf = self.font_main.render(end_text, True, (255, 255, 255))
            end_rect = end_surf.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(end_surf, end_rect)

    def _create_match_particles(self, matches):
        for r, c in matches:
            gem_type = self.board[r, c]
            if gem_type == 0: continue # This can happen in chains
            color = self.GEM_COLORS[gem_type - 1]
            x = self.board_offset_x + c * self.GEM_SIZE + self.GEM_SIZE // 2
            y = self.board_offset_y + r * self.GEM_SIZE + self.GEM_SIZE // 2
            for _ in range(10): # 10 particles per gem
                self.particles.append({
                    'pos': [x, y],
                    'vel': [self._np_random.uniform(-2, 2), self._np_random.uniform(-2, 2)],
                    'life': self._np_random.integers(15, 30),
                    'color': color,
                    'size': self._np_random.integers(2, 5)
                })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Gem Swap")
    
    running = True
    terminated = False
    
    # Game loop
    while running:
        action = [0, 0, 0] # no-op, buttons released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if terminated:
                    continue
                
                # --- Map keys to actions for human play ---
                movement = 0
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                
                space_held = 1 if pygame.key.get_pressed()[pygame.K_SPACE] else 0
                shift_held = 1 if pygame.key.get_pressed()[pygame.K_SHIFT] else 0
                
                # For turn-based, we only step on a key press
                action = [movement, space_held, shift_held]
                if event.key == pygame.K_SPACE:
                    action[1] = 1

                if action != [0,0,0]:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
    env.close()