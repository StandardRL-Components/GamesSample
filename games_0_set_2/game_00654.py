
# Generated: 2025-08-27T14:21:17.761185
# Source Brief: brief_00654.md
# Brief Index: 654

        
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
        "Controls: Use arrow keys to move the cursor. Hold space while pressing an arrow key to swap tiles."
    )

    game_description = (
        "Swap adjacent gems to create matches of 3 or more. Clear the entire board before you run out of moves!"
    )

    auto_advance = False

    # --- Game Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_COLORS = 5
    TILE_SIZE = 40
    TILE_SPACING = 4
    MAX_MOVES = 30
    MAX_STEPS = 1000

    # --- Rewards ---
    REWARD_PER_TILE = 1.0
    REWARD_COMBO_BONUS = 5.0
    REWARD_INVALID_SWAP = -0.1
    REWARD_WIN = 100.0

    # --- Colors ---
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 55)
    TILE_COLORS = [
        (220, 50, 50),   # Red
        (50, 220, 50),   # Green
        (50, 100, 220),  # Blue
        (220, 220, 50),  # Yellow
        (180, 50, 220),  # Purple
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        
        self.grid_pixel_width = self.GRID_WIDTH * (self.TILE_SIZE + self.TILE_SPACING) - self.TILE_SPACING
        self.grid_pixel_height = self.GRID_HEIGHT * (self.TILE_SIZE + self.TILE_SPACING) - self.TILE_SPACING
        self.grid_top_left_x = (640 - self.grid_pixel_width) // 2
        self.grid_top_left_y = (400 - self.grid_pixel_height) // 2

        self.board = None
        self.cursor_pos = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = []
        self.last_matched_tiles = []
        self.last_swap_info = {}

        self.reset()
        
        try:
            self.validate_implementation()
        except AssertionError as e:
            print(f"Implementation validation failed: {e}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.particles = []
        self.last_matched_tiles = []
        self.last_swap_info = {}

        self._create_board()
        while not self._find_all_possible_moves():
            self._create_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.last_matched_tiles = []
        self.last_swap_info = {}
        
        reward = 0
        terminated = False

        movement, space_held, _ = action
        space_pressed = space_held == 1

        # --- Action Handling ---
        cursor_moved = False
        if movement > 0:
            original_pos = list(self.cursor_pos)
            if movement == 1: self.cursor_pos[1] -= 1
            elif movement == 2: self.cursor_pos[1] += 1
            elif movement == 3: self.cursor_pos[0] -= 1
            elif movement == 4: self.cursor_pos[0] += 1
            
            self.cursor_pos[0] %= self.GRID_WIDTH
            self.cursor_pos[1] %= self.GRID_HEIGHT
            cursor_moved = original_pos != self.cursor_pos

        # A swap action is a move + spacebar
        if space_pressed and movement > 0:
            reward, terminated = self._handle_swap(movement)
        elif not cursor_moved and movement == 0:
            # Penalty for doing nothing, encourages action
            reward = -0.01

        if not terminated:
            if self.moves_left <= 0:
                terminated = True
            elif np.sum(self.board) == 0: # All tiles cleared
                reward += self.REWARD_WIN
                terminated = True
            elif self.steps >= self.MAX_STEPS:
                terminated = True
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_swap(self, direction):
        self.moves_left -= 1
        r, c = self.cursor_pos
        
        r2, c2 = r, c
        if direction == 1: r2 -= 1
        elif direction == 2: r2 += 1
        elif direction == 3: c2 -= 1
        elif direction == 4: c2 += 1
        
        # Handle wrapping for the swap target
        r2 %= self.GRID_HEIGHT
        c2 %= self.GRID_WIDTH

        # Perform swap
        self.board[r, c], self.board[r2, c2] = self.board[r2, c2], self.board[r, c]
        self.last_swap_info = {'pos1': (r, c), 'pos2': (r2, c2)}

        # Check for matches
        matches = self._find_all_matches()
        if not matches:
            # Invalid swap, swap back
            self.board[r, c], self.board[r2, c2] = self.board[r2, c2], self.board[r, c]
            return self.REWARD_INVALID_SWAP, False

        # Valid swap, process chain reactions
        total_reward = 0
        chain_level = 0
        while matches:
            cleared_count = len(matches)
            total_reward += cleared_count * self.REWARD_PER_TILE
            if cleared_count > 3:
                total_reward += self.REWARD_COMBO_BONUS
            if chain_level > 0:
                total_reward += self.REWARD_COMBO_BONUS * chain_level

            # Clear matched tiles and spawn particles
            for r_m, c_m in matches:
                # sfx: match_sound.wav
                self._spawn_particles(r_m, c_m, self.board[r_m, c_m])
                self.board[r_m, c_m] = 0 # 0 for empty
                self.last_matched_tiles.append((r_m, c_m))
            
            self._apply_gravity_and_refill()
            
            matches = self._find_all_matches()
            chain_level += 1

        # Anti-softlock: reshuffle if no moves are possible
        if not self._find_all_possible_moves() and np.sum(self.board) > 0:
            self._reshuffle_board()
            # sfx: reshuffle_sound.wav

        return total_reward, False

    def _create_board(self):
        self.board = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        # Ensure no matches on creation
        while self._find_all_matches():
            matches = self._find_all_matches()
            for r, c in matches:
                # Replace with a color different from its neighbors
                forbidden_colors = {self.board[r, c]}
                if r > 0: forbidden_colors.add(self.board[r-1, c])
                if r < self.GRID_HEIGHT - 1: forbidden_colors.add(self.board[r+1, c])
                if c > 0: forbidden_colors.add(self.board[r, c-1])
                if c < self.GRID_WIDTH - 1: forbidden_colors.add(self.board[r, c+1])
                
                possible_colors = list(set(range(1, self.NUM_COLORS + 1)) - forbidden_colors)
                if not possible_colors: possible_colors = list(range(1, self.NUM_COLORS + 1))
                
                self.board[r, c] = self.np_random.choice(possible_colors)

    def _find_all_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.board[r, c] != 0 and self.board[r, c] == self.board[r, c+1] == self.board[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        # Vertical
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.board[r, c] != 0 and self.board[r, c] == self.board[r+1, c] == self.board[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[r, c] != 0:
                    if r != empty_row:
                        self.board[empty_row, c] = self.board[r, c]
                        self.board[r, c] = 0
                    empty_row -= 1
            # Refill
            for r in range(empty_row, -1, -1):
                self.board[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _find_all_possible_moves(self):
        moves = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Swap right
                if c < self.GRID_WIDTH - 1:
                    self.board[r, c], self.board[r, c+1] = self.board[r, c+1], self.board[r, c]
                    if self._find_all_matches(): moves.append(((r,c), (r,c+1)))
                    self.board[r, c], self.board[r, c+1] = self.board[r, c+1], self.board[r, c] # Swap back
                # Swap down
                if r < self.GRID_HEIGHT - 1:
                    self.board[r, c], self.board[r+1, c] = self.board[r+1, c], self.board[r, c]
                    if self._find_all_matches(): moves.append(((r,c), (r+1,c)))
                    self.board[r, c], self.board[r+1, c] = self.board[r+1, c], self.board[r, c] # Swap back
        return moves

    def _reshuffle_board(self):
        flat_board = self.board.flatten()
        non_empty_tiles = flat_board[flat_board != 0]
        self.np_random.shuffle(non_empty_tiles)
        
        new_board = np.zeros_like(self.board)
        fill_idx = 0
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.board[r,c] != 0:
                    new_board[r,c] = non_empty_tiles[fill_idx]
                    fill_idx += 1
        self.board = new_board
        
        # Keep reshuffling until a valid move exists or we give up
        for _ in range(10): # Max 10 reshuffles to prevent infinite loop
            if self._find_all_matches():
                self._create_board() # Start over if board has matches
            elif self._find_all_possible_moves():
                return # Found a valid state
            else: # No matches, no moves, shuffle again
                self.np_random.shuffle(self.board.ravel())
        # If still no moves, just create a fresh board
        self._create_board()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_top_left_x, self.grid_top_left_y, self.grid_pixel_width, self.grid_pixel_height)
        self._draw_rounded_rect(self.screen, grid_rect, self.COLOR_GRID, 10)

        # Update and draw particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            alpha = max(0, min(255, int(p['life'] * 5)))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0]-p['size']), int(p['pos'][1]-p['size'])))

        # Draw tiles
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                tile_val = self.board[r, c]
                if tile_val == 0: continue

                x = self.grid_top_left_x + c * (self.TILE_SIZE + self.TILE_SPACING)
                y = self.grid_top_left_y + r * (self.TILE_SIZE + self.TILE_SPACING)
                rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
                color = self.TILE_COLORS[tile_val - 1]

                # Highlight effect for recently matched tiles
                if (r, c) in self.last_matched_tiles:
                    pulse = abs(math.sin(self.steps * 0.8))
                    highlight_color = (255, 255, 255)
                    self._draw_rounded_rect(self.screen, rect.inflate(4,4), highlight_color, 8, alpha=int(150*pulse))

                self._draw_rounded_rect(self.screen, rect, color, 8)
                
        # Draw cursor
        cursor_x = self.grid_top_left_x + self.cursor_pos[0] * (self.TILE_SIZE + self.TILE_SPACING)
        cursor_y = self.grid_top_left_y + self.cursor_pos[1] * (self.TILE_SIZE + self.TILE_SPACING)
        cursor_rect = pygame.Rect(cursor_x - 3, cursor_y - 3, self.TILE_SIZE + 6, self.TILE_SIZE + 6)
        
        # Breathing effect for cursor
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 # 0 to 1
        alpha = 150 + pulse * 105
        width = int(2 + pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width, border_radius=12, )

    def _render_ui(self):
        self._draw_text(f"Score: {self.score}", (30, 25), self.font_large, anchor="midleft")
        self._draw_text(f"Moves: {self.moves_left}", (610, 25), self.font_large, anchor="midright")
        if self.game_over:
            win_text = "BOARD CLEARED!" if np.sum(self.board) == 0 else "GAME OVER"
            self._draw_text(win_text, (320, 200), self.font_large, anchor="center")

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
        }

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, anchor="center"):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        shadow_surf = font.render(text, True, shadow_color)
        shadow_rect = shadow_surf.get_rect()

        setattr(text_rect, anchor, pos)
        setattr(shadow_rect, anchor, (pos[0] + 2, pos[1] + 2))
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)
        
    def _draw_rounded_rect(self, surface, rect, color, corner_radius, alpha=255):
        if rect.width < 2 * corner_radius or rect.height < 2 * corner_radius:
            return
        
        temp_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        temp_surf.set_alpha(alpha)
        
        pygame.draw.rect(temp_surf, color, (0, 0, rect.width, rect.height), border_radius=corner_radius)
        surface.blit(temp_surf, rect.topleft)

    def _spawn_particles(self, r, c, tile_val):
        x = self.grid_top_left_x + c * (self.TILE_SIZE + self.TILE_SPACING) + self.TILE_SIZE / 2
        y = self.grid_top_left_y + r * (self.TILE_SIZE + self.TILE_SPACING) + self.TILE_SIZE / 2
        color = self.TILE_COLORS[tile_val - 1]
        
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': random.randint(20, 40),
                'size': random.randint(2, 5)
            })

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame window for human interaction ---
    pygame.display.set_caption("Match-3 Gym Environment")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    # Game loop for human player
    while not done:
        action = [0, 0, 0] # no-op, release, release
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        move_action = 0
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Only register an action if a key is pressed to advance the turn
        if move_action > 0 or space_action > 0:
            action = [move_action, space_action, shift_action]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Limit frame rate for human playability
        
    env.close()