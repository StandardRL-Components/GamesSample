
# Generated: 2025-08-28T00:21:30.204704
# Source Brief: brief_03766.md
# Brief Index: 3766

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a tile. "
        "Move the cursor to an adjacent tile and press Space again to swap. "
        "Press Shift to deselect."
    )

    game_description = (
        "A colorful match-3 puzzle game. Swap adjacent tiles to create lines of 3 or more of the "
        "same color. Clear 50% of the board in 25 moves to win. Create chain reactions for bonus points!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 8
        self.NUM_COLORS = 6
        self.TILE_SIZE = 40
        self.BOARD_OFFSET_X = (640 - self.GRID_WIDTH * self.TILE_SIZE) // 2
        self.BOARD_OFFSET_Y = (400 - self.GRID_HEIGHT * self.TILE_SIZE) // 2 + 20
        self.MAX_MOVES = 25
        self.WIN_PERCENTAGE = 0.5
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (44, 62, 80)  # Dark blue-gray
        self.COLOR_GRID = (52, 73, 94)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_CURSOR = (241, 196, 15) # Yellow
        self.TILE_COLORS = [
            (231, 76, 60),    # Red
            (46, 204, 113),   # Green
            (52, 152, 219),   # Blue
            (241, 196, 15),   # Yellow
            (155, 89, 182),   # Purple
            (230, 126, 34),   # Orange
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)

        # Game state variables (initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.visual_cursor_pos = None
        self.selected_tile = None
        self.moves_left = 0
        self.score = 0
        self.tiles_cleared = 0
        self.initial_tile_count = self.GRID_WIDTH * self.GRID_HEIGHT
        self.game_over = False
        self.steps = 0
        self.particles = []
        self.last_action_reward = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.tiles_cleared = 0
        self.game_over = False
        self.selected_tile = None
        self.cursor_pos = (self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2)
        self.visual_cursor_pos = (
            self.BOARD_OFFSET_X + self.cursor_pos[1] * self.TILE_SIZE,
            self.BOARD_OFFSET_Y + self.cursor_pos[0] * self.TILE_SIZE,
        )
        self.particles = []
        self.last_action_reward = 0

        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # --- Action Handling ---
        action_taken = False
        if shift_pressed and self.selected_tile is not None:
            self.selected_tile = None
            action_taken = True
            # sfx: deselect_sound

        elif movement != 0:
            r, c = self.cursor_pos
            if movement == 1: r = max(0, r - 1) # Up
            elif movement == 2: r = min(self.GRID_HEIGHT - 1, r + 1) # Down
            elif movement == 3: c = max(0, c - 1) # Left
            elif movement == 4: c = min(self.GRID_WIDTH - 1, c + 1) # Right
            self.cursor_pos = (r, c)
            action_taken = True
        
        if space_pressed:
            action_taken = True
            if self.selected_tile is None:
                self.selected_tile = self.cursor_pos
                # sfx: select_sound
            else:
                # Attempt a swap
                if self._is_adjacent(self.selected_tile, self.cursor_pos):
                    reward = self._handle_swap(self.selected_tile, self.cursor_pos)
                    self.moves_left -= 1
                else: # Invalid swap (not adjacent)
                    self.selected_tile = None # Deselect
                    # sfx: error_sound
        
        self.last_action_reward = reward

        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if (self.tiles_cleared / self.initial_tile_count) >= self.WIN_PERCENTAGE:
                reward += 100 # Win bonus
                # sfx: win_sound
            else:
                reward -= 10 # Loss penalty
                # sfx: lose_sound

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_render_particles()
        self._render_grid_and_tiles()
        self._render_cursor_and_selection()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cleared_percent": self.tiles_cleared / self.initial_tile_count,
        }

    # --- Game Logic ---
    
    def _generate_board(self):
        self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while self._find_all_matches():
            self._remove_matches(self._find_all_matches())
            self._apply_gravity()
            self._refill_board()

    def _is_adjacent(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2) == 1

    def _handle_swap(self, pos1, pos2):
        # Perform swap
        r1, c1 = pos1
        r2, c2 = pos2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

        # Check for matches
        matches = self._find_all_matches()
        if not matches:
            # No match, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            self.selected_tile = None
            # sfx: invalid_swap_sound
            return -0.2
        else:
            # Match found, process it
            self.selected_tile = None
            total_reward = 0
            chain_level = 0
            while matches:
                # sfx: match_sound
                if chain_level > 0:
                    total_reward += 10 # Chain reaction bonus
                
                num_cleared = len(matches)
                self.tiles_cleared += num_cleared
                total_reward += num_cleared # +1 per tile
                self.score += num_cleared * (chain_level + 1)
                
                if num_cleared > 3:
                    total_reward += 5 # Large match bonus

                self._remove_matches(matches)
                self._apply_gravity()
                self._refill_board()
                
                matches = self._find_all_matches()
                chain_level += 1
            return total_reward

    def _find_all_matches(self):
        matches = set()
        # Horizontal matches
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.add((r, c)); matches.add((r, c+1)); matches.add((r, c+2))
        # Vertical matches
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.add((r, c)); matches.add((r+1, c)); matches.add((r+2, c))
        return matches

    def _remove_matches(self, matches):
        for r, c in matches:
            self._create_particles(r, c, self.grid[r, c])
            self.grid[r, c] = -1 # Mark as empty

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
    
    def _refill_board(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_COLORS)

    def _check_termination(self):
        if self.moves_left <= 0:
            return True
        if (self.tiles_cleared / self.initial_tile_count) >= self.WIN_PERCENTAGE:
            return True
        return False

    # --- Rendering ---

    def _render_grid_and_tiles(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.BOARD_OFFSET_X, self.BOARD_OFFSET_Y, self.GRID_WIDTH * self.TILE_SIZE, self.GRID_HEIGHT * self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)

        # Draw tiles
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.grid[r, c]
                if color_idx != -1:
                    tile_rect = pygame.Rect(
                        self.BOARD_OFFSET_X + c * self.TILE_SIZE + 2,
                        self.BOARD_OFFSET_Y + r * self.TILE_SIZE + 2,
                        self.TILE_SIZE - 4,
                        self.TILE_SIZE - 4
                    )
                    color = self.TILE_COLORS[color_idx]
                    pygame.draw.rect(self.screen, color, tile_rect, border_radius=6)
                    # Add a subtle highlight for 3D effect
                    highlight_color = tuple(min(255, x + 30) for x in color)
                    pygame.gfxdraw.arc(self.screen, tile_rect.centerx, tile_rect.centery, tile_rect.width//2 - 2, 135, 315, highlight_color)


    def _render_cursor_and_selection(self):
        # Smoothly interpolate visual cursor position
        target_x = self.BOARD_OFFSET_X + self.cursor_pos[1] * self.TILE_SIZE
        target_y = self.BOARD_OFFSET_Y + self.cursor_pos[0] * self.TILE_SIZE
        self.visual_cursor_pos = (
            self.visual_cursor_pos[0] * 0.6 + target_x * 0.4,
            self.visual_cursor_pos[1] * 0.6 + target_y * 0.4
        )
        
        # Draw cursor
        cursor_rect = pygame.Rect(
            int(self.visual_cursor_pos[0]), int(self.visual_cursor_pos[1]),
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=3, border_radius=8)
        
        # Draw selection highlight
        if self.selected_tile is not None:
            r, c = self.selected_tile
            selection_rect = pygame.Rect(
                self.BOARD_OFFSET_X + c * self.TILE_SIZE,
                self.BOARD_OFFSET_Y + r * self.TILE_SIZE,
                self.TILE_SIZE, self.TILE_SIZE
            )
            # Pulsing alpha effect for selection
            alpha = 128 + math.sin(pygame.time.get_ticks() * 0.005) * 64
            overlay = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(overlay, (255, 255, 255, alpha), overlay.get_rect(), border_radius=8)
            self.screen.blit(overlay, selection_rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Moves
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.screen.get_width() - moves_text.get_width() - 15, 10))
        
        # Cleared percentage bar
        cleared_ratio = min(1.0, (self.tiles_cleared / self.initial_tile_count) / self.WIN_PERCENTAGE)
        bar_width = 400
        bar_height = 20
        bar_x = (self.screen.get_width() - bar_width) // 2
        bar_y = self.screen.get_height() - bar_height - 10
        
        # Background bar
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        # Fill bar
        fill_width = int(bar_width * cleared_ratio)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.TILE_COLORS[2], (bar_x, bar_y, fill_width, bar_height), border_radius=5)
        # Win marker
        win_marker_x = bar_x + bar_width
        pygame.draw.line(self.screen, self.COLOR_TEXT, (win_marker_x, bar_y), (win_marker_x, bar_y + bar_height), 2)
        
        percent_text = self.font_small.render(f"{int(self.tiles_cleared / self.initial_tile_count * 100)}% Cleared", True, self.COLOR_TEXT)
        self.screen.blit(percent_text, (bar_x + bar_width / 2 - percent_text.get_width() / 2, bar_y + 2))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            won = (self.tiles_cleared / self.initial_tile_count) >= self.WIN_PERCENTAGE
            msg = "YOU WIN!" if won else "GAME OVER"
            color = self.TILE_COLORS[1] if won else self.TILE_COLORS[0]
            
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.screen.get_width() / 2 - end_text.get_width() / 2, self.screen.get_height() / 2 - end_text.get_height() / 2))

    # --- Particles ---
    
    def _create_particles(self, r, c, color_idx):
        px = self.BOARD_OFFSET_X + c * self.TILE_SIZE + self.TILE_SIZE / 2
        py = self.BOARD_OFFSET_Y + r * self.TILE_SIZE + self.TILE_SIZE / 2
        color = self.TILE_COLORS[color_idx]
        for _ in range(20): # Create 20 particles
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.randint(20, 40)
            self.particles.append([px, py, vx, vy, lifetime, color])

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # lifetime -= 1
            if p[4] <= 0:
                self.particles.remove(p)
            else:
                # Fade out effect
                alpha = int(255 * (p[4] / 40))
                color = p[5] + (alpha,)
                radius = int(p[4] / 8)
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), radius, color)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Match-3 Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
        
        # Only step if an action was taken
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print("Game Over! Press 'R' to restart.")

        # Render the environment observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Run at 60 FPS for smooth visuals

    pygame.quit()