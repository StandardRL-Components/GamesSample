
# Generated: 2025-08-28T02:51:47.347463
# Source Brief: brief_01823.md
# Brief Index: 1823

        
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
        "Controls: Use Space and Shift to move the cursor right and down. "
        "Use Arrow Keys to swap the highlighted crystal with an adjacent one."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Swap crystals to match three or more of the "
        "same color. Clear the entire board within the move limit to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 8
        self.NUM_COLORS = 3  # 1: Red, 2: Green, 3: Blue
        self.MAX_MOVES = 15

        # Visual constants
        self.TILE_W_HALF, self.TILE_H_HALF = 24, 12
        self.CRYSTAL_H = 20
        self.OFFSET_X = self.WIDTH // 2
        self.OFFSET_Y = 100

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 80)
        self.COLOR_CURSOR = (255, 255, 100)
        self.CRYSTAL_COLORS = {
            1: ((255, 50, 50), (200, 20, 20), (150, 10, 10)),  # Red
            2: ((50, 255, 50), (20, 200, 20), (10, 150, 10)),  # Green
            3: ((80, 80, 255), (40, 40, 200), (20, 20, 150)),  # Blue
        }
        self.COLOR_TEXT = (220, 220, 240)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_popup = pygame.font.SysFont("monospace", 16, bold=True)

        # State variables (initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.steps = None
        self.last_cleared_info = {} # {'coords': set(), 'score': 0, 'bonus': False}
        self.particles = []

        self.reset()
        self.validate_implementation()
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_grid()
        self.cursor_pos = (self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2)
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.steps = 0
        self.last_cleared_info = {}
        self.particles = []
        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while True:
            matches = self._find_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.last_cleared_info = {}
        reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Action Processing ---
        # 1. Swap attempt (consumes a move)
        swap_attempted = movement > 0
        if swap_attempted:
            self.moves_left -= 1
            r, c = self.cursor_pos
            dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
            nr, nc = r + dr, c + dc

            if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH:
                self._swap_crystals(r, c, nr, nc)
                
                # Process matches and chain reactions
                total_cleared_this_turn = set()
                total_score_this_turn = 0
                
                while True:
                    matches = self._find_matches()
                    if not matches:
                        break # No more matches, chain reaction ends
                    
                    # Score calculation
                    num_cleared = len(matches)
                    reward += num_cleared
                    total_score_this_turn += num_cleared
                    if num_cleared >= 4:
                        reward += 5
                        total_score_this_turn += 5
                        self.last_cleared_info['bonus'] = True
                    
                    # Clear crystals and create particles
                    for mr, mc in matches:
                        self._create_particles(mr, mc, self.grid[mr, mc])
                    total_cleared_this_turn.update(matches)

                    # Remove from grid and apply gravity/refill
                    for mr, mc in sorted(list(matches), key=lambda p: p[0], reverse=True):
                         self._remove_and_drop(mr, mc)
                    self._refill_top_rows()
                
                # If no matches resulted from the swap, swap back. Move is still consumed.
                if not total_cleared_this_turn:
                    self._swap_crystals(r, c, nr, nc) 
                else:
                    self.score += total_score_this_turn
                    self.last_cleared_info['coords'] = total_cleared_this_turn
                    self.last_cleared_info['score'] = total_score_this_turn
            
        # 2. Cursor movement (free action)
        if space_held:
            r, c = self.cursor_pos
            self.cursor_pos = (r, (c + 1) % self.GRID_WIDTH)
        if shift_held:
            r, c = self.cursor_pos
            self.cursor_pos = ((r + 1) % self.GRID_HEIGHT, c)

        # --- Termination Check ---
        terminated = False
        board_cleared = not np.any(self.grid > 0)
        if board_cleared:
            terminated = True
            reward += 100
        elif self.moves_left <= 0:
            terminated = True
            if not board_cleared: # only apply penalty if not a win
                reward -= 100
        
        # Max steps termination
        if self.steps >= 1000:
             terminated = True

        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _swap_crystals(self, r1, c1, r2, c2):
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == 0: continue
                # Horizontal
                if c < self.GRID_WIDTH - 2 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_HEIGHT - 2 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _remove_and_drop(self, r, c):
        for row in range(r, 0, -1):
            self.grid[row, c] = self.grid[row-1, c]
        self.grid[0, c] = 0

    def _refill_top_rows(self):
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT):
                if self.grid[r,c] == 0:
                    self.grid[r,c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _create_particles(self, r, c, color_id):
        # Sound effect placeholder: # sfx_crystal_break()
        sx, sy = self._get_iso_coords(r, c)
        sy += self.TILE_H_HALF
        main_color = self.CRYSTAL_COLORS[color_id][0]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(10, 20)
            self.particles.append([sx, sy, vx, vy, life, main_color])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _get_iso_coords(self, r, c):
        sx = self.OFFSET_X + (c - r) * self.TILE_W_HALF
        sy = self.OFFSET_Y + (c + r) * self.TILE_H_HALF
        return int(sx), int(sy)
    
    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            p1 = self._get_iso_coords(r, 0)
            p2 = self._get_iso_coords(r, self.GRID_WIDTH)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        for c in range(self.GRID_WIDTH + 1):
            p1 = self._get_iso_coords(0, c)
            p2 = self._get_iso_coords(self.GRID_HEIGHT, c)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        
        # Draw crystals
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_id = self.grid[r, c]
                if color_id > 0:
                    is_flashing = (r, c) in self.last_cleared_info.get('coords', set())
                    self._render_crystal(r, c, color_id, is_flashing)

        # Draw cursor
        r, c = self.cursor_pos
        sx, sy = self._get_iso_coords(r, c)
        points = [
            (sx, sy + self.TILE_H_HALF),
            (sx + self.TILE_W_HALF, sy + self.TILE_H_HALF * 2),
            (sx, sy + self.TILE_H_HALF * 3),
            (sx - self.TILE_W_HALF, sy + self.TILE_H_HALF * 2)
        ]
        pygame.draw.aalines(self.screen, self.COLOR_CURSOR, True, points)
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, points, 2)
        
        # Update and draw particles
        for p in self.particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # life -= 1
            alpha = max(0, min(255, int(255 * (p[4] / 20.0))))
            color = p[5]
            radius = max(0, int(p[4]/4))
            # Custom alpha blending for filled circle
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*color, alpha))
            self.screen.blit(temp_surf, (int(p[0]) - radius, int(p[1]) - radius))
        self.particles = [p for p in self.particles if p[4] > 0]

    def _render_crystal(self, r, c, color_id, is_flashing):
        sx, sy = self._get_iso_coords(r, c)
        
        colors = self.CRYSTAL_COLORS[color_id]
        c_top, c_left, c_right = colors[0], colors[1], colors[2]

        if is_flashing:
            c_top = (255, 255, 255)
            c_left = (200, 200, 200)
            c_right = (180, 180, 180)

        # Top face
        points_top = [
            (sx, sy),
            (sx + self.TILE_W_HALF, sy + self.TILE_H_HALF),
            (sx, sy + self.TILE_H_HALF * 2),
            (sx - self.TILE_W_HALF, sy + self.TILE_H_HALF)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points_top, c_top)
        pygame.gfxdraw.filled_polygon(self.screen, points_top, c_top)
        
        # Left face
        points_left = [
            (sx - self.TILE_W_HALF, sy + self.TILE_H_HALF),
            (sx, sy + self.TILE_H_HALF * 2),
            (sx, sy + self.TILE_H_HALF * 2 + self.CRYSTAL_H),
            (sx - self.TILE_W_HALF, sy + self.TILE_H_HALF + self.CRYSTAL_H)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points_left, c_left)
        pygame.gfxdraw.filled_polygon(self.screen, points_left, c_left)

        # Right face
        points_right = [
            (sx + self.TILE_W_HALF, sy + self.TILE_H_HALF),
            (sx, sy + self.TILE_H_HALF * 2),
            (sx, sy + self.TILE_H_HALF * 2 + self.CRYSTAL_H),
            (sx + self.TILE_W_HALF, sy + self.TILE_H_HALF + self.CRYSTAL_H)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points_right, c_right)
        pygame.gfxdraw.filled_polygon(self.screen, points_right, c_right)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Moves left
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 10))
        
        # Score popup
        if 'score' in self.last_cleared_info and self.last_cleared_info['score'] > 0:
            score = self.last_cleared_info['score']
            is_bonus = self.last_cleared_info.get('bonus', False)
            
            popup_text_str = f"+{score}"
            if is_bonus: popup_text_str += " Combo!"
            
            color = (255, 220, 0) if is_bonus else (255, 255, 255)
            popup_text = self.font_popup.render(popup_text_str, True, color)
            
            # Position popup near the cursor
            r, c = self.cursor_pos
            sx, sy = self._get_iso_coords(r, c)
            text_rect = popup_text.get_rect(center=(sx, sy - 20))
            self.screen.blit(popup_text, text_rect)

        # Game Over message
        if self.game_over:
            board_cleared = not np.any(self.grid > 0)
            msg = "YOU WIN!" if board_cleared else "GAME OVER"
            color = (100, 255, 100) if board_cleared else (255, 100, 100)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            
            game_over_text = pygame.font.SysFont("monospace", 48, bold=True).render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            overlay.blit(game_over_text, text_rect)
            self.screen.blit(overlay, (0, 0))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")