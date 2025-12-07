
# Generated: 2025-08-28T05:16:07.707814
# Source Brief: brief_05512.md
# Brief Index: 5512

        
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
        "Controls: ←→ to select a column. SPACE to pick up the next crystal. "
        "SHIFT to drop it in the selected column. SPACE to cancel a drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Drop crystals into the grid to form lines of five "
        "of the same color (horizontally, vertically, or diagonally) to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_size = (640, 400)
        self.screen = pygame.Surface(self.screen_size)
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game Constants ---
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 12
        self.MAX_STEPS = 500
        
        # --- Visual Constants ---
        self.ISO_TILE_WIDTH_HALF = 20
        self.ISO_TILE_HEIGHT_HALF = 10
        self.CRYSTAL_HEIGHT_OFFSET = 15

        self.ORIGIN_X = self.screen_size[0] // 2
        self.ORIGIN_Y = 80

        self.COLOR_BG = (25, 20, 35)
        self.COLOR_GRID = (60, 50, 80)
        self.COLOR_TEXT = (240, 240, 255)
        self.COLOR_TEXT_SHADOW = (10, 10, 15)
        self.CRYSTAL_COLORS = [
            (0, 0, 0), # 0: Empty
            (255, 80, 80),   # 1: Red
            (80, 255, 80),   # 2: Green
            (80, 120, 255),  # 3: Blue
            (255, 255, 80),  # 4: Yellow
            (200, 80, 255),  # 5: Purple
        ]
        self.CRYSTAL_HIGHLIGHT_COLORS = [
            (0,0,0),
            (255, 150, 150),
            (150, 255, 150),
            (150, 180, 255),
            (255, 255, 150),
            (230, 150, 255),
        ]

        # --- State Variables ---
        self.grid = None
        self.cursor_x = 0
        self.game_phase = "SELECT"  # 'SELECT', 'DROP'
        self.next_crystal_color_idx = 1
        self.selected_column = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        self.reset()
        # self.validate_implementation() # Uncomment to run validation
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        # Create an interesting, uneven starting floor
        for _ in range(self.GRID_WIDTH * 2):
            col = self.np_random.integers(0, self.GRID_WIDTH)
            color_idx = self.np_random.integers(1, len(self.CRYSTAL_COLORS))
            self._drop_crystal(col, color_idx)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []
        self.cursor_x = self.GRID_WIDTH // 2
        self.game_phase = "SELECT"
        self.selected_column = None
        self._generate_next_crystal()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = -0.01 # Small cost for taking a step
        terminated = False

        if self.game_phase == "SELECT":
            if movement == 3:  # Left
                self.cursor_x = max(0, self.cursor_x - 1)
            elif movement == 4:  # Right
                self.cursor_x = min(self.GRID_WIDTH - 1, self.cursor_x + 1)

            if space_press:
                if self.grid[0][self.cursor_x] == 0:
                    self.selected_column = self.cursor_x
                    self.game_phase = "DROP"
                    # SFX: select_crystal
                else:
                    reward -= 0.1 # Penalty for selecting a full column
        
        elif self.game_phase == "DROP":
            if shift_press:
                drop_row = self._drop_crystal(self.selected_column, self.next_crystal_color_idx)
                
                if drop_row is not None:
                    # SFX: crystal_land
                    match_reward, is_win = self._check_and_score_matches(drop_row, self.selected_column)
                    reward += match_reward
                    self.score += int(match_reward)

                    if is_win:
                        reward += 50
                        self.score += 50
                        terminated = True
                        self.game_over = True
                        self.win_message = "YOU WIN!"
                        # SFX: win_jingle
                    else:
                        if self._is_board_full():
                            reward -= 10
                            terminated = True
                            self.game_over = True
                            self.win_message = "GAME OVER"
                            # SFX: loss_sound
                
                    self._generate_next_crystal()
                    self.game_phase = "SELECT"
                else:
                    # Should not be reachable due to check in SELECT phase
                    self.game_phase = "SELECT"
                    reward -= 1

            elif space_press: # Cancel drop
                self.game_phase = "SELECT"
                self.selected_column = None
                # SFX: cancel_action

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True
            self.win_message = "TIME UP"
            reward -= 10

        self._update_particles()
        
        return self._get_observation(), float(reward), terminated, False, self._get_info()

    def _iso_to_screen(self, r, c):
        x = self.ORIGIN_X + (c - r) * self.ISO_TILE_WIDTH_HALF
        y = self.ORIGIN_Y + (c + r) * self.ISO_TILE_HEIGHT_HALF
        return int(x), int(y)

    def _drop_crystal(self, col, color_idx):
        for r in range(self.GRID_HEIGHT - 1, -1, -1):
            if self.grid[r][col] == 0:
                self.grid[r][col] = color_idx
                return r
        return None

    def _check_and_score_matches(self, r, c):
        total_reward = 0
        is_win = False
        color = self.grid[r][c]
        if color == 0: return 0, False

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        all_matched_coords = set()

        for dr, dc in directions:
            line_coords = [(r, c)]
            # Positive direction
            for i in range(1, 5):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH and self.grid[nr][nc] == color:
                    line_coords.append((nr, nc))
                else:
                    break
            # Negative direction
            for i in range(1, 5):
                nr, nc = r - i * dr, c - i * dc
                if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH and self.grid[nr][nc] == color:
                    line_coords.append((nr, nc))
                else:
                    break
            
            count = len(line_coords)
            if count >= 2:
                reward_map = {2: 1, 3: 2, 4: 3, 5: 10}
                line_reward = reward_map.get(min(count, 5), 0)
                total_reward += line_reward
                if count >= 5:
                    is_win = True
                
                for coord in line_coords:
                    all_matched_coords.add(coord)

        for pr, pc in all_matched_coords:
            self._spawn_match_particles(pr, pc, self.grid[pr][pc])

        return total_reward, is_win

    def _spawn_match_particles(self, r, c, color_idx):
        sx, sy = self._iso_to_screen(r, c)
        sy -= self.CRYSTAL_HEIGHT_OFFSET // 2
        base_color = self.CRYSTAL_COLORS[color_idx]
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'x': sx, 'y': sy,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': life, 'max_life': life,
                'color': base_color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vx'] *= 0.95
            p['vy'] *= 0.95
            p['life'] -= 1

    def _is_board_full(self):
        return np.all(self.grid[0, :] != 0)

    def _generate_next_crystal(self):
        self.next_crystal_color_idx = self.np_random.integers(1, len(self.CRYSTAL_COLORS))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid_and_crystals()
        self._render_cursor_and_selection()
        self._render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_and_crystals(self):
        # Render grid lines
        for r in range(self.GRID_HEIGHT + 1):
            p1 = self._iso_to_screen(r, 0)
            p2 = self._iso_to_screen(r, self.GRID_WIDTH)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        for c in range(self.GRID_WIDTH + 1):
            p1 = self._iso_to_screen(0, c)
            p2 = self._iso_to_screen(self.GRID_HEIGHT, c)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        
        # Render crystals
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.grid[r][c]
                if color_idx > 0:
                    sx, sy = self._iso_to_screen(r, c)
                    self._draw_crystal(self.screen, sx, sy, color_idx)

    def _draw_crystal(self, surface, x, y, color_idx):
        top_face_points = [
            (x, y - self.CRYSTAL_HEIGHT_OFFSET),
            (x + self.ISO_TILE_WIDTH_HALF, y - self.CRYSTAL_HEIGHT_OFFSET + self.ISO_TILE_HEIGHT_HALF),
            (x, y),
            (x - self.ISO_TILE_WIDTH_HALF, y - self.CRYSTAL_HEIGHT_OFFSET + self.ISO_TILE_HEIGHT_HALF),
        ]
        
        color = self.CRYSTAL_COLORS[color_idx]
        highlight_color = self.CRYSTAL_HIGHLIGHT_COLORS[color_idx]
        shadow_color = tuple(max(0, val - 50) for val in color)

        left_face_points = [ (x,y), (x - self.ISO_TILE_WIDTH_HALF, y + self.ISO_TILE_HEIGHT_HALF), (x - self.ISO_TILE_WIDTH_HALF, y - self.CRYSTAL_HEIGHT_OFFSET + self.ISO_TILE_HEIGHT_HALF), (x, y - self.CRYSTAL_HEIGHT_OFFSET) ]
        right_face_points = [ (x,y), (x + self.ISO_TILE_WIDTH_HALF, y + self.ISO_TILE_HEIGHT_HALF), (x + self.ISO_TILE_WIDTH_HALF, y - self.CRYSTAL_HEIGHT_OFFSET + self.ISO_TILE_HEIGHT_HALF), (x, y - self.CRYSTAL_HEIGHT_OFFSET) ]

        pygame.gfxdraw.filled_polygon(surface, left_face_points, shadow_color)
        pygame.gfxdraw.filled_polygon(surface, right_face_points, shadow_color)
        pygame.gfxdraw.filled_polygon(surface, top_face_points, color)
        pygame.gfxdraw.aapolygon(surface, top_face_points, highlight_color)

    def _render_cursor_and_selection(self):
        if self.game_over: return

        if self.game_phase == "SELECT":
            # Draw cursor over the selected column at the top
            sx, sy = self._iso_to_screen(-0.5, self.cursor_x)
            points = [
                (sx, sy),
                (sx + self.ISO_TILE_WIDTH_HALF, sy + self.ISO_TILE_HEIGHT_HALF),
                (sx, sy + self.ISO_TILE_HEIGHT_HALF * 2),
                (sx - self.ISO_TILE_WIDTH_HALF, sy + self.ISO_TILE_HEIGHT_HALF),
            ]
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            alpha = int(100 + pulse * 100)
            pygame.gfxdraw.filled_polygon(self.screen, points, (255, 255, 255, alpha))
            pygame.gfxdraw.aapolygon(self.screen, points, (255, 255, 255))
        
        elif self.game_phase == "DROP":
            # Highlight the entire column
            c = self.selected_column
            for r in range(self.GRID_HEIGHT):
                if self.grid[r][c] == 0:
                    sx, sy = self._iso_to_screen(r, c)
                    points = [
                        (sx, sy),
                        (sx + self.ISO_TILE_WIDTH_HALF, sy + self.ISO_TILE_HEIGHT_HALF),
                        (sx, sy + self.ISO_TILE_HEIGHT_HALF * 2),
                        (sx - self.ISO_TILE_WIDTH_HALF, sy + self.ISO_TILE_HEIGHT_HALF),
                    ]
                    pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
                    alpha = int(50 + pulse * 50)
                    pygame.gfxdraw.filled_polygon(self.screen, points, (255, 255, 255, alpha))

            # Draw the crystal to be dropped
            sx, sy = self._iso_to_screen(-0.5, self.selected_column)
            self._draw_crystal(self.screen, sx, sy, self.next_crystal_color_idx)

    def _render_particles(self):
        for p in self.particles:
            alpha = (p['life'] / p['max_life'])
            radius = int(alpha * 5)
            color = p['color']
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), radius, (*color, int(alpha*255)))

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (15, 10), self.font_small)
        
        # Next Crystal
        self._draw_text("NEXT:", (540, 10), self.font_small)
        if self.game_phase == "SELECT":
            self._draw_crystal(self.screen, 585, 50, self.next_crystal_color_idx)

        # Game Over Message
        if self.game_over:
            self._draw_text(self.win_message, (self.screen_size[0] // 2, self.screen_size[1] // 2 - 20), self.font_large, center=True)

    def _draw_text(self, text, pos, font, color=None, shadow_color=None, center=False):
        color = color or self.COLOR_TEXT
        shadow_color = shadow_color or self.COLOR_TEXT_SHADOW
        
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        print("Running implementation validation...")
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
        assert 'score' in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert 'steps' in info

        # Test initial board has valid moves
        self.reset()
        assert not self._is_board_full(), "Initial board should not be full"
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # For human play, we need to detect key presses, not holds
    key_cooldown = 0
    COOLDOWN_TIME = 150 # ms

    # Create a display for the game
    display_screen = pygame.display.set_mode(env.screen_size)
    pygame.display.set_caption("Crystal Cave")
    game_clock = pygame.time.Clock()

    # Game loop
    while running:
        time_delta = game_clock.tick(60)
        action = np.array([0, 0, 0])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Handle key presses with a cooldown for turn-based action
        now = pygame.time.get_ticks()
        meaningful_action = False
        if now > key_cooldown:
            if keys[pygame.K_LEFT]:
                action[0] = 3
                meaningful_action = True
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
                meaningful_action = True
            
            if keys[pygame.K_SPACE]:
                action[1] = 1
                meaningful_action = True
                
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1
                meaningful_action = True
        
        if not terminated and meaningful_action:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            key_cooldown = now + COOLDOWN_TIME
        
        # Get the current frame from the environment
        frame = obs if 'obs' in locals() else env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        
        # Display the frame
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print("Game Over! Press R to reset.")
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                        wait_for_reset = False
                        key_cooldown = pygame.time.get_ticks() + COOLDOWN_TIME * 2
        
    env.close()