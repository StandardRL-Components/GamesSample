
# Generated: 2025-08-27T17:36:40.386467
# Source Brief: brief_01582.md
# Brief Index: 1582

        
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
        "Controls: Use arrow keys to move the selector. "
        "Hold an arrow key and press Space to shift the corresponding row or column."
    )

    game_description = (
        "Strategically shift rows and columns of shimmering crystals to create matches of three or more. "
        "Clear the board before you run out of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_WIDTH = 8
        self.GRID_HEIGHT = 8
        self.NUM_COLORS = 7
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Visuals
        self.TILE_W_HALF = 28
        self.TILE_H_HALF = 14
        self.ORIGIN_X = self.SCREEN_WIDTH // 2
        self.ORIGIN_Y = self.SCREEN_HEIGHT // 2 - self.GRID_HEIGHT * self.TILE_H_HALF // 2 + 20
        self.CRYSTAL_Z_OFFSET = 8

        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 80)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 100)
        self.CRYSTAL_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 150, 255),
            (255, 150, 80), (255, 80, 255), (80, 255, 255),
            (255, 255, 80)
        ]

        # Reward structure
        self.REWARD_MOVE_COST = -0.2
        self.REWARD_WIN = 100
        self.REWARD_LOSS = -10
        self.REWARD_PER_CRYSTAL = 1.0
        self.REWARD_BONUS_MATCH = 5.0

        # Initialize state variables (will be set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 0
        self.grid = None
        self.cursor = (0, 0)
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.cursor = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.particles = []

        while True:
            self._populate_initial_grid()
            if not self._find_matches():
                break
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        action_taken = space_held or shift_held # For this game, space/shift are the same action trigger
        is_shift_move = action_taken and movement in [1, 2, 3, 4]
        is_cursor_move = not action_taken and movement in [1, 2, 3, 4]

        if is_cursor_move:
            # sfx: cursor_move.wav
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            self.cursor = (
                (self.cursor[0] + dx) % self.GRID_WIDTH,
                (self.cursor[1] + dy) % self.GRID_HEIGHT,
            )
        elif is_shift_move:
            # sfx: crystal_shift.wav
            self.moves_left -= 1
            reward += self.REWARD_MOVE_COST
            
            cx, cy = self.cursor
            direction = movement
            
            if direction in [1, 2]: # Up/Down shift column
                col = self.grid[:, cy].copy()
                roll_amount = -1 if direction == 1 else 1
                self.grid[:, cy] = np.roll(col, roll_amount)
            elif direction in [3, 4]: # Left/Right shift row
                row = self.grid[cx, :].copy()
                roll_amount = -1 if direction == 3 else 1
                self.grid[cx, :] = np.roll(row, roll_amount)

            reward += self._handle_cascades()

        # Check termination conditions
        terminated = False
        if np.all(self.grid == 0):
            reward += self.REWARD_WIN
            terminated = True
            # sfx: win_fanfare.wav
        elif self.moves_left <= 0:
            reward += self.REWARD_LOSS
            terminated = True
            # sfx: lose_sound.wav
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _populate_initial_grid(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))

    def _handle_cascades(self):
        total_cascade_reward = 0
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            # sfx: match_clear.wav
            num_cleared = len(matches)
            cascade_reward = num_cleared * self.REWARD_PER_CRYSTAL
            if num_cleared >= 4:
                cascade_reward += self.REWARD_BONUS_MATCH
            
            total_cascade_reward += cascade_reward
            self.score += cascade_reward

            self._clear_matches(matches)
            self._apply_gravity()
            self._refill_board()
        return total_cascade_reward

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[c, r] == 0:
                    continue
                
                # Horizontal check
                if c < self.GRID_WIDTH - 2 and self.grid[c, r] == self.grid[c+1, r] == self.grid[c+2, r]:
                    matches.update([(c, r), (c+1, r), (c+2, r)])
                
                # Vertical check
                if r < self.GRID_HEIGHT - 2 and self.grid[c, r] == self.grid[c, r+1] == self.grid[c, r+2]:
                    matches.update([(c, r), (c, r+1), (c, r+2)])
        return list(matches)

    def _clear_matches(self, matches):
        for c, r in matches:
            if self.grid[c, r] != 0:
                self._create_particles(c, r, self.grid[c, r])
                self.grid[c, r] = 0

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            col = self.grid[c, :]
            non_zeros = col[col != 0]
            new_col = np.zeros(self.GRID_HEIGHT, dtype=int)
            new_col[self.GRID_HEIGHT - len(non_zeros):] = non_zeros
            self.grid[c, :] = new_col

    def _refill_board(self):
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT):
                if self.grid[c, r] == 0:
                    self.grid[c, r] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_base()
        self._render_crystals()
        self._render_cursor()
        self._update_and_render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor,
        }

    def _iso_to_screen(self, x, y, z=0):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_W_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_H_HALF - z
        return int(screen_x), int(screen_y)

    def _render_grid_base(self):
        for r in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, r)
            end = self._iso_to_screen(self.GRID_WIDTH, r)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for c in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(c, 0)
            end = self._iso_to_screen(c, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

    def _render_crystals(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                crystal_type = self.grid[c, r]
                if crystal_type > 0:
                    self._draw_crystal(c, r, crystal_type)

    def _draw_crystal(self, c, r, crystal_type):
        x, y = self._iso_to_screen(c, r)
        color = self.CRYSTAL_COLORS[crystal_type - 1]
        
        # Shimmer effect
        shimmer = (math.sin(self.steps * 0.1 + c * 0.5 + r * 0.3) + 1) / 2
        highlight_color = tuple(min(255, val + 40 * shimmer) for val in color)

        # Points for isometric cube
        top_y = y - self.CRYSTAL_Z_OFFSET
        bottom_y = y
        
        p_top = (x, top_y - self.TILE_H_HALF)
        p_left = (x - self.TILE_W_HALF, top_y)
        p_right = (x + self.TILE_W_HALF, top_y)
        p_center = (x, top_y)
        p_bottom = (x, bottom_y)
        p_bottom_left = (x - self.TILE_W_HALF, bottom_y - self.TILE_H_HALF)
        p_bottom_right = (x + self.TILE_W_HALF, bottom_y - self.TILE_H_HALF)

        # Draw faces
        darker_color = tuple(v * 0.6 for v in color)
        pygame.gfxdraw.filled_polygon(self.screen, [p_left, p_center, p_bottom, p_bottom_left], darker_color)
        pygame.gfxdraw.filled_polygon(self.screen, [p_right, p_center, p_bottom, p_bottom_right], color)
        pygame.gfxdraw.filled_polygon(self.screen, [p_left, p_top, p_right, p_center], highlight_color)

        # Draw outlines
        pygame.gfxdraw.aapolygon(self.screen, [p_left, p_center, p_bottom, p_bottom_left], darker_color)
        pygame.gfxdraw.aapolygon(self.screen, [p_right, p_center, p_bottom, p_bottom_right], color)
        pygame.gfxdraw.aapolygon(self.screen, [p_left, p_top, p_right, p_center], highlight_color)

        # Draw pattern for accessibility
        self._draw_crystal_pattern(x, top_y, crystal_type)

    def _draw_crystal_pattern(self, x, y, crystal_type):
        pat_color = (255, 255, 255, 150)
        s = 4
        if crystal_type == 1: # Circle
            pygame.gfxdraw.filled_circle(self.screen, x, y, s, pat_color)
        elif crystal_type == 2: # Square
            pygame.draw.rect(self.screen, pat_color, (x-s, y-s, 2*s, 2*s))
        elif crystal_type == 3: # Triangle
            pygame.gfxdraw.filled_trigon(self.screen, x, y-s, x-s, y+s, x+s, y+s, pat_color)
        elif crystal_type == 4: # X
            pygame.draw.line(self.screen, pat_color, (x-s, y-s), (x+s, y+s), 2)
            pygame.draw.line(self.screen, pat_color, (x-s, y+s), (x+s, y-s), 2)
        elif crystal_type == 5: # Diamond
            pygame.gfxdraw.filled_polygon(self.screen, [(x, y-s), (x-s, y), (x, y+s), (x+s, y)], pat_color)
        elif crystal_type == 6: # Horizontal lines
            pygame.draw.line(self.screen, pat_color, (x-s, y-s//2), (x+s, y-s//2), 2)
            pygame.draw.line(self.screen, pat_color, (x-s, y+s//2), (x+s, y+s//2), 2)
        elif crystal_type == 7: # Vertical lines
            pygame.draw.line(self.screen, pat_color, (x-s//2, y-s), (x-s//2, y+s), 2)
            pygame.draw.line(self.screen, pat_color, (x+s//2, y-s), (x+s//2, y+s), 2)

    def _render_cursor(self):
        cx, cy = self.cursor
        x, y = self._iso_to_screen(cx, cy)
        
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        alpha = 150 + 105 * pulse
        
        color = self.COLOR_CURSOR + (int(alpha),)
        
        points = [
            self._iso_to_screen(cx, cy, -2),
            self._iso_to_screen(cx + 1, cy, -2),
            self._iso_to_screen(cx + 1, cy + 1, -2),
            self._iso_to_screen(cx, cy + 1, -2),
        ]
        
        pygame.gfxdraw.polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)


    def _create_particles(self, c, r, crystal_type):
        x, y = self._iso_to_screen(c, r, self.CRYSTAL_Z_OFFSET)
        color = self.CRYSTAL_COLORS[crystal_type - 1]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 2 # Add upward bias
            life = random.randint(20, 40)
            self.particles.append([x, y, vx, vy, life, color])

    def _update_and_render_particles(self):
        active_particles = []
        for p in self.particles:
            p[0] += p[1] # x += vx
            p[0] += random.uniform(-0.5, 0.5) # Wiggle
            p[2] += p[3] # y += vy
            p[3] += 0.2  # Gravity
            p[4] -= 1    # life -= 1
            if p[4] > 0:
                active_particles.append(p)
                size = max(0, int(p[4] / 8))
                alpha = max(0, min(255, p[4] * 10))
                color_with_alpha = p[5] + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[2]), size, color_with_alpha)
        self.particles = active_particles

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 20))

        if self.game_over:
            outcome_text_str = "BOARD CLEARED!" if np.all(self.grid == 0) else "OUT OF MOVES"
            outcome_text = self.font_large.render(outcome_text_str, True, self.COLOR_CURSOR)
            text_rect = outcome_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 50))
            self.screen.blit(outcome_text, text_rect)

    def close(self):
        pygame.quit()

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Play ---
    # This part is for testing and demonstration purposes.
    # It will not be part of the final single-file submission.
    
    obs, info = env.reset()
    done = False
    
    # Create a display for human play
    pygame.display.set_caption("Crystal Caverns")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    print(env.user_guide)
    
    while not done:
        # Construct action from keyboard input
        keys = pygame.key.get_pressed()
        
        move_action = 0
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        
        # Poll for events
        should_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                # For turn-based, we step on any key press
                should_step = True
        
        if should_step:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")

        # Update display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human play
        
    env.close()