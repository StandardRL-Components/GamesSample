
# Generated: 2025-08-27T23:48:19.231919
# Source Brief: brief_03580.md
# Brief Index: 3580

        
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
        "Controls: ↑↓←→ to move cursor. Press space to select a tile. "
        "Move to an adjacent tile and press space again to swap. Press shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent gems to create matches of 3 or more. Clear the board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Game constants
        self.GRID_WIDTH = 8
        self.GRID_HEIGHT = 8
        self.NUM_COLORS = 5
        self.MAX_MOVES = 50
        self.MAX_STEPS = 1000
        self.TILE_SIZE = 40
        self.GRID_OFFSET_X = (640 - self.GRID_WIDTH * self.TILE_SIZE) // 2
        self.GRID_OFFSET_Y = (400 - self.GRID_HEIGHT * self.TILE_SIZE) // 2
        self.GEM_SIZE = int(self.TILE_SIZE * 0.8)
        
        # Colors & Fonts
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_CURSOR = (255, 255, 0, 100)
        self.COLOR_SELECT = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.UI_FONT = pygame.font.Font(None, 32)
        self.UI_FONT_LARGE = pygame.font.Font(None, 64)

        # Game state variables
        self.grid = None
        self.cursor_pos = None
        self.selected_tile = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tile = None
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self._generate_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        
        reward += self._handle_input(action)
        
        # Update particles for one frame (for static burst effect)
        self.particles = [p for p in self.particles if p[4] > 0]
        for i in range(len(self.particles)):
            p = self.particles[i]
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1 # Decrement lifespan
            self.particles[i] = p

        if not self.game_over:
            if self.moves_left <= 0:
                self.game_over = True
                terminated = True
                reward = -10.0
            elif self._is_board_clear():
                self.game_over = True
                terminated = True
                reward += 100.0
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        moved = False
        if movement == 1: # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
            moved = True
        elif movement == 2: # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_HEIGHT
            moved = True
        elif movement == 3: # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
            moved = True
        elif movement == 4: # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_WIDTH
            moved = True
        
        if moved and self.selected_tile and self.grid[self.cursor_pos[1], self.cursor_pos[0]] == -1:
            # Don't allow moving cursor into an empty space while a tile is selected
            # This prevents swapping with empty spaces.
            if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_HEIGHT
            if movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
            if movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_WIDTH
            if movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_WIDTH) % self.GRID_WIDTH

        if shift_pressed and self.selected_tile is not None:
            self.selected_tile = None
            # sfx: deselect_sound

        if space_pressed:
            cx, cy = self.cursor_pos
            if self.grid[cy, cx] == -1: # Cannot select an empty space
                # sfx: error_sound
                return 0

            if self.selected_tile is None:
                self.selected_tile = (cx, cy)
                # sfx: select_sound
            else:
                sx, sy = self.selected_tile
                if (cx, cy) == (sx, sy):
                    self.selected_tile = None
                    # sfx: deselect_sound
                elif abs(cx - sx) + abs(cy - sy) == 1:
                    # sfx: swap_sound
                    reward = self._process_swap(sx, sy, cx, cy)
                    self.selected_tile = None
                    return reward
                else:
                    self.selected_tile = (cx, cy)
                    # sfx: select_sound
        return 0

    def _process_swap(self, x1, y1, x2, y2):
        self.moves_left -= 1
        self._swap_tiles(x1, y1, x2, y2)
        
        matches = self._find_all_matches()
        if not matches:
            self._swap_tiles(x1, y1, x2, y2) # Swap back
            # sfx: invalid_swap_sound
            return -0.1

        total_reward = 0
        is_combo = False
        while matches:
            # sfx: match_clear_sound
            num_cleared = len(matches)
            turn_reward = num_cleared
            if num_cleared > 3: turn_reward += 5
            if is_combo: turn_reward += 10
            
            total_reward += turn_reward
            self.score += turn_reward

            for x, y in matches:
                self._create_particles(x, y, self.grid[y, x])
                self.grid[y, x] = -1
            
            self._apply_gravity_and_refill()
            matches = self._find_all_matches()
            is_combo = True
        
        return total_reward

    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int8)
            
            # Ensure no initial matches
            initial_matches = self._find_all_matches()
            while initial_matches:
                for x, y in initial_matches:
                    self.grid[y, x] = -1
                self._apply_gravity_and_refill()
                initial_matches = self._find_all_matches()

            # Ensure at least one move is possible
            if self._is_move_possible():
                break

    def _is_move_possible(self):
        temp_grid = np.copy(self.grid)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Check swap right
                if x < self.GRID_WIDTH - 1:
                    self._swap_tiles(x, y, x + 1, y)
                    if self._find_all_matches():
                        self.grid = np.copy(temp_grid)
                        return True
                    self._swap_tiles(x, y, x + 1, y) # Swap back
                # Check swap down
                if y < self.GRID_HEIGHT - 1:
                    self._swap_tiles(x, y, x, y + 1)
                    if self._find_all_matches():
                        self.grid = np.copy(temp_grid)
                        return True
                    self._swap_tiles(x, y, x, y + 1) # Swap back
        self.grid = np.copy(temp_grid)
        return False

    def _find_all_matches(self):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] == -1: continue
                # Horizontal
                if x < self.GRID_WIDTH - 2 and self.grid[y, x] == self.grid[y, x+1] == self.grid[y, x+2]:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
                # Vertical
                if y < self.GRID_HEIGHT - 2 and self.grid[y, x] == self.grid[y+1, x] == self.grid[y+2, x]:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return matches

    def _swap_tiles(self, x1, y1, x2, y2):
        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y, x] != -1:
                    self._swap_tiles(x, y, x, empty_row)
                    empty_row -= 1
            for y in range(empty_row, -1, -1):
                self.grid[y, x] = self.np_random.integers(0, self.NUM_COLORS)

    def _create_particles(self, grid_x, grid_y, color_idx):
        if color_idx < 0: return
        px = self.GRID_OFFSET_X + grid_x * self.TILE_SIZE + self.TILE_SIZE // 2
        py = self.GRID_OFFSET_Y + grid_y * self.TILE_SIZE + self.TILE_SIZE // 2
        color = self.GEM_COLORS[color_idx]
        for _ in range(15): # Number of particles
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifespan = random.randint(10, 20)
            self.particles.append([px, py, vx, vy, lifespan, color])

    def _is_board_clear(self):
        return np.all(self.grid == -1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.TILE_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.TILE_SIZE, y))

        # Draw gems
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.grid[y, x]
                if color_idx != -1:
                    self._draw_gem(x, y, color_idx)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), max(0, int(p[4] / 4)))

        # Draw selection highlight
        if self.selected_tile is not None:
            sx, sy = self.selected_tile
            rect = pygame.Rect(
                self.GRID_OFFSET_X + sx * self.TILE_SIZE,
                self.GRID_OFFSET_Y + sy * self.TILE_SIZE,
                self.TILE_SIZE, self.TILE_SIZE
            )
            pulse = (math.sin(self.steps * 0.3) + 1) / 2 * 3 + 2
            pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, int(pulse), border_radius=8)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + cx * self.TILE_SIZE,
            self.GRID_OFFSET_Y + cy * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, cursor_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR[:3], cursor_rect, 2, border_radius=8)

    def _draw_gem(self, grid_x, grid_y, color_idx):
        center_x = self.GRID_OFFSET_X + grid_x * self.TILE_SIZE + self.TILE_SIZE // 2
        center_y = self.GRID_OFFSET_Y + grid_y * self.TILE_SIZE + self.TILE_SIZE // 2
        
        color = self.GEM_COLORS[color_idx]
        shadow_color = tuple(max(0, c - 50) for c in color)
        highlight_color = tuple(min(255, c + 80) for c in color)
        
        gem_rect = pygame.Rect(0, 0, self.GEM_SIZE, self.GEM_SIZE)
        gem_rect.center = (center_x, center_y)

        # Main gem body with antialiasing
        pygame.gfxdraw.box(self.screen, gem_rect, shadow_color)
        pygame.gfxdraw.box(self.screen, gem_rect.move(0, -2), color)

        # Highlight
        pygame.gfxdraw.filled_polygon(self.screen, [
            (gem_rect.left + 2, gem_rect.top + 2),
            (gem_rect.left + self.GEM_SIZE * 0.4, gem_rect.top + 2),
            (gem_rect.left + 2, gem_rect.top + self.GEM_SIZE * 0.4)
        ], highlight_color)

    def _render_ui(self):
        # Render Moves Left
        moves_text = self.UI_FONT.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Render Score
        score_text = self.UI_FONT.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (620 - score_text.get_width(), 20))

        # Render Game Over
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_status = "Board Cleared!" if self._is_board_clear() else "Out of Moves"
            status_surf = self.UI_FONT_LARGE.render(win_status, True, self.COLOR_SELECT)
            status_rect = status_surf.get_rect(center=(320, 180))
            self.screen.blit(status_surf, status_rect)

            final_score_surf = self.UI_FONT.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(320, 230))
            self.screen.blit(final_score_surf, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
            "selected_tile": self.selected_tile,
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Use a dummy window to capture key presses
    pygame.display.set_caption("Match-3 Game Test")
    screen = pygame.display.set_mode((640, 400))
    
    terminated = False
    running = True
    clock = pygame.time.Clock()

    while running and not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")
        
        if terminated:
            print("Game Over!")
            pygame.time.wait(3000) # Pause for 3 seconds before closing
            running = False

        clock.tick(10) # Limit manual play speed

    env.close()