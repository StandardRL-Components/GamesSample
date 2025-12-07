
# Generated: 2025-08-27T18:16:04.685280
# Source Brief: brief_01778.md
# Brief Index: 1778

        
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
        "Use arrow keys to move the cursor. Press space to select a gem, "
        "then use an arrow key to swap it. Match 3+ to score. Press shift to reset."
    )

    game_description = (
        "A vibrant match-3 puzzle game. Strategically swap gems to create "
        "cascades and clear the board before you run out of 50 moves."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 8
        self.NUM_GEM_TYPES = 6
        self.GEM_SIZE = 40
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH * self.GEM_SIZE) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT * self.GEM_SIZE) // 2
        self.MAX_MOVES = 50
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 255, 0)
        self.GEM_COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 150, 255),   # Blue
            (255, 255, 80),   # Yellow
            (255, 80, 255),   # Magenta
            (80, 255, 255),   # Cyan
        ]

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
        self.font_large = pygame.font.SysFont('sans-serif', 28, bold=True)
        self.font_small = pygame.font.SysFont('sans-serif', 20, bold=True)

        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.score = None
        self.moves_left = None
        self.steps = None
        self.game_over = None
        self.particles = []
        self.last_cleared_coords = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.steps = 0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_pos = None
        self.particles = []
        self.last_cleared_coords = []

        self._generate_board()
        
        return self._get_observation(), self._get_info()

    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            
            # Ensure no initial matches
            while self._find_matches():
                matches = self._find_matches()
                for r, c in matches:
                    # Avoid matching with neighbors
                    forbidden = set()
                    if r > 0: forbidden.add(self.grid[r-1, c])
                    if c > 0: forbidden.add(self.grid[r, c-1])
                    
                    possible_gems = [g for g in range(self.NUM_GEM_TYPES) if g not in forbidden]
                    if not possible_gems: possible_gems = list(range(self.NUM_GEM_TYPES))
                    
                    self.grid[r, c] = self.np_random.choice(possible_gems)

            # Ensure at least one move is possible
            if self._find_all_possible_moves():
                break

    def _find_all_possible_moves(self):
        moves = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Check swap right
                if c < self.GRID_WIDTH - 1:
                    self._swap_gems(r, c, r, c + 1)
                    if self._find_matches():
                        moves.append(((r, c), (r, c + 1)))
                    self._swap_gems(r, c, r, c + 1) # Swap back
                # Check swap down
                if r < self.GRID_HEIGHT - 1:
                    self._swap_gems(r, c, r + 1, c)
                    if self._find_matches():
                        moves.append(((r, c), (r + 1, c)))
                    self._swap_gems(r, c, r + 1, c) # Swap back
        return moves

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        self.last_cleared_coords = []

        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

        if shift_press:
            # Game over on shift press as per brief
            reward = -50
            terminated = True
            self.game_over = True
        else:
            reward, terminated = self._handle_player_action(movement, space_press)

        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_action(self, movement, space_press):
        reward, terminated = 0, False

        # --- Input Handling ---
        if self.selected_pos is None:
            # State: SELECTING (moving cursor)
            if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 2: self.cursor_pos[0] = min(self.GRID_HEIGHT - 1, self.cursor_pos[0] + 1)
            elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 4: self.cursor_pos[1] = min(self.GRID_WIDTH - 1, self.cursor_pos[1] + 1)

            if space_press:
                self.selected_pos = list(self.cursor_pos)
        
        else:
            # State: SWAPPING (gem selected, waiting for swap direction)
            if space_press: # Deselect
                self.selected_pos = None
                return 0, False

            if movement != 0:
                r1, c1 = self.selected_pos
                r2, c2 = r1, c1

                if movement == 1: r2 = r1 - 1
                elif movement == 2: r2 = r1 + 1
                elif movement == 3: c2 = c1 - 1
                elif movement == 4: c2 = c1 + 1

                self.selected_pos = None # Action is consumed

                if 0 <= r2 < self.GRID_HEIGHT and 0 <= c2 < self.GRID_WIDTH:
                    self.moves_left -= 1
                    self._swap_gems(r1, c1, r2, c2)
                    
                    # --- Game Logic: Match and Cascade ---
                    match_found_on_swap, turn_reward = self._process_cascades()

                    if not match_found_on_swap:
                        self._swap_gems(r1, c1, r2, c2) # Swap back
                        reward = -0.1
                    else:
                        reward = turn_reward
                        self.cursor_pos = [r2, c2] # Move cursor to swapped gem

        # --- Check Termination Conditions ---
        if np.all(self.grid == -1): # All gems cleared
            reward += 50
            terminated = True
        elif self.moves_left <= 0:
            reward -= 50
            terminated = True
        elif not self._find_all_possible_moves() and not self.game_over:
            # No moves left, end game
            reward -= 50
            terminated = True
            
        return reward, terminated

    def _process_cascades(self):
        total_reward = 0
        is_first_cascade = True
        match_found_on_swap = False

        while True:
            matches = self._find_matches()
            if not matches:
                break

            if is_first_cascade:
                match_found_on_swap = True
            is_first_cascade = False

            # Calculate reward for this cascade
            num_cleared = len(matches)
            total_reward += num_cleared  # +1 per gem
            if num_cleared > 3:
                total_reward += 5  # Bonus for larger matches

            # Clear gems and create effects
            for r, c in matches:
                if self.grid[r, c] != -1:
                    self._create_particles(r, c)
                    self.last_cleared_coords.append((r, c))
                    self.grid[r, c] = -1 # Mark as empty
            self.score += num_cleared

            # Gravity
            self._apply_gravity()
            self._refill_board()
        
        return match_found_on_swap, total_reward

    def _swap_gems(self, r1, c1, r2, c2):
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == -1: continue
                
                # Horizontal match
                if c < self.GRID_WIDTH - 2 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                
                # Vertical match
                if r < self.GRID_HEIGHT - 2 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self._swap_gems(r, c, empty_row, c)
                    empty_row -= 1

    def _refill_board(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _create_particles(self, r, c):
        gem_type = self.grid[r,c]
        if gem_type == -1: return
        color = self.GEM_COLORS[gem_type]
        center_x = self.GRID_X_OFFSET + c * self.GEM_SIZE + self.GEM_SIZE // 2
        center_y = self.GRID_Y_OFFSET + r * self.GEM_SIZE + self.GEM_SIZE // 2
        
        for _ in range(15): # Create 15 particles
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append([center_x, center_y, vx, vy, life, color])

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # life -= 1
            p[2] *= 0.98 # friction
            p[3] *= 0.98
            if p[4] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self._update_particles()
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_X_OFFSET + i * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.GEM_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y_OFFSET + i * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH * self.GEM_SIZE, y))

        # Draw gems
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    color = self.GEM_COLORS[gem_type]
                    rect = pygame.Rect(self.GRID_X_OFFSET + c * self.GEM_SIZE, self.GRID_Y_OFFSET + r * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
                    
                    center_x, center_y = rect.center
                    radius = int(self.GEM_SIZE * 0.4)
                    
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    
                    # Highlight effect
                    highlight_color = (min(255, c+80) for c in color)
                    pygame.gfxdraw.aacircle(self.screen, center_x-radius//4, center_y-radius//4, radius//3, tuple(highlight_color))

        # Draw match flash effect
        for r, c in self.last_cleared_coords:
            rect = pygame.Rect(self.GRID_X_OFFSET + c * self.GEM_SIZE, self.GRID_Y_OFFSET + r * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
            flash_surface = pygame.Surface((self.GEM_SIZE, self.GEM_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 200, 150))
            self.screen.blit(flash_surface, rect.topleft)
            
        # Draw particles
        for x, y, vx, vy, life, color in self.particles:
            size = max(1, int(life / 6))
            pygame.draw.circle(self.screen, color, (int(x), int(y)), size)

        # Draw cursor
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
        alpha = int(100 + 100 * pulse)
        cursor_color = self.COLOR_CURSOR + (alpha,)
        cursor_rect = pygame.Rect(
            self.GRID_X_OFFSET + self.cursor_pos[1] * self.GEM_SIZE,
            self.GRID_Y_OFFSET + self.cursor_pos[0] * self.GEM_SIZE,
            self.GEM_SIZE, self.GEM_SIZE
        )
        s = pygame.Surface((self.GEM_SIZE, self.GEM_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, cursor_color, s.get_rect(), border_radius=6, width=4)
        self.screen.blit(s, cursor_rect.topleft)

        # Draw selection
        if self.selected_pos:
            select_rect = pygame.Rect(
                self.GRID_X_OFFSET + self.selected_pos[1] * self.GEM_SIZE,
                self.GRID_Y_OFFSET + self.selected_pos[0] * self.GEM_SIZE,
                self.GEM_SIZE, self.GEM_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, select_rect, border_radius=6, width=4)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = np.all(self.grid == -1)
            end_text_str = "BOARD CLEARED!" if win_condition else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
            "selected_pos": list(self.selected_pos) if self.selected_pos else None,
        }
        
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Match-3 Gym Environment")
    
    running = True
    terminated = False
    
    # Game loop
    while running:
        # Get user input
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                elif event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    terminated = False
        
        action = [movement, space, shift]
        
        # Only step if an action is taken or the game ended
        if any(action) or terminated:
            if not terminated:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        env.clock.tick(30) # Limit FPS

    env.close()