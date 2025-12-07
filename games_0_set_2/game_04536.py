
# Generated: 2025-08-28T02:42:02.202913
# Source Brief: brief_04536.md
# Brief Index: 4536

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrows to move the cursor. Press space to select a gem. "
        "Move to an adjacent gem and press space again to swap. Press shift to cancel."
    )

    game_description = (
        "Match gems to reach a target score in this fast-paced, grid-based puzzle game."
    )

    auto_advance = False

    class Particle:
        def __init__(self, x, y, color, rng):
            self.x = x
            self.y = y
            self.color = color
            angle = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(1, 4)
            self.vx = math.cos(angle) * speed
            self.vy = math.sin(angle) * speed
            self.lifespan = rng.integers(20, 40)
            self.size = rng.integers(4, 8)

        def update(self):
            self.x += self.vx
            self.y += self.vy
            self.lifespan -= 1
            self.size = max(0, self.size - 0.2)

        def draw(self, surface):
            if self.lifespan > 0:
                alpha = int(255 * (self.lifespan / 40))
                temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.color, alpha), (self.size, self.size), self.size)
                surface.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size)), special_flags=pygame.BLEND_RGBA_ADD)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_GEM_TYPES = 6
        self.TARGET_SCORE = 5000
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000

        self.GEM_SIZE = 40
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_SIZE * self.GEM_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.GEM_SIZE) // 2

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID_BG = (25, 30, 50)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 160, 80),  # Orange
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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 50)

        # State variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.score = None
        self.moves_left = None
        self.steps = None
        self.game_over = None
        self.last_score_milestone = None
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_gem_pos = None
        self.last_score_milestone = 0
        self.space_pressed_last_frame = True # Prevent action on first frame
        self.shift_pressed_last_frame = True # Prevent action on first frame
        self.particles = []

        self._create_stable_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.space_pressed_last_frame
        shift_press = shift_held and not self.shift_pressed_last_frame

        reward += self._handle_input(movement, space_press, shift_press)

        self.space_pressed_last_frame = space_held
        self.shift_pressed_last_frame = shift_held

        # Check for score milestones
        if self.score // 1000 > self.last_score_milestone:
            self.last_score_milestone = self.score // 1000
            reward += 10

        terminated = self._check_termination()
        if terminated:
            if self.score >= self.TARGET_SCORE:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Lose penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement, space_press, shift_press):
        # Handle shift to cancel selection
        if shift_press and self.selected_gem_pos:
            self.selected_gem_pos = None
            # sound: cancel_select.wav
            return 0

        # Handle cursor movement
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)

        # Handle space press for selection/swapping
        if space_press:
            r, c = self.cursor_pos
            if not self.selected_gem_pos:
                # Select a gem
                self.selected_gem_pos = [r, c]
                # sound: select_gem.wav
            else:
                # Attempt a swap
                sr, sc = self.selected_gem_pos
                is_adjacent = abs(sr - r) + abs(sc - c) == 1
                if is_adjacent:
                    return self._attempt_swap(sr, sc, r, c)
                else:
                    # If not adjacent, treat as a new selection
                    self.selected_gem_pos = [r, c]
                    # sound: select_gem.wav
        return 0

    def _attempt_swap(self, r1, c1, r2, c2):
        self.moves_left -= 1
        self.selected_gem_pos = None

        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        # sound: swap.wav
        
        matches = self._find_matches()
        if not matches:
            # Invalid swap, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            # sound: invalid_swap.wav
            return -0.2

        # Valid swap, process matches
        total_reward = 0
        chain = 0
        while matches:
            chain += 1
            num_matched = len(matches)
            
            # Calculate score and reward
            # Base score: 10 per gem. Match-3=30, Match-4=40 etc.
            # Reward: 1 for 3, 2 for 4, etc.
            # Chain bonus: score and reward multiplied by chain number
            base_score = num_matched * 10
            base_reward = max(0, num_matched - 2)
            
            self.score += base_score * chain
            total_reward += base_reward * chain

            # sound: match.wav
            for r, c in matches:
                self._create_particles(r, c)
                self.grid[r, c] = -1 # Mark for removal

            self._apply_gravity()
            self._fill_new_gems()
            
            matches = self._find_matches()

        # After all chains, check for soft-lock
        if not self._check_for_possible_moves():
            self._regenerate_grid()
            # No penalty/reward for regeneration, it's a game mechanic

        return total_reward

    def _create_stable_grid(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_matches():
                if self._check_for_possible_moves():
                    break
    
    def _regenerate_grid(self):
        self._create_stable_grid()
        # Create particles everywhere for visual feedback
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                self._create_particles(r, c, count=3)

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem = self.grid[r, c]
                if gem == -1: continue
                
                # Horizontal
                if c < self.GRID_SIZE - 2 and self.grid[r, c+1] == gem and self.grid[r, c+2] == gem:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                
                # Vertical
                if r < self.GRID_SIZE - 2 and self.grid[r+1, c] == gem and self.grid[r+2, c] == gem:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1

    def _fill_new_gems(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _check_for_possible_moves(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem = self.grid[r, c]
                # Check swap right
                if c < self.GRID_SIZE - 1:
                    neighbor = self.grid[r, c+1]
                    self.grid[r, c], self.grid[r, c+1] = neighbor, gem
                    if self._find_matches():
                        self.grid[r, c], self.grid[r, c+1] = gem, neighbor # Swap back
                        return True
                    self.grid[r, c], self.grid[r, c+1] = gem, neighbor # Swap back
                # Check swap down
                if r < self.GRID_SIZE - 1:
                    neighbor = self.grid[r+1, c]
                    self.grid[r, c], self.grid[r+1, c] = neighbor, gem
                    if self._find_matches():
                        self.grid[r, c], self.grid[r+1, c] = gem, neighbor # Swap back
                        return True
                    self.grid[r, c], self.grid[r+1, c] = gem, neighbor # Swap back
        return False

    def _check_termination(self):
        if self.score >= self.TARGET_SCORE:
            self.game_over = True
        if self.moves_left <= 0:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_SIZE * self.GEM_SIZE, self.GRID_SIZE * self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=10)

        # Draw gems
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    self._draw_gem(c, r, gem_type)

        # Draw selection and cursor
        self._draw_cursor()
        if self.selected_gem_pos:
            self._draw_selection_highlight()

        # Update and draw particles
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
            p.draw(self.screen)

    def _grid_to_screen(self, r, c):
        x = self.GRID_OFFSET_X + c * self.GEM_SIZE + self.GEM_SIZE // 2
        y = self.GRID_OFFSET_Y + r * self.GEM_SIZE + self.GEM_SIZE // 2
        return x, y

    def _draw_gem(self, c, r, gem_type):
        x, y = self._grid_to_screen(r, c)
        color = self.GEM_COLORS[gem_type]
        
        # Main gem body
        rect = pygame.Rect(x - self.GEM_SIZE//2 + 4, y - self.GEM_SIZE//2 + 4, self.GEM_SIZE - 8, self.GEM_SIZE - 8)
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        
        # Highlight
        highlight_color = tuple(min(255, val + 60) for val in color)
        pygame.draw.rect(self.screen, highlight_color, (rect.x + 2, rect.y + 2, rect.width - 10, rect.height // 3), border_radius=5)
        
        # Shadow
        shadow_color = tuple(max(0, val - 40) for val in color)
        pygame.draw.line(self.screen, shadow_color, (rect.left, rect.bottom), (rect.right, rect.bottom), 3)
        pygame.draw.line(self.screen, shadow_color, (rect.right, rect.top), (rect.right, rect.bottom), 3)

    def _draw_cursor(self):
        r, c = self.cursor_pos
        x, y = self._grid_to_screen(r, c)
        rect = pygame.Rect(x - self.GEM_SIZE // 2, y - self.GEM_SIZE // 2, self.GEM_SIZE, self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=10)

    def _draw_selection_highlight(self):
        r, c = self.selected_gem_pos
        x, y = self._grid_to_screen(r, c)
        
        # Pulsing glow effect
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
        radius = int(self.GEM_SIZE * 0.5 + pulse * 5)
        alpha = int(100 + pulse * 100)
        
        temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*self.COLOR_SELECTED, alpha))
        pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius, (*self.COLOR_SELECTED, alpha))
        self.screen.blit(temp_surf, (x-radius, y-radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _create_particles(self, r, c, count=10):
        x, y = self._grid_to_screen(r, c)
        color = self.GEM_COLORS[self.grid[r, c]]
        for _ in range(count):
            self.particles.append(self.Particle(x, y, color, self.np_random))

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Moves display
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        text_rect = moves_text.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(moves_text, text_rect)
        
        # Game over screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.score >= self.TARGET_SCORE:
                win_text = self.font_title.render("YOU WIN!", True, (100, 255, 100))
                text_rect = win_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
                overlay.blit(win_text, text_rect)
            else:
                lose_text = self.font_title.render("GAME OVER", True, (255, 100, 100))
                text_rect = lose_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
                overlay.blit(lose_text, text_rect)
            
            final_score_text = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 30))
            overlay.blit(final_score_text, score_rect)
            
            self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This block allows you to play the game manually.
    # It requires pygame to be installed with display support.
    try:
        import os
        # Set a display if not available (e.g., in a remote environment)
        if "DISPLAY" not in os.environ:
             os.environ["SDL_VIDEODRIVER"] = "dummy"
             raise ImportError("No display available for manual play, running headless check.")

        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Gem Matcher")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            movement, space, shift = 0, 0, 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Since auto_advance is False, we need a small delay for human playability
            pygame.time.wait(50) 
            
            if done:
                print(f"Game Over! Final Info: {info}")
                pygame.time.wait(2000) # Wait 2 seconds before closing
        
        env.close()

    except (ImportError, pygame.error) as e:
        print(f"Manual play unavailable ({e}). Running a short headless test.")
        
        # --- Headless Test ---
        # This block runs a simple test without a display.
        obs, info = env.reset()
        total_reward = 0
        for i in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps. Final Info: {info}, Total Reward: {total_reward}")
                obs, info = env.reset()
                total_reward = 0
        env.close()
        print("Headless test completed.")