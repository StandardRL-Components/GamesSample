
# Generated: 2025-08-28T02:39:00.054870
# Source Brief: brief_01765.md
# Brief Index: 1765

        
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
        "Controls: Use arrow keys to move the cursor. Hold Space and press an arrow key to swap gems."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent gems to create matches of 3 or more. Race against a 15-move limit to reach the target score of 100. Earn points for each gem matched and bonus points for cascades!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_GEM_TYPES = 6
        self.CELL_SIZE = 40
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.TARGET_SCORE = 100
        self.MAX_MOVES = 15
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 140, 50),  # Orange
        ]
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_remaining = None
        self.game_over = None
        self.effects = None
        self.steps = 0
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.effects = []
        
        self._create_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0
        self.game_over = self.score >= self.TARGET_SCORE or self.moves_remaining <= 0
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        self._update_effects()

        if not space_held:
            # Cursor movement only, no turn taken
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)
        else:
            # Swap attempt, a turn is taken
            self.moves_remaining -= 1
            
            p1 = self.cursor_pos
            p2 = list(p1)

            if movement == 1: p2[1] -= 1 # Up
            elif movement == 2: p2[1] += 1 # Down
            elif movement == 3: p2[0] -= 1 # Left
            elif movement == 4: p2[0] += 1 # Right
            
            if self._is_valid_swap(p1, p2):
                self._swap_gems(p1, p2)
                
                matches, cascades = self._process_all_matches()
                if matches:
                    # Successful match
                    reward += len(matches) # +1 per gem
                    self.score += len(matches)
                    if cascades > 0:
                        reward += 5 * cascades # Cascade bonus
                        # Sound: Cascade bonus!
                else:
                    # Invalid swap (no match), swap back
                    self._swap_gems(p1, p2)
                    reward -= 0.2
                    # Sound: Invalid move
            else:
                # Attempted swap off-board or with self
                reward -= 0.2
                # Sound: Invalid move
        
        # Check for termination and apply terminal rewards
        terminated = self.score >= self.TARGET_SCORE or self.moves_remaining <= 0
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.TARGET_SCORE:
                reward += 100
                # Sound: Victory!
            else:
                reward -= 10
                # Sound: Game over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, 
                                self.GRID_SIZE * self.CELL_SIZE, self.GRID_SIZE * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)
        
        # Draw gems
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    self._draw_gem(c, r, gem_type)
        
        # Draw visual effects
        for effect in self.effects:
            if effect['type'] == 'vanish':
                progress = 1 - (effect['life'] / effect['max_life'])
                x = self.GRID_OFFSET_X + effect['pos'][0] * self.CELL_SIZE + self.CELL_SIZE // 2
                y = self.GRID_OFFSET_Y + effect['pos'][1] * self.CELL_SIZE + self.CELL_SIZE // 2
                radius = int(progress * self.CELL_SIZE * 0.7)
                alpha = int(255 * (1 - progress))
                
                # Draw a sparkling explosion
                for i in range(5):
                    angle = progress * 360 + i * 72
                    px = x + int(radius * math.cos(math.radians(angle)))
                    py = y + int(radius * math.sin(math.radians(angle)))
                    pygame.gfxdraw.filled_circle(self.screen, px, py, int(self.CELL_SIZE * 0.1), (255, 255, 255, alpha))

    def _render_ui(self):
        # Draw cursor
        c, r = self.cursor_pos
        rect = pygame.Rect(self.GRID_OFFSET_X + c * self.CELL_SIZE,
                           self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                           self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=5)

        # Draw score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Draw moves remaining
        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 10))

        # Draw game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.TARGET_SCORE:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "cursor_pos": list(self.cursor_pos),
        }

    # --- Helper Methods ---
    def _create_board(self):
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
        while self._find_matches():
            matches = self._find_matches()
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2] != -1:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c] != -1:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _process_all_matches(self):
        all_matched_gems = set()
        cascades = -1
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            cascades += 1
            # Sound: Match found
            
            for r, c in matches:
                all_matched_gems.add((r, c))
                if self.grid[r,c] != -1:
                    self.effects.append({'type': 'vanish', 'pos': (c, r), 'life': 10, 'max_life': 10})
                    self.grid[r, c] = -1

            self._apply_gravity()
            self._refill_board()
        
        return all_matched_gems, max(0, cascades)

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            write_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != write_row:
                        self.grid[write_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    write_row -= 1

    def _refill_board(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)
                    # Sound: Gem fall

    def _is_valid_swap(self, p1, p2):
        if not (0 <= p2[0] < self.GRID_SIZE and 0 <= p2[1] < self.GRID_SIZE):
            return False
        dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        return dist == 1

    def _swap_gems(self, p1, p2):
        c1, r1 = p1
        c2, r2 = p2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        # Sound: Gem swap

    def _draw_gem(self, c, r, gem_type):
        x = self.GRID_OFFSET_X + c * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
        center_x, center_y = x + self.CELL_SIZE // 2, y + self.CELL_SIZE // 2
        radius = int(self.CELL_SIZE * 0.4)
        color = self.GEM_COLORS[gem_type % len(self.GEM_COLORS)]
        
        # Base color with a slight gradient
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        
        # Highlight for 3D effect
        highlight_color = tuple(min(255, val + 60) for val in color)
        pygame.gfxdraw.arc(self.screen, center_x, center_y, radius - 1, 135, 315, highlight_color)
        
        # Shape overlay for accessibility
        shape_color = (255, 255, 255)
        p = int(radius * 0.6)
        if gem_type == 0: # Circle
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(radius * 0.3), shape_color)
        elif gem_type == 1: # Square
            pygame.draw.rect(self.screen, shape_color, (center_x - p//2, center_y - p//2, p, p))
        elif gem_type == 2: # Triangle
            points = [(center_x, center_y - p//2), (center_x - p//2, center_y + p//2), (center_x + p//2, center_y + p//2)]
            pygame.gfxdraw.filled_polygon(self.screen, points, shape_color)
        elif gem_type == 3: # Diamond
            points = [(center_x, center_y - p//2), (center_x + p//2, center_y), (center_x, center_y + p//2), (center_x - p//2, center_y)]
            pygame.gfxdraw.filled_polygon(self.screen, points, shape_color)
        elif gem_type == 4: # Hexagon
             points = [(center_x + int(p//2 * math.cos(math.radians(a))), center_y + int(p//2 * math.sin(math.radians(a)))) for a in range(30, 390, 60)]
             pygame.gfxdraw.filled_polygon(self.screen, points, shape_color)
        elif gem_type == 5: # Star
            points = []
            for i in range(10):
                angle = math.radians(i * 36)
                rad = p//2 if i % 2 == 0 else p//4
                points.append((center_x + int(rad * math.cos(angle)), center_y + int(rad * math.sin(angle))))
            pygame.gfxdraw.filled_polygon(self.screen, points, shape_color)
        
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, (0,0,0,50))


    def _update_effects(self):
        self.effects = [e for e in self.effects if e['life'] > 0]
        for e in self.effects:
            e['life'] -= 1

    def close(self):
        pygame.font.quit()
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up a window to display the game
    pygame.display.set_caption("Gem Swap")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        # Human input mapping
        action = [0, 0, 0] # no-op, no-space, no-shift
        keys = pygame.key.get_pressed()
        
        move_made = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                
                # Non-swap moves (cursor movement)
                if not keys[pygame.K_SPACE]:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    if action[0] != 0: move_made = True
                
                # Swap moves
                if keys[pygame.K_SPACE]:
                    action[1] = 1
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    if action[0] != 0: move_made = True

        if move_made:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")
            if terminated:
                print("Game Over!")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Limit frame rate
        env.clock.tick(30)
        
    env.close()