
# Generated: 2025-08-28T00:40:08.057823
# Source Brief: brief_03864.md
# Brief Index: 3864

        
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
        "Use arrow keys to swap the selected piece with an adjacent one. "
        "Press space to select a new random piece. Matches of 3+ clear pieces."
    )

    game_description = (
        "A strategic match-3 puzzle game. Swap colored pieces on a 5x5 grid to create "
        "matches of three or more. Plan your moves to create cascading combos and "
        "clear the board before you run out of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_SIZE = 5
        self.NUM_COLORS = 5
        self.MAX_MOVES = 10
        self.MAX_STEPS = 1000  # Failsafe termination

        # Visual constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 64
        self.GRID_LINE_WIDTH = 2
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2

        # Colors
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_GRID = (50, 60, 80)
        self.PIECE_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SELECTOR = (255, 255, 255)
        
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
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # Game state variables (initialized in reset)
        self.grid = None
        self.selected_pos = None
        self.moves_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._create_initial_grid()
        self.selected_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.moves_remaining = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False

        movement = action[0]
        space_pressed = action[1] == 1
        
        if space_pressed:
            # Select a new random piece, does not consume a move
            self.selected_pos = (
                self.np_random.integers(0, self.GRID_SIZE),
                self.np_random.integers(0, self.GRID_SIZE),
            )
        else:
            # An actual move is attempted
            self.moves_remaining -= 1
            
            if movement == 0: # No-op
                reward += -0.2
            else:
                reward += self._attempt_swap(movement)

        # Check for termination conditions
        if self.moves_remaining <= 0 or self._is_board_clear() or self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            if self._is_board_clear():
                reward += 100 # Goal-oriented reward for clearing the board

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _attempt_swap(self, movement_action):
        r, c = self.selected_pos
        dr, dc = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement_action]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE):
            return -0.2 # Invalid move (out of bounds)

        temp_grid = self.grid.copy()
        temp_grid[r, c], temp_grid[nr, nc] = temp_grid[nr, nc], temp_grid[r, c]

        matches = self._find_matches(temp_grid)

        if not matches:
            # No match created, swap is invalid. Penalize.
            # play 'error' sound
            return -0.2
        else:
            # Valid swap, update grid and start chain reaction
            self.grid = temp_grid
            self.selected_pos = (nr, nc) # Move selector to the swapped position
            
            total_reward = 0
            chain_level = 1
            while matches:
                # Calculate reward for current matches
                num_matched = len(matches)
                total_reward += num_matched * chain_level
                
                # Check for full row/column clears
                cleared_rows, cleared_cols = set(), set()
                for mr, mc in matches:
                    cleared_rows.add(mr)
                    cleared_cols.add(mc)
                
                for row_idx in cleared_rows:
                    if all((row_idx, col) in matches for col in range(self.GRID_SIZE)):
                        total_reward += 5
                for col_idx in cleared_cols:
                    if all((row, col_idx) in matches for row in range(self.GRID_SIZE)):
                        total_reward += 5

                self.score += num_matched * 10 * chain_level

                # Clear matched pieces and create particles
                for mr, mc in matches:
                    self._create_particles(mr, mc, self.grid[mr, mc])
                    self.grid[mr, mc] = 0
                
                # play 'match' sound, maybe pitch increases with chain_level

                # Apply gravity and fill new pieces
                self._apply_gravity()
                self._fill_new_pieces()

                # Check for new matches
                matches = self._find_matches(self.grid)
                chain_level += 1
            
            return total_reward

    def _create_initial_grid(self):
        while True:
            grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_matches(grid):
                return grid

    def _find_matches(self, grid):
        matched_coords = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if grid[r, c] == 0:
                    continue
                # Horizontal matches
                if c < self.GRID_SIZE - 2 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matched_coords.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical matches
                if r < self.GRID_SIZE - 2 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matched_coords.update([(r, c), (r+1, c), (r+2, c)])
        return matched_coords

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
    
    def _fill_new_pieces(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _is_board_clear(self):
        return np.all(self.grid == 0)

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
            "moves_remaining": self.moves_remaining,
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end_pos = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, self.GRID_LINE_WIDTH)
            # Horizontal
            start_pos = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            end_pos = (self.GRID_X_OFFSET + self.GRID_WIDTH, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, self.GRID_LINE_WIDTH)

        # Draw pieces
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                piece = self.grid[r, c]
                if piece > 0:
                    color = self.PIECE_COLORS[piece - 1]
                    rect = pygame.Rect(
                        self.GRID_X_OFFSET + c * self.CELL_SIZE + self.GRID_LINE_WIDTH,
                        self.GRID_Y_OFFSET + r * self.CELL_SIZE + self.GRID_LINE_WIDTH,
                        self.CELL_SIZE - 2 * self.GRID_LINE_WIDTH,
                        self.CELL_SIZE - 2 * self.GRID_LINE_WIDTH
                    )
                    # Draw a slight 3D effect
                    shadow_color = tuple(max(0, val - 40) for val in color)
                    highlight_color = tuple(min(255, val + 40) for val in color)
                    pygame.draw.rect(self.screen, shadow_color, rect.move(3, 3))
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, int(self.CELL_SIZE * 0.3), highlight_color)
                    pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, int(self.CELL_SIZE * 0.3), highlight_color)


        # Draw selector
        sel_r, sel_c = self.selected_pos
        sel_rect = pygame.Rect(
            self.GRID_X_OFFSET + sel_c * self.CELL_SIZE,
            self.GRID_Y_OFFSET + sel_r * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        # Pulsating alpha effect for selector
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # Varies between 0 and 1
        alpha = 100 + 155 * pulse
        
        # Create a temporary surface for transparency
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_SELECTOR, alpha), s.get_rect(), 8, border_radius=8)
        self.screen.blit(s, sel_rect.topleft)

        # Update and draw particles
        self._update_and_render_particles()

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "BOARD CLEARED!" if self._is_board_clear() else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, score_rect)

    def _create_particles(self, r, c, piece_type):
        px = self.GRID_X_OFFSET + c * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.GRID_Y_OFFSET + r * self.CELL_SIZE + self.CELL_SIZE / 2
        color = self.PIECE_COLORS[piece_type - 1]
        
        for _ in range(20): # Number of particles per piece
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(20, 40) # Frames
            size = random.uniform(2, 5)
            self.particles.append([ [px, py], vel, color, life, size ])

    def _update_and_render_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[1][1] += 0.1 # gravity on particles
            p[3] -= 1 # life -= 1
            
            pos = p[0]
            color = p[2]
            life = p[3]
            size = p[4]
            
            if life > 0:
                # Fade out effect
                alpha_color = (*color, max(0, min(255, int(255 * (life / 30)))))
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, alpha_color, (size, size), size)
                self.screen.blit(temp_surf, (int(pos[0] - size), int(pos[1] - size)))

        self.particles = [p for p in self.particles if p[3] > 0]

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

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To display the game, we need a Pygame screen
    pygame.display.set_caption("Match-3 Gym Environment")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1 # Swap up
                elif event.key == pygame.K_DOWN:
                    action[0] = 2 # Swap down
                elif event.key == pygame.K_LEFT:
                    action[0] = 3 # Swap left
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4 # Swap right
                elif event.key == pygame.K_SPACE:
                    action[1] = 1 # Select random
                
                # A key press triggers a step in auto_advance=False mode
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated:
                    print(f"Game Over! Final Score: {info['score']}")
                    # Optional: auto-reset after a delay
                    pygame.time.wait(2000)
                    obs, info = env.reset()

        # Render the observation to the display screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()