
# Generated: 2025-08-28T04:02:41.401208
# Source Brief: brief_02183.md
# Brief Index: 2183

        
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
        "Controls: ↑↓←→ to draw the line. Connect the green start to the red end."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A line-drawing puzzle. Connect the green start point to the red end point by drawing a "
        "continuous line within the move limit. Each move extends the line one step."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_large = pygame.font.SysFont("sans-serif", 48, bold=True)
        self.font_small = pygame.font.SysFont("sans-serif", 24, bold=True)
        
        self.COLOR_BG = (15, 18, 26)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_START = (26, 188, 156)
        self.COLOR_END = (231, 76, 60)
        self.COLOR_PATH = (52, 152, 219)
        self.COLOR_PATH_HEAD = (241, 196, 15)
        self.COLOR_TEXT = (236, 240, 241)

        # Game constants
        self.grid_size = 5
        self.max_moves = 10
        
        # Calculate grid rendering properties
        self.padding = 40
        game_area_size = min(self.width - 2 * self.padding, self.height - 2 * self.padding)
        self.cell_size = game_area_size / self.grid_size
        self.grid_offset_x = (self.width - self.grid_size * self.cell_size) / 2
        self.grid_offset_y = (self.height - self.grid_size * self.cell_size) / 2
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.start_pos = (0, 0)
        self.end_pos = (0, 0)
        self.current_pos = (0, 0)
        self.path = []
        self.previous_distance_to_end = 0
        self.win_message = ""
        self.pulse_timer = 0
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_maze()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.pulse_timer = 0
        
        self.current_pos = self.start_pos
        self.path = [self.current_pos]
        self.previous_distance_to_end = self._manhattan_distance(self.current_pos, self.end_pos)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.pulse_timer += 1
        reward = 0
        terminated = False
        
        is_move_valid = False
        old_pos = self.current_pos

        if movement > 0: # 1=up, 2=down, 3=left, 4=right
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            next_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
            
            # Check for invalid moves
            if not (0 <= next_pos[0] < self.grid_size and 0 <= next_pos[1] < self.grid_size):
                # Out of bounds
                pass
            elif next_pos in self.path:
                # Self-intersection
                pass
            else:
                is_move_valid = True
                self.current_pos = next_pos
                self.path.append(next_pos)
        
        self.steps += 1
        
        # --- Reward Calculation ---
        if not is_move_valid:
            reward = -1.0  # Penalty for invalid move or no-op
        else:
            dist_now = self._manhattan_distance(self.current_pos, self.end_pos)
            if dist_now < self.previous_distance_to_end:
                reward = 0.5  # Moved closer
            else:
                reward = -0.5 # Moved further or same (but valid move)
            self.previous_distance_to_end = dist_now
        
        # --- Termination Check ---
        if self.current_pos == self.end_pos:
            reward += 100.0  # Big win bonus
            terminated = True
            self.win_message = "COMPLETE!"
            # Sound effect placeholder: play_win_sound()
        elif self.steps >= self.max_moves:
            reward -= 10.0 # Penalty for running out of moves
            terminated = True
            self.win_message = "OUT OF MOVES"
            # Sound effect placeholder: play_lose_sound()

        if terminated:
            self.game_over = True
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
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
            "moves_left": self.max_moves - self.steps,
            "is_game_over": self.game_over,
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Vertical
            start_x = self.grid_offset_x + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, self.grid_offset_y), (start_x, self.grid_offset_y + self.grid_size * self.cell_size), 1)
            # Horizontal
            start_y = self.grid_offset_y + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, start_y), (self.grid_offset_x + self.grid_size * self.cell_size, start_y), 1)

        # Draw start and end points
        self._draw_grid_square(self.start_pos, self.COLOR_START, 0.8)
        self._draw_grid_square(self.end_pos, self.COLOR_END, 0.8)

        # Draw path
        if len(self.path) > 1:
            path_points = [self._get_pixel_coords(pos) for pos in self.path]
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, path_points, int(self.cell_size * 0.2))
        
        # Draw rounded corners and head
        for pos in self.path:
            pygame.draw.circle(self.screen, self.COLOR_PATH, self._get_pixel_coords(pos), int(self.cell_size * 0.1))

        # Draw pulsing head
        pulse_size = int(self.cell_size * 0.2) + int(3 * math.sin(self.pulse_timer * 0.5))
        pygame.draw.circle(self.screen, self.COLOR_PATH_HEAD, self._get_pixel_coords(self.current_pos), max(0, pulse_size))
        pygame.gfxdraw.aacircle(self.screen, *self._get_pixel_coords(self.current_pos), max(0, pulse_size), self.COLOR_PATH_HEAD)

    def _render_ui(self):
        # Moves left
        moves_text = self.font_small.render(f"MOVES: {self.max_moves - self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.padding, 10))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.width - self.padding, 10))
        self.screen.blit(score_text, score_rect)
        
        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_PATH_HEAD)
            end_rect = end_text.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(end_text, end_rect)

    def _generate_maze(self):
        max_attempts = 100
        for _ in range(max_attempts):
            path_len = self.np_random.integers(7, self.max_moves)  # Path of 7-9 nodes
            
            start_node = tuple(self.np_random.integers(0, self.grid_size, size=2).tolist())
            path = [start_node]
            visited = {start_node}
            
            stuck = False
            for _ in range(path_len - 1):
                x, y = path[-1]
                possible_moves = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) not in visited:
                        possible_moves.append((nx, ny))
                
                if not possible_moves:
                    stuck = True
                    break
                
                next_idx = self.np_random.integers(0, len(possible_moves))
                next_node = possible_moves[next_idx]
                path.append(next_node)
                visited.add(next_node)

            if not stuck and len(path) == path_len:
                self.start_pos = path[0]
                self.end_pos = path[-1]
                if self.start_pos != self.end_pos:
                    return

        # Failsafe if generation fails
        self.start_pos = (0, self.np_random.integers(0, self.grid_size))
        self.end_pos = (self.grid_size - 1, self.np_random.integers(0, self.grid_size))

    def _get_pixel_coords(self, grid_pos):
        x = int(self.grid_offset_x + (grid_pos[0] + 0.5) * self.cell_size)
        y = int(self.grid_offset_y + (grid_pos[1] + 0.5) * self.cell_size)
        return x, y

    def _draw_grid_square(self, grid_pos, color, scale=1.0):
        px, py = self._get_pixel_coords(grid_pos)
        size = self.cell_size * scale
        half_size = size / 2
        rect = pygame.Rect(px - half_size, py - half_size, size, size)
        pygame.draw.rect(self.screen, color, rect, border_radius=int(size*0.2))

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

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
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame window for display
    pygame.display.set_caption("Line Drawer")
    screen_display = pygame.display.set_mode((env.width, env.height))
    
    action = [0, 0, 0] # Start with a no-op
    
    print("\n" + "="*30)
    print("MANUAL PLAY INSTRUCTIONS")
    print(env.user_guide)
    print("Press ESC or close window to quit.")
    print("="*30 + "\n")
    
    while not terminated:
        # Get observation from the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.1f}")
            pygame.time.wait(2000) # Pause to show final message
            break

        # Reset action for next frame
        action = [0, 0, 0] # Default to no-op
        
        # Poll for user input
        input_received = False
        while not input_received:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    input_received = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                        input_received = True
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                        input_received = True
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                        input_received = True
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                        input_received = True
                    elif event.key == pygame.K_ESCAPE:
                        terminated = True
                        input_received = True
        
        if terminated:
            break

    env.close()