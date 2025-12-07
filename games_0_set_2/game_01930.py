
# Generated: 2025-08-27T18:43:40.309920
# Source Brief: brief_01930.md
# Brief Index: 1930

        
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
        "Controls: Arrows to move cursor. Space to select/deselect a crystal. "
        "With a crystal selected, arrows move it to an adjacent empty space, which uses a turn. "
        "Shift cycles selection."
    )

    game_description = (
        "An isometric puzzle game. Move crystals to create matches of 5 or more of the same color. "
        "Score 1000 points within 20 moves to win."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.BOARD_ROWS, self.BOARD_COLS = 8, 10
        self.NUM_CRYSTAL_TYPES = 5
        self.TARGET_SCORE = 1000
        self.MAX_MOVES = 20
        self.MATCH_MIN_SIZE = 5

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 50, 64)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTION = (100, 255, 255)
        self.CRYSTAL_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # Isometric projection constants
        self.TILE_WIDTH = 48
        self.TILE_HEIGHT = self.TILE_WIDTH // 2
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 100

        # State variables (initialized in reset)
        self.board = None
        self.cursor_pos = None
        self.selected_crystal_pos = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.particles = None
        self.last_reward = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.BOARD_ROWS // 2, self.BOARD_COLS // 2]
        self.selected_crystal_pos = None
        self.particles = []
        self.last_reward = 0
        
        # Generate a board with no initial matches
        while True:
            self.board = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1, size=(self.BOARD_ROWS, self.BOARD_COLS))
            if not self._find_matches():
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        is_major_action = False

        # --- Action Interpretation ---

        # Major Action: Move a selected crystal
        if self.selected_crystal_pos and movement in [1, 2, 3, 4]:
            r, c = self.selected_crystal_pos
            dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
            nr, nc = r + dr, c + dc

            if 0 <= nr < self.BOARD_ROWS and 0 <= nc < self.BOARD_COLS and self.board[nr, nc] == 0:
                is_major_action = True
                # Swap crystal with empty space
                self.board[nr, nc], self.board[r, c] = self.board[r, c], self.board[nr, nc]
                self.selected_crystal_pos = None # Deselect after move
                # Sound: Crystal slide
                
                chain_reward = 0
                chain_score = 0
                chain_count = 0
                
                # Process matches in a loop for chain reactions
                while True:
                    matches = self._find_matches()
                    if not matches:
                        break
                    
                    chain_count += 1
                    # Sound: Match success (louder for chains)
                    
                    for match_group in matches:
                        match_size = len(match_group)
                        
                        # Calculate score
                        points = match_size * 10
                        chain_score += points
                        
                        # Calculate reward
                        chain_reward += 1 * (match_size // 3) # Continuous
                        if match_size == 5: chain_reward += 10
                        elif match_size == 6: chain_reward += 20
                        else: chain_reward += 30
                        
                        # Remove crystals and create particles
                        for mr, mc in match_group:
                            color_idx = self.board[mr, mc] - 1
                            if color_idx >= 0:
                                self._create_particles(self._iso_to_screen(mr, mc), self.CRYSTAL_COLORS[color_idx], 10)
                            self.board[mr, mc] = 0
                
                    self._handle_gravity_and_refill()
                
                if chain_count > 0:
                    self.score += chain_score
                    reward += chain_reward
                else:
                    # Move with no match
                    reward = -0.1
                
                self.moves_left -= 1
        
        # Minor Actions (if no major action was taken)
        if not is_major_action:
            if shift_pressed: # Cycle selection
                self._cycle_selection()
                # Sound: Select UI
            elif space_pressed: # Toggle selection at cursor
                r, c = self.cursor_pos
                if self.selected_crystal_pos == [r, c]:
                    self.selected_crystal_pos = None
                    # Sound: Deselect UI
                elif self.board[r, c] != 0:
                    self.selected_crystal_pos = [r, c]
                    # Sound: Select UI
            elif movement in [1, 2, 3, 4]: # Move cursor
                dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
                self.cursor_pos[0] = max(0, min(self.BOARD_ROWS - 1, self.cursor_pos[0] + dr))
                self.cursor_pos[1] = max(0, min(self.BOARD_COLS - 1, self.cursor_pos[1] + dc))
                # Sound: Cursor tick

        # --- Termination Check ---
        terminated = False
        if self.score >= self.TARGET_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
            # Sound: Victory
        elif self.moves_left <= 0:
            reward -= 10
            terminated = True
            self.game_over = True
            # Sound: Failure
        
        self.last_reward = reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self._draw_background()
        self._draw_grid()
        self._draw_crystals()
        self._draw_cursor_and_selection()
        self._update_and_draw_particles()
        self._draw_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "moves_left": self.moves_left, "last_reward": self.last_reward}

    # --- Drawing Methods ---

    def _iso_to_screen(self, r, c):
        x = self.ORIGIN_X + (c - r) * (self.TILE_WIDTH // 2)
        y = self.ORIGIN_Y + (c + r) * (self.TILE_HEIGHT // 2)
        return int(x), int(y)

    def _draw_background(self):
        self.screen.fill(self.COLOR_BG)
        # Add subtle texture
        for _ in range(100):
            x = self.np_random.integers(0, self.WIDTH)
            y = self.np_random.integers(0, self.HEIGHT)
            c = self.np_random.integers(28, 38)
            self.screen.set_at((x, y), (c, c, c+5))

    def _draw_grid(self):
        for r in range(self.BOARD_ROWS + 1):
            p1 = self._iso_to_screen(r, -0.5)
            p2 = self._iso_to_screen(r, self.BOARD_COLS - 0.5)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        for c in range(self.BOARD_COLS + 1):
            p1 = self._iso_to_screen(-0.5, c)
            p2 = self._iso_to_screen(self.BOARD_ROWS - 0.5, c)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

    def _draw_crystals(self):
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                if self.board[r, c] != 0:
                    self._draw_crystal(r, c, self.board[r, c] - 1)

    def _draw_crystal(self, r, c, color_idx):
        x, y = self._iso_to_screen(r, c)
        color = self.CRYSTAL_COLORS[color_idx]
        
        # Darker shades for 3D effect
        color_dark = tuple(max(0, val - 60) for val in color)
        color_darker = tuple(max(0, val - 90) for val in color)

        h = self.TILE_HEIGHT
        w = self.TILE_WIDTH
        
        # Points for isometric cube
        top_point = (x, y - h // 2)
        left_point = (x - w // 2, y)
        right_point = (x + w // 2, y)
        bottom_point = (x, y + h // 2)
        
        # Draw faces
        # Top face
        pygame.gfxdraw.filled_polygon(self.screen, [top_point, left_point, (x, y), right_point], color)
        # Left face
        pygame.gfxdraw.filled_polygon(self.screen, [left_point, (x, y), bottom_point, (x - w // 2, y + h // 2)], color_dark)
        # Right face
        pygame.gfxdraw.filled_polygon(self.screen, [right_point, (x, y), bottom_point, (x + w // 2, y + h // 2)], color_darker)
        
        # Anti-aliased outline
        pygame.gfxdraw.aapolygon(self.screen, [top_point, left_point, (x - w // 2, y + h // 2), bottom_point, (x + w // 2, y + h // 2), right_point], (0,0,0,50))


    def _draw_cursor_and_selection(self):
        # Draw cursor
        r, c = self.cursor_pos
        x, y = self._iso_to_screen(r, c)
        points = [
            (x, y - self.TILE_HEIGHT // 2), (x - self.TILE_WIDTH // 2, y),
            (x, y + self.TILE_HEIGHT // 2), (x + self.TILE_WIDTH // 2, y)
        ]
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, [p for p in points], 2)

        # Draw selection highlight
        if self.selected_crystal_pos:
            r, c = self.selected_crystal_pos
            x, y = self._iso_to_screen(r, c)
            pygame.gfxdraw.filled_circle(self.screen, x, y + self.TILE_HEIGHT // 2, 12, (*self.COLOR_SELECTION, 80))
            pygame.gfxdraw.aacircle(self.screen, x, y + self.TILE_HEIGHT // 2, 12, self.COLOR_SELECTION)


    def _draw_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        moves_text = self.font_main.render(f"MOVES: {self.moves_left}", True, (255, 255, 255))
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.score >= self.TARGET_SCORE else "GAME OVER"
            color = (100, 255, 100) if self.score >= self.TARGET_SCORE else (255, 100, 100)
            end_text = self.font_main.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, 50))
            self.screen.blit(end_text, end_rect)


    # --- Particle System ---

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(20, 40)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'color': color})

    def _update_and_draw_particles(self):
        remaining_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifetime'] -= 1
            if p['lifetime'] > 0:
                remaining_particles.append(p)
                alpha = max(0, min(255, int(255 * (p['lifetime'] / 30))))
                color = (*p['color'], alpha)
                radius = int(max(1, p['lifetime'] * 0.1))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)
        self.particles = remaining_particles

    # --- Game Logic Helpers ---

    def _find_matches(self):
        visited = set()
        all_matches = []
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                if (r, c) in visited or self.board[r, c] == 0:
                    continue
                
                color = self.board[r, c]
                q = [(r, c)]
                current_match = set([(r, c)])
                visited.add((r, c))

                head = 0
                while head < len(q):
                    curr_r, curr_c = q[head]
                    head += 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < self.BOARD_ROWS and 0 <= nc < self.BOARD_COLS and \
                           (nr, nc) not in visited and self.board[nr, nc] == color:
                            visited.add((nr, nc))
                            current_match.add((nr, nc))
                            q.append((nr, nc))
                
                if len(current_match) >= self.MATCH_MIN_SIZE:
                    all_matches.append(current_match)
        return all_matches

    def _handle_gravity_and_refill(self):
        for c in range(self.BOARD_COLS):
            empty_row = self.BOARD_ROWS - 1
            for r in range(self.BOARD_ROWS - 1, -1, -1):
                if self.board[r, c] != 0:
                    self.board[empty_row, c], self.board[r, c] = self.board[r, c], self.board[empty_row, c]
                    empty_row -= 1
            # Refill from top
            for r in range(empty_row, -1, -1):
                self.board[r, c] = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1)

    def _cycle_selection(self):
        crystals = []
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                if self.board[r,c] != 0:
                    crystals.append([r,c])
        
        if not crystals:
            self.selected_crystal_pos = None
            return

        if self.selected_crystal_pos is None:
            self.selected_crystal_pos = crystals[0]
        else:
            try:
                current_idx = crystals.index(self.selected_crystal_pos)
                next_idx = (current_idx + 1) % len(crystals)
                self.selected_crystal_pos = crystals[next_idx]
            except ValueError: # If selected is somehow not in list
                self.selected_crystal_pos = crystals[0]

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set a small delay between frames for human playability
    FRAME_DELAY_MS = 100
    
    print("\n" + "="*30)
    print("      Cavern Crystals Test")
    print("="*30)
    print(env.user_guide)
    print("="*30)

    # Game loop
    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        # Pygame event handling for human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
        
        # If any key was pressed, step the environment
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            print(f"Action: {action} -> Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")

        # Render the current state to the screen
        # This is for display purposes; the agent would use the 'obs' array
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        
        # We need a display for human play
        try:
            display_screen = pygame.display.get_surface()
            if display_screen is None:
                 display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
        except pygame.error:
            # Running in a headless environment
            pass

        pygame.time.wait(FRAME_DELAY_MS)

    print("\nGame Over!")
    print(f"Final Score: {info['score']}, Moves Left: {info['moves_left']}")
    env.close()