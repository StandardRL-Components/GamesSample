
# Generated: 2025-08-27T15:28:43.057784
# Source Brief: brief_01000.md
# Brief Index: 1000

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a crystal, "
        "then use arrow keys to push it. Press Space again to cancel selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Shift crystals to match three or more of the same color. "
        "Clear the board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_SIZE = 8
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MOVES_LIMIT = 10
    NUM_CRYSTAL_TYPES = 3

    # --- Colors ---
    COLOR_BG = (25, 20, 45)
    COLOR_GRID = (45, 40, 65)
    CRYSTAL_COLORS = {
        1: ((255, 50, 50), (180, 20, 20)),  # Red (main, side)
        2: ((50, 255, 50), (20, 180, 20)),  # Green
        3: ((50, 100, 255), (20, 60, 180)), # Blue
    }
    CRYSTAL_GLOW_COLORS = {
        1: (255, 100, 100, 50),
        2: (100, 255, 100, 50),
        3: (100, 150, 255, 50),
    }
    CURSOR_COLOR = (255, 255, 0)
    CURSOR_SELECT_COLOR = (255, 255, 255)
    PARTICLE_COLORS = {
        1: (255, 150, 150),
        2: (150, 255, 150),
        3: (150, 200, 255),
    }

    # --- Isometric Projection ---
    TILE_WIDTH_ISO = 48
    TILE_HEIGHT_ISO = 24
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 120

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 16)
        self.font_small = pygame.font.SysFont("monospace", 12)

        # Game state variables initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_crystal_pos = None
        self.game_phase = None # 'SELECTING' or 'PUSHING'
        self.moves_remaining = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.last_reward = 0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self._generate_initial_grid()
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_crystal_pos = None
        self.game_phase = 'SELECTING'
        self.moves_remaining = self.MOVES_LIMIT
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.last_reward = 0

        return self._get_observation(), self._get_info()

    def _generate_initial_grid(self):
        while True:
            grid = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
            # Ensure at least one move is possible.
            # This is a simple proxy: just ensure not too many of one color are clustered.
            # A full check is too slow for reset().
            if self._is_board_potentially_playable(grid):
                return grid

    def _is_board_potentially_playable(self, grid):
        # Simple heuristic to avoid impossible boards, not a guarantee.
        # Checks if any color takes up more than 60% of the board.
        for color in range(1, self.NUM_CRYSTAL_TYPES + 1):
            if np.sum(grid == color) > (self.GRID_SIZE * self.GRID_SIZE * 0.6):
                return False
        return True

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0

        # --- Action Handling ---
        if self.game_phase == 'SELECTING':
            reward = self._handle_selecting_phase(movement, space_pressed)
        elif self.game_phase == 'PUSHING':
            reward = self._handle_pushing_phase(movement, space_pressed)
        
        self.score += reward
        self.last_reward = reward

        # --- Termination Check ---
        num_crystals = np.count_nonzero(self.grid)
        if num_crystals == 0:
            self.game_over = True
            reward += 100  # Win bonus
            self.score += 100
        elif self.moves_remaining <= 0 and self.game_phase == 'SELECTING':
            self.game_over = True
            reward -= 50 # Loss penalty
            self.score -= 50
        
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_selecting_phase(self, movement, space_pressed):
        if movement != 0: # Move cursor
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_SIZE
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_SIZE
        
        if space_pressed:
            cy, cx = self.cursor_pos
            if self.grid[cy, cx] != 0:
                self.game_phase = 'PUSHING'
                self.selected_crystal_pos = list(self.cursor_pos) # Make a copy
        return 0

    def _handle_pushing_phase(self, movement, space_pressed):
        if space_pressed: # Cancel selection
            self.game_phase = 'SELECTING'
            self.selected_crystal_pos = None
            return 0
        
        if movement != 0: # Execute push
            self.moves_remaining -= 1
            direction = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            push_reward = self._execute_push(direction)
            
            # Add continuous penalty for remaining crystals
            num_crystals = np.count_nonzero(self.grid)
            penalty = num_crystals * 0.2
            return push_reward - penalty
        
        return 0

    def _execute_push(self, direction):
        if direction == (0, 0): return 0
        
        dy, dx = direction
        start_y, start_x = self.selected_crystal_pos

        # Trace the line of crystals to be pushed
        line_to_push = []
        curr_x, curr_y = start_x, start_y
        while 0 <= curr_y < self.GRID_SIZE and 0 <= curr_x < self.GRID_SIZE:
            if self.grid[curr_y, curr_x] == 0:
                break # Found an empty space
            line_to_push.append((curr_y, curr_x))
            curr_y += dy
            curr_x += dx
        
        # If we can't push (hit edge or empty space immediately)
        if not line_to_push or not (0 <= curr_y < self.GRID_SIZE and 0 <= curr_x < self.GRID_SIZE):
             self.game_phase = 'SELECTING'
             self.selected_crystal_pos = None
             return -1.0 # Penalize invalid move attempt

        # Push the line of crystals one step
        for y, x in reversed(line_to_push):
            self.grid[y + dy, x + dx] = self.grid[y, x]
        self.grid[start_y, start_x] = 0

        # Process matches and gravity
        match_reward, made_match = self._process_cascades()

        # Reset state after move
        self.game_phase = 'SELECTING'
        self.selected_crystal_pos = None

        if not made_match:
            return -1.0 # Penalize moves that don't result in a match
        
        return match_reward

    def _process_cascades(self):
        total_reward = 0
        made_match_at_all = False
        
        while True:
            matched_coords = self._find_all_matches()
            if not matched_coords:
                break
            
            made_match_at_all = True
            
            # Calculate reward and remove crystals
            num_matched = len(matched_coords)
            total_reward += num_matched * 10 # Base reward
            if num_matched > 3:
                total_reward += (num_matched - 3) * 5 # Combo bonus
            
            for y, x in matched_coords:
                color = self.grid[y, x]
                if color != 0:
                    # sound effect: crystal shatter
                    self._spawn_particles(x, y, color)
                    self.grid[y, x] = 0 # Clear crystal
            
            # Apply gravity
            self._apply_gravity()
        
        return total_reward, made_match_at_all

    def _find_all_matches(self):
        matched_coords = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color = self.grid[r, c]
                if color == 0: continue
                
                # Horizontal match
                if c + 2 < self.GRID_SIZE and self.grid[r, c+1] == color and self.grid[r, c+2] == color:
                    for i in range(3): matched_coords.add((r, c+i))
                
                # Vertical match
                if r + 2 < self.GRID_SIZE and self.grid[r+1, c] == color and self.grid[r+2, c] == color:
                    for i in range(3): matched_coords.add((r+i, c))
        return matched_coords

    def _apply_gravity(self):
        # sound effect: crystals falling
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
    
    def _spawn_particles(self, x, y, color):
        screen_x, screen_y = self._iso_to_cart(x, y)
        for _ in range(15): # Spawn 15 particles
            self.particles.append({
                'pos': [screen_x, screen_y],
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -1)],
                'life': self.np_random.uniform(15, 30),
                'color': self.PARTICLE_COLORS[color]
            })

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
            "crystals_remaining": np.count_nonzero(self.grid),
            "game_phase": self.game_phase
        }

    def _iso_to_cart(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * (self.TILE_WIDTH_ISO / 2)
        screen_y = self.ORIGIN_Y + (x + y) * (self.TILE_HEIGHT_ISO / 2)
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Update and draw particles
        self._update_and_draw_particles()
        
        # Draw grid lines
        for r in range(self.GRID_SIZE + 1):
            p1 = self._iso_to_cart(-0.5, r - 0.5)
            p2 = self._iso_to_cart(self.GRID_SIZE - 0.5, r - 0.5)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for c in range(self.GRID_SIZE + 1):
            p1 = self._iso_to_cart(c - 0.5, -0.5)
            p2 = self._iso_to_cart(c - 0.5, self.GRID_SIZE - 0.5)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
            
        # Draw crystals
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_id = self.grid[r, c]
                if color_id != 0:
                    self._draw_iso_cube(c, r, color_id)

        # Draw cursor
        if not self.game_over:
            cursor_y, cursor_x = self.cursor_pos
            is_selected = self.game_phase == 'PUSHING'
            self._draw_cursor(cursor_x, cursor_y, is_selected)

    def _draw_iso_cube(self, x, y, color_id):
        screen_x, screen_y = self._iso_to_cart(x, y)
        w, h = self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO
        
        main_color, side_color = self.CRYSTAL_COLORS[color_id]
        glow_color = self.CRYSTAL_GLOW_COLORS[color_id]

        # Glow effect
        glow_surface = pygame.Surface((w * 1.5, h * 1.5), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surface, glow_color, glow_surface.get_rect())
        self.screen.blit(glow_surface, (screen_x - w * 0.75, screen_y - h * 0.75))

        # Cube points
        top_face = [
            (screen_x, screen_y - h / 2),
            (screen_x + w / 2, screen_y),
            (screen_x, screen_y + h / 2),
            (screen_x - w / 2, screen_y)
        ]
        left_face = [
            (screen_x - w / 2, screen_y),
            (screen_x, screen_y + h / 2),
            (screen_x, screen_y + h / 2 + h),
            (screen_x - w / 2, screen_y + h)
        ]
        right_face = [
            (screen_x + w / 2, screen_y),
            (screen_x, screen_y + h / 2),
            (screen_x, screen_y + h / 2 + h),
            (screen_x + w / 2, screen_y + h)
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, left_face, side_color)
        pygame.gfxdraw.filled_polygon(self.screen, right_face, side_color)
        pygame.gfxdraw.filled_polygon(self.screen, top_face, main_color)
        
        pygame.gfxdraw.aapolygon(self.screen, left_face, side_color)
        pygame.gfxdraw.aapolygon(self.screen, right_face, side_color)
        pygame.gfxdraw.aapolygon(self.screen, top_face, main_color)

    def _draw_cursor(self, x, y, is_selected):
        screen_x, screen_y = self._iso_to_cart(x, y)
        w, h = self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO
        
        color = self.CURSOR_SELECT_COLOR if is_selected else self.CURSOR_COLOR
        thickness = 3 if is_selected else 2
        
        points = [
            (screen_x, screen_y - h / 2 - 4),
            (screen_x + w / 2 + 4, screen_y),
            (screen_x, screen_y + h / 2 + 4),
            (screen_x - w / 2 - 4, screen_y)
        ]
        pygame.draw.lines(self.screen, color, True, points, thickness)

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                size = max(1, int(p['life'] / 5))
                pygame.draw.circle(self.screen, p['color'], p['pos'], size)

    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_medium.render(f"Moves: {self.moves_remaining}", True, (255, 255, 255))
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))

        # Crystals remaining
        crystals_text = self.font_medium.render(f"Crystals: {np.count_nonzero(self.grid)}", True, (255, 255, 255))
        self.screen.blit(crystals_text, (10, 10))
        
        # Score
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

        # Game Over / Win message
        if self.game_over:
            num_crystals = np.count_nonzero(self.grid)
            msg = "YOU WIN!" if num_crystals == 0 else "GAME OVER"
            color = (100, 255, 100) if num_crystals == 0 else (255, 100, 100)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_text.get_height() // 2))

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # no-op, released, released
        
        # This human-play loop only sends an action on a key-down event
        # to match the turn-based nature of the game.
        event_happened = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                event_happened = True
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: action[2] = 1
                elif event.key == pygame.K_r: # Reset on 'r'
                    print("Resetting environment.")
                    env.reset()
        
        if event_happened:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Reward: {reward:.2f}, Score: {info['score']:.2f}")
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}")
                env.reset()

        # Render the environment's observation to the screen
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2)) # Transpose back for pygame
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(60) # Limit frame rate
        
    env.close()