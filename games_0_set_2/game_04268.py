
# Generated: 2025-08-28T01:54:08.701704
# Source Brief: brief_04268.md
# Brief Index: 4268

        
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
        "Controls: Use arrow keys (↑↓←→) to move the cursor. "
        "Press Shift to toggle push axis (row/column). "
        "Press Space to push the selected row/column."
    )

    game_description = (
        "Strategically shift rows and columns of shimmering crystals in an isometric cavern. "
        "Create matching sets of three or more to clear them from the board and score points."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_CRYSTAL_TYPES = 3  # 1, 2, 3
        self.MAX_STEPS = 1000

        # --- Visuals ---
        self.TILE_WIDTH_HALF = 24
        self.TILE_HEIGHT_HALF = 12
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # --- Colors ---
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_CURSOR = (255, 255, 100)
        self.CRYSTAL_COLORS = {
            1: ((255, 80, 80), (220, 50, 50), (160, 20, 20)),  # Red
            2: ((80, 255, 80), (50, 220, 50), (20, 160, 20)),  # Green
            3: ((80, 120, 255), (50, 90, 220), (20, 60, 160)), # Blue
        }

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        # --- State ---
        self.np_random = None
        self.board = None
        self.cursor_pos = None
        self.push_axis = None # 0 for row (horizontal), 1 for column (vertical)
        self.steps = None
        self.score = None
        self.game_over = None
        self.animations = []
        self.particles = []
        
        # Action handling helpers
        self.shift_lock = False
        self.space_lock = False

        self.reset()
        self.validate_implementation()

    def _iso_to_screen(self, r, c):
        x = self.ORIGIN_X + (c - r) * self.TILE_WIDTH_HALF
        y = self.ORIGIN_Y + (c + r) * self.TILE_HEIGHT_HALF
        return int(x), int(y)

    def _draw_iso_tile(self, surface, r, c, color_tuple):
        x, y = self._iso_to_screen(r, c)
        main_color, side_color, top_color = color_tuple
        
        points_top = [
            (x, y - self.TILE_HEIGHT_HALF),
            (x + self.TILE_WIDTH_HALF, y),
            (x, y + self.TILE_HEIGHT_HALF),
            (x - self.TILE_WIDTH_HALF, y),
        ]
        
        # We draw a simple rhombus for performance and clarity
        pygame.gfxdraw.aapolygon(surface, points_top, top_color)
        pygame.gfxdraw.filled_polygon(surface, points_top, top_color)
        
        # Highlight/Shine
        shine_points = [
            (x - self.TILE_WIDTH_HALF * 0.8, y + self.TILE_HEIGHT_HALF * 0.2),
            (x, y - self.TILE_HEIGHT_HALF * 0.8),
            (x + self.TILE_WIDTH_HALF * 0.2, y - self.TILE_HEIGHT_HALF * 0.6)
        ]
        pygame.draw.lines(surface, (255,255,255, 80), False, shine_points, 2)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.push_axis = 0 # Start with horizontal push
        self.animations = []
        self.particles = []
        self.shift_lock = False
        self.space_lock = False
        
        self._generate_board()

        return self._get_observation(), self._get_info()

    def _generate_board(self):
        while True:
            self.board = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            
            # Ensure no initial matches
            while True:
                matches, _ = self._find_matches(self.board)
                if not np.any(matches):
                    break
                self.board[matches] = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1, size=np.sum(matches))

            # Ensure there's at least one possible move
            if self._check_for_possible_moves(self.board):
                break
    
    def _check_for_possible_moves(self, board):
        # Check horizontal pushes
        for r in range(self.GRID_SIZE):
            temp_board = np.copy(board)
            temp_board[r, :] = np.roll(temp_board[r, :], 1)
            matches, _ = self._find_matches(temp_board)
            if np.any(matches):
                return True
        # Check vertical pushes
        for c in range(self.GRID_SIZE):
            temp_board = np.copy(board)
            temp_board[:, c] = np.roll(temp_board[:, c], 1, axis=0)
            matches, _ = self._find_matches(temp_board)
            if np.any(matches):
                return True
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_val, shift_val = action[0], action[1], action[2]
        space_held = space_val == 1
        shift_held = shift_val == 1
        
        reward = 0
        self.steps += 1

        # --- Handle Input ---
        # Toggle push axis on Shift press
        if shift_held and not self.shift_lock:
            self.push_axis = 1 - self.push_axis
            self.shift_lock = True
        elif not shift_held:
            self.shift_lock = False

        # Move cursor
        if movement == 1: # Up
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: # Down
            self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
        elif movement == 3: # Left
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: # Right
            self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)

        # Execute push on Space press
        if space_held and not self.space_lock:
            reward = self._execute_push()
            self.space_lock = True
        elif not space_held:
            self.space_lock = False

        # --- Check Termination ---
        if np.all(self.board == 0): # Win condition
            reward += 100
            self.game_over = True
        elif not self._check_for_possible_moves(self.board): # Loss condition
            self.game_over = True
        elif self.steps >= self.MAX_STEPS: # Step limit
            self.game_over = True

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _execute_push(self):
        r, c = self.cursor_pos
        
        # --- Perform the shift ---
        if self.push_axis == 0: # Horizontal
            self.board[r, :] = np.roll(self.board[r, :], 1)
        else: # Vertical
            self.board[:, c] = np.roll(self.board[:, c], 1, axis=0)
            
        # --- Resolve matches and gravity in a loop for chain reactions ---
        total_match_reward = 0
        any_match_in_turn = False
        
        while True:
            matches, match_counts = self._find_matches(self.board)
            if not np.any(matches):
                break
            
            any_match_in_turn = True
            
            # Calculate reward and create particles
            for length, count in match_counts.items():
                if length == 3: total_match_reward += 10 * count
                elif length == 4: total_match_reward += 20 * count
                else: total_match_reward += 30 * count
            
            for r_m, c_m in np.argwhere(matches):
                self._create_particles(r_m, c_m, self.board[r_m, c_m])
            
            # Clear matched crystals
            self.board[matches] = 0
            
            # Apply gravity
            for col in range(self.GRID_SIZE):
                empty_slots = np.where(self.board[:, col] == 0)[0]
                filled_slots = np.where(self.board[:, col] != 0)[0]
                
                if len(empty_slots) > 0 and len(filled_slots) > 0:
                    for r_fill in reversed(filled_slots):
                        # Find first empty slot below the current filled one
                        fall_to = empty_slots[empty_slots > r_fill]
                        if len(fall_to) > 0:
                            target_r = np.max(fall_to)
                            self.board[target_r, col] = self.board[r_fill, col]
                            self.board[r_fill, col] = 0
                            # Update empty_slots for next iteration in this column
                            empty_slots = np.where(self.board[:, col] == 0)[0]

        if not any_match_in_turn:
            return -0.2 # Penalty for a move with no match
        
        return total_match_reward

    def _find_matches(self, board):
        matches = np.zeros_like(board, dtype=bool)
        match_counts = {}

        # Horizontal matches
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                val = board[r, c]
                if val != 0 and val == board[r, c+1] == board[r, c+2]:
                    length = 3
                    while c + length < self.GRID_SIZE and board[r, c+length] == val:
                        length += 1
                    for i in range(length):
                        matches[r, c+i] = True
                    match_counts[length] = match_counts.get(length, 0) + 1
                    c += length -1 # Skip checked cells

        # Vertical matches
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                val = board[r, c]
                if val != 0 and val == board[r+1, c] == board[r+2, c]:
                    length = 3
                    while r + length < self.GRID_SIZE and board[r+length, c] == val:
                        length += 1
                    for i in range(length):
                        matches[r+i, c] = True
                    match_counts[length] = match_counts.get(length, 0) + 1
                    r += length - 1 # Skip checked cells
        
        return matches, match_counts

    def _create_particles(self, r, c, crystal_type):
        # sound: crystal_shatter.wav
        if crystal_type == 0: return
        color = self.CRYSTAL_COLORS[crystal_type][0]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            pos = list(self._iso_to_screen(r, c))
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([pos, vel, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _get_observation(self):
        self._update_particles()
        
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_SIZE + 1):
            p1 = self._iso_to_screen(r, 0)
            p2 = self._iso_to_screen(r, self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, (p1[0] - self.TILE_WIDTH_HALF, p1[1]), (p2[0], p2[1] + self.TILE_HEIGHT_HALF), 1)
        for c in range(self.GRID_SIZE + 1):
            p1 = self._iso_to_screen(0, c)
            p2 = self._iso_to_screen(self.GRID_SIZE, c)
            pygame.draw.line(self.screen, self.COLOR_GRID, (p1[0] + self.TILE_WIDTH_HALF, p1[1]), (p2[0], p2[1] - self.TILE_HEIGHT_HALF), 1)

        # Draw crystals
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                crystal_type = self.board[r, c]
                if crystal_type != 0:
                    self._draw_iso_tile(self.screen, r, c, self.CRYSTAL_COLORS[crystal_type])

        # Draw cursor
        r, c = self.cursor_pos
        x, y = self._iso_to_screen(r, c)
        cursor_points = [
            (x, y - self.TILE_HEIGHT_HALF - 4),
            (x + self.TILE_WIDTH_HALF + 4, y),
            (x, y + self.TILE_HEIGHT_HALF + 4),
            (x - self.TILE_WIDTH_HALF - 4, y),
        ]
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, cursor_points, 2)
        
        # Draw push axis indicator
        if self.push_axis == 0: # Horizontal
            p1 = (x - self.TILE_WIDTH_HALF - 8, y)
            p2 = (x + self.TILE_WIDTH_HALF + 8, y)
        else: # Vertical
            p1 = (x, y - self.TILE_HEIGHT_HALF - 8)
            p2 = (x, y + self.TILE_HEIGHT_HALF + 8)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, p1, p2, 2)


        # Draw particles
        for pos, vel, lifetime, color in self.particles:
            alpha = max(0, min(255, int(255 * (lifetime / 30.0))))
            size = max(1, int(4 * (lifetime / 30.0)))
            pygame.draw.rect(self.screen, (*color, alpha), (int(pos[0]), int(pos[1]), size, size))

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        moves_text = self.font_main.render(f"MOVES: {self.steps}/{self.MAX_STEPS}", True, (255, 255, 255))
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": list(self.cursor_pos),
            "possible_moves": self._check_for_possible_moves(self.board)
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Crystal Cavern")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = [0, 0, 0] # No-op, no space, no shift
    
    print("\n" + "="*30)
    print("      CRYSTAL CAVERN")
    print("="*30)
    print(env.user_guide)
    print("="*30)


    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                print("--- Game Reset ---")

        if terminated:
            action = [0, 0, 0] # No-op
        else:
            keys = pygame.key.get_pressed()
            mov = 0
            if keys[pygame.K_UP]: mov = 1
            elif keys[pygame.K_DOWN]: mov = 2
            elif keys[pygame.K_LEFT]: mov = 3
            elif keys[pygame.K_RIGHT]: mov = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [mov, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.1f}, Score: {info['score']}")

        if terminated:
            print(f"--- GAME OVER ---")
            print(f"Final Score: {info['score']} in {info['steps']} moves.")
            
        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we control the step rate here
        env.clock.tick(30) # Limit to 30 FPS for smooth controls
        
    env.close()