
# Generated: 2025-08-27T23:42:03.786529
# Source Brief: brief_03550.md
# Brief Index: 3550

        
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
        "Controls: Use arrow keys to move the selection cursor. "
        "Press space to swap the selected block with the one in the direction you last moved."
    )

    game_description = (
        "Match-3 puzzle game. Swap adjacent blocks to clear the board before you run out of moves."
    )

    auto_advance = False
    
    # Class-level variable for difficulty progression
    total_clears = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000

        self.BOARD_DIM = 360
        self.CELL_SIZE = self.BOARD_DIM // self.GRID_SIZE
        self.BOARD_OFFSET_X = (self.WIDTH - self.BOARD_DIM) // 2
        self.BOARD_OFFSET_Y = (self.HEIGHT - self.BOARD_DIM) // 2 + 20

        # --- Colors ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TARGET = (255, 165, 0, 150)
        self.BLOCK_COLORS = {
            -1: (80, 80, 80),   # Obstacle
            1: (220, 50, 50),   # Red
            2: (50, 220, 50),   # Green
            3: (50, 100, 220),  # Blue
        }
        self.UI_COLOR = (240, 240, 240)

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
        self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24)
        
        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.last_move_dir = None
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.animations = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.last_move_dir = None # 1:up, 2:down, 3:left, 4:right
        self.animations = []

        self._generate_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_press, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Action ---
        # 1. Move cursor
        if movement != 0:
            self.last_move_dir = movement
            prev_pos = list(self.cursor_pos)
            if movement == 1: # Up
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 2: # Down
                self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
            elif movement == 3: # Left
                self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 4: # Right
                self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
            # sound: cursor_move.wav

        # 2. Handle swap attempt
        if space_press:
            self.moves_left -= 1
            reward += self._attempt_swap()

        # --- Process Game Logic (Matches, Gravity, Refill) ---
        chain_multiplier = 1
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            # sound: match_clear.wav
            clear_reward = self._clear_matches(matches, chain_multiplier)
            reward += clear_reward
            
            self._apply_gravity()
            self._refill_board()
            chain_multiplier += 0.5

        # --- Check for no possible moves ---
        if not self._find_possible_moves():
            self._shuffle_board()
            # sound: board_shuffle.wav

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self._is_board_clear():
            # Lost by running out of moves
            pass
        elif terminated and self._is_board_clear():
            # Won by clearing board
            reward += 100
            GameEnv.total_clears += 1
            # sound: win_fanfare.wav

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _attempt_swap(self):
        if self.last_move_dir is None:
            return -0.1 # Invalid attempt

        r1, c1 = self.cursor_pos
        r2, c2 = r1, c1

        if self.last_move_dir == 1: r2 -= 1 # Up
        elif self.last_move_dir == 2: r2 += 1 # Down
        elif self.last_move_dir == 3: c2 -= 1 # Left
        elif self.last_move_dir == 4: c2 += 1 # Right

        # Check bounds and if swapping with obstacle
        if not (0 <= r2 < self.GRID_SIZE and 0 <= c2 < self.GRID_SIZE):
            # sound: error.wav
            return -0.1 # Out of bounds
        if self.grid[r1, c1] == -1 or self.grid[r2, c2] == -1:
            # sound: error.wav
            return -0.1 # Swapping with obstacle

        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        # Check if swap created a match
        if not self._find_matches_at([ (r1,c1), (r2,c2) ]):
            # No match, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            # sound: swap_fail.wav
            return 0 # No penalty for a valid but non-matching swap, but move is consumed
        
        # sound: swap_success.wav
        # The reward for the match itself is calculated in the main loop
        return 0

    def _clear_matches(self, matches, multiplier):
        reward = 0
        num_cleared = len(matches)
        
        reward += num_cleared * 1 * multiplier
        if num_cleared == 4:
            reward += 5 * multiplier
        elif num_cleared >= 5:
            reward += 10 * multiplier
        
        for r, c in matches:
            self.grid[r, c] = 0 # 0 for empty
            # Create a particle effect animation
            self._add_particle_burst(r, c, self.BLOCK_COLORS[self.grid[r, c] if (r,c) not in matches else 1])

        self.score += int(reward)
        return int(reward)

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1

    def _refill_board(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, 4)

    def _generate_board(self):
        # Determine number of obstacles based on progression
        num_obstacles = 3 + (GameEnv.total_clears // 50)

        while True:
            self.grid = self.np_random.integers(1, 4, size=(self.GRID_SIZE, self.GRID_SIZE))
            
            # Place obstacles
            obstacle_pos = set()
            while len(obstacle_pos) < num_obstacles:
                r, c = self.np_random.integers(0, self.GRID_SIZE, size=2)
                obstacle_pos.add((r, c))
            for r, c in obstacle_pos:
                self.grid[r, c] = -1

            # Ensure no initial matches
            while self._find_matches():
                for r, c in self._find_matches():
                    if self.grid[r,c] != -1:
                        self.grid[r, c] = self.np_random.integers(1, 4)

            # Ensure at least one valid move exists
            if self._find_possible_moves():
                break

    def _shuffle_board(self):
        movable_blocks = []
        movable_indices = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] != -1:
                    movable_blocks.append(self.grid[r, c])
                    movable_indices.append((r, c))
        
        while True:
            self.np_random.shuffle(movable_blocks)
            temp_grid = self.grid.copy()
            for i, (r, c) in enumerate(movable_indices):
                temp_grid[r, c] = movable_blocks[i]
            
            self.grid = temp_grid
            if not self._find_matches() and self._find_possible_moves():
                break

    def _find_matches_at(self, positions):
        all_matches = set()
        for r_start, c_start in positions:
            color = self.grid[r_start, c_start]
            if color in {0, -1}:
                continue
            
            # Horizontal check
            h_match = {(r_start, c_start)}
            for c in range(c_start - 1, -1, -1):
                if self.grid[r_start, c] == color: h_match.add((r_start, c))
                else: break
            for c in range(c_start + 1, self.GRID_SIZE):
                if self.grid[r_start, c] == color: h_match.add((r_start, c))
                else: break
            if len(h_match) >= 3: all_matches.update(h_match)
            
            # Vertical check
            v_match = {(r_start, c_start)}
            for r in range(r_start - 1, -1, -1):
                if self.grid[r, c_start] == color: v_match.add((r, c_start))
                else: break
            for r in range(r_start + 1, self.GRID_SIZE):
                if self.grid[r, c_start] == color: v_match.add((r, c_start))
                else: break
            if len(v_match) >= 3: all_matches.update(v_match)
        return all_matches

    def _find_matches(self):
        return self._find_matches_at([(r,c) for r in range(self.GRID_SIZE) for c in range(self.GRID_SIZE)])

    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r,c] == -1: continue
                # Check swap right
                if c < self.GRID_SIZE - 1 and self.grid[r, c+1] != -1:
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                    if self._find_matches_at([(r,c), (r,c+1)]): moves.append(((r,c), (r,c+1)))
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c] # Swap back
                # Check swap down
                if r < self.GRID_SIZE - 1 and self.grid[r+1, c] != -1:
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                    if self._find_matches_at([(r,c), (r+1,c)]): moves.append(((r,c), (r+1,c)))
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c] # Swap back
        return moves

    def _is_board_clear(self):
        return not np.any((self.grid != -1) & (self.grid != 0))

    def _check_termination(self):
        if self.game_over: return True
        
        if self.moves_left <= 0 or self._is_board_clear() or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_draw_animations()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _add_particle_burst(self, r, c, color):
        px, py = self._get_pixel_pos(r, c)
        px += self.CELL_SIZE // 2
        py += self.CELL_SIZE // 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
            life = random.randint(10, 20)
            self.animations.append(['particle', [px, py], [vx, vy], life, color])
    
    def _update_and_draw_animations(self):
        # This is for transient visual effects like particles
        # Since auto_advance=False, these will only update and draw once per step.
        # For a smoother look, auto_advance=True would be needed.
        # Given the constraints, we will have 'bursts' of particles per step.
        
        active_animations = []
        for anim in self.animations:
            anim_type, pos, vel, life, color = anim
            if anim_type == 'particle':
                pos[0] += vel[0]
                pos[1] += vel[1]
                life -= 1
                if life > 0:
                    active_animations.append(anim)
                
                size = int(max(0, (life / 20) * 6))
                if size > 0:
                    pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), size)
        self.animations = active_animations

    def _get_pixel_pos(self, r, c):
        return (
            self.BOARD_OFFSET_X + c * self.CELL_SIZE,
            self.BOARD_OFFSET_Y + r * self.CELL_SIZE
        )

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.BOARD_OFFSET_X + i * self.CELL_SIZE, self.BOARD_OFFSET_Y)
            end_pos = (self.BOARD_OFFSET_X + i * self.CELL_SIZE, self.BOARD_OFFSET_Y + self.BOARD_DIM)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.BOARD_OFFSET_X, self.BOARD_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.BOARD_OFFSET_X + self.BOARD_DIM, self.BOARD_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw blocks
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_id = self.grid[r, c]
                if color_id == 0: continue

                x, y = self._get_pixel_pos(r, c)
                color = self.BLOCK_COLORS[color_id]
                
                block_rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                inner_rect = block_rect.inflate(-6, -6)
                
                pygame.draw.rect(self.screen, color, inner_rect, border_radius=4)
                
                # Add a subtle 3D effect
                light_color = tuple(min(255, val + 30) for val in color)
                dark_color = tuple(max(0, val - 30) for val in color)
                pygame.draw.line(self.screen, light_color, inner_rect.topleft, inner_rect.topright, 2)
                pygame.draw.line(self.screen, light_color, inner_rect.topleft, inner_rect.bottomleft, 2)
                pygame.draw.line(self.screen, dark_color, inner_rect.bottomright, inner_rect.topright, 2)
                pygame.draw.line(self.screen, dark_color, inner_rect.bottomright, inner_rect.bottomleft, 2)


        # Draw cursor
        cur_r, cur_c = self.cursor_pos
        cur_x, cur_y = self._get_pixel_pos(cur_r, cur_c)
        cursor_rect = pygame.Rect(cur_x, cur_y, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

        # Draw target indicator
        if self.last_move_dir is not None:
            tr, tc = cur_r, cur_c
            if self.last_move_dir == 1: tr -= 1
            elif self.last_move_dir == 2: tr += 1
            elif self.last_move_dir == 3: tc -= 1
            elif self.last_move_dir == 4: tc += 1
            
            if 0 <= tr < self.GRID_SIZE and 0 <= tc < self.GRID_SIZE:
                tx, ty = self._get_pixel_pos(tr, tc)
                target_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(target_surface, self.COLOR_TARGET, (0, 0, self.CELL_SIZE, self.CELL_SIZE), border_radius=5)
                self.screen.blit(target_surface, (tx, ty))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.UI_COLOR)
        self.screen.blit(score_text, (20, 10))

        # Moves
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.UI_COLOR)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 10))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self._is_board_clear():
                msg = "Board Cleared!"
            else:
                msg = "Game Over"
            
            end_text = self.font_large.render(msg, True, (255, 255, 100))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # --- Example Usage & Interactive Play ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for interactive play
    pygame.display.set_caption("Match-3 Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    print("\n--- Interactive Mode ---")
    print(env.user_guide)
    
    while running:
        action = [0, 0, 0] # no-op, release, release
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                
                # Note: This interactive loop sends one action per key press.
                # A more advanced agent could hold keys.
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                
                if event.key == pygame.K_SPACE: action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1

                # If any key was pressed, step the environment
                if any(a != 0 for a in action):
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Moves Left: {info['moves_left']}")
                    
                    if terminated:
                        print(f"--- Episode Finished --- Final Score: {info['score']}")

        # Draw the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    env.close()