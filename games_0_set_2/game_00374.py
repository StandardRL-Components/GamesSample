
# Generated: 2025-08-27T13:27:44.965126
# Source Brief: brief_00374.md
# Brief Index: 374

        
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
        "Controls: Arrow keys to move selector. Space to swap with the tile in the direction of your last movement."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent colored gems to create matches of 3 or more. Clear the entire board before you run out of moves. Grey stones are locked and can be unlocked by making matches next to them."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # Class variable to track difficulty across resets
    _current_level_locked_tiles = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = 6
        self.TILE_SIZE = 50
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_SIZE * self.TILE_SIZE) / 2 - 100
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_SIZE * self.TILE_SIZE) / 2
        self.NUM_TILE_TYPES = 5  # Number of colors
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (45, 50, 55)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_LOCKED = (110, 120, 130)
        self.COLOR_LOCKED_ICON = (80, 90, 100)
        self.TILE_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (100, 150, 255), # Blue
            (255, 255, 100), # Yellow
            (200, 100, 255), # Purple
        ]
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_score = pygame.font.SysFont("Consolas", 18)

        # Initialize state variables
        self.grid = None
        self.locked_tile_health = None
        self.cursor_pos = None
        self.last_move_direction = None
        self.moves_remaining = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.cursor_pos = np.array([0, 0])
        self.last_move_direction = np.array([0, -1]) # Default to Up
        self.particles = []

        # Board generation
        while True:
            self._generate_board()
            if self._check_for_possible_moves():
                break

        return self._get_observation(), self._get_info()
    
    def _generate_board(self):
        self.grid = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        self.locked_tile_health = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        # Ensure no initial matches
        while True:
            matches = self._find_matches()
            if not matches:
                break
            for r, c in matches:
                # Avoid re-matching with the same color
                forbidden_colors = set()
                if r > 0: forbidden_colors.add(self.grid[r-1, c])
                if c > 0: forbidden_colors.add(self.grid[r, c-1])
                possible_colors = [i for i in range(1, self.NUM_TILE_TYPES + 1) if i not in forbidden_colors]
                if not possible_colors: possible_colors = list(range(1, self.NUM_TILE_TYPES + 1))
                self.grid[r, c] = self.np_random.choice(possible_colors)

        # Add locked tiles
        num_locked = min(GameEnv._current_level_locked_tiles, (self.GRID_SIZE**2) // 2)
        locked_positions = set()
        while len(locked_positions) < num_locked:
            r, c = self.np_random.integers(0, self.GRID_SIZE, size=2)
            if (r,c) not in locked_positions:
                self.grid[r, c] = -1 # -1 for locked
                self.locked_tile_health[r, c] = 1 # Needs 1 adjacent match to unlock
                locked_positions.add((r,c))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_pressed = action[1] == 1
        
        self.steps += 1
        reward = 0
        
        # Every step call where an action is attempted consumes a move
        if movement != 0 or space_pressed:
            self.moves_remaining -= 1
            reward -= 0.01 # Small penalty for any move to encourage efficiency
        
        # 1. Handle cursor movement
        if movement != 0:
            move_map = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]} # up, down, left, right
            direction = np.array(move_map[movement])
            self.cursor_pos = (self.cursor_pos + direction) % self.GRID_SIZE
            self.last_move_direction = direction

        # 2. Handle swap action
        if space_pressed:
            swap_target_pos = (self.cursor_pos + self.last_move_direction)
            
            # Check if swap is valid (within bounds, not with a locked tile)
            r1, c1 = self.cursor_pos
            if 0 <= swap_target_pos[0] < self.GRID_SIZE and 0 <= swap_target_pos[1] < self.GRID_SIZE:
                r2, c2 = swap_target_pos
                # Cannot swap a locked tile
                if self.grid[r1, c1] != -1 and self.grid[r2, c2] != -1:
                    # Perform swap
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                    
                    # Process matches and chain reactions
                    chain_reward = self._process_all_matches()
                    
                    if chain_reward == 0:
                        # No match found, swap was unproductive. Revert swap.
                        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                    else:
                        reward += chain_reward
                # else: Mechanically invalid swap (with locked tile), move is consumed, no effect.
            # else: Mechanically invalid swap (off-grid), move is consumed, no effect.
        
        # 3. Handle board state and termination
        # Anti-softlock: Reshuffle if no moves are possible
        if not self._check_for_possible_moves() and not self._is_board_clear():
            self._reshuffle_board()
            # This is a free action, so we refund the move cost
            if movement != 0 or space_pressed:
                self.moves_remaining += 1

        terminated = self._check_termination()
        if terminated:
            if self._is_board_clear():
                reward += 100 # Win bonus
                GameEnv._current_level_locked_tiles = min(10, GameEnv._current_level_locked_tiles + 1)
            else:
                reward -= 10 # Loss penalty

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_all_matches(self):
        total_chain_reward = 0
        while True:
            matches = self._find_matches()
            if not matches:
                break

            # Calculate reward for current matches
            match_counts = {}
            for r, c in matches:
                color = self.grid[r, c]
                match_counts[color] = match_counts.get(color, 0) + 1
            
            # Simplified reward based on total tiles cleared in this wave
            num_cleared = len(matches)
            if num_cleared == 3: total_chain_reward += 0.1
            elif num_cleared == 4: total_chain_reward += 0.2
            else: total_chain_reward += 0.3
            
            # Clear matched tiles and handle unlocks
            for r, c in matches:
                # Sound effect placeholder
                # play_sound('match.wav')
                self._create_particles(r, c, self.grid[r, c])
                self.grid[r, c] = 0 # 0 for empty
                
                # Check for unlocking adjacent locked tiles
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and self.grid[nr, nc] == -1:
                        self.locked_tile_health[nr, nc] -= 1
                        if self.locked_tile_health[nr, nc] <= 0:
                            self.grid[nr, nc] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
                            total_chain_reward += 5 # Unlock bonus
                            # Sound effect placeholder
                            # play_sound('unlock.wav')

            self._apply_gravity()
            self._fill_top_rows()
        
        return total_chain_reward

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color = self.grid[r, c]
                if color <= 0: continue

                # Horizontal
                if c < self.GRID_SIZE - 2 and self.grid[r, c+1] == color and self.grid[r, c+2] == color:
                    matches.add((r, c)); matches.add((r, c+1)); matches.add((r, c+2))
                # Vertical
                if r < self.GRID_SIZE - 2 and self.grid[r+1, c] == color and self.grid[r+2, c] == color:
                    matches.add((r, c)); matches.add((r+1, c)); matches.add((r+2, c))
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1

    def _fill_top_rows(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)

    def _check_for_possible_moves(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == -1: continue # Can't move locked tiles
                # Check swap right
                if c < self.GRID_SIZE - 1 and self.grid[r, c+1] != -1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches():
                        self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                # Check swap down
                if r < self.GRID_SIZE - 1 and self.grid[r+1, c] != -1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches():
                        self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return False

    def _reshuffle_board(self):
        # Sound effect placeholder
        # play_sound('reshuffle.wav')
        while True:
            movable_tiles = []
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    if self.grid[r, c] > 0:
                        movable_tiles.append(self.grid[r, c])
            
            self.np_random.shuffle(movable_tiles)
            
            idx = 0
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    if self.grid[r, c] > 0:
                        self.grid[r, c] = movable_tiles[idx]
                        idx += 1
            
            if not self._find_matches() and self._check_for_possible_moves():
                break

    def _is_board_clear(self):
        return np.all(self.grid <= 0)

    def _check_termination(self):
        if self.game_over:
            return True
        if self._is_board_clear():
            self.game_over = True
            return True
        if self.moves_remaining <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_render_particles()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "level_locked_tiles": GameEnv._current_level_locked_tiles,
        }

    def _render_game(self):
        # Draw grid background
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.TILE_SIZE,
                    self.GRID_OFFSET_Y + r * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_type = self.grid[r, c]
                if tile_type == 0: continue # Empty

                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.TILE_SIZE + 4,
                    self.GRID_OFFSET_Y + r * self.TILE_SIZE + 4,
                    self.TILE_SIZE - 8, self.TILE_SIZE - 8
                )
                
                if tile_type > 0: # Colored tile
                    color = self.TILE_COLORS[tile_type - 1]
                    pygame.draw.rect(self.screen, color, rect, border_radius=5)
                    # Add a subtle highlight for 3D effect
                    highlight_color = tuple(min(255, x + 40) for x in color)
                    pygame.draw.rect(self.screen, highlight_color, (rect.x, rect.y, rect.width, 5), border_top_left_radius=5, border_top_right_radius=5)
                elif tile_type == -1: # Locked tile
                    pygame.draw.rect(self.screen, self.COLOR_LOCKED, rect, border_radius=5)
                    # Draw a lock icon
                    icon_cx, icon_cy = rect.centerx, rect.centery
                    pygame.draw.rect(self.screen, self.COLOR_LOCKED_ICON, (icon_cx - 6, icon_cy, 12, 8), border_radius=2)
                    pygame.draw.arc(self.screen, self.COLOR_LOCKED_ICON, (icon_cx - 6, icon_cy - 8, 12, 12), 0, math.pi, 3)


        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + cursor_c * self.TILE_SIZE,
            self.GRID_OFFSET_Y + cursor_r * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)
        
        # Draw swap direction indicator
        indicator_start = np.array(cursor_rect.center)
        indicator_end = indicator_start + self.last_move_direction * (self.TILE_SIZE / 2.5)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, indicator_start, indicator_end, 2)
        
    def _render_ui(self):
        # Game Title and Info Panel
        panel_rect = pygame.Rect(self.GRID_OFFSET_X + self.GRID_SIZE * self.TILE_SIZE + 20, self.GRID_OFFSET_Y, 200, 300)
        pygame.draw.rect(self.screen, (35, 40, 45), panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, (55, 60, 65), panel_rect, 2, border_radius=10)

        title_text = self.font_main.render("GEM SWAP", True, (200, 220, 255))
        self.screen.blit(title_text, (panel_rect.x + (panel_rect.width - title_text.get_width()) / 2, panel_rect.y + 20))

        moves_text = self.font_score.render(f"Moves: {self.moves_remaining}", True, (255, 255, 255))
        self.screen.blit(moves_text, (panel_rect.x + 20, panel_rect.y + 70))
        
        score_val = f"{self.score:.2f}" if isinstance(self.score, float) else str(self.score)
        score_text = self.font_score.render(f"Score: {score_val}", True, (255, 255, 255))
        self.screen.blit(score_text, (panel_rect.x + 20, panel_rect.y + 100))
        
        level_text = self.font_score.render(f"Locked: {GameEnv._current_level_locked_tiles}", True, (150, 160, 170))
        self.screen.blit(level_text, (panel_rect.x + 20, panel_rect.y + 130))

        if self.game_over:
            status = "BOARD CLEARED!" if self._is_board_clear() else "OUT OF MOVES"
            color = (100, 255, 100) if self._is_board_clear() else (255, 100, 100)
            end_text = self.font_main.render(status, True, color)
            self.screen.blit(end_text, (panel_rect.x + (panel_rect.width - end_text.get_width()) / 2, panel_rect.y + 200))
            
    def _create_particles(self, r, c, tile_type):
        if tile_type <= 0: return
        px = self.GRID_OFFSET_X + c * self.TILE_SIZE + self.TILE_SIZE / 2
        py = self.GRID_OFFSET_Y + r * self.TILE_SIZE + self.TILE_SIZE / 2
        color = self.TILE_COLORS[tile_type - 1]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([px, py, vx, vy, lifetime, color])

    def _update_and_render_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2]  # x
            p[1] += p[3]  # y
            p[4] -= 1     # lifetime
            
            # Gravity
            p[3] += 0.1
            
            size = max(0, int(p[4] / 5))
            if size > 0:
                alpha = int(255 * (p[4] / 30))
                color = p[5]
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color + (alpha,), (size, size), size)
                self.screen.blit(s, (int(p[0]) - size, int(p[1]) - size))

    def close(self):
        pygame.font.quit()
        pygame.quit()
        super().close()

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override screen for direct rendering
    env.screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Swap Gym Environment")

    print(env.user_guide)
    print(env.game_description)

    while not done:
        action = np.array([0, 0, 0]) # Default no-op
        
        # Human controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
                    print("--- GAME RESET ---")
                
                if not done:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    print(f"Action: {action}, Reward: {reward:.3f}, Score: {info['score']:.2f}, Moves: {info['moves_remaining']}")
                    if terminated:
                        print("--- GAME OVER ---")

        # Render the current state to the display
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS

    env.close()