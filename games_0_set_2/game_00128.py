
# Generated: 2025-08-27T12:41:20.079101
# Source Brief: brief_00128.md
# Brief Index: 128

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a gem, then space on an adjacent gem to swap. Shift to reshuffle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced match-3 puzzle game. Swap gems to create matches of 3 or more and clear the board before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.GEM_SIZE = 40
        self.BOARD_OFFSET_X = (self.WIDTH - self.GRID_SIZE * self.GEM_SIZE) // 2
        self.BOARD_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.GEM_SIZE) // 2
        self.MAX_STEPS = 1000
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 180

        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_GRID = (50, 50, 70)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_SELECT = (255, 255, 255)
        self.GEM_COLORS = [
            (0, 0, 0),  # 0: Empty
            (255, 50, 50),  # 1: Red
            (50, 255, 50),  # 2: Green
            (50, 100, 255), # 3: Blue
        ]
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18)
        
        # Internal state variables
        self.board = None
        self.cursor_pos = None
        self.selected_gem = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        
        # Animation state machine
        self.game_phase = 'IDLE' # IDLE, SWAP, MATCH, FALL
        self.animation_timer = 0
        self.animation_data = {}
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.board = self._create_initial_board()
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_gem = None
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        
        self.game_phase = 'IDLE'
        self.animation_timer = 0
        self.animation_data = {}
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0
        
        # --- Animation State Machine ---
        if self.game_phase != 'IDLE':
            self._update_animation()
        else:
            # --- Player Input Processing ---
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            space_pressed = space_held and not self.last_space_held
            shift_pressed = shift_held and not self.last_shift_held
            self.last_space_held = space_held
            self.last_shift_held = shift_held

            self._handle_movement(movement)

            if shift_pressed:
                # Reshuffle is a free action, no reward/penalty
                self._reshuffle_board()
                self.game_phase = 'IDLE'

            if space_pressed:
                reward = self._handle_selection()

        terminated = self._check_termination()
        if terminated and not self.game_over:
            if np.sum(self.board > 0) == 0: # Win
                reward += 100
            else: # Loss
                reward += -50
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --------------------------------------------------------------------------
    # Core Game Logic
    # --------------------------------------------------------------------------

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[0] -= 1
        elif movement == 2: self.cursor_pos[0] += 1
        elif movement == 3: self.cursor_pos[1] -= 1
        elif movement == 4: self.cursor_pos[1] += 1
        
        self.cursor_pos[0] = self.cursor_pos[0] % self.GRID_SIZE
        self.cursor_pos[1] = self.cursor_pos[1] % self.GRID_SIZE

    def _handle_selection(self):
        r, c = self.cursor_pos
        if self.selected_gem is None:
            self.selected_gem = [r, c]
            # sound: select_gem.wav
            return 0
        else:
            sel_r, sel_c = self.selected_gem
            is_adjacent = abs(r - sel_r) + abs(c - sel_c) == 1
            
            if not is_adjacent:
                self.selected_gem = [r, c] # Select new gem
                # sound: select_gem.wav
                return 0
            
            # Perform swap
            self._swap_gems(self.selected_gem, self.cursor_pos)
            self.selected_gem = None
            
            # Check if this swap creates a match
            matches = self._find_matches()
            if not matches:
                # Invalid swap, swap back
                self.animation_data['is_invalid'] = True
                # sound: invalid_swap.wav
                return -0.1
            else:
                self.animation_data['is_invalid'] = False
                # sound: swap_valid.wav
                return 0 # Reward is handled in the animation phase

    def _process_matches(self):
        matches = self._find_matches()
        if not matches:
            self.game_phase = 'IDLE'
            # Check for no valid moves after cascade
            if not self._find_valid_moves():
                self._reshuffle_board()
            return 0

        # sound: match_found.wav
        num_cleared = len(matches)
        reward = num_cleared * 1.0
        if num_cleared > 3:
            reward += 5

        self.score += reward

        self.animation_data = {'type': 'match', 'gems': list(matches), 'timer': 10}
        self.game_phase = 'MATCH'
        self.animation_timer = 10
        return reward

    def _swap_gems(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]
        
        self.game_phase = 'SWAP'
        self.animation_timer = 10
        self.animation_data = {
            'type': 'swap',
            'pos1': pos1,
            'pos2': pos2,
            'timer': 10
        }

    def _update_animation(self):
        self.animation_timer -= 1
        if self.animation_timer > 0:
            return

        if self.game_phase == 'SWAP':
            if self.animation_data.get('is_invalid'):
                # Swap back
                self._swap_gems(self.animation_data['pos1'], self.animation_data['pos2'])
                self.animation_data['is_invalid'] = False # Prevent infinite loop
            else:
                self._process_matches()

        elif self.game_phase == 'MATCH':
            # Remove matched gems from board
            for r, c in self.animation_data['gems']:
                self.board[r, c] = 0
            self.game_phase = 'FALL'
            self.animation_timer = 10
            self.animation_data = {'type': 'fall', 'timer': 10}
            # sound: gems_fall.wav

        elif self.game_phase == 'FALL':
            self._apply_gravity()
            self._refill_board()
            # After falling and refilling, check for new cascade matches
            self._process_matches()

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.board[r, c] != 0:
                    if r != empty_row:
                        self.board[empty_row, c] = self.board[r, c]
                        self.board[r, c] = 0
                    empty_row -= 1
                    
    def _refill_board(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.board[r, c] == 0:
                    self.board[r, c] = self.np_random.integers(1, 4)

    # --------------------------------------------------------------------------
    # Board Generation and Validation
    # --------------------------------------------------------------------------

    def _create_initial_board(self):
        while True:
            board = self.np_random.integers(1, 4, size=(self.GRID_SIZE, self.GRID_SIZE))
            # Ensure no matches on creation
            while self._find_matches_on_board(board):
                board = self.np_random.integers(1, 4, size=(self.GRID_SIZE, self.GRID_SIZE))
            
            # Ensure at least one valid move exists
            if self._find_valid_moves_on_board(board):
                return board

    def _reshuffle_board(self):
        # sound: reshuffle.wav
        self.board = self._create_initial_board()
        self.selected_gem = None

    def _find_matches(self):
        return self._find_matches_on_board(self.board)

    def _find_matches_on_board(self, board):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem = board[r, c]
                if gem == 0: continue
                # Horizontal
                if c < self.GRID_SIZE - 2 and board[r, c+1] == gem and board[r, c+2] == gem:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_SIZE - 2 and board[r+1, c] == gem and board[r+2, c] == gem:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _find_valid_moves(self):
        return self._find_valid_moves_on_board(self.board)

    def _find_valid_moves_on_board(self, board):
        temp_board = np.copy(board)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Try swapping right
                if c < self.GRID_SIZE - 1:
                    temp_board[r,c], temp_board[r,c+1] = temp_board[r,c+1], temp_board[r,c]
                    if self._find_matches_on_board(temp_board):
                        return True
                    temp_board[r,c], temp_board[r,c+1] = temp_board[r,c+1], temp_board[r,c] # Swap back
                # Try swapping down
                if r < self.GRID_SIZE - 1:
                    temp_board[r,c], temp_board[r+1,c] = temp_board[r+1,c], temp_board[r,c]
                    if self._find_matches_on_board(temp_board):
                        return True
                    temp_board[r,c], temp_board[r+1,c] = temp_board[r+1,c], temp_board[r,c] # Swap back
        return False

    # --------------------------------------------------------------------------
    # Gymnasium Interface
    # --------------------------------------------------------------------------

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
            "gems_left": np.sum(self.board > 0),
            "cursor_pos": self.cursor_pos,
            "selected_gem": self.selected_gem
        }

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        if np.sum(self.board > 0) == 0: # All gems cleared
            return True
        return False

    def _calculate_reward(self):
        # Rewards are calculated and assigned during specific events
        # This function is not used in this event-driven reward model
        return 0

    # --------------------------------------------------------------------------
    # Rendering
    # --------------------------------------------------------------------------

    def _render_game(self):
        self._draw_grid()
        self._draw_gems()
        self._draw_cursor_and_selection()

    def _draw_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.BOARD_OFFSET_X + i * self.GEM_SIZE, self.BOARD_OFFSET_Y)
            end_pos = (self.BOARD_OFFSET_X + i * self.GEM_SIZE, self.BOARD_OFFSET_Y + self.GRID_SIZE * self.GEM_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
            # Horizontal lines
            start_pos = (self.BOARD_OFFSET_X, self.BOARD_OFFSET_Y + i * self.GEM_SIZE)
            end_pos = (self.BOARD_OFFSET_X + self.GRID_SIZE * self.GEM_SIZE, self.BOARD_OFFSET_Y + i * self.GEM_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
    
    def _draw_gems(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.board[r, c]
                if gem_type == 0:
                    continue

                x = self.BOARD_OFFSET_X + c * self.GEM_SIZE
                y = self.BOARD_OFFSET_Y + r * self.GEM_SIZE
                
                # Handle animations
                if self.game_phase == 'SWAP' and self.animation_data['type'] == 'swap':
                    p1, p2 = self.animation_data['pos1'], self.animation_data['pos2']
                    progress = 1.0 - (self.animation_timer / self.animation_data['timer'])
                    if [r, c] == p1:
                        x = int(self.BOARD_OFFSET_X + (c + (p2[1] - c) * progress) * self.GEM_SIZE)
                        y = int(self.BOARD_OFFSET_Y + (r + (p2[0] - r) * progress) * self.GEM_SIZE)
                    elif [r, c] == p2:
                        x = int(self.BOARD_OFFSET_X + (c + (p1[1] - c) * progress) * self.GEM_SIZE)
                        y = int(self.BOARD_OFFSET_Y + (r + (p1[0] - r) * progress) * self.GEM_SIZE)

                elif self.game_phase == 'MATCH' and self.animation_data['type'] == 'match':
                    if (r,c) in self.animation_data['gems']:
                        # Flash effect
                        if self.animation_timer % 4 < 2:
                            self._draw_single_gem(gem_type, x, y, self.GEM_SIZE)
                        continue

                self._draw_single_gem(gem_type, x, y, self.GEM_SIZE)

    def _draw_single_gem(self, gem_type, x, y, size):
        color = self.GEM_COLORS[gem_type]
        rect = pygame.Rect(x, y, size, size)
        center_x, center_y = x + size // 2, y + size // 2
        radius = int(size * 0.4)

        if gem_type == 1: # Red Circle
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        elif gem_type == 2: # Green Square
            padding = int(size * 0.1)
            pygame.draw.rect(self.screen, color, rect.inflate(-padding*2, -padding*2), border_radius=4)
        elif gem_type == 3: # Blue Triangle
            padding = int(size * 0.15)
            points = [
                (center_x, y + padding),
                (x + size - padding, y + size - padding),
                (x + padding, y + size - padding)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_cursor_and_selection(self):
        # Draw selected gem highlight
        if self.selected_gem is not None:
            r, c = self.selected_gem
            rect = pygame.Rect(
                self.BOARD_OFFSET_X + c * self.GEM_SIZE,
                self.BOARD_OFFSET_Y + r * self.GEM_SIZE,
                self.GEM_SIZE, self.GEM_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, 3)

        # Draw cursor
        r, c = self.cursor_pos
        rect = pygame.Rect(
            self.BOARD_OFFSET_X + c * self.GEM_SIZE,
            self.BOARD_OFFSET_Y + r * self.GEM_SIZE,
            self.GEM_SIZE, self.GEM_SIZE
        )
        # Pulsating effect for cursor
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 * 3
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, int(2 + pulse))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        minutes = int(time_left) // 60
        seconds = int(time_left) % 60
        timer_text = self.font_large.render(f"Time: {minutes:02}:{seconds:02}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        # Gems Remaining
        gems_left = np.sum(self.board > 0)
        gems_text = self.font_small.render(f"{gems_left} / {self.GRID_SIZE**2} Gems Left", True, self.COLOR_TEXT)
        self.screen.blit(gems_text, (self.WIDTH // 2 - gems_text.get_width() // 2, self.HEIGHT - 30))

    def validate_implementation(self):
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
        
        # Test that initial board has a move
        assert self._find_valid_moves()
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # pip install gymnasium[classic-control]
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'mac', 'dummy'
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # In a turn-based game, we only want to step when an action is taken
        # This simple human player loop steps every frame a key is held.
        # A more advanced loop would only step on key *presses*.
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause for 2 seconds before restarting

        clock.tick(30) # Run at 30 FPS

    env.close()
    pygame.quit()