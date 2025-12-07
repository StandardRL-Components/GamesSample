
# Generated: 2025-08-27T23:26:26.842269
# Source Brief: brief_03460.md
# Brief Index: 3460

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem, "
        "then move to an adjacent gem and press Space again to swap. Press Shift to cancel a selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent gems to create matches of 3 or more in a race against time to reach a target score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 8
    NUM_GEM_TYPES = 3  # Red, Green, Blue
    FPS = 30
    TIME_LIMIT_SECONDS = 180
    TARGET_SCORE = 1000
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (50, 60, 70)
    COLOR_TEXT = (220, 220, 230)
    COLOR_GEMS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (50, 100, 255),  # Blue
    ]
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTED = (255, 255, 255, 100) # Semi-transparent white

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Board layout calculation
        self.board_area_size = int(self.HEIGHT * 0.9)
        self.cell_size = self.board_area_size // self.GRID_SIZE
        self.gem_radius = int(self.cell_size * 0.4)
        self.board_offset_x = (self.WIDTH - self.board_area_size) // 2
        self.board_offset_y = (self.HEIGHT - self.board_area_size) // 2
        
        # State variables (initialized in reset)
        self.board = None
        self.cursor_pos = None
        self.selected_pos = None
        self.score = None
        self.steps = None
        self.time_remaining_frames = None
        self.game_over = None
        self.last_space_held = None
        self.last_shift_held = None
        
        # Animation related state
        self.animations = [] # List to hold all active animations
        self.reward_this_step = 0
        
        # Initialize state variables
        self.reset()
        self.validate_implementation() # Self-check
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.score = 0
        self.steps = 0
        self.time_remaining_frames = self.MAX_STEPS
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_pos = None
        self.last_space_held = False
        self.last_shift_held = False
        self.animations = []

        # Generate a valid starting board (no initial matches)
        self._initialize_board()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # Update game logic
        self.steps += 1
        self.time_remaining_frames -= 1
        self.reward_this_step = 0

        # Process input only if no major animations are running
        if not self._is_animating():
            self._handle_input(action)
        
        # Update animations and game logic (like cascades)
        self._update_animations()
        if not self._is_animating():
             self._run_cascade_cycle()

        reward = self.reward_this_step
        terminated = self._check_termination()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_board()
        self._render_gems()
        self._render_animations()
        self._render_cursor()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    # --- Game Logic Helpers ---

    def _initialize_board(self):
        self.board = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        while self._find_matches():
            matches = self._find_matches()
            for r, c in matches:
                self.board[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

    def _handle_input(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Detect key presses (rising edge)
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        # Handle cursor movement
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Up
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1) # Down
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Left
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1) # Right
        
        if shift_pressed:
            # sfx: cancel_selection
            self.selected_pos = None
        
        if space_pressed:
            r, c = self.cursor_pos
            if self.selected_pos is None:
                # sfx: select_gem
                self.selected_pos = (r, c)
            else:
                sr, sc = self.selected_pos
                if abs(r - sr) + abs(c - sc) == 1:
                    # sfx: swap_gem
                    self._create_swap_animation((sr, sc), (r, c))
                else:
                    # sfx: invalid_selection
                    self.selected_pos = (r, c)

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _run_cascade_cycle(self):
        matches = self._find_matches()
        if not matches:
            return

        # sfx: match_found
        num_cleared = len(matches)
        self.reward_this_step += num_cleared
        self.score += num_cleared
        
        if num_cleared == 4:
            self.reward_this_step += 10
            self.score += 10
        elif num_cleared >= 5:
            self.reward_this_step += 20
            self.score += 20
        
        self._create_match_animation(matches)
        for r, c in matches:
            self.board[r, c] = 0 # 0 represents an empty space
            
        self._apply_gravity()
        self._refill_board()

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.board[r, c]
                if gem_type == 0: continue
                
                if c < self.GRID_SIZE - 2 and self.board[r, c+1] == gem_type and self.board[r, c+2] == gem_type:
                    for i in range(3): matches.add((r, c+i))
                
                if r < self.GRID_SIZE - 2 and self.board[r+1, c] == gem_type and self.board[r+2, c] == gem_type:
                    for i in range(3): matches.add((r+i, c))
        return list(matches)

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.board[r, c] != 0:
                    if r != empty_row:
                        self.board[empty_row, c] = self.board[r, c]
                        self.board[r, c] = 0
                        self._create_fall_animation((r,c), (empty_row, c))
                    empty_row -= 1
    
    def _refill_board(self):
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE):
                if self.board[r, c] == 0:
                    self.board[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
                    self._create_fall_animation((-1, c), (r, c))

    def _check_termination(self):
        if self.score >= self.TARGET_SCORE:
            self.reward_this_step += 100 # Victory bonus
            self.game_over = True
            return True
        if self.time_remaining_frames <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
        
    # --- Animation System ---

    def _is_animating(self):
        return len(self.animations) > 0

    def _create_swap_animation(self, pos1, pos2):
        self.animations.append({
            'type': 'swap', 'pos1': pos1, 'pos2': pos2,
            'gem1_type': self.board[pos1], 'gem2_type': self.board[pos2],
            'progress': 0.0, 'duration': 8
        })
        self.board[pos1] = 0
        self.board[pos2] = 0

    def _create_match_animation(self, positions):
        for pos in positions:
            self.animations.append({'type': 'match', 'pos': pos, 'progress': 0.0, 'duration': 10})
    
    def _create_fall_animation(self, start_pos, end_pos):
        self.animations.append({
            'type': 'fall', 'start_pos': start_pos, 'end_pos': end_pos,
            'gem_type': self.board[end_pos], 'progress': 0.0, 'duration': 6
        })
        self.board[end_pos] = 0

    def _update_animations(self):
        # sfx: particle_fizzle
        finished_animations = []
        for anim in self.animations:
            anim['progress'] += 1.0 / anim['duration']
            if anim['progress'] >= 1.0:
                finished_animations.append(anim)

        for anim in finished_animations:
            if anim['type'] == 'swap':
                r1, c1 = anim['pos1']; r2, c2 = anim['pos2']
                self.board[r1, c1], self.board[r2, c2] = anim['gem2_type'], anim['gem1_type']
                if not self._find_matches():
                    # sfx: invalid_swap
                    self._create_swap_animation(anim['pos2'], anim['pos1'])
                self.selected_pos = None
            
            elif anim['type'] == 'fall':
                r, c = anim['end_pos']
                self.board[r, c] = anim['gem_type']

            self.animations.remove(anim)

    # --- Rendering ---

    def _grid_to_pixel(self, r, c):
        x = self.board_offset_x + c * self.cell_size + self.cell_size // 2
        y = self.board_offset_y + r * self.cell_size + self.cell_size // 2
        return int(x), int(y)

    def _render_board(self):
        for i in range(self.GRID_SIZE + 1):
            start_x = self.board_offset_x + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, self.board_offset_y), (start_x, self.board_offset_y + self.board_area_size), 1)
            start_y = self.board_offset_y + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.board_offset_x, start_y), (self.board_offset_x + self.board_area_size, start_y), 1)

    def _render_gems(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.board[r, c]
                if gem_type > 0:
                    px, py = self._grid_to_pixel(r, c)
                    color = self.COLOR_GEMS[gem_type - 1]
                    pygame.gfxdraw.filled_circle(self.screen, px, py, self.gem_radius, color)
                    pygame.gfxdraw.aacircle(self.screen, px, py, self.gem_radius, tuple(min(255, x+50) for x in color))

    def _render_cursor(self):
        if self._is_animating(): return
        r, c = self.cursor_pos
        px, py = self._grid_to_pixel(r, c)
        
        if self.selected_pos:
            sr, sc = self.selected_pos
            spx, spy = self._grid_to_pixel(sr, sc)
            s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            s.fill(self.COLOR_SELECTED)
            self.screen.blit(s, (spx - self.cell_size//2, spy - self.cell_size//2))

        rect = pygame.Rect(px - self.cell_size//2, py - self.cell_size//2, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3)

    def _render_animations(self):
        for anim in self.animations:
            if anim['type'] == 'swap':
                p = anim['progress']
                x1, y1 = self._grid_to_pixel(*anim['pos1']); x2, y2 = self._grid_to_pixel(*anim['pos2'])
                
                curr_x1 = int(x1 + (x2 - x1) * p); curr_y1 = int(y1 + (y2 - y1) * p)
                color1 = self.COLOR_GEMS[anim['gem1_type'] - 1]
                pygame.gfxdraw.filled_circle(self.screen, curr_x1, curr_y1, self.gem_radius, color1)
                
                curr_x2 = int(x2 + (x1 - x2) * p); curr_y2 = int(y2 + (y1 - y2) * p)
                color2 = self.COLOR_GEMS[anim['gem2_type'] - 1]
                pygame.gfxdraw.filled_circle(self.screen, curr_x2, curr_y2, self.gem_radius, color2)
            
            elif anim['type'] == 'match':
                p = anim['progress']
                px, py = self._grid_to_pixel(*anim['pos'])
                alpha = int(255 * (1 - p)); radius = int(self.gem_radius * (1 + p * 0.5))
                pygame.gfxdraw.filled_circle(self.screen, px, py, radius, (255, 255, 255, alpha))

            elif anim['type'] == 'fall':
                p = anim['progress']
                sx, sy_start = self._grid_to_pixel(*anim['start_pos']); ex, ey_end = self._grid_to_pixel(*anim['end_pos'])
                curr_y = int(sy_start + (ey_end - sy_start) * p)
                color = self.COLOR_GEMS[anim['gem_type'] - 1]
                pygame.gfxdraw.filled_circle(self.screen, ex, curr_y, self.gem_radius, color)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left_sec = max(0, self.time_remaining_frames // self.FPS)
        minutes, seconds = divmod(time_left_sec, 60)
        timer_text = self.font_large.render(f"Time: {minutes:02}:{seconds:02}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.score >= self.TARGET_SCORE else "TIME UP"
            msg_surf = self.font_large.render(msg, True, (255, 215, 0))
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    key_to_action = {
        pygame.K_UP:    [1, 0, 0], pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0], pygame.K_RIGHT: [4, 0, 0],
        pygame.K_SPACE: [0, 1, 0], pygame.K_LSHIFT:[0, 0, 1], pygame.K_RSHIFT: [0, 0, 1],
    }

    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    terminated = False
    
    print("--- Gem Swap ---"); print(env.user_guide)

    while True:
        action = np.array([0, 0, 0])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close(); pygame.quit(); exit()
        
        keys = pygame.key.get_pressed()
        for key, act in key_to_action.items():
            if keys[key]: action += np.array(act)
        
        action[0] = min(action[0], 4)
        action[1] = min(action[1], 1)
        action[2] = min(action[2], 1)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0: print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward}")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000)
            obs, info = env.reset()

        clock.tick(GameEnv.FPS)