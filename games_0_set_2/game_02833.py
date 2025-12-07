
# Generated: 2025-08-28T06:07:14.385010
# Source Brief: brief_02833.md
# Brief Index: 2833

        
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
        "Controls: Arrow keys to move the selector. Press Space to swap the selected gem with an adjacent one "
        "in the direction of your last move."
    )

    game_description = (
        "A fast-paced match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Race against the clock to reach the target score by creating combos and chain reactions."
    )

    auto_advance = True

    # --- Constants ---
    # Game parameters
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    NUM_GEM_TYPES = 5
    GEM_SIZE = 40
    ANIMATION_SPEED = 0.2  # Seconds per swap/fall
    WIN_SCORE = 1000
    GAME_TIME_SECONDS = 60.0

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 120, 255),   # Blue
        (255, 255, 80),   # Yellow
        (200, 80, 255),   # Purple
    ]
    COLOR_WHITE = (255, 255, 255)
    COLOR_TIMER_START = (0, 220, 0)
    COLOR_TIMER_END = (220, 0, 0)

    # UI
    UI_HEIGHT = 60
    BOARD_OFFSET_X = (640 - GRID_WIDTH * GEM_SIZE) // 2
    BOARD_OFFSET_Y = UI_HEIGHT + (400 - UI_HEIGHT - GRID_HEIGHT * GEM_SIZE) // 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.render_mode = render_mode
        self.np_random = None

        # State variables are initialized in reset()
        self.grid = None
        self.selector = None
        self.last_move_dir = None
        self.score = None
        self.steps = None
        self.timer = None
        self.game_state = None # 'IDLE', 'SWAPPING', 'MATCHING', 'FALLING'
        self.animation_progress = None
        self.swap_info = None
        self.matched_gems = None
        self.particles = None
        self.previous_space_held = None
        self.previous_shift_held = None
        self.step_reward = None
        
        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.GAME_TIME_SECONDS
        self.game_state = 'IDLE'
        self.animation_progress = 0.0
        self.swap_info = {}
        self.matched_gems = set()
        self.particles = []
        self.previous_space_held = 0
        self.previous_shift_held = 0
        self.step_reward = 0.0

        self.selector = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.last_move_dir = [0, 1] # Default to right
        
        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.step_reward = 0.0
        terminated = False
        
        # --- Time progression ---
        # Fixed timestep for 30 FPS gameplay feel
        dt = 1/30.0 
        self.timer = max(0, self.timer - dt)
        self.steps += 1
        
        # --- Handle Game State Machine ---
        if self.game_state == 'IDLE':
            self._handle_input(action)
        elif self.game_state in ['SWAPPING', 'SWAPPING_BACK']:
            self._update_swap_animation(dt)
        elif self.game_state == 'MATCHING':
            self._update_match_animation(dt)
        elif self.game_state == 'FALLING':
            self._update_fall_animation(dt)
            
        self._update_particles(dt)

        # --- Check for termination ---
        if self.timer <= 0:
            self.step_reward -= 10 # Penalty for running out of time
            terminated = True
        if self.score >= self.WIN_SCORE:
            self.step_reward += 100 # Bonus for winning
            terminated = True
        if self.steps >= 1800: # 60 seconds * 30 fps
            terminated = True

        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1], action[2]

        # --- Selector Movement ---
        if movement != 0:
            dy, dx = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[movement]
            self.selector[0] = np.clip(self.selector[0] + dy, 0, self.GRID_HEIGHT - 1)
            self.selector[1] = np.clip(self.selector[1] + dx, 0, self.GRID_WIDTH - 1)
            self.last_move_dir = [dy, dx]
        
        # --- Swap Action (on key press) ---
        if space_held and not self.previous_space_held:
            self._initiate_swap()
        
        self.previous_space_held = space_held
        self.previous_shift_held = shift_held # Reserved for future use

    def _initiate_swap(self):
        y1, x1 = self.selector
        y2, x2 = y1 + self.last_move_dir[0], x1 + self.last_move_dir[1]

        if not (0 <= y2 < self.GRID_HEIGHT and 0 <= x2 < self.GRID_WIDTH):
            return # Invalid swap location

        # Perform swap
        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]

        matches1 = self._find_matches_at(y1, x1)
        matches2 = self._find_matches_at(y2, x2)
        self.matched_gems = matches1.union(matches2)

        self.swap_info = {'pos1': (y1, x1), 'pos2': (y2, x2)}

        if self.matched_gems:
            self.game_state = 'SWAPPING'
        else:
            # No match, swap back
            self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
            self.game_state = 'SWAPPING_BACK'
            self.step_reward -= 0.1 # Penalty for invalid move

    def _update_swap_animation(self, dt):
        self.animation_progress += dt / self.ANIMATION_SPEED
        if self.animation_progress >= 1.0:
            self.animation_progress = 0.0
            self.swap_info = {}
            if self.game_state == 'SWAPPING_BACK':
                self.game_state = 'IDLE'
            else:
                self.game_state = 'MATCHING'

    def _update_match_animation(self, dt):
        self.animation_progress += dt / (self.ANIMATION_SPEED * 0.5)
        if self.animation_progress >= 1.0:
            self.animation_progress = 0.0
            self._process_matches()
            self.game_state = 'FALLING'

    def _update_fall_animation(self, dt):
        self.animation_progress += dt / self.ANIMATION_SPEED
        if self.animation_progress >= 1.0:
            self.animation_progress = 0.0
            self._apply_gravity()
            new_matches = self._find_all_matches()
            if new_matches:
                self.matched_gems = new_matches
                self.game_state = 'MATCHING'
            else:
                self.game_state = 'IDLE'
                if not self._find_possible_moves():
                    self._generate_board() # Reshuffle if no moves left
                    self.step_reward -= 5

    def _process_matches(self):
        num_matched = len(self.matched_gems)
        self.score += num_matched * 10
        self.step_reward += num_matched # Base reward
        if num_matched == 4: self.step_reward += 10
        if num_matched >= 5: self.step_reward += 20
        
        for y, x in self.matched_gems:
            self._create_particles(y, x, self.grid[y, x])
            self.grid[y, x] = -1 # Mark as empty
        # sfx: gem_match.wav

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y, x] != -1:
                    if y != empty_row:
                        self.grid[empty_row, x] = self.grid[y, x]
                        self.grid[y, x] = -1
                    empty_row -= 1
            # Fill new empty spaces at the top
            for y in range(empty_row, -1, -1):
                self.grid[y, x] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_gems()
        self._render_selector()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Board Generation & Validation ---
    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if not self._find_all_matches() and self._find_possible_moves():
                break

    def _find_all_matches(self):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                matches.update(self._find_matches_at(y, x))
        return matches

    def _find_matches_at(self, y, x):
        matches = set()
        gem_type = self.grid[y, x]
        if gem_type == -1: return matches

        # Horizontal
        h_matches = {(y, x)}
        for i in range(x - 1, -1, -1):
            if self.grid[y, i] == gem_type: h_matches.add((y, i))
            else: break
        for i in range(x + 1, self.GRID_WIDTH):
            if self.grid[y, i] == gem_type: h_matches.add((y, i))
            else: break
        if len(h_matches) >= 3: matches.update(h_matches)

        # Vertical
        v_matches = {(y, x)}
        for i in range(y - 1, -1, -1):
            if self.grid[i, x] == gem_type: v_matches.add((i, x))
            else: break
        for i in range(y + 1, self.GRID_HEIGHT):
            if self.grid[i, x] == gem_type: v_matches.add((i, x))
            else: break
        if len(v_matches) >= 3: matches.update(v_matches)
        
        return matches

    def _find_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Try swapping right
                if x < self.GRID_WIDTH - 1:
                    self.grid[y, x], self.grid[y, x + 1] = self.grid[y, x + 1], self.grid[y, x]
                    if self._find_matches_at(y, x) or self._find_matches_at(y, x + 1):
                        self.grid[y, x], self.grid[y, x + 1] = self.grid[y, x + 1], self.grid[y, x]
                        return True
                    self.grid[y, x], self.grid[y, x + 1] = self.grid[y, x + 1], self.grid[y, x]
                # Try swapping down
                if y < self.GRID_HEIGHT - 1:
                    self.grid[y, x], self.grid[y + 1, x] = self.grid[y + 1, x], self.grid[y, x]
                    if self._find_matches_at(y, x) or self._find_matches_at(y + 1, x):
                        self.grid[y, x], self.grid[y + 1, x] = self.grid[y + 1, x], self.grid[y, x]
                        return True
                    self.grid[y, x], self.grid[y + 1, x] = self.grid[y + 1, x], self.grid[y, x]
        return False

    # --- Rendering ---
    def _render_grid(self):
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.BOARD_OFFSET_X, self.BOARD_OFFSET_Y + y * self.GEM_SIZE)
            end_pos = (self.BOARD_OFFSET_X + self.GRID_WIDTH * self.GEM_SIZE, self.BOARD_OFFSET_Y + y * self.GEM_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.BOARD_OFFSET_X + x * self.GEM_SIZE, self.BOARD_OFFSET_Y)
            end_pos = (self.BOARD_OFFSET_X + x * self.GEM_SIZE, self.BOARD_OFFSET_Y + self.GRID_HEIGHT * self.GEM_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_gems(self):
        p = self.animation_progress
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[y, x]
                if gem_type == -1: continue

                pos_x = self.BOARD_OFFSET_X + x * self.GEM_SIZE
                pos_y = self.BOARD_OFFSET_Y + y * self.GEM_SIZE
                
                # Handle animations
                if self.game_state in ['SWAPPING', 'SWAPPING_BACK'] and self.swap_info:
                    y1, x1 = self.swap_info['pos1']
                    y2, x2 = self.swap_info['pos2']
                    if (y, x) == (y1, x1):
                        pos_x = self._lerp(pos_x, self.BOARD_OFFSET_X + x2 * self.GEM_SIZE, p)
                        pos_y = self._lerp(pos_y, self.BOARD_OFFSET_Y + y2 * self.GEM_SIZE, p)
                    elif (y, x) == (y2, x2):
                        pos_x = self._lerp(pos_x, self.BOARD_OFFSET_X + x1 * self.GEM_SIZE, p)
                        pos_y = self._lerp(pos_y, self.BOARD_OFFSET_Y + y1 * self.GEM_SIZE, p)
                
                elif self.game_state == 'FALLING':
                    fall_dist = 0
                    for r in range(y - 1, -1, -1):
                        if self.grid[r, x] == -1: fall_dist += 1
                    if fall_dist > 0:
                        start_y = self.BOARD_OFFSET_Y + (y - fall_dist) * self.GEM_SIZE
                        end_y = self.BOARD_OFFSET_Y + y * self.GEM_SIZE
                        pos_y = self._lerp(start_y, end_y, p)

                size = self.GEM_SIZE
                if self.game_state == 'MATCHING' and (y, x) in self.matched_gems:
                    size = self._lerp(self.GEM_SIZE, 0, p)

                self._draw_gem(pos_x, pos_y, gem_type, size)

    def _draw_gem(self, x, y, gem_type, size):
        if size <= 0: return
        center_x, center_y = int(x + self.GEM_SIZE / 2), int(y + self.GEM_SIZE / 2)
        radius = int(size * 0.45)
        
        color = self.GEM_COLORS[gem_type]
        highlight = tuple(min(255, c + 60) for c in color)
        
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, highlight)
        
        # Add a little shine
        shine_x = center_x - radius // 2
        shine_y = center_y - radius // 2
        pygame.gfxdraw.filled_circle(self.screen, shine_x, shine_y, radius // 4, self.COLOR_WHITE)
        pygame.gfxdraw.aacircle(self.screen, shine_x, shine_y, radius // 4, self.COLOR_WHITE)

    def _render_selector(self):
        if self.game_state != 'IDLE': return
        y, x = self.selector
        rect = pygame.Rect(
            self.BOARD_OFFSET_X + x * self.GEM_SIZE,
            self.BOARD_OFFSET_Y + y * self.GEM_SIZE,
            self.GEM_SIZE,
            self.GEM_SIZE
        )
        alpha = int(128 + 127 * math.sin(self.steps * 0.2))
        color = (*self.COLOR_WHITE, alpha)
        
        # Create a temporary surface for the glowing rectangle
        temp_surface = pygame.Surface((self.GEM_SIZE, self.GEM_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(temp_surface, color, (0, 0, self.GEM_SIZE, self.GEM_SIZE), 4, border_radius=5)
        self.screen.blit(temp_surface, rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (20, 10))

        # Timer bar
        timer_ratio = self.timer / self.GAME_TIME_SECONDS
        bar_width = 200
        bar_height = 20
        bar_x = 640 - bar_width - 20
        bar_y = 20
        
        current_color = self._lerp_color(self.COLOR_TIMER_END, self.COLOR_TIMER_START, timer_ratio)
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, current_color, (bar_x, bar_y, bar_width * timer_ratio, bar_height))

    def _create_particles(self, y, x, gem_type):
        px = self.BOARD_OFFSET_X + x * self.GEM_SIZE + self.GEM_SIZE / 2
        py = self.BOARD_OFFSET_Y + y * self.GEM_SIZE + self.GEM_SIZE / 2
        color = self.GEM_COLORS[gem_type]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.uniform(0.3, 0.7)
            self.particles.append([px, py, vx, vy, lifetime, lifetime, color])

    def _update_particles(self, dt):
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[4] -= dt
        self.particles = [p for p in self.particles if p[4] > 0]

    def _render_particles(self):
        for x, y, vx, vy, life, max_life, color in self.particles:
            alpha = int(255 * (life / max_life))
            radius = int(3 * (life / max_life))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), radius, (*color, alpha))

    # --- Helper functions ---
    @staticmethod
    def _lerp(a, b, t):
        return a + (b - a) * t

    @staticmethod
    def _lerp_color(c1, c2, t):
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )
        
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

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            env.reset()

        clock.tick(30) # Run at 30 FPS

    pygame.quit()