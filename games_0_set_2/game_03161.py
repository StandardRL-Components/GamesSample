import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space while pressing an arrow key to swap gems."
    )

    # Short, user-facing description of the game:
    game_description = (
        "Strategically swap adjacent gems to create matches of 3 or more in a race against time to clear the board."
    )

    # Frames only advance when an action is received.
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    NUM_GEM_TYPES = 6
    GEM_SIZE = 40
    GRID_LINE_WIDTH = 2
    BOARD_OFFSET_X = (640 - (GRID_WIDTH * (GEM_SIZE + GRID_LINE_WIDTH))) // 2
    BOARD_OFFSET_Y = (400 - (GRID_HEIGHT * (GEM_SIZE + GRID_LINE_WIDTH))) + 10

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_SELECTOR = (255, 255, 255)
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 120, 255),   # Blue
        (255, 255, 80),   # Yellow
        (255, 80, 255),   # Magenta
        (80, 255, 255),   # Cyan
    ]
    
    # Game settings
    MAX_TIME = 30  # seconds
    TIME_PER_STEP = 0.05 # Each step advances time by this much

    # Animation speeds
    SWAP_SPEED = 0.25 # seconds
    FALL_SPEED = 0.2 # seconds
    CLEAR_SPEED = 0.3 # seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.screen_width = 640
        self.screen_height = 400

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)
        
        # Internal state variables
        self.board = None
        self.selector_pos = None
        self.time_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.game_state = None
        self.animation_timer = None
        self.swap_info = None
        self.clearing_gems = None
        self.falling_gems = None
        self.particles = None
        self.last_reward = 0.0
        
        # Initialize state, but don't call reset here as super().reset() will be called by the wrapper
        # The validation call requires a seeded RNG, which is only set in reset().
        # self.reset()
        
        # CRITICAL: Validate implementation
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_TIME
        
        self.selector_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self._generate_board()
        
        self.game_state = "IDLE" # States: IDLE, SWAP_ANIM, REVERT_ANIM, CLEAR_ANIM, FALL_ANIM
        self.animation_timer = 0
        self.swap_info = {}
        self.clearing_gems = set()
        self.falling_gems = []
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.time_left = max(0, self.time_left - self.TIME_PER_STEP)
        
        reward = 0.0
        
        # Process game state machine
        if self.game_state == "IDLE":
            reward += self._handle_input(action)
        else:
            self._update_animations()

        # Update particles
        self._update_particles()
        
        # Check for termination conditions
        terminated = bool(self.time_left <= 0 or self._is_board_clear())
        if terminated and not self.game_over:
            self.game_over = True
            if self._is_board_clear():
                reward += 100 # Win bonus
            else:
                reward -= 10 # Timeout penalty

        self.last_reward = reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action
        reward = 0.0

        # Handle selector movement
        if movement > 0 and not space_held:
            if movement == 1 and self.selector_pos[1] > 0: self.selector_pos[1] -= 1
            elif movement == 2 and self.selector_pos[1] < self.GRID_HEIGHT - 1: self.selector_pos[1] += 1
            elif movement == 3 and self.selector_pos[0] > 0: self.selector_pos[0] -= 1
            elif movement == 4 and self.selector_pos[0] < self.GRID_WIDTH - 1: self.selector_pos[0] += 1
        
        # Handle swap action
        elif movement > 0 and space_held:
            x1, y1 = self.selector_pos
            x2, y2 = x1, y1
            if movement == 1 and y1 > 0: y2 -= 1
            elif movement == 2 and y1 < self.GRID_HEIGHT - 1: y2 += 1
            elif movement == 3 and x1 > 0: x2 -= 1
            elif movement == 4 and x1 < self.GRID_WIDTH - 1: x2 += 1
            
            if (x1, y1) != (x2, y2): # A valid swap direction was chosen
                # Perform the swap on the board data
                self.board[y1, x1], self.board[y2, x2] = self.board[y2, x2], self.board[y1, x1]
                
                # Check if this swap creates a match
                matches = self._find_all_matches()
                
                self.swap_info = {'pos1': (x1, y1), 'pos2': (x2, y2)}
                
                if not matches:
                    # Invalid swap, revert
                    self.board[y1, x1], self.board[y2, x2] = self.board[y2, x2], self.board[y1, x1]
                    self.game_state = "REVERT_ANIM"
                    self.animation_timer = self.SWAP_SPEED
                    reward -= 0.1
                    # sfx: invalid_swap
                else:
                    # Valid swap, start clearing
                    self.game_state = "SWAP_ANIM"
                    self.animation_timer = self.SWAP_SPEED
                    # sfx: swap_success

        return reward

    def _update_animations(self):
        self.animation_timer = max(0, self.animation_timer - self.TIME_PER_STEP)
        
        if self.animation_timer == 0:
            if self.game_state in ["SWAP_ANIM", "REVERT_ANIM"]:
                self._start_match_check()
            elif self.game_state == "CLEAR_ANIM":
                self._apply_gravity()
            elif self.game_state == "FALL_ANIM":
                self._finish_fall()
                self._start_match_check()

    def _start_match_check(self):
        matches = self._find_all_matches()
        if matches:
            reward = 0
            for match in matches:
                # Base reward
                reward += len(match)
                # Bonus rewards
                if len(match) == 4: reward += 5
                if len(match) >= 5: reward += 10
            
            self.score += reward
            
            self.clearing_gems = {pos for match in matches for pos in match}
            for x, y in self.clearing_gems:
                self._create_particles(x, y, self.board[y, x])
                self.board[y, x] = -1 # Mark for clearing
            
            self.game_state = "CLEAR_ANIM"
            self.animation_timer = self.CLEAR_SPEED
            # sfx: match_clear
        else:
            self.game_state = "IDLE"

    def _apply_gravity(self):
        self.clearing_gems.clear()
        self.falling_gems = []
        
        for x in range(self.GRID_WIDTH):
            empty_count = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[y, x] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    gem_type = self.board[y, x]
                    self.board[y + empty_count, x] = gem_type
                    self.board[y, x] = -1
                    self.falling_gems.append({'from': (x, y), 'to': (x, y + empty_count), 'type': gem_type})
        
        if self.falling_gems:
            self.game_state = "FALL_ANIM"
            self.animation_timer = self.FALL_SPEED
            # sfx: gems_fall
        else:
            self._refill_board()
            self._start_match_check()

    def _finish_fall(self):
        self.falling_gems.clear()
        self._refill_board()

    def _refill_board(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.board[y, x] == -1:
                    self.board[y, x] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _generate_board(self):
        self.board = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while self._find_all_matches():
             matches = self._find_all_matches()
             for match in matches:
                 for x, y in match:
                     self.board[y, x] = self.np_random.integers(0, self.NUM_GEM_TYPES)
    
    def _find_all_matches(self):
        all_matches = []
        matched_pos = set()
        
        # Horizontal
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.board[y, x] == self.board[y, x+1] == self.board[y, x+2] and self.board[y, x] != -1:
                    match = {(x, y), (x+1, y), (x+2, y)}
                    # Extend match
                    i = x + 3
                    while i < self.GRID_WIDTH and self.board[y, i] == self.board[y, x]:
                        match.add((i, y))
                        i += 1
                    
                    is_new_match = not any(pos in matched_pos for pos in match)
                    if is_new_match:
                        all_matches.append(list(match))
                        matched_pos.update(match)

        # Vertical
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.board[y, x] == self.board[y+1, x] == self.board[y+2, x] and self.board[y, x] != -1:
                    match = {(x, y), (x, y+1), (x, y+2)}
                    # Extend match
                    i = y + 3
                    while i < self.GRID_HEIGHT and self.board[i, x] == self.board[y, x]:
                        match.add((x, i))
                        i += 1
                        
                    is_new_match = not any(pos in matched_pos for pos in match)
                    if is_new_match:
                        all_matches.append(list(match))
                        matched_pos.update(match)
                        
        return all_matches

    def _is_board_clear(self):
        return bool(np.all(self.board == -1))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.BOARD_OFFSET_X + i * (self.GEM_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH // 2
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.BOARD_OFFSET_Y), (x, self.BOARD_OFFSET_Y + self.GRID_HEIGHT * (self.GEM_SIZE + self.GRID_LINE_WIDTH)), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_HEIGHT + 1):
            y = self.BOARD_OFFSET_Y + i * (self.GEM_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH // 2
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_OFFSET_X, y), (self.BOARD_OFFSET_X + self.GRID_WIDTH * (self.GEM_SIZE + self.GRID_LINE_WIDTH), y), self.GRID_LINE_WIDTH)

        # Draw gems
        visible_gems = set((x, y) for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH))
        
        # Handle animations
        anim_progress = 1.0
        if self.game_state in ["SWAP_ANIM", "REVERT_ANIM"] and self.SWAP_SPEED > 0:
            anim_progress = 1.0 - (self.animation_timer / self.SWAP_SPEED)
        
        if self.game_state in ["SWAP_ANIM", "REVERT_ANIM"]:
            p1 = self.swap_info['pos1']
            p2 = self.swap_info['pos2']
            type1, type2 = self.board[p2[1], p2[0]], self.board[p1[1], p1[0]]
            if self.game_state == "REVERT_ANIM":
                type1, type2 = self.board[p1[1], p1[0]], self.board[p2[1], p2[0]]

            self._render_animated_gem(p1, p2, anim_progress, type1)
            self._render_animated_gem(p2, p1, anim_progress, type2)
            visible_gems.discard(p1)
            visible_gems.discard(p2)

        elif self.game_state == "CLEAR_ANIM":
            clear_progress = self.animation_timer / self.CLEAR_SPEED if self.CLEAR_SPEED > 0 else 0.0
            for x, y in self.clearing_gems:
                self._render_gem(x, y, self.board[y,x], scale=clear_progress, alpha=int(255 * clear_progress))
                visible_gems.discard((x, y))

        elif self.game_state == "FALL_ANIM":
            fall_progress = 1.0 - (self.animation_timer / self.FALL_SPEED) if self.FALL_SPEED > 0 else 1.0
            for fall in self.falling_gems:
                self._render_animated_gem(fall['from'], fall['to'], fall_progress, fall['type'])
                visible_gems.discard(fall['from'])
                visible_gems.discard(fall['to'])
        
        for x, y in list(visible_gems):
            gem_type = self.board[y, x]
            if gem_type != -1:
                self._render_gem(x, y, gem_type)
        
        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), p['color'])

        # Draw selector
        if not self.game_over:
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            sel_alpha = 150 + int(pulse * 105)
            
            x, y = self.selector_pos
            rect = pygame.Rect(
                self.BOARD_OFFSET_X + x * (self.GEM_SIZE + self.GRID_LINE_WIDTH),
                self.BOARD_OFFSET_Y + y * (self.GEM_SIZE + self.GRID_LINE_WIDTH),
                self.GEM_SIZE, self.GEM_SIZE
            )
            # Use a surface for alpha blending
            sel_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(sel_surf, (*self.COLOR_SELECTOR, sel_alpha), sel_surf.get_rect(), 4, border_radius=8)
            self.screen.blit(sel_surf, rect.topleft)


    def _render_gem(self, x, y, gem_type, scale=1.0, alpha=255):
        if gem_type == -1: return
        center_x = self.BOARD_OFFSET_X + x * (self.GEM_SIZE + self.GRID_LINE_WIDTH) + self.GEM_SIZE // 2
        center_y = self.BOARD_OFFSET_Y + y * (self.GEM_SIZE + self.GRID_LINE_WIDTH) + self.GEM_SIZE // 2
        radius = int((self.GEM_SIZE // 2 - 2) * scale)
        if radius <= 0: return

        color = self.GEM_COLORS[gem_type]
        
        # Create a temporary surface for transparency
        temp_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.aacircle(temp_surface, radius, radius, radius, color)
        pygame.gfxdraw.filled_circle(temp_surface, radius, radius, radius, color)
        
        # Inner highlight
        highlight_color = tuple(min(255, c + 60) for c in color)
        pygame.gfxdraw.filled_circle(temp_surface, radius - radius // 4, radius - radius // 4, radius // 3, (*highlight_color, 100))
        
        temp_surface.set_alpha(alpha)
        self.screen.blit(temp_surface, (center_x - radius, center_y - radius))

    def _render_animated_gem(self, pos_from, pos_to, progress, gem_type):
        x1, y1 = pos_from
        x2, y2 = pos_to
        
        # Lerp grid coordinates
        render_x = x1 + (x2 - x1) * progress
        render_y = y1 + (y2 - y1) * progress
        
        # Convert to pixel coordinates
        center_x = self.BOARD_OFFSET_X + render_x * (self.GEM_SIZE + self.GRID_LINE_WIDTH) + self.GEM_SIZE // 2
        center_y = self.BOARD_OFFSET_Y + render_y * (self.GEM_SIZE + self.GRID_LINE_WIDTH) + self.GEM_SIZE // 2
        
        radius = self.GEM_SIZE // 2 - 2
        color = self.GEM_COLORS[gem_type]
        
        pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), radius, color)
        pygame.gfxdraw.filled_circle(self.screen, int(center_x), int(center_y), radius, color)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 20))

        # Time
        time_color = (255, 255, 255) if self.time_left > 10 else (255, 100, 100)
        time_text = self.font_small.render(f"Time: {self.time_left:.1f}", True, time_color)
        self.screen.blit(time_text, (self.screen_width - time_text.get_width() - 20, 20))
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self._is_board_clear() else "TIME'S UP!"
            text_surface = self.font_large.render(message, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(text_surface, text_rect)
    
    def _create_particles(self, grid_x, grid_y, gem_type):
        center_x = self.BOARD_OFFSET_X + grid_x * (self.GEM_SIZE + self.GRID_LINE_WIDTH) + self.GEM_SIZE // 2
        center_y = self.BOARD_OFFSET_Y + grid_y * (self.GEM_SIZE + self.GRID_LINE_WIDTH) + self.GEM_SIZE // 2
        color = self.GEM_COLORS[gem_type]
        
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': vel,
                'lifetime': self.np_random.uniform(0.3, 0.7),
                'size': self.np_random.uniform(2, 5),
                'color': color
            })
    
    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifetime'] -= self.TIME_PER_STEP
            if p['lifetime'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for internal testing and is not part of the standard API.
        # It's called here to ensure the environment is correctly implemented.
        print("Running implementation validation...")
        
        # Test reset
        obs, info = self.reset(seed=42)
        assert self.observation_space.contains(obs), "Reset observation is not in the observation space."
        assert isinstance(info, dict), "Reset info is not a dictionary."
        
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        assert self.observation_space.shape == (self.screen_height, self.screen_width, 3)
        assert self.observation_space.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert self.observation_space.contains(obs), "Step observation is not in the observation space."
        assert isinstance(reward, (int, float)), f"Reward is not a number, but {type(reward)}."
        assert isinstance(term, bool), f"Terminated is not a boolean, but {type(term)}."
        assert isinstance(trunc, bool), f"Truncated is not a boolean, but {type(trunc)}."
        assert isinstance(info, dict), "Step info is not a dictionary."
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    # The validation call is helpful for development.
    # We instantiate the env, then run the validation.
    env_for_validation = GameEnv()
    env_for_validation.validate_implementation()
    env_for_validation.close()
    
    # --- To play manually ---
    # This requires a window, so we re-init pygame for display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit() # Quit the dummy driver
    pygame.init() # Re-init with default driver
    
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Gem Swap")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    obs, info = env.reset(seed=123)
    done = False
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op
    action[1] = 0
    action[2] = 0

    while not done:
        # --- Manual Control ---
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for manual play

    env.close()