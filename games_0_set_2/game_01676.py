# Generated: 2025-08-27T17:55:44.037210
# Source Brief: brief_01676.md
# Brief Index: 1676

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to select a gem. "
        "Move to an adjacent gem and press space again to swap. Press shift to deselect."
    )

    game_description = (
        "Swap adjacent gems to create matches of 3 or more. Plan your moves to create "
        "cascading combos and reach the target score before you run out of moves."
    )

    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    NUM_GEM_TYPES = 6
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_AREA_HEIGHT = SCREEN_HEIGHT
    CELL_SIZE = GRID_AREA_HEIGHT // GRID_HEIGHT
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_OFFSET_Y = (GRID_AREA_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2
    GEM_RADIUS = int(CELL_SIZE * 0.4)
    
    # Animation timings (in steps/frames)
    SWAP_DURATION = 8
    EXPLOSION_DURATION = 10
    FALL_DURATION = 6
    SHAKE_DURATION = 10

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_LINES = (40, 50, 60)
    COLOR_SELECTOR = (255, 255, 0)
    COLOR_SELECTED = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COMBO_COLOR = (255, 200, 0)
    GEM_COLORS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (50, 150, 255),  # Blue
        (255, 255, 50),  # Yellow
        (200, 50, 255),  # Purple
        (255, 150, 50),  # Orange
    ]

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
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 18)

        self.grid = None
        self.selector_pos = None
        self.selected_gem = None
        self.game_phase = "IDLE"
        self.animation_timer = 0
        self.swapping_gems = None
        self.falling_gems = None
        self.particles = []
        self.combo_multiplier = 1
        self.combo_display_timer = 0
        self.last_action_time = 0
        self.last_move_action = 0

        # This will call reset, which needs a random generator.
        # So we initialize one here, which will be re-seeded in reset.
        self.np_random = np.random.default_rng()
        self.reset()
        # self.validate_implementation() # Commented out for submission as it prints to stdout

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = 25
        self.game_over = False
        self.terminal_reward_given = False

        self.selector_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_gem = None
        self.game_phase = "IDLE"
        self.animation_timer = 0
        self.swapping_gems = None
        self.falling_gems = []
        self.particles = []
        self.combo_multiplier = 1
        
        self._generate_valid_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        step_reward = 0

        # --- Update ongoing animations and game logic ---
        self._update_game_phase()
        self._update_particles()
        if self.combo_display_timer > 0:
            self.combo_display_timer -= 1

        # --- Process player input only when IDLE ---
        if self.game_phase == "IDLE" and not self.game_over:
            # Handle deselection
            if shift_press:
                self.selected_gem = None
            
            # Handle cursor movement
            if movement != self.last_move_action or pygame.time.get_ticks() - self.last_action_time > 200:
                if movement == 1: self.selector_pos[1] = max(0, self.selector_pos[1] - 1)
                elif movement == 2: self.selector_pos[1] = min(self.GRID_HEIGHT - 1, self.selector_pos[1] + 1)
                elif movement == 3: self.selector_pos[0] = max(0, self.selector_pos[0] - 1)
                elif movement == 4: self.selector_pos[0] = min(self.GRID_WIDTH - 1, self.selector_pos[0] + 1)
                if movement != 0:
                    self.last_action_time = pygame.time.get_ticks()
            self.last_move_action = movement

            # Handle selection/swap
            if space_press:
                if not self.selected_gem:
                    # Select a gem
                    self.selected_gem = list(self.selector_pos)
                    # sfx: select_gem
                else:
                    # Attempt a swap
                    p1 = self.selected_gem
                    p2 = self.selector_pos
                    if self._are_adjacent(p1, p2):
                        self.moves_left -= 1
                        self._start_swap(p1, p2)
                        step_reward = -0.2 # Penalty for any swap, overwritten on match
                    else:
                        # Invalid swap (not adjacent), treat as new selection
                        self.selected_gem = list(self.selector_pos)
                        # sfx: invalid_move

        # --- Update score and check for termination ---
        current_phase_reward = self._get_phase_reward()
        if current_phase_reward != 0:
            step_reward = current_phase_reward

        self.score += step_reward
        terminated = self._check_termination()
        if terminated and not self.terminal_reward_given:
            if self.score >= 1000:
                step_reward += 100
                self.score += 100
            elif self.moves_left <= 0:
                step_reward -= 10
                self.score -= 10
            self.terminal_reward_given = True
            
        return (
            self._get_observation(),
            float(step_reward),
            terminated,
            False,
            self._get_info()
        )

    # --- Game Logic Helpers ---

    def _update_game_phase(self):
        if self.game_phase == "IDLE": return
        self.animation_timer -= 1
        if self.animation_timer > 0: return

        if self.game_phase == "SWAPPING":
            self._finish_swap()
            self.game_phase = "CHECK_MATCHES"
        
        elif self.game_phase == "SWAP_BACK":
            self._finish_swap() # Swaps back to original state
            self.game_phase = "IDLE"

        elif self.game_phase == "CHECK_MATCHES":
            matches = self._find_all_matches()
            if matches:
                if self.combo_multiplier > 1:
                    self.combo_display_timer = 30 # Show combo text for 1 second
                self._process_matches(matches)
                self.game_phase = "EXPLODING"
                self.animation_timer = self.EXPLOSION_DURATION
            elif self.swapping_gems: # First check after a swap failed
                self._start_swap_back() # Swap back if no match
            else: # No matches from a cascade
                self.combo_multiplier = 1
                if not self._find_possible_moves():
                    self._generate_valid_board() # Reshuffle if no moves left
                self.game_phase = "IDLE"

        elif self.game_phase == "EXPLODING":
            self._apply_gravity()
            self.game_phase = "FALLING"
            self.animation_timer = self.FALL_DURATION

        elif self.game_phase == "FALLING":
            self.falling_gems = []
            self._refill_board()
            self.combo_multiplier += 1
            self.game_phase = "CHECK_MATCHES"

    def _get_phase_reward(self):
        if self.game_phase == "CHECK_MATCHES":
            matches = self._find_all_matches()
            if matches:
                reward = len(matches)
                if self.combo_multiplier > 1:
                    reward += 5 # Combo bonus
                return reward
        return 0

    def _generate_valid_board(self):
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while self._find_all_matches() or not self._find_possible_moves():
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2] != -1:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c] != -1:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _find_possible_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Test swap right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                    if self._find_all_matches():
                        self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                        return True
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                # Test swap down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                    if self._find_all_matches():
                        self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                        return True
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
        return False

    def _process_matches(self, matches):
        # sfx: match_found
        for r, c in matches:
            self._create_particles(r, c, self.grid[r, c])
            self.grid[r, c] = -1 # Mark as empty
    
    def _apply_gravity(self):
        self.falling_gems = []
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                        self.falling_gems.append({'from': (r, c), 'to': (empty_row, c), 'type': self.grid[empty_row, c]})
                    empty_row -= 1
    
    def _refill_board(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)
                    # Add to falling gems to animate them coming from off-screen
                    self.falling_gems.append({'from': (-1, c), 'to': (r, c), 'type': self.grid[r,c]})

    def _start_swap(self, p1, p2):
        self.game_phase = "SWAPPING"
        self.animation_timer = self.SWAP_DURATION
        self.swapping_gems = (list(p1), list(p2))
        self.selected_gem = None
        # sfx: gem_swap

    def _finish_swap(self):
        p1, p2 = self.swapping_gems
        c1, r1 = p1
        c2, r2 = p2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
    def _start_swap_back(self):
        # sfx: invalid_swap
        self.game_phase = "SWAP_BACK"
        self.animation_timer = self.SWAP_DURATION
        # No change to swapping_gems, it will just swap them back
        
    def _are_adjacent(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= 1000 or self.moves_left <= 0:
            self.game_over = True
            self.game_phase = "GAME_OVER"
            return True
        return False

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._draw_gems()
        self._draw_particles()
        self._draw_selector()
        self._draw_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left
        }

    def _draw_grid(self):
        for r in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE))

    def _draw_gems(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type == -1: continue

                pos_x = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                pos_y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                radius = self.GEM_RADIUS

                # Handle animations
                if self.game_phase in ["SWAPPING", "SWAP_BACK"] and self.swapping_gems:
                    p1, p2 = self.swapping_gems
                    progress = 1 - self.animation_timer / self.SWAP_DURATION
                    if [c, r] == p1:
                        pos_x = int(pygame.math.lerp(pos_x, self.GRID_OFFSET_X + p2[0] * self.CELL_SIZE + self.CELL_SIZE // 2, progress))
                        pos_y = int(pygame.math.lerp(pos_y, self.GRID_OFFSET_Y + p2[1] * self.CELL_SIZE + self.CELL_SIZE // 2, progress))
                    elif [c, r] == p2:
                        pos_x = int(pygame.math.lerp(pos_x, self.GRID_OFFSET_X + p1[0] * self.CELL_SIZE + self.CELL_SIZE // 2, progress))
                        pos_y = int(pygame.math.lerp(pos_y, self.GRID_OFFSET_Y + p1[1] * self.CELL_SIZE + self.CELL_SIZE // 2, progress))
                
                if self.game_phase == "FALLING":
                    for fall in self.falling_gems:
                        if fall['to'] == (r, c):
                            progress = 1 - self.animation_timer / self.FALL_DURATION
                            start_y = self.GRID_OFFSET_Y + fall['from'][0] * self.CELL_SIZE + self.CELL_SIZE // 2
                            end_y = self.GRID_OFFSET_Y + fall['to'][0] * self.CELL_SIZE + self.CELL_SIZE // 2
                            pos_y = int(pygame.math.lerp(start_y, end_y, progress))
                            break
                
                if self.game_phase == "EXPLODING":
                    is_exploding = False
                    # Logic to shrink exploding gems can be added here
                
                self._draw_single_gem(pos_x, pos_y, gem_type, radius)
    
    def _draw_single_gem(self, x, y, gem_type, radius):
        color = self.GEM_COLORS[gem_type]
        
        # Body
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, tuple(min(255, c+30) for c in color))
        
        # Highlight
        highlight_color = (255, 255, 255, 100)
        highlight_radius = int(radius * 0.5)
        highlight_pos = (x - int(radius * 0.3), y - int(radius * 0.3))
        
        # Simple highlight using a filled circle with alpha
        s = pygame.Surface((highlight_radius*2, highlight_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, highlight_color, (highlight_radius, highlight_radius), highlight_radius)
        self.screen.blit(s, (highlight_pos[0] - highlight_radius, highlight_pos[1] - highlight_radius))

    def _draw_selector(self):
        # Draw selected gem highlight
        if self.selected_gem:
            c, r = self.selected_gem
            rect = pygame.Rect(self.GRID_OFFSET_X + c * self.CELL_SIZE, self.GRID_OFFSET_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, 3, border_radius=4)
        
        # Draw cursor
        c, r = self.selector_pos
        rect = pygame.Rect(self.GRID_OFFSET_X + c * self.CELL_SIZE, self.GRID_OFFSET_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        # Pulsing effect for selector
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
        color = (
            int(pygame.math.lerp(180, 255, pulse)),
            int(pygame.math.lerp(180, 255, pulse)),
            0
        )
        pygame.draw.rect(self.screen, color, rect, 3, border_radius=4)

    def _draw_ui(self):
        # Score
        self._render_text(f"Score: {self.score}", (20, 10), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        # Moves
        self._render_text(f"Moves: {self.moves_left}", (self.SCREEN_WIDTH - 150, 10), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Combo
        if self.combo_multiplier > 1 and self.combo_display_timer > 0:
            alpha = min(255, int(255 * (self.combo_display_timer / 20)))
            text = f"Combo x{self.combo_multiplier}!"
            size = self.font_large.size(text)
            color = (*self.COMBO_COLOR, alpha)
            self._render_text(text, (self.SCREEN_WIDTH//2 - size[0]//2, 100), self.font_large, color, use_alpha=True)

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "You Win!" if self.score >= 1000 else "Game Over"
            color = (100, 255, 100) if self.score >= 1000 else (255, 100, 100)
            
            size = self.font_large.size(msg)
            self._render_text(msg, (self.SCREEN_WIDTH//2 - size[0]//2, self.SCREEN_HEIGHT//2 - 50), self.font_large, color)
            
            final_score_msg = f"Final Score: {self.score}"
            size = self.font_medium.size(final_score_msg)
            self._render_text(final_score_msg, (self.SCREEN_WIDTH//2 - size[0]//2, self.SCREEN_HEIGHT//2 + 10), self.font_medium, self.COLOR_TEXT)

    def _render_text(self, text, pos, font, color, shadow_color=None, use_alpha=False):
        if use_alpha:
            text_surf = font.render(text, True, color)
            text_surf.set_alpha(color[3])
        else:
            if shadow_color:
                shadow_surf = font.render(text, True, shadow_color)
                self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    # --- Particles ---
    
    def _create_particles(self, r, c, gem_type):
        pos_x = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        pos_y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.GEM_COLORS[gem_type]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particles.append({
                'x': pos_x, 'y': pos_y, 'vx': vx, 'vy': vy, 
                'life': random.randint(15, 30), 'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            size = int(p['life'] / 10) + 1
            
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (size, size), size)
            self.screen.blit(s, (int(p['x']) - size, int(p['y']) - size))

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play ---
    # To control, run this script and have the Pygame window focused.
    # This is a simple manual control loop for testing.
    
    pygame.display.set_caption("Gem Swap")
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = [0, 0, 0] # no-op, no-space, no-shift
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # None
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        # Actions
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

        if terminated or truncated:
            print("Game Over!")
            done = True
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we control the speed here
        env.clock.tick(30)

    pygame.quit()