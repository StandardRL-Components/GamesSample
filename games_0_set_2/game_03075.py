
# Generated: 2025-08-28T06:54:02.186476
# Source Brief: brief_03075.md
# Brief Index: 3075

        
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
        "then move to an adjacent gem and press Space again to swap. "
        "Hold Shift and press Space to reshuffle the board (costs a move)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Collect 50 gems within 20 moves to win. Create combos for bonus points!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Game parameters
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_GEM_TYPES = 6
    MOVES_LIMIT = 20
    WIN_SCORE = 50
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (60, 70, 80)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED = (255, 255, 0)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 255, 80),   # Yellow
        (255, 80, 255),   # Magenta
        (80, 255, 255),   # Cyan
    ]

    # Visuals
    TILE_WIDTH_ISO = 64
    TILE_HEIGHT_ISO = 32
    GEM_SIZE_MOD = 0.7

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.screen_width, self.screen_height = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 24)

        # Isometric grid origin for centering
        self.origin_x = self.screen_width // 2
        self.origin_y = self.screen_height // 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_ISO) // 4 + 20

        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.message = ""
        self.message_timer = 0
        
        self.reset()

        self.validate_implementation()

    def _to_iso(self, r, c):
        """Converts grid coordinates (row, col) to isometric screen coordinates (x, y)."""
        iso_x = self.origin_x + (c - r) * self.TILE_WIDTH_ISO / 2
        iso_y = self.origin_y + (c + r) * self.TILE_HEIGHT_ISO / 2
        return int(iso_x), int(iso_y)

    def _generate_grid(self):
        """Generates a new grid, ensuring no initial matches and at least one possible move."""
        while True:
            self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            # Remove pre-existing matches
            while self._find_all_matches():
                matches = self._find_all_matches()
                for r, c in matches:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
            # Ensure at least one move is possible
            if self._find_possible_swaps():
                break

    def _find_all_matches(self):
        """Finds all horizontal and vertical matches of 3 or more."""
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == 0: continue
                # Horizontal check
                if c < self.GRID_WIDTH - 2 and self.grid[r, c] == self.grid[r, c + 1] == self.grid[r, c + 2]:
                    matches.update([(r, c), (r, c + 1), (r, c + 2)])
                # Vertical check
                if r < self.GRID_HEIGHT - 2 and self.grid[r, c] == self.grid[r + 1, c] == self.grid[r + 2, c]:
                    matches.update([(r, c), (r + 1, c), (r + 2, c)])
        return list(matches)

    def _find_possible_swaps(self):
        """Finds all swaps that would result in a match."""
        possible_swaps = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Test swap with right neighbor
                if c < self.GRID_WIDTH - 1:
                    self.grid[r, c], self.grid[r, c + 1] = self.grid[r, c + 1], self.grid[r, c]
                    if self._find_all_matches():
                        possible_swaps.append(((r, c), (r, c + 1)))
                    self.grid[r, c], self.grid[r, c + 1] = self.grid[r, c + 1], self.grid[r, c] # Swap back
                # Test swap with down neighbor
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r, c], self.grid[r + 1, c] = self.grid[r + 1, c], self.grid[r, c]
                    if self._find_all_matches():
                        possible_swaps.append(((r, c), (r + 1, c)))
                    self.grid[r, c], self.grid[r + 1, c] = self.grid[r + 1, c], self.grid[r, c] # Swap back
        return possible_swaps

    def _apply_gravity_and_refill(self):
        """Makes gems fall into empty spaces and refills the top of the grid."""
        for c in range(self.GRID_WIDTH):
            empty_spaces = 0
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] == 0:
                    empty_spaces += 1
                elif empty_spaces > 0:
                    self.grid[r + empty_spaces, c] = self.grid[r, c]
                    self.grid[r, c] = 0
            # Refill
            for r in range(empty_spaces):
                self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

    def _create_particles(self, pos, color, count):
        """Creates a burst of particles for visual effect."""
        # sound: gem_match.wav
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_grid()
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_gem_pos = None
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MOVES_LIMIT
        self.game_over = False
        
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.message = ""
        self.message_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_space_held # Use space as activator for shift

        self.steps += 1
        reward = 0
        terminated = False
        
        # --- Handle Actions ---
        # 1. Cursor Movement
        if movement == 1 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1 # Up
        if movement == 2 and self.cursor_pos[0] < self.GRID_HEIGHT - 1: self.cursor_pos[0] += 1 # Down
        if movement == 3 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1 # Left
        if movement == 4 and self.cursor_pos[1] < self.GRID_WIDTH - 1: self.cursor_pos[1] += 1 # Right

        # 2. Reshuffle
        if shift_pressed:
            if self.moves_left > 0:
                self.moves_left -= 1
                reward -= 5  # Penalty for reshuffling
                self._generate_grid()
                self.selected_gem_pos = None
                self.message = "Board Reshuffled!"
                self.message_timer = 60
                # sound: reshuffle.wav

        # 3. Select / Swap
        elif space_pressed:
            # sound: select.wav
            if self.selected_gem_pos is None:
                self.selected_gem_pos = list(self.cursor_pos)
            else:
                r1, c1 = self.selected_gem_pos
                r2, c2 = self.cursor_pos
                
                # Check for adjacency
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    self.moves_left -= 1
                    
                    # Perform swap
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                    
                    total_gems_matched = 0
                    combo_multiplier = 1
                    is_stable = False
                    
                    while not is_stable:
                        matches = self._find_all_matches()
                        if matches:
                            num_matched = len(matches)
                            total_gems_matched += num_matched
                            
                            # Calculate reward for this wave of matches
                            reward += num_matched * combo_multiplier
                            if num_matched >= 4: reward += 5  # 4-gem bonus
                            if num_matched >= 5: reward += 5  # 5-gem bonus (total +10)
                            
                            for r, c in matches:
                                gem_type = self.grid[r, c]
                                if gem_type > 0:
                                    iso_pos = self._to_iso(r, c)
                                    color = self.GEM_COLORS[gem_type - 1]
                                    self._create_particles(iso_pos, color, 10)
                                    self.grid[r, c] = 0 # Mark as empty
                            
                            self._apply_gravity_and_refill()
                            combo_multiplier += 1.5 # Increase combo multiplier for next wave
                        else:
                            is_stable = True
                    
                    if total_gems_matched == 0:
                        # Invalid swap, swap back
                        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                        self.moves_left += 1 # Refund move
                        reward = -0.1 # Small penalty for invalid move
                        # sound: invalid_swap.wav
                    else:
                        self.score += total_gems_matched

                    self.selected_gem_pos = None
                else:
                    # Not adjacent, re-select at new cursor position
                    self.selected_gem_pos = list(self.cursor_pos)

        # --- Post-Action Logic ---
        # Anti-softlock check
        if not self._find_possible_swaps() and not self.game_over:
            self._generate_grid()
            self.selected_gem_pos = None
            self.message = "No moves left! Reshuffling."
            self.message_timer = 60
            # sound: reshuffle.wav

        # Check termination conditions
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            terminated = True
            reward += 50 # Win bonus
            self.message = "You Win!"
            self.message_timer = 180
            # sound: win.wav
        elif self.moves_left <= 0:
            self.game_over = True
            terminated = True
            reward = -10 # Loss penalty
            self.message = "Game Over!"
            self.message_timer = 180
            # sound: lose.wav
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # 1. Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            start = self._to_iso(r, 0)
            end = self._to_iso(r, self.GRID_WIDTH)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for c in range(self.GRID_WIDTH + 1):
            start = self._to_iso(0, c)
            end = self._to_iso(self.GRID_HEIGHT, c)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # 2. Draw gems and highlights
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    iso_pos = self._to_iso(r, c)
                    color = self.GEM_COLORS[gem_type - 1]
                    self._draw_iso_gem(iso_pos, color)
        
        # 3. Draw cursor and selection
        cursor_iso = self._to_iso(self.cursor_pos[0], self.cursor_pos[1])
        self._draw_iso_highlight(cursor_iso, self.COLOR_CURSOR, 3)

        if self.selected_gem_pos:
            selected_iso = self._to_iso(self.selected_gem_pos[0], self.selected_gem_pos[1])
            self._draw_iso_highlight(selected_iso, self.COLOR_SELECTED, 4)

        # 4. Update and draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                # Fade out effect
                alpha = max(0, min(255, int(p['life'] * 15)))
                size = max(1, int(p['life'] / 6))
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, p['color'] + (alpha,), (size, size), size)
                self.screen.blit(s, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

    def _draw_iso_gem(self, pos, color):
        """Draws a single polished isometric gem."""
        w, h = self.TILE_WIDTH_ISO * self.GEM_SIZE_MOD, self.TILE_HEIGHT_ISO * self.GEM_SIZE_MOD
        points = [
            (pos[0], pos[1] - h / 2),
            (pos[0] + w / 2, pos[1]),
            (pos[0], pos[1] + h / 2),
            (pos[0] - w / 2, pos[1])
        ]
        
        # Darker base for depth
        dark_color = tuple(max(0, val - 60) for val in color)
        pygame.gfxdraw.filled_polygon(self.screen, points, dark_color)
        
        # Main color fill
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # Top highlight
        highlight_color = tuple(min(255, val + 80) for val in color)
        highlight_points = [
            (pos[0], pos[1] - h / 2),
            (pos[0] + w / 4, pos[1] - h/4),
            (pos[0], pos[1]),
            (pos[0] - w / 4, pos[1] - h/4)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, highlight_points, highlight_color)
        
        # Outline for crispness
        pygame.gfxdraw.aapolygon(self.screen, points, (0,0,0,100))

    def _draw_iso_highlight(self, pos, color, thickness):
        """Draws a highlight outline for cursor/selection."""
        w, h = self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO
        points = [
            (pos[0], pos[1] - h / 2),
            (pos[0] + w / 2, pos[1]),
            (pos[0], pos[1] + h / 2),
            (pos[0] - w / 2, pos[1])
        ]
        pygame.draw.lines(self.screen, color, True, points, thickness)

    def _render_ui(self):
        # Semi-transparent background for UI
        ui_panel = pygame.Surface((self.screen_width, 40), pygame.SRCALPHA)
        ui_panel.fill((0, 0, 0, 128))
        self.screen.blit(ui_panel, (0, 0))

        # Score display
        score_text = f"Gems: {self.score} / {self.WIN_SCORE}"
        self._draw_text(score_text, (10, 10), self.COLOR_TEXT, self.font_ui)
        
        # Moves display
        moves_text = f"Moves: {self.moves_left}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        self._draw_text(moves_text, (self.screen_width - moves_surf.get_width() - 10, 10), self.COLOR_TEXT, self.font_ui)

        # Message display
        if self.message and self.message_timer > 0:
            self.message_timer -= 1
            alpha = min(255, self.message_timer * 5)
            font_color = self.COLOR_TEXT[:3] + (alpha,)
            shadow_color = self.COLOR_TEXT_SHADOW[:3] + (alpha,)
            
            msg_surf = self.font_msg.render(self.message, True, font_color)
            shadow_surf = self.font_msg.render(self.message, True, shadow_color)
            
            x = self.screen_width // 2 - msg_surf.get_width() // 2
            y = self.screen_height - 50
            
            self.screen.blit(shadow_surf, (x + 1, y + 1))
            self.screen.blit(msg_surf, (x, y))

    def _draw_text(self, text, pos, color, font):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (pos[0] + 1, pos[1] + 1))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
            "selected_gem": self.selected_gem_pos,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Gem Swap")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()
    running = True

    # Game loop
    while running:
        movement = 0 # no-op
        space_held = False
        shift_held = False

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Key handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        # Since auto_advance is False, we only step on an action
        # For human play, we need to decide when to step. Let's step on any key press.
        # A simple way is to check if the action is not a total no-op.
        # However, the environment handles rising edges, so we can step every frame.
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Moves Left: {info['moves_left']}")
            pygame.time.wait(3000) # Wait 3 seconds before resetting
            obs, info = env.reset()

        clock.tick(30) # Limit to 30 FPS for human play

    env.close()