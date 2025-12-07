
# Generated: 2025-08-27T16:19:51.746695
# Source Brief: brief_01192.md
# Brief Index: 1192

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select up to two gems. "
        "A swap is attempted when two are selected. Press shift to clear selections."
    )

    game_description = (
        "Strategically match gems in a grid to collect 50 gems before you run out of 20 moves."
    )

    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 10
    GRID_HEIGHT = 8
    GEM_SIZE = 40
    WIN_SCORE = 50
    MAX_MOVES = 20
    
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTION = (255, 255, 0)
    COLOR_WHITE = (240, 240, 240)
    COLOR_SHADOW = (10, 10, 10)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width = 640
        self.screen_height = 400

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        self.grid_offset_x = (self.screen_width - self.GRID_WIDTH * self.GEM_SIZE) // 2
        self.grid_offset_y = (self.screen_height - self.GRID_HEIGHT * self.GEM_SIZE) // 2
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.grid = None
        self.cursor = None
        self.selections = []
        self.particles = []
        self.last_action = None
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.win_message = ""
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win_message = ""
        
        self.cursor = (self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2)
        self.selections = []
        self.particles = []
        self.last_action = self.action_space.sample() * 0

        self._create_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        prev_space = self.last_action[1] == 1
        prev_shift = self.last_action[2] == 1
        self.last_action = action

        space_just_pressed = space_held and not prev_space
        shift_just_pressed = shift_held and not prev_shift

        # 1. Handle cursor movement
        self._move_cursor(movement)

        # 2. Handle selections and swaps
        if shift_just_pressed:
            self.selections.clear()
            # Sound: Deselect sound

        if space_just_pressed:
            if self.cursor in self.selections:
                self.selections.remove(self.cursor)
                # Sound: Deselect sound
            elif len(self.selections) < 2:
                self.selections.append(self.cursor)
                # Sound: Select sound

            if len(self.selections) == 2:
                reward = self._attempt_swap(self.selections[0], self.selections[1])
                self.selections.clear()

        # 3. Update game state
        self.steps += 1
        self.score += reward
        
        # 4. Check for termination
        terminated = self.score >= self.WIN_SCORE or self.moves_left <= 0
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 50  # Goal-oriented reward
                self.win_message = "YOU WIN!"
                # Sound: Win jingle
            else:
                self.win_message = "GAME OVER"
                # Sound: Lose sound
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, movement):
        r, c = self.cursor
        if movement == 1: r -= 1  # Up
        if movement == 2: r += 1  # Down
        if movement == 3: c -= 1  # Left
        if movement == 4: c += 1  # Right
        self.cursor = (max(0, min(r, self.GRID_HEIGHT - 1)), max(0, min(c, self.GRID_WIDTH - 1)))

    def _attempt_swap(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2

        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return 0  # Not adjacent

        self.moves_left -= 1
        
        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        # Sound: Swap sound

        total_gems_cleared = 0
        chain_reaction = True
        while chain_reaction:
            matches = self._find_matches()
            if not matches:
                # If the initial swap caused no match, swap back
                if total_gems_cleared == 0:
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                chain_reaction = False
            else:
                # Sound: Match sound
                total_gems_cleared += len(matches)
                self._create_particles(matches)
                self._remove_gems(matches)
                self._drop_gems()
                self._refill_grid()
        
        # After a chain, check if any moves are possible. If not, reshuffle.
        if not self._check_for_possible_moves():
            self._create_board(preserve_state=True)
            # Sound: Reshuffle sound

        return total_gems_cleared

    def _create_board(self, preserve_state=False):
        if not preserve_state:
            self.grid = self.rng.integers(0, len(self.GEM_COLORS), size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        
        while True:
            # Remove initial matches
            while True:
                matches = self._find_matches()
                if not matches:
                    break
                self._remove_gems(matches)
                self._drop_gems()
                self._refill_grid()
            
            # Check for possible moves
            if self._check_for_possible_moves():
                break
            else: # Reshuffle
                self.grid = self.rng.integers(0, len(self.GEM_COLORS), size=(self.GRID_HEIGHT, self.GRID_WIDTH))

    def _find_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        # Vertical
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _remove_gems(self, matches):
        for r, c in matches:
            self.grid[r, c] = -1

    def _drop_gems(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
    
    def _refill_grid(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.rng.integers(0, len(self.GEM_COLORS))

    def _check_for_possible_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Check swap right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if len(self._find_matches()) > 0:
                        self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                # Check swap down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if len(self._find_matches()) > 0:
                        self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._draw_gems()
        self._update_and_draw_particles()
        self._draw_cursor_and_selections()
        self._draw_ui()
        if self.game_over:
            self._draw_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor,
        }

    def _draw_grid(self):
        grid_rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y, self.GRID_WIDTH * self.GEM_SIZE, self.GRID_HEIGHT * self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)
        for r in range(1, self.GRID_HEIGHT):
            y = self.grid_offset_y + r * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (self.grid_offset_x, y), (self.grid_offset_x + self.GRID_WIDTH * self.GEM_SIZE, y))
        for c in range(1, self.GRID_WIDTH):
            x = self.grid_offset_x + c * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (x, self.grid_offset_y), (x, self.grid_offset_y + self.GRID_HEIGHT * self.GEM_SIZE))

    def _draw_gems(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type == -1: continue
                
                color = self.GEM_COLORS[gem_type]
                rect = pygame.Rect(
                    self.grid_offset_x + c * self.GEM_SIZE + 2,
                    self.grid_offset_y + r * self.GEM_SIZE + 2,
                    self.GEM_SIZE - 4, self.GEM_SIZE - 4
                )
                
                # Beveled effect
                light_color = tuple(min(255, x + 60) for x in color)
                dark_color = tuple(max(0, x - 60) for x in color)
                
                pygame.draw.rect(self.screen, dark_color, rect, border_radius=8)
                inner_rect = rect.inflate(-4, -4)
                pygame.draw.rect(self.screen, color, inner_rect, border_radius=8)
                
                # Highlight
                pygame.gfxdraw.arc(self.screen, inner_rect.centerx, inner_rect.centery, 
                                   int(self.GEM_SIZE * 0.25), 110, 160, (255, 255, 255, 100))
                pygame.gfxdraw.arc(self.screen, inner_rect.centerx, inner_rect.centery, 
                                   int(self.GEM_SIZE * 0.25)-1, 110, 160, (255, 255, 255, 100))

    def _draw_cursor_and_selections(self):
        # Draw selections
        for r, c in self.selections:
            rect = pygame.Rect(
                self.grid_offset_x + c * self.GEM_SIZE,
                self.grid_offset_y + r * self.GEM_SIZE,
                self.GEM_SIZE, self.GEM_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTION, rect, 4, border_radius=8)
        
        # Draw cursor
        r, c = self.cursor
        pulse = abs(math.sin(self.steps * 0.2)) * 4
        rect = pygame.Rect(
            self.grid_offset_x + c * self.GEM_SIZE - pulse / 2,
            self.grid_offset_y + r * self.GEM_SIZE - pulse / 2,
            self.GEM_SIZE + pulse, self.GEM_SIZE + pulse
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=8)

    def _draw_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf = font.render(text, True, self.COLOR_SHADOW)
            self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _draw_ui(self):
        self._draw_text(f"SCORE: {self.score}/{self.WIN_SCORE}", self.font_small, self.COLOR_WHITE, (20, 20))
        self._draw_text(f"MOVES: {self.moves_left}", self.font_small, self.COLOR_WHITE, (self.screen_width - 150, 20))

    def _draw_game_over(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text_surf = self.font_large.render(self.win_message, True, self.COLOR_WHITE)
        text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
        self.screen.blit(text_surf, text_rect.move(2,2)) # Shadow
        
        text_surf = self.font_large.render(self.win_message, True, self.COLOR_SELECTION)
        text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, positions):
        for r, c in positions:
            gem_type = self.grid[r,c]
            if gem_type == -1: continue # Gem might have been part of two matches
            color = self.GEM_COLORS[gem_type]
            px = self.grid_offset_x + c * self.GEM_SIZE + self.GEM_SIZE / 2
            py = self.grid_offset_y + r * self.GEM_SIZE + self.GEM_SIZE / 2
            for _ in range(10):
                angle = self.rng.random() * 2 * math.pi
                speed = self.rng.random() * 2 + 1
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = self.rng.integers(20, 40)
                self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            radius = int(self.GEM_SIZE * 0.1 * (p['life'] / p['max_life']))
            if radius > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def validate_implementation(self):
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Gem Puzzle Environment")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # no-op, release space, release shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        # Since auto_advance is False, we only step when there's an action
        # For human play, we want to step on every key press/release
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward > 0:
            print(f"Reward: {reward}, Score: {info['score']}, Moves left: {info['moves_left']}")
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(1000)

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit human play speed

    pygame.quit()