
# Generated: 2025-08-28T00:22:51.078730
# Source Brief: brief_03774.md
# Brief Index: 3774

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a gem, then move to an adjacent "
        "gem and press space again to swap. Press shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of 3 or more of the same color. "
        "Clear all 20 gems within 15 moves to win!"
    )

    # Frames only advance when an action is received.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 8
        self.NUM_GEM_TYPES = 5
        self.GEM_SIZE = 40
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.GEM_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.GEM_SIZE) // 2
        self.TOTAL_GEMS_TO_COLLECT = 20
        self.MAX_MOVES = 15
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 65)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_WARN = (255, 100, 100)
        self.COLOR_TEXT_WIN = (100, 255, 150)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.GEM_HIGHLIGHTS = [pygame.Color(c).lerp((255, 255, 255), 0.5) for c in self.GEM_COLORS]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # --- Game State Initialization ---
        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.moves_left = None
        self.gems_collected_this_game = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win_state = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.selection_pulse = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.gems_collected_this_game = 0
        self.game_over = False
        self.win_state = None
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_gem_pos = None
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        self.selection_pulse = (self.selection_pulse + 5) % 360

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        self._move_cursor(movement)

        if shift_press and self.selected_gem_pos:
            self.selected_gem_pos = None
            # sfx: deselect sound

        if space_press:
            if not self.selected_gem_pos:
                self.selected_gem_pos = list(self.cursor_pos)
                # sfx: select sound
            else:
                if self._are_adjacent(self.selected_gem_pos, self.cursor_pos):
                    reward += self._attempt_swap(self.selected_gem_pos, self.cursor_pos)
                    self.selected_gem_pos = None
                else:
                    self.selected_gem_pos = list(self.cursor_pos) # Reselect at new cursor
                    # sfx: error/reselect sound

        self._update_particles()
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        terminated = self._check_termination()
        if terminated:
            if self.win_state:
                reward += 100
            else:
                reward -= 100
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "moves_left": self.moves_left,
            "gems_collected": self.gems_collected_this_game,
        }
        
    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            while self._find_all_matches():
                matches = self._find_all_matches()
                for r, c in matches:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
            if self._find_possible_moves():
                break

    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Check swap right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r, c], self.grid[r, c + 1] = self.grid[r, c + 1], self.grid[r, c]
                    if self._find_all_matches():
                        moves.append(((r, c), (r, c + 1)))
                    self.grid[r, c], self.grid[r, c + 1] = self.grid[r, c + 1], self.grid[r, c] # Swap back
                # Check swap down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r, c], self.grid[r + 1, c] = self.grid[r + 1, c], self.grid[r, c]
                    if self._find_all_matches():
                        moves.append(((r, c), (r + 1, c)))
                    self.grid[r, c], self.grid[r + 1, c] = self.grid[r + 1, c], self.grid[r, c] # Swap back
        return moves

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

    def _are_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _attempt_swap(self, pos1, pos2):
        self.moves_left -= 1
        # sfx: swap sound
        r1, c1 = pos1
        r2, c2 = pos2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

        cascade_reward = self._handle_cascades()

        if cascade_reward == 0:
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1] # Swap back
            self.moves_left += 1
            # sfx: invalid swap sound
            return -0.1
        
        return cascade_reward

    def _handle_cascades(self):
        total_reward = 0
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            
            # sfx: match sound
            num_cleared = len(matches)
            if num_cleared >= 4:
                total_reward += 5 # Bonus for large match
            
            total_reward += num_cleared * 1
            self.gems_collected_this_game = min(self.TOTAL_GEMS_TO_COLLECT, self.gems_collected_this_game + num_cleared)

            for r, c in matches:
                self._spawn_particles(c, r, self.GEM_COLORS[self.grid[r,c]-1])
                self.grid[r, c] = 0
            
            self._apply_gravity()
            self._fill_new_gems()

        return total_reward

    def _find_all_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r, c + 1] == self.grid[r, c + 2]:
                    matches.update([(r, c), (r, c + 1), (r, c + 2)])
        # Vertical
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r + 1, c] == self.grid[r + 2, c]:
                    matches.update([(r, c), (r + 1, c), (r + 2, c)])
        return matches
    
    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
                    
    def _fill_new_gems(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
    
    def _spawn_particles(self, c, r, color):
        x = self.GRID_OFFSET_X + c * self.GEM_SIZE + self.GEM_SIZE // 2
        y = self.GRID_OFFSET_Y + r * self.GEM_SIZE + self.GEM_SIZE // 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({'pos': [x, y], 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color, 'radius': radius})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1

    def _check_termination(self):
        if self.game_over:
            return True
        if self.gems_collected_this_game >= self.TOTAL_GEMS_TO_COLLECT:
            self.game_over = True
            self.win_state = True
            return True
        if self.moves_left <= 0:
            self.game_over = True
            self.win_state = False
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win_state = False
            return True
        return False
        
    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_WIDTH * self.GEM_SIZE, self.GRID_HEIGHT * self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)
        
        # Draw gems
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    x = self.GRID_OFFSET_X + c * self.GEM_SIZE
                    y = self.GRID_OFFSET_Y + r * self.GEM_SIZE
                    color = self.GEM_COLORS[gem_type - 1]
                    highlight_color = self.GEM_HIGHLIGHTS[gem_type-1]
                    
                    rect = pygame.Rect(x + 4, y + 4, self.GEM_SIZE - 8, self.GEM_SIZE - 8)
                    pygame.gfxdraw.box(self.screen, rect, color)
                    pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, int(self.GEM_SIZE * 0.25), highlight_color)
                    pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, int(self.GEM_SIZE * 0.25), highlight_color)

        # Draw selected gem highlight
        if self.selected_gem_pos:
            r, c = self.selected_gem_pos
            x = self.GRID_OFFSET_X + c * self.GEM_SIZE
            y = self.GRID_OFFSET_Y + r * self.GEM_SIZE
            pulse_alpha = 100 + 100 * (1 + math.sin(math.radians(self.selection_pulse))) / 2
            
            surf = pygame.Surface((self.GEM_SIZE, self.GEM_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(surf, (255, 255, 200, pulse_alpha), (2, 2, self.GEM_SIZE - 4, self.GEM_SIZE - 4), 4, border_radius=8)
            self.screen.blit(surf, (x, y))
            
        # Draw cursor
        cx, cy = self.cursor_pos
        x = self.GRID_OFFSET_X + cx * self.GEM_SIZE
        y = self.GRID_OFFSET_Y + cy * self.GEM_SIZE
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (x, y, self.GEM_SIZE, self.GEM_SIZE), 3, border_radius=6)

        # Draw particles
        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            color = (*p['color'], alpha)
            surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(surf, int(p['radius']), int(p['radius']), int(p['radius']), color)
            pygame.gfxdraw.filled_circle(surf, int(p['radius']), int(p['radius']), int(p['radius']), color)
            self.screen.blit(surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

    def _render_ui(self):
        # Moves Left
        moves_color = self.COLOR_TEXT_WARN if self.moves_left <= 3 and not self.game_over else self.COLOR_TEXT
        moves_surf = self.font_main.render(f"Moves: {self.moves_left}", True, moves_color)
        self.screen.blit(moves_surf, (20, 15))

        # Gems Collected
        progress = self.gems_collected_this_game / self.TOTAL_GEMS_TO_COLLECT
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 20
        bar_y = 20
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        fill_width = max(0, int(bar_width * progress))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.GEM_COLORS[2], (bar_x, bar_y, fill_width, bar_height), border_radius=5)
        
        gems_text = f"{self.gems_collected_this_game}/{self.TOTAL_GEMS_TO_COLLECT} Gems"
        gems_surf = self.font_main.render(gems_text, True, self.COLOR_TEXT)
        self.screen.blit(gems_surf, (bar_x + bar_width/2 - gems_surf.get_width()/2, bar_y - 2))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_state:
                text = "YOU WIN!"
                color = self.COLOR_TEXT_WIN
            else:
                text = "GAME OVER"
                color = self.COLOR_TEXT_WARN
            
            text_surf = self.font_large.render(text, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def validate_implementation(self):
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Matcher")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + env.game_description)
    print(env.user_guide + "\n")

    while running:
        # --- Action mapping for human input ---
        movement = 0 # no-op
        space_held = False
        shift_held = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True

        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        # --- Handle Pygame events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting Environment ---")
                obs, info = env.reset()
                total_reward = 0

        # --- Step the environment ---
        # Since auto_advance is False, we only step on an action.
        # For human play, we can step every frame to register key holds.
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Moves: {info['moves_left']}")

        if terminated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # In a real scenario, you might wait for a key press to reset
            # For this demo, we'll just let it sit on the end screen
            # running = False # uncomment to exit on termination

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for human play

    pygame.quit()