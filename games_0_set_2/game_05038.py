
# Generated: 2025-08-28T03:47:45.648898
# Source Brief: brief_05038.md
# Brief Index: 5038

        
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
        "Controls: Arrows to move selector. Space to select a gem. Arrows again to swap with an adjacent gem. Space again to cancel selection."
    )

    game_description = (
        "Swap adjacent gems to create matches of 3 or more. Create chain reactions to get combo bonuses and reach the target score before you run out of moves."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.NUM_GEM_TYPES = 6
        self.CELL_SIZE = 36
        self.BOARD_OFFSET_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.BOARD_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2 + 20
        self.INITIAL_MOVES = 30
        self.TARGET_SCORE = 1000
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (25, 30, 45)
        self.COLOR_GRID = (50, 60, 80)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_SCORE = (255, 215, 0)
        self.COLOR_MOVES = (173, 216, 230)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        self.font_combo = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.np_random = None
        self.board = None
        self.selector_pos = None
        self.selected_gem_pos = None
        self.interaction_state = None
        self.last_space_state = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.steps = None
        self.particles = []
        self.combo_text_alpha = 0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.INITIAL_MOVES
        self.game_over = False
        self.interaction_state = 'browse'
        self.selector_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_gem_pos = None
        self.last_space_state = False
        self.particles = []
        self.combo_text_alpha = 0
        
        self._generate_initial_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_state
        self.last_space_state = space_held
        
        # --- Handle player input and game logic ---
        if self.interaction_state == 'browse':
            if space_pressed:
                self.selected_gem_pos = list(self.selector_pos)
                self.interaction_state = 'select'
                # sfx: select_gem.wav
            elif movement != 0:
                self._move_selector(movement)
        
        elif self.interaction_state == 'select':
            if space_pressed: # Cancel selection
                self.selected_gem_pos = None
                self.interaction_state = 'browse'
                # sfx: cancel.wav
            elif movement != 0: # Attempt swap
                reward, terminated = self._handle_swap_attempt(movement)
        
        self._update_particles()
        if self.combo_text_alpha > 0:
            self.combo_text_alpha = max(0, self.combo_text_alpha - 10)

        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_swap_attempt(self, direction):
        if self.selected_gem_pos is None:
            return 0, False

        r1, c1 = self.selected_gem_pos
        r2, c2 = r1, c1
        if direction == 1: r2 -= 1 # Up
        elif direction == 2: r2 += 1 # Down
        elif direction == 3: c2 -= 1 # Left
        elif direction == 4: c2 += 1 # Right

        # Check if target is valid and adjacent
        if not (0 <= r2 < self.GRID_HEIGHT and 0 <= c2 < self.GRID_WIDTH and (abs(r1-r2) + abs(c1-c2) == 1)):
            return 0, False # Invalid swap direction, do nothing

        # --- This is a full turn ---
        self.moves_left -= 1
        # sfx: swap.wav
        
        # Perform swap
        self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]

        matches1 = self._find_matches_at([ (r1,c1), (r2,c2) ])
        
        if not matches1:
            # No match, swap back
            self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]
            self.interaction_state = 'browse'
            self.selected_gem_pos = None
            self.selector_pos = [r2, c2]
            # sfx: invalid_swap.wav
            
            reward = -0.2
            terminated = self.moves_left <= 0
            if terminated:
                reward -= 10 # Terminal penalty for losing
            return reward, terminated

        # Match found, handle cascades
        total_reward = self._handle_cascades(initial_matches=matches1)

        self.interaction_state = 'browse'
        self.selected_gem_pos = None
        self.selector_pos = [r2, c2]

        terminated = self.score >= self.TARGET_SCORE or self.moves_left <= 0
        if terminated:
            if self.score >= self.TARGET_SCORE:
                total_reward += 100 # Terminal reward for winning
                # sfx: win_game.wav
            else:
                total_reward -= 10 # Terminal penalty for losing
                # sfx: lose_game.wav

        return total_reward, terminated
        
    def _handle_cascades(self, initial_matches):
        total_reward = 0
        combo_multiplier = 1
        
        current_matches = initial_matches
        
        while current_matches:
            # Score and reward for this wave
            num_cleared = len(current_matches)
            self.score += num_cleared * 10 * combo_multiplier
            total_reward += num_cleared * 1 # +1 per gem
            if combo_multiplier > 1:
                total_reward += 5 # Combo bonus
            
            self._display_combo_text(combo_multiplier, num_cleared)
            # sfx: match_wave.wav

            # Clear gems and create particles
            for r, c in current_matches:
                self._create_particles(r, c)
                self.board[r, c] = 0 # 0 represents an empty space
            
            self._apply_gravity()
            self._refill_board()
            
            combo_multiplier += 1
            current_matches = self._find_all_matches()

        return total_reward

    def _move_selector(self, movement):
        if movement == 1: self.selector_pos[0] = max(0, self.selector_pos[0] - 1)
        elif movement == 2: self.selector_pos[0] = min(self.GRID_HEIGHT - 1, self.selector_pos[0] + 1)
        elif movement == 3: self.selector_pos[1] = max(0, self.selector_pos[1] - 1)
        elif movement == 4: self.selector_pos[1] = min(self.GRID_WIDTH - 1, self.selector_pos[1] + 1)
        # sfx: cursor_move.wav

    def _find_matches_at(self, positions):
        matches = set()
        for r_start, c_start in positions:
            if not (0 <= r_start < self.GRID_HEIGHT and 0 <= c_start < self.GRID_WIDTH): continue
            gem_type = self.board[r_start, c_start]
            if gem_type == 0: continue

            # Horizontal
            h_matches = {(r_start, c) for c in range(self.GRID_WIDTH) if self.board[r_start, c] == gem_type}
            for c in range(self.GRID_WIDTH - 2):
                if all((r_start, c+i) in h_matches for i in range(3)):
                    matches.update([(r_start, c+i) for i in range(3)])

            # Vertical
            v_matches = {(r, c_start) for r in range(self.GRID_HEIGHT) if self.board[r, c_start] == gem_type}
            for r in range(self.GRID_HEIGHT - 2):
                if all((r+i, c_start) in v_matches for i in range(3)):
                    matches.update([(r+i, c_start) for i in range(3)])
        return self._find_all_matches(matches) # Expand to full connected lines

    def _find_all_matches(self, initial_set=None):
        matches = set() if initial_set is None else set(initial_set)
        
        # Horizontal check
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                gem_type = self.board[r, c]
                if gem_type != 0 and self.board[r, c+1] == gem_type and self.board[r, c+2] == gem_type:
                    matches.update([(r, c), (r, c+1), (r, c+2)])

        # Vertical check
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                gem_type = self.board[r, c]
                if gem_type != 0 and self.board[r+1, c] == gem_type and self.board[r+2, c] == gem_type:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[r, c] != 0:
                    if r != empty_row:
                        self.board[empty_row, c] = self.board[r, c]
                        self.board[r, c] = 0
                    empty_row -= 1
    
    def _refill_board(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.board[r, c] == 0:
                    self.board[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
    
    def _generate_initial_board(self):
        self.board = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while self._find_all_matches():
            self.board = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_HEIGHT + 1):
            y = self.BOARD_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_OFFSET_X, y), (self.BOARD_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.BOARD_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.BOARD_OFFSET_Y), (x, self.BOARD_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE))

        # Draw gems
        gem_radius = self.CELL_SIZE // 2 - 4
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.board[r, c]
                if gem_type > 0:
                    color = self.GEM_COLORS[gem_type - 1]
                    center_x = self.BOARD_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                    center_y = self.BOARD_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                    
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, gem_radius, color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, gem_radius, tuple(min(255, c_val+50) for c_val in color))
                    
        # Draw selector
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.005)) * 100 + 50
        sel_color = (255, 255, 255, pulse)
        sel_rect = pygame.Rect(
            self.BOARD_OFFSET_X + self.selector_pos[1] * self.CELL_SIZE,
            self.BOARD_OFFSET_Y + self.selector_pos[0] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        sel_surf = pygame.Surface(sel_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(sel_surf, sel_color, sel_surf.get_rect(), 4, border_radius=4)
        self.screen.blit(sel_surf, sel_rect.topleft)

        # Draw selected gem highlight
        if self.selected_gem_pos:
            pulse_sel = abs(math.sin(pygame.time.get_ticks() * 0.01)) * 150 + 100
            sel_color = (255, 255, 0, pulse_sel)
            sel_rect = pygame.Rect(
                self.BOARD_OFFSET_X + self.selected_gem_pos[1] * self.CELL_SIZE,
                self.BOARD_OFFSET_Y + self.selected_gem_pos[0] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            sel_surf = pygame.Surface(sel_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(sel_surf, sel_color, sel_surf.get_rect(), 2, border_radius=4)
            self.screen.blit(sel_surf, sel_rect.topleft)

        self._render_particles()

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 10))

        # Moves
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_MOVES)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 10))
        
        # Combo Text
        if self.combo_text_alpha > 0 and hasattr(self, 'last_combo_text'):
            self.last_combo_text.set_alpha(self.combo_text_alpha)
            text_rect = self.last_combo_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(self.last_combo_text, text_rect)
            
        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.TARGET_SCORE else "GAME OVER"
            color = (0, 255, 0) if self.score >= self.TARGET_SCORE else (255, 0, 0)
            
            end_text = self.font_main.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
            self.screen.blit(end_text, text_rect)
            
            score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = score_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 20))
            self.screen.blit(score_text, score_rect)

    def _display_combo_text(self, combo, num_cleared):
        self.combo_text_alpha = 255
        text = f"COMBO x{combo}!" if combo > 1 else f"+{num_cleared}"
        color = self.GEM_COLORS[combo % len(self.GEM_COLORS)]
        self.last_combo_text = self.font_combo.render(text, True, color)

    def _create_particles(self, r, c):
        gem_type = self.board[r,c]
        if gem_type == 0: return
        color = self.GEM_COLORS[gem_type - 1]
        center_x = self.BOARD_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.BOARD_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(20, 40)
            self.particles.append({'x': center_x, 'y': center_y, 'vx': vx, 'vy': vy, 'life': life, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * (255 / 40))))
            color = p['color'] + (alpha,)
            size = int(p['life'] * (8 / 40))
            if size > 0:
                rect = pygame.Rect(int(p['x'] - size/2), int(p['y'] - size/2), size, size)
                surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                surf.fill(color)
                self.screen.blit(surf, rect.topleft)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "game_over": self.game_over,
        }

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Info: {info}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30) # Since auto_advance is False, this just limits human input rate
        
    env.close()