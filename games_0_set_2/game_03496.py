
# Generated: 2025-08-27T23:31:31.176865
# Source Brief: brief_03496.md
# Brief Index: 3496

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to select a gem. "
        "Select an adjacent gem to swap. Shift to reshuffle the board (costs a move)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match cascading gems to reach a target score in a limited number of moves. "
        "Create chains and combos for higher scores."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Game Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_GEM_TYPES = 6
    GEM_SIZE = 40
    GRID_OFFSET_X = 160
    GRID_OFFSET_Y = 40
    
    SCORE_TARGET = 5000
    MOVE_LIMIT = 20

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (30, 40, 50)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_GAMEOVER_TEXT = (255, 255, 255)
    COLOR_WIN_TEXT = (255, 255, 100)
    
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 255, 80),   # Yellow
        (255, 80, 255),   # Magenta
        (80, 255, 255),   # Cyan
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width = 640
        self.screen_height = 400
        
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
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables (initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.selected_gem = None
        self.moves_left = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.space_was_pressed = False
        self.shift_was_pressed = False
        self.particles = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.MOVE_LIMIT
        self.game_over = False
        self.win = False
        self.steps = 0
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_gem = None
        self.space_was_pressed = False
        self.shift_was_pressed = False
        self.particles = []

        self._generate_initial_grid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Input ---
        if not self.game_over:
            # 1. Cursor Movement
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)
            
            # 2. Shift to Reshuffle
            if shift_pressed and not self.shift_was_pressed:
                self._shuffle_grid()
                self.moves_left -= 1
                reward -= 5 # Cost for reshuffling
                # SFX: board_shuffle.wav
            
            # 3. Space to Select/Swap
            if space_pressed and not self.space_was_pressed:
                # SFX: select.wav
                if self.selected_gem is None:
                    self.selected_gem = tuple(self.cursor_pos)
                else:
                    # Second selection
                    if tuple(self.cursor_pos) == self.selected_gem:
                        # Deselect
                        self.selected_gem = None
                    elif self._is_adjacent(self.selected_gem, self.cursor_pos):
                        # Attempt swap
                        reward += self._attempt_swap(self.selected_gem, tuple(self.cursor_pos))
                        self.selected_gem = None
                    else:
                        # Invalid second selection, move selection to new cursor
                        self.selected_gem = tuple(self.cursor_pos)

        self.space_was_pressed = space_pressed
        self.shift_was_pressed = shift_pressed
        self.steps += 1
        self._update_particles()
        
        # --- Check Termination Conditions ---
        if not self.game_over:
            if self.score >= self.SCORE_TARGET:
                self.game_over = True
                self.win = True
                reward += 100
                # SFX: win_jingle.wav
            elif self.moves_left <= 0:
                self.game_over = True
                reward -= 10
                # SFX: lose_sound.wav

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _attempt_swap(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Swap gems
        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]

        matches1 = self._find_matches_at(x1, y1)
        matches2 = self._find_matches_at(x2, y2)
        all_matches = matches1.union(matches2)

        if not all_matches:
            # No match, swap back
            self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
            # SFX: invalid_swap.wav
            return -0.1 # Small penalty for invalid move
        else:
            # Match found, consume a move and process cascades
            self.moves_left -= 1
            # SFX: match_success.wav
            return self._process_cascades(all_matches)
    
    def _process_cascades(self, initial_matches):
        total_reward = 0
        combo_multiplier = 1.0
        matches_to_process = deque([initial_matches])

        while matches_to_process:
            current_matches = matches_to_process.popleft()
            if not current_matches:
                continue

            # Calculate score and reward for this wave
            score_gain, reward_gain = self._calculate_match_rewards(current_matches, combo_multiplier)
            self.score += score_gain
            total_reward += reward_gain
            
            # Create particles and remove gems
            for x, y in current_matches:
                self._create_particles(x, y, self.grid[y, x])
                self.grid[y, x] = -1 # Mark for removal
            
            # SFX: gem_destroy.wav with pitch increasing with combo_multiplier

            # Apply gravity and refill
            self._gems_fall()
            self._refill_grid()

            # Find new matches caused by the cascade
            new_matches = self._find_all_matches()
            if new_matches:
                matches_to_process.append(new_matches)
                combo_multiplier += 0.5 # Increase combo for next wave
                # SFX: combo_chime.wav
        
        # After all cascades, check for no-more-moves scenario
        if not self._find_possible_moves():
            self._shuffle_grid()
            # SFX: board_shuffle.wav
            
        return total_reward

    def _calculate_match_rewards(self, matches, combo_multiplier):
        num_gems = len(matches)
        base_score = num_gems * 10
        base_reward = num_gems
        
        if num_gems == 4:
            base_score *= 2
            base_reward += 5
        elif num_gems >= 5:
            base_score *= 3
            base_reward += 10
            
        return int(base_score * combo_multiplier), base_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(
            self.GRID_OFFSET_X - 5, self.GRID_OFFSET_Y - 5,
            self.GRID_WIDTH * self.GEM_SIZE + 10, self.GRID_HEIGHT * self.GEM_SIZE + 10
        )
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=10)

        # Draw gems
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[y, x]
                if gem_type != -1:
                    self._draw_gem(x, y, gem_type)
        
        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + cursor_x * self.GEM_SIZE,
            self.GRID_OFFSET_Y + cursor_y * self.GEM_SIZE,
            self.GEM_SIZE, self.GEM_SIZE
        )
        pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, 3, border_radius=5)

        # Draw selection highlight
        if self.selected_gem:
            sel_x, sel_y = self.selected_gem
            sel_rect = pygame.Rect(
                self.GRID_OFFSET_X + sel_x * self.GEM_SIZE,
                self.GRID_OFFSET_Y + sel_y * self.GEM_SIZE,
                self.GEM_SIZE, self.GEM_SIZE
            )
            # Pulsating effect
            pulse = (math.sin(self.steps * 0.3) + 1) / 2
            color = (255, 255, int(150 + pulse * 105))
            pygame.draw.rect(self.screen, color, sel_rect, 4, border_radius=5)

        # Draw particles
        for p in self.particles:
            p_x, p_y = int(p['pos'][0]), int(p['pos'][1])
            pygame.draw.circle(self.screen, p['color'], (p_x, p_y), int(p['size']))

    def _draw_gem(self, x, y, gem_type):
        center_x = self.GRID_OFFSET_X + x * self.GEM_SIZE + self.GEM_SIZE // 2
        center_y = self.GRID_OFFSET_Y + y * self.GEM_SIZE + self.GEM_SIZE // 2
        radius = self.GEM_SIZE // 2 - 4
        color = self.GEM_COLORS[gem_type]
        
        # Draw shadow
        pygame.gfxdraw.filled_circle(self.screen, center_x + 2, center_y + 2, radius, (0,0,0,50))
        # Draw main gem body
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
        # Draw highlight
        h_color = (min(255, color[0]+80), min(255, color[1]+80), min(255, color[2]+80))
        pygame.gfxdraw.filled_circle(self.screen, center_x - radius//3, center_y - radius//3, radius//3, h_color)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Moves
        moves_text = self.font_medium.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.screen_width - moves_text.get_width() - 20, 20))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg_text = self.font_large.render("YOU WIN!", True, self.COLOR_WIN_TEXT)
            else:
                msg_text = self.font_large.render("GAME OVER", True, self.COLOR_GAMEOVER_TEXT)
            
            text_rect = msg_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg_text, text_rect)

    def _generate_initial_grid(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if not self._find_all_matches() and self._find_possible_moves():
                break

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Horizontal
                if c < self.GRID_WIDTH - 2 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2] != -1:
                    matches.add((c, r)); matches.add((c+1, r)); matches.add((c+2, r))
                # Vertical
                if r < self.GRID_HEIGHT - 2 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c] != -1:
                    matches.add((c, r)); matches.add((c, r+1)); matches.add((c, r+2))
        return matches

    def _find_matches_at(self, c, r):
        gem_type = self.grid[r, c]
        if gem_type == -1: return set()
        
        h_matches, v_matches = { (c, r) }, { (c, r) }
        
        # Horizontal
        for i in range(c - 1, -1, -1):
            if self.grid[r, i] == gem_type: h_matches.add((i, r))
            else: break
        for i in range(c + 1, self.GRID_WIDTH):
            if self.grid[r, i] == gem_type: h_matches.add((i, r))
            else: break
        
        # Vertical
        for i in range(r - 1, -1, -1):
            if self.grid[i, c] == gem_type: v_matches.add((c, i))
            else: break
        for i in range(r + 1, self.GRID_HEIGHT):
            if self.grid[i, c] == gem_type: v_matches.add((c, i))
            else: break
            
        matches = set()
        if len(h_matches) >= 3: matches.update(h_matches)
        if len(v_matches) >= 3: matches.update(v_matches)
        return matches

    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Try swapping right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                    if self._find_matches_at(c, r) or self._find_matches_at(c+1, r):
                        moves.append(((c,r), (c+1,r)))
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c] # Swap back
                # Try swapping down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                    if self._find_matches_at(c, r) or self._find_matches_at(c, r+1):
                        moves.append(((c,r), (c,r+1)))
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c] # Swap back
        return moves
    
    def _gems_fall(self):
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
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _shuffle_grid(self):
        flat_gems = self.grid.flatten().tolist()
        self.np_random.shuffle(flat_gems)
        self.grid = np.array(flat_gems).reshape((self.GRID_HEIGHT, self.GRID_WIDTH))
        
        # Ensure no matches and at least one move after shuffling
        while self._find_all_matches() or not self._find_possible_moves():
            self.np_random.shuffle(flat_gems)
            self.grid = np.array(flat_gems).reshape((self.GRID_HEIGHT, self.GRID_WIDTH))

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _create_particles(self, x, y, gem_type):
        center_x = self.GRID_OFFSET_X + x * self.GEM_SIZE + self.GEM_SIZE // 2
        center_y = self.GRID_OFFSET_Y + y * self.GEM_SIZE + self.GEM_SIZE // 2
        color = self.GEM_COLORS[gem_type]
        for _ in range(20): # Number of particles
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'size': self.np_random.uniform(2, 5),
                'life': 30 # Frames
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['size'] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0.5]

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part is for human testing and requires a display.
    # It will not run in a headless environment.
    try:
        # Re-initialize pygame with a display for manual play
        pygame.display.init()
        screen = pygame.display.set_mode((env.screen_width, env.screen_height))
        pygame.display.set_caption("Gem Matcher")
        
        obs, info = env.reset()
        done = False
        
        print(env.user_guide)
        
        while not done:
            movement = 0 # No-op
            space = 0
            shift = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Draw the observation to the display screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Limit frame rate for playability
            
        print(f"Game Over! Final Score: {info['score']}")
        
    except pygame.error as e:
        print(f"Pygame display error: {e}")
        print("Skipping manual play. This is expected in a headless environment.")
    
    pygame.quit()