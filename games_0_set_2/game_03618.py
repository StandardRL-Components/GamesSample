import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a gem and make a match."
    )

    game_description = (
        "Match-3 puzzle game. Select a gem to match it with 3 or more adjacent gems of the same color. Create combos with cascading matches to maximize your score and reach 500 points."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 8
    NUM_GEM_TYPES = 6
    GEM_SIZE = 40
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_SIZE * GEM_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_SIZE * GEM_SIZE) // 2
    MIN_MATCH_SIZE = 3
    WIN_SCORE = 500
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (30, 35, 40)
    COLOR_GRID_LINES = (50, 55, 60)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 255, 100)
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 255, 80),   # Yellow
        (200, 80, 255),   # Purple
        (255, 150, 80),   # Orange
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_combo = pygame.font.SysFont("Arial", 20, bold=True)

        # --- Game State Initialization ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.steps = None
        self.game_over = None
        
        # --- Effects ---
        self.particles = []
        self.combo_popups = []

        # --- Initialize state variables ---
        # The reset call is deferred to the first call to reset()
        
        # --- Validation ---
        # self.validate_implementation() # Validation can be run separately if needed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.particles.clear()
        self.combo_popups.clear()

        self._generate_initial_grid()
        self._ensure_moves_exist()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Clear single-frame effects from the previous step
        self.particles.clear()
        self.combo_popups.clear()

        movement = action[0]
        space_pressed = action[1] == 1
        
        reward = 0
        
        # Prioritize movement over selection
        if movement != 0:
            self._handle_movement(movement)
            # Small penalty for spending a step just moving
            reward = -0.1
        elif space_pressed:
            reward = self._handle_selection()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and self.score >= self.WIN_SCORE:
            reward += 100 # Win bonus

        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_SIZE
        self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_SIZE

    def _handle_selection(self):
        col, row = self.cursor_pos
        
        if self.grid[row, col] == -1:
            return -1 # Penalty for clicking empty space

        connected_gems = self._find_connected_gems(col, row)

        if len(connected_gems) < self.MIN_MATCH_SIZE:
            return -1 # Penalty for invalid match attempt

        # Successful match, start cascade
        total_reward = 0
        combo = 1
        
        current_matches = [connected_gems]
        
        while current_matches:
            match_points = 0
            for match_group in current_matches:
                match_points += len(match_group)
                self._create_match_effects(match_group)
                for c, r in match_group:
                    self.grid[r, c] = -1 # Mark for removal
            
            # Score based on number of gems and combo
            total_reward += match_points + (5 * combo)
            self.score += match_points * combo

            if combo > 1:
                # Find center of first match group for popup
                avg_x = sum(c for c, r in current_matches[0]) / len(current_matches[0])
                avg_y = sum(r for c, r in current_matches[0]) / len(current_matches[0])
                self.combo_popups.append({
                    "text": f"COMBO x{combo}",
                    "pos": (avg_x, avg_y),
                })
                # sfx: combo_sound

            self._apply_gravity_and_refill()
            
            current_matches = self._find_all_board_matches()
            if current_matches:
                combo += 1
                # sfx: cascade_sound
        
        # After all cascades, check if any moves are left
        self._ensure_moves_exist()
        
        return total_reward

    def _generate_initial_grid(self):
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
        # Prevent matches on spawn
        while self._find_all_board_matches():
            matches = self._find_all_board_matches()
            for group in matches:
                for c, r in group:
                    # Reroll the gem, avoiding its previous color
                    original_color = self.grid[r, c]
                    new_color = original_color
                    while new_color == original_color:
                        new_color = self.np_random.integers(0, self.NUM_GEM_TYPES)
                    self.grid[r, c] = new_color

    def _ensure_moves_exist(self):
        if not self._find_all_possible_matches():
            # No moves left, reshuffle the board
            flat_grid = self.grid.flatten()
            self.np_random.shuffle(flat_grid)
            self.grid = flat_grid.reshape((self.GRID_SIZE, self.GRID_SIZE))
            # Recursively call until a valid board is made
            self._generate_initial_grid()
            self._ensure_moves_exist()

    def _find_connected_gems(self, start_col, start_row):
        target_color = self.grid[start_row, start_col]
        if target_color == -1:
            return []

        q = deque([(start_col, start_row)])
        visited = set([(start_col, start_row)])
        connected = []

        while q:
            c, r = q.popleft()
            connected.append((c, r))

            for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nc, nr = c + dc, r + dr
                if 0 <= nc < self.GRID_SIZE and 0 <= nr < self.GRID_SIZE:
                    if (nc, nr) not in visited and self.grid[nr, nc] == target_color:
                        visited.add((nc, nr))
                        q.append((nc, nr))
        return connected

    def _find_all_board_matches(self):
        matches = []
        checked = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (c, r) in checked or self.grid[r,c] == -1:
                    continue
                
                # Check horizontal
                h_match = [(c, r)]
                for i in range(1, self.GRID_SIZE - c):
                    if self.grid[r, c+i] == self.grid[r, c]:
                        h_match.append((c+i, r))
                    else: break
                if len(h_match) >= self.MIN_MATCH_SIZE:
                    matches.append(h_match)
                    for pos in h_match: checked.add(pos)

                # Check vertical
                v_match = [(c, r)]
                for i in range(1, self.GRID_SIZE - r):
                    if self.grid[r+i, c] == self.grid[r, c]:
                        v_match.append((c, r+i))
                    else: break
                if len(v_match) >= self.MIN_MATCH_SIZE:
                    # Avoid adding subsets of horizontal matches
                    is_new = True
                    for pos in v_match:
                        if pos in checked:
                            is_new = False
                            break
                    if is_new:
                        matches.append(v_match)
                        for pos in v_match: checked.add(pos)
        return matches

    def _find_all_possible_matches(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == -1: continue
                
                # Temporarily swap with neighbors and check for matches
                for dc, dr in [(0, 1), (1, 0)]:
                    nc, nr = c + dc, r + dr
                    if 0 <= nc < self.GRID_SIZE and 0 <= nr < self.GRID_SIZE:
                        # Swap
                        self.grid[r, c], self.grid[nr, nc] = self.grid[nr, nc], self.grid[r, c]
                        
                        # Check if the swap created a match at either location
                        if len(self._find_connected_gems(c, r)) >= self.MIN_MATCH_SIZE or \
                           len(self._find_connected_gems(nc, nr)) >= self.MIN_MATCH_SIZE:
                            # Swap back and return True
                            self.grid[r, c], self.grid[nr, nc] = self.grid[nr, nc], self.grid[r, c]
                            return True
                        
                        # Swap back
                        self.grid[r, c], self.grid[nr, nc] = self.grid[nr, nc], self.grid[r, c]
        return False

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_SIZE):
            empty_slots = 0
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] == -1:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[r + empty_slots, c] = self.grid[r, c]
                    self.grid[r, c] = -1
            
            # Refill top
            for r in range(empty_slots):
                self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        if not self._find_all_possible_matches():
            return True
        return False

    def _get_observation(self):
        # Ensure grid is initialized if reset() hasn't been called yet
        if self.grid is None:
            self.reset()
        
        self.screen.fill(self.COLOR_BG)
        self._render_grid_bg()
        self._render_gems()
        self._render_cursor()
        self._render_particles()
        self._render_combo_popups()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_grid_bg(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            x = self.GRID_OFFSET_X + i * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_SIZE * self.GEM_SIZE))
            # Horizontal lines
            y = self.GRID_OFFSET_Y + i * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_SIZE * self.GEM_SIZE, y))

    def _render_gems(self):
        gem_radius = self.GEM_SIZE // 2 - 4
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    color = self.GEM_COLORS[gem_type]
                    center_x = self.GRID_OFFSET_X + c * self.GEM_SIZE + self.GEM_SIZE // 2
                    center_y = self.GRID_OFFSET_Y + r * self.GEM_SIZE + self.GEM_SIZE // 2
                    
                    # Draw filled circle with anti-aliasing
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, gem_radius, color)
                    # FIX: The generator expression must be converted to a tuple or list
                    outline_color = tuple(min(255, val + 50) for val in color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, gem_radius, outline_color)

    def _render_cursor(self):
        c, r = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_OFFSET_X + c * self.GEM_SIZE,
            self.GRID_OFFSET_Y + r * self.GEM_SIZE,
            self.GEM_SIZE, self.GEM_SIZE
        )
        # Pulsing glow effect
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
        line_width = int(2 + pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, line_width, border_radius=4)
    
    def _create_match_effects(self, matched_gems):
        # sfx: match_sound
        for c, r in matched_gems:
            center_x = self.GRID_OFFSET_X + c * self.GEM_SIZE + self.GEM_SIZE // 2
            center_y = self.GRID_OFFSET_Y + r * self.GEM_SIZE + self.GEM_SIZE // 2
            color = self.GEM_COLORS[self.grid[r,c]]
            
            for _ in range(10): # 10 particles per gem
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 4)
                self.particles.append({
                    "pos": [center_x, center_y],
                    "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                    "radius": random.uniform(2, 5),
                    "color": color
                })

    def _render_particles(self):
        # In auto_advance=False, particles are drawn once and then cleared.
        # This gives a static "burst" effect for one frame.
        for p in self.particles:
            pos = (int(p["pos"][0] + p["vel"][0]), int(p["pos"][1] + p["vel"][1]))
            pygame.draw.circle(self.screen, p["color"], pos, int(p["radius"]))

    def _render_combo_popups(self):
        for popup in self.combo_popups:
            text_surf = self.font_combo.render(popup["text"], True, self.COLOR_UI_TEXT)
            center_x = self.GRID_OFFSET_X + popup["pos"][0] * self.GEM_SIZE + self.GEM_SIZE // 2
            center_y = self.GRID_OFFSET_Y + popup["pos"][1] * self.GEM_SIZE + self.GEM_SIZE // 2
            text_rect = text_surf.get_rect(center=(center_x, center_y))
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        score_text = f"SCORE: {self.score} / {self.WIN_SCORE}"
        steps_text = f"MOVES: {self.steps} / {self.MAX_STEPS}"

        score_surf = self.font_main.render(score_text, True, self.COLOR_UI_TEXT)
        steps_surf = self.font_main.render(steps_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(score_surf, (20, 10))
        self.screen.blit(steps_surf, (self.SCREEN_WIDTH - steps_surf.get_width() - 20, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_surf = self.font_main.render(end_text, True, (255, 255, 255))
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_surf, end_rect)

    def close(self):
        pygame.quit()

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

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It will create a visible pygame window
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Matcher")
    
    running = True
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0] # no-op, space released, shift released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            if terminated:
                print("--- GAME OVER ---")
                
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for human play

    env.close()