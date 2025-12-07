
# Generated: 2025-08-28T00:56:28.246580
# Source Brief: brief_03946.md
# Brief Index: 3946

        
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

    # User-facing control string (describes the agent's actions)
    user_guide = (
        "The cursor scans the grid. Use ↑↓←→ to swap the "
        "current gem with an adjacent one. Do nothing to skip."
    )

    # User-facing description of the game
    game_description = (
        "Match 3 or more gems to score points. Create chain reactions for big bonuses. "
        "Reach 5000 points in 30 moves to win!"
    )

    # Frames only advance when an action is received
    auto_advance = False
    
    # --- Game Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    CELL_SIZE = 44
    GRID_X_OFFSET = (640 - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y_OFFSET = (400 - GRID_HEIGHT * CELL_SIZE) // 2 + 20
    
    NUM_GEM_TYPES = 6
    WIN_SCORE = 5000
    MAX_MOVES = 30
    
    # --- Colors ---
    COLOR_BG = (25, 30, 45)
    COLOR_GRID_BG = (40, 45, 60)
    COLOR_GRID_LINE = (55, 60, 75)
    COLOR_CURSOR = (255, 255, 0, 100)
    COLOR_UI_TEXT = (240, 240, 240)
    
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]
    GEM_SHADOW_COLORS = [
        (180, 50, 50),
        (50, 180, 50),
        (50, 100, 180),
        (180, 180, 50),
        (180, 50, 180),
        (50, 180, 180),
    ]

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
        
        self.font_ui = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_score_popup = pygame.font.SysFont("sans-serif", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("sans-serif", 48, bold=True)

        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.steps = None
        self.no_match_moves_in_a_row = None
        self.particles = []
        self.score_popups = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.no_match_moves_in_a_row = 0
        self.particles = []
        self.score_popups = []
        
        self._create_initial_grid()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.particles.clear()
        self.score_popups.clear()

        movement = action[0]
        reward = 0
        
        # A movement action constitutes a "turn"
        if movement != 0:
            self.moves_left -= 1
            
            cx, cy = self.cursor_pos
            nx, ny = cx, cy
            
            if movement == 1: ny -= 1  # Up
            elif movement == 2: ny += 1  # Down
            elif movement == 3: nx -= 1  # Left
            elif movement == 4: nx += 1  # Right
            
            # Check for valid swap
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                # Perform swap
                self.grid[cy, cx], self.grid[ny, nx] = self.grid[ny, nx], self.grid[cy, cx]
                
                # Resolve matches
                total_score_this_turn, matches_found = self._resolve_board()
                
                if matches_found:
                    # Successful move
                    self.score += total_score_this_turn
                    reward += total_score_this_turn / 100.0  # Scale score to reward
                    self.no_match_moves_in_a_row = 0
                    # placeholder for sound effect: pygame.mixer.Sound("match.wav").play()
                else:
                    # Invalid move, swap back
                    self.grid[cy, cx], self.grid[ny, nx] = self.grid[ny, nx], self.grid[cy, cx]
                    reward = -0.2
                    self.no_match_moves_in_a_row += 1
                    # placeholder for sound effect: pygame.mixer.Sound("invalid_move.wav").play()
            else:
                # Attempted swap out of bounds
                reward = -0.2
                self.no_match_moves_in_a_row += 1
        
        # Advance cursor after every action (swap or no-op)
        self.cursor_pos[0] += 1
        if self.cursor_pos[0] >= self.GRID_WIDTH:
            self.cursor_pos[0] = 0
            self.cursor_pos[1] += 1
            if self.cursor_pos[1] >= self.GRID_HEIGHT:
                self.cursor_pos[1] = 0
        
        # Anti-softlock: Reshuffle board if no matches are made for a while
        if self.no_match_moves_in_a_row >= 5:
            self._create_initial_grid() # Reshuffle
            self.no_match_moves_in_a_row = 0
            reward -= 1 # Penalty for needing a reshuffle

        terminated = self.moves_left <= 0 or self.score >= self.WIN_SCORE
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
                # placeholder for sound effect: pygame.mixer.Sound("win.wav").play()
            else:
                reward += -10
                # placeholder for sound effect: pygame.mixer.Sound("lose.wav").play()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_initial_grid(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if not self._find_matches():
                break

    def _resolve_board(self):
        total_score = 0
        any_matches = False
        chain_multiplier = 1

        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            any_matches = True
            # placeholder for sound effect: pygame.mixer.Sound(f"chain_{chain_multiplier}.wav").play()
            
            # Calculate score and clear matched gems
            score_this_cascade = 0
            gems_cleared = set()
            for match in matches:
                gems_cleared.update(match)
            
            match_score = len(gems_cleared) * 10 * chain_multiplier
            if len(gems_cleared) >= 6: match_score *= 2 # Bonus for large clears
            score_this_cascade += match_score

            for r, c in gems_cleared:
                self._create_particles(c, r, self.grid[r, c])
                self.grid[r, c] = 0
            
            if score_this_cascade > 0:
                # Find a representative position for the score popup
                avg_r = int(sum(r for r, c in gems_cleared) / len(gems_cleared))
                avg_c = int(sum(c for r, c in gems_cleared) / len(gems_cleared))
                self.score_popups.append({
                    "text": f"+{score_this_cascade}", "pos": [avg_c, avg_r], "life": 60
                })
            
            total_score += score_this_cascade

            # Apply gravity
            for c in range(self.GRID_WIDTH):
                empty_row = self.GRID_HEIGHT - 1
                for r in range(self.GRID_HEIGHT - 1, -1, -1):
                    if self.grid[r, c] != 0:
                        self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                        empty_row -= 1
            
            # Refill board
            for r in range(self.GRID_HEIGHT):
                for c in range(self.GRID_WIDTH):
                    if self.grid[r, c] == 0:
                        self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
            
            chain_multiplier += 1
        
        return total_score, any_matches

    def _find_matches(self):
        matches = set()
        # Horizontal matches
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2] != 0:
                    matches.add(frozenset([(r, c), (r, c+1), (r, c+2)]))
        # Vertical matches
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c] != 0:
                    matches.add(frozenset([(r, c), (r+1, c), (r+2, c)]))
        
        # Consolidate overlapping matches
        if not matches:
            return []
        
        consolidated = []
        while matches:
            current_match = set(matches.pop())
            while True:
                found_overlap = False
                remaining_matches = set()
                for other_match in matches:
                    if not current_match.isdisjoint(other_match):
                        current_match.update(other_match)
                        found_overlap = True
                    else:
                        remaining_matches.add(other_match)
                matches = remaining_matches
                if not found_overlap:
                    break
            consolidated.append(current_match)
        return consolidated

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
            "cursor_pos": list(self.cursor_pos),
        }

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, 
                                self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=8)

        # Draw gems and grid lines
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    self._draw_gem(c, r, gem_type)
        
        # Draw grid lines on top
        for r in range(1, self.GRID_HEIGHT):
            y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, y))
        for c in range(1, self.GRID_WIDTH):
            x = self.GRID_X_OFFSET + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE))

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_X_OFFSET + cx * self.CELL_SIZE, 
                                  self.GRID_Y_OFFSET + cy * self.CELL_SIZE, 
                                  self.CELL_SIZE, self.CELL_SIZE)
        
        # Pulsing effect for cursor
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 # 0 to 1
        alpha = 80 + pulse * 60
        cursor_surface = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (255, 255, 0, alpha), cursor_surface.get_rect(), border_radius=8)
        pygame.draw.rect(cursor_surface, (255, 255, 200), cursor_surface.get_rect(), width=3, border_radius=8)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

        # Draw and update particles
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['vel'][1] += 0.1 # gravity
                alpha = max(0, p['life'] * 4)
                pygame.draw.circle(self.screen, p['color'] + (alpha,), p['pos'], int(p['size']))
                p['size'] = max(1, p['size'] * 0.98)

        # Draw and update score popups
        for s in self.score_popups[:]:
            s['life'] -= 1
            if s['life'] <= 0:
                self.score_popups.remove(s)
            else:
                s['pos'][1] -= 0.5
                alpha = max(0, s['life'] * 4)
                text_surf = self.font_score_popup.render(s['text'], True, self.COLOR_UI_TEXT)
                text_surf.set_alpha(alpha)
                px = self.GRID_X_OFFSET + s['pos'][0] * self.CELL_SIZE + self.CELL_SIZE // 2 - text_surf.get_width() // 2
                py = self.GRID_Y_OFFSET + s['pos'][1] * self.CELL_SIZE + self.CELL_SIZE // 2 - text_surf.get_height() // 2
                self.screen.blit(text_surf, (px, py))

    def _draw_gem(self, c, r, gem_type):
        gem_color = self.GEM_COLORS[gem_type - 1]
        shadow_color = self.GEM_SHADOW_COLORS[gem_type - 1]
        
        x = self.GRID_X_OFFSET + c * self.CELL_SIZE
        y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
        
        gem_rect = pygame.Rect(x + 4, y + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
        shadow_rect = gem_rect.copy()
        shadow_rect.y += 2

        pygame.draw.rect(self.screen, shadow_color, shadow_rect, border_radius=8)
        pygame.draw.rect(self.screen, gem_color, gem_rect, border_radius=8)

        # Highlight/glint
        highlight_pos = (gem_rect.left + 6, gem_rect.top + 6)
        pygame.draw.circle(self.screen, (255, 255, 255, 100), highlight_pos, 4)

    def _create_particles(self, c, r, gem_type):
        x = self.GRID_X_OFFSET + c * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_Y_OFFSET + r * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.GEM_COLORS[gem_type - 1]

        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(20, 40),
                'size': random.uniform(2, 5),
                'color': color
            })

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Moves left display
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        moves_rect = moves_text.get_rect(topright=(620, 10))
        self.screen.blit(moves_text, moves_rect)

        # Game Over display
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.score >= self.WIN_SCORE:
                end_text = self.font_game_over.render("YOU WIN!", True, (100, 255, 100))
            else:
                end_text = self.font_game_over.render("GAME OVER", True, (255, 100, 100))
            
            end_rect = end_text.get_rect(center=(320, 200))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part allows a human to play the game for testing purposes.
    # It requires a windowed display.
    import sys
    
    # Check if we can create a display
    try:
        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Match-3 Puzzle")
        is_playable = True
    except pygame.error:
        print("Display not available. Skipping interactive test.")
        is_playable = False

    if is_playable:
        obs, info = env.reset()
        terminated = False
        
        action_map = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }
        
        while not terminated:
            action = [0, 0, 0] # Default action is no-op
            
            # Wait for a key press
            event = pygame.event.wait()
            
            if event.type == pygame.QUIT:
                break
            if event.type == pygame.KEYDOWN:
                if event.key in action_map:
                    action[0] = action_map[event.key]
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    continue
                elif event.key == pygame.K_q:
                    break
            
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

            # Render to the display window
            frame = np.transpose(env._get_observation(), (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
    env.close()