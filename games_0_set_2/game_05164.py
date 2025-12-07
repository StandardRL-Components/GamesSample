
# Generated: 2025-08-28T04:11:58.549174
# Source Brief: brief_05164.md
# Brief Index: 5164

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a gem. "
        "Move the cursor to an adjacent tile and press space again to swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric gem-matching puzzle. Match 3 or more gems to score points. "
        "Reach the target score of 20 gems within 10 moves to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    NUM_GEM_TYPES = 6
    WIN_SCORE = 20
    MAX_MOVES = 10

    # Visuals
    TILE_WIDTH = 40
    TILE_HEIGHT = 20
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 255, 80),   # Yellow
        (255, 80, 255),   # Magenta
        (80, 255, 255),   # Cyan
    ]
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (50, 60, 80)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 48)

        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - self.GRID_HEIGHT * self.TILE_HEIGHT // 2 + 20

        # Initialize state variables to be defined in reset()
        self.np_random = None
        self.grid = None
        self.cursor_pos = None
        self.selected_gem = None
        self.moves_left = None
        self.score = None
        self.game_over = None
        self.pending_reward = 0
        self.terminal_reward_given = False
        self.last_space_pressed = False
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game_over = False
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.selected_gem = None
        self.pending_reward = 0
        self.terminal_reward_given = False
        self.last_space_pressed = False
        
        self._initialize_grid()
        
        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        # Ensure no matches on start by repeatedly resolving them
        while self._find_and_remove_matches():
            self._apply_gravity_and_refill()

    def _find_and_remove_matches(self, award_points=False):
        matches = self._find_matches()
        if not matches:
            return False
        
        if award_points:
            num_matched = len(matches)
            self.score += num_matched
            self.pending_reward += num_matched
            # Placeholder for particle effects
            # self.create_particles_at(matches)

        for x, y in matches:
            self.grid[x, y] = 0 # Mark as empty
        return True

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_slots = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[x, y + empty_slots] = self.grid[x, y]
                    self.grid[x, y] = 0
            # Refill top rows with new gems
            for y in range(empty_slots):
                self.grid[x, y] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_val, shift_val = action
        space_pressed = space_val == 1

        self.pending_reward = 0
        
        # --- Handle Input ---
        # 1. Cursor Movement
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            self.cursor_pos = (
                max(0, min(self.GRID_WIDTH - 1, self.cursor_pos[0] + dx)),
                max(0, min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + dy))
            )
        
        # 2. Selection/Swap (on press, not hold, to create discrete events)
        is_press_event = space_pressed and not self.last_space_pressed
        if is_press_event:
            cx, cy = self.cursor_pos
            if self.selected_gem is None:
                self.selected_gem = (cx, cy)
                # Sound: select_gem.wav
            else:
                sx, sy = self.selected_gem
                # Deselect if clicking the same gem
                if (cx, cy) == (sx, sy):
                    self.selected_gem = None
                    # Sound: deselect.wav
                # Check for adjacent swap
                elif abs(cx - sx) + abs(cy - sy) == 1:
                    self._attempt_swap((sx, sy), (cx, cy))
                    self.selected_gem = None
                # Select a new gem if not adjacent
                else:
                    self.selected_gem = (cx, cy)
                    # Sound: select_gem.wav

        self.last_space_pressed = space_pressed

        # --- Check Termination ---
        terminated = self.moves_left <= 0 or self.score >= self.WIN_SCORE
        if terminated and not self.terminal_reward_given:
            if self.score >= self.WIN_SCORE:
                self.pending_reward += 100 # Win bonus
                # Sound: win_game.wav
            else:
                self.pending_reward += -10 # Loss penalty
                # Sound: lose_game.wav
            self.terminal_reward_given = True
            self.game_over = True

        return (
            self._get_observation(),
            self.pending_reward,
            self.game_over,
            False,
            self._get_info()
        )
        
    def _attempt_swap(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Perform swap
        self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]
        
        matches1 = self._find_matches_at_pos(x1, y1)
        matches2 = self._find_matches_at_pos(x2, y2)
        all_matches = matches1.union(matches2)

        if not all_matches:
            # Invalid move, swap back, no move cost
            self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]
            # Sound: invalid_swap.wav
        else:
            # Valid move, consume a move
            self.moves_left -= 1
            # Sound: gem_swap.wav
            
            # Process chain reactions
            chain = 1
            while self._find_and_remove_matches(award_points=True):
                # Sound: match_{chain}.wav (e.g., match_1, match_2 for combos)
                self._apply_gravity_and_refill()
                chain += 1

    def _find_matches(self):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == 0: continue
                # Horizontal check for 3+
                if x < self.GRID_WIDTH - 2 and self.grid[x, y] == self.grid[x+1, y] == self.grid[x+2, y]:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
                    # Check for 4+
                    if x < self.GRID_WIDTH - 3 and self.grid[x,y] == self.grid[x+3, y]: matches.add((x+3, y))
                    if x < self.GRID_WIDTH - 4 and self.grid[x,y] == self.grid[x+4, y]: matches.add((x+4, y))
                # Vertical check for 3+
                if y < self.GRID_HEIGHT - 2 and self.grid[x, y] == self.grid[x, y+1] == self.grid[x, y+2]:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
                    # Check for 4+
                    if y < self.GRID_HEIGHT - 3 and self.grid[x,y] == self.grid[x, y+3]: matches.add((x, y+3))
                    if y < self.GRID_HEIGHT - 4 and self.grid[x,y] == self.grid[x, y+4]: matches.add((x, y+4))
        return matches

    def _find_matches_at_pos(self, x, y):
        matches = set()
        gem_type = self.grid[x, y]
        if gem_type == 0: return matches

        # Horizontal check
        line = []
        for i in range(self.GRID_WIDTH):
            if self.grid[i, y] == gem_type: line.append((i, y))
            else:
                if len(line) >= 3: matches.update(line)
                line = []
        if len(line) >= 3: matches.update(line)

        # Vertical check
        line = []
        for j in range(self.GRID_HEIGHT):
            if self.grid[x, j] == gem_type: line.append((x, j))
            else:
                if len(line) >= 3: matches.update(line)
                line = []
        if len(line) >= 3: matches.update(line)

        return matches

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
            "selected_gem": self.selected_gem,
            "is_game_over": self.game_over,
        }

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * self.TILE_WIDTH // 2
        screen_y = self.origin_y + (x + y) * self.TILE_HEIGHT // 2
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Draw grid lines
        for y in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # Draw gems
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[x, y]
                if gem_type > 0:
                    self._draw_gem(x, y, gem_type)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_poly = [
            self._iso_to_screen(cx, cy), self._iso_to_screen(cx + 1, cy),
            self._iso_to_screen(cx + 1, cy + 1), self._iso_to_screen(cx, cy + 1),
        ]
        pygame.gfxdraw.aapolygon(self.screen, cursor_poly, self.COLOR_CURSOR)
        
        # Draw selection highlight
        if self.selected_gem is not None:
            sx, sy = self.selected_gem
            selection_poly = [
                self._iso_to_screen(sx, sy), self._iso_to_screen(sx + 1, sy),
                self._iso_to_screen(sx + 1, sy + 1), self._iso_to_screen(sx, sy + 1),
            ]
            for i in range(1, 3): # Draw multiple lines for thickness
                pygame.gfxdraw.aapolygon(self.screen, [(p[0], p[1]-i) for p in selection_poly], self.COLOR_SELECTED)
                pygame.gfxdraw.aapolygon(self.screen, [(p[0], p[1]+i) for p in selection_poly], self.COLOR_SELECTED)
                pygame.gfxdraw.aapolygon(self.screen, [(p[0]-i, p[1]) for p in selection_poly], self.COLOR_SELECTED)
                pygame.gfxdraw.aapolygon(self.screen, [(p[0]+i, p[1]) for p in selection_poly], self.COLOR_SELECTED)

    def _draw_gem(self, x, y, gem_type):
        center_x, center_y = self._iso_to_screen(x + 0.5, y + 0.5)
        color = self.GEM_COLORS[gem_type - 1]
        
        poly = [
            (center_x, center_y - self.TILE_HEIGHT // 2),
            (center_x + self.TILE_WIDTH // 2, center_y),
            (center_x, center_y + self.TILE_HEIGHT // 2),
            (center_x - self.TILE_WIDTH // 2, center_y),
        ]
        
        border_color = (max(0, c-50) for c in color)
        
        pygame.gfxdraw.filled_polygon(self.screen, poly, color)
        pygame.gfxdraw.aapolygon(self.screen, poly, tuple(border_color))
        
        highlight_color = tuple(min(255, c+80) for c in color)
        highlight_poly = [
            (poly[0][0], poly[0][1] + 2),
            (poly[1][0] - 5, poly[1][1] - 2),
            (poly[3][0] + 5, poly[3][1] - 2),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, highlight_poly, highlight_color)
        pygame.gfxdraw.aapolygon(self.screen, highlight_poly, highlight_color)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))

        if self.game_over:
            msg, color = ("You Win!", (100, 255, 100)) if self.score >= self.WIN_SCORE else ("Game Over", (255, 100, 100))
            
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            bg_rect = msg_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(msg_surf, msg_rect)

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    pygame.display.set_caption("Gemstone Grid")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Track held keys for single press events
    key_last_state = pygame.key.get_pressed()
    
    while running:
        # --- Action Mapping for Human Play ---
        movement = 0 # none
        space_pressed = 0
        
        current_keys = pygame.key.get_pressed()
        
        # Check for key press events, not just held keys
        if current_keys[pygame.K_UP] and not key_last_state[pygame.K_UP]: movement = 1
        elif current_keys[pygame.K_DOWN] and not key_last_state[pygame.K_DOWN]: movement = 2
        elif current_keys[pygame.K_LEFT] and not key_last_state[pygame.K_LEFT]: movement = 3
        elif current_keys[pygame.K_RIGHT] and not key_last_state[pygame.K_RIGHT]: movement = 4
        
        if current_keys[pygame.K_SPACE]: space_pressed = 1
        
        action = [movement, space_pressed, 0] # shift is unused
        
        key_last_state = current_keys
        
        # --- Step the environment ---
        # Since auto_advance is False, we only step on an action.
        # We step every frame to ensure cursor movement is responsive.
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if reward != 0:
            print(f"Reward: {reward:.2f}, Total: {total_reward:.2f}, Moves left: {info['moves_left']}")

        if terminated:
            print(f"Episode Finished. Final Score: {info['score']}. Press 'R' to restart.")

        # --- Handle Pygame Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("--- RESETTING ---")
                    obs, info = env.reset()
                    total_reward = 0
                elif event.key == pygame.K_q:
                    running = False

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    pygame.quit()