
# Generated: 2025-08-27T16:07:05.949821
# Source Brief: brief_01126.md
# Brief Index: 1126

        
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
        "Controls: Arrow keys to move cursor. Press space to select a fruit, "
        "move to an adjacent fruit, and press space again to swap."
    )

    game_description = (
        "A classic match-3 puzzle game. Swap adjacent fruits to create lines of 3 or more. "
        "Clear fruits to score points and reach the target before you run out of moves!"
    )

    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 10
    CELL_SIZE = 36
    GRID_OFFSET_X = (WIDTH - GRID_SIZE * CELL_SIZE) // 2
    GRID_OFFSET_Y = (HEIGHT - GRID_SIZE * CELL_SIZE) // 2
    FRUIT_TYPES = 3
    
    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (45, 55, 65)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECT_GLOW = (255, 255, 255, 100)
    
    FRUIT_COLORS = {
        1: (220, 50, 50),   # Apple (Red)
        2: (50, 220, 50),   # Lime (Green)
        3: (80, 80, 255),   # Blueberry (Blue)
    }

    # --- Game settings ---
    INITIAL_MOVES = 50
    TARGET_SCORE = 1000
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 28, bold=True)
        
        # State variables initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        
        # Visual effect state
        self.effects = [] # For match flashes, invalid swap shakes, etc.

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.INITIAL_MOVES
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_pos = None
        self.prev_space_held = False
        self.effects = []
        
        self._initialize_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.effects = [] # Clear effects from the previous step
        reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
        
        # 2. Handle selection and swap logic
        if space_pressed:
            # SFX: Play select sound
            cursor_tuple = tuple(self.cursor_pos)
            if self.selected_pos is None:
                self.selected_pos = cursor_tuple
            else:
                if self._are_adjacent(self.selected_pos, cursor_tuple):
                    # This is a move attempt
                    self.moves_left -= 1
                    
                    self._swap_fruits(self.selected_pos, cursor_tuple)
                    
                    total_cleared, cascade_reward = self._handle_matches()
                    
                    if total_cleared > 0:
                        # Successful swap
                        self.score += total_cleared
                        reward += cascade_reward
                        # SFX: Play match/combo sound
                    else:
                        # Failed swap, swap back
                        self._swap_fruits(self.selected_pos, cursor_tuple)
                        reward = -0.2
                        # SFX: Play invalid move sound
                        self.effects.append(("shake", [self.selected_pos, cursor_tuple]))

                    # After any swap attempt, deselect
                    self.selected_pos = None

                    # Anti-softlock check
                    if not self._find_possible_moves():
                        self._reshuffle_board()
                        # SFX: Play reshuffle sound
                        self.effects.append(("reshuffle",))

                elif self.selected_pos == cursor_tuple:
                    # Deselect if clicking the same fruit
                    self.selected_pos = None
                else:
                    # Select a new fruit if not adjacent
                    self.selected_pos = cursor_tuple

        self.prev_space_held = space_held

        # 3. Check for termination
        terminated = False
        if self.score >= self.TARGET_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
            self.effects.append(("win",))
        elif self.moves_left <= 0:
            reward -= 10
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _initialize_board(self):
        self.grid = self.np_random.integers(1, self.FRUIT_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        # Ensure no initial matches and at least one move is possible
        while self._handle_matches()[0] > 0 or not self._find_possible_moves():
            if self._handle_matches()[0] > 0: # Clear initial matches
                continue # The loop will re-check
            if not self._find_possible_moves(): # Reshuffle if stuck
                self._reshuffle_board(initial=True)

    def _reshuffle_board(self, initial=False):
        if not initial:
            # Small penalty for auto-reshuffle
            self.score = max(0, self.score - 10)
            
        valid_board = False
        while not valid_board:
            flat_grid = self.grid.flatten()
            self.np_random.shuffle(flat_grid)
            self.grid = flat_grid.reshape((self.GRID_SIZE, self.GRID_SIZE))
            
            # A valid board has no initial matches and at least one possible move
            if self._check_for_matches() is None and self._find_possible_moves():
                valid_board = True

    def _handle_matches(self):
        total_cleared = 0
        total_reward = 0
        while True:
            to_clear = self._check_for_matches()
            if to_clear:
                num_cleared = len(to_clear)
                total_cleared += num_cleared
                total_reward += num_cleared  # +1 per fruit
                if num_cleared > 3:
                    total_reward += 5  # Combo bonus
                
                for x, y in to_clear:
                    fruit_type = self.grid[x, y]
                    self.grid[x, y] = 0
                    self.effects.append(("flash", (x, y), self.FRUIT_COLORS[fruit_type]))
                
                self._apply_gravity()
                self._fill_new_fruits()
            else:
                break
        return total_cleared, total_reward

    def _check_for_matches(self):
        to_clear = set()
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[x, y] == 0: continue
                
                # Check horizontal
                if x < self.GRID_SIZE - 2 and self.grid[x, y] == self.grid[x+1, y] == self.grid[x+2, y]:
                    to_clear.update([(x, y), (x+1, y), (x+2, y)])
                # Check vertical
                if y < self.GRID_SIZE - 2 and self.grid[x, y] == self.grid[x, y+1] == self.grid[x, y+2]:
                    to_clear.update([(x, y), (x, y+1), (x, y+2)])
        return to_clear if to_clear else None

    def _apply_gravity(self):
        for x in range(self.GRID_SIZE):
            empty_y = self.GRID_SIZE - 1
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[x, y] != 0:
                    self._swap_fruits((x, y), (x, empty_y))
                    empty_y -= 1

    def _fill_new_fruits(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[x, y] == 0:
                    self.grid[x, y] = self.np_random.integers(1, self.FRUIT_TYPES + 1)

    def _find_possible_moves(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                # Check swap right
                if x < self.GRID_SIZE - 1:
                    self._swap_fruits((x, y), (x+1, y))
                    if self._check_for_matches():
                        self._swap_fruits((x, y), (x+1, y)) # Swap back
                        return True
                    self._swap_fruits((x, y), (x+1, y)) # Swap back
                # Check swap down
                if y < self.GRID_SIZE - 1:
                    self._swap_fruits((x, y), (x, y+1))
                    if self._check_for_matches():
                        self._swap_fruits((x, y), (x, y+1)) # Swap back
                        return True
                    self._swap_fruits((x, y), (x, y+1)) # Swap back
        return False

    def _swap_fruits(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]

    def _are_adjacent(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) + abs(y1 - y2) == 1
    
    def _pos_to_pixels(self, x, y):
        return (
            self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2,
        )

    def _render_game(self):
        # Draw grid background
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = (self.GRID_OFFSET_X + x * self.CELL_SIZE, 
                        self.GRID_OFFSET_Y + y * self.CELL_SIZE, 
                        self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Handle one-frame visual effects
        shake_offsets = {}
        for effect in self.effects:
            if effect[0] == "shake":
                for pos in effect[1]:
                    shake_offsets[pos] = (self.np_random.integers(-2, 3), self.np_random.integers(-2, 3))
            elif effect[0] == "flash":
                _, (x, y), color = effect
                px, py = self._pos_to_pixels(x, y)
                self._draw_starburst(px, py, color)

        # Draw fruits
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                fruit_type = self.grid[x, y]
                if fruit_type == 0: continue
                
                px, py = self._pos_to_pixels(x, y)
                
                # Apply shake effect if any
                if (x, y) in shake_offsets:
                    off_x, off_y = shake_offsets[(x, y)]
                    px += off_x
                    py += off_y

                is_selected = self.selected_pos == (x, y)
                self._draw_fruit(fruit_type, px, py, is_selected)

        # Draw cursor
        cursor_rect = (self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE,
                       self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE,
                       self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _draw_fruit(self, fruit_type, px, py, is_selected):
        color = self.FRUIT_COLORS[fruit_type]
        radius = self.CELL_SIZE // 2 - 5
        
        if is_selected:
            # Draw a pulsating glow effect for the selected fruit
            glow_radius = radius + 4 + int(2 * math.sin(pygame.time.get_ticks() * 0.01))
            glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_SELECT_GLOW, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (px - glow_radius, py - glow_radius))
            radius += 1 # Make selected fruit slightly larger

        if fruit_type == 1: # Apple (Circle)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)
        elif fruit_type == 2: # Lime (Square)
            size = radius * 1.6
            rect = pygame.Rect(px - size/2, py - size/2, size, size)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
        elif fruit_type == 3: # Blueberry (Triangle)
            points = [
                (px, py - radius),
                (px - radius, py + radius * 0.7),
                (px + radius, py + radius * 0.7)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_starburst(self, px, py, color, num_rays=8, length=25):
        for i in range(num_rays):
            angle = i * (2 * math.pi / num_rays)
            start_pos = (px + 5 * math.cos(angle), py + 5 * math.sin(angle))
            end_pos = (px + length * math.cos(angle), py + length * math.sin(angle))
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Moves display
        moves_text = self.font_large.render(f"MOVES: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 15, 10))
        
        # Game over text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            
            if self.score >= self.TARGET_SCORE:
                end_text_str = "YOU WIN!"
                end_color = (100, 255, 100)
            else:
                end_text_str = "GAME OVER"
                end_color = (255, 100, 100)
                
            end_text = self.font_large.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            overlay.blit(end_text, text_rect)
            self.screen.blit(overlay, (0, 0))

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

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Override screen to be a display window
    env.screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Match-3 Game")
    
    terminated = False
    
    # Game loop
    while not terminated:
        movement = 0 # no-op
        space_held = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = True
            
        action = [movement, 1 if space_held else 0, 0] # shift is unused
        
        # Only step if an action is taken or space is involved
        if movement != 0 or space_held or env.prev_space_held:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

        # Render the current state to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit FPS for manual play

    pygame.quit()