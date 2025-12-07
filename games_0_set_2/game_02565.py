import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import os
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame


# Set Pygame to run in a headless mode, suitable for server environments
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move selector. Space to select a fruit group to match."
    )

    game_description = (
        "Match cascading fruits in a grid to reach a target score before running out of moves. "
        "Select groups of 3 or more adjacent fruits of the same type to score points."
    )

    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_AREA_WIDTH = 400
    CELL_SIZE = GRID_AREA_WIDTH // GRID_WIDTH
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_AREA_WIDTH) // 2

    NUM_FRUIT_TYPES = 5
    MIN_MATCH_SIZE = 3
    TARGET_SCORE = 1000
    STARTING_MOVES = 20
    
    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (30, 45, 60)
    COLOR_GRID_LINES = (50, 70, 90)
    
    FRUIT_COLORS = {
        1: (255, 80, 80),   # Red (Cherry)
        2: (80, 255, 80),   # Green (Apple)
        3: (80, 150, 255),  # Blue (Blueberry)
        4: (255, 200, 80),  # Orange (Orange)
        5: (200, 80, 255),  # Purple (Grape)
    }
    
    COLOR_SELECTOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 36)
        
        # --- State Variables ---
        self.grid = None
        self.selector_pos = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.steps = None
        self.effects = [] # For rendering transient effects like pops

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.STARTING_MOVES
        self.game_over = False
        self.steps = 0
        self.selector_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.effects = []
        
        self._generate_valid_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.effects = [] # Clear effects from previous step
        reward = 0
        
        movement, space_press, _ = action
        
        # Handle selector movement
        if movement > 0:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            self.selector_pos[0] = (self.selector_pos[0] + dx) % self.GRID_WIDTH
            self.selector_pos[1] = (self.selector_pos[1] + dy) % self.GRID_HEIGHT
        
        # Handle selection
        if space_press == 1:
            self.moves_left -= 1
            reward += self._process_selection()
        
        # Check for termination conditions
        terminated = self.score >= self.TARGET_SCORE or self.moves_left <= 0
        if terminated:
            self.game_over = True
            if self.score >= self.TARGET_SCORE:
                reward += 100 # Win bonus
            else:
                reward -= 10 # Loss penalty

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_selection(self):
        """Processes a selection action at the current selector position."""
        sel_x, sel_y = self.selector_pos
        fruit_type = self.grid[sel_y, sel_x]
        
        if fruit_type == 0:
            return -0.1 # Penalty for selecting empty space

        connected_fruits = self._find_connected_group(sel_x, sel_y)
        
        if len(connected_fruits) < self.MIN_MATCH_SIZE:
            # Invalid move
            self.effects.append({'type': 'fail', 'pos': (sel_x, sel_y)})
            return -0.1

        # Valid match
        total_reward = 0
        
        # --- Chain reaction loop ---
        chain_level = 0
        while len(connected_fruits) >= self.MIN_MATCH_SIZE:
            chain_level += 1
            num_matched = len(connected_fruits)
            
            # Calculate reward for this part of the chain
            reward_this_chain = num_matched
            if num_matched >= 4:
                reward_this_chain += 5 # Bonus for 4+ match
            
            self.score += int(reward_this_chain * (1 + 0.5 * (chain_level - 1))) # Chain bonus
            total_reward += reward_this_chain

            # Add visual effects
            for (fx, fy) in connected_fruits:
                self.effects.append({'type': 'pop', 'pos': (fx, fy), 'color': self.FRUIT_COLORS[self.grid[fy, fx]]})
            
            # Remove fruits from grid
            for (fx, fy) in connected_fruits:
                self.grid[fy, fx] = 0
            
            # Cascade and refill
            self._apply_gravity()
            self._refill_grid()
            
            # Find next matches for potential chain reaction
            all_new_matches = self._find_all_matches()
            if all_new_matches:
                connected_fruits = all_new_matches[0] # Process the first found chain
            else:
                break # No more chains

        # After all cascades, ensure the board is not stuck
        self._ensure_moves_exist()

        return total_reward

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
        }

    # --- Game Logic Helpers ---
    
    def _generate_valid_grid(self):
        """Generates a grid and ensures it has at least one valid move."""
        # This loop is to ensure the generated board is playable.
        # It's statistically very unlikely to fail, but this is a safeguard.
        for _ in range(100): # Try up to 100 times to generate a board with moves
            self.grid = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if self._find_all_matches():
                return # Found a board with at least one match, we're good.
                
        # Fallback: If 100 random boards have no moves, manually create one.
        # This guarantees termination and a playable board.
        self.grid = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        fruit_type = self.grid[0, 2] # Pick a fruit type from the generated board
        self.grid[0, 0] = fruit_type
        self.grid[0, 1] = fruit_type
        self.grid[1, 0] = fruit_type

    def _ensure_moves_exist(self):
        """If no moves are possible on the board, it is regenerated."""
        if not self._find_all_matches():
            # The board is stuck, regenerate it completely.
            self._generate_valid_grid()

    def _find_connected_group(self, start_x, start_y):
        """Finds all connected fruits of the same type using BFS."""
        if self.grid[start_y, start_x] == 0:
            return set()
            
        target_type = self.grid[start_y, start_x]
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        
        while q:
            x, y = q.popleft()
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[ny, nx] == target_type:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return visited

    def _find_all_matches(self):
        """Finds all match groups of 3 or more on the grid."""
        matches = []
        checked = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y,x] > 0 and (x, y) not in checked:
                    group = self._find_connected_group(x, y)
                    if len(group) >= self.MIN_MATCH_SIZE:
                        matches.append(group)
                    checked.update(group)
        return matches

    def _clear_all_matches(self):
        """Removes all matched fruits from the grid."""
        all_matches = self._find_all_matches()
        for group in all_matches:
            for x, y in group:
                self.grid[y, x] = 0

    def _apply_gravity(self):
        """Makes fruits fall down into empty spaces."""
        for x in range(self.GRID_WIDTH):
            col = self.grid[:, x]
            non_zeros = col[col != 0]
            new_col = np.zeros(self.GRID_HEIGHT, dtype=int)
            new_col[self.GRID_HEIGHT - len(non_zeros):] = non_zeros
            self.grid[:, x] = new_col
            
    def _refill_grid(self):
        """Fills empty spaces at the top with new random fruits."""
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] == 0:
                    self.grid[y, x] = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1)

    # --- Rendering Helpers ---

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_AREA_WIDTH, self.GRID_AREA_WIDTH)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw fruits
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                fruit_type = self.grid[y, x]
                if fruit_type > 0:
                    self._draw_fruit(x, y, fruit_type)
        
        # Draw effects
        for effect in self.effects:
            if effect['type'] == 'pop':
                self._draw_pop_effect(effect['pos'], effect['color'])
            elif effect['type'] == 'fail':
                self._draw_fail_effect(effect['pos'])
                
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x_pos = self.GRID_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x_pos, self.GRID_OFFSET_Y), (x_pos, self.GRID_OFFSET_Y + self.GRID_AREA_WIDTH))
        for i in range(self.GRID_HEIGHT + 1):
            y_pos = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_OFFSET_X, y_pos), (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, y_pos))

        # Draw selector
        self._draw_selector()

    def _draw_fruit(self, grid_x, grid_y, fruit_type):
        cx = self.GRID_OFFSET_X + int((grid_x + 0.5) * self.CELL_SIZE)
        cy = self.GRID_OFFSET_Y + int((grid_y + 0.5) * self.CELL_SIZE)
        radius = int(self.CELL_SIZE * 0.4)
        color = self.FRUIT_COLORS[fruit_type]
        
        # Draw a slightly darker version for depth
        dark_color = tuple(max(0, c - 50) for c in color)
        pygame.draw.circle(self.screen, dark_color, (cx, cy + 2), radius)
        
        # Main fruit body
        pygame.draw.circle(self.screen, color, (cx, cy), radius)
        
        # Highlight for 3D effect
        highlight_color = tuple(min(255, c + 80) for c in color)
        pygame.gfxdraw.aacircle(self.screen, cx - radius // 3, cy - radius // 3, radius // 4, highlight_color)
        pygame.gfxdraw.filled_circle(self.screen, cx - radius // 3, cy - radius // 3, radius // 4, highlight_color)

    def _draw_selector(self):
        sel_x, sel_y = self.selector_pos
        
        # Highlight the entire potential match group
        if self.grid[sel_y, sel_x] > 0:
            group = self._find_connected_group(sel_x, sel_y)
            if len(group) >= self.MIN_MATCH_SIZE:
                for gx, gy in group:
                    rect = pygame.Rect(
                        self.GRID_OFFSET_X + gx * self.CELL_SIZE,
                        self.GRID_OFFSET_Y + gy * self.CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                    # Create a semi-transparent surface for the highlight
                    highlight_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    highlight_surface.fill((255, 255, 0, 80))
                    self.screen.blit(highlight_surface, rect.topleft)

        # Draw the main selector box
        rect = pygame.Rect(
            self.GRID_OFFSET_X + sel_x * self.CELL_SIZE,
            self.GRID_OFFSET_Y + sel_y * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        # Pulsing effect for thickness
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        thickness = 2 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, thickness)

    def _draw_pop_effect(self, pos, color):
        """Draws a static starburst effect for a matched fruit."""
        cx = self.GRID_OFFSET_X + int((pos[0] + 0.5) * self.CELL_SIZE)
        cy = self.GRID_OFFSET_Y + int((pos[1] + 0.5) * self.CELL_SIZE)
        
        for i in range(8):
            angle = i * (math.pi / 4)
            length1 = self.CELL_SIZE * 0.4
            length2 = self.CELL_SIZE * 0.6
            start_pos = (cx + length1 * math.cos(angle), cy + length1 * math.sin(angle))
            end_pos = (cx + length2 * math.cos(angle), cy + length2 * math.sin(angle))
            pygame.draw.line(self.screen, color, start_pos, end_pos, 3)

    def _draw_fail_effect(self, pos):
        """Draws a red 'X' for a failed match attempt."""
        cx = self.GRID_OFFSET_X + int((pos[0] + 0.5) * self.CELL_SIZE)
        cy = self.GRID_OFFSET_Y + int((pos[1] + 0.5) * self.CELL_SIZE)
        size = self.CELL_SIZE * 0.3
        color = (255, 50, 50)
        pygame.draw.line(self.screen, color, (cx - size, cy - size), (cx + size, cy + size), 4)
        pygame.draw.line(self.screen, color, (cx - size, cy + size), (cx + size, cy - size), 4)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Moves
        moves_text = self.font_medium.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.TARGET_SCORE:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly.
    # Note: This will fail if SDL_VIDEODRIVER is set to "dummy". 
    # To run this, you might need to comment out the os.environ line at the top of the file.
    # pip install gymnasium[pygame]
    
    # To run in interactive mode, we must unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode='rgb_array')
    env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Matcher")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space = 1
                elif event.key == pygame.K_r: # Reset
                    env.reset()
                elif event.key == pygame.K_q:
                    running = False
        
        action = [movement, space, 0] # Shift is not used
        
        # Only step if an action is taken
        if any(action):
            obs, reward, terminated, _, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}, Terminated: {terminated}")
        else:
            obs = env._get_observation()

        # Render to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()