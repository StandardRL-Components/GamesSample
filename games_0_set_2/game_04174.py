
# Generated: 2025-08-28T01:37:20.364629
# Source Brief: brief_04174.md
# Brief Index: 4174

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a tile, "
        "then move to an adjacent tile and press Space again to swap."
    )

    game_description = (
        "Swap adjacent tiles to create matches of three or more. Clear the board to win, "
        "but the game ends if no more moves are possible."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 10, 8
    TILE_SIZE = 40
    TILE_BORDER_RADIUS = 6
    GRID_WIDTH = GRID_COLS * TILE_SIZE
    GRID_HEIGHT = GRID_ROWS * TILE_SIZE
    X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH) // 2
    Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT) // 2
    NUM_COLORS = 5
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (25, 30, 35)
    COLOR_GRID_BG = (40, 45, 55)
    TILE_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 255, 80),   # Yellow
        (200, 80, 255),   # Purple
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED = (255, 255, 0)
    COLOR_HINT = (255, 255, 255, 50) # RGBA
    COLOR_TEXT = (220, 220, 220)

    # --- Rewards ---
    REWARD_MATCH_3 = 1
    REWARD_MATCH_4 = 2
    REWARD_MATCH_5_PLUS = 5
    REWARD_INVALID_SWAP = -0.1
    REWARD_WIN = 100
    REWARD_LOSE = -10

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selection_state = "IDLE" # "IDLE" or "SELECTED"
        self.first_selection = None
        self.particles = []
        self.possible_moves = []
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selection_state = "IDLE"
        self.first_selection = None
        self.particles = []
        
        self._generate_initial_board()
        self.possible_moves = self._find_possible_moves()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        # --- Handle Input ---
        # Shift cancels selection
        if shift_pressed and self.selection_state == "SELECTED":
            self.selection_state = "IDLE"
            self.first_selection = None
            # sound: selection cancel

        # Movement moves cursor
        if self.selection_state == "IDLE" and movement != 0:
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
            # sound: cursor move

        # Space handles selection/swapping
        if space_pressed:
            if self.selection_state == "IDLE":
                self.selection_state = "SELECTED"
                self.first_selection = list(self.cursor_pos)
                # sound: select tile
            else: # self.selection_state == "SELECTED"
                is_adjacent = abs(self.cursor_pos[0] - self.first_selection[0]) + \
                              abs(self.cursor_pos[1] - self.first_selection[1]) == 1
                if is_adjacent:
                    # --- Perform Swap and Process Consequences ---
                    self.steps += 1
                    reward, terminated = self._handle_swap(self.first_selection, self.cursor_pos)
                
                self.selection_state = "IDLE"
                self.first_selection = None
        
        # Check step limit
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_swap(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2

        # Swap tiles in grid
        self.grid[y1][x1], self.grid[y2][x2] = self.grid[y2][x2], self.grid[y1][x1]
        # sound: swap

        total_cascade_reward = 0
        is_first_match = True

        while True:
            matches = self._find_matches()
            if not matches:
                break

            # Calculate reward for this cascade step
            match_counts = {}
            for x, y in matches:
                match_counts.setdefault(id(matches), []).append((x,y))

            for match_set in match_counts.values():
                count = len(match_set)
                if count == 3: total_cascade_reward += self.REWARD_MATCH_3
                elif count == 4: total_cascade_reward += self.REWARD_MATCH_4
                else: total_cascade_reward += self.REWARD_MATCH_5_PLUS

            self.score += total_cascade_reward if is_first_match else 0 # Score only on first part of cascade
            is_first_match = False

            # Visual/Audio Feedback
            self._spawn_particles(matches)
            # sound: match clear

            # Clear and refill
            for x, y in matches:
                self.grid[y][x] = -1 # Mark for removal
            self._apply_gravity_and_refill()

        # If no matches were made by the swap, it's an invalid move
        if total_cascade_reward == 0:
            # Swap back
            self.grid[y1][x1], self.grid[y2][x2] = self.grid[y2][x2], self.grid[y1][x1]
            # sound: invalid swap
            return self.REWARD_INVALID_SWAP, False
        
        # After a successful cascade, check for game end states
        self.possible_moves = self._find_possible_moves()
        
        # Check for win (board clear)
        if all(tile == -1 for row in self.grid for tile in row):
            self.game_over = True
            return self.REWARD_WIN + total_cascade_reward, True

        # Check for loss (no more moves)
        if not self.possible_moves:
            self.game_over = True
            return self.REWARD_LOSE + total_cascade_reward, True

        return total_cascade_reward, False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Game Logic Helpers ---
    def _generate_initial_board(self):
        self.grid = [[self.np_random.integers(0, self.NUM_COLORS) for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        # Ensure no initial matches and at least one possible move
        while self._find_matches() or not self._find_possible_moves():
            self.grid = [[self.np_random.integers(0, self.NUM_COLORS) for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]

    def _find_matches(self):
        matches = set()
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS - 2):
                if self.grid[y][x] == self.grid[y][x+1] == self.grid[y][x+2] != -1:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS - 2):
                if self.grid[y][x] == self.grid[y+1][x] == self.grid[y+2][x] != -1:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return matches

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_COLS):
            empty_slots = []
            for y in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[y][x] == -1:
                    empty_slots.append(y)
                elif empty_slots:
                    new_y = empty_slots.pop(0)
                    self.grid[new_y][x] = self.grid[y][x]
                    self.grid[y][x] = -1
                    empty_slots.append(y)
            # Refill
            for y in empty_slots:
                self.grid[y][x] = self.np_random.integers(0, self.NUM_COLORS)

    def _find_possible_moves(self):
        possible_moves = []
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                # Test swap right
                if x < self.GRID_COLS - 1:
                    self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x]
                    if self._find_matches():
                        possible_moves.append(((x, y), (x+1, y)))
                    self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x] # Swap back
                # Test swap down
                if y < self.GRID_ROWS - 1:
                    self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x]
                    if self._find_matches():
                        possible_moves.append(((x, y), (x, y+1)))
                    self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x] # Swap back
        return possible_moves

    # --- Rendering Helpers ---
    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.X_OFFSET, self.Y_OFFSET, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=10)

        # Draw hint glows
        hint_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        for move in self.possible_moves:
            for x, y in move:
                hint_surface.fill((0,0,0,0))
                pygame.draw.rect(hint_surface, self.COLOR_HINT, (0, 0, self.TILE_SIZE, self.TILE_SIZE), border_radius=self.TILE_BORDER_RADIUS)
                self.screen.blit(hint_surface, (self.X_OFFSET + x * self.TILE_SIZE, self.Y_OFFSET + y * self.TILE_SIZE))

        # Draw tiles
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                tile_val = self.grid[y][x]
                if tile_val != -1:
                    color = self.TILE_COLORS[tile_val]
                    rect = pygame.Rect(self.X_OFFSET + x * self.TILE_SIZE + 2,
                                       self.Y_OFFSET + y * self.TILE_SIZE + 2,
                                       self.TILE_SIZE - 4, self.TILE_SIZE - 4)
                    pygame.draw.rect(self.screen, color, rect, border_radius=self.TILE_BORDER_RADIUS)

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(self.X_OFFSET + cursor_x * self.TILE_SIZE,
                                  self.Y_OFFSET + cursor_y * self.TILE_SIZE,
                                  self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=self.TILE_BORDER_RADIUS)

        # Draw selection
        if self.selection_state == "SELECTED":
            sel_x, sel_y = self.first_selection
            sel_rect = pygame.Rect(self.X_OFFSET + sel_x * self.TILE_SIZE,
                                   self.Y_OFFSET + sel_y * self.TILE_SIZE,
                                   self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, sel_rect, 4, border_radius=self.TILE_BORDER_RADIUS + 1)
        
        self._update_and_draw_particles()

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 15))
        
        # Steps
        steps_text = self.font_main.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 15, 15))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            is_win = all(tile == -1 for row in self.grid for tile in row)
            msg = "BOARD CLEARED!" if is_win else "NO MORE MOVES"
            
            end_text = self.font_large.render(msg, True, self.COLOR_SELECTED)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _spawn_particles(self, locations):
        for x, y in locations:
            tile_color = self.TILE_COLORS[self.grid[y][x]]
            center_x = self.X_OFFSET + x * self.TILE_SIZE + self.TILE_SIZE / 2
            center_y = self.Y_OFFSET + y * self.TILE_SIZE + self.TILE_SIZE / 2
            for _ in range(15): # Number of particles per tile
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                lifespan = self.np_random.integers(15, 30)
                self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': lifespan, 'color': tile_color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30))
                color = (*p['color'], alpha)
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                # Using gfxdraw for anti-aliased circles
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], max(0, int(p['life'] / 5)), color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(0, int(p['life'] / 5)), color)

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be executed when the environment is used by an RL agent
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print("--- Playing Game: Match-3 ---")
    print(env.game_description)
    print(env.user_guide)

    # Main game loop
    while not done:
        # Human input mapping
        keys = pygame.key.get_pressed()
        move_action = 0
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        
        # Process events
        should_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                should_step = True

        if should_step:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Rendering for human player
        # In a real gym usage, this would be env.render()
        render_surface = pygame.transform.scale(env.screen, (env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        if 'window' not in locals():
            pygame.display.set_caption("Match-3 Gym Environment")
            window = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        window.blit(render_surface, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we only need to tick the clock to prevent high CPU usage
        env.clock.tick(30)

    print("Game Over!")
    pygame.quit()