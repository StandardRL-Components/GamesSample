import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a tile, "
        "then move to an adjacent tile and press Shift to swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the board by swapping adjacent colored tiles to create matches of three or more. "
        "Plan your moves carefully, as you have a limited number of swaps to clear the entire grid."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_TILE_TYPES = 5
    MAX_STEPS = 1000
    INITIAL_MOVES = 30

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (50, 60, 70)
    COLOR_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 255, 0, 100)
    COLOR_SELECT_GLOW = (255, 255, 255, 120)
    TILE_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 120, 255),   # Blue
        (255, 255, 80),   # Yellow
        (200, 80, 255),   # Purple
        (255, 150, 50),   # Orange
    ]

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
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_fx = pygame.font.SysFont("Arial Black", 30)

        self.grid_area_size = self.SCREEN_HEIGHT
        self.cell_size = self.grid_area_size // self.GRID_HEIGHT
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_area_size) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_area_size) // 2
        self.tile_radius = int(self.cell_size * 0.4)

        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.last_action = np.array([0, 0, 0])
        self.effects = []
        
        # self.reset() # reset is called by the wrapper/runner, not needed here
        
        # self.validate_implementation() # Uncomment to run validation check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.moves_left = self.INITIAL_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_pos = None
        self.last_action = np.array([0, 0, 0])
        self.effects = []

        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        self._handle_input(action)
        reward += self._process_swap(action)
        
        # Process visual effects timers
        self.effects = [fx for fx in self.effects if fx.get("timer", 1) > 0]
        for fx in self.effects:
            fx["timer"] = fx.get("timer", 1) - 1

        terminated = self._check_termination()
        
        if not terminated:
            if not self._check_for_possible_matches() and np.any(self.grid > 0):
                self._reshuffle_board()
                # sound: shuffle_sound
                self.effects.append({"type": "text_popup", "text": "Reshuffling!", "pos": (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2), "timer": 60, "color": self.COLOR_CURSOR})

        if terminated and np.all(self.grid == 0): # Win condition
            reward += 100 # Main win bonus
            reward += 5   # Extra bonus
            self.effects.append({"type": "text_popup", "text": "BOARD CLEARED!", "pos": (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2), "timer": 120, "color": (100, 255, 100)})

        self.last_action = action

        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info(),
        )
    
    def _handle_input(self, action):
        movement, _, _ = action
        
        if movement != 0:
            # Debounce movement
            last_movement = self.last_action[0]
            if movement != last_movement:
                if movement == 1: self.cursor_pos[1] -= 1
                elif movement == 2: self.cursor_pos[1] += 1
                elif movement == 3: self.cursor_pos[0] -= 1
                elif movement == 4: self.cursor_pos[0] += 1

                self.cursor_pos[0] = self.cursor_pos[0] % self.GRID_WIDTH
                self.cursor_pos[1] = self.cursor_pos[1] % self.GRID_HEIGHT
                # sound: cursor_move_sound
    
    def _process_swap(self, action):
        reward = 0
        _, space_btn, shift_btn = action
        space_press = space_btn == 1 and self.last_action[1] == 0
        shift_press = shift_btn == 1 and self.last_action[2] == 0

        if space_press:
            self.selected_pos = list(self.cursor_pos)
            # sound: select_tile_sound
        
        if shift_press and self.selected_pos is not None:
            pos2 = list(self.cursor_pos)
            pos1 = self.selected_pos

            # Check for adjacency
            dx = abs(pos1[0] - pos2[0])
            dy = abs(pos1[1] - pos2[1])
            if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                self.moves_left -= 1
                
                # Perform hypothetical swap
                temp_grid = self.grid.copy()
                val1, val2 = temp_grid[pos1[0], pos1[1]], temp_grid[pos2[0], pos2[1]]
                temp_grid[pos1[0], pos1[1]], temp_grid[pos2[0], pos2[1]] = val2, val1
                
                matches1 = self._find_matches_at(temp_grid, pos1[0], pos1[1])
                matches2 = self._find_matches_at(temp_grid, pos2[0], pos2[1])

                if matches1 or matches2:
                    # Valid swap, process matches
                    # sound: valid_swap_sound
                    self.grid = temp_grid
                    combo = 0
                    while True:
                        all_matches = self._find_all_matches(self.grid)
                        if not all_matches:
                            break
                        
                        combo += 1
                        num_cleared = len(all_matches)
                        reward += num_cleared * combo # Combo bonus provides +1 per tile on 1st combo, +2 on 2nd, etc.
                        self.score += num_cleared * combo
                        
                        for x, y in all_matches:
                            # sound: tile_clear_sound
                            self.effects.append({"type": "flash", "pos": (x, y), "timer": 15})
                            self.grid[x, y] = 0
                        
                        self._apply_gravity()
                else:
                    # Invalid swap
                    # sound: invalid_swap_sound
                    reward = -0.1
                    self.effects.append({"type": "shake", "timer": 10})

            else: # Not adjacent
                reward = -0.01

            self.selected_pos = None # Clear selection after any shift press attempt
        elif shift_press and self.selected_pos is None:
            reward = -0.01 # Penalty for pressing shift with no selection
            
        return reward

    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
            # Ensure no initial matches
            while self._find_all_matches(self.grid):
                self.grid = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
            
            if self._check_for_possible_matches():
                break

    def _find_matches_at(self, grid, x, y):
        if grid[x, y] == 0:
            return set()
        
        color = grid[x, y]
        # Horizontal
        h_matches = {(x, y)}
        for i in range(x - 1, -1, -1):
            if grid[i, y] == color: h_matches.add((i, y))
            else: break
        for i in range(x + 1, self.GRID_WIDTH):
            if grid[i, y] == color: h_matches.add((i, y))
            else: break
        
        # Vertical
        v_matches = {(x, y)}
        for j in range(y - 1, -1, -1):
            if grid[x, j] == color: v_matches.add((x, j))
            else: break
        for j in range(y + 1, self.GRID_HEIGHT):
            if grid[x, j] == color: v_matches.add((x, j))
            else: break
            
        matches = set()
        if len(h_matches) >= 3: matches.update(h_matches)
        if len(v_matches) >= 3: matches.update(v_matches)
        return matches

    def _find_all_matches(self, grid):
        all_matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if grid[x, y] > 0:
                    matches = self._find_matches_at(grid, x, y)
                    if matches:
                        all_matches.update(matches)
        return all_matches

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_y = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] != 0:
                    if y != empty_y:
                        self.grid[x, empty_y] = self.grid[x, y]
                        self.grid[x, y] = 0
                    empty_y -= 1

    def _check_for_possible_matches(self):
        temp_grid = self.grid.copy()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if temp_grid[x, y] == 0: continue
                # Swap right
                if x < self.GRID_WIDTH - 1 and temp_grid[x+1, y] != 0:
                    temp_grid[x, y], temp_grid[x+1, y] = temp_grid[x+1, y], temp_grid[x, y]
                    if self._find_matches_at(temp_grid, x, y) or self._find_matches_at(temp_grid, x+1, y):
                        return True
                    temp_grid[x, y], temp_grid[x+1, y] = temp_grid[x+1, y], temp_grid[x, y] # Swap back
                # Swap down
                if y < self.GRID_HEIGHT - 1 and temp_grid[x, y+1] != 0:
                    temp_grid[x, y], temp_grid[x, y+1] = temp_grid[x, y+1], temp_grid[x, y]
                    if self._find_matches_at(temp_grid, x, y) or self._find_matches_at(temp_grid, x, y+1):
                        return True
                    temp_grid[x, y], temp_grid[x, y+1] = temp_grid[x, y+1], temp_grid[x, y] # Swap back
        return False

    def _reshuffle_board(self):
        if not np.any(self.grid > 0): return

        for _ in range(10): # Try to reshuffle up to 10 times
            tiles = self.grid[self.grid > 0].tolist()
            self.np_random.shuffle(tiles)
            
            new_grid = np.zeros_like(self.grid)
            tile_idx = 0
            for x in range(self.GRID_WIDTH):
                # Count how many tiles should be in this column
                num_in_col = np.count_nonzero(self.grid[:,x] > 0)
                for i in range(num_in_col):
                    if tile_idx < len(tiles):
                        new_grid[x, self.GRID_HEIGHT - 1 - i] = tiles[tile_idx]
                        tile_idx += 1
            
            self.grid = new_grid
            if self._check_for_possible_matches():
                return
        # Failsafe: if 10 shuffles fail, just generate a new valid board fragment
        self._generate_board()


    def _check_termination(self):
        self.game_over = (
            self.moves_left <= 0 or
            self.steps >= self.MAX_STEPS or
            np.all(self.grid == 0)
        )
        return bool(self.game_over)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Apply screen shake
        shake_offset = (0, 0)
        shake_effect = next((fx for fx in self.effects if fx["type"] == "shake"), None)
        if shake_effect:
            shake_offset = (random.randint(-5, 5), random.randint(-5, 5))

        grid_rect = pygame.Rect(self.grid_offset_x + shake_offset[0], self.grid_offset_y + shake_offset[1], self.grid_area_size, self.grid_area_size)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, 0, 10)
        
        # Draw grid lines
        for i in range(1, self.GRID_WIDTH):
            x = self.grid_offset_x + i * self.cell_size + shake_offset[0]
            pygame.draw.line(self.screen, self.COLOR_BG, (x, grid_rect.top), (x, grid_rect.bottom), 2)
        for i in range(1, self.GRID_HEIGHT):
            y = self.grid_offset_y + i * self.cell_size + shake_offset[1]
            pygame.draw.line(self.screen, self.COLOR_BG, (grid_rect.left, y), (grid_rect.right, y), 2)

        # Draw tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                tile_val = self.grid[x, y]
                if tile_val > 0:
                    center_x = int(self.grid_offset_x + (x + 0.5) * self.cell_size) + shake_offset[0]
                    center_y = int(self.grid_offset_y + (y + 0.5) * self.cell_size) + shake_offset[1]
                    color = self.TILE_COLORS[tile_val - 1]
                    
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.tile_radius, color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.tile_radius, tuple(int(c*0.8) for c in color))

        # Draw selected tile glow
        if self.selected_pos is not None:
            x, y = self.selected_pos
            center_x = int(self.grid_offset_x + (x + 0.5) * self.cell_size) + shake_offset[0]
            center_y = int(self.grid_offset_y + (y + 0.5) * self.cell_size) + shake_offset[1]
            glow_radius = int(self.tile_radius * (1.2 + 0.1 * math.sin(self.steps * 0.3)))
            
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_SELECT_GLOW, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (center_x - glow_radius, center_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        rect = pygame.Rect(
            self.grid_offset_x + cursor_x * self.cell_size + shake_offset[0],
            self.grid_offset_y + cursor_y * self.cell_size + shake_offset[1],
            self.cell_size, self.cell_size
        )
        s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), 5, 4)
        self.screen.blit(s, rect.topleft)
        
        # Draw effects
        for fx in self.effects:
            if fx['type'] == 'flash':
                x, y = fx['pos']
                center_x = int(self.grid_offset_x + (x + 0.5) * self.cell_size) + shake_offset[0]
                center_y = int(self.grid_offset_y + (y + 0.5) * self.cell_size) + shake_offset[1]
                alpha = max(0, int(255 * (fx['timer'] / 15)))
                flash_surf = pygame.Surface((self.tile_radius*2, self.tile_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(flash_surf, (255,255,255,alpha), (self.tile_radius, self.tile_radius), self.tile_radius)
                self.screen.blit(flash_surf, (center_x - self.tile_radius, center_y - self.tile_radius), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))

        # Moves Left
        moves_surf = self.font_main.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (self.SCREEN_WIDTH - moves_surf.get_width() - 15, 10))
        
        # Popup effects
        for fx in self.effects:
            if fx['type'] == 'text_popup':
                alpha = max(0, int(255 * (fx['timer'] / 60)))
                popup_surf = self.font_fx.render(fx['text'], True, fx['color'])
                popup_surf.set_alpha(alpha)
                popup_rect = popup_surf.get_rect(center=fx['pos'])
                self.screen.blit(popup_surf, popup_rect)
        
        # Game Over Text
        if self.game_over:
            text = "GAME OVER"
            if np.all(self.grid == 0): text = "YOU WIN!"
            
            over_surf = self.font_fx.render(text, True, self.COLOR_TEXT)
            over_rect = over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            bg_rect = over_rect.inflate(20, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0,0,0,150))
            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(over_surf, over_rect)


    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To run, you might need to unset the dummy video driver
    # comment out: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # or run with: SDL_VIDEODRIVER=x11 python your_file.py
    
    # For this example, we'll assume the dummy driver is active and just test the logic
    # To play visually, you need a display.
    try:
        real_display = True
        pygame.display.init()
        pygame.display.set_mode((1,1))
    except pygame.error:
        real_display = False
        print("No display available. Running in headless mode. Visuals will not be shown.")

    env = GameEnv()
    obs, info = env.reset()
    
    if not real_display:
        # Simple test loop for headless mode
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Game ended. Final score: {info['score']}")
                obs, info = env.reset()
        env.close()
        exit()

    # --- Interactive Game Loop (requires a display) ---
    running = True
    terminated = False
    
    # Game loop
    while running:
        action = np.array([0, 0, 0]) # Default no-op action
        
        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        should_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                should_step = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        if not terminated:
            # Since auto_advance is False, we only step when a key is pressed
            if should_step:
                 obs, reward, terminated, truncated, info = env.step(action)
                 if reward != 0:
                    print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")
            else:
                 # If no keys are pressed, just update the last action to release buttons
                 env.last_action = action
                 obs = env._get_observation() # Re-render for animations like glow

        # Render the observation to the screen
        real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate

    env.close()