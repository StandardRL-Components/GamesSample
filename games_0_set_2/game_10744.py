import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:52:48.762522
# Source Brief: brief_00744.md
# Brief Index: 744
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A mystical puzzle game where you manipulate a grid of colored symbols to match specific patterns, "
        "unlocking powers and weaving a dream."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to use the selected power on the grid, "
        "and press shift to cycle through your unlocked powers."
    )
    auto_advance = False

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (15, 10, 35)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_ACCENT = (255, 215, 50)
    COLOR_CURSOR = (255, 255, 150)
    TILE_COLORS = {
        0: (50, 150, 255),  # Blue (Calmness)
        1: (255, 80, 80),   # Red (Passion)
        2: (80, 255, 150),  # Green (Growth)
        3: (255, 215, 50),  # Gold (Insight)
    }
    SYMBOL_COLOR = (255, 255, 255)

    # Dimensions
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 12, 8
    TILE_SIZE = 32
    GRID_MARGIN_X = 40
    GRID_MARGIN_Y = 40
    GRID_WIDTH = GRID_COLS * TILE_SIZE
    GRID_HEIGHT = GRID_ROWS * TILE_SIZE
    
    # Game Parameters
    MAX_TURNS = 150
    MAX_STEPS = 1000
    NUM_SYMBOLS = 4
    NUM_COLORS = 4
    NARRATIVE_GOAL = 10

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
        self.font_small = pygame.font.SysFont("sans-serif", 16)
        self.font_medium = pygame.font.SysFont("sans-serif", 20)
        self.font_large = pygame.font.SysFont("sans-serif", 32)
        
        self.grid = None
        self.cursor_pos = None
        self.steps = None
        self.turns = None
        self.score = None
        self.game_over = None
        self.reward_this_step = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.animations = None
        self.all_powers = None
        self.unlocked_powers = None
        self.current_power_idx_ptr = None
        self.narrative_progress = None
        self.target_pattern = None

        # self.reset() # reset is called by the wrapper/runner
        # self.validate_implementation() # this is for debugging, not needed in production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.turns = self.MAX_TURNS
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.prev_space_held = True
        self.prev_shift_held = True
        
        self.animations = []

        self._initialize_powers()
        self._initialize_grid()
        
        self.narrative_progress = 0
        self._update_target_pattern()

        # Ensure no matches on reset
        while self._find_and_process_matches(False) > 0:
            self._initialize_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        self.game_over = (self.turns <= 0) or (self.narrative_progress >= self.NARRATIVE_GOAL) or (self.steps >= self.MAX_STEPS)

        if self.game_over:
            if self.narrative_progress >= self.NARRATIVE_GOAL:
                self.reward_this_step += 100 # Win reward
            else:
                self.reward_this_step -= 100 # Loss reward
            
            return self._get_observation(), self.reward_this_step, True, False, self._get_info()

        self._handle_input(action)
        self._update_animations()
        
        self.steps += 1
        terminated = self.game_over

        return (
            self._get_observation(),
            self.reward_this_step,
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
            "turns": self.turns,
            "narrative_progress": self.narrative_progress,
        }

    def _initialize_grid(self):
        self.grid = np.zeros((self.GRID_COLS, self.GRID_ROWS, 2), dtype=int)
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                self.grid[x, y] = [self.np_random.integers(0, self.NUM_SYMBOLS), self.np_random.integers(0, self.NUM_COLORS)]

    def _initialize_powers(self):
        self.all_powers = [
            {"name": "SHIFT ROW", "desc": "Shift row right", "cost": 1},
            {"name": "SHIFT COL", "desc": "Shift column down", "cost": 1},
            {"name": "ROTATE 2x2", "desc": "Rotate 2x2 block", "cost": 1},
            {"name": "SHUFFLE", "desc": "Randomize board", "cost": 3},
        ]
        self.unlocked_powers = [0]
        self.current_power_idx_ptr = 0

    def _handle_input(self, action):
        movement, space_val, shift_val = action
        space_pressed = space_val == 1 and not self.prev_space_held
        shift_pressed = shift_val == 1 and not self.prev_shift_held

        self.prev_space_held = space_val == 1
        self.prev_shift_held = shift_val == 1

        if any(anim['type'] == 'slide' for anim in self.animations):
            return # Don't allow input during slide animations

        # --- Movement ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
        
        # --- Cycle Power (Shift) ---
        if shift_pressed:
            self.current_power_idx_ptr = (self.current_power_idx_ptr + 1) % len(self.unlocked_powers)
            # SFX: UI_CYCLE_SOUND

        # --- Execute Power (Space) ---
        if space_pressed:
            self._execute_current_power()

    def _execute_current_power(self):
        power_idx = self.unlocked_powers[self.current_power_idx_ptr]
        power = self.all_powers[power_idx]
        
        if self.turns < power['cost']:
            # SFX: ERROR_SOUND
            return

        self.turns -= power['cost']
        # SFX: ACTION_SOUND
        
        cx, cy = self.cursor_pos
        
        if power['name'] == "SHIFT ROW":
            row = self.grid[:, cy].copy()
            self.grid[:, cy] = np.roll(row, 1, axis=0)
            for i in range(self.GRID_COLS):
                self._add_slide_animation(i, cy, (i - 1 + self.GRID_COLS) % self.GRID_COLS, cy)
        
        elif power['name'] == "SHIFT COL":
            col = self.grid[cx, :].copy()
            self.grid[cx, :] = np.roll(col, 1, axis=0)
            for i in range(self.GRID_ROWS):
                self._add_slide_animation(cx, i, cx, (i - 1 + self.GRID_ROWS) % self.GRID_ROWS)

        elif power['name'] == "ROTATE 2x2":
            x0, y0 = cx, cy
            x1, y1 = (cx + 1) % self.GRID_COLS, (cy + 1) % self.GRID_ROWS
            
            temp = self.grid[x0, y0].copy()
            self.grid[x0, y0] = self.grid[x0, y1].copy()
            self.grid[x0, y1] = self.grid[x1, y1].copy()
            self.grid[x1, y1] = self.grid[x1, y0].copy()
            self.grid[x1, y0] = temp
            
            self._add_slide_animation(x0, y1, x0, y0)
            self._add_slide_animation(x1, y1, x0, y1)
            self._add_slide_animation(x1, y0, x1, y1)
            self._add_slide_animation(x0, y0, x1, y0)

        elif power['name'] == "SHUFFLE":
            self.reward_this_step -= 0.1
            self._initialize_grid()
            while self._find_and_process_matches(False) > 0:
                self._initialize_grid()

        # Post-action processing
        self._process_turn_end()

    def _process_turn_end(self):
        chain_reaction_level = 0
        while True:
            matches_found = self._find_and_process_matches(True)
            if matches_found > 0:
                self.reward_this_step += matches_found * (1 + chain_reaction_level * 0.5)
                self.score += matches_found * 10 * (1 + chain_reaction_level)
                self._refill_grid()
                chain_reaction_level += 1
                # SFX: CHAIN_REACTION_SOUND
            else:
                break
        self._update_progression()

    def _find_and_process_matches(self, process_for_real):
        matched_tiles = set()
        symbol_target, color_target, length_target = self.target_pattern

        # Check rows
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS - length_target + 1):
                is_match = True
                for i in range(length_target):
                    tile_symbol, tile_color = self.grid[x + i, y]
                    if not (tile_symbol == symbol_target and tile_color == color_target):
                        is_match = False
                        break
                if is_match:
                    for i in range(length_target):
                        matched_tiles.add((x + i, y))
        
        # Check columns
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS - length_target + 1):
                is_match = True
                for i in range(length_target):
                    tile_symbol, tile_color = self.grid[x, y + i]
                    if not (tile_symbol == symbol_target and tile_color == color_target):
                        is_match = False
                        break
                if is_match:
                    for i in range(length_target):
                        matched_tiles.add((x, y + i))

        if process_for_real and matched_tiles:
            for x, y in matched_tiles:
                self.grid[x, y, 0] = -1 # Mark for removal
                self._add_particle_burst(x, y)
            self.narrative_progress += 1
            self.reward_this_step += 10 # Narrative stage reward
            self._update_target_pattern()
            # SFX: MATCH_SUCCESS_SOUND
        
        return len(matched_tiles)

    def _refill_grid(self):
        for x in range(self.GRID_COLS):
            empty_count = 0
            for y in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[x, y, 0] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    self.grid[x, y + empty_count] = self.grid[x, y].copy()
                    self._add_slide_animation(x, y, x, y + empty_count, fall=True)

            for y in range(empty_count):
                self.grid[x, y] = [self.np_random.integers(0, self.NUM_SYMBOLS), self.np_random.integers(0, self.NUM_COLORS)]
                self._add_slide_animation(x, -1 -y, x, y, fall=True)

    def _update_progression(self):
        # Unlock SWIFT COL
        if self.score >= 100 and 1 not in self.unlocked_powers:
            self.unlocked_powers.append(1)
            self.reward_this_step += 5
        # Unlock ROTATE 2x2
        if self.score >= 300 and 2 not in self.unlocked_powers:
            self.unlocked_powers.append(2)
            self.reward_this_step += 5
        # Unlock SHUFFLE
        if self.score >= 500 and 3 not in self.unlocked_powers:
            self.unlocked_powers.append(3)
            self.reward_this_step += 5
    
    def _update_target_pattern(self):
        # symbol, color, length
        self.target_pattern = [
            self.np_random.integers(0, self.NUM_SYMBOLS),
            self.np_random.integers(0, self.NUM_COLORS),
            3
        ]

    # --- ANIMATIONS ---
    def _add_particle_burst(self, grid_x, grid_y):
        px, py = self._grid_to_pixel(grid_x, grid_y)
        px += self.TILE_SIZE // 2
        py += self.TILE_SIZE // 2
        color = self.TILE_COLORS[self.grid[grid_x, grid_y, 1]]
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.animations.append({'type': 'particle', 'pos': [px, py], 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _add_slide_animation(self, from_x, from_y, to_x, to_y, fall=False):
        tile_data = self.grid[to_x, to_y].copy()
        start_pos = self._grid_to_pixel(from_x, from_y)
        end_pos = self._grid_to_pixel(to_x, to_y)
        duration = 8 if fall else 5
        self.animations.append({'type': 'slide', 'tile': tile_data, 'start': start_pos, 'end': end_pos, 'progress': 0, 'duration': duration})

    def _update_animations(self):
        for anim in self.animations[:]:
            if anim['type'] == 'particle':
                anim['pos'][0] += anim['vel'][0]
                anim['pos'][1] += anim['vel'][1]
                anim['vel'][1] += 0.1 # Gravity
                anim['life'] -= 1
                if anim['life'] <= 0:
                    self.animations.remove(anim)
            elif anim['type'] == 'slide':
                anim['progress'] += 1
                if anim['progress'] >= anim['duration']:
                    self.animations.remove(anim)

    # --- RENDERING ---
    def _render_game(self):
        # Draw grid lines (subtle)
        for x in range(self.GRID_COLS + 1):
            px = self.GRID_MARGIN_X + x * self.TILE_SIZE
            pygame.draw.line(self.screen, (30, 25, 55), (px, self.GRID_MARGIN_Y), (px, self.GRID_MARGIN_Y + self.GRID_HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = self.GRID_MARGIN_Y + y * self.TILE_SIZE
            pygame.draw.line(self.screen, (30, 25, 55), (self.GRID_MARGIN_X, py), (self.GRID_MARGIN_X + self.GRID_WIDTH, py))

        # Draw static tiles
        is_sliding = any(a['type'] == 'slide' for a in self.animations)
        if not is_sliding:
            for x in range(self.GRID_COLS):
                for y in range(self.GRID_ROWS):
                    self._draw_tile(self.grid[x,y], x, y)

        # Draw animated tiles and particles
        self._render_animated_elements()
        
        # Draw cursor
        if not is_sliding:
            cx, cy = self.cursor_pos
            px, py = self._grid_to_pixel(cx, cy)
            rect = (px, py, self.TILE_SIZE, self.TILE_SIZE)
            
            # Glow effect
            glow_size = int(self.TILE_SIZE * 0.75)
            glow_alpha = 100 + 50 * math.sin(self.steps * 0.2)
            for i in range(glow_size, 0, -2):
                alpha = glow_alpha * (1 - (i / glow_size))**2
                pygame.gfxdraw.aacircle(self.screen, int(px + self.TILE_SIZE/2), int(py + self.TILE_SIZE/2), int(i/2), (*self.COLOR_CURSOR, int(alpha)))
            
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=4)
    
    def _render_animated_elements(self):
        # Draw sliding tiles first
        sliding_tiles = [a for a in self.animations if a['type'] == 'slide']
        for anim in sliding_tiles:
            t = anim['progress'] / anim['duration']
            t = t*t # Ease-in
            curr_x = anim['start'][0] + (anim['end'][0] - anim['start'][0]) * t
            curr_y = anim['start'][1] + (anim['end'][1] - anim['start'][1]) * t
            self._draw_tile_at_pixel(anim['tile'], curr_x, curr_y)

        # Draw particles on top
        for anim in self.animations:
            if anim['type'] == 'particle':
                p = anim['life'] / anim['max_life']
                size = int(p * 5)
                if size > 0:
                    color = (*anim['color'], int(p * 255))
                    temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (size, size), size)
                    self.screen.blit(temp_surf, (int(anim['pos'][0] - size), int(anim['pos'][1] - size)))

    def _grid_to_pixel(self, grid_x, grid_y):
        return self.GRID_MARGIN_X + grid_x * self.TILE_SIZE, self.GRID_MARGIN_Y + grid_y * self.TILE_SIZE

    def _draw_tile(self, tile_data, grid_x, grid_y):
        px, py = self._grid_to_pixel(grid_x, grid_y)
        self._draw_tile_at_pixel(tile_data, px, py)

    def _draw_tile_at_pixel(self, tile_data, px, py):
        symbol, color_id = tile_data
        if symbol == -1: return # Don't draw removed tiles
        
        rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
        
        # Tile background
        border_color = tuple(c*0.7 for c in self.TILE_COLORS[color_id])
        pygame.draw.rect(self.screen, border_color, rect, 0, border_radius=4)
        inner_rect = rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, self.TILE_COLORS[color_id], inner_rect, 0, border_radius=3)
        
        # Symbol
        center_x, center_y = px + self.TILE_SIZE / 2, py + self.TILE_SIZE / 2
        s = self.TILE_SIZE * 0.3
        
        if symbol == 0: # Circle
            pygame.draw.circle(self.screen, self.SYMBOL_COLOR, (center_x, center_y), s)
        elif symbol == 1: # Square
            pygame.draw.rect(self.screen, self.SYMBOL_COLOR, (center_x - s/2, center_y - s/2, s, s))
        elif symbol == 2: # Triangle
            points = [(center_x, center_y - s/1.5), (center_x - s/1.5, center_y + s/2), (center_x + s/1.5, center_y + s/2)]
            pygame.draw.polygon(self.screen, self.SYMBOL_COLOR, points)
        elif symbol == 3: # Plus
            pygame.draw.line(self.screen, self.SYMBOL_COLOR, (center_x - s/1.5, center_y), (center_x + s/1.5, center_y), 3)
            pygame.draw.line(self.screen, self.SYMBOL_COLOR, (center_x, center_y - s/1.5), (center_x, center_y + s/1.5), 3)

    def _render_ui(self):
        ui_x_base = self.GRID_MARGIN_X + self.GRID_WIDTH + 30
        
        # Score and Turns
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (ui_x_base, 40))
        turns_text = self.font_medium.render(f"TURNS: {self.turns}", True, self.COLOR_UI_TEXT)
        self.screen.blit(turns_text, (ui_x_base, 65))

        # Narrative Progress
        narrative_title = self.font_small.render("DREAM WEAVE", True, self.COLOR_UI_TEXT)
        self.screen.blit(narrative_title, (ui_x_base, 100))
        bar_w, bar_h = 150, 15
        progress = min(1.0, self.narrative_progress / self.NARRATIVE_GOAL)
        pygame.draw.rect(self.screen, (30, 25, 55), (ui_x_base, 120, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, (ui_x_base, 120, int(bar_w * progress), bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (ui_x_base, 120, bar_w, bar_h), 1)

        # Target Pattern
        target_title = self.font_small.render("CURRENT OBJECTIVE", True, self.COLOR_UI_TEXT)
        self.screen.blit(target_title, (ui_x_base, 155))
        symbol, color, length = self.target_pattern
        for i in range(length):
            self._draw_tile_at_pixel((symbol, color), ui_x_base + i * (self.TILE_SIZE * 0.8), 175)

        # Powers
        powers_title = self.font_small.render("POWERS (SHIFT TO CYCLE)", True, self.COLOR_UI_TEXT)
        self.screen.blit(powers_title, (ui_x_base, 230))
        for i, unlocked_idx in enumerate(self.unlocked_powers):
            power = self.all_powers[unlocked_idx]
            is_selected = i == self.current_power_idx_ptr
            color = self.COLOR_UI_ACCENT if is_selected else self.COLOR_UI_TEXT
            
            y_pos = 255 + i * 40
            
            power_name = self.font_medium.render(power['name'], True, color)
            self.screen.blit(power_name, (ui_x_base + 10, y_pos))
            power_desc = self.font_small.render(f"{power['desc']} (Cost: {power['cost']})", True, self.COLOR_UI_TEXT)
            self.screen.blit(power_desc, (ui_x_base + 10, y_pos + 18))
            
            if is_selected:
                pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, (ui_x_base - 5, y_pos-2, 160, 38), 1, border_radius=4)
        
        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((10, 5, 25, 200))
            self.screen.blit(overlay, (0, 0))
            
            if self.narrative_progress >= self.NARRATIVE_GOAL:
                end_text = self.font_large.render("DREAM REALIZED", True, self.COLOR_UI_ACCENT)
            else:
                end_text = self.font_large.render("DREAM FADED", True, (200, 50, 50))
                
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)


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

if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Dream Weaver")
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # no-op, released, released
    
    print("\n--- Controls ---")
    print("Arrows: Move cursor")
    print("Space: Use selected power")
    print("Shift: Cycle through powers")
    print("Q: Quit")
    
    while not done:
        # Human input mapping
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                done = True

        keys = pygame.key.get_pressed()
        
        # Movement
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # Buttons
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    print(f"Game Over! Final Score: {info['score']}")
    pygame.quit()