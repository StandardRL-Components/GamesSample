import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:55:26.138912
# Source Brief: brief_01993.md
# Brief Index: 1993
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Temporal Threads: A match-3 puzzle game where the player manipulates seasons
    to maximize resource collection.

    The player controls a cursor on a grid of time fragments. Matching 3 or more
    fragments of the same color yields 'Temporal Threads'. These threads can be
    spent on crafting items that manipulate the game's season, which in turn
    affects the number of threads gained from matches.

    The goal is to reach a target number of threads before the episode ends.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = "A match-3 puzzle game where you swap time fragments to gather resources. Manipulate the seasons to multiply your gains and craft powerful abilities."
    user_guide = "Use arrow keys to move the cursor. Press space to select/swap tiles or craft items. Press shift to switch focus between the grid and the crafting menu."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 8, 8
    TILE_SIZE = 40
    GRID_MARGIN_TOP, GRID_MARGIN_LEFT = 40, 40
    GRID_WIDTH = GRID_COLS * TILE_SIZE
    GRID_HEIGHT = GRID_ROWS * TILE_SIZE
    UI_X_START = GRID_MARGIN_LEFT + GRID_WIDTH + 20

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_GRID_BG = (20, 30, 50)
    COLOR_GRID_LINES = (40, 60, 90)
    COLOR_UI_BG = (15, 25, 45)
    COLOR_UI_BORDER = (50, 70, 110)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_DIM = (150, 150, 170)
    COLOR_TEXT_SUCCESS = (100, 255, 150)
    COLOR_TEXT_FAIL = (255, 100, 100)
    
    TILE_COLORS = {
        1: (66, 135, 245),   # Blue (Winter)
        2: (78, 194, 84),    # Green (Spring)
        3: (219, 70, 70),    # Red (Summer)
        4: (237, 195, 62),   # Yellow (Autumn)
    }
    TILE_SYMBOLS = { # For potential future use, e.g. colorblind mode
        1: "❄", 2: "⚘", 3: "☀", 4: "☁"
    }

    SEASON_COLORS = {
        "WINTER": (15, 25, 55),
        "SPRING": (15, 45, 25),
        "SUMMER": (55, 25, 15),
        "AUTUMN": (55, 45, 15),
    }
    SEASON_MULTIPLIERS = {"SPRING": 1.5, "SUMMER": 1.0, "AUTUMN": 2.0, "WINTER": 1.0}
    SEASONS = ["SPRING", "SUMMER", "AUTUMN", "WINTER"]

    # Game parameters
    TARGET_THREADS = 1000
    MAX_STEPS = 2000
    NUM_TILE_TYPES = 4
    CURSOR_LERP_RATE = 0.4
    SCORE_LERP_RATE = 0.1

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
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)
        
        self.render_mode = render_mode

        # Game state variables are initialized in reset()
        self._initialize_crafting_recipes()
        # self.reset() is called by the wrapper, but we can call it here for standalone use
        
    def _initialize_crafting_recipes(self):
        self.crafting_recipes = [
            {"name": "Advance Season", "cost": 50, "effect": self._craft_advance_season, "unlocked_at": 0},
            {"name": "Repeat Season", "cost": 100, "effect": self._craft_repeat_season, "unlocked_at": 250},
            {"name": "Board Reshuffle", "cost": 75, "effect": self._craft_reshuffle, "unlocked_at": 500},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0 # Temporal Threads
        self.display_score = 0.0
        self.game_over = False
        self.win = False

        self.board = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.cursor_pos = [0, 0]
        self.visual_cursor_pos = self._get_pixel_pos(self.cursor_pos[0], self.cursor_pos[1])
        self.selected_tile_pos = None

        self.focus = "GRID"  # "GRID" or "CRAFT"
        self.craft_selection_index = 0
        self.available_recipes = []
        self._update_available_recipes()

        self.current_season_index = 0 # Start in Spring
        
        self.particles = []
        self.floating_texts = []
        self.action_cooldown = 0
        self.last_action_time = 0

        self._generate_initial_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.action_cooldown = max(0, self.action_cooldown - 1)
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.action_cooldown == 0:
            movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

            if shift_press:
                self.focus = "CRAFT" if self.focus == "GRID" else "GRID"
                # sfx: UI_focus_swap
            
            if self.focus == "GRID":
                reward += self._handle_grid_input(movement, space_press)
            elif self.focus == "CRAFT":
                reward += self._handle_craft_input(movement, space_press)

        self._update_game_state()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if self.win and not self.game_over:
            reward += 100
            self.floating_texts.append(self._create_floating_text("VICTORY!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.COLOR_TEXT_SUCCESS, size='large'))
            self.game_over = True

        if terminated and not self.win:
             self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_grid_input(self, movement, space_press):
        # --- Handle Movement ---
        if movement > 0:
            prev_pos = list(self.cursor_pos)
            if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 2: self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
            elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 4: self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)
            if prev_pos != self.cursor_pos:
                # sfx: cursor_move
                pass
        
        # --- Handle Selection/Swap ---
        if space_press:
            if self.selected_tile_pos is None:
                self.selected_tile_pos = list(self.cursor_pos)
                # sfx: tile_select
            else:
                # Attempt to swap
                r1, c1 = self.selected_tile_pos
                r2, c2 = self.cursor_pos
                
                # Check for adjacency
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    self._swap_tiles(r1, c1, r2, c2)
                    matches1 = self._find_matches_at(r1, c1)
                    matches2 = self._find_matches_at(r2, c2)
                    
                    if matches1 or matches2:
                        # sfx: swap_success
                        self.action_cooldown = 5 # Give time for animations
                        return self._process_all_matches()
                    else:
                        # Invalid swap, swap back
                        self._swap_tiles(r1, c1, r2, c2)
                        self.selected_tile_pos = None
                        # sfx: swap_fail
                        return -0.1 # Small penalty for invalid move
                else:
                    # Not adjacent, just change selection
                    self.selected_tile_pos = list(self.cursor_pos)
                    # sfx: tile_select
        return 0

    def _handle_craft_input(self, movement, space_press):
        if not self.available_recipes:
            return 0
        
        if movement in [1, 2]: # Up/Down
            if movement == 1: self.craft_selection_index = (self.craft_selection_index - 1) % len(self.available_recipes)
            elif movement == 2: self.craft_selection_index = (self.craft_selection_index + 1) % len(self.available_recipes)
            # sfx: UI_cycle
        
        if space_press:
            recipe = self.available_recipes[self.craft_selection_index]
            if self.score >= recipe["cost"]:
                self.score -= recipe["cost"]
                recipe["effect"]()
                self.floating_texts.append(self._create_floating_text(f"-{recipe['cost']} Threads", (self.UI_X_START + 100, 80), self.COLOR_TEXT_FAIL))
                # sfx: craft_success
                self.focus = "GRID"
                self.action_cooldown = 5
                return 5 # Reward for crafting
            else:
                # sfx: craft_fail
                return -0.1 # Small penalty for trying to craft without funds
        return 0

    def _update_game_state(self):
        # Smoothly update display score
        self.display_score += (self.score - self.display_score) * self.SCORE_LERP_RATE
        
        # Smoothly update visual cursor
        target_pixel_pos = self._get_pixel_pos(self.cursor_pos[0], self.cursor_pos[1])
        self.visual_cursor_pos[0] += (target_pixel_pos[0] - self.visual_cursor_pos[0]) * self.CURSOR_LERP_RATE
        self.visual_cursor_pos[1] += (target_pixel_pos[1] - self.visual_cursor_pos[1]) * self.CURSOR_LERP_RATE

        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

        # Update floating texts
        self.floating_texts = [t for t in self.floating_texts if t['lifespan'] > 0]
        for t in self.floating_texts:
            t['pos'][1] -= t['vel']
            t['lifespan'] -= 1
            
        # Update available recipes
        self._update_available_recipes()

    def _update_available_recipes(self):
        self.available_recipes = [r for r in self.crafting_recipes if self.score >= r['unlocked_at']]
        if self.available_recipes:
            self.craft_selection_index = min(self.craft_selection_index, len(self.available_recipes) - 1)

    def _check_termination(self):
        if self.score >= self.TARGET_THREADS:
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_grid()
        self._render_particles()
        self._render_ui()
        self._render_floating_texts()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "season": self.SEASONS[self.current_season_index]}

    # --- Rendering Methods ---
    def _render_background(self):
        season_bg = self.SEASON_COLORS[self.SEASONS[self.current_season_index]]
        self.screen.fill(self.COLOR_BG)
        # Blend season color into background
        season_overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        season_overlay.fill(season_bg + (80,)) # Use alpha for subtlety
        self.screen.blit(season_overlay, (0, 0))

    def _render_grid(self):
        # Grid background and border
        grid_rect = pygame.Rect(self.GRID_MARGIN_LEFT, self.GRID_MARGIN_TOP, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)
        
        # Tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_type = self.board[r, c]
                if tile_type > 0:
                    px, py = self._get_pixel_pos(r, c)
                    color = self.TILE_COLORS[tile_type]
                    
                    # Draw subtle shadow
                    pygame.gfxdraw.filled_circle(self.screen, int(px + 2), int(py + 2), self.TILE_SIZE // 2 - 2, (0,0,0,50))
                    # Draw main tile
                    pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), self.TILE_SIZE // 2 - 2, color)
                    pygame.gfxdraw.aacircle(self.screen, int(px), int(py), self.TILE_SIZE // 2 - 2, color)

        # Selected tile highlight
        if self.selected_tile_pos is not None:
            r, c = self.selected_tile_pos
            px, py = self._get_pixel_pos(r, c)
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2 * 5
            pygame.draw.circle(self.screen, (255, 255, 255), (px, py), self.TILE_SIZE // 2, 2 + int(pulse))

        # Cursor
        cx, cy = self.visual_cursor_pos
        cursor_rect = pygame.Rect(cx - self.TILE_SIZE // 2, cy - self.TILE_SIZE // 2, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, (255, 255, 0), cursor_rect, 3, border_radius=5)

    def _render_particles(self):
        for p in self.particles:
            color = p['color'] + (int(255 * (p['lifespan'] / p['max_lifespan'])),) # Fade out
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)

    def _render_ui(self):
        # UI Panel
        ui_rect = pygame.Rect(self.UI_X_START, self.GRID_MARGIN_TOP, self.SCREEN_WIDTH - self.UI_X_START - 20, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect, border_radius=10)
        
        # Focus indicators
        if self.focus == "GRID":
            pygame.draw.rect(self.screen, (255,255,0), (self.GRID_MARGIN_LEFT-4, self.GRID_MARGIN_TOP-4, self.GRID_WIDTH+8, self.GRID_HEIGHT+8), 2, border_radius=5)
        else: # CRAFT
            pygame.draw.rect(self.screen, (255,255,0), (ui_rect.x-4, ui_rect.y-4, ui_rect.width+8, ui_rect.height+8), 2, border_radius=5)

        # Score / Threads
        score_text = self.font_large.render(f"{math.floor(self.display_score):,}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.UI_X_START + 20, 50))
        target_text = self.font_small.render(f"GOAL: {self.TARGET_THREADS:,}", True, self.COLOR_TEXT_DIM)
        self.screen.blit(target_text, (self.UI_X_START + 20, 95))

        # Season
        season = self.SEASONS[self.current_season_index]
        season_color = self.TILE_COLORS[self.current_season_index % 4 + 1]
        season_text = self.font_medium.render("Season:", True, self.COLOR_TEXT_DIM)
        self.screen.blit(season_text, (self.UI_X_START + 20, 140))
        season_name_text = self.font_medium.render(season, True, season_color)
        self.screen.blit(season_name_text, (self.UI_X_START + 20, 165))
        mult_text = self.font_small.render(f"({self.SEASON_MULTIPLIERS[season]}x threads)", True, self.COLOR_TEXT_DIM)
        self.screen.blit(mult_text, (self.UI_X_START + 20, 190))

        # Crafting
        craft_title = self.font_medium.render("Crafting", True, self.COLOR_TEXT)
        self.screen.blit(craft_title, (self.UI_X_START + 20, 230))
        
        y_offset = 260
        for i, recipe in enumerate(self.available_recipes):
            is_selected = (i == self.craft_selection_index) and (self.focus == "CRAFT")
            can_afford = self.score >= recipe['cost']
            
            card_color = self.COLOR_UI_BORDER if is_selected else self.COLOR_GRID_LINES
            card_rect = pygame.Rect(self.UI_X_START + 15, y_offset, ui_rect.width - 30, 40)
            pygame.draw.rect(self.screen, card_color, card_rect, 2, 5)
            
            name_color = self.COLOR_TEXT if can_afford else self.COLOR_TEXT_DIM
            name_surf = self.font_small.render(recipe['name'], True, name_color)
            self.screen.blit(name_surf, (card_rect.x + 10, card_rect.y + 12))
            
            cost_color = self.COLOR_TEXT_SUCCESS if can_afford else self.COLOR_TEXT_FAIL
            cost_surf = self.font_small.render(f"{recipe['cost']}", True, cost_color)
            cost_rect = cost_surf.get_rect(right=card_rect.right - 10, centery=card_rect.centery)
            self.screen.blit(cost_surf, cost_rect)
            y_offset += 50
            
    def _render_floating_texts(self):
        for t in self.floating_texts:
            alpha = int(255 * (t['lifespan'] / t['max_lifespan']))
            font = self.font_large if t['size'] == 'large' else self.font_medium
            text_surf = font.render(t['text'], True, t['color'])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=t['pos'])
            self.screen.blit(text_surf, text_rect)

    # --- Game Logic Methods ---
    def _generate_initial_board(self):
        while True:
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    self.board[r, c] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
            if not self._find_all_matches() and self._has_valid_moves():
                break

    def _swap_tiles(self, r1, c1, r2, c2):
        self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]

    def _find_matches_at(self, r, c):
        tile_type = self.board[r, c]
        if tile_type == 0: return set()
        
        # Horizontal
        h_matches = {(r, c)}
        for i in range(c - 1, -1, -1):
            if self.board[r, i] == tile_type: h_matches.add((r, i))
            else: break
        for i in range(c + 1, self.GRID_COLS):
            if self.board[r, i] == tile_type: h_matches.add((r, i))
            else: break
        
        # Vertical
        v_matches = {(r, c)}
        for i in range(r - 1, -1, -1):
            if self.board[i, c] == tile_type: v_matches.add((i, c))
            else: break
        for i in range(r + 1, self.GRID_ROWS):
            if self.board[i, c] == tile_type: v_matches.add((i, c))
            else: break
            
        final_matches = set()
        if len(h_matches) >= 3: final_matches.update(h_matches)
        if len(v_matches) >= 3: final_matches.update(v_matches)
        return final_matches

    def _find_all_matches(self):
        all_matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                all_matches.update(self._find_matches_at(r,c))
        return all_matches

    def _process_all_matches(self):
        total_reward = 0
        chain = 0
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            
            chain += 1
            # sfx: match_success_chain_{chain}
            
            # Group matches by length for reward calculation
            # This is an approximation, but good enough
            match_count = len(matches)
            if match_count == 3: total_reward += 1
            elif match_count == 4: total_reward += 2
            else: total_reward += 3

            base_threads = match_count * chain
            season_multiplier = self.SEASON_MULTIPLIERS[self.SEASONS[self.current_season_index]]
            threads_gained = int(base_threads * season_multiplier)
            self.score += threads_gained

            for r, c in matches:
                self._create_particles(r, c)
                self.board[r, c] = 0
            
            self.floating_texts.append(self._create_floating_text(f"+{threads_gained}", self._get_pixel_pos(*list(matches)[0])))

            self._apply_gravity_and_refill()
            self.selected_tile_pos = None

        if not self._has_valid_moves():
            self._reshuffle_board()
            self.floating_texts.append(self._create_floating_text("Reshuffle!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.COLOR_TEXT, size='large'))
        
        return total_reward

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.board[r, c] != 0:
                    if r != empty_row:
                        self.board[empty_row, c] = self.board[r, c]
                        self.board[r, c] = 0
                    empty_row -= 1
            for r in range(empty_row, -1, -1):
                self.board[r, c] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
        # sfx: tiles_fall

    def _has_valid_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Check swap right
                if c < self.GRID_COLS - 1:
                    self._swap_tiles(r, c, r, c + 1)
                    if self._find_all_matches():
                        self._swap_tiles(r, c, r, c + 1)
                        return True
                    self._swap_tiles(r, c, r, c + 1)
                # Check swap down
                if r < self.GRID_ROWS - 1:
                    self._swap_tiles(r, c, r + 1, c)
                    if self._find_all_matches():
                        self._swap_tiles(r, c, r + 1, c)
                        return True
                    self._swap_tiles(r, c, r + 1, c)
        return False

    def _reshuffle_board(self):
        while True:
            flat_board = self.board.flatten()
            self.np_random.shuffle(flat_board)
            self.board = flat_board.reshape((self.GRID_ROWS, self.GRID_COLS))
            if not self._find_all_matches() and self._has_valid_moves():
                break
        # sfx: board_reshuffle

    # --- Crafting Effects ---
    def _craft_advance_season(self):
        self.current_season_index = (self.current_season_index + 1) % len(self.SEASONS)
        self.floating_texts.append(self._create_floating_text("Season Advanced!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.COLOR_TEXT, size='large'))

    def _craft_repeat_season(self):
        # Effectively does nothing but consume resources, but in a game could be used to stay in a beneficial season
        self.floating_texts.append(self._create_floating_text("Season Stabilized!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.COLOR_TEXT, size='large'))

    def _craft_reshuffle(self):
        self._reshuffle_board()
        self.floating_texts.append(self._create_floating_text("Board Reshuffled!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.COLOR_TEXT, size='large'))
        
    # --- Helper & Utility Methods ---
    def _get_pixel_pos(self, r, c):
        px = self.GRID_MARGIN_LEFT + c * self.TILE_SIZE + self.TILE_SIZE // 2
        py = self.GRID_MARGIN_TOP + r * self.TILE_SIZE + self.TILE_SIZE // 2
        return [px, py]

    def _create_particles(self, r, c):
        px, py = self._get_pixel_pos(r, c)
        color = self.TILE_COLORS[self.board[r, c]]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'lifespan': 30,
                'max_lifespan': 30,
                'size': self.np_random.uniform(3, 7)
            })
    
    def _create_floating_text(self, text, pos, color=(255,255,255), vel=1, lifespan=60, size='medium'):
        return {'text': text, 'pos': list(pos), 'color': color, 'vel': vel, 'lifespan': lifespan, 'max_lifespan': lifespan, 'size': size}

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Temporal Threads - Manual Test")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")

    while not done:
        # Action mapping from keyboard
        movement = 0  # no-op
        space_press = 0
        shift_press = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_press = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_press = 1
                elif event.key == pygame.K_q: done = True

        action = [movement, space_press, shift_press]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    print(f"\nGame Over! Final Score: {info['score']}, Total Reward: {total_reward}, Steps: {info['steps']}")
    env.close()