import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:31:51.985372
# Source Brief: brief_01043.md
# Brief Index: 1043
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Expand your settlement on a shrinking hexagonal island by placing tiles. "
        "Solve word puzzles to unlock new types of terrain and reach the target settlement size before you run out of space."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to place a tile or interact with letters. "
        "Press shift to toggle between placing tiles and solving the word puzzle."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    TILE_SIZE = 20
    WIN_CONDITION_TILES = 100
    MAX_STEPS = 2000

    # --- Colors ---
    COLOR_BG = (15, 23, 42)
    COLOR_LANDSCAPE = (30, 41, 59)
    COLOR_LANDSCAPE_BORDER = (220, 38, 38)
    COLOR_GRID = (51, 65, 85)
    COLOR_UI_TEXT = (226, 232, 240)
    COLOR_UI_ACCENT = (56, 189, 248)
    COLOR_UI_MODE_PLACE = (34, 197, 94)
    COLOR_UI_MODE_WORD = (249, 115, 22)
    
    TILE_DEFINITIONS = {
        "grass": {"color": (74, 222, 128), "unlockable": False},
        "water": {"color": (59, 130, 246), "unlockable": True},
        "desert": {"color": (251, 191, 36), "unlockable": True},
        "mountain": {"color": (168, 85, 247), "unlockable": True},
        "forest": {"color": (22, 101, 52), "unlockable": True},
    }
    
    WORD_LIST = [
        ("WATER", "water"),
        ("DESERT", "desert"),
        ("MOUNTAIN", "mountain"),
        ("FOREST", "forest"),
        ("RIVER", "water"),
        ("PEAK", "mountain"),
        ("DUNE", "desert"),
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        # State variables to be initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.settlement = None
        self.landscape_poly = None
        self.landscape_center = None
        self.landscape_radius = None
        self.shrink_timer = None
        self.shrink_interval = None
        self.cursor_hex_pos = None
        self.cursor_pixel_pos = None
        self.game_mode = None
        self.available_tiles = None
        self.selected_tile_idx = None
        self.particles = None
        self.current_word = None
        self.scrambled_word = None
        self.tile_to_unlock = None
        self.word_cursor_idx = None
        self.word_held_idx = None
        self.last_space_held = None
        self.last_shift_held = None
        self.reward_this_step = None
        
        # The original code called reset() and validate_implementation() here.
        # It's better practice to let the user call reset() explicitly.
        # We'll remove them to align with standard Gym environment design.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.landscape_center = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.landscape_radius = self.HEIGHT / 2 * 0.95
        self._update_landscape_poly()

        self.settlement = {}  # { (q, r): type }
        self.cursor_hex_pos = (0, 0)
        self.cursor_pixel_pos = self._hex_to_pixel(self.cursor_hex_pos)

        self.shrink_interval = 15
        self.shrink_timer = self.shrink_interval

        self.game_mode = "PLACE"
        self.available_tiles = ["grass"]
        self.selected_tile_idx = 0
        self.particles = []
        
        self._setup_new_word()

        self.word_cursor_idx = 0
        self.word_held_idx = None
        
        self.last_space_held = False
        self.last_shift_held = False

        self.reward_this_step = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = 0
        self.steps += 1
        
        self._handle_input(action)
        self._update_game_state()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        
        if shift_press:
            self._toggle_game_mode()
            # sfx: UI_mode_switch.wav

        if self.game_mode == "PLACE":
            self._handle_place_mode_input(movement, space_press)
        elif self.game_mode == "WORD":
            self._handle_word_mode_input(movement, space_press)

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _handle_place_mode_input(self, movement, space_press):
        if movement != 0:
            # Hex grid neighbors: E, W, SE, NW, SW, NE
            # Our mapping: R, L, D, U, (D+L), (U+R)
            q, r = self.cursor_hex_pos
            if movement == 1: self.cursor_hex_pos = self._hex_neighbor(q, r, 3) # Up -> NW
            elif movement == 2: self.cursor_hex_pos = self._hex_neighbor(q, r, 2) # Down -> SE
            elif movement == 3: self.cursor_hex_pos = self._hex_neighbor(q, r, 1) # Left -> W
            elif movement == 4: self.cursor_hex_pos = self._hex_neighbor(q, r, 0) # Right -> E
            # sfx: UI_cursor_move.wav
        
        if space_press:
            self._place_tile()

    def _handle_word_mode_input(self, movement, space_press):
        if movement in [3, 4] and self.steps % 3 == 0: # Slower cursor for word mode
            if movement == 3: self.word_cursor_idx = max(0, self.word_cursor_idx - 1)
            if movement == 4: self.word_cursor_idx = min(len(self.scrambled_word) - 1, self.word_cursor_idx + 1)
            # sfx: UI_char_select.wav

        if space_press:
            if self.word_held_idx is None:
                self.word_held_idx = self.word_cursor_idx
                # sfx: UI_char_pickup.wav
            else:
                # Swap letters
                held_char = self.scrambled_word[self.word_held_idx]
                self.scrambled_word[self.word_held_idx] = self.scrambled_word[self.word_cursor_idx]
                self.scrambled_word[self.word_cursor_idx] = held_char
                self.word_held_idx = None
                # sfx: UI_char_swap.wav
                self._check_word_solved()

    def _update_game_state(self):
        # Interpolate cursor
        target_pixel_pos = self._hex_to_pixel(self.cursor_hex_pos)
        self.cursor_pixel_pos = self.cursor_pixel_pos.lerp(target_pixel_pos, 0.4)

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # Landscape shrink logic
        self.shrink_timer -= 1
        if self.shrink_timer <= 0:
            self.landscape_radius = max(0, self.landscape_radius - 2)
            self._update_landscape_poly()
            self.shrink_interval = max(5, 15 - self.steps // 200)
            self.shrink_timer = self.shrink_interval
            # sfx: landscape_shrink.wav
    
    def _check_termination(self):
        if len(self.settlement) >= self.WIN_CONDITION_TILES:
            self.reward_this_step += 100
            return True
        if self.landscape_radius < self.TILE_SIZE:
            self.reward_this_step -= 100
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "settlement_size": len(self.settlement)}

    # --- Game Logic Helpers ---

    def _toggle_game_mode(self):
        if self.game_mode == "PLACE":
            self.game_mode = "WORD"
        else:
            self.game_mode = "PLACE"
            self.word_held_idx = None # Reset word selection
    
    def _place_tile(self):
        pos = self.cursor_hex_pos
        if pos in self.settlement:
            self.reward_this_step -= 0.1 # Penalty for trying to place on existing tile
            # sfx: placement_fail.wav
            return

        pixel_pos = self._hex_to_pixel(pos)
        if not self._is_point_in_poly(pixel_pos, self.landscape_poly):
            self.reward_this_step -= 0.1 # Penalty for placing outside landscape
            # sfx: placement_fail.wav
            return

        tile_type = self.available_tiles[self.selected_tile_idx % len(self.available_tiles)]
        
        # Placement rule: Water must be adjacent to water or edge of map
        if tile_type == 'water':
            is_valid_water_placement = False
            for i in range(6):
                neighbor = self._hex_neighbor(pos[0], pos[1], i)
                if self.settlement.get(neighbor) == 'water':
                    is_valid_water_placement = True
                    break
                if not self._is_point_in_poly(self._hex_to_pixel(neighbor), self.landscape_poly):
                    is_valid_water_placement = True # Adjacent to edge
                    break
            if not is_valid_water_placement and len(self.settlement) > 0:
                self.reward_this_step -= 0.1
                # sfx: placement_fail.wav
                return

        # Place the tile
        self.settlement[pos] = tile_type
        self.score += 1
        self.reward_this_step += 1
        # sfx: placement_success.wav
        
        # Add particles for effect
        color = self.TILE_DEFINITIONS[tile_type]["color"]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'pos': pixel_pos.copy(),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': random.randint(15, 30),
                'color': color
            })
        
        # Cycle selected tile type after placement
        self.selected_tile_idx = (self.selected_tile_idx + 1) % len(self.available_tiles)

    def _setup_new_word(self):
        unlockable = [t for t, d in self.TILE_DEFINITIONS.items() if d["unlockable"] and t not in self.available_tiles]
        if not unlockable:
            self.current_word = "COMPLETE"
            self.scrambled_word = list("COMPLETE")
            self.tile_to_unlock = None
            return

        potential_words = [w for w in self.WORD_LIST if w[1] in unlockable]
        if not potential_words:
            # Fallback if we run out of specific words
            self.tile_to_unlock = random.choice(unlockable)
            self.current_word = self.tile_to_unlock.upper()
        else:
            self.current_word, self.tile_to_unlock = random.choice(potential_words)
        
        scrambled = list(self.current_word)
        random.shuffle(scrambled)
        self.scrambled_word = scrambled

    def _check_word_solved(self):
        if "".join(self.scrambled_word) == self.current_word:
            self.reward_this_step += 5
            self.score += 5
            if self.tile_to_unlock and self.tile_to_unlock not in self.available_tiles:
                self.available_tiles.append(self.tile_to_unlock)
                # sfx: new_tile_unlocked.wav
            self._setup_new_word()
            self.game_mode = "PLACE" # Switch back to placing
            self.word_held_idx = None
    
    # --- Rendering Helpers ---

    def _render_game(self):
        # Draw landscape
        pygame.gfxdraw.filled_polygon(self.screen, self.landscape_poly, self.COLOR_LANDSCAPE)
        pygame.gfxdraw.aapolygon(self.screen, self.landscape_poly, self.COLOR_LANDSCAPE_BORDER)
        
        # Draw grid lines
        for q in range(-15, 16):
            for r in range(-15, 16):
                pixel_pos = self._hex_to_pixel((q, r))
                if self._is_point_in_poly(pixel_pos, self.landscape_poly):
                    self._draw_hexagon(self.screen, self.COLOR_GRID, pixel_pos, self.TILE_SIZE, 1)

        # Draw settlement
        for pos, tile_type in self.settlement.items():
            pixel_pos = self._hex_to_pixel(pos)
            color = self.TILE_DEFINITIONS[tile_type]["color"]
            self._draw_hexagon(self.screen, color, pixel_pos, self.TILE_SIZE, 0)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            try:
                # Use a temporary surface for blending if needed, but simple alpha drawing works
                s = pygame.Surface((int(p['life'] / 5) + 2, int(p['life'] / 5) + 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p['color'], alpha), (s.get_width()//2, s.get_height()//2), int(p['life'] / 10) + 1)
                self.screen.blit(s, (p['pos'].x - s.get_width()//2, p['pos'].y - s.get_height()//2))
            except (ValueError, TypeError): # Catch potential issues with color or alpha
                 pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['life'] / 10) + 1)
        
        # Draw cursor
        if self.game_mode == "PLACE":
            selected_type = self.available_tiles[self.selected_tile_idx]
            cursor_color = self.TILE_DEFINITIONS[selected_type]["color"]
            self._draw_glowing_hexagon(self.screen, cursor_color, self.cursor_pixel_pos, self.TILE_SIZE)
            self._draw_hexagon(self.screen, (255, 255, 255), self.cursor_pixel_pos, self.TILE_SIZE, 2)

    def _render_ui(self):
        # Score / Settlement Size
        score_text = self.font_medium.render(f"SIZE: {len(self.settlement)} / {self.WIN_CONDITION_TILES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Game Mode
        mode_text_str = f"MODE: {self.game_mode}"
        mode_color = self.COLOR_UI_MODE_PLACE if self.game_mode == "PLACE" else self.COLOR_UI_MODE_WORD
        mode_text = self.font_small.render(mode_text_str, True, mode_color)
        self.screen.blit(mode_text, (10, self.HEIGHT - 25))
        shift_icon_text = self.font_small.render("[SHIFT to toggle]", True, self.COLOR_UI_TEXT)
        self.screen.blit(shift_icon_text, (mode_text.get_width() + 15, self.HEIGHT - 25))

        # Available Tiles Palette
        for i, tile_type in enumerate(self.available_tiles):
            color = self.TILE_DEFINITIONS[tile_type]["color"]
            x_pos = self.WIDTH - 30 - (len(self.available_tiles) - 1 - i) * 35
            rect = pygame.Rect(x_pos, 10, 30, 30)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            if i == self.selected_tile_idx % len(self.available_tiles) and self.game_mode == "PLACE":
                pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, rect, 2, border_radius=4)

        # Word Scramble UI
        if self.game_mode == "WORD":
            self._draw_word_puzzle()
        else: # Dimmed version in place mode
            self._draw_word_puzzle(dimmed=True)

    def _draw_word_puzzle(self, dimmed=False):
        puzzle_bg_rect = pygame.Rect(0, 0, 400, 80)
        puzzle_bg_rect.center = (self.WIDTH / 2, self.HEIGHT - 50)
        
        s = pygame.Surface(puzzle_bg_rect.size, pygame.SRCALPHA)
        alpha = 80 if dimmed else 180
        pygame.draw.rect(s, (*self.COLOR_GRID, alpha), s.get_rect(), border_radius=10)
        self.screen.blit(s, puzzle_bg_rect.topleft)

        char_spacing = 40
        start_x = puzzle_bg_rect.centerx - (len(self.scrambled_word) - 1) * char_spacing / 2
        
        for i, char in enumerate(self.scrambled_word):
            color = self.COLOR_UI_TEXT if not dimmed else self.COLOR_GRID
            char_surf = self.font_large.render(char, True, color)
            char_rect = char_surf.get_rect(center=(start_x + i * char_spacing, puzzle_bg_rect.centery))
            self.screen.blit(char_surf, char_rect)

            if not dimmed:
                # Draw cursor
                if i == self.word_cursor_idx:
                    underline_rect = pygame.Rect(char_rect.left, char_rect.bottom + 2, char_rect.width, 3)
                    pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, underline_rect, border_radius=2)
                # Draw held indicator
                if i == self.word_held_idx:
                    pygame.draw.circle(self.screen, self.COLOR_UI_ACCENT, (char_rect.centerx, char_rect.bottom + 10), 5, 2)

    # --- Geometry & Drawing Helpers ---

    def _hex_to_pixel(self, hex_coord):
        q, r = hex_coord
        x = self.TILE_SIZE * (3./2. * q)
        y = self.TILE_SIZE * (math.sqrt(3)/2. * q + math.sqrt(3) * r)
        return pygame.Vector2(x, y) + self.landscape_center

    def _hex_neighbor(self, q, r, direction):
        # 0: E, 1: W, 2: SE, 3: NW, 4: SW, 5: NE
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, -1)]
        dq, dr = neighbors[direction]
        return (q + dq, r + dr)

    def _update_landscape_poly(self):
        self.landscape_poly = []
        for i in range(6):
            angle = math.pi / 3 * i - math.pi / 6
            x = self.landscape_center.x + self.landscape_radius * math.cos(angle)
            y = self.landscape_center.y + self.landscape_radius * math.sin(angle)
            self.landscape_poly.append((int(x), int(y)))

    def _is_point_in_poly(self, point, poly):
        x, y = point
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _draw_hexagon(self, surface, color, position, radius, width=0):
        points = []
        for i in range(6):
            angle = math.pi / 3 * i + math.pi / 6
            x = position.x + radius * math.cos(angle)
            y = position.y + radius * math.sin(angle)
            points.append((x, y))
        if width == 0:
            pygame.gfxdraw.filled_polygon(surface, points, color)
        else:
            pygame.draw.lines(surface, color, True, points, width)

    def _draw_glowing_hexagon(self, surface, color, position, radius):
        temp_surf = pygame.Surface((radius * 2.5, radius * 2.5), pygame.SRCALPHA)
        center = (temp_surf.get_width() / 2, temp_surf.get_height() / 2)
        
        for i in range(5, 0, -1):
            alpha = 100 - i * 15
            size = radius + i * 1.5
            glow_color = (*color, alpha)
            
            points = []
            for j in range(6):
                angle = math.pi / 3 * j + math.pi / 6
                x = center[0] + size * math.cos(angle)
                y = center[1] + size * math.sin(angle)
                points.append((x, y))
            pygame.gfxdraw.filled_polygon(temp_surf, points, glow_color)
            pygame.gfxdraw.aapolygon(temp_surf, points, glow_color)
            
        surface.blit(temp_surf, (position.x - center[0], position.y - center[1]))

if __name__ == '__main__':
    # Set the video driver to a real one for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Controls ---
    # Arrows: Move cursor / select letter
    # Space: Place tile / swap letter
    # Shift: Toggle mode
    # Q: Quit
    
    pygame.display.set_caption("Terraform Tessellation")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    # Convert MultiDiscrete to keyboard presses
    action = [0, 0, 0] # no-op, released, released
    
    terminated, truncated = False, False
    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True

        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space and Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)
        
    pygame.quit()
    print(f"Game Over! Final Score: {env.score}, Settlement Size: {len(env.settlement)}")