import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Arrow keys to move the cursor. "
        "Shift to cycle crystal type. Space to place a crystal."
    )

    game_description = (
        "Place light-refracting crystals to illuminate all gems before time or crystals run out. "
        "A turn passes only when you place a crystal."
    )

    auto_advance = False

    # --- Constants ---
    # Game parameters
    GRID_WIDTH = 20
    GRID_HEIGHT = 15
    TILE_WIDTH_HALF = 16
    TILE_HEIGHT_HALF = 8
    MAX_BOUNCES = 15
    STARTING_TIME = 60
    NUM_GEMS = 5

    # Crystal Types
    CRYSTAL_BEND = 0
    CRYSTAL_SPLIT = 1
    CRYSTAL_TYPES = [CRYSTAL_BEND, CRYSTAL_SPLIT]
    CRYSTAL_NAMES = {CRYSTAL_BEND: "Bender", CRYSTAL_SPLIT: "Splitter"}
    
    # Directions (Grid-based)
    # Using complex numbers for easy rotation: N, E, S, W
    DIRECTIONS = {'N': 0 - 1j, 'E': 1 + 0j, 'S': 0 + 1j, 'W': -1 + 0j}
    DIR_VECTORS = list(DIRECTIONS.values())
    DIR_NAMES = list(DIRECTIONS.keys())

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 60)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_BEAM = (255, 255, 100)
    
    GEM_COLORS = {
        'unlit': (80, 80, 100),
        'lit': (100, 255, 255),
        'lit_glow': (100, 255, 255, 50)
    }
    CRYSTAL_COLORS = {
        CRYSTAL_BEND: ((180, 100, 255), (120, 60, 200)),
        CRYSTAL_SPLIT: ((255, 100, 100), (200, 60, 60)),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width = 640
        self.screen_height = 400
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_crystal_select = pygame.font.Font(None, 20)

        self.origin_x = self.screen_width // 2
        self.origin_y = 80
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = None
        self.light_source_pos = None
        self.gems = []
        self.crystals = {}
        self.crystal_inventory = {}
        self.time_left = 0
        self.selected_crystal_type = self.CRYSTAL_BEND
        self.light_beams = []
        self.lit_gem_positions = set()
        self.prev_shift_state = 0
        self.prev_space_state = 0
        self.np_random = None

    def _iso_to_cart(self, iso_pos):
        iso_x, iso_y = iso_pos
        cart_x = self.origin_x + (iso_x - iso_y) * self.TILE_WIDTH_HALF
        cart_y = self.origin_y + (iso_x + iso_y) * self.TILE_HEIGHT_HALF
        return int(cart_x), int(cart_y)

    def _generate_level(self):
        self.light_source_pos = (self.GRID_WIDTH // 2, 0)
        
        possible_gem_positions = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                pos = (c, r)
                if pos != self.light_source_pos:
                    possible_gem_positions.append(pos)
        
        self.np_random.shuffle(possible_gem_positions)
        
        self.gems = [{'pos': pos, 'lit': False} for pos in possible_gem_positions[:self.NUM_GEMS]]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self._generate_level()
        self.crystals = {}
        self.crystal_inventory = {self.CRYSTAL_BEND: 8, self.CRYSTAL_SPLIT: 4}
        self.time_left = self.STARTING_TIME
        self.selected_crystal_type = self.CRYSTAL_BEND
        
        self.prev_shift_state = 0
        self.prev_space_state = 0
        
        self._update_light_paths()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_val, shift_val = action
        reward = 0.0
        
        # --- Handle Actions ---
        # 1. Cursor Movement (always processed)
        cx, cy = self.cursor_pos
        if movement == 1: # Up (NW)
            cy -= 1
        elif movement == 2: # Down (SE)
            cy += 1
        elif movement == 3: # Left (SW)
            cx -= 1
        elif movement == 4: # Right (NE)
            cx += 1
        self.cursor_pos = (max(0, min(self.GRID_WIDTH - 1, cx)), max(0, min(self.GRID_HEIGHT - 1, cy)))

        # 2. Cycle Crystal Type (on press, not hold)
        if shift_val == 1 and self.prev_shift_state == 0:
            current_index = self.CRYSTAL_TYPES.index(self.selected_crystal_type)
            next_index = (current_index + 1) % len(self.CRYSTAL_TYPES)
            self.selected_crystal_type = self.CRYSTAL_TYPES[next_index]
        self.prev_shift_state = shift_val

        # 3. Place Crystal (on press, not hold)
        if space_val == 1 and self.prev_space_state == 0:
            # A "turn" happens now
            self.steps += 1
            
            can_place = (
                self.cursor_pos not in self.crystals
                and self.cursor_pos != self.light_source_pos
                and not any(g['pos'] == self.cursor_pos for g in self.gems)
                and self.crystal_inventory[self.selected_crystal_type] > 0
            )
            
            if can_place:
                # Place crystal
                self.crystals[self.cursor_pos] = {'type': self.selected_crystal_type}
                self.crystal_inventory[self.selected_crystal_type] -= 1
                self.time_left -= 1
                # sfx: crystal_place.wav

                # Calculate reward
                prev_lit_count = len(self.lit_gem_positions)
                self._update_light_paths()
                newly_lit_count = len(self.lit_gem_positions) - prev_lit_count
                
                if newly_lit_count > 0:
                    reward += newly_lit_count * 5.0 # Reward for illuminating new gems
                    # sfx: gem_lit.wav
                else:
                    reward -= 1.0 # Penalty for useless placement
            else:
                reward -= 0.5 # Small penalty for trying to place illegally
                # sfx: error.wav

        self.prev_space_state = space_val

        # --- Check Termination Conditions ---
        all_gems_lit = len(self.lit_gem_positions) == len(self.gems)
        no_time = self.time_left <= 0
        no_crystals = all(count == 0 for count in self.crystal_inventory.values())
        
        terminated = False
        if all_gems_lit:
            reward += 100.0
            terminated = True
            self.game_over = True
            # sfx: victory.wav
        elif no_time or no_crystals:
            reward -= 100.0
            terminated = True
            self.game_over = True
            # sfx: failure.wav

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_light_paths(self):
        self.light_beams = []
        self.lit_gem_positions.clear()
        for g in self.gems:
            g['lit'] = False

        initial_ray = (self.light_source_pos, self.DIRECTIONS['S'])
        active_rays = [initial_ray]

        processed_rays = set()

        while active_rays:
            pos, direction = active_rays.pop(0)
            
            # The complex number representing the direction is hashable and can be used directly.
            if (pos, direction) in processed_rays:
                continue
            processed_rays.add((pos, direction))

            beam_path = [pos]
            current_pos = pos
            
            for _ in range(self.MAX_BOUNCES):
                next_pos = (int(current_pos[0] + direction.real), int(current_pos[1] + direction.imag))

                # Check boundaries
                if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                    beam_path.append(next_pos)
                    break

                # Check for gems
                for gem in self.gems:
                    if gem['pos'] == next_pos:
                        gem['lit'] = True
                        self.lit_gem_positions.add(gem['pos'])
                
                # Check for crystals
                if next_pos in self.crystals:
                    beam_path.append(next_pos)
                    crystal = self.crystals[next_pos]
                    
                    if crystal['type'] == self.CRYSTAL_BEND:
                        # 90 degree clockwise bend
                        new_direction = direction * -1j 
                        direction = new_direction
                        current_pos = next_pos
                    elif crystal['type'] == self.CRYSTAL_SPLIT:
                        # Split: one continues, one bends 90 deg clockwise
                        new_ray_dir = direction * -1j
                        active_rays.append((next_pos, new_ray_dir))
                        current_pos = next_pos # Continue straight
                    else:
                        current_pos = next_pos # Pass through
                else:
                    current_pos = next_pos
            
            if len(beam_path) > 1:
                self.light_beams.append(beam_path)


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid lines
        for r in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_cart((0, r))
            end = self._iso_to_cart((self.GRID_WIDTH, r))
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for c in range(self.GRID_WIDTH + 1):
            start = self._iso_to_cart((c, 0))
            end = self._iso_to_cart((c, self.GRID_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Render light source
        ls_points = [
            self._iso_to_cart((self.light_source_pos[0], self.light_source_pos[1])),
            self._iso_to_cart((self.light_source_pos[0] + 1, self.light_source_pos[1])),
            self._iso_to_cart((self.light_source_pos[0] + 1, self.light_source_pos[1] + 1)),
            self._iso_to_cart((self.light_source_pos[0], self.light_source_pos[1] + 1)),
        ]
        pygame.draw.polygon(self.screen, self.COLOR_BEAM, ls_points)

        # Render light beams
        for beam_path in self.light_beams:
            if len(beam_path) < 2: continue
            
            path_cart = [self._iso_to_cart(p) for p in beam_path]
            for i in range(len(path_cart) - 1):
                p1 = path_cart[i]
                p2 = path_cart[i+1]
                # Glow effect
                pygame.draw.aaline(self.screen, (255, 255, 150, 50), p1, p2, 4)
                pygame.draw.aaline(self.screen, (255, 255, 150, 100), p1, p2, 2)
                pygame.draw.aaline(self.screen, (255, 255, 200), p1, p2, 1)


        # Render gems
        for gem in self.gems:
            color = self.GEM_COLORS['lit'] if gem['lit'] else self.GEM_COLORS['unlit']
            pos_cart = self._iso_to_cart(gem['pos'])
            diamond_points = [
                (pos_cart[0], pos_cart[1] - self.TILE_HEIGHT_HALF),
                (pos_cart[0] + self.TILE_WIDTH_HALF / 2, pos_cart[1]),
                (pos_cart[0], pos_cart[1] + self.TILE_HEIGHT_HALF),
                (pos_cart[0] - self.TILE_WIDTH_HALF / 2, pos_cart[1]),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, diamond_points, color)
            pygame.gfxdraw.aapolygon(self.screen, diamond_points, color)
            if gem['lit']:
                 pygame.gfxdraw.filled_circle(self.screen, pos_cart[0], pos_cart[1], 10, self.GEM_COLORS['lit_glow'])


        # Render crystals
        for pos, crystal in self.crystals.items():
            colors = self.CRYSTAL_COLORS[crystal['type']]
            center_cart = self._iso_to_cart(pos)
            
            top_point = (center_cart[0], center_cart[1] - self.TILE_HEIGHT_HALF)
            right_point = (center_cart[0] + self.TILE_WIDTH_HALF, center_cart[1])
            bottom_point = (center_cart[0], center_cart[1] + self.TILE_HEIGHT_HALF)
            left_point = (center_cart[0] - self.TILE_WIDTH_HALF, center_cart[1])
            
            pygame.gfxdraw.filled_polygon(self.screen, [top_point, right_point, bottom_point], colors[1])
            pygame.gfxdraw.filled_polygon(self.screen, [top_point, left_point, bottom_point], colors[0])
            pygame.gfxdraw.aapolygon(self.screen, [top_point, right_point, bottom_point, left_point], (200,200,220))


        # Render cursor
        cursor_points = [
            self._iso_to_cart((self.cursor_pos[0], self.cursor_pos[1])),
            self._iso_to_cart((self.cursor_pos[0] + 1, self.cursor_pos[1])),
            self._iso_to_cart((self.cursor_pos[0] + 1, self.cursor_pos[1] + 1)),
            self._iso_to_cart((self.cursor_pos[0], self.cursor_pos[1] + 1)),
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, cursor_points, 2)

    def _render_ui(self):
        # Time remaining
        time_text = self.font_ui.render(f"Turns Left: {self.time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.screen_width - time_text.get_width() - 10, 10))

        # Crystals remaining
        y_offset = 10
        for c_type in self.CRYSTAL_TYPES:
            count = self.crystal_inventory[c_type]
            name = self.CRYSTAL_NAMES[c_type]
            is_selected = self.selected_crystal_type == c_type
            
            text_color = self.COLOR_TEXT if not is_selected else self.COLOR_BEAM
            text = self.font_ui.render(f"{name}: {count}", True, text_color)
            self.screen.blit(text, (10, y_offset))
            if is_selected:
                pygame.draw.rect(self.screen, self.COLOR_BEAM, (6, y_offset-2, text.get_width()+8, text.get_height()+4), 1)
            y_offset += 25
        
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.screen_width//2 - score_text.get_width()//2, 10))
        
        # Gems lit
        gems_lit_text = self.font_ui.render(f"Gems Lit: {len(self.lit_gem_positions)}/{self.NUM_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(gems_lit_text, (self.screen_width//2 - gems_lit_text.get_width()//2, 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "gems_lit": len(self.lit_gem_positions),
            "crystals_left": sum(self.crystal_inventory.values()),
        }

    def close(self):
        pygame.quit()