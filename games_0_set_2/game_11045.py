import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:24:20.612912
# Source Brief: brief_01045.md
# Brief Index: 1045
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
        "Build a thriving civilization on a hex grid by answering riddles to unlock resources. "
        "Grow your population to win before the world shrinks or time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor during placement. "
        "Press space to confirm your choice or place a resource. Use shift to cycle through riddle answers."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Visuals & Game Constants
        self._init_visuals()
        self._init_constants()
        self._init_riddles()

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.population = 0
        self.grid_radius = 0
        self.hex_grid = {}
        self.cursor_q, self.cursor_r = 0, 0
        self.game_phase = "RIDDLE" # or "PLACEMENT"
        self.current_riddle = {}
        self.riddle_answer_idx = 0
        self.resource_to_place = None
        self.riddles_solved = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.available_resources = []
        self.target_population = 1000
        self.max_steps = 2000
        
        self.reset()
        # self.validate_implementation() # This is a self-check, can be commented out

    def _init_visuals(self):
        """Initialize colors and fonts."""
        self.COLOR_BG_TOP = (15, 25, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 70)
        self.COLOR_GRID_LINE = (60, 80, 110)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_TEXT_SHADOW = (10, 15, 20)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UNUSABLE = (40, 45, 50)
        self.COLOR_EMPTY = (25, 35, 55)
        
        self.RES_COLORS = {
            "RESIDENTIAL": (240, 220, 100),
            "FERTILE": (50, 200, 80),
            "WATER": (60, 120, 255),
            "MINING": (180, 100, 80),
        }
        
        self.FONT_UI = pygame.font.Font(None, 36)
        self.FONT_RIDDLE = pygame.font.Font(None, 28)
        self.FONT_RIDDLE_OPTIONS = pygame.font.Font(None, 24)

    def _init_constants(self):
        """Initialize game constants."""
        self.hex_size = 20
        self.grid_center_x = self.screen_width // 2
        self.grid_center_y = self.screen_height // 2 - 20
        self.initial_grid_radius = 5

    def _init_riddles(self):
        """Initialize the riddle database."""
        self.riddles = [
            {'text': "A new settlement needs a home.", 'options': ['RESIDENTIAL'], 'correct': 0},
            {'text': "To grow, a city needs people.", 'options': ['RESIDENTIAL'], 'correct': 0},
            {'text': "Growth requires fertile ground.", 'options': ['FERTILE', 'WATER'], 'correct': 0},
            {'text': "Crops cannot grow without soil.", 'options': ['FERTILE', 'MINING'], 'correct': 0},
            {'text': "A thirsty farm grows best.", 'options': ['WATER', 'FERTILE'], 'correct': 0},
            {'text': "Rivers bring life to the land.", 'options': ['WATER', 'MINING'], 'correct': 0},
            {'text': "Industry requires raw materials.", 'options': ['MINING', 'WATER'], 'correct': 0},
            {'text': "Wealth is dug from the earth.", 'options': ['MINING', 'FERTILE'], 'correct': 0},
            {'text': "A city needs a foundation.", 'options': ['RESIDENTIAL', 'WATER'], 'correct': 0},
            {'text': "Expand the population.", 'options': ['RESIDENTIAL', 'MINING'], 'correct': 0},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.population = 0
        self.riddles_solved = 0
        
        self.grid_radius = self.initial_grid_radius
        self.hex_grid = {}
        for q in range(-self.grid_radius, self.grid_radius + 1):
            for r in range(-self.grid_radius, self.grid_radius + 1):
                if -q - r >= -self.grid_radius and -q - r <= self.grid_radius:
                    self.hex_grid[(q, r)] = {'type': 'EMPTY', 'pop': 0}

        # Place initial residential hex
        self.hex_grid[(0, 0)] = {'type': 'RESIDENTIAL', 'pop': 10}
        self.cursor_q, self.cursor_r = 0, 1
        
        self._update_population()
        self._update_available_resources()
        self._generate_riddle()
        
        self.game_phase = "RIDDLE"
        self.last_space_held = True # Prevent action on first frame
        self.last_shift_held = True
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        reward = 0
        self.steps += 1

        if self.game_phase == "RIDDLE":
            if shift_pressed:
                # SFX: UI_CYCLE
                self.riddle_answer_idx = (self.riddle_answer_idx + 1) % len(self.current_riddle['options'])
            if space_pressed:
                if self.riddle_answer_idx == self.current_riddle['correct']:
                    # SFX: RIDDLE_CORRECT
                    reward += 5
                    self.score += 5
                    self.riddles_solved += 1
                    self.resource_to_place = self.current_riddle['options'][self.riddle_answer_idx]
                    self.game_phase = "PLACEMENT"
                    self._update_available_resources()
                else:
                    # SFX: RIDDLE_INCORRECT
                    reward -= 1 # Small penalty for wrong answer
                    self.score -= 1
                    self._generate_riddle() # New riddle
        
        elif self.game_phase == "PLACEMENT":
            self._handle_movement(movement)
            if space_pressed:
                reward += self._place_resource()

        # Update game state
        if self.steps > 0 and self.steps % 50 == 0:
            self._shrink_grid()
            # SFX: GRID_SHRINK

        terminated = self._check_termination()
        if terminated:
            if self.population >= self.target_population:
                reward += 100 # Victory bonus
                self.score += 100
            else:
                reward -= 100 # Loss penalty
                self.score -= 100
            self.game_over = True
        
        # Penalize for being slow
        reward -= 0.01 
        self.score -= 0.01

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        truncated = self.steps >= self.max_steps
        if truncated:
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement):
        """Updates cursor based on movement action."""
        # SFX: CURSOR_MOVE
        q, r = self.cursor_q, self.cursor_r
        if movement == 1: r -= 1 # Up
        elif movement == 2: r += 1 # Down
        elif movement == 3: q -= 1 # Left
        elif movement == 4: q += 1 # Right
        
        # Keep cursor within the active grid
        dist = (abs(q) + abs(r) + abs(-q - r)) / 2
        if dist <= self.grid_radius:
            self.cursor_q, self.cursor_r = q, r

    def _place_resource(self):
        """Places the current resource at the cursor and calculates rewards."""
        q, r = self.cursor_q, self.cursor_r
        if self.hex_grid.get((q, r), {}).get('type') == 'EMPTY':
            # SFX: PLACE_RESOURCE
            self.hex_grid[(q, r)] = {'type': self.resource_to_place, 'pop': 0}
            
            pop_before = self.population
            self._update_population()
            pop_gain = self.population - pop_before
            
            adj_penalty = self._calculate_adjacency_penalty(q, r)
            
            self._generate_riddle()
            self.game_phase = "RIDDLE"
            
            return pop_gain + adj_penalty
        # SFX: PLACE_FAIL
        return 0

    def _update_population(self):
        """Recalculates the entire population based on adjacencies."""
        self.population = 0
        # Reset all pops first
        for hex_data in self.hex_grid.values():
            if hex_data['type'] == 'RESIDENTIAL':
                hex_data['pop'] = 0

        # Calculate new pops
        for (q, r), hex_data in self.hex_grid.items():
            if hex_data['type'] == 'RESIDENTIAL':
                base_pop = 10
                bonus = 0
                for nq, nr in self._get_neighbors(q, r):
                    neighbor = self.hex_grid.get((nq, nr))
                    if neighbor:
                        if neighbor['type'] == 'WATER': bonus += 5
                        elif neighbor['type'] == 'FERTILE': bonus += 3
                        elif neighbor['type'] == 'MINING': bonus += 2
                hex_data['pop'] = base_pop + bonus
                self.population += hex_data['pop']

    def _calculate_adjacency_penalty(self, q, r):
        """Calculates penalty for placing next to unusable hexes."""
        penalty = 0
        for nq, nr in self._get_neighbors(q, r):
            neighbor = self.hex_grid.get((nq, nr))
            if neighbor is None or neighbor['type'] == 'UNUSABLE':
                penalty -= 0.1
        return penalty

    def _shrink_grid(self):
        """Shrinks the usable grid by one layer."""
        if self.grid_radius > 1:
            self.grid_radius -= 1
            for (q, r), hex_data in self.hex_grid.items():
                dist = (abs(q) + abs(r) + abs(-q - r)) / 2
                if dist > self.grid_radius:
                    hex_data['type'] = 'UNUSABLE'
            self._update_population() # Population might change if resources are lost

    def _check_termination(self):
        """Checks for win, loss, or max steps."""
        if self.population >= self.target_population:
            return True
        
        # Check if there are any empty hexes left
        for (q, r), hex_data in self.hex_grid.items():
            dist = (abs(q) + abs(r) + abs(-q - r)) / 2
            if dist <= self.grid_radius and hex_data['type'] == 'EMPTY':
                return False # Found a valid spot
        
        return True # No valid spots left

    def _update_available_resources(self):
        """Updates the list of resources player can get from riddles."""
        self.available_resources = ["RESIDENTIAL", "FERTILE"]
        if self.riddles_solved >= 5:
            self.available_resources.append("WATER")
        if self.riddles_solved >= 10:
            self.available_resources.append("MINING")

    def _generate_riddle(self):
        """Generates a new riddle for the player."""
        possible_riddles = [r for r in self.riddles if set(r['options']).issubset(self.available_resources)]
        if not possible_riddles: # Fallback
            possible_riddles = self.riddles
        self.current_riddle = random.choice(possible_riddles)
        self.riddle_answer_idx = 0

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        """Draws a vertical gradient background."""
        for y in range(self.screen_height):
            ratio = y / self.screen_height
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))

    def _render_game(self):
        """Renders the hexagonal grid."""
        # Draw all hexes
        for (q, r), hex_data in self.hex_grid.items():
            dist = (abs(q) + abs(r) + abs(-q - r)) / 2
            if dist <= self.initial_grid_radius:
                pixel_x, pixel_y = self._hex_to_pixel(q, r)
                corners = self._get_hex_corners(pixel_x, pixel_y, self.hex_size)
                
                res_type = hex_data['type']
                if res_type == 'UNUSABLE':
                    color = self.COLOR_UNUSABLE
                elif res_type == 'EMPTY':
                    color = self.COLOR_EMPTY
                else:
                    color = self.RES_COLORS.get(res_type, (255,0,255))

                pygame.gfxdraw.filled_polygon(self.screen, corners, color)
                pygame.gfxdraw.aapolygon(self.screen, corners, self.COLOR_GRID_LINE)
        
        # Draw cursor
        if self.game_phase == "PLACEMENT":
            cursor_x, cursor_y = self._hex_to_pixel(self.cursor_q, self.cursor_r)
            # Glow effect
            for i in range(5, 0, -1):
                glow_size = self.hex_size + i * 1.5
                glow_alpha = 60 - i * 10
                glow_corners = self._get_hex_corners(cursor_x, cursor_y, glow_size)
                pygame.gfxdraw.aapolygon(self.screen, glow_corners, (*self.COLOR_CURSOR, glow_alpha))
            
            cursor_corners = self._get_hex_corners(cursor_x, cursor_y, self.hex_size)
            pygame.gfxdraw.aapolygon(self.screen, cursor_corners, self.COLOR_CURSOR)
            pygame.gfxdraw.aapolygon(self.screen, cursor_corners, self.COLOR_CURSOR)

    def _render_ui(self):
        """Renders UI elements like score, population, and riddles."""
        # Population Display
        pop_text = f"POP: {self.population} / {self.target_population}"
        self._draw_text(pop_text, (20, 20), self.FONT_UI, self.COLOR_TEXT, shadow=True)

        # Score Display
        score_text = f"SCORE: {self.score:.0f}"
        self._draw_text(score_text, (20, 55), self.FONT_UI, self.COLOR_TEXT, shadow=True)
        
        # Step Display
        step_text = f"STEP: {self.steps} / {self.max_steps}"
        self._draw_text(step_text, (self.screen_width - 180, 20), self.FONT_UI, self.COLOR_TEXT, shadow=True)

        # Game Phase / Riddle Display
        ui_box_rect = pygame.Rect(10, self.screen_height - 80, self.screen_width - 20, 70)
        pygame.draw.rect(self.screen, (*self.COLOR_BG_TOP, 200), ui_box_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, ui_box_rect, 2, border_radius=10)

        if self.game_phase == "RIDDLE":
            self._draw_text(self.current_riddle['text'], (25, self.screen_height - 70), self.FONT_RIDDLE, self.COLOR_TEXT)
            options = self.current_riddle['options']
            for i, option in enumerate(options):
                color = self.COLOR_CURSOR if i == self.riddle_answer_idx else self.COLOR_TEXT
                prefix = "> " if i == self.riddle_answer_idx else "  "
                self._draw_text(f"{prefix}{option}", (40 + i * 150, self.screen_height - 45), self.FONT_RIDDLE_OPTIONS, color)
        
        elif self.game_phase == "PLACEMENT":
            text = f"Place {self.resource_to_place}"
            self._draw_text(text, (25, self.screen_height - 60), self.FONT_UI, self.RES_COLORS[self.resource_to_place])

    def _draw_text(self, text, pos, font, color, shadow=False):
        if shadow:
            shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "population": self.population,
            "grid_radius": self.grid_radius,
            "riddles_solved": self.riddles_solved,
            "game_phase": self.game_phase,
        }

    # --- Hexagon Math Helpers ---
    def _hex_to_pixel(self, q, r):
        x = self.hex_size * (3./2. * q) + self.grid_center_x
        y = self.hex_size * (math.sqrt(3)/2. * q + math.sqrt(3) * r) + self.grid_center_y
        return int(x), int(y)

    def _get_hex_corners(self, center_x, center_y, size):
        corners = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            corners.append(
                (int(center_x + size * math.cos(angle_rad)),
                 int(center_y + size * math.sin(angle_rad)))
            )
        return corners

    def _get_neighbors(self, q, r):
        return [
            (q + 1, r), (q - 1, r),
            (q, r + 1), (q, r - 1),
            (q + 1, r - 1), (q - 1, r + 1)
        ]

    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop for human play
    running = True
    while running:
        # Default action is "do nothing"
        action = [0, 0, 0] # [movement, space, shift]
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.0f}, Population: {info['population']}")
            # Optional: auto-reset after a delay
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        # Render the environment to the screen
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # Create a display if one doesn't exist
        try:
            display_surface = pygame.display.get_surface()
            if display_surface is None:
                raise Exception
        except Exception:
            display_surface = pygame.display.set_mode((env.screen_width, env.screen_height))
            pygame.display.set_caption("Hex Civilization")

        display_surface.blit(render_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()