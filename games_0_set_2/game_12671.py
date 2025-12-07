import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:15:01.307724
# Source Brief: brief_02671.md
# Brief Index: 2671
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Mayan City Builder Gymnasium Environment

    Manages a Mayan city through seasonal cycles. The player builds houses and
    farms on a grid to grow the population, balancing resources against
    seasonal effects.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]` (Movement): 0=none, 1=up, 2=down, 3=left, 4=right (moves the build cursor)
    - `actions[1]` (Space): 0=released, 1=held (builds a House)
    - `actions[2]` (Shift): 0=released, 1=held (builds a Farm)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Reward Structure:**
    - +1.0 for constructing a building.
    - +0.1 for each point of population growth.
    - -0.1 for each point of population decline.
    - +100 for winning (population >= 500).
    - -100 for losing (population <= 0).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manage a Mayan city through seasonal cycles. Build houses and farms on a grid to grow the population, "
        "balancing resources against seasonal effects."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press space to build a house or shift to build a farm."
    )
    auto_advance = False

    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    UI_HEIGHT = 60
    GRID_WIDTH = 10
    GRID_HEIGHT = 6
    TILE_SIZE = 50

    # Colors
    COLOR_BG = (26, 28, 33)
    COLOR_UI_BG = (44, 47, 51)
    COLOR_UI_BORDER = (62, 66, 73)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SUCCESS = (116, 196, 118)
    COLOR_TEXT_FAIL = (240, 101, 101)
    
    COLOR_GRID_LINE = (54, 57, 61)
    COLOR_RIVER = (79, 111, 166)
    COLOR_FERTILE = (89, 133, 90)
    COLOR_EMPTY = (139, 118, 86)
    COLOR_CURSOR = (46, 204, 211)
    COLOR_CURSOR_INVALID = (220, 50, 50)
    
    COLOR_HOUSE = (214, 153, 67)
    COLOR_HOUSE_ROOF = (168, 95, 52)
    COLOR_FARM = (186, 204, 100)
    COLOR_FARM_CROPS = (146, 163, 75)

    # Game Parameters
    MAX_STEPS = 1000
    WIN_POPULATION = 500
    LOSE_POPULATION = 0
    
    INITIAL_POPULATION = 10
    INITIAL_CORN = 100
    
    HOUSE_COST = {'corn': 50}
    FARM_COST = {'corn': 20}
    
    HOUSE_CAPACITY = 5
    CORN_CONSUMPTION_PER_CAPITA = 1
    POP_GROWTH_RATE = 0.1
    POP_STARVATION_RATE = 0.2
    
    FARM_YIELD_BASE = 10
    SEASON_YIELD_MOD = {0: 0.5, 1: 1.0, 2: 2.0} # 0:Dry, 1:Wet, 2:Harvest
    SEASON_NAMES = ["DRY", "WET", "HARVEST"]
    SEASON_COLORS = [(240, 180, 90), (100, 150, 250), (120, 220, 120)]
    
    # Tile Types
    T_EMPTY = 0
    T_FERTILE = 1
    T_RIVER = 2
    T_HOUSE = 3
    T_FARM = 4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        self.grid_pixel_width = self.GRID_WIDTH * self.TILE_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.TILE_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_offset_y = self.UI_HEIGHT + (self.SCREEN_HEIGHT - self.UI_HEIGHT - self.grid_pixel_height) // 2

        self.particles = []

    def _initialize_state_variables(self):
        """Initializes all state variables. Called by reset."""
        self.steps = 0
        self.population = 0
        self.corn = 0
        self.housing_capacity = 0
        self.season = 0
        self.cursor_pos = [0, 0]
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.game_over = False
        self.last_pop_change = 0
        self.last_build_feedback = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._initialize_state_variables()
        
        self.steps = 0
        self.population = self.INITIAL_POPULATION
        self.corn = self.INITIAL_CORN
        self.season = 2 # Start in Harvest for a good start
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.game_over = False
        self.particles.clear()
        
        self._generate_map()
        self._update_housing_capacity()
        
        return self._get_observation(), self._get_info()

    def _generate_map(self):
        self.grid.fill(self.T_EMPTY)
        river_y = self.np_random.integers(1, self.GRID_HEIGHT - 1)
        for x in range(self.GRID_WIDTH):
            self.grid[x, river_y] = self.T_RIVER
            if river_y > 0:
                self.grid[x, river_y - 1] = self.T_FERTILE
            if river_y < self.GRID_HEIGHT - 1:
                self.grid[x, river_y + 1] = self.T_FERTILE

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.last_pop_change = 0
        self.last_build_feedback = ""

        # 1. Handle Player Action
        movement, build_house_action, build_farm_action = action[0], action[1] == 1, action[2] == 1
        self._move_cursor(movement)

        built_something = False
        if build_house_action:
            built_something = self._try_build(self.T_HOUSE, self.HOUSE_COST)
        elif build_farm_action:
            built_something = self._try_build(self.T_FARM, self.FARM_COST)
        
        if built_something:
            reward += 1.0

        # 2. Update Game State (Season & Resources)
        self._advance_season_and_resources()
        
        # 3. Calculate Population-based Reward
        if self.last_pop_change > 0:
            reward += self.last_pop_change * 0.1
        elif self.last_pop_change < 0:
            reward += self.last_pop_change * 0.1 # a negative change will result in negative reward

        # 4. Check for Termination
        self.steps += 1
        terminated = False
        truncated = False
        if self.population >= self.WIN_POPULATION:
            reward += 100
            terminated = True
        elif self.population <= self.LOSE_POPULATION:
            reward -= 100
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Per Gymnasium docs, terminated should also be true if truncated
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _move_cursor(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] -= 1
        elif movement == 2: # Down
            self.cursor_pos[1] += 1
        elif movement == 3: # Left
            self.cursor_pos[0] -= 1
        elif movement == 4: # Right
            self.cursor_pos[0] += 1
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

    def _try_build(self, building_type, cost):
        x, y = self.cursor_pos
        tile = self.grid[x, y]
        
        can_build = False
        if building_type == self.T_HOUSE and tile in [self.T_EMPTY, self.T_FERTILE]:
            can_build = True
        elif building_type == self.T_FARM and tile == self.T_FERTILE:
            can_build = True

        if not can_build:
            self.last_build_feedback = "Invalid Location!"
            return False

        if self.corn >= cost['corn']:
            self.corn -= cost['corn']
            self.grid[x, y] = building_type
            self._update_housing_capacity()
            self._create_particles(x, y, self.COLOR_CURSOR)
            self.last_build_feedback = "Build successful!"
            return True
        else:
            self.last_build_feedback = "Not enough corn!"
            return False

    def _update_housing_capacity(self):
        num_houses = np.count_nonzero(self.grid == self.T_HOUSE)
        self.housing_capacity = num_houses * self.HOUSE_CAPACITY

    def _advance_season_and_resources(self):
        # Advance season
        self.season = (self.season + 1) % 3

        # Calculate corn production
        num_farms = np.count_nonzero(self.grid == self.T_FARM)
        yield_mod = self.SEASON_YIELD_MOD[self.season]
        corn_produced = int(num_farms * self.FARM_YIELD_BASE * yield_mod)
        self.corn += corn_produced

        # Calculate corn consumption
        corn_consumed = int(self.population * self.CORN_CONSUMPTION_PER_CAPITA)
        self.corn -= corn_consumed
        
        # Update population
        prev_pop = self.population
        if self.corn >= 0: # Surplus or balanced
            if self.population < self.housing_capacity:
                growth = max(1, int(self.population * self.POP_GROWTH_RATE))
                self.population = min(self.population + growth, self.housing_capacity)
        else: # Starvation
            starvation = max(1, int(self.population * self.POP_STARVATION_RATE))
            self.population -= starvation
            self.corn = 0 # Can't have negative corn

        self.population = max(0, self.population)
        self.last_pop_change = self.population - prev_pop

    def _get_info(self):
        return {
            "score": self.population,
            "steps": self.steps,
            "corn": self.corn,
            "season": self.SEASON_NAMES[self.season],
            "housing_capacity": self.housing_capacity
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._update_and_draw_particles()
        self._draw_cursor()

    def _draw_grid(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.TILE_SIZE,
                    self.grid_offset_y + y * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                tile_type = self.grid[x, y]
                
                # Draw base tile
                if tile_type == self.T_RIVER:
                    color = self.COLOR_RIVER
                elif tile_type == self.T_FERTILE:
                    color = self.COLOR_FERTILE
                else: # Empty, or built-upon
                    color = self.COLOR_EMPTY
                pygame.draw.rect(self.screen, color, rect)

                # Draw buildings
                if tile_type == self.T_HOUSE:
                    self._draw_house(rect)
                elif tile_type == self.T_FARM:
                    self._draw_farm(rect)

                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)

    def _draw_house(self, rect):
        base_rect = pygame.Rect(rect.left + 5, rect.centery, rect.width - 10, rect.height / 2 - 5)
        pygame.draw.rect(self.screen, self.COLOR_HOUSE, base_rect)
        
        roof_points = [
            (rect.left + 2, rect.centery),
            (rect.centerx, rect.top + 5),
            (rect.right - 2, rect.centery)
        ]
        pygame.gfxdraw.aapolygon(self.screen, roof_points, self.COLOR_HOUSE_ROOF)
        pygame.gfxdraw.filled_polygon(self.screen, roof_points, self.COLOR_HOUSE_ROOF)

    def _draw_farm(self, rect):
        pygame.draw.rect(self.screen, self.COLOR_FARM, rect)
        for i in range(1, 4):
            y_pos = rect.top + i * (rect.height // 4)
            pygame.draw.line(self.screen, self.COLOR_FARM_CROPS, (rect.left + 5, y_pos), (rect.right - 5, y_pos), 2)

    def _draw_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(
            self.grid_offset_x + x * self.TILE_SIZE,
            self.grid_offset_y + y * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        
        alpha = 128 + int(127 * math.sin(pygame.time.get_ticks() * 0.005))
        
        tile = self.grid[x, y]
        can_build_anything = tile in [self.T_EMPTY, self.T_FERTILE]
        color = self.COLOR_CURSOR if can_build_anything else self.COLOR_CURSOR_INVALID

        cursor_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*color, alpha), (0, 0, self.TILE_SIZE, self.TILE_SIZE), 4, border_radius=4)
        self.screen.blit(cursor_surface, rect.topleft)

    def _render_ui(self):
        ui_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_UI_BORDER, (0, self.UI_HEIGHT - 1), (self.SCREEN_WIDTH, self.UI_HEIGHT - 1), 2)

        pop_text = self.font_main.render(f"POP: {self.population}/{self.housing_capacity}", True, self.COLOR_TEXT)
        self.screen.blit(pop_text, (15, 10))
        if self.last_pop_change != 0:
            change_str = f"+{self.last_pop_change}" if self.last_pop_change > 0 else str(self.last_pop_change)
            change_color = self.COLOR_TEXT_SUCCESS if self.last_pop_change > 0 else self.COLOR_TEXT_FAIL
            change_text = self.font_small.render(change_str, True, change_color)
            self.screen.blit(change_text, (20, 35))

        corn_text = self.font_main.render(f"CORN: {int(self.corn)}", True, self.COLOR_TEXT)
        self.screen.blit(corn_text, (200, 10))
        
        season_name = self.SEASON_NAMES[self.season]
        season_color = self.SEASON_COLORS[self.season]
        season_text = self.font_main.render(f"SEASON: {season_name}", True, season_color)
        self.screen.blit(season_text, (380, 10))

        step_text = self.font_main.render(f"DAY: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(step_text, (380, 35))

        if self.last_build_feedback:
            feedback_text = self.font_small.render(self.last_build_feedback, True, self.COLOR_TEXT)
            self.screen.blit(feedback_text, (200, 35))

    def _create_particles(self, grid_x, grid_y, color):
        cx = self.grid_offset_x + grid_x * self.TILE_SIZE + self.TILE_SIZE // 2
        cy = self.grid_offset_y + grid_y * self.TILE_SIZE + self.TILE_SIZE // 2
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            particle = {
                'pos': [cx, cy],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(20, 40),
                'color': color,
                'radius': random.uniform(2, 5)
            }
            self.particles.append(particle)
            
    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)
            else:
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you need to unset the dummy video driver
    # and install pygame: pip install pygame
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Mayan City Builder")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Arrows: Move cursor")
    print("Space: Build House")
    print("Shift: Build Farm")
    print("Enter: Do nothing (pass turn)")
    print("Q: Quit")
    
    running = True
    while running:
        action_taken = False
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                
                if event.key == pygame.K_RETURN:
                    action = [0, 0, 0]
                    action_taken = True
                elif event.key == pygame.K_SPACE:
                    action = [0, 1, 0]
                    action_taken = True
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action = [0, 0, 1]
                    action_taken = True
                elif event.key == pygame.K_UP:
                    action = [1, 0, 0]
                    action_taken = True
                elif event.key == pygame.K_DOWN:
                    action = [2, 0, 0]
                    action_taken = True
                elif event.key == pygame.K_LEFT:
                    action = [3, 0, 0]
                    action_taken = True
                elif event.key == pygame.K_RIGHT:
                    action = [4, 0, 0]
                    action_taken = True

        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step: {info['steps']}, Pop: {info['score']}, Corn: {info['corn']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

            if terminated or truncated:
                print("Game Over!")
                print(f"Final Score (Population): {info['score']}")
                print(f"Total Reward: {total_reward}")
                obs, info = env.reset()
                total_reward = 0

        # Render the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10)

    env.close()