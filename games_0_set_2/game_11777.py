import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:35:11.695370
# Source Brief: brief_01777.md
# Brief Index: 1777
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Cultivate a vibrant kelp forest on the seabed. Plant different kelp species, create matches to earn resources, and attract diverse marine life to build a thriving ecosystem."
    user_guide = "Use ←→ arrow keys to move the cursor and ↑↓ to select a kelp type. Press space to plant kelp and shift to match adjacent kelp of the same type."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 20
    CELL_WIDTH = SCREEN_WIDTH // GRID_COLS
    SEABED_Y = 370
    MAX_STEPS = 2000

    # Game Rules
    INITIAL_RESOURCES = 50
    BASE_PLANT_COST = 2
    COST_INCREASE_INTERVAL = 25
    WIN_CONDITION_HEIGHT = 1500
    WIN_CONDITION_SPECIES = 3

    # Colors
    COLOR_BG_TOP = (10, 25, 50)
    COLOR_BG_BOTTOM = (5, 10, 30)
    COLOR_SEABED = (40, 30, 20)
    COLOR_CURSOR = (255, 255, 0, 100) # RGBA for transparency
    COLOR_UI_TEXT = (220, 220, 255)
    
    KELP_TYPES = {
        "blue": {"color": (50, 100, 255), "growth_rate": 0.05, "match_reward": 2},
        "green": {"color": (50, 220, 100), "growth_rate": 0.08, "match_reward": 4},
        "red": {"color": (255, 80, 80), "growth_rate": 0.12, "match_reward": 6},
    }
    KELP_COLOR_NAMES = list(KELP_TYPES.keys())

    SPECIES_DATA = {
        "Guppy": {"color": (200, 200, 200), "trigger": lambda k: k["blue"] >= 3},
        "Clownfish": {"color": (255, 150, 50), "trigger": lambda k: k["red"] >= 3 and k["green"] >= 1},
        "Angelfish": {"color": (255, 220, 0), "trigger": lambda k: k["green"] >= 5},
    }

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
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 24)
        
        self.render_mode = render_mode
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.resources = 0
        self.plant_cost = 0
        self.kelp_planted_count = 0
        self.total_kelp_height = 0
        self.kelp_grid = []
        self.cursor_pos = 0
        self.selected_kelp_index = 0
        self.marine_life = []
        self.attracted_species = set()
        self.particles = []
        self.god_rays = []
        self.last_space_held = False
        self.last_shift_held = False
        
        # self.reset() is called by the test suite, no need to call it here.
        # However, we need to initialize state for validate_implementation
        self._initialize_state()
        
        self.validate_implementation()

    def _initialize_state(self):
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.resources = self.INITIAL_RESOURCES
        self.plant_cost = self.BASE_PLANT_COST
        self.kelp_planted_count = 0
        self.total_kelp_height = 0
        
        self.kelp_grid = [None] * self.GRID_COLS
        self.cursor_pos = self.GRID_COLS // 2
        self.selected_kelp_index = 0

        self.marine_life = []
        self.attracted_species = set()
        self.particles = []
        
        self.last_space_held = False
        self.last_shift_held = False

        self._create_god_rays()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- 1. Process Actions ---
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        # Handle movement and kelp selection
        if movement == 1: # Up
            self.selected_kelp_index = (self.selected_kelp_index + 1) % len(self.KELP_COLOR_NAMES)
        elif movement == 2: # Down
            self.selected_kelp_index = (self.selected_kelp_index - 1 + len(self.KELP_COLOR_NAMES)) % len(self.KELP_COLOR_NAMES)
        elif movement == 3: # Left
            self.cursor_pos = max(0, self.cursor_pos - 1)
        elif movement == 4: # Right
            self.cursor_pos = min(self.GRID_COLS - 1, self.cursor_pos + 1)
            
        # Handle planting
        if space_press:
            reward += self._plant_kelp()
            
        # Handle matching
        if shift_press:
            reward += self._match_kelp()

        # --- 2. Update Game State ---
        self.steps += 1
        self._update_kelp()
        self._update_particles()
        self._update_marine_life()
        
        # --- 3. Check for Events & Calculate Rewards ---
        new_species_reward = self._check_for_new_species()
        reward += new_species_reward
        
        # --- 4. Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.total_kelp_height >= self.WIN_CONDITION_HEIGHT and len(self.attracted_species) >= self.WIN_CONDITION_SPECIES:
                reward += 100 # Win
            else:
                reward -= 100 # Loss
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Game Logic Helpers ---
    def _plant_kelp(self):
        if self.kelp_grid[self.cursor_pos] is None and self.resources >= self.plant_cost:
            self.resources -= self.plant_cost
            kelp_type = self.KELP_COLOR_NAMES[self.selected_kelp_index]
            x_pos = self.cursor_pos * self.CELL_WIDTH + self.CELL_WIDTH / 2
            self.kelp_grid[self.cursor_pos] = Kelp(x_pos, kelp_type, self.KELP_TYPES[kelp_type])
            self.kelp_planted_count += 1
            if self.kelp_planted_count % self.COST_INCREASE_INTERVAL == 0:
                self.plant_cost += 1
            # sfx: gentle 'plop' sound
            return 0.1 # Small reward for a successful action
        return 0

    def _match_kelp(self):
        kelp = self.kelp_grid[self.cursor_pos]
        if kelp is None or kelp.height < 5:
            return 0

        match_reward = 0
        match_count = 1
        
        # Check left
        if self.cursor_pos > 0:
            left_kelp = self.kelp_grid[self.cursor_pos - 1]
            if left_kelp and left_kelp.type == kelp.type:
                left_kelp.grow(kelp.height * 0.5)
                match_count += 1
        
        # Check right
        if self.cursor_pos < self.GRID_COLS - 1:
            right_kelp = self.kelp_grid[self.cursor_pos + 1]
            if right_kelp and right_kelp.type == kelp.type:
                right_kelp.grow(kelp.height * 0.5)
                match_count += 1

        if match_count > 1:
            resource_gain = int(kelp.height * 0.2 * self.KELP_TYPES[kelp.type]["match_reward"])
            self.resources += resource_gain
            match_reward = resource_gain # Reward is equal to resources gained
            kelp.grow(kelp.height * 0.5 * (match_count - 1))
            self._create_particles(kelp.x, self.SEABED_Y - kelp.height, kelp.color, resource_gain)
            # sfx: shimmering 'chime' sound
        return match_reward

    def _update_kelp(self):
        self.total_kelp_height = 0
        for kelp in self.kelp_grid:
            if kelp:
                kelp.update(self.steps)
                self.total_kelp_height += kelp.height

    def _check_for_new_species(self):
        reward = 0
        kelp_counts = {"blue": 0, "green": 0, "red": 0}
        for kelp in self.kelp_grid:
            if kelp and kelp.height > 20: # Only count established kelp
                kelp_counts[kelp.type] += 1
        
        for species_name, data in self.SPECIES_DATA.items():
            if species_name not in self.attracted_species and data["trigger"](kelp_counts):
                self.attracted_species.add(species_name)
                x = random.uniform(50, self.SCREEN_WIDTH - 50)
                y = random.uniform(50, self.SEABED_Y - 50)
                self.marine_life.append(Fish((x, y), species_name, data["color"]))
                reward += 5 # Event-based reward for new species
                # sfx: happy 'bloop' discovery sound
        return reward

    def _check_termination(self):
        win = self.total_kelp_height >= self.WIN_CONDITION_HEIGHT and len(self.attracted_species) >= self.WIN_CONDITION_SPECIES
        
        can_plant = self.resources >= self.plant_cost
        can_match = any(self._can_match_at(i) for i in range(self.GRID_COLS))
        loss = not can_plant and not can_match

        max_steps_reached = self.steps >= self.MAX_STEPS
        
        return win or loss or max_steps_reached

    def _can_match_at(self, i):
        kelp = self.kelp_grid[i]
        if not kelp: return False
        
        left_kelp = self.kelp_grid[i-1] if i > 0 else None
        right_kelp = self.kelp_grid[i+1] if i < self.GRID_COLS - 1 else None

        return (left_kelp and left_kelp.type == kelp.type) or \
               (right_kelp and right_kelp.type == kelp.type)

    # --- Rendering ---
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Gradient background
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        self._render_god_rays()

        # Seabed
        pygame.draw.rect(self.screen, self.COLOR_SEABED, (0, self.SEABED_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.SEABED_Y))

    def _render_game(self):
        for particle in self.particles:
            particle.draw(self.screen)
        for kelp in self.kelp_grid:
            if kelp:
                kelp.draw(self.screen, self.SEABED_Y, self.steps)
        for fish in self.marine_life:
            fish.draw(self.screen)
        self._render_cursor()

    def _render_ui(self):
        # Resources
        res_text = f"Resources: {self.resources}"
        res_surf = self.font_large.render(res_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(res_surf, (10, 10))

        # Forest Size
        height_text = f"Forest Size: {int(self.total_kelp_height)} / {self.WIN_CONDITION_HEIGHT}"
        height_surf = self.font_large.render(height_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(height_surf, (10, 40))

        # Species
        species_text = f"Species: {len(self.attracted_species)} / {self.WIN_CONDITION_SPECIES}"
        species_surf = self.font_large.render(species_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(species_surf, (10, 70))
        
        # Selected Kelp
        selected_type = self.KELP_COLOR_NAMES[self.selected_kelp_index]
        selected_color = self.KELP_TYPES[selected_type]["color"]
        plant_text = f"Planting: {selected_type.capitalize()} (Cost: {self.plant_cost})"
        plant_surf = self.font_large.render(plant_text, True, selected_color)
        self.screen.blit(plant_surf, (self.SCREEN_WIDTH - plant_surf.get_width() - 10, 10))

    def _render_cursor(self):
        cursor_rect = pygame.Rect(
            self.cursor_pos * self.CELL_WIDTH, 
            self.SEABED_Y - 5, 
            self.CELL_WIDTH, 
            10
        )
        # Create a temporary surface for transparency
        s = pygame.Surface((cursor_rect.width, cursor_rect.height), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, cursor_rect.topleft)

    def _create_god_rays(self):
        self.god_rays = []
        for _ in range(random.randint(3, 5)):
            x1 = random.uniform(-100, self.SCREEN_WIDTH + 100)
            width = random.uniform(80, 200)
            angle = random.uniform(0.1, 0.4) * random.choice([-1, 1])
            self.god_rays.append([x1, width, angle, random.uniform(0.01, 0.03)])

    def _render_god_rays(self):
        for i, ray in enumerate(self.god_rays):
            x1, width, angle, speed = ray
            x1_moved = x1 + self.steps * speed
            
            p1 = (x1_moved, 0)
            p2 = (x1_moved + width, 0)
            p3 = (p2[0] + self.SCREEN_HEIGHT * angle, self.SCREEN_HEIGHT)
            p4 = (p1[0] + self.SCREEN_HEIGHT * angle, self.SCREEN_HEIGHT)
            
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3, p4), (255, 255, 255, 2))
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3, p4), (255, 255, 255, 2))

    def _create_particles(self, x, y, color, count):
        for _ in range(count * 2):
            self.particles.append(Particle(x, y, color))
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _update_marine_life(self):
        for fish in self.marine_life:
            fish.update(self.SCREEN_WIDTH, self.SEABED_Y)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "resources": self.resources,
            "plant_cost": self.plant_cost,
            "total_kelp_height": self.total_kelp_height,
            "attracted_species_count": len(self.attracted_species),
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Create a temporary clean state for validation
        original_state = self.__dict__.copy()
        self.reset(seed=42)

        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Restore original state
        self.__dict__.update(original_state)
        print("✓ Implementation validated successfully")

# --- Helper Classes ---

class Kelp:
    def __init__(self, x, kelp_type, type_data):
        self.x = x
        self.type = kelp_type
        self.color = type_data["color"]
        self.growth_rate = type_data["growth_rate"]
        
        self.height = 1
        self.target_height = 5
        self.sway = random.uniform(0.5, 1.5)
        self.sway_speed = random.uniform(0.01, 0.03)
        self.segments = 10

    def grow(self, amount):
        self.target_height += amount

    def update(self, time_step):
        # Lerp height for smooth growth
        if self.height < self.target_height:
            self.height = self.height * 0.95 + self.target_height * 0.05
        # Passive growth
        self.target_height += self.growth_rate

    def draw(self, surface, base_y, time_step):
        points = []
        for i in range(self.segments + 1):
            progress = i / self.segments
            y = base_y - progress * self.height
            # Top sways more than the bottom
            x_offset = math.sin(self.x + time_step * self.sway_speed) * self.sway * progress**2 * 5
            points.append((int(self.x + x_offset), int(y)))
        
        if len(points) > 1:
            pygame.draw.lines(surface, self.color, False, points, max(2, int(self.height / 20)))

class Particle:
    def __init__(self, x, y, color):
        self.pos = [x + random.uniform(-5, 5), y + random.uniform(-5, 5)]
        self.vel = [random.uniform(-0.5, 0.5), random.uniform(-1.5, -0.5)]
        self.lifespan = random.randint(30, 60)
        self.color = color
        self.radius = random.uniform(2, 5)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] *= 0.98 # Slow down vertical movement
        self.lifespan -= 1
        return self.lifespan > 0

    def draw(self, surface):
        alpha = max(0, 255 * (self.lifespan / 60))
        color_with_alpha = self.color + (alpha,)
        
        s = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, color_with_alpha, (self.radius, self.radius), self.radius)
        surface.blit(s, (int(self.pos[0] - self.radius), int(self.pos[1] - self.radius)))

class Fish:
    def __init__(self, pos, species, color):
        self.pos = list(pos)
        self.species = species
        self.color = color
        self.vel = [random.uniform(-1, 1), random.uniform(-0.5, 0.5)]
        self.size = random.uniform(8, 15)
        self.turn_timer = 0

    def update(self, screen_width, seabed_y):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        self.turn_timer -= 1
        if self.turn_timer <= 0:
            self.vel[0] += random.uniform(-0.2, 0.2)
            self.vel[1] += random.uniform(-0.2, 0.2)
            self.vel[0] = np.clip(self.vel[0], -1.5, 1.5)
            self.vel[1] = np.clip(self.vel[1], -0.7, 0.7)
            self.turn_timer = random.randint(20, 60)

        # Boundary checks
        if self.pos[0] < self.size or self.pos[0] > screen_width - self.size:
            self.vel[0] *= -1
        if self.pos[1] < self.size or self.pos[1] > seabed_y - self.size:
            self.vel[1] *= -1
            
        self.pos[0] = np.clip(self.pos[0], self.size, screen_width - self.size)
        self.pos[1] = np.clip(self.pos[1], self.size, seabed_y - self.size)


    def draw(self, surface):
        direction = 1 if self.vel[0] > 0 else -1
        p1 = (self.pos[0], self.pos[1])
        p2 = (self.pos[0] - self.size * direction, self.pos[1] - self.size / 2)
        p3 = (self.pos[0] - self.size * direction, self.pos[1] + self.size / 2)
        
        pygame.gfxdraw.aapolygon(surface, (p1, p2, p3), self.color)
        pygame.gfxdraw.filled_polygon(surface, (p1, p2, p3), self.color)
        
        # Eye
        eye_x = self.pos[0] - 2 * direction
        eye_y = self.pos[1]
        pygame.draw.circle(surface, (0,0,0), (int(eye_x), int(eye_y)), 1)

# Example usage
if __name__ == '__main__':
    # Un-dummy the video driver for local rendering
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    pygame.display.set_caption("Kelp Forest")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0

    while not terminated:
        # --- Manual Control Example ---
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # no-op, released, released
        
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        # Convert observation back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        env.clock.tick(30) # Limit to 30 FPS for smooth viewing

    print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
    env.close()