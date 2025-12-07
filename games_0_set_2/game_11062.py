import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:32:41.502933
# Source Brief: brief_01062.md
# Brief Index: 1062
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class Plant:
    """Represents a single plant in the garden."""
    def __init__(self, x, y, species_info, plant_id):
        self.x = x
        self.y = y
        self.species_info = species_info
        self.id = plant_id
        
        self.base_size = random.uniform(15, 25)
        self.size_modifier = 0.0
        self.target_size_modifier = 0.0
        
        self.max_size_modifier = 50.0
        self.min_size_modifier = -10.0
        
        self.animation_phase = random.uniform(0, 2 * math.pi)

    @property
    def current_size(self):
        return self.base_size + self.size_modifier

    def update(self, dt):
        # Smoothly interpolate size
        self.size_modifier += (self.target_size_modifier - self.size_modifier) * 0.1 * dt * 30
        self.animation_phase += dt * self.species_info.get('anim_speed', 1.0)

    def grow(self):
        # SFX: grow_sound.play()
        self.target_size_modifier = min(self.max_size_modifier, self.target_size_modifier + 10)

    def shrink(self):
        # SFX: shrink_sound.play()
        self.target_size_modifier = max(self.min_size_modifier, self.target_size_modifier - 10)

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, x, y, color, lifespan):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.color = color
        self.lifespan = lifespan
        self.initial_lifespan = lifespan

    def update(self, dt):
        self.x += self.vx * dt * 30
        self.y += self.vy * dt * 30
        self.vy += 0.1 * dt * 30 # Gravity
        self.lifespan -= dt

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.initial_lifespan))
            radius = int(2 * (self.lifespan / self.initial_lifespan))
            if radius > 0:
                pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), radius, (*self.color, alpha))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Cultivate a mystical garden of alien plants. Nurture different species, discover synergies, "
        "and arrange them to create the most magnificent display."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selector. Press space to grow the selected plant "
        "and shift to shrink it."
    )
    auto_advance = True

    # --- Persistent state across episodes ---
    CUMULATIVE_SCORE = 0
    PLANT_SPECIES_CATALOG = [
        {'name': 'Sunpetal', 'color': (255, 190, 0), 'petal_color': (230, 80, 30)},
        {'name': 'Skyvine', 'color': (50, 200, 50), 'leaf_color': (100, 220, 100), 'anim_speed': 0.5},
        {'name': 'Glow-shroom', 'color': (150, 100, 255), 'glow_color': (200, 160, 255), 'anim_speed': 2.0},
        {'name': 'Hydro-fern', 'color': (40, 150, 180), 'detail_color': (80, 200, 220), 'anim_speed': 0.8},
        {'name': 'Crystal-bloom', 'color': (220, 220, 255), 'petal_color': (255, 255, 255), 'anim_speed': 0.3}
    ]
    UNLOCKED_SPECIES_INDICES = {0, 1} # Start with Sunpetal and Skyvine

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.dt = 1 / 30.0 # Assume 30 FPS for physics/animation updates

        # --- Visuals ---
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 16)
        self.font_small = pygame.font.SysFont("Consolas", 12)
        self.COLOR_BG = (25, 20, 40)
        self.COLOR_GRID = (45, 40, 65)
        self.COLOR_WATER = (80, 120, 255)
        self.COLOR_UI_TEXT = (255, 255, 240)
        self.COLOR_SELECTOR = (0, 255, 255)
        
        # --- Game Grid ---
        self.grid_size = (10, 6)
        self.cell_size = (self.width // self.grid_size[0], self.height // self.grid_size[1])

        # --- State variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.high_score_session = 0
        self.plants = []
        self.water_sources = []
        self.selected_coords = (0, 0)
        self.particles = []
        self.notifications = deque(maxlen=3)
        self.last_action_time = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Update persistent state ---
        GameEnv.CUMULATIVE_SCORE += self.score
        
        newly_unlocked = False
        current_unlock_count = len(GameEnv.UNLOCKED_SPECIES_INDICES)
        required_unlocks = 1 + (GameEnv.CUMULATIVE_SCORE // 1000)
        
        if required_unlocks > current_unlock_count:
            for i in range(len(GameEnv.PLANT_SPECIES_CATALOG)):
                if i not in GameEnv.UNLOCKED_SPECIES_INDICES:
                    GameEnv.UNLOCKED_SPECIES_INDICES.add(i)
                    species_name = GameEnv.PLANT_SPECIES_CATALOG[i]['name']
                    self._add_notification(f"New Species: {species_name}!", (255, 215, 0))
                    newly_unlocked = True
                    # SFX: unlock_ fanfare.play()
                    break

        # --- Initialize episode state ---
        self.steps = 0
        self.score = 0
        if 'high_score' in (options or {}):
            self.high_score_session = options['high_score']
        
        self.plants = []
        self.particles = []
        if not newly_unlocked:
            self.notifications.clear()

        # Place water sources
        self.water_sources = []
        num_water = self.np_random.integers(2, 4)
        for _ in range(num_water):
            self.water_sources.append((self.np_random.integers(0, self.grid_size[0]), self.np_random.integers(0, self.grid_size[1])))

        # Place initial plants
        num_plants = self.np_random.integers(3, 6)
        occupied_coords = set(self.water_sources)
        for i in range(num_plants):
            while True:
                x, y = self.np_random.integers(0, self.grid_size[0]), self.np_random.integers(0, self.grid_size[1])
                if (x, y) not in occupied_coords:
                    occupied_coords.add((x, y))
                    species_idx = self.np_random.choice(list(GameEnv.UNLOCKED_SPECIES_INDICES))
                    species_info = GameEnv.PLANT_SPECIES_CATALOG[species_idx]
                    self.plants.append(Plant(x, y, species_info, i))
                    break
        
        if self.plants:
            self.selected_coords = (self.plants[0].x, self.plants[0].y)
        else:
            self.selected_coords = (self.grid_size[0] // 2, self.grid_size[1] // 2)

        self.last_action_time = pygame.time.get_ticks()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_btn, shift_btn = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Actions ---
        self._handle_actions(movement, space_btn, shift_btn)

        # --- Update Game State ---
        for plant in self.plants:
            plant.update(self.dt)
        
        for particle in self.particles[:]:
            particle.update(self.dt)
            if particle.lifespan <= 0:
                self.particles.remove(particle)
        
        # --- Calculate Reward ---
        old_score = self.score
        new_score = self._calculate_magnificence()
        
        reward = 0
        score_delta = new_score - old_score
        
        if score_delta > 0:
            reward += score_delta * 0.1 # Scale down score reward
        else:
            reward -= 0.01 # Small penalty for stagnation
            
        self.score = new_score

        if self.score > self.high_score_session:
            reward += 10 # Capped bonus for new high score
            self.high_score_session = self.score

        self.steps += 1
        terminated = self.steps >= 2000
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_actions(self, movement, grow, shrink):
        # --- Movement ---
        if movement != 0:
            x, y = self.selected_coords
            if movement == 1: y = (y - 1 + self.grid_size[1]) % self.grid_size[1] # Up
            elif movement == 2: y = (y + 1) % self.grid_size[1] # Down
            elif movement == 3: x = (x - 1 + self.grid_size[0]) % self.grid_size[0] # Left
            elif movement == 4: x = (x + 1) % self.grid_size[0] # Right
            self.selected_coords = (x, y)
        
        # --- Grow/Shrink ---
        selected_plant = self._get_plant_at(*self.selected_coords)
        if selected_plant:
            if grow:
                selected_plant.grow()
                self._create_particles(selected_plant, 'grow')
            elif shrink:
                selected_plant.shrink()
                self._create_particles(selected_plant, 'shrink')

    def _calculate_magnificence(self):
        total_score = 0
        plant_coords = { (p.x, p.y): p for p in self.plants }
        
        # 1. Diversity Score
        present_species = set(p.species_info['name'] for p in self.plants)
        total_score += len(present_species) * 20

        for plant in self.plants:
            plant_score = 0
            
            # Base score for size
            plant_score += plant.current_size * 0.5

            # Species-specific scoring
            species = plant.species_info['name']
            is_shaded = False
            if plant.y > 0:
                plant_above = plant_coords.get((plant.x, plant.y - 1))
                if plant_above and plant_above.current_size > plant.current_size * 1.5:
                    is_shaded = True

            if species == 'Skyvine':
                plant_score += plant.current_size * 0.5 # Extra score for height
            elif species == 'Sunpetal' and not is_shaded:
                plant_score += 15
            elif species == 'Glow-shroom' and is_shaded:
                plant_score += 20
            elif species == 'Crystal-bloom':
                plant_score += plant.current_size * 0.2 + 10 # High base value

            # Proximity bonuses
            is_near_water = any(abs(plant.x - wx) <= 1 and abs(plant.y - wy) <= 1 for wx, wy in self.water_sources)
            if is_near_water:
                plant_score += 10
                if species == 'Hydro-fern':
                    plant_score += 25
            
            # Synergy bonuses
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = plant_coords.get((plant.x + dx, plant.y + dy))
                if neighbor:
                    neighbor_species = neighbor.species_info['name']
                    if (species, neighbor_species) in [('Glow-shroom', 'Hydro-fern'), ('Hydro-fern', 'Glow-shroom')]:
                        plant_score += 15
                    if species == 'Crystal-bloom':
                        plant_score += 5 # Buffs neighbors

            # Overgrowth/Competition penalty (simple overlap check)
            for other in self.plants:
                if plant.id != other.id:
                    dist_sq = ((plant.x - other.x) * self.cell_size[0])**2 + ((plant.y - other.y) * self.cell_size[1])**2
                    radii_sum_sq = ((plant.current_size/2) + (other.current_size/2))**2
                    if dist_sq < radii_sum_sq * 0.7: # Penalize significant overlap
                        plant_score -= 5

            total_score += max(0, plant_score)

        return int(total_score)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_structures()
        self._render_grid()
        self._render_water_sources()
        self._render_plants()
        self._render_selector()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "high_score": self.high_score_session}

    # --- Rendering Methods ---
    def _render_background_structures(self):
        for i in range(15):
            x = int(math.sin(i * 0.4 + self.steps * 0.001) * self.width * 0.3 + self.width * 0.5)
            y = int(math.cos(i * 0.7) * self.height * 0.3 + self.height * 0.5)
            w = int(100 + math.sin(i + self.steps * 0.002) * 50)
            h = int(100 + math.cos(i * 1.2 + self.steps * 0.002) * 50)
            color = (
                self.COLOR_BG[0] + 10 + int(math.sin(i) * 5),
                self.COLOR_BG[1] + 10 + int(math.cos(i) * 5),
                self.COLOR_BG[2] + 15 + int(math.sin(i*0.5) * 5)
            )
            pygame.draw.ellipse(self.screen, color, (x-w//2, y-h//2, w, h), 1)

    def _render_grid(self):
        for x in range(self.grid_size[0] + 1):
            start_pos = (x * self.cell_size[0], 0)
            end_pos = (x * self.cell_size[0], self.height)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.grid_size[1] + 1):
            start_pos = (0, y * self.cell_size[1])
            end_pos = (self.width, y * self.cell_size[1])
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

    def _render_water_sources(self):
        for x, y in self.water_sources:
            center_x = int((x + 0.5) * self.cell_size[0])
            center_y = int((y + 0.5) * self.cell_size[1])
            radius = int(self.cell_size[0] * 0.3)
            pulse = (math.sin(self.steps * 0.1) + 1) / 2
            
            # Outer glow
            glow_radius = int(radius * (1.2 + pulse * 0.2))
            glow_alpha = int(80 + pulse * 40)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, (*self.COLOR_WATER, glow_alpha))
            # Inner circle
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_WATER)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, (200, 220, 255))

    def _render_plants(self):
        sorted_plants = sorted(self.plants, key=lambda p: p.y)
        for plant in sorted_plants:
            cx = int((plant.x + 0.5) * self.cell_size[0])
            cy = int((plant.y + 0.5) * self.cell_size[1])
            self._draw_plant_by_species(self.screen, plant, cx, cy)

    def _draw_plant_by_species(self, surface, plant, cx, cy):
        size = plant.current_size
        phase = plant.animation_phase
        species = plant.species_info['name']
        
        if species == 'Sunpetal':
            num_petals = 8
            for i in range(num_petals):
                angle = 2 * math.pi * i / num_petals + phase * 0.2
                len1 = size * 0.6
                len2 = size * 0.8 + math.sin(phase + i) * 5
                p1 = (cx + math.cos(angle) * len1, cy + math.sin(angle) * len1)
                p2 = (cx + math.cos(angle) * len2, cy + math.sin(angle) * len2)
                pygame.draw.line(surface, plant.species_info['petal_color'], p1, p2, max(1, int(size/8)))
            pygame.gfxdraw.filled_circle(surface, cx, cy, int(size/3), plant.species_info['color'])
            pygame.gfxdraw.aacircle(surface, cx, cy, int(size/3), (255,255,255))
        
        elif species == 'Skyvine':
            vine_width = max(2, int(size / 8))
            num_segments = 5
            points = [(cx, cy + size/2)]
            for i in range(1, num_segments + 1):
                px, py = points[-1]
                offset = math.sin(phase + i * 0.5) * size * 0.1
                points.append((px + offset, py - size / num_segments))
            if len(points) > 1:
                pygame.draw.lines(surface, plant.species_info['color'], False, points, vine_width)
                for i, p in enumerate(points):
                    if i > 0:
                        leaf_size = max(1, int(size/10))
                        leaf_dir = 1 if i % 2 == 0 else -1
                        pygame.draw.line(surface, plant.species_info['leaf_color'], p, (p[0] + leaf_size * leaf_dir, p[1] - leaf_size), 2)

        elif species == 'Glow-shroom':
            pulse = (math.sin(phase) + 1) / 2
            glow_size = int(size * 0.7 * (1 + pulse * 0.2))
            glow_alpha = int(100 + pulse * 50)
            pygame.gfxdraw.filled_circle(surface, cx, cy, glow_size, (*plant.species_info['glow_color'], glow_alpha))
            pygame.gfxdraw.filled_circle(surface, cx, cy, int(size * 0.5), plant.species_info['color'])
            pygame.draw.rect(surface, plant.species_info['color'], (cx - size*0.1, cy, size*0.2, size*0.2))

        elif species == 'Hydro-fern':
            self._draw_fractal_leaf(surface, cx, cy, -math.pi/2, size*0.4, 6, plant.species_info['color'], plant.species_info['detail_color'], phase)

        elif species == 'Crystal-bloom':
            num_crystals = 6
            for i in range(num_crystals):
                angle = 2 * math.pi * i / num_crystals + phase * 0.1
                r = size * 0.6 + math.sin(phase * 2 + i) * 5
                p1 = (cx, cy)
                p2 = (cx + math.cos(angle) * r, cy + math.sin(angle) * r)
                p3 = (cx + math.cos(angle + 0.1) * r * 0.7, cy + math.sin(angle + 0.1) * r * 0.7)
                color_phase = (phase + i) % (2*math.pi)
                color = (
                    int(128 + 127 * math.sin(color_phase)),
                    int(128 + 127 * math.sin(color_phase + 2*math.pi/3)),
                    int(128 + 127 * math.sin(color_phase + 4*math.pi/3))
                )
                pygame.gfxdraw.filled_trigon(surface, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), color)

    def _draw_fractal_leaf(self, surface, x, y, angle, length, depth, color1, color2, phase):
        if depth <= 0 or length < 2:
            return
        
        end_x = x + length * math.cos(angle)
        end_y = y + length * math.sin(angle)
        
        width = max(1, int(depth / 2))
        pygame.draw.line(surface, color1, (x, y), (end_x, end_y), width)
        
        sway = math.sin(phase + depth) * 0.1
        self._draw_fractal_leaf(surface, end_x, end_y, angle - 0.5 + sway, length * 0.7, depth - 1, color2, color1, phase)
        self._draw_fractal_leaf(surface, end_x, end_y, angle + 0.5 + sway, length * 0.7, depth - 1, color2, color1, phase)

    def _render_selector(self):
        x, y = self.selected_coords
        rect = pygame.Rect(x * self.cell_size[0], y * self.cell_size[1], self.cell_size[0], self.cell_size[1])
        
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
        alpha = int(150 + pulse * 105)
        color = (*self.COLOR_SELECTOR, alpha)
        
        pygame.gfxdraw.rectangle(self.screen, rect, color)
        
        # Info box for selected plant
        plant = self._get_plant_at(x, y)
        if plant:
            info_text = f"{plant.species_info['name']} [{int(plant.current_size)}]"
            text_surf = self.font_medium.render(info_text, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(centerx=rect.centerx, bottom=rect.top - 5)
            # Clamp to screen
            text_rect.clamp_ip(self.screen.get_rect())
            pygame.draw.rect(self.screen, (*self.COLOR_BG, 180), text_rect.inflate(8, 4))
            self.screen.blit(text_surf, text_rect)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = f"Magnificence: {self.score}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # High Score
        hs_text = f"Best: {self.high_score_session}"
        hs_surf = self.font_medium.render(hs_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(hs_surf, (10, 40))

        # Steps
        time_left = 2000 - self.steps
        time_text = f"Time: {time_left}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_UI_TEXT)
        time_rect = time_surf.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(time_surf, time_rect)

        # Notifications
        for i, (text, color) in enumerate(self.notifications):
            notif_surf = self.font_medium.render(text, True, color)
            notif_rect = notif_surf.get_rect(centerx=self.width/2, y=10 + i * 25)
            self.screen.blit(notif_surf, notif_rect)
    
    # --- Utility Methods ---
    def _get_plant_at(self, x, y):
        for plant in self.plants:
            if plant.x == x and plant.y == y:
                return plant
        return None
        
    def _create_particles(self, plant, p_type):
        cx = (plant.x + 0.5) * self.cell_size[0]
        cy = (plant.y + 0.5) * self.cell_size[1]
        
        if p_type == 'grow':
            color = (200, 255, 200) # Greenish-white for growth
            # SFX: particle_burst_up.play()
        else: # shrink
            color = (255, 200, 200) # Reddish-white for shrinking
            # SFX: particle_burst_down.play()

        for _ in range(15):
            self.particles.append(Particle(cx, cy, color, lifespan=random.uniform(0.5, 1.2)))

    def _add_notification(self, text, color):
        self.notifications.append((text, color))

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # The original code had a validate_implementation method that was called in __init__.
    # This is not standard practice for gym environments and can cause issues with
    # environment creation wrappers. It has been removed from the __init__ but is
    # kept here for reference during development.
    def validate_implementation(env_instance):
        print("Running validation...")
        # Test action space
        assert env_instance.action_space.shape == (3,)
        assert env_instance.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = env_instance._get_observation()
        assert test_obs.shape == (env_instance.height, env_instance.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = env_instance.reset()
        assert obs.shape == (env_instance.height, env_instance.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = env_instance.action_space.sample()
        obs, reward, term, trunc, info = env_instance.step(test_action)
        assert obs.shape == (env_instance.height, env_instance.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    # --- Manual Play Example ---
    # This part is for manual testing and is not part of the environment definition.
    # We need to unset the SDL_VIDEODRIVER to see the display.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    validate_implementation(env) # Manually call validation
    obs, info = env.reset(options={'high_score': 0})
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Hanging Gardens")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset(options={'high_score': info['high_score']})
                    total_reward = 0
                    print("--- Episode Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset(options={'high_score': info['high_score']})
            total_reward = 0
            
        clock.tick(30) # Limit to 30 FPS for manual play

    env.close()