import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    """
    Gymnasium environment for a resource collection and building game.

    The player controls a collector at the bottom of the screen, moving it left and
    right to catch falling resources (wood, stone, gold). These resources are
    used to build structures.

    The goal is to build 5 houses within the time limit.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
      Only left (3) and right (4) affect the collector.
    - actions[1]: Space button (0=released, 1=held)
      On press, selects the next available building site.
    - actions[2]: Shift button (0=released, 1=held)
      On press, cycles through the available structure types to build (House, Tower).

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +0.1 for collecting any resource.
    - +1.0 for completing a House.
    - +2.0 for completing a Tower.
    - +100 for winning (building 5 houses).
    - -100 for losing (time runs out).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a collector to catch falling resources. Use the collected wood, stone, "
        "and gold to build houses and towers to win before time runs out."
    )
    user_guide = (
        "Use ←→ arrow keys to move the collector. Press space to select a building site "
        "and shift to cycle through build options (House/Tower)."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60 # Internal simulation FPS, not RL steps per second
    MAX_STEPS = 18000 # 5 minutes at 60 FPS

    # Colors
    COLOR_BG = (15, 23, 42)
    COLOR_UI_TEXT = (226, 232, 240)
    COLOR_UI_BG = (30, 41, 59, 180)
    COLOR_COLLECTOR = (56, 189, 248)
    COLOR_COLLECTOR_GLOW = (14, 116, 144, 50)
    COLOR_WOOD = (161, 98, 7)
    COLOR_STONE = (113, 113, 122)
    COLOR_GOLD = (252, 211, 77)
    COLOR_HOUSE = (34, 197, 94)
    COLOR_TOWER = (99, 102, 241)
    COLOR_PROGRESS_BAR = (245, 158, 11)
    COLOR_SITE_OUTLINE = (100, 116, 139)
    COLOR_SITE_SELECTED = (250, 250, 250)

    # Game Parameters
    COLLECTOR_WIDTH = 80
    COLLECTOR_HEIGHT = 12
    COLLECTOR_SPEED = 6.0
    COLLECTOR_FRICTION = 0.85
    RESOURCE_SIZE = 12
    PARTICLE_LIFESPAN = 30
    BUILD_TIME = 180 # steps to complete a building

    STRUCTURE_TYPES = ['house', 'tower']
    STRUCTURE_COSTS = {
        'house': {'wood': 2, 'stone': 1, 'gold': 0},
        'tower': {'wood': 1, 'stone': 2, 'gold': 1}
    }
    WIN_CONDITION_HOUSES = 5

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
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        self.collector_pos = None
        self.collector_vel = None
        self.resources = None
        self.falling_resources = None
        self.particles = None
        self.building_sites = None
        self.selected_site_idx = None
        self.selected_build_type_idx = None
        self.last_space_held = None
        self.last_shift_held = None
        self.steps = None
        self.score = None
        self.houses_built = None
        self.game_over = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.collector_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 30)
        self.collector_vel = pygame.Vector2(0, 0)

        self.resources = {'wood': 0, 'stone': 0, 'gold': 0}
        self.falling_resources = []
        self.particles = []

        self._initialize_building_sites()
        self.selected_site_idx = 0
        self.selected_build_type_idx = 0

        self.last_space_held = False
        self.last_shift_held = False

        self.steps = 0
        self.score = 0
        self.houses_built = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        self._handle_input(action)
        self._update_collector()
        reward += self._update_falling_resources()
        reward += self._update_building()
        self._update_particles()

        terminated = self._check_termination()
        if terminated:
            if self.houses_built >= self.WIN_CONDITION_HOUSES:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Lose penalty
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _initialize_building_sites(self):
        self.building_sites = []
        num_sites = self.WIN_CONDITION_HOUSES + 2 # 5 for houses, 2 for potential towers
        site_y = self.SCREEN_HEIGHT - 80
        total_width = (num_sites * 60) + ((num_sites - 1) * 20)
        start_x = (self.SCREEN_WIDTH - total_width) / 2
        for i in range(num_sites):
            self.building_sites.append({
                'pos': pygame.Vector2(start_x + i * 80, site_y),
                'size': pygame.Vector2(60, 40),
                'is_complete': False,
                'is_constructing': False,
                'progress': 0,
                'structure_type': None
            })

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 3: # Left
            self.collector_vel.x -= self.COLLECTOR_SPEED
        elif movement == 4: # Right
            self.collector_vel.x += self.COLLECTOR_SPEED

        # Cycle build type (Shift)
        shift_pressed = shift_held and not self.last_shift_held
        if shift_pressed:
            self.selected_build_type_idx = (self.selected_build_type_idx + 1) % len(self.STRUCTURE_TYPES)

        # Cycle selected site (Space)
        space_pressed = space_held and not self.last_space_held
        if space_pressed:
            # Find next non-complete site
            for _ in range(len(self.building_sites)):
                self.selected_site_idx = (self.selected_site_idx + 1) % len(self.building_sites)
                if not self.building_sites[self.selected_site_idx]['is_complete']:
                    break

        self.last_space_held, self.last_shift_held = space_held, shift_held

    def _update_collector(self):
        self.collector_vel.x *= self.COLLECTOR_FRICTION
        if abs(self.collector_vel.x) < 0.1:
            self.collector_vel.x = 0
        self.collector_pos += self.collector_vel
        self.collector_pos.x = np.clip(
            self.collector_pos.x, self.COLLECTOR_WIDTH / 2, self.SCREEN_WIDTH - self.COLLECTOR_WIDTH / 2
        )

    def _update_falling_resources(self):
        reward = 0
        # Spawn new resources
        if self.np_random.random() < 0.05: # Wood (common)
            self._spawn_resource('wood')
        if self.np_random.random() < 0.03: # Stone
            self._spawn_resource('stone')
        if self.np_random.random() < 0.01: # Gold (rare)
            self._spawn_resource('gold')

        collector_rect = pygame.Rect(
            self.collector_pos.x - self.COLLECTOR_WIDTH / 2,
            self.collector_pos.y - self.COLLECTOR_HEIGHT / 2,
            self.COLLECTOR_WIDTH, self.COLLECTOR_HEIGHT
        )

        for res in self.falling_resources[:]:
            res['pos'].y += res['speed']
            if res['pos'].y > self.SCREEN_HEIGHT:
                self.falling_resources.remove(res)
                continue

            res_rect = pygame.Rect(res['pos'].x - self.RESOURCE_SIZE/2, res['pos'].y - self.RESOURCE_SIZE/2, self.RESOURCE_SIZE, self.RESOURCE_SIZE)
            if collector_rect.colliderect(res_rect):
                self.resources[res['type']] += 1
                reward += 0.1
                self._spawn_particles(res['pos'], self._get_color_for_type(res['type']))
                self.falling_resources.remove(res)

        return reward

    def _spawn_resource(self, res_type):
        self.falling_resources.append({
            'pos': pygame.Vector2(self.np_random.uniform(20, self.SCREEN_WIDTH - 20), -20),
            'type': res_type,
            'speed': self.np_random.uniform(1.5, 3.5)
        })

    def _update_building(self):
        reward = 0
        site = self.building_sites[self.selected_site_idx]

        # Start new construction if conditions are met
        if not site['is_complete'] and not site['is_constructing']:
            build_type = self.STRUCTURE_TYPES[self.selected_build_type_idx]
            costs = self.STRUCTURE_COSTS[build_type]
            can_afford = all(self.resources[res] >= cost for res, cost in costs.items())

            if can_afford:
                for res, cost in costs.items():
                    self.resources[res] -= cost
                site['is_constructing'] = True
                site['structure_type'] = build_type

        # Update ongoing construction
        for s in self.building_sites:
            if s['is_constructing']:
                s['progress'] += 1 / self.BUILD_TIME
                if s['progress'] >= 1.0:
                    s['progress'] = 1.0
                    s['is_constructing'] = False
                    s['is_complete'] = True
                    self._spawn_particles(s['pos'] + s['size']/2, self.COLOR_GOLD, 30)

                    if s['structure_type'] == 'house':
                        self.houses_built += 1
                        reward += 1.0
                    elif s['structure_type'] == 'tower':
                        reward += 2.0
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _spawn_particles(self, pos, color, count=10):
        for _ in range(count):
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
                'lifespan': self.np_random.integers(15, self.PARTICLE_LIFESPAN),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _check_termination(self):
        return self.houses_built >= self.WIN_CONDITION_HOUSES or self.steps >= self.MAX_STEPS

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
            "resources": self.resources,
            "houses_built": self.houses_built,
            "time_left": self.MAX_STEPS - self.steps,
        }

    def _render_game(self):
        # Render building sites
        for i, site in enumerate(self.building_sites):
            site_rect = pygame.Rect(site['pos'], site['size'])
            if site['is_complete']:
                color = self._get_color_for_type(site['structure_type'])
                pygame.draw.rect(self.screen, color, site_rect, border_radius=4)
                if site['structure_type'] == 'house':
                    pygame.draw.polygon(self.screen, color, [
                        (site_rect.left, site_rect.top),
                        (site_rect.centerx, site_rect.top - 20),
                        (site_rect.right, site_rect.top)
                    ])
                else: # tower
                    pygame.draw.rect(self.screen, color, (site_rect.centerx - 5, site_rect.top-20, 10, 20))
            else:
                # Base outline
                pygame.draw.rect(self.screen, self.COLOR_SITE_OUTLINE, site_rect, 2, border_radius=4)
                if site['is_constructing']:
                    progress_width = site['size'].x * site['progress']
                    progress_rect = pygame.Rect(site['pos'].x, site['pos'].y, progress_width, site['size'].y)
                    pygame.draw.rect(self.screen, self._get_color_for_type(site['structure_type']), progress_rect, border_radius=4)

            # Selection highlight
            if i == self.selected_site_idx:
                pygame.draw.rect(self.screen, self.COLOR_SITE_SELECTED, site_rect, 2, border_radius=6)

        # Render falling resources
        for res in self.falling_resources:
            pygame.draw.circle(self.screen, self._get_color_for_type(res['type']), res['pos'], self.RESOURCE_SIZE / 2)

        # Render particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / self.PARTICLE_LIFESPAN))
            color_with_alpha = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), color_with_alpha)

        # Render collector
        collector_rect = pygame.Rect(
            self.collector_pos.x - self.COLLECTOR_WIDTH / 2,
            self.collector_pos.y - self.COLLECTOR_HEIGHT / 2,
            self.COLLECTOR_WIDTH, self.COLLECTOR_HEIGHT
        )
        # Glow effect
        glow_surface = pygame.Surface((self.COLLECTOR_WIDTH + 20, self.COLLECTOR_HEIGHT + 20), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, self.COLOR_COLLECTOR_GLOW, glow_surface.get_rect(), border_radius=12)
        self.screen.blit(glow_surface, (collector_rect.x-10, collector_rect.y-10))
        # Main body
        pygame.draw.rect(self.screen, self.COLOR_COLLECTOR, collector_rect, border_radius=6)

    def _render_ui(self):
        # UI Background Panel
        panel_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 40)
        s = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (0,0))
        pygame.draw.line(self.screen, self.COLOR_SITE_OUTLINE, (0, 40), (self.SCREEN_WIDTH, 40))

        # Resources
        res_text_wood = self.font_main.render(f"Wood: {self.resources['wood']}", True, self.COLOR_WOOD)
        res_text_stone = self.font_main.render(f"Stone: {self.resources['stone']}", True, self.COLOR_STONE)
        res_text_gold = self.font_main.render(f"Gold: {self.resources['gold']}", True, self.COLOR_GOLD)
        self.screen.blit(res_text_wood, (10, 10))
        self.screen.blit(res_text_stone, (150, 10))
        self.screen.blit(res_text_gold, (300, 10))

        # Time
        time_left_sec = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_main.render(f"Time: {int(time_left_sec // 60):02}:{int(time_left_sec % 60):02}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - 130, 10))

        # Selected build type
        build_type = self.STRUCTURE_TYPES[self.selected_build_type_idx]
        costs = self.STRUCTURE_COSTS[build_type]
        build_text = self.font_main.render(f"Build: {build_type.capitalize()}", True, self._get_color_for_type(build_type))
        cost_text = self.font_small.render(f"Cost: W:{costs['wood']} S:{costs['stone']} G:{costs['gold']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(build_text, (440, 5))
        self.screen.blit(cost_text, (440, 22))

    def _get_color_for_type(self, res_type):
        return {
            'wood': self.COLOR_WOOD,
            'stone': self.COLOR_STONE,
            'gold': self.COLOR_GOLD,
            'house': self.COLOR_HOUSE,
            'tower': self.COLOR_TOWER
        }.get(res_type, (255, 255, 255))

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == "__main__":
    # This block will not run in the test environment, but is useful for human play
    # For the main script to run, we need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Use a display screen for human play
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Resource Collector")
    clock = pygame.time.Clock()
    
    # Action state
    action = np.array([0, 0, 0]) # [movement, space, shift]

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0

        # Actions
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Houses: {info['houses_built']}")

        # Render the observation to the display screen
        # Need to transpose back for pygame's blit
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print("Game Over!")
    print(f"Final Score: {info['score']:.2f}")
    print(f"Houses Built: {info['houses_built']}")
    
    env.close()