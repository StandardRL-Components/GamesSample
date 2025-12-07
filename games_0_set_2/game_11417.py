import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:00:33.307146
# Source Brief: brief_01417.md
# Brief Index: 1417
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for Plants
class Plant:
    """Represents a single plant in the greenhouse."""
    def __init__(self, pos, species_id, is_target=False):
        self.pos = pygame.Vector2(pos)
        self.species_id = species_id
        self.is_target = is_target
        self.health = 100.0
        self.growth = 0.0  # 0 to 100
        self.max_growth = 100
        self.radius = 5
        self.color = (50, 180, 50) if not is_target else (100, 255, 100)

    def update(self, resources):
        """Updates plant growth and health based on available resources."""
        health_lost = 0
        
        has_water = resources['water'] > 0
        has_nutrients = resources['nutrients'] > 0
        has_light = resources['light'] > 0

        if has_water and has_nutrients and has_light:
            if self.health > 50:
                self.growth = min(self.max_growth, self.growth + 0.25)
        else:
            initial_health = self.health
            if not has_water: self.health -= 0.2
            if not has_nutrients: self.health -= 0.2
            if not has_light: self.health -= 0.2
            self.health = max(0, self.health)
            health_lost = initial_health - self.health

        self.radius = 5 + int(self.growth * 0.25)
        return health_lost

    def take_damage(self, amount):
        """Reduces plant health and returns the amount lost."""
        initial_health = self.health
        self.health = max(0, self.health - amount)
        return initial_health - self.health

    def is_alive(self):
        return self.health > 0

    def is_mature(self):
        return self.growth >= self.max_growth

    def draw(self, surface):
        """Renders the plant on the given surface."""
        if not self.is_alive():
            return
            
        # Main plant body with anti-aliasing
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
        
        # Target indicator glow effect
        if self.is_target:
            glow_radius = self.radius + 5 + int(abs(math.sin(pygame.time.get_ticks() * 0.002)) * 3)
            glow_color = (200, 255, 200, 60)
            s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
            surface.blit(s, (int(self.pos.x - glow_radius), int(self.pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Health bar
        bar_width = 40
        bar_height = 5
        bar_x = self.pos.x - bar_width / 2
        bar_y = self.pos.y - self.radius - 15
        health_percent = self.health / 100.0
        
        pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        if health_percent > 0:
            pygame.draw.rect(surface, (0, 255, 0), (bar_x, bar_y, int(bar_width * health_percent), bar_height))


# Helper class for Pests
class Pest:
    """Represents a pest that attacks plants."""
    def __init__(self, pos, screen_dims):
        self.pos = pygame.Vector2(pos)
        self.screen_dims = screen_dims
        self.speed = 1.5
        self.damage = 5.0
        self.health = 10
        self.radius = 6
        self.target_plant = None
        self.color = (255, 100, 0)

    def update(self, plants):
        """Moves the pest towards the nearest plant and attacks it."""
        health_lost_by_plant = 0
        living_plants = [p for p in plants if p.is_alive()]
        if not living_plants:
            return health_lost_by_plant

        if self.target_plant is None or not self.target_plant.is_alive():
            self.target_plant = min(living_plants, key=lambda p: self.pos.distance_to(p.pos))

        if self.target_plant:
            direction = (self.target_plant.pos - self.pos)
            if direction.length() > 0:
                direction.normalize_ip()
            self.pos += direction * self.speed
            
            if self.pos.distance_to(self.target_plant.pos) < self.target_plant.radius + self.radius:
                health_lost_by_plant += self.target_plant.take_damage(self.damage)
                self.health = 0 # Pest is destroyed after one attack

        self.pos.x = np.clip(self.pos.x, 0, self.screen_dims[0])
        self.pos.y = np.clip(self.pos.y, 0, self.screen_dims[1])
        
        return health_lost_by_plant

    def is_alive(self):
        return self.health > 0

    def draw(self, surface):
        """Renders the pest as a spiky star."""
        points = []
        for i in range(5):
            angle = math.radians(i * 72 + pygame.time.get_ticks() * 0.1)
            outer_p = self.pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * self.radius
            angle += math.radians(36)
            inner_p = self.pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * self.radius * 0.5
            points.append((int(outer_p.x), int(outer_p.y)))
            points.append((int(inner_p.x), int(inner_p.y)))
        pygame.gfxdraw.aapolygon(surface, points, self.color)
        pygame.gfxdraw.filled_polygon(surface, points, self.color)


# Helper class for Particles
class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, pos, vel, color, lifespan, radius):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.radius = radius

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1

    def is_alive(self):
        return self.lifespan > 0

    def draw(self, surface):
        alpha = int(255 * (self.lifespan / self.max_lifespan))
        color_with_alpha = (*self.color, alpha)
        s = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, color_with_alpha, (self.radius, self.radius), self.radius)
        surface.blit(s, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)), special_flags=pygame.BLEND_RGBA_ADD)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Defend your greenhouse from pests and manage resources to grow a target plant to maturity in zero-g."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the reticle. Press space to fire your laser at pests and hold shift over a resource pad to collect it."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Consolas", 20, bold=True)
            self.large_font = pygame.font.SysFont("Consolas", 50, bold=True)
        except pygame.error:
            self.font = pygame.font.SysFont(None, 24)
            self.large_font = pygame.font.SysFont(None, 60)
        
        self.MAX_STEPS = 2000
        self.RETICLE_SPEED = 10
        self.MAX_PESTS = 20

        self.COLOR_BG = (10, 20, 30)
        self.COLOR_RETICLE = (0, 255, 255)
        self.COLOR_WATER = (0, 150, 255)
        self.COLOR_NUTRIENTS = (180, 0, 255)
        self.COLOR_LIGHT = (255, 255, 0)
        
        self.steps = 0
        self.score = 0
        self.game_over_message = ""
        self.reticle_pos = None
        self.plants = []
        self.pests = []
        self.particles = []
        self.resources = {}
        self.resource_pads = {}
        self.target_plant_index = 0
        self.pest_spawn_rate = 0.01
        
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over_message = ""
        
        self.reticle_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        
        self.plants = [
            Plant((150, 250), 0),
            Plant((320, 200), 1),
            Plant((490, 250), 2)
        ]
        self.target_plant_index = self.np_random.integers(0, len(self.plants))
        for i, plant in enumerate(self.plants):
            plant.is_target = (i == self.target_plant_index)

        self.pests = []
        self.particles = []
        
        self.resources = {'water': 50, 'nutrients': 50, 'light': 50}
        
        pad_y = 30
        pad_h = 40
        pad_w = 100
        self.resource_pads = {
            'water': pygame.Rect(self.WIDTH // 2 - 1.6 * pad_w, pad_y, pad_w, pad_h),
            'nutrients': pygame.Rect(self.WIDTH // 2 - 0.5 * pad_w, pad_y, pad_w, pad_h),
            'light': pygame.Rect(self.WIDTH // 2 + 0.6 * pad_w, pad_y, pad_w, pad_h),
        }
        
        self.pest_spawn_rate = 0.01

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over_message:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1

        # 1. Handle Player Action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.reticle_pos.y -= self.RETICLE_SPEED
        elif movement == 2: self.reticle_pos.y += self.RETICLE_SPEED
        elif movement == 3: self.reticle_pos.x -= self.RETICLE_SPEED
        elif movement == 4: self.reticle_pos.x += self.RETICLE_SPEED
        self.reticle_pos.x = np.clip(self.reticle_pos.x, 0, self.WIDTH)
        self.reticle_pos.y = np.clip(self.reticle_pos.y, 0, self.HEIGHT)

        if space_held:
            # sfx: laser_beam
            self._create_laser_beam_effect()
            pests_hit = [p for p in self.pests if p.pos.distance_to(self.reticle_pos) < 15]
            for pest in pests_hit:
                pest.health -= 2
                if not pest.is_alive():
                    reward += 5.0
                    self._create_explosion(pest.pos, (255, 150, 50))
                    # sfx: pest_zap

        if shift_held:
            for res_type, pad in self.resource_pads.items():
                if pad.collidepoint(self.reticle_pos):
                    # sfx: resource_add
                    if self.resources[res_type] < 100:
                        self.resources[res_type] = min(100, self.resources[res_type] + 1.5)
                        reward += 0.05
                    self._create_resource_gain_effect(self.reticle_pos, res_type)
        
        # 2. Update Game State
        total_health_lost_from_pests = 0
        for pest in self.pests:
            total_health_lost_from_pests += pest.update(self.plants)
        reward -= 0.5 * total_health_lost_from_pests

        total_health_lost_from_starvation = 0
        living_plants = [p for p in self.plants if p.is_alive()]
        for plant in living_plants:
            health_lost = plant.update(self.resources)
            total_health_lost_from_starvation += health_lost
            reward += 0.01
        reward -= 0.5 * total_health_lost_from_starvation

        if living_plants:
            resource_drain = len(living_plants) * 0.05
            self.resources['water'] = max(0, self.resources['water'] - resource_drain)
            self.resources['nutrients'] = max(0, self.resources['nutrients'] - resource_drain)
            self.resources['light'] = max(0, self.resources['light'] - resource_drain)

        self.pests = [p for p in self.pests if p.is_alive()]
        self.particles = [p for p in self.particles if p.is_alive()]

        if self.np_random.random() < self.pest_spawn_rate and len(self.pests) < self.MAX_PESTS:
            spawn_edge = self.np_random.integers(0, 4)
            if spawn_edge == 0: pos = (self.np_random.random() * self.WIDTH, 0)
            elif spawn_edge == 1: pos = (self.np_random.random() * self.WIDTH, self.HEIGHT)
            elif spawn_edge == 2: pos = (0, self.np_random.random() * self.HEIGHT)
            else: pos = (self.WIDTH, self.np_random.random() * self.HEIGHT)
            self.pests.append(Pest(pos, (self.WIDTH, self.HEIGHT)))
            
        if self.steps > 0 and self.steps % 200 == 0:
            self.pest_spawn_rate = min(0.1, self.pest_spawn_rate + 0.01)

        # 3. Check for Termination
        if self.plants[self.target_plant_index].is_mature():
            reward += 100
            terminated = True
            self.game_over_message = "VICTORY!"
        elif not any(p.is_alive() for p in self.plants):
            reward -= 100
            terminated = True
            self.game_over_message = "ALL PLANTS DIED"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over_message = "TIME LIMIT REACHED"

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        for p in self.particles:
            p.update()
            p.draw(self.screen)
            
        for plant in self.plants:
            plant.draw(self.screen)
            
        for pest in self.pests:
            pest.draw(self.screen)
            
        x, y = int(self.reticle_pos.x), int(self.reticle_pos.y)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (x - 10, y), (x + 10, y), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (x, y - 10), (x, y + 10), 2)
        pygame.gfxdraw.aacircle(self.screen, x, y, 12, self.COLOR_RETICLE)

    def _render_ui(self):
        for res_type, pad in self.resource_pads.items():
            color = getattr(self, f"COLOR_{res_type.upper()}")
            dark_color = tuple(c * 0.5 for c in color)
            pygame.draw.rect(self.screen, dark_color, pad, border_radius=5)
            pygame.draw.rect(self.screen, color, pad, width=2, border_radius=5)
            text = self.font.render(res_type.upper(), True, color)
            self.screen.blit(text, text.get_rect(center=pad.center))

        res_y = 10
        for i, (res_type, value) in enumerate(self.resources.items()):
            color = getattr(self, f"COLOR_{res_type.upper()}")
            bar_x = 10 + i * 110
            pygame.draw.rect(self.screen, (50,50,50), (bar_x, res_y, 100, 15))
            if value > 0:
                pygame.draw.rect(self.screen, color, (bar_x, res_y, int(value), 15))
        
        score_text = self.font.render(f"Score: {self.score:.1f}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, self.HEIGHT - 30))
        
        steps_text = self.font.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, (255, 255, 255))
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, self.HEIGHT - 30))

        target_plant = self.plants[self.target_plant_index]
        if target_plant.is_alive():
            target_info_text = self.font.render("TARGET", True, (200, 255, 200))
            self.screen.blit(target_info_text, (target_plant.pos.x - target_info_text.get_width()/2, target_plant.pos.y + target_plant.radius + 5))

        if self.game_over_message:
            text_surface = self.large_font.render(self.game_over_message, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surface, text_rect)

    def _create_explosion(self, pos, color):
        for _ in range(30):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(15, 30)
            radius = self.np_random.integers(2, 5)
            self.particles.append(Particle(pos, vel, color, lifespan, radius))

    def _create_laser_beam_effect(self):
        for i in range(2):
            vel = (self.np_random.random() - 0.5, self.np_random.random() - 0.5)
            pos = self.reticle_pos + pygame.Vector2(vel) * 5
            lifespan = 5
            radius = 3
            self.particles.append(Particle(pos, (0,0), self.COLOR_RETICLE, lifespan, radius))

    def _create_resource_gain_effect(self, pos, res_type):
        color = getattr(self, f"COLOR_{res_type.upper()}")
        for _ in range(2):
            vel = (self.np_random.random() * 2 - 1, self.np_random.random() * 2 - 1)
            lifespan = 10
            radius = 4
            self.particles.append(Particle(pos, vel, color, lifespan, radius))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zero-G Gardener")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space_held, shift_held = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            obs, info = env.reset()
            pygame.time.wait(2000)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()