import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper function for drawing anti-aliased thick lines
def draw_line_antialiased(surface, color, start_pos, end_pos, width=1):
    x1, y1 = start_pos
    x2, y2 = end_pos
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0: return
    
    unit_dx, unit_dy = dx / length, dy / length
    perp_dx, perp_dy = -unit_dy, unit_dx

    points = []
    for i in range(width):
        offset = (i - (width - 1) / 2)
        points.append((x1 + perp_dx * offset, y1 + perp_dy * offset))
        points.append((x2 + perp_dx * offset, y2 + perp_dy * offset))
        points.append((x2 - perp_dx * offset, y2 - perp_dy * offset))
        points.append((x1 - perp_dx * offset, y1 - perp_dy * offset))
    
    if len(points) >= 3:
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

class Planet:
    def __init__(self, pos, radius, target_color):
        self.pos = np.array(pos, dtype=np.float32)
        self.radius = radius
        self.base_color = (25, 28, 36)
        self.target_color = np.array(target_color)
        self.color_accumulator = np.array([0.0, 0.0, 0.0])
        self.current_color = self.base_color
        self.is_upgraded = False
        self.plants = []
        self.prev_color_dist = self._get_color_distance()

    def _get_color_distance(self):
        # Calculate Euclidean distance in RGB space
        return np.linalg.norm(self.current_color - self.target_color)

    def add_plant(self, seed_color_type):
        # seed_color_type is 0 for R, 1 for G, 2 for B
        self.plants.append({'type': seed_color_type, 'growth': 0.0})
        # Sound: Soft "thud" or "splash"

    def update(self, time_factor):
        if self.is_upgraded:
            return 0

        # Grow plants
        for plant in self.plants:
            if plant['growth'] < 1.0:
                plant['growth'] += 0.01 * time_factor
                plant['growth'] = min(1.0, plant['growth'])

        # Update planet color based on plants
        self.color_accumulator.fill(0)
        for plant in self.plants:
            plant_color_contribution = np.array([0, 0, 0])
            plant_color_contribution[plant['type']] = 255 * plant['growth']
            self.color_accumulator += plant_color_contribution
        
        # Blend base color with accumulated plant colors
        final_color_np = np.clip(self.base_color + self.color_accumulator, 0, 255)
        self.current_color = tuple(int(c) for c in final_color_np)
        
        # Check for upgrade
        color_dist = self._get_color_distance()
        if color_dist < 20: # Threshold for considering the color matched
            self.is_upgraded = True
            self.current_color = (255, 215, 0) # Gold
            return 10.0 # Planet upgrade reward
        
        # Continuous reward for getting closer
        reward = (self.prev_color_dist - color_dist) * 0.01
        self.prev_color_dist = color_dist
        return max(0, reward) # Only positive rewards

    def draw(self, surface):
        # Glow effect
        for i in range(self.radius, self.radius + 15, 2):
            alpha = 50 * (1 - (i - self.radius) / 15)
            glow_color = (*self.current_color, int(alpha))
            if self.is_upgraded:
                glow_color = (255, 215, 0, int(alpha))
            pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), i, glow_color)
        
        # Planet body
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, self.current_color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, self.current_color)

        # Target color indicator
        if not self.is_upgraded:
            indicator_size = 10
            indicator_pos = (int(self.pos[0]) - indicator_size // 2, int(self.pos[1]) - self.radius - 20)
            pygame.draw.rect(surface, self.target_color, (*indicator_pos, indicator_size, indicator_size))
            pygame.draw.rect(surface, (255,255,255), (*indicator_pos, indicator_size, indicator_size), 1)

class Seed:
    def __init__(self, pos, angle, color, color_type):
        self.pos = np.array(pos, dtype=np.float32)
        self.velocity = np.array([math.cos(angle), -math.sin(angle)]) * 8
        self.color = color
        self.color_type = color_type # 0:R, 1:G, 2:B
        self.radius = 5
        self.trail = []

    def update(self):
        self.trail.append(self.pos.copy())
        if len(self.trail) > 10:
            self.trail.pop(0)
        self.pos += self.velocity

    def draw(self, surface):
        # Draw trail
        for i, pos in enumerate(self.trail):
            alpha = int(255 * (i / len(self.trail)))
            trail_color = (*self.color, alpha)
            pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), self.radius - 2, trail_color)

        # Draw seed head
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, self.color)

class Particle:
    def __init__(self, pos, color, life):
        self.pos = np.array(pos, dtype=np.float32)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
        self.color = color
        self.life = life
        self.max_life = life

    def update(self):
        self.pos += self.velocity
        self.velocity *= 0.98 # Friction
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            size = int(5 * (self.life / self.max_life))
            if size > 0:
                rect = pygame.Rect(int(self.pos[0] - size/2), int(self.pos[1] - size/2), size, size)
                shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(shape_surf, (*self.color, alpha), shape_surf.get_rect())
                surface.blit(shape_surf, rect)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Cultivate barren planets by shooting colored seeds from your launcher. "
        "Match the target color of each planet to bring the galaxy to life."
    )
    user_guide = (
        "Controls: ←→ to aim, ↑↓ to control time speed. "
        "Press space to fire a seed and shift to cycle seed type."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        # Colors
        self.COLOR_BG = (10, 12, 18)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.SEED_COLORS = [(255, 50, 50), (50, 255, 50), (50, 100, 255)]
        self.SEED_NAMES = ["RED", "GREEN", "BLUE"]

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_factor = 1.0
        self.aim_angle = math.pi / 2 # Pointing straight up
        self.launcher_pos = (self.WIDTH // 2, self.HEIGHT - 20)
        self.selected_seed_idx = 0
        self.last_shift_held = False
        self.last_space_held = False
        self.galaxy_level = 0
        self.planets = []
        self.seeds_in_flight = []
        self.particles = []
        self.stars = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_factor = 1.0
        self.aim_angle = math.pi / 2
        self.selected_seed_idx = 0
        self.last_shift_held = False
        self.last_space_held = False
        self.galaxy_level = 1
        
        self.planets.clear()
        self.seeds_in_flight.clear()
        self.particles.clear()
        self._setup_galaxy()
        
        self.stars = [(random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.uniform(0.5, 1.5)) for _ in range(150)]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        self._handle_input(action)
        reward += self._update_game_state()
        self.score += reward

        self.steps += 1
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Aiming
        if movement == 3: # Left
            self.aim_angle += 0.05
        if movement == 4: # Right
            self.aim_angle -= 0.05
        self.aim_angle = max(0.1, min(math.pi - 0.1, self.aim_angle))

        # Time manipulation
        if movement == 1: # Up
            self.time_factor += 0.1
        if movement == 2: # Down
            self.time_factor -= 0.1
        self.time_factor = max(0.1, min(2.0, self.time_factor))

        # Plant seed (on press)
        if space_held and not self.last_space_held:
            color = self.SEED_COLORS[self.selected_seed_idx]
            color_type = self.selected_seed_idx
            new_seed = Seed(self.launcher_pos, self.aim_angle, color, color_type)
            self.seeds_in_flight.append(new_seed)
            # Sound: "pew" or "launch"

        # Cycle seed (on press)
        if shift_held and not self.last_shift_held:
            self.selected_seed_idx = (self.selected_seed_idx + 1) % len(self.SEED_COLORS)
            # Sound: "click" or "switch"

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_game_state(self):
        step_reward = 0

        # Update seeds
        for seed in self.seeds_in_flight[:]:
            seed.update()
            # Check collision with planets
            for planet in self.planets:
                if not planet.is_upgraded:
                    dist = np.linalg.norm(seed.pos - planet.pos)
                    if dist < planet.radius:
                        planet.add_plant(seed.color_type)
                        self.seeds_in_flight.remove(seed)
                        self._create_particles(seed.pos, seed.color, 20)
                        break
            # Remove if out of bounds
            if not (0 < seed.pos[0] < self.WIDTH and 0 < seed.pos[1] < self.HEIGHT):
                if seed in self.seeds_in_flight:
                    self.seeds_in_flight.remove(seed)

        # Update planets
        for planet in self.planets:
            planet_reward = planet.update(self.time_factor)
            if planet_reward > 0 and planet.is_upgraded:
                # Sound: "Success" or "Chime"
                self._create_particles(planet.pos, (255, 215, 0), 100)
            step_reward += planet_reward

        # Update particles
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)
        
        return step_reward

    def _check_termination(self):
        if self.steps >= 1000:
            return True
        
        if all(p.is_upgraded for p in self.planets):
            self.score += 100 # Galaxy completion bonus
            self.galaxy_level += 1
            if self.galaxy_level > 3: # Cap level for this simple setup
                return True
            self._setup_galaxy()
            # Sound: "Level Up" or "Galaxy Complete" fanfare
            
        return False
        
    def _setup_galaxy(self):
        self.planets.clear()
        
        # Define planet layouts and targets per galaxy level
        if self.galaxy_level == 1:
            targets = [(255, 0, 0)]
            positions = [(self.WIDTH // 2, 150)]
        elif self.galaxy_level == 2:
            targets = [(0, 255, 0), (0, 0, 255)]
            positions = [(self.WIDTH // 4, 150), (self.WIDTH * 3 // 4, 150)]
        else: # For level 3 and beyond
            targets = [(255, 255, 0), (255, 0, 255), (0, 255, 255)]
            positions = [(self.WIDTH // 2, 100), (self.WIDTH // 4, 200), (self.WIDTH * 3 // 4, 200)]

        for pos, target in zip(positions, targets):
            self.planets.append(Planet(pos, radius=40, target_color=target))

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
            "galaxy_level": self.galaxy_level,
            "planets_upgraded": sum(1 for p in self.planets if p.is_upgraded),
            "total_planets": len(self.planets)
        }

    def _render_game(self):
        self._render_stars()
        self._render_launcher_and_trajectory()
        for particle in self.particles:
            particle.draw(self.screen)
        for planet in self.planets:
            planet.draw(self.screen)
        for seed in self.seeds_in_flight:
            seed.draw(self.screen)

    def _render_stars(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), size)

    def _render_launcher_and_trajectory(self):
        # Launcher
        p1 = self.launcher_pos
        end_point = (p1[0] + 30 * math.cos(self.aim_angle), p1[1] - 30 * math.sin(self.aim_angle))
        
        base_angle_left = self.aim_angle + math.pi / 2
        base_angle_right = self.aim_angle - math.pi / 2
        
        p2 = (p1[0] + 15 * math.cos(base_angle_left), p1[1] - 15 * math.sin(base_angle_left))
        p3 = (p1[0] + 15 * math.cos(base_angle_right), p1[1] - 15 * math.sin(base_angle_right))
        
        launcher_color = self.SEED_COLORS[self.selected_seed_idx]
        pygame.gfxdraw.aapolygon(self.screen, [end_point, p2, p3], launcher_color)
        pygame.gfxdraw.filled_polygon(self.screen, [end_point, p2, p3], launcher_color)

        # Trajectory
        traj_end = (p1[0] + 200 * math.cos(self.aim_angle), p1[1] - 200 * math.sin(self.aim_angle))
        draw_line_antialiased(self.screen, (*launcher_color, 100), end_point, traj_end, 3)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/1000", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (10, 45))

        # Galaxy Info
        galaxy_text = self.font_large.render(f"GALAXY {self.galaxy_level}", True, self.COLOR_UI_TEXT)
        text_rect = galaxy_text.get_rect(center=(self.WIDTH // 2, 30))
        self.screen.blit(galaxy_text, text_rect)

        # Time Factor
        time_text = self.font_small.render(f"TIME x{self.time_factor:.1f}", True, self.COLOR_UI_TEXT)
        text_rect = time_text.get_rect(right=self.WIDTH - 10, top=10)
        self.screen.blit(time_text, text_rect)
        
        # Current Seed
        seed_name = self.SEED_NAMES[self.selected_seed_idx]
        seed_color = self.SEED_COLORS[self.selected_seed_idx]
        seed_text = self.font_small.render("SEED", True, self.COLOR_UI_TEXT)
        seed_name_text = self.font_small.render(seed_name, True, seed_color)
        
        self.screen.blit(seed_text, (self.WIDTH - 80, self.HEIGHT - 40))
        self.screen.blit(seed_name_text, (self.WIDTH - 80, self.HEIGHT - 25))
        pygame.draw.rect(self.screen, seed_color, (self.WIDTH - 95, self.HEIGHT - 35, 10, 10))
        
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(Particle(pos, color, life=random.randint(20, 40)))

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It is not part of the required environment implementation
    # but is useful for testing.
    
    # Re-enable video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Planet Cultivator")
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

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    pygame.quit()