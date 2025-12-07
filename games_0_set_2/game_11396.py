import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:54:41.185301
# Source Brief: brief_01396.md
# Brief Index: 1396
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for Planets
class Planet:
    def __init__(self, x, y, radius, mass):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.radius = radius
        self.mass = mass
        self.color = (random.randint(100, 200), random.randint(150, 255), random.randint(100, 200))
        self.glow_color = (*self.color, 50)

    def apply_force(self, force):
        acceleration = force / self.mass
        self.vel += acceleration

    def update(self, dt, screen_width, screen_height):
        # Inter-planetary gravity (weak)
        # In a real game, this would iterate over all other planets.
        # For this env, we'll keep it simple and focus on wave effects.

        self.pos += self.vel * dt
        self.vel *= 0.98  # Damping

        # Boundary collision
        if self.pos.x - self.radius < 0:
            self.pos.x = self.radius
            self.vel.x *= -0.5
        if self.pos.x + self.radius > screen_width:
            self.pos.x = screen_width - self.radius
            self.vel.x *= -0.5
        if self.pos.y - self.radius < 0:
            self.pos.y = self.radius
            self.vel.y *= -0.5
        if self.pos.y + self.radius > screen_height:
            self.pos.y = screen_height - self.radius
            self.vel.y *= -0.5

    def draw(self, surface, is_selected):
        # Glow effect
        glow_radius = int(self.radius * 1.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.glow_color, (glow_radius, glow_radius), glow_radius)
        surface.blit(s, (int(self.pos.x - glow_radius), int(self.pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Planet body
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), self.color)

        # Selection indicator
        if is_selected:
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.radius + 8), (255, 255, 0))
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.radius + 9), (255, 255, 0))

# Helper class for Gravity Waves
class GravityWave:
    def __init__(self, pos, wave_type, max_radius=200, duration=30):
        self.pos = pos
        self.wave_type = wave_type
        self.max_radius = max_radius
        self.duration = duration
        self.age = 0
        self.radius = 0
        if wave_type == "push":
            self.color = (255, 100, 0)
        else: # pull
            self.color = (0, 150, 255)

    def update(self):
        self.age += 1
        progress = self.age / self.duration
        self.radius = self.max_radius * math.sin(progress * math.pi / 2) # Ease-out expansion
        return self.age >= self.duration

    def draw(self, surface):
        if self.age < self.duration:
            alpha = int(200 * (1 - (self.age / self.duration)))
            color_with_alpha = (*self.color, alpha)
            
            # Use a surface for transparency
            s = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(s, int(self.radius), int(self.radius), int(self.radius) - 1, color_with_alpha)
            pygame.gfxdraw.aacircle(s, int(self.radius), int(self.radius), int(self.radius), color_with_alpha)
            surface.blit(s, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Guide your fleet through a treacherous nebula by manipulating planetary gravity. "
        "Create push and pull waves to move planets and clear a safe path to the exit portal."
    )
    user_guide = (
        "Controls: Use ↑ and ↓ arrow keys to select a planet. Press Shift to toggle between 'push' and 'pull' "
        "wave types. Press Space to deploy the selected wave."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 16)
        self.font_m = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 48, bold=True)

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_FLEET = (255, 255, 0)
        self.COLOR_EXIT = (0, 255, 128)
        self.COLOR_NEBULA = (180, 50, 220)
        self.COLOR_TEXT = (220, 220, 240)

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.fleet_pos = None
        self.fleet_radius = 8
        self.exit_pos = None
        self.exit_radius = 20
        self.planets = []
        self.active_waves = []
        self.particles = []
        self.selected_planet_idx = 0
        self.wave_types = ["push", "pull"]
        self.selected_wave_idx = 0
        
        # Input handling for one-shot actions
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Pre-generate starfield
        self.starfield = self._generate_starfield(200)

        self.reset()
        
        # Run validation
        # self.validate_implementation() # Commented out for submission

    def _generate_starfield(self, num_stars):
        stars = []
        for _ in range(num_stars):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.choice([1, 1, 1, 2])
            brightness = random.randint(50, 150)
            stars.append(((x, y), size, (brightness, brightness, brightness)))
        return stars

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.game_over and self.score > 0: # Won previous level
            self.level += 1
        elif self.game_over and self.score <= 0: # Lost previous level
            pass # Keep same level
            
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.fleet_pos = pygame.Vector2(30, self.HEIGHT / 2)
        self.exit_pos = pygame.Vector2(self.WIDTH - 40, self.HEIGHT / 2)
        
        self.planets.clear()
        self.active_waves.clear()
        self.particles.clear()
        
        num_planets = min(1 + self.level, 10)
        for _ in range(num_planets):
            placed = False
            while not placed:
                radius = random.randint(15, 25)
                x = random.randint(radius + 50, self.WIDTH - radius - 50)
                y = random.randint(radius + 50, self.HEIGHT - radius - 50)
                
                # Ensure no overlap with other planets or start/end points
                new_planet = Planet(x, y, radius, radius * 10)
                if self.fleet_pos.distance_to(new_planet.pos) < new_planet.radius + 50:
                    continue
                if self.exit_pos.distance_to(new_planet.pos) < new_planet.radius + 50:
                    continue
                
                overlap = False
                for p in self.planets:
                    if p.pos.distance_to(new_planet.pos) < p.radius + new_planet.radius + 20:
                        overlap = True
                        break
                if not overlap:
                    self.planets.append(new_planet)
                    placed = True
        
        self.selected_planet_idx = 0
        self.selected_wave_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Action Processing ---
        # Planet selection (up/down)
        if movement == 1: # Up
            self.selected_planet_idx = (self.selected_planet_idx - 1) % len(self.planets)
        elif movement == 2: # Down
            self.selected_planet_idx = (self.selected_planet_idx + 1) % len(self.planets)
        
        # Wave type switching (Shift) - rising edge
        if shift_held and not self.prev_shift_held:
            self.selected_wave_idx = (self.selected_wave_idx + 1) % len(self.wave_types)
            # SFX: UI_SWITCH
        
        # Wave deployment (Space) - rising edge
        if space_held and not self.prev_space_held and self.planets:
            target_planet = self.planets[self.selected_planet_idx]
            wave_type = self.wave_types[self.selected_wave_idx]
            new_wave = GravityWave(target_planet.pos, wave_type)
            self.active_waves.append(new_wave)
            # SFX: WAVE_DEPLOY_PUSH or WAVE_DEPLOY_PULL

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Game Logic & Physics Update ---
        self.steps += 1
        
        # Update waves and apply forces
        for wave in self.active_waves[:]:
            if wave.update():
                self.active_waves.remove(wave)
            else:
                for planet in self.planets:
                    dist_vec = planet.pos - wave.pos
                    dist = dist_vec.length()
                    if 0 < dist < wave.radius:
                        force_magnitude = 50000 / (dist + 1) # Inverse distance force
                        if wave.wave_type == "push":
                            force = dist_vec.normalize() * force_magnitude
                            # SFX: PLANET_PUSH_IMPACT
                        else: # pull
                            force = -dist_vec.normalize() * force_magnitude
                            # SFX: PLANET_PULL_IMPACT
                        planet.apply_force(force)

        # Update planets
        for planet in self.planets:
            planet.update(1.0 / 30, self.WIDTH, self.HEIGHT)
        
        # Update fleet (moves steadily towards exit)
        move_dir = (self.exit_pos - self.fleet_pos)
        if move_dir.length() > 0:
            self.fleet_pos += move_dir.normalize() * 1.0 # Fleet speed
            # Add particles for engine trail
            if self.steps % 3 == 0:
                self.particles.append([self.fleet_pos.copy(), [random.uniform(-1, 0.5), random.uniform(-0.5, 0.5)], random.randint(15, 25), self.COLOR_FLEET])

        # Update particles
        for p in self.particles[:]:
            p[0] += p[1]
            p[2] -= 1
            if p[2] <= 0:
                self.particles.remove(p)

        # --- Termination and Reward ---
        reward += 0.01 # Survival reward
        
        # Check for collision with planets
        for planet in self.planets:
            if self.fleet_pos.distance_to(planet.pos) < self.fleet_radius + planet.radius:
                self.game_over = True
                reward = -100
                # SFX: EXPLOSION
                break
        
        # Check for collision with nebula
        nebula_strength = self._get_nebula_strength_at(self.fleet_pos)
        if nebula_strength > 0.8:
            self.game_over = True
            reward = -100
            # SFX: SHIP_DAMAGE_SIZZLE
        elif nebula_strength > 0.5:
            reward -= 0.1 # Proximity penalty

        # Check for victory
        if not self.game_over and self.fleet_pos.distance_to(self.exit_pos) < self.exit_radius:
            self.game_over = True
            reward = 100
            self.score = 1 # Mark as a win for level progression
            # SFX: VICTORY_FANFARE
        
        # Check for timeout
        if self.steps >= 2000:
            self.game_over = True
            reward = -50 # Penalty for timeout
        
        self.score += reward
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_nebula_strength_at(self, pos):
        # Calculate nebula "density" at a point based on proximity to planets
        strength = 0
        for p in self.planets:
            dist_sq = (p.pos - pos).length_squared()
            # Nebula influence radius is proportional to planet radius
            influence_radius_sq = (p.radius * 5)**2
            if dist_sq < influence_radius_sq:
                strength += (influence_radius_sq - dist_sq) / influence_radius_sq
        return min(strength, 1.5) # Cap strength

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for pos, size, color in self.starfield:
            pygame.draw.rect(self.screen, color, (*pos, size, size))

    def _render_nebula(self):
        # Use large, additive blended circles for a fast, good-looking nebula
        nebula_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for planet in self.planets:
            radius = int(planet.radius * 5)
            color = (*self.COLOR_NEBULA, 20) # Low alpha for blending
            pygame.draw.circle(nebula_surface, color, (int(planet.pos.x), int(planet.pos.y)), radius)
        self.screen.blit(nebula_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_game(self):
        self._render_background()
        self._render_nebula()

        # Exit portal
        pygame.gfxdraw.filled_circle(self.screen, int(self.exit_pos.x), int(self.exit_pos.y), self.exit_radius, (*self.COLOR_EXIT, 50))
        pygame.gfxdraw.aacircle(self.screen, int(self.exit_pos.x), int(self.exit_pos.y), self.exit_radius, self.COLOR_EXIT)
        
        # Planets
        for i, planet in enumerate(self.planets):
            planet.draw(self.screen, i == self.selected_planet_idx)
            
        # Active Waves
        for wave in self.active_waves:
            wave.draw(self.screen)

        # Particles
        for p in self.particles:
            color = (*p[3], int(255 * (p[2] / 25)))
            r = max(1, int(4 * (p[2] / 25)))
            pygame.draw.circle(self.screen, color, p[0], r)

        # Fleet
        pygame.gfxdraw.filled_circle(self.screen, int(self.fleet_pos.x), int(self.fleet_pos.y), self.fleet_radius, (*self.COLOR_FLEET, 60))
        pygame.gfxdraw.aacircle(self.screen, int(self.fleet_pos.x), int(self.fleet_pos.y), self.fleet_radius, self.COLOR_FLEET)
        pygame.gfxdraw.filled_circle(self.screen, int(self.fleet_pos.x), int(self.fleet_pos.y), int(self.fleet_radius * 0.7), self.COLOR_FLEET)

    def _render_ui(self):
        # Top UI bar
        score_text = self.font_m.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        level_text = self.font_m.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        steps_text = self.font_m.render(f"STEPS: {self.steps}/2000", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(level_text, (self.WIDTH // 2 - level_text.get_width() // 2, 10))
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        # Bottom UI bar for wave selection
        wave_type_text = self.font_m.render("WAVE TYPE:", True, self.COLOR_TEXT)
        self.screen.blit(wave_type_text, (10, self.HEIGHT - 40))
        for i, w_type in enumerate(self.wave_types):
            color = (255, 255, 0) if i == self.selected_wave_idx else self.COLOR_TEXT
            text = self.font_s.render(f"[{w_type.upper()}]", True, color)
            self.screen.blit(text, (150 + i * 100, self.HEIGHT - 38))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "MISSION SUCCESS" if self.score > 0 else "MISSION FAILED"
            color = self.COLOR_EXIT if self.score > 0 else (255, 50, 50)
            end_text = self.font_l.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH//2 - end_text.get_width()//2, self.HEIGHT//2 - end_text.get_height()//2 - 20))
            reset_text = self.font_m.render("Call reset() to continue", True, self.COLOR_TEXT)
            self.screen.blit(reset_text, (self.WIDTH//2 - reset_text.get_width()//2, self.HEIGHT//2 + 20))


    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "fleet_pos": (self.fleet_pos.x, self.fleet_pos.y),
            "planets": len(self.planets),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage:
if __name__ == '__main__':
    # To run with display, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we'll need a screen
    pygame.display.set_caption("Gravity Forge Nebula")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop for human play
    running = True
    while running:
        # Map keyboard inputs to MultiDiscrete action
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation to the display screen
        # Need to transpose back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS

    env.close()