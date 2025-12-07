import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:36:45.363771
# Source Brief: brief_01091.md
# Brief Index: 1091
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game objects
class Element:
    def __init__(self, x, y, vx, vy, size, color_name, color_value):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.size = size
        self.color_name = color_name
        self.color_value = color_value
        self.glow_color = tuple(min(255, c + 50) for c in color_value)

    def update(self, nudge, size_change, speed_multiplier, bounds):
        # Apply size change
        self.size = np.clip(self.size + size_change, 10, 50)

        # Apply nudge from action
        nudge_strength = 2.0
        self.x += nudge[0] * nudge_strength
        self.y += nudge[1] * nudge_strength

        # Apply base velocity
        self.x += self.vx * speed_multiplier
        self.y += self.vy * speed_multiplier

        # Screen wrap
        w, h = bounds
        if self.x < -self.size: self.x = w + self.size
        if self.x > w + self.size: self.x = -self.size
        if self.y < -self.size: self.y = h + self.size
        if self.y > h + self.size: self.y = -self.size

    def draw(self, surface):
        # Draw glow effect
        glow_radius = int(self.size * 1.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.glow_color + (50,), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (int(self.x) - glow_radius, int(self.y) - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw main element body with anti-aliasing
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), int(self.size), self.color_value)
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.size), self.color_value)

class Portal:
    def __init__(self, x, y, size, color_name, color_value):
        self.x, self.y = x, y
        self.size = size
        self.color_name = color_name
        self.color_value = color_value
        self.angle = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-1, 1)

    def update(self):
        self.angle += self.rotation_speed

    def draw(self, surface):
        # Draw glowing portal ring
        for i in range(5):
            alpha = 150 - i * 30
            color = self.color_value + (alpha,)
            pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.size + i, color)

        # Draw rotating inner particles for effect
        for i in range(4):
            angle_rad = math.radians(self.angle + i * 90)
            px = self.x + math.cos(angle_rad) * (self.size - 5)
            py = self.y + math.sin(angle_rad) * (self.size - 5)
            pygame.draw.circle(surface, (255, 255, 255), (int(px), int(py)), 2)


class Particle:
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.x, self.y = x, y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = random.randint(20, 40)
        self.color = color
        self.size = random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)
        return self.lifetime > 0 and self.size > 0

    def draw(self, surface):
        alpha = int(255 * (self.lifetime / 40))
        if alpha > 0:
            color_with_alpha = self.color + (alpha,)
            temp_surf = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (self.size, self.size), self.size)
            surface.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size)), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide drifting elemental orbs into matching cosmic portals. Nudge all orbs at once and alter their size to achieve stability before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to nudge all orbs. Hold space to grow orbs and shift to shrink them."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and timing
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000

        # Colors and Fonts
        self.COLOR_BG = (10, 5, 30)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_SUCCESS = (100, 255, 100)
        self.COLOR_FAILURE = (255, 100, 100)
        self.ELEMENT_COLORS = {
            'red': (255, 50, 50),
            'blue': (80, 120, 255),
            'green': (50, 255, 80),
            'yellow': (255, 255, 80),
            'purple': (200, 80, 255)
        }
        self.ALL_POSSIBLE_COLORS = list(self.ELEMENT_COLORS.keys())

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stability = 100.0
        self.elements = []
        self.portals = []
        self.particles = []
        self.stars = []
        self.num_active_pairs = 1
        self.base_element_speed = 1.0
        self.available_colors = []

        self._generate_stars()
        # self.reset() # This is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stability = 100.0
        self.elements.clear()
        self.portals.clear()
        self.particles.clear()

        # Reset difficulty
        self.num_active_pairs = 1
        self.base_element_speed = 1.0
        self.available_colors = self.ALL_POSSIBLE_COLORS[:2]

        for _ in range(self.num_active_pairs):
            self._spawn_element_portal_pair()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Unpack action and update game state
        movement, space_held, shift_held = action
        nudge_vec = self._get_nudge_vector(movement)
        size_change = (1 if space_held else 0) - (1 if shift_held else 0)

        self._update_particles()
        self._update_portals()
        self._update_elements(nudge_vec, size_change)

        # 2. Handle reactions and event-based rewards
        event_reward = self._handle_reactions()

        # 3. Update stability and calculate continuous reward
        self.stability = max(0, self.stability - 0.05) # Stability decay
        continuous_reward = 0.1 if self.stability > 50 else -0.1
        
        total_reward = event_reward + continuous_reward
        self.score += total_reward

        # 4. Update difficulty and spawn new entities
        self._update_difficulty()
        while len(self.elements) < self.num_active_pairs:
            self._spawn_element_portal_pair()

        # 5. Check for termination and apply terminal rewards
        self.steps += 1
        terminated = self.stability <= 0 or self.steps >= self.MAX_STEPS
        truncated = False # This environment does not truncate based on time limits in the same way as some others.
        if terminated and not self.game_over:
            self.game_over = True
            terminal_reward = 100 if self.steps >= self.MAX_STEPS else -100
            self.score += terminal_reward
            total_reward += terminal_reward

        return self._get_observation(), total_reward, terminated, truncated, self._get_info()

    def _get_nudge_vector(self, movement_action):
        if movement_action == 1: return (0, -1)  # Up
        if movement_action == 2: return (0, 1)   # Down
        if movement_action == 3: return (-1, 0)  # Left
        if movement_action == 4: return (1, 0)   # Right
        return (0, 0) # None

    def _update_elements(self, nudge_vec, size_change):
        for element in self.elements:
            element.update(nudge_vec, size_change, self.base_element_speed, (self.WIDTH, self.HEIGHT))

    def _update_portals(self):
        for portal in self.portals:
            portal.update()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _handle_reactions(self):
        reward = 0
        consumed_elements = []
        for element in self.elements:
            for portal in self.portals:
                dist = math.hypot(element.x - portal.x, element.y - portal.y)
                if dist < portal.size:
                    if element.color_name == portal.color_name:
                        # Correct reaction
                        reward += 1.0
                        self.stability = min(100, self.stability + 10)
                        self._create_particle_burst(element.x, element.y, self.COLOR_SUCCESS)
                        # sound: success_chime.wav
                    else:
                        # Incorrect reaction
                        reward -= 5.0
                        self.stability = max(0, self.stability - 20)
                        self._create_particle_burst(element.x, element.y, self.COLOR_FAILURE)
                        # sound: error_buzz.wav
                    
                    consumed_elements.append(element)
                    self.portals.remove(portal)
                    break
            if element in consumed_elements:
                continue
        
        self.elements = [e for e in self.elements if e not in consumed_elements]
        return reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 100 == 0:
            self.num_active_pairs = min(5, self.num_active_pairs + 1)
            self.base_element_speed = min(2.5, self.base_element_speed + 0.05)
        
        if self.steps > 0 and self.steps % 200 == 0:
            if len(self.available_colors) < len(self.ALL_POSSIBLE_COLORS):
                self.available_colors.append(self.ALL_POSSIBLE_COLORS[len(self.available_colors)])

    def _spawn_element_portal_pair(self):
        color_name = random.choice(self.available_colors)
        color_value = self.ELEMENT_COLORS[color_name]
        
        # Spawn portal away from edges
        portal_x = random.uniform(80, self.WIDTH - 80)
        portal_y = random.uniform(80, self.HEIGHT - 80)
        self.portals.append(Portal(portal_x, portal_y, 40, color_name, color_value))
        
        # Spawn element on an edge
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            ex, ey = random.uniform(0, self.WIDTH), -20
            evx, evy = random.uniform(-1, 1), random.uniform(0.5, 1.5)
        elif edge == 'bottom':
            ex, ey = random.uniform(0, self.WIDTH), self.HEIGHT + 20
            evx, evy = random.uniform(-1, 1), random.uniform(-1.5, -0.5)
        elif edge == 'left':
            ex, ey = -20, random.uniform(0, self.HEIGHT)
            evx, evy = random.uniform(0.5, 1.5), random.uniform(-1, 1)
        else: # right
            ex, ey = self.WIDTH + 20, random.uniform(0, self.HEIGHT)
            evx, evy = random.uniform(-1.5, -0.5), random.uniform(-1, 1)
            
        self.elements.append(Element(ex, ey, evx, evy, random.randint(15, 35), color_name, color_value))

    def _create_particle_burst(self, x, y, color):
        for _ in range(50):
            self.particles.append(Particle(x, y, color))

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stability": self.stability}

    def _generate_stars(self):
        self.stars = []
        for i in range(200):
            self.stars.append({
                'x': random.randint(0, self.WIDTH),
                'y': random.randint(0, self.HEIGHT),
                'size': random.uniform(0.5, 1.5),
                'speed': random.uniform(0.1, 0.3) * (1 + i/200.0) # Parallax
            })

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_portals()
        self._render_particles()
        self._render_elements()
        self._render_ui()
        if self.game_over:
            self._render_game_over()

    def _render_background(self):
        for star in self.stars:
            star['x'] -= star['speed']
            if star['x'] < 0:
                star['x'] = self.WIDTH
                star['y'] = random.randint(0, self.HEIGHT)
            
            alpha = int(100 * star['size'])
            pygame.draw.circle(self.screen, (200, 200, 255, alpha), (star['x'], star['y']), star['size'])

    def _render_portals(self):
        for portal in self.portals:
            portal.draw(self.screen)

    def _render_particles(self):
        for particle in self.particles:
            particle.draw(self.screen)

    def _render_elements(self):
        for element in self.elements:
            element.draw(self.screen)

    def _render_ui(self):
        # Stability Bar
        bar_width, bar_height = 200, 20
        bar_x, bar_y = 20, 20
        fill_width = int(bar_width * (self.stability / 100.0))
        
        if self.stability > 60:
            bar_color = (100, 220, 100)
        elif self.stability > 30:
            bar_color = (220, 220, 100)
        else:
            bar_color = (220, 100, 100)
            
        pygame.draw.rect(self.screen, (50, 50, 70), (bar_x, bar_y, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)
        stability_text = self.font_ui.render("STABILITY", True, self.COLOR_UI_TEXT)
        self.screen.blit(stability_text, (bar_x + bar_width + 10, bar_y))

        # Score and Steps Text
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 20))
        steps_text = self.font_ui.render(f"STEP: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 20, 50))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.steps >= self.MAX_STEPS:
            text = "VICTORY"
            color = self.COLOR_SUCCESS
        else:
            text = "DEFEATED"
            color = self.COLOR_FAILURE
        
        game_over_surf = self.font_game_over.render(text, True, color)
        pos_x = self.WIDTH / 2 - game_over_surf.get_width() / 2
        pos_y = self.HEIGHT / 2 - game_over_surf.get_height() / 2
        self.screen.blit(game_over_surf, (pos_x, pos_y))

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # The `validate_implementation` function was removed as it's not part of the standard API
    # and was likely for internal development. The main block is for human play.
    
    # Set the SDL_VIDEODRIVER to a real driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cosmic Alchemy")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    # --- Human player controls ---
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    # space: 0=released, 1=held
    # shift: 0=released, 1=held
    action = [0, 0, 0] 

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0

        # Size change
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Render one last time to show the game over screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            pygame.time.wait(2000) # Wait before resetting
            obs, info = env.reset()

        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)
        
    env.close()