import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:40:46.467384
# Source Brief: brief_01213.md
# Brief Index: 1213
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper function for drawing anti-aliased thick lines
def draw_line_antialiased(surface, color, start_pos, end_pos, width=1):
    """Draws a thick anti-aliased line."""
    x0, y0 = start_pos
    x1, y1 = end_pos
    mid_pos = ((x0 + x1) / 2, (y0 + y1) / 2)
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0: return
    angle = math.atan2(y0 - y1, x0 - x1)
    
    # Create a polygon for the line
    ul = (mid_pos[0] + (length / 2.) * math.cos(angle) - (width / 2.) * math.sin(angle),
          mid_pos[1] + (width / 2.) * math.cos(angle) + (length / 2.) * math.sin(angle))
    ur = (mid_pos[0] - (length / 2.) * math.cos(angle) - (width / 2.) * math.sin(angle),
          mid_pos[1] + (width / 2.) * math.cos(angle) - (length / 2.) * math.sin(angle))
    bl = (mid_pos[0] + (length / 2.) * math.cos(angle) + (width / 2.) * math.sin(angle),
          mid_pos[1] - (width / 2.) * math.cos(angle) + (length / 2.) * math.sin(angle))
    br = (mid_pos[0] - (length / 2.) * math.cos(angle) + (width / 2.) * math.sin(angle),
          mid_pos[1] - (width / 2.) * math.cos(angle) - (length / 2.) * math.sin(angle))
    
    pygame.gfxdraw.aapolygon(surface, (ul, ur, br, bl), color)
    pygame.gfxdraw.filled_polygon(surface, (ul, ur, br, bl), color)

class Particle:
    def __init__(self, pos, color, target_idx):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.color = color
        self.target_idx = target_idx
        self.radius = 6
        self.trail = [self.pos.copy() for _ in range(15)]

    def update(self, total_force, bounds):
        # Physics update
        self.vel += total_force
        self.vel *= 0.98  # Drag
        if self.vel.length() > 5: # Speed limit
            self.vel.scale_to_length(5)
        self.pos += self.vel

        # Boundary collision
        if self.pos.x - self.radius < 0:
            self.pos.x = self.radius
            self.vel.x *= -0.8
        elif self.pos.x + self.radius > bounds[0]:
            self.pos.x = bounds[0] - self.radius
            self.vel.x *= -0.8
        if self.pos.y - self.radius < 0:
            self.pos.y = self.radius
            self.vel.y *= -0.8
        elif self.pos.y + self.radius > bounds[1]:
            self.pos.y = bounds[1] - self.radius
            self.vel.y *= -0.8
            
        # Update trail
        self.trail.pop(0)
        self.trail.append(self.pos.copy())

    def draw(self, surface):
        # Draw trail
        for i, p in enumerate(self.trail):
            alpha = int(255 * (i / len(self.trail)) * 0.5)
            color = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(p.x), int(p.y), int(self.radius * (i / len(self.trail))), color)

        # Draw glow
        glow_radius = int(self.radius * 1.8)
        glow_color = self.color + (50,)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), glow_radius, glow_color)
        
        # Draw core particle
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)


class MagneticField:
    def __init__(self, pos, color):
        self.pos = pygame.Vector2(pos)
        self.strength = 0.0
        self.direction = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
        self.color = color
        self.radius = 15
        self.min_strength = -1.5
        self.max_strength = 1.5

    def adjust_strength(self, amount):
        self.strength = np.clip(self.strength + amount, self.min_strength, self.max_strength)

    def adjust_direction(self, d_vec):
        self.direction += d_vec
        if self.direction.length() > 0:
            self.direction.normalize_ip()

    def draw(self, surface, is_selected):
        # Draw force lines
        num_lines = int(abs(self.strength) * 6)
        for i in range(num_lines):
            alpha = 100 - i * 10
            if alpha > 10:
                radius = self.radius + 10 + i * 8
                pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), radius, self.color + (alpha,))

        # Draw direction indicator
        end_pos = self.pos + self.direction * (self.radius + 10 + abs(self.strength) * 15)
        indicator_color = (255, 255, 255, 200)
        draw_line_antialiased(surface, indicator_color, self.pos, end_pos, width=3)
        
        # Draw core
        core_color = self.color
        if is_selected:
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius + 4, (255, 255, 255, 80))
            core_color = (255, 255, 255)

        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, core_color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, core_color)
        
        # Draw strength indicator (positive/negative)
        strength_color = (100, 255, 100) if self.strength > 0 else (255, 100, 100)
        if abs(self.strength) > 0.01:
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius - 4, strength_color)


class TargetZone:
    def __init__(self, pos, color, radius):
        self.pos = pygame.Vector2(pos)
        self.color = color
        self.radius = radius
        self.pulse = 0
        self.pulse_speed = 0.1

    def update(self):
        self.pulse += self.pulse_speed
        if self.pulse > math.pi * 2:
            self.pulse = 0

    def draw(self, surface):
        alpha = int(70 + 30 * math.sin(self.pulse))
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color + (alpha // 2,))
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color + (alpha,))
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius - 2, self.color + (alpha,))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    game_description = (
        "Use controllable magnetic fields to guide particles into their matching target zones before time runs out."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to aim the selected magnetic field. Hold space to increase its strength, hold shift to decrease it. Press shift to cycle between fields."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_TIME = 90  # seconds

        # EXACT spaces:
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_info = pygame.font.SysFont("Consolas", 18)
        
        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.PARTICLE_COLORS = [(255, 80, 80), (80, 255, 80), (80, 150, 255)]
        self.FIELD_COLOR = (200, 200, 255)

        # Game entities (initialized in reset)
        self.particles = []
        self.fields = []
        self.zones = []
        self.effects = []
        self.last_distances = []

        # Game state (initialized in reset)
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.selected_field_index = 0
        self.prev_shift_state = 0
        self.terminated = False
        
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.display = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Magnetic Manipulation")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_TIME
        self.terminated = False
        self.selected_field_index = 0
        self.prev_shift_state = 0
        self.effects = []

        # Create Target Zones
        self.zones = [
            TargetZone((80, self.HEIGHT // 2), self.PARTICLE_COLORS[0], 35),
            TargetZone((self.WIDTH // 2, 80), self.PARTICLE_COLORS[1], 35),
            TargetZone((self.WIDTH - 80, self.HEIGHT // 2), self.PARTICLE_COLORS[2], 35)
        ]

        # Create Magnetic Fields
        self.fields = [
            MagneticField((self.WIDTH * 0.25, self.HEIGHT * 0.25), self.FIELD_COLOR),
            MagneticField((self.WIDTH * 0.75, self.HEIGHT * 0.25), self.FIELD_COLOR),
            MagneticField((self.WIDTH * 0.5, self.HEIGHT * 0.5), self.FIELD_COLOR),
            MagneticField((self.WIDTH * 0.25, self.HEIGHT * 0.75), self.FIELD_COLOR),
            MagneticField((self.WIDTH * 0.75, self.HEIGHT * 0.75), self.FIELD_COLOR),
        ]
        for field in self.fields:
            field.strength = 0.0
            field.direction = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize()

        # Create Particles
        self.particles = []
        num_particles = 25
        for i in range(num_particles):
            target_idx = i % len(self.zones)
            pos = (
                self.np_random.uniform(self.WIDTH * 0.4, self.WIDTH * 0.6),
                self.np_random.uniform(self.HEIGHT * 0.4, self.HEIGHT * 0.6)
            )
            particle = Particle(pos, self.PARTICLE_COLORS[target_idx], target_idx)
            self.particles.append(particle)
            
        self.last_distances = [p.pos.distance_to(self.zones[p.target_idx].pos) for p in self.particles]

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Action Handling ---
        # Adaptation: Shift-press (0->1 transition) cycles selected field
        if shift_held and not self.prev_shift_state:
            self.selected_field_index = (self.selected_field_index + 1) % len(self.fields)
            # sfx: UI_Cycle.wav
        self.prev_shift_state = shift_held
        
        selected_field = self.fields[self.selected_field_index]
        
        # Action 0: Movement -> Adjust direction
        dir_change = pygame.Vector2(0, 0)
        if movement == 1: dir_change.y = -1 # Up
        elif movement == 2: dir_change.y = 1  # Down
        elif movement == 3: dir_change.x = -1 # Left
        elif movement == 4: dir_change.x = 1  # Right
        if dir_change.length() > 0:
            selected_field.adjust_direction(dir_change * 0.1)
            
        # Action 1: Space -> Increase strength
        if space_held:
            selected_field.adjust_strength(0.05)
            # sfx: Field_Charge.wav
            
        # Action 2: Shift (hold) -> Decrease strength (Note: press cycles, hold decreases)
        if shift_held:
            selected_field.adjust_strength(-0.05)
            # sfx: Field_Discharge.wav

        # --- Game Logic & Physics ---
        for i, p in enumerate(self.particles):
            total_force = pygame.Vector2(0, 0)
            for field in self.fields:
                vec_to_particle = p.pos - field.pos
                dist_sq = vec_to_particle.length_squared()
                if dist_sq > 1:
                    force_magnitude = field.strength / dist_sq
                    # In this model, "direction" is a uniform vector, not a radial one
                    total_force += field.direction * force_magnitude * 100
            
            p.update(total_force, (self.WIDTH, self.HEIGHT))
        
        for zone in self.zones:
            zone.update()
        
        self.effects = [e for e in self.effects if e['life'] > 0]
        for effect in self.effects:
            effect['life'] -= 1

        # --- Reward Calculation & Events ---
        captured_this_step = []
        remaining_particles = []
        
        # Continuous reward for moving closer
        current_distances = [p.pos.distance_to(self.zones[p.target_idx].pos) for p in self.particles]
        for i in range(len(self.particles)):
            reward += (self.last_distances[i] - current_distances[i]) * 0.01
        self.last_distances = current_distances

        for p in self.particles:
            target_zone = self.zones[p.target_idx]
            if p.pos.distance_to(target_zone.pos) < target_zone.radius:
                captured_this_step.append(p)
                self.score += 10
                reward += 10
                # sfx: Particle_Capture.wav
                self._add_effect('capture', target_zone.pos, p.color)
            else:
                remaining_particles.append(p)
        
        self.particles = remaining_particles
        self.last_distances = [p.pos.distance_to(self.zones[p.target_idx].pos) for p in self.particles]

        # Chain reaction bonus
        if len(captured_this_step) > 1:
            chain_bonus = 5 * (len(captured_this_step) - 1)
            self.score += chain_bonus
            reward += chain_bonus
            # sfx: Chain_Reaction.wav

        # --- Termination Check ---
        self.steps += 1
        self.time_remaining -= 1 / self.FPS
        
        if not self.particles:
            self.terminated = True
            reward += 50 # Win bonus
            self.score += 50
            # sfx: Win.wav
        elif self.time_remaining <= 0:
            self.terminated = True
            # No explicit negative reward, letting the lack of a win bonus be the penalty

        return self._get_observation(), reward, self.terminated, False, self._get_info()

    def render(self):
        if self.render_mode == "human":
            self.display.blit(self.screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.FPS)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.render_mode == "human":
            self.render()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, (25, 30, 45), (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, (25, 30, 45), (0, y), (self.WIDTH, y))

        for zone in self.zones:
            zone.draw(self.screen)
            
        for i, field in enumerate(self.fields):
            field.draw(self.screen, i == self.selected_field_index)

        for particle in self.particles:
            particle.draw(self.screen)
            
        for effect in self.effects:
            if effect['type'] == 'capture':
                for p_info in effect['particles']:
                    p_info['pos'] += p_info['vel']
                    p_info['vel'].y += 0.1 # gravity
                    alpha = 255 * (effect['life'] / effect['max_life'])
                    color = effect['color'] + (int(alpha),)
                    pygame.gfxdraw.filled_circle(self.screen, int(p_info['pos'].x), int(p_info['pos'].y), 3, color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Timer
        timer_color = (255, 255, 255) if self.time_remaining > 10 else (255, 100, 100)
        timer_text = self.font_ui.render(f"TIME: {max(0, self.time_remaining):.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Selected Field Info
        field_text = self.font_info.render(f"FIELD: {self.selected_field_index + 1}/{len(self.fields)} (SHIFT-press to cycle)", True, (200, 200, 200))
        self.screen.blit(field_text, (10, self.HEIGHT - field_text.get_height() - 10))
        
        if self.terminated:
            msg = "ALL PARTICLES CAPTURED!" if not self.particles else "TIME UP!"
            color = (100, 255, 100) if not self.particles else (255, 100, 100)
            end_text = self.font_ui.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), end_rect.inflate(20,20))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "particles_left": len(self.particles),
            "selected_field": self.selected_field_index
        }

    def _add_effect(self, type, pos, color):
        if type == 'capture':
            effect = {'type': type, 'pos': pos, 'color': color, 'life': 30, 'max_life': 30, 'particles': []}
            for _ in range(15):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 4)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                effect['particles'].append({'pos': pos.copy(), 'vel': vel})
            self.effects.append(effect)
            
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Controls ---
    # Arrows: Adjust Direction
    # Space: Increase Strength
    # Left Shift: Decrease Strength AND Press to cycle field
    
    while not done:
        # Simple human control mapping for testing
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

    env.close()