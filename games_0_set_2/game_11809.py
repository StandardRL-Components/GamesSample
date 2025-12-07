import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:38:09.651072
# Source Brief: brief_01809.md
# Brief Index: 1809
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Spirit:
    """Represents a single elemental spirit."""
    def __init__(self, pos, spirit_type, color):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.type = spirit_type
        self.color = color
        self.radius = 10
        self.target_radius = 10
        self.creation_time = pygame.time.get_ticks()

    def update(self, attraction_force, repulsion_force, center, bounds):
        """Updates the spirit's position and velocity."""
        # Apply forces
        self.vel += attraction_force
        self.vel += repulsion_force

        # Apply drag
        self.vel *= 0.9

        # Update position
        self.pos += self.vel

        # Bounce off walls
        if self.pos.x - self.radius < 0 or self.pos.x + self.radius > bounds[0]:
            self.vel.x *= -0.8
            self.pos.x = np.clip(self.pos.x, self.radius, bounds[0] - self.radius)
        if self.pos.y - self.radius < 0 or self.pos.y + self.radius > bounds[1]:
            self.vel.y *= -0.8
            self.pos.y = np.clip(self.pos.y, self.radius, bounds[1] - self.radius)

        # Pulsing effect for visual flair
        pulse_speed = 4
        self.radius = self.target_radius + math.sin((pygame.time.get_ticks() - self.creation_time) / 1000 * pulse_speed) * 2


class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, pos, vel, color, start_radius, life):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.color = color
        self.radius = start_radius
        self.life = life
        self.max_life = life

    def update(self):
        """Updates particle's position and life."""
        self.pos += self.vel
        self.vel *= 0.98  # Particle drag
        self.life -= 1
        return self.life > 0

class Wave:
    """Represents an incoming energy wave."""
    def __init__(self, angle, speed, screen_size):
        self.radius = max(screen_size) / 2 + 50
        self.speed = speed
        self.angle = angle # For rendering an arc
        self.width = 10
        self.damage = 25
        self.color = (255, 100, 0)

    def update(self):
        """Moves the wave towards the center."""
        self.radius -= self.speed
        return self.radius > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend the core by guiding elemental spirits to combine and strengthen your shield against incoming energy waves."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to attract spirits. Hold space to repel spirits from the center. Combine spirits near the center to recharge your shield."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    CENTER = (WIDTH // 2, HEIGHT // 2)
    SHIELD_RADIUS = 60
    SPIRIT_COMBINE_RADIUS = 40
    ATTRACTION_FORCE = 0.25
    REPULSION_FORCE = 0.5
    MAX_STEPS = 1500

    # --- Colors ---
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 10)
    COLOR_SHIELD = (170, 100, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BAR_BG = (50, 50, 80)
    COLOR_UI_BAR_FG = (100, 200, 255)

    SPIRIT_DATA = {
        'fire': {'color': (255, 80, 80)},
        'water': {'color': (80, 150, 255)},
        'earth': {'color': (80, 255, 120)},
        'air': {'color': (255, 255, 100)},
    }
    SPIRIT_UNLOCK_ORDER = ['fire', 'water', 'earth', 'air']

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("sans", 24)
        self.font_small = pygame.font.SysFont("sans", 16)

        self.unlocked_spirits = [self.SPIRIT_UNLOCK_ORDER[0]]
        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.shield_strength = 100.0
        self.max_shield_strength = 100.0
        self.target_shield_strength = 100.0

        self.current_wave_num = 1
        self.last_spirit_unlock_wave = 0
        self.wave_timer = 0
        self.wave_cooldown = 250
        self.wave_speed = 1.0

        self.spirits = []
        self.waves = []
        self.particles = []
        
        for _ in range(3):
            self._spawn_spirit()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # --- Update Game Logic ---
        self._handle_input(action)
        
        reward += self._update_spirits()
        wave_reward, damage = self._update_waves()
        reward += wave_reward
        self.target_shield_strength -= damage

        # Smoothly interpolate shield strength for visual appeal
        self.shield_strength = self.shield_strength * 0.9 + self.target_shield_strength * 0.1

        self._update_particles()
        
        progression_reward = self._update_progression()
        reward += progression_reward

        # --- Calculate Rewards & Termination ---
        reward += 0.01  # Small survival reward per step

        self.steps += 1
        terminated = (self.shield_strength <= 0.1) or (self.steps >= self.MAX_STEPS)
        truncated = False # Not using truncation

        if terminated:
            if self.shield_strength <= 0.1:
                reward = -100.0  # Game over penalty
                # Sound: game_over.wav
            else: # Max steps reached
                reward += 100.0 # Victory bonus
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action
        attraction_force = pygame.math.Vector2(0, 0)
        repulsion_force = pygame.math.Vector2(0, 0)

        if movement == 1: attraction_force.y = -self.ATTRACTION_FORCE
        elif movement == 2: attraction_force.y = self.ATTRACTION_FORCE
        elif movement == 3: attraction_force.x = -self.ATTRACTION_FORCE
        elif movement == 4: attraction_force.x = self.ATTRACTION_FORCE

        for spirit in self.spirits:
            repel_vec = pygame.math.Vector2(0,0)
            if space_held:
                # Sound: repel_pulse.wav (on press)
                dist_from_center = spirit.pos.distance_to(self.CENTER)
                if dist_from_center > 1: # Avoid division by zero
                    repel_vec = (spirit.pos - pygame.math.Vector2(self.CENTER)).normalize() * self.REPULSION_FORCE
            
            spirit.update(attraction_force, repel_vec, self.CENTER, (self.WIDTH, self.HEIGHT))

    def _update_spirits(self):
        """Handle spirit movement, collisions, and combinations."""
        reward = 0
        to_combine = []
        for i, s1 in enumerate(self.spirits):
            if s1.pos.distance_to(self.CENTER) < self.SPIRIT_COMBINE_RADIUS:
                for j, s2 in enumerate(self.spirits):
                    if i < j and s2.pos.distance_to(self.CENTER) < self.SPIRIT_COMBINE_RADIUS:
                        if s1.pos.distance_to(s2.pos) < s1.radius + s2.radius:
                            to_combine.append((i, j))
        
        if to_combine:
            # For simplicity, combine the first detected pair
            i, j = to_combine[0]
            s1, s2 = self.spirits[i], self.spirits[j]
            
            # Sound: spirit_combine.wav
            self._create_particles(s1.pos.lerp(s2.pos, 0.5), 30, self.COLOR_SHIELD, 5)
            self.target_shield_strength = min(self.max_shield_strength, self.target_shield_strength + 20)
            self.score += 10
            reward += 1.0

            # Remove combined spirits and spawn new ones
            self.spirits.pop(max(i, j))
            self.spirits.pop(min(i, j))
            self._spawn_spirit()
            self._spawn_spirit()

        return reward

    def _update_waves(self):
        """Handle wave spawning, movement, and collision with shield."""
        reward = 0
        damage_taken = 0
        self.wave_timer += 1
        if self.wave_timer >= self.wave_cooldown:
            # A wave cycle is complete
            self.wave_timer = 0
            self.current_wave_num += 1
            self.score += 5
            reward += 5.0 # Survived a wave
            self._spawn_wave()
            # Sound: new_wave_warning.wav

        hit_waves = []
        for i, wave in enumerate(self.waves):
            wave.update()
            if wave.radius < self.SHIELD_RADIUS:
                # Sound: shield_hit.wav
                damage_taken += wave.damage
                hit_waves.append(i)
                self._create_particles(pygame.math.Vector2(self.CENTER), 20, wave.color, 4, offset_radius=self.SHIELD_RADIUS)
        
        for i in sorted(hit_waves, reverse=True):
            del self.waves[i]

        return reward, damage_taken

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _update_progression(self):
        """Update difficulty and unlock new spirits."""
        reward = 0
        # Increase wave speed
        if self.current_wave_num > 0 and self.current_wave_num % 5 == 0:
            self.wave_speed = min(3.0, self.wave_speed + 0.001) # Incremental increase
        
        # Unlock new spirits
        if self.current_wave_num // 10 > self.last_spirit_unlock_wave:
            self.last_spirit_unlock_wave = self.current_wave_num // 10
            if len(self.unlocked_spirits) < len(self.SPIRIT_UNLOCK_ORDER):
                new_spirit = self.SPIRIT_UNLOCK_ORDER[len(self.unlocked_spirits)]
                self.unlocked_spirits.append(new_spirit)
                reward += 50.0
                self.score += 100
                # Sound: spirit_unlocked.wav
                # Create a special particle effect for unlocking
                self._create_particles(pygame.math.Vector2(self.CENTER), 100, self.SPIRIT_DATA[new_spirit]['color'], 8)
        return reward

    def _spawn_spirit(self):
        side = self.np_random.integers(4)
        if side == 0: x, y = self.np_random.uniform(0, self.WIDTH), -20
        elif side == 1: x, y = self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20
        elif side == 2: x, y = -20, self.np_random.uniform(0, self.HEIGHT)
        else: x, y = self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT)
        
        spirit_type = self.np_random.choice(self.unlocked_spirits)
        color = self.SPIRIT_DATA[spirit_type]['color']
        self.spirits.append(Spirit((x, y), spirit_type, color))

    def _spawn_wave(self):
        angle = self.np_random.uniform(0, 2 * math.pi)
        self.waves.append(Wave(angle, self.wave_speed, (self.WIDTH, self.HEIGHT)))
    
    def _create_particles(self, pos, count, color, speed_mult, offset_radius=0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            start_pos = pos + vel.normalize() * offset_radius
            life = self.np_random.integers(20, 40)
            radius = self.np_random.uniform(2, 5)
            self.particles.append(Particle(start_pos, vel, color, radius, life))

    def _get_observation(self):
        self._render_background()
        self._render_shield()
        for wave in self.waves: self._render_wave(wave)
        for particle in self.particles: self._render_particle(particle)
        for spirit in self.spirits: self._render_spirit(spirit)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave_num}

    def _draw_glow_circle(self, surface, pos, radius, color, glow_width=10):
        """Draws a circle with a glowing aura."""
        x, y = int(pos[0]), int(pos[1])
        r, g, b = color
        
        # Draw multiple transparent layers for the glow
        for i in range(glow_width, 0, -2):
            alpha = int(100 * (1 - i / glow_width))
            pygame.gfxdraw.filled_circle(surface, x, y, int(radius + i), (r, g, b, alpha))
            pygame.gfxdraw.aacircle(surface, x, y, int(radius + i), (r, g, b, alpha))

        # Draw the solid center
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius), color)
        pygame.gfxdraw.aacircle(surface, x, y, int(radius), color)

    def _render_background(self):
        """Draws a vertical gradient for the background."""
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        # Faint summoning circles
        for i in range(1, 5):
            radius = self.SHIELD_RADIUS + i * 40
            alpha = 30 - i * 5
            pygame.gfxdraw.aacircle(self.screen, self.CENTER[0], self.CENTER[1], radius, (*self.COLOR_SHIELD, alpha))

    def _render_shield(self):
        # Draw shield bar
        bar_angle = (self.shield_strength / self.max_shield_strength) * 360
        bar_rect = pygame.Rect(self.CENTER[0] - self.SHIELD_RADIUS - 10, self.CENTER[1] - self.SHIELD_RADIUS - 10, (self.SHIELD_RADIUS+10)*2, (self.SHIELD_RADIUS+10)*2)
        pygame.draw.arc(self.screen, self.COLOR_UI_BAR_BG, bar_rect, 0, 2*math.pi, 5)
        if bar_angle > 0:
            pygame.draw.arc(self.screen, self.COLOR_UI_BAR_FG, bar_rect, math.pi/2, math.pi/2 + math.radians(bar_angle), 5)

        # Draw shield visual effect
        num_layers = int(self.shield_strength / 20)
        for i in range(num_layers):
            radius = self.SHIELD_RADIUS - i * 4
            alpha = int(20 + (self.shield_strength / self.max_shield_strength) * 80)
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, self.CENTER[0], self.CENTER[1], int(radius), (*self.COLOR_SHIELD, alpha))

    def _render_spirit(self, spirit):
        self._draw_glow_circle(self.screen, spirit.pos, spirit.radius, spirit.color, glow_width=15)

    def _render_wave(self, wave):
        rect = pygame.Rect(self.CENTER[0] - wave.radius, self.CENTER[1] - wave.radius, wave.radius * 2, wave.radius * 2)
        if rect.width > 0 and rect.height > 0:
            arc_length = math.pi / 2 # How wide the arc is
            start_angle = wave.angle - arc_length / 2
            end_angle = wave.angle + arc_length / 2
            
            # Glow effect for wave
            for i in range(5, 0, -1):
                alpha = int(150 * (1 - i / 5))
                pygame.draw.arc(self.screen, (*wave.color, alpha), rect, start_angle, end_angle, wave.width + i*2)
            pygame.draw.arc(self.screen, wave.color, rect, start_angle, end_angle, wave.width)

    def _render_particle(self, particle):
        alpha = int(255 * (particle.life / particle.max_life))
        r, g, b = particle.color
        color = (r, g, b, alpha)
        if particle.radius > 0:
            pygame.gfxdraw.filled_circle(self.screen, int(particle.pos.x), int(particle.pos.y), int(particle.radius), color)

    def _render_ui(self):
        # Wave number
        wave_text = self.font_main.render(f"Wave: {self.current_wave_num}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Unlocked spirits
        unlocked_text = self.font_small.render("Unlocked Spirits:", True, self.COLOR_UI_TEXT)
        self.screen.blit(unlocked_text, (10, self.HEIGHT - 35))
        for i, spirit_type in enumerate(self.unlocked_spirits):
            color = self.SPIRIT_DATA[spirit_type]['color']
            pos = (130 + i * 30, self.HEIGHT - 25)
            self._draw_glow_circle(self.screen, pos, 8, color, 5)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the headless environment but is useful for local development
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display for local testing
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Spirit Shield")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

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
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            done = True
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()