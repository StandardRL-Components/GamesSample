import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:26:12.035809
# Source Brief: brief_02163.md
# Brief Index: 2163
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for musical chords
class Chord:
    def __init__(self, x, y, chord_type, chord_data, width, height):
        self.x = x
        self.y = y
        self.type = chord_type
        self.data = chord_data
        self.width = width
        self.height = height
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.alive = True

    def update(self, gravity, speed):
        self.y += speed * gravity
        self.rect.topleft = (int(self.x), int(self.y))

    def draw(self, surface, font):
        # Glow effect
        glow_rect = self.rect.inflate(8, 8)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.data['color'] + (60,), (0, 0, glow_rect.width, glow_rect.height), border_radius=10)
        surface.blit(glow_surf, glow_rect.topleft)

        # Main rectangle
        pygame.draw.rect(surface, self.data['color'], self.rect, border_radius=8)
        pygame.draw.rect(surface, (255, 255, 255), self.rect, 2, border_radius=8)

        # Text
        text_surf = font.render(self.type, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

# Helper class for Dissonance Waves
class Wave:
    def __init__(self, y, speed, screen_width):
        self.x = -150.0
        self.y = y
        self.speed = speed
        self.screen_width = screen_width
        self.amplitude = 15
        self.frequency = 0.04
        self.alive = True
        self.collided = False

    def update(self, chords):
        self.x += self.speed
        if self.x > self.screen_width + 150:
            self.alive = False
            return
        
        # Simplified collision check for performance
        wave_y_min = self.y - self.amplitude
        wave_y_max = self.y + self.amplitude
        for chord in chords:
            if chord.rect.top < wave_y_max and chord.rect.bottom > wave_y_min:
                # AABB collision on a segment of the wave
                # Check if the chord's horizontal span overlaps with the wave's current position
                if chord.rect.left < self.x < chord.rect.right:
                    self.collided = True
                    return

    def draw(self, surface):
        points = []
        num_points = int(self.screen_width + 300)
        for i in range(num_points):
            px = self.x + i - 150
            py = self.y + math.sin((self.x + i) * self.frequency) * self.amplitude
            points.append((px, py))
        
        # Draw multiple lines for a glowing/electric effect
        if len(points) > 1:
            pygame.draw.aalines(surface, (255, 0, 100, 100), False, points, 5)
            pygame.draw.aalines(surface, (255, 100, 150, 150), False, points, 3)
            pygame.draw.aalines(surface, (255, 255, 255), False, points, 1)

# Helper class for visual effect particles
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.radius = random.uniform(3, 7)
        self.color = color
        self.lifetime = random.randint(30, 60)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        self.radius -= 0.1

    def draw(self, surface):
        if self.lifetime > 0 and self.radius > 0:
            alpha = max(0, min(255, int(255 * (self.lifetime / 60))))
            try:
                pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), self.color + (alpha,))
            except OverflowError: # Catches potential errors if radius becomes huge/invalid
                pass


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Dodge dissonance waves by flipping gravity. Combine harmonically compatible chords "
        "to score points and survive in this musical arcade challenge."
    )
    user_guide = (
        "Controls: Press space to flip gravity. Press shift to combine nearby compatible chords for points. "
        "Avoid the dissonance waves!"
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Pygame setup
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_chord = pygame.font.SysFont("Consolas", 18, bold=True)
        
        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Game constants
        self.COLOR_BG_TOP = (10, 5, 25)
        self.COLOR_BG_BOTTOM = (30, 15, 50)
        self.COLOR_WAVE = (255, 0, 100)
        self.COLOR_TEXT = (220, 220, 255)
        
        self.CHORD_FALL_SPEED = 1.0
        self.CHORD_SPAWN_INTERVAL = 90
        self.WAVE_SPAWN_INTERVAL = 250
        self.MAX_STEPS = 5000
        
        # Chord progression data
        self.CHORD_DATA = {
            'Cmaj': {'color': (66, 135, 245), 'next': 'Gmaj'},
            'Gmaj': {'color': (245, 132, 66), 'next': 'Amin'},
            'Amin': {'color': (188, 66, 245), 'next': 'Fmaj'},
            'Fmaj': {'color': (66, 245, 158), 'next': 'Cmaj'},
        }
        self.CHORD_TYPES = list(self.CHORD_DATA.keys())

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gravity = 1
        self.chords = []
        self.waves = []
        self.particles = []
        self.chord_spawn_timer = 0
        self.wave_spawn_timer = 0
        self.wave_speed = 0.0
        self.score_milestone = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gravity = 1  # 1 for down, -1 for up
        
        self.chords = []
        self.waves = []
        self.particles = []
        
        self.chord_spawn_timer = self.CHORD_SPAWN_INTERVAL
        self.wave_spawn_timer = self.WAVE_SPAWN_INTERVAL
        self.wave_speed = 1.0
        self.score_milestone = 100
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        # 1. Unpack and handle actions (on press, not hold)
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        if space_held and not self.prev_space_held:
            self._flip_gravity() # Sound: "Gravity_Flip.wav"
        if shift_held and not self.prev_shift_held:
            reward += self._combine_chords() # Sound: "Chord_Combine.wav"
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # 2. Update game logic
        self.steps += 1
        self._spawn_entities()
        
        # Update chords
        for chord in self.chords:
            chord.update(self.gravity, self.CHORD_FALL_SPEED)
        self.chords = [c for c in self.chords if -c.height < c.y < self.HEIGHT]

        # Update waves and check for collisions
        passed_waves = 0
        for wave in self.waves:
            wave.update(self.chords)
            if wave.collided:
                self.game_over = True # Sound: "Dissonance_Hit.wav"
                break
            if not wave.alive:
                passed_waves += 1
        if not self.game_over:
            reward += passed_waves * 1.0
        self.waves = [w for w in self.waves if w.alive]
        
        # Update particles
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifetime > 0 and p.radius > 0]

        # 3. Calculate rewards and check termination
        if self.score >= self.score_milestone:
            reward += 10.0
            self.score_milestone += 100

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        if self.game_over:
            reward = -100.0
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _flip_gravity(self):
        self.gravity *= -1
        # Visual feedback for gravity flip
        self._create_particles(self.WIDTH / 2, self.HEIGHT / 2, (200, 200, 255), 20)

    def _combine_chords(self):
        to_remove = []
        reward = 0.0
        for i in range(len(self.chords)):
            for j in range(i + 1, len(self.chords)):
                c1 = self.chords[i]
                c2 = self.chords[j]
                
                if not c1.alive or not c2.alive:
                    continue

                # Check for vertical alignment and proximity
                if abs(c1.rect.centerx - c2.rect.centerx) < 20 and abs(c1.rect.centery - c2.rect.centery) < 60:
                    # Check for harmonic compatibility
                    c1_next = self.CHORD_DATA[c1.type]['next']
                    c2_next = self.CHORD_DATA[c2.type]['next']
                    
                    if c1_next == c2.type or c2_next == c1.type:
                        c1.alive = False
                        c2.alive = False
                        self.score += 10
                        reward += 0.1
                        
                        # Update wave speed based on new score
                        self.wave_speed = 1.0 + (self.score // 500) * 0.05

                        mid_x = (c1.rect.centerx + c2.rect.centerx) / 2
                        mid_y = (c1.rect.centery + c2.rect.centery) / 2
                        avg_color = tuple( (c1.data['color'][k] + c2.data['color'][k]) // 2 for k in range(3) )
                        self._create_particles(mid_x, mid_y, avg_color, 40)
                        
        if any(not c.alive for c in self.chords):
            self.chords = [c for c in self.chords if c.alive]
        return reward

    def _spawn_entities(self):
        # Spawn chords
        self.chord_spawn_timer -= 1
        if self.chord_spawn_timer <= 0:
            self.chord_spawn_timer = self.CHORD_SPAWN_INTERVAL
            spawn_x = random.uniform(20, self.WIDTH - 100)
            spawn_y = -40 if self.gravity > 0 else self.HEIGHT
            chord_type = random.choice(self.CHORD_TYPES)
            self.chords.append(Chord(spawn_x, spawn_y, chord_type, self.CHORD_DATA[chord_type], 80, 40))
        
        # Spawn waves
        self.wave_spawn_timer -= 1
        if self.wave_spawn_timer <= 0:
            self.wave_spawn_timer = self.WAVE_SPAWN_INTERVAL
            spawn_y = random.uniform(50, self.HEIGHT - 50)
            self.waves.append(Wave(spawn_y, self.wave_speed, self.WIDTH))

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_background(self):
        # Draw a smooth vertical gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp),
            )
            self.screen.fill(color, (0, y, self.WIDTH, 1))

    def _render_game(self):
        for wave in self.waves:
            wave.draw(self.screen)
        for chord in self.chords:
            chord.draw(self.screen, self.font_chord)
        for particle in self.particles:
            particle.draw(self.screen)

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Gravity indicator
        arrow_color = (200, 200, 255)
        if self.gravity > 0: # Down
            points = [(self.WIDTH - 30, 20), (self.WIDTH - 20, 35), (self.WIDTH - 10, 20)]
            pygame.draw.polygon(self.screen, arrow_color, points, 0)
        else: # Up
            points = [(self.WIDTH - 30, 35), (self.WIDTH - 20, 20), (self.WIDTH - 10, 35)]
            pygame.draw.polygon(self.screen, arrow_color, points, 0)

    def close(self):
        pygame.quit()
    
    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block is for interactive testing and will not be run by the evaluation system.
    # It has been modified to use a separate display window.
    env = GameEnv()
    
    # Override render method for human play
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Chord Gravity")
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        # Map keyboard keys to actions for human play
        # 0=none, 1=up, 2=down, 3=left, 4=right
        # space = 0 released, 1 held
        # shift = 0 released, 1 held
        
        # This part is for human interaction, not used in the gym environment itself
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # Default no-op action
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to the human-visible screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS for smooth visuals

    env.close()
    print("Game Over. Final Info:", info)