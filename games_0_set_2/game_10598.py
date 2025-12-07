import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:43:57.514353
# Source Brief: brief_00598.md
# Brief Index: 598
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes for Game Entities ---

class Particle:
    """A single particle for effects like explosions and trails."""
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = list(pos)
        self.vel = list(vel)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifetime -= 1
        self.radius *= 0.98

    def draw(self, surface):
        if self.lifetime > 0 and self.radius > 0.5:
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            # Use a simple circle for performance
            # Create a temporary surface for the particle to handle alpha properly
            temp_surf = pygame.Surface((int(self.radius*2), int(self.radius*2)), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (int(self.radius), int(self.radius)), int(self.radius))
            surface.blit(temp_surf, (int(self.pos[0] - self.radius), int(self.pos[1] - self.radius)))


class Missile:
    """An incoming missile that the player must destroy."""
    def __init__(self, x_pos, speed, frequency, screen_width):
        self.pos = [x_pos, -20.0]
        self.speed = speed
        self.frequency = frequency  # Target frequency (0.0 to 1.0)
        self.color = (255, 68, 68) # Bright Red
        self.trail_particles = []
        self.width = 12
        self.height = 25

    def update(self):
        self.pos[1] += self.speed
        # Add trail particles
        if random.random() < 0.8:
            trail_pos = (self.pos[0], self.pos[1] - self.height / 2)
            trail_vel = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
            trail_radius = random.uniform(2, 4)
            trail_lifetime = random.randint(10, 20)
            self.trail_particles.append(Particle(trail_pos, trail_vel, trail_radius, (255, 120, 0), trail_lifetime))

        # Update and remove old trail particles
        for p in self.trail_particles:
            p.update()
        self.trail_particles = [p for p in self.trail_particles if p.lifetime > 0]


    def draw(self, surface):
        # Draw trail first
        for p in self.trail_particles:
            p.draw(surface)

        # Draw missile body (a sharp triangle)
        p1 = (int(self.pos[0]), int(self.pos[1] - self.height / 2))
        p2 = (int(self.pos[0] - self.width / 2), int(self.pos[1] + self.height / 2))
        p3 = (int(self.pos[0] + self.width / 2), int(self.pos[1] + self.height / 2))
        
        # Glow effect
        pygame.gfxdraw.filled_trigon(surface, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], (*self.color, 50))
        pygame.gfxdraw.aatrigon(surface, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], (*self.color, 50))
        
        # Main body
        pygame.gfxdraw.filled_trigon(surface, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.color)
        pygame.gfxdraw.aatrigon(surface, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.color)


class SonicWave:
    """A wave emitted by the player."""
    def __init__(self, pos, frequency, amplitude):
        self.pos = list(pos)
        self.frequency = frequency # 0.0 to 1.0
        self.amplitude = amplitude # 0.1 to 1.0
        self.speed = 8.0
        self.radius = 10.0
        self.max_radius = 20 + self.amplitude * 250 # Wave width depends on amplitude
        self.lifetime = 60 # How many frames the wave visual persists
        self.color_start = (68, 170, 255) # Blue
        self.color_end = (68, 255, 170) # Green
        # Interpolate color based on frequency
        self.color = tuple(int(s + (e - s) * self.frequency) for s, e in zip(self.color_start, self.color_end))

    def update(self):
        self.pos[1] -= self.speed
        self.radius += self.max_radius / 60 # Grow to max radius over lifetime
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(150 * (self.lifetime / 60))
            # Draw expanding ring
            pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), int(self.radius), (*self.color, alpha))
            pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), int(self.radius)-2, (*self.color, alpha))
    
    def get_hitbox(self):
        return pygame.Rect(self.pos[0] - self.radius, self.pos[1] - 5, self.radius * 2, 10)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your base from incoming missiles by firing sonic waves. Match the frequency and amplitude of your "
        "waves to intercept and destroy targets before they reach the city."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to adjust wave frequency and ←→ to adjust amplitude. "
        "Press space to fire a wave."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    INITIAL_MISSILES = 10
    
    COLOR_BG = (13, 10, 31)
    COLOR_GRID = (37, 33, 60)
    COLOR_UI_TEXT = (200, 200, 255)
    COLOR_EMITTER = (0, 255, 255)

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.emitter_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 20)
        self.emitter_freq = 0.5
        self.emitter_amp = 0.1
        self.wave_cooldown = 0
        self.missiles = []
        self.waves = []
        self.particles = []
        self.missile_speed_mod = 0.0
        self.missile_freq_range_mod = 0.0
        self.missiles_to_spawn = self.INITIAL_MISSILES
        self.missiles_destroyed = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        
        self.emitter_freq = 0.5
        self.emitter_amp = 0.1
        self.wave_cooldown = 0
        
        self.missiles.clear()
        self.waves.clear()
        self.particles.clear()
        
        self.missile_speed_mod = 1.0
        self.missile_freq_range_mod = 0.1
        self.missiles_to_spawn = self.INITIAL_MISSILES
        self.missiles_destroyed = 0
        
        for _ in range(self.missiles_to_spawn):
            self._spawn_missile()
            
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = -0.01 # Small penalty for time passing

        self._handle_input(action)
        self._update_game_state()
        
        interaction_reward = self._handle_interactions()
        reward += interaction_reward

        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        self.game_over = terminated

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, _ = action
        
        # Update emitter frequency
        if movement == 1: # Up
            self.emitter_freq = min(1.0, self.emitter_freq + 0.02)
        elif movement == 2: # Down
            self.emitter_freq = max(0.0, self.emitter_freq - 0.02)
            
        # Update emitter amplitude
        if movement == 4: # Right
            self.emitter_amp = min(1.0, self.emitter_amp + 0.02)
        elif movement == 3: # Left
            self.emitter_amp = max(0.1, self.emitter_amp - 0.02)

        # Fire wave
        if space_pressed and self.wave_cooldown <= 0:
            # sfx: Player shoots wave
            self.waves.append(SonicWave(self.emitter_pos, self.emitter_freq, self.emitter_amp))
            self.wave_cooldown = 15 # Cooldown of 15 frames

    def _update_game_state(self):
        if self.wave_cooldown > 0:
            self.wave_cooldown -= 1

        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.missile_speed_mod += 0.1
            self.missile_freq_range_mod = min(0.5, self.missile_freq_range_mod + 0.05)

        for m in self.missiles: m.update()
        for w in self.waves: w.update()
        for p in self.particles: p.update()
        
        # Cleanup
        self.waves = [w for w in self.waves if w.pos[1] > -w.radius]
        self.particles = [p for p in self.particles if p.lifetime > 0]

    def _handle_interactions(self):
        step_reward = 0
        
        destroyed_missiles = []
        for missile in self.missiles:
            missile_hitbox = pygame.Rect(missile.pos[0] - missile.width/2, missile.pos[1] - missile.height/2, missile.width, missile.height)
            
            # Continuous reward for "good aiming"
            for wave in self.waves:
                freq_diff = abs(wave.frequency - missile.frequency)
                freq_tolerance = wave.amplitude * 0.25 # Wider amplitude = more tolerant
                
                # Check if wave is horizontally aligned and frequency is correct
                if wave.get_hitbox().x < missile.pos[0] < wave.get_hitbox().right and freq_diff < freq_tolerance:
                    step_reward += 0.1
            
            # Check for actual collisions
            for wave in self.waves:
                if missile in destroyed_missiles: continue
                
                freq_diff = abs(wave.frequency - missile.frequency)
                freq_tolerance = wave.amplitude * 0.25
                
                if wave.get_hitbox().colliderect(missile_hitbox) and freq_diff < freq_tolerance:
                    # sfx: Missile explosion
                    self._create_explosion(missile.pos, missile.color)
                    destroyed_missiles.append(missile)
                    step_reward += 10
                    self.score += 100
                    self.missiles_destroyed += 1
                    break
        
        self.missiles = [m for m in self.missiles if m not in destroyed_missiles]
        return step_reward

    def _check_termination(self):
        # Win condition
        if self.missiles_destroyed == self.INITIAL_MISSILES:
            self.win_state = True
            return True, 100 # Win

        # Lose condition
        for missile in self.missiles:
            if missile.pos[1] > self.SCREEN_HEIGHT:
                # sfx: Game over, missile hit base
                self.win_state = False
                return True, -100 # Lose

        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.win_state = False
            return True, 0

        return False, 0
    
    def _spawn_missile(self):
        x = random.uniform(50, self.SCREEN_WIDTH - 50)
        speed = self.missile_speed_mod
        freq = np.clip(random.uniform(0.5 - self.missile_freq_range_mod, 0.5 + self.missile_freq_range_mod), 0, 1)
        self.missiles.append(Missile(x, speed, freq, self.SCREEN_WIDTH))

    def _create_explosion(self, pos, color):
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 6)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            radius = random.uniform(2, 6)
            lifetime = random.randint(20, 40)
            p_color = random.choice([(255, 170, 0), (255, 255, 0), (255, 255, 255)])
            self.particles.append(Particle(pos, vel, radius, p_color, lifetime))

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
            "missiles_left": len(self.missiles),
            "emitter_freq": self.emitter_freq,
            "emitter_amp": self.emitter_amp,
        }

    def _render_game(self):
        # Draw grid background
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        for p in self.particles: p.draw(self.screen)
        for w in self.waves: w.draw(self.screen)
        for m in self.missiles: m.draw(self.screen)
        
        # Draw emitter
        emitter_width = 80
        p1 = (int(self.emitter_pos[0] - emitter_width / 2), int(self.emitter_pos[1] + 10))
        p2 = (int(self.emitter_pos[0] + emitter_width / 2), int(self.emitter_pos[1] + 10))
        p3 = (int(self.emitter_pos[0]), int(self.emitter_pos[1] - 10))
        pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_EMITTER)
        pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_EMITTER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Missiles left
        missiles_text = self.font_ui.render(f"TARGETS: {self.missiles_destroyed}/{self.INITIAL_MISSILES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(missiles_text, (self.SCREEN_WIDTH - missiles_text.get_width() - 10, 10))

        # Emitter stats
        bar_width = 100
        bar_height = 10
        
        # Frequency Bar
        freq_color_start = (68, 170, 255)
        freq_color_end = (68, 255, 170)
        freq_color = tuple(int(s + (e - s) * self.emitter_freq) for s, e in zip(freq_color_start, freq_color_end))
        
        freq_label = self.font_ui.render("FREQ", True, self.COLOR_UI_TEXT)
        self.screen.blit(freq_label, (self.emitter_pos[0] - 100, self.SCREEN_HEIGHT - 35))
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.emitter_pos[0] - 100, self.SCREEN_HEIGHT - 20, bar_width, bar_height))
        pygame.draw.rect(self.screen, freq_color, (self.emitter_pos[0] - 100, self.SCREEN_HEIGHT - 20, bar_width * self.emitter_freq, bar_height))
        
        # Amplitude Bar
        amp_color = (255, 100, 255)
        amp_label = self.font_ui.render("AMP", True, self.COLOR_UI_TEXT)
        self.screen.blit(amp_label, (self.emitter_pos[0] + 10, self.SCREEN_HEIGHT - 35))
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.emitter_pos[0] + 10, self.SCREEN_HEIGHT - 20, bar_width, bar_height))
        pygame.draw.rect(self.screen, amp_color, (self.emitter_pos[0] + 10, self.SCREEN_HEIGHT - 20, bar_width * self.emitter_amp, bar_height))

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "MISSION COMPLETE" if self.win_state else "MISSION FAILED"
            color = (100, 255, 100) if self.win_state else (255, 100, 100)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not be run by the autograder
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Sonic Disruptor")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # None
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    env.close()