import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:21:39.423603
# Source Brief: brief_00417.md
# Brief Index: 417
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a rhythmic particle accelerator game.

    The player controls a proton on a circular track, adjusting its phase to
    collide with target particles and avoid hazards. The goal is to build
    progressively more complex hadrons, culminating in the creation of a
    stable Higgs boson, without letting the proton's energy level overload.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a proton in a rhythmic particle accelerator. Collide with targets to build complex hadrons "
        "while avoiding hazards and managing your energy."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to adjust the proton's phase on the track. "
        "Hold space and shift for fine-tuning."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_TRACK = (30, 50, 80)
    COLOR_PULSE = (50, 80, 120)
    COLOR_PROTON = (0, 150, 255)
    COLOR_PROTON_GLOW = (0, 100, 200)
    COLOR_TARGET = (0, 255, 150)
    COLOR_HAZARD = (255, 50, 50)
    COLOR_TEXT = (220, 220, 255)
    COLOR_ENERGY_BAR_BG = (40, 40, 60)
    COLOR_ENERGY_HIGH = (255, 200, 0)
    COLOR_ENERGY_LOW = (0, 200, 100)
    COLOR_GOLD = (255, 215, 0)

    # Physics & Gameplay
    ACCELERATOR_RADIUS = 120
    PROTON_RADIUS = 10
    PARTICLE_RADIUS = 7
    BEAT_DURATION = 45  # frames for one rhythm pulse
    ENERGY_THRESHOLD = 70.0
    MAX_ENERGY = 100.0

    # Action mapping
    PHASE_ADJUST_MAJOR = 0.15
    PHASE_ADJUST_MINOR = 0.05
    PHASE_ADJUST_FINE = 0.025
    
    # Hadron progression
    HADRON_SEQUENCE = [
        "Proton", "Deuteron", "Triton", "Alpha Particle",
        "Carbon-12", "Gold Nucleus", "HIGGS BOSON"
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_hadron = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_win = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy = 0.0
        self.proton_phase = 0.0
        self.particles = []
        self.effects = []
        self.current_hadron_index = 0
        self.particle_speed_modifier = 1.0
        self.last_action_feedback = [0.0, 0.0] # [movement, fine-tune] for visual feedback

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy = 10.0
        self.proton_phase = self.np_random.uniform(0, 2 * math.pi)
        self.particles = []
        self.effects = []
        self.current_hadron_index = 0
        self.particle_speed_modifier = 1.0
        self.last_action_feedback = [0.0, 0.0]

        # Spawn initial particles
        for _ in range(3):
            self._spawn_particle('target')
        for _ in range(2):
            self._spawn_particle('hazard')

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Calculate pre-action reward ---
        reward = 0
        if self.energy < self.ENERGY_THRESHOLD:
            reward += 0.01  # Small survival reward
        else:
            reward -= 0.1 * ((self.energy - self.ENERGY_THRESHOLD) / (self.MAX_ENERGY - self.ENERGY_THRESHOLD))

        # --- 2. Update game state based on action ---
        self._update_proton(action)
        self._update_particles()
        collision_reward = self._handle_collisions()
        reward += collision_reward
        self._update_effects()
        self._update_game_state()

        # --- 3. Check for termination ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Max steps reached
             self.game_over = True # Ensure game over state is set
        
        if self.game_over:
            if self.energy >= self.MAX_ENERGY:
                reward = -100.0 # Terminal penalty for losing
            elif self.current_hadron_index == len(self.HADRON_SEQUENCE) - 1:
                reward = 100.0 # Terminal reward for winning

        self.score += reward
        
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_proton(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        phase_change = 0
        fine_tune_change = 0

        # Movement actions
        if movement == 1: phase_change += self.PHASE_ADJUST_MAJOR  # Up
        elif movement == 2: phase_change -= self.PHASE_ADJUST_MAJOR  # Down
        elif movement == 3: phase_change -= self.PHASE_ADJUST_MINOR  # Left
        elif movement == 4: phase_change += self.PHASE_ADJUST_MINOR  # Right

        # Fine-tune actions
        if space_held: fine_tune_change += self.PHASE_ADJUST_FINE
        if shift_held: fine_tune_change -= self.PHASE_ADJUST_FINE
        
        self.proton_phase += phase_change + fine_tune_change
        self.proton_phase %= (2 * math.pi)

        # Store for visual feedback
        self.last_action_feedback = [phase_change, fine_tune_change]

    def _update_particles(self):
        for p in self.particles:
            p['phase'] += p['speed'] * self.particle_speed_modifier
            p['phase'] %= (2 * math.pi)

    def _update_effects(self):
        self.effects = [e for e in self.effects if e['life'] > 0]
        for e in self.effects:
            e['life'] -= 1
            e['pos'] = (e['pos'][0] + e['vel'][0], e['pos'][1] + e['vel'][1])

    def _update_game_state(self):
        self.steps += 1
        self.energy = max(0, self.energy - 0.05) # Natural energy decay

        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.particle_speed_modifier += 0.05
        
        # Spawn new particles periodically to keep the game active
        if self.steps > 0 and self.steps % 75 == 0:
            if self.np_random.random() < 0.6:
                self._spawn_particle('target')
            else:
                self._spawn_particle('hazard')
        
        # Introduction of new particle types (conceptual, here just more hazards)
        if self.steps > 0 and self.steps % 500 == 0:
             self._spawn_particle('hazard')

        if self.energy >= self.MAX_ENERGY:
            self.game_over = True
            # sfx: game_over_sound
    
    def _spawn_particle(self, p_type):
        particle = {
            'type': p_type,
            'phase': self.np_random.uniform(0, 2 * math.pi),
            'radius': self.ACCELERATOR_RADIUS + self.np_random.uniform(-15, 15),
            'speed': self.np_random.uniform(0.01, 0.03) * (1 if p_type == 'target' else 1.2),
        }
        self.particles.append(particle)

    def _handle_collisions(self):
        reward = 0
        proton_pos = self._phase_to_cartesian(self.proton_phase, self.ACCELERATOR_RADIUS)
        
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p_pos = self._phase_to_cartesian(p['phase'], p['radius'])
            dist = math.hypot(proton_pos[0] - p_pos[0], proton_pos[1] - p_pos[1])
            
            if dist < self.PROTON_RADIUS + self.PARTICLE_RADIUS:
                particles_to_remove.append(i)
                self._create_particle_effect(p_pos, p['type'])
                
                if p['type'] == 'target':
                    # sfx: target_hit_sound
                    reward += 1.0
                    self.energy += 5.0
                    if self.current_hadron_index < len(self.HADRON_SEQUENCE) - 1:
                        self.current_hadron_index += 1
                        reward += 50.0
                        # sfx: hadron_upgrade_sound
                        if self.current_hadron_index == len(self.HADRON_SEQUENCE) - 1:
                            self.game_over = True # VICTORY
                            # sfx: win_sound
                elif p['type'] == 'hazard':
                    # sfx: hazard_hit_sound
                    reward -= 5.0
                    self.energy += 25.0
        
        # Remove collided particles (iterate backwards to avoid index errors)
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]
            
        return reward
    
    def _create_particle_effect(self, pos, p_type):
        num_sparks = 20 if p_type == 'target' else 15
        color = self.COLOR_TARGET if p_type == 'target' else self.COLOR_HAZARD
        for _ in range(num_sparks):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.effects.append({
                'pos': list(pos),
                'vel': (math.cos(angle) * speed, math.sin(angle) * speed),
                'life': self.np_random.integers(10, 20),
                'color': color,
            })

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Render all elements
        self._render_background()
        self._render_effects()
        self._render_particles()
        self._render_proton()
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "current_hadron": self.HADRON_SEQUENCE[self.current_hadron_index]
        }
        
    def _phase_to_cartesian(self, phase, radius):
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2
        x = cx + radius * math.cos(phase)
        y = cy + radius * math.sin(phase)
        return (x, y)

    def _render_antialiased_circle(self, surface, color, center, radius):
        pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), int(radius), color)
        pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius), color)

    # --- Rendering Methods ---

    def _render_background(self):
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2
        # Accelerator track
        self._render_antialiased_circle(self.screen, self.COLOR_TRACK, (cx, cy), self.ACCELERATOR_RADIUS + 25)
        self._render_antialiased_circle(self.screen, self.COLOR_BG, (cx, cy), self.ACCELERATOR_RADIUS - 25)
        
        # Rhythm pulse
        beat_progress = (self.steps % self.BEAT_DURATION) / self.BEAT_DURATION
        pulse_radius = int(beat_progress * (self.ACCELERATOR_RADIUS + 25))
        pulse_alpha = int(255 * (1 - beat_progress))
        if pulse_alpha > 0:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(s, cx, cy, pulse_radius, self.COLOR_PULSE + (pulse_alpha,))
            self.screen.blit(s, (0, 0))

    def _render_effects(self):
        for e in self.effects:
            alpha = max(0, 255 * (e['life'] / 20.0))
            color = e['color'] + (alpha,)
            size = max(1, int(2 * (e['life'] / 20.0)))
            pygame.draw.circle(self.screen, color, (int(e['pos'][0]), int(e['pos'][1])), size)

    def _render_particles(self):
        for p in self.particles:
            pos = self._phase_to_cartesian(p['phase'], p['radius'])
            color = self.COLOR_TARGET if p['type'] == 'target' else self.COLOR_HAZARD
            self._render_antialiased_circle(self.screen, color, pos, self.PARTICLE_RADIUS)

    def _render_proton(self):
        pos = self._phase_to_cartesian(self.proton_phase, self.ACCELERATOR_RADIUS)
        
        # Glow effect
        glow_radius = self.PROTON_RADIUS + 15
        for i in range(15):
            alpha = 100 * (1 - i / 15.0)**2
            radius = self.PROTON_RADIUS + i
            color = self.COLOR_PROTON_GLOW + (alpha,)
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (radius, radius), radius)
            self.screen.blit(s, (int(pos[0] - radius), int(pos[1] - radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Core proton
        self._render_antialiased_circle(self.screen, self.COLOR_PROTON, pos, self.PROTON_RADIUS)

        # Action feedback indicators
        for i in range(1, 4):
            # Movement feedback
            if abs(self.last_action_feedback[0]) > 0.01:
                angle = self.proton_phase + (math.pi/2 if self.last_action_feedback[0] > 0 else -math.pi/2)
                p1_offset = self.PROTON_RADIUS + i*2
                p2_offset = self.PROTON_RADIUS + i*2 + 5
                p1 = self._phase_to_cartesian(angle, p1_offset)
                p2 = self._phase_to_cartesian(angle, p2_offset)
                pygame.draw.line(self.screen, self.COLOR_PROTON, p1, p2, 1)

            # Fine-tune feedback
            if abs(self.last_action_feedback[1]) > 0.01:
                angle = self.proton_phase + (0 if self.last_action_feedback[1] > 0 else math.pi)
                p_offset = self.PROTON_RADIUS + 8
                p = self._phase_to_cartesian(angle, p_offset)
                pygame.draw.circle(self.screen, self.COLOR_PROTON, (int(p[0]), int(p[1])), 2)


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        # Energy Bar
        bar_x, bar_y, bar_w, bar_h = 10, self.HEIGHT - 30, self.WIDTH - 20, 20
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        energy_ratio = min(1.0, self.energy / self.MAX_ENERGY)
        energy_w = int(bar_w * energy_ratio)
        energy_color = self.COLOR_ENERGY_LOW if energy_ratio < 0.7 else self.COLOR_ENERGY_HIGH
        pygame.draw.rect(self.screen, energy_color, (bar_x, bar_y, energy_w, bar_h), border_radius=5)
        energy_text = self.font_ui.render(f"ENERGY", True, self.COLOR_TEXT)
        self.screen.blit(energy_text, (bar_x + 5, bar_y))
        
        # Current Hadron
        hadron_name = self.HADRON_SEQUENCE[self.current_hadron_index]
        hadron_color = self.COLOR_GOLD if hadron_name == "HIGGS BOSON" else self.COLOR_TEXT
        hadron_text = self.font_hadron.render(f"PARTICLE: {hadron_name}", True, hadron_color)
        self.screen.blit(hadron_text, (self.WIDTH // 2 - hadron_text.get_width() // 2, 10))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.current_hadron_index == len(self.HADRON_SEQUENCE) - 1:
                win_text = self.font_win.render("HIGGS BOSON CREATED!", True, self.COLOR_GOLD)
                self.screen.blit(win_text, (self.WIDTH // 2 - win_text.get_width() // 2, self.HEIGHT // 2 - 50))
            else:
                lose_text = self.font_win.render("ENERGY OVERLOAD", True, self.COLOR_HAZARD)
                self.screen.blit(lose_text, (self.WIDTH // 2 - lose_text.get_width() // 2, self.HEIGHT // 2 - 50))
            
            final_score_text = self.font_hadron.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
            self.screen.blit(final_score_text, (self.WIDTH // 2 - final_score_text.get_width() // 2, self.HEIGHT // 2 + 10))

    def close(self):
        pygame.quit()
        
# Example usage for interactive play or testing
if __name__ == "__main__":
    # The main loop is for interactive testing and visualization.
    # It will create a window, which is fine for this purpose,
    # even though the environment itself is headless-compatible.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Interactive Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    pygame.display.set_caption("Particle Accelerator")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Keyboard to MultiDiscrete Action Mapping ---
        keys = pygame.key.get_pressed()
        
        # Movement
        mov_action = 0 # no-op
        if keys[pygame.K_UP]: mov_action = 1
        elif keys[pygame.K_DOWN]: mov_action = 2
        elif keys[pygame.K_LEFT]: mov_action = 3
        elif keys[pygame.K_RIGHT]: mov_action = 4
        
        # Space and Shift
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()