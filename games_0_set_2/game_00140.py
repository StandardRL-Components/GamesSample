
# Generated: 2025-08-27T12:43:17.614602
# Source Brief: brief_00140.md
# Brief Index: 140

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the crosshair. Press Space to fire. "
        "Press Shift to switch between Standard (green) and Risky (red) ammo."
    )

    game_description = (
        "A top-down target practice game. Hit all 25 targets before you run out of "
        "ammo. Standard shots always hit for 1 point. Risky shots have a 50% "
        "chance to hit but award 2 points."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.TARGET_COUNT = 25
        self.TARGET_RADIUS = 12
        self.INITIAL_AMMO = 40
        self.CROSSHAIR_SPEED = 15
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_TARGET = (220, 220, 220)
        self.COLOR_TARGET_OUTLINE = (100, 100, 110)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_STANDARD = (0, 255, 128)
        self.COLOR_RISKY = (255, 80, 80)
        self.COLOR_PARTICLE_HIT = (255, 200, 80)
        self.COLOR_PARTICLE_RISKY_HIT = (255, 100, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_feedback = pygame.font.SysFont("Consolas", 28, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State Variables ---
        self.crosshair_pos = None
        self.targets = None
        self.ammo = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.is_risky_mode = None
        self.particles = None
        self.shot_tracer = None
        self.feedback_message = None
        self.feedback_timer = None
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.crosshair_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self._generate_targets()
        
        self.ammo = self.INITIAL_AMMO
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.is_risky_mode = False
        
        self.particles = []
        self.shot_tracer = None # Stores {'end_pos', 'color', 'timer'}
        self.feedback_message = ""
        self.feedback_timer = 0

        return self._get_observation(), self._get_info()

    def _generate_targets(self):
        self.targets = []
        margin = self.TARGET_RADIUS + 10
        for _ in range(self.TARGET_COUNT):
            placed = False
            while not placed:
                pos = np.array([
                    self.np_random.uniform(margin, self.SCREEN_WIDTH - margin),
                    self.np_random.uniform(margin, self.SCREEN_HEIGHT - margin - 50) # Keep top clear for UI
                ])
                
                if not any(np.linalg.norm(pos - t['pos']) < self.TARGET_RADIUS * 2.5 for t in self.targets):
                    self.targets.append({'pos': pos, 'active': True, 'hit_timer': 0})
                    placed = True
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.shot_tracer = None

        movement, space_action, shift_action = action
        fire_pressed = space_action == 1
        shift_pressed = shift_action == 1

        # 1. Handle Input
        if shift_pressed:
            self.is_risky_mode = not self.is_risky_mode
            # // Play switch sound

        self._move_crosshair(movement)

        # 2. Process Firing Action
        if fire_pressed and self.ammo > 0:
            self.ammo -= 1
            # // Play fire sound
            
            hit_target_idx = self._check_for_hit()
            
            if hit_target_idx is not None:
                is_hit_successful = True
                if self.is_risky_mode:
                    is_hit_successful = self.np_random.random() < 0.5

                if is_hit_successful:
                    target = self.targets[hit_target_idx]
                    target['active'] = False
                    target['hit_timer'] = 10 # For shrink animation
                    
                    if self.is_risky_mode:
                        self.score += 2
                        reward += 2
                        self._create_particles(target['pos'], self.COLOR_PARTICLE_RISKY_HIT, 30)
                        self._set_feedback("RISKY HIT! [+2]", self.COLOR_RISKY)
                        # // Play risky hit sound
                    else:
                        self.score += 1
                        reward += 1
                        self._create_particles(target['pos'], self.COLOR_PARTICLE_HIT, 20)
                        self._set_feedback("HIT! [+1]", self.COLOR_STANDARD)
                        # // Play standard hit sound
                else: # Risky shot miss
                    reward -= 0.1
                    self._set_feedback("RISKY MISS!", self.COLOR_RISKY)
                    # // Play miss sound
            else: # Missed, no target
                reward -= 0.1
                self._set_feedback("MISS", self.COLOR_TEXT)
                # // Play miss sound

            tracer_color = self.COLOR_RISKY if self.is_risky_mode else self.COLOR_STANDARD
            self.shot_tracer = {'end_pos': self.crosshair_pos.copy(), 'color': tracer_color, 'timer': 5}

        # 3. Update Game State
        self._update_particles()
        for t in self.targets:
            if t['hit_timer'] > 0:
                t['hit_timer'] -= 1

        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        
        # 4. Check for Termination
        terminated = False
        targets_remaining = sum(1 for t in self.targets if t['active'])

        if targets_remaining == 0:
            terminated = True
            self.game_over = True
            reward += 50
            self._set_feedback("ALL TARGETS CLEARED!", self.COLOR_STANDARD, 999)
        elif self.ammo == 0 and targets_remaining > 0:
            terminated = True
            self.game_over = True
            reward -= 50
            self._set_feedback("OUT OF AMMO", self.COLOR_RISKY, 999)
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            reward -= 50 # Penalize running out of time
            self._set_feedback("TIME LIMIT REACHED", self.COLOR_RISKY, 999)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _move_crosshair(self, movement):
        if movement == 1: # Up
            self.crosshair_pos[1] -= self.CROSSHAIR_SPEED
        elif movement == 2: # Down
            self.crosshair_pos[1] += self.CROSSHAIR_SPEED
        elif movement == 3: # Left
            self.crosshair_pos[0] -= self.CROSSHAIR_SPEED
        elif movement == 4: # Right
            self.crosshair_pos[0] += self.CROSSHAIR_SPEED
        
        self.crosshair_pos[0] = np.clip(self.crosshair_pos[0], 0, self.SCREEN_WIDTH)
        self.crosshair_pos[1] = np.clip(self.crosshair_pos[1], 0, self.SCREEN_HEIGHT)

    def _check_for_hit(self):
        for i, target in enumerate(self.targets):
            if target['active']:
                dist = np.linalg.norm(self.crosshair_pos - target['pos'])
                if dist < self.TARGET_RADIUS:
                    return i
        return None

    def _set_feedback(self, message, color, duration=30):
        self.feedback_message = (message, color)
        self.feedback_timer = duration

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifetime': lifetime, 'color': color})
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifetime'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw targets
        for t in self.targets:
            pos_int = (int(t['pos'][0]), int(t['pos'][1]))
            if t['active']:
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.TARGET_RADIUS, self.COLOR_TARGET_OUTLINE)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.TARGET_RADIUS, self.COLOR_TARGET)
            elif t['hit_timer'] > 0:
                # Shrink animation on hit
                radius = int(self.TARGET_RADIUS * (t['hit_timer'] / 10.0))
                alpha = int(255 * (t['hit_timer'] / 10.0))
                color = self.COLOR_PARTICLE_HIT + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)

        # Draw shot tracer
        if self.shot_tracer and self.shot_tracer['timer'] > 0:
            alpha = int(255 * (self.shot_tracer['timer'] / 5.0))
            color = self.shot_tracer['color'] + (alpha,)
            start_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT)
            end_pos = (int(self.shot_tracer['end_pos'][0]), int(self.shot_tracer['end_pos'][1]))
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
            self.shot_tracer['timer'] -= 1

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 30.0))
            color = p['color'] + (alpha,)
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['lifetime'] / 6)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)

        # Draw crosshair
        crosshair_color = self.COLOR_RISKY if self.is_risky_mode else self.COLOR_STANDARD
        x, y = int(self.crosshair_pos[0]), int(self.crosshair_pos[1])
        size = 12
        pygame.draw.line(self.screen, crosshair_color, (x - size, y), (x + size, y), 2)
        pygame.draw.line(self.screen, crosshair_color, (x, y - size), (x, y + size), 2)
        pygame.gfxdraw.aacircle(self.screen, x, y, size // 2, crosshair_color)

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Ammo
        ammo_color = self.COLOR_RISKY if self.ammo <= 5 else self.COLOR_TEXT
        ammo_surf = self.font_main.render(f"AMMO: {self.ammo}", True, ammo_color)
        self.screen.blit(ammo_surf, (self.SCREEN_WIDTH - ammo_surf.get_width() - 10, 10))

        # Projectile Mode
        mode_text = "RISKY" if self.is_risky_mode else "STANDARD"
        mode_color = self.COLOR_RISKY if self.is_risky_mode else self.COLOR_STANDARD
        mode_surf = self.font_main.render(f"MODE: {mode_text}", True, mode_color)
        self.screen.blit(mode_surf, (self.SCREEN_WIDTH // 2 - mode_surf.get_width() // 2, 10))
        
        # Feedback Text
        if self.feedback_timer > 0:
            msg, color = self.feedback_message
            alpha = 255
            if self.feedback_timer < 15:
                alpha = int(255 * (self.feedback_timer / 15.0))
            
            feedback_surf = self.font_feedback.render(msg, True, color)
            feedback_surf.set_alpha(alpha)
            pos = (self.SCREEN_WIDTH // 2 - feedback_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - 50)
            self.screen.blit(feedback_surf, pos)

        # Game Over Text
        if self.game_over and self.feedback_timer > 30:
            msg, color = self.feedback_message
            game_over_surf = self.font_game_over.render(msg, True, color)
            pos = (self.SCREEN_WIDTH // 2 - game_over_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - game_over_surf.get_height() // 2)
            self.screen.blit(game_over_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.ammo,
            "targets_remaining": sum(1 for t in self.targets if t['active']),
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        # Test game logic assertions
        self.reset()
        assert self.ammo > 0
        self.step([0, 1, 0]) # Fire a shot
        assert self.ammo == self.INITIAL_AMMO - 1
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Target Practice")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_action, shift_action]
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'R' key
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for human play
        
    env.close()