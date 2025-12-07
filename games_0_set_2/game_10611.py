import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:46:37.172977
# Source Brief: brief_00611.md
# Brief Index: 611
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a shape-shifting cube
    to deflect incoming energy pulses. The goal is to achieve a 5x combo
    multiplier before the 60-second timer runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a shape-shifting cube to deflect incoming energy pulses. "
        "Achieve a 5x combo multiplier before the timer runs out to win."
    )
    user_guide = (
        "Use arrow keys to move. Hold space to Absorb or shift to Reflect."
    )
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_PULSE = (0, 255, 255) # Cyan
    COLOR_PLAYER_PIERCE = (50, 150, 255) # Blue
    COLOR_PLAYER_ABSORB = (50, 255, 150) # Green
    COLOR_PLAYER_REFLECT = (255, 100, 100) # Red
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_COMBO_BASE = (255, 200, 0) # Yellow

    # Player
    PLAYER_SIZE = 25
    PLAYER_SPEED = 5

    # Pulse
    PULSE_BASE_SPEED = 1.5
    PULSE_SPAWN_RATE_SECONDS = 0.5
    PULSE_WIDTH = 4
    PULSE_TRAIL_LENGTH = 15

    # Forms
    FORM_PIERCE = 0
    FORM_ABSORB = 1
    FORM_REFLECT = 2
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_combo = pygame.font.Font(None, 64)

        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_form = None
        self.pulses = None
        self.particles = None
        self.pulse_speed = None
        self.pulse_spawn_timer = None
        self.combo_multiplier = None
        self.combo_counter = None
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_form = self.FORM_PIERCE
        
        self.pulses = []
        self.particles = []
        
        self.pulse_speed = self.PULSE_BASE_SPEED
        self.pulse_spawn_timer = 0
        
        self.combo_multiplier = 1.0
        self.combo_counter = 0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        self._handle_input(action)
        reward += self._update_pulses()
        self._update_particles()
        reward += self._handle_collisions()
        
        self.steps += 1
        self._update_difficulty()
        self.pulse_spawn_timer -= 1
        if self.pulse_spawn_timer <= 0:
            self._spawn_pulse()
            self.pulse_spawn_timer = self.PULSE_SPAWN_RATE_SECONDS * self.FPS
        
        terminated = False
        if self.combo_multiplier >= 5.0:
            reward += 100 # Victory reward
            terminated = True
            # Sound: Victory fanfare
        
        time_left = self.GAME_DURATION_SECONDS - (self.steps / self.FPS)
        if time_left <= 0:
            reward -= 10 # Time out penalty
            terminated = True
            # Sound: Game over buzzer
        
        self.game_over = terminated
        self.score += reward

        # The 'truncated' flag is not used in this game logic, so it's always False.
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3: self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos.x += self.PLAYER_SPEED

        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

        if space_held: self.player_form = self.FORM_ABSORB
        elif shift_held: self.player_form = self.FORM_REFLECT
        else: self.player_form = self.FORM_PIERCE

    def _update_pulses(self):
        center = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        reward_penalty = 0
        pulses_to_remove = []
        for i, pulse in enumerate(self.pulses):
            pulse['pos'] += pulse['vel'] * self.pulse_speed
            pulse['trail'].append(pulse['pos'].copy())
            if len(pulse['trail']) > self.PULSE_TRAIL_LENGTH:
                pulse['trail'].pop(0)

            if not self.screen.get_rect().collidepoint(pulse['pos']):
                pulses_to_remove.append(i)
                continue

            if pulse['pos'].distance_to(center) < self.PULSE_WIDTH * 2:
                pulses_to_remove.append(i)
                if self.player_form != self.FORM_PIERCE:
                    reward_penalty -= 0.1
                    self.combo_counter = 0
                    self.combo_multiplier = 1.0
                    # Sound: Failure sizzle
        
        for i in sorted(pulses_to_remove, reverse=True): del self.pulses[i]
        return reward_penalty

    def _handle_collisions(self):
        reward = 0
        pulses_to_remove = []
        player_rect = pygame.Rect(
            self.player_pos.x - self.PLAYER_SIZE, self.player_pos.y - self.PLAYER_SIZE,
            self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2
        )

        if self.player_form == self.FORM_PIERCE: return 0

        for i, pulse in enumerate(self.pulses):
            if player_rect.collidepoint(pulse['pos']):
                if self.player_form == self.FORM_ABSORB:
                    pulses_to_remove.append(i)
                    self._create_particles(pulse['pos'], self.COLOR_PLAYER_ABSORB)
                    reward += 0.1
                    self.combo_counter += 1
                    # Sound: Absorb pop
                
                elif self.player_form == self.FORM_REFLECT:
                    pulse['vel'] *= -1
                    pulse['pos'] += pulse['vel'] * self.PLAYER_SIZE * 0.5 
                    self._create_particles(pulse['pos'], self.COLOR_PLAYER_REFLECT)
                    reward += 0.1
                    self.combo_counter += 1
                    # Sound: Reflect ping
        
        for i in sorted(pulses_to_remove, reverse=True): del self.pulses[i]

        new_combo = min(5.0, 1.0 + self.combo_counter * 0.2)
        if new_combo > self.combo_multiplier:
            reward += 1.0
            # Sound: Combo increase chime
        self.combo_multiplier = new_combo
        
        return reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.pulse_speed += 0.5

    def _spawn_pulse(self):
        edge = self.np_random.integers(0, 4)
        if edge == 0: pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), 0)
        elif edge == 1: pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT)
        elif edge == 2: pos = pygame.math.Vector2(0, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        else: pos = pygame.math.Vector2(self.SCREEN_WIDTH, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        
        center_target = pygame.math.Vector2(
            self.SCREEN_WIDTH / 2 + self.np_random.uniform(-50, 50),
            self.SCREEN_HEIGHT / 2 + self.np_random.uniform(-50, 50)
        )
        vel = (center_target - pos).normalize()
        self.pulses.append({'pos': pos, 'vel': vel, 'trail': []})

    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'lifetime': self.np_random.integers(20, 40), 'color': color
            })

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['lifetime'] -= 1
            if p['lifetime'] <= 0: particles_to_remove.append(i)
        for i in sorted(particles_to_remove, reverse=True): del self.particles[i]
            
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
            "combo": self.combo_multiplier,
            "time_left": self.GAME_DURATION_SECONDS - (self.steps / self.FPS),
        }

    def _render_game(self):
        self._render_particles()
        self._render_pulses()
        self._render_player()

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        size = self.PLAYER_SIZE
        
        if self.player_form == self.FORM_PIERCE:
            color = self.COLOR_PLAYER_PIERCE
            for i in range(5, 0, -1):
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size + i * 2, (*color, 50 - i * 8))
            rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
            pygame.draw.rect(self.screen, color, rect, 3)

        elif self.player_form == self.FORM_ABSORB:
            color = self.COLOR_PLAYER_ABSORB
            for i in range(8, 0, -1):
                glow_color = (*color, 80 - i * 8)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + i, glow_color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size + i, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)

        elif self.player_form == self.FORM_REFLECT:
            color = self.COLOR_PLAYER_REFLECT
            rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
            for i in range(5, 0, -1):
                glow_rect = rect.inflate(i*4, i*4)
                pygame.draw.rect(self.screen, (*color, 50 - i * 8), glow_rect, 1)
            pygame.draw.rect(self.screen, color, rect)

    def _render_pulses(self):
        for pulse in self.pulses:
            for i, p in enumerate(pulse['trail']):
                alpha = int(255 * (i / self.PULSE_TRAIL_LENGTH))
                if i > 0:
                    pygame.draw.line(self.screen, (*self.COLOR_PULSE, alpha), p, pulse['trail'][i-1], self.PULSE_WIDTH)
            end_pos = pulse['pos']
            start_pos = pulse['pos'] - pulse['vel'] * self.pulse_speed * 2
            pygame.draw.line(self.screen, self.COLOR_PULSE, start_pos, end_pos, self.PULSE_WIDTH)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['lifetime'] / 40.0))
            pos = (int(p['pos'].x), int(p['pos'].y))
            size = max(1, int(p['lifetime'] / 8))
            pygame.draw.circle(self.screen, (*p['color'], alpha), pos, size)

    def _render_ui(self):
        time_left = max(0, self.GAME_DURATION_SECONDS - (self.steps / self.FPS))
        timer_text = self.font_ui.render(f"Time: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        if self.combo_multiplier > 1.0:
            t = (self.combo_multiplier - 1.0) / 4.0
            combo_color = tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(self.COLOR_COMBO_BASE, (255, 255, 100)))
            combo_str = f"{self.combo_multiplier:.1f}x"
            combo_surf = self.font_combo.render(combo_str, True, combo_color)
            combo_pos = (self.SCREEN_WIDTH/2 - combo_surf.get_width()/2, self.SCREEN_HEIGHT - combo_surf.get_height() - 20)
            self.screen.blit(combo_surf, combo_pos)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not used by the evaluation system.
    # It is safe to remove or modify.
    
    # Un-dummy the video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Shape Shifter")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    running = True
    
    while running:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.0f}, Combo: {info['combo']:.1f}x")
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause for 2 seconds before restarting
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)
        
    env.close()