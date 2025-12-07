import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:17:31.199011
# Source Brief: brief_00907.md
# Brief Index: 907
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player guides a boson through a maze of
    accelerating protons. The goal is to reach the exit by using trajectory
    shifts to trigger chain reactions among the protons.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a boson through a maze of protons. Use trajectory shifts to trigger "
        "chain reactions and reach the exit before time runs out."
    )
    user_guide = (
        "Controls: Press Shift to change your launch angle and Space to launch the boson. "
        "Avoid colliding with protons."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    TIME_LIMIT_SECONDS = 30

    # Colors
    COLOR_BG = (10, 10, 30) # Dark space blue
    COLOR_WALL = (150, 150, 170)
    COLOR_BOSON = (0, 255, 255) # Bright cyan
    COLOR_PROTON = (255, 60, 60) # Bright red
    COLOR_EXIT = (0, 255, 128) # Glowing green
    COLOR_PULSE = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)

    # Entity Properties
    BOSON_RADIUS = 8
    BOSON_SPEED = 4.0
    PROTON_RADIUS = 10
    PROTON_BASE_SPEED = 1.5
    PULSE_MAX_RADIUS = 100
    PULSE_DURATION = 15 # frames
    PROTON_TRAIL_LENGTH = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.win_message = ""
        self.boson = None
        self.protons = []
        self.walls = []
        self.exit_rect = None
        self.energy_pulses = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.proton_speed_multiplier = 1.0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS
        self.game_over = False
        self.win_message = ""
        self.proton_speed_multiplier = 1.0

        self.prev_space_held = False
        self.prev_shift_held = False

        self._create_maze()
        self._spawn_boson()
        self._spawn_protons(num_protons=8)
        
        self.energy_pulses = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            is_truncated = self.win_message == "TIME OUT"
            is_terminated = self.game_over and not is_truncated
            return self._get_observation(), 0, is_terminated, is_truncated, self._get_info()
            
        reward = 0
        terminated = False
        truncated = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        self.steps += 1
        self.time_remaining -= 1

        # Handle player actions
        if shift_press and not self.boson['launched']:
            self.boson['angle_deg'] = (self.boson['angle_deg'] + 45) % 360
            # SFX: Angle change beep

        if space_press and not self.boson['launched']:
            self.boson['launched'] = True
            angle_rad = math.radians(self.boson['angle_deg'])
            self.boson['vel'] = pygame.math.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * self.BOSON_SPEED
            # SFX: Boson launch swoosh

        # Update difficulty
        if self.steps > 0 and self.steps % 1000 == 0:
            self.proton_speed_multiplier += 0.05

        self._update_boson()
        self._update_protons()
        self._update_pulses()

        collision_reward, collision_termination = self._handle_collisions()
        reward += collision_reward
        if collision_termination:
            terminated = True
        
        if self.boson['launched'] and not terminated:
             reward += 0.1

        if self.time_remaining <= 0 and not terminated:
            reward -= 50
            truncated = True
            self.game_over = True
            self.win_message = "TIME OUT"
            # SFX: Timeout buzzer
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_boson(self):
        if self.boson['launched']:
            self.boson['pos'] += self.boson['vel']
            # Boundary bouncing
            if self.boson['pos'].x - self.BOSON_RADIUS < 0 or self.boson['pos'].x + self.BOSON_RADIUS > self.SCREEN_WIDTH:
                self.boson['vel'].x *= -1
                self.boson['pos'].x = max(self.BOSON_RADIUS, min(self.boson['pos'].x, self.SCREEN_WIDTH - self.BOSON_RADIUS))
                # SFX: Boson bounce
            if self.boson['pos'].y - self.BOSON_RADIUS < 0 or self.boson['pos'].y + self.BOSON_RADIUS > self.SCREEN_HEIGHT:
                self.boson['vel'].y *= -1
                self.boson['pos'].y = max(self.BOSON_RADIUS, min(self.boson['pos'].y, self.SCREEN_HEIGHT - self.BOSON_RADIUS))
                # SFX: Boson bounce

    def _update_protons(self):
        for p in self.protons:
            p['trail'].append(p['pos'].copy())
            p['pos'] += p['vel'] * self.proton_speed_multiplier

            # Boundary collision
            if p['pos'].x - self.PROTON_RADIUS < 0 or p['pos'].x + self.PROTON_RADIUS > self.SCREEN_WIDTH:
                p['vel'].x *= -1
                p['pos'].x = max(self.PROTON_RADIUS, min(p['pos'].x, self.SCREEN_WIDTH - self.PROTON_RADIUS))
            if p['pos'].y - self.PROTON_RADIUS < 0 or p['pos'].y + self.PROTON_RADIUS > self.SCREEN_HEIGHT:
                p['vel'].y *= -1
                p['pos'].y = max(self.PROTON_RADIUS, min(p['pos'].y, self.SCREEN_HEIGHT - self.PROTON_RADIUS))
            
            # Maze wall collision
            proton_rect = pygame.Rect(p['pos'].x - self.PROTON_RADIUS, p['pos'].y - self.PROTON_RADIUS, self.PROTON_RADIUS*2, self.PROTON_RADIUS*2)
            for wall in self.walls:
                if proton_rect.colliderect(wall):
                    overlap_x = min(proton_rect.right, wall.right) - max(proton_rect.left, wall.left)
                    overlap_y = min(proton_rect.bottom, wall.bottom) - max(proton_rect.top, wall.top)
                    if overlap_x < overlap_y:
                        p['vel'].x *= -1
                    else:
                        p['vel'].y *= -1
                    p['pos'] += p['vel'].normalize() * 1.5

    def _update_pulses(self):
        for pulse in self.energy_pulses:
            pulse['age'] += 1
            pulse['radius'] = (pulse['age'] / self.PULSE_DURATION) * self.PULSE_MAX_RADIUS
        self.energy_pulses = [p for p in self.energy_pulses if p['age'] <= self.PULSE_DURATION]

    def _handle_collisions(self):
        reward = 0
        terminated = False

        if self.exit_rect.collidepoint(self.boson['pos']):
            reward += 100
            terminated = True
            self.game_over = True
            self.win_message = "VICTORY!"
            # SFX: Victory chime
            return reward, terminated

        if self.boson['launched']:
            for p in self.protons:
                dist = self.boson['pos'].distance_to(p['pos'])
                if dist < self.BOSON_RADIUS + self.PROTON_RADIUS:
                    reward -= 100
                    reward += 1
                    terminated = True
                    self.game_over = True
                    self.win_message = "DESTROYED"
                    self._create_energy_pulse(self.boson['pos'])
                    # SFX: Explosion and pulse activation
                    return reward, terminated
        
        for pulse in self.energy_pulses:
            for p in self.protons:
                if not p.get('hit_by_pulse', False):
                    dist = pulse['pos'].distance_to(p['pos'])
                    if dist < pulse['radius'] + self.PROTON_RADIUS:
                        p['vel'] = p['vel'].rotate(45)
                        p['hit_by_pulse'] = True
                        # SFX: Proton direction change zap

        for p in self.protons:
            if 'hit_by_pulse' in p:
                p['hit_by_pulse'] = False

        return reward, terminated

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
            "time_remaining": self.time_remaining,
            "boson_launched": self.boson['launched']
        }

    def _render_game(self):
        self._render_glow_rect(self.exit_rect, self.COLOR_EXIT, 15)
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        for p in self.protons:
            for i, pos in enumerate(p['trail']):
                alpha = int(255 * (i / self.PROTON_TRAIL_LENGTH))
                color = (*self.COLOR_PROTON, alpha)
                radius = int(self.PROTON_RADIUS * 0.5 * (i / self.PROTON_TRAIL_LENGTH))
                if radius > 0:
                   self._render_transparent_circle(pos, radius, color)

        for p in self.protons:
            self._render_glow_circle(p['pos'], self.PROTON_RADIUS, self.COLOR_PROTON, 10)

        for pulse in self.energy_pulses:
            progress = pulse['age'] / self.PULSE_DURATION
            alpha = int(255 * (1 - progress)**2)
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, int(pulse['pos'].x), int(pulse['pos'].y), int(pulse['radius']), (*self.COLOR_PULSE, alpha))

        self._render_glow_circle(self.boson['pos'], self.BOSON_RADIUS, self.COLOR_BOSON, 20)
        
        if not self.boson['launched']:
            angle_rad = math.radians(self.boson['angle_deg'])
            start_pos = self.boson['pos']
            end_pos = self.boson['pos'] + pygame.math.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * 30
            pygame.draw.aaline(self.screen, self.COLOR_BOSON, start_pos, end_pos)

    def _render_ui(self):
        time_text = f"Time: {max(0, self.time_remaining / self.FPS):.1f}"
        text_surface = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 10, 10))

        score_text = f"Score: {self.score:.2f}"
        score_surface = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surface, (10, 10))
        
        if self.game_over:
            over_surface = self.font_game_over.render(self.win_message, True, self.COLOR_TEXT)
            over_rect = over_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_surface, over_rect)

    def _render_glow_circle(self, pos, radius, color, glow_size):
        for i in range(glow_size, 0, -2):
            alpha = int(100 * (1 - (i / glow_size)))
            s = pygame.Surface((radius * 2 + i, radius * 2 + i), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (s.get_width() // 2, s.get_height() // 2), radius + i // 2)
            self.screen.blit(s, (int(pos.x - s.get_width() // 2), int(pos.y - s.get_height() // 2)))
        
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius, color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, color)

    def _render_glow_rect(self, rect, color, glow_size):
        for i in range(glow_size, 0, -2):
            alpha = int(100 * (1 - (i / glow_size)))
            glow_rect = rect.inflate(i, i)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, alpha), s.get_rect(), border_radius=i)
            self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        
    def _render_transparent_circle(self, pos, radius, color_with_alpha):
        s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, color_with_alpha, (radius, radius), radius)
        self.screen.blit(s, (int(pos.x - radius), int(pos.y - radius)))

    def _create_maze(self):
        self.walls = []
        wall_thickness = 10
        self.walls.append(pygame.Rect(150, 0, wall_thickness, 200))
        self.walls.append(pygame.Rect(150, 200, 250, wall_thickness))
        self.walls.append(pygame.Rect(400, 200, wall_thickness, 200))
        self.walls.append(pygame.Rect(250, 0, wall_thickness, 100))
        self.exit_rect = pygame.Rect(self.SCREEN_WIDTH - 60, self.SCREEN_HEIGHT - 60, 40, 40)
        
    def _spawn_boson(self):
        self.boson = {
            'pos': pygame.math.Vector2(50, 50),
            'vel': pygame.math.Vector2(0, 0),
            'launched': False,
            'angle_deg': 0.0
        }

    def _spawn_protons(self, num_protons):
        self.protons = []
        for _ in range(num_protons):
            while True:
                pos = pygame.math.Vector2(
                    self.np_random.uniform(self.PROTON_RADIUS, self.SCREEN_WIDTH - self.PROTON_RADIUS),
                    self.np_random.uniform(self.PROTON_RADIUS, self.SCREEN_HEIGHT - self.PROTON_RADIUS)
                )
                proton_rect = pygame.Rect(pos.x - self.PROTON_RADIUS, pos.y - self.PROTON_RADIUS, self.PROTON_RADIUS*2, self.PROTON_RADIUS*2)
                in_wall = any(wall.colliderect(proton_rect) for wall in self.walls)
                too_close_to_start = pos.distance_to(pygame.math.Vector2(50, 50)) < 100
                too_close_to_exit = self.exit_rect.inflate(50,50).collidepoint(pos)
                if not in_wall and not too_close_to_start and not too_close_to_exit:
                    break
            
            angle = self.np_random.uniform(0, 360)
            vel = pygame.math.Vector2(1, 0).rotate(angle) * self.PROTON_BASE_SPEED
            self.protons.append({
                'pos': pos,
                'vel': vel,
                'trail': deque(maxlen=self.PROTON_TRAIL_LENGTH)
            })

    def _create_energy_pulse(self, pos):
        self.energy_pulses.append({
            'pos': pos.copy(),
            'age': 0,
            'radius': 0
        })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # The dummy video driver should prevent a window from showing up
    # even with the display.set_mode call.
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Boson Maze")
        clock = pygame.time.Clock()
    except pygame.error:
        # Running in a truly headless environment, no display available
        screen = None
        clock = None
        print("Pygame display unavailable. Running simulation without rendering.")

    
    while running:
        movement, space, shift = 0, 0, 0
        
        if screen:
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
        else: # If no display, just step with random actions for demonstration
            action = env.action_space.sample()
            movement, space, shift = action.tolist()

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)

        if screen:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        if terminated or truncated:
            print(f"Episode Finished! Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
            if screen:
                pygame.time.wait(2000)
            obs, info = env.reset()

        if clock:
            clock.tick(GameEnv.FPS)
        
    env.close()