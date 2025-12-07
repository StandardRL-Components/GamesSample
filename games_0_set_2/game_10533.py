import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a cyberpunk-themed arcade survival game.
    The player controls a time-bending agent, defending a neon city
    from waves of monstrous enemies. The core mechanics involve strategic
    positioning, activating devastating time pulses, and flipping gravity.
    """
    
    game_description = (
        "A cyberpunk-themed arcade survival game where the player defends a neon city "
        "from waves of enemies using time pulses and gravity flips."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press Space to activate a time pulse and Shift to flip gravity."
    )
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2500
        self.MAX_WAVES = 20
        self.PLAYER_SIZE = 20
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_DRAG = 0.92
        self.GRAVITY = 0.4
        self.PULSE_COOLDOWN_MAX = 45  # 1.5 seconds at 30fps
        self.PULSE_SPEED = 5
        self.PULSE_MAX_RADIUS = 200

        # --- Color Palette (Cyberpunk Neon) ---
        self.COLOR_BG = (16, 5, 32)          # Dark Purple
        self.COLOR_GRID = (40, 20, 70)       # Faint Purple
        self.COLOR_SKYLINE = (25, 10, 50)    # Darker Purple/Blue
        self.COLOR_PLAYER = (0, 255, 255)    # Cyan
        self.COLOR_ENEMY = (255, 64, 64)     # Bright Red
        self.COLOR_PULSE = (0, 255, 255)     # Cyan
        self.COLOR_TEXT = (220, 220, 255)    # Light Lavender
        self.COLOR_GAMEOVER = (255, 0, 100)  # Magenta

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_gameover = pygame.font.SysFont('Consolas', 50, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_pos = None
        self.player_vel = None
        self.gravity_direction = 1  # 1 for down, -1 for up
        self.enemies = []
        self.pulses = []
        self.particles = []
        self.wave = 0
        self.pulse_cooldown = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        # --- Pre-generate static background elements ---
        self.skyline = self._generate_skyline()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.gravity_direction = 1

        self.enemies = []
        self.pulses = []
        self.particles = []

        self.wave = 1
        self._spawn_wave()

        self.pulse_cooldown = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- 1. Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1: self.player_vel.y -= self.PLAYER_ACCEL  # Up
        if movement == 2: self.player_vel.y += self.PLAYER_ACCEL  # Down
        if movement == 3: self.player_vel.x -= self.PLAYER_ACCEL  # Left
        if movement == 4: self.player_vel.x += self.PLAYER_ACCEL  # Right

        # Gravity Flip (on button press)
        if shift_held and not self.prev_shift_held:
            self.gravity_direction *= -1
            # sfx: gravity_flip

        # Time Pulse (on button press)
        if space_held and not self.prev_space_held and self.pulse_cooldown <= 0:
            self.pulses.append({'pos': self.player_pos.copy(), 'radius': 10, 'max_radius': self.PULSE_MAX_RADIUS})
            self.pulse_cooldown = self.PULSE_COOLDOWN_MAX
            # sfx: player_pulse
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # --- 2. Update Game State ---
        # Cooldowns
        if self.pulse_cooldown > 0:
            self.pulse_cooldown -= 1

        # Player physics
        self.player_vel.y += self.GRAVITY * self.gravity_direction
        self.player_vel *= self.PLAYER_DRAG
        self.player_pos += self.player_vel
        self._clamp_player_position()
        if self.player_vel.length() > 1:
            vel_base = -self.player_vel.normalize() * 2
            self._create_particles(self.player_pos, self.COLOR_PLAYER, 1, 5, vel_base=vel_base)


        # Enemies
        for enemy in self.enemies:
            if (self.player_pos - enemy['pos']).length() > 0:
                direction = (self.player_pos - enemy['pos']).normalize()
                enemy['pos'] += direction * enemy['speed']

        # Pulses
        for pulse in self.pulses[:]:
            pulse['radius'] += self.PULSE_SPEED
            if pulse['radius'] > pulse['max_radius']:
                self.pulses.remove(pulse)

        # Particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # --- 3. Collisions and Events ---
        # Pulse-Enemy collisions
        for pulse in self.pulses:
            for enemy in self.enemies[:]:
                if pulse['pos'].distance_to(enemy['pos']) < pulse['radius'] + enemy['size']:
                    self.enemies.remove(enemy)
                    reward += 0.1
                    self.score += 10
                    self._create_particles(enemy['pos'], self.COLOR_ENEMY, 20, 15, speed_scale=3)
                    # sfx: enemy_destroy

        # Player-Enemy collisions
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for enemy in self.enemies:
            if player_rect.collidepoint(enemy['pos']):
                self.game_over = True
                reward = -1.0 # Changed from -100 to be less punishing
                self.score -= 1000
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 50, 25, speed_scale=5)
                # sfx: player_death
                break
        
        # Wave progression
        if not self.enemies and not self.game_over:
            reward += 1.0
            self.score += 100
            self.wave += 1
            if self.wave > self.MAX_WAVES:
                self.game_over = True
                self.win = True
                reward += 10.0 # Changed from 100
                self.score += 10000
            else:
                self._spawn_wave()
                # sfx: wave_clear

        # --- 4. Termination ---
        truncated = self.steps >= self.MAX_STEPS
        terminated = self.game_over
        if truncated and not terminated:
            # Timeout is a form of loss
            reward = -1.0 # Changed from -100

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _clamp_player_position(self):
        half_size = self.PLAYER_SIZE / 2
        self.player_pos.x = np.clip(self.player_pos.x, half_size, self.WIDTH - half_size)
        self.player_pos.y = np.clip(self.player_pos.y, half_size, self.HEIGHT - half_size)
        
    def _spawn_wave(self):
        num_enemies = 2 + self.wave
        enemy_speed = 0.8 + (self.wave * 0.05)
        for _ in range(num_enemies):
            # Spawn enemies on the edges of the screen
            edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -20)
            elif edge == 'bottom':
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20)
            elif edge == 'left':
                pos = pygame.Vector2(-20, self.np_random.uniform(0, self.HEIGHT))
            else: # right
                pos = pygame.Vector2(self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT))

            self.enemies.append({
                'pos': pos,
                'speed': enemy_speed,
                'size': 8,
                'phase': self.np_random.uniform(0, 2 * math.pi)
            })

    def _create_particles(self, pos, color, count, max_life, speed_scale=1.0, vel_base=None):
        for _ in range(count):
            if vel_base is not None:
                vel = vel_base + pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)) * 0.5
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(0.5, 2.5) * speed_scale
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(max_life // 2, max_life + 1),
                'color': color,
                'size': self.np_random.uniform(1, 3)
            })

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave}

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_frame(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        self.screen.blit(self.skyline, (0, 0))
        self._render_grid()
        
        # --- Game Elements ---
        self._render_particles()
        self._render_pulses()
        self._render_enemies()
        self._render_player()

        # --- UI ---
        self._render_ui()

    def _generate_skyline(self):
        skyline_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        temp_rng = np.random.default_rng(123) # Use a fixed seed for deterministic skyline
        for i in range(50):
            x = temp_rng.integers(0, self.WIDTH)
            w = temp_rng.integers(10, 40)
            h = temp_rng.integers(20, 100)
            y = self.HEIGHT - h
            pygame.draw.rect(skyline_surface, self.COLOR_SKYLINE, (x, y, w, h))
        return skyline_surface

    def _render_grid(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 15))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_pulses(self):
        for pulse in self.pulses:
            r = int(pulse['radius'])
            pos = (int(pulse['pos'].x), int(pulse['pos'].y))
            if r > 0:
                alpha = int(255 * (1 - (pulse['radius'] / pulse['max_radius'])))
                color = (*self.COLOR_PULSE, alpha)
                
                if alpha > 0:
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], r + 2, (*color[:3], alpha // 4))
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], r + 1, (*color[:3], alpha // 2))
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], r, color)
                    if r > 1:
                        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], r-1, color)


    def _render_enemies(self):
        for enemy in self.enemies:
            pulse_factor = (math.sin(self.steps * 0.2 + enemy['phase']) + 1) / 2
            size = int(enemy['size'] + pulse_factor * 3)
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))

            glow_color = (*self.COLOR_ENEMY, 50)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 4, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 2, glow_color)
            
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_ENEMY)

    def _render_player(self):
        pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        size = self.PLAYER_SIZE
        rect = pygame.Rect(pos_int[0] - size / 2, pos_int[1] - size / 2, size, size)

        glow_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER, 50), glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, (rect.x - size / 2, rect.y - size / 2), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=3)

    def _render_ui(self):
        wave_text = self.font_ui.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        arrow_y = 40 if self.gravity_direction == 1 else self.HEIGHT - 40
        arrow_points = [(self.WIDTH - 30, arrow_y), (self.WIDTH - 20, arrow_y + 10 * self.gravity_direction), (self.WIDTH - 10, arrow_y)]
        pygame.draw.lines(self.screen, self.COLOR_TEXT, False, arrow_points, 2)

        if self.game_over:
            if self.win:
                msg = "SYSTEM DEFENDED"
                color = self.COLOR_PLAYER
            else:
                msg = "CONNECTION LOST"
                color = self.COLOR_GAMEOVER
            
            over_text = self.font_gameover.render(msg, True, color)
            over_rect = over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(over_text, over_rect)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # Example usage: play the game manually
    # This requires a display. Set SDL_VIDEODRIVER to something other than "dummy".
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Neon Pulse Defender")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Wave: {info['wave']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30)

    env.close()