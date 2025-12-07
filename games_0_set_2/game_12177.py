import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:13:34.451794
# Source Brief: brief_02177.md
# Brief Index: 2177
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Surf on quantum waves, collect particles for boosts, and race upwards to escape a collapsing singularity."
    user_guide = "Controls: ←→ to surf left and right. Press space to jump; use ↑ or ↓ while jumping to alter jump height. Press shift to use a speed boost."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.SINGULARITY_Y = 20.0
        
        # Colors
        self.COLOR_BG_TOP = (5, 0, 10)
        self.COLOR_BG_BOTTOM = (20, 0, 30)
        self.COLOR_PLAYER = (255, 150, 0)
        self.COLOR_PLAYER_GLOW = (255, 100, 0)
        self.COLOR_STABLE_WAVE = (0, 150, 255)
        self.COLOR_STABLE_WAVE_GLOW = (0, 200, 255)
        self.COLOR_UNSTABLE_WAVE = (150, 50, 255)
        self.COLOR_UNSTABLE_WAVE_GLOW = (200, 100, 255)
        self.COLOR_PARTICLE = (50, 255, 100)
        self.COLOR_SINGULARITY = (255, 255, 255)
        self.COLOR_SINGULARITY_GLOW = (200, 200, 255)
        self.COLOR_TEXT = (220, 220, 220)

        # Physics
        self.GRAVITY = 0.35
        self.JUMP_STRENGTH = -8.0
        self.JUMP_HEIGHT_MODIFIER = 2.0
        self.HORIZONTAL_ACCEL = 0.8
        self.AIR_CONTROL_FACTOR = 0.5
        self.FRICTION = 0.95
        self.MAX_VEL_X = 6.0
        self.COYOTE_TIME_FRAMES = 5
        
        # Super init
        super().__init__()
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        except pygame.error:
            self.font = pygame.font.Font(None, 24)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.last_world_y = None
        self.waves = None
        self.collectibles = None
        self.effects = None
        self.player_trail = None
        self.on_wave_id = -1
        self.coyote_timer = 0
        self.particle_inventory = None
        self.active_effects = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.world_y_offset = 0.0
        self.base_wave_freq = 0.0
        self.base_decay_rate = 0.0
        self.bg_stars = None
        
        self.np_random = None # Will be seeded in reset
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        
        self.waves = []
        self.collectibles = []
        self.effects = []
        self.player_trail = []
        
        self.on_wave_id = -1
        self.coyote_timer = 0
        self.particle_inventory = {"speed_boost": 1} # Start with one
        self.active_effects = {}

        self.prev_space_held = False
        self.prev_shift_held = False

        self.world_y_offset = 0.0
        self.base_wave_freq = 0.02
        self.base_decay_rate = 0.5
        
        self.bg_stars = [
            (self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT), self.np_random.uniform(0.1, 0.5))
            for _ in range(100)
        ]

        # Generate initial waves ensuring the player starts on one
        initial_y = self.HEIGHT / 2 + self.world_y_offset
        self._generate_wave(initial_y, force_stable=True)
        for y in np.arange(initial_y + 50, self.HEIGHT + 100, 50):
            self._generate_wave(y)
        for y in np.arange(initial_y - 50, 50, -50):
            self._generate_wave(y)

        self.last_world_y = self.player_pos[1] - self.world_y_offset

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        self._handle_input(movement, space_held, shift_held)
        reward += self._update_player_and_world()
        
        self.steps += 1
        
        current_world_y = self.player_pos[1] - self.world_y_offset
        y_delta = self.last_world_y - current_world_y
        reward += y_delta * 0.1 # Reward for upward movement
        self.last_world_y = current_world_y

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            if current_world_y <= self.SINGULARITY_Y:
                reward += 100 # Reached singularity
            else:
                reward -= 10 # Fell off
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        accel_factor = 1.0 if self.on_wave_id != -1 else self.AIR_CONTROL_FACTOR
        if movement == 3: # Left
            self.player_vel[0] -= self.HORIZONTAL_ACCEL * accel_factor
        elif movement == 4: # Right
            self.player_vel[0] += self.HORIZONTAL_ACCEL * accel_factor

        jump_pressed = space_held and not self.prev_space_held
        if jump_pressed and self.coyote_timer > 0:
            jump_mod = 0
            if movement == 1: jump_mod = -self.JUMP_HEIGHT_MODIFIER
            elif movement == 2: jump_mod = self.JUMP_HEIGHT_MODIFIER
            
            self.player_vel[1] = self.JUMP_STRENGTH + jump_mod
            self.coyote_timer = 0
            self.on_wave_id = -1
            self._create_particles(self.player_pos, 20, self.COLOR_PLAYER, 3, 5)

        use_pressed = shift_held and not self.prev_shift_held
        if use_pressed and self.particle_inventory["speed_boost"] > 0:
            self.particle_inventory["speed_boost"] -= 1
            self.active_effects["speed_boost"] = 90 # 3 seconds at 30fps
            self._create_particles(self.player_pos, 30, self.COLOR_PARTICLE, 5, 8)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_player_and_world(self):
        reward = 0
        
        # --- Update Player ---
        if "speed_boost" in self.active_effects:
            self.active_effects["speed_boost"] -= 1
            self.player_vel[0] *= 1.05
            if self.active_effects["speed_boost"] <= 0:
                del self.active_effects["speed_boost"]

        if self.on_wave_id == -1:
            self.player_vel[1] += self.GRAVITY
        
        self.player_vel[0] *= self.FRICTION
        self.player_vel[0] = np.clip(self.player_vel[0], -self.MAX_VEL_X, self.MAX_VEL_X)
        self.player_pos += self.player_vel
        
        self.player_trail.append(list(self.player_pos) + [10])
        for trail in self.player_trail: trail[2] -= 1
        self.player_trail = [t for t in self.player_trail if t[2] > 0]
        
        if self.player_pos[0] < 0: self.player_pos[0] = self.WIDTH
        if self.player_pos[0] > self.WIDTH: self.player_pos[0] = 0

        self.on_wave_id = -1
        for i, wave in enumerate(self.waves):
            world_wave_y = self._get_wave_y(wave, self.player_pos[0])
            screen_wave_y = world_wave_y + self.world_y_offset
            
            if self.player_pos[1] >= screen_wave_y - 5 and self.player_pos[1] <= screen_wave_y + 10 and self.player_vel[1] >= 0:
                self.player_pos[1] = screen_wave_y
                self.player_vel[1] = 0
                self.on_wave_id = i
                self.coyote_timer = self.COYOTE_TIME_FRAMES
                
                slope = self._get_wave_slope(wave, self.player_pos[0])
                self.player_vel[0] += slope * 0.1

                if abs(self.player_vel[0]) > 2.0:
                    reward += 0.05 # Grinding reward
                    if self.steps % 5 == 0:
                        self._create_particles(self.player_pos, 1, (200, 200, 200), 1, 2)

                if wave["unstable"]: wave["decay"] += self.base_decay_rate
                break
        
        if self.on_wave_id == -1: self.coyote_timer = max(0, self.coyote_timer - 1)

        # --- Update World ---
        if self.steps > 0 and self.steps % 100 == 0: self.base_wave_freq = min(0.06, self.base_wave_freq + 0.002)
        if self.steps > 0 and self.steps % 200 == 0: self.base_decay_rate = min(2.0, self.base_decay_rate + 0.05)

        scroll_threshold = self.HEIGHT / 2.5
        if self.player_pos[1] < scroll_threshold:
            scroll_amount = scroll_threshold - self.player_pos[1]
            self.world_y_offset += scroll_amount
            self.player_pos[1] += scroll_amount
        
        self.waves = [w for w in self.waves if not (w["unstable"] and w["decay"] > 100)]
        self.waves = [w for w in self.waves if w['y_base'] + self.world_y_offset < self.HEIGHT + 50]

        top_most_wave_y = min(w['y_base'] for w in self.waves) if self.waves else self.world_y_offset
        if top_most_wave_y > self.world_y_offset - self.HEIGHT:
             self._generate_wave(top_most_wave_y - 50)
        
        for p in self.collectibles[:]:
            screen_pos = p["pos"] + np.array([0, self.world_y_offset])
            dist = np.linalg.norm(self.player_pos - screen_pos)
            if dist < 20:
                self.collectibles.remove(p)
                self.particle_inventory["speed_boost"] += 1
                reward += 1.0 # Collectible reward
                break
        
        for p in self.effects: p["pos"] += p["vel"]; p["life"] -= 1
        self.effects = [p for p in self.effects if p["life"] > 0]
        
        return reward

    def _check_termination(self):
        current_world_y = self.player_pos[1] - self.world_y_offset
        if current_world_y <= self.SINGULARITY_Y or self.player_pos[1] > self.HEIGHT + 20:
            self.game_over = True
            return True
        return False

    def _get_wave_y(self, wave, x):
        return wave['y_base'] + math.sin(x * wave['freq'] + wave['phase']) * wave['amp']

    def _get_wave_slope(self, wave, x):
        return math.cos(x * wave['freq'] + wave['phase']) * wave['amp'] * wave['freq']

    def _generate_wave(self, y_pos, force_stable=False):
        freq = self.base_wave_freq + self.np_random.uniform(-0.005, 0.005)
        amp = self.np_random.uniform(20, 50)
        unstable = not force_stable and self.np_random.random() < 0.3 + (self.world_y_offset / 5000)
        
        wave = {
            "id": self.steps + self.np_random.integers(10000), "y_base": y_pos, "amp": amp, "freq": freq, 
            "phase": self.np_random.uniform(0, 2 * math.pi), "unstable": unstable, "decay": 0
        }
        self.waves.append(wave)

        if self.np_random.random() < 0.15:
            x = self.np_random.uniform(50, self.WIDTH - 50)
            y = self._get_wave_y(wave, x)
            self.collectibles.append({"pos": np.array([x, y])})

    def _create_particles(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.effects.append({"pos": pos.copy(), "vel": vel, "life": self.np_random.integers(10, 20), "color": color})

    def _get_observation(self):
        self._render_bg()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_bg(self):
        self.screen.fill(self.COLOR_BG_BOTTOM)
        rect = pygame.Rect(0, 0, self.WIDTH, self.HEIGHT * 0.7)
        shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, (*self.COLOR_BG_TOP, 200), shape_surf.get_rect())
        self.screen.blit(shape_surf, rect)

        for x, y, speed in self.bg_stars:
            y_on_screen = (y + self.world_y_offset * speed) % self.HEIGHT
            brightness = int(100 + 155 * speed)
            pygame.gfxdraw.pixel(self.screen, int(x), int(y_on_screen), (brightness, brightness, brightness))

    def _render_game(self):
        sing_y = self.SINGULARITY_Y + self.world_y_offset
        pulse = 1 + 0.2 * math.sin(self.steps * 0.1)
        self._draw_glow_circle(self.screen, (self.WIDTH / 2, sing_y), 15 * pulse, self.COLOR_SINGULARITY, self.COLOR_SINGULARITY_GLOW)

        for wave in self.waves:
            points = [(x, self._get_wave_y(wave, x) + self.world_y_offset) for x in range(0, self.WIDTH + 1, 5)]
            color, glow = (self.COLOR_UNSTABLE_WAVE, self.COLOR_UNSTABLE_WAVE_GLOW) if wave["unstable"] else (self.COLOR_STABLE_WAVE, self.COLOR_STABLE_WAVE_GLOW)
            if len(points) > 1:
                pygame.draw.lines(self.screen, glow, False, points, 5)
                pygame.draw.lines(self.screen, color, False, points, 2)
            if wave["unstable"]:
                s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                pygame.draw.lines(s, (255, 0, 0, int(min(255, wave["decay"] * 2.55))), False, points, 2)
                self.screen.blit(s, (0, 0))

        for p in self.collectibles:
            pos = p["pos"] + np.array([0, self.world_y_offset])
            self._draw_glow_circle(self.screen, pos, 6, self.COLOR_PARTICLE, self.COLOR_PARTICLE)

        for t in self.player_trail:
            pos = t[:2]
            self._draw_glow_circle(self.screen, pos, 4 * (t[2] / 10), (*self.COLOR_PLAYER_GLOW, int(255 * (t[2] / 10))), None, is_alpha=True)

        player_render_pos = self.player_pos
        self._draw_glow_circle(self.screen, player_render_pos, 8, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
        if "speed_boost" in self.active_effects:
            self._draw_glow_circle(self.screen, player_render_pos, 12, (*self.COLOR_PARTICLE, 150), None, width=2, is_alpha=True)

        for p in self.effects:
            pos = p["pos"]
            # Use SRALPHA surface for alpha blending particles
            s = pygame.Surface((4, 4), pygame.SRCALPHA)
            alpha = int(255 * (p["life"] / 20.0))
            pygame.draw.circle(s, (*p["color"], alpha), (2, 2), 2)
            self.screen.blit(s, (int(pos[0]) - 2, int(pos[1]) - 2))

    def _render_ui(self):
        dist = max(0, int((self.player_pos[1] - self.world_y_offset) - self.SINGULARITY_Y))
        dist_text = self.font.render(f"DISTANCE: {dist}", True, self.COLOR_TEXT)
        self.screen.blit(dist_text, (10, 10))

        ui_text = self.font.render("BOOST", True, self.COLOR_TEXT)
        self.screen.blit(ui_text, (self.WIDTH - 150, self.HEIGHT - 30))
        for i in range(self.particle_inventory["speed_boost"]):
            self._draw_glow_circle(self.screen, (self.WIDTH - 80 + i * 20, self.HEIGHT - 22), 6, self.COLOR_PARTICLE, self.COLOR_PARTICLE)

    def _draw_glow_circle(self, surface, pos, radius, color, glow_color, width=0, is_alpha=False):
        pos = (int(pos[0]), int(pos[1]))
        radius = int(radius)
        if radius <= 0: return

        if is_alpha:
            target_rect = pygame.Rect(pos[0] - radius, pos[1] - radius, radius * 2, radius * 2)
            shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            if width == 0:
                pygame.draw.circle(shape_surf, color, (radius, radius), radius)
            else:
                pygame.draw.circle(shape_surf, color, (radius, radius), radius, width)
            surface.blit(shape_surf, target_rect)
            return

        if glow_color:
            glow_radius = radius + 4
            target_rect = pygame.Rect(pos[0] - glow_radius, pos[1] - glow_radius, glow_radius * 2, glow_radius * 2)
            shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            pygame.draw.circle(shape_surf, (*glow_color[:3], 50), (glow_radius, glow_radius), glow_radius)
            pygame.draw.circle(shape_surf, (*glow_color[:3], 80), (glow_radius, glow_radius), int(glow_radius*0.8))
            surface.blit(shape_surf, target_rect)

        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_to_singularity": max(0, int((self.player_pos[1] - self.world_y_offset) - self.SINGULARITY_Y)),
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the game and play it manually
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    pygame.display.set_caption("Quantum Wave Surfer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True

    while running:
        action = [0, 0, 0] # Default no-op
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Key presses for continuous actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        
        if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
             action[0] = 3
        elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
             action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(30)

    env.close()
    pygame.quit()