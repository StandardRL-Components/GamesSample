import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:32:17.654209
# Source Brief: brief_01796.md
# Brief Index: 1796
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a drone in a shrinking arena, collecting power-ups to survive. "
        "Dodge shifting winds and escape the collapsing boundaries to achieve victory."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to pilot your drone and collect the green power-ups."
    )
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_POWERUP = (0, 255, 150)
    COLOR_POWERUP_GLOW = (0, 255, 150, 60)
    COLOR_ARENA = (255, 50, 50)
    COLOR_ARENA_GLOW = (255, 50, 50, 40)
    COLOR_WIND = (200, 200, 255, 20)
    COLOR_TEXT = (220, 220, 240)
    COLOR_PARTICLE = (200, 220, 255)

    # Player
    PLAYER_ACCEL = 0.8
    PLAYER_DRAG = 0.92
    PLAYER_MAX_SPEED = 8.0
    PLAYER_SIZE = 12
    SPEED_BOOST_MULTIPLIER = 2.0
    SPEED_BOOST_DURATION = 90  # frames

    # Arena
    INITIAL_ARENA_SHRINK_RATE = 0.1  # pixels per step
    ARENA_SHRINK_ACCELERATION = 0.005 # rate increase per 100 steps
    ARENA_MIN_AREA_FACTOR = 0.2
    SHRINK_SLOW_DURATION = 120 # frames

    # Powerups
    POWERUP_RADIUS = 8
    POWERUP_COUNT = 3
    POWERUP_WIN_CONDITION = 12

    # Wind
    WIND_MAX_STRENGTH = 0.2
    WIND_OSCILLATION_SPEED = 0.01

    # Episode
    MAX_STEPS = 1500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.powerups = []
        self.particles = []
        self.powerups_collected = 0
        self.arena_size = np.zeros(2, dtype=np.float32)
        self.arena_shrink_rate = 0.0
        self.initial_arena_area = self.SCREEN_WIDTH * self.SCREEN_HEIGHT
        self.wind_strength = 0.0
        self.wind_phase = 0.0
        self.speed_boost_timer = 0
        self.shrink_slow_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50], dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)

        self.powerups_collected = 0
        self.powerups = []
        
        self.arena_size = np.array([self.SCREEN_WIDTH, self.SCREEN_HEIGHT], dtype=np.float32)
        self.arena_shrink_rate = self.INITIAL_ARENA_SHRINK_RATE
        
        self.wind_phase = self.np_random.uniform(0, 2 * math.pi)
        self.wind_strength = 0.0
        
        self.speed_boost_timer = 0
        self.shrink_slow_timer = 0

        self.particles = []
        
        for _ in range(self.POWERUP_COUNT):
            self._spawn_powerup()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = 0.1 # Survival reward

        # --- Update Game Logic ---
        self._update_wind()
        self._update_player(movement)
        self._update_arena()
        self._update_particles()

        # --- Collision Detection ---
        # Powerups
        for powerup in self.powerups[:]:
            if np.linalg.norm(self.player_pos - powerup) < self.PLAYER_SIZE + self.POWERUP_RADIUS:
                # SFX: Powerup collect
                self.powerups.remove(powerup)
                self.powerups_collected += 1
                self.score += 1
                reward += 10.0
                self.speed_boost_timer = self.SPEED_BOOST_DURATION
                self.shrink_slow_timer = self.SHRINK_SLOW_DURATION
                self._create_collection_effect(powerup)
                self._spawn_powerup()

        # Arena boundary
        arena_half = self.arena_size / 2
        arena_center = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        if (abs(self.player_pos[0] - arena_center[0]) > arena_half[0] or
            abs(self.player_pos[1] - arena_center[1]) > arena_half[1]):
            # SFX: Player death / collision
            self.game_over = True
            reward = -100.0

        # --- Check Termination Conditions ---
        terminated = self.game_over
        
        if self.powerups_collected >= self.POWERUP_WIN_CONDITION:
            # SFX: Victory
            terminated = True
            reward += 100.0
        
        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Gymnasium standard
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, movement):
        # Apply acceleration from input
        accel = np.zeros(2, dtype=np.float32)
        if movement == 1: accel[1] = -self.PLAYER_ACCEL  # Up
        elif movement == 2: accel[1] = self.PLAYER_ACCEL   # Down
        elif movement == 3: accel[0] = -self.PLAYER_ACCEL  # Left
        elif movement == 4: accel[0] = self.PLAYER_ACCEL   # Right

        # Apply wind force
        accel[0] += self.wind_strength
        
        self.player_vel += accel
        self.player_vel *= self.PLAYER_DRAG

        # Speed boost
        max_speed = self.PLAYER_MAX_SPEED
        if self.speed_boost_timer > 0:
            self.speed_boost_timer -= 1
            max_speed *= self.SPEED_BOOST_MULTIPLIER

        # Clamp velocity
        speed = np.linalg.norm(self.player_vel)
        if speed > max_speed:
            self.player_vel = self.player_vel / speed * max_speed
            
        self.player_pos += self.player_vel

        # Clamp position to screen bounds (visual only, logic uses arena)
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.SCREEN_WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT)

    def _update_arena(self):
        if self.shrink_slow_timer > 0:
            self.shrink_slow_timer -= 1
            return # Pause shrinking
            
        current_shrink_rate = self.arena_shrink_rate + (self.steps // 100) * self.ARENA_SHRINK_ACCELERATION
        
        aspect_ratio = self.SCREEN_WIDTH / self.SCREEN_HEIGHT
        shrink_x = current_shrink_rate
        shrink_y = current_shrink_rate / aspect_ratio
        
        self.arena_size -= np.array([shrink_x, shrink_y]) * 2
        
        min_size = np.array([self.SCREEN_WIDTH, self.SCREEN_HEIGHT]) * math.sqrt(self.ARENA_MIN_AREA_FACTOR)
        self.arena_size[0] = max(self.arena_size[0], min_size[0])
        self.arena_size[1] = max(self.arena_size[1], min_size[1])

    def _update_wind(self):
        self.wind_phase += self.WIND_OSCILLATION_SPEED
        self.wind_strength = math.sin(self.wind_phase) * self.WIND_MAX_STRENGTH

    def _spawn_powerup(self):
        arena_half = self.arena_size / 2
        arena_center = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        
        # Spawn away from player and edges
        while True:
            x = self.np_random.uniform(arena_center[0] - arena_half[0] * 0.8, arena_center[0] + arena_half[0] * 0.8)
            y = self.np_random.uniform(arena_center[1] - arena_half[1] * 0.8, arena_center[1] + arena_half[1] * 0.8)
            pos = np.array([x, y])
            if np.linalg.norm(pos - self.player_pos) > 100:
                self.powerups.append(pos)
                break

    def _create_collection_effect(self, pos):
        # SFX: Particle burst
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifetime = self.np_random.integers(20, 40)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifetime': lifetime, 'size': 3})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['lifetime'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
            if p['lifetime'] <= 0:
                self.particles.remove(p)

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
            "powerups_collected": self.powerups_collected,
            "arena_percentage": (self.arena_size[0] * self.arena_size[1]) / self.initial_arena_area * 100
        }

    def _render_game(self):
        arena_center = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        # Render Wind
        self._render_wind(arena_center)

        # Render Arena
        arena_rect = pygame.Rect(
            arena_center[0] - self.arena_size[0] / 2,
            arena_center[1] - self.arena_size[1] / 2,
            self.arena_size[0],
            self.arena_size[1]
        )
        pygame.draw.rect(self.screen, self.COLOR_ARENA, arena_rect, 3, border_radius=5)
        
        # Render Powerups
        for pos in self.powerups:
            self._render_glowing_circle(pos, self.POWERUP_RADIUS, self.COLOR_POWERUP, self.COLOR_POWERUP_GLOW)

        # Render Player Trail
        if self.speed_boost_timer > 0:
            self._render_player_trail()

        # Render Player
        self._render_player()

        # Render Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, p['pos'].astype(int), int(p['size']))

    def _render_player(self):
        pos = self.player_pos.astype(int)
        size = self.PLAYER_SIZE
        
        # Glow effect
        glow_color = self.COLOR_PLAYER_GLOW
        if self.speed_boost_timer > 0:
            # Pulsing glow for speed boost
            pulse = (math.sin(self.steps * 0.5) + 1) / 2
            glow_size = int(size * (2.5 + pulse * 1.5))
            glow_color = (self.COLOR_PLAYER[0], self.COLOR_PLAYER[1], self.COLOR_PLAYER[2], 20 + int(pulse * 30))
        else:
            glow_size = int(size * 3)

        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surf, (pos[0] - glow_size, pos[1] - glow_size), special_flags=pygame.BLEND_RGBA_ADD)

        # Player triangle
        angle = math.atan2(self.player_vel[1], self.player_vel[0]) if np.linalg.norm(self.player_vel) > 0.1 else -math.pi/2
        points = [
            (pos[0] + math.cos(angle) * size, pos[1] + math.sin(angle) * size),
            (pos[0] + math.cos(angle + 2.2) * size, pos[1] + math.sin(angle + 2.2) * size),
            (pos[0] + math.cos(angle - 2.2) * size, pos[1] + math.sin(angle - 2.2) * size),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_player_trail(self):
        trail_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        trail_color = list(self.COLOR_PLAYER) + [100]
        
        length = int(np.linalg.norm(self.player_vel) * 2)
        if length < 5: return
        
        start_pos = self.player_pos
        end_pos = self.player_pos - self.player_vel / np.linalg.norm(self.player_vel) * length
        
        pygame.draw.line(trail_surf, trail_color, start_pos, end_pos, width=int(self.PLAYER_SIZE * 1.5))
        self.screen.blit(trail_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_wind(self, arena_center):
        num_arrows = 15
        for i in range(num_arrows):
            y = (i / num_arrows) * self.SCREEN_HEIGHT
            x_offset = self.wind_strength * 200 * (math.sin(self.wind_phase + i * 0.5) + 1.5)
            x_center = arena_center[0] + x_offset
            
            start_pos = (x_center - 15 * self.wind_strength, y)
            end_pos = (x_center + 15 * self.wind_strength, y)
            
            if self.wind_strength > 0:
                pygame.gfxdraw.line(self.screen, int(start_pos[0]), int(y), int(end_pos[0]), int(y), self.COLOR_WIND)
            else:
                pygame.gfxdraw.line(self.screen, int(end_pos[0]), int(y), int(start_pos[0]), int(y), self.COLOR_WIND)

    def _render_glowing_circle(self, pos, radius, color, glow_color):
        pos_int = pos.astype(int)
        glow_radius = int(radius * 2.5)
        
        # Pulsing glow
        pulse = (math.sin(self.steps * 0.1 + pos[0]) + 1) / 2
        current_glow_radius = int(glow_radius * (0.8 + pulse * 0.2))

        glow_surf = pygame.Surface((current_glow_radius * 2, current_glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (current_glow_radius, current_glow_radius), current_glow_radius)
        self.screen.blit(glow_surf, (pos_int[0] - current_glow_radius, pos_int[1] - current_glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, color)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)

    def _render_ui(self):
        # Powerups collected
        score_text = self.font_main.render(f"Power-ups: {self.powerups_collected} / {self.POWERUP_WIN_CONDITION}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Arena size
        arena_percent = (self.arena_size[0] * self.arena_size[1]) / self.initial_arena_area * 100
        arena_text = self.font_main.render(f"Arena: {arena_percent:.1f}%", True, self.COLOR_TEXT)
        self.screen.blit(arena_text, (self.SCREEN_WIDTH - arena_text.get_width() - 10, 10))
        
        # Speed boost timer
        if self.speed_boost_timer > 0:
            boost_text = self.font_small.render("SPEED BOOST", True, self.COLOR_PLAYER)
            self.screen.blit(boost_text, (self.SCREEN_WIDTH / 2 - boost_text.get_width() / 2, self.SCREEN_HEIGHT - 30))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not run in the evaluation environment.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    pygame.display.set_caption("Drone Arena")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    # Map keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_w: 1,
        pygame.K_DOWN: 2,
        pygame.K_s: 2,
        pygame.K_LEFT: 3,
        pygame.K_a: 3,
        pygame.K_RIGHT: 4,
        pygame.K_d: 4,
    }

    while not done:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Movement
        for key, move_action in key_map.items():
            if keys[key]:
                action[0] = move_action
                break

        # Space and Shift
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for consistent feel

    env.close()