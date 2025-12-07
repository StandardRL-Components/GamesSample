import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:54:46.396961
# Source Brief: brief_03227.md
# Brief Index: 3227
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Navigate your spaceship through a dangerous asteroid field to collect energy orbs before time runs out."
    )
    user_guide = (
        "Controls: Use ↑ to accelerate, ↓ to brake, and ←→ to turn your ship."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_FPS = 30
    MAX_EPISODE_STEPS = 900 # 30 seconds at 30 FPS

    # --- Colors ---
    COLOR_BG = (10, 15, 30)
    COLOR_SHIP = (50, 150, 255)
    COLOR_SHIP_GLOW = (150, 200, 255)
    COLOR_ORB = (50, 255, 150)
    COLOR_ORB_GLOW = (150, 255, 200)
    COLOR_ASTEROID = (160, 80, 40)
    COLOR_ASTEROID_OUTLINE = (100, 50, 20)
    COLOR_PARTICLE = (255, 200, 150)
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_UI_SUCCESS = (100, 255, 100)
    COLOR_UI_FAILURE = (255, 100, 100)

    # --- Game Parameters ---
    SHIP_ROTATION_SPEED = 5.0  # degrees per frame
    SHIP_ACCELERATION = 0.2
    SHIP_MAX_SPEED = 6.0
    SHIP_DRAG = 0.985 # Multiplier to velocity each frame
    SHIP_RADIUS = 12

    ASTEROID_MIN_RADIUS = 20
    ASTEROID_MAX_RADIUS = 40
    ASTEROID_MIN_VERTS = 5
    ASTEROID_MAX_VERTS = 10
    ASTEROID_ROTATION_SPEED = 0.5

    ORB_RADIUS = 8
    ORB_PULSE_SPEED = 0.1
    ORB_PULSE_AMOUNT = 2

    PARTICLE_LIFETIME = 20
    PARTICLE_SPEED = 2.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.orbs_to_collect_for_level = 0
        self.timer = 0

        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.player_angle = 0.0 # In degrees

        self.orbs = []
        self.asteroids = []
        self.particles = []
        self.stars = []

        # self.reset() # Removed to avoid calling reset before seed is set by wrapper
        # self.validate_implementation() # Removed as it's for dev testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Check if previous run was a success to level up
        if self.score >= self.orbs_to_collect_for_level and self.orbs_to_collect_for_level > 0:
            self.level += 1
            # Sound: Level Up!
        else:
            self.level = 1 # Reset level on failure

        # --- Initialize State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_EPISODE_STEPS

        # --- Player State ---
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.player_angle = -90.0  # Pointing up

        # --- Entity Generation ---
        num_asteroids = 3 + self.level * 2
        self.orbs_to_collect_for_level = 8 + self.level * 2

        self.orbs = []
        self.asteroids = []
        self.particles = []

        # Generate stars for background
        self.stars = [
            (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT), self.np_random.integers(1, 3))
            for _ in range(100)
        ]

        # Generate asteroids
        for _ in range(num_asteroids):
            self._spawn_asteroid()

        # Generate orbs
        for _ in range(self.orbs_to_collect_for_level):
            self._spawn_orb()

        return self._get_observation(), self._get_info()

    def _spawn_entity(self, radius):
        """Helper to find a safe spawn location for an entity."""
        for _ in range(100): # Attempt to find a spot 100 times
            pos = np.array([
                self.np_random.uniform(radius, self.SCREEN_WIDTH - radius),
                self.np_random.uniform(radius, self.SCREEN_HEIGHT - radius)
            ])
            # Check distance from player start
            if np.linalg.norm(pos - self.player_pos) < 100:
                continue
            # Check distance from other asteroids
            too_close = False
            for ast in self.asteroids:
                if np.linalg.norm(pos - ast['pos']) < ast['radius'] + radius + 20:
                    too_close = True
                    break
            if not too_close:
                return pos
        return np.array([radius, radius]) # Failsafe spawn

    def _spawn_asteroid(self):
        radius = self.np_random.uniform(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS)
        pos = self._spawn_entity(radius)
        angle = self.np_random.uniform(0, 360)
        rot_speed = self.np_random.uniform(-self.ASTEROID_ROTATION_SPEED, self.ASTEROID_ROTATION_SPEED)
        num_verts = self.np_random.integers(self.ASTEROID_MIN_VERTS, self.ASTEROID_MAX_VERTS + 1)
        
        verts = []
        for i in range(num_verts):
            a = 2 * math.pi * i / num_verts
            r = radius + self.np_random.uniform(-radius * 0.2, radius * 0.2)
            verts.append((r * math.cos(a), r * math.sin(a)))

        self.asteroids.append({
            'pos': pos, 'radius': radius, 'angle': angle,
            'rot_speed': rot_speed, 'verts': verts
        })

    def _spawn_orb(self):
        pos = self._spawn_entity(self.ORB_RADIUS)
        self.orbs.append({'pos': pos})

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False

        # --- Update Game Logic ---
        self.steps += 1
        self.timer -= 1

        # --- Player Action Handling ---
        # Rotation
        if movement == 3:  # Left
            self.player_angle -= self.SHIP_ROTATION_SPEED
        if movement == 4:  # Right
            self.player_angle += self.SHIP_ROTATION_SPEED

        # Acceleration
        speed = np.linalg.norm(self.player_vel)
        if movement == 1:  # Up
            rad_angle = math.radians(self.player_angle)
            accel_vec = np.array([math.cos(rad_angle), math.sin(rad_angle)]) * self.SHIP_ACCELERATION
            self.player_vel += accel_vec
            # Visual: Spawn particles
            self._spawn_particles()
        elif movement == 2: # Down
             if speed > 0:
                self.player_vel -= self.player_vel / speed * self.SHIP_ACCELERATION
        
        # Cap speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.SHIP_MAX_SPEED:
            self.player_vel = (self.player_vel / speed) * self.SHIP_MAX_SPEED
        
        # Apply drag
        self.player_vel *= self.SHIP_DRAG

        # Update position
        self.player_pos += self.player_vel

        # Screen wrap
        self.player_pos[0] %= self.SCREEN_WIDTH
        self.player_pos[1] %= self.SCREEN_HEIGHT

        # --- Update World Entities ---
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        
        # Update asteroids
        for ast in self.asteroids:
            ast['angle'] += ast['rot_speed']

        # --- Collision Detection & Rewards ---
        # Orb collection
        remaining_orbs = []
        for orb in self.orbs:
            if np.linalg.norm(self.player_pos - orb['pos']) < self.SHIP_RADIUS + self.ORB_RADIUS:
                self.score += 1
                reward += 1.0
            else:
                remaining_orbs.append(orb)
        self.orbs = remaining_orbs

        # Asteroid collision
        for ast in self.asteroids:
            if np.linalg.norm(self.player_pos - ast['pos']) < self.SHIP_RADIUS + ast['radius']:
                terminated = True
                reward = -100.0
                break

        # --- Termination Conditions ---
        if not terminated:
            reward += 0.01 # Small survival reward

            if self.score >= self.orbs_to_collect_for_level:
                terminated = True
                reward = 100.0

            elif self.timer <= 0:
                terminated = True
                reward = -100.0 # Time out penalty

        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_particles()
        self._render_asteroids()
        self._render_orbs()
        self._render_ship()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "timer": self.timer,
            "orbs_to_collect": self.orbs_to_collect_for_level
        }

    def _render_stars(self):
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, (200, 200, 220), (x, y, size, size))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / self.PARTICLE_LIFETIME))
            color = (*self.COLOR_PARTICLE, alpha)
            size = int(p['size'] * (p['life'] / self.PARTICLE_LIFETIME))
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_asteroids(self):
        for ast in self.asteroids:
            points = []
            for vx, vy in ast['verts']:
                angle_rad = math.radians(ast['angle'])
                x = ast['pos'][0] + vx * math.cos(angle_rad) - vy * math.sin(angle_rad)
                y = ast['pos'][1] + vx * math.sin(angle_rad) + vy * math.cos(angle_rad)
                points.append((int(x), int(y)))
            
            if len(points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID_OUTLINE)

    def _render_orbs(self):
        pulse = math.sin(self.steps * self.ORB_PULSE_SPEED) * self.ORB_PULSE_AMOUNT
        for orb in self.orbs:
            pos = (int(orb['pos'][0]), int(orb['pos'][1]))
            radius = int(self.ORB_RADIUS + pulse)
            
            # Glow effect
            glow_radius = int(radius * 1.8)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_ORB_GLOW, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ORB)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ORB_GLOW)
            
    def _render_ship(self):
        # The ship is always rendered at the center, but its logical position is player_pos
        # To render everything else correctly, we need to offset them by the player's position.
        # This implementation renders the ship at a fixed position and moves the world around it.
        # This is incorrect. The ship should be rendered at self.player_pos. Let's fix this.
        
        # The original code renders the ship at the center and moves the world. This is a common
        # technique for scrolling games, but it makes rendering complex. Let's adjust rendering
        # to draw everything in its world coordinates.
        
        # Let's revert the ship rendering to be at its world position `self.player_pos`
        center = self.player_pos
        rad = math.radians(self.player_angle)
        
        # Ship body points
        p1 = (center[0] + self.SHIP_RADIUS * math.cos(rad), 
              center[1] + self.SHIP_RADIUS * math.sin(rad))
        p2 = (center[0] + self.SHIP_RADIUS * math.cos(rad + 2.4), 
              center[1] + self.SHIP_RADIUS * math.sin(rad + 2.4))
        p3 = (center[0] + self.SHIP_RADIUS * 0.5 * math.cos(rad + math.pi), 
              center[1] + self.SHIP_RADIUS * 0.5 * math.sin(rad + math.pi))
        p4 = (center[0] + self.SHIP_RADIUS * math.cos(rad - 2.4), 
              center[1] + self.SHIP_RADIUS * math.sin(rad - 2.4))
        
        points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3, p4]]

        # To handle screen wrapping, we need to draw the ship up to 9 times
        for dx in [-self.SCREEN_WIDTH, 0, self.SCREEN_WIDTH]:
            for dy in [-self.SCREEN_HEIGHT, 0, self.SCREEN_HEIGHT]:
                wrapped_points = [(p[0] + dx, p[1] + dy) for p in points]
                wrapped_center = (center[0] + dx, center[1] + dy)

                # Glow effect
                glow_radius = int(self.SHIP_RADIUS * 2.5)
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*self.COLOR_SHIP_GLOW, 40), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surf, (wrapped_center[0] - glow_radius, wrapped_center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
                
                pygame.gfxdraw.filled_polygon(self.screen, wrapped_points, self.COLOR_SHIP)
                pygame.gfxdraw.aapolygon(self.screen, wrapped_points, self.COLOR_SHIP_GLOW)

    def _render_ui(self):
        # Orbs collected
        orb_text = f"ORBS: {self.score} / {self.orbs_to_collect_for_level}"
        orb_surf = self.font_ui.render(orb_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(orb_surf, (10, 10))

        # Time remaining
        time_sec = max(0, self.timer / self.GAME_FPS)
        time_color = self.COLOR_UI_TEXT if time_sec > 5 else self.COLOR_UI_FAILURE
        time_text = f"TIME: {time_sec:.1f}"
        time_surf = self.font_ui.render(time_text, True, time_color)
        self.screen.blit(time_surf, (10, 30))

        # Level
        level_text = f"LEVEL: {self.level}"
        level_surf = self.font_ui.render(level_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(level_surf, (self.SCREEN_WIDTH - level_surf.get_width() - 10, 10))

        # Game Over message
        if self.timer <=0 or (self.steps >= self.MAX_EPISODE_STEPS and self.score < self.orbs_to_collect_for_level):
            msg = "MISSION FAILED"
            color = self.COLOR_UI_FAILURE
            msg_surf = self.font_big.render(msg, True, color)
            self.screen.blit(msg_surf, (self.SCREEN_WIDTH/2 - msg_surf.get_width()/2, self.SCREEN_HEIGHT/2 - msg_surf.get_height()/2))
        elif self.score >= self.orbs_to_collect_for_level:
            msg = "LEVEL COMPLETE"
            color = self.COLOR_UI_SUCCESS
            msg_surf = self.font_big.render(msg, True, color)
            self.screen.blit(msg_surf, (self.SCREEN_WIDTH/2 - msg_surf.get_width()/2, self.SCREEN_HEIGHT/2 - msg_surf.get_height()/2))

    def _spawn_particles(self):
        num_particles = 3
        rad = math.radians(self.player_angle + 180) # Opposite direction of movement
        for _ in range(num_particles):
            offset_angle = rad + self.np_random.uniform(-0.4, 0.4)
            speed = self.PARTICLE_SPEED + self.np_random.uniform(-0.5, 0.5)
            vel = np.array([math.cos(offset_angle), math.sin(offset_angle)]) * speed
            
            # Spawn behind the ship
            spawn_pos_offset = np.array([math.cos(rad), math.sin(rad)]) * self.SHIP_RADIUS
            
            self.particles.append({
                'pos': self.player_pos + spawn_pos_offset,
                'vel': vel + self.player_vel * 0.5, # Inherit some ship velocity
                'life': self.PARTICLE_LIFETIME,
                'size': self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It requires pygame to be installed and a display available
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Collector")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0.0

    while not done:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0
                print("--- Game Reset ---")

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds before reset
            obs, info = env.reset()
            total_reward = 0.0
        
        clock.tick(GameEnv.GAME_FPS)

    env.close()