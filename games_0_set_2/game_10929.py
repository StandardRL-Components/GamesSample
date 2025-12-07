import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:13:21.956557
# Source Brief: brief_00929.md
# Brief Index: 929
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
        "Launch a shape-shifting object in an isometric arena to hit targets. Aim your shot, "
        "choose your power, and transform between a cube and a sphere to score points before time runs out."
    )
    user_guide = (
        "Aiming: Use ↑/↓ for power, ←/→ for angle. Press space to launch. In-flight: Press shift to transform."
    )
    auto_advance = True

    # --- Constants ---
    # Game settings
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GAME_FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * GAME_FPS
    WIN_SCORE = 1000

    # Colors
    COLOR_BG = (15, 18, 23)
    COLOR_FLOOR = (230, 230, 230)
    COLOR_WALLS = (150, 150, 150)
    COLOR_CUBE = (0, 150, 255)
    COLOR_CUBE_GLOW = (0, 150, 255, 50)
    COLOR_SPHERE = (255, 80, 80)
    COLOR_SPHERE_GLOW = (255, 80, 80, 50)
    COLOR_TARGET = (255, 220, 0)
    COLOR_SHADOW = (0, 0, 0, 50)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_POWER_BAR_BG = (50, 50, 50)
    COLOR_POWER_BAR_FILL = (200, 200, 200)

    # Physics
    WORLD_WIDTH, WORLD_DEPTH = 500, 300
    GRAVITY = -0.3
    FLOOR_BOUNCINESS = 0.6
    WALL_BOUNCINESS = 0.8
    FRICTION = 0.98
    STOP_VELOCITY_THRESHOLD = 0.1
    MAX_LAUNCH_POWER = 15.0
    
    # Isometric Projection
    ISO_ANGLE = math.pi / 6
    ISO_COS = math.cos(ISO_ANGLE)
    ISO_SIN = math.sin(ISO_ANGLE)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.game_state = 'AIMING' # 'AIMING' or 'LAUNCHED'
        self.player_form = 'CUBE' # 'CUBE' or 'SPHERE'
        
        self.player_pos = np.array([0.0, 0.0, 0.0]) # x, y (height), z
        self.player_vel = np.array([0.0, 0.0, 0.0])
        self.launch_angle = 0.0
        self.launch_power = self.MAX_LAUNCH_POWER / 2
        
        self.targets = []
        self.particles = []
        
        self.target_spawn_timer = 0
        self.base_target_spawn_cooldown = 2.0 # seconds
        
        self.shift_was_pressed = False

        self.launch_origin = np.array([0.0, 0.0])
        
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.game_over = False
        self.game_state = 'AIMING'
        self.player_form = 'CUBE'
        
        self.player_pos = np.array([0.0, 50.0, 0.0]) # Start slightly elevated
        self.player_vel = np.array([0.0, 0.0, 0.0])
        
        self.launch_angle = self.np_random.uniform(0, 2 * math.pi)
        self.launch_power = self.MAX_LAUNCH_POWER / 2
        
        self.targets = []
        for _ in range(5):
            self._spawn_target()
            
        self.particles = []
        self.target_spawn_timer = self.base_target_spawn_cooldown
        self.shift_was_pressed = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        if self.game_state == 'AIMING':
            # Adjust launch parameters
            if movement == 1: self.launch_power = min(self.MAX_LAUNCH_POWER, self.launch_power + 0.5)
            if movement == 2: self.launch_power = max(0, self.launch_power - 0.5)
            if movement == 3: self.launch_angle -= 0.1
            if movement == 4: self.launch_angle += 0.1
            
            # Launch
            if space_held:
                # sfx: launch_sound()
                self.game_state = 'LAUNCHED'
                self.player_vel[0] = math.cos(self.launch_angle) * self.launch_power
                self.player_vel[2] = math.sin(self.launch_angle) * self.launch_power
                self.player_vel[1] = self.launch_power * 0.8 # Initial upward velocity
                self.launch_origin = self.player_pos[[0, 2]].copy()

        elif self.game_state == 'LAUNCHED':
            # Transform
            if shift_held and not self.shift_was_pressed:
                # sfx: transform_sound()
                self.player_form = 'SPHERE' if self.player_form == 'CUBE' else 'CUBE'
            self.shift_was_pressed = shift_held

        # --- Game Logic Update ---
        self.steps += 1
        time_delta = 1.0 / self.GAME_FPS
        self.time_remaining -= time_delta

        # Target Spawning
        self.target_spawn_timer -= time_delta
        if self.target_spawn_timer <= 0:
            self._spawn_target()
            # Difficulty scaling: spawn rate increases
            spawn_rate_increase = self.steps / self.MAX_STEPS * 0.9 # Scale from 0 to 0.9
            self.target_spawn_timer = self.base_target_spawn_cooldown * (1 - spawn_rate_increase)

        # Physics and Collisions
        if self.game_state == 'LAUNCHED':
            old_pos_xz = self.player_pos[[0, 2]].copy()
            reward += self._update_physics()
            new_pos_xz = self.player_pos[[0, 2]].copy()
            
            # Continuous proximity reward
            if len(self.targets) > 0:
                distances_old = [np.linalg.norm(old_pos_xz - t[:2]) for t in self.targets]
                distances_new = [np.linalg.norm(new_pos_xz - t[:2]) for t in self.targets]
                if min(distances_new) < min(distances_old):
                    reward += 0.01
                else:
                    reward -= 0.01

        # Particle Update
        self._update_particles()

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            if self.steps >= self.MAX_STEPS:
                truncated = True
                terminated = False
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_physics(self):
        reward = 0
        
        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        
        # Apply friction if on/near floor
        if self.player_pos[1] <= 0.1:
            self.player_vel[[0, 2]] *= self.FRICTION

        # Update position
        self.player_pos += self.player_vel

        # Floor collision
        if self.player_pos[1] < 0:
            self.player_pos[1] = 0
            self.player_vel[1] *= -self.FLOOR_BOUNCINESS
            # sfx: bounce_floor()

        # Wall collisions
        wall_hit = False
        if abs(self.player_pos[0]) > self.WORLD_WIDTH / 2:
            self.player_pos[0] = np.sign(self.player_pos[0]) * self.WORLD_WIDTH / 2
            self.player_vel[0] *= -self.WALL_BOUNCINESS
            wall_hit = True
        if abs(self.player_pos[2]) > self.WORLD_DEPTH / 2:
            self.player_pos[2] = np.sign(self.player_pos[2]) * self.WORLD_DEPTH / 2
            self.player_vel[2] *= -self.WALL_BOUNCINESS
            wall_hit = True
        
        if wall_hit:
            # sfx: bounce_wall()
            reward -= 1

        # Target collisions
        hit_targets = []
        player_radius = 15 if self.player_form == 'SPHERE' else 20
        if self.player_pos[1] < player_radius: # Must be near floor to hit
            for i, target in enumerate(self.targets):
                dist = np.linalg.norm(self.player_pos[[0, 2]] - target[:2])
                if dist < player_radius + target[2]: # target[2] is radius
                    hit_targets.append(i)
        
        for i in sorted(hit_targets, reverse=True):
            # sfx: hit_target()
            target_pos = self.targets.pop(i)
            self._create_particles(target_pos[0], 0, target_pos[1], self.COLOR_TARGET)
            
            # Calculate score and reward based on launch distance
            launch_dist = np.linalg.norm(self.launch_origin - target_pos[:2])
            max_dist = np.linalg.norm([self.WORLD_WIDTH, self.WORLD_DEPTH])
            dist_multiplier = max(0, min(1, launch_dist / max_dist))
            
            base_reward = 1 + dist_multiplier * 9 # Reward from 1 to 10
            reward += base_reward
            self.score += int(base_reward * 10)

        # Check if stopped
        if np.linalg.norm(self.player_vel) < self.STOP_VELOCITY_THRESHOLD and self.player_pos[1] < 1:
            self.game_state = 'AIMING'
            self.player_pos = np.array([0.0, 10.0, 0.0]) # Reset to center
            self.player_vel = np.array([0.0, 0.0, 0.0])
            self.launch_power = self.MAX_LAUNCH_POWER / 2

        return reward

    def _spawn_target(self):
        padding = 20
        x = self.np_random.uniform(-self.WORLD_WIDTH/2 + padding, self.WORLD_WIDTH/2 - padding)
        z = self.np_random.uniform(-self.WORLD_DEPTH/2 + padding, self.WORLD_DEPTH/2 - padding)
        radius = 15
        self.targets.append(np.array([x, z, radius]))

    def _create_particles(self, x, y, z, color):
        for _ in range(20):
            vel = self.np_random.uniform(-2, 2, size=3)
            vel[1] = self.np_random.uniform(1, 4) # Upward burst
            lifespan = self.np_random.uniform(10, 20)
            self.particles.append({
                'pos': np.array([x, y, z], dtype=float),
                'vel': vel,
                'lifespan': lifespan,
                'color': color
            })
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][1] += self.GRAVITY * 0.5 # Particles have less gravity
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _project(self, x, y, z):
        """Converts 3D world coordinates to 2D screen coordinates."""
        iso_x = (x - z) * self.ISO_COS
        iso_y = (x + z) * self.ISO_SIN - y
        return int(iso_x + self.SCREEN_WIDTH / 2), int(iso_y + self.SCREEN_HEIGHT / 2 + 50)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Render Floor and Walls ---
        floor_points_3d = [
            (-self.WORLD_WIDTH/2, 0, -self.WORLD_DEPTH/2),
            ( self.WORLD_WIDTH/2, 0, -self.WORLD_DEPTH/2),
            ( self.WORLD_WIDTH/2, 0,  self.WORLD_DEPTH/2),
            (-self.WORLD_WIDTH/2, 0,  self.WORLD_DEPTH/2),
        ]
        floor_points_2d = [self._project(x,y,z) for x,y,z in floor_points_3d]
        pygame.gfxdraw.filled_polygon(self.screen, floor_points_2d, self.COLOR_FLOOR)
        pygame.gfxdraw.aapolygon(self.screen, floor_points_2d, self.COLOR_WALLS)

        # --- Sort and Render Objects by Depth (z-order) ---
        renderables = []
        # Targets
        for t in self.targets:
            renderables.append({'type': 'target', 'pos': np.array([t[0], 0, t[1]]), 'radius': t[2]})
        # Player
        renderables.append({'type': 'player', 'pos': self.player_pos})
        # Particles
        for p in self.particles:
             renderables.append({'type': 'particle', 'pos': p['pos'], 'color': p['color'], 'lifespan': p['lifespan']})

        # Sort by z-coordinate (approximated by y in screen space) for correct layering
        renderables.sort(key=lambda r: self._project(*r['pos'])[1])

        for item in renderables:
            if item['type'] == 'target':
                sx, sy = self._project(item['pos'][0], item['pos'][1], item['pos'][2])
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, int(item['radius']), self.COLOR_TARGET)
                pygame.gfxdraw.aacircle(self.screen, sx, sy, int(item['radius']), tuple(c*0.8 for c in self.COLOR_TARGET))
            elif item['type'] == 'player':
                self._render_player()
            elif item['type'] == 'particle':
                sx, sy = self._project(item['pos'][0], item['pos'][1], item['pos'][2])
                size = int(max(1, item['lifespan'] / 5))
                pygame.draw.rect(self.screen, item['color'], (sx, sy, size, size))

    def _render_player(self):
        x, y, z = self.player_pos
        sx, sy = self._project(x, y, z)
        shadow_x, shadow_y = self._project(x, 0, z)
        
        # Shadow
        shadow_size = int(max(5, 20 - y / 10))
        shadow_surface = pygame.Surface((shadow_size * 2, shadow_size * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, self.COLOR_SHADOW, (0, 0, shadow_size * 2, shadow_size * 2))
        self.screen.blit(shadow_surface, (shadow_x - shadow_size, shadow_y - shadow_size))

        # Player object
        if self.player_form == 'CUBE':
            size = 15
            color = self.COLOR_CUBE
            glow_color = self.COLOR_CUBE_GLOW
            self._draw_iso_cube(sx, sy, size, color, glow_color)
        else: # SPHERE
            size = 15
            color = self.COLOR_SPHERE
            glow_color = self.COLOR_SPHERE_GLOW
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, int(size * 1.5), glow_color)
            # Main sphere
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, size, color)
            pygame.gfxdraw.aacircle(self.screen, sx, sy, size, tuple(c*0.8 for c in color))

    def _draw_iso_cube(self, sx, sy, size, color, glow_color):
        # Pre-calculate points relative to center (sx, sy)
        top_points = [
            (sx, sy - size),
            (sx + size * self.ISO_COS * 1.5, sy - size + size * self.ISO_SIN * 1.5),
            (sx, sy - size + 2 * size * self.ISO_SIN * 1.5),
            (sx - size * self.ISO_COS * 1.5, sy - size + size * self.ISO_SIN * 1.5),
        ]
        
        # Glow effect
        glow_surface = pygame.Surface((size*4, size*4), pygame.SRCALPHA)
        glow_points = [(p[0] - sx + size*2, p[1] - sy + size*2) for p in top_points]
        pygame.gfxdraw.filled_polygon(glow_surface, glow_points, glow_color)
        self.screen.blit(glow_surface, (sx - size*2, sy - size*2))

        # Draw faces
        c_dark = tuple(int(c * 0.6) for c in color)
        c_med = tuple(int(c * 0.8) for c in color)

        # Right face
        right_face = [top_points[1], top_points[2], (top_points[2][0], top_points[2][1] + size), (top_points[1][0], top_points[1][1] + size)]
        pygame.gfxdraw.filled_polygon(self.screen, right_face, c_med)
        
        # Left face
        left_face = [top_points[3], top_points[2], (top_points[2][0], top_points[2][1] + size), (top_points[3][0], top_points[3][1] + size)]
        pygame.gfxdraw.filled_polygon(self.screen, left_face, c_dark)
        
        # Top face
        pygame.gfxdraw.filled_polygon(self.screen, top_points, color)

        # Outlines for definition
        pygame.gfxdraw.aapolygon(self.screen, top_points, c_dark)
        pygame.gfxdraw.aapolygon(self.screen, right_face, c_dark)
        pygame.gfxdraw.aapolygon(self.screen, left_face, c_dark)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        timer_text = self.font_large.render(f"TIME: {max(0, math.ceil(self.time_remaining))}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Aiming UI
        if self.game_state == 'AIMING':
            # Power bar
            bar_w, bar_h = 150, 20
            bar_x, bar_y = self.SCREEN_WIDTH / 2 - bar_w / 2, self.SCREEN_HEIGHT - 40
            power_ratio = self.launch_power / self.MAX_LAUNCH_POWER
            fill_w = int(bar_w * power_ratio)
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_FILL, (bar_x, bar_y, fill_w, bar_h))
            power_text = self.font_small.render("POWER", True, self.COLOR_UI_TEXT)
            self.screen.blit(power_text, (bar_x + bar_w/2 - power_text.get_width()/2, bar_y + bar_h))
            
            # Aiming indicator
            sx, sy = self._project(*self.player_pos)
            end_x = sx + math.cos(self.launch_angle) * (20 + self.launch_power * 2)
            end_y = sy + math.sin(self.launch_angle) * (20 + self.launch_power * 2) * self.ISO_SIN * 2
            pygame.draw.aaline(self.screen, self.COLOR_SPHERE, (sx, sy), (end_x, end_y), 2)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "player_form": self.player_form,
            "game_state": self.game_state,
        }

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # This block is for human play and is not used by the evaluation system.
    # It will not be run, so it does not need to be modified.
    # To run this, you will need to `pip install pygame`.
    
    # We need to unset the dummy video driver to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play Loop ---
    running = True
    pygame.display.set_caption("Bounce & Transform")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    total_reward = 0
    terminated = False
    truncated = False

    while running:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

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
        
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Reset after a delay
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            terminated = False
            truncated = False

        clock.tick(GameEnv.GAME_FPS)
        
    env.close()