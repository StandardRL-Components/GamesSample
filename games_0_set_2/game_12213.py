import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:16:01.611621
# Source Brief: brief_02213.md
# Brief Index: 2213
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    def __init__(self, pos, vel, lifespan, color, radius_start, radius_end):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color
        self.radius_start = radius_start
        self.radius_end = radius_end

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.vel *= 0.98  # Damping

    def draw(self, surface):
        if self.lifespan > 0:
            life_ratio = self.lifespan / self.max_lifespan
            current_radius = int(self.radius_start * life_ratio + self.radius_end * (1 - life_ratio))
            alpha = int(255 * life_ratio)
            color = self.color + (alpha,)
            
            temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (current_radius, current_radius), current_radius)
            surface.blit(temp_surf, (int(self.pos.x - current_radius), int(self.pos.y - current_radius)), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Launch a projectile through a complex gravity field to hit all targets. "
        "Manipulate time and aim carefully to navigate past obstacles and through portals."
    )
    user_guide = (
        "Controls: ↑↓ to aim, ←→ to control time dilation. "
        "Press space to launch and shift to reset your aim."
    )
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2500

        # Colors
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 200, 0)
        self.COLOR_TARGET = (0, 255, 150)
        self.COLOR_TARGET_GLOW = (0, 200, 120)
        self.COLOR_LAUNCHER = (200, 200, 220)
        self.COLOR_PREVIEW = (255, 255, 255, 100)
        self.COLOR_UI_TEXT = (220, 220, 240)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_level = pygame.font.SysFont("Consolas", 24, bold=True)

        # State variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.launches_left = 0
        self.launcher_pos = pygame.math.Vector2(60, self.HEIGHT // 2)
        self.launcher_angle = 0.0
        self.time_dilation = 1.0
        self.object_state = 'IDLE' # 'IDLE', 'IN_FLIGHT'
        self.object_pos = pygame.math.Vector2(0, 0)
        self.object_vel = pygame.math.Vector2(0, 0)
        self.object_radius = 8
        self.targets = []
        self.fractal_lines = []
        self.portals = []
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.reward_this_step = 0.0

        # self.reset() is called by the wrapper, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0.0
        
        if options and 'level' in options:
            self.level = options['level']
        else:
            self.level = 1

        self._generate_level()
        
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def _generate_level(self):
        self.launches_left = 3 + self.level // 2
        self.launcher_angle = -45.0
        self.time_dilation = 1.0
        self.object_state = 'IDLE'
        self.particles.clear()
        
        # Generate fractal gravity lines
        self.fractal_lines.clear()
        num_lines = min(10, 2 + self.level)
        for _ in range(num_lines):
            p1 = pygame.math.Vector2(
                self.np_random.uniform(self.WIDTH * 0.2, self.WIDTH * 0.9),
                self.np_random.uniform(self.HEIGHT * 0.1, self.HEIGHT * 0.9)
            )
            angle = self.np_random.uniform(0, 2 * math.pi)
            length = self.np_random.uniform(50, 150)
            p2 = p1 + pygame.math.Vector2(math.cos(angle), math.sin(angle)) * length
            strength = self.np_random.uniform(100, 300 + self.level * 20) # Gravity strength
            self.fractal_lines.append({'p1': p1, 'p2': p2, 'strength': strength})

        # Generate targets
        self.targets.clear()
        num_targets = 1 + self.level // 3
        for _ in range(num_targets):
            while True:
                pos = pygame.math.Vector2(
                    self.np_random.uniform(self.WIDTH * 0.4, self.WIDTH * 0.9),
                    self.np_random.uniform(self.HEIGHT * 0.1, self.HEIGHT * 0.9)
                )
                if pos.distance_to(self.launcher_pos) > 150:
                    self.targets.append(pos)
                    break
        
        # Generate portals (from level 2 onwards)
        self.portals.clear()
        if self.level >= 2:
            p1 = pygame.math.Vector2(
                self.np_random.uniform(self.WIDTH * 0.2, self.WIDTH * 0.8),
                self.np_random.uniform(self.HEIGHT * 0.2, self.HEIGHT * 0.8)
            )
            p2 = pygame.math.Vector2(
                self.np_random.uniform(self.WIDTH * 0.2, self.WIDTH * 0.8),
                self.np_random.uniform(self.HEIGHT * 0.2, self.HEIGHT * 0.8)
            )
            self.portals = [{'pos': p1, 'linked_pos': p2}, {'pos': p2, 'linked_pos': p1}]

    def step(self, action):
        self.reward_this_step = 0.0
        self.game_over = False
        
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        if self.object_state == 'IDLE':
            self._handle_aiming_input(movement, space_pressed, shift_pressed)
        elif self.object_state == 'IN_FLIGHT':
            self._update_projectile()
        
        self._update_particles()
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        self.steps += 1
        
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_aiming_input(self, movement, space_pressed, shift_pressed):
        # Adjust angle
        if movement == 1: self.launcher_angle -= 1.0
        if movement == 2: self.launcher_angle += 1.0
        self.launcher_angle = max(-90, min(90, self.launcher_angle))
        
        # Adjust time dilation
        if movement == 3: self.time_dilation -= 0.05
        if movement == 4: self.time_dilation += 0.05
        self.time_dilation = max(0.1, min(2.0, self.time_dilation))
        
        # Reset aim
        if shift_pressed:
            self.launcher_angle = -45.0
            self.time_dilation = 1.0

        # Launch
        if space_pressed and self.launches_left > 0:
            self.launches_left -= 1
            self.object_state = 'IN_FLIGHT'
            self.object_pos = self.launcher_pos.copy()
            launch_angle_rad = math.radians(self.launcher_angle)
            initial_speed = 15.0
            self.object_vel = pygame.math.Vector2(math.cos(launch_angle_rad), math.sin(launch_angle_rad)) * initial_speed
            # sfx: launch_sound

    def _update_projectile(self):
        # Simulate with time dilation
        dt = self.time_dilation
        
        # Calculate gravity from fractal lines
        total_force = pygame.math.Vector2(0, 0)
        for line in self.fractal_lines:
            force = self._get_force_from_line(line, self.object_pos)
            total_force += force
        
        self.object_vel += total_force * (dt / self.FPS)
        self.object_pos += self.object_vel * (dt / self.FPS)
        
        # Damping
        self.object_vel *= 0.999
        
        # Check collisions
        self._check_collisions()

        # Check if stopped or out of bounds
        is_out_of_bounds = not (0 < self.object_pos.x < self.WIDTH and 0 < self.object_pos.y < self.HEIGHT)
        is_stopped = self.object_vel.length() < 0.1
        
        if is_out_of_bounds or is_stopped:
            self.object_state = 'IDLE'
            if is_out_of_bounds:
                # sfx: fizzle_sound
                self._create_sparks(self.object_pos, 20, (200, 200, 200))

    def _check_collisions(self):
        # Targets
        for target_pos in self.targets[:]:
            if self.object_pos.distance_to(target_pos) < self.object_radius + 10:
                self.targets.remove(target_pos)
                self.score += 1
                self.reward_this_step += 5.0
                self._create_sparks(target_pos, 50, self.COLOR_TARGET)
                # sfx: target_hit_sound
                if not self.targets:
                    self.reward_this_step += 50.0 # Level complete bonus
                    self.game_over = True

        # Portals
        for portal in self.portals:
            if self.object_pos.distance_to(portal['pos']) < 15:
                self.object_pos = portal['linked_pos'].copy()
                self.object_vel = self.object_vel.rotate(self.np_random.uniform(-10, 10)) # Slight random rotation on exit
                self._create_sparks(portal['pos'], 30, (150, 0, 255))
                self._create_sparks(portal['linked_pos'], 30, (150, 0, 255))
                # sfx: portal_whoosh_sound
                break # Prevent teleport loops in one frame

    def _get_force_from_line(self, line, pos):
        p1, p2 = line['p1'], line['p2']
        line_vec = p2 - p1
        if line_vec.length() == 0: return pygame.math.Vector2(0,0)
        
        point_vec = pos - p1
        t = line_vec.dot(point_vec) / line_vec.length_squared()
        t = max(0, min(1, t))
        
        closest_point = p1 + t * line_vec
        dist_vec = pos - closest_point
        dist = dist_vec.length()
        
        if dist < 1: return pygame.math.Vector2(0,0)
        
        force_mag = line['strength'] / (dist * dist + 10)
        force_vec = dist_vec.normalize() * force_mag
        
        # Force is perpendicular to the line
        normal = line_vec.rotate(90).normalize()
        return normal * normal.dot(force_vec)

    def _calculate_reward(self):
        # Proximity reward
        if self.object_state == 'IN_FLIGHT' and self.targets:
            min_dist = min(self.object_pos.distance_to(t) for t in self.targets)
            self.reward_this_step += 0.1 * (1 - min(1, min_dist / self.WIDTH))
        
        if self._check_termination():
            if self.launches_left <= 0 and self.targets:
                self.reward_this_step -= 50.0
        
        return self.reward_this_step

    def _check_termination(self):
        if self.game_over: return True
        if self.launches_left <= 0 and self.object_state == 'IDLE' and self.targets:
            return True
        if not self.targets:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_fractal_field()
        self._render_portals()
        self._render_targets()
        self._render_launcher()

        if self.object_state == 'IDLE':
            self._render_trajectory_preview()
        elif self.object_state == 'IN_FLIGHT':
            self._render_object()
            
        self._render_particles()

    def _render_fractal_field(self):
        for line in self.fractal_lines:
            strength_ratio = min(1, (line['strength'] - 100) / 400)
            color = (
                int(60 + 195 * strength_ratio),
                60,
                int(255 - 195 * strength_ratio)
            )
            pygame.draw.aaline(self.screen, color, line['p1'], line['p2'], 1)

    def _render_portals(self):
        for portal in self.portals:
            radius = 15
            angle = (self.steps * 3) % 360
            for i in range(5):
                offset_angle = math.radians(angle + i * 72)
                x_off = math.cos(offset_angle) * radius * 0.5
                y_off = math.sin(offset_angle) * radius * 0.5
                color_phase = (self.steps + i * 20) % 255
                color = (150 + int(105 * math.sin(color_phase * 0.1)), 50, 255 - int(105 * math.sin(color_phase * 0.1)))
                pygame.gfxdraw.aacircle(self.screen, int(portal['pos'].x + x_off), int(portal['pos'].y + y_off), 5, color)

    def _render_targets(self):
        for pos in self.targets:
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 12, self.COLOR_TARGET_GLOW + (50,))
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 10, self.COLOR_TARGET)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 10, self.COLOR_TARGET)

    def _render_launcher(self):
        angle_rad = math.radians(self.launcher_angle)
        end_pos = self.launcher_pos + pygame.math.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * 30
        pygame.draw.aaline(self.screen, self.COLOR_LAUNCHER, self.launcher_pos, end_pos, 3)
        p1 = self.launcher_pos + pygame.math.Vector2(0, 10)
        p2 = self.launcher_pos + pygame.math.Vector2(0, -10)
        p3 = self.launcher_pos - pygame.math.Vector2(10, 0)
        pygame.gfxdraw.aapolygon(self.screen, [p1,p2,p3], self.COLOR_LAUNCHER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1,p2,p3], self.COLOR_LAUNCHER)

    def _render_object(self):
        # Glow
        glow_radius = int(self.object_radius * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW + (80,), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(self.object_pos.x - glow_radius), int(self.object_pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Main object
        pygame.gfxdraw.filled_circle(self.screen, int(self.object_pos.x), int(self.object_pos.y), self.object_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(self.object_pos.x), int(self.object_pos.y), self.object_radius, self.COLOR_PLAYER)

    def _render_trajectory_preview(self):
        if self.launches_left > 0:
            sim_pos = self.launcher_pos.copy()
            launch_angle_rad = math.radians(self.launcher_angle)
            sim_vel = pygame.math.Vector2(math.cos(launch_angle_rad), math.sin(launch_angle_rad)) * 15.0
            dt = self.time_dilation

            points = []
            for i in range(150):
                total_force = pygame.math.Vector2(0, 0)
                for line in self.fractal_lines:
                    force = self._get_force_from_line(line, sim_pos)
                    total_force += force
                
                sim_vel += total_force * (dt / self.FPS)
                sim_pos += sim_vel * (dt / self.FPS)
                sim_vel *= 0.999

                if i % 5 == 0:
                    points.append((int(sim_pos.x), int(sim_pos.y)))
                
                if not (0 < sim_pos.x < self.WIDTH and 0 < sim_pos.y < self.HEIGHT):
                    break

            if len(points) > 1:
                for i in range(len(points) - 1):
                    p1 = points[i]
                    p2 = points[i+1]
                    alpha = 150 * (1 - i / len(points))
                    pygame.draw.line(self.screen, (255, 255, 255, alpha), p1, p2, 1)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        level_text = self.font_level.render(f"Level: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.WIDTH // 2 - level_text.get_width() // 2, 10))
        
        launches_text = self.font_main.render(f"Launches: {self.launches_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(launches_text, (10, 10))

        dilation_text = self.font_main.render(f"Time Dilation: {self.time_dilation:.2f}x", True, self.COLOR_UI_TEXT)
        self.screen.blit(dilation_text, (10, self.HEIGHT - 30))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _create_sparks(self, position, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 30)
            self.particles.append(Particle(position, vel, lifespan, color, 5, 0))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "launches_left": self.launches_left,
            "targets_left": len(self.targets),
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # The original code had a `validate_implementation` method that was called
    # in __init__. This is not standard practice for Gymnasium environments
    # and can cause issues with environment creation wrappers. It has been removed
    # from the __init__ method. For local testing, you can still call it manually.
    
    # --- Pygame setup for human play ---
    # This part of the script is for human interaction and will not run in
    # the headless evaluation environment.
    pygame.display.init()
    pygame.font.init()
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fractal Gravity Field")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        movement = 0 # none
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        if keys[pygame.K_r]:
            obs, info = env.reset()
            continue
        
        # Construct the action
        action = [movement, space_held, shift_held]
        
        obs, reward, term, trunc, info = env.step(action)
        
        if term or trunc:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
            if not info['targets_left']:
                print("Level Complete! Advancing to next level.")
                obs, info = env.reset(options={'level': info['level'] + 1})
            else:
                print("Game Over. Resetting level.")
                obs, info = env.reset(options={'level': info['level']})

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()
    pygame.quit()