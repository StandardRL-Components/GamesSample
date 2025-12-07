import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:23:53.102233
# Source Brief: brief_01669.md
# Brief Index: 1669
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Acorn Descent: A fast-paced arcade racing game where a sentient acorn
    races down a procedurally generated mountain. The player must use
    gravity flipping and teleportation to avoid obstacles, collect nutrients,
    and reach the finish line before time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Race a sentient acorn down a procedurally generated mountain. Flip gravity and teleport to dodge obstacles, "
        "collect nutrients, and reach the finish line before time runs out."
    )
    user_guide = (
        "Controls: Use ←→ arrows to move left and right. Press space to flip gravity and shift to teleport."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Assumed FPS for smooth interpolation

        # Colors
        self.COLOR_BG = (15, 20, 40)
        self.COLOR_ACORN = (50, 150, 255)
        self.COLOR_ACORN_GLOW = (150, 200, 255)
        self.COLOR_OBSTACLE = (100, 80, 70)
        self.COLOR_NUTRIENT = (100, 255, 100)
        self.COLOR_FINISH = (255, 255, 255)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_UI_SHADOW = (20, 20, 20)

        # Game Mechanics
        self.MOUNTAIN_LENGTH = 20000
        self.MAX_STEPS = 5000
        self.PLAYER_ACCEL = 1.0
        self.PLAYER_FRICTION = 0.90
        self.GRAVITY = 0.5
        self.PLAYER_MAX_VX = 8.0
        self.PLAYER_MAX_VY = 12.0
        self.PLAYER_RADIUS = 12
        self.TELEPORT_BASE_DIST = 150
        self.TELEPORT_NUTRIENT_BONUS = 15
        self.TELEPORT_COOLDOWN_MAX = 60  # 2 seconds
        self.GRAVITY_FLIP_COOLDOWN_MAX = 30 # 1 second
        self.GRAVITY_FLIP_NUTRIENT_BONUS = 1 # frames

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 20)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 26)


        # --- Persistent State ---
        self.best_time_steps = float('inf')

        # Initialize state variables by calling reset
        self.reset()
        
        # --- Critical Self-Check ---
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Episode State
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS

        # Player State
        self.player_pos = np.array([self.WIDTH / 2.0, 100.0])
        self.player_vel = np.array([0.0, 0.0])
        self.gravity_direction = 1

        # Game World State
        self.nutrient_count = 0
        self.teleport_cooldown = 0
        self.gravity_flip_cooldown = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.camera_y = 0.0
        
        self.obstacles = []
        self.nutrients = []
        self.particles = []
        self.bg_layers = self._generate_background_layers()
        
        self.obstacle_density = 0.02
        self.last_generated_y = 0
        self._generate_world_chunk(0, self.HEIGHT * 2)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        prev_y_pos = self.player_pos[1]

        self._handle_input(action)
        self._update_player_state()
        self._update_world_state()
        
        reward = self._handle_collisions_and_events()
        
        # Reward for downward progress
        y_progress_delta = self.player_pos[1] - prev_y_pos
        reward += y_progress_delta * 0.01

        self.score += reward
        self.steps += 1
        self.time_remaining -= 1

        terminated, term_reward = self._check_termination()
        self.score += term_reward
        reward += term_reward
        if terminated:
            self.game_over = True
            if self.player_pos[1] >= self.MOUNTAIN_LENGTH:
                # Update best time only on successful completion
                self.best_time_steps = min(self.best_time_steps, self.steps)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL

        # Gravity Flip (on press)
        if space_held and not self.last_space_held and self.gravity_flip_cooldown <= 0:
            self.gravity_direction *= -1
            flip_cooldown_reduction = self.nutrient_count * self.GRAVITY_FLIP_NUTRIENT_BONUS
            self.gravity_flip_cooldown = max(5, self.GRAVITY_FLIP_COOLDOWN_MAX - flip_cooldown_reduction)
            # SFX: whoosh_gravity.wav
            self._create_particles(self.player_pos, 20, (200, 200, 255), 2.0, 360)

        # Teleport (on press)
        if shift_held and not self.last_shift_held and self.teleport_cooldown <= 0 and self.nutrient_count > 0:
            teleport_dist = self.TELEPORT_BASE_DIST + self.nutrient_count * self.TELEPORT_NUTRIENT_BONUS
            start_pos = self.player_pos.copy()
            self.player_pos[1] += teleport_dist * self.gravity_direction
            self.player_vel[1] = 0 # Stop momentum after teleport
            self.teleport_cooldown = self.TELEPORT_COOLDOWN_MAX
            self.nutrient_count -= 1
            # SFX: warp.wav
            self._create_particles(start_pos, 30, (255, 255, 100), 3.0, 180)
            self._create_particles(self.player_pos, 30, (100, 255, 255), 3.0, 180)


        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_player_state(self):
        # Apply gravity
        self.player_vel[1] += self.GRAVITY * self.gravity_direction
        
        # Apply friction
        self.player_vel[0] *= self.PLAYER_FRICTION

        # Clamp velocity
        self.player_vel[0] = np.clip(self.player_vel[0], -self.PLAYER_MAX_VX, self.PLAYER_MAX_VX)
        self.player_vel[1] = np.clip(self.player_vel[1], -self.PLAYER_MAX_VY, self.PLAYER_MAX_VY)

        # Update position
        self.player_pos += self.player_vel

        # Screen boundaries (horizontal)
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        
        # Cooldowns
        if self.teleport_cooldown > 0:
            self.teleport_cooldown -= 1
        if self.gravity_flip_cooldown > 0:
            self.gravity_flip_cooldown -= 1

    def _update_world_state(self):
        # Update camera to follow player
        self.camera_y = self.player_pos[1] - self.HEIGHT / 2

        # Procedurally generate new world chunks
        if self.camera_y + self.HEIGHT > self.last_generated_y - self.HEIGHT:
            self._generate_world_chunk(self.last_generated_y, self.last_generated_y + self.HEIGHT)
            self.last_generated_y += self.HEIGHT
            
            # Increase difficulty over time
            self.obstacle_density = min(0.1, self.obstacle_density + 0.005)

        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            
        # Clean up off-screen objects
        view_top = self.camera_y - 100 # Buffer
        self.obstacles = [o for o in self.obstacles if o['pos'][1] + o['radius'] > view_top]
        self.nutrients = [n for n in self.nutrients if n['pos'][1] + n['radius'] > view_top]

    def _handle_collisions_and_events(self):
        reward = 0
        
        # Player vs Nutrients
        for nutrient in self.nutrients[:]:
            dist = np.linalg.norm(self.player_pos - nutrient['pos'])
            if dist < self.PLAYER_RADIUS + nutrient['radius']:
                self.nutrients.remove(nutrient)
                self.nutrient_count += 1
                reward += 1.0
                # SFX: collect_nutrient.wav
                self._create_particles(nutrient['pos'], 10, self.COLOR_NUTRIENT, 1.5)
        
        # Player vs Obstacles
        for obstacle in self.obstacles:
            dist = np.linalg.norm(self.player_pos - obstacle['pos'])
            if dist < self.PLAYER_RADIUS + obstacle['radius']:
                self.game_over = True
                reward = -100.0
                # SFX: crash.wav
                self._create_particles(self.player_pos, 50, (255, 50, 50), 5.0)
                break
        return reward

    def _check_termination(self):
        terminated = False
        reward = 0

        if self.game_over: # From obstacle collision
            return True, 0

        if self.player_pos[1] >= self.MOUNTAIN_LENGTH:
            terminated = True
            time_bonus = (self.time_remaining / self.MAX_STEPS) * 50
            reward = 100 + time_bonus
            # SFX: victory.wav
        elif self.time_remaining <= 0:
            terminated = True
            reward = -50.0
            # SFX: time_up.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward = -50.0 # Same penalty as time out
            
        return terminated, reward
    
    def _generate_world_chunk(self, start_y, end_y):
        y = start_y
        while y < end_y:
            # Generate a row of potential objects
            row_y = y + self.np_random.uniform(50, 150)
            if row_y > self.MOUNTAIN_LENGTH - 200: # Don't spawn near finish
                break

            if self.np_random.random() < self.obstacle_density:
                num_obstacles = self.np_random.integers(1, 4)
                for _ in range(num_obstacles):
                    pos = np.array([self.np_random.uniform(50, self.WIDTH - 50), row_y])
                    radius = self.np_random.uniform(20, 50)
                    self.obstacles.append({'pos': pos, 'radius': radius})

            if self.np_random.random() < 0.1: # Nutrient spawn chance
                pos = np.array([self.np_random.uniform(50, self.WIDTH - 50), row_y + self.np_random.uniform(-30, 30)])
                self.nutrients.append({'pos': pos, 'radius': 8})
            y = row_y

    def _generate_background_layers(self):
        layers = []
        for i in range(4):
            depth = (i + 1) * 0.25
            color = tuple(c * (1 - depth * 0.8) for c in self.COLOR_OBSTACLE)
            layer = {'points': [], 'depth': depth, 'color': color}
            y = -200
            while y < self.MOUNTAIN_LENGTH + self.HEIGHT:
                num_peaks = self.np_random.integers(2, 5)
                for j in range(num_peaks):
                    base_y = y + self.np_random.uniform(0, 200)
                    height = self.np_random.uniform(100, 400)
                    width = self.np_random.uniform(200, 500)
                    x_pos = self.np_random.uniform(-width/2, self.WIDTH + width/2)
                    
                    if self.np_random.random() < 0.5: # Left side
                        p1 = (x_pos - width/2, base_y + height)
                        p2 = (x_pos, base_y)
                        p3 = (x_pos + width/2, base_y + height)
                    else: # Right side
                        p1 = (x_pos - width/2, base_y + height)
                        p2 = (x_pos, base_y)
                        p3 = (x_pos + width/2, base_y + height)
                    layer['points'].append((p1, p2, p3))
                y += 400
            layers.append(layer)
        return layers

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_world()
        self._render_player()
        self._render_particles()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for layer in self.bg_layers:
            scroll_y = self.camera_y * layer['depth']
            for p1, p2, p3 in layer['points']:
                points_on_screen = [
                    (int(p1[0]), int(p1[1] - scroll_y)),
                    (int(p2[0]), int(p2[1] - scroll_y)),
                    (int(p3[0]), int(p3[1] - scroll_y))
                ]
                pygame.gfxdraw.aapolygon(self.screen, points_on_screen, layer['color'])
                pygame.gfxdraw.filled_polygon(self.screen, points_on_screen, layer['color'])

    def _render_game_world(self):
        # Render Finish Line
        finish_y_on_screen = int(self.MOUNTAIN_LENGTH - self.camera_y)
        if 0 < finish_y_on_screen < self.HEIGHT:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (0, finish_y_on_screen), (self.WIDTH, finish_y_on_screen), 5)

        # Render Obstacles
        for o in self.obstacles:
            pos = (int(o['pos'][0]), int(o['pos'][1] - self.camera_y))
            radius = int(o['radius'])
            if -radius < pos[1] < self.HEIGHT + radius:
                 pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_OBSTACLE)
                 pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_OBSTACLE)

        # Render Nutrients
        for n in self.nutrients:
            pos = (int(n['pos'][0]), int(n['pos'][1] - self.camera_y))
            radius = int(n['radius'])
            if -radius < pos[1] < self.HEIGHT + radius:
                pulse = int(abs(math.sin(self.steps * 0.1)) * 3)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + pulse, self.COLOR_NUTRIENT)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + pulse, self.COLOR_NUTRIENT)

    def _render_player(self):
        if self.game_over: return
        pos = (int(self.player_pos[0]), int(self.player_pos[1] - self.camera_y))
        
        # Glow effect
        glow_radius = self.PLAYER_RADIUS + 8 + int(abs(math.sin(self.steps * 0.2)) * 4)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*self.COLOR_ACORN_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

        # Acorn body
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_ACORN)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_ACORN)
        
        # Gravity indicator
        indicator_y = pos[1] + (self.PLAYER_RADIUS + 3) * self.gravity_direction
        pygame.draw.line(self.screen, self.COLOR_ACORN_GLOW, pos, (pos[0], indicator_y), 2)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1] - self.camera_y))
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            if alpha > 0:
                temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(temp_surf, (pos[0] - p['size'], pos[1] - p['size']))

    def _render_ui(self):
        def draw_text(text, font, color, pos, shadow=True):
            if shadow:
                text_surf_s = font.render(text, True, self.COLOR_UI_SHADOW)
                self.screen.blit(text_surf_s, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Top-left: Nutrients
        nutrient_text = f"Nutrients: {self.nutrient_count}"
        draw_text(nutrient_text, self.font_small, self.COLOR_NUTRIENT, (10, 10))

        # Top-right: Time
        time_sec = self.time_remaining / self.FPS
        time_text = f"Time: {time_sec:.1f}s"
        draw_text(time_text, self.font_small, self.COLOR_UI_TEXT, (self.WIDTH - 180, 10))

        # Top-right: Best Time
        if self.best_time_steps != float('inf'):
            best_sec = self.best_time_steps / self.FPS
            best_text = f"Best: {best_sec:.1f}s"
        else:
            best_text = "Best: N/A"
        draw_text(best_text, self.font_small, self.COLOR_UI_TEXT, (self.WIDTH - 180, 35))
        
        # Center message on game over
        if self.game_over:
            if self.player_pos[1] >= self.MOUNTAIN_LENGTH:
                msg = "FINISH!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            shadow_surf = self.font_large.render(msg, True, self.COLOR_UI_SHADOW)
            self.screen.blit(shadow_surf, text_rect.move(3,3))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "nutrients": self.nutrient_count,
            "progress": self.player_pos[1] / self.MOUNTAIN_LENGTH,
            "time_remaining": self.time_remaining
        }

    def _create_particles(self, pos, count, color, max_speed, spread_angle=360):
        for _ in range(count):
            angle = self.np_random.uniform(0, spread_angle)
            rad = math.radians(angle - 90) # Adjust for pygame coordinates
            speed = self.np_random.uniform(1.0, max_speed)
            vel = np.array([math.cos(rad) * speed, math.sin(rad) * speed])
            lifespan = self.np_random.integers(15, 40)
            self.particles.append({
                'pos': pos.copy() + self.np_random.uniform(-5, 5, 2),
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.display.init()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Acorn Descent")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # The main loop in __main__ was not runnable as-is because the dummy driver
    # was still set. I've added a line to unset it for local play.
    # The original code would have crashed here.
    
    while not done:
        # --- Manual Control Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        # --- Rendering ---
        # The observation is already the rendered image
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()