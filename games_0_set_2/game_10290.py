import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame.gfxdraw
import math

# Helper classes for game entities
class Target:
    """Represents a single target entity in the game."""
    def __init__(self, pos, radius, pattern='stationary', pattern_params=None):
        self.pos = pygame.Vector2(pos)
        self.initial_pos = pygame.Vector2(pos)
        self.radius = radius
        self.alive = True
        self.pattern = pattern
        self.pattern_params = pattern_params or {}
        self.age = 0
        self.pulse_speed = 0.05
        self.pulse_amplitude = 3

    def update(self, dt):
        """Updates the target's position based on its movement pattern."""
        self.age += dt
        if self.pattern == 'circular':
            radius = self.pattern_params.get('radius', 50)
            speed = self.pattern_params.get('speed', 1)
            self.pos.x = self.initial_pos.x + math.cos(self.age * speed) * radius
            self.pos.y = self.initial_pos.y + math.sin(self.age * speed) * radius
        elif self.pattern == 'figure_eight':
            radius_x = self.pattern_params.get('radius_x', 60)
            radius_y = self.pattern_params.get('radius_y', 30)
            speed = self.pattern_params.get('speed', 1)
            self.pos.x = self.initial_pos.x + math.sin(self.age * speed) * radius_x
            self.pos.y = self.initial_pos.y + math.sin(self.age * speed * 2) * radius_y

    def draw(self, surface):
        """Renders the target with a pulsating effect."""
        if not self.alive:
            return
        
        pulse = math.sin(self.age * self.pulse_speed * 20) * self.pulse_amplitude
        current_radius = int(self.radius + pulse)
        if current_radius <= 0: return
        
        pos_int = (int(self.pos.x), int(self.pos.y))
        
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], current_radius, (255, 80, 80))
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], current_radius, (255, 80, 80))
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(current_radius * 0.7), (255, 150, 150))

class Particle:
    """Represents a single particle for explosion effects."""
    def __init__(self, pos, vel, size, lifetime, color):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.size = size
        self.lifetime = lifetime
        self.initial_lifetime = lifetime
        self.color = color

    def update(self, dt):
        """Updates the particle's position and lifetime."""
        self.pos += self.vel * dt * 60 # Frame-rate independent velocity
        self.vel *= 0.95 # Damping
        self.lifetime -= dt

    def draw(self, surface):
        """Renders the particle with a fade-out effect."""
        if self.lifetime <= 0:
            return
        
        alpha = int(255 * (self.lifetime / self.initial_lifetime))
        current_size = int(self.size * (self.lifetime / self.initial_lifetime))
        if current_size <= 0: return

        temp_surf = pygame.Surface((current_size * 2, current_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.color + (alpha,), (current_size, current_size), current_size)
        surface.blit(temp_surf, (int(self.pos.x - current_size), int(self.pos.y - current_size)), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Use a laser to destroy all targets on screen. Create chain reactions for bonus points before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the laser origin. Align the laser beam with targets to destroy them."
    )
    auto_advance = True
    
    WIDTH, HEIGHT = 640, 400
    TARGET_FPS = 60
    MAX_TIME = 60.0
    
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 50)
    COLOR_LASER_BEAM = (100, 200, 255)
    COLOR_LASER_ORIGIN = (200, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CHAIN_BAR = (50, 255, 50)
    COLOR_CHAIN_BAR_BG = (50, 50, 50)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.level = 1
        
        self.laser_pos = pygame.Vector2(0, 0)
        self.targets = []
        self.particles = []
        self.timer = 0.0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.chain_progress = 0
        self.chain_cooldown = 0.0
        self.screen_shake = 0

    def _spawn_targets(self):
        self.targets.clear()
        
        if self.level == 1:
            positions = [
                (self.WIDTH * 0.25, self.HEIGHT * 0.5),
                (self.WIDTH * 0.5, self.HEIGHT * 0.5),
                (self.WIDTH * 0.75, self.HEIGHT * 0.5)
            ]
            for pos in positions:
                self.targets.append(Target(pos, 15))
        else:
            num_stationary = 3
            num_moving = 2 * (self.level - 1)
            
            for i in range(num_stationary):
                x = self.WIDTH * (0.2 + 0.6 * (i / max(1, num_stationary - 1)))
                y = self.np_random.uniform(0.3, 0.7) * self.HEIGHT
                self.targets.append(Target((x, y), 15))
            
            for i in range(num_moving):
                start_x = self.np_random.uniform(self.WIDTH * 0.1, self.WIDTH * 0.9)
                start_y = self.np_random.uniform(self.HEIGHT * 0.2, self.HEIGHT * 0.8)
                pattern_choice = self.np_random.choice(['circular', 'figure_eight'])
                if pattern_choice == 'circular':
                    params = {'radius': self.np_random.uniform(40, 80), 'speed': self.np_random.uniform(0.5, 1.5)}
                else:
                    params = {'radius_x': self.np_random.uniform(50, 100), 'radius_y': self.np_random.uniform(20, 50), 'speed': self.np_random.uniform(0.5, 1.2)}
                self.targets.append(Target((start_x, start_y), 15, pattern=pattern_choice, pattern_params=params))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.laser_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.particles.clear()
        self._spawn_targets()
        
        self.timer = self.MAX_TIME
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.chain_progress = 0
        self.chain_cooldown = 0.0
        self.screen_shake = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        dt = self.clock.tick(self.TARGET_FPS) / 1000.0
        self.steps += 1
        reward = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        laser_speed = 300
        if movement == 1: self.laser_pos.y -= laser_speed * dt
        elif movement == 2: self.laser_pos.y += laser_speed * dt
        elif movement == 3: self.laser_pos.x -= laser_speed * dt
        elif movement == 4: self.laser_pos.x += laser_speed * dt
        
        self.laser_pos.x = max(0, min(self.WIDTH, self.laser_pos.x))
        self.laser_pos.y = max(0, min(self.HEIGHT, self.laser_pos.y))

        self.timer = max(0, self.timer - dt)
        
        for target in self.targets:
            if target.alive: target.update(dt)
                
        for particle in self.particles[:]:
            particle.update(dt)
            if particle.lifetime <= 0: self.particles.remove(particle)

        if self.chain_cooldown > 0:
            self.chain_cooldown -= dt
            if self.chain_cooldown <= 0:
                self.chain_progress = 0

        if self.screen_shake > 0: self.screen_shake -= 1

        laser_width = 2 + self.chain_progress * 2
        for target in self.targets:
            if target.alive and abs(target.pos.x - self.laser_pos.x) < target.radius + laser_width / 2:
                target.alive = False
                reward += 0.1
                self.chain_progress = min(5, self.chain_progress + 1)
                self.chain_cooldown = 2.0
                self._create_explosion(target.pos)
                self.screen_shake = 5
                
                if self.chain_progress == 5:
                    reward += 1.0
                    self.chain_progress = 0
                    self.chain_cooldown = 0

        terminated = False
        all_targets_destroyed = all(not t.alive for t in self.targets)
        
        if self.timer <= 0:
            reward -= 100.0
            terminated = True
            self.game_over = True
        
        if all_targets_destroyed:
            reward += 100.0
            terminated = True
            self.game_over = True
            self.level += 1
            
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_explosion(self, pos):
        num_particles = 20 + self.chain_progress * 10
        colors = [(255, 180, 50), (255, 120, 0), (255, 220, 150)]
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(50, 200)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            size = self.np_random.uniform(3, 8)
            lifetime = self.np_random.uniform(0.5, 1.2)
            color_idx = self.np_random.integers(len(colors))
            color = colors[color_idx]
            self.particles.append(Particle(pos, vel, size, lifetime, color))

    def _get_observation(self):
        render_offset = (0, 0)
        if self.screen_shake > 0:
            render_offset = (self.np_random.integers(-3, 4), self.np_random.integers(-3, 4))
        
        temp_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        temp_surface.fill(self.COLOR_BG)
        
        self._render_background(temp_surface)
        self._render_laser(temp_surface)
        self._render_targets(temp_surface)
        self._render_particles(temp_surface)
        
        self.screen.blit(temp_surface, render_offset)
        self._render_ui(self.screen)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, surface):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(surface, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(surface, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

    def _render_laser(self, surface):
        laser_width = 2 + self.chain_progress * 2
        laser_alpha = 100 + self.chain_progress * 30
        laser_rect = pygame.Rect(self.laser_pos.x - laser_width // 2, 0, laser_width, self.HEIGHT)
        
        beam_surf = pygame.Surface(laser_rect.size, pygame.SRCALPHA)
        beam_surf.fill(self.COLOR_LASER_BEAM + (laser_alpha,))
        surface.blit(beam_surf, laser_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)

        x, y = int(self.laser_pos.x), int(self.laser_pos.y)
        glow_size = 15 + self.chain_progress
        for i in range(glow_size, 0, -2):
            alpha = int(50 * (1 - i / glow_size))
            pygame.gfxdraw.aacircle(surface, x, y, i, self.COLOR_LASER_ORIGIN + (alpha,))
        
        line_len = 8
        pygame.draw.line(surface, self.COLOR_LASER_ORIGIN, (x - line_len, y), (x + line_len, y), 1)
        pygame.draw.line(surface, self.COLOR_LASER_ORIGIN, (x, y - line_len), (x, y + line_len), 1)

    def _render_targets(self, surface):
        for target in self.targets:
            target.draw(surface)

    def _render_particles(self, surface):
        for particle in self.particles:
            particle.draw(surface)

    def _render_ui(self, surface):
        target_text = f"TARGETS: {sum(1 for t in self.targets if t.alive)}"
        text_surf = self.font_large.render(target_text, True, self.COLOR_TEXT)
        surface.blit(text_surf, (10, 10))
        
        mins, secs = divmod(self.timer, 60)
        timer_text = f"{int(mins):02}:{int(secs):02}"
        text_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        surface.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))
        
        bar_width, bar_height = 200, 15
        bar_x, bar_y = self.WIDTH // 2 - bar_width // 2, self.HEIGHT - 30
        fill_ratio = self.chain_progress / 5.0
        fill_width = int(bar_width * fill_ratio)
        
        pygame.draw.rect(surface, self.COLOR_CHAIN_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        if fill_width > 0:
            pygame.draw.rect(surface, self.COLOR_CHAIN_BAR, (bar_x, bar_y, fill_width, bar_height), border_radius=4)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "targets_left": sum(1 for t in self.targets if t.alive),
            "timer": self.timer,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be run by the evaluation system
    # You can use it for testing and debugging
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Laser Chain Reaction")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    while True:
        movement = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting Game ---")
                env.level = 1
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if terminated:
            if info.get('timer', 1) > 0 and info.get('targets_left', 1) == 0:
                print(f"Level {info['level'] - 1} Cleared! Episode Reward: {total_reward:.2f}")
                print(f"Starting Level {info['level']}...")
                pygame.time.wait(2000)
                obs, info = env.reset()
                terminated = False
                total_reward = 0
            else:
                print(f"Game Over! Final Score: {total_reward:.2f}. Press 'R' to restart game.")
                while True:
                    event = pygame.event.wait()
                    if event.type == pygame.QUIT:
                        env.close()
                        quit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("--- Restarting Game ---")
                        env.level = 1
                        obs, info = env.reset()
                        total_reward = 0
                        terminated = False
                        break