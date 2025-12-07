
# Generated: 2025-08-27T22:16:38.520987
# Source Brief: brief_03068.md
# Brief Index: 3068

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Star:
    """Represents a single falling star."""
    def __init__(self, x, y, speed, size, color):
        self.pos = [x, y]
        self.speed = speed
        self.size = size
        self.color = color
        self.trail = []

    def update(self):
        """Move the star and update its trail."""
        self.trail.append(list(self.pos))
        if len(self.trail) > 10:
            self.trail.pop(0)
        self.pos[1] += self.speed

class Projectile:
    """Represents a player-fired laser bolt."""
    def __init__(self, x, y):
        self.pos = [x, y]
        self.speed = 15
    
    def update(self):
        """Move the projectile upwards."""
        self.pos[1] -= self.speed

class Particle:
    """Represents a single particle for effects like explosions."""
    def __init__(self, x, y, color, lifespan, np_random):
        self.pos = [x, y]
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.lifespan = lifespan
        self.initial_lifespan = lifespan
        self.color = color

    def update(self):
        """Move the particle and decrease its lifespan."""
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move your ship. Press space to fire your laser. "
        "Avoid the falling stars and survive for 60 seconds!"
    )

    game_description = (
        "Pilot a ship to shoot falling stars and survive for 60 seconds in a visually stunning top-down arcade environment."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and clock
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Game constants
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS
        self.PLAYER_SPEED = 7
        self.PLAYER_FIRE_COOLDOWN = 6
        self.PLAYER_INVINCIBILITY_FRAMES = 90 # 3 seconds

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 200)
        self.COLOR_LASER = (100, 255, 100)
        self.STAR_COLORS = [(255, 200, 0), (255, 100, 0), (255, 50, 50)]
        self.COLOR_EXPLOSION = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEART = (255, 0, 80)
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        
        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_lives = None
        self.player_invincible_timer = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.stars = []
        self.projectiles = []
        self.particles = []
        self.background_stars = []
        self.projectile_cooldown_timer = None
        self.consecutive_no_ops = None
        self.star_spawn_rate = None
        self.star_base_speed = None
        self.star_spawn_timer = None
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.width // 2, self.height - 50]
        self.player_lives = 3
        self.player_invincible_timer = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.stars.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.projectile_cooldown_timer = 0
        self.consecutive_no_ops = 0

        self.star_spawn_rate = 0.5  # Stars per second
        self.star_base_speed = 2.0  # Pixels per frame
        self.star_spawn_timer = 0

        if not self.background_stars:
            for _ in range(150):
                self.background_stars.append(
                    (
                        self.np_random.integers(0, self.width),
                        self.np_random.integers(0, self.height),
                        self.np_random.integers(1, 4), # size
                        self.np_random.uniform(0.1, 0.5) # speed
                    )
                )

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.steps += 1
        reward = 0.1  # Survival reward

        # --- Handle Input and Player Logic ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 0:
            self.consecutive_no_ops += 1
        else:
            self.consecutive_no_ops = 0
            if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
            elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED
            elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
            elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        
        if self.consecutive_no_ops > 5:
            reward -= 0.2

        self.player_pos[0] = np.clip(self.player_pos[0], 10, self.width - 10)
        self.player_pos[1] = np.clip(self.player_pos[1], 10, self.height - 10)

        if self.projectile_cooldown_timer > 0:
            self.projectile_cooldown_timer -= 1
            
        if space_held and self.projectile_cooldown_timer == 0:
            # Fire on press or while held (if cooldown allows)
            self.projectiles.append(Projectile(self.player_pos[0], self.player_pos[1] - 20))
            self.projectile_cooldown_timer = self.PLAYER_FIRE_COOLDOWN
            # sfx: player_shoot.wav
        
        if self.player_invincible_timer > 0:
            self.player_invincible_timer -= 1

        # --- Update Game World ---
        self._update_difficulty()
        self._update_stars()
        self._update_projectiles()
        self._update_particles()
        
        # --- Handle Collisions ---
        reward += self._handle_collisions()

        # --- Check Termination ---
        terminated = self.player_lives <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.player_lives > 0: # Survived
                reward += 50
                # sfx: victory.wav
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.star_spawn_rate += 0.05
            self.star_base_speed += 0.1

    def _spawn_star(self):
        x = self.np_random.uniform(20, self.width - 20)
        size_idx = self.np_random.choice(3, p=[0.6, 0.3, 0.1])
        size = [8, 12, 16][size_idx]
        speed_multiplier = [1, 1.2, 1.5][size_idx]
        speed = self.star_base_speed * speed_multiplier
        color = self.STAR_COLORS[size_idx]
        self.stars.append(Star(x, -size, speed, size, color))
        # sfx: star_spawn.wav

    def _update_stars(self):
        self.star_spawn_timer += self.star_spawn_rate / self.FPS
        if self.star_spawn_timer >= 1:
            self.star_spawn_timer -= 1
            self._spawn_star()
        
        for star in self.stars[:]:
            star.update()
            if star.pos[1] > self.height + star.size:
                self.stars.remove(star)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj.update()
            if proj.pos[1] < -10:
                self.projectiles.remove(proj)

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0:
                self.particles.remove(p)
    
    def _handle_collisions(self):
        reward = 0
        # Projectiles vs Stars
        for proj in self.projectiles[:]:
            for star in self.stars[:]:
                dist = math.hypot(proj.pos[0] - star.pos[0], proj.pos[1] - star.pos[1])
                if dist < star.size + 3: # 3 is projectile radius
                    self._create_explosion(star.pos, 30, self.COLOR_EXPLOSION)
                    # sfx: explosion.wav
                    if star in self.stars: self.stars.remove(star)
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    self.score += 1
                    reward += 1
                    break

        # Player vs Stars
        if self.player_invincible_timer == 0:
            player_rect = pygame.Rect(self.player_pos[0] - 8, self.player_pos[1] - 8, 16, 16)
            for star in self.stars[:]:
                star_rect = pygame.Rect(star.pos[0] - star.size, star.pos[1] - star.size, star.size*2, star.size*2)
                if player_rect.colliderect(star_rect):
                    self._create_explosion(self.player_pos, 50, self.COLOR_PLAYER)
                    # sfx: player_hit.wav
                    if star in self.stars: self.stars.remove(star)
                    self.player_lives -= 1
                    self.player_invincible_timer = self.PLAYER_INVINCIBILITY_FRAMES
                    reward -= 10
                    break
        return reward

    def _create_explosion(self, position, count, color):
        for _ in range(count):
            self.particles.append(Particle(position[0], position[1], color, self.np_random.integers(15, 30), self.np_random))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_stars()
        self._render_projectiles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.player_lives}
        
    def _render_background(self):
        # Update and draw parallax stars
        for i in range(len(self.background_stars)):
            x, y, size, speed = self.background_stars[i]
            y = (y + speed) % self.height
            self.background_stars[i] = (x, y, size, speed)
            
            # Twinkle effect
            brightness = self.np_random.integers(50, 150)
            color = (brightness, brightness, brightness)
            pygame.draw.circle(self.screen, color, (int(x), int(y)), size // 2)

    def _render_player(self):
        p = self.player_pos
        points = [(p[0], p[1] - 15), (p[0] - 10, p[1] + 10), (p[0] + 10, p[1] + 10)]
        
        # Invincibility flash
        if self.player_invincible_timer > 0 and (self.steps // 3) % 2 == 0:
            return

        # Glow effect
        glow_points = [(p[0], p[1] - 18), (p[0] - 13, p[1] + 13), (p[0] + 13, p[1] + 13)]
        pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        
        # Main ship
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_stars(self):
        for star in self.stars:
            # Trail
            if star.trail:
                for i, pos in enumerate(star.trail):
                    alpha = int(255 * (i / len(star.trail)))
                    trail_color = (star.color[0], star.color[1], star.color[2], alpha)
                    temp_surf = pygame.Surface((star.size*2, star.size*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, trail_color, (star.size, star.size), int(star.size * (i / len(star.trail))))
                    self.screen.blit(temp_surf, (int(pos[0] - star.size), int(pos[1] - star.size)))

            # Star body
            pygame.gfxdraw.aacircle(self.screen, int(star.pos[0]), int(star.pos[1]), star.size, star.color)
            pygame.gfxdraw.filled_circle(self.screen, int(star.pos[0]), int(star.pos[1]), star.size, star.color)

    def _render_projectiles(self):
        for proj in self.projectiles:
            start_pos = (int(proj.pos[0]), int(proj.pos[1]))
            end_pos = (int(proj.pos[0]), int(proj.pos[1] - 10))
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, 4)
    
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p.lifespan / p.initial_lifespan))
            color = (p.color[0], p.color[1], p.color[2], alpha)
            radius = int(3 * (p.lifespan / p.initial_lifespan))
            if radius > 0:
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (int(p.pos[0] - radius), int(p.pos[1] - radius)))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.width - timer_text.get_width() - 10, 10))

        # Lives
        for i in range(self.player_lives):
            self._draw_heart(self.screen, 20 + i * 35, self.height - 25, 15)
            
    def _draw_heart(self, surface, x, y, size):
        points = [
            (x, y - size // 3),
            (x + size // 2, y - size * 2 // 3),
            (x + size, y - size // 3),
            (x + size, y + size // 4),
            (x + size // 2, y + size * 2 // 3),
            (x, y + size // 4)
        ]
        pygame.gfxdraw.aapolygon(surface, points, self.COLOR_HEART)
        pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_HEART)

    def close(self):
        pygame.quit()

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Starfall Survivor")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            env.reset()
            pygame.time.wait(2000)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)
        
    env.close()