import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship, dodging and destroying asteroids to survive increasingly difficult waves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.MAX_WAVES = 10
        self.INITIAL_LIVES = 3

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PROJECTILE = (180, 255, 255)
        self.COLOR_ASTEROID = (160, 160, 160)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_EXPLOSION = [(255, 200, 0), (255, 100, 0), (200, 50, 0)]
        
        # Player settings
        self.PLAYER_ACCELERATION = 0.6
        self.PLAYER_FRICTION = 0.96
        self.PLAYER_MAX_SPEED = 7.0
        self.PLAYER_SIZE = 12
        self.FIRE_COOLDOWN_FRAMES = 6 # 5 shots per second
        self.INVINCIBILITY_FRAMES = 90 # 3 seconds

        # Projectile settings
        self.PROJECTILE_SPEED = 12.0
        
        # Asteroid settings
        self.ASTEROID_SIZES = {
            'large': {'radius': 40, 'score': 20, 'reward': 0.25},
            'medium': {'radius': 20, 'score': 50, 'reward': 0.5},
            'small': {'radius': 10, 'score': 100, 'reward': 1.0},
        }
        
        # Gymnasium spaces
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
            self.font_ui = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 72)
        except IOError:
            self.font_ui = pygame.font.SysFont("sans", 24)
            self.font_game_over = pygame.font.SysFont("sans", 72)

        # State variables (will be initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.player_pos = None
        self.player_vel = None
        self.player_last_move_dir = None
        self.player_lives = 0
        self.invincibility_timer = 0
        self.current_wave = 0
        self.asteroids = []
        self.projectiles = []
        self.particles = []
        self.stars = []
        self.fire_cooldown = 0
        self.last_space_held = False
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_last_move_dir = pygame.math.Vector2(0, -1) # Start facing up
        self.player_lives = self.INITIAL_LIVES
        self.invincibility_timer = self.INVINCIBILITY_FRAMES # Start with brief invincibility
        
        self.current_wave = 1
        self.asteroids = []
        self.projectiles = []
        self.particles = []
        self.fire_cooldown = 0
        self.last_space_held = False

        if not self.stars:
            self._create_stars()

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def _create_stars(self):
        self.stars = []
        for _ in range(150): # Deep space stars
            self.stars.append({
                'pos': (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                'size': 1, 'speed': 0.1
            })
        for _ in range(50): # Closer stars
            self.stars.append({
                'pos': (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                'size': 2, 'speed': 0.2
            })
    
    def _spawn_wave(self):
        num_asteroids = 8 + 2 * self.current_wave
        for _ in range(num_asteroids):
            self._create_asteroid('large')

    def _create_asteroid(self, size, pos=None, vel=None):
        if pos is None:
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -self.ASTEROID_SIZES['large']['radius'])
            elif edge == 1: # Bottom
                pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ASTEROID_SIZES['large']['radius'])
            elif edge == 2: # Left
                pos = pygame.math.Vector2(-self.ASTEROID_SIZES['large']['radius'], self.np_random.uniform(0, self.HEIGHT))
            else: # Right
                pos = pygame.math.Vector2(self.WIDTH + self.ASTEROID_SIZES['large']['radius'], self.np_random.uniform(0, self.HEIGHT))
        
        if vel is None:
            base_speed = 1.0 + (self.current_wave - 1) * 0.2
            speed = self.np_random.uniform(base_speed, base_speed + 0.5)
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed

        radius = self.ASTEROID_SIZES[size]['radius']
        num_points = self.np_random.integers(8, 15)
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            dist = self.np_random.uniform(radius * 0.7, radius * 1.3)
            points.append(pygame.math.Vector2(math.cos(angle), math.sin(angle)) * dist)

        self.asteroids.append({
            'pos': pos, 'vel': vel, 'radius': radius, 'size': size, 'points': points
        })

    def _create_explosion(self, pos, num_particles, radius):
        # sfx: explosion
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'lifetime': self.np_random.integers(20, 40),
                'max_lifetime': 40,
                'color': self.COLOR_EXPLOSION[self.np_random.integers(len(self.COLOR_EXPLOSION))],
                'size': self.np_random.uniform(1, radius/4)
            })

    def step(self, action):
        reward = 0.01 # Small reward for surviving a frame
        
        # --- Handle Input ---
        movement = action[0]
        space_held = action[1] == 1
        
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length() > 0:
            self.player_last_move_dir = move_vec.normalize()
            self.player_vel += self.player_last_move_dir * self.PLAYER_ACCELERATION
        
        if space_held and not self.last_space_held and self.fire_cooldown <= 0 and not self.game_over:
            # sfx: laser_fire
            proj_vel = self.player_last_move_dir * self.PROJECTILE_SPEED + self.player_vel * 0.5
            self.projectiles.append({
                'pos': pygame.math.Vector2(self.player_pos),
                'vel': proj_vel,
                'lifetime': 60,
                'hit': False
            })
            self.fire_cooldown = self.FIRE_COOLDOWN_FRAMES
        
        self.last_space_held = space_held
        if self.fire_cooldown > 0: self.fire_cooldown -= 1
        if self.invincibility_timer > 0: self.invincibility_timer -= 1

        # --- Update Game State ---
        # Player
        self.player_vel *= self.PLAYER_FRICTION
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)
        self.player_pos += self.player_vel
        self.player_pos.x %= self.WIDTH
        self.player_pos.y %= self.HEIGHT
        
        # Projectiles
        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['lifetime'] <= 0 or not (0 < p['pos'].x < self.WIDTH and 0 < p['pos'].y < self.HEIGHT):
                projectiles_to_remove.append(i)
                if not p['hit']:
                    reward -= 0.2 # Penalty for missed shot
        for i in reversed(projectiles_to_remove):
            self.projectiles.pop(i)

        # Asteroids
        for a in self.asteroids:
            a['pos'] += a['vel']
            a['pos'].x %= self.WIDTH
            a['pos'].y %= self.HEIGHT

        # Particles
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'] += p['vel']
            p['vel'] *= 0.98
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                particles_to_remove.append(i)
        for i in reversed(particles_to_remove):
            self.particles.pop(i)

        # --- Collisions ---
        if not self.game_over:
            # Player vs Asteroids
            if self.invincibility_timer <= 0:
                for a in self.asteroids:
                    if self.player_pos.distance_to(a['pos']) < a['radius'] + self.PLAYER_SIZE * 0.5:
                        # sfx: player_explosion
                        self.player_lives -= 1
                        reward -= 10.0
                        self._create_explosion(self.player_pos, 50, 30)
                        if self.player_lives > 0:
                            self.invincibility_timer = self.INVINCIBILITY_FRAMES
                            self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
                            self.player_vel = pygame.math.Vector2(0, 0)
                        else:
                            self.game_over = True
                        break # Only one hit per frame
            
            # Projectiles vs Asteroids
            asteroids_to_add = []
            asteroids_to_remove = []
            projectiles_to_remove = []
            for i, p in enumerate(self.projectiles):
                if i in projectiles_to_remove: continue
                for j, a in enumerate(self.asteroids):
                    if j in asteroids_to_remove: continue
                    if p['pos'].distance_to(a['pos']) < a['radius']:
                        p['hit'] = True
                        projectiles_to_remove.append(i)
                        asteroids_to_remove.append(j)
                        
                        self.score += self.ASTEROID_SIZES[a['size']]['score']
                        reward += self.ASTEROID_SIZES[a['size']]['reward']
                        self._create_explosion(a['pos'], int(a['radius']), a['radius'])
                        
                        if a['size'] == 'large':
                            for _ in range(2):
                                asteroids_to_add.append(('medium', pygame.math.Vector2(a['pos']), None))
                        elif a['size'] == 'medium':
                            for _ in range(2):
                                asteroids_to_add.append(('small', pygame.math.Vector2(a['pos']), None))
                        
                        break # Projectile can only hit one asteroid
            
            for i in sorted(list(set(projectiles_to_remove)), reverse=True):
                self.projectiles.pop(i)
            for i in sorted(list(set(asteroids_to_remove)), reverse=True):
                self.asteroids.pop(i)
            for size, pos, vel in asteroids_to_add:
                self._create_asteroid(size, pos, vel)

        # --- Game Flow ---
        if not self.asteroids and not self.game_over:
            self.current_wave += 1
            if self.current_wave > self.MAX_WAVES:
                self.game_won = True
                self.game_over = True
                reward += 100.0
            else:
                # sfx: wave_clear
                self._spawn_wave()
                self.invincibility_timer = self.INVINCIBILITY_FRAMES # Brief grace period

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render stars
        for star in self.stars:
            pos = (int(star['pos'][0] - self.player_vel.x * star['speed']) % self.WIDTH,
                   int(star['pos'][1] - self.player_vel.y * star['speed']) % self.HEIGHT)
            pygame.draw.circle(self.screen, self.COLOR_TEXT, pos, star['size'] // 2)

        # Render asteroids
        for a in self.asteroids:
            points = [(p + a['pos']) for p in a['points']]
            points_int = [(int(p.x), int(p.y)) for p in points]
            pygame.gfxdraw.aapolygon(self.screen, points_int, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, points_int, self.COLOR_ASTEROID)

        # Render projectiles
        for p in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(p['pos'].x), int(p['pos'].y)), 3)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

        # Render player
        if not (self.player_lives <= 0):
            is_blinking = self.invincibility_timer > 0 and (self.invincibility_timer // 4) % 2 == 0
            if not is_blinking:
                angle = self.player_last_move_dir.angle_to(pygame.math.Vector2(0, -1))
                p1 = pygame.math.Vector2(0, -self.PLAYER_SIZE).rotate(-angle) + self.player_pos
                p2 = pygame.math.Vector2(-self.PLAYER_SIZE * 0.6, self.PLAYER_SIZE * 0.6).rotate(-angle) + self.player_pos
                p3 = pygame.math.Vector2(self.PLAYER_SIZE * 0.6, self.PLAYER_SIZE * 0.6).rotate(-angle) + self.player_pos
                
                pygame.gfxdraw.aapolygon(self.screen, [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y))], self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(self.screen, [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y))], self.COLOR_PLAYER)

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Lives
        for i in range(self.player_lives):
            p1 = pygame.math.Vector2(0, -self.PLAYER_SIZE * 0.6) + pygame.math.Vector2(30 + i * 25, self.HEIGHT - 20)
            p2 = pygame.math.Vector2(-self.PLAYER_SIZE * 0.4, self.PLAYER_SIZE * 0.4) + pygame.math.Vector2(30 + i * 25, self.HEIGHT - 20)
            p3 = pygame.math.Vector2(self.PLAYER_SIZE * 0.4, self.PLAYER_SIZE * 0.4) + pygame.math.Vector2(30 + i * 25, self.HEIGHT - 20)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)])
        
        # Game Over / Win message
        if self.game_over:
            msg = "YOU WON!" if self.game_won else "GAME OVER"
            color = (0, 255, 0) if self.game_won else (255, 0, 0)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.current_wave,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Gymnasium's 'human' render mode is not used here to keep the
    # implementation self-contained as requested. We render directly to a window.
    
    # Un-comment the line below to run with a visible window
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Asteroid Annihilation")
        clock = pygame.time.Clock()
        
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            movement = 0 # No-op
            space_held = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space_held = 1
                
            action = [movement, space_held, 0] # Shift is unused
            
            obs, reward, term, trunc, info = env.step(action)
            terminated = term
            truncated = trunc
            
            # Draw the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(env.FPS)
            
        print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
        
        # Wait for a bit before closing
        pygame.time.wait(3000)
    
    except pygame.error as e:
        print(f"Caught pygame error: {e}")
        print("This is expected if you are running in a headless environment.")
        print("The environment itself is still valid for training.")

    finally:
        env.close()