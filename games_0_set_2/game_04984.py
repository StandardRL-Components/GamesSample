import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your ship. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship in a top-down arcade environment, blasting asteroids for points and survival."
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Set dummy video driver for headless operation
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        
        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_ASTEROID = (180, 180, 180)
        self.COLOR_INVINCIBLE = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.PARTICLE_COLORS = [(255, 69, 0), (255, 165, 0), (255, 215, 0)]

        # Fonts
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Game constants
        self.PLAYER_ACCELERATION = 0.4
        self.PLAYER_FRICTION = 0.96
        self.PLAYER_MAX_SPEED = 5
        self.PLAYER_RADIUS = 12
        self.PROJECTILE_SPEED = 8
        self.PROJECTILE_LIFETIME = 60 # frames
        self.FIRE_COOLDOWN = 8 # frames
        self.INVINCIBILITY_DURATION = 90 # frames
        self.MAX_ASTEROIDS = 10
        self.ASTEROIDS_TO_WIN = 50
        self.MAX_EPISODE_STEPS = 1000
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.last_move_dir = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.asteroids_destroyed = None
        self.base_asteroid_speed = None
        self.fire_cooldown_timer = None
        self.invincible_timer = None
        self.projectiles = None
        self.asteroids = None
        self.particles = None
        self.stars = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = -90 # Pointing up
        self.last_move_dir = pygame.Vector2(0, -1) # Default fire direction
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.win = False
        self.asteroids_destroyed = 0
        self.base_asteroid_speed = 1.0
        self.fire_cooldown_timer = 0
        self.invincible_timer = self.INVINCIBILITY_DURATION
        
        # Initialize entity lists
        self.projectiles = []
        self.asteroids = []
        self.particles = []
        
        # Create static starfield
        if self.stars is None:
             self.stars = [
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
                for _ in range(100)
            ]

        # Spawn initial asteroids
        while len(self.asteroids) < self.MAX_ASTEROIDS // 2:
            self._spawn_asteroid()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = -0.02 # Small penalty per step to encourage efficiency

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_held = action[1] == 1  # Boolean
            # shift_held = action[2] == 1 is unused as per brief
            
            # Process inputs
            self._handle_input(movement, space_held)
            
            # Update game logic
            self._update_player()
            self._update_projectiles()
            self._update_asteroids()
            self._update_particles()
            
            # Handle collisions and calculate rewards
            reward += self._handle_collisions()

            # Spawn new asteroids if needed
            if len(self.asteroids) < self.MAX_ASTEROIDS:
                self._spawn_asteroid()

        # Check for termination conditions
        terminated, term_reward = self._check_termination()
        reward += term_reward
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        move_dir = pygame.Vector2(0, 0)
        if movement == 1: # Up
            move_dir.y = -1
        elif movement == 2: # Down
            move_dir.y = 1
        elif movement == 3: # Left
            move_dir.x = -1
        elif movement == 4: # Right
            move_dir.x = 1

        if move_dir.length() > 0:
            self.player_vel += move_dir.normalize() * self.PLAYER_ACCELERATION
            self.last_move_dir = move_dir.normalize()

        # Firing
        if space_held and self.fire_cooldown_timer == 0:
            self._fire_projectile()
            self.fire_cooldown_timer = self.FIRE_COOLDOWN
            # sfx: player_shoot.wav

    def _fire_projectile(self):
        # Calculate spawn position at the tip of the ship
        offset = self.last_move_dir * (self.PLAYER_RADIUS + 5)
        spawn_pos = self.player_pos + offset
        
        # Create projectile
        projectile = {
            'pos': spawn_pos,
            'vel': self.last_move_dir * self.PROJECTILE_SPEED,
            'lifetime': self.PROJECTILE_LIFETIME
        }
        self.projectiles.append(projectile)

    def _update_player(self):
        # Apply friction and cap speed
        self.player_vel *= self.PLAYER_FRICTION
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)
        
        # Update position
        self.player_pos += self.player_vel
        
        # Screen wrapping
        if self.player_pos.x < 0: self.player_pos.x = self.WIDTH
        if self.player_pos.x > self.WIDTH: self.player_pos.x = 0
        if self.player_pos.y < 0: self.player_pos.y = self.HEIGHT
        if self.player_pos.y > self.HEIGHT: self.player_pos.y = 0

        # Update timers
        if self.fire_cooldown_timer > 0:
            self.fire_cooldown_timer -= 1
        if self.invincible_timer > 0:
            self.invincible_timer -= 1

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['lifetime'] <= 0 or not (0 < p['pos'].x < self.WIDTH and 0 < p['pos'].y < self.HEIGHT):
                self.projectiles.remove(p)

    def _update_asteroids(self):
        for a in self.asteroids:
            a['pos'] += a['vel']
            # Screen wrapping
            if a['pos'].x < -a['radius']: a['pos'].x = self.WIDTH + a['radius']
            if a['pos'].x > self.WIDTH + a['radius']: a['pos'].x = -a['radius']
            if a['pos'].y < -a['radius']: a['pos'].y = self.HEIGHT + a['radius']
            if a['pos'].y > self.HEIGHT + a['radius']: a['pos'].y = -a['radius']

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _spawn_asteroid(self, size=3, position=None, velocity=None):
        if position is None:
            # Spawn off-screen
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -30)
            elif edge == 1: # Right
                pos = pygame.Vector2(self.WIDTH + 30, self.np_random.uniform(0, self.HEIGHT))
            elif edge == 2: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 30)
            else: # Left
                pos = pygame.Vector2(-30, self.np_random.uniform(0, self.HEIGHT))
        else:
            pos = position

        if velocity is None:
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * self.base_asteroid_speed
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        else:
            vel = velocity
            
        radius = size * 10
        num_points = self.np_random.integers(8, 13)
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            dist = self.np_random.uniform(radius * 0.8, radius * 1.2)
            points.append(pygame.Vector2(dist * math.cos(angle), dist * math.sin(angle)))

        self.asteroids.append({'pos': pos, 'vel': vel, 'radius': radius, 'size': size, 'points': points})

    def _handle_collisions(self):
        reward = 0
        
        # Projectile-Asteroid collisions
        for p in self.projectiles[:]:
            for a in self.asteroids[:]:
                if p['pos'].distance_to(a['pos']) < a['radius']:
                    self.projectiles.remove(p)
                    self.asteroids.remove(a)
                    
                    # Add reward based on size
                    if a['size'] == 3: reward += 3
                    elif a['size'] == 2: reward += 2
                    else: reward += 1
                    
                    self.score += a['size'] * 10
                    self.asteroids_destroyed += 1

                    # Increase difficulty
                    if self.asteroids_destroyed > 0 and self.asteroids_destroyed % 10 == 0:
                        self.base_asteroid_speed += 0.05

                    # Create explosion
                    self._create_explosion(a['pos'], a['radius'])
                    # sfx: explosion.wav

                    # Break into smaller asteroids
                    if a['size'] > 1:
                        for _ in range(2):
                            new_vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize()
                            new_vel *= self.base_asteroid_speed * 1.5
                            self._spawn_asteroid(a['size'] - 1, a['pos'].copy(), new_vel)
                    break # Move to next projectile
        
        # Player-Asteroid collisions
        if self.invincible_timer == 0:
            for a in self.asteroids[:]:
                if self.player_pos.distance_to(a['pos']) < self.PLAYER_RADIUS + a['radius']:
                    self.asteroids.remove(a)
                    self.lives -= 1
                    reward -= 5
                    self.invincible_timer = self.INVINCIBILITY_DURATION
                    self._create_explosion(self.player_pos, self.PLAYER_RADIUS * 2)
                    # sfx: player_hit.wav
                    if self.lives > 0:
                        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
                        self.player_vel = pygame.Vector2(0, 0)
                    break
        
        return reward

    def _create_explosion(self, position, size):
        num_particles = int(size)
        for _ in range(num_particles):
            vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            vel = vel.normalize() * self.np_random.uniform(1, 4)
            lifetime = self.np_random.integers(20, 40)
            color = random.choice(self.PARTICLE_COLORS)
            self.particles.append({'pos': position.copy(), 'vel': vel, 'lifetime': lifetime, 'max_life': lifetime, 'color': color})

    def _check_termination(self):
        reward = 0
        if self.lives <= 0:
            self.game_over = True
            self.win = False
            reward = -100
            return True, reward
        if self.asteroids_destroyed >= self.ASTEROIDS_TO_WIN:
            self.game_over = True
            self.win = True
            reward = 100
            return True, reward
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            self.win = False
            return True, reward
        
        return False, reward

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_stars()
        self._render_asteroids()
        self._render_projectiles()
        self._render_particles()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_stars(self):
        for x, y, size in self.stars:
            color = (size * 40, size * 40, size * 40)
            pygame.draw.rect(self.screen, color, (x, y, size, size))

    def _render_player(self):
        # Determine angle from last move direction
        angle = self.last_move_dir.angle_to(pygame.Vector2(1, 0))
        
        # Create 3 points for the triangle
        p1 = self.player_pos + pygame.Vector2(self.PLAYER_RADIUS, 0).rotate(-angle)
        p2 = self.player_pos + pygame.Vector2(-self.PLAYER_RADIUS * 0.5, self.PLAYER_RADIUS * 0.8).rotate(-angle)
        p3 = self.player_pos + pygame.Vector2(-self.PLAYER_RADIUS * 0.5, -self.PLAYER_RADIUS * 0.8).rotate(-angle)
        
        # Draw ship
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)])
        
        # Draw invincibility shield
        if self.invincible_timer > 0:
            alpha = 100 + (math.sin(self.steps * 0.5) * 50)
            alpha = max(0, min(255, alpha))
            radius = self.PLAYER_RADIUS + 4
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), int(radius), (*self.COLOR_INVINCIBLE, int(alpha)))
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), int(radius), (*self.COLOR_INVINCIBLE, int(alpha/4)))

    def _render_asteroids(self):
        for a in self.asteroids:
            points = [(p + a['pos']).xy for p in a['points']]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_projectiles(self):
        for p in self.projectiles:
            start_pos = p['pos']
            end_pos = p['pos'] - p['vel'].normalize() * 8
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 3)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['lifetime'] / p['max_life']
            alpha = int(255 * life_ratio)
            color = (*p['color'], alpha)
            size = int(life_ratio * 5)
            
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'].x - size), int(p['pos'].y - size)))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_main.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Asteroids remaining
        asteroids_rem_text = self.font_main.render(f"DESTROYED: {self.asteroids_destroyed}/{self.ASTEROIDS_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(asteroids_rem_text, (self.WIDTH/2 - asteroids_rem_text.get_width()/2, self.HEIGHT - 30))

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else (255, 50, 50)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "asteroids_destroyed": self.asteroids_destroyed,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    import os
    # Set a visible video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11", "windows", "cocoa", etc.
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Arcade")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults
        movement = 0 # none
        space_held = 0 # released
        shift_held = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
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
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Limit human play to 30 FPS
        
    env.close()