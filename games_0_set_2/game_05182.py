# Generated: 2025-08-28T04:13:58.257617
# Source Brief: brief_05182.md
# Brief Index: 5182

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Use arrow keys to move. Hold Shift for a speed boost. Press Space for a temporary shield."
    )

    game_description = (
        "Pilot a spaceship in a top-down arcade environment. Collect 50 golden asteroids to win, but watch out for hazardous red debris. You have 3 lives."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_THRUSTER = (255, 255, 255)
    COLOR_ASTEROID = (255, 200, 0)
    COLOR_DEBRIS = (255, 50, 50)
    COLOR_SHIELD = (100, 200, 255)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BAR_BG = (50, 50, 80)
    COLOR_UI_SHIELD_BAR = (100, 200, 255)
    COLOR_UI_BOOST_BAR = (255, 150, 0)

    # Screen dimensions
    WIDTH, HEIGHT = 640, 400

    # Game parameters
    PLAYER_SIZE = 12
    PLAYER_ACCELERATION = 0.4
    PLAYER_DAMPING = 0.96
    PLAYER_MAX_SPEED = 5.0
    BOOST_MULTIPLIER = 2.0
    ASTEROID_COUNT = 10
    ASTEROID_SIZE_RANGE = (10, 18)
    DEBRIS_SIZE_RANGE = (8, 14)
    WIN_SCORE = 50
    INITIAL_LIVES = 3
    MAX_STEPS = 3000 # Increased to allow more time to win

    # Cooldowns & Durations (in frames/steps)
    SHIELD_DURATION = 90 # 3 seconds at 30fps
    SHIELD_COOLDOWN = 300 # 10 seconds
    BOOST_DURATION = 60 # 2 seconds
    BOOST_COOLDOWN = 240 # 8 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        self.render_mode = render_mode
        
        # This will be initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.lives = None
        self.score = None
        self.asteroids_collected = None
        self.steps = None
        self.game_over = None
        
        self.asteroids = []
        self.debris = []
        self.particles = []
        self.stars = []
        
        self.shield_timer = 0
        self.shield_cooldown_timer = 0
        self.boost_timer = 0
        self.boost_cooldown_timer = 0

        self.initial_debris_spawn_interval = 60 # 2 seconds
        self.debris_spawn_interval = self.initial_debris_spawn_interval
        self.debris_spawn_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_angle = -90.0

        self.lives = self.INITIAL_LIVES
        self.score = 0
        self.asteroids_collected = 0
        self.steps = 0
        self.game_over = False

        self.shield_timer = 0
        self.shield_cooldown_timer = 0
        self.boost_timer = 0
        self.boost_cooldown_timer = 0
        
        self.debris_spawn_interval = self.initial_debris_spawn_interval
        self.debris_spawn_timer = 0

        self.asteroids = [self._create_asteroid() for _ in range(self.ASTEROID_COUNT)]
        self.debris = [self._create_debris() for _ in range(3)]
        self.particles = []
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
            for _ in range(100)
        ]
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01  # Small reward for surviving
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        self._update_player()
        self._update_entities()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        if self.boost_timer > 0:
            reward += 0.02 # Small reward for using boost effectively

        self._spawn_entities()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated or truncated:
            if self.asteroids_collected >= self.WIN_SCORE:
                reward += 100 # Win bonus
            elif self.lives <= 0:
                reward -= 100 # Lose penalty
            self.game_over = True
        
        self.score += collision_reward # Only update score based on game events
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        if movement != 0:
            direction_map = {1: -90, 2: 90, 3: 180, 4: 0} # up, down, left, right
            target_angle = direction_map[movement]
            
            # Smoothly turn player
            angle_diff = (target_angle - self.player_angle + 180) % 360 - 180
            self.player_angle += angle_diff * 0.2
            
            rad = math.radians(self.player_angle)
            acceleration = np.array([math.cos(rad), math.sin(rad)]) * self.PLAYER_ACCELERATION
            self.player_vel += acceleration
        
        # Shield
        if space_held and self.shield_cooldown_timer == 0:
            self.shield_timer = self.SHIELD_DURATION
            self.shield_cooldown_timer = self.SHIELD_COOLDOWN + self.SHIELD_DURATION

        # Boost
        if shift_held and self.boost_cooldown_timer == 0:
            self.boost_timer = self.BOOST_DURATION
            self.boost_cooldown_timer = self.BOOST_COOLDOWN + self.BOOST_DURATION

    def _update_player(self):
        # Timers
        self.shield_timer = max(0, self.shield_timer - 1)
        self.shield_cooldown_timer = max(0, self.shield_cooldown_timer - 1)
        self.boost_timer = max(0, self.boost_timer - 1)
        self.boost_cooldown_timer = max(0, self.boost_cooldown_timer - 1)

        # Speed limit
        current_max_speed = self.PLAYER_MAX_SPEED * self.BOOST_MULTIPLIER if self.boost_timer > 0 else self.PLAYER_MAX_SPEED
        speed = np.linalg.norm(self.player_vel)
        if speed > current_max_speed:
            self.player_vel = self.player_vel / speed * current_max_speed

        # Damping
        self.player_vel *= self.PLAYER_DAMPING
        self.player_pos += self.player_vel

        # Screen boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _update_entities(self):
        # Move debris
        for d in self.debris:
            d['pos'] += d['vel']
            d['angle'] += d['rot_speed']
        self.debris = [d for d in self.debris if -50 < d['pos'][0] < self.WIDTH + 50 and -50 < d['pos'][1] < self.HEIGHT + 50]

        # Rotate asteroids
        for a in self.asteroids:
            a['angle'] += a['rot_speed']

        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _spawn_entities(self):
        self.debris_spawn_timer += 1
        if self.debris_spawn_timer >= self.debris_spawn_interval:
            self.debris.append(self._create_debris())
            self.debris_spawn_timer = 0

    def _handle_collisions(self):
        reward = 0
        
        # Player-Asteroid
        for asteroid in self.asteroids[:]:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_SIZE + asteroid['size']:
                self.asteroids.remove(asteroid)
                self.asteroids_collected += 1
                
                if self.shield_timer > 0:
                    reward += 2.0 # Bonus for collecting with shield
                else:
                    reward += 1.0

                self._create_particles(asteroid['pos'], 30, self.COLOR_ASTEROID, 2.5, 40)
                self.asteroids.append(self._create_asteroid()) # Respawn
                
                # Increase difficulty
                if self.asteroids_collected > 0 and self.asteroids_collected % 10 == 0:
                    self.debris_spawn_interval = max(20, self.debris_spawn_interval * 0.9)

        # Player-Debris
        is_invulnerable = self.shield_timer > 0
        for d in self.debris:
            dist = np.linalg.norm(self.player_pos - d['pos'])
            if dist < self.PLAYER_SIZE + d['size']:
                if is_invulnerable:
                    self._create_particles(d['pos'], 20, self.COLOR_SHIELD, 2, 30)
                    d['pos'] = np.array([-100, -100], dtype=np.float32) # Fling it off-screen
                else:
                    self.lives -= 1
                    reward -= 5.0
                    self._create_particles(self.player_pos, 50, self.COLOR_DEBRIS, 4, 60)
                    # Reset player position
                    self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
                    self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
                    if self.lives <= 0:
                        self.game_over = True
                break # only one collision per frame
        return reward

    def _check_termination(self):
        return self.lives <= 0 or self.asteroids_collected >= self.WIN_SCORE

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
            "lives": self.lives,
            "asteroids_collected": self.asteroids_collected,
        }

    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            c = int(120 + math.sin(self.steps * 0.01 + x) * 40)
            pygame.draw.rect(self.screen, (c, c, c), (x, y, size, size))
            
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = p['color']
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['lifespan'] * 0.1 + 1), (*color, alpha))

        # Asteroids
        for a in self.asteroids:
            self._draw_rotated_polygon(self.screen, a['points'], a['pos'], a['angle'], self.COLOR_ASTEROID, a['size'])

        # Debris
        for d in self.debris:
            self._draw_rotated_polygon(self.screen, d['points'], d['pos'], d['angle'], self.COLOR_DEBRIS, d['size'])

        # Player
        is_moving = np.linalg.norm(self.player_vel) > 0.5
        if is_moving:
            self._render_thruster()
        
        # Glow
        glow_size = self.PLAYER_SIZE + 10 + (math.sin(self.steps * 0.1) * 3)
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), int(glow_size), self.COLOR_PLAYER_GLOW)
        
        # Ship body
        player_points = [
            (self.PLAYER_SIZE, 0),
            (-self.PLAYER_SIZE * 0.7, self.PLAYER_SIZE * 0.7),
            (-self.PLAYER_SIZE * 0.4, 0),
            (-self.PLAYER_SIZE * 0.7, -self.PLAYER_SIZE * 0.7),
        ]
        self._draw_rotated_polygon(self.screen, player_points, self.player_pos, self.player_angle, self.COLOR_PLAYER, self.PLAYER_SIZE)
        
        # Shield
        if self.shield_timer > 0:
            shield_alpha = 100 + int(abs(math.sin(self.steps * 0.2)) * 100)
            radius = self.PLAYER_SIZE + 5
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), radius, (*self.COLOR_SHIELD, shield_alpha))
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), radius+2, (*self.COLOR_SHIELD, int(shield_alpha*0.5)))
    
    def _render_thruster(self):
        flame_length = np.linalg.norm(self.player_vel) * 2
        if self.boost_timer > 0:
            flame_length *= 2.5
        
        rad = math.radians(self.player_angle + 180) # Opposite direction of ship
        
        for i in range(1, int(flame_length)):
            p = self.np_random.random()
            pos = self.player_pos + np.array([math.cos(rad), math.sin(rad)]) * i * 1.5 * p
            size = max(1, (flame_length - i) * 0.3)
            alpha = 50 + int(p * 150)
            color = self.COLOR_THRUSTER if self.np_random.random() > 0.1 else self.COLOR_PLAYER
            if self.boost_timer > 0:
                color = self.COLOR_UI_BOOST_BAR
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(size), (*color, alpha))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"ASTEROIDS: {self.asteroids_collected} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            points = [ (10, 0), (-7, 7), (-4, 0), (-7, -7) ]
            pos = (self.WIDTH - 25 - i * 25, 22)
            self._draw_rotated_polygon(self.screen, points, pos, -90, self.COLOR_PLAYER, 10)

        # Cooldown bars
        bar_width = 100
        bar_height = 10
        # Shield
        shield_ratio = 1.0 - min(1.0, self.shield_cooldown_timer / self.SHIELD_COOLDOWN) if self.SHIELD_COOLDOWN > 0 else 1.0
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 35, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_SHIELD_BAR, (10, 35, bar_width * shield_ratio, bar_height))
        shield_text = self.font_small.render("SHIELD", True, self.COLOR_UI_SHIELD_BAR if shield_ratio == 1.0 else self.COLOR_UI_TEXT)
        self.screen.blit(shield_text, (120, 31))
        
        # Boost
        boost_ratio = 1.0 - min(1.0, self.boost_cooldown_timer / self.BOOST_COOLDOWN) if self.BOOST_COOLDOWN > 0 else 1.0
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 55, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_BOOST_BAR, (10, 55, bar_width * boost_ratio, bar_height))
        boost_text = self.font_small.render("BOOST", True, self.COLOR_UI_BOOST_BAR if boost_ratio == 1.0 else self.COLOR_UI_TEXT)
        self.screen.blit(boost_text, (120, 51))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "GAME OVER"
            if self.asteroids_collected >= self.WIN_SCORE:
                message = "YOU WIN!"
            
            text = self.font_large.render(message, True, self.COLOR_UI_TEXT)
            text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text, text_rect)

    # --- Helper Functions ---
    def _create_asteroid(self):
        pos = self.np_random.random(size=2) * np.array([self.WIDTH, self.HEIGHT])
        while np.linalg.norm(pos - self.player_pos) < 100: # Don't spawn on player
            pos = self.np_random.random(size=2) * np.array([self.WIDTH, self.HEIGHT])
            
        size = self.np_random.uniform(*self.ASTEROID_SIZE_RANGE)
        num_points = self.np_random.integers(7, 12)
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dist = size * self.np_random.uniform(0.7, 1.1)
            points.append((math.cos(angle) * dist, math.sin(angle) * dist))
        
        return {
            'pos': pos.astype(np.float32), 'size': size, 'angle': 0, 
            'rot_speed': self.np_random.uniform(-1, 1), 'points': points
        }

    def _create_debris(self):
        edge = self.np_random.integers(4)
        if edge == 0: # top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -30.0])
        elif edge == 1: # bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 30.0])
        elif edge == 2: # left
            pos = np.array([-30.0, self.np_random.uniform(0, self.HEIGHT)])
        else: # right
            pos = np.array([self.WIDTH + 30.0, self.np_random.uniform(0, self.HEIGHT)])

        target = self.player_pos + self.np_random.uniform(-50, 50, size=2)
        delta = target - pos
        norm = np.linalg.norm(delta)
        
        if norm > 1e-6:
            direction = delta / norm
        else:
            # If target and pos are the same, pick a random direction
            angle = self.np_random.uniform(0, 2 * np.pi)
            direction = np.array([math.cos(angle), math.sin(angle)])

        speed = self.np_random.uniform(1.5, 3.5)
        
        size = self.np_random.uniform(*self.DEBRIS_SIZE_RANGE)
        points = [ (size, 0), (-size * 0.5, size * 0.8), (-size * 0.5, -size * 0.8) ]
        
        return {
            'pos': pos.astype(np.float32), 'vel': (direction * speed).astype(np.float32), 'size': size,
            'angle': math.degrees(math.atan2(direction[1], direction[0])),
            'rot_speed': self.np_random.uniform(-2, 2), 'points': points
        }
    
    def _create_particles(self, pos, count, color, speed_range, life_range):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, speed_range)
            lifespan = self.np_random.integers(life_range // 2, life_range)
            self.particles.append({
                'pos': pos.copy(),
                'vel': (np.array([math.cos(angle), math.sin(angle)]) * speed).astype(np.float32),
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color
            })
    
    def _draw_rotated_polygon(self, surface, points, pos, angle, color, size):
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        
        rotated_points = []
        for x, y in points:
            x_rot = x * cos_a - y * sin_a + pos[0]
            y_rot = x * sin_a + y * cos_a + pos[1]
            rotated_points.append((x_rot, y_rot))
        
        if len(rotated_points) > 2:
            pygame.gfxdraw.aapolygon(surface, rotated_points, color)
            pygame.gfxdraw.filled_polygon(surface, rotated_points, color)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # We need to switch the render_mode to "human" to see the screen
    # For that, we need a display.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    import sys
    if sys.platform == "win32":
        os.environ["SDL_VIDEODRIVER"] = "directx"

    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display screen
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    truncated = False
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print("\n" + "="*50)
    print(env.game_description)
    print(env.user_guide)
    print("="*50 + "\n")

    while not terminated and not truncated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No movement
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
            
        # Actions
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Info: {info}")
    env.close()