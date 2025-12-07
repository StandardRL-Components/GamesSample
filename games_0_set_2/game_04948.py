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
        "Controls: ↑↓←→ to move. Hold space to fire at the nearest asteroid. Avoid the red enemy ships."
    )

    game_description = (
        "A top-down arcade shooter. Pilot your ship, collect 50 asteroids by shooting them, and avoid colliding with enemies."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 70)
    COLOR_ASTEROID = (150, 150, 150)
    COLOR_PROJECTILE = (200, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    
    # Game parameters
    PLAYER_ACCELERATION = 0.4
    PLAYER_FRICTION = 0.92
    PLAYER_MAX_SPEED = 5
    PLAYER_LIVES = 3
    PLAYER_INVINCIBILITY_STEPS = 60 # 2 seconds at 30fps

    ENEMY_COUNT = 4
    ENEMY_BASE_SPEED = 1.0
    
    ASTEROID_COUNT = 15
    ASTEROID_MIN_SIZE = 10
    ASTEROID_MAX_SIZE = 25
    ASTEROID_WIN_SCORE = 50

    PROJECTILE_SPEED = 10
    FIRE_COOLDOWN = 6 # frames

    MAX_STEPS = 1500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_lives = 0
        self.player_invincible_timer = 0
        self.fire_cooldown_timer = 0
        self.enemy_speed_modifier = 0.0

        self.enemies = []
        self.asteroids = []
        self.projectiles = []
        self.particles = []
        self.stars = []

        self.np_random = None
        self.dist_to_nearest_asteroid = float('inf')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_lives = self.PLAYER_LIVES
        self.player_invincible_timer = 0
        self.fire_cooldown_timer = 0
        self.enemy_speed_modifier = 0.0

        self.enemies = [self._spawn_enemy() for _ in range(self.ENEMY_COUNT)]
        self.asteroids = [self._spawn_asteroid() for _ in range(self.ASTEROID_COUNT)]
        self.projectiles = []
        self.particles = []
        
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': np.array([self.np_random.random() * self.SCREEN_WIDTH, self.np_random.random() * self.SCREEN_HEIGHT]),
                'size': self.np_random.integers(1, 3),
                'speed': 0.1 + self.np_random.random() * 0.4
            })
        
        self.dist_to_nearest_asteroid = self._get_dist_to_nearest_asteroid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        # If the game is over, the agent should call reset().
        # We still process the action to return a valid transition, but the state won't change meaningfully.
        if not self.game_over:
            movement, space_held, _ = action
            space_held = (space_held == 1)
            
            # --- Update Game Logic ---
            self._handle_input(movement, space_held)
            self._update_player()
            self._update_projectiles()
            self._update_enemies()
            self._update_particles()
        
        reward = 0.0
        
        # --- Handle Collisions and Game Events ---
        if not self.game_over:
            reward += self._handle_collisions()
            self._respawn_asteroids()

            # --- Continuous Reward ---
            new_dist = self._get_dist_to_nearest_asteroid()
            if new_dist < self.dist_to_nearest_asteroid:
                reward += 0.1 # Small reward for getting closer
            self.dist_to_nearest_asteroid = new_dist

        # --- Check Termination ---
        self.steps += 1
        terminated = False
        truncated = False
        
        if self.score >= self.ASTEROID_WIN_SCORE:
            if not self.game_over: reward += 100.0
            terminated = True
        elif self.player_lives <= 0:
            if not self.game_over: reward -= 100.0
            terminated = True
        
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True

        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Helper methods for game logic ---

    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 1: self.player_vel[1] -= self.PLAYER_ACCELERATION  # Up
        if movement == 2: self.player_vel[1] += self.PLAYER_ACCELERATION  # Down
        if movement == 3: self.player_vel[0] -= self.PLAYER_ACCELERATION  # Left
        if movement == 4: self.player_vel[0] += self.PLAYER_ACCELERATION  # Right

        # Shooting
        if self.fire_cooldown_timer > 0:
            self.fire_cooldown_timer -= 1
        
        if space_held and self.fire_cooldown_timer == 0:
            target_asteroid = self._find_nearest_asteroid()
            if target_asteroid:
                # Fire projectile
                direction = target_asteroid['pos'] - self.player_pos
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                    self.projectiles.append({
                        'pos': self.player_pos.copy(),
                        'vel': direction * self.PROJECTILE_SPEED
                    })
                    self.fire_cooldown_timer = self.FIRE_COOLDOWN

    def _update_player(self):
        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION
        
        # Clamp speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = (self.player_vel / speed) * self.PLAYER_MAX_SPEED
        
        # Update position
        self.player_pos += self.player_vel
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], 10, self.SCREEN_WIDTH - 10)
        self.player_pos[1] = np.clip(self.player_pos[1], 10, self.SCREEN_HEIGHT - 10)

        # Invincibility
        if self.player_invincible_timer > 0:
            self.player_invincible_timer -= 1

    def _update_projectiles(self):
        projectiles_to_keep = []
        for p in self.projectiles:
            p['pos'] += p['vel']
            if 0 < p['pos'][0] < self.SCREEN_WIDTH and 0 < p['pos'][1] < self.SCREEN_HEIGHT:
                projectiles_to_keep.append(p)
        self.projectiles = projectiles_to_keep

    def _update_enemies(self):
        for enemy in self.enemies:
            # Simple circular motion pattern
            enemy['angle'] += enemy['angle_speed'] * (self.ENEMY_BASE_SPEED + self.enemy_speed_modifier)
            enemy['pos'][0] = enemy['center'][0] + math.cos(enemy['angle']) * enemy['radius']
            enemy['pos'][1] = enemy['center'][1] + math.sin(enemy['angle']) * enemy['radius']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.2)

    def _handle_collisions(self):
        reward = 0.0
        
        # Projectile-Asteroid
        hit_projectile_indices = set()
        hit_asteroid_indices = set()
        for p_idx, p in enumerate(self.projectiles):
            for a_idx, a in enumerate(self.asteroids):
                if a_idx in hit_asteroid_indices:
                    continue
                dist = np.linalg.norm(p['pos'] - a['pos'])
                if dist < a['size']:
                    reward += 10.0
                    self.score += 1
                    self._create_explosion(a['pos'], (255, 200, 50), 30)
                    hit_asteroid_indices.add(a_idx)
                    hit_projectile_indices.add(p_idx)
                    
                    if self.score % 10 == 0 and self.score > 0:
                        self.enemy_speed_modifier += 0.2

        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in hit_projectile_indices]
        self.asteroids = [a for i, a in enumerate(self.asteroids) if i not in hit_asteroid_indices]

        # Player-Enemy
        if self.player_invincible_timer == 0:
            for enemy in self.enemies:
                dist = np.linalg.norm(self.player_pos - enemy['pos'])
                if dist < 15: # Player size + enemy size
                    self.player_lives -= 1
                    reward -= 25.0 # Heavier penalty for getting hit
                    self.player_invincible_timer = self.PLAYER_INVINCIBILITY_STEPS
                    self._create_explosion(self.player_pos, self.COLOR_ENEMY, 20)
                    break
        
        return reward

    def _respawn_asteroids(self):
        while len(self.asteroids) < self.ASTEROID_COUNT:
            self.asteroids.append(self._spawn_asteroid())

    # --- Spawning and Creation ---

    def _spawn_enemy(self):
        side = self.np_random.integers(0, 4)
        if side == 0: # Top
            center_pos = [self.np_random.random() * self.SCREEN_WIDTH, -50]
        elif side == 1: # Bottom
            center_pos = [self.np_random.random() * self.SCREEN_WIDTH, self.SCREEN_HEIGHT + 50]
        elif side == 2: # Left
            center_pos = [-50, self.np_random.random() * self.SCREEN_HEIGHT]
        else: # Right
            center_pos = [self.SCREEN_WIDTH + 50, self.np_random.random() * self.SCREEN_HEIGHT]

        return {
            'pos': np.array(center_pos, dtype=float),
            'center': np.array(center_pos, dtype=float),
            'radius': self.np_random.integers(100, 300),
            'angle': self.np_random.random() * 2 * math.pi,
            'angle_speed': (self.np_random.random() * 0.01 + 0.005) * (1 if self.np_random.random() > 0.5 else -1)
        }

    def _spawn_asteroid(self):
        pos = self.np_random.random(2) * [self.SCREEN_WIDTH, self.SCREEN_HEIGHT]
        while np.linalg.norm(pos - self.player_pos) < 100: # Avoid spawning on player
            pos = self.np_random.random(2) * [self.SCREEN_WIDTH, self.SCREEN_HEIGHT]
        
        num_vertices = self.np_random.integers(5, 9)
        size = self.np_random.integers(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE + 1)
        vertices = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = size * (0.7 + self.np_random.random() * 0.6)
            vertices.append((dist * math.cos(angle), dist * math.sin(angle)))

        return {'pos': pos, 'size': size, 'vertices': vertices}

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 4
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'size': self.np_random.integers(4, 8),
                'color': color
            })
    
    # --- Distance Helpers ---

    def _find_nearest_asteroid(self):
        if not self.asteroids:
            return None
        
        min_dist = float('inf')
        nearest = None
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
                nearest = asteroid
        return nearest

    def _get_dist_to_nearest_asteroid(self):
        nearest = self._find_nearest_asteroid()
        if nearest:
            return np.linalg.norm(self.player_pos - nearest['pos'])
        return float('inf')

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_asteroids()
        self._render_enemies()
        self._render_projectiles()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        player_vel_norm = np.linalg.norm(self.player_vel)
        if player_vel_norm > 0.1:
            direction = self.player_vel / player_vel_norm
        else:
            direction = np.array([0,0])

        for star in self.stars:
            star['pos'] -= direction * star['speed']
            if star['pos'][0] < 0: star['pos'][0] = self.SCREEN_WIDTH
            if star['pos'][0] > self.SCREEN_WIDTH: star['pos'][0] = 0
            if star['pos'][1] < 0: star['pos'][1] = self.SCREEN_HEIGHT
            if star['pos'][1] > self.SCREEN_HEIGHT: star['pos'][1] = 0
            
            color_val = int(star['speed'] * 150)
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), star['pos'], star['size'])

    def _render_asteroids(self):
        for a in self.asteroids:
            points = [(v[0] + a['pos'][0], v[1] + a['pos'][1]) for v in a['vertices']]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_enemies(self):
        for e in self.enemies:
            x, y = int(e['pos'][0]), int(e['pos'][1])
            points = [(x, y - 8), (x - 7, y + 5), (x + 7, y + 5)]
            pygame.gfxdraw.filled_trigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.COLOR_ENEMY)
            pygame.gfxdraw.aatrigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 12, self.COLOR_ENEMY_GLOW)


    def _render_projectiles(self):
        for p in self.projectiles:
            start = p['pos']
            end = p['pos'] - p['vel'] * 0.5
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), 2)

    def _render_player(self):
        if self.player_lives <= 0: return

        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        
        # Invincibility flash
        if self.player_invincible_timer > 0 and self.steps % 10 < 5:
            return

        # Player ship shape (triangle)
        angle = math.atan2(self.player_vel[1], self.player_vel[0]) if np.linalg.norm(self.player_vel) > 0.1 else -math.pi / 2
        p1 = (x + 10 * math.cos(angle), y + 10 * math.sin(angle))
        p2 = (x + 8 * math.cos(angle + 2.5), y + 8 * math.sin(angle + 2.5))
        p3 = (x + 8 * math.cos(angle - 2.5), y + 8 * math.sin(angle - 2.5))
        
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, x, y, 15, self.COLOR_PLAYER_GLOW)

        # Ship body
        pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)
        pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)

        # Engine trail
        if np.linalg.norm(self.player_vel) > 1:
            for _ in range(2):
                offset = (self.np_random.random(2) - 0.5) * 6
                pos = self.player_pos - self.player_vel * 1.5 + offset
                self.particles.append({
                    'pos': pos,
                    'vel': (self.np_random.random(2) - 0.5) * 0.5,
                    'life': 10,
                    'size': self.np_random.integers(2, 5),
                    'color': (255, 150, 0)
                })

    def _render_particles(self):
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), p['color'])

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"ASTEROIDS: {self.score} / {self.ASTEROID_WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.player_lives):
            x, y = self.SCREEN_WIDTH - 20 - (i * 20), 20
            points = [(x, y - 7), (x - 5, y + 4), (x + 5, y + 4)]
            pygame.gfxdraw.filled_trigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.COLOR_PLAYER)

        # Game Over message
        if self.game_over:
            if self.score >= self.ASTEROID_WIN_SCORE:
                msg = "YOU WIN!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires pygame to be installed with display support.
    # The environment itself runs headlessly.
    # To run, ensure you have a display: `unset SDL_VIDEODRIVER` or similar.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Arcade Asteroids")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False
                print("--- Game Reset ---")

        clock.tick(30) # Limit to 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()