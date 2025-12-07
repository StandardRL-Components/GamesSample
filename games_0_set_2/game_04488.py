import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Ensure Pygame runs headless
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold space to fire your mining laser."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship through an asteroid field, mining valuable ore while avoiding collisions to reach a target resource quota."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    WIN_SCORE = 200
    MAX_STEPS = 5000
    
    # Player settings
    PLAYER_MAX_HEALTH = 3
    PLAYER_ACCELERATION = 0.8
    PLAYER_FRICTION = 0.92
    PLAYER_MAX_SPEED = 8
    PLAYER_SIZE = 12

    # Asteroid settings
    ASTEROID_SPAWN_RATE = 20 # Lower is more frequent
    ASTEROID_ORE_MIN = 10
    ASTEROID_ORE_MAX = 50
    ASTEROID_MIN_SPEED = 1.0
    ASTEROID_MAX_SPEED = 2.5
    DIFFICULTY_INTERVAL = 200
    DIFFICULTY_SCALING = 0.01

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (50, 150, 255, 50)
    COLOR_THRUSTER = (255, 180, 50)
    COLOR_ASTEROID = (120, 110, 100)
    COLOR_ASTEROID_OUTLINE = (80, 70, 60)
    COLOR_ORE = (255, 215, 0)
    COLOR_LASER = (255, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (50, 200, 50)
    COLOR_HEALTH_BAR_BG = (200, 50, 50)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        self.render_mode = render_mode
        self.np_random = None
        self._last_action = None

        # This is here to quiet the linter, but reset() is the source of truth
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.player_pos = np.zeros(2)
        self.player_vel = np.zeros(2)
        self.player_health = 0
        self.asteroid_base_speed = 0
        self.asteroids = []
        self.particles = []
        self.stars = []
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT * 0.8], dtype=float)
        self.player_vel = np.zeros(2, dtype=float)
        self.player_health = self.PLAYER_MAX_HEALTH

        self.asteroid_base_speed = self.ASTEROID_MIN_SPEED
        self.asteroids = []
        for _ in range(5):
            self._spawn_asteroid(random_y=True)

        self.particles = []
        self.stars = self._create_stars(200)
        self._last_action = self.action_space.sample() * 0 # A neutral starting action

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self._last_action = action
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_held, _ = self._unpack_action(action)

        self._update_player(movement)
        
        mined_this_frame = self._handle_mining(space_held)
        if mined_this_frame > 0:
            reward += mined_this_frame # +1 per ore
        else:
            reward -= 0.01 # Small penalty for not mining

        reward += self._update_asteroids() # Collision and destruction rewards/penalties
        self._update_particles()
        self._spawn_new_asteroids()
        self._update_difficulty()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.victory:
                reward += 100
            else: # Lost or truncated
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _unpack_action(self, action):
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        return movement, space_held, shift_held

    def _update_player(self, movement):
        acc = np.zeros(2, dtype=float)
        if movement == 1: acc[1] -= self.PLAYER_ACCELERATION # Up
        if movement == 2: acc[1] += self.PLAYER_ACCELERATION # Down
        if movement == 3: acc[0] -= self.PLAYER_ACCELERATION # Left
        if movement == 4: acc[0] += self.PLAYER_ACCELERATION # Right

        self.player_vel += acc
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED
        
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_FRICTION

        # Screen boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        # Thruster particles
        if np.linalg.norm(acc) > 0:
            angle = math.atan2(-acc[1], -acc[0]) + (self.np_random.uniform(-0.3, 0.3))
            self._create_particles(1, self.player_pos, angle, 2, 4, self.COLOR_THRUSTER, 10, 0.9)


    def _handle_mining(self, space_held):
        total_ore_mined = 0
        if not space_held:
            return total_ore_mined

        target_asteroid = None
        min_dist = float('inf')
        
        for asteroid in self.asteroids:
            if asteroid['pos'][0] - asteroid['size'] < self.player_pos[0] < asteroid['pos'][0] + asteroid['size']:
                if asteroid['pos'][1] < self.player_pos[1]:
                    dist = self.player_pos[1] - asteroid['pos'][1]
                    if dist < min_dist:
                        min_dist = dist
                        target_asteroid = asteroid

        if target_asteroid:
            # sfx: mining_laser_active
            ore_mined = min(1, target_asteroid['ore'])
            target_asteroid['ore'] -= ore_mined
            self.score += ore_mined
            total_ore_mined += ore_mined

            # Create ore particles
            laser_hit_pos = np.array([self.player_pos[0], target_asteroid['pos'][1] + target_asteroid['size']])
            self._create_particles(1, laser_hit_pos, math.pi / 2, 1, 3, self.COLOR_ORE, 20, 0.95, gravity=0.1)
        
        return total_ore_mined


    def _update_asteroids(self):
        reward = 0
        asteroids_to_remove = []
        for i, asteroid in enumerate(self.asteroids):
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] += asteroid['rotation_speed']

            # Player collision
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_SIZE + asteroid['size']:
                self.player_health -= 1
                # sfx: player_hit
                self._create_particles(30, self.player_pos, 0, 1, 5, (255, 100, 100), 40, 0.9, spread=math.pi * 2)
                asteroids_to_remove.append(asteroid)
                continue

            # Asteroid-asteroid collision
            for j in range(i + 1, len(self.asteroids)):
                other = self.asteroids[j]
                dist_vec = asteroid['pos'] - other['pos']
                dist = np.linalg.norm(dist_vec)
                min_dist = asteroid['size'] + other['size']
                if dist < min_dist:
                    overlap = min_dist - dist
                    direction = dist_vec / dist if dist > 0 else np.array([1.0, 0.0])
                    if asteroid['size'] > other['size']:
                        other['pos'] -= direction * overlap
                    else:
                        asteroid['pos'] += direction * overlap

            # Check if mined out or off-screen
            if asteroid['ore'] <= 0 or asteroid['pos'][1] > self.HEIGHT + asteroid['size']:
                if asteroid['ore'] <= 0:
                    reward += 5 # Destruction reward
                    # sfx: asteroid_explode
                    self._create_particles(50, asteroid['pos'], 0, 1, 4, self.COLOR_ASTEROID, 50, 0.92, spread=math.pi * 2)
                asteroids_to_remove.append(asteroid)
        
        if asteroids_to_remove:
            ids_to_remove = {id(a) for a in asteroids_to_remove}
            self.asteroids = [a for a in self.asteroids if id(a) not in ids_to_remove]
        return reward
    
    def _spawn_new_asteroids(self):
        if self.np_random.integers(0, self.ASTEROID_SPAWN_RATE) == 0:
            self._spawn_asteroid()

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.asteroid_base_speed = min(self.asteroid_base_speed + self.DIFFICULTY_SCALING, self.ASTEROID_MAX_SPEED - 0.5)

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.victory = True
            return True
        if self.player_health <= 0:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "victory": self.victory
        }

    def _render_game(self):
        self._draw_stars()
        self._draw_asteroids()
        self._draw_particles()
        self._draw_player()
        self._draw_laser()

    def _draw_stars(self):
        for star in self.stars:
            color_val = int(star['brightness'])
            color = (color_val, color_val, color_val + 20)
            pygame.draw.circle(self.screen, color, (int(star['pos'][0]), int(star['pos'][1])), star['size'])
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.HEIGHT:
                star['pos'][0] = self.np_random.uniform(0, self.WIDTH)
                star['pos'][1] = 0

    def _draw_asteroids(self):
        for asteroid in self.asteroids:
            points = []
            for vertex in asteroid['vertices']:
                rotated_vertex = vertex.rotate(asteroid['angle'])
                points.append((int(asteroid['pos'][0] + rotated_vertex.x), int(asteroid['pos'][1] + rotated_vertex.y)))
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID_OUTLINE)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _draw_player(self):
        # Glow
        glow_size = int(self.PLAYER_SIZE * 2.5)
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surf, (int(self.player_pos[0] - glow_size), int(self.player_pos[1] - glow_size)))

        # Ship body
        p1 = (self.player_pos[0], self.player_pos[1] - self.PLAYER_SIZE)
        p2 = (self.player_pos[0] - self.PLAYER_SIZE * 0.7, self.player_pos[1] + self.PLAYER_SIZE * 0.7)
        p3 = (self.player_pos[0] + self.PLAYER_SIZE * 0.7, self.player_pos[1] + self.PLAYER_SIZE * 0.7)
        points = [p1, p2, p3]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _draw_laser(self):
        if self._last_action is None:
            return
        _, space_held, _ = self._unpack_action(self._last_action)
        if not space_held: return
        
        target_asteroid = None
        min_dist = float('inf')
        for asteroid in self.asteroids:
            if asteroid['pos'][0] - asteroid['size'] < self.player_pos[0] < asteroid['pos'][0] + asteroid['size']:
                if asteroid['pos'][1] < self.player_pos[1]:
                    dist = self.player_pos[1] - asteroid['pos'][1]
                    if dist < min_dist:
                        min_dist = dist
                        target_asteroid = asteroid
        
        if target_asteroid:
            start_pos = (int(self.player_pos[0]), int(self.player_pos[1] - self.PLAYER_SIZE))
            end_pos = (int(self.player_pos[0]), int(target_asteroid['pos'][1] + target_asteroid['size']))
            
            # Laser beam with glow
            pygame.draw.line(self.screen, (255, 150, 150), start_pos, end_pos, 5)
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, 3)

            # Impact sparks
            pygame.draw.circle(self.screen, (255, 255, 100), end_pos, 8)
            pygame.draw.circle(self.screen, self.COLOR_LASER, end_pos, 5)

    def _draw_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), int(p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"ORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health Bar
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width = 100
        bar_height = 15
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, bar_width * health_pct, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        if self.game_over:
            message = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_HEALTH_BAR if self.victory else self.COLOR_HEALTH_BAR_BG
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _spawn_asteroid(self, random_y=False):
        size = self.np_random.uniform(15, 40)
        pos = np.array([
            self.np_random.uniform(size, self.WIDTH - size),
            self.np_random.uniform(-self.HEIGHT * 0.5, -size) if not random_y else self.np_random.uniform(0, self.HEIGHT * 0.6)
        ], dtype=float)
        speed = self.np_random.uniform(self.asteroid_base_speed, self.asteroid_base_speed + (self.ASTEROID_MAX_SPEED - self.ASTEROID_MIN_SPEED))
        
        self.asteroids.append({
            'pos': pos,
            'vel': np.array([0, speed], dtype=float),
            'size': size,
            'ore': self.np_random.integers(self.ASTEROID_ORE_MIN, self.ASTEROID_ORE_MAX + 1),
            'vertices': self._create_asteroid_shape(size),
            'angle': self.np_random.uniform(0, 360),
            'rotation_speed': self.np_random.uniform(-1, 1)
        })

    def _create_asteroid_shape(self, radius):
        num_vertices = self.np_random.integers(7, 12)
        vertices = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = self.np_random.uniform(radius * 0.7, radius)
            vertices.append(pygame.math.Vector2(dist * math.cos(angle), dist * math.sin(angle)))
        return vertices

    def _create_particles(self, num, pos, angle, speed_min, speed_max, color, life, drag, spread=0.5, gravity=0.0):
        for _ in range(num):
            p_angle = angle + self.np_random.uniform(-spread, spread)
            p_speed = self.np_random.uniform(speed_min, speed_max)
            vel = np.array([math.cos(p_angle) * p_speed, math.sin(p_angle) * p_speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'size': self.np_random.uniform(2, 5),
                'life': self.np_random.uniform(life * 0.8, life * 1.2),
                'max_life': life,
                'color': color,
                'drag': drag,
                'gravity': gravity
            })

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= p['drag']
            p['vel'][1] += p.get('gravity', 0)
            p['life'] -= 1
            p['size'] *= 0.98
            if p['life'] <= 0 or p['size'] < 0.5:
                particles_to_remove.append(p)
        if particles_to_remove:
            ids_to_remove = {id(p) for p in particles_to_remove}
            self.particles = [p for p in self.particles if id(p) not in ids_to_remove]

    def _create_stars(self, num_stars):
        stars = []
        for _ in range(num_stars):
            stars.append({
                'pos': self.np_random.uniform(0, [self.WIDTH, self.HEIGHT], size=2),
                'size': self.np_random.integers(1, 3),
                'brightness': self.np_random.uniform(50, 150),
                'speed': self.np_random.uniform(0.1, 0.5)
            })
        return stars

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # The environment is headless by default, but for playing, we need a window.
    # Unset the dummy video driver to allow a window to be created.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    while running:
        # Player controls
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Rendering
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'R' to restart
            wait_for_restart = True
            while wait_for_restart:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_restart = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_restart = False
                clock.tick(30)

    env.close()