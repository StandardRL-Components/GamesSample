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

    user_guide = "Controls: ↑↓←→ to move. Hold Space near an asteroid to mine. Avoid the red enemy ships."
    game_description = "Mine asteroids for valuable ore while dodging enemy ships in a retro top-down space arcade. Collect 100 ore to win, but lose all 3 lives and you fail."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_SIZE = 12
    ENEMY_SIZE = 10
    ASTEROID_BASE_SIZE = 25
    NUM_ASTEROIDS = 8
    NUM_ENEMIES = 5
    WIN_SCORE = 100
    MAX_STEPS = 10000
    PLAYER_ACCELERATION = 0.5
    PLAYER_DAMPING = 0.92
    MINING_DISTANCE = 50
    ASTEROID_RESPAWN_TIME = 300  # 10 seconds at 30fps

    # --- Colors ---
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 60)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 80)
    COLOR_ASTEROID = (120, 120, 130)
    COLOR_ASTEROID_DEPLETED = (50, 50, 60)
    COLOR_LASER = (100, 150, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_STAR = (200, 200, 220)

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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 64)

        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.asteroids = []
        self.enemies = []
        self.particles = []
        self.stars = []

        # Initialize attributes that are part of the game state
        self.is_mining = False
        self.mining_target = None
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.game_won = False
        self.invincibility_timer = 0
        
        # self.reset() is called to set the initial state
        # self.validate_implementation() can be called after reset if needed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.game_won = False
        self.invincibility_timer = 90  # 3 seconds
        self.is_mining = False
        self.mining_target = None

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)

        self._init_stars()
        self._init_asteroids()
        self._init_enemies()

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        reward = 0
        self.steps += 1
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

        self._update_player(movement)
        mining_reward = self._update_asteroids_and_mining(space_held)
        reward += mining_reward
        self._update_enemies()
        self._update_particles()

        collision_reward = self._handle_collisions()
        reward += collision_reward

        # Continuous reward/penalty
        nearest_asteroid_dist = self._get_dist_to_nearest_asteroid()
        if nearest_asteroid_dist > self.MINING_DISTANCE * 2:
            reward -= 0.001  # Small penalty for being idle/far away

        terminated = self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                self.game_won = True
                reward += 100  # Win bonus
            else:
                reward -= 100  # Loss penalty
        
        terminated = self.lives <= 0 or self.score >= self.WIN_SCORE

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)

        self._render_stars()
        self._render_particles()
        self._render_asteroids()
        if self.is_mining:
            self._render_laser()
        self._render_enemies()
        self._render_player()
        self._render_ui()

        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives}

    # --- Initialization Helpers ---
    def _init_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': (self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                'size': self.np_random.uniform(0.5, 1.5)
            })

    def _init_asteroids(self):
        self.asteroids = []
        for _ in range(self.NUM_ASTEROIDS):
            self.asteroids.append(self._create_asteroid())

    def _init_enemies(self):
        self.enemies = []
        for i in range(self.NUM_ENEMIES):
            pos = self._get_safe_spawn_pos()
            pattern = self.np_random.choice(['circular', 'horizontal', 'vertical'])
            enemy = {
                'pos': pos,
                'pattern': pattern,
                'angle': self.np_random.uniform(0, 2 * math.pi),
                'center': pos.copy(),
                'radius': self.np_random.uniform(50, 150),
                'direction': 1,
            }
            self.enemies.append(enemy)

    def _create_asteroid(self, respawn=False):
        pos = self._get_safe_spawn_pos() if respawn else np.array([
            self.np_random.uniform(50, self.WIDTH - 50),
            self.np_random.uniform(50, self.HEIGHT - 50)
        ], dtype=np.float32)

        max_ore = self.np_random.integers(50, 150)
        shape_points = []
        num_vertices = self.np_random.integers(7, 12)
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = self.np_random.uniform(0.7, 1.1) * self.ASTEROID_BASE_SIZE
            shape_points.append((math.cos(angle) * dist, math.sin(angle) * dist))

        return {
            'pos': pos,
            'ore': max_ore,
            'max_ore': max_ore,
            'shape_points': shape_points,
            'respawn_timer': 0,
        }

    def _get_safe_spawn_pos(self):
        while True:
            pos = np.array([
                self.np_random.uniform(0, self.WIDTH),
                self.np_random.uniform(0, self.HEIGHT)
            ], dtype=np.float32)
            if np.linalg.norm(pos - self.player_pos) > 100:
                return pos

    # --- Update Logic ---
    def _update_player(self, movement):
        acc = np.array([0.0, 0.0])
        if movement == 1: acc[1] -= self.PLAYER_ACCELERATION
        elif movement == 2: acc[1] += self.PLAYER_ACCELERATION
        elif movement == 3: acc[0] -= self.PLAYER_ACCELERATION
        elif movement == 4: acc[0] += self.PLAYER_ACCELERATION

        self.player_vel += acc
        self.player_vel *= self.PLAYER_DAMPING
        self.player_pos += self.player_vel

        # Screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        # Thrust particles
        if np.linalg.norm(acc) > 0 and self.steps % 2 == 0:
            angle = math.atan2(self.player_vel[1], self.player_vel[0]) + math.pi
            self._spawn_particles(1, self.player_pos, angle, 0.5, (255, 200, 100), 1, 15, 2)

    def _update_asteroids_and_mining(self, space_held):
        mining_reward = 0
        self.is_mining = False
        self.mining_target = None

        # Handle respawning
        for asteroid in self.asteroids:
            if asteroid['respawn_timer'] > 0:
                asteroid['respawn_timer'] -= 1
                if asteroid['respawn_timer'] == 0:
                    new_asteroid = self._create_asteroid(respawn=True)
                    asteroid.update(new_asteroid)

        if space_held:
            closest_asteroid = None
            min_dist = float('inf')
            for asteroid in self.asteroids:
                if asteroid['ore'] > 0:
                    dist = np.linalg.norm(self.player_pos - asteroid['pos'])
                    if dist < min_dist:
                        min_dist = dist
                        closest_asteroid = asteroid

            if closest_asteroid and min_dist < self.MINING_DISTANCE:
                self.is_mining = True
                self.mining_target = closest_asteroid

                # sfx: mining_laser_loop
                ore_mined = min(closest_asteroid['ore'], 2)  # Mine 2 ore per frame
                closest_asteroid['ore'] -= ore_mined
                self.score += ore_mined
                mining_reward += ore_mined * 0.1

                # Mining particles
                self._spawn_particles(1, closest_asteroid['pos'], 0, 2 * math.pi, (255, 255, 0), 2, 20, 1)

                if closest_asteroid['ore'] <= 0:
                    # sfx: asteroid_depleted
                    closest_asteroid['respawn_timer'] = self.ASTEROID_RESPAWN_TIME
                    self.is_mining = False

        return mining_reward

    def _update_enemies(self):
        speed_bonus = (self.score // 25) * 0.2
        base_speed = 1.0 + speed_bonus

        for enemy in self.enemies:
            if enemy['pattern'] == 'circular':
                enemy['angle'] += 0.02 * (base_speed / 1.5)
                enemy['pos'][0] = enemy['center'][0] + math.cos(enemy['angle']) * enemy['radius']
                enemy['pos'][1] = enemy['center'][1] + math.sin(enemy['angle']) * enemy['radius']
            elif enemy['pattern'] == 'horizontal':
                enemy['pos'][0] += base_speed * enemy['direction']
                if enemy['pos'][0] > self.WIDTH or enemy['pos'][0] < 0:
                    enemy['direction'] *= -1
            elif enemy['pattern'] == 'vertical':
                enemy['pos'][1] += base_speed * enemy['direction']
                if enemy['pos'][1] > self.HEIGHT or enemy['pos'][1] < 0:
                    enemy['direction'] *= -1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] *= 0.98

    def _handle_collisions(self):
        if self.invincibility_timer > 0:
            return 0

        for enemy in self.enemies:
            dist = np.linalg.norm(self.player_pos - enemy['pos'])
            if dist < self.PLAYER_SIZE + self.ENEMY_SIZE:
                # sfx: player_explosion
                self.lives -= 1
                self._spawn_particles(50, self.player_pos, 0, 2 * math.pi, (255, 100, 0), 5, 60, 4)
                if self.lives > 0:
                    self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
                    self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
                    self.invincibility_timer = 90
                return -10  # Collision penalty
        return 0

    # --- Helper Methods ---
    def _spawn_particles(self, count, pos, angle, spread, color, speed, lifespan, size):
        for _ in range(count):
            p_angle = angle + self.np_random.uniform(-spread, spread)
            p_speed = speed * self.np_random.uniform(0.5, 1.5)
            vel = np.array([math.cos(p_angle), math.sin(p_angle)]) * p_speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(lifespan // 2, lifespan),
                'color': color,
                'size': size * self.np_random.uniform(0.8, 1.2)
            })

    def _get_dist_to_nearest_asteroid(self):
        min_dist = float('inf')
        for asteroid in self.asteroids:
            if asteroid['ore'] > 0:
                dist = np.linalg.norm(self.player_pos - asteroid['pos'])
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    # --- Rendering ---
    def _render_stars(self):
        for star in self.stars:
            size = star['size']
            if self.np_random.random() < 0.005:  # Twinkle
                size *= 2
            pygame.draw.circle(self.screen, self.COLOR_STAR, star['pos'], size)

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, p['size']))

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            if asteroid['respawn_timer'] > 0:
                continue

            size_ratio = asteroid['ore'] / asteroid['max_ore']
            color = self.COLOR_ASTEROID if size_ratio > 0 else self.COLOR_ASTEROID_DEPLETED

            points = [(p[0] * size_ratio + asteroid['pos'][0], p[1] * size_ratio + asteroid['pos'][1]) for p in
                      asteroid['shape_points']]
            if len(points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ENEMY_SIZE + 4, self.COLOR_ENEMY_GLOW)
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ENEMY_SIZE, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ENEMY_SIZE, self.COLOR_ENEMY)

    def _render_player(self):
        if self.lives <= 0: return

        # Blink when invincible
        if self.invincibility_timer > 0 and self.steps % 6 < 3:
            return

        angle = math.atan2(self.player_vel[1], self.player_vel[0])
        points = []
        for i in range(3):
            a = angle + (i * 2 * math.pi / 3)
            if i == 0:  # Pointy front
                p_size = self.PLAYER_SIZE * 1.5
            else:  # Flat back
                p_size = self.PLAYER_SIZE * 0.8
            points.append((
                self.player_pos[0] + p_size * math.cos(a),
                self.player_pos[1] + p_size * math.sin(a)
            ))

        # Glow
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER_GLOW)
        # Body
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_laser(self):
        if self.mining_target:
            start_pos = tuple(self.player_pos.astype(int))
            end_pos = tuple(self.mining_target['pos'].astype(int))
            width = int(self.np_random.uniform(1, 4))  # Pulsing width
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, width)
            pygame.draw.aaline(self.screen, self.COLOR_LASER, start_pos, end_pos)

    def _render_ui(self):
        score_text = self.font_ui.render(f"ORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

    def _render_game_over(self):
        text_str = "VICTORY!" if self.game_won else "GAME OVER"
        color = self.COLOR_PLAYER if self.game_won else self.COLOR_ENEMY

        text_surf = self.font_game_over.render(text_str, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))

        bg_rect = text_rect.inflate(20, 20)
        s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, bg_rect)
        self.screen.blit(text_surf, text_rect)


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # This block will not be run by the verifier
    env = GameEnv()
    obs, info = env.reset()

    # To render, you need to unset the dummy video driver
    # and create a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    pygame.display.init()
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))

    running = True
    total_reward = 0.0
    terminated = False
    truncated = False

    while running:
        movement = 0  # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    if terminated or truncated:
                        obs, info = env.reset()
                        total_reward = 0.0
                        terminated = False
                        truncated = False

        if not (terminated or truncated):
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Convert the observation (H, W, C) back to a Pygame surface (W, H)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait for 'R' to reset

        env.clock.tick(30)

    env.close()
    pygame.quit()