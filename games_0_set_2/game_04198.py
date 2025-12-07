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


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to fly your ship. Hold [SPACE] to fire the mining laser. "
        "Evade the red patrol ships!"
    )

    game_description = (
        "Pilot a mining ship, blast asteroids for valuable minerals, and evade relentless enemy patrols. "
        "Collect 50 minerals to win, but be careful - three hits and your ship is destroyed."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    WIN_SCORE = 50
    STARTING_LIVES = 3
    
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_THRUST = (255, 180, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_MINERAL = (50, 150, 255)
    COLOR_ASTEROID = (120, 110, 100)
    COLOR_LASER = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_EXPLOSION = [(255, 50, 50), (255, 150, 0), (255, 255, 0)]

    # Game settings
    PLAYER_ACCEL = 0.6
    PLAYER_FRICTION = 0.96
    PLAYER_MAX_SPEED = 6
    PLAYER_SIZE = 12
    PLAYER_INVINCIBILITY_FRAMES = 90 # 3 seconds

    ENEMY_COUNT = 2
    ENEMY_SPEED_INITIAL = 1.0
    ENEMY_SPEED_INCREASE_INTERVAL = 500
    ENEMY_SPEED_INCREASE_AMOUNT = 0.1
    ENEMY_SIZE = 14

    ASTEROID_COUNT = 7
    ASTEROID_HEALTH = 30 # Ticks to mine
    ASTEROID_MINERALS = 5
    ASTEROID_MIN_SIZE = 20
    ASTEROID_MAX_SIZE = 40

    MINERAL_SPEED = 2.0
    MINERAL_LIFESPAN = 450 # 15 seconds

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

        self.np_random = None
        self.last_action = None # Will be initialized in reset
        
        # self.reset() is called here, which is fine, but validation calls it again.
        # To avoid double-reset, we can let the first call be from validation.
        # However, the original code had it, so we keep it for consistency.
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             self.np_random = np.random.default_rng(seed=seed)
        else:
             self.np_random = np.random.default_rng()

        self.last_action = [0, 0, 0] # Initialize last_action
        self.steps = 0
        self.score = 0
        self.lives = self.STARTING_LIVES
        self.game_over = False
        self.game_won = False

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_angle = -90.0
        self.invincibility_timer = 0

        self.enemy_speed = self.ENEMY_SPEED_INITIAL
        self.enemies = [self._spawn_enemy() for _ in range(self.ENEMY_COUNT)]
        self.asteroids = [self._spawn_asteroid() for _ in range(self.ASTEROID_COUNT)]
        self.minerals = []
        self.particles = []
        self.stars = self._generate_stars(200)

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.last_action = action
        reward = 0
        terminated = False
        
        # --- Pre-step state for reward calculation ---
        closest_asteroid_dist_before = self._get_closest_asteroid_dist()

        # --- Handle Action and Update Player ---
        self._handle_input(action)
        self._update_player()
        
        # --- Update Game Objects ---
        mining_reward = self._update_laser_and_asteroids(action[1] == 1)
        reward += mining_reward
        self._update_enemies()
        self._update_minerals()
        self._update_particles()

        # --- Handle Collisions ---
        collision_reward, collected_minerals = self._handle_collisions()
        reward += collision_reward
        self.score += collected_minerals

        # --- Continuous Reward ---
        closest_asteroid_dist_after = self._get_closest_asteroid_dist()
        if closest_asteroid_dist_after < closest_asteroid_dist_before:
            reward += 0.01 # Small reward for getting closer
        else:
            reward -= 0.002 # Small penalty for moving away

        # --- Update Game State ---
        self.steps += 1
        if self.steps % self.ENEMY_SPEED_INCREASE_INTERVAL == 0:
            self.enemy_speed += self.ENEMY_SPEED_INCREASE_AMOUNT

        # --- Check Termination Conditions ---
        if self.score >= self.WIN_SCORE:
            self.game_won = True
            terminated = True
            reward += 100
        elif self.lives <= 0:
            self.game_over = True
            terminated = True
            reward -= 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        if self.auto_advance:
            self.clock.tick(self.FPS)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement = action[0]
        
        if movement == 1:  # Up
            self.player_vel[1] -= self.PLAYER_ACCEL
        elif movement == 2:  # Down
            self.player_vel[1] += self.PLAYER_ACCEL
        elif movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL
        
        if movement != 0:
            self._create_thrust_particles()

    def _update_player(self):
        # Apply friction and clamp speed
        self.player_vel *= self.PLAYER_FRICTION
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED
        
        # Update position and wrap around screen
        self.player_pos += self.player_vel
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT
        
        # Update angle to face velocity
        if np.linalg.norm(self.player_vel) > 0.1:
            self.player_angle = math.degrees(math.atan2(self.player_vel[1], self.player_vel[0]))
            
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

    def _update_laser_and_asteroids(self, space_held):
        reward = 0
        if not space_held:
            return reward
            
        laser_end = self.player_pos + np.array([math.cos(math.radians(self.player_angle)), math.sin(math.radians(self.player_angle))]) * (self.PLAYER_SIZE * 2)
        
        for i in range(len(self.asteroids) - 1, -1, -1):
            asteroid = self.asteroids[i]
            dist = np.linalg.norm(laser_end - asteroid['pos'])
            if dist < asteroid['size']:
                # sfx: mining_laser_hit
                asteroid['health'] -= 1
                self._create_hit_spark(laser_end)
                if asteroid['health'] <= 0:
                    # sfx: asteroid_explosion
                    self._create_explosion(asteroid['pos'], int(asteroid['size'] * 1.5))
                    for _ in range(asteroid['minerals']):
                        self._spawn_mineral(asteroid['pos'])
                    
                    # Check for risky mining bonus
                    is_near_enemy = False
                    for enemy in self.enemies:
                        if np.linalg.norm(asteroid['pos'] - enemy['pos']) < 150:
                            is_near_enemy = True
                            break
                    if is_near_enemy:
                        reward += 2.0 # Risky mining bonus
                    
                    self.asteroids[i] = self._spawn_asteroid() # Respawn
                break # Laser hits only one asteroid
        return reward

    def _update_enemies(self):
        for enemy in self.enemies:
            target_point = enemy['path'][enemy['target_idx']]
            direction = target_point - enemy['pos']
            dist = np.linalg.norm(direction)
            
            if dist < self.enemy_speed:
                enemy['pos'] = target_point.copy()
                enemy['target_idx'] = (enemy['target_idx'] + 1) % len(enemy['path'])
            else:
                enemy['pos'] += (direction / dist) * self.enemy_speed

    def _update_minerals(self):
        for i in range(len(self.minerals) - 1, -1, -1):
            mineral = self.minerals[i]
            mineral['pos'] += mineral['vel']
            mineral['lifespan'] -= 1
            if mineral['lifespan'] <= 0:
                self.minerals.pop(i)

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.pop(i)

    def _handle_collisions(self):
        reward = 0
        collected_minerals = 0
        
        # Player vs Enemy
        if self.invincibility_timer == 0:
            for enemy in self.enemies:
                dist = np.linalg.norm(self.player_pos - enemy['pos'])
                if dist < self.PLAYER_SIZE + self.ENEMY_SIZE:
                    # sfx: player_hit
                    self.lives -= 1
                    reward -= 10
                    self.invincibility_timer = self.PLAYER_INVINCIBILITY_FRAMES
                    self._create_explosion(self.player_pos, 30)
                    if self.lives > 0:
                        # Reset position to center to give player a chance
                        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
                        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
                    break
        
        # Player vs Mineral
        for i in range(len(self.minerals) - 1, -1, -1):
            mineral = self.minerals[i]
            dist = np.linalg.norm(self.player_pos - mineral['pos'])
            if dist < self.PLAYER_SIZE + 5: # Mineral size is 5
                # sfx: mineral_collect
                self.minerals.pop(i)
                collected_minerals += 1
                reward += 1.0
        
        return reward, collected_minerals

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_stars()
        self._render_minerals()
        self._render_asteroids()
        self._render_enemies()
        if self.lives > 0:
            self._render_player()
        self._render_particles()
        if self.game_over or self.game_won:
            self._render_end_screen()

    def _render_stars(self):
        for star in self.stars:
            size = star[2]
            color_val = 10 + size * 20
            color = (color_val, color_val, color_val + 20)
            self.screen.set_at((int(star[0]), int(star[1])), color)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = [(p[0] + asteroid['pos'][0], p[1] + asteroid['pos'][1]) for p in asteroid['shape']]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_enemies(self):
        for enemy in self.enemies:
            p = enemy['pos']
            s = self.ENEMY_SIZE
            points = [(p[0], p[1] - s), (p[0] + s, p[1]), (p[0], p[1] + s), (p[0] - s, p[1])]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
            # Glow effect
            pygame.gfxdraw.filled_polygon(self.screen, points, (*self.COLOR_ENEMY, 40))

    def _render_minerals(self):
        for mineral in self.minerals:
            pos = (int(mineral['pos'][0]), int(mineral['pos'][1]))
            alpha = max(0, min(255, int(mineral['lifespan'] * 2)))
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, (*self.COLOR_MINERAL, alpha // 4))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_MINERAL)

    def _render_player(self):
        if self.invincibility_timer > 0 and (self.invincibility_timer // 3) % 2 == 0:
            return # Blink effect
        
        angle_rad = math.radians(self.player_angle)
        s = self.PLAYER_SIZE
        p1 = (self.player_pos[0] + s * math.cos(angle_rad), self.player_pos[1] + s * math.sin(angle_rad))
        p2 = (self.player_pos[0] + s * math.cos(angle_rad + 2.3), self.player_pos[1] + s * math.sin(angle_rad + 2.3))
        p3 = (self.player_pos[0] + s * math.cos(angle_rad - 2.3), self.player_pos[1] + s * math.sin(angle_rad - 2.3))
        
        points = [p1, p2, p3]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        # Glow effect
        pygame.gfxdraw.aapolygon(self.screen, points, (*self.COLOR_PLAYER, 60))

    def _render_particles(self):
        # Laser
        if self.last_action is not None and len(self.last_action) > 1 and self.last_action[1] == 1 and not self.game_over:
             laser_start = self.player_pos + np.array([math.cos(math.radians(self.player_angle)), math.sin(math.radians(self.player_angle))]) * self.PLAYER_SIZE
             laser_end = self.player_pos + np.array([math.cos(math.radians(self.player_angle)), math.sin(math.radians(self.player_angle))]) * (self.PLAYER_SIZE * 2.5)
             pygame.draw.line(self.screen, self.COLOR_LASER, laser_start, laser_end, 3)
             pygame.draw.circle(self.screen, self.COLOR_LASER, laser_start, 4)

        for p in self.particles:
            alpha = max(0, int(255 * (p['lifespan'] / p['max_lifespan'])))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _render_ui(self):
        score_text = self.font_small.render(f"MINERALS: {self.score} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        for i in range(self.lives):
            s = self.PLAYER_SIZE * 0.8
            pos_x = self.WIDTH - 20 - i * (s * 2.5)
            pos_y = 20
            points = [(pos_x+s, pos_y), (pos_x-s*0.7, pos_y-s*0.7), (pos_x-s*0.7, pos_y+s*0.7)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_end_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        if self.game_won:
            end_text_str = "VICTORY"
            color = (100, 255, 100)
        else:
            end_text_str = "GAME OVER"
            color = (255, 100, 100)
            
        end_text = self.font_large.render(end_text_str, True, color)
        text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives}

    # --- Spawning and Creation Helpers ---
    def _generate_stars(self, n):
        return [(self.np_random.random() * self.WIDTH, self.np_random.random() * self.HEIGHT, self.np_random.random()) for _ in range(n)]

    def _spawn_asteroid(self):
        pos = np.array([self.np_random.random() * self.WIDTH, self.np_random.random() * self.HEIGHT], dtype=np.float32)
        # Ensure it doesn't spawn on the player
        while np.linalg.norm(pos - self.player_pos) < 100:
            pos = np.array([self.np_random.random() * self.WIDTH, self.np_random.random() * self.HEIGHT], dtype=np.float32)
        
        size = self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        num_vertices = self.np_random.integers(7, 12)
        shape = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = size + self.np_random.uniform(-size * 0.2, size * 0.2)
            shape.append((dist * math.cos(angle), dist * math.sin(angle)))
            
        return {
            'pos': pos, 'size': size, 'shape': shape, 'health': self.ASTEROID_HEALTH, 'minerals': self.ASTEROID_MINERALS
        }

    def _spawn_enemy(self):
        w, h = self.WIDTH, self.HEIGHT
        margin = 50
        path_w = self.np_random.uniform(w * 0.3, w - 2 * margin)
        path_h = self.np_random.uniform(h * 0.3, h - 2 * margin)
        top_left_x = self.np_random.uniform(margin, w - margin - path_w)
        top_left_y = self.np_random.uniform(margin, h - margin - path_h)
        
        path = [
            np.array([top_left_x, top_left_y]),
            np.array([top_left_x + path_w, top_left_y]),
            np.array([top_left_x + path_w, top_left_y + path_h]),
            np.array([top_left_x, top_left_y + path_h]),
        ]
        start_idx = self.np_random.integers(0, 4)
        return {'pos': path[start_idx].copy(), 'path': path, 'target_idx': (start_idx + 1) % 4}

    def _spawn_mineral(self, pos):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(self.MINERAL_SPEED * 0.5, self.MINERAL_SPEED * 1.5)
        vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * speed
        self.minerals.append({'pos': pos.copy(), 'vel': vel, 'lifespan': self.MINERAL_LIFESPAN})
    
    def _create_explosion(self, pos, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(15, 40)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'radius': self.np_random.uniform(2, 5),
                'color': random.choice(self.COLOR_EXPLOSION), 'lifespan': lifespan, 'max_lifespan': lifespan
            })

    def _create_thrust_particles(self):
        if self.np_random.random() > 0.3: # Don't spawn every frame
            return
        angle_rad = math.radians(self.player_angle + 180) # Opposite direction of movement
        offset = np.array([math.cos(angle_rad), math.sin(angle_rad)]) * self.PLAYER_SIZE * 0.8
        
        angle_spread = self.np_random.uniform(-0.4, 0.4)
        speed = self.np_random.uniform(1, 3)
        vel = np.array([math.cos(angle_rad + angle_spread), math.sin(angle_rad + angle_spread)]) * speed
        lifespan = self.np_random.integers(10, 20)
        self.particles.append({
            'pos': self.player_pos + offset, 'vel': vel, 'radius': self.np_random.uniform(1, 3),
            'color': self.COLOR_PLAYER_THRUST, 'lifespan': lifespan, 'max_lifespan': lifespan
        })

    def _create_hit_spark(self, pos):
        for _ in range(2):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(5, 15)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'radius': self.np_random.uniform(1, 2),
                'color': self.COLOR_LASER, 'lifespan': lifespan, 'max_lifespan': lifespan
            })

    def _get_closest_asteroid_dist(self):
        if not self.asteroids:
            return float('inf')
        return min(np.linalg.norm(self.player_pos - a['pos']) for a in self.asteroids)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires a graphical display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # --- Human Input to Action ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        if terminated:
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
                total_reward = 0
            
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()