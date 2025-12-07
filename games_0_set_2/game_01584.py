
# Generated: 2025-08-27T17:36:27.426761
# Source Brief: brief_01584.md
# Brief Index: 1584

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Hold space to mine nearby asteroids. Avoid the red enemy patrols."
    )

    game_description = (
        "Pilot a mining ship, collect ore from asteroids, and avoid enemy patrols to gather 100 ore units."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_ASTEROID = (120, 120, 120)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 60)
    COLOR_ORE = (255, 220, 0)
    COLOR_EXPLOSION = (255, 150, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_STAR = (200, 200, 220)

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Parameters
    MAX_STEPS = 1000
    WIN_SCORE = 100
    NUM_ASTEROIDS = 8
    NUM_ENEMIES = 5
    NUM_STARS = 100
    
    # Player
    PLAYER_SIZE = 12
    PLAYER_ACCEL = 0.4
    PLAYER_FRICTION = 0.96
    PLAYER_MAX_SPEED = 5
    
    # Mining
    MINING_RANGE = 60
    MINING_RATE = 1
    
    # Asteroids
    ASTEROID_BASE_SIZE = 20
    ASTEROID_ORE_CAPACITY = 25
    ASTEROID_RESPAWN_TIME = 450 # 15 seconds at 30fps

    # Enemies
    ENEMY_SIZE = 10
    ENEMY_BASE_SPEED = 1.0
    ENEMY_SPEED_INCREASE_INTERVAL = 200
    ENEMY_SPEED_INCREASE_AMOUNT = 0.05
    
    # Rewards
    REWARD_ORE_COLLECTED = 1.0
    REWARD_TIME_PENALTY = -0.01
    REWARD_RISKY_MINE = 5.0
    REWARD_SAFE_MINE = -2.0
    REWARD_WIN = 100.0
    REWARD_LOSE = -50.0
    RISK_RADIUS = 120

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = 0.0

        self.asteroids = []
        self.enemies = []
        self.particles = []
        self.stars = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.enemy_speed_modifier = 1.0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.enemy_speed_modifier = 1.0
        
        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = -90

        self.particles.clear()
        
        self._spawn_stars()
        self._spawn_asteroids()
        self._spawn_enemies()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = self.REWARD_TIME_PENALTY
        
        if not self.game_over:
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            reward += self._handle_input(movement, space_held)
            self._update_player()
            self._update_enemies()
            self._update_asteroids()
            self._check_collisions()

        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += self.REWARD_WIN
            elif self.steps >= self.MAX_STEPS:
                pass # No specific reward for timeout
            self.game_over = True
        
        # Clamp score to win condition
        self.score = min(self.score, self.WIN_SCORE)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        accel = pygame.math.Vector2(0, 0)
        if movement == 1:  # Up
            accel.y = -self.PLAYER_ACCEL
        elif movement == 2:  # Down
            accel.y = self.PLAYER_ACCEL
        elif movement == 3:  # Left
            accel.x = -self.PLAYER_ACCEL
        elif movement == 4:  # Right
            accel.x = self.PLAYER_ACCEL
        
        self.player_vel += accel
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)

        if accel.length() > 0:
            self.player_angle = self.player_vel.angle_to(pygame.math.Vector2(1, 0))
            self._create_thruster_particles()
        
        mining_reward = 0
        if space_held:
            mining_reward = self._mine_asteroid()
        
        return mining_reward

    def _update_player(self):
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_FRICTION

        self.player_pos.x = np.clip(self.player_pos.x, 0, self.SCREEN_WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.SCREEN_HEIGHT)

    def _update_enemies(self):
        if self.steps > 0 and self.steps % self.ENEMY_SPEED_INCREASE_INTERVAL == 0:
            self.enemy_speed_modifier += self.ENEMY_SPEED_INCREASE_AMOUNT
        
        for enemy in self.enemies:
            speed = enemy['base_speed'] * self.enemy_speed_modifier
            if enemy['type'] == 'circular':
                enemy['angle'] += speed * 0.05
                enemy['pos'].x = enemy['center'].x + math.cos(enemy['angle']) * enemy['radius']
                enemy['pos'].y = enemy['center'].y + math.sin(enemy['angle']) * enemy['radius']
            elif enemy['type'] in ['horizontal', 'vertical']:
                if enemy['type'] == 'horizontal':
                    enemy['pos'].x += speed * enemy['dir']
                    if enemy['pos'].x < 0 or enemy['pos'].x > self.SCREEN_WIDTH:
                        enemy['dir'] *= -1
                else: # vertical
                    enemy['pos'].y += speed * enemy['dir']
                    if enemy['pos'].y < 0 or enemy['pos'].y > self.SCREEN_HEIGHT:
                        enemy['dir'] *= -1
            elif enemy['type'] == 'diagonal':
                enemy['pos'] += enemy['dir'] * speed
                if enemy['pos'].x < 0 or enemy['pos'].x > self.SCREEN_WIDTH:
                    enemy['dir'].x *= -1
                if enemy['pos'].y < 0 or enemy['pos'].y > self.SCREEN_HEIGHT:
                    enemy['dir'].y *= -1
            elif enemy['type'] == 'random':
                if enemy['pos'].distance_to(enemy['target']) < 10:
                    enemy['target'] = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT))
                dir_vec = (enemy['target'] - enemy['pos']).normalize()
                enemy['pos'] += dir_vec * speed

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            if asteroid['active']:
                asteroid['angle'] += asteroid['rot_speed']
            else:
                asteroid['respawn_timer'] -= 1
                if asteroid['respawn_timer'] <= 0:
                    asteroid['active'] = True
                    asteroid['ore'] = self.ASTEROID_ORE_CAPACITY

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_collisions(self):
        for enemy in self.enemies:
            if self.player_pos.distance_to(enemy['pos']) < self.PLAYER_SIZE + self.ENEMY_SIZE:
                if not self.game_over:
                    self.game_over = True
                    self._create_explosion()
                    # SFX: Player explosion
                    # The terminal reward is handled in step() after termination is confirmed
                    return self.REWARD_LOSE
        return 0

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_particles()
        self._render_asteroids()
        self._render_enemies()
        if not self.game_over or (self.game_over and self.score >= self.WIN_SCORE):
             self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }
        
    def _spawn_stars(self):
        self.stars.clear()
        for _ in range(self.NUM_STARS):
            self.stars.append((
                self.np_random.integers(0, self.SCREEN_WIDTH),
                self.np_random.integers(0, self.SCREEN_HEIGHT),
                self.np_random.choice([1, 2])
            ))

    def _spawn_asteroids(self):
        self.asteroids.clear()
        for i in range(self.NUM_ASTEROIDS):
            while True:
                pos = pygame.math.Vector2(
                    self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
                    self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
                )
                if pos.distance_to(self.player_pos) > 100:
                    break
            
            num_vertices = self.np_random.integers(7, 12)
            base_size = self.ASTEROID_BASE_SIZE + self.np_random.uniform(-5, 5)
            vertices = []
            for j in range(num_vertices):
                angle = 2 * math.pi * j / num_vertices
                radius = base_size + self.np_random.uniform(-base_size/4, base_size/4)
                vertices.append((math.cos(angle) * radius, math.sin(angle) * radius))

            self.asteroids.append({
                'pos': pos,
                'ore': self.ASTEROID_ORE_CAPACITY,
                'active': True,
                'respawn_timer': 0,
                'vertices': vertices,
                'angle': self.np_random.uniform(0, 2 * math.pi),
                'rot_speed': self.np_random.uniform(-0.02, 0.02)
            })

    def _spawn_enemies(self):
        self.enemies.clear()
        patterns = ['circular', 'horizontal', 'vertical', 'diagonal', 'random']
        for i in range(self.NUM_ENEMIES):
            pattern = patterns[i % len(patterns)]
            enemy = {
                'type': pattern,
                'base_speed': self.ENEMY_BASE_SPEED + self.np_random.uniform(-0.2, 0.2),
            }
            if pattern == 'circular':
                enemy['center'] = pygame.math.Vector2(self.np_random.uniform(100, self.SCREEN_WIDTH-100), self.np_random.uniform(100, self.SCREEN_HEIGHT-100))
                enemy['radius'] = self.np_random.uniform(50, 150)
                enemy['angle'] = self.np_random.uniform(0, 2*math.pi)
                enemy['pos'] = enemy['center'] + pygame.math.Vector2(math.cos(enemy['angle']), math.sin(enemy['angle'])) * enemy['radius']
            elif pattern in ['horizontal', 'vertical']:
                enemy['pos'] = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT))
                enemy['dir'] = self.np_random.choice([-1, 1])
            elif pattern == 'diagonal':
                enemy['pos'] = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT))
                enemy['dir'] = pygame.math.Vector2(self.np_random.choice([-1, 1]), self.np_random.choice([-1, 1])).normalize()
            elif pattern == 'random':
                enemy['pos'] = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT))
                enemy['target'] = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT))
            self.enemies.append(enemy)

    def _mine_asteroid(self):
        closest_asteroid = None
        min_dist = float('inf')
        
        for asteroid in self.asteroids:
            if asteroid['active'] and asteroid['ore'] > 0:
                dist = self.player_pos.distance_to(asteroid['pos'])
                if dist < min_dist:
                    min_dist = dist
                    closest_asteroid = asteroid
        
        if closest_asteroid and min_dist < self.MINING_RANGE:
            ore_mined = min(self.MINING_RATE, closest_asteroid['ore'])
            self.score += ore_mined
            closest_asteroid['ore'] -= ore_mined
            
            if closest_asteroid['ore'] <= 0:
                closest_asteroid['active'] = False
                closest_asteroid['respawn_timer'] = self.ASTEROID_RESPAWN_TIME

            # SFX: Mining laser sound
            self._create_mining_particles(closest_asteroid['pos'])
            
            # Calculate risk reward
            min_enemy_dist = float('inf')
            for enemy in self.enemies:
                min_enemy_dist = min(min_enemy_dist, self.player_pos.distance_to(enemy['pos']))
            
            risk_reward = self.REWARD_RISKY_MINE if min_enemy_dist < self.RISK_RADIUS else self.REWARD_SAFE_MINE
            
            return self.REWARD_ORE_COLLECTED * ore_mined + risk_reward
            
        return 0

    def _create_thruster_particles(self):
        angle_rad = math.radians(self.player_angle)
        for _ in range(2):
            offset = pygame.math.Vector2(self.PLAYER_SIZE * 0.8, 0).rotate(-self.player_angle + self.np_random.uniform(-25, 25))
            pos = self.player_pos - offset
            vel = -self.player_vel + pygame.math.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5))
            self.particles.append({
                'pos': pos, 'vel': vel, 'life': self.np_random.integers(10, 20),
                'color': random.choice([self.COLOR_ORE, self.COLOR_EXPLOSION]), 'radius': self.np_random.uniform(1, 3)
            })

    def _create_mining_particles(self, asteroid_pos):
        for _ in range(3):
            direction = (self.player_pos - asteroid_pos).normalize()
            vel = direction * self.np_random.uniform(2, 4) + pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            self.particles.append({
                'pos': asteroid_pos.copy(), 'vel': vel, 'life': self.np_random.integers(15, 25),
                'color': self.COLOR_ORE, 'radius': self.np_random.uniform(2, 4)
            })

    def _create_explosion(self):
        for _ in range(50):
            vel = pygame.math.Vector2(0, 1).rotate(self.np_random.uniform(0, 360)) * self.np_random.uniform(1, 6)
            self.particles.append({
                'pos': self.player_pos.copy(), 'vel': vel, 'life': self.np_random.integers(20, 40),
                'color': random.choice([self.COLOR_EXPLOSION, self.COLOR_ORE, (255,255,255)]), 
                'radius': self.np_random.uniform(2, 5)
            })
            
    def _render_stars(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size / 2)

    def _render_player(self):
        # Glow effect
        glow_surf = pygame.Surface((self.PLAYER_SIZE * 4, self.PLAYER_SIZE * 4), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, int(self.PLAYER_SIZE*2), int(self.PLAYER_SIZE*2), int(self.PLAYER_SIZE*1.5), self.COLOR_PLAYER_GLOW)
        self.screen.blit(glow_surf, (int(self.player_pos.x - self.PLAYER_SIZE*2), int(self.player_pos.y - self.PLAYER_SIZE*2)))

        # Ship body
        angle_rad = math.radians(self.player_angle)
        p1 = self.player_pos + pygame.math.Vector2(self.PLAYER_SIZE, 0).rotate(-self.player_angle)
        p2 = self.player_pos + pygame.math.Vector2(-self.PLAYER_SIZE/2, self.PLAYER_SIZE*0.75).rotate(-self.player_angle)
        p3 = self.player_pos + pygame.math.Vector2(-self.PLAYER_SIZE/2, -self.PLAYER_SIZE*0.75).rotate(-self.player_angle)
        points = [(int(p.x), int(p.y)) for p in [p1, p2, p3]]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
    def _render_asteroids(self):
        for asteroid in self.asteroids:
            if asteroid['active']:
                points = []
                for vx, vy in asteroid['vertices']:
                    rotated_x = vx * math.cos(asteroid['angle']) - vy * math.sin(asteroid['angle'])
                    rotated_y = vx * math.sin(asteroid['angle']) + vy * math.cos(asteroid['angle'])
                    points.append((int(asteroid['pos'].x + rotated_x), int(asteroid['pos'].y + rotated_y)))
                
                ore_ratio = asteroid['ore'] / self.ASTEROID_ORE_CAPACITY
                color = tuple(int(c * ore_ratio + self.COLOR_BG[i] * (1 - ore_ratio)) for i, c in enumerate(self.COLOR_ASTEROID))

                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_int = (int(enemy['pos'].x), int(enemy['pos'].y))
            # Glow
            glow_surf = pygame.Surface((self.ENEMY_SIZE * 4, self.ENEMY_SIZE * 4), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, int(self.ENEMY_SIZE*2), int(self.ENEMY_SIZE*2), int(self.ENEMY_SIZE*1.8), self.COLOR_ENEMY_GLOW)
            self.screen.blit(glow_surf, (pos_int[0] - self.ENEMY_SIZE*2, pos_int[1] - self.ENEMY_SIZE*2))
            # Body
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ENEMY_SIZE, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ENEMY_SIZE, self.COLOR_ENEMY)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            color_with_alpha = p['color'] + (alpha,)
            
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

    def _render_ui(self):
        score_text = self.font.render(f"ORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            msg = "MISSION COMPLETE" if self.score >= self.WIN_SCORE else "SHIP DESTROYED"
            msg_font = pygame.font.SysFont("monospace", 40, bold=True)
            msg_text = msg_font.render(msg, True, self.COLOR_PLAYER if self.score >= self.WIN_SCORE else self.COLOR_ENEMY)
            text_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_text, text_rect)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Change to 'windows' or 'dummy' as needed

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Space Miner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # None
        space_held = 0
        shift_held = 0
        
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
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30)
        
    pygame.quit()