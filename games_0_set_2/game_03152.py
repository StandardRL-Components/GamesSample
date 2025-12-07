
# Generated: 2025-08-27T22:31:39.115508
# Source Brief: brief_03152.md
# Brief Index: 3152

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to mine asteroids. Hold Shift to boost (consumes ore)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship through an asteroid field, collecting valuable ore while dodging dangerous enemy patrols."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 50)
    COLOR_ENEMY_1 = (255, 50, 50)
    COLOR_ENEMY_2 = (255, 120, 50)
    COLOR_ENEMY_3 = (255, 200, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 60)
    COLOR_ASTEROID = (120, 130, 140)
    COLOR_ORE = (255, 220, 0)
    COLOR_ORE_GLOW = (255, 220, 0, 80)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR = (50, 200, 50)
    COLOR_HEALTH_BAR_BG = (150, 50, 50)
    COLOR_BOOST_TRAIL = (100, 200, 255)

    # Game Parameters
    WIDTH, HEIGHT = 640, 400
    PLAYER_ACCELERATION = 0.4
    PLAYER_FRICTION = 0.96
    PLAYER_MAX_HEALTH = 3
    PLAYER_INVINCIBILITY_FRAMES = 60
    BOOST_MULTIPLIER = 1.8
    BOOST_ORE_COST = 1
    BOOST_COST_INTERVAL = 10
    WIN_ORE_COUNT = 50
    MAX_STEPS = 1500
    INITIAL_ASTEROIDS = 12
    MIN_ASTEROID_ORE = 50
    MAX_ASTEROID_ORE = 150
    MINING_RATE = 2
    MINING_RANGE = 60
    INITIAL_ENEMIES = 4
    ENEMY_BASE_SPEED_MIN = 0.6
    ENEMY_BASE_SPEED_MAX = 1.2
    ENEMY_SPEED_INCREASE_INTERVAL = 200
    ENEMY_SPEED_INCREASE_AMOUNT = 0.05
    ENEMY_MAX_SPEED = 3.0
    STAR_COUNT = 150

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.stars = [
            (
                random.randint(0, self.WIDTH),
                random.randint(0, self.HEIGHT),
                random.uniform(0.2, 0.8), # parallax speed factor
                random.randint(1, 2) # size
            ) for _ in range(self.STAR_COUNT)
        ]
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = 0
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ore = 0
        self.invincibility_timer = 0
        
        self.boost_active = False
        self.boost_cost_timer = 0
        
        self.asteroids = []
        for _ in range(self.INITIAL_ASTEROIDS):
            self._spawn_asteroid()

        self.enemies = []
        for _ in range(self.INITIAL_ENEMIES):
            self._spawn_enemy()
            
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01  # Cost of living
        
        self._handle_input(movement, shift_held)
        self._update_player()
        
        if space_held:
            reward += self._handle_mining()

        self._update_enemies()
        self._update_particles()
        self._update_asteroids()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        self.steps += 1
        self.score += reward
        
        terminated = self.player_health <= 0 or self.player_ore >= self.WIN_ORE_COUNT or self.steps >= self.MAX_STEPS
        
        if terminated:
            self.game_over = True
            if self.player_ore >= self.WIN_ORE_COUNT:
                self.win = True
                reward += 100
                self.score += 100
            elif self.player_health <= 0:
                reward -= 100
                self.score -= 100
        
        # Scale enemy speed
        if self.steps > 0 and self.steps % self.ENEMY_SPEED_INCREASE_INTERVAL == 0:
            for enemy in self.enemies:
                enemy['speed'] = min(self.ENEMY_MAX_SPEED, enemy['speed'] + self.ENEMY_SPEED_INCREASE_AMOUNT)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, shift_held):
        # Boosting
        self.boost_active = shift_held and self.player_ore >= self.BOOST_ORE_COST
        
        # Movement
        acceleration = pygame.math.Vector2(0, 0)
        if movement == 1: acceleration.y = -1
        elif movement == 2: acceleration.y = 1
        elif movement == 3: acceleration.x = -1
        elif movement == 4: acceleration.x = 1
        
        if acceleration.length() > 0:
            acceleration.normalize_ip()
            acceleration *= self.PLAYER_ACCELERATION
            if self.boost_active:
                acceleration *= self.BOOST_MULTIPLIER
        
        self.player_vel += acceleration

    def _update_player(self):
        # Friction
        self.player_vel *= self.PLAYER_FRICTION
        if self.player_vel.length() < 0.1:
            self.player_vel.x, self.player_vel.y = 0, 0

        # Move
        self.player_pos += self.player_vel
        
        # Boost cost and effect
        if self.boost_active:
            self.boost_cost_timer += 1
            if self.boost_cost_timer >= self.BOOST_COST_INTERVAL:
                self.player_ore = max(0, self.player_ore - self.BOOST_ORE_COST)
                self.boost_cost_timer = 0
            # sfx: boost_sound_loop()
            self._create_particles(1, self.player_pos, self.COLOR_BOOST_TRAIL, 2, 20, -self.player_vel * 0.5)
        
        # Update angle for rendering
        if self.player_vel.length() > 0.1:
            self.player_angle = self.player_vel.angle_to(pygame.math.Vector2(1, 0))

        # Keep on screen
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.HEIGHT)
        
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

    def _handle_mining(self):
        reward = 0
        closest_asteroid = None
        min_dist = float('inf')

        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
                closest_asteroid = asteroid
        
        if closest_asteroid and min_dist < self.MINING_RANGE + closest_asteroid['size']:
            # sfx: mining_laser_sound()
            mined_amount = self.MINING_RATE
            closest_asteroid['ore'] -= mined_amount
            
            # Create ore particle stream
            for _ in range(2):
                start_pos = pygame.math.Vector2(closest_asteroid['pos'])
                particle_vel = (self.player_pos - start_pos).normalize() * 3
                self._create_particles(1, start_pos, self.COLOR_ORE, 3, 40, particle_vel, particle_type='ore')

            if closest_asteroid['ore'] <= 0:
                # sfx: asteroid_destroyed_sound()
                self.asteroids.remove(closest_asteroid)
                reward += 1.0 # Reward for destroying an asteroid
        return reward

    def _update_enemies(self):
        for enemy in self.enemies:
            pattern = enemy['pattern']
            if pattern == 'circular':
                enemy['state']['angle'] += enemy['speed'] * 0.02
                offset_x = math.cos(enemy['state']['angle']) * enemy['state']['radius']
                offset_y = math.sin(enemy['state']['angle']) * enemy['state']['radius']
                enemy['pos'] = enemy['state']['center'] + pygame.math.Vector2(offset_x, offset_y)
            
            elif pattern == 'back-and-forth':
                target = enemy['state']['points'][enemy['state']['target_idx']]
                direction = (target - enemy['pos'])
                if direction.length() < enemy['speed']:
                    enemy['state']['target_idx'] = 1 - enemy['state']['target_idx']
                else:
                    enemy['pos'] += direction.normalize() * enemy['speed']

            elif pattern == 'random_walk':
                if self.np_random.random() < 0.02 or enemy['state']['target'].distance_to(enemy['pos']) < 20:
                    enemy['state']['target'] = pygame.math.Vector2(self.np_random.integers(50, self.WIDTH-50), self.np_random.integers(50, self.HEIGHT-50))
                direction = (enemy['state']['target'] - enemy['pos']).normalize()
                enemy['pos'] += direction * enemy['speed']

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_asteroids(self):
        # Respawn asteroids
        if len(self.asteroids) < self.INITIAL_ASTEROIDS and self.np_random.random() < 0.01:
            self._spawn_asteroid()

    def _handle_collisions(self):
        reward = 0
        # Player vs Enemy
        if self.invincibility_timer == 0:
            for enemy in self.enemies:
                if self.player_pos.distance_to(enemy['pos']) < 20: # 10 for player, 10 for enemy
                    # sfx: player_hit_sound()
                    self.player_health -= 1
                    self.invincibility_timer = self.PLAYER_INVINCIBILITY_FRAMES
                    self.player_vel += (self.player_pos - enemy['pos']).normalize() * 5 # Knockback
                    self._create_particles(30, self.player_pos, self.COLOR_ENEMY_1, 4, 30, spread=3.0)
                    break
        
        # Player vs Ore Particles
        for p in self.particles[:]:
            if p['type'] == 'ore' and self.player_pos.distance_to(p['pos']) < 20:
                # sfx: ore_collect_sound()
                self.player_ore = min(self.WIN_ORE_COUNT, self.player_ore + 1)
                self.particles.remove(p)
                reward += 0.1
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, speed, size in self.stars:
            px = (x - self.player_vel.x * speed) % self.WIDTH
            py = (y - self.player_vel.y * speed) % self.HEIGHT
            pygame.draw.rect(self.screen, (255, 255, 255), (px, py, size, size))

        # Asteroids
        for asteroid in self.asteroids:
            asteroid['angle'] = (asteroid['angle'] + asteroid['rot_speed']) % 360
            points = []
            for i in range(asteroid['num_points']):
                angle = math.radians(360 / asteroid['num_points'] * i + asteroid['angle'])
                x = asteroid['pos'].x + math.cos(angle) * asteroid['size'] * asteroid['shape'][i]
                y = asteroid['pos'].y + math.sin(angle) * asteroid['size'] * asteroid['shape'][i]
                points.append((int(x), int(y)))
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            color = [self.COLOR_ENEMY_1, self.COLOR_ENEMY_2, self.COLOR_ENEMY_3][enemy['type']-1]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, color)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = p['color'] + (alpha,)
            pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color)

        # Player
        if self.player_health > 0:
            player_alpha = 255 if self.invincibility_timer % 10 < 5 else 100
            player_color = self.COLOR_PLAYER
            
            # Glow
            glow_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (30, 30), 30)
            self.screen.blit(glow_surf, (int(self.player_pos.x - 30), int(self.player_pos.y - 30)), special_flags=pygame.BLEND_RGBA_ADD)

            # Ship body
            ship_points_base = [(-10, -8), (12, 0), (-10, 8)]
            ship_points_rotated = []
            angle_rad = math.radians(-self.player_angle)
            for x, y in ship_points_base:
                rx = x * math.cos(angle_rad) - y * math.sin(angle_rad) + self.player_pos.x
                ry = x * math.sin(angle_rad) + y * math.cos(angle_rad) + self.player_pos.y
                ship_points_rotated.append((int(rx), int(ry)))
            
            final_color = (player_color[0], player_color[1], player_color[2], player_alpha)
            pygame.gfxdraw.aapolygon(self.screen, ship_points_rotated, final_color)
            pygame.gfxdraw.filled_polygon(self.screen, ship_points_rotated, final_color)

    def _render_ui(self):
        # Ore count
        ore_text = self.font_ui.render(f"ORE: {self.player_ore}/{self.WIN_ORE_COUNT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ore_text, (10, 10))

        # Health bar
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width = 100
        bar_height = 15
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (self.WIDTH - bar_width - 10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (self.WIDTH - bar_width - 10, 10, bar_width * health_ratio, bar_height))

        # Game Over / Win Message
        if self.game_over:
            msg_text = "MISSION COMPLETE" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ENEMY_1
            rendered_text = self.font_msg.render(msg_text, True, color)
            text_rect = rendered_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(rendered_text, text_rect)
            
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "ore": self.player_ore, "health": self.player_health}

    def _spawn_asteroid(self):
        pos = pygame.math.Vector2(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT))
        # Ensure it doesn't spawn on the player
        while pos.distance_to(self.player_pos) < 100:
            pos = pygame.math.Vector2(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT))
        
        ore = self.np_random.integers(self.MIN_ASTEROID_ORE, self.MAX_ASTEROID_ORE + 1)
        size = int(15 + 15 * (ore / self.MAX_ASTEROID_ORE))
        num_points = self.np_random.integers(7, 12)
        shape_factors = [self.np_random.uniform(0.8, 1.2) for _ in range(num_points)]

        self.asteroids.append({
            'pos': pos, 'ore': ore, 'size': size, 'angle': self.np_random.uniform(0, 360),
            'rot_speed': self.np_random.uniform(-0.5, 0.5), 'num_points': num_points, 'shape': shape_factors
        })

    def _spawn_enemy(self):
        pos = pygame.math.Vector2(self.np_random.choice([0, self.WIDTH]), self.np_random.integers(0, self.HEIGHT))
        while pos.distance_to(self.player_pos) < 150:
            pos = pygame.math.Vector2(self.np_random.choice([0, self.WIDTH]), self.np_random.integers(0, self.HEIGHT))

        enemy_type = self.np_random.integers(1, 4)
        speed = self.np_random.uniform(self.ENEMY_BASE_SPEED_MIN, self.ENEMY_BASE_SPEED_MAX) * (0.8 + 0.2 * enemy_type)
        
        pattern_choice = self.np_random.choice(['circular', 'back-and-forth', 'random_walk'])
        state = {}
        if pattern_choice == 'circular':
            state = {'center': pygame.math.Vector2(self.np_random.integers(100, self.WIDTH-100), self.np_random.integers(100, self.HEIGHT-100)),
                     'radius': self.np_random.integers(50, 150), 'angle': self.np_random.uniform(0, 2*math.pi)}
        elif pattern_choice == 'back-and-forth':
            p1 = pygame.math.Vector2(self.np_random.integers(50, self.WIDTH-50), self.np_random.integers(50, self.HEIGHT-50))
            p2 = pygame.math.Vector2(self.np_random.integers(50, self.WIDTH-50), self.np_random.integers(50, self.HEIGHT-50))
            state = {'points': [p1, p2], 'target_idx': 1}
        elif pattern_choice == 'random_walk':
            state = {'target': pygame.math.Vector2(self.np_random.integers(50, self.WIDTH-50), self.np_random.integers(50, self.HEIGHT-50))}
        
        self.enemies.append({'pos': pos, 'type': enemy_type, 'speed': speed, 'pattern': pattern_choice, 'state': state})

    def _create_particles(self, num, pos, color, size, life, vel=None, spread=1.0, particle_type='generic'):
        for _ in range(num):
            if vel:
                p_vel = pygame.math.Vector2(vel)
                p_vel.x += self.np_random.uniform(-spread, spread)
                p_vel.y += self.np_random.uniform(-spread, spread)
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(0.5, 2.0) * spread
                p_vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            
            self.particles.append({
                'pos': pygame.math.Vector2(pos), 'vel': p_vel, 'color': color, 'size': size, 'life': life, 'max_life': life, 'type': particle_type
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Use a dummy window to display the game from the rgb_array
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Get player input from keyboard
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
        
        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle game over
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            
        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Cap the frame rate
        env.clock.tick(30)
        
    env.close()