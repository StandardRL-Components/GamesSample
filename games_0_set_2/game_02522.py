
# Generated: 2025-08-27T20:37:25.735490
# Source Brief: brief_02522.md
# Brief Index: 2522

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Arrow keys to move. Hold Space to mine nearby asteroids."

    # Must be a short, user-facing description of the game:
    game_description = "Pilot a mining ship through an asteroid field, extracting ore while avoiding collisions to amass a fortune."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.SCREEN_CENTER = np.array([self.WIDTH / 2, self.HEIGHT / 2])

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
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 30)
        self.COLOR_ASTEROID = (160, 160, 170)
        self.COLOR_ORE = (255, 220, 50)
        self.COLOR_BEAM = (255, 255, 255, 150)
        self.COLOR_EXPLOSION = (255, 100, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR_BG = (80, 0, 0)
        self.COLOR_HEALTH_BAR = (220, 0, 0)
        self.COLOR_WIN = (50, 255, 150)
        self.COLOR_LOSE = (220, 0, 0)

        # Game constants
        self.MAX_STEPS = 2000
        self.WIN_ORE = 100
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_ACCELERATION = 0.4
        self.PLAYER_MAX_SPEED = 4.0
        self.PLAYER_DRAG = 0.96
        self.PLAYER_RADIUS = 12
        self.MINING_RANGE = 120
        self.MINING_RATE = 0.5
        self.ASTEROID_MIN_SIZE = 15
        self.ASTEROID_MAX_SIZE = 45
        self.INITIAL_ASTEROIDS = 15
        self.SPAWN_DISTANCE = 500

        # State variables will be initialized in reset()
        self.steps = 0
        self.ore_collected = 0
        self.game_over = False
        self.player_health = 0
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_angle = 0.0
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.spawn_rate = 0.0
        self.np_random = None
        self.collision_cooldowns = {}

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.ore_collected = 0
        self.game_over = False
        self.player_health = self.PLAYER_MAX_HEALTH
        
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_angle = -math.pi / 2

        self.asteroids = []
        self.particles = []
        self.collision_cooldowns = {}
        self.spawn_rate = 0.01

        self.stars = []
        for _ in range(200):
            self.stars.append({
                'pos': self.np_random.random(2) * np.array([self.WIDTH, self.HEIGHT]),
                'depth': 0.1 + self.np_random.random() * 0.9,
            })

        for i in range(self.INITIAL_ASTEROIDS):
            self._spawn_asteroid(i, is_initial=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = -0.05  # Penalty for each step

        if not self.game_over:
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            self._handle_input(movement)
            self._update_player()
            mining_reward = self._update_mining(space_held)
            collision_reward = self._handle_collisions()
            
            reward += mining_reward + collision_reward
            
            self._update_particles()
            self._spawn_new_asteroids()
            self._update_difficulty()

        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.ore_collected >= self.WIN_ORE:
                reward += 100
            elif self.player_health <= 0:
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement):
        direction_vector = np.array([0.0, 0.0])
        if movement == 1: direction_vector[1] = -1 # Up
        elif movement == 2: direction_vector[1] = 1  # Down
        elif movement == 3: direction_vector[0] = -1 # Left
        elif movement == 4: direction_vector[0] = 1  # Right

        if np.linalg.norm(direction_vector) > 0:
            self.player_vel += direction_vector * self.PLAYER_ACCELERATION
            self.player_angle = math.atan2(direction_vector[1], direction_vector[0])
            # Engine flare particles
            if self.steps % 3 == 0:
                self._create_particles(1, self.player_pos, 'engine_flare', angle_offset=math.pi)

    def _update_player(self):
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED
        
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_DRAG

    def _update_mining(self, space_held):
        reward = 0
        if not space_held:
            return reward

        target_asteroid = None
        min_dist = self.MINING_RANGE
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos']) - asteroid['size']
            if dist < min_dist:
                min_dist = dist
                target_asteroid = asteroid
        
        if target_asteroid:
            # sound: mining_beam.wav
            mined_amount = min(self.MINING_RATE, target_asteroid['ore'])
            target_asteroid['ore'] -= mined_amount
            target_asteroid['size'] = self.ASTEROID_MIN_SIZE + (target_asteroid['ore'] / target_asteroid['max_ore']) * (self.ASTEROID_MAX_SIZE - self.ASTEROID_MIN_SIZE)
            
            self.ore_collected += mined_amount
            reward += mined_amount * 0.1
            
            if self.steps % 2 == 0:
                self._create_particles(1, target_asteroid['pos'], 'ore', target_pos=self.player_pos)

            if target_asteroid['ore'] <= 0:
                # sound: asteroid_depleted.wav
                self.asteroids.remove(target_asteroid)
                reward += 1

        return reward

    def _handle_collisions(self):
        reward = 0
        cooldown_keys_to_remove = []
        for key in self.collision_cooldowns:
            self.collision_cooldowns[key] -= 1
            if self.collision_cooldowns[key] <= 0:
                cooldown_keys_to_remove.append(key)
        for key in cooldown_keys_to_remove:
            del self.collision_cooldowns[key]

        for asteroid in self.asteroids:
            if asteroid['id'] in self.collision_cooldowns:
                continue

            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['size']:
                # sound: collision.wav
                damage = asteroid['size'] * 0.5
                self.player_health = max(0, self.player_health - damage)
                reward -= 5
                
                self._create_particles(int(asteroid['size']), asteroid['pos'], 'explosion')
                
                # Apply impulse
                collision_normal = (self.player_pos - asteroid['pos']) / dist
                self.player_vel += collision_normal * 3.0
                
                self.collision_cooldowns[asteroid['id']] = 30 # 1 second cooldown
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['lifespan'] -= 1
            if p['type'] == 'ore':
                target_dir = p['target_pos'] - p['pos']
                dist = np.linalg.norm(target_dir)
                if dist > 1:
                    p['vel'] = target_dir / dist * 5.0
                else:
                    p['lifespan'] = 0 # Reached target
            p['pos'] += p['vel']
            
    def _create_particles(self, count, pos, p_type, target_pos=None, angle_offset=0.0):
        for _ in range(count):
            if p_type == 'explosion':
                angle = self.np_random.random() * 2 * math.pi
                speed = 1 + self.np_random.random() * 3
                vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                lifespan = 20 + self.np_random.integers(10)
                color = self.COLOR_EXPLOSION
            elif p_type == 'ore':
                vel = np.array([0.0, 0.0])
                lifespan = 60
                color = self.COLOR_ORE
            elif p_type == 'engine_flare':
                angle = self.player_angle + angle_offset + (self.np_random.random() - 0.5) * 0.5
                speed = 1 + self.np_random.random() * 2
                vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                lifespan = 10 + self.np_random.integers(5)
                color = self.COLOR_ORE
            
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'color': color, 'type': p_type, 'target_pos': target_pos
            })

    def _spawn_new_asteroids(self):
        if self.np_random.random() < self.spawn_rate:
            new_id = max([a['id'] for a in self.asteroids] + [-1]) + 1
            self._spawn_asteroid(new_id)

    def _spawn_asteroid(self, asteroid_id, is_initial=False):
        size = self.np_random.integers(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE + 1)
        ore = size * 2.0
        
        if is_initial:
            angle = self.np_random.random() * 2 * math.pi
            dist = 100 + self.np_random.random() * self.SPAWN_DISTANCE
        else:
            angle = self.np_random.random() * 2 * math.pi
            dist = self.SPAWN_DISTANCE
            
        pos = self.player_pos + np.array([math.cos(angle) * dist, math.sin(angle) * dist])
        
        num_points = 8
        shape_points = []
        for i in range(num_points):
            angle_p = 2 * math.pi * i / num_points
            radius_p = size * (0.8 + self.np_random.random() * 0.4)
            shape_points.append(np.array([math.cos(angle_p) * radius_p, math.sin(angle_p) * radius_p]))

        self.asteroids.append({
            'id': asteroid_id,
            'pos': pos,
            'size': float(size),
            'ore': ore,
            'max_ore': ore,
            'shape_points': shape_points
        })

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 500 == 0:
            self.spawn_rate = min(0.1, self.spawn_rate + 0.01)

    def _check_termination(self):
        return (self.player_health <= 0 or
                self.ore_collected >= self.WIN_ORE or
                self.steps >= self.MAX_STEPS)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": int(self.ore_collected), "steps": self.steps}

    def _render_game(self):
        camera_offset = self.player_pos - self.SCREEN_CENTER

        # Render stars with parallax
        for star in self.stars:
            star_screen_pos = (star['pos'] - camera_offset * star['depth'])
            star_screen_pos[0] %= self.WIDTH
            star_screen_pos[1] %= self.HEIGHT
            pygame.draw.circle(self.screen, (200, 200, 255), star_screen_pos.astype(int), 1)

        # Render particles
        for p in self.particles:
            pos = (p['pos'] - camera_offset).astype(int)
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 20.0))))
            color = p['color'] + (alpha,) if len(p['color']) == 3 else p['color']
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, pos - np.array([2, 2]), special_flags=pygame.BLEND_RGBA_ADD)

        # Render mining beam
        if not self.game_over and self.action_space.sample()[1] == 1: # A bit of a hack to check if space is held
            target_asteroid = None
            min_dist = self.MINING_RANGE
            for asteroid in self.asteroids:
                dist = np.linalg.norm(self.player_pos - asteroid['pos']) - asteroid['size']
                if dist < min_dist:
                    min_dist = dist
                    target_asteroid = asteroid
            if target_asteroid:
                start_pos = self.SCREEN_CENTER
                end_pos = (target_asteroid['pos'] - camera_offset).astype(int)
                pygame.draw.aaline(self.screen, self.COLOR_BEAM, start_pos, end_pos, 1)

        # Render asteroids
        for asteroid in self.asteroids:
            points = [(p + asteroid['pos'] - camera_offset).astype(int) for p in asteroid['shape_points']]
            if len(points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
        
        # Render player
        player_points = [
            np.array([self.PLAYER_RADIUS, 0]),
            np.array([-self.PLAYER_RADIUS * 0.5, -self.PLAYER_RADIUS * 0.8]),
            np.array([-self.PLAYER_RADIUS * 0.5, self.PLAYER_RADIUS * 0.8])
        ]
        rotation_matrix = np.array([
            [math.cos(self.player_angle), -math.sin(self.player_angle)],
            [math.sin(self.player_angle), math.cos(self.player_angle)]
        ])
        rotated_points = [p @ rotation_matrix.T + self.SCREEN_CENTER for p in player_points]
        
        # Glow effect
        glow_surf = pygame.Surface((self.PLAYER_RADIUS*4, self.PLAYER_RADIUS*4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2), self.PLAYER_RADIUS*1.5)
        self.screen.blit(glow_surf, (self.SCREEN_CENTER - np.array([self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2])).astype(int), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.filled_polygon(self.screen, [p.astype(int) for p in rotated_points], self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, [p.astype(int) for p in rotated_points], self.COLOR_PLAYER)

    def _render_ui(self):
        # Health bar
        health_pct = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_pct), 20))
        health_text = self.font_ui.render("HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Ore collected
        ore_text = self.font_ui.render(f"ORE: {int(self.ore_collected)} / {self.WIN_ORE}", True, self.COLOR_UI_TEXT)
        text_rect = ore_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(ore_text, text_rect)
        
        # Game over text
        if self.game_over:
            if self.ore_collected >= self.WIN_ORE:
                text = self.font_game_over.render("VICTORY", True, self.COLOR_WIN)
            else:
                text = self.font_game_over.render("GAME OVER", True, self.COLOR_LOSE)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)

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
    # This block allows you to run the file directly to test the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    # Mapping from Pygame keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while running:
        # Default action is NO-OP
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        # Space and Shift
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            
    pygame.quit()