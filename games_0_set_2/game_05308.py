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
        "Controls: Arrow keys to move. Hold Space to mine nearby asteroids. Avoid the red patrol ships."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship, collect minerals from asteroids, and evade enemy patrols in this top-down arcade space miner. Collect 50 minerals to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 70)
    COLOR_ASTEROID = (120, 120, 130)
    COLOR_MINERAL = (255, 220, 0)
    COLOR_BEAM = (255, 255, 100, 150)
    COLOR_EXPLOSION = [(255, 200, 0), (255, 100, 0), (255, 50, 0)]
    COLOR_UI_TEXT = (220, 220, 240)
    
    # Game parameters
    PLAYER_ACCELERATION = 0.6
    PLAYER_FRICTION = 0.94
    PLAYER_MAX_SPEED = 6
    PLAYER_RADIUS = 12
    
    ENEMY_BASE_SPEED = 1.0
    ENEMY_RADIUS = 10
    
    MINING_RANGE = 100
    MINING_RATE = 4  # minerals per second
    
    WIN_MINERALS = 50
    MAX_STEPS = 1800 # 60 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_minerals = None
        self.enemies = []
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.mining_beam_target = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # The environment must be reset before validation to initialize the state
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0, 0], dtype=np.float32)
        self.player_minerals = 0
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mining_beam_target = None
        
        self.enemies = self._spawn_enemies(3)
        self.asteroids = self._spawn_asteroids(15)
        self.particles = []
        self._init_stars(200)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_held, _ = action
        
        # Update game logic
        self._update_player(movement)
        self._update_enemies()
        
        reward = -0.01  # Time penalty
        
        # Mining logic
        self.mining_beam_target = None
        if space_held == 1:
            reward += self._handle_mining()
        
        self._update_particles()
        
        # Check for game end conditions
        if self._check_collisions():
            self.game_over = True
            reward = -100.0
            self._create_explosion(self.player_pos, 50)
            # sfx: player_explosion.wav
        elif self.player_minerals >= self.WIN_MINERALS:
            self.game_over = True
            reward = 100.0
        
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        terminated = self.game_over or truncated

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_stars()
        self._render_asteroids()
        self._render_enemies()
        self._render_player()
        self._render_particles()
        self._render_mining_beam()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "minerals": self.player_minerals,
        }
        
    def _wrap_position(self, pos):
        pos[0] %= self.SCREEN_WIDTH
        pos[1] %= self.SCREEN_HEIGHT
        return pos
        
    # --- Update Logic ---

    def _update_player(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: self.player_vel[1] -= self.PLAYER_ACCELERATION
        if movement == 2: self.player_vel[1] += self.PLAYER_ACCELERATION
        if movement == 3: self.player_vel[0] -= self.PLAYER_ACCELERATION
        if movement == 4: self.player_vel[0] += self.PLAYER_ACCELERATION
        
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED
            
        self.player_vel *= self.PLAYER_FRICTION
        self.player_pos += self.player_vel
        self.player_pos = self._wrap_position(self.player_pos)

    def _update_enemies(self):
        speed_multiplier = 1.0 + (self.player_minerals // 10) * 0.05
        current_speed = self.ENEMY_BASE_SPEED * speed_multiplier
        
        for enemy in self.enemies:
            enemy['angle'] += enemy['angular_velocity'] * current_speed
            enemy['pos'][0] = enemy['patrol_center'][0] + math.cos(enemy['angle']) * enemy['patrol_radius']
            enemy['pos'][1] = enemy['patrol_center'][1] + math.sin(enemy['angle']) * enemy['patrol_radius']
            enemy['pos'] = self._wrap_position(enemy['pos'])

    def _handle_mining(self):
        reward = 0
        
        # Find closest minable asteroid
        closest_asteroid = None
        min_dist_sq = self.MINING_RANGE ** 2
        
        for asteroid in self.asteroids:
            if asteroid['minerals'] > 0:
                dist_sq = self._toroidal_distance_sq(self.player_pos, asteroid['pos'])
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_asteroid = asteroid
        
        if closest_asteroid:
            self.mining_beam_target = closest_asteroid['pos']
            # sfx: mining_laser_loop.wav
            
            # Mine minerals based on a cooldown
            mine_cooldown_frames = self.FPS / self.MINING_RATE
            if self.steps % int(mine_cooldown_frames) == 0:
                mined_amount = 1
                closest_asteroid['minerals'] -= mined_amount
                self.player_minerals += mined_amount
                reward += 1.0
                
                # sfx: mineral_collect.wav
                self._create_mineral_particle(closest_asteroid['pos'])
                
                if closest_asteroid['minerals'] <= 0:
                    closest_asteroid['minerals'] = 0
                    if closest_asteroid['is_large']:
                        reward += 5.0 # Bonus for depleting a large asteroid
                    else:
                        # Respawn small asteroid elsewhere
                        self.asteroids.remove(closest_asteroid)
                        self.asteroids.extend(self._spawn_asteroids(1))
        
        return reward

    def _check_collisions(self):
        for enemy in self.enemies:
            dist_sq = self._toroidal_distance_sq(self.player_pos, enemy['pos'])
            if dist_sq < (self.PLAYER_RADIUS + self.ENEMY_RADIUS) ** 2:
                return True
        return False
        
    # --- Spawning and Initialization ---
    
    def _spawn_enemies(self, count):
        enemies = []
        for _ in range(count):
            patrol_center = self.np_random.uniform(low=0, high=[self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
            enemies.append({
                'pos': np.array([0,0], dtype=np.float32),
                'patrol_center': patrol_center,
                'patrol_radius': self.np_random.uniform(50, 150),
                'angle': self.np_random.uniform(0, 2 * math.pi),
                'angular_velocity': self.np_random.choice([-1, 1]) * self.np_random.uniform(0.02, 0.04)
            })
        return enemies
        
    def _spawn_asteroids(self, count):
        asteroids = []
        for _ in range(count):
            size = self.np_random.uniform(10, 30)
            is_large = size > 20
            minerals = int(size * (2 if is_large else 1))
            
            pos = self.np_random.uniform(low=0, high=[self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
            
            num_vertices = self.np_random.integers(7, 12)
            vertices = []
            angle_step = 2 * math.pi / num_vertices
            for i in range(num_vertices):
                angle = i * angle_step + self.np_random.uniform(-0.1, 0.1)
                radius = size + self.np_random.uniform(-size * 0.2, size * 0.2)
                vertices.append((math.cos(angle) * radius, math.sin(angle) * radius))
            
            asteroids.append({
                'pos': np.array(pos, dtype=np.float32),
                'size': size,
                'minerals': minerals,
                'max_minerals': minerals,
                'is_large': is_large,
                'vertices': vertices,
                'angle': self.np_random.uniform(0, 2 * math.pi),
                'rot_speed': self.np_random.uniform(-0.01, 0.01)
            })
        return asteroids
        
    def _init_stars(self, count):
        self.stars = []
        for _ in range(count):
            self.stars.append({
                'pos': np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)]),
                'depth': self.np_random.uniform(0.1, 0.6) # Determines parallax speed and brightness
            })
            
    # --- Particle System ---

    def _create_explosion(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.integers(20, 40)
            color = self.COLOR_EXPLOSION[self.np_random.integers(0, len(self.COLOR_EXPLOSION))]
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'color': color, 'type': 'explosion'})
            
    def _create_mineral_particle(self, start_pos):
        lifespan = 30
        self.particles.append({'pos': start_pos.copy(), 'vel': np.zeros(2), 'lifespan': lifespan, 'max_lifespan': lifespan, 'type': 'mineral'})

    def _update_particles(self):
        for p in self.particles[:]:
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
                continue

            if p['type'] == 'explosion':
                p['pos'] += p['vel']
                p['vel'] *= 0.95 # friction
            elif p['type'] == 'mineral':
                # Move towards player
                target_vec = self.player_pos - p['pos']
                dist = np.linalg.norm(target_vec)
                if dist > 1:
                    target_vec /= dist
                
                # Simple interpolation towards player
                progress = 1.0 - (p['lifespan'] / p['max_lifespan'])
                p['pos'] += target_vec * 10 * progress
        
    # --- Rendering ---
    
    def _render_stars(self):
        # Move stars based on player velocity for parallax effect
        for star in self.stars:
            star['pos'] -= self.player_vel * star['depth']
            star['pos'] = self._wrap_position(star['pos'])
            
            brightness = int(200 * star['depth'])
            color = (brightness, brightness, brightness)
            size = int(star['depth'] * 2)
            pygame.draw.rect(self.screen, color, (int(star['pos'][0]), int(star['pos'][1]), size, size))
            
    def _render_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['angle'] += asteroid['rot_speed']
            points = []
            for vx, vy in asteroid['vertices']:
                # Rotate vertex
                rotated_x = vx * math.cos(asteroid['angle']) - vy * math.sin(asteroid['angle'])
                rotated_y = vx * math.sin(asteroid['angle']) + vy * math.cos(asteroid['angle'])
                # Translate to position
                points.append((int(asteroid['pos'][0] + rotated_x), int(asteroid['pos'][1] + rotated_y)))
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
                
                # Mineral sheen for non-depleted asteroids
                if asteroid['minerals'] > 0:
                    sheen_color = list(self.COLOR_MINERAL)
                    sheen_color.append(int(30 * (asteroid['minerals'] / asteroid['max_minerals'])))
                    pygame.gfxdraw.filled_polygon(self.screen, points, tuple(sheen_color))

    def _render_enemies(self):
        for enemy in self.enemies:
            # Glow effect
            glow_surf = pygame.Surface((self.ENEMY_RADIUS*4, self.ENEMY_RADIUS*4), pygame.SRCALPHA)
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            radius = int(self.ENEMY_RADIUS * 2 + pulse * 5)
            pygame.draw.circle(glow_surf, self.COLOR_ENEMY_GLOW, (self.ENEMY_RADIUS*2, self.ENEMY_RADIUS*2), radius)
            self.screen.blit(glow_surf, (int(enemy['pos'][0] - self.ENEMY_RADIUS*2), int(enemy['pos'][1] - self.ENEMY_RADIUS*2)))

            # Main body
            p1 = (enemy['pos'][0] + self.ENEMY_RADIUS, enemy['pos'][1])
            p2 = (enemy['pos'][0] - self.ENEMY_RADIUS/2, enemy['pos'][1] - self.ENEMY_RADIUS*0.866)
            p3 = (enemy['pos'][0] - self.ENEMY_RADIUS/2, enemy['pos'][1] + self.ENEMY_RADIUS*0.866)
            points = [p1, p2, p3]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

    def _render_player(self):
        if self.game_over: return
        
        # Glow effect
        glow_surf = pygame.Surface((self.PLAYER_RADIUS*4, self.PLAYER_RADIUS*4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2), self.PLAYER_RADIUS*2)
        self.screen.blit(glow_surf, (int(self.player_pos[0] - self.PLAYER_RADIUS*2), int(self.player_pos[1] - self.PLAYER_RADIUS*2)))
        
        # Ship body
        angle = math.atan2(self.player_vel[1], self.player_vel[0]) if np.linalg.norm(self.player_vel) > 0.1 else -math.pi/2
        
        p1 = (self.player_pos[0] + math.cos(angle) * self.PLAYER_RADIUS, self.player_pos[1] + math.sin(angle) * self.PLAYER_RADIUS)
        p2 = (self.player_pos[0] + math.cos(angle + 2.5) * self.PLAYER_RADIUS * 0.8, self.player_pos[1] + math.sin(angle + 2.5) * self.PLAYER_RADIUS * 0.8)
        p3 = (self.player_pos[0] + math.cos(angle - 2.5) * self.PLAYER_RADIUS * 0.8, self.player_pos[1] + math.sin(angle - 2.5) * self.PLAYER_RADIUS * 0.8)
        points = [p1,p2,p3]
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            if p['type'] == 'explosion':
                alpha = 255 * (p['lifespan'] / 40)
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((10, 10), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (5, 5), 5)
                self.screen.blit(temp_surf, (int(p['pos'][0]-5), int(p['pos'][1]-5)), special_flags=pygame.BLEND_RGBA_ADD)
            elif p['type'] == 'mineral':
                pygame.draw.circle(self.screen, self.COLOR_MINERAL, (int(p['pos'][0]), int(p['pos'][1])), 3)

    def _render_mining_beam(self):
        if self.mining_beam_target is not None and not self.game_over:
            start_pos = (int(self.player_pos[0]), int(self.player_pos[1]))
            end_pos = (int(self.mining_beam_target[0]), int(self.mining_beam_target[1]))
            
            # Create a list of points for a thick, jittery line
            num_segments = 10
            points = [start_pos]
            for i in range(1, num_segments):
                t = i / num_segments
                x = start_pos[0] * (1-t) + end_pos[0] * t + self.np_random.uniform(-3, 3)
                y = start_pos[1] * (1-t) + end_pos[1] * t + self.np_random.uniform(-3, 3)
                points.append((x, y))
            points.append(end_pos)
            
            pygame.draw.aalines(self.screen, self.COLOR_BEAM, False, points, 1)

    def _render_ui(self):
        mineral_text = f"MINERALS: {self.player_minerals}/{self.WIN_MINERALS}"
        text_surface = self.font.render(mineral_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 10))

    # --- Utility ---
    def _toroidal_distance_sq(self, pos1, pos2):
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        dx = min(dx, self.SCREEN_WIDTH - dx)
        dy = min(dy, self.SCREEN_HEIGHT - dy)
        
        return dx**2 + dy**2

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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment.
    # To see the game, comment out the os.environ line at the top.
    try:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        # --- Pygame setup for human play ---
        pygame.display.set_caption("Space Miner")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        print("\n" + "="*30)
        print(GameEnv.game_description)
        print(GameEnv.user_guide)
        print("="*30 + "\n")

        while running:
            # --- Action mapping from keyboard to MultiDiscrete ---
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            # --- Gym step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # --- Pygame rendering ---
            # The observation is already a rendered frame, so we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}, Steps: {info['steps']}, Minerals: {info['minerals']}")
                total_reward = 0
                obs, info = env.reset()
                # Add a small delay to let the player see the end state
                pygame.time.wait(2000)

            # --- Event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    total_reward = 0
                    obs, info = env.reset()

        env.close()
    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("This might be because the environment is running in headless mode.")
        print("To play visually, comment out the `os.environ` line at the top of the file.")