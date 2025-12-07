
# Generated: 2025-08-27T18:19:47.083059
# Source Brief: brief_01798.md
# Brief Index: 1798

        
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
        "Controls: Arrow keys to move your ship. Hold space to fire your mining laser at asteroids."
    )

    game_description = (
        "Pilot a mining ship in a dangerous asteroid field. Blast asteroids to collect ore while evading hostile patrol drones. Collect 100 ore to win, but be careful - one collision means game over."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    WIN_ORE = 100

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (60, 180, 255)
    COLOR_PLAYER_GLOW = (30, 90, 128)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (128, 25, 25)
    COLOR_ASTEROID = (150, 150, 150)
    COLOR_ORE = (255, 220, 50)
    COLOR_LASER = (255, 100, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_EXPLOSION = (255, 150, 50)

    # Player settings
    PLAYER_ACCEL = 0.4
    PLAYER_DRAG = 0.94
    PLAYER_MAX_SPEED = 5.0
    PLAYER_RADIUS = 12

    # Enemy settings
    INITIAL_ENEMIES = 5
    ENEMY_RADIUS = 10
    INITIAL_ENEMY_SPEED = 0.8
    ENEMY_SPEED_INCREASE_INTERVAL = 2000
    ENEMY_SPEED_INCREASE_AMOUNT = 0.05
    ENEMY_MAX_SPEED = 2.0

    # Asteroid settings
    ASTEROID_COUNT = 10
    ASTEROID_MIN_ORE = 20
    ASTEROID_MAX_ORE = 50

    # Mining settings
    MINING_RANGE = 120
    MINING_RATE = 1

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        self.game_over = False
        self.steps = 0
        self.score = 0
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.player_ore = 0
        self.win = False
        
        self.enemies = []
        self.asteroids = []
        self.ore_particles = []
        self.fx_particles = [] # For laser, explosions, etc.
        self.stars = []
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-check

    def _generate_stars(self, count=100):
        self.stars = []
        for _ in range(count):
            self.stars.append({
                "pos": self.np_random.random(2) * np.array([self.WIDTH, self.HEIGHT]),
                "brightness": self.np_random.uniform(50, 150),
            })

    def _spawn_asteroids(self, count):
        self.asteroids = []
        for _ in range(count):
            ore = self.np_random.integers(self.ASTEROID_MIN_ORE, self.ASTEROID_MAX_ORE + 1)
            self.asteroids.append({
                "pos": self.np_random.random(2) * np.array([self.WIDTH, self.HEIGHT]),
                "ore": ore,
                "max_ore": ore,
                "radius": 10 + ore * 0.3,
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "rotation_speed": self.np_random.uniform(-0.02, 0.02),
                "shape": self._generate_asteroid_shape(10 + ore * 0.3)
            })

    def _generate_asteroid_shape(self, radius):
        num_vertices = self.np_random.integers(7, 12)
        angles = np.linspace(0, 2 * math.pi, num_vertices, endpoint=False)
        noise = self.np_random.uniform(0.7, 1.1, num_vertices)
        return [(math.cos(a) * radius * n, math.sin(a) * radius * n) for a, n in zip(angles, noise)]

    def _spawn_enemies(self, count):
        self.enemies = []
        for _ in range(count):
            self.enemies.append({
                "center": self.np_random.random(2) * np.array([self.WIDTH, self.HEIGHT]),
                "path_radius": self.np_random.uniform(50, 150),
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "speed": self.INITIAL_ENEMY_SPEED
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.player_ore = 0
        self.game_over = False
        self.win = False

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0, 0], dtype=np.float32)

        self._generate_stars()
        self._spawn_asteroids(self.ASTEROID_COUNT)
        self._spawn_enemies(self.INITIAL_ENEMIES)

        self.ore_particles = []
        self.fx_particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01 # Small penalty for time passing
        
        # --- Handle Actions ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.player_vel[1] -= self.PLAYER_ACCEL
        if movement == 2: self.player_vel[1] += self.PLAYER_ACCEL
        if movement == 3: self.player_vel[0] -= self.PLAYER_ACCEL
        if movement == 4: self.player_vel[0] += self.PLAYER_ACCEL

        # --- Update Player ---
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED
        
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_DRAG

        # Screen wrap
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

        # --- Update Game Logic ---
        self._update_enemies()
        self._update_asteroids()
        
        if space_held:
            mined_something, mine_reward = self._handle_mining()
            reward += mine_reward
        
        collect_reward = self._update_ore_particles()
        reward += collect_reward
        
        self._update_fx_particles()

        # Respawn asteroids if all are mined
        if not self.asteroids:
            self._spawn_asteroids(self.ASTEROID_COUNT)

        # --- Check Collisions and Termination ---
        for enemy in self.enemies:
            pos = self._get_enemy_pos(enemy)
            if np.linalg.norm(self.player_pos - pos) < self.PLAYER_RADIUS + self.ENEMY_RADIUS:
                self.game_over = True
                reward -= 100
                self._create_explosion(self.player_pos, 50, self.PLAYER_RADIUS * 2)
                # Sound: Player explosion
                break
        
        if self.player_ore >= self.WIN_ORE:
            self.player_ore = self.WIN_ORE
            self.game_over = True
            self.win = True
            reward += 100

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        self.score += reward

        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _update_enemies(self):
        current_speed = self.INITIAL_ENEMY_SPEED + (self.steps // self.ENEMY_SPEED_INCREASE_INTERVAL) * self.ENEMY_SPEED_INCREASE_AMOUNT
        current_speed = min(current_speed, self.ENEMY_MAX_SPEED)

        for enemy in self.enemies:
            enemy['speed'] = current_speed
            enemy['angle'] += enemy['speed'] / enemy['path_radius']
    
    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['angle'] += asteroid['rotation_speed']

    def _handle_mining(self):
        # Find closest asteroid in range
        target_asteroid = None
        min_dist = self.MINING_RANGE
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
                target_asteroid = asteroid
        
        if target_asteroid:
            # Sound: Mining laser loop
            target_asteroid['ore'] -= self.MINING_RATE
            
            # Create laser particles
            for _ in range(2):
                self.fx_particles.append(self._create_particle(
                    self.player_pos,
                    target_asteroid['pos'],
                    self.COLOR_LASER,
                    size=2,
                    lifetime=int(min_dist / 8),
                    speed=8
                ))

            if target_asteroid['ore'] <= 0:
                # Sound: Asteroid break
                self._create_explosion(target_asteroid['pos'], 20, target_asteroid['radius'])
                
                # Create ore particles
                for _ in range(int(target_asteroid['max_ore'] / 2)):
                    self.ore_particles.append(self._create_particle(
                        target_asteroid['pos'],
                        self.player_pos,
                        self.COLOR_ORE,
                        size=4,
                        lifetime=100,
                        speed=self.np_random.uniform(1, 3),
                        attracted=True
                    ))
                self.asteroids.remove(target_asteroid)
                return True, 1.0 # Reward for destroying asteroid
        return False, 0.0

    def _update_ore_particles(self):
        collected_reward = 0
        for p in self.ore_particles[:]:
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.ore_particles.remove(p)
                continue
            
            if p.get('attracted', False):
                direction = self.player_pos - p['pos']
                dist = np.linalg.norm(direction)
                if dist < 2: # Close enough to collect
                    self.player_ore += 1
                    collected_reward += 0.1
                    self.ore_particles.remove(p)
                    # Sound: Ore collect
                    continue
                
                direction /= dist
                p['vel'] = p['vel'] * 0.9 + direction * 1.5
            
            p['pos'] += p['vel']
        return collected_reward

    def _update_fx_particles(self):
        for p in self.fx_particles[:]:
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.fx_particles.remove(p)
                continue
            p['pos'] += p['vel']
            p['size'] = max(0, p['size'] * 0.95)

    def _create_particle(self, start_pos, target_pos, color, size, lifetime, speed, attracted=False):
        direction = target_pos - start_pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            direction /= dist
        
        return {
            "pos": np.copy(start_pos),
            "vel": direction * speed + self.np_random.normal(0, 0.2, 2),
            "color": color,
            "size": size,
            "lifetime": lifetime,
            "attracted": attracted
        }

    def _create_explosion(self, pos, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.fx_particles.append({
                "pos": np.copy(pos),
                "vel": vel,
                "color": self.COLOR_EXPLOSION,
                "size": self.np_random.uniform(2, 5),
                "lifetime": self.np_random.integers(10, 20)
            })

    def _get_enemy_pos(self, enemy):
        return enemy['center'] + np.array([
            math.cos(enemy['angle']) * enemy['path_radius'],
            math.sin(enemy['angle']) * enemy['path_radius']
        ])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for star in self.stars:
            color = tuple(min(255, c + star['brightness']) for c in self.COLOR_BG)
            self.screen.set_at(star['pos'].astype(int), color)

        # Ore particles
        for p in self.ore_particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), int(p['size']))

        # Asteroids
        for asteroid in self.asteroids:
            points = [(p[0] * math.cos(asteroid['angle']) - p[1] * math.sin(asteroid['angle']) + asteroid['pos'][0],
                       p[0] * math.sin(asteroid['angle']) + p[1] * math.cos(asteroid['angle']) + asteroid['pos'][1])
                      for p in asteroid['shape']]
            
            points_int = [(int(x), int(y)) for x, y in points]
            if len(points_int) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points_int, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points_int, self.COLOR_ASTEROID)

        # Enemies
        for enemy in self.enemies:
            pos = self._get_enemy_pos(enemy)
            pos_int = pos.astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS-2, self.COLOR_ENEMY)

        # Player
        if not (self.game_over and not self.win):
            pos_int = self.player_pos.astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS + 4, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS + 4, self.COLOR_PLAYER_GLOW)
            
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

        # FX particles (lasers, explosions)
        for p in self.fx_particles:
            pos_int = p['pos'].astype(int)
            size = int(p['size'])
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], pos_int, size)

    def _render_ui(self):
        # Ore count
        ore_text = self.font_ui.render(f"ORE: {self.player_ore} / {self.WIN_ORE}", True, self.COLOR_TEXT)
        self.screen.blit(ore_text, (10, 10))

        # Game over text
        if self.game_over:
            if self.win:
                end_text = self.font_game_over.render("YOU WIN!", True, self.COLOR_ORE)
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.COLOR_ENEMY)
            
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ore": self.player_ore,
        }
        
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
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Change to 'windows' or 'Quartz' if needed

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0

    print(env.user_guide)
    
    while running:
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Ore: {info['ore']}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()