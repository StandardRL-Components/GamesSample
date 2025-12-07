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
        "Controls: ↑↓←→ to move. Hold SPACE to mine nearby asteroids. Avoid the red enemy ships."
    )

    game_description = (
        "Pilot a mining ship in a dangerous asteroid field. Collect 100 ore to win, but watch out for hostile drones. "
        "The longer you take, the faster they get."
    )

    auto_advance = True

    # --- Colors ---
    COLOR_BG = (16, 0, 32)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_ENEMY = (255, 0, 64)
    COLOR_ASTEROID = (128, 128, 144)
    COLOR_ORE = (255, 255, 0)
    COLOR_UI_TEXT = (224, 224, 255)
    COLOR_BEAM = (0, 255, 255, 150) # With alpha

    # --- Game Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_SIZE = 2000
    PLAYER_ACCELERATION = 0.4
    PLAYER_DRAG = 0.96
    PLAYER_MAX_SPEED = 6.0
    PLAYER_BRAKE_FORCE = 0.90
    PLAYER_RADIUS = 10
    ENEMY_RADIUS = 8
    ASTEROID_BASE_RADIUS = 20
    MINING_RANGE = 100
    MINING_RATE = 0.2 # Ore per step
    WIN_SCORE = 100
    MAX_STEPS = 1800 # 60 seconds at 30fps

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.game_time = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.camera_pos = np.zeros(2, dtype=np.float32)

        self.asteroids = []
        self.enemies = []
        self.particles = []
        self.stars = []
        
        self.enemy_base_speed = 1.0
        self.enemy_current_speed = 1.0
        self.last_difficulty_increase_time = 0
        self.last_asteroid_spawn_time = 0
        
        self.mining_target = None  # FIX: Initialize attribute

        self.reset()
        
        # This is a non-standard call, but useful for development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_time = 0.0
        self.game_over = False
        self.game_won = False

        self.player_pos = np.array([self.WORLD_SIZE / 2, self.WORLD_SIZE / 2], dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.camera_pos = self.player_pos.copy()

        self.last_difficulty_increase_time = 0
        self.last_asteroid_spawn_time = 0
        self.enemy_current_speed = self.enemy_base_speed
        
        self.mining_target = None # FIX: Ensure reset to None

        self.particles.clear()
        self._initialize_stars()
        self._initialize_asteroids(15)
        self._initialize_enemies(5)

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        if not self.game_over:
            self._update_player(movement)
            self._update_enemies()
            self._update_particles()
            self._handle_mining(space_held)
            
            # --- Ore collection reward ---
            ore_mined = self.score - self._get_info()['score']
            if ore_mined > 0:
                reward += ore_mined * 0.1

            # --- Asteroid destruction reward ---
            num_asteroids_before = len(self.asteroids)
            self.asteroids = [a for a in self.asteroids if a['ore'] > 0]
            if len(self.asteroids) < num_asteroids_before:
                reward += 1.0 # Sound: asteroid_destroyed.wav

            self._check_collisions()
            self._update_game_state()

        self.steps += 1
        self.game_time += 1.0 / 30.0 # Assuming 30 FPS

        terminated = self.game_over or self.game_won or self.steps >= self.MAX_STEPS
        
        if self.game_won and not self.game_over: # Prevent double reward/penalty
            reward += 100
        elif self.game_over:
            reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, movement):
        # Movement
        accel = np.zeros(2, dtype=np.float32)
        if movement == 1: accel[1] = -self.PLAYER_ACCELERATION # Up
        # movement 2 (down) is brake
        if movement == 3: accel[0] = -self.PLAYER_ACCELERATION # Left
        if movement == 4: accel[0] = self.PLAYER_ACCELERATION # Right
        
        self.player_vel += accel
        
        # Braking
        if movement == 2:
            self.player_vel *= self.PLAYER_BRAKE_FORCE
        
        # Drag
        self.player_vel *= self.PLAYER_DRAG
        
        # Speed limit
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED
            
        self.player_pos += self.player_vel
        
        # World wrap-around
        self.player_pos %= self.WORLD_SIZE

        # Add thruster particles
        if np.linalg.norm(accel) > 0 or speed > 1.0:
            if self.np_random.random() < 0.5:
                angle = math.atan2(self.player_vel[1], self.player_vel[0]) + math.pi
                angle += self.np_random.uniform(-0.5, 0.5)
                p_vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(1, 3)
                self.particles.append({
                    'pos': self.player_pos.copy() - self.player_vel,
                    'vel': p_vel,
                    'lifespan': self.np_random.integers(10, 20),
                    'color': (100, 100, 200),
                    'radius': self.np_random.uniform(1, 3)
                })

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['angle'] += self.enemy_current_speed / enemy['orbit_radius']
            offset_x = math.cos(enemy['angle']) * enemy['orbit_radius']
            offset_y = math.sin(enemy['angle']) * enemy['orbit_radius']
            enemy['pos'] = enemy['orbit_center'] + np.array([offset_x, offset_y])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _handle_mining(self, space_held):
        self.mining_target = None
        if not space_held:
            return

        min_dist = float('inf')
        target_asteroid = None
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.MINING_RANGE and dist < min_dist:
                min_dist = dist
                target_asteroid = asteroid
        
        if target_asteroid:
            self.mining_target = target_asteroid
            # Sound: mining_beam.wav
            mined_amount = self.MINING_RATE
            target_asteroid['ore'] -= mined_amount
            self.score = min(self.WIN_SCORE, self.score + mined_amount)

            if self.np_random.random() < 0.3:
                # Create ore particle effect
                self.particles.append({
                    'pos': target_asteroid['pos'].copy() + self.np_random.uniform(-10, 10, 2),
                    'vel': (self.player_pos - target_asteroid['pos']) / 30.0,
                    'lifespan': self.np_random.integers(25, 35),
                    'color': self.COLOR_ORE,
                    'radius': self.np_random.uniform(2, 4)
                })

    def _check_collisions(self):
        # Player-Enemy collision
        for enemy in self.enemies:
            dist = np.linalg.norm(self.player_pos - enemy['pos'])
            if dist < self.PLAYER_RADIUS + self.ENEMY_RADIUS:
                self.game_over = True
                # Sound: explosion.wav
                self._create_explosion(self.player_pos, self.COLOR_PLAYER)
                break

    def _update_game_state(self):
        # Win condition
        if self.score >= self.WIN_SCORE:
            self.game_won = True

        # Difficulty scaling
        if self.game_time - self.last_difficulty_increase_time > 10:
            self.enemy_current_speed += 0.5
            self.last_difficulty_increase_time = self.game_time
        
        # Asteroid spawning
        if self.game_time - self.last_asteroid_spawn_time > 20:
            self._initialize_asteroids(1)
            self.last_asteroid_spawn_time = self.game_time

    def _get_observation(self):
        # Smooth camera
        self.camera_pos = 0.9 * self.camera_pos + 0.1 * self.player_pos

        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_game_elements()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _world_to_screen(self, pos):
        screen_center = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        return (pos - self.camera_pos + screen_center).astype(int)

    def _render_stars(self):
        for star in self.stars:
            # Parallax effect
            screen_pos = (star['pos'] - self.camera_pos * star['depth'] + 
                          np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2]))
            screen_pos[0] %= self.SCREEN_WIDTH
            screen_pos[1] %= self.SCREEN_HEIGHT
            
            color_val = int(star['brightness'] * (150 + 105 * math.sin(self.game_time + star['pos'][0])))
            color = (color_val, color_val, color_val)
            pygame.draw.circle(self.screen, color, screen_pos.astype(int), int(star['radius']))

    def _render_game_elements(self):
        # Render particles
        for p in self.particles:
            screen_pos = self._world_to_screen(p['pos'])
            radius = int(p['radius'] * (p['lifespan'] / 20.0))
            if radius > 0:
                pygame.draw.circle(self.screen, p['color'], screen_pos, radius)

        # Render mining beam
        if self.mining_target and not self.game_over:
            start_pos = self._world_to_screen(self.player_pos)
            end_pos = self._world_to_screen(self.mining_target['pos'])
            
            # Pulsing width
            width = int(2 + math.sin(self.game_time * 30))
            pygame.draw.line(self.screen, self.COLOR_BEAM, start_pos, end_pos, width)
            pygame.gfxdraw.filled_circle(self.screen, start_pos[0], start_pos[1], 4, self.COLOR_BEAM)
            pygame.gfxdraw.filled_circle(self.screen, end_pos[0], end_pos[1], 4, self.COLOR_BEAM)

        # Render asteroids
        for asteroid in self.asteroids:
            screen_pos = self._world_to_screen(asteroid['pos'])
            radius = int(asteroid['radius'] * (asteroid['ore'] / 10.0))
            if radius > 2:
                # Create a jagged look
                points = []
                for i in range(12):
                    angle = i * (2 * math.pi / 12) + asteroid['angle']
                    r = radius + self.np_random.uniform(-0.2, 0.2) * radius
                    x = screen_pos[0] + r * math.cos(angle)
                    y = screen_pos[1] + r * math.sin(angle)
                    points.append((x, y))
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)

        # Render enemies
        for enemy in self.enemies:
            screen_pos = self._world_to_screen(enemy['pos'])
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)

        # Render player
        if not self.game_over:
            screen_pos = self._world_to_screen(self.player_pos)
            angle = math.atan2(self.player_vel[1], self.player_vel[0]) if np.linalg.norm(self.player_vel) > 0.1 else 0
            
            # Triangle ship
            p1 = (screen_pos[0] + self.PLAYER_RADIUS * math.cos(angle), screen_pos[1] + self.PLAYER_RADIUS * math.sin(angle))
            p2 = (screen_pos[0] + self.PLAYER_RADIUS * math.cos(angle + 2.5), screen_pos[1] + self.PLAYER_RADIUS * math.sin(angle + 2.5))
            p3 = (screen_pos[0] + self.PLAYER_RADIUS * math.cos(angle - 2.5), screen_pos[1] + self.PLAYER_RADIUS * math.sin(angle - 2.5))
            
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)
            pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = f"ORE: {int(self.score)} / {self.WIN_SCORE}"
        text_surf = self.font_main.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Progress bar
        bar_width = 200
        bar_height = 10
        progress = self.score / self.WIN_SCORE
        pygame.draw.rect(self.screen, (50,50,80), (10, 45, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_ORE, (10, 45, int(bar_width * progress), bar_height))
        
        # Game Over / Win Text
        if self.game_over:
            msg = "GAME OVER"
            color = self.COLOR_ENEMY
        elif self.game_won:
            msg = "MISSION COMPLETE"
            color = self.COLOR_PLAYER
        else:
            return
            
        text_surf = self.font_main.render(msg, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "game_time": self.game_time,
            "enemy_speed": self.enemy_current_speed
        }
    
    def _initialize_stars(self):
        self.stars.clear()
        for _ in range(200):
            self.stars.append({
                'pos': self.np_random.uniform(0, self.SCREEN_WIDTH, 2),
                'depth': self.np_random.uniform(0.1, 0.8),
                'brightness': self.np_random.uniform(0.5, 1.0),
                'radius': self.np_random.uniform(0.5, 1.5)
            })

    def _initialize_asteroids(self, count):
        for _ in range(count):
            pos = self.np_random.uniform(0, self.WORLD_SIZE, 2)
            # Ensure not too close to the player's initial spawn
            while np.linalg.norm(pos - self.player_pos) < 300:
                pos = self.np_random.uniform(0, self.WORLD_SIZE, 2)
            
            self.asteroids.append({
                'pos': pos,
                'ore': 10.0,
                'radius': self.ASTEROID_BASE_RADIUS,
                'angle': self.np_random.uniform(0, 2 * math.pi)
            })

    def _initialize_enemies(self, count):
        self.enemies.clear()
        world_center = np.array([self.WORLD_SIZE / 2, self.WORLD_SIZE / 2])
        for _ in range(count):
            self.enemies.append({
                'pos': np.zeros(2, dtype=np.float32),
                'orbit_center': world_center + self.np_random.uniform(-300, 300, 2),
                'orbit_radius': self.np_random.uniform(100, 400),
                'angle': self.np_random.uniform(0, 2 * math.pi)
            })

    def _create_explosion(self, pos, base_color):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 8)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            color_mod = self.np_random.uniform(0.5, 1.2, 3)
            color = np.clip(np.array(base_color) * color_mod, 0, 255).astype(int)
            
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 40),
                'color': tuple(color),
                'radius': self.np_random.uniform(2, 5)
            })
            
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # Set this to "human" to play the game, or "rgb_array" for no-op rendering
    render_mode = "human" 
    
    if render_mode == "human":
        # Pygame needs a display for human rendering
        os.environ["SDL_VIDEODRIVER"] = "x11"
        pygame.display.set_caption("Asteroid Miner")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    terminated = False
    total_reward = 0
    
    # --- Human Controls Mapping ---
    keys_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    print(GameEnv.user_guide)

    while True:
        # --- Human Player Input ---
        if render_mode == "human":
            action = [0, 0, 0] # [movement, space, shift]
            keys = pygame.key.get_pressed()

            # Movement (only one direction at a time)
            for key, move_action in keys_to_action.items():
                if keys[key]:
                    action[0] = move_action
                    break

            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
        else: # Random Agent
            action = env.action_space.sample()

        if terminated:
            print(f"Episode finished. Total Reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0
            terminated = False
            if render_mode != "human": # Let human player reset manually
                continue

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render_mode == "human":
            # Convert the observation back to a Pygame surface and display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30) # Control FPS