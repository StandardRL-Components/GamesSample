
# Generated: 2025-08-28T05:16:35.932242
# Source Brief: brief_05524.md
# Brief Index: 5524

        
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
        "Controls: ↑↓←→ to move. Hold space to mine nearby asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a space miner through an asteroid field, collecting ore while dodging collisions to reach a target yield."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        self.WIN_ORE = 100
        self.INITIAL_LIVES = 3
        
        self.PLAYER_SPEED = 5
        self.PLAYER_SIZE = 12
        self.MINING_RANGE = 80
        self.MINING_RATE = 1
        self.INVINCIBILITY_FRAMES = 90 # 3 seconds at 30 FPS

        self.ASTEROID_SPAWN_RATE = 25 # Lower is more frequent
        self.MAX_ASTEROIDS = 15
        self.INITIAL_ASTEROID_SPEED = 0.5
        self.MAX_ASTEROID_SPEED = 2.0
        self.DIFFICULTY_RAMP_STEPS = 500

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 150)
        self.COLOR_LOW_ORE = (115, 110, 105)
        self.COLOR_MID_ORE = (160, 130, 90)
        self.COLOR_HIGH_ORE = (230, 190, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_MINING_BEAM = (255, 255, 100)
        
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.player_pos = None
        self.player_lives = None
        self.ore_collected = None
        self.asteroids = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.last_move_dir = None
        self.invincibility_timer = None
        self.asteroid_drift_speed = None
        self.starfield = None
        
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_lives = self.INITIAL_LIVES
        self.ore_collected = 0
        self.invincibility_timer = 0
        self.last_move_dir = np.array([0, -1], dtype=np.float32) # Facing up

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.asteroid_drift_speed = self.INITIAL_ASTEROID_SPEED
        self.asteroids = []
        for _ in range(self.MAX_ASTEROIDS // 2):
            self._spawn_asteroid()

        self.particles = []
        self._init_starfield()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.02 # Small penalty for existing

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        self._handle_input(movement)
        self._update_player()
        self._update_asteroids()
        
        if space_held:
            reward += self._handle_mining()

        reward += self._check_collisions()
        self._update_particles()
        self._spawn_new_asteroids()
        self._update_difficulty()

        self.score += reward
        terminated = self._check_termination()

        if terminated and not self.game_over:
             if self.ore_collected >= self.WIN_ORE:
                 reward += 100
                 self.score += 100
             self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1: # Up
            move_vec[1] -= 1
        elif movement == 2: # Down
            move_vec[1] += 1
        elif movement == 3: # Left
            move_vec[0] -= 1
        elif movement == 4: # Right
            move_vec[0] += 1
        
        if np.linalg.norm(move_vec) > 0:
            self.last_move_dir = move_vec / np.linalg.norm(move_vec)
            self.player_pos += self.last_move_dir * self.PLAYER_SPEED
        
        # Update starfield based on player movement
        for layer in self.starfield:
            for star in layer['stars']:
                star[0] -= self.last_move_dir[0] * self.PLAYER_SPEED * layer['speed_factor'] if np.linalg.norm(move_vec) > 0 else 0
                star[1] -= self.last_move_dir[1] * self.PLAYER_SPEED * layer['speed_factor'] if np.linalg.norm(move_vec) > 0 else 0
                self._wrap_around(star)

    def _update_player(self):
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)

        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
    
    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            self._wrap_around(asteroid['pos'])

    def _handle_mining(self):
        closest_asteroid, min_dist = None, float('inf')
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
                closest_asteroid = asteroid
        
        if closest_asteroid and min_dist < self.MINING_RANGE + closest_asteroid['size']:
            # sfx: mining_laser_start
            mined_amount = min(closest_asteroid['ore'], self.MINING_RATE)
            if mined_amount > 0:
                closest_asteroid['ore'] -= mined_amount
                self.ore_collected += mined_amount
                
                # Spawn mining particles
                for _ in range(2):
                    self._spawn_particle(
                        pos=closest_asteroid['pos'].copy(),
                        type='mining',
                        target=self.player_pos
                    )
                
                if closest_asteroid['ore'] <= 0:
                    # sfx: asteroid_depleted
                    self.asteroids.remove(closest_asteroid)
                return mined_amount * 0.1 # Reward for mining
        return 0

    def _check_collisions(self):
        if self.invincibility_timer > 0:
            return 0
        
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_SIZE + asteroid['size']:
                self.player_lives -= 1
                self.invincibility_timer = self.INVINCIBILITY_FRAMES
                # sfx: player_explosion
                for _ in range(50):
                    self._spawn_particle(pos=self.player_pos.copy(), type='explosion')
                return -10 # Penalty for collision
        return 0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            if p['type'] == 'mining':
                target_vec = p['target'] - p['pos']
                dist = np.linalg.norm(target_vec)
                if dist > 5:
                    p['vel'] = target_vec / dist * 6.0
                else: # Reached target
                    p['lifespan'] = 0
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _spawn_new_asteroids(self):
        if self.np_random.integers(0, self.ASTEROID_SPAWN_RATE) == 0 and len(self.asteroids) < self.MAX_ASTEROIDS:
            self._spawn_asteroid()

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_RAMP_STEPS == 0:
            new_speed = self.asteroid_drift_speed + 0.1
            self.asteroid_drift_speed = min(new_speed, self.MAX_ASTEROID_SPEED)
            for asteroid in self.asteroids:
                 # Update existing asteroid velocities
                 norm = np.linalg.norm(asteroid['vel'])
                 if norm > 0:
                     asteroid['vel'] = (asteroid['vel'] / norm) * self.asteroid_drift_speed
    
    def _check_termination(self):
        return (
            self.player_lives <= 0 or
            self.ore_collected >= self.WIN_ORE or
            self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ore": self.ore_collected,
            "lives": self.player_lives,
        }

    def _render_game(self):
        # Render starfield
        for layer in self.starfield:
            color = layer['color']
            for x, y, size in layer['stars']:
                pygame.draw.circle(self.screen, color, (int(x), int(y)), int(size))

        # Render asteroids
        for asteroid in self.asteroids:
            ore_ratio = asteroid['ore'] / asteroid['max_ore']
            if ore_ratio > 0.66: color = self.COLOR_HIGH_ORE
            elif ore_ratio > 0.33: color = self.COLOR_MID_ORE
            else: color = self.COLOR_LOW_ORE
            
            pos = (int(asteroid['pos'][0]), int(asteroid['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(asteroid['size']), color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(asteroid['size']), tuple(c*0.7 for c in color))

        # Render mining beam
        if not self.game_over and any(p['type'] == 'mining' for p in self.particles):
             closest_asteroid, min_dist = None, float('inf')
             for asteroid in self.asteroids:
                dist = np.linalg.norm(self.player_pos - asteroid['pos'])
                if dist < min_dist:
                    min_dist = dist
                    closest_asteroid = asteroid
             if closest_asteroid and min_dist < self.MINING_RANGE + closest_asteroid['size']:
                 start_pos = (int(self.player_pos[0]), int(self.player_pos[1]))
                 end_pos = (int(closest_asteroid['pos'][0]), int(closest_asteroid['pos'][1]))
                 pygame.draw.aaline(self.screen, self.COLOR_MINING_BEAM, start_pos, end_pos, 2)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Render player
        if not self.game_over and self.player_lives > 0:
            if self.invincibility_timer == 0 or self.steps % 10 < 5:
                # Calculate triangle points for the ship
                p1 = self.player_pos + self.last_move_dir * self.PLAYER_SIZE
                perp_vec = np.array([-self.last_move_dir[1], self.last_move_dir[0]])
                p2 = self.player_pos - self.last_move_dir * self.PLAYER_SIZE * 0.5 + perp_vec * self.PLAYER_SIZE * 0.8
                p3 = self.player_pos - self.last_move_dir * self.PLAYER_SIZE * 0.5 - perp_vec * self.PLAYER_SIZE * 0.8
                points = [tuple(p1), tuple(p2), tuple(p3)]
                
                # Draw glow
                pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_PLAYER_GLOW)
                # Draw main ship
                pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_PLAYER)

    def _render_ui(self):
        # Ore counter
        ore_text = self.font_ui.render(f"ORE: {self.ore_collected}/{self.WIN_ORE}", True, self.COLOR_TEXT)
        self.screen.blit(ore_text, (10, 10))

        # Lives display
        for i in range(self.player_lives):
            ship_icon_points = [
                (self.WIDTH - 20 - i*25, 15), 
                (self.WIDTH - 30 - i*25, 30), 
                (self.WIDTH - 10 - i*25, 30)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, ship_icon_points)

        # Game over message
        if self.game_over:
            if self.ore_collected >= self.WIN_ORE:
                msg = "VICTORY"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            text = self.font_game_over.render(msg, True, color)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _spawn_asteroid(self):
        edge = self.np_random.integers(0, 4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -30], dtype=np.float32)
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 30], dtype=np.float32)
        elif edge == 2: # Left
            pos = np.array([-30, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32)
        else: # Right
            pos = np.array([self.WIDTH + 30, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32)
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * self.asteroid_drift_speed
        
        size = self.np_random.uniform(10, 30)
        ore = int(size * self.np_random.uniform(1.5, 3.0))

        self.asteroids.append({
            'pos': pos, 'vel': vel, 'size': size, 'ore': ore, 'max_ore': ore
        })

    def _spawn_particle(self, pos, type, target=None):
        if type == 'explosion':
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(15, 40)
            color = random.choice([(255, 100, 0), (255, 200, 0), (200, 200, 200)])
            size = self.np_random.integers(1, 4)
        elif type == 'mining':
            vel = np.array([0, 0])
            lifespan = self.np_random.integers(20, 40)
            color = self.COLOR_MINING_BEAM
            size = self.np_random.integers(2, 4)
        else:
            return

        self.particles.append({
            'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'max_lifespan': lifespan,
            'color': color, 'type': type, 'target': target, 'size': size
        })
    
    def _init_starfield(self):
        self.starfield = [
            {'stars': [], 'speed_factor': 0.1, 'color': (50, 60, 80)},
            {'stars': [], 'speed_factor': 0.2, 'color': (100, 110, 130)},
            {'stars': [], 'speed_factor': 0.3, 'color': (180, 180, 200)}
        ]
        for i, layer in enumerate(self.starfield):
            num_stars = (i + 1) * 50
            for _ in range(num_stars):
                layer['stars'].append([
                    self.np_random.uniform(0, self.WIDTH),
                    self.np_random.uniform(0, self.HEIGHT),
                    self.np_random.uniform(0.5, 1.5)
                ])

    def _wrap_around(self, pos_array):
        buffer = 40
        if pos_array[0] < -buffer: pos_array[0] = self.WIDTH + buffer
        if pos_array[0] > self.WIDTH + buffer: pos_array[0] = -buffer
        if pos_array[1] < -buffer: pos_array[1] = self.HEIGHT + buffer
        if pos_array[1] > self.HEIGHT + buffer: pos_array[1] = -buffer

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Set up Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Miner")
    clock = pygame.time.Clock()
    
    running = True
    total_score = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_score = 0
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_score += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Ore: {info['ore']}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_score = 0
            print("--- New Game ---")

        clock.tick(env.FPS)
        
    env.close()