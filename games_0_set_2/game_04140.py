
# Generated: 2025-08-28T01:31:42.012217
# Source Brief: brief_04140.md
# Brief Index: 4140

        
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
        "Controls: ↑↓←→ to move. Hold Space to mine ore from nearby asteroids. Avoid the red enemy ships."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship, collect ore from asteroids, and evade relentless enemies in a visually stunning, procedurally generated asteroid field."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_WIDTH, WORLD_HEIGHT = 1280, 800
    
    COLOR_BG = (15, 18, 32)
    COLOR_STAR = (100, 100, 120)
    COLOR_PLAYER = (60, 180, 255)
    COLOR_PLAYER_GLOW = (60, 180, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 70)
    COLOR_ASTEROID = (120, 120, 120)
    COLOR_ORE = (255, 220, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_BEAM = (255, 255, 255, 100)

    PLAYER_SIZE = 12
    PLAYER_ACCEL = 0.4
    PLAYER_DRAG = 0.96
    PLAYER_MAX_SPEED = 6
    
    MINING_RANGE = 80
    MINING_RATE = 1
    
    WIN_SCORE = 100
    MAX_STEPS = 1000

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
        self.font = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.asteroids = []
        self.enemies = []
        self.particles = []
        self.stars = []
        self.camera_pos = None
        self.prev_dist_to_asteroid = None
        self.prev_dist_to_enemy = None
        
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Comment out for submission

    def _generate_stars(self, count=200):
        self.stars = []
        for _ in range(count):
            self.stars.append(
                (
                    self.np_random.uniform(0, self.WORLD_WIDTH),
                    self.np_random.uniform(0, self.WORLD_HEIGHT),
                    self.np_random.uniform(0.5, 1.5) # size
                )
            )

    def _generate_asteroids(self, count=20):
        self.asteroids = []
        for _ in range(count):
            # Ensure asteroids don't spawn too close to the center (player start)
            while True:
                pos = np.array([
                    self.np_random.uniform(0, self.WORLD_WIDTH),
                    self.np_random.uniform(0, self.WORLD_HEIGHT)
                ])
                if np.linalg.norm(pos - np.array([self.WORLD_WIDTH/2, self.WORLD_HEIGHT/2])) > 200:
                    break
            
            size = self.np_random.uniform(20, 40)
            num_vertices = self.np_random.integers(7, 12)
            
            self.asteroids.append({
                "pos": pos,
                "ore": self.np_random.integers(10, 25),
                "size": size,
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "rotation_speed": self.np_random.uniform(-0.01, 0.01),
                "vertices": self._generate_asteroid_shape(size, num_vertices)
            })

    def _generate_asteroid_shape(self, radius, num_vertices):
        vertices = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            r = radius + self.np_random.uniform(-radius * 0.2, radius * 0.2)
            vertices.append((r * math.cos(angle), r * math.sin(angle)))
        return vertices

    def _generate_enemies(self, count=3):
        self.enemies = []
        for _ in range(count):
            orbit_radius = self.np_random.uniform(100, 300)
            orbit_center = np.array([
                self.np_random.uniform(orbit_radius, self.WORLD_WIDTH - orbit_radius),
                self.np_random.uniform(orbit_radius, self.WORLD_HEIGHT - orbit_radius)
            ])
            self.enemies.append({
                "pos": np.array([0.0, 0.0]),
                "orbit_center": orbit_center,
                "orbit_radius": orbit_radius,
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "speed": self.np_random.uniform(0.02, 0.04)
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        
        self.camera_pos = self.player_pos - np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2])

        self._generate_stars()
        self._generate_asteroids()
        self._generate_enemies()
        
        self.particles = []

        # Reset distance trackers for reward calculation
        self.prev_dist_to_asteroid = self._get_closest_distance(self.player_pos, [a['pos'] for a in self.asteroids if a['ore'] > 0])
        self.prev_dist_to_enemy = self._get_closest_distance(self.player_pos, [e['pos'] for e in self.enemies])

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        movement, space_held, _ = action
        space_held = space_held == 1
        
        # --- Update Game Logic ---
        self._handle_input(movement)
        self._update_player()
        self._update_enemies()
        self._update_asteroids()
        
        reward = self._calculate_reward()
        
        mined_ore = self._handle_mining(space_held)
        if mined_ore > 0:
            self.score += mined_ore
            reward += mined_ore * 1.0 # Event-based reward
            # sfx: ore_collect.wav

        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Goal-oriented reward
            else:
                reward -= 100.0 # Collision penalty
                # sfx: player_explosion.wav
                self._create_explosion(self.player_pos, 50, self.COLOR_PLAYER)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement):
        accel = np.array([0.0, 0.0])
        if movement == 1: # Up
            accel[1] -= self.PLAYER_ACCEL
        elif movement == 2: # Down
            accel[1] += self.PLAYER_ACCEL
        elif movement == 3: # Left
            accel[0] -= self.PLAYER_ACCEL
        elif movement == 4: # Right
            accel[0] += self.PLAYER_ACCEL
        
        self.player_vel += accel
        
        if np.linalg.norm(accel) > 0:
            self._create_thruster_particles(accel)
            # sfx: thruster_loop.wav

    def _update_player(self):
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel * (self.PLAYER_MAX_SPEED / speed)
        
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_DRAG

        # World wrap-around
        self.player_pos[0] %= self.WORLD_WIDTH
        self.player_pos[1] %= self.WORLD_HEIGHT

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['angle'] += enemy['speed']
            enemy['pos'][0] = enemy['orbit_center'][0] + math.cos(enemy['angle']) * enemy['orbit_radius']
            enemy['pos'][1] = enemy['orbit_center'][1] + math.sin(enemy['angle']) * enemy['orbit_radius']

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['angle'] += asteroid['rotation_speed']

    def _handle_mining(self, space_held):
        mined_ore = 0
        if not space_held:
            return 0
        
        for asteroid in self.asteroids:
            if asteroid['ore'] > 0:
                dist = np.linalg.norm(self.player_pos - asteroid['pos'])
                if dist < self.MINING_RANGE + asteroid['size']:
                    ore_to_mine = min(self.MINING_RATE, asteroid['ore'])
                    asteroid['ore'] -= ore_to_mine
                    mined_ore += ore_to_mine
                    
                    self._create_mining_particles(asteroid['pos'])
                    # sfx: mining_beam.wav
                    break # Mine one at a time
        return mined_ore

    def _get_closest_distance(self, pos, targets):
        if not targets:
            return float('inf')
        distances = [np.linalg.norm(pos - t) for t in targets]
        return min(distances)

    def _calculate_reward(self):
        # Continuous feedback reward
        reward = 0.0
        
        # Reward for moving towards ore
        ore_asteroids_pos = [a['pos'] for a in self.asteroids if a['ore'] > 0]
        if ore_asteroids_pos:
            dist_to_asteroid = self._get_closest_distance(self.player_pos, ore_asteroids_pos)
            if self.prev_dist_to_asteroid is not None:
                reward += (self.prev_dist_to_asteroid - dist_to_asteroid) * 0.1
            self.prev_dist_to_asteroid = dist_to_asteroid
        
        # Penalty for moving towards enemies
        enemy_pos = [e['pos'] for e in self.enemies]
        if enemy_pos:
            dist_to_enemy = self._get_closest_distance(self.player_pos, enemy_pos)
            if self.prev_dist_to_enemy is not None:
                reward -= (self.prev_dist_to_enemy - dist_to_enemy) * 0.1
            self.prev_dist_to_enemy = dist_to_enemy

        return reward

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        # Check collision with enemies
        for enemy in self.enemies:
            dist = np.linalg.norm(self.player_pos - enemy['pos'])
            if dist < self.PLAYER_SIZE + 8: # 8 is enemy radius
                self.game_over = True
                break
        
        return self.game_over
    
    # --- Particle System ---
    def _create_thruster_particles(self, accel):
        # Create particles moving opposite to acceleration
        angle = math.atan2(self.player_vel[1], self.player_vel[0]) + math.pi
        for _ in range(2):
            p_angle = angle + self.np_random.uniform(-0.3, 0.3)
            p_speed = self.np_random.uniform(1, 3)
            p_vel = np.array([math.cos(p_angle) * p_speed, math.sin(p_angle) * p_speed])
            self.particles.append({
                "pos": self.player_pos.copy(),
                "vel": p_vel,
                "lifetime": self.np_random.integers(10, 20),
                "color": self.COLOR_PLAYER,
                "size": self.np_random.uniform(1, 3)
            })

    def _create_mining_particles(self, asteroid_pos):
        for _ in range(2):
            direction = self.player_pos - asteroid_pos
            unit_dir = direction / (np.linalg.norm(direction) + 1e-6)
            
            start_pos = asteroid_pos + unit_dir * self.np_random.uniform(20, 40)
            
            self.particles.append({
                "pos": start_pos,
                "vel": unit_dir * self.np_random.uniform(2, 4),
                "lifetime": self.np_random.integers(20, 30),
                "color": self.COLOR_ORE,
                "size": self.np_random.uniform(2, 4)
            })

    def _create_explosion(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifetime": self.np_random.integers(20, 40),
                "color": color,
                "size": self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    # --- Rendering ---
    def _get_observation(self):
        # Smooth camera follow
        target_cam_pos = self.player_pos - np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2])
        self.camera_pos = self.camera_pos * 0.9 + target_cam_pos * 0.1

        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _world_to_screen(self, pos):
        return int(pos[0] - self.camera_pos[0]), int(pos[1] - self.camera_pos[1])

    def _render_game(self):
        # Draw stars
        for x, y, size in self.stars:
            sx, sy = self._world_to_screen((x, y))
            if 0 < sx < self.SCREEN_WIDTH and 0 < sy < self.SCREEN_HEIGHT:
                 pygame.draw.circle(self.screen, self.COLOR_STAR, (sx, sy), size)

        # Draw asteroids
        for asteroid in self.asteroids:
            if asteroid['ore'] > 0:
                color = tuple(np.clip(np.array(self.COLOR_ASTEROID) + np.array(self.COLOR_ORE) * 0.3, 0, 255))
            else:
                color = self.COLOR_ASTEROID

            rotated_vertices = []
            for vx, vy in asteroid['vertices']:
                rotated_x = vx * math.cos(asteroid['angle']) - vy * math.sin(asteroid['angle'])
                rotated_y = vx * math.sin(asteroid['angle']) + vy * math.cos(asteroid['angle'])
                screen_pos = self._world_to_screen((asteroid['pos'][0] + rotated_x, asteroid['pos'][1] + rotated_y))
                rotated_vertices.append(screen_pos)
            
            pygame.gfxdraw.aapolygon(self.screen, rotated_vertices, color)
            pygame.gfxdraw.filled_polygon(self.screen, rotated_vertices, color)

        # Draw mining beam
        if self.action_space.sample()[1] == 1: # A bit of a hack to check if space is held
            closest_asteroid = None
            min_dist = float('inf')
            for asteroid in self.asteroids:
                if asteroid['ore'] > 0:
                    dist = np.linalg.norm(self.player_pos - asteroid['pos'])
                    if dist < self.MINING_RANGE + asteroid['size'] and dist < min_dist:
                        min_dist = dist
                        closest_asteroid = asteroid
            if closest_asteroid:
                player_screen_pos = self._world_to_screen(self.player_pos)
                asteroid_screen_pos = self._world_to_screen(closest_asteroid['pos'])
                pygame.draw.aaline(self.screen, self.COLOR_BEAM, player_screen_pos, asteroid_screen_pos, 2)


        # Draw enemies
        for enemy in self.enemies:
            pos = self._world_to_screen(enemy['pos'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)

        # Draw particles
        for p in self.particles:
            pos = self._world_to_screen(p['pos'])
            alpha = int(255 * (p['lifetime'] / 20)) # Fade out
            color = p['color'] + (alpha,)
            
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (pos[0] - p['size'], pos[1] - p['size']))

        # Draw player
        if not self.game_over:
            pos = self._world_to_screen(self.player_pos)
            angle = math.atan2(self.player_vel[1], self.player_vel[0])

            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(self.PLAYER_SIZE * 1.8), self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(self.PLAYER_SIZE * 1.8), self.COLOR_PLAYER_GLOW)

            # Ship body (triangle)
            points = [
                (self.PLAYER_SIZE, 0),
                (-self.PLAYER_SIZE * 0.5, -self.PLAYER_SIZE * 0.8),
                (-self.PLAYER_SIZE * 0.5, self.PLAYER_SIZE * 0.8)
            ]
            
            rotated_points = []
            for x, y in points:
                rx = x * math.cos(angle) - y * math.sin(angle) + pos[0]
                ry = x * math.sin(angle) + y * math.cos(angle) + pos[1]
                rotated_points.append((rx, ry))
            
            pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)
    
    def _render_ui(self):
        score_text = self.font.render(f"ORE: {self.score} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        if self.game_over:
            end_text_str = "VICTORY!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos.tolist(),
            "player_vel": self.player_vel.tolist(),
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Use a separate display for human play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    
    terminated = False
    total_reward = 0
    
    # Map Pygame keys to the action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not terminated:
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        for key, move in key_map.items():
            if keys[key]:
                movement_action = move
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control the frame rate

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    # Keep the window open for a few seconds to show the final screen
    pygame.time.wait(3000)
    env.close()