
# Generated: 2025-08-27T21:43:54.420208
# Source Brief: brief_02888.md
# Brief Index: 2888

        
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
        "Controls: Arrow keys to move your ship. Hold Space to activate your mining beam on the nearest asteroid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship through a dense asteroid field. Collect 50 valuable minerals to win, but be careful! Colliding with asteroids will damage your ship. Lose all 3 lives, and it's game over."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 1280, 800
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.TARGET_MINERALS = 50
        self.INITIAL_LIVES = 3
        self.NUM_ASTEROIDS = 25
        self.NUM_STARS = 150

        # Player constants
        self.PLAYER_ACCELERATION = 0.4
        self.PLAYER_FRICTION = 0.96
        self.PLAYER_MAX_SPEED = 6.0
        self.PLAYER_SIZE = 12
        self.PLAYER_INVINCIBILITY_FRAMES = 90
        self.MINING_RANGE = 120
        self.MINING_RATE = 20 # frames per mineral

        # Asteroid constants
        self.BASE_ASTEROID_SPEED = 1.0
        self.ASTEROID_SPEED_INCREASE = 0.05
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 150)
        self.COLOR_ASTEROID = (160, 140, 120)
        self.COLOR_MINERAL = (255, 215, 0)
        self.COLOR_EXPLOSION = [(255, 50, 50), (255, 150, 50), (255, 255, 255)]
        self.COLOR_BEAM = (255, 255, 0)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_STAR = (220, 220, 240)

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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_lives = 0
        self.minerals_collected = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.invincibility_timer = 0
        self.mining_timer = 0
        self.mining_target = None
        
        self.asteroids = []
        self.particles = []
        self.stars = []
        
        self.current_asteroid_speed = self.BASE_ASTEROID_SPEED
        self.camera_pos = np.array([0.0, 0.0])
        
        # This will be set in reset()
        self.np_random = None

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_lives = self.INITIAL_LIVES
        self.minerals_collected = 0
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.invincibility_timer = 0
        self.mining_timer = 0
        self.mining_target = None
        
        self.current_asteroid_speed = self.BASE_ASTEROID_SPEED
        
        self.asteroids.clear()
        for _ in range(self.NUM_ASTEROIDS):
            self._spawn_asteroid(initial=True)

        self.particles.clear()
        
        self.stars.clear()
        for _ in range(self.NUM_STARS):
            self.stars.append({
                'pos': np.array([self.np_random.uniform(0, self.WORLD_WIDTH), self.np_random.uniform(0, self.WORLD_HEIGHT)]),
                'depth': self.np_random.uniform(0.1, 0.6)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            reward += self._update_asteroids()
            reward += self._update_particles()
            reward += self._check_collisions()
            reward += self._calculate_proximity_penalty()

        self._update_camera()

        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.game_won:
                reward += 100
            else:
                reward -= 100
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Movement
        if movement == 1: self.player_vel[1] -= self.PLAYER_ACCELERATION  # Up
        if movement == 2: self.player_vel[1] += self.PLAYER_ACCELERATION  # Down
        if movement == 3: self.player_vel[0] -= self.PLAYER_ACCELERATION  # Left
        if movement == 4: self.player_vel[0] += self.PLAYER_ACCELERATION  # Right

        # Mining
        self.mining_target = None
        if space_held:
            closest_asteroid = None
            min_dist_sq = self.MINING_RANGE ** 2
            
            for asteroid in self.asteroids:
                if asteroid['minerals'] > 0:
                    dist_sq = np.sum((self.player_pos - asteroid['pos']) ** 2)
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_asteroid = asteroid
            
            if closest_asteroid:
                self.mining_target = closest_asteroid
                self.mining_timer += 1
                if self.mining_timer >= self.MINING_RATE:
                    self.mining_timer = 0
                    closest_asteroid['minerals'] -= 1
                    # sfx: mining_zap
                    self._create_mineral_particle(closest_asteroid['pos'])
            else:
                self.mining_timer = 0
        else:
            self.mining_timer = 0

    def _update_player(self):
        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION
        
        # Clamp speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel * (self.PLAYER_MAX_SPEED / speed)
            
        # Update position
        self.player_pos += self.player_vel
        
        # Clamp to world bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WORLD_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.WORLD_HEIGHT - self.PLAYER_SIZE)
        
        # Invincibility
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
            
        # Engine particles
        if np.linalg.norm(self.player_vel) > 1.0:
            if self.steps % 3 == 0:
                self._create_engine_particle()

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] = (asteroid['angle'] + asteroid['rot_speed']) % 360
            
            # Respawn if off-screen or mined out
            if not (-50 < asteroid['pos'][0] < self.WORLD_WIDTH + 50 and \
                    -50 < asteroid['pos'][1] < self.WORLD_HEIGHT + 50) or \
                    asteroid['minerals'] <= 0:
                self._spawn_asteroid(asteroid_to_replace=asteroid)
        return 0.0

    def _update_particles(self):
        reward = 0
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['lifespan'] -= 1
            p['pos'] += p['vel']
            if p['type'] == 'mineral':
                dist_sq = np.sum((self.player_pos - p['pos'])**2)
                if dist_sq < (self.PLAYER_SIZE + 5)**2:
                    p['lifespan'] = 0
                    self.minerals_collected += 1
                    # sfx: collect_mineral
                    reward += 5.0 # Event-based reward for collecting
                    if self.minerals_collected > 0 and (self.minerals_collected // 50) > ((self.minerals_collected - 1) // 50):
                         self.current_asteroid_speed += self.ASTEROID_SPEED_INCREASE
        return reward

    def _check_collisions(self):
        if self.invincibility_timer > 0:
            return 0.0
            
        for asteroid in self.asteroids:
            dist_sq = np.sum((self.player_pos - asteroid['pos']) ** 2)
            if dist_sq < (self.PLAYER_SIZE + asteroid['size']) ** 2:
                self.player_lives -= 1
                self.invincibility_timer = self.PLAYER_INVINCIBILITY_FRAMES
                self._create_explosion(self.player_pos)
                # sfx: player_explosion
                self.player_pos = np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2], dtype=float)
                self.player_vel = np.array([0.0, 0.0])
                # No reward penalty here, handled at termination
                return 0.0
        return 0.0

    def _calculate_proximity_penalty(self):
        penalty = 0.0
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < asteroid['size'] + self.PLAYER_SIZE + 5:
                penalty -= 0.1
                break # Only penalize for the closest one
        return penalty

    def _check_termination(self):
        if self.player_lives <= 0:
            self.game_over = True
            return True
        if self.minerals_collected >= self.TARGET_MINERALS:
            self.game_over = True
            self.game_won = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

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
            "lives": self.player_lives,
            "minerals": self.minerals_collected,
        }
        
    def _update_camera(self):
        target_cam_pos = self.player_pos - np.array([self.WIDTH / 2, self.HEIGHT / 2])
        # Smooth camera movement
        self.camera_pos = self.camera_pos * 0.9 + target_cam_pos * 0.1
        
        self.camera_pos[0] = np.clip(self.camera_pos[0], 0, self.WORLD_WIDTH - self.WIDTH)
        self.camera_pos[1] = np.clip(self.camera_pos[1], 0, self.WORLD_HEIGHT - self.HEIGHT)
        
    def _world_to_screen(self, pos):
        return (pos - self.camera_pos).astype(int)

    def _render_game(self):
        # Render stars with parallax
        for star in self.stars:
            screen_pos = (star['pos'] - self.camera_pos * star['depth'])
            screen_pos[0] %= self.WIDTH
            screen_pos[1] %= self.HEIGHT
            pygame.draw.circle(self.screen, self.COLOR_STAR, screen_pos.astype(int), 1)
            
        # Render particles
        for p in self.particles:
            screen_pos = self._world_to_screen(p['pos'])
            size = p.get('size', 3)
            if p['type'] == 'explosion':
                alpha = max(0, 255 * (p['lifespan'] / p['max_lifespan']))
                p['color'].a = int(alpha)
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], int(size), p['color'])
            else:
                 pygame.draw.rect(self.screen, p['color'], (*screen_pos, size, size))
            
        # Render asteroids
        for asteroid in self.asteroids:
            screen_pos = self._world_to_screen(asteroid['pos'])
            if -50 < screen_pos[0] < self.WIDTH + 50 and -50 < screen_pos[1] < self.HEIGHT + 50:
                # Create a rotated polygon for the asteroid
                points = []
                for i in range(asteroid['num_points']):
                    angle = math.radians(asteroid['angle'] + (360 / asteroid['num_points']) * i)
                    x = screen_pos[0] + asteroid['point_dists'][i] * math.cos(angle)
                    y = screen_pos[1] + asteroid['point_dists'][i] * math.sin(angle)
                    points.append((x, y))
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)

        # Render mining beam
        if self.mining_target:
            start_pos = self._world_to_screen(self.player_pos)
            end_pos = self._world_to_screen(self.mining_target['pos'])
            pygame.draw.aaline(self.screen, self.COLOR_BEAM, start_pos, end_pos, 2)
            
        # Render player
        if self.player_lives > 0:
            if self.invincibility_timer == 0 or self.steps % 10 < 5:
                player_screen_pos = self._world_to_screen(self.player_pos)
                
                # Glow effect
                glow_radius = int(self.PLAYER_SIZE * 1.8)
                temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(temp_surf, (player_screen_pos[0] - glow_radius, player_screen_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

                # Ship body
                p1 = (player_screen_pos[0], player_screen_pos[1] - self.PLAYER_SIZE)
                p2 = (player_screen_pos[0] - self.PLAYER_SIZE / 2, player_screen_pos[1] + self.PLAYER_SIZE / 2)
                p3 = (player_screen_pos[0] + self.PLAYER_SIZE / 2, player_screen_pos[1] + self.PLAYER_SIZE / 2)
                
                angle = math.atan2(self.player_vel[0], -self.player_vel[1])
                points = [p1, p2, p3]
                center = player_screen_pos
                rotated_points = [
                    (center[0] + (p[0] - center[0]) * math.cos(angle) - (p[1] - center[1]) * math.sin(angle),
                     center[1] + (p[0] - center[0]) * math.sin(angle) + (p[1] - center[1]) * math.cos(angle))
                    for p in points
                ]

                pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)
                pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Minerals collected
        mineral_text = self.font_small.render(f"MINERALS: {self.minerals_collected}/{self.TARGET_MINERALS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(mineral_text, (10, 10))

        # Lives remaining
        lives_text = self.font_small.render(f"LIVES: {self.player_lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Game Over / Win Message
        if self.game_over:
            message = "MISSION COMPLETE" if self.game_won else "GAME OVER"
            color = self.COLOR_MINERAL if self.game_won else self.COLOR_EXPLOSION[0]
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _spawn_asteroid(self, initial=False, asteroid_to_replace=None):
        edge = self.np_random.integers(4)
        if edge == 0:  # Top
            pos = np.array([self.np_random.uniform(0, self.WORLD_WIDTH), -40.0])
        elif edge == 1:  # Right
            pos = np.array([self.WORLD_WIDTH + 40.0, self.np_random.uniform(0, self.WORLD_HEIGHT)])
        elif edge == 2:  # Bottom
            pos = np.array([self.np_random.uniform(0, self.WORLD_WIDTH), self.WORLD_HEIGHT + 40.0])
        else:  # Left
            pos = np.array([-40.0, self.np_random.uniform(0, self.WORLD_HEIGHT)])
        
        if initial:
            pos = np.array([self.np_random.uniform(0, self.WORLD_WIDTH), self.np_random.uniform(0, self.WORLD_HEIGHT)])

        angle_to_center = math.atan2(self.WORLD_HEIGHT/2 - pos[1], self.WORLD_WIDTH/2 - pos[0])
        angle = self.np_random.uniform(angle_to_center - math.pi / 4, angle_to_center + math.pi / 4)
        speed = self.current_asteroid_speed * self.np_random.uniform(0.8, 1.2)
        vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        
        size = self.np_random.integers(15, 35)
        num_points = self.np_random.integers(7, 12)
        
        asteroid_data = {
            'pos': pos,
            'vel': vel,
            'size': size,
            'minerals': self.np_random.integers(1, 6),
            'angle': self.np_random.uniform(0, 360),
            'rot_speed': self.np_random.uniform(-1.5, 1.5),
            'num_points': num_points,
            'point_dists': [size * self.np_random.uniform(0.8, 1.2) for _ in range(num_points)]
        }
        
        if asteroid_to_replace:
            asteroid_to_replace.update(asteroid_data)
        else:
            self.asteroids.append(asteroid_data)

    def _create_mineral_particle(self, origin_pos):
        self.particles.append({
            'pos': origin_pos.copy(),
            'vel': (self.player_pos - origin_pos) / 50.0, # Move towards player
            'lifespan': 100,
            'color': self.COLOR_MINERAL,
            'type': 'mineral',
            'size': 4
        })

    def _create_explosion(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            max_lifespan = self.np_random.integers(20, 50)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'lifespan': max_lifespan,
                'max_lifespan': max_lifespan,
                'color': pygame.Color(random.choice(self.COLOR_EXPLOSION)),
                'type': 'explosion',
                'size': self.np_random.uniform(2, 8)
            })

    def _create_engine_particle(self):
        angle = math.atan2(-self.player_vel[0], self.player_vel[1]) + self.np_random.uniform(-0.3, 0.3)
        speed = np.linalg.norm(self.player_vel) * 0.5
        offset = np.array([math.sin(angle), -math.cos(angle)]) * self.PLAYER_SIZE * 0.7
        self.particles.append({
            'pos': self.player_pos + offset,
            'vel': np.array([math.sin(angle), -math.cos(angle)]) * speed,
            'lifespan': 20,
            'color': self.COLOR_PLAYER,
            'type': 'engine',
            'size': 2
        })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to get initial observation
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {obs.shape}"
        assert obs.dtype == np.uint8
        
        # Test reset return types
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