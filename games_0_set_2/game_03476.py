
# Generated: 2025-08-27T23:28:56.206132
# Source Brief: brief_03476.md
# Brief Index: 3476

        
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
        "Controls: Use arrow keys to turn and thrust your ship. Hold space near an asteroid to mine it. Avoid collisions!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship, mine asteroids for minerals, and avoid collisions in a procedurally generated asteroid field."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
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
        self.FONT_UI = pygame.font.Font(None, 24)
        self.FONT_MSG = pygame.font.Font(None, 50)
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_STAR = (200, 200, 220)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_ASTEROID = (100, 100, 110)
        self.COLOR_ASTEROID_OUTLINE = (140, 140, 150)
        self.COLOR_MINERAL = (255, 220, 0)
        self.COLOR_BEAM = (100, 255, 100)
        self.COLOR_UI = (230, 230, 230)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)
        self.EXPLOSION_COLORS = [(255, 50, 50), (255, 150, 50), (255, 255, 100)]

        # Game parameters
        self.MAX_STEPS = 5000
        self.WIN_SCORE = 50
        self.INITIAL_LIVES = 3
        self.INITIAL_ASTEROIDS = 8
        self.MIN_ASTEROIDS = 5
        self.INVINCIBILITY_DURATION = 90  # frames (3 seconds at 30fps)
        
        # Player parameters
        self.PLAYER_RADIUS = 15
        self.PLAYER_TURN_SPEED = 4.5
        self.PLAYER_THRUST = 0.25
        self.PLAYER_BRAKE_THRUST = 0.15
        self.PLAYER_DRAG = 0.985
        self.PLAYER_MAX_SPEED = 5.0
        
        # Asteroid parameters
        self.ASTEROID_BASE_SPEED = 0.5
        self.DIFFICULTY_INTERVAL = 500 # steps
        
        # Mining parameters
        self.MINING_RADIUS = 100
        self.MINING_RATE = 0.2
        self.MINERAL_PARTICLE_LIFETIME = 40
        
        # Reward parameters
        self.REWARD_MINERAL = 1.0
        self.REWARD_COLLISION = -10.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSS = -100.0
        self.REWARD_DISTANCE_FACTOR = 0.01
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.step_reward = 0.0
        
        # Player state
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_angle = -90.0  # Pointing up
        self.lives = self.INITIAL_LIVES
        self.invincibility_timer = 0

        # Game objects
        self.asteroids = []
        self.particles = []
        self.mining_target = None
        self.asteroid_speed_modifier = 1.0
        self._spawn_asteroids(self.INITIAL_ASTEROIDS)
        
        # Static background
        self.stars = [
            (self.np_random.integers(0, self.SCREEN_WIDTH), 
             self.np_random.integers(0, self.SCREEN_HEIGHT), 
             self.np_random.integers(1, 4)) for _ in range(150)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.step_reward = 0.0
        dist_before = self._get_closest_asteroid_dist()

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Update game logic
        self._handle_input(movement)
        self._update_difficulty()
        self._update_player()
        self._update_asteroids()
        self._update_particles()
        self._handle_mining(space_held)
        self._handle_collisions()
        
        dist_after = self._get_closest_asteroid_dist()
        if dist_before is not None and dist_after is not None:
            self.step_reward += (dist_before - dist_after) * self.REWARD_DISTANCE_FACTOR

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                self.step_reward += self.REWARD_WIN
            else:
                self.step_reward += self.REWARD_LOSS
        
        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        # 3=left, 4=right for turning
        if movement == 3: self.player_angle -= self.PLAYER_TURN_SPEED
        if movement == 4: self.player_angle += self.PLAYER_TURN_SPEED

        # 1=up for thrust, 2=down for braking/reverse
        thrust = 0
        if movement == 1:
            thrust = self.PLAYER_THRUST
            # sound: player_thrust.wav
        elif movement == 2:
            thrust = -self.PLAYER_BRAKE_THRUST
        
        if thrust != 0:
            rad_angle = math.radians(self.player_angle)
            acceleration = np.array([math.cos(rad_angle), math.sin(rad_angle)]) * thrust
            self.player_vel += acceleration

        self.player_vel *= self.PLAYER_DRAG
        
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = (self.player_vel / speed) * self.PLAYER_MAX_SPEED
            
    def _update_player(self):
        self.player_pos += self.player_vel
        self.player_pos[0] %= self.SCREEN_WIDTH
        self.player_pos[1] %= self.SCREEN_HEIGHT
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
            
    def _update_asteroids(self):
        for ast in self.asteroids:
            ast['pos'] += ast['vel'] * self.asteroid_speed_modifier
            ast['pos'][0] %= self.SCREEN_WIDTH
            ast['pos'][1] %= self.SCREEN_HEIGHT
            ast['angle'] = (ast['angle'] + ast['rot_speed']) % 360
        
        self.asteroids = [ast for ast in self.asteroids if ast['minerals'] > 0]
        
        if len(self.asteroids) < self.MIN_ASTEROIDS:
            self._spawn_asteroids(self.INITIAL_ASTEROIDS - len(self.asteroids))

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            
            collected = False
            if p['type'] == 'mineral' and np.linalg.norm(p['pos'] - self.player_pos) < self.PLAYER_RADIUS:
                self.score = min(self.score + 1, self.WIN_SCORE)
                self.step_reward += self.REWARD_MINERAL
                collected = True
                # sound: mineral_collect.wav
            
            if p['lifetime'] > 0 and not collected:
                if p['type'] == 'explosion':
                    p['vel'] *= 0.95 # Slow down explosion particles
                active_particles.append(p)
        self.particles = active_particles
        
    def _handle_mining(self, space_held):
        self.mining_target = None
        if not space_held: return

        closest_ast, min_dist = None, self.MINING_RADIUS
        for ast in self.asteroids:
            dist = np.linalg.norm(self.player_pos - ast['pos'])
            if dist < min_dist:
                min_dist, closest_ast = dist, ast
        
        if closest_ast:
            self.mining_target = closest_ast
            closest_ast['minerals'] -= self.MINING_RATE
            
            if self.np_random.random() < 0.6: # Spawn particles intermittently
                # sound: mining_zap.wav
                particle_vel = (self.player_pos - closest_ast['pos']) / self.np_random.uniform(20, 30)
                self.particles.append({
                    'pos': closest_ast['pos'].copy(), 'vel': particle_vel, 'type': 'mineral',
                    'lifetime': self.MINERAL_PARTICLE_LIFETIME, 'color': self.COLOR_MINERAL
                })

    def _handle_collisions(self):
        if self.invincibility_timer > 0: return
        
        for ast in self.asteroids:
            dist = np.linalg.norm(self.player_pos - ast['pos'])
            if dist < self.PLAYER_RADIUS + ast['radius']:
                self.lives -= 1
                self.step_reward += self.REWARD_COLLISION
                self.invincibility_timer = self.INVINCIBILITY_DURATION
                self._spawn_explosion(self.player_pos, self.player_vel)
                # sound: explosion.wav
                
                repulsion_vec = self.player_pos - ast['pos']
                repulsion_vec /= dist if dist > 0 else 1
                self.player_vel += repulsion_vec * 2.0
                ast['vel'] -= repulsion_vec * 0.5
                break

    def _update_difficulty(self):
        self.asteroid_speed_modifier = 1.0 + (self.steps // self.DIFFICULTY_INTERVAL) * 0.1

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.game_over, self.win = True, True
        elif self.lives <= 0:
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_asteroids()
        self._render_particles()
        self._render_player()
        if self.mining_target:
            self._render_mining_beam()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))
    
    def _render_asteroids(self):
        for ast in self.asteroids:
            rad = math.radians(ast['angle'])
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            
            points = []
            for p in ast['shape_points']:
                x = p[0] * cos_a - p[1] * sin_a + ast['pos'][0]
                y = p[0] * sin_a + p[1] * cos_a + ast['pos'][1]
                points.append((int(x), int(y)))
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID_OUTLINE)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_particles(self):
        for p in self.particles:
            size = 4 if p['type'] == 'explosion' else 2
            alpha = int(255 * (p['lifetime'] / (100 if p['type'] == 'explosion' else self.MINERAL_PARTICLE_LIFETIME)))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, size, size))
            self.screen.blit(temp_surf, (int(p['pos'][0] - size/2), int(p['pos'][1] - size/2)))

    def _render_player(self):
        if self.invincibility_timer > 0 and (self.steps // 3) % 2 == 0:
            return

        rad = math.radians(self.player_angle)
        p1 = self.player_pos + np.array([math.cos(rad), math.sin(rad)]) * self.PLAYER_RADIUS
        
        rad2 = math.radians(self.player_angle + 140)
        p2 = self.player_pos + np.array([math.cos(rad2), math.sin(rad2)]) * self.PLAYER_RADIUS * 0.8
        
        rad3 = math.radians(self.player_angle - 140)
        p3 = self.player_pos + np.array([math.cos(rad3), math.sin(rad3)]) * self.PLAYER_RADIUS * 0.8
        
        points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_mining_beam(self):
        start_pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        end_pos = (int(self.mining_target['pos'][0]), int(self.mining_target['pos'][1]))
        width = int(2 + math.sin(self.steps * 0.5) * 1.5)
        pygame.draw.aaline(self.screen, self.COLOR_BEAM, start_pos, end_pos, width)

    def _render_ui(self):
        score_text = self.FONT_UI.render(f"MINERALS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.FONT_UI.render(f"LIVES: {self.lives}", True, self.COLOR_UI)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        msg = "YOU WIN!" if self.win else "GAME OVER"
        color = self.COLOR_WIN if self.win else self.COLOR_LOSE
        text = self.FONT_MSG.render(msg, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)
        
    def _get_info(self):
        return { "score": self.score, "steps": self.steps, "lives": self.lives }
        
    def _get_closest_asteroid_dist(self):
        if not self.asteroids: return None
        min_dist = float('inf')
        for ast in self.asteroids:
            dist = np.linalg.norm(self.player_pos - ast['pos'])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _spawn_asteroids(self, num):
        for _ in range(num):
            while True:
                pos = self.np_random.uniform(low=0, high=[self.SCREEN_WIDTH, self.SCREEN_HEIGHT], size=2).astype(np.float32)
                if np.linalg.norm(pos - self.player_pos) > self.PLAYER_RADIUS + 70:
                    break
            
            radius = self.np_random.integers(15, 35)
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * self.ASTEROID_BASE_SPEED
            
            ast = {
                'pos': pos, 'vel': vel, 'radius': radius,
                'minerals': radius * 2.5,
                'angle': self.np_random.uniform(0, 360),
                'rot_speed': self.np_random.uniform(-1.5, 1.5),
                'shape_points': self._create_irregular_polygon_points(radius)
            }
            self.asteroids.append(ast)

    def _create_irregular_polygon_points(self, radius):
        num_vertices = self.np_random.integers(7, 12)
        points = []
        for i in range(num_vertices):
            angle = (2 * math.pi / num_vertices) * i
            dist = self.np_random.uniform(radius * 0.7, radius * 1.0)
            points.append((dist * math.cos(angle), dist * math.sin(angle)))
        return points

    def _spawn_explosion(self, position, initial_vel):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed + initial_vel * 0.5
            self.particles.append({
                'pos': position.copy(), 'vel': vel, 'type': 'explosion',
                'lifetime': self.np_random.integers(40, 100),
                'color': random.choice(self.EXPLOSION_COLORS)
            })
            
    def validate_implementation(self):
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