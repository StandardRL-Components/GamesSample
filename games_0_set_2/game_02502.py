
# Generated: 2025-08-27T20:33:47.715745
# Source Brief: brief_02502.md
# Brief Index: 2502

        
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
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    user_guide = (
        "Controls: ↑ to thrust, ←→ to turn, and ↓ to brake. "
        "Hold space to activate your mining beam on nearby asteroids."
    )

    game_description = (
        "Pilot a spaceship to mine asteroids for ore. Gather 100 ore units to win, "
        "but avoid collisions. The game has three stages with increasing difficulty."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (60, 180, 255)
        self.COLOR_THRUSTER = (255, 180, 60)
        self.COLOR_ASTEROID = (120, 100, 90)
        self.COLOR_ORE = (255, 220, 50)
        self.COLOR_EXPLOSION = [(255, 100, 0), (255, 50, 0), (200, 0, 0)]
        self.COLOR_BEAM = (100, 255, 100, 100) # RGBA
        self.COLOR_UI = (230, 230, 230)
        self.COLOR_STAR = (200, 200, 220)

        # Game constants
        self.FPS = 30
        self.MAX_LIVES = 3
        self.WIN_SCORE = 100
        self.STAGE_TIMELIMIT_SECONDS = 60
        self.INVULNERABILITY_FRAMES = 90 # 3 seconds

        # Physics constants
        self.PLAYER_ACCELERATION = 0.25
        self.PLAYER_ROTATION_SPEED = 4.5
        self.PLAYER_FRICTION = 0.985
        self.PLAYER_BRAKE_FRICTION = 0.95
        self.PLAYER_MAX_SPEED = 6

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.player_radius = None
        self.lives = None
        self.score = None
        self.stage = None
        self.stage_timer = None
        self.invulnerable_timer = None
        self.asteroids = None
        self.particles = None
        self.starfield = None
        self.game_over = None
        self.steps = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = -90
        self.player_radius = 12

        self.lives = self.MAX_LIVES
        self.score = 0
        self.stage = 1
        self.stage_timer = self.STAGE_TIMELIMIT_SECONDS * self.FPS
        self.invulnerable_timer = self.INVULNERABILITY_FRAMES

        self.asteroids = []
        self.particles = []
        self._spawn_asteroids(10)

        if self.starfield is None:
            self.starfield = [
                (self.np_random.integers(0, self.WIDTH),
                 self.np_random.integers(0, self.HEIGHT),
                 self.np_random.uniform(0.5, 1.5))
                for _ in range(150)
            ]

        self.game_over = False
        self.steps = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        if not self.game_over:
            self.steps += 1
            self._handle_input(movement)
            self._update_player()
            self._update_asteroids()
            self._update_particles()

            ore_mined, large_asteroids_destroyed = self._handle_mining(space_held)
            if ore_mined > 0:
                self.score = min(self.WIN_SCORE, self.score + ore_mined)
                reward += ore_mined * 0.1 # Continuous reward for mining
            else:
                reward -= 0.001 # Small penalty for not mining

            reward += large_asteroids_destroyed * 1.0 # Event reward for destroying large asteroids

            if self.invulnerable_timer > 0:
                self.invulnerable_timer -= 1
            else:
                if self._handle_collisions():
                    self.lives -= 1
                    self.invulnerable_timer = self.INVULNERABILITY_FRAMES
                    self._spawn_particles(self.player_pos, 50, self.COLOR_EXPLOSION, (20, 50), (1, 4), 'explosion')
                    # sfx: player_explosion.wav
                    self.player_pos.update(self.WIDTH / 2, self.HEIGHT / 2)
                    self.player_vel.update(0, 0)

            self.stage_timer -= 1
            self._check_stage_progression()
            if not self.asteroids:
                self._spawn_asteroids(10)


        terminated = self._check_termination()
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward = 100.0 # Goal-oriented win reward
            else:
                reward = -50.0 # Goal-oriented loss reward
        
        return (
            self._get_observation(space_held),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        # Rotate
        if movement == 3: # Left
            self.player_angle -= self.PLAYER_ROTATION_SPEED
        if movement == 4: # Right
            self.player_angle += self.PLAYER_ROTATION_SPEED

        # Thrust / Brake
        if movement == 1: # Up
            rad_angle = math.radians(self.player_angle)
            acceleration = pygame.math.Vector2(math.cos(rad_angle), math.sin(rad_angle)) * self.PLAYER_ACCELERATION
            self.player_vel += acceleration
            # sfx: thruster_loop.wav
            # Spawn thruster particles
            if self.steps % 2 == 0:
                self._spawn_particles(self.player_pos, 1, [self.COLOR_THRUSTER], (10, 20), (1, 2), 'thruster', -self.player_angle)
        elif movement == 2: # Down
            self.player_vel *= self.PLAYER_BRAKE_FRICTION
        
    def _update_player(self):
        # Apply friction and speed limit
        self.player_vel *= self.PLAYER_FRICTION
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)
        
        self.player_pos += self.player_vel

        # Screen wrap
        self.player_pos.x %= self.WIDTH
        self.player_pos.y %= self.HEIGHT

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] += asteroid['rot_speed']
            asteroid['pos'].x %= self.WIDTH
            asteroid['pos'].y %= self.HEIGHT

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['pos'] += p['vel']

    def _handle_collisions(self):
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < self.player_radius + asteroid['size']:
                return True
        return False

    def _handle_mining(self, space_held):
        ore_mined_total = 0
        large_asteroids_destroyed = 0
        if not space_held:
            return 0, 0

        rad_angle = math.radians(self.player_angle)
        player_forward = pygame.math.Vector2(math.cos(rad_angle), math.sin(rad_angle))

        for asteroid in self.asteroids[:]:
            vec_to_asteroid = asteroid['pos'] - self.player_pos
            dist = vec_to_asteroid.length()

            if 20 < dist < 100: # Mining range
                if vec_to_asteroid.length() > 0:
                    angle_diff = player_forward.angle_to(vec_to_asteroid)
                    if abs(angle_diff) < 25: # Mining cone angle
                        # sfx: mining_beam.wav
                        ore_to_mine = 1
                        asteroid['ore'] -= ore_to_mine
                        ore_mined_total += ore_to_mine
                        
                        # Spawn ore particles
                        self._spawn_particles(asteroid['pos'], 1, [self.COLOR_ORE], (30, 60), (0.5, 1.5), 'ore', target=self.player_pos)

                        if asteroid['ore'] <= 0:
                            if asteroid['initial_ore'] >= 20:
                                large_asteroids_destroyed += 1
                            # sfx: asteroid_destroyed.wav
                            self._spawn_particles(asteroid['pos'], int(asteroid['size']), [self.COLOR_ASTEROID], (15, 30), (0.5, 2), 'explosion')
                            self.asteroids.remove(asteroid)
        return ore_mined_total, large_asteroids_destroyed

    def _check_stage_progression(self):
        new_stage = self.stage
        if self.score > 66:
            new_stage = 3
        elif self.score > 33:
            new_stage = 2
        
        if new_stage > self.stage:
            self.stage = new_stage
            self.stage_timer = self.STAGE_TIMELIMIT_SECONDS * self.FPS
            self.asteroids.clear()
            self.particles.clear()
            self._spawn_asteroids(10)
            # sfx: stage_up.wav

    def _check_termination(self):
        if self.score >= self.WIN_SCORE or self.lives <= 0 or self.stage_timer <= 0:
            self.game_over = True
            return True
        return False

    def _spawn_asteroids(self, count):
        for _ in range(count):
            while True:
                pos = pygame.math.Vector2(
                    self.np_random.integers(0, self.WIDTH),
                    self.np_random.integers(0, self.HEIGHT)
                )
                if pos.distance_to(self.player_pos) > 100: # Don't spawn on player
                    break
            
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(0.5, 1.5) + (self.stage - 1) * 0.5
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            size = self.np_random.uniform(15, 35)
            
            num_points = self.np_random.integers(7, 12)
            shape_points = []
            for i in range(num_points):
                angle = (i / num_points) * 360 + self.np_random.uniform(-10, 10)
                dist = size + self.np_random.uniform(-size*0.3, size*0.3)
                shape_points.append(
                    (dist * math.cos(math.radians(angle)), dist * math.sin(math.radians(angle)))
                )
            
            initial_ore = int(size * 0.8)

            self.asteroids.append({
                'pos': pos, 'vel': vel, 'size': size,
                'angle': self.np_random.uniform(0, 360),
                'rot_speed': self.np_random.uniform(-1, 1),
                'shape': shape_points,
                'ore': initial_ore, 'initial_ore': initial_ore
            })

    def _spawn_particles(self, pos, count, colors, life_range, speed_range, p_type='explosion', angle_offset=0, target=None):
        for _ in range(count):
            if p_type == 'explosion':
                angle = self.np_random.uniform(0, 360)
                speed = self.np_random.uniform(*speed_range)
                vel = pygame.math.Vector2(math.cos(math.radians(angle)), math.sin(math.radians(angle))) * speed
            elif p_type == 'thruster':
                angle = angle_offset + 180 + self.np_random.uniform(-15, 15)
                speed = self.np_random.uniform(*speed_range)
                vel = pygame.math.Vector2(math.cos(math.radians(angle)), math.sin(math.radians(angle))) * speed
            elif p_type == 'ore':
                vec_to_target = (target - pos)
                if vec_to_target.length() > 0:
                    vel = vec_to_target.normalize() * self.np_random.uniform(*speed_range)
                else:
                    vel = pygame.math.Vector2(0,0)
            
            self.particles.append({
                'pos': pos.copy(), 'vel': vel,
                'life': self.np_random.integers(*life_range),
                'color': random.choice(colors)
            })

    def _get_observation(self, space_held=False):
        self.screen.fill(self.COLOR_BG)
        self._render_starfield()
        self._render_asteroids()
        self._render_particles()
        self._render_player(space_held)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_starfield(self):
        for x, y, size in self.starfield:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = []
            for p in asteroid['shape']:
                rotated_p = pygame.math.Vector2(p).rotate(asteroid['angle'])
                points.append(asteroid['pos'] + rotated_p)
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], max(1, p['life'] / 10))

    def _render_player(self, space_held):
        # Draw mining beam first
        if space_held:
            rad_angle = math.radians(self.player_angle)
            p1 = self.player_pos
            p2 = p1 + pygame.math.Vector2(math.cos(rad_angle - 0.4), math.sin(rad_angle - 0.4)) * 80
            p3 = p1 + pygame.math.Vector2(math.cos(rad_angle + 0.4), math.sin(rad_angle + 0.4)) * 80
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_BEAM)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_BEAM)

        # Draw player ship
        rad_angle = math.radians(self.player_angle)
        s, c = math.sin(rad_angle), math.cos(rad_angle)
        
        p1 = self.player_pos + pygame.math.Vector2(c, s) * self.player_radius
        p2 = self.player_pos + pygame.math.Vector2(-s, c) * (self.player_radius * 0.7) - pygame.math.Vector2(c, s) * (self.player_radius * 0.7)
        p3 = self.player_pos + pygame.math.Vector2(s, -c) * (self.player_radius * 0.7) - pygame.math.Vector2(c, s) * (self.player_radius * 0.7)
        
        color = self.COLOR_PLAYER
        if self.invulnerable_timer > 0 and (self.invulnerable_timer // 4) % 2 == 0:
             color = (128, 220, 255) # Lighter color when invulnerable

        pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), color)
        pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), color)

    def _render_ui(self):
        # Ore Count
        ore_text = self.font_small.render(f"ORE: {self.score} / {self.WIN_SCORE}", True, self.COLOR_UI)
        self.screen.blit(ore_text, (10, 10))

        # Stage Timer
        time_left = max(0, self.stage_timer // self.FPS)
        timer_text = self.font_small.render(f"TIME: {time_left}", True, self.COLOR_UI)
        self.screen.blit(timer_text, (self.WIDTH // 2 - timer_text.get_width() // 2, 10))

        # Lives
        for i in range(self.lives):
            ship_icon_points = [
                (self.WIDTH - 25 - i * 25, 15),
                (self.WIDTH - 35 - i * 25, 30),
                (self.WIDTH - 15 - i * 25, 30)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_UI, ship_icon_points)
        
        # Game Over / Win Text
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_UI)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
            "time_left": max(0, self.stage_timer // self.FPS)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        clock.tick(env.FPS)
        
    env.close()