
# Generated: 2025-08-27T20:42:35.023601
# Source Brief: brief_02552.md
# Brief Index: 2552

        
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
        "Controls: Use arrow keys to move. Hold space near an asteroid to activate your mining beam."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship through asteroid fields, collecting valuable ore while dodging collisions to reach a target yield."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WIN_SCORE = 100
    MAX_STEPS = 1500 # Extended for more gameplay time

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (50, 150, 255, 30)
    COLOR_THRUSTER = (255, 180, 50)
    COLOR_ASTEROID = (120, 120, 120)
    COLOR_ASTEROID_DEPLETED = (60, 60, 60)
    COLOR_ORE = (255, 220, 0)
    COLOR_BEAM = (255, 255, 100, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_EXPLOSION = [(255, 100, 0), (255, 180, 50), (255, 50, 0)]

    # Player settings
    SHIP_SIZE = 12
    SHIP_ACCEL = 0.4
    SHIP_FRICTION = 0.98
    SHIP_MAX_SPEED = 6

    # Asteroid settings
    ASTEROID_COUNT = 7
    ASTEROID_MIN_SIZE = 20
    ASTEROID_MAX_SIZE = 40
    ASTEROID_MIN_SPEED = 0.1
    ASTEROID_MAX_SPEED = 0.6
    ASTEROID_MIN_ORE = 50
    ASTEROID_MAX_ORE = 150

    # Mining settings
    MINING_RANGE = 120
    MINING_RATE = 0.5
    
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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = 0
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # This will be initialized properly in reset()
        self.np_random = None
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = -90

        self.asteroids = []
        for _ in range(self.ASTEROID_COUNT):
            self.asteroids.append(self._create_asteroid())

        self.particles = []
        self._create_stars()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        terminated = False

        if not self.game_over:
            self._update_player(movement)
            self._update_asteroids()
            
            ore_mined, bonus = self._handle_mining(space_held)
            if ore_mined > 0:
                # Continuous reward for mining
                reward += ore_mined * 0.1
                # Event-based reward for crossing thresholds
                reward += bonus

            # Proximity penalty
            closest_asteroid, min_dist = self._get_closest_asteroid(self.player_pos)
            if min_dist > self.MINING_RANGE * 1.5:
                reward -= 0.001

            self._check_collisions()

        self._update_particles()
        self.steps += 1

        if self.game_over:
            if self.win:
                reward += 100 # Goal-oriented reward for winning
            else:
                reward -= 50 # Goal-oriented penalty for losing
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        accel = pygame.Vector2(0, 0)
        if movement == 1: accel.y = -self.SHIP_ACCEL
        elif movement == 2: accel.y = self.SHIP_ACCEL
        elif movement == 3: accel.x = -self.SHIP_ACCEL
        elif movement == 4: accel.x = self.SHIP_ACCEL

        self.player_vel += accel
        if self.player_vel.length() > self.SHIP_MAX_SPEED:
            self.player_vel.scale_to_length(self.SHIP_MAX_SPEED)

        self.player_vel *= self.SHIP_FRICTION
        self.player_pos += self.player_vel

        # Screen boundary checks
        self.player_pos.x = max(self.SHIP_SIZE, min(self.player_pos.x, self.SCREEN_WIDTH - self.SHIP_SIZE))
        self.player_pos.y = max(self.SHIP_SIZE, min(self.player_pos.y, self.SCREEN_HEIGHT - self.SHIP_SIZE))

        # Update angle
        if self.player_vel.length() > 0.1:
            self.player_angle = self.player_vel.angle_to(pygame.Vector2(1, 0))

        # Thruster particles
        if accel.length() > 0:
            # sfx: ship_thruster_loop
            self._create_thruster_particles()

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            if asteroid['pos'].x < -self.ASTEROID_MAX_SIZE: asteroid['pos'].x = self.SCREEN_WIDTH + self.ASTEROID_MAX_SIZE
            if asteroid['pos'].x > self.SCREEN_WIDTH + self.ASTEROID_MAX_SIZE: asteroid['pos'].x = -self.ASTEROID_MAX_SIZE
            if asteroid['pos'].y < -self.ASTEROID_MAX_SIZE: asteroid['pos'].y = self.SCREEN_HEIGHT + self.ASTEROID_MAX_SIZE
            if asteroid['pos'].y > self.SCREEN_HEIGHT + self.ASTEROID_MAX_SIZE: asteroid['pos'].y = -self.ASTEROID_MAX_SIZE

    def _handle_mining(self, space_held):
        ore_this_frame = 0
        bonus_reward = 0
        if space_held:
            closest_asteroid, min_dist = self._get_closest_asteroid(self.player_pos)
            if closest_asteroid and min_dist < self.MINING_RANGE and closest_asteroid['ore'] > 0:
                # sfx: mining_beam_loop
                mine_amount = (1 - (min_dist / self.MINING_RANGE)) * self.MINING_RATE
                mine_amount = min(mine_amount, closest_asteroid['ore'])
                
                old_score_bucket = math.floor(self.score / 10)
                self.score += mine_amount
                new_score_bucket = math.floor(self.score / 10)
                
                if new_score_bucket > old_score_bucket:
                    bonus_reward = 1.0 # Event-based reward
                
                closest_asteroid['ore'] -= mine_amount
                ore_this_frame = mine_amount

                self._create_ore_particles(closest_asteroid['pos'])
                if self.score >= self.WIN_SCORE:
                    self.score = self.WIN_SCORE
                    self.win = True
                    self.game_over = True
        return ore_this_frame, bonus_reward

    def _check_collisions(self):
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < self.SHIP_SIZE + asteroid['size']:
                # sfx: explosion_sound
                self.game_over = True
                self._create_explosion(self.player_pos)
                break

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_asteroids()
        
        # Render mining beam if active
        if not self.game_over and self.action_space.sample()[1] == 1:
             closest_asteroid, min_dist = self._get_closest_asteroid(self.player_pos)
             if closest_asteroid and min_dist < self.MINING_RANGE and closest_asteroid['ore'] > 0:
                self._render_mining_beam(closest_asteroid['pos'])
        
        self._render_particles()

        if not self.game_over:
            self._render_player()
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    # --- Helper and Rendering Methods ---

    def _render_player(self):
        # Glow effect
        glow_radius = int(self.SHIP_SIZE * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (self.player_pos.x - glow_radius, self.player_pos.y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Ship body
        angle_rad = math.radians(self.player_angle)
        p1 = self.player_pos + pygame.Vector2(self.SHIP_SIZE, 0).rotate(-self.player_angle)
        p2 = self.player_pos + pygame.Vector2(-self.SHIP_SIZE/2, self.SHIP_SIZE * 0.8).rotate(-self.player_angle)
        p3 = self.player_pos + pygame.Vector2(-self.SHIP_SIZE/2, -self.SHIP_SIZE * 0.8).rotate(-self.player_angle)
        points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            color = self.COLOR_ASTEROID if asteroid['ore'] > 0 else self.COLOR_ASTEROID_DEPLETED
            points = [(p + asteroid['pos']) for p in asteroid['shape']]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_mining_beam(self, target_pos):
        angle_rad = math.radians(self.player_angle)
        p_front = self.player_pos + pygame.Vector2(self.SHIP_SIZE, 0).rotate(-self.player_angle)
        
        vec_to_target = (target_pos - self.player_pos).normalize()
        p_side1 = target_pos + vec_to_target.rotate(90) * 10
        p_side2 = target_pos + vec_to_target.rotate(-90) * 10
        
        points = [p_front, p_side1, p_side2]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_BEAM)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BEAM)

    def _render_ui(self):
        score_text = self.font_small.render(f"ORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_small.render(f"TIME: {self.MAX_STEPS - self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        if self.game_over:
            if self.win:
                end_text = self.font_large.render("MISSION COMPLETE", True, self.COLOR_ORE)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_EXPLOSION[2])
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _create_asteroid(self):
        pos = pygame.Vector2(
            self.np_random.uniform(0, self.SCREEN_WIDTH),
            self.np_random.uniform(0, self.SCREEN_HEIGHT)
        )
        # Ensure asteroids don't spawn on the player
        while pos.distance_to(self.player_pos) < self.MINING_RANGE:
            pos = pygame.Vector2(
                self.np_random.uniform(0, self.SCREEN_WIDTH),
                self.np_random.uniform(0, self.SCREEN_HEIGHT)
            )
        
        angle = self.np_random.uniform(0, 360)
        speed = self.np_random.uniform(self.ASTEROID_MIN_SPEED, self.ASTEROID_MAX_SPEED)
        vel = pygame.Vector2(speed, 0).rotate(angle)
        size = self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        ore = self.np_random.uniform(self.ASTEROID_MIN_ORE, self.ASTEROID_MAX_ORE)
        
        num_vertices = self.np_random.integers(7, 12)
        irregularity = self.np_random.uniform(0.6, 1.0)
        shape_points = []
        for i in range(num_vertices):
            angle = (2 * math.pi / num_vertices) * i
            dist = size * self.np_random.uniform(irregularity, 1.0)
            shape_points.append(pygame.Vector2(math.cos(angle), math.sin(angle)) * dist)

        return {'pos': pos, 'vel': vel, 'size': size, 'ore': ore, 'shape': shape_points}
    
    def _create_stars(self):
        self.stars = []
        for _ in range(100):
            self.stars.append({
                'pos': pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                'size': self.np_random.uniform(0.5, 1.5)
            })

    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, (200, 200, 220), star['pos'], star['size'])

    def _get_closest_asteroid(self, pos):
        closest = None
        min_dist = float('inf')
        for asteroid in self.asteroids:
            dist = pos.distance_to(asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
                closest = asteroid
        return closest, min_dist

    # --- Particle System ---
    def _create_thruster_particles(self):
        for _ in range(2):
            vel_angle = self.player_vel.angle_to(pygame.Vector2(1,0))
            offset = pygame.Vector2(-self.SHIP_SIZE, 0).rotate(-vel_angle)
            pos = self.player_pos + offset
            vel = pygame.Vector2(-3, 0).rotate(-vel_angle) + self.player_vel * 0.5
            vel.rotate_ip(self.np_random.uniform(-15, 15))
            self.particles.append({
                'pos': pos, 'vel': vel, 'lifespan': 15, 'color': self.COLOR_THRUSTER, 'type': 'thruster'
            })

    def _create_ore_particles(self, asteroid_pos):
        for _ in range(3):
            start_pos = asteroid_pos + pygame.Vector2(self.np_random.uniform(-10, 10), self.np_random.uniform(-10, 10))
            vel = (self.player_pos - start_pos).normalize() * self.np_random.uniform(2, 4)
            self.particles.append({
                'pos': start_pos, 'vel': vel, 'lifespan': 60, 'color': self.COLOR_ORE, 'type': 'ore'
            })

    def _create_explosion(self, pos):
        for _ in range(50):
            vel = pygame.Vector2(1,0).rotate(self.np_random.uniform(0, 360)) * self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'lifespan': 40, 'color': random.choice(self.COLOR_EXPLOSION), 'type': 'explosion'
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['type'] == 'explosion':
                p['vel'] *= 0.95
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 20))))
            color = (*p['color'], alpha)
            size = 2 if p['type'] != 'explosion' else max(1, p['lifespan'] * 0.1)
            
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (p['pos'].x - size, p['pos'].y - size), special_flags=pygame.BLEND_RGBA_ADD)


    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=0)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to test the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # Create a window to display the game
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            print("Press 'R' to reset.")
        
        # Control the frame rate
        env.clock.tick(30)

    pygame.quit()