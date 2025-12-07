
# Generated: 2025-08-27T20:13:57.147785
# Source Brief: brief_02394.md
# Brief Index: 2394

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold space near an asteroid to mine ore. Dodge the red lasers!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Mine asteroids for valuable ore in deep space, but watch out for deadly security lasers. "
        "Collect 1000 ore to win, but lose all your lives and it's game over."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.W, self.H = 640, 400
        self.FPS = 30
        self.SHIP_SIZE = 12
        self.SHIP_ACCEL = 0.8
        self.SHIP_FRICTION = 0.92
        self.SHIP_MAX_SPEED = 8
        self.SHIP_INVULNERABLE_DURATION = 90  # 3 seconds at 30fps
        self.MINE_RADIUS = 60
        self.MINE_RATE = 1 # ore per frame
        self.LASER_SPEED = 5
        self.ASTEROID_MIN_ORE = 10
        self.ASTEROID_MAX_ORE = 50
        self.MAX_ASTEROIDS = 8
        self.ORE_GOAL = 1000
        self.MAX_STEPS = 5000
        self.INITIAL_LIVES = 3

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_SHIP = (60, 180, 255)
        self.COLOR_SHIP_THRUST = (255, 200, 100)
        self.COLOR_ASTEROID = (120, 120, 130)
        self.COLOR_LASER = (255, 20, 20)
        self.COLOR_ORE = (255, 220, 0)
        self.COLOR_EXPLOSION = [(255, 50, 50), (255, 150, 50), (255, 255, 100)]
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.win = False

        self.ship_pos = np.array([self.W / 2, self.H / 2], dtype=np.float32)
        self.ship_vel = np.array([0, 0], dtype=np.float32)
        self.ship_angle = -90.0
        self.ship_invulnerable_timer = 0
        
        self.asteroids = []
        self.lasers = []
        self.particles = []
        self.max_lasers = 1

        self.stars = [(self.np_random.integers(0, self.W), self.np_random.integers(0, self.H), self.np_random.integers(1, 3)) for _ in range(150)]
        
        while len(self.asteroids) < self.MAX_ASTEROIDS:
            self._spawn_asteroid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = self.game_over

        if not terminated:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            self._handle_input(movement)
            self._update_ship(movement)
            
            mining_reward, is_mining = self._update_mining(space_held)
            reward += mining_reward
            
            laser_reward = self._update_lasers()
            reward += laser_reward
            
            self._update_asteroids(is_mining)
            self._update_particles()
            self._update_difficulty()

            self.steps += 1

            if self.score >= self.ORE_GOAL:
                self.game_over = True
                self.win = True
                terminated = True
                reward += 100
            elif self.lives <= 0:
                self.game_over = True
                self.win = False
                terminated = True
                reward += -50
            elif self.steps >= self.MAX_STEPS:
                terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        accel = np.array([0, 0], dtype=np.float32)
        if movement == 1: accel[1] -= self.SHIP_ACCEL # Up
        if movement == 2: accel[1] += self.SHIP_ACCEL # Down
        if movement == 3: accel[0] -= self.SHIP_ACCEL # Left
        if movement == 4: accel[0] += self.SHIP_ACCEL # Right
        
        self.ship_vel += accel
        speed = np.linalg.norm(self.ship_vel)
        if speed > self.SHIP_MAX_SPEED:
            self.ship_vel = self.ship_vel / speed * self.SHIP_MAX_SPEED

    def _update_ship(self, movement):
        self.ship_vel *= self.SHIP_FRICTION
        self.ship_pos += self.ship_vel

        # Screen bounds
        self.ship_pos[0] = np.clip(self.ship_pos[0], self.SHIP_SIZE, self.W - self.SHIP_SIZE)
        self.ship_pos[1] = np.clip(self.ship_pos[1], self.SHIP_SIZE, self.H - self.SHIP_SIZE)
        
        # Angle for rendering
        if np.linalg.norm(self.ship_vel) > 0.1:
            self.ship_angle = math.degrees(math.atan2(self.ship_vel[1], self.ship_vel[0]))

        if self.ship_invulnerable_timer > 0:
            self.ship_invulnerable_timer -= 1
        
        # Thruster particles
        if movement != 0 and not self.game_over:
            angle_rad = math.radians(self.ship_angle + 180)
            offset = np.array([math.cos(angle_rad), math.sin(angle_rad)]) * self.SHIP_SIZE
            for _ in range(2):
                p_vel = -self.ship_vel * 0.2 + self.np_random.uniform(-0.5, 0.5, 2)
                self.particles.append({
                    "pos": self.ship_pos + offset,
                    "vel": p_vel,
                    "life": 10,
                    "color": self.COLOR_SHIP_THRUST,
                    "radius": self.np_random.integers(1, 4)
                })

    def _update_mining(self, space_held):
        reward = 0
        is_mining = False
        
        if not self.asteroids: return reward, is_mining

        dists = [np.linalg.norm(self.ship_pos - a['pos']) for a in self.asteroids]
        closest_idx = np.argmin(dists)
        closest_asteroid = self.asteroids[closest_idx]
        dist_to_closest = dists[closest_idx]

        if dist_to_closest < self.MINE_RADIUS + closest_asteroid['radius']:
            if space_held and closest_asteroid['ore'] > 0:
                is_mining = True
                # # Sound: mining_beam.wav
                if not closest_asteroid.get('is_being_mined', False):
                    reward += 1 # Bonus for starting to mine an asteroid
                    closest_asteroid['is_being_mined'] = True

                ore_mined = min(self.MINE_RATE, closest_asteroid['ore'])
                self.score += ore_mined
                closest_asteroid['ore'] -= ore_mined
                reward += ore_mined * 0.1

                for _ in range(int(ore_mined * 2)):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    start_pos = closest_asteroid['pos'] + np.array([math.cos(angle), math.sin(angle)]) * closest_asteroid['radius']
                    self.particles.append({
                        "pos": start_pos,
                        "vel": (self.ship_pos - start_pos) / 20.0, # Move towards ship
                        "life": 20,
                        "color": self.COLOR_ORE,
                        "radius": 2
                    })
            else:
                reward -= 0.01 # Penalty for being near but not mining
                closest_asteroid['is_being_mined'] = False
        
        for i, asteroid in enumerate(self.asteroids):
            if i != closest_idx:
                asteroid['is_being_mined'] = False

        return reward, is_mining

    def _update_lasers(self):
        reward = 0
        if len(self.lasers) < self.max_lasers and self.np_random.random() < 0.02:
            self._spawn_laser()
        
        ship_rect = pygame.Rect(self.ship_pos[0] - self.SHIP_SIZE / 2, self.ship_pos[1] - self.SHIP_SIZE / 2, self.SHIP_SIZE, self.SHIP_SIZE)
        
        for laser in self.lasers[:]:
            laser['pos'] += laser['vel']
            if not self.screen.get_rect().colliderect(laser['rect']):
                self.lasers.remove(laser)
                continue
            
            laser_rect = laser['rect']
            laser_rect.topleft = laser['pos']

            if laser_rect.colliderect(ship_rect) and self.ship_invulnerable_timer == 0:
                self.lives -= 1
                reward -= 10
                self.ship_invulnerable_timer = self.SHIP_INVULNERABLE_DURATION
                self._create_explosion(self.ship_pos, 30)
                # # Sound: ship_hit.wav
                if self.lives <= 0:
                    # # Sound: game_over.wav
                    self._create_explosion(self.ship_pos, 80)
                break
        return reward

    def _update_asteroids(self, is_mining):
        for asteroid in self.asteroids[:]:
            if asteroid['ore'] <= 0:
                self.asteroids.remove(asteroid)
                self._create_explosion(asteroid['pos'], int(asteroid['radius']))
                # # Sound: asteroid_break.wav
            else:
                if not is_mining: # Asteroids drift when not being mined
                    asteroid['pos'] += asteroid['vel']
                    if asteroid['pos'][0] < -50 or asteroid['pos'][0] > self.W + 50 or \
                       asteroid['pos'][1] < -50 or asteroid['pos'][1] > self.H + 50:
                       self.asteroids.remove(asteroid)

        while len(self.asteroids) < self.MAX_ASTEROIDS:
            self._spawn_asteroid()

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_difficulty(self):
        self.max_lasers = 1 + self.score // 200
        if self.max_lasers > 5:
            self.max_lasers = 5

    def _spawn_asteroid(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.W), -50.0])
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.W), self.H + 50.0])
        elif edge == 2: # Left
            pos = np.array([-50.0, self.np_random.uniform(0, self.H)])
        else: # Right
            pos = np.array([self.W + 50.0, self.np_random.uniform(0, self.H)])
        
        vel = (np.array([self.W/2, self.H/2]) - pos) / 500.0 # Move towards center
        vel += self.np_random.uniform(-0.1, 0.1, 2)
        
        ore = self.np_random.integers(self.ASTEROID_MIN_ORE, self.ASTEROID_MAX_ORE + 1)
        radius = 10 + (ore / self.ASTEROID_MAX_ORE) * 20
        
        num_points = self.np_random.integers(7, 12)
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dist = self.np_random.uniform(0.7, 1.0) * radius
            points.append([math.cos(angle) * dist, math.sin(angle) * dist])
        
        self.asteroids.append({'pos': pos, 'vel': vel, 'ore': ore, 'radius': radius, 'points': points, 'is_being_mined': False})

    def _spawn_laser(self):
        # # Sound: laser_fire.wav
        side = self.np_random.integers(4)
        if side == 0: # From top
            pos = np.array([self.np_random.uniform(0, self.W), -20.0])
            vel = np.array([0, self.LASER_SPEED])
            rect = pygame.Rect(0, 0, 4, 20)
        elif side == 1: # From bottom
            pos = np.array([self.np_random.uniform(0, self.W), self.H])
            vel = np.array([0, -self.LASER_SPEED])
            rect = pygame.Rect(0, 0, 4, 20)
        elif side == 2: # From left
            pos = np.array([-20.0, self.np_random.uniform(0, self.H)])
            vel = np.array([self.LASER_SPEED, 0])
            rect = pygame.Rect(0, 0, 20, 4)
        else: # From right
            pos = np.array([self.W, self.np_random.uniform(0, self.H)])
            vel = np.array([-self.LASER_SPEED, 0])
            rect = pygame.Rect(0, 0, 20, 4)
        
        self.lasers.append({'pos': pos, 'vel': vel, 'rect': rect})

    def _create_explosion(self, pos, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": self.np_random.choice(self.COLOR_EXPLOSION),
                "radius": self.np_random.integers(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_particles()
        self._render_asteroids()
        self._render_lasers()
        if self.lives > 0:
            self._render_ship()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for x, y, r in self.stars:
            c = self.np_random.integers(50, 100)
            pygame.draw.circle(self.screen, (c, c, c), (x, y), r)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            if alpha < 0: alpha = 0
            color = p['color']
            
            # Using gfxdraw for antialiasing and alpha blending
            px, py = int(p['pos'][0]), int(p['pos'][1])
            pr = int(p['radius'] * (p['life'] / 30))
            if pr > 0:
                pygame.gfxdraw.filled_circle(self.screen, px, py, pr, (*color, alpha))
                pygame.gfxdraw.aacircle(self.screen, px, py, pr, (*color, alpha))

    def _render_asteroids(self):
        for a in self.asteroids:
            points = [(p[0] + a['pos'][0], p[1] + a['pos'][1]) for p in a['points']]
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_lasers(self):
        for laser in self.lasers:
            # Glow effect
            glow_rect = laser['rect'].inflate(laser['rect'].width*2, laser['rect'].height*2)
            glow_rect.center = laser['rect'].center
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_LASER, 50), s.get_rect(), border_radius=3)
            self.screen.blit(s, glow_rect.topleft)
            # Core beam
            pygame.draw.rect(self.screen, (255, 200, 200), laser['rect'], border_radius=3)
    
    def _render_ship(self):
        if self.ship_invulnerable_timer > 0 and self.steps % 6 < 3:
            return # Flicker effect

        x, y = self.ship_pos
        angle_rad = math.radians(self.ship_angle)
        
        p1 = (x + math.cos(angle_rad) * self.SHIP_SIZE, y + math.sin(angle_rad) * self.SHIP_SIZE)
        p2 = (x + math.cos(angle_rad + 2.2) * self.SHIP_SIZE, y + math.sin(angle_rad + 2.2) * self.SHIP_SIZE)
        p3 = (x + math.cos(angle_rad - 2.2) * self.SHIP_SIZE, y + math.sin(angle_rad - 2.2) * self.SHIP_SIZE)
        
        points = [p1, p2, p3]
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_SHIP)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"ORE: {self.score}/{self.ORE_GOAL}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            ship_icon_points = [
                (self.W - 20 - i*25, 25),
                (self.W - 35 - i*25, 15),
                (self.W - 35 - i*25, 35)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_SHIP, ship_icon_points)

        if self.game_over:
            if self.win:
                msg = "MISSION COMPLETE"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # This allows you to play the game with keyboard controls.
    
    # Set SDL to use a real video driver
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # 'windows', 'x11', or 'macOS'
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Asteroid Miner")
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause before closing

    env.close()