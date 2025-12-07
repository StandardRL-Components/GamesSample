
# Generated: 2025-08-28T03:05:05.544574
# Source Brief: brief_04816.md
# Brief Index: 4816

        
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

    user_guide = (
        "Controls: Arrow keys to move your ship. Hold Space to mine nearby asteroids for ore. Avoid collisions!"
    )

    game_description = (
        "Pilot a spaceship through a dense asteroid field. Mine asteroids for valuable ore to "
        "reach the collection goal, but be careful: one collision means game over."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        self.WIN_SCORE = 100
        self.NUM_ASTEROIDS = 10
        self.PLAYER_SPEED = 5
        self.PLAYER_DRAG = 0.92
        self.ASTEROID_MIN_ORE = 20
        self.ASTEROID_MAX_ORE = 80
        self.MINING_RANGE = 80
        self.MINING_RATE = 1.0

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 64)

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 200)
        self.COLOR_ASTEROID_DARK = (50, 50, 60)
        self.COLOR_ASTEROID_LIGHT = (80, 80, 90)
        self.COLOR_ORE = (255, 200, 0)
        self.COLOR_BEAM = (255, 255, 100, 150)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_EXPLOSION = [(255, 50, 50), (255, 150, 0), (255, 255, 0)]
        self.STAR_COLORS = [(50, 50, 70), (100, 100, 120), (150, 150, 180)]

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self._generate_stars()
        
        # This will be properly initialized in reset()
        self.np_random = None

        self.reset()
        
        # Self-validation check
        self.validate_implementation()

    def _generate_stars(self):
        self.stars = []
        for _ in range(200):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.choice([1, 1, 1, 2, 2, 3])
            color = self.STAR_COLORS[size - 1]
            self.stars.append({'pos': [x, y], 'size': size, 'color': color})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float64)

        self.asteroids = []
        while len(self.asteroids) < self.NUM_ASTEROIDS:
            self._spawn_asteroid()

        self.particles = []

        return self._get_observation(), self._get_info()

    def _spawn_asteroid(self, position=None):
        if position is None:
            # Spawn off-screen to avoid popping in
            side = self.np_random.integers(4)
            if side == 0: # top
                x = self.np_random.uniform(0, self.WIDTH)
                y = -50
            elif side == 1: # right
                x = self.WIDTH + 50
                y = self.np_random.uniform(0, self.HEIGHT)
            elif side == 2: # bottom
                x = self.np_random.uniform(0, self.WIDTH)
                y = self.HEIGHT + 50
            else: # left
                x = -50
                y = self.np_random.uniform(0, self.HEIGHT)
            pos = np.array([x, y], dtype=np.float64)
        else:
            pos = np.array(position, dtype=np.float64)

        size = self.np_random.uniform(15, 40)
        ore_content = int(size / 40 * (self.ASTEROID_MAX_ORE - self.ASTEROID_MIN_ORE) + self.ASTEROID_MIN_ORE)
        
        # Create a procedural shape
        num_vertices = self.np_random.integers(7, 12)
        base_vertices = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            radius = size * self.np_random.uniform(0.8, 1.2)
            base_vertices.append((radius * math.cos(angle), radius * math.sin(angle)))

        self.asteroids.append({
            'pos': pos,
            'vel': self.np_random.uniform(-0.5, 0.5, size=2),
            'size': size,
            'ore': ore_content,
            'max_ore': ore_content,
            'angle': self.np_random.uniform(0, 2 * math.pi),
            'rot_speed': self.np_random.uniform(-0.02, 0.02),
            'base_vertices': base_vertices
        })

    def step(self, action):
        reward = -0.02  # Small penalty for each step

        if not self.game_over:
            # Unpack action
            movement, space_held, _ = action
            space_held = space_held == 1

            # === Update Player ===
            if movement == 1: self.player_vel[1] -= self.PLAYER_SPEED / self.FPS
            if movement == 2: self.player_vel[1] += self.PLAYER_SPEED / self.FPS
            if movement == 3: self.player_vel[0] -= self.PLAYER_SPEED / self.FPS
            if movement == 4: self.player_vel[0] += self.PLAYER_SPEED / self.FPS
            
            self.player_pos += self.player_vel
            self.player_vel *= self.PLAYER_DRAG

            # Boundary checks
            self.player_pos[0] = np.clip(self.player_pos[0], 15, self.WIDTH - 15)
            self.player_pos[1] = np.clip(self.player_pos[1], 15, self.HEIGHT - 15)

            # === Update Asteroids ===
            for a in self.asteroids:
                a['pos'] += a['vel']
                a['angle'] += a['rot_speed']
                # Screen wrap for asteroids
                if a['pos'][0] < -60: a['pos'][0] = self.WIDTH + 60
                if a['pos'][0] > self.WIDTH + 60: a['pos'][0] = -60
                if a['pos'][1] < -60: a['pos'][1] = self.HEIGHT + 60
                if a['pos'][1] > self.HEIGHT + 60: a['pos'][1] = -60

            # === Handle Mining ===
            mined_this_step = False
            if space_held:
                closest_asteroid = None
                min_dist = self.MINING_RANGE
                for a in self.asteroids:
                    dist = np.linalg.norm(self.player_pos - a['pos'])
                    if dist < min_dist:
                        min_dist = dist
                        closest_asteroid = a
                
                if closest_asteroid:
                    # sfx: mining_beam_loop.wav
                    mined_amount = self.MINING_RATE
                    mined_amount = min(mined_amount, closest_asteroid['ore'])
                    
                    closest_asteroid['ore'] -= mined_amount
                    self.score += mined_amount
                    reward += mined_amount * 0.1
                    mined_this_step = True

                    # Mining particles
                    for _ in range(2):
                        angle = self.np_random.uniform(0, 2 * math.pi)
                        start_pos = closest_asteroid['pos'] + np.array([math.cos(angle), math.sin(angle)]) * closest_asteroid['size'] * 0.5
                        self.particles.append(self._create_particle(start_pos, self.COLOR_ORE, 2, 0.8, target=self.player_pos))
            
            # If no ore was mined, the step penalty remains
            if mined_this_step:
                reward -= -0.02 # cancel out the step penalty

            # === Handle Asteroid Depletion ===
            asteroids_to_remove = [a for a in self.asteroids if a['ore'] <= 0]
            for a in asteroids_to_remove:
                # sfx: asteroid_depleted.wav
                self.asteroids.remove(a)
                reward += 1.0 # Bonus for finishing an asteroid
                self._spawn_asteroid()

            # === Handle Collisions ===
            for a in self.asteroids:
                dist = np.linalg.norm(self.player_pos - a['pos'])
                if dist < a['size'] + 10: # 10 is player radius
                    self.game_over = True
                    reward = -50.0
                    # sfx: explosion.wav
                    self._create_explosion(self.player_pos, 100)
                    break
        
        # === Update Particles ===
        self._update_particles()

        # === Check Termination Conditions ===
        self.steps += 1
        terminated = self.game_over
        if not terminated:
            if self.score >= self.WIN_SCORE:
                self.score = self.WIN_SCORE
                self.game_over = True
                self.win = True
                terminated = True
                reward = 100.0
                # sfx: win_jingle.wav
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
                terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particle(self, pos, color, size, lifespan, vel=None, target=None, spread=0.5):
        if target is not None:
            direction = (target - pos)
            dist = np.linalg.norm(direction)
            if dist < 1: dist = 1
            vel = (direction / dist) * self.np_random.uniform(2, 4)
            vel += self.np_random.uniform(-spread, spread, size=2)
        elif vel is None:
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        
        return {
            'pos': np.array(pos, dtype=np.float64),
            'vel': vel,
            'lifespan': lifespan,
            'max_lifespan': lifespan,
            'size': size,
            'color': color
        }

    def _create_explosion(self, pos, count):
        for _ in range(count):
            color = random.choice(self.COLOR_EXPLOSION)
            particle = self._create_particle(pos, color, self.np_random.uniform(1, 4), self.np_random.uniform(0.5, 1.5))
            particle['vel'] *= 1.5 # Explosions are faster
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1.0 / self.FPS
            p['vel'] *= 0.98 # Particle drag
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render stars
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['size'] / 2)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Render mining beam if active
        space_held = self.action_space.sample()[1] == 1 # A bit of a hack to get last action for rendering
        if not self.game_over and space_held:
            closest_asteroid = None
            min_dist = self.MINING_RANGE
            for a in self.asteroids:
                dist = np.linalg.norm(self.player_pos - a['pos'])
                if dist < min_dist:
                    min_dist = dist
                    closest_asteroid = a
            if closest_asteroid:
                flicker = self.np_random.uniform(0.8, 1.2)
                start_pos = tuple(self.player_pos.astype(int))
                end_pos = tuple(closest_asteroid['pos'].astype(int))
                pygame.draw.line(self.screen, self.COLOR_BEAM, start_pos, end_pos, int(3 * flicker))

        # Render asteroids
        for a in self.asteroids:
            # Rotate vertices
            rotated_vertices = []
            for vx, vy in a['base_vertices']:
                rx = vx * math.cos(a['angle']) - vy * math.sin(a['angle']) + a['pos'][0]
                ry = vx * math.sin(a['angle']) + vy * math.cos(a['angle']) + a['pos'][1]
                rotated_vertices.append((int(rx), int(ry)))
            
            if len(rotated_vertices) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, rotated_vertices, self.COLOR_ASTEROID_DARK)
                pygame.gfxdraw.aapolygon(self.screen, rotated_vertices, self.COLOR_ASTEROID_LIGHT)
        
        # Render player
        if not self.game_over:
            # Ship body
            p1 = (self.player_pos[0], self.player_pos[1] - 14)
            p2 = (self.player_pos[0] - 10, self.player_pos[1] + 10)
            p3 = (self.player_pos[0] + 10, self.player_pos[1] + 10)
            
            # Glow effect
            glow_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.draw.polygon(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), [(20, 6), (10, 30), (30, 30)])
            self.screen.blit(glow_surf, (int(self.player_pos[0] - 20), int(self.player_pos[1] - 20)))

            # Main ship
            points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TEXT)

            # Cargo indicator
            fill_ratio = min(1.0, self.score / self.WIN_SCORE)
            if fill_ratio > 0:
                cargo_height = int(16 * fill_ratio)
                cargo_y = self.player_pos[1] + 8 - cargo_height
                cargo_rect = pygame.Rect(self.player_pos[0] - 4, cargo_y, 8, cargo_height)
                pygame.draw.rect(self.screen, self.COLOR_ORE, cargo_rect)


    def _render_ui(self):
        # Ore counter
        score_text = self.font_ui.render(f"ORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Steps timer
        time_left = self.MAX_STEPS - self.steps
        time_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Game over message
        if self.game_over:
            if self.win:
                msg = "MISSION COMPLETE"
                color = self.COLOR_ORE
            else:
                msg = "SHIP DESTROYED"
                color = self.COLOR_EXPLOSION[0]
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

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


if __name__ == "__main__":
    # To play the game manually, you can run this file.
    # This is for demonstration and debugging purposes.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        # Action defaults
        movement = 0 # none
        space_held = 0 # released
        shift_held = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
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
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()