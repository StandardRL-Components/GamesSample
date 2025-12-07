import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    """
    Gymnasium environment for a top-down arcade space mining game.

    The player pilots a ship, mines asteroids for minerals, and avoids hostile
    patrols. The goal is to collect 100 mineral units.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold Space to mine nearby asteroids. Avoid red enemy ships."
    )
    game_description = (
        "Pilot a mining ship, collect valuable minerals from asteroids while dodging hostile alien patrols."
    )

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (100, 180, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 100, 100, 70)
    COLOR_MINERAL = (255, 220, 50)
    COLOR_LASER = (100, 255, 100)
    COLOR_TEXT = (255, 255, 255)
    ASTEROID_COLORS = [(100, 100, 100), (140, 120, 100), (110, 90, 130)]

    # Screen dimensions
    WIDTH, HEIGHT = 640, 400

    # Game parameters
    MAX_STEPS = 5000
    WIN_SCORE = 100
    PLAYER_ACCELERATION = 0.2
    PLAYER_TURN_SPEED = 0.1
    PLAYER_DRAG = 0.98
    PLAYER_MAX_SPEED = 4.0
    NUM_ASTEROIDS = 10
    NUM_ENEMIES = 3
    ENEMY_INITIAL_SPEED = 1.0
    ENEMY_SPEED_INCREASE_INTERVAL = 200
    ENEMY_SPEED_INCREASE_AMOUNT = 0.05
    ENEMY_MAX_SPEED = 5.0
    MINING_RANGE = 80
    MINING_RATE = 0.1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 16)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.player_angle = 0.0
        self.asteroids = []
        self.enemies = []
        self.particles = []
        self.engine_trail = []
        self.mining_target = None
        self.np_random = None
        self.stars = []

        # Initialize state for the first time
        # self.reset() is called by the validation function
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mining_target = None

        # Player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float64)
        self.player_angle = -math.pi / 2  # Start facing up

        # Generate game elements
        self._generate_stars()
        self._generate_asteroids()
        self._generate_enemies()

        self.particles = []
        self.engine_trail = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False

        # --- Update game logic ---
        self._handle_input(action)
        self._update_player()
        self._update_enemies()
        is_mining = self._handle_mining(action)
        self._update_particles()

        # --- Reward calculation ---
        if not is_mining:
            reward -= 0.01  # Penalty for being idle

        # Collect rewards from particles
        collected_minerals, destroyed_asteroids = self._process_collections()
        reward += collected_minerals * 0.1
        reward += destroyed_asteroids * 1.0
        self.score += collected_minerals
        self.score = min(self.score, self.WIN_SCORE)

        # --- Check for collisions and termination ---
        if self._check_player_enemy_collision():
            reward = -100.0
            terminated = True
            self.game_over = True
            self._create_explosion(self.player_pos, self.COLOR_PLAYER)
            # Sound placeholder: pygame.mixer.Sound('explosion.wav').play()

        if self.score >= self.WIN_SCORE:
            reward = 100.0
            terminated = True
            self.game_over = True
            # Sound placeholder: pygame.mixer.Sound('win.wav').play()

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_stars()
        self._render_asteroids()
        self._render_mining_laser()
        self._render_enemies()
        self._render_player()
        self._render_particles()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Helper methods for game logic ---

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Turning
        if movement == 3:  # Left
            self.player_angle -= self.PLAYER_TURN_SPEED
        if movement == 4:  # Right
            self.player_angle += self.PLAYER_TURN_SPEED

        # Acceleration
        thrust = np.array([0.0, 0.0])
        if movement == 1:  # Up
            thrust = np.array([math.cos(self.player_angle), math.sin(self.player_angle)]) * self.PLAYER_ACCELERATION
            # Add engine trail particles
            if self.steps % 2 == 0:
                self.engine_trail.append({'pos': self.player_pos.copy(), 'life': 20})
        if movement == 2:  # Down (Brake)
            thrust = -self.player_vel * 0.05
        
        self.player_vel += thrust

    def _update_player(self):
        # Apply drag
        self.player_vel *= self.PLAYER_DRAG
        
        # Cap speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = (self.player_vel / speed) * self.PLAYER_MAX_SPEED

        # Update position
        self.player_pos += self.player_vel

        # Screen wrap
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

        # Update engine trail
        for trail in self.engine_trail:
            trail['life'] -= 1
        self.engine_trail = [t for t in self.engine_trail if t['life'] > 0]

    def _update_enemies(self):
        # Increase speed periodically
        if self.steps > 0 and self.steps % self.ENEMY_SPEED_INCREASE_INTERVAL == 0:
            for enemy in self.enemies:
                enemy['speed'] = min(self.ENEMY_MAX_SPEED, enemy['speed'] + self.ENEMY_SPEED_INCREASE_AMOUNT)

        for enemy in self.enemies:
            enemy['angle'] += enemy['speed'] * 0.02
            offset = np.array([math.cos(enemy['angle']), math.sin(enemy['angle'])]) * enemy['radius']
            enemy['pos'] = enemy['center'] + offset

    def _handle_mining(self, action):
        space_held = action[1] == 1
        is_mining = False

        if space_held:
            # Find closest asteroid in range
            closest_asteroid = None
            min_dist = float('inf')
            for asteroid in self.asteroids:
                dist = np.linalg.norm(self.player_pos - asteroid['pos'])
                if dist < self.MINING_RANGE and dist < min_dist:
                    min_dist = dist
                    closest_asteroid = asteroid
            
            self.mining_target = closest_asteroid

            if self.mining_target:
                is_mining = True
                # Sound placeholder: pygame.mixer.Sound('laser.wav').play(loops=-1)
                minerals_before = math.floor(self.mining_target['minerals'])
                self.mining_target['minerals'] -= self.MINING_RATE
                minerals_after = math.floor(self.mining_target['minerals'])

                # Spawn particle if a whole unit is mined
                if minerals_before > minerals_after:
                    self._spawn_mineral_particle(self.mining_target['pos'])
        else:
            self.mining_target = None
            # Sound placeholder: pygame.mixer.Sound('laser.wav').stop()

        return is_mining

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _process_collections(self):
        collected_minerals = 0
        destroyed_asteroids = 0

        # Collect mineral particles
        player_radius = 10
        for p in self.particles[:]:
            if p['type'] == 'mineral':
                if np.linalg.norm(p['pos'] - self.player_pos) < player_radius:
                    collected_minerals += 1
                    self.particles.remove(p)
                    # Sound placeholder: pygame.mixer.Sound('collect.wav').play()

        # Remove depleted asteroids
        for a in self.asteroids[:]:
            if a['minerals'] <= 0:
                self.asteroids.remove(a)
                destroyed_asteroids += 1
                self._create_explosion(a['pos'], random.choice(self.ASTEROID_COLORS))
                # Sound placeholder: pygame.mixer.Sound('asteroid_break.wav').play()

        return collected_minerals, destroyed_asteroids

    def _check_player_enemy_collision(self):
        player_radius = 8  # Collision radius for player
        enemy_size = 12
        for enemy in self.enemies:
            dist = np.linalg.norm(self.player_pos - enemy['pos'])
            if dist < player_radius + enemy_size / 2:
                return True
        return False

    # --- Helper methods for generation ---

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            pos = (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT))
            brightness = self.np_random.integers(50, 150)
            self.stars.append({'pos': pos, 'color': (brightness, brightness, brightness)})

    def _generate_asteroids(self):
        self.asteroids = []
        for _ in range(self.NUM_ASTEROIDS):
            radius = self.np_random.integers(15, 30)
            pos = self.np_random.uniform(low=radius, high=[self.WIDTH-radius, self.HEIGHT-radius], size=2)
            
            # Ensure asteroids don't spawn too close to the center
            while np.linalg.norm(pos - np.array([self.WIDTH/2, self.HEIGHT/2])) < 100:
                 pos = self.np_random.uniform(low=radius, high=[self.WIDTH-radius, self.HEIGHT-radius], size=2)

            minerals = self.np_random.integers(3, 8) * (radius / 15)
            color_idx = self.np_random.integers(0, len(self.ASTEROID_COLORS))
            self.asteroids.append({
                'pos': pos, 'radius': radius, 'minerals': minerals,
                'max_minerals': minerals, 'color': self.ASTEROID_COLORS[color_idx]
            })

    def _generate_enemies(self):
        self.enemies = []
        for i in range(self.NUM_ENEMIES):
            patrol_radius = self.np_random.uniform(100, 200)
            patrol_center = self.np_random.uniform(low=patrol_radius, high=[self.WIDTH-patrol_radius, self.HEIGHT-patrol_radius], size=2)
            angle = self.np_random.uniform(0, 2 * math.pi)
            pos = patrol_center + np.array([math.cos(angle), math.sin(angle)]) * patrol_radius
            
            self.enemies.append({
                'pos': pos, 'center': patrol_center, 'radius': patrol_radius,
                'angle': angle, 'speed': self.ENEMY_INITIAL_SPEED
            })

    def _spawn_mineral_particle(self, origin_pos):
        angle_to_player = math.atan2(self.player_pos[1] - origin_pos[1], self.player_pos[0] - origin_pos[0])
        angle = self.np_random.normal(loc=angle_to_player, scale=0.5)
        speed = self.np_random.uniform(2, 4)
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        self.particles.append({
            'pos': origin_pos.copy(), 'vel': vel, 'life': 100,
            'type': 'mineral', 'color': self.COLOR_MINERAL
        })

    def _create_explosion(self, pos, base_color):
        num_particles = 30
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(20, 40)
            r, g, b = base_color
            color_variation = self.np_random.integers(-30, 30)
            final_color = (max(0, min(255, r + color_variation)), 
                           max(0, min(255, g + color_variation)), 
                           max(0, min(255, b + color_variation)))
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': life,
                'type': 'fx', 'color': final_color
            })

    # --- Helper methods for rendering ---

    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], star['pos'], 1)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pos_int = asteroid['pos'].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], asteroid['radius'], asteroid['color'])
            
            # FIX: Convert generator expression to a tuple for the color argument
            highlight_color = tuple(min(c + 20, 255) for c in asteroid['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], asteroid['radius'], highlight_color)
            
            # Health bar
            health_ratio = asteroid['minerals'] / asteroid['max_minerals']
            if health_ratio < 1.0:
                bar_width = asteroid['radius'] * 1.5
                bar_height = 5
                bar_pos_x = pos_int[0] - bar_width / 2
                bar_pos_y = pos_int[1] - asteroid['radius'] - 10
                pygame.draw.rect(self.screen, (50, 50, 50), (bar_pos_x, bar_pos_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, self.COLOR_MINERAL, (bar_pos_x, bar_pos_y, bar_width * health_ratio, bar_height))

    def _render_enemies(self):
        size = 12
        for enemy in self.enemies:
            pos_int = enemy['pos'].astype(int)
            rect = pygame.Rect(pos_int[0] - size/2, pos_int[1] - size/2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect)
            # Simple glow effect
            glow_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_ENEMY_GLOW, (size, size), size)
            self.screen.blit(glow_surf, (pos_int[0] - size, pos_int[1] - size))

    def _render_player(self):
        if self.game_over: return

        # Draw engine trail
        for trail in self.engine_trail:
            alpha = max(0, min(255, int(trail['life'] * 12)))
            color = (255, 200, 100, alpha)
            glow_surf = pygame.Surface((16, 16), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, color, (8, 8), int(trail['life'] * 0.3))
            self.screen.blit(glow_surf, (int(trail['pos'][0] - 8), int(trail['pos'][1] - 8)))

        # Player ship as a triangle
        p1 = self.player_pos + np.array([math.cos(self.player_angle), math.sin(self.player_angle)]) * 15
        p2 = self.player_pos + np.array([math.cos(self.player_angle + 2.5), math.sin(self.player_angle + 2.5)]) * 8
        p3 = self.player_pos + np.array([math.cos(self.player_angle - 2.5), math.sin(self.player_angle - 2.5)]) * 8
        points = [p1.astype(int), p2.astype(int), p3.astype(int)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Glow effect
        pos_int = self.player_pos.astype(int)
        glow_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (30, 30), 25)
        self.screen.blit(glow_surf, (pos_int[0]-30, pos_int[1]-30))
        
    def _render_mining_laser(self):
        if self.mining_target:
            start_pos = self.player_pos.astype(int)
            end_pos = self.mining_target['pos'].astype(int)
            pygame.draw.aaline(self.screen, self.COLOR_LASER, start_pos, end_pos, 2)
            # Add some pulsating effect
            pulse = abs(math.sin(self.steps * 0.5)) * 3
            pygame.draw.circle(self.screen, self.COLOR_LASER, end_pos, int(5 + pulse), 1)

    def _render_particles(self):
        for p in self.particles:
            pos_int = p['pos'].astype(int)
            if p['type'] == 'mineral':
                pygame.draw.rect(self.screen, p['color'], (pos_int[0]-1, pos_int[1]-1, 3, 3))
            elif p['type'] == 'fx':
                alpha = max(0, min(255, int(p['life'] * 6)))
                color = (*p['color'], alpha)
                radius = int((40 - p['life']) * 0.2)
                fx_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(fx_surf, color, (radius, radius), radius)
                self.screen.blit(fx_surf, (pos_int[0] - radius, pos_int[1] - radius))


    def _render_ui(self):
        score_text = self.font.render(f"MINERALS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        step_text = self.small_font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(step_text, (self.WIDTH - step_text.get_width() - 10, 10))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Reset to initialize np_random and other state variables
        obs, info = self.reset()

        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    
    # Unset the dummy video driver if you want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(GameEnv.user_guide)

    while running:
        # Action defaults
        movement = 0  # no-op
        space_held = 0
        shift_held = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            total_reward = 0
            # Wait a bit before resetting
            pygame.time.wait(2000)
            env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()