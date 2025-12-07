import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:51:50.001085
# Source Brief: brief_00713.md
# Brief Index: 713
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a spaceship to collect asteroids.
    The core mechanic involves synchronizing the ship's rotation with a periodic
    magnetic pulse to double the points gained from collected asteroids.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a spaceship to collect asteroids. Synchronize your ship's rotation with a periodic "
        "magnetic pulse to double the points from collected asteroids."
    )
    user_guide = (
        "Controls: ↑↓←→ to move, space to rotate clockwise, and shift to rotate counter-clockwise."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    SCORE_TO_WIN = 200
    MAX_STEPS = 3000 # Increased from brief for better playability

    # Colors
    COLOR_BG = (5, 0, 15)
    COLOR_SHIP = (255, 255, 255)
    COLOR_SHIP_GLOW = (180, 180, 255)
    COLOR_PULSE_LINE = (0, 255, 150)
    COLOR_PULSE_ACTIVE = (150, 255, 200)
    COLOR_UI_TEXT = (220, 220, 255)
    ASTEROID_COLORS = [
        (100, 100, 110), (120, 120, 130), (140, 140, 150),
        (160, 160, 170), (180, 180, 190)
    ]

    # Player
    PLAYER_SPEED = 6.0
    PLAYER_ROTATION_SPEED = 5.0 # degrees per step
    PLAYER_RADIUS = 12

    # Asteroids
    MIN_ASTEROIDS = 8
    INITIAL_ASTEROID_SPEED = 1.0
    DIFFICULTY_INTERVAL = 50 # steps
    DIFFICULTY_SCALING = 0.01

    # Pulse
    PULSE_INTERVAL = 10 * FPS # 10 seconds
    PULSE_DURATION = 1 * FPS   # 1 second
    PULSE_TOLERANCE = 15 # degrees

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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # State variables are initialized in reset()
        self.world_offset = None
        self.player_angle = None
        self.asteroids = None
        self.particles = None
        self.stars = None
        self.score = None
        self.steps = None
        self.pulse_timer = None
        self.pulse_active_timer = None
        self.pulse_angle = None
        self.pulse_sync_achieved = None
        self.current_asteroid_speed = None

        # self.reset() # reset is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.world_offset = pygame.math.Vector2(0, 0)
        self.player_angle = 0.0
        self.asteroids = []
        self.particles = []
        self.score = 0
        self.steps = 0
        
        self.pulse_timer = self.PULSE_INTERVAL
        self.pulse_active_timer = 0
        self.pulse_angle = self.np_random.uniform(0, 360)
        self.pulse_sync_achieved = False
        
        self.current_asteroid_speed = self.INITIAL_ASTEROID_SPEED

        self._generate_starfield()
        while len(self.asteroids) < self.MIN_ASTEROIDS:
            self._spawn_asteroid(on_screen=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        self.steps += 1

        self._handle_input(movement, space_held, shift_held)
        self._update_pulse()
        self._update_asteroids()
        self._update_particles()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        if len(self.asteroids) < self.MIN_ASTEROIDS:
            self._spawn_asteroid()

        self._apply_difficulty_scaling()

        terminated = self.score >= self.SCORE_TO_WIN or self.steps >= self.MAX_STEPS
        truncated = False
        if terminated and self.score >= self.SCORE_TO_WIN:
            reward += 100.0  # Goal-oriented reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Movement (moves the world, not the player)
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = self.PLAYER_SPEED  # Up
        elif movement == 2: move_vec.y = -self.PLAYER_SPEED # Down
        elif movement == 3: move_vec.x = self.PLAYER_SPEED  # Left
        elif movement == 4: move_vec.x = -self.PLAYER_SPEED # Right
        self.world_offset += move_vec

        # Rotation
        if space_held: self.player_angle += self.PLAYER_ROTATION_SPEED
        if shift_held: self.player_angle -= self.PLAYER_ROTATION_SPEED
        self.player_angle %= 360

    def _update_pulse(self):
        self.pulse_sync_achieved = False
        if self.pulse_active_timer > 0:
            self.pulse_active_timer -= 1
        else:
            self.pulse_timer -= 1
            if self.pulse_timer <= 0:
                self.pulse_timer = self.PULSE_INTERVAL
                self.pulse_active_timer = self.PULSE_DURATION
                self.pulse_angle = self.np_random.uniform(0, 360)
                # sfx: pulse_charge

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            # World wrapping logic is handled in rendering

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _apply_difficulty_scaling(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.current_asteroid_speed += self.DIFFICULTY_SCALING

    def _handle_collisions(self):
        reward = 0.0
        player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        # Check for pulse synchronization success
        if self.pulse_active_timer > 0:
            angle_diff = abs((self.player_angle - self.pulse_angle + 180) % 360 - 180)
            if angle_diff <= self.PULSE_TOLERANCE:
                self.pulse_sync_achieved = True
                reward += 1.0 # Reward for successful sync
                # sfx: pulse_sync_success

        asteroids_to_remove = []
        for asteroid in self.asteroids:
            asteroid_screen_pos = asteroid['pos'] - self.world_offset
            if player_pos.distance_to(asteroid_screen_pos) < self.PLAYER_RADIUS + asteroid['radius']:
                points = asteroid['points']
                if self.pulse_sync_achieved:
                    points *= 2 # Double points on sync
                    # sfx: asteroid_collect_boosted
                else:
                    # sfx: asteroid_collect
                    pass

                self.score = min(self.SCORE_TO_WIN, self.score + points)
                reward += 0.1 # Continuous reward for collection
                self._create_explosion(asteroid_screen_pos, asteroid['color'])
                asteroids_to_remove.append(asteroid)
        
        self.asteroids = [a for a in self.asteroids if a not in asteroids_to_remove]
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_pulse()
        self._render_asteroids()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.stars:
            pos_x = int((star['pos'].x - self.world_offset.x * star['parallax']) % self.SCREEN_WIDTH)
            pos_y = int((star['pos'].y - self.world_offset.y * star['parallax']) % self.SCREEN_HEIGHT)
            pygame.draw.circle(self.screen, star['color'], (pos_x, pos_y), star['size'])

    def _render_pulse(self):
        center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        if self.pulse_active_timer > 0:
            # Draw expanding circle for active pulse
            radius = (self.PULSE_DURATION - self.pulse_active_timer) * 4
            alpha = max(0, 255 - int(radius / 2))
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, center[0], center[1], int(radius), (*self.COLOR_PULSE_ACTIVE, alpha))
                pygame.gfxdraw.aacircle(self.screen, center[0], center[1], int(radius-2), (*self.COLOR_PULSE_ACTIVE, alpha))

        # Draw rotating indicator line
        rad_angle = math.radians(self.pulse_angle)
        end_pos = (center[0] + 50 * math.cos(rad_angle), center[1] - 50 * math.sin(rad_angle))
        pygame.draw.aaline(self.screen, self.COLOR_PULSE_LINE, center, end_pos)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            screen_pos = asteroid['pos'] - self.world_offset
            # Wrap rendering for seamless scrolling
            for dx in [-self.SCREEN_WIDTH, 0, self.SCREEN_WIDTH]:
                for dy in [-self.SCREEN_HEIGHT, 0, self.SCREEN_HEIGHT]:
                    pos = (int(screen_pos.x + dx), int(screen_pos.y + dy))
                    if -50 < pos[0] < self.SCREEN_WIDTH + 50 and -50 < pos[1] < self.SCREEN_HEIGHT + 50:
                        points = [(p[0] + pos[0], p[1] + pos[1]) for p in asteroid['shape']]
                        pygame.gfxdraw.aapolygon(self.screen, points, asteroid['color'])
                        pygame.gfxdraw.filled_polygon(self.screen, points, asteroid['color'])

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_player(self):
        center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        rad_angle = math.radians(self.player_angle)
        
        # Ship shape points relative to (0,0)
        p1 = (self.PLAYER_RADIUS, 0)
        p2 = (-self.PLAYER_RADIUS / 2, -self.PLAYER_RADIUS * 0.8)
        p3 = (-self.PLAYER_RADIUS / 2, self.PLAYER_RADIUS * 0.8)

        # Rotate points
        def rotate(p, angle):
            x, y = p
            return (x * math.cos(angle) - y * math.sin(angle),
                    x * math.sin(angle) + y * math.cos(angle))

        rp1 = rotate(p1, -rad_angle)
        rp2 = rotate(p2, -rad_angle)
        rp3 = rotate(p3, -rad_angle)
        
        # Translate points to screen center
        ship_points = [
            (center_x + rp1[0], center_y + rp1[1]),
            (center_x + rp2[0], center_y + rp2[1]),
            (center_x + rp3[0], center_y + rp3[1])
        ]

        # Draw glow
        glow_surf = pygame.Surface((self.PLAYER_RADIUS*4, self.PLAYER_RADIUS*4), pygame.SRCALPHA)
        glow_center = (self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
        pygame.draw.circle(glow_surf, (*self.COLOR_SHIP_GLOW, 50), glow_center, self.PLAYER_RADIUS*1.8)
        pygame.draw.circle(glow_surf, (*self.COLOR_SHIP_GLOW, 30), glow_center, self.PLAYER_RADIUS*2.2)
        self.screen.blit(glow_surf, (center_x - glow_center[0], center_y - glow_center[1]), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw ship
        pygame.gfxdraw.aapolygon(self.screen, ship_points, self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(self.screen, ship_points, self.COLOR_SHIP)

    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        text_surface = self.font.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        # Pulse sync indicator
        if self.pulse_sync_achieved:
            sync_text = "SYNC!"
            sync_surface = self.font.render(sync_text, True, self.COLOR_PULSE_ACTIVE)
            text_rect = sync_surface.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30))
            self.screen.blit(sync_surface, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _spawn_asteroid(self, on_screen=False):
        size_class = self.np_random.integers(0, 5)
        radius = 8 + size_class * 4
        points = 1 + size_class

        if on_screen:
            pos = pygame.math.Vector2(
                self.np_random.uniform(0, self.SCREEN_WIDTH),
                self.np_random.uniform(0, self.SCREEN_HEIGHT)
            )
        else:
            edge = self.np_random.integers(0, 4)
            if edge == 0: # Top
                pos = pygame.math.Vector2(self.np_random.uniform(-radius, self.SCREEN_WIDTH + radius), -radius)
            elif edge == 1: # Right
                pos = pygame.math.Vector2(self.SCREEN_WIDTH + radius, self.np_random.uniform(-radius, self.SCREEN_HEIGHT + radius))
            elif edge == 2: # Bottom
                pos = pygame.math.Vector2(self.np_random.uniform(-radius, self.SCREEN_WIDTH + radius), self.SCREEN_HEIGHT + radius)
            else: # Left
                pos = pygame.math.Vector2(-radius, self.np_random.uniform(-radius, self.SCREEN_HEIGHT + radius))
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.current_asteroid_speed + self.np_random.uniform(-0.5, 0.5)
        vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed

        # Generate a pseudo-random polygon shape
        shape_points = []
        num_vertices = 8
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            r = radius + self.np_random.uniform(-radius*0.3, radius*0.3)
            shape_points.append((r * math.cos(angle), r * math.sin(angle)))

        self.asteroids.append({
            'pos': pos + self.world_offset,
            'vel': vel,
            'radius': radius,
            'points': points,
            'color': self.ASTEROID_COLORS[size_class],
            'shape': shape_points
        })

    def _generate_starfield(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                'size': self.np_random.choice([0, 1, 1, 2]),
                'color': random.choice([(50,50,70), (70,70,90), (100,100,120)]),
                'parallax': self.np_random.uniform(0.1, 0.5)
            })

    def _create_explosion(self, position, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': position.copy(),
                'vel': pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'lifespan': self.np_random.integers(15, 30),
                'max_lifespan': 30,
                'color': color,
                'size': self.np_random.integers(1, 4)
            })

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    # To see the game, comment out the os.environ line at the top.
    
    # Re-enable video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Pulse")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")

    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term or trunc

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()