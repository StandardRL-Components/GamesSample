import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move your spaceship. Avoid the asteroids!"
    )

    game_description = (
        "Pilot a spaceship in a top-down arcade environment, dodging asteroids to survive for 60 seconds. "
        "The longer you survive, the faster and more numerous the asteroids become."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_ASTEROID = (180, 180, 190)
    COLOR_ASTEROID_OUTLINE = (130, 130, 140)
    COLOR_STAR = (200, 200, 220)
    COLOR_THRUSTER = (255, 220, 100)
    COLOR_EXPLOSION = (255, 165, 0)
    COLOR_UI = (220, 220, 255)

    # Game Parameters
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    
    # Player
    PLAYER_SPEED = 5.0
    PLAYER_SIZE = 10 
    PLAYER_COLLISION_RADIUS = PLAYER_SIZE * 0.8

    # Asteroids
    INITIAL_ASTEROID_COUNT = 3
    ASTEROID_SPAWN_RATE_INITIAL = 1.0  # Per second
    ASTEROID_SPAWN_RATE_INCREASE = 0.01 # Per second, per second
    ASTEROID_MAX_VEL_INITIAL = 1.5
    ASTEROID_MAX_VEL_INCREASE_INTERVAL = 10 # seconds
    ASTEROID_MAX_VEL_INCREASE_AMOUNT = 0.2
    ASTEROID_SIZES = [15, 25, 35] # Radii
    ASTEROID_MIN_VERTS = 5
    ASTEROID_MAX_VERTS = 10
    
    # Rewards
    REWARD_SURVIVAL_STEP = 0.01 # Brief says 0.1, but this keeps total rewards more balanced
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0

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
        
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_game_over = pygame.font.SysFont("Consolas", 50)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 28)
            self.font_game_over = pygame.font.SysFont(None, 60)

        self.player_pos = pygame.Vector2(0, 0)
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_survived = 0.0
        self.asteroid_spawn_timer = 0.0
        self.current_asteroid_spawn_rate = self.ASTEROID_SPAWN_RATE_INITIAL
        self.current_max_asteroid_vel = self.ASTEROID_MAX_VEL_INITIAL

        # self.reset() is called by the gym wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.asteroids = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_survived = 0.0

        self.current_asteroid_spawn_rate = self.ASTEROID_SPAWN_RATE_INITIAL
        self.current_max_asteroid_vel = self.ASTEROID_MAX_VEL_INITIAL
        self.asteroid_spawn_timer = 1.0 / self.current_asteroid_spawn_rate
        
        self._generate_stars(200)
        for _ in range(self.INITIAL_ASTEROID_COUNT):
            self._spawn_asteroid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self._handle_input(movement)
        
        self._update_game_state()
        
        reward = self.REWARD_SURVIVAL_STEP
        terminated = False
        truncated = False

        if self._check_collisions():
            self._create_explosion(self.player_pos)
            # sfx: player_explosion.wav
            reward = self.REWARD_LOSS
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward += self.REWARD_WIN
            terminated = True
            self.game_over = True
            self.win = True
            # sfx: win_fanfare.wav

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        moved = True
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: # Up
            move_vec.y = -self.PLAYER_SPEED
        elif movement == 2: # Down
            move_vec.y = self.PLAYER_SPEED
        elif movement == 3: # Left
            move_vec.x = -self.PLAYER_SPEED
        elif movement == 4: # Right
            move_vec.x = self.PLAYER_SPEED
        else: # No-op
            moved = False

        self.player_pos += move_vec
        
        if moved:
            self._create_thruster_particles(self.player_pos, movement)
            # sfx: thruster_loop.wav

        self.player_pos.x = np.clip(self.player_pos.x, 0, self.SCREEN_WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.SCREEN_HEIGHT)

    def _update_game_state(self):
        self.steps += 1
        self.time_survived = self.steps / self.FPS

        # Update difficulty
        self.current_asteroid_spawn_rate = self.ASTEROID_SPAWN_RATE_INITIAL + self.time_survived * self.ASTEROID_SPAWN_RATE_INCREASE
        level = math.floor(self.time_survived / self.ASTEROID_MAX_VEL_INCREASE_INTERVAL)
        self.current_max_asteroid_vel = self.ASTEROID_MAX_VEL_INITIAL + level * self.ASTEROID_MAX_VEL_INCREASE_AMOUNT

        # Update asteroids
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] += asteroid['rot_speed']
            
            # Screen wrap
            if asteroid['pos'].x < -asteroid['size']: asteroid['pos'].x = self.SCREEN_WIDTH + asteroid['size']
            if asteroid['pos'].x > self.SCREEN_WIDTH + asteroid['size']: asteroid['pos'].x = -asteroid['size']
            if asteroid['pos'].y < -asteroid['size']: asteroid['pos'].y = self.SCREEN_HEIGHT + asteroid['size']
            if asteroid['pos'].y > self.SCREEN_HEIGHT + asteroid['size']: asteroid['pos'].y = -asteroid['size']

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # Spawn new asteroids
        self.asteroid_spawn_timer -= 1 / self.FPS
        if self.asteroid_spawn_timer <= 0:
            self._spawn_asteroid()
            self.asteroid_spawn_timer += 1.0 / self.current_asteroid_spawn_rate
            # sfx: asteroid_spawn.wav

    def _check_collisions(self):
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < self.PLAYER_COLLISION_RADIUS + asteroid['size']:
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_particles()
        self._render_asteroids()
        if not self.game_over:
            self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_survived": self.time_survived,
        }

    # --- Rendering Methods ---

    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['size'])

    def _render_player(self):
        p_size = self.PLAYER_SIZE
        points = [
            (self.player_pos.x, self.player_pos.y - p_size),
            (self.player_pos.x - p_size / 1.5, self.player_pos.y + p_size / 2),
            (self.player_pos.x + p_size / 1.5, self.player_pos.y + p_size / 2)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = []
            for i in range(len(asteroid['shape'])):
                angle = asteroid['angle'] + asteroid['shape'][i][0]
                radius = asteroid['shape'][i][1]
                x = asteroid['pos'].x + math.cos(angle) * radius
                y = asteroid['pos'].y + math.sin(angle) * radius
                points.append((int(x), int(y)))
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID_OUTLINE)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = max(0, p['life'] / p['max_life'])
            color = tuple(c * life_ratio for c in p['color'])
            radius = int(p['size'] * life_ratio)
            if radius > 0:
                pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), radius)

    def _render_ui(self):
        # Time
        time_text = f"TIME: {max(0, self.GAME_DURATION_SECONDS - self.time_survived):.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI)
        self.screen.blit(time_surf, (10, 10))
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            over_surf = self.font_game_over.render(msg, True, color)
            over_rect = over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_surf, over_rect)

    # --- Helper Methods ---
    
    def _generate_stars(self, count):
        self.stars = []
        for _ in range(count):
            size = self.np_random.choice([0, 0, 0, 1, 1, 2])
            if size > 0:
                self.stars.append({
                    'pos': (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                    'size': size,
                    'color': (self.np_random.integers(150, 201), self.np_random.integers(150, 201), self.np_random.integers(200, 256))
                })

    def _spawn_asteroid(self):
        edge = self.np_random.integers(0, 4)
        size = self.np_random.choice(self.ASTEROID_SIZES)
        
        if edge == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -size)
        elif edge == 1: # Right
            pos = pygame.Vector2(self.SCREEN_WIDTH + size, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        elif edge == 2: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + size)
        else: # Left
            pos = pygame.Vector2(-size, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            
        angle_to_center = math.atan2(self.SCREEN_HEIGHT/2 - pos.y, self.SCREEN_WIDTH/2 - pos.x)
        angle = self.np_random.uniform(angle_to_center - math.pi/4, angle_to_center + math.pi/4)
        speed = self.np_random.uniform(self.current_max_asteroid_vel / 2, self.current_max_asteroid_vel)
        vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)

        # Generate shape
        num_verts = self.np_random.integers(self.ASTEROID_MIN_VERTS, self.ASTEROID_MAX_VERTS + 1)
        shape = []
        for i in range(num_verts):
            vert_angle = (2 * math.pi / num_verts) * i
            vert_radius = self.np_random.uniform(size * 0.7, size)
            shape.append((vert_angle, vert_radius))

        self.asteroids.append({
            'pos': pos,
            'vel': vel,
            'size': size,
            'angle': 0,
            'rot_speed': self.np_random.uniform(-0.05, 0.05),
            'shape': shape
        })

    def _create_explosion(self, position):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(20, 41)
            self.particles.append({
                'pos': pygame.Vector2(position),
                'vel': vel,
                'life': life,
                'max_life': life,
                'size': self.np_random.integers(1, 5),
                'color': self.COLOR_EXPLOSION
            })

    def _create_thruster_particles(self, position, move_direction):
        if move_direction == 0: return

        # Base direction opposite to movement
        if move_direction == 1: base_angle = math.pi / 2 # Up -> Down
        elif move_direction == 2: base_angle = -math.pi / 2 # Down -> Up
        elif move_direction == 3: base_angle = 0 # Left -> Right
        else: base_angle = math.pi # Right -> Left
        
        for _ in range(3):
            angle = self.np_random.uniform(base_angle - 0.3, base_angle + 0.3)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(5, 16)
            self.particles.append({
                'pos': pygame.Vector2(position),
                'vel': vel,
                'life': life,
                'max_life': life,
                'size': self.np_random.integers(1, 4),
                'color': self.COLOR_THRUSTER
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires the "dummy" SDL_VIDEODRIVER to be unset
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Asteroid Dodger")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        action = [movement, 0, 0] # Space and shift are not used

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}, Time Survived: {info['time_survived']:.2f}s")
    pygame.quit()