
# Generated: 2025-08-28T02:21:27.391216
# Source Brief: brief_04420.md
# Brief Index: 4420

        
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
        "Controls: Arrow keys to move. Hold space to fire. Survive the asteroid field."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down arcade shooter. Survive a 60-second asteroid onslaught for a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_ASTEROID = (150, 150, 150)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_EXPLOSION = (255, 150, 0)
        self.COLOR_UI = (255, 255, 255)
        self.COLOR_STAR = (100, 100, 120)
        
        # Player settings
        self.PLAYER_SPEED = 3.0
        self.PLAYER_SIZE = 12
        self.PLAYER_FRICTION = 0.95 # Added for smoother movement feel
        
        # Projectile settings
        self.PROJECTILE_SPEED = 8.0
        self.PROJECTILE_COOLDOWN = 10 # frames
        self.PROJECTILE_SIZE = 2

        # Asteroid settings
        self.ASTEROID_MIN_SPEED = 0.5
        self.ASTEROID_MAX_SPEED = 2.0
        self.ASTEROID_MIN_SIZE = 15
        self.ASTEROID_MAX_SIZE = 40
        self.ASTEROID_MIN_VERTICES = 7
        self.ASTEROID_MAX_VERTICES = 12
        self.INITIAL_SPAWN_PERIOD = 50 # frames
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables to be set in reset()
        self.player_pos = None
        self.player_vel = None
        self.last_move_direction = None
        self.asteroids = []
        self.projectiles = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fire_cooldown = 0
        self.spawn_timer = 0
        self.np_random = None
        
        # Initialize state
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.last_move_direction = np.array([0.0, -1.0], dtype=float) # Default to up
        
        self.asteroids = []
        self.projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fire_cooldown = 0
        self.spawn_timer = 0

        # Generate a static starfield
        if not self.stars: # Only generate stars once
            for _ in range(150):
                self.stars.append(
                    (
                        self.np_random.integers(0, self.WIDTH),
                        self.np_random.integers(0, self.HEIGHT),
                        self.np_random.choice([1, 2]),
                    )
                )

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Survival reward per frame
        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1 # Unused in this game
        
        # Player movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1:  # Up
            move_vec[1] -= 1.0
        elif movement == 2:  # Down
            move_vec[1] += 1.0
        elif movement == 3:  # Left
            move_vec[0] -= 1.0
        elif movement == 4:  # Right
            move_vec[0] += 1.0
        
        if np.linalg.norm(move_vec) > 0:
            norm_move_vec = move_vec / np.linalg.norm(move_vec)
            self.player_vel += norm_move_vec * 0.4 # Acceleration
            self.last_move_direction = norm_move_vec
        
        # Limit max speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_SPEED:
            self.player_vel = (self.player_vel / speed) * self.PLAYER_SPEED

        # Firing
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
            
        if space_held and self.fire_cooldown <= 0:
            self._spawn_projectile()
            self.fire_cooldown = self.PROJECTILE_COOLDOWN
            # sfx: player_shoot.wav

        # Update game logic
        self._update_player()
        reward += self._update_projectiles()
        self._update_asteroids()
        self._update_spawner()
        self._update_particles()
        
        collision_reward, player_hit = self._handle_collisions()
        reward += collision_reward
        
        if player_hit:
            self.game_over = True
            reward = -100.0 # Terminal loss penalty
            # sfx: player_explosion.wav

        # Check Termination Conditions
        terminated = self.game_over
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.game_over: # Survived the whole time
                reward = 100.0 # Terminal win bonus
                # sfx: win_jingle.wav
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _update_player(self):
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_FRICTION # Apply friction

        # Toroidal world wrap
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

    def _update_projectiles(self):
        miss_penalty = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p['pos'] += p['vel']
            p['life'] -= 1
            
            on_screen = (0 <= p['pos'][0] < self.WIDTH and 0 <= p['pos'][1] < self.HEIGHT)
            
            if p['life'] > 0 and on_screen:
                projectiles_to_keep.append(p)
            elif p['life'] > 0 and not on_screen: # Went off-screen (miss)
                miss_penalty -= 0.2
                # sfx: miss_whoosh.wav
        
        self.projectiles = projectiles_to_keep
        return miss_penalty
        
    def _update_asteroids(self):
        for a in self.asteroids:
            a['pos'] += a['vel']
            # Toroidal world wrap
            a['pos'][0] = (a['pos'][0] + a['radius']) % (self.WIDTH + 2 * a['radius']) - a['radius']
            a['pos'][1] = (a['pos'][1] + a['radius']) % (self.HEIGHT + 2 * a['radius']) - a['radius']

    def _update_spawner(self):
        # Difficulty scaling: spawn period decreases over time
        reduction_factor = self.steps // 600
        current_spawn_period = self.INITIAL_SPAWN_PERIOD - reduction_factor * 5
        current_spawn_period = max(15, current_spawn_period)

        self.spawn_timer += 1
        if self.spawn_timer >= current_spawn_period and len(self.asteroids) < 15:
            self.spawn_timer = 0
            self._spawn_asteroid()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _handle_collisions(self):
        reward = 0
        player_hit = False
        
        asteroids_to_remove = set()
        projectiles_to_remove = set()

        for i, p in enumerate(self.projectiles):
            if i in projectiles_to_remove: continue
            for j, a in enumerate(self.asteroids):
                if j in asteroids_to_remove: continue
                dist = np.linalg.norm(p['pos'] - a['pos'])
                if dist < a['radius'] + self.PROJECTILE_SIZE:
                    asteroids_to_remove.add(j)
                    projectiles_to_remove.add(i)
                    reward += 1.0
                    self._create_explosion(a['pos'], a['radius'])
                    # sfx: asteroid_explosion.wav
                    break
        
        if asteroids_to_remove:
            self.asteroids = [a for i, a in enumerate(self.asteroids) if i not in asteroids_to_remove]
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
            self.score += len(asteroids_to_remove)

        if not self.game_over:
            for a in self.asteroids:
                dist = np.linalg.norm(self.player_pos - a['pos'])
                if dist < a['radius'] + self.PLAYER_SIZE * 0.7:
                    player_hit = True
                    self._create_explosion(self.player_pos, self.PLAYER_SIZE * 2)
                    break
        
        return reward, player_hit

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
            "time_remaining": (self.MAX_STEPS - self.steps) // self.FPS
        }

    def _render_game(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size)
            
        for a in self.asteroids:
            self._draw_toroidal_polygon(a['pos'], a['shape'], self.COLOR_ASTEROID)
            
        for p in self.projectiles:
            start_pos = p['pos']
            end_pos = p['pos'] - p['vel'] * 0.5
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start_pos.astype(int), end_pos.astype(int), 2)

        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((int(p['radius']*2), int(p['radius']*2)), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))
            
        if not self.game_over:
            self._draw_player()

    def _draw_player(self):
        angle = math.atan2(self.last_move_direction[1], self.last_move_direction[0]) + math.pi / 2
        s = self.PLAYER_SIZE
        points = [np.array([0, -s]), np.array([-s/2, s/2]), np.array([s/2, s/2])]
        rot_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        
        for dx in [-self.WIDTH, 0, self.WIDTH]:
            for dy in [-self.HEIGHT, 0, self.HEIGHT]:
                translated_points = [(rot_matrix @ p + self.player_pos + np.array([dx, dy])).astype(int) for p in points]
                pygame.gfxdraw.aapolygon(self.screen, translated_points, self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(self.screen, translated_points, self.COLOR_PLAYER)

    def _draw_toroidal_polygon(self, pos, shape_pts, color):
        for dx in [-self.WIDTH, 0, self.WIDTH]:
            for dy in [-self.HEIGHT, 0, self.HEIGHT]:
                points = [(p[0] + pos[0] + dx, p[1] + pos[1] + dy) for p in shape_pts]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        time_left = max(0, (self.MAX_STEPS - self.steps) // self.FPS)
        time_text = self.font.render(f"TIME: {time_left}", True, self.COLOR_UI)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

    def _spawn_projectile(self):
        pos = self.player_pos.copy() + self.last_move_direction * self.PLAYER_SIZE
        vel = self.last_move_direction * self.PROJECTILE_SPEED
        self.projectiles.append({'pos': pos, 'vel': vel, 'life': self.FPS})

    def _spawn_asteroid(self):
        edge = self.np_random.integers(4)
        if edge == 0: pos = np.array([self.np_random.uniform(0, self.WIDTH), -self.ASTEROID_MAX_SIZE], dtype=float)
        elif edge == 1: pos = np.array([self.WIDTH + self.ASTEROID_MAX_SIZE, self.np_random.uniform(0, self.HEIGHT)], dtype=float)
        elif edge == 2: pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ASTEROID_MAX_SIZE], dtype=float)
        else: pos = np.array([-self.ASTEROID_MAX_SIZE, self.np_random.uniform(0, self.HEIGHT)], dtype=float)

        target = np.array([self.np_random.uniform(self.WIDTH*0.2, self.WIDTH*0.8), self.np_random.uniform(self.HEIGHT*0.2, self.HEIGHT*0.8)])
        direction = (target - pos) / np.linalg.norm(target - pos)
        
        speed = self.np_random.uniform(self.ASTEROID_MIN_SPEED, self.ASTEROID_MAX_SPEED)
        radius = self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        
        self.asteroids.append({
            'pos': pos, 'vel': direction * speed, 'radius': radius, 'shape': self._create_random_asteroid_shape(radius)
        })

    def _create_random_asteroid_shape(self, radius):
        num_vertices = self.np_random.integers(self.ASTEROID_MIN_VERTICES, self.ASTEROID_MAX_VERTICES + 1)
        angles = np.linspace(0, 2 * math.pi, num_vertices, endpoint=False)
        return [(math.cos(a) * radius * self.np_random.uniform(0.7, 1.3), math.sin(a) * radius * self.np_random.uniform(0.7, 1.3)) for a in angles]

    def _create_explosion(self, pos, size):
        num_particles = int(size)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * (1 + size / self.ASTEROID_MAX_SIZE)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': life, 'max_life': life,
                'radius': self.np_random.uniform(2, 5), 'color': self.COLOR_EXPLOSION
            })

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

if __name__ == '__main__':
    import os
    try:
        # Attempt to set display for human play
        pygame.display.init()
        pygame.font.init()
    except pygame.error:
        # Fallback to dummy driver for headless environments
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.font.init()

    env = GameEnv(render_mode="rgb_array")
    
    # Setup window for human play if possible
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption(GameEnv.game_description)
        human_play = True
    except pygame.error:
        print("Pygame display not available. Running headless check.")
        human_play = False

    if human_play:
        obs, info = env.reset()
        terminated = False
        
        print("--- Human Play Test ---")
        print(GameEnv.user_guide)
        
        while not terminated:
            keys = pygame.key.get_pressed()
            
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
                
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            env.clock.tick(env.FPS)

        print(f"Game Over! Final Score: {info['score']}")
        env.close()