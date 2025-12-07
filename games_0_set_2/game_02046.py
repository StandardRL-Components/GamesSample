import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your ship. Survive the asteroid field for 60 seconds."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship through a treacherous asteroid field. Dodge space rocks and survive as long as you can as the field gets denser."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds of gameplay

        # Colors
        self.COLOR_BG = (10, 10, 25)
        self.COLOR_SHIP = (255, 255, 255)
        self.COLOR_ASTEROID = (180, 180, 180)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_EFFECT = (200, 200, 255)

        # Ship properties
        self.SHIP_SIZE = 15
        self.SHIP_SPEED = 5
        self.SHIP_HITBOX_RADIUS = self.SHIP_SIZE * 0.6

        # Asteroid properties
        self.ASTEROID_MIN_RADIUS = 10
        self.ASTEROID_MAX_RADIUS = 40
        self.ASTEROID_MIN_SPEED = 1
        self.ASTEROID_MAX_SPEED = 3
        self.INITIAL_SPAWN_PROB = 0.05
        self.SPAWN_PROB_INCREASE = 0.001 / self.FPS # per step
        self.MAX_ASTEROIDS = 10

        # Reward properties
        self.REWARD_SURVIVE = 0.1
        self.REWARD_WIN = 100
        self.REWARD_LOSE = -100
        self.REWARD_NEAR_MISS = 5
        self.PENALTY_MOVE_AWAY = -2
        self.NEAR_MISS_DISTANCE = 75

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        if render_mode == "rgb_array":
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.large_font = pygame.font.SysFont("monospace", 48, bold=True)
        
        # State variables (initialized in reset)
        self.ship_pos = None
        self.asteroids = []
        self.near_miss_effects = []
        self.starfield = []
        self.steps = 0
        self.score = 0
        self.current_spawn_prob = self.INITIAL_SPAWN_PROB
        self.game_over = False
        self.win = False

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.ship_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
        self.asteroids = []
        self.near_miss_effects = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.current_spawn_prob = self.INITIAL_SPAWN_PROB
        
        self._generate_starfield()

        # Ensure a random agent can survive for a bit
        for _ in range(5):
             self._spawn_asteroid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # Unpack factorized action
        movement = action[0]
        
        # Update game logic
        self.steps += 1
        
        # 1. Handle player movement
        move_vec = self._handle_movement(movement)
        
        # 2. Update asteroids
        self._update_asteroids()

        # 3. Calculate rewards
        reward += self.REWARD_SURVIVE
        
        nearest_asteroid, min_dist = self._get_nearest_asteroid()
        
        if nearest_asteroid:
            # Penalty for moving away from nearest asteroid
            if np.linalg.norm(move_vec) > 0: # only if moving
                vec_to_asteroid = nearest_asteroid['pos'] - self.ship_pos
                # Normalize vectors for dot product
                norm_move_vec = move_vec / np.linalg.norm(move_vec)
                norm_vec_to_asteroid = vec_to_asteroid / np.linalg.norm(vec_to_asteroid)
                
                if np.dot(norm_move_vec, norm_vec_to_asteroid) < -0.5: # Moving mostly away
                    reward += self.PENALTY_MOVE_AWAY
            
            # Reward for near miss
            if min_dist < self.NEAR_MISS_DISTANCE and min_dist > nearest_asteroid['radius'] + self.SHIP_HITBOX_RADIUS:
                reward += self.REWARD_NEAR_MISS
                self._create_near_miss_effect()
        
        # 4. Check for termination conditions
        terminated = False
        if self._check_collision(nearest_asteroid, min_dist):
            self.game_over = True
            terminated = True
            reward = self.REWARD_LOSE
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = True
            terminated = True
            reward = self.REWARD_WIN
        
        self.score += reward
        
        # Update clock for auto-advance
        self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        move_vec = np.array([0, 0], dtype=np.float64)
        if movement == 1:  # Up
            move_vec[1] = -1
        elif movement == 2:  # Down
            move_vec[1] = 1
        elif movement == 3:  # Left
            move_vec[0] = -1
        elif movement == 4:  # Right
            move_vec[0] = 1

        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec) * self.SHIP_SPEED

        self.ship_pos += move_vec
        
        # Clamp ship position to screen bounds
        self.ship_pos[0] = np.clip(self.ship_pos[0], 0, self.WIDTH)
        self.ship_pos[1] = np.clip(self.ship_pos[1], 0, self.HEIGHT)
        
        return move_vec

    def _update_asteroids(self):
        # Move existing asteroids and remove off-screen ones
        self.asteroids = [a for a in self.asteroids if self._move_asteroid(a)]

        # Spawn new asteroids
        if len(self.asteroids) < self.MAX_ASTEROIDS and self.np_random.random() < self.current_spawn_prob:
            self._spawn_asteroid()
        
        # Increase spawn probability over time
        self.current_spawn_prob += self.SPAWN_PROB_INCREASE

    def _move_asteroid(self, asteroid):
        asteroid['pos'] += asteroid['vel']
        # Check if asteroid is off-screen (with a margin)
        margin = asteroid['radius'] * 2
        return (
            -margin < asteroid['pos'][0] < self.WIDTH + margin and
            -margin < asteroid['pos'][1] < self.HEIGHT + margin
        )

    def _spawn_asteroid(self):
        edge = self.np_random.integers(4)
        radius = self.np_random.integers(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS + 1)
        
        if edge == 0: # Top
            pos = np.array([self.np_random.random() * self.WIDTH, -radius], dtype=np.float64)
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.random() * self.WIDTH, self.HEIGHT + radius], dtype=np.float64)
        elif edge == 2: # Left
            pos = np.array([-radius, self.np_random.random() * self.HEIGHT], dtype=np.float64)
        else: # Right
            pos = np.array([self.WIDTH + radius, self.np_random.random() * self.HEIGHT], dtype=np.float64)
            
        # Target a point within the central 50% of the screen for more interesting paths
        target = np.array([
            self.np_random.uniform(self.WIDTH * 0.25, self.WIDTH * 0.75),
            self.np_random.uniform(self.HEIGHT * 0.25, self.HEIGHT * 0.75)
        ])
        
        direction = target - pos
        direction = direction / np.linalg.norm(direction)
        
        speed = self.np_random.uniform(self.ASTEROID_MIN_SPEED, self.ASTEROID_MAX_SPEED)
        vel = direction * speed
        
        self.asteroids.append({'pos': pos, 'vel': vel, 'radius': radius})

    def _get_nearest_asteroid(self):
        if not self.asteroids:
            return None, float('inf')
        
        min_dist = float('inf')
        nearest_asteroid = None
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.ship_pos - asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
                nearest_asteroid = asteroid
        return nearest_asteroid, min_dist

    def _check_collision(self, nearest_asteroid, min_dist):
        if nearest_asteroid:
            collision_dist = nearest_asteroid['radius'] + self.SHIP_HITBOX_RADIUS
            if min_dist < collision_dist:
                # sfx: explosion
                return True
        return False

    def _create_near_miss_effect(self):
        # sfx: whoosh
        self.near_miss_effects.append({
            'pos': self.ship_pos.copy(),
            'radius': self.SHIP_SIZE,
            'max_radius': self.NEAR_MISS_DISTANCE,
            'alpha': 255
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw starfield
        for star in self.starfield:
            self.screen.set_at(star['pos'], star['color'])

        # Draw near miss effects
        for effect in self.near_miss_effects[:]:
            color = (*self.COLOR_EFFECT, effect['alpha'])
            pygame.gfxdraw.aacircle(self.screen, int(effect['pos'][0]), int(effect['pos'][1]), int(effect['radius']), color)
            effect['radius'] += 2
            effect['alpha'] -= 10
            if effect['alpha'] <= 0:
                self.near_miss_effects.remove(effect)

        # Draw asteroids
        for asteroid in self.asteroids:
            pos = asteroid['pos'].astype(int)
            radius = int(asteroid['radius'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)

        # Draw ship
        if self.ship_pos is not None and (not self.game_over or self.win):
            x, y = self.ship_pos
            # Points of the triangle
            p1 = (x, y - self.SHIP_SIZE)
            p2 = (x - self.SHIP_SIZE / 2, y + self.SHIP_SIZE / 2)
            p3 = (x + self.SHIP_SIZE / 2, y + self.SHIP_SIZE / 2)
            points = [p1, p2, p3]
            
            int_points = [(int(px), int(py)) for px, py in points]
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_SHIP)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_SHIP)

    def _render_ui(self):
        # Display time remaining
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"TIME: {time_left:.1f}"
        text_surface = self.font.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Display score
        score_text = f"SCORE: {self.score:.1f}"
        score_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surface.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_surface, score_rect)
        
        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            message_surface = self.large_font.render(message, True, color)
            message_rect = message_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(message_surface, message_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "asteroids_on_screen": len(self.asteroids),
        }
    
    def _generate_starfield(self):
        self.starfield = []
        for _ in range(200):
            pos = (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT))
            brightness = self.np_random.integers(50, 150)
            color = (brightness, brightness, brightness)
            self.starfield.append({'pos': pos, 'color': color})

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset, which initializes the environment state
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)

        # Test observation space (now that env is reset)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        # print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    # pip install pygame gymnasium
    import os
    # Set a display driver. Use 'dummy' for headless execution.
    # For interactive play, you might need 'x11', 'windows', or 'cocoa'.
    try:
        pygame.display.init()
    except pygame.error:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    try:
        game_screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Asteroid Dodger")
        interactive = True
    except pygame.error:
        print("No display available. Running in non-interactive mode.")
        interactive = False

    running = True
    total_reward = 0
    
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0] # [movement, space, shift]

        if interactive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            
            # Movement
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Other actions (not used in this game)
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1
        else: # Simple non-interactive loop
            action = env.action_space.sample()
            if env.steps > env.MAX_STEPS * 2: # Prevent infinite loops
                running = False


        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if interactive:
            # Render the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            game_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            if interactive:
                pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()