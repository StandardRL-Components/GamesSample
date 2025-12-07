import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ← to hop left, ↑ or → to hop forward. Survive the asteroid field and reach the finish line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a hopping spacecraft through a treacherous asteroid field to reach the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 1000
        self.FINISH_LINE_X = 6000 # World coordinate of the finish line
        self.FPS = 60 # For manual play mode

        # Physics & Gameplay
        self.GRAVITY = 0.5
        self.JUMP_VELOCITY_Y = -11
        self.JUMP_VELOCITY_X = 7
        self.GROUND_Y = self.SCREEN_HEIGHT - 40
        self.PLAYER_RADIUS = 12

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (150, 255, 150)
        self.COLOR_ASTEROID = (255, 80, 80)
        self.COLOR_ASTEROID_OUTLINE = (200, 50, 50)
        self.COLOR_FINISH = (80, 80, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_PARTICLE_JUMP = (200, 200, 255)
        self.COLOR_PARTICLE_CRASH = (255, 150, 50)
        
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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # Initialize state variables (to be properly set in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_world_pos = np.zeros(2)
        self.player_vel = np.zeros(2)
        self.on_ground = True
        self.camera_x = 0.0
        self.asteroids = []
        self.cleared_asteroids = set()
        self.asteroid_spawn_timer = 0
        self.base_asteroid_speed = 0.0
        self.stars = []
        self.particles = []
        self.player_squash = 0.0 # For animation 'juice'

        # Initial reset
        # self.reset() is called by validate_implementation which is more thorough
        
        # Self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # Player state
        self.player_world_pos = np.array([self.SCREEN_WIDTH / 4.0, self.GROUND_Y])
        self.player_vel = np.array([0.0, 0.0])
        self.on_ground = True
        self.player_squash = 0.0
        
        # World state
        self.camera_x = 0.0
        self.asteroids = []
        self.cleared_asteroids = set()
        self.asteroid_spawn_timer = 0
        self.base_asteroid_speed = 2.0
        self.particles = []

        # Generate starfield
        self.stars = []
        for _ in range(200):
            self.stars.append({
                'pos': [self.np_random.uniform(0, self.FINISH_LINE_X + self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)],
                'depth': self.np_random.uniform(0.1, 0.8) # For parallax
            })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # --- 1. Handle Input ---
        self._handle_input(action)

        # --- 2. Update Game State ---
        self._update_player()
        self._update_camera()
        self._update_asteroids()
        self._update_particles()

        # --- 3. Calculate Reward & Termination ---
        reward = 0.1  # Survival reward

        # Reward for clearing asteroids
        for asteroid in self.asteroids:
            if asteroid['id'] not in self.cleared_asteroids and self.player_world_pos[0] > asteroid['pos'][0]:
                reward += 1.0
                self.cleared_asteroids.add(asteroid['id'])
                # sfx: positive chime

        # Check for termination conditions
        terminated = False
        if self._check_collision():
            reward = -100.0
            terminated = True
            self._create_explosion(self.player_world_pos)
            # sfx: explosion
        elif self.player_world_pos[0] >= self.FINISH_LINE_X:
            reward += 100.0
            terminated = True
            # sfx: victory fanfare
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        if self.game_over: return

        movement = action[0]
        
        if self.on_ground:
            jump_dir = 0
            # Action 1 (up) or 4 (right) -> hop forward
            if movement == 1 or movement == 4:
                jump_dir = 1
                # sfx: jump
            # Action 3 (left) -> hop backward
            elif movement == 3:
                jump_dir = -1
                # sfx: jump
            
            if jump_dir != 0:
                self.on_ground = False
                self.player_vel[1] = self.JUMP_VELOCITY_Y
                self.player_vel[0] = self.JUMP_VELOCITY_X * jump_dir
                self.player_squash = -1.0 # Stretch for jump
                self._create_jump_particles(self.player_world_pos)

    def _update_player(self):
        if self.game_over: return

        # Apply physics if in the air
        if not self.on_ground:
            self.player_vel[1] += self.GRAVITY
            self.player_world_pos += self.player_vel

        # Check for landing
        if not self.on_ground and self.player_world_pos[1] >= self.GROUND_Y:
            self.on_ground = True
            self.player_world_pos[1] = self.GROUND_Y
            self.player_vel = np.array([0.0, 0.0])
            self.player_squash = 1.0 # Squash on landing
            # sfx: land
        
        # Animation decay
        self.player_squash *= 0.85

    def _update_camera(self):
        # Camera follows player's x-position, keeping them centered
        self.camera_x = self.player_world_pos[0] - self.SCREEN_WIDTH / 2

    def _update_asteroids(self):
        # Difficulty scaling
        if self.steps > 0 and self.steps % 250 == 0:
            self.base_asteroid_speed = min(5.0, self.base_asteroid_speed + 0.05)

        # Spawn new asteroids
        self.asteroid_spawn_timer -= 1
        if self.asteroid_spawn_timer <= 0:
            self._spawn_asteroid()
            spawn_delay = self.np_random.integers(30, 90)
            self.asteroid_spawn_timer = int(spawn_delay / (self.base_asteroid_speed / 2.0))

        # Move and despawn existing asteroids
        asteroids_to_keep = []
        for asteroid in self.asteroids:
            asteroid['pos'][0] -= asteroid['speed']
            if asteroid['pos'][0] + asteroid['radius'] > self.camera_x:
                asteroids_to_keep.append(asteroid)
        self.asteroids = asteroids_to_keep


    def _spawn_asteroid(self):
        radius = self.np_random.uniform(15, 40)
        self.asteroids.append({
            'id': self.steps + self.np_random.uniform(),
            'pos': np.array([
                self.camera_x + self.SCREEN_WIDTH + radius,
                self.np_random.uniform(50, self.GROUND_Y - radius * 2)
            ]),
            'radius': radius,
            'speed': self.base_asteroid_speed + self.np_random.uniform(-0.5, 1.0)
        })

    def _check_collision(self):
        player_center = self.player_world_pos.copy()
        player_center[1] -= self.PLAYER_RADIUS # Adjust for base of player
        for asteroid in self.asteroids:
            dist = np.linalg.norm(player_center - asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius']:
                return True
        return False

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw parallax starfield
        for star in self.stars:
            screen_x = (star['pos'][0] - self.camera_x * star['depth']) % self.SCREEN_WIDTH
            brightness = int(255 * star['depth'])
            color = (brightness, brightness, brightness)
            pygame.draw.circle(self.screen, color, (int(screen_x), int(star['pos'][1])), 1)

        # Draw finish line if visible
        finish_screen_x = self.FINISH_LINE_X - self.camera_x
        if 0 < finish_screen_x < self.SCREEN_WIDTH:
            finish_surf = pygame.Surface((20, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            for i in range(0, self.SCREEN_HEIGHT, 20):
                pygame.draw.rect(finish_surf, self.COLOR_FINISH + (150,), (0, i, 10, 10))
                pygame.draw.rect(finish_surf, self.COLOR_FINISH + (150,), (10, i + 10, 10, 10))
            self.screen.blit(finish_surf, (int(finish_screen_x), 0))

        # Draw asteroids
        for asteroid in self.asteroids:
            screen_pos = (int(asteroid['pos'][0] - self.camera_x), int(asteroid['pos'][1]))
            radius = int(asteroid['radius'])
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, self.COLOR_ASTEROID)
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius, self.COLOR_ASTEROID_OUTLINE)

        # Draw particles
        for p in self.particles:
            screen_pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            alpha = max(0, min(255, int(255 * p['lifespan'] / p['max_lifespan'])))
            color = p['color'] + (alpha,)
            radius = int(p['radius'])
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (screen_pos[0] - radius, screen_pos[1] - radius))

        # Draw player
        if not (self.game_over and self._check_collision()):
            self._draw_player()

    def _draw_player(self):
        # Player is always in the middle of the screen horizontally
        screen_pos = np.array([self.SCREEN_WIDTH / 2.0, self.player_world_pos[1]])
        
        # Squash and stretch animation
        h = self.PLAYER_RADIUS * 2 * (1 - self.player_squash * 0.3)
        w = self.PLAYER_RADIUS * 2 * (1 + self.player_squash * 0.3)
        
        player_rect = pygame.Rect(0, 0, int(w), int(h))
        player_rect.center = (int(screen_pos[0]), int(screen_pos[1] - h / 2))

        # Draw glow
        glow_surf = pygame.Surface((w * 2, h * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, self.COLOR_PLAYER_GLOW + (50,), glow_surf.get_rect())
        self.screen.blit(glow_surf, (player_rect.centerx - w, player_rect.centery - h))

        # Draw body
        pygame.draw.ellipse(self.screen, self.COLOR_PLAYER, player_rect)
    
    def _render_ui(self):
        # Display score and steps
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        dist_text = self.font_ui.render(f"DISTANCE: {int(self.player_world_pos[0])}/{self.FINISH_LINE_X}", True, self.COLOR_TEXT)
        self.screen.blit(dist_text, (10, 35))

        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        if self.game_over:
            msg = "MISSION FAILED"
            if self.player_world_pos[0] >= self.FINISH_LINE_X:
                msg = "MISSION COMPLETE!"
            elif self.steps >= self.MAX_STEPS:
                msg = "TIME UP"
            
            over_text = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": self.player_world_pos[0],
        }

    def _create_explosion(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'lifespan': self.np_random.integers(20, 40),
                'max_lifespan': 40,
                'color': self.COLOR_PARTICLE_CRASH,
                'radius': self.np_random.integers(2, 5),
            })

    def _create_jump_particles(self, pos):
        for _ in range(5):
            self.particles.append({
                'pos': pos.copy() + np.array([self.np_random.uniform(-5, 5), 0]),
                'vel': np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(1, 3)]),
                'lifespan': self.np_random.integers(10, 20),
                'max_lifespan': 20,
                'color': self.COLOR_PARTICLE_JUMP,
                'radius': self.np_random.integers(1, 4),
            })

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space after reset
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        # Test game-specific assertions
        assert self.base_asteroid_speed <= 5.0, "Asteroid speed exceeds max value."
        # FIX: Check only the y-component of player_world_pos, not the entire array.
        assert 0 <= self.player_world_pos[1] <= self.GROUND_Y + 1, "Player y-position is out of bounds."

        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play ---
    # This part requires a display and is for testing/demonstration.
    # It won't run in a purely headless environment but is useful for development.
    try:
        import sys
        
        # Re-initialize pygame with a visible display
        pygame.quit()
        os.environ.pop("SDL_VIDEODRIVER", None)
        pygame.init()
        pygame.font.init()
        
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Asteroid Hopper")
        
        obs, info = env.reset()
        
        # Game loop
        while True:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Get key presses for manual control
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP] or keys[pygame.K_RIGHT]:
                movement = 1 # Hop forward
            elif keys[pygame.K_LEFT]:
                movement = 3 # Hop backward
            
            action = [movement, 0, 0] # Space and Shift are unused

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
                obs, info = env.reset()
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Control frame rate
            env.clock.tick(env.FPS)

    except (ImportError, pygame.error) as e:
        print(f"Pygame display not available, cannot run manual play. Error: {e}")
    except Exception as e:
        print(f"An error occurred during manual play: {e}")
        pygame.quit()