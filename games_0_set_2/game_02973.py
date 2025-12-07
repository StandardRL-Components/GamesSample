
# Generated: 2025-08-28T06:41:02.259254
# Source Brief: brief_02973.md
# Brief Index: 2973

        
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
        "Controls: Hold space for a small jump or shift for a large jump. Avoid the red obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a glowing green line through a procedurally generated obstacle course. Time your jumps to survive as long as possible and reach the 100m finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.PIXELS_PER_METER = 50.0

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_GROUND = (150, 150, 150)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (0, 255, 128)

        # Game constants
        self.GROUND_Y = self.HEIGHT - 50
        self.PLAYER_X = self.WIDTH * 0.2
        self.PLAYER_LINE_HEIGHT = 25
        self.PLAYER_LINE_WIDTH = 4
        self.GRAVITY = 0.6
        self.SMALL_JUMP_VEL = -9
        self.LARGE_JUMP_VEL = -13
        self.MAX_STEPS = 10000

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.distance_traveled = 0.0
        self.player_y = 0
        self.player_vy = 0
        self.on_ground = True
        self.world_scroll_speed = 0
        self.min_obstacle_gap = 0
        self.obstacles = []
        self.particles = []
        self.last_jump_action = False
        
        # Initialize state
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.distance_traveled = 0.0
        
        # Player state
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.on_ground = True
        
        # World state
        self.world_scroll_speed = 1.0 * self.PIXELS_PER_METER / self.FPS # 1 m/s
        self.min_obstacle_gap = 250 # pixels
        
        self.obstacles = []
        self.particles = []
        self._spawn_initial_obstacles()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # Unpack factorized action
        # movement = action[0]  # 0-4: none/up/down/left/right (unused)
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # Handle input
        self._handle_input(space_held, shift_held)
        
        # Update game logic
        self._update_player()
        self._update_particles()
        reward += self._update_obstacles() # Reward for passing obstacles
        self._update_difficulty()

        # Update distance and score
        self.distance_traveled += self.world_scroll_speed / self.PIXELS_PER_METER
        reward += 0.01 # Small reward for surviving
        
        # Check termination conditions
        terminated = False
        collision = self._check_collisions()
        
        if collision:
            reward = -100.0
            terminated = True
            # Sound effect placeholder: # pygame.mixer.Sound('crash.wav').play()
        elif self.distance_traveled >= 100:
            reward = 100.0
            terminated = True
            # Sound effect placeholder: # pygame.mixer.Sound('win.wav').play()
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.steps += 1
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, space_held, shift_held):
        # Allow jump only if on the ground and the key was just pressed
        jump_action = space_held or shift_held
        just_pressed_jump = jump_action and not self.last_jump_action
        
        if self.on_ground and just_pressed_jump:
            if shift_held:
                self.player_vy = self.LARGE_JUMP_VEL
            else: # space_held
                self.player_vy = self.SMALL_JUMP_VEL
            self.on_ground = False
            self._spawn_jump_particles()
            # Sound effect placeholder: # pygame.mixer.Sound('jump.wav').play()

        self.last_jump_action = jump_action

    def _update_player(self):
        if not self.on_ground:
            self.player_vy += self.GRAVITY
            self.player_y += self.player_vy
        
        if self.player_y >= self.GROUND_Y:
            self.player_y = self.GROUND_Y
            self.player_vy = 0
            if not self.on_ground:
                self._spawn_land_particles()
                # Sound effect placeholder: # pygame.mixer.Sound('land.wav').play()
            self.on_ground = True

    def _update_obstacles(self):
        passed_obstacle_reward = 0
        
        last_obstacle_x = 0
        if self.obstacles:
            last_obstacle_x = self.obstacles[-1]['rect'].x
        
        for obs in self.obstacles:
            obs['rect'].x -= self.world_scroll_speed
            # Check for passing
            if not obs['passed'] and obs['rect'].right < self.PLAYER_X:
                obs['passed'] = True
                passed_obstacle_reward += 1.0
                
        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]
        
        # Spawn new obstacles
        if not self.obstacles or self.obstacles[-1]['rect'].right < self.WIDTH - self.min_obstacle_gap:
            self._spawn_obstacle()
            
        return passed_obstacle_reward

    def _update_difficulty(self):
        # Increase speed every 50 steps
        if self.steps > 0 and self.steps % 50 == 0:
            speed_increase = 0.05 * self.PIXELS_PER_METER / self.FPS
            self.world_scroll_speed += speed_increase
        
        # Decrease gap every 100 steps
        if self.steps > 0 and self.steps % 100 == 0:
            self.min_obstacle_gap = max(150, self.min_obstacle_gap - 10)

    def _check_collisions(self):
        player_rect = pygame.Rect(
            self.PLAYER_X, self.player_y - self.PLAYER_LINE_HEIGHT,
            self.PLAYER_LINE_WIDTH, self.PLAYER_LINE_HEIGHT
        )
        for obs in self.obstacles:
            if player_rect.colliderect(obs['rect']):
                return True
        return False

    def _spawn_initial_obstacles(self):
        x = self.WIDTH * 0.6
        while x < self.WIDTH * 2:
            self._spawn_obstacle(x_pos=x)
            if self.obstacles:
                x = self.obstacles[-1]['rect'].right + self.min_obstacle_gap + self.np_random.integers(-50, 50)
            else:
                break
    
    def _spawn_obstacle(self, x_pos=None):
        if x_pos is None:
            x_pos = self.WIDTH + self.np_random.integers(0, 100)
            
        height = self.np_random.integers(20, 100)
        width = self.np_random.integers(30, 60)
        
        rect = pygame.Rect(x_pos, self.GROUND_Y - height, width, height)
        self.obstacles.append({'rect': rect, 'passed': False})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _spawn_jump_particles(self):
        for _ in range(10):
            self.particles.append({
                'pos': [self.PLAYER_X + self.PLAYER_LINE_WIDTH / 2, self.GROUND_Y],
                'vel': [self.np_random.uniform(-1, 1), self.np_random.uniform(-3, -1)],
                'life': self.np_random.integers(15, 25),
                'radius': self.np_random.uniform(1, 4)
            })

    def _spawn_land_particles(self):
         for _ in range(5):
            self.particles.append({
                'pos': [self.PLAYER_X + self.PLAYER_LINE_WIDTH / 2, self.GROUND_Y],
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-1, 0)],
                'life': self.np_random.integers(10, 20),
                'radius': self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Ground
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 2)
        
        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 25.0))
            color = (*self.COLOR_PARTICLE, alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color
            )

        # Player
        player_rect = pygame.Rect(
            self.PLAYER_X, self.player_y - self.PLAYER_LINE_HEIGHT,
            self.PLAYER_LINE_WIDTH, self.PLAYER_LINE_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)
        # Add a glow effect
        glow_rect = player_rect.inflate(6, 6)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER, 50), glow_surf.get_rect(), border_radius=4)
        self.screen.blit(glow_surf, glow_rect.topleft)

    def _render_ui(self):
        dist_text = f"{min(self.distance_traveled, 100.0):.1f}m / 100m"
        text_surface = self.font.render(dist_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        score_text = f"Score: {self.score:.1f}"
        score_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surface, (self.WIDTH - score_surface.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_traveled": self.distance_traveled,
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

        # Test specific game mechanics from brief
        self.reset()
        initial_speed_pixels_per_frame = 1.0 * self.PIXELS_PER_METER / self.FPS
        assert math.isclose(self.world_scroll_speed, initial_speed_pixels_per_frame), "Initial speed is not 1 m/s"

        # Test terminal rewards
        self.reset()
        self.distance_traveled = 100
        _, reward, _, _, _ = self.step(self.action_space.sample())
        assert math.isclose(reward, 100.0), "Reward for reaching 100m is not +100"

        self.reset()
        # Force a collision
        self.player_y = self.GROUND_Y
        self.obstacles = [{'rect': pygame.Rect(self.PLAYER_X, self.GROUND_Y - 20, 20, 20), 'passed': False}]
        _, reward, _, _, _ = self.step(self.action_space.sample())
        assert math.isclose(reward, -100.0), "Reward for collision is not -100"
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    # pip install pygame gymnasium
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Line Jumper")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    print("\n" + "="*30)
    print("      LINE JUMPER")
    print("="*30)
    print(env.game_description)
    print("\n" + env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        space_held = False
        shift_held = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = True
        if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
            terminated = True
            
        action = [0, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
        if terminated:
            print(f"Game Over! Final Info: {info}")

    env.close()