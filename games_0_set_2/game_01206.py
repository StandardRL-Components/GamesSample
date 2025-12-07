
# Generated: 2025-08-27T16:22:39.454011
# Source Brief: brief_01206.md
# Brief Index: 1206

        
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
        "Press SPACE to jump. Combine with SHIFT for a low jump, ← for a short jump, or → for a long jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced rhythm runner. Jump between platforms to the beat, timing risky landings for a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.BPM = 120
        self.BEAT_DURATION = (60 / self.BPM) * self.FPS
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_OUTLINE = (200, 255, 255)
        self.COLOR_OBSTACLE = (200, 0, 100)
        self.COLOR_SAFE_ZONE = (0, 200, 100)
        self.COLOR_RISKY_ZONE = (255, 200, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_BEAT = (255, 255, 255)
        
        # Physics constants
        self.GRAVITY = 0.8
        self.PLAYER_SIZE = 20
        self.PLAYER_X_TARGET = 120
        self.JUMP_VEL_HIGH = -14
        self.JUMP_VEL_LOW = -10
        self.JUMP_MOD_SHORT = -3
        self.JUMP_MOD_LONG = 3
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_score = pygame.font.SysFont("monospace", 32, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.is_grounded = None
        self.platforms = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.beat_progress = None
        self.jumps_cleared = None
        self.obstacle_speed = None
        self.last_action_time = None

        self.reset()

        # Run validation check
        self.validate_implementation()

    def _generate_platforms(self):
        last_platform = self.platforms[-1]
        
        for _ in range(5): # Generate a few platforms ahead
            gap_x = random.uniform(80, 200)
            gap_y = random.uniform(-80, 80)
            
            width = random.uniform(150, 400)
            height = 80
            
            new_x = last_platform.rect.right + gap_x
            new_y = np.clip(last_platform.rect.y + gap_y, 150, self.HEIGHT - 50)
            
            new_platform = self.Platform(new_x, new_y, width, height)
            self.platforms.append(new_platform)
            last_platform = new_platform

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = pygame.Vector2(self.PLAYER_X_TARGET, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = True
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.jumps_cleared = 0
        self.obstacle_speed = 4.0
        
        # World state
        initial_platform = self.Platform( -self.PLAYER_SIZE, self.HEIGHT - 80, self.WIDTH, 80)
        self.platforms = [initial_platform]
        self._generate_platforms()
        
        # Effects
        self.particles = []
        self.beat_progress = 0
        self.last_action_time = -100 # Allow immediate first action
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Beat and Timing ---
        self.beat_progress = (self.beat_progress + 1) % self.BEAT_DURATION

        # --- Player Logic ---
        # Action cooldown to prevent spamming jumps
        can_act = self.steps > self.last_action_time + 5

        if self.is_grounded and space_held and can_act:
            self.is_grounded = False
            self.last_action_time = self.steps
            # Sound: Jump
            
            # Determine jump type
            if shift_held: # Low jump
                self.player_vel.y = self.JUMP_VEL_LOW
            else: # High jump
                self.player_vel.y = self.JUMP_VEL_HIGH
            
            if movement == 3: # Short jump
                self.player_vel.x = self.JUMP_MOD_SHORT
            elif movement == 4: # Long jump
                self.player_vel.x = self.JUMP_MOD_LONG
            else: # Normal horizontal
                self.player_vel.x = 0

            # Add particles on jump
            for _ in range(10):
                self.particles.append(self.Particle(self.player_pos.copy(), self.COLOR_PLAYER_OUTLINE, self.np_random))

        # --- Physics and World Update ---
        # Update platforms and remove off-screen ones
        for p in self.platforms:
            p.rect.x -= self.obstacle_speed
        self.platforms = [p for p in self.platforms if p.rect.right > 0]
        
        # Generate new platforms if needed
        if self.platforms[-1].rect.right < self.WIDTH + 200:
            self._generate_platforms()
            
        # Apply gravity and update player position
        if not self.is_grounded:
            self.player_vel.y += self.GRAVITY
            reward -= 0.05 # Small penalty for being in the air, encouraging efficiency
        
        self.player_pos += self.player_vel
        
        # Horizontal correction towards center
        self.player_vel.x *= 0.9 # Air drag
        if self.is_grounded:
             self.player_vel.x *= 0.7 # Ground friction
        
        # Correct player's X position back to the target
        correction_force = (self.PLAYER_X_TARGET - self.player_pos.x) * 0.1
        self.player_vel.x += correction_force
        
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # --- Collision Detection ---
        if not self.is_grounded and self.player_vel.y > 0:
            for p in self.platforms:
                if player_rect.colliderect(p.rect) and player_rect.bottom < p.rect.bottom:
                    # Check for landing on top surface
                    if abs(player_rect.bottom - p.rect.top) < self.player_vel.y + 1:
                        self.is_grounded = True
                        self.player_pos.y = p.rect.top - self.PLAYER_SIZE
                        self.player_vel.y = 0
                        # Sound: Land
                        
                        # Determine landing zone and reward
                        if p.risky_zone_left.collidepoint(player_rect.centerx, p.rect.top) or \
                           p.risky_zone_right.collidepoint(player_rect.centerx, p.rect.top):
                            reward += 5
                            self.score += 5
                            # Sound: Risky Land
                            for _ in range(20):
                                self.particles.append(self.Particle(self.player_pos.copy(), self.COLOR_RISKY_ZONE, self.np_random, intensity=1.5))
                        else:
                            reward += 1
                            self.score += 1
                            # Sound: Safe Land
                            for _ in range(5):
                                self.particles.append(self.Particle(self.player_pos.copy(), self.COLOR_SAFE_ZONE, self.np_random))
                        
                        self.jumps_cleared += 1
                        # Difficulty scaling
                        if self.jumps_cleared > 0 and self.jumps_cleared % 20 == 0:
                            self.obstacle_speed = min(8.0, self.obstacle_speed + 0.05)
                        
                        break # Stop checking other platforms
        
        # Check for side collision or falling
        is_colliding_side = False
        if not self.is_grounded:
             for p in self.platforms:
                if player_rect.colliderect(p.rect):
                    is_colliding_side = True
                    break
        
        if is_colliding_side or self.player_pos.y > self.HEIGHT:
            self.game_over = True
            reward = -100
            self.score -= 100
            # Sound: Fail/Explode
            for _ in range(50):
                self.particles.append(self.Particle(self.player_pos.copy(), self.COLOR_OBSTACLE, self.np_random, intensity=2.0))

        # --- Particle Update ---
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]
        
        # --- Termination ---
        terminated = self.game_over or self.steps >= 1000
        if self.steps >= 1000 and not self.game_over:
            reward += 50
            self.score += 50

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        
        # Pulsing grid
        pulse = abs(math.sin(self.beat_progress / self.BEAT_DURATION * math.pi))
        alpha = int(30 + pulse * 40)
        grid_color = (*self.COLOR_GRID[:3], alpha)
        
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, grid_color, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, grid_color, (0, i), (self.WIDTH, i))

    def _render_game(self):
        # Render platforms
        for p in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, p.rect)
            pygame.draw.rect(self.screen, self.COLOR_SAFE_ZONE, p.safe_zone)
            pygame.draw.rect(self.screen, self.COLOR_RISKY_ZONE, p.risky_zone_left)
            pygame.draw.rect(self.screen, self.COLOR_RISKY_ZONE, p.risky_zone_right)
            
        # Render particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Render player
        player_rect = pygame.Rect(int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.gfxdraw.box(self.screen, player_rect, self.COLOR_PLAYER)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, 2)

    def _render_ui(self):
        # Render score
        score_text = self.font_score.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Render beat visualizer
        pulse = abs(math.sin(self.beat_progress / self.BEAT_DURATION * math.pi))
        radius = int(15 + pulse * 10)
        alpha = int(50 + pulse * 205)
        beat_color = (*self.COLOR_BEAT[:3], alpha)
        
        beat_pos = (int(self.player_pos.x + self.PLAYER_SIZE / 2), int(self.player_pos.y - 30))
        pygame.gfxdraw.filled_circle(self.screen, beat_pos[0], beat_pos[1], radius, beat_color)
        pygame.gfxdraw.aacircle(self.screen, beat_pos[0], beat_pos[1], radius, beat_color)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "jumps": self.jumps_cleared,
            "speed": self.obstacle_speed,
        }

    def close(self):
        pygame.quit()

    class Platform:
        def __init__(self, x, y, width, height):
            self.rect = pygame.Rect(x, y, width, height)
            risky_width = min(30, width * 0.15)
            
            self.safe_zone = pygame.Rect(
                x + risky_width, y,
                width - 2 * risky_width, 5
            )
            self.risky_zone_left = pygame.Rect(x, y, risky_width, 5)
            self.risky_zone_right = pygame.Rect(x + width - risky_width, y, risky_width, 5)

        def __getstate__(self):
            return (self.rect.topleft, self.rect.size)

        def __setstate__(self, state):
            topleft, size = state
            self.__init__(topleft[0], topleft[1], size[0], size[1])


    class Particle:
        def __init__(self, pos, color, rng, intensity=1.0):
            self.pos = pos.copy()
            angle = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(1, 4) * intensity
            self.vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.lifespan = rng.integers(15, 30)
            self.initial_lifespan = self.lifespan
            self.color = color
            self.radius = rng.uniform(2, 5) * intensity

        def update(self):
            self.pos += self.vel
            self.vel *= 0.95
            self.lifespan -= 1

        def draw(self, surface):
            alpha = int(255 * (self.lifespan / self.initial_lifespan))
            color_with_alpha = (*self.color[:3], alpha)
            pygame.gfxdraw.filled_circle(
                surface, int(self.pos.x), int(self.pos.y), int(self.radius), color_with_alpha
            )
    
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
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # For human play, we need a different setup
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Rhythm Runner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    running = True
    total_reward = 0
    
    # Main game loop
    while running:
        # Map keyboard inputs to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        if terminated:
            # If the game is over, wait for a key press to reset
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
        else:
            # If the game is running, step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # --- Pygame-specific event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Display the observation ---
        # The observation is (H, W, C), but pygame wants (W, H) surface
        # and surfarray.make_surface expects (W, H, C)
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(env.FPS)
        
    env.close()