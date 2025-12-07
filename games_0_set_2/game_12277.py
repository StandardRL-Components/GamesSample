import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:31:06.080160
# Source Brief: brief_02277.md
# Brief Index: 2277
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Ball:
    """Represents a player-controlled ball."""
    def __init__(self, x, y, radius, color, speed_multiplier=1.0):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.radius = radius
        self.color = color
        self.tilt = 0  # -1 for left, 0 for none, 1 for right
        self.speed_multiplier = speed_multiplier

    def update(self, gravity, friction, tilt_strength, max_x, max_y):
        # Apply tilt acceleration
        self.vel.x += self.tilt * tilt_strength * self.speed_multiplier

        # Apply gravity
        self.vel.y += gravity * self.speed_multiplier

        # Apply friction
        self.vel.x *= friction

        # Update position
        self.pos += self.vel

        # Wall bounce
        if self.pos.x - self.radius < 0:
            self.pos.x = self.radius
            self.vel.x *= -0.9
        elif self.pos.x + self.radius > max_x:
            self.pos.x = max_x - self.radius
            self.vel.x *= -0.9

        # Floor/Ceiling bounce
        if self.pos.y - self.radius < 0:
            self.pos.y = self.radius
            self.vel.y *= -0.9
        elif self.pos.y + self.radius > max_y:
            self.pos.y = max_y - self.radius
            self.vel.y *= -0.9

    def draw(self, surface):
        # Draw a shadow
        shadow_pos = (int(self.pos.x + 3), int(self.pos.y + 3))
        pygame.gfxdraw.filled_circle(surface, shadow_pos[0], shadow_pos[1], self.radius, (0, 0, 0, 100))
        
        # Draw the ball
        pos_int = (int(self.pos.x), int(self.pos.y))
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], self.radius, self.color)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], self.radius, self.color)

class Orb:
    """Represents a collectible orb."""
    def __init__(self, x, y, radius, orb_type):
        self.pos = pygame.Vector2(x, y)
        self.radius = radius
        self.type = orb_type # 'green' or 'red'
        self.color = (50, 255, 50) if orb_type == 'green' else (255, 50, 50)
        self.glow_color = (150, 255, 150) if orb_type == 'green' else (255, 150, 150)
        self.pulse_timer = random.uniform(0, 2 * math.pi)

    def draw(self, surface):
        self.pulse_timer += 0.1
        pulse_radius = self.radius + 2 * math.sin(self.pulse_timer)
        
        # Draw glow
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(pulse_radius) + 3, self.glow_color + (50,))
        
        # Draw orb
        pos_int = (int(self.pos.x), int(self.pos.y))
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(pulse_radius), self.color)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(pulse_radius), self.color)

class Particle:
    """Represents a single particle for effects."""
    def __init__(self, x, y, color):
        self.pos = pygame.Vector2(x, y)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        self.lifespan = random.randint(20, 40)
        self.color = color

    def update(self):
        self.pos += self.vel
        self.vel *= 0.98 # friction
        self.lifespan -= 1

    def draw(self, surface):
        alpha = max(0, 255 * (self.lifespan / 40))
        size = max(1, int(4 * (self.lifespan / 40)))
        pygame.draw.rect(surface, self.color + (alpha,), (int(self.pos.x), int(self.pos.y), size, size))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control three balls at once to collect green orbs and avoid red ones. "
        "Earn a bonus for synchronized collections."
    )
    user_guide = (
        "Use arrow keys (←→) to tilt the first ball. Use secondary actions "
        "(like holding Shift or another key) to tilt the other two balls."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1800 # 60 seconds at 30fps
        
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (26, 26, 46)
        self.COLOR_UI_TEXT = (230, 230, 255)
        self.BALL_COLORS = [(233, 69, 96), (22, 199, 154), (248, 180, 0)] # Red, Green, Yellow

        # Game Physics Constants
        self.GRAVITY = 0.15
        self.FRICTION = 0.99
        self.TILT_STRENGTH = 0.25
        self.ORB_SPAWN_INTERVAL = 40
        
        # Initialize state variables
        self.balls = []
        self.orbs = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sync_flash_timer = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sync_flash_timer = 0
        
        self.balls = [
            Ball(self.WIDTH * 0.25, self.HEIGHT / 2, 15, self.BALL_COLORS[0]),
            Ball(self.WIDTH * 0.50, self.HEIGHT / 2, 15, self.BALL_COLORS[1]),
            Ball(self.WIDTH * 0.75, self.HEIGHT / 2, 15, self.BALL_COLORS[2]),
        ]
        
        self.orbs = []
        self.particles = []
        for _ in range(5):
            self._spawn_orb()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.01 # Survival reward
        
        # 1. Handle Input
        self._handle_input(action)
        
        # 2. Update Physics
        for ball in self.balls:
            ball.update(self.GRAVITY, self.FRICTION, self.TILT_STRENGTH, self.WIDTH, self.HEIGHT)
            
        # 3. Handle Collisions
        reward += self._handle_collisions()

        # 4. Update Particles and Effects
        self._update_particles()
        if self.sync_flash_timer > 0:
            self.sync_flash_timer -= 1
        
        # 5. Spawn new orbs
        if self.steps % self.ORB_SPAWN_INTERVAL == 0:
            self._spawn_orb()

        # 6. Update Step Counter and Check Termination
        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS
        if terminated:
            reward += 10 # Survival bonus
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, action):
        """Maps MultiDiscrete action to ball tilts."""
        # Ball 1 Tilt: action[0] (3=left, 4=right)
        if action[0] == 3: self.balls[0].tilt = -1
        elif action[0] == 4: self.balls[0].tilt = 1
        else: self.balls[0].tilt = 0

        # Ball 2 Tilt: action[1] (0=right, 1=left)
        self.balls[1].tilt = 1 if action[1] == 1 else -1

        # Ball 3 Tilt: action[2] (0=right, 1=left)
        self.balls[2].tilt = 1 if action[2] == 1 else -1

    def _handle_collisions(self):
        reward_delta = 0
        collected_orbs_indices = set()
        collected_by_ball = [False, False, False]

        for i, ball in enumerate(self.balls):
            for j, orb in enumerate(self.orbs):
                if j in collected_orbs_indices:
                    continue
                
                distance = ball.pos.distance_to(orb.pos)
                if distance < ball.radius + orb.radius:
                    collected_orbs_indices.add(j)
                    
                    if orb.type == 'green':
                        # // SFX: Positive chime
                        self.score += 1
                        reward_delta += 1
                        collected_by_ball[i] = True
                        self._spawn_particles(orb.pos, orb.color, 15)
                    elif orb.type == 'red':
                        # // SFX: Negative buzz
                        self.score -= 2
                        reward_delta -= 2
                        for b in self.balls:
                            b.speed_multiplier *= 1.15
                        self._spawn_particles(orb.pos, orb.color, 25)
        
        # Check for sync bonus
        if all(collected_by_ball):
            # // SFX: Big success sound
            self.score += 5
            reward_delta += 5
            self.sync_flash_timer = 10 # frames
        
        # Remove collected orbs
        if collected_orbs_indices:
            self.orbs = [orb for i, orb in enumerate(self.orbs) if i not in collected_orbs_indices]

        return reward_delta

    def _spawn_orb(self):
        if len(self.orbs) >= 15: return # Max orbs on screen
        
        orb_type = 'green' if random.random() > 0.3 else 'red'
        radius = 10
        pos = pygame.Vector2(
            random.randint(radius, self.WIDTH - radius),
            random.randint(radius, self.HEIGHT - radius)
        )
        self.orbs.append(Orb(pos.x, pos.y, radius, orb_type))
        
    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(Particle(pos.x, pos.y, color))
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles (background)
        for p in self.particles:
            p.draw(self.screen)

        # Render orbs
        for orb in self.orbs:
            orb.draw(self.screen)
            
        # Render balls
        for ball in self.balls:
            ball.draw(self.screen)

        # Render sync flash
        if self.sync_flash_timer > 0:
            alpha = int(200 * (self.sync_flash_timer / 10))
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Time Left
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_large.render(f"Time: {time_left // 30}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        # Ball Speeds
        y_offset = 55
        for i, ball in enumerate(self.balls):
            speed_val = ball.speed_multiplier
            speed_text = self.font_small.render(f"Ball {i+1} Speed: {speed_val:.2f}x", True, ball.color)
            self.screen.blit(speed_text, (10, y_offset + i * 20))
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
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
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To control:
    # Ball 1 (Red):    'A' (left) / 'D' (right)
    # Ball 2 (Green):  'J' (left) / 'L' (right)
    # Ball 3 (Yellow): Left Arrow / Right Arrow
    
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Trio Tilt")
    screen_display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample() 
    action[0] = 0 # Start with no tilt for ball 1
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Ball 1 Control
        if keys[pygame.K_a]: action[0] = 3 # Left
        elif keys[pygame.K_d]: action[0] = 4 # Right
        else: action[0] = 0 # None
        
        # Ball 2 Control
        action[1] = 1 if keys[pygame.K_j] else 0 # 1=left, 0=right (J/L)
        if not keys[pygame.K_j] and not keys[pygame.K_l]:
            # This is tricky. The space doesn't allow 'no action'.
            # We can't set tilt to 0, only left or right.
            # For human play, we can just alternate to keep it still.
            action[1] = env.steps % 2

        # Ball 3 Control
        action[2] = 1 if keys[pygame.K_LEFT] else 0 # 1=left, 0=right
        if not keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            action[2] = env.steps % 2

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play
        
    env.close()
    print(f"Game Over! Final Score: {info['score']}")