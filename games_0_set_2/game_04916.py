
# Generated: 2025-08-28T03:23:43.914605
# Source Brief: brief_04916.md
# Brief Index: 4916

        
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


# --- Constants ---
WIDTH, HEIGHT = 640, 400
PADDLE_WIDTH, PADDLE_THICKNESS = 120, 15
BALL_RADIUS = 10
PADDLE_SPEED = 10
GRAVITY_STRENGTH = 0.35
BOUNCE_DAMPING = 0.98
MAX_SCORE = 5
MAX_MISSES = 2
MAX_STEPS = 1000

# --- Colors ---
COLOR_BG = (25, 28, 40)
COLOR_PADDLE = (230, 230, 255)
COLOR_BALL = (255, 100, 100)
COLOR_WALL = (45, 50, 70)
COLOR_GRAVITY = (100, 255, 120)
COLOR_PARTICLE = (255, 220, 80)
COLOR_TEXT = (240, 240, 240)
COLOR_TEXT_SHADOW = (20, 20, 30)

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, vel, radius, lifetime, color):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.radius = radius
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.color = color

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95  # Friction
        self.lifetime -= 1

    def draw(self, surface):
        if self.is_alive():
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                temp_surf,
                (*self.color, alpha),
                (self.radius, self.radius),
                self.radius,
            )
            surface.blit(temp_surf, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)))

    def is_alive(self):
        return self.lifetime > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move paddle. Press space to cycle gravity."
    )

    game_description = (
        "A minimalist arcade game where you control a paddle under shifting gravity to keep a ball in play. Score points by making the ball exit the screen on the side opposite your paddle."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 40, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)
        
        self.ball_pos = pygame.math.Vector2(0, 0)
        self.ball_vel = pygame.math.Vector2(0, 0)
        self.paddle_pos = 0.0
        self.gravity_dir = 0  # 0:down, 1:left, 2:up, 3:right
        self.gravity_vectors = [
            pygame.math.Vector2(0, GRAVITY_STRENGTH),
            pygame.math.Vector2(-GRAVITY_STRENGTH, 0),
            pygame.math.Vector2(0, -GRAVITY_STRENGTH),
            pygame.math.Vector2(GRAVITY_STRENGTH, 0),
        ]
        self.score = 0
        self.misses = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.prev_space_held = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.particles = []
        self.prev_space_held = False
        self.gravity_dir = 0
        
        self._reset_ball(scored=False)
        
        if self.gravity_dir in [0, 2]: # Horizontal paddle
            self.paddle_pos = WIDTH / 2
        else: # Vertical paddle
            self.paddle_pos = HEIGHT / 2
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
        
        reward = 0
        
        if not self.game_over:
            events = self._handle_input_and_update_state(action)
            
            if events.get("paddle_hit"):
                reward += 0.1
                # SFX: Paddle hit sound
            if events.get("scored"):
                reward += 1.0
                # SFX: Score point sound
            if events.get("missed"):
                # SFX: Miss sound
                pass

        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= MAX_SCORE:
                reward += 10.0 # Win bonus
                # SFX: Win jingle
            else:
                reward -= 10.0 # Lose penalty
                # SFX: Lose sound
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input_and_update_state(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle_pos -= PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_pos += PADDLE_SPEED
            
        if space_held and not self.prev_space_held:
            self.gravity_dir = (self.gravity_dir + 1) % 4
            # SFX: Gravity shift sound
        self.prev_space_held = space_held
        
        # Clamp paddle position
        if self.gravity_dir in [0, 2]: # Horizontal
            self.paddle_pos = np.clip(self.paddle_pos, PADDLE_WIDTH / 2, WIDTH - PADDLE_WIDTH / 2)
        else: # Vertical
            self.paddle_pos = np.clip(self.paddle_pos, PADDLE_WIDTH / 2, HEIGHT - PADDLE_WIDTH / 2)

        # --- Update Game State ---
        events = {}
        
        # Apply gravity
        self.ball_vel += self.gravity_vectors[self.gravity_dir]
        
        # Move ball
        self.ball_pos += self.ball_vel
        
        # Collisions
        paddle_rect = self._get_paddle_rect()
        
        # Paddle bounce
        if paddle_rect.colliderect(self._get_ball_rect()):
            # SFX: Paddle bounce
            events["paddle_hit"] = True
            self._create_particles(self.ball_pos, 15)
            
            if self.gravity_dir == 0: # Down
                self.ball_pos.y = paddle_rect.top - BALL_RADIUS
                self.ball_vel.y *= -BOUNCE_DAMPING
                self.ball_vel.x += (self.ball_pos.x - self.paddle_pos) * 0.1
            elif self.gravity_dir == 1: # Left
                self.ball_pos.x = paddle_rect.right + BALL_RADIUS
                self.ball_vel.x *= -BOUNCE_DAMPING
                self.ball_vel.y += (self.ball_pos.y - self.paddle_pos) * 0.1
            elif self.gravity_dir == 2: # Up
                self.ball_pos.y = paddle_rect.bottom + BALL_RADIUS
                self.ball_vel.y *= -BOUNCE_DAMPING
                self.ball_vel.x += (self.ball_pos.x - self.paddle_pos) * 0.1
            elif self.gravity_dir == 3: # Right
                self.ball_pos.x = paddle_rect.left - BALL_RADIUS
                self.ball_vel.x *= -BOUNCE_DAMPING
                self.ball_vel.y += (self.ball_pos.y - self.paddle_pos) * 0.1
            
            self.ball_vel.x = np.clip(self.ball_vel.x, -15, 15)
            self.ball_vel.y = np.clip(self.ball_vel.y, -15, 15)

        # Wall bounces (non-play-axis walls)
        if self.gravity_dir in [0, 2]: # Horizontal paddle, vertical play
            if self.ball_pos.x < BALL_RADIUS:
                self.ball_pos.x = BALL_RADIUS
                self.ball_vel.x *= -BOUNCE_DAMPING
                self._create_particles(self.ball_pos, 5)
            elif self.ball_pos.x > WIDTH - BALL_RADIUS:
                self.ball_pos.x = WIDTH - BALL_RADIUS
                self.ball_vel.x *= -BOUNCE_DAMPING
                self._create_particles(self.ball_pos, 5)
        else: # Vertical paddle, horizontal play
            if self.ball_pos.y < BALL_RADIUS:
                self.ball_pos.y = BALL_RADIUS
                self.ball_vel.y *= -BOUNCE_DAMPING
                self._create_particles(self.ball_pos, 5)
            elif self.ball_pos.y > HEIGHT - BALL_RADIUS:
                self.ball_pos.y = HEIGHT - BALL_RADIUS
                self.ball_vel.y *= -BOUNCE_DAMPING
                self._create_particles(self.ball_pos, 5)

        # Score / Miss check
        scored_or_missed = False
        if self.gravity_dir == 0 and self.ball_pos.y > HEIGHT + BALL_RADIUS:
            self.misses += 1
            events["missed"] = True
            scored_or_missed = True
        elif self.gravity_dir == 0 and self.ball_pos.y < -BALL_RADIUS:
            self.score += 1
            events["scored"] = True
            scored_or_missed = True
        elif self.gravity_dir == 1 and self.ball_pos.x > WIDTH + BALL_RADIUS:
            self.misses += 1
            events["missed"] = True
            scored_or_missed = True
        elif self.gravity_dir == 1 and self.ball_pos.x < -BALL_RADIUS:
            self.score += 1
            events["scored"] = True
            scored_or_missed = True
        elif self.gravity_dir == 2 and self.ball_pos.y < -BALL_RADIUS:
            self.misses += 1
            events["missed"] = True
            scored_or_missed = True
        elif self.gravity_dir == 2 and self.ball_pos.y > HEIGHT + BALL_RADIUS:
            self.score += 1
            events["scored"] = True
            scored_or_missed = True
        elif self.gravity_dir == 3 and self.ball_pos.x < -BALL_RADIUS:
            self.misses += 1
            events["missed"] = True
            scored_or_missed = True
        elif self.gravity_dir == 3 and self.ball_pos.x > WIDTH + BALL_RADIUS:
            self.score += 1
            events["scored"] = True
            scored_or_missed = True
            
        if scored_or_missed:
            self._reset_ball(scored=events.get("scored", False))

        # Update particles
        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles:
            p.update()
            
        return events

    def _reset_ball(self, scored):
        if self.gravity_dir == 0: # Down
            self.ball_pos = pygame.math.Vector2(self.np_random.uniform(WIDTH * 0.2, WIDTH * 0.8), HEIGHT * 0.2)
            self.ball_vel = pygame.math.Vector2(self.np_random.uniform(-2, 2), 0)
        elif self.gravity_dir == 1: # Left
            self.ball_pos = pygame.math.Vector2(WIDTH * 0.8, self.np_random.uniform(HEIGHT * 0.2, HEIGHT * 0.8))
            self.ball_vel = pygame.math.Vector2(0, self.np_random.uniform(-2, 2))
        elif self.gravity_dir == 2: # Up
            self.ball_pos = pygame.math.Vector2(self.np_random.uniform(WIDTH * 0.2, WIDTH * 0.8), HEIGHT * 0.8)
            self.ball_vel = pygame.math.Vector2(self.np_random.uniform(-2, 2), 0)
        elif self.gravity_dir == 3: # Right
            self.ball_pos = pygame.math.Vector2(WIDTH * 0.2, self.np_random.uniform(HEIGHT * 0.2, HEIGHT * 0.8))
            self.ball_vel = pygame.math.Vector2(0, self.np_random.uniform(-2, 2))

    def _check_termination(self):
        return (
            self.score >= MAX_SCORE
            or self.misses >= MAX_MISSES
            or self.steps >= MAX_STEPS
        )

    def _get_observation(self):
        self.screen.fill(COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "misses": self.misses}

    def _get_paddle_rect(self):
        if self.gravity_dir == 0: # Down
            return pygame.Rect(self.paddle_pos - PADDLE_WIDTH / 2, HEIGHT - PADDLE_THICKNESS, PADDLE_WIDTH, PADDLE_THICKNESS)
        elif self.gravity_dir == 1: # Left
            return pygame.Rect(WIDTH - PADDLE_THICKNESS, self.paddle_pos - PADDLE_WIDTH / 2, PADDLE_THICKNESS, PADDLE_WIDTH)
        elif self.gravity_dir == 2: # Up
            return pygame.Rect(self.paddle_pos - PADDLE_WIDTH / 2, 0, PADDLE_WIDTH, PADDLE_THICKNESS)
        else: # Right
            return pygame.Rect(0, self.paddle_pos - PADDLE_WIDTH / 2, PADDLE_THICKNESS, PADDLE_WIDTH)

    def _get_ball_rect(self):
        return pygame.Rect(self.ball_pos.x - BALL_RADIUS, self.ball_pos.y - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)

    def _create_particles(self, pos, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            radius = self.np_random.uniform(1, 4)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append(Particle(pos, vel, radius, lifetime, COLOR_PARTICLE))

    def _render_game(self):
        # Draw play area walls
        pygame.draw.rect(self.screen, COLOR_WALL, (0, 0, WIDTH, HEIGHT), 2)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
        
        # Draw paddle
        paddle_rect = self._get_paddle_rect()
        pygame.draw.rect(self.screen, COLOR_PADDLE, paddle_rect, border_radius=3)
        
        # Draw ball with antialiasing and glow
        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, *ball_pos_int, BALL_RADIUS, COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, *ball_pos_int, BALL_RADIUS, COLOR_BALL)
        
        # Draw gravity indicator
        center = (WIDTH // 2, HEIGHT // 2)
        arrow_size = 20
        if self.gravity_dir == 0: # Down
            points = [(center[0], center[1] + arrow_size), (center[0] - arrow_size/2, center[1]), (center[0] + arrow_size/2, center[1])]
        elif self.gravity_dir == 1: # Left
            points = [(center[0] - arrow_size, center[1]), (center[0], center[1] - arrow_size/2), (center[0], center[1] + arrow_size/2)]
        elif self.gravity_dir == 2: # Up
            points = [(center[0], center[1] - arrow_size), (center[0] - arrow_size/2, center[1]), (center[0] + arrow_size/2, center[1])]
        else: # Right
            points = [(center[0] + arrow_size, center[1]), (center[0], center[1] - arrow_size/2), (center[0], center[1] + arrow_size/2)]
        pygame.gfxdraw.aapolygon(self.screen, points, COLOR_GRAVITY)
        pygame.gfxdraw.filled_polygon(self.screen, points, COLOR_GRAVITY)
        
    def _render_ui(self):
        # Draw score
        score_text = f"SCORE: {self.score}"
        text_surf = self.font_small.render(score_text, True, COLOR_TEXT)
        shadow_surf = self.font_small.render(score_text, True, COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (WIDTH // 2 - text_surf.get_width() // 2 + 2, 12))
        self.screen.blit(text_surf, (WIDTH // 2 - text_surf.get_width() // 2, 10))
        
        # Draw misses
        miss_text = "●" * self.misses + "○" * (MAX_MISSES - self.misses)
        miss_surf = self.font_small.render(miss_text, True, COLOR_BALL)
        self.screen.blit(miss_surf, (WIDTH - miss_surf.get_width() - 20, 10))
        
        # Draw game over message
        if self.game_over:
            msg = "YOU WIN!" if self.score >= MAX_SCORE else "GAME OVER"
            text_surf = self.font_large.render(msg, True, COLOR_TEXT)
            shadow_surf = self.font_large.render(msg, True, COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (WIDTH // 2 - text_surf.get_width() // 2 + 3, HEIGHT // 2 - text_surf.get_height() // 2 + 3))
            self.screen.blit(text_surf, (WIDTH // 2 - text_surf.get_width() // 2, HEIGHT // 2 - text_surf.get_height() // 2))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Interactive Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up Pygame window for display
    pygame.display.set_caption("Gravity Paddle")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    
    # Game loop
    running = True
    while running:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Key state
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
        
        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause before reset
            env.reset()

    env.close()