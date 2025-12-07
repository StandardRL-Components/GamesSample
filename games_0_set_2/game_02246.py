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
    """
    An arcade-style brick breaker game.

    The player controls a paddle at the bottom of the screen to bounce a ball
    upwards, breaking bricks to score points. The game ends if the player
    loses all three lives or achieves the target score.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ← and → to move the paddle. Break all the bricks to win!"
    )

    game_description = (
        "Bounce a ball to break bricks and score points in this fast-paced arcade game. "
        "Bricks have different point values (Green: 1, Blue: 2, Red: 3). "
        "The ball speeds up as you break more bricks. Don't let the ball fall!"
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 10
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 4.0
    BALL_SPEED_INCREMENT = 0.05
    BALL_MAX_X_VEL_FACTOR = 1.2
    MAX_STEPS = 10000
    WIN_SCORE = 75
    INITIAL_LIVES = 3
    ANTISOFTLOCK_STEPS = 500

    # --- Colors ---
    COLOR_BG = (15, 15, 35)
    COLOR_BG_GRID = (30, 30, 60)
    COLOR_PADDLE = (220, 220, 255)
    COLOR_PADDLE_OUTLINE = (150, 150, 200)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 255, 100, 50)
    COLOR_TEXT = (255, 255, 255)
    BRICK_COLORS = {
        1: (0, 200, 100),   # Green
        2: (0, 150, 255),   # Blue
        3: (255, 80, 80),   # Red
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 64)
        self.font_small = pygame.font.Font(None, 32)
        
        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.lives = None
        self.game_over = None
        self.win = None
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.current_ball_speed = None
        self.bricks = None
        self.particles = None
        self.bricks_destroyed_count = None
        self.steps_since_brick_hit = None
        
        # Note: self.reset() is called with a seed for the first time by the environment wrapper,
        # so we don't need to call it here. However, to allow standalone instantiation,
        # we ensure a seed is available for the first reset.
        self.reset(seed=0)
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.win = False

        self.current_ball_speed = self.INITIAL_BALL_SPEED

        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - self.PADDLE_HEIGHT * 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self._reset_ball()
        
        self.particles = []
        self.bricks_destroyed_count = 0
        self.steps_since_brick_hit = 0

        self._spawn_bricks()

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        """Resets the ball's position and velocity after losing a life or at the start."""
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        # FIX: Swapped arguments to ensure low < high
        angle = self.np_random.uniform(math.radians(-120), math.radians(-60))
        self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.current_ball_speed

    def _spawn_bricks(self):
        """Creates the grid of bricks."""
        self.bricks = []
        brick_width = 50
        brick_height = 20
        gap = 4
        rows = 5
        cols = self.WIDTH // (brick_width + gap)
        start_x = (self.WIDTH - cols * (brick_width + gap) + gap) // 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                points = 1 if r >= 3 else (2 if r >= 1 else 3)
                color = self.BRICK_COLORS[points]
                brick = {
                    "rect": pygame.Rect(
                        start_x + c * (brick_width + gap),
                        start_y + r * (brick_height + gap),
                        brick_width,
                        brick_height,
                    ),
                    "points": points,
                    "color": color,
                }
                self.bricks.append(brick)
    
    def step(self, action):
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0

        # 1. Handle player input
        moved = self._handle_input(movement)
        if moved:
            reward -= 0.01

        # 2. Update game state
        self._update_ball()
        self._update_particles()
        
        # 3. Handle collisions and calculate rewards
        reward += self._handle_collisions()

        # 4. Update progression and check anti-softlock
        self.steps_since_brick_hit += 1
        if self.steps_since_brick_hit > self.ANTISOFTLOCK_STEPS:
            self._spawn_bricks()
            self.steps_since_brick_hit = 0

        # 5. Check for termination conditions
        self.win = self.score >= self.WIN_SCORE
        self.game_over = self.lives <= 0
        
        if self.win:
            reward += 100
        if self.game_over:
            reward -= 100

        terminated = self.win or self.game_over
        truncated = self.steps >= self.MAX_STEPS
        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, movement):
        moved = False
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            moved = True
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            moved = True
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)
        return moved

    def _update_ball(self):
        self.ball_pos += self.ball_vel
    
    def _update_particles(self):
        # Update and remove old particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] = max(0, p["radius"] - 0.2)

    def _handle_collisions(self):
        reward = 0.0
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)

        # Paddle collision
        if ball_rect.colliderect(self.paddle):
            # sound: paddle_hit
            reward += 0.1
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS # Prevent sticking

            # Add horizontal velocity based on hit location
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * self.BALL_MAX_X_VEL_FACTOR
            self.ball_vel.normalize_ip()
            self.ball_vel *= self.current_ball_speed

        # Brick collisions
        for brick in self.bricks[:]:
            if ball_rect.colliderect(brick["rect"]):
                # sound: brick_break
                self.bricks.remove(brick)
                reward += brick["points"]
                self.score += brick["points"]
                self._spawn_particles(brick["rect"].center, brick["color"])
                
                self.ball_vel.y *= -1
                self.bricks_destroyed_count += 1
                self.steps_since_brick_hit = 0

                # Difficulty scaling
                if self.bricks_destroyed_count % 10 == 0:
                    self.current_ball_speed += self.BALL_SPEED_INCREMENT
                    self.ball_vel = self.ball_vel.normalize() * self.current_ball_speed
                break 

        # Bottom wall (lose life)
        if self.ball_pos.y + self.BALL_RADIUS >= self.HEIGHT:
            # sound: lose_life
            self.lives -= 1
            reward -= 5
            if self.lives > 0:
                self._reset_ball()

        return reward

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "radius": self.np_random.uniform(2, 5),
                "color": color
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_BG_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_BG_GRID, (0, y), (self.WIDTH, y))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((int(p["radius"])*2, int(p["radius"])*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, p["pos"] - pygame.Vector2(p["radius"], p["radius"]), special_flags=pygame.BLEND_RGBA_ADD)

        # Bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick["color"], brick["rect"])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in brick["color"]), brick["rect"], 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_OUTLINE, self.paddle, 2, border_radius=3)

        # Ball with glow
        glow_surf = pygame.Surface((self.BALL_RADIUS * 4, self.BALL_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW, (self.BALL_RADIUS*2, self.BALL_RADIUS*2), self.BALL_RADIUS * 1.5)
        self.screen.blit(glow_surf, (int(self.ball_pos.x - self.BALL_RADIUS*2), int(self.ball_pos.y - self.BALL_RADIUS*2)), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        
    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over or self.win:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives}

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Re-enable the normal video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Brick Breaker")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    truncated = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated and not truncated:
            # --- Human Controls ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step the environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- Render the observation to the display window ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        else: # Game is over
            # Allow resetting the game with a key press
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
                truncated = False

        clock.tick(30) # Run at 30 FPS
        
    env.close()