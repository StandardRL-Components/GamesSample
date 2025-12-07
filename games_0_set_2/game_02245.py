
# Generated: 2025-08-28T04:14:05.403937
# Source Brief: brief_02245.md
# Brief Index: 2245

        
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


# Helper classes can be defined outside the main environment class for clarity.
class Particle:
    """A simple particle for effects."""
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95  # Damping
        self.lifetime -= 1
        self.radius -= 0.1
        
    def draw(self, surface, iso_transform_func):
        if self.lifetime > 0 and self.radius > 0:
            screen_pos = iso_transform_func(self.pos.x, self.pos.y)
            # Draw a simple circle for particles, no need for complex aa
            pygame.draw.circle(surface, self.color, (int(screen_pos[0]), int(screen_pos[1])), int(self.radius))

class Brick:
    """Represents a single brick in the game."""
    def __init__(self, world_pos, world_dims, color):
        self.rect = pygame.Rect(world_pos, world_dims)
        self.color = color
        # Pre-calculate darker shades for 3D effect
        self.color_light = tuple(min(255, c + 40) for c in color)
        self.color_shadow_1 = tuple(max(0, c - 40) for c in color)
        self.color_shadow_2 = tuple(max(0, c - 60) for c in color)
        self.alive = True

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the paddle. Your goal is to destroy all the bricks."
    )

    game_description = (
        "An isometric arcade game. Bounce the ball off your paddle to destroy all the bricks."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 32, 44
        self.ISO_TILE_WIDTH_HALF, self.ISO_TILE_HEIGHT_HALF = 14, 7
        self.ISO_OFFSET_X, self.ISO_OFFSET_Y = self.SCREEN_WIDTH // 2, 50

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.BRICK_COLORS = [
            (227, 91, 91), (91, 227, 159), (91, 159, 227),
            (227, 227, 91), (159, 91, 227), (91, 227, 227)
        ]

        # Game constants
        self.PADDLE_WIDTH, self.PADDLE_DEPTH, self.PADDLE_HEIGHT = 6, 2, 8
        self.PADDLE_Y = self.WORLD_HEIGHT - 5
        self.PADDLE_SPEED = 1.0
        self.BALL_RADIUS = 0.7
        self.BALL_SPEED = 0.75
        self.PADDLE_SPIN_EFFECT = 0.5
        self.MAX_STEPS = 1000
        self.INITIAL_LIVES = 3

        # Game state variables (will be initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.paddle_pos_x = 0
        self.ball_pos = pygame.math.Vector2(0, 0)
        self.ball_vel = pygame.math.Vector2(0, 0)
        self.bricks = []
        self.particles = []
        self.softlock_counter = 0

        # Final check
        self.validate_implementation()
    
    def _iso_to_screen(self, x, y):
        """Converts world coordinates to screen coordinates."""
        screen_x = self.ISO_OFFSET_X + (x - y) * self.ISO_TILE_WIDTH_HALF
        screen_y = self.ISO_OFFSET_Y + (x + y) * self.ISO_TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, world_rect, height, top_color, side_1_color, side_2_color):
        """Draws a 3D-looking isometric cube."""
        x, y, w, d = world_rect.x, world_rect.y, world_rect.width, world_rect.height
        
        # Calculate 8 corners of the cube
        corners = [
            (x, y), (x + w, y), (x + w, y + d), (x, y + d)
        ]
        screen_corners_top = [self._iso_to_screen(cx, cy) for cx, cy in corners]
        screen_corners_bottom = [(sx, sy + height) for sx, sy in screen_corners_top]

        # Draw faces (order matters for overlap)
        # Top face
        pygame.gfxdraw.aapolygon(surface, screen_corners_top, top_color)
        pygame.gfxdraw.filled_polygon(surface, screen_corners_top, top_color)

        # Right side face
        side_1_points = [screen_corners_top[1], screen_corners_top[2], screen_corners_bottom[2], screen_corners_bottom[1]]
        pygame.gfxdraw.aapolygon(surface, side_1_points, side_1_color)
        pygame.gfxdraw.filled_polygon(surface, side_1_points, side_1_color)

        # Bottom side face
        side_2_points = [screen_corners_top[2], screen_corners_top[3], screen_corners_bottom[3], screen_corners_bottom[2]]
        pygame.gfxdraw.aapolygon(surface, side_2_points, side_2_color)
        pygame.gfxdraw.filled_polygon(surface, side_2_points, side_2_color)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        
        # Reset paddle
        self.paddle_pos_x = (self.WORLD_WIDTH - self.PADDLE_WIDTH) / 2
        
        # Reset ball
        self._reset_ball()

        # Generate bricks
        self.bricks = []
        brick_w, brick_d = 3, 1.5
        rows, cols = 3, 5
        start_x = (self.WORLD_WIDTH - (cols * (brick_w + 1) - 1)) / 2
        start_y = 8
        for r in range(rows):
            for c in range(cols):
                brick_x = start_x + c * (brick_w + 1)
                brick_y = start_y + r * (brick_d + 1)
                color = self.BRICK_COLORS[(r * cols + c) % len(self.BRICK_COLORS)]
                self.bricks.append(Brick(
                    world_pos=(brick_x, brick_y),
                    world_dims=(brick_w, brick_d),
                    color=color
                ))

        self.particles = []
        self.softlock_counter = 0
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        """Puts the ball back on the paddle with a new random velocity."""
        self.ball_pos.x = self.paddle_pos_x + self.PADDLE_WIDTH / 2
        self.ball_pos.y = self.PADDLE_Y - self.BALL_RADIUS
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards angle
        self.ball_vel.x = math.cos(angle) * self.BALL_SPEED
        self.ball_vel.y = math.sin(angle) * self.BALL_SPEED
    
    def step(self, action):
        reward = 0.0
        terminated = False

        # 1. Handle Input
        movement = action[0]
        if movement == 3:  # Left
            self.paddle_pos_x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_pos_x += self.PADDLE_SPEED
        
        self.paddle_pos_x = np.clip(self.paddle_pos_x, 0, self.WORLD_WIDTH - self.PADDLE_WIDTH)
        
        # 2. Update Game Logic
        # Update ball position
        prev_ball_pos = self.ball_pos.copy()
        self.ball_pos += self.ball_vel

        # Ball collisions
        # Walls
        if self.ball_pos.x < self.BALL_RADIUS or self.ball_pos.x > self.WORLD_WIDTH - self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WORLD_WIDTH - self.BALL_RADIUS)
            self.softlock_counter += 1
        if self.ball_pos.y < self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            self.softlock_counter = 0

        # Paddle
        paddle_rect = pygame.Rect(self.paddle_pos_x, self.PADDLE_Y, self.PADDLE_WIDTH, self.PADDLE_DEPTH)
        if self.ball_vel.y > 0 and paddle_rect.collidepoint(self.ball_pos.x, self.ball_pos.y):
            # sound: paddle_hit
            self.ball_pos.y = self.PADDLE_Y - self.BALL_RADIUS
            self.ball_vel.y *= -1
            
            offset = (self.ball_pos.x - paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * self.PADDLE_SPIN_EFFECT
            
            reward += 0.1
            self.softlock_counter = 0

        # Bricks
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        for i in range(len(self.bricks) - 1, -1, -1):
            brick = self.bricks[i]
            if brick.alive and brick.rect.colliderect(ball_rect):
                # sound: brick_destroy
                brick.alive = False
                reward += 1.0
                self.score += 1
                self.softlock_counter = 0

                # Create explosion particles
                for _ in range(15):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(0.1, 0.5)
                    vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
                    self.particles.append(Particle(
                        pos=brick.rect.center,
                        vel=vel,
                        radius=self.np_random.uniform(2, 5),
                        color=brick.color_light,
                        lifetime=self.np_random.integers(15, 30)
                    ))

                # Collision response
                # Determine if it's a horizontal or vertical collision
                prev_ball_rect = pygame.Rect(prev_ball_pos.x - self.BALL_RADIUS, prev_ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
                if prev_ball_rect.bottom <= brick.rect.top or prev_ball_rect.top >= brick.rect.bottom:
                    self.ball_vel.y *= -1
                else:
                    self.ball_vel.x *= -1
                break # Only hit one brick per frame

        self.bricks = [b for b in self.bricks if b.alive]

        # Normalize ball speed
        speed = self.ball_vel.length()
        if speed > 0:
            self.ball_vel = (self.ball_vel / speed) * self.BALL_SPEED

        # Anti-softlock
        if self.softlock_counter > 20:
            self.ball_vel.y += self.np_random.uniform(-0.1, 0.1)
            self.softlock_counter = 0
        
        # Update particles
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifetime > 0]
        
        # 3. Check for Termination
        self.steps += 1
        
        # Loss of life
        if self.ball_pos.y > self.WORLD_HEIGHT:
            # sound: lose_life
            self.lives -= 1
            if self.lives > 0:
                self._reset_ball()
            else:
                terminated = True
                reward = -100.0

        # Win condition
        if not self.bricks:
            # sound: win_game
            terminated = True
            reward = 100.0

        # Max steps
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Draw grid
        for i in range(0, self.WORLD_WIDTH + 1, 2):
            p1 = self._iso_to_screen(i, 0)
            p2 = self._iso_to_screen(i, self.WORLD_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for i in range(0, self.WORLD_HEIGHT + 1, 2):
            p1 = self._iso_to_screen(0, i)
            p2 = self._iso_to_screen(self.WORLD_WIDTH, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

    def _render_game(self):
        # Draw bricks
        for brick in self.bricks:
            self._draw_iso_cube(self.screen, brick.rect, 6, brick.color_light, brick.color_shadow_1, brick.color_shadow_2)

        # Draw paddle
        paddle_rect = pygame.Rect(self.paddle_pos_x, self.PADDLE_Y, self.PADDLE_WIDTH, self.PADDLE_DEPTH)
        self._draw_iso_cube(self.screen, paddle_rect, self.PADDLE_HEIGHT, self.COLOR_PADDLE, (150,150,180), (100,100,120))
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen, self._iso_to_screen)

        # Draw ball and its shadow
        ball_screen_pos = self._iso_to_screen(self.ball_pos.x, self.ball_pos.y)
        ball_height = (self.PADDLE_Y - self.ball_pos.y) * self.ISO_TILE_HEIGHT_HALF * 0.2
        ball_render_pos = (ball_screen_pos[0], ball_screen_pos[1] - max(0, ball_height))
        shadow_size = int(self.BALL_RADIUS * self.ISO_TILE_WIDTH_HALF * (1 - max(0, ball_height)/100))
        
        if shadow_size > 0:
            shadow_surf = pygame.Surface((shadow_size*2, shadow_size*2), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, (0,0,0,80), shadow_surf.get_rect())
            self.screen.blit(shadow_surf, (ball_screen_pos[0]-shadow_size, ball_screen_pos[1]-shadow_size//2))
        
        pygame.gfxdraw.aacircle(self.screen, int(ball_render_pos[0]), int(ball_render_pos[1]), int(self.BALL_RADIUS * self.ISO_TILE_WIDTH_HALF), self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(ball_render_pos[0]), int(ball_render_pos[1]), int(self.BALL_RADIUS * self.ISO_TILE_WIDTH_HALF), self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        life_icon_screen_pos = self._iso_to_screen(0, 0)
        life_icon_radius = int(self.BALL_RADIUS * self.ISO_TILE_WIDTH_HALF * 0.8)
        for i in range(self.lives):
            pos_x = self.SCREEN_WIDTH - 20 - i * (life_icon_radius * 2.5)
            pygame.gfxdraw.aacircle(self.screen, int(pos_x), 20, life_icon_radius, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(pos_x), 20, life_icon_radius, self.COLOR_BALL)

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
            "lives": self.lives,
            "bricks_remaining": len(self.bricks)
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
        # Reset first to ensure all state is initialized for observation
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import time

    # For human play
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a separate window for rendering
    pygame.display.set_caption("Isometric Breakout")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    
    # Game loop
    while not terminated:
        # Get keyboard input
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        action = [movement, 0, 0] # space and shift are unused

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # Cap the frame rate
        time.sleep(1/30)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()