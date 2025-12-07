
# Generated: 2025-08-27T23:17:14.268067
# Source Brief: brief_03420.md
# Brief Index: 3420

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An expert implementation of a visually-rich, arcade-style block breaker game
    adhering to the Gymnasium API.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Short, user-facing description of the game
    game_description = (
        "A fast-paced block breaker where risky paddle hits are rewarded. Clear all blocks to win."
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 60)
    COLOR_PADDLE = (230, 230, 255)
    COLOR_PADDLE_SHADOW = (150, 150, 180)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (200, 200, 0)
    COLOR_TEXT = (220, 220, 240)
    BLOCK_COLORS = {
        10: (50, 205, 50),   # Green
        20: (65, 105, 225),  # Blue
        30: (220, 20, 60),   # Red
    }
    
    # Game parameters
    PADDLE_HEIGHT = 12
    PADDLE_SPEED = 10
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 6.0
    MAX_LIVES = 3
    MAX_STAGES = 3
    MAX_STEPS = 5000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.lives = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.ball_attached_to_paddle = True
        self.blocks = []
        self.particles = []

        # This will be properly seeded in reset()
        self.np_random = np.random.default_rng()
        
        self.reset()
        
        # Run self-check after initialization
        # self.validate_implementation() # Commented out for submission, but useful for testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.lives = self.MAX_LIVES
        self.particles = []
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Initializes the game state for the current stage."""
        self.paddle = pygame.Rect(
            self.SCREEN_WIDTH // 2 - 50, 
            self.SCREEN_HEIGHT - 30, 
            100, 
            self.PADDLE_HEIGHT
        )
        
        self.ball_attached_to_paddle = True
        self.ball_speed = self.INITIAL_BALL_SPEED + (self.stage - 1) * 0.5
        self._reset_ball()

        # Generate blocks
        self.blocks = []
        block_width = 50
        block_height = 20
        rows = 4 + self.stage  # More rows in later stages
        cols = 11
        
        # Increase density with stage
        block_probability = 0.5 + self.stage * 0.15 
        
        for r in range(rows):
            for c in range(cols):
                if self.np_random.random() < block_probability:
                    points = self.np_random.choice(list(self.BLOCK_COLORS.keys()))
                    color = self.BLOCK_COLORS[points]
                    block_rect = pygame.Rect(
                        c * (block_width + 6) + 32,
                        r * (block_height + 6) + 50,
                        block_width,
                        block_height
                    )
                    self.blocks.append({"rect": block_rect, "points": points, "color": color})

    def _reset_ball(self):
        """Attaches the ball to the paddle."""
        self.ball_attached_to_paddle = True
        self.ball_pos = pygame.math.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.math.Vector2(0, 0)
        # Sound placeholder: // Sound: Ball reset

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed
        
        if not self.game_over:
            # --- Handle Input ---
            movement = action[0]
            space_held = action[1] == 1
            
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            
            # Clamp paddle to screen
            self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.paddle.width, self.paddle.x))

            if self.ball_attached_to_paddle and space_held:
                self.ball_attached_to_paddle = False
                angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
                self.ball_vel = pygame.math.Vector2(math.sin(angle), -math.cos(angle)) * self.ball_speed
                # Sound placeholder: // Sound: Ball launch
            
            # --- Update Game State ---
            reward += self._update_game_state()

        # --- Check for Termination ---
        if self.lives <= 0:
            self.game_over = True
        
        if not self.blocks and self.stage > self.MAX_STAGES:
             self.game_over = True # Game won

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self):
        """Main game logic update function."""
        reward = 0
        
        # Update ball position
        if self.ball_attached_to_paddle:
            self.ball_pos.x = self.paddle.centerx
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
        else:
            self.ball_pos += self.ball_vel
            
        # --- Collisions ---
        # Walls
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.SCREEN_WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.SCREEN_WIDTH - self.BALL_RADIUS, self.ball_pos.x))
            # Sound placeholder: // Sound: Wall bounce
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # Sound placeholder: // Sound: Wall bounce

        # Miss
        if self.ball_pos.y - self.BALL_RADIUS > self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 10
            if self.lives > 0:
                self._reset_ball()
            # Sound placeholder: // Sound: Lose life

        # Paddle
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
            # Sound placeholder: // Sound: Paddle bounce
            
            # Reposition ball to prevent sticking
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
            
            # Calculate bounce angle and risk reward
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.paddle.width / 2)
            offset = max(-1.0, min(1.0, offset))
            
            if abs(offset) > 0.6: # Risky edge hit
                reward += 0.1
            elif abs(offset) < 0.2: # Safe center hit
                reward -= 0.02

            bounce_angle = offset * (math.pi / 2.5) # Max angle ~72 degrees
            self.ball_vel = pygame.math.Vector2(math.sin(bounce_angle), -math.cos(bounce_angle)) * self.ball_speed

        # Blocks
        hit_block = None
        for block in self.blocks:
            if block["rect"].colliderect(ball_rect):
                hit_block = block
                break
        
        if hit_block:
            self.blocks.remove(hit_block)
            reward += hit_block["points"]
            self.score += hit_block["points"]
            self.ball_vel.y *= -1 # Simple bounce
            self._create_particles(hit_block["rect"].center, hit_block["color"])
            # Sound placeholder: // Sound: Block break

        # Stage clear
        if not self.blocks and self.stage <= self.MAX_STAGES:
            reward += 100
            self.stage += 1
            if self.stage <= self.MAX_STAGES:
                self._setup_stage()
                # Sound placeholder: // Sound: Stage clear
            else:
                self.game_over = True # Won the game
                # Sound placeholder: // Sound: Game win
                
        # Update particles
        self._update_particles()
        
        return reward
        
    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.math.Vector2(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "radius": self.np_random.uniform(1, 4),
                "color": color,
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

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
        # Background Grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Blocks
        for block in self.blocks:
            r = block["rect"]
            pygame.draw.rect(self.screen, block["color"], r, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block["color"]), r, width=2, border_radius=3)

        # Paddle
        shadow_rect = self.paddle.copy()
        shadow_rect.y += 4
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_SHADOW, shadow_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=6)
        
        # Ball Glow
        glow_radius = int(self.BALL_RADIUS * 1.8)
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.ball_pos.x), int(self.ball_pos.y),
            glow_radius, (*self.COLOR_BALL_GLOW, 50)
        )
        # Ball
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.ball_pos.x), int(self.ball_pos.y),
            self.BALL_RADIUS, self.COLOR_BALL
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(self.ball_pos.x), int(self.ball_pos.y),
            self.BALL_RADIUS, self.COLOR_BALL
        )
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 20))))
            color_with_alpha = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"].x), int(p["pos"].y),
                int(p["radius"]), color_with_alpha
            )

    def _render_ui(self):
        # Score
        score_surf = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))

        # Lives
        for i in range(self.lives):
            pos_x = self.SCREEN_WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 10))
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_BALL)
            
        # Stage
        stage_surf = self.font_small.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        stage_rect = stage_surf.get_rect(centerx=self.SCREEN_WIDTH / 2)
        stage_rect.top = 10
        self.screen.blit(stage_surf, stage_rect)

        # Game Over / Win message
        if self.game_over:
            if self.lives <= 0:
                msg = "GAME OVER"
            else:
                msg = "YOU WIN!"
            
            end_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Simple shadow for text
            shadow_surf = self.font_large.render(msg, True, (0,0,0))
            shadow_rect = shadow_surf.get_rect(center=(self.SCREEN_WIDTH / 2 + 3, self.SCREEN_HEIGHT / 2 + 3))
            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(end_surf, end_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
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

if __name__ == '__main__':
    # --- Example of how to run the environment ---
    env = GameEnv(render_mode="rgb_array")
    
    # --- To display the game, we need a display surface ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # --- Main Game Loop ---
    running = True
    while running:
        # --- Action gathering from keyboard ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is the rendered frame. We just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if done:
            print(f"Episode finished. Final Info: {info}")
            obs, info = env.reset()
            
        env.clock.tick(30) # Control the frame rate

    env.close()