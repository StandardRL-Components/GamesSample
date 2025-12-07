# Generated: 2025-08-28T03:27:28.494328
# Source Brief: brief_02029.md
# Brief Index: 2029

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, isometric block breaker with a retro-neon aesthetic. "
        "Clear all blocks to advance, but lose a life if the ball hits the bottom."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_BALL = (100, 255, 180)
        self.COLOR_BALL_GLOW = (150, 255, 200, 50)
        self.COLOR_GRID = (25, 35, 60)
        self.BLOCK_COLORS = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]
        self.COLOR_TEXT = (255, 255, 255)

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.stage = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached_to_paddle = True
        self.ball_base_speed = 0
        self.ball_speed_multiplier = 1.0
        self.blocks = []
        self.particles = []
        self.ball_trail = []
        
        # Initialize state
        # The reset call is deferred to the first call to reset()
        
        # Validate implementation
        # self.validate_implementation() # Validation is better done after first reset
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.stage = 1
        self.ball_base_speed = 7 # pixels per frame
        self.ball_speed_multiplier = 1.0

        self.paddle = pygame.Rect(
            self.WIDTH // 2 - 50, self.HEIGHT - 40, 100, 10
        )
        
        self.particles = []
        self.ball_trail = []
        
        self._setup_stage()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _setup_stage(self):
        """Resets ball and generates blocks for the current stage."""
        self.ball_attached_to_paddle = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - 10]
        self.ball_vel = [0, 0]
        self.blocks = []
        
        block_width, block_height = 50, 20
        
        if self.stage == 1:
            rows, cols = 4, 10
            for r in range(rows):
                for c in range(cols):
                    if (r + c) % 2 == 0: continue # Checkerboard pattern
                    x = 70 + c * (block_width + 5)
                    y = 60 + r * (block_height + 5)
                    self.blocks.append({
                        'rect': pygame.Rect(x, y, block_width, block_height),
                        'color': self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                    })
        elif self.stage == 2:
            rows, cols = 6, 10
            for r in range(rows):
                for c in range(cols):
                    if c < 2 or c > 7 or (r > 1 and r < 4): # Hollow rectangle
                        x = 70 + c * (block_width + 5)
                        y = 60 + r * (block_height + 5)
                        self.blocks.append({
                            'rect': pygame.Rect(x, y, block_width, block_height),
                            'color': self.BLOCK_COLORS[c % len(self.BLOCK_COLORS)]
                        })
        elif self.stage == 3:
            rows, cols = 7, 10
            for r in range(rows):
                for c in range(cols):
                    if abs(c - 4.5) < r: # Triangle
                        x = 70 + c * (block_width + 5)
                        y = 60 + r * (block_height + 5)
                        self.blocks.append({
                            'rect': pygame.Rect(x, y, block_width, block_height),
                            'color': self.BLOCK_COLORS[(r+c) % len(self.BLOCK_COLORS)]
                        })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.steps += 1
        reward = -0.01  # Time penalty

        # 1. Handle Input
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        paddle_speed = 12
        if movement == 3:  # Left
            self.paddle.x -= paddle_speed
        elif movement == 4:  # Right
            self.paddle.x += paddle_speed
        
        self.paddle.clamp_ip(self.screen.get_rect()) # Keep paddle on screen

        # 2. Update Game Logic
        reward += self._update_ball(space_held)
        self._update_particles()
        
        # 3. Check for stage clear
        if not self.blocks and not self.game_over:
            reward += 10
            self.stage += 1
            self.ball_speed_multiplier += 0.05
            if self.stage > 3:
                self.game_over = True
                reward += 100 # Win game bonus
            else:
                self._setup_stage()
                # sound: stage_clear.wav

        # 4. Check for termination
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_ball(self, space_held):
        reward = 0
        
        if self.ball_attached_to_paddle:
            self.ball_pos[0] = self.paddle.centerx
            if space_held:
                self.ball_attached_to_paddle = False
                angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
                speed = self.ball_base_speed * self.ball_speed_multiplier
                self.ball_vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                # sound: launch.wav
            return reward

        # Update ball trail
        self.ball_trail.append(list(self.ball_pos))
        if len(self.ball_trail) > 5:
            self.ball_trail.pop(0)

        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        ball_radius = 8
        ball_rect = pygame.Rect(self.ball_pos[0] - ball_radius, self.ball_pos[1] - ball_radius, ball_radius * 2, ball_radius * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.WIDTH, ball_rect.right)
            # sound: wall_bounce.wav
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            # sound: wall_bounce.wav
            
        # Bottom wall (lose life)
        if ball_rect.top >= self.HEIGHT:
            self.lives -= 1
            # sound: lose_life.wav
            if self.lives <= 0:
                self.game_over = True
            else:
                self._setup_stage() # Resets ball to paddle
            return reward

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            self.ball_vel[0] += offset * 4
            
            # Clamp velocity to prevent extreme angles and maintain speed
            speed = math.hypot(*self.ball_vel)
            target_speed = self.ball_base_speed * self.ball_speed_multiplier
            if speed > 0:
                scale = target_speed / speed
                self.ball_vel[0] *= scale
                self.ball_vel[1] *= scale

            # Risk/reward for hitting with edge of paddle
            if abs(offset) > 0.8: # outer 20%
                reward += 0.1
                # sound: edge_hit.wav
            else:
                # sound: paddle_hit.wav
                pass

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                reward += 1
                self.score += 10
                self.blocks.remove(block)
                # sound: block_break.wav

                # Create particles
                for _ in range(15):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 4)
                    life = self.np_random.integers(15, 30)
                    self.particles.append([
                        list(ball_rect.center),
                        [math.cos(angle) * speed, math.sin(angle) * speed],
                        life,
                        block['color']
                    ])
                
                # Collision response: determine if hit was more horizontal or vertical
                dx = ball_rect.centerx - block['rect'].centerx
                dy = ball_rect.centery - block['rect'].centery
                
                if abs(dx / block['rect'].width) > abs(dy / block['rect'].height):
                    self.ball_vel[0] *= -1
                else:
                    self.ball_vel[1] *= -1
                
                break # Only handle one block collision per frame

        return reward
        
    def _update_particles(self):
        for p in self.particles[:]:
            p[0][0] += p[1][0] # Update x
            p[0][1] += p[1][1] # Update y
            p[1][1] += 0.1 # Gravity
            p[2] -= 1 # Decrement life
            if p[2] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _render_iso_rect(self, surface, rect, color, depth=8):
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        
        top_face = [(x, y + h/2), (x + w/2, y), (x + w, y + h/2), (x + w/2, y + h)]
        side_face_y = y + h/2 + depth
        
        # Darken color for sides
        dark_color = tuple(max(0, c - 60) for c in color)
        
        # Draw side faces
        pygame.draw.polygon(surface, dark_color, [(x, y + h/2), (x + w/2, y + h), (x + w/2, side_face_y + h), (x, side_face_y)])
        pygame.draw.polygon(surface, dark_color, [(x + w, y + h/2), (x + w/2, y + h), (x + w/2, side_face_y + h), (x + w, side_face_y)])
        
        # Draw top face
        pygame.draw.polygon(surface, color, top_face)
        # FIX: Use pygame.gfxdraw.aapolygon with correct argument order
        pygame.gfxdraw.aapolygon(surface, top_face, (0,0,0)) # Black outline for clarity

    def _render_game(self):
        # Render blocks
        for block in self.blocks:
            self._render_iso_rect(self.screen, block['rect'], block['color'])

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        pygame.draw.rect(self.screen, (0,0,0), self.paddle.inflate(2,2), width=1, border_radius=3)

        # Render ball trail
        if len(self.ball_trail) > 1:
            for i, pos in enumerate(self.ball_trail):
                alpha = (i + 1) / len(self.ball_trail) * 100
                color = (*self.COLOR_BALL, alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 7, color)

        # Render ball with glow
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, *ball_pos_int, 12, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, *ball_pos_int, 12, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, *ball_pos_int, 8, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, *ball_pos_int, 8, self.COLOR_BALL)
        
        # Render particles
        for p in self.particles:
            pos, _, life, color = p
            alpha = max(0, min(255, life * 15))
            size = max(1, int(life / 6))
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (size, size), size)
            self.screen.blit(s, (int(pos[0]) - size, int(pos[1]) - size))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_large.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width()//2, 10))
        
        if self.game_over:
            msg = "GAME OVER" if self.lives <= 0 else "YOU WIN!"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Validating implementation...")
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
        assert trunc is False
        assert isinstance(info, dict)
        
        # Test ball speed increase
        self.reset()
        self.stage = 1
        self.ball_speed_multiplier = 1.0
        self.blocks = [] # Force a stage clear
        self.step(self.action_space.sample())
        assert self.stage == 2
        assert math.isclose(self.ball_speed_multiplier, 1.05)
        
        # Assert game over after 3 lives
        self.reset()
        self.lives = 1
        self.ball_pos = [self.WIDTH/2, self.HEIGHT + 10] # force ball out of bounds
        self.ball_vel = [0, 10]
        self.ball_attached_to_paddle = False
        _, _, terminated, _, _ = self.step(self.action_space.sample())
        assert self.lives == 0
        assert self.game_over is True
        assert terminated is True
        
        self.reset() # Reset to a clean state
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    
    # Run validation
    try:
        env.validate_implementation()
    except Exception as e:
        print(f"Validation failed: {e}")
        env.close()
        exit()

    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    # To play, you might need to remove/comment the os.environ line at the top
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Isometric Block Breaker")
    except pygame.error as e:
        print(f"Could not create display: {e}")
        print("Running in headless mode. No visual output will be shown.")
        screen = None

    clock = pygame.time.Clock()

    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
        if screen: # Only process events if a display is available
            # Construct the action from keyboard input
            movement = 0 # no-op
            space_held = 0
            shift_held = 0

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            if keys[pygame.K_RIGHT]:
                movement = 4
            if keys[pygame.K_SPACE]:
                space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_held = 1

            action = [movement, space_held, shift_held]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
        else: # If headless, just step with a random action
             action = env.action_space.sample()
             if done:
                 print(f"Game over. Score: {info['score']}. Resetting.")
                 obs, info = env.reset()
                 done = False


        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation from the environment to the screen
        if screen:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        clock.tick(env.FPS)

    env.close()